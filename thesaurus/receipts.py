"""Bounded local receipt reading. The result is always a private, unconfirmed draft.

Embedded PDF text is preferred. Scans/photos use local OCR only; fiscal IDs are
hashed for deduplication, never returned together with receipt/customer details.
"""
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from .wine_tools import _identity, _image

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 10
MAX_TEXT = 100_000
MAX_ITEMS = 200
_MONEY = r"(?:\d{1,3}(?:\s\d{3})+|\d+)(?:[.,]\d{1,2})?"
_QTY = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d{1,3})?)\s*(?:шт\.?|кг|л)?\s*[xх×*]\s*(" + _MONEY + r")(?![\d.,])", re.IGNORECASE)
_TOTAL = re.compile(r"^(?:ИТОГ(?:О)?|К\s+ОПЛАТЕ|ВСЕГО(?:\s+К\s+ОПЛАТЕ)?|СУММА\s+ЧЕКА)\s*[:=]?\s*(" + _MONEY + r")\s*(?:₽|руб\.?)?$", re.IGNORECASE)
_TOTAL_LABEL = re.compile(r"^(?:ИТОГ(?:О)?|К\s+ОПЛАТЕ|ВСЕГО(?:\s+К\s+ОПЛАТЕ)?|СУММА\s+ЧЕКА)\s*[:=]?$", re.IGNORECASE)
_META = re.compile(r"^(?:кассовый\s+чек|чек\s+передан|приход\b|о[оа]о\b|пао\b|зао\b|ао\b|ип\b|"
                   r"адрес\b|место\s+расч|кассир\b|номер\b|регистрацион|инн\b|фн\b|фд\b|фпд?\b|"
                   r"офд\b|эл\.?\s*адр|версия\s+ффд|признак\b|расчета\b|расчёта\b|подакциз|полный\s+расчет|"
                   r"мера\s+кол|в\s*т\.?\s*ч\.?|сумма\s+(?:ндс|с\s+ндс|по\s+чеку)|ндс\b|"
                   r"наличными\b|безналичными\b|система\s+налого|карта\b|ваша\s+скидка|баллов\b|"
                   r"списано\b|начислено\b|дата\b|покупатель\b|телефон\b|e-?mail\b|сайт\b|www\.|https?://)", re.IGNORECASE)
_VOLUME_UNIT = re.compile(r"(?<![\d,−+\-])(?<!\d\.)(\d+(?:[.,]\d+)?)\s*"
                          r"(мл|ml|сл|cl|литров|литра|литр|л|l)(?!\w)", re.IGNORECASE)
_VOLUME_SUFFIX = re.compile(r"(?<![\d,−+\-])(?<!\d\.)(\d+[.,]\d{1,3})(?![\d.,])\s*\)?\s*$")
_VOLUME_OTHER_FIELD = re.compile(r"(?:\b(?:цена|стоимость|сумма|количество|кол-во|qty|price|скидка|"
                                 r"крепость|abv|алк(?:оголь)?|арт(?:икул)?|код|no)|№)\s*[:=.\-]?\s*$", re.IGNORECASE)
# Unitless integers could be a vintage, item count, code or price. Only common
# fractional bottle formats at the very end of a wine name are safe suggestions.
_SUFFIX_BOTTLE_ML = {187, 200, 250, 375, 500, 750, 1500}


def _clean(text, limit=MAX_TEXT):
    if not isinstance(text, str) or len(text) > limit:
        raise ValueError("Слишком большой или некорректный текст чека")
    text = unicodedata.normalize("NFKC", text)
    if len(text) > limit:
        raise ValueError("Слишком большой текст чека")
    return "".join(c for c in text if c in "\n\r\t" or unicodedata.category(c) not in {"Cc", "Cf"})


def _money(value):
    value = re.sub(r"\s", "", value).replace(",", ".")
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if not number.is_finite() or not 0 <= number <= 1_000_000_000:
        return None
    return number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _fiscal_fingerprint(text):
    patterns = {
        "fn": r"(?:\bФН\b|фискальный\s+накопитель)[\s:№#]*(\d{10,20})\b|(?:[?&]|\b)fn=(\d{10,20})\b",
        "fd": r"(?:\b(?:номер\s+)?ФД\b|фискальный\s+документ)[\s:№#]*(\d{1,10})\b|(?:[?&]|\b)i=(\d{1,10})\b",
        "fp": r"(?:\bФПД?\b|фискальный\s+признак(?:\s+документа)?)[\s:№#]*(\d{5,12})\b|(?:[?&]|\b)fp=(\d{5,12})\b",
    }
    values = []
    for pattern in patterns.values():
        found = {str(int(next(value for value in match if value)))
                 for match in re.findall(pattern, text, re.IGNORECASE)}
        if len(found) != 1:
            return None
        values.append(found.pop())
    return hashlib.sha256(("fiscal:v1:" + ":".join(values)).encode("ascii")).hexdigest()


def _receipt_date(text):
    patterns = (r"\b(?:дата(?:\s+выдачи)?|чек\s+от)\s*[:]?\s*(\d{2}[./-]\d{2}[./-](?:20)?\d{2})",
                r"\b(\d{2}\.\d{2}\.(?:20)?\d{2})\b", r"\b(20\d{2}-\d{2}-\d{2})\b")
    for pattern in patterns:
        for value in re.findall(pattern, text, re.IGNORECASE):
            try:
                parts = re.split(r"[./-]", value)
                if len(parts[0]) == 4:
                    result = date(*map(int, parts))
                else:
                    day, month, year = map(int, parts)
                    result = date(year if year >= 100 else 2000 + year, month, day)
                if 1990 <= result.year <= datetime.now(timezone.utc).year + 1:
                    return result.isoformat()
            except ValueError:
                continue
    return None


def receipt_volume(name, *, is_wine=True):
    """Suggest a bottle size from a receipt name; unitless suffixes need review."""
    empty = {"volume_ml": None, "volume_source": None, "volume_requires_confirmation": False}
    if not is_wine or not isinstance(name, str) or len(name) > 500:
        return empty
    name = unicodedata.normalize("NFKC", name)
    explicit, malformed = set(), False
    for match in _VOLUME_UNIT.finditer(name):
        number, unit = match.groups()
        multiplier = 10 if unit.casefold() in {"cl", "сл"} else 1 if unit.casefold() in {"ml", "мл"} else 1000
        amount = Decimal(number.replace(",", ".")) * multiplier
        if amount != amount.to_integral_value() or not 20 <= amount <= 30000:
            malformed = True
        else:
            explicit.add(int(amount))
    suffix = _VOLUME_SUFFIX.search(name)
    inferred = None
    if suffix and not _VOLUME_OTHER_FIELD.search(name[:suffix.start()]):
        # A second bare decimal in the name makes a unitless size ambiguous;
        # explicitly labelled alcohol percentages and sizes remain distinguishable.
        prefix = _VOLUME_UNIT.sub("", name[:suffix.start()])
        prefix = re.sub(r"\d+[.,]\d+\s*%", "", prefix)
        if not re.search(r"\d+[.,]\d+", prefix):
            amount = Decimal(suffix.group(1).replace(",", ".")) * 1000
            if amount in _SUFFIX_BOTTLE_ML:
                inferred = int(amount)
    values = explicit | ({inferred} if inferred is not None else set())
    if malformed or len(values) != 1:
        return empty
    return {"volume_ml": values.pop(),
            "volume_source": "name_explicit_unit" if explicit else "name_decimal_suffix",
            "volume_requires_confirmation": not bool(explicit)}


def receipt_identity(name, *, is_wine=True):
    """Re-read identity fields derived from a receipt name after a correction."""
    # An unusual explicit size such as 2020 ml is not a vintage year.
    identity = _identity({"name": _VOLUME_UNIT.sub("", name)}) if is_wine else {}
    return {"vintage": identity.get("vintage"), **receipt_volume(name, is_wine=is_wine)}


def _item(name, quantity, listed_price, explicit_amount=None):
    name = re.sub(r"^\d{1,3}[.)]\s+", "", " ".join(name.split())).strip(" ;")[:500]
    if not name or not any(c.isalpha() for c in name) or "@" in name:
        return None
    is_wine = bool(re.search(r"\b(?:вино|игрист\w*|шампанск\w*|wine|champagne|prosecco|cava)\b", name, re.IGNORECASE))
    if re.search(r"\b(?:уксус|соус|бокал|штопор|открывашка)\b", name, re.IGNORECASE):
        is_wine = False
    return {"name": name, "quantity": float(quantity), "unit_price": float(listed_price),
            "amount": None, "is_wine": is_wine, **receipt_identity(name, is_wine=is_wine),
            "winery": "", "_quantity": quantity,
            "_listed": listed_price, "_explicit": explicit_amount, "_discount": Decimal(0)}


def _finish(item):
    amount = item["_explicit"]
    if amount is None:
        amount = (item["_quantity"] * item["_listed"] - item["_discount"]).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if not amount.is_finite() or not 0 <= amount <= 1_000_000_000:
        item["amount"] = None
        item["unit_price"] = None
    else:
        item["amount"] = float(amount)
        item["unit_price"] = float((amount / item["_quantity"]).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
    return {key: value for key, value in item.items() if not key.startswith("_")}


def _empty(status="unavailable", warnings=None):
    return {"status": status, "date": None, "merchant": "", "total": None, "currency": "RUB",
            "items": [], "warnings": warnings or [], "fingerprint": "", "requires_confirmation": True,
            "reconciled": False}


def parse_receipt_text(text):
    """Read explicit quantity × price rows and their final amounts, excluding VAT/bonus totals."""
    try:
        text = _clean(text)
    except ValueError:
        return _empty(warnings=["Текст чека слишком большой или не прочитан; заполните сведения вручную."])
    result = _empty()
    result["date"] = _receipt_date(text)
    result["fingerprint"] = _fiscal_fingerprint(text) or hashlib.sha256(text.encode("utf-8")).hexdigest()
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    merchant = next((line for line in lines[:100] if re.match(r"^(?:ООО|АО|ПАО|ЗАО|ИП)\b", line, re.IGNORECASE)), "")
    result["merchant"] = merchant[:200]
    if re.search(r"\b(?:USD|EUR|GBP|CNY)\b|[$€£]", text):
        result["warnings"].append("Обнаружена другая валюта; автоматический разбор поддерживает только рублёвые чеки.")
        return result
    if re.search(r"\bвозврат\s+прихода\b", text, re.IGNORECASE):
        result["warnings"].append("Это чек возврата; его нельзя автоматически записать как новый расход.")
        return result
    items, buffer, current, totals = [], [], None, []
    pending_total = False
    ended = False
    for line in lines:
        total = _TOTAL.fullmatch(line)
        if total or pending_total:
            value = total.group(1) if total else line if re.fullmatch(_MONEY + r"\s*(?:₽|руб\.?)?", line, re.IGNORECASE) else None
            pending_total = False
            if value:
                amount = _money(re.sub(r"\s*(?:₽|руб\.?)$", "", value, flags=re.IGNORECASE))
                if amount is not None:
                    totals.append(amount)
                    if current:
                        items.append(_finish(current))
                        current = None
                    ended = True
                    continue
        if _TOTAL_LABEL.fullmatch(line):
            pending_total = True
            continue
        if ended:
            continue
        quantity_price = _QTY.search(line)
        if quantity_price:
            if current:
                items.append(_finish(current))
            quantity = Decimal(quantity_price.group(1).replace(",", "."))
            listed = _money(quantity_price.group(2))
            prefix = line[:quantity_price.start()].strip()
            name = prefix if any(c.isalpha() for c in prefix) else " ".join(buffer)
            suffix = line[quantity_price.end():].strip()
            explicit = re.fullmatch(r"=\s*(" + _MONEY + r")\s*(?:₽|руб\.?)?", suffix, re.IGNORECASE)
            if quantity <= 0 or quantity > 100_000 or listed is None:
                current = None
                result["warnings"].append("Позиция с некорректным количеством или ценой не добавлена.")
            else:
                current = _item(name, quantity, listed, _money(explicit.group(1)) if explicit else None)
                if current is None:
                    result["warnings"].append("У позиции не прочитано название; проверьте состав чека.")
            buffer = []
            if len(items) > MAX_ITEMS:
                return _empty(warnings=["В чеке больше 200 позиций; требуется ручное добавление."])
            continue
        final = re.fullmatch(r"=\s*(" + _MONEY + r")\s*(?:₽|руб\.?)?", line, re.IGNORECASE)
        if current and final:
            current["_explicit"] = _money(final.group(1))
            continue
        discount = re.fullmatch(r"скидка(?:\s+по\s+товару)?\s*[:=-]?\s*(" + _MONEY + r")\s*(?:₽|руб\.?)?", line, re.IGNORECASE)
        if current and discount:
            current["_discount"] += _money(discount.group(1)) or Decimal(0)
            continue
        if _META.search(line) or re.fullmatch(r"[\d\s.,=:%#/-]+", line):
            buffer = []
            continue
        if any(c.isalpha() for c in line) and len(buffer) < 8:
            buffer.append(line[:500])
    if current:
        items.append(_finish(current))
    if len(items) > MAX_ITEMS:
        return _empty(warnings=["В чеке больше 200 позиций; требуется ручное добавление."])
    result["items"] = items
    unique_totals = set(totals)
    if len(unique_totals) == 1:
        result["total"] = float(unique_totals.pop())
    elif len(unique_totals) > 1:
        result["warnings"].append("В чеке найдены разные итоги; укажите правильную сумму вручную.")
    if result["total"] is None:
        result["warnings"].append("Итог чека не прочитан однозначно; укажите его вручную.")
    elif items and all(item["amount"] is not None for item in items):
        difference = abs(sum(Decimal(str(item["amount"])) for item in items) - Decimal(str(result["total"])))
        result["reconciled"] = difference <= Decimal("0.02")
        if not result["reconciled"]:
            result["warnings"].append("Сумма позиций не совпала с итогом чека; проверьте скидки и непрочитанные строки.")
    if not items:
        result["warnings"].append("Позиции не прочитаны; их можно добавить вручную. Сумма расхода требует подтверждения.")
    if any(item["is_wine"] and (item["vintage"] is None or item["volume_ml"] is None) for item in items):
        result["warnings"].append("У некоторых вин не опубликованы год урожая или единицы объёма; подтвердите их вручную.")
    if any(item["volume_requires_confirmation"] for item in items):
        result["warnings"].append("Объём некоторых вин определён из названия без единиц измерения; проверьте объём каждой бутылки перед добавлением в винотеку.")
    if any(item["is_wine"] and not item["quantity"].is_integer() for item in items):
        result["warnings"].append("Дробное количество вина требует проверки; в запас добавляются целые бутылки.")
    if not result["date"]:
        result["warnings"].append("Дата покупки не прочитана; укажите её вручную.")
    result["status"] = "candidate" if items or result["total"] is not None else "unavailable"
    result["warnings"].append("Проверьте дату, итог и позиции перед сохранением. Расход и пополнение запасов ещё не записаны.")
    return result


def _ocr_text(image):
    """Use a text-only local OCR executable; never call a vision LLM or cloud service."""
    language = os.getenv("RECEIPT_OCR_LANG", "rus+eng").strip()
    if not re.fullmatch(r"[A-Za-z_]+(?:\+[A-Za-z_]+)*", language) or len(language) > 50:
        raise ValueError("Некорректный язык OCR")
    tesseract = shutil.which("tesseract")
    powershell = shutil.which("powershell.exe") if sys.platform == "win32" else None
    helper = Path(__file__).resolve().parents[1] / "scripts" / "wine_windows_ocr.ps1"
    with tempfile.TemporaryDirectory(prefix="receipt-ocr-") as directory:
        path = Path(directory) / "receipt.jpg"
        path.write_bytes(image)
        attempts = []
        if tesseract:
            attempts.append(("tesseract", [tesseract, str(path), "stdout", "-l", language, "--psm", "6"]))
        if powershell and helper.is_file():
            attempts.append(("windows_ocr", [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                                            "-File", str(helper), "-ImagePath", str(path), "-LanguageTag", "ru-RU"]))
        for provider, argv in attempts:
            try:
                result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=45,
                                        shell=False, check=False, encoding="utf-8", errors="replace")
                if not result.returncode and 0 < len(result.stdout.strip()) <= MAX_TEXT:
                    return _clean(result.stdout), provider
            except (OSError, subprocess.SubprocessError):
                continue
    raise ValueError("Локальный OCR не смог прочитать чек. Нужен русский язык OCR Windows или Tesseract rus+eng; можно ввести расход вручную.")


def _normalize_ocr_money(text):
    """Correct O/О only in a two-digit monetary fraction on explicit money rows."""
    lines, pending_total = [], False
    for line in text.splitlines(keepends=True):
        monetary = (pending_total or bool(re.match(r"\s*(?:=|ИТОГ(?:О)?\b|К\s+ОПЛАТЕ\b|ВСЕГО\b|СУММА\s+ЧЕКА\b|СКИДКА\b)", line, re.IGNORECASE))
                    or bool(re.search(r"\d\s*(?:шт\.?|кг|л)?\s*[xх×*]\s*\d", line, re.IGNORECASE)))
        if monetary:
            line = re.sub(r"(?<![\w.,])(\d[\d ]*[.,])([\dOoОо]{2})(?![\w.,])",
                          lambda match: match.group(1) + re.sub(r"[OoОо]", "0", match.group(2)), line)
        pending_total = bool(_TOTAL_LABEL.fullmatch(line.strip()))
        lines.append(line)
    return "".join(lines)


def _render_page(raw, index):
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(raw, password="") as document, document[index] as page:
        width, height = page.get_size()
        if not (0 < width <= 20_000 and 0 < height <= 20_000):
            raise ValueError("Слишком большой размер страницы")
        scale = min(2, 2200 / max(width, height))
        with page.render(scale=scale) as bitmap:
            image = bitmap.to_pil().convert("RGB")
            try:
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=95)
                return output.getvalue()
            finally:
                image.close()


def _pdf_text(raw):
    from pypdf import PdfReader, apply_configuration, get_configuration
    from pypdf.errors import PyPdfError

    warnings = []
    # Context-local limits are restored on failure; unrelated PDF work is not changed.
    configuration = get_configuration()
    limit_names = ("zlib_maximum_output_length", "lzw_maximum_output_length", "run_length_maximum_output_length",
                   "array_based_stream_maximum_output_length", "maximum_declared_stream_length",
                   "brotli_maximum_output_length", "jbig2_maximum_output_length")
    limits = {name: min(getattr(configuration, name), 4 * 1024 * 1024) for name in limit_names}
    with apply_configuration(**limits, image_maximum_buffer_size=16 * 1024 * 1024,
                             stream_decoding_work_maximum_length=32 * 1024 * 1024,
                             page_tree_maximum_entries=1000, page_tree_maximum_depth=30,
                             xform_maximum_invocations_per_extraction=500, jbig2dec_binary=None,
                             disable_legacy_handling=True):
        try:
            reader = PdfReader(io.BytesIO(raw), strict=True)
            if reader.is_encrypted:
                if not reader.decrypt(""):
                    raise ValueError("PDF защищён паролем; пришлите открытый PDF или фото")
                warnings.append("PDF прочитан без пароля; ограничения редактирования не мешают распознаванию.")
            if not 1 <= len(reader.pages) <= MAX_PDF_PAGES:
                raise ValueError("В PDF должно быть от 1 до 10 страниц")
            texts, providers = [], []
            for index, page in enumerate(reader.pages):
                resources = page.get("/Resources") or {}
                resources = resources.get_object() if hasattr(resources, "get_object") else resources
                objects = resources.get("/XObject") or {}
                objects = objects.get_object() if hasattr(objects, "get_object") else objects
                for value in objects.values():
                    obj = value.get_object()
                    if obj.get("/Subtype") == "/Image" and int(obj.get("/Width", 0)) * int(obj.get("/Height", 0)) > 36_000_000:
                        raise ValueError("Слишком большое встроенное изображение PDF")
                text = page.extract_text() or ""
                if len(text.strip()) < 20:
                    text, provider = _ocr_text(_render_page(raw, index))
                    providers.append(provider)
                else:
                    providers.append("pypdf")
                texts.append(_clean(text))
                if sum(map(len, texts)) > MAX_TEXT:
                    raise ValueError("Слишком много текста в PDF")
            return "\n".join(texts), "+".join(sorted(set(providers))), warnings
        except PyPdfError:
            raise ValueError("PDF не удалось прочитать безопасно") from None


def recognize_receipt(path, *, dry_run=False):
    """Read bounded PDF/photo locally. Never create an expense or cellar entry."""
    if dry_run:
        return _empty("dry_run", ["Распознавание не запускалось; файл и сеть не использовались."])
    try:
        path = Path(path)
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_PDF_BYTES:
            raise ValueError("Чек должен быть файлом до 10 МБ")
        with path.open("rb") as stream:
            raw = stream.read(MAX_PDF_BYTES + 1)
        if not 0 < len(raw) <= MAX_PDF_BYTES:
            raise ValueError("Чек должен быть файлом до 10 МБ")
        if raw.startswith(b"%PDF-"):
            text, provider, warnings = _pdf_text(raw)
        else:
            text, provider = _ocr_text(_image(path))
            warnings = ["Текст распознан с фото локально; особенно внимательно проверьте цифры."]
        if "ocr" in provider:
            normalized = _normalize_ocr_money(text)
            if normalized != text:
                warnings.append("В копейках OCR прочитал букву O вместо нуля; суммы нормализованы и требуют проверки.")
            text = normalized
        result = parse_receipt_text(text)
        result["fingerprint"] = _fiscal_fingerprint(text) or hashlib.sha256(raw).hexdigest()
        result["provider"] = provider
        result["warnings"] = warnings + result["warnings"]
        return result
    except ImportError:
        return _empty(warnings=["Для PDF нужен локальный pypdf>=6.10; для сканов — pypdfium2 и локальный OCR. Доступен ручной ввод."])
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError, OverflowError, subprocess.SubprocessError) as exc:
        message = str(exc) if isinstance(exc, ValueError) and str(exc).startswith(("PDF защищён", "В PDF должно", "Чек должен", "Для безопасного", "Слишком", "Локальный OCR")) else "Не удалось безопасно прочитать чек локально"
        return _empty(warnings=[message + ". Отправьте более чёткое фото или заполните сведения вручную."])
