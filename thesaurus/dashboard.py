"""Loopback-only administration UI; mutations enqueue work, never send messages.

The app deliberately has no Telegram imports. Browser sessions are random,
server-side and expire on restart. The password is retained only as a salted
PBKDF2 digest inside the application, never in SQLite or the browser cookie.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import ipaddress
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.background import BackgroundTask

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
COOKIE_NAME = "thesaurus_admin"
SESSION_SECONDS = 8 * 60 * 60
MAX_FORM_BYTES = 32_768
NAVIGATION = [
    {"key": "overview", "url": "/", "label": "Обзор"},
    {"key": "history", "url": "/history", "label": "История данных"},
    {"key": "compare", "url": "/compare", "label": "Сравнение"},
    {"key": "reports", "url": "/reports", "label": "Выпуски"},
    {"key": "forecasts", "url": "/forecasts", "label": "Прогнозы"},
    {"key": "jobs", "url": "/jobs", "label": "Задания и доставка"},
    {"key": "subscriptions", "url": "/subscriptions", "label": "Подписки"},
    {"key": "settings", "url": "/settings", "label": "Запросы и лимиты"},
]
REPORT_TYPES = {"ruonia": "RUONIA", "ai": "AI-обзор", "wordstat": "Wordstat", "regions": "Регионы"}
STATUS_LABELS = {
    "pending": "В очереди", "queued": "В очереди", "running": "Выполняется",
    "completed": "Готово", "done": "Готово", "succeeded": "Готово", "failed": "Ошибка",
    "partial": "Частично", "sent": "Доставлено", "sending": "Отправляется",
    "uncertain": "Нужна проверка", "cancelled": "Отменено", "skipped": "Пропущено",
    "ready": "Готово", "blocked": "Приостановлено", "retry": "Повтор",
    "numbers_checked": "Числа сверены", "unverified": "Не подтверждено",
    "ok": "Полные", "missing": "Пропуск", "interrupted": "Прервано",
}


@dataclass
class _Session:
    csrf: str
    expires: float
    authenticated: bool = False


class _Access:
    def __init__(self, password: str):
        self.salt = secrets.token_bytes(32)
        self.digest = self._digest(password)
        self.sessions: dict[str, _Session] = {}
        self.failures: deque[float] = deque()
        self.lock = threading.RLock()

    def _digest(self, password: str) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), self.salt, 600_000)

    def new_session(self, authenticated: bool = False) -> tuple[str, _Session]:
        with self.lock:
            now = time.monotonic()
            self.sessions = {key: value for key, value in self.sessions.items() if value.expires > now}
            if len(self.sessions) >= 256:
                oldest = min(self.sessions, key=lambda key: self.sessions[key].expires)
                del self.sessions[oldest]
            token = secrets.token_urlsafe(32)
            session = _Session(secrets.token_urlsafe(32), now + SESSION_SECONDS, authenticated)
            self.sessions[token] = session
            return token, session

    def get(self, token: str | None) -> _Session | None:
        with self.lock:
            session = self.sessions.get(token or "")
            if session and session.expires > time.monotonic():
                return session
            if token:
                self.sessions.pop(token, None)
            return None

    def remove(self, token: str | None) -> None:
        with self.lock:
            self.sessions.pop(token or "", None)

    def verify_password(self, password: str) -> bool:
        with self.lock:
            now = time.monotonic()
            while self.failures and now - self.failures[0] >= 60:
                self.failures.popleft()
            if len(self.failures) >= 5:
                raise HTTPException(429, "Слишком много попыток. Повторите вход через минуту.")
            valid = len(password) <= 4096 and hmac.compare_digest(self.digest, self._digest(password))
            if valid:
                self.failures.clear()
            else:
                self.failures.append(now)
            return valid


def _config(settings: Any, key: str, default: Any = None) -> Any:
    if isinstance(settings, dict):
        return settings.get(key, default)
    return getattr(settings, key, default)


def _local_request(request: Request) -> bool:
    """Reject alternate DNS names, proxy headers and remote connections."""
    try:
        parsed = urlsplit("http://" + request.headers.get("host", ""))
        # Reading port also validates malformed or out-of-range values.
        _ = parsed.port
        host_ok = parsed.hostname in {"127.0.0.1", "localhost"}
        clean_host = not parsed.username and not parsed.password and not parsed.path and not parsed.query and not parsed.fragment
        client_ok = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
        return bool(host_ok and clean_host and client_ok)
    except ValueError:
        return False


async def _form(request: Request) -> dict[str, str]:
    """Only small URL-encoded forms are needed; no multipart dependency."""
    if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/x-www-form-urlencoded":
        raise HTTPException(415, "Неподдерживаемый формат формы.")
    chunks = bytearray()
    async for chunk in request.stream():
        chunks.extend(chunk)
        if len(chunks) > MAX_FORM_BYTES:
            raise HTTPException(413, "Форма слишком большая.")
    try:
        data = parse_qs(chunks.decode("utf-8"), keep_blank_values=True, max_num_fields=80)
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(400, "Не удалось прочитать форму.") from None
    if any(len(values) != 1 for values in data.values()):
        raise HTTPException(400, "Поля формы не должны повторяться.")
    return {key: values[0] for key, values in data.items()}


async def _protected_form(request: Request, access: _Access) -> dict[str, str]:
    expected_origin = f"http://{request.headers.get('host', '')}"
    if request.headers.get("origin") != expected_origin:
        raise HTTPException(403, "Источник запроса не разрешён.")
    session = access.get(request.cookies.get(COOKIE_NAME))
    if not session:
        raise HTTPException(403, "Сеанс истёк. Войдите снова.")
    values = await _form(request)
    if not hmac.compare_digest(values.pop("csrf", ""), session.csrf):
        raise HTTPException(403, "Форма устарела. Обновите страницу.")
    return values


def _set_cookie(response: Any, token: str) -> None:
    response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True,
                        samesite="strict", secure=False, path="/")


def _safe_report_path(root: Path, relative: str) -> Path:
    """Only regular, registered report files inside the configured root."""
    root = root.resolve()
    candidate = Path(relative)
    if ".." in candidate.parts:
        raise HTTPException(404, "Файл не найден.")
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise HTTPException(404, "Файл не найден.")
    return resolved


def _date_filter(value: str) -> str:
    if value:
        try:
            date.fromisoformat(value)
        except ValueError:
            raise HTTPException(400, "Укажите дату в формате ГГГГ-ММ-ДД.") from None
    return value


def _positive_id(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise HTTPException(400, "Укажите числовой ID.") from None
    if result <= 0:
        raise HTTPException(400, "ID должен быть положительным числом.")
    return result


def create_app(store: Any, settings: Any = None, *, password: str | None = None,
               report_root: str | Path | None = None) -> FastAPI:
    """Create the UI without starting servers, jobs or any external clients."""
    enabled = _config(settings, "dashboard_enabled", os.getenv("DASHBOARD_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"})
    if enabled is not True:
        raise ValueError("Веб-панель отключена. Для её запуска задайте DASHBOARD_ENABLED=1.")
    password = password if password is not None else _config(settings, "dashboard_password", os.getenv("DASHBOARD_PASSWORD", ""))
    if not password or not password.strip():
        raise ValueError("Задайте DASHBOARD_PASSWORD перед запуском локальной панели.")
    access = _Access(password)
    # Do not retain the settings object or plaintext password in app state.
    del password
    data_dir = Path(_config(settings, "data_dir", HERE.parent / "data"))
    reports_dir = Path(report_root) if report_root is not None else Path(_config(settings, "reports_dir") or HERE.parent / "src" / "reports")
    app = FastAPI(title="Thesaurus", docs_url=None, redoc_url=None, openapi_url=None)
    templates = Environment(loader=FileSystemLoader(HERE / "templates"),
                            autoescape=select_autoescape(("html", "xml")))
    templates.filters["report_name"] = lambda value: REPORT_TYPES.get(value, value)
    templates.filters["status_name"] = lambda value: STATUS_LABELS.get(value, value)
    templates.filters["number"] = lambda value: "—" if value is None else f"{value:,.3f}".rstrip("0").rstrip(".").replace(",", " ")
    templates.filters["percentage"] = lambda value: "—" if value is None else f"{value * 100:.1f}%"
    app.state.store = store

    def render(request: Request, template: str, status_code: int = 200, **context: Any) -> HTMLResponse:
        session = access.get(request.cookies.get(COOKIE_NAME))
        defaults = {"navigation": NAVIGATION, "csrf": session.csrf if session else "", "chart_url": "",
                    "report_types": REPORT_TYPES, "error": "", "message": ""}
        defaults.update(context)
        return HTMLResponse(templates.get_template(template).render(**defaults), status_code=status_code)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if not _local_request(request):
            response = HTMLResponse("Доступ разрешён только с этого компьютера.", status_code=403)
        else:
            public = request.url.path in {"/login", "/static/dashboard.css", "/favicon.ico"}
            session = access.get(request.cookies.get(COOKIE_NAME))
            if not public and not (session and session.authenticated):
                response = RedirectResponse("/login", status_code=303)
            else:
                try:
                    response = await call_next(request)
                except Exception as exc:  # noqa: BLE001 - HTTP boundary hides internal and provider details.
                    LOGGER.error("Dashboard request failed (%s)", type(exc).__name__)
                    response = render(request, "error.html", 500, title="Не удалось выполнить действие",
                                      message="Обновите страницу. Если ошибка повторится, проверьте журнал сервиса.")
        response.headers.update({
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
            "Referrer-Policy": "same-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
        })
        return response

    @app.exception_handler(HTTPException)
    async def known_error(request: Request, exc: HTTPException):
        return render(request, "error.html", exc.status_code, title="Действие не выполнено", message=exc.detail)

    @app.get("/login")
    async def login_page(request: Request):
        old = access.get(request.cookies.get(COOKIE_NAME))
        if old and old.authenticated:
            return RedirectResponse("/", status_code=303)
        if old:
            # Parallel tabs or incidental login redirects must not invalidate
            # the CSRF token already visible in the user's first login form.
            return render(request, "login.html", csrf=old.csrf)
        token, session = access.new_session()
        response = render(request, "login.html", csrf=session.csrf)
        _set_cookie(response, token)
        return response

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @app.post("/login")
    async def login(request: Request):
        values = await _protected_form(request, access)
        if not access.verify_password(values.get("password", "")):
            return render(request, "login.html", 400, error="Неверный пароль.")
        access.remove(request.cookies.get(COOKIE_NAME))
        token, _ = access.new_session(authenticated=True)
        response = RedirectResponse("/", status_code=303)
        _set_cookie(response, token)
        return response

    @app.post("/logout")
    async def logout(request: Request):
        await _protected_form(request, access)
        access.remove(request.cookies.get(COOKIE_NAME))
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    @app.get("/static/plotly.min.js")
    async def plotly_asset():
        import plotly
        asset = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
        if not asset.is_file():
            raise HTTPException(503, "Компонент графиков недоступен.")
        return FileResponse(asset, media_type="text/javascript")

    _register_views(app, store, render, access, reports_dir, data_dir / "exports")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    return app


def _register_views(app: FastAPI, store: Any, render: Any, access: _Access,
                    reports_dir: Path, exports_dir: Path) -> None:
    """Register read views and validated queue/configuration actions."""
    from .store import dumps

    def decoded(value, default):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default

    def page(request, key, title, description, **context):
        messages = {"saved": "Настройки сохранены.", "queued": "Задание добавлено в очередь. Оно выполнится при работающем сервисе.",
                    "resolved": "Решение о доставке сохранено.", "approved": "Список запросов обновлён."}
        return render(request, "pages.html", page=key, title=title, description=description,
                      message=messages.get(request.query_params.get("notice"), ""), **context)

    def redirect(path, notice="saved"):
        return RedirectResponse(path + "?" + urlencode({"notice": notice}), status_code=303)

    def numbers(raw):
        try:
            values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
        except (TypeError, ValueError):
            raise HTTPException(400, "Укажите числовые ID через запятую.") from None
        if len(values) > 100 or any(value <= 0 for value in values):
            raise HTTPException(400, "Можно выбрать до 100 положительных ID.")
        return values

    def filters(request):
        p = request.query_params
        values = {key: p.get(key, "").strip() for key in ("source", "frequency", "phrases", "regions", "date_from", "date_to")}
        for key in ("date_from", "date_to"):
            _date_filter(values[key])
        if values["date_from"] and values["date_to"] and values["date_from"] > values["date_to"]:
            raise HTTPException(400, "Начало периода должно быть не позднее его конца.")
        phrases = list(dict.fromkeys(value.strip() for value in values["phrases"].split(",") if value.strip()))
        if len(phrases) > 100 or any(len(value) > 120 for value in phrases):
            raise HTTPException(400, "Можно выбрать до 100 запросов длиной до 120 символов.")
        regions = numbers(values["regions"])
        if values["frequency"] not in {"", "daily", "weekly", "monthly"}:
            raise HTTPException(400, "Неизвестная частота наблюдений.")
        return values, phrases, regions

    def observations(request, limit=501):
        values, phrases, regions = filters(request)
        clauses, params = [], []
        for key in ("source", "frequency"):
            if values[key]:
                clauses.append(key + "=?")
                params.append(values[key])
        for key, operator in (("date_from", ">="), ("date_to", "<=")):
            if values[key]:
                clauses.append("date" + operator + "?")
                params.append(values[key])
        for column, selected in (("metric", phrases), ("region_id", regions)):
            if selected:
                clauses.append(column + " IN (" + ",".join("?" for _ in selected) + ")")
                params.extend(selected)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = store.query("SELECT source,metric,region_id,device,frequency,date,value,unit,quality,retrieved_at FROM observations" + where + " ORDER BY date DESC,metric,region_id LIMIT ?", (*params, limit))
        return values, rows

    @app.get("/")
    async def overview(request: Request):
        summary = store.one("SELECT (SELECT COUNT(*) FROM observations) observations,(SELECT COUNT(*) FROM reports) reports,(SELECT COUNT(*) FROM jobs WHERE state IN ('pending','running')) jobs,(SELECT COUNT(*) FROM deliveries WHERE state='uncertain') uncertain")
        runs = store.query("SELECT id,section,status,started_at,completed_at,quality FROM runs ORDER BY started_at DESC LIMIT 8")
        latest = store.query("SELECT source,frequency,MAX(date) latest,COUNT(*) amount FROM observations GROUP BY source,frequency ORDER BY source,frequency")
        alerts = store.query("SELECT metric,region_id,observed_date,created_at FROM alerts ORDER BY created_at DESC LIMIT 8")
        return page(request, "overview", "Состояние наблюдений", "Свежесть данных, последние запуски и сообщения, которым нужна проверка.", summary=summary, runs=runs, latest=latest, alerts=alerts)

    @app.get("/history")
    @app.get("/compare")
    async def history(request: Request):
        key = "compare" if request.url.path == "/compare" else "history"
        values, rows = observations(request)
        query = urlencode({k: v for k, v in values.items() if v})
        return page(request, key, "Сравнение показателей" if key == "compare" else "История данных",
                    "Выберите период, запросы и регионы. Наблюдения разных частот и единиц измерения сохраняются отдельно.",
                    rows=rows[:500], truncated=len(rows) > 500, filters=values, query=query,
                    sources=store.query("SELECT DISTINCT source FROM observations ORDER BY source"),
                    chart_url="/api/chart?" + query)

    @app.get("/api/chart")
    async def chart(request: Request):
        _, rows = observations(request, 5001)
        units = {row["unit"] for row in rows}
        if len(units) > 1:
            return JSONResponse({"traces": [], "message": "Выберите один источник: значения в разных единицах нельзя сравнивать на одной шкале."})
        if len(rows) > 5000:
            return JSONResponse({"traces": [], "message": "Уточните период и фильтры: для графика выбрано больше 5 000 наблюдений."})
        grouped = {}
        for row in reversed(rows):
            key = (row["source"], row["metric"], row["region_id"], row["device"], row["frequency"])
            if key not in grouped:
                grouped[key] = {"name": html.escape(f"{row['metric']} · {row['region_id']} · {row['frequency']} · {row['device']}"), "dates": [], "values": []}
            grouped[key]["dates"].append(row["date"])
            grouped[key]["values"].append(row["value"])
        if len(grouped) > 20:
            return JSONResponse({"traces": [], "message": "Выберите до 20 рядов, чтобы график оставался читаемым."})
        return JSONResponse({"traces": list(grouped.values()), "unit": next(iter(units), "")})

    @app.get("/export/{format_name}")
    async def export(request: Request, format_name: str):
        if format_name not in {"csv", "xlsx"}:
            raise HTTPException(404, "Неизвестный формат выгрузки.")
        values, phrases, regions = filters(request)
        from .exporting import export_data
        exports_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="observations-", suffix="." + format_name, dir=exports_dir, delete=False) as file:
            target = Path(file.name)
        try:
            export_data(store, target, format=format_name, date_from=values["date_from"] or None,
                        date_to=values["date_to"] or None, phrases=phrases or None,
                        regions=regions or None, source=values["source"] or None,
                        frequency=values["frequency"] or None)
        except ValueError:
            target.unlink(missing_ok=True)
            raise HTTPException(400, "Проверьте формат и фильтры выгрузки.") from None
        except Exception:
            target.unlink(missing_ok=True)
            raise
        media = "text/csv; charset=utf-8" if format_name == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        return FileResponse(target, filename="thesaurus-observations." + format_name, media_type=media,
                            background=BackgroundTask(target.unlink, missing_ok=True))

    @app.get("/reports")
    async def reports(request: Request):
        rows = store.query("SELECT id,section,created_at,period_start,period_end,keyword_version,cohort_version FROM reports ORDER BY created_at DESC LIMIT 100")
        return page(request, "reports", "Выпуски", "Сохранённые отчёты и материалы. Повторная доставка использует уже подготовленный выпуск.", reports=rows)

    @app.get("/reports/{report_id}")
    async def report(request: Request, report_id: str):
        row = store.one("SELECT * FROM reports WHERE id=?", (report_id,))
        if not row:
            raise HTTPException(404, "Выпуск не найден.")
        parts = decoded(row["parts"], [])
        files = [{"index": index, "name": Path(part.get("path", "")).name or "Файл выпуска"}
                 for index, part in enumerate(parts) if isinstance(part, dict) and part.get("kind") in {"photo", "document"} and part.get("path")]
        news = decoded(row["news"], [])
        safe_news = []
        for item in news if isinstance(news, list) else []:
            if isinstance(item, dict):
                url = str(item.get("url", ""))
                safe_news.append({"title": item.get("title", "Источник"), "url": url if urlsplit(url).scheme in {"http", "https"} else "",
                                  "status": item.get("status", "unverified"), "published_at": item.get("published_at") or "Дата не указана",
                                  "availability": {"available": "Доступна", "unavailable": "Недоступна", "unreachable": "Не отвечает",
                                      "blocked": "Проверка заблокирована", "redirect_limit": "Цепочка перенаправлений", "not_checked": "Не проверялась"}.get(
                                      (item.get("source_availability") or {}).get("availability"), "Не проверялась")})
        return page(request, "reports", "Выпуск: " + REPORT_TYPES.get(row["section"], row["section"]),
                    "Содержание сохранено на дату выпуска. Статус источников указан отдельно.", report=row, files=files, news=safe_news)

    @app.get("/reports/{report_id}/files/{part_index}")
    async def report_file(report_id: str, part_index: int):
        row = store.one("SELECT parts FROM reports WHERE id=?", (report_id,))
        parts = decoded(row["parts"], []) if row else []
        if part_index < 0 or part_index >= len(parts):
            raise HTTPException(404, "Файл не найден.")
        part = parts[part_index]
        if not isinstance(part, dict) or part.get("kind") not in {"photo", "document"} or not part.get("path"):
            raise HTTPException(404, "Файл не найден.")
        target = _safe_report_path(reports_dir, part["path"])
        return FileResponse(target, filename=target.name, content_disposition_type="attachment")

    @app.get("/forecasts")
    async def forecasts(request: Request):
        rows = store.query("SELECT metric,region_id,origin_date,target_date,horizon,prediction,lower,upper,baseline,actual,model FROM forecasts ORDER BY origin_date DESC,metric,horizon LIMIT 100")
        scores = store.query("""SELECT source,metric,region_id,device,frequency,model,horizon,COUNT(*) amount,
            AVG(ABS(prediction-actual)) mae,AVG(ABS(baseline-actual)) baseline_mae,
            SUM(ABS(prediction-actual))/NULLIF(SUM(ABS(actual)),0) wape,
            SUM(CASE WHEN baseline IS NOT NULL THEN ABS(baseline-actual) END)
                /NULLIF(SUM(CASE WHEN baseline IS NOT NULL THEN ABS(actual) END),0) baseline_wape,
            AVG(CASE WHEN lower IS NOT NULL AND upper IS NOT NULL
                THEN CASE WHEN actual BETWEEN lower AND upper THEN 1.0 ELSE 0.0 END END) interval_coverage
            FROM forecasts WHERE actual IS NOT NULL AND datetime(created_at)<datetime(target_date)
            AND ABS(prediction)<1e308 AND ABS(actual)<1e308
            AND (baseline IS NULL OR ABS(baseline)<1e308)
            AND (lower IS NULL OR ABS(lower)<1e308) AND (upper IS NULL OR ABS(upper)<1e308)
            GROUP BY source,metric,region_id,device,frequency,model,horizon
            ORDER BY metric,region_id,horizon LIMIT 200""")
        return page(request, "forecasts", "Проверка прогнозов", "Ошибка оценивается после появления факта. Базовый прогноз повторяет последнее известное значение.", rows=rows, scores=scores)

    @app.post("/actions/reports/{report_id}/verify-sources")
    async def verify_sources(request: Request, report_id: str):
        await _protected_form(request, access)
        if not store.one("SELECT id FROM reports WHERE id=?", (report_id,)):
            raise HTTPException(404, "Выпуск не найден.")
        store.enqueue("verify_sources", {"report_id": report_id})
        return redirect("/jobs", "queued")

    @app.get("/jobs")
    async def jobs(request: Request):
        rows = store.query("SELECT id,kind,payload,state,created_at,started_at,finished_at FROM jobs ORDER BY id DESC LIMIT 100")
        for row in rows:
            payload = decoded(row.pop("payload"), {})
            row["section"] = payload.get("section", "")
            row["recipient"] = payload.get("recipient") or "Только подготовка"
            row["job_name"] = REPORT_TYPES.get(row["section"], {"propose_keywords": "Предложения запросов", "verify_sources": "Проверка источников"}.get(row["kind"], "Задание"))
        pending = store.query("SELECT id,report_id,recipient,part_index,attempts,updated_at FROM deliveries WHERE state='uncertain' ORDER BY updated_at LIMIT 100")
        recipients = store.query("SELECT DISTINCT s.recipient FROM subscriptions s LEFT JOIN users u ON u.user_id=s.user_id WHERE s.enabled=1 AND (s.user_id IS NULL OR u.enabled=1) ORDER BY s.recipient")
        return page(request, "jobs", "Задания и доставка", "Действия выполняет сервис. Если он остановлен, новые задания остаются в очереди.", jobs=rows, uncertain=pending, recipients=recipients)

    @app.post("/actions/report")
    async def enqueue_report(request: Request):
        values = await _protected_form(request, access)
        section = values.get("section", "")
        if section not in REPORT_TYPES:
            raise HTTPException(400, "Выберите тип отчёта.")
        payload = {"section": section}
        recipient = values.get("recipient", "").strip()
        if recipient == "*":
            subscriptions = store.query("SELECT s.* FROM subscriptions s LEFT JOIN users u ON u.user_id=s.user_id WHERE s.section=? AND s.enabled=1 AND (s.user_id IS NULL OR u.enabled=1)", (section,))
            if not subscriptions:
                raise HTTPException(400, "Активных подписок на этот отчёт нет. Можно подготовить выпуск без доставки.")
            for subscriber in subscriptions:
                store.enqueue("report", {"section": section, "recipient": subscriber["recipient"], "detail": subscriber["detail"],
                                         "phrases": decoded(subscriber["phrases"], []), "regions": decoded(subscriber["regions"], [])})
            return redirect("/jobs", "queued")
        if recipient:
            subscriber = store.one("SELECT s.* FROM subscriptions s LEFT JOIN users u ON u.user_id=s.user_id WHERE s.recipient=? AND s.section=? AND s.enabled=1 AND (s.user_id IS NULL OR u.enabled=1)", (recipient, section))
            if not subscriber:
                raise HTTPException(400, "У получателя нет активной подписки на этот отчёт.")
            payload.update(recipient=recipient, detail=subscriber["detail"], phrases=decoded(subscriber["phrases"], []), regions=decoded(subscriber["regions"], []))
        store.enqueue("report", payload)
        return redirect("/jobs", "queued")

    @app.post("/actions/delivery/{delivery_id}")
    async def resolve_delivery(request: Request, delivery_id: int):
        values = await _protected_form(request, access)
        action = values.get("action", "")
        if action not in {"retry", "sent"}:
            raise HTTPException(400, "Неизвестное решение о доставке.")
        from .delivery import resolve_uncertain
        try:
            resolve_uncertain(store, delivery_id, action)
        except ValueError:
            raise HTTPException(409, "Доставка уже обработана или больше не требует решения.") from None
        return redirect("/jobs", "resolved")

    @app.get("/subscriptions")
    async def subscriptions(request: Request):
        rows = store.query("SELECT * FROM subscriptions ORDER BY recipient,section")
        for row in rows:
            for key in ("phrases", "regions", "times", "weekdays"):
                row[key] = decoded(row[key], [])
        editing = None
        if request.query_params.get("edit"):
            editing = next((row for row in rows if row["id"] == _positive_id(request.query_params["edit"])), None)
            if not editing:
                raise HTTPException(404, "Подписка не найдена.")
            if editing["section"] == "ruonia":
                editing["available_phrases"] = ["RUONIA", "1 мес", "3 мес", "6 мес"]
                editing["available_regions"] = [225]
            elif editing["section"] == "wordstat":
                editing["available_phrases"] = [entry["phrase"] for entry in store.keywords()]
                editing["available_regions"] = [225]
            elif editing["section"] == "regions":
                editing["available_phrases"] = store.get_setting("incident_phrases", [])
                editing["available_regions"] = store.regions()
            else:
                editing["available_phrases"] = []
                editing["available_regions"] = []
        users = store.query("SELECT user_id,role,enabled FROM users ORDER BY role,user_id")
        return page(request, "subscriptions", "Подписки и доступ", "Разрешённые пользователи управляют своими подписками в Telegram. Каждое расписание использует свой часовой пояс.", subscriptions=rows, users=users, editing=editing)

    @app.post("/actions/user")
    async def user_action(request: Request):
        values = await _protected_form(request, access)
        user_id = _positive_id(values.get("user_id", ""))
        role = values.get("role", "subscriber")
        if role != 'subscriber':
            raise HTTPException(400, 'Администраторы задаются через ADMIN_TELEGRAM_IDS.')
        if values.get("enabled") not in {"0", "1"}:
            raise HTTPException(400, "Выберите состояние доступа.")
        from .market_store import change_access
        try:
            change_access(store, 0, user_id, values['enabled'] == '1', authenticated_dashboard=True)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return redirect("/subscriptions")

    @app.post("/actions/subscription")
    async def subscription_action(request: Request):
        values = await _protected_form(request, access)
        subscription_id = _positive_id(values.get("subscription_id", ""))
        subscription = store.one("SELECT id,section FROM subscriptions WHERE id=?", (subscription_id,))
        if not subscription:
            raise HTTPException(404, "Подписка не найдена.")
        from .scheduling import validate_filters, validate_subscription
        try:
            days = [int(value.strip()) - 1 for value in values.get("weekdays", "").split(",") if value.strip()]
            normalized = validate_subscription({"enabled": int(values.get("enabled", "0")), "detail": values.get("detail", "brief"),
                "times": [v.strip() for v in values.get("times", "").split(",") if v.strip()], "weekdays": days,
                "timezone": values.get("timezone", "Europe/Moscow"), "phrases": [v.strip() for v in values.get("phrases", "").splitlines() if v.strip()],
                "regions": numbers(values.get("regions", ""))})
        except ValueError:
            raise HTTPException(400, "Проверьте расписание: время ЧЧ:ММ, дни от 1 до 7, корректные запросы и регионы.") from None
        try:
            validate_filters(store, subscription["section"], normalized["phrases"], normalized["regions"])
        except ValueError:
            raise HTTPException(400, "Выбирайте запросы и регионы, доступные для этого типа отчёта. AI-обзор не использует эти фильтры.") from None
        encoded = {key: dumps(value) if isinstance(value, list) else value for key, value in normalized.items()}
        store.execute("UPDATE subscriptions SET " + ",".join(key + "=?" for key in encoded) + " WHERE id=?", (*encoded.values(), subscription_id))
        return redirect("/subscriptions")

    @app.get("/settings")
    async def settings_page(request: Request):
        proposals = store.query("SELECT id,created_at,entries FROM keyword_proposals WHERE state='pending' ORDER BY id DESC LIMIT 10")
        for proposal in proposals:
            proposal["entries"] = decoded(proposal["entries"], [])
        usage = store.query("SELECT provider,day,SUM(calls) calls,SUM(tokens_in) tokens_in,SUM(tokens_out) tokens_out,SUM(cost) cost,COUNT(cost) priced,COUNT(*) attempts FROM usage GROUP BY provider,day ORDER BY day DESC,provider LIMIT 20")
        return page(request, "settings", "Запросы, регионы и лимиты", "Изменения наборов сохраняются с версией. Предложения AI применяются только после вашего решения.",
                    keywords=store.keywords(), regions=store.regions(), keyword_version=store.version("keyword"), cohort_version=store.version("cohort"),
                    budgets=store.get_setting("budgets", {}), proposals=proposals, usage=usage,
                    rules=store.query("SELECT * FROM alert_rules ORDER BY id"),
                    incident_phrases=store.get_setting("incident_phrases", []))

    @app.post("/actions/keywords")
    async def keywords_action(request: Request):
        values = await _protected_form(request, access)
        entries = []
        for line in values.get("entries", "").splitlines():
            if line.strip():
                phrase, _, theme = line.partition("|")
                entries.append({"phrase": phrase.strip(), "theme": theme.strip() or "другое"})
        try:
            store.set_keywords(entries)
        except ValueError:
            raise HTTPException(400, "Укажите от 1 до 100 запросов, каждый длиной до 120 символов.") from None
        return redirect("/settings")

    @app.post("/actions/propose-keywords")
    async def propose_keywords(request: Request):
        await _protected_form(request, access)
        store.enqueue("propose_keywords")
        return redirect("/jobs", "queued")

    @app.post("/actions/proposals/{proposal_id}")
    async def proposal_action(request: Request, proposal_id: int):
        from .management import resolve_proposal
        values = await _protected_form(request, access)
        row = store.one("SELECT entries FROM keyword_proposals WHERE id=? AND state='pending'", (proposal_id,))
        if not row:
            raise HTTPException(404, "Предложение уже рассмотрено или не найдено.")
        try:
            state = resolve_proposal(store, proposal_id, values.get("action"))
        except ValueError:
            raise HTTPException(400, "Предложение не удалось применить. Проверьте действие и список запросов.") from None
        return redirect("/settings", "approved" if state == "approved" else "saved")

    @app.post("/actions/regions")
    async def regions_action(request: Request):
        values = await _protected_form(request, access)
        try:
            store.set_regions(numbers(values.get("regions", "")))
        except ValueError:
            raise HTTPException(400, "Выберите от 1 до 100 регионов.") from None
        return redirect("/settings")

    @app.post("/actions/budgets")
    async def budget_action(request: Request):
        values = await _protected_form(request, access)
        import math

        from .budget import set_budgets
        try:
            changes = {key: int(values.get(key, "")) for key in ("wordstat_calls", "ai_calls", "ai_output_tokens")}
            if not (0 <= changes["wordstat_calls"] <= 1_000_000 and 0 <= changes["ai_calls"] <= 10_000 and 128 <= changes["ai_output_tokens"] <= 100_000):
                raise ValueError()
        except (ValueError, TypeError):
            raise HTTPException(400, "Укажите неотрицательные лимиты запросов и от 128 до 100 000 выходных токенов.") from None
        prices = {}
        try:
            for provider in ("ai", "wordstat"):
                tariff = {}
                for field in ("per_call", "input_per_million", "output_per_million"):
                    raw = values.get(f"{provider}_{field}", "").strip()
                    if raw:
                        number = float(raw)
                        if not math.isfinite(number) or number < 0:
                            raise ValueError()
                        tariff[field] = number
                if tariff:
                    prices[provider] = tariff
            changes["prices"] = prices
            if "currency" in values:
                currency = values["currency"].strip()
                if not 1 <= len(currency) <= 12 or any(ord(char) < 32 for char in currency):
                    raise ValueError()
                changes["currency"] = currency
            set_budgets(store, changes)
        except (ValueError, TypeError):
            raise HTTPException(400, "Проверьте тарифы и единицу стоимости. Цены должны быть неотрицательными; единицу нельзя менять после первой учтённой стоимости.") from None
        return redirect("/settings")

    def alert_filters(values):
        metric = values.get("metric", "").strip() or None
        region_id = _positive_id(values["region_id"]) if values.get("region_id", "").strip() else None
        if metric and metric not in set(store.get_setting("incident_phrases", [])):
            raise HTTPException(400, "Выберите запрос из списка региональных наблюдений.")
        if region_id and region_id not in set(store.regions()):
            raise HTTPException(400, "Выберите регион из закреплённого набора.")
        return metric, region_id

    @app.post("/actions/alerts")
    async def add_alert(request: Request):
        values = await _protected_form(request, access)
        name = values.get("name", "").strip()
        if not 1 <= len(name) <= 120:
            raise HTTPException(400, "Название правила должно содержать от 1 до 120 символов.")
        metric, region_id = alert_filters(values)
        store.execute("INSERT INTO alert_rules(name,metric,region_id,enabled) VALUES(?,?,?,0)", (name, metric, region_id))
        return redirect("/settings")

    @app.post("/actions/alerts/{rule_id}")
    async def alert_action(request: Request, rule_id: int):
        values = await _protected_form(request, access)
        if not store.one("SELECT id FROM alert_rules WHERE id=?", (rule_id,)):
            raise HTTPException(404, "Правило не найдено.")
        import math
        keys = ("pct_threshold", "z_threshold", "min_coverage", "min_baseline", "cooldown_hours")
        try:
            fields = [float(values.get(key, "")) for key in keys]
            if any(not math.isfinite(value) or value < 0 for value in fields) or fields[2] > 1 or values.get("enabled") not in {"0", "1"}:
                raise ValueError()
        except (ValueError, TypeError):
            raise HTTPException(400, "Проверьте пороги: значения должны быть неотрицательными, покрытие от 0 до 1.") from None
        metric, region_id = alert_filters(values)
        store.execute("UPDATE alert_rules SET pct_threshold=?,z_threshold=?,min_coverage=?,min_baseline=?,cooldown_hours=?,enabled=?,metric=?,region_id=? WHERE id=?", (*fields, int(values["enabled"]), metric, region_id, rule_id))
        return redirect("/settings")
