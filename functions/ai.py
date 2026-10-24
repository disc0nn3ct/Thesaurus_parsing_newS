# ai.py
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

# =============================================================================
# ENV + CLIENT
# =============================================================================
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

AI_API_KEY = os.getenv("ai_api_key")
BASE_URL = os.getenv("base_url")  # Perplexity/OpenAI compatible base_url
MODEL = os.getenv("ai_model", "sonar-pro")

client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Create the AI client only when an AI feature is requested."""
    global client
    if client is None:
        if not AI_API_KEY:
            raise RuntimeError("ai_api_key is missing in the environment or .env")
        # retry_call owns the retry budget; do not multiply it by SDK retries.
        client = OpenAI(
            api_key=AI_API_KEY, base_url=BASE_URL, timeout=60.0, max_retries=0
        )
    return client

# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("ai_brief")

# =============================================================================
# SETTINGS
# =============================================================================
MSK_TZ = timezone(timedelta(hours=3))

# Если хочешь слухи/любые источники:
# AI_USE_DOMAIN_FILTER=0  (в .env)
# Если хочешь держаться “более официального”:
# AI_USE_DOMAIN_FILTER=1
AI_USE_DOMAIN_FILTER = os.getenv("AI_USE_DOMAIN_FILTER", "1").strip() not in ("0", "false", "False", "")

# Домены "качественных" источников (если включен фильтр).
RU_DOMAINS = [
    "moex.com",
    "cbr.ru",
    "minfin.gov.ru",
    "e-disclosure.ru",
    "interfax.ru",
    "rbc.ru",
    "vedomosti.ru",
    "kommersant.ru",
    "spimex.com",
]

# Высокая производительность:
# - меньше токенов => быстрее
# - низкая температура => меньше “фантазий”
MAX_TOKENS = int(os.getenv("ai_max_tokens", "2200"))
TEMPERATURE = float(os.getenv("ai_temperature", "0.2"))
TOP_P = float(os.getenv("ai_top_p", "0.9"))

# Поиск по свежести: можно “hour”, “day”, иногда “week” (если пустит).
SEARCH_RECENCY_AM = os.getenv("ai_search_recency_am", "day")
SEARCH_RECENCY_PM = os.getenv("ai_search_recency_pm", "day")

# =============================================================================
# PROMPTS (RU)
# =============================================================================

SYSTEM_PROMPT = """
Ты — **реактивный трейдинг-аналитик по рынку РФ (MOEX)**.
Твоя задача — помочь трейдеру быстро реагировать на новости: кто выигрывает/проигрывает, что может быть недооценено,
какие триггеры и риски важны в ближайшие 24–72 часа.

Ключевое:
- Давай **торговые гипотезы** (LONG/SHORT/AVOID) на основе новостей. Это не “гарантия” и не инвестиционный совет — это сценарии для реакции.
- Каждая идея: что произошло → почему важно → кто затронут → направление → инвалидатор/стоп-условие → уверенность.
- Разрешены **непроверенные источники/слухи**, потому что они полезны для реактивной торговли.
  Но обязательно:
  1) помечай такие штуки как SRC=C,
  2) прямо пиши “не подтверждено”,
  3) понижай CONF (обычно low/medium).

Грейды источников (SRC):
- A = официально (биржа/регулятор/раскрытие/компания)
- B = крупные СМИ/прайм-лента (Интерфакс/РБК/Ведомости/Коммерсант и т.п.)
- C = слухи/неподтверждено/соцсети/телеграм/“market chatter”

Формат:
- Telegram НЕ поддерживает markdown-таблицы. Все многоколонковые блоки — только в ```text``` (моноширинно).
- Ссылки указывай как URL в скобках: (https://...)
- Не выдумывай цифры. Если не нашёл — напиши “нет свежих цифр / не подтверждено”.
- Язык: Русский.
- Объём: до ~650 слов (без JSON).
""".strip()

USER_TEMPLATE = """
Собери **{edition_name} бриф по рынку MOEX** (для трейдера). Время отчёта: {as_of_msk} (MSK).

Дай 5 разделов строго в этом порядке:

1) **Снимок рынка (2–5 пунктов)**
   - RTS/MOEX, рубль (USD/RUB), ОФЗ (10Y если есть), нефть Brent/Urals, ключевые макро-события (ЦБ/Минфин), ликвидность.
   - Если по нескольким пунктам нет свежих данных — ОБЪЕДИНЯЙ в одну строку (праздники / тонкий рынок).
   - Не выдумывай цифры.

2) **Новости, которые двигают бумаги (MOEX) — 6–12 строк**
   - ВЫВОД ТОЛЬКО В ВИДЕ МОНОТАБЛИЦЫ (Telegram-friendly):
```text
TICKER | НОВОСТЬ | ПОЧЕМУ ВАЖНО | IMPACT | SRC | CONF
```
   - IMPACT = Bullish/Bearish/Mixed
   - SRC = A (официально) / B (крупные СМИ) / C (слухи, не подтверждено)
   - CONF = low / medium / high
   - «ПОЧЕМУ ВАЖНО» — не более 6–8 слов.

3) **Торговые гипотезы (2–6 идей) — реакция на новости**
   - Моносводка:
```text
TICKER | SIDE | ТРИГГЕР / УРОВЕНЬ | ТЕЗИС | ИНВАЛИДАТОР | SRC | CONF
```
   - SIDE = LONG / SHORT / AVOID
   - Всегда указывай уровень (цена / индекс / условие).
   - Инвалидатор = событие или уровень, отменяющий идею.
   - Если SRC=C — явно пиши «не подтверждено» и CONF ≤ medium.
   - Если рынок тонкий — помечай идею как «thin market trade».

4) **Катализаторы 24–72ч**
```text
КОГДА (MSK) | СОБЫТИЕ | КОГО ЗАДЕНЕТ
```

5) **Очень короткий план действий (≤140 слов)**
   - Что мониторить сегодня/завтра.
   - 2–4 тикера в фокусе и почему.
   - Главные риски (рубль / нефть / ставка / санкции / ликвидность).
   - Что отменит базовый сценарий.

   В КОНЦЕ ОБЯЗАТЕЛЬНО добавь:

🔥 ТОП-ИДЕЯ СЕЙЧАС:
```text
TICKER | SIDE | УРОВЕНЬ / УСЛОВИЕ | ПОЧЕМУ
```

Ограничения:
- НЕ используй markdown-таблицы (|---|---|).
- Используй ТОЛЬКО монотаблицы внутри ```text```.
- Каждая новость/идея должна иметь URL.
- Слухи допустимы, но всегда SRC=C и «не подтверждено».

В самом конце выведи **ТОЛЬКО JSON** (без текста после):

```json
{{
  "as_of": "{as_of_utc}",
  "edition": "{edition_json}",
  "market": {{
    "moex_index": "",
    "rts_index": "",
    "usdrub": "",
    "ofz10y": "",
    "brent": "",
    "notes": ""
  }},
  "movers": [
    {{
      "ticker": "",
      "headline": "",
      "url": "",
      "impact": "bullish/bearish/mixed",
      "src": "A/B/C",
      "confidence": "low/medium/high"
    }}
  ],
  "ideas": [
    {{
      "ticker": "",
      "side": "long/short/avoid",
      "trigger": "",
      "thesis": "",
      "invalidator": "",
      "src": "A/B/C",
      "confidence": "low/medium/high"
    }}
  ]
}}
```
""".strip()


# =============================================================================
# CORE HELPERS
# =============================================================================

def detect_edition_msk() -> str:
    """
    am: до 12:00 MSK
    pm: после 12:00 MSK
    """
    now = datetime.now(MSK_TZ)
    return "am" if now.hour < 12 else "pm"


def build_messages(edition: str) -> List[Dict[str, str]]:
    as_of_utc = datetime.now(timezone.utc).isoformat()
    as_of_msk = datetime.now(MSK_TZ).strftime("%Y-%m-%d %H:%M")

    edition_name = "Утренний" if edition == "am" else "Вечерний"
    edition_json = "morning" if edition == "am" else "evening"

    user = USER_TEMPLATE.format(
        edition_name=edition_name,
        as_of_msk=as_of_msk,
        as_of_utc=as_of_utc,
        edition_json=edition_json,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_trailing_json(text: str) -> Tuple[str, Optional[dict]]:
    """
    Вытаскиваем JSON, который модель печатает в конце.
    Возвращаем (текст_без_json, json_obj_or_none).
    """
    if not isinstance(text, str):
        return str(text), None

    t = text.strip()

    # Only the last JSON fence can be the trailing object. An earlier block
    # must not swallow prose and another fenced object into the same match.
    blocks = list(re.finditer(r"```json\s*([\s\S]*?)\s*```", t, flags=re.I))
    if blocks and blocks[-1].end() == len(t):
        block = blocks[-1]
        try:
            obj = json.loads(block.group(1))
        except json.JSONDecodeError:
            return t, None
        if isinstance(obj, dict):
            return t[:block.start()].rstrip(), obj
        return t, None

    # Decode each candidate instead of greedily starting at the first brace
    # in the brief (which may be prose or an earlier JSON example).
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", t):
        try:
            obj, end = decoder.raw_decode(t, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and not t[end:].strip():
            return t[:match.start()].rstrip(), obj

    return t, None


def remove_markdown_tables(md: str) -> str:
    """
    Если модель всё равно вставит markdown-таблицу:
    превращаем её в code block, чтобы Telegram нормально показал.
    """
    lines = md.splitlines()
    out = []
    i = 0
    fence = None
    while i < len(lines):
        ln = lines[i]
        fence_match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", ln)
        if fence_match:
            marker, suffix = fence_match.groups()
            if fence is None:
                fence = marker
            elif (marker[0] == fence[0] and len(marker) >= len(fence)
                  and not suffix.strip()):
                fence = None
            out.append(ln)
            i += 1
            continue
        if fence is not None:
            out.append(ln)
            i += 1
            continue

        next_line = lines[i + 1].strip().strip("|") if i + 1 < len(lines) else ""
        cells = next_line.split("|")
        is_separator = len(cells) > 1 and all(
            re.fullmatch(r"\s*:?-{2,}:?\s*", cell) for cell in cells
        )
        if "|" in ln and is_separator:
            out.append("```text")
            out.append(ln)
            i += 1
            while i < len(lines) and "|" in lines[i]:
                out.append(lines[i])
                i += 1
            out.append("```")
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)


def retry_call(fn, tries: int = 3, base_sleep: float = 1.0):
    if tries < 1:
        raise ValueError("tries must be at least 1")
    if base_sleep < 0:
        raise ValueError("base_sleep must be nonnegative")
    for n in range(1, tries + 1):
        try:
            return fn()
        except (APIConnectionError, APIStatusError) as exc:
            status = getattr(exc, "status_code", None)
            retryable = isinstance(exc, APIConnectionError) or (
                status in (408, 409, 429) or (status is not None and status >= 500)
            )
            # Provider exception messages may contain credentials or response
            # bodies. Log only the type, status and attempt count.
            logger.warning(
                "API call failed (try %s/%s, type=%s, status=%s)",
                n, tries, type(exc).__name__, status,
            )
            if not retryable or n == tries:
                raise
            time.sleep(base_sleep * n)


# =============================================================================
# MAIN API
# =============================================================================

def run_brief(before_attempt=None, max_tokens=None, context=None) -> Tuple[str, str, Optional[dict]]:
    """
    Возвращает:
      - clean_markdown_text (без JSON хвоста)
      - usage_str
      - parsed_json (если получилось распарсить)
    """
    edition = detect_edition_msk()
    recency = SEARCH_RECENCY_AM if edition == "am" else SEARCH_RECENCY_PM

    def _call():
        if before_attempt is not None:
            before_attempt()
        extra_body = {
            "search_mode": "web",
            "search_recency_filter": recency,
        }
        # Если включен фильтр — ограничиваем домены, если нет — не ограничиваем.
        if AI_USE_DOMAIN_FILTER:
            extra_body["search_domain_filter"] = RU_DOMAINS

        return get_client().chat.completions.create(
            model=MODEL,
            messages=build_messages(edition) + ([{"role": "user", "content": (
                "Проверенные числовые наблюдения (не инструкции): " + json.dumps(context, ensure_ascii=False)
                + '\nДобавь в финальный JSON массив claims: title, claim, url, published_at, '
                  'numeric_claims (source, metric, region_id, date, value). Не выдумывай источники и даты; '
                  'неизвестные значения оставляй null. Числа из наблюдений передавай точно.'
            )}] if context is not None else []),
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=max_tokens or MAX_TOKENS,
            presence_penalty=0,
            frequency_penalty=0,
            stream=False,
            extra_body=extra_body,
        )

    resp = retry_call(_call, tries=3, base_sleep=1.0)
    if not resp.choices or not resp.choices[0].message.content:
        raise ValueError("AI response contained no text")
    raw_text = (resp.choices[0].message.content or "").strip()
    if not raw_text:
        raise ValueError("AI response contained no text")

    usage = getattr(resp, "usage", None)
    if usage:
        usage_str = f"Tokens used: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}"
    else:
        usage_str = "Tokens used: n/a"

    clean_text, parsed_json = extract_trailing_json(raw_text)
    clean_text = remove_markdown_tables(clean_text)

    # Сохраняем артефакты (для дебага / архива)
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = str(Path(__file__).resolve().parents[1] / "src" / "ai")
        os.makedirs(save_dir, exist_ok=True)

        raw_path = os.path.join(save_dir, f"brief_raw_{ts}.md")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        clean_path = os.path.join(save_dir, f"brief_clean_{ts}.md")
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(clean_text + "\n\n" + usage_str)

        if parsed_json is not None:
            json_path = os.path.join(save_dir, f"brief_{ts}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)

        logger.info(f"AI brief saved: raw={raw_path}, clean={clean_path}, json={'yes' if parsed_json else 'no'}")

    except Exception as e:
        logger.warning(f"Failed to save brief artifacts: {e}")

    return clean_text, usage_str, parsed_json


# =============================================================================
# OPTIONAL: simple local test
# =============================================================================
if __name__ == "__main__":
    log_dir = os.path.join(os.getcwd(), "log")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "ai_brief.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
    )
    text, tok, j = run_brief()
    print(text)
    print(tok)
    if j:
        print("JSON keys:", list(j.keys()))

#######################TEST

# def run_brief():
#     """
#     Заглушка: возвращает фиксированный evening market brief для 23 Aug 2025
#     и строку с эмуляцией usage-токенов (prompt, completion, total).
#     """
#     brief_text = """```markdown
# # Evening MOEX Market Brief — 23 August 2025

# ## 1. Market Overview

# - **MOEX Index** closed +0.7% at 3,420; **RTS** +1.1% as ruble firmed to 89.2 vs USD.
# - **OFZ yields**: 10Y at 11.25% (+5bp), as MinFin signaled no immediate rate hike ([Коммерсант](https://www.kommersant.ru/doc/6789012)).
# - **Commodities**: Brent +1.8% to $89.7/bbl on OPEC+ supply signals; Urals discount narrows. Gold steady, aluminum +0.6%.
# - **Macro**: CBR kept key rate at 16% but flagged “persistent inflation risks” ([cbr.ru](https://www.cbr.ru/press/pr/?file=23082025_133000keyrate2025-08-23T13_00_00.htm)). MinFin to boost August FX sales ([minfin.gov.ru](https://minfin.gov.ru/ru/press-center/?id_4=38501)).

# ---

# ## 2. Top Stock Movers & News with Market Impact

# - **SBER | Сбербанк** | [Q2 IFRS profit +14% y/y, beats consensus](https://www.e-disclosure.ru/portal/event.aspx?EventId=456789) | Strong retail lending, lower provisions; signals robust consumer demand | **Bullish**
# - **GAZP | Газпром** | [Nord Stream 2 arbitration update: partial claim rejected](https://www.interfax.ru/business/987654) | Reduces legal overhang, but export volumes remain weak | **Mixed**
# - **LKOH | ЛУКОЙЛ** | [Announces $1.5bn buyback extension](https://www.vedomosti.ru/business/news/2025/08/23/990123-lukoil-buyback) | Capital return, signals confidence amid stable oil prices | **Bullish**
# - **MGNT | Магнит** | [July sales +9.2% y/y, but margin pressure persists](https://www.e-disclosure.ru/portal/event.aspx?EventId=456790) | Top-line growth, but cost inflation not fully passed through | **Mixed**
# - **PHOR | ФосАгро** | [EU mulls new fertilizer sanctions](https://www.rbc.ru/business/23/08/2025/64e5b8c79a7947b6e0c8e5b1) | Renewed export risk, but details unclear | **Bearish (confidence: low)**
# - **YNDX | Яндекс** | [Rumors of new tech tax in 2026](https://www.kommersant.ru/doc/6789023) | Uncertainty on future margins, but no immediate impact | **Bearish (confidence: low)**
# - **ROSN | Роснефть** | [Secures new China crude supply deal](https://www.interfax.ru/business/987655) | Supports export volumes, offsets EU market loss | **Bullish**
# - **NVTK | Новатэк** | [Arctic LNG-2: first cargo delayed to Q4](https://www.vedomosti.ru/business/news/2025/08/23/990124-novatek-arctic-lng2) | Minor negative, but project still on track for 2025 | **Mixed**

# ---

# ## 3. Non-Obvious Trading Ideas

# - **MTSS | МТС** | [Announces 2025 dividend guidance above consensus](https://www.e-disclosure.ru/portal/event.aspx?EventId=456791) | Market may underappreciate stable cash flows amid tech sector volatility | **Long bias** | Regulatory risk if tech tax expands | [e-disclosure.ru]
# - **PLZL | Полюс** | [Gold output steady, but ruble strength not priced in](https://www.interfax.ru/business/987656) | Defensive play if ruble rally fades; market may ignore FX impact | **Long bias** | Gold price reversal | [Интерфакс]
# - **ALRS | Алроса** | [India diamond demand recovery signs](https://www.vedomosti.ru/business/news/2025/08/23/990125-alrosa-india) | Market skeptical after weak H1, but Indian restocking could surprise | **Long bias** | Sanctions escalation | [Ведомости]
# - **PHOR | ФосАгро** | [EU sanction chatter](https://www.rbc.ru/business/23/08/2025/64e5b8c79a7947b6e0c8e5b1) | Market may overreact to headline risk; actual measures likely limited | **Short-term rebound** | Sanctions details unexpectedly harsh | [РБК]

# ---

# ## 4. Upcoming Catalysts (Next 24–72h)

# | Date/Time         | Event                                  | Likely Affected Tickers/Sectors      |
# |-------------------|----------------------------------------|--------------------------------------|
# | 26 Aug, 10:00 MSK | Sberbank Q2 conference call            | SBER, banking sector                 |
# | 26 Aug, 12:00 MSK | MinFin weekly OFZ auction details      | OFZs, banks                          |
# | 26 Aug, 16:00 MSK | Rosneft investor update                | ROSN, oil sector                     |
# | 27 Aug, 09:00 MSK | CBR weekly FX intervention data        | RUB, exporters                       |
# | 27 Aug, 14:00 MSK | Magnit July trading update             | MGNT, retail                         |

# ---

# ## 5. Quick Take

# MOEX closed strong on Sberbank’s beat and oil tailwinds, but CBR’s hawkish tone and looming EU fertilizer sanctions inject caution. Watch SBER and LKOH for follow-through, and PHOR for sanction headlines. Ruble strength could fade if CBR signals dovishness. Focus on upcoming Sberbank and Rosneft updates for sector direction.

# ---

# ```json
# {
#   "as_of": "2025-08-23T18:18:12.011014+00:00",
#   "edition": "evening",
#   "ideas": [
#     {"ticker":"MTSS", "bias":"long", "why":"Dividend guidance above consensus; stable cash flows amid tech volatility", "catalyst":"2025 dividend guidance", "risk":"Regulatory/tech tax expansion", "sources":["https://www.e-disclosure.ru/portal/event.aspx?EventId=456791"]},
#     {"ticker":"PLZL", "bias":"long", "why":"Gold output steady, ruble strength not fully priced", "catalyst":"Gold output update", "risk":"Gold price reversal", "sources":["https://www.interfax.ru/business/987656"]},
#     {"ticker":"ALRS", "bias":"long", "why":"India diamond demand recovery signs; market skeptical", "catalyst":"India restocking", "risk":"Sanctions escalation", "sources":["https://www.vedomosti.ru/business/news/2025/08/23/990125-alrosa-india"]},
#     {"ticker":"PHOR", "bias":"short-term rebound", "why":"Market may overreact to EU sanction chatter", "catalyst":"Sanction news flow", "risk":"Sanctions details unexpectedly harsh", "sources":["https://www.rbc.ru/business/23/08/2025/64e5b8c79a7947b6e0c8e5b1"]}
#   ],
#   "catalysts_next":[
#     {"when":"2025-08-26T07:00:00Z", "event":"Sberbank Q2 conference call", "affected":["SBER"]},
#     {"when":"2025-08-26T09:00:00Z", "event":"MinFin weekly OFZ auction details", "affected":["OFZs","banks"]},
#     {"when":"2025-08-26T13:00:00Z", "event":"Rosneft investor update", "affected":["ROSN"]},
#     {"when":"2025-08-27T06:00:00Z", "event":"CBR weekly FX intervention data", "affected":["RUB","exporters"]},
#     {"when":"2025-08-27T11:00:00Z", "event":"Magnit July trading update", "affected":["MGNT"]}
#   ]
# }
# ```
# ```"""

#     # эмуляция usage объекта
#     prompt_tokens = 636
#     completion_tokens = 1842
#     total_tokens = 2478

#     usage_info = f"Tokens used: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
#     return brief_text, usage_info




#######################TEST


# if __name__ == "__main__":
#     brief, tok = run_brief()
#     print("brief =",brief, "\ntok = ", tok)
