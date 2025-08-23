# russia_trader_brief_timebased.py
# -*- coding: utf-8 -*-

import os
from datetime import datetime, timezone, timedelta
from openai import OpenAI
from dotenv import load_dotenv

# ===== 1) ENV =====
load_dotenv()
AI_API_KEY = os.getenv("ai_api_key")
BASE_URL = os.getenv("base_url")

if not AI_API_KEY:
    raise RuntimeError("AI_API_KEY is missing in .env")

client = OpenAI(api_key=AI_API_KEY, base_url=BASE_URL)

# ===== 2) PROMPT (только Россия) =====
SYSTEM_PROMPT = """You are a **Russian Equity Trader’s Assistant**. Your task is to generate **morning and evening actionable briefs** about the MOEX equity market.
Be concise but **highly informative**. Prioritize **fresh Russian news (≤24h for morning, ≤12h for evening)** that can move individual stocks or sectors.
Use reputable sources (moex.com, cbr.ru, minfin.gov.ru, e-disclosure.ru, Интерфакс, РБК, Ведомости, Коммерсант) and **always cite them inline with links**. Avoid generic headlines — focus on what actually changes positioning.

Think like a trader:
- Which MOEX stocks/sectors are **positively/negatively impacted**?
- What is **non-obvious** or under-reacted by the market?
- What are the **near-term catalysts** (today / this week)?
- What are the **risks or stop-loss triggers**?
- How could this fit into a **buy/sell/avoid** tactical decision?

Do not give financial advice; instead, present **scenarios** and **decision-relevant context**.
If confidence is low, mark it. Keep total words ≤500.
"""

USER_TEMPLATE = """Produce today’s **{edition_name} MOEX market brief** for a professional equity trader.

**Deliverables (strict order):**

1. **Market Overview** (2–4 bullets)
   - Moves in RTS/MOEX index, ruble, OFZ yields, commodities (oil, gas, metals).
   - Key macro/CBR/MinFin news.

2. **Top Stock Movers & News with Market Impact (MOEX only)**
   - 5–8 bullets. Each = **Ticker | Company | Headline (linked) | Why it matters | Expected direction (bullish/bearish/mixed)**.

3. **Non-Obvious Trading Ideas (2–4 names)**
   `Ticker | Name | News trigger (with link) | Why market may under-react | Trade angle (long/short bias) | Key risk`

4. **Upcoming Catalysts (Next 24–72h)**
    Calendar with **date/time** and likely affected tickers/sectors.

5. **Quick Take (≤100 words)**
   - Base case, upside/downside risks, what to monitor into next session.

**Constraints & Style:**
- Output in **Markdown**.
- Every news/idea must include at least one **source link**.
- If confidence is low, mark `(confidence: low)`.
- End with compact JSON for machine parsing:
```json
{{
  "as_of": "{as_of}",
  "edition": "{edition_json}",
  "ideas": [
    {{"ticker":"", "bias":"long/short", "why":"", "catalyst":"", "risk":"", "sources":[""]}}
  ],
  "catalysts_next":[
    {{"when":"", "event":"", "affected":[""]}}
  ]
}}
```
"""

RU_DOMAINS = [
    "moex.com","cbr.ru","minfin.gov.ru","e-disclosure.ru","interfax.ru","rbc.ru",
    "vedomosti.ru","kommersant.ru","spimex.com"
]


def detect_edition() -> str:
    now = datetime.now()  # локальное время сервера/машины
    hour = now.hour
    if hour < 12:
        return "am"
    else:
        return "pm"


def build_messages(edition: str):
    as_of = datetime.now(timezone.utc).isoformat()
    edition_name = "Morning" if edition == "am" else "Evening"
    edition_json = "morning" if edition == "am" else "evening"
    user = USER_TEMPLATE.format(as_of=as_of, edition_name=edition_name, edition_json=edition_json)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user}
    ]

################################
def run_brief() -> str:
    edition = detect_edition()
    recency = "day"
    resp = client.chat.completions.create(
        model="sonar-pro",
        messages=build_messages(edition),
        temperature=0.2,
        top_p=0.9,
        max_tokens=2500,
        presence_penalty=0,
        frequency_penalty=0,
        stream=False,
        extra_body={
            "search_mode": "web",
            "search_domain_filter": RU_DOMAINS,
            "search_recency_filter": recency
        }
    )
    text = resp.choices[0].message.content

    # токены
    usage = getattr(resp, "usage", None)
    if usage:
        print(f"Tokens used: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
    
    print("\n\n\n", text)

    # === Сохраняем text в файл ===
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("src")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"brief_{ts}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Бриф сохранён в {file_path}")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения брифа: {e}")

    return text, str(f"Tokens used: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")


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
