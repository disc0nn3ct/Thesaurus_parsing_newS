# ThesaurusV2

import pandas as pd
from datetime import datetime, timedelta, time
import os
import requests
from requests.exceptions import RequestException
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging
import matplotlib.dates as mdates
import re

import sys


import holidays
from datetime import date

# Настройка логгера
log_dir = os.path.join(os.getcwd(), "log")
os.makedirs(log_dir, exist_ok=True)

# Полный путь к лог-файлу
log_file = os.path.join(log_dir, "ruonia_log.txt")

# Настройка логгера
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)


def is_russian_workday(check_date=None):
    if check_date is None:
        check_date = date.today()

    ru_holidays = holidays.Russia()




    return check_date not in ru_holidays # выходные сб вс и    
    # return check_date.weekday() < 5 and check_date not in ru_holidays # выходные сб вс и

#### 
# ======================= WORDSTAT: кризисные ключи -> графики -> рассылка =======================
import os
import re
import json
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

import numpy as np
from sklearn.linear_model import Ridge

load_dotenv()

# из .env:
#  - WORDSTAT_OAUTH  — OAuth-токен, который ты получил по инструкции ...
#  - WORDSTAT_CLIENT_ID — ClientId приложения (с той же страницы)
WORDSTAT_OAUTH = os.getenv("WORDSTAT_OAUTH")
WORDSTAT_CLIENT_ID = os.getenv("WORDSTAT_CLIENT_ID")

# базовый URL API из официальной доки
WORDSTAT_API = "https://api.wordstat.yandex.net"
WORDSTAT_REGION_RUSSIA = 225  # Россия
WORDSTAT_SAVE_DIR = os.path.join("src", "wordstat")
WORDSTAT_RAW_DIR = os.path.join(WORDSTAT_SAVE_DIR, "raw")  # сюда будем складывать все ответы API
WORDSTAT_MAPS_DIR = os.path.join(WORDSTAT_SAVE_DIR, "maps")
os.makedirs(WORDSTAT_MAPS_DIR, exist_ok=True)

os.makedirs(WORDSTAT_SAVE_DIR, exist_ok=True)
os.makedirs(WORDSTAT_RAW_DIR, exist_ok=True)
111
# клиент Perplexity из твоего ai.py
from functions.ai import client as ai_client
from functions.wordstat_api import _wordstat_post

import hashlib

def slugify_phrase(s: str, max_len: int = 60) -> str:
    """
    Делает безопасный slug, сохраняя кириллицу (через \w с UNICODE),
    плюс добавляет короткий хэш для уникальности.
    """
    s = (s or "").strip().lower()
    base = re.sub(r"[^\w.-]+", "_", s, flags=re.UNICODE).strip("_")
    if not base:
        base = "phrase"
    if len(base) > max_len:
        base = base[:max_len].rstrip("_")
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}"

def _series_metrics(y: np.ndarray, window_short: int = 8, window_long: int = 52) -> dict:
    """
    Метрики по времянке:
      - last, mean_long, std_long
      - z_last (насколько текущий уровень аномальный)
      - mom_short: изменение краткосрочного среднего относительно долгосрочного
      - vol_short: волатильность на коротком окне
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 10:
        return {}

    wL = min(window_long, len(y))
    wS = min(window_short, len(y))

    last = float(y[-1])
    mean_long = float(np.mean(y[-wL:]))
    std_long = float(np.std(y[-wL:]) + 1e-9)

    mean_short = float(np.mean(y[-wS:]))
    z_last = float((last - mean_long) / std_long)
    mom_short = float((mean_short - mean_long) / (mean_long + 1e-9))  # относительное смещение
    vol_short = float(np.std(y[-wS:]) / (mean_short + 1e-9))          # относительная волатильность

    return {
        "last": last,
        "mean_long": mean_long,
        "std_long": std_long,
        "z_last": z_last,
        "mom_short": mom_short,
        "vol_short": vol_short,
    }


def _forecast_ridge_log1p(y: np.ndarray, horizon: int = 4, train_weeks: int = 156) -> dict:
    """
    Прогноз на horizon недель:
      - ridge регрессия по времени на log1p(y)
      - возвращаем прогнозы (yhat) и доверительный коридор (примерный) из std остатков
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 30:
        return {"yhat": [], "lo": [], "hi": []}

    # ограничим обучение последними train_weeks (примерно 3 года)
    start = max(0, n - train_weeks)
    y_train = y[start:]
    t = np.arange(len(y_train)).reshape(-1, 1)

    # log1p для устойчивости к пикам
    y_log = np.log1p(np.clip(y_train, 0, None))

    model = Ridge(alpha=1.0)
    model.fit(t, y_log)

    # остатки -> оценка шума
    y_log_pred = model.predict(t)
    resid = y_log - y_log_pred
    resid_std = float(np.std(resid) + 1e-9)

    # прогноз
    t_future = np.arange(len(y_train), len(y_train) + horizon).reshape(-1, 1)
    fut_log = model.predict(t_future)

    # грубый 80% коридор (можно заменить на 95%)
    z = 1.28
    lo_log = fut_log - z * resid_std
    hi_log = fut_log + z * resid_std

    yhat = np.expm1(fut_log)
    lo = np.expm1(lo_log)
    hi = np.expm1(hi_log)

    # неотрицательно
    yhat = np.clip(yhat, 0, None)
    lo = np.clip(lo, 0, None)
    hi = np.clip(hi, 0, None)

    return {
        "yhat": [float(v) for v in yhat],
        "lo": [float(v) for v in lo],
        "hi": [float(v) for v in hi],
        "resid_std_log": resid_std,
    }


def interpret_signal(metrics: dict) -> str:
    """
    Короткая интерпретация по метрикам.
    """
    if not metrics:
        return "Недостаточно данных для уверенной интерпретации."

    z = metrics["z_last"]
    mom = metrics["mom_short"]
    vol = metrics["vol_short"]

    parts = []

    # уровень
    if z >= 2.0:
        parts.append("аномально высокий уровень интереса (возможна паника/событие)")
    elif z <= -2.0:
        parts.append("аномально низкий уровень интереса (затухание)")
    elif z >= 1.0:
        parts.append("выше обычного")
    elif z <= -1.0:
        parts.append("ниже обычного")
    else:
        parts.append("в пределах нормы")

    # импульс
    if mom >= 0.25:
        parts.append("устойчивый рост последних недель")
    elif mom <= -0.25:
        parts.append("устойчивое снижение последних недель")

    # волатильность
    if vol >= 0.35:
        parts.append("высокая волатильность (новостной шум/резкие всплески)")

    return "; ".join(parts).capitalize() + "."


def make_phrase_report(phrase: str, df: pd.DataFrame, horizon: int = 4) -> dict:
    """
    Возвращает словарь с метриками + прогнозом + текстовой интерпретацией.
    """
    y = df["count"].astype(float).to_numpy()
    metrics = _series_metrics(y)
    fc = _forecast_ridge_log1p(y, horizon=horizon)

    # базовые числа прогноза
    yhat = fc.get("yhat", [])
    lo = fc.get("lo", [])
    hi = fc.get("hi", [])

    report = {
        "phrase": phrase,
        "metrics": metrics,
        "forecast": fc,
        "interpretation": interpret_signal(metrics),
    }

    # удобные поля
    if yhat:
        report["forecast_next"] = yhat[0]
        report["forecast_horizon"] = yhat
        report["forecast_ci_next"] = (lo[0], hi[0]) if lo and hi else None
    else:
        report["forecast_next"] = None
        report["forecast_horizon"] = []

    return report


def build_and_save_charts(df_all: pd.DataFrame, out_dir: str = WORDSTAT_SAVE_DIR) -> tuple[list[str], list[dict]]:
    """
    По каждой фразе рисуем PNG: фактические значения + прогноз на 4 недели.
    Возвращаем (files, reports).
    """
    saved = []
    reports = []

    for phrase, df in df_all.groupby("phrase"):
        if df.empty:
            continue

        df = df.sort_values("date").reset_index(drop=True)
        rep = make_phrase_report(phrase, df, horizon=4)
        reports.append(rep)

        # подготовка прогноза к рисованию
        yhat = rep.get("forecast_horizon", [])
        lohi = rep.get("forecast", {})
        lo = lohi.get("lo", [])
        hi = lohi.get("hi", [])

        last_date = df["date"].iloc[-1]
        future_dates = [last_date + pd.Timedelta(days=7*(i+1)) for i in range(len(yhat))]

        plt.figure(figsize=(12, 5))
        plt.plot(df["date"], df["count"], linewidth=2, label="Факт")

        if yhat:
            plt.plot(future_dates, yhat, linewidth=2, linestyle="--", label="Прогноз (4 нед.)")
            # коридор
            if lo and hi:
                plt.fill_between(future_dates, lo, hi, alpha=0.2, label="Коридор (≈80%)")

        title = f"Wordstat weekly: {phrase}"
        plt.title(title)
        plt.xlabel("Дата (недели)")
        plt.ylabel("Число запросов")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # safe_phrase = re.sub(r"[^a-zA-Z0-9_.-]+", "_", phrase.strip())
        # fname = f"wordstat_{safe_phrase}.png"
        safe_phrase = slugify_phrase(phrase)
        fname = f"wordstat_{safe_phrase}.png"
        fpath = os.path.join(out_dir, fname)
        base, ext = os.path.splitext(fpath)
        k = 1
        while os.path.exists(fpath):
            k += 1
            fpath = f"{base}_{k}{ext}"

        plt.savefig(fpath)
        plt.close()
        saved.append(fpath)
        logger.info(f"💾 Сохранён график: {fpath}")

    return saved, reports


def _last_sunday_on_or_before(d: date) -> date:
    # Monday=0..Sunday=6 -> до ближайшего прошедшего воскресенья
    return d - timedelta(days=(d.weekday() + 1) % 7)


def _first_monday_on_or_after(d: date) -> date:
    return d + timedelta(days=(7 - d.weekday()) % 7)


def _prepare_week_bounds():
    """
    fromDate: первый понедельник >= 2018-01-01
    toDate: последнее воскресенье не позднее сегодня-2
    """
    today = datetime.today().date()
    to_date = today - timedelta(days=2)
    to_date = _last_sunday_on_or_before(to_date)

    from_date = date(2018, 1, 1)
    from_date = _first_monday_on_or_after(from_date)
    return from_date, to_date


from functions.regions_digest import send_incidents_daily_regions_digest






def plot_daily_heatmap(df: pd.DataFrame, phrase: str, out_path: str, top_regions: int = 25):
    """
    Heatmap: строки=regionId, столбцы=даты, значение=count.
    """
    sdf = df[df["phrase"] == phrase].copy()
    if sdf.empty:
        logger.warning(f"plot_daily_heatmap: пусто для '{phrase}'")
        return None

    # упорядочим регионы по сумме интереса за период
    order = (sdf.groupby("regionId")["count"].sum()
               .sort_values(ascending=False)
               .head(top_regions)
               .index.tolist())
    sdf = sdf[sdf["regionId"].isin(order)]

    pivot = (sdf.pivot_table(index="regionId", columns="date", values="count", aggfunc="sum")
               .fillna(0))

    plt.figure(figsize=(14, max(6, 0.28 * len(pivot.index))))
    plt.imshow(pivot.values, aspect="auto")
    plt.title(f"Wordstat daily heatmap (top {top_regions} regions): {phrase}")
    plt.xlabel("Дата")
    plt.ylabel("Регион (regionId)")

    cols = list(pivot.columns)
    step = max(1, len(cols) // 10)
    xticks = list(range(0, len(cols), step))
    plt.xticks(xticks, [pd.to_datetime(cols[i]).strftime("%m-%d") for i in xticks], rotation=45)

    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info(f"💾 heatmap сохранён: {out_path}")
    return out_path



def get_crisis_keywords_via_perplexity() -> list[str]:
    """
    Берём 8 русскоязычных ключевых фраз про кризисы/проблемы/стрессы населения
    через того же клиента, что и run_brief() (ai.py).

    Формат ответа — JSON-массив строк (ровно 8).
    """
    sys_prompt = (
    "Ты — аналитический ИИ-агент, формирующий поисковые ИНДИКАТОРЫ общественной тревоги "
    "и экономической ситуации для анализа динамики в Яндекс Вордстате. "
    "Сформируй ровно 8 русскоязычных поисковых запросов, которые люди РЕАЛЬНО вводят "
    "в Яндекс или Google, и по динамике которых (рост, пики, затухание) можно делать выводы "
    "о кризисах, нестабильности, ухудшении или стабилизации ситуации в стране и мире. "
    "Запросы должны быть короткими (1–4 слова), в формате телеграфного поискового ввода. "
    "Допускается использование ТОЛЬКО оператора '+' для фиксации служебных слов "
    "(например: 'работа +дома', 'деньги +на карте'). "
    "Запрещено использовать любые другие операторы Вордстата "
    "(-, !, кавычки, скобки, |). "
    "Каждый запрос должен быть МАКРО-ИНДИКАТОРОМ одной из сфер: "
    "1) цены и инфляция, "
    "2) работа и доходы, "
    "3) банки и сбережения, "
    "4) валюта и курс, "
    "5) долги и кредиты, "
    "6) доступность базовых услуг, "
    "7) лекарства и здоровье, "
    "8) неопределённость и страх будущего. "
    "Используй только устойчивые массовые запросы, существующие много лет "
    "и реагирующие на кризисы. "
    "Ответ — ТОЛЬКО JSON-массив из ровно 8 строк, без пояснений."
)

    resp = ai_client.chat.completions.create(
        model="sonar-pro",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Дай массив из 8 фраз. Только JSON, без текста до/после."}
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=400,
        stream=False,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        arr = json.loads(raw)
        if not isinstance(arr, list) or len(arr) != 8:
            raise ValueError("Ожидался массив из 8 элементов")
        return [str(x).strip() for x in arr]
    except Exception as e:
        logger.warning(f"Ключевые слова от Perplexity не распарсились: {e}. Использую дефолтный набор.")
        return [
            "обвал", "рост цен", "дефолт", "безработица",
            "кредитные каникулы", "закрытие банков", "дефицит лекарств", "отключение электричества"
        ]


def fetch_wordstat_dynamics(phrase: str, regions=None, devices=None) -> pd.DataFrame:
    """
    Достаёт weekly-динамику counts/share по фразе с 2018-01-01 до сегодня-2 (последнее воскресенье).
    Использует метод /v1/dynamics.
    Дополнительно сохраняет *сырой* ответ API в JSON-файл в src/wordstat/raw.
    """
    if regions is None:
        regions = [WORDSTAT_REGION_RUSSIA]
    if devices is None:
        devices = ["all"]

    from_date, to_date = _prepare_week_bounds()
    payload = {
        "phrase": phrase,  # в этом методе допустим только оператор '+', но простая фраза тоже ок
        "period": "weekly",
        "fromDate": from_date.strftime("%Y-%m-%d"),
        "toDate": to_date.strftime("%Y-%m-%d"),
        "regions": regions,
        "devices": devices,
    }
    data = _wordstat_post("/v1/dynamics", payload)

    # === Сохраняем сырой ответ в JSON ===
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # safe_phrase = re.sub(r"[^a-zA-Z0-9_.-]+", "_", phrase.strip())
        # raw_fname = f"wordstat_dynamics_{safe_phrase}_{ts}.json"
        safe_phrase = slugify_phrase(phrase)
        raw_fname = f"wordstat_dynamics_{safe_phrase}_{ts}.json"

        raw_fpath = os.path.join(WORDSTAT_RAW_DIR, raw_fname)
        with open(raw_fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Сырой ответ Wordstat по '{phrase}' сохранён в {raw_fpath}")
    except Exception as e:
        logger.exception(f"⚠️ Не удалось сохранить сырой ответ Wordstat по '{phrase}': {e}")

    # === Преобразуем в DataFrame ===
    dyn = data.get("dynamics", [])
    df = pd.DataFrame(dyn)
    if df.empty:
        df = pd.DataFrame(columns=["date", "count", "share"])
    else:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    df["phrase"] = phrase
    return df


def send_wordstat_digest(tg_client, recipients):
    """
    Главный раннер:
      1) берём 8 ключей у Perplexity,
      2) тянем динамику из Wordstat (и сохраняем сырые JSON-ответы),
      3) строим графики (+прогноз/аналитика),
      4) рассылаем в TG: график + интерпретация + прогнозные значения.

    При любой ошибке Wordstat просто шлём текстовое уведомление, не валим всю программу.
    """
    # 1) ключевые фразы
    try:
        keywords = get_crisis_keywords_via_perplexity()
    except Exception as e:
        logger.exception(f"Perplexity не вернул ключи: {e}")
        keywords = [
            "обвал", "рост цен", "дефолт", "безработица",
            "кредитные каникулы", "закрытие банков", "дефицит лекарств", "отключение электричества"
        ]

    logger.info(f"Wordstat: ключевые фразы: {keywords}")

    # 2) динамика по каждому слову
    frames = []
    errors = []
    for kw in keywords:
        try:
            df_kw = fetch_wordstat_dynamics(kw)
            if not df_kw.empty:
                frames.append(df_kw)
                logger.info(f"Wordstat: по фразе '{kw}' получено {len(df_kw)} точек")
            else:
                logger.warning(f"Wordstat: по фразе '{kw}' пришёл пустой ответ (dynamics=[])")
            time.sleep(0.3)  # чуть поддросим, чтобы не забанили по RPS
        except Exception as e:
            logger.exception(f"Ошибка Wordstat по '{kw}': {e}")
            errors.append((kw, str(e)))

    # Если НИ по одной фразе данных нет — шлём предупреждение и выходим из функции
    if not frames:
        msg = (
            "⚠️ Не удалось получить данные из Ворда ни по одной фразе.\n"
            "Возможные причины:\n"
            "• неверный WORDSTAT_OAUTH или WORDSTAT_CLIENT_ID;\n"
            "• приложению не выдан доступ к API Ворда;\n"
            "• исчерпана квота запросов или превышен лимит RPS;\n"
            "• временная ошибка сервиса.\n\n"
            "Подробности смотри в логах (ищи сообщения с префиксом 'Wordstat')."
        )
        for chat_id in recipients:
            try:
                tg_client.send_message(chat_id, msg)
            except Exception as e:
                logger.exception(f"⚠️ Ошибка отправки уведомления в {chat_id}: {e}")
        return

    # 3) строим графики + получаем отчёты (интерпретация и прогноз)
    all_df = pd.concat(frames, ignore_index=True)
    files, reports = build_and_save_charts(all_df)  # важно: функция должна возвращать (files, reports)

    if not files:
        msg = (
            "⚠️ Wordstat вернул данные, но не удалось построить ни одного графика "
            "(возможно, все DataFrame оказались пустыми после фильтрации)."
        )
        for chat_id in recipients:
            try:
                tg_client.send_message(chat_id, msg)
            except Exception as e:
                logger.exception(f"⚠️ Ошибка отправки уведомления в {chat_id}: {e}")
        return

    # 4) рассылка в Telegram: заголовок + по каждому индикатору картинка и аналитика
    header = (
        "📊 Еженедельные тренды поисковых запросов (Ворд)\n"
        "Период: с 2018-01-01 по последние доступные недели.\n"
        "Источник: API /v1/dynamics.\n\n"
        "🧠 Аналитика: отклонение от нормы (z-score), импульс (8н vs 52н), прогноз на 4 недели."
    )

    for chat_id in recipients:
        try:
            tg_client.send_message(chat_id, header)

            for rep in reports:
                phrase = rep.get("phrase", "")
                metrics = rep.get("metrics", {}) or {}
                interp = rep.get("interpretation", "") or ""

                fnext = rep.get("forecast_next", None)
                fci = rep.get("forecast_ci_next", None)
                fh = rep.get("forecast_horizon", []) or []

                last = metrics.get("last", None)
                z = metrics.get("z_last", None)
                mom = metrics.get("mom_short", None)
                vol = metrics.get("vol_short", None)

                # найдём файл графика по safe_phrase (учитываем суффиксы _2 и т.п.)
                # safe_phrase = re.sub(r"[^a-zA-Z0-9_.-]+", "_", phrase.strip())
                # candidates = [fp for fp in files if f"wordstat_{safe_phrase}" in os.path.basename(fp)]
                safe_phrase = slugify_phrase(phrase)
                candidates = [fp for fp in files if f"wordstat_{safe_phrase}" in os.path.basename(fp)]
                chart_path = candidates[-1] if candidates else None

                lines = [f"🔎 {phrase}"]

                if interp:
                    lines.append(f"• Интерпретация: {interp}")

                if last is not None:
                    lines.append(f"• Текущее значение (посл. неделя): {int(round(last))}")

                if z is not None:
                    lines.append(f"• Отклонение от нормы (z-score): {z:.2f}")

                if mom is not None:
                    lines.append(f"• Импульс (8н vs 52н): {mom*100:.0f}%")

                if vol is not None:
                    lines.append(f"• Волатильность (8н): {vol*100:.0f}%")

                if fnext is not None:
                    if fci and isinstance(fci, (list, tuple)) and len(fci) == 2:
                        lo, hi = fci
                        lines.append(
                            f"• Прогноз на след. неделю: {int(round(fnext))} "
                            f"(≈{int(round(lo))}…{int(round(hi))})"
                        )
                    else:
                        lines.append(f"• Прогноз на след. неделю: {int(round(fnext))}")

                if fh:
                    fh_int = [int(round(v)) for v in fh]
                    lines.append(f"• Прогноз 4 недели: {fh_int}")

                caption = "\n".join(lines)

                if chart_path:
                    tg_client.send_photo(chat_id, photo=chart_path, caption=caption)
                else:
                    tg_client.send_message(chat_id, caption)

            # при желании — коротко сообщаем об ошибках по отдельным ключам
            if errors:
                err_lines = ["⚠️ Ошибки по отдельным фразам:"]
                for kw, err in errors[:8]:
                    err_lines.append(f"• {kw}: {err[:200]}")
                tg_client.send_message(chat_id, "\n".join(err_lines))

            logger.info(f"✅ Wordstat-дайджест отправлен в {chat_id}")

        except Exception as e:
            logger.exception(f"⚠️ Ошибка отправки в {chat_id}: {e}")





# ======================= /WORDSTAT =======================






# Проверяет за какой промежутек нужен запрос данных и загружает актуальную информацию на сегодня, 
def check_if_need_new_rec(FILENAME="ruonia_data.xlsx"):
    try:
        today = datetime.today().date()
        end_date = today.strftime("%d.%m.%Y")
        start_date = "11.01.2010"
        from_dt = datetime.strptime(start_date, "%d.%m.%Y").strftime("%m/%d/%Y")
        to_dt = today.strftime("%m/%d/%Y")

        if os.path.exists(FILENAME):
            try:
                local_df = pd.read_excel(FILENAME)
                local_df["Дата"] = pd.to_datetime(local_df["Дата"], dayfirst=True)
                last_date = local_df["Дата"].max().date()
                from_date = local_df["Дата"].min().date()

                logger.info(f"Файл найден. С {from_date} по {last_date}")

                if not is_russian_workday():
                    # logger.info("Сегодня выходной или праздник — обновление не требуется.")
                    logger.info("Праздник — обновление не требуется.")
                    return 0

                if last_date.strftime('%d.%m.%Y') == today.strftime('%d.%m.%Y') or (
                    last_date.strftime('%d.%m.%Y') == (today - timedelta(days=1)).strftime('%d.%m.%Y') and
                    datetime.now().time() < time(14, 0)
                ):
                    logger.info("Данные уже актуальны. Обновление не требуется.")
                    return 0
                else:
                    logger.info("Обнаружены новые данные. Загружаем обновление.")
            except Exception as e:
                logger.warning(f"Ошибка чтения файла: {e}")
                logger.info("Выполняем загрузку заново.")
        else:
            logger.info(f"Файл {FILENAME} не найден. Загружаем с {start_date} по {end_date}.")

        url = f"https://cbr.ru/Queries/UniDbQuery/DownloadExcel/125022?Posted=True&From={start_date}&To={end_date}&I1=true&M1=true&M3=true&M6=true&FromDate={from_dt}&ToDate={to_dt}"
        logger.info(f"Запрос по ссылке: {url}")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса данных: {e}")
            return -1

        with open(FILENAME, "wb") as f:
            f.write(response.content)
            logger.info(f"Файл успешно сохранён: {FILENAME}")

        return 1

    except RequestException as e:
        logger.error(f"Ошибка загрузки с сайта ЦБ: {e}")
        return -1
    except Exception as e:
        logger.exception(f"Необработанная ошибка: {e}")
        return -2


# Построение графиков, похоже на скользящие средние
def analitics(FILENAME="ruonia_data.xlsx"):
    today_str = datetime.today().strftime("%Y-%m-%d")
    base_filename = f"ruonia_trend_{today_str}"
    ext = ".png"

    output_dir = os.path.join(os.getcwd(), "src")
    os.makedirs(output_dir, exist_ok=True)

    # Генерация имени для полного графика
    version = 1
    output_path = os.path.join(output_dir, base_filename + ext)
    while os.path.exists(output_path):
        version += 1
        output_path = os.path.join(output_dir, f"{base_filename}_v{version}{ext}")

    try:
        # Загружаем и обрабатываем данные
        df = pd.read_excel(FILENAME)
        df = df.rename(columns={
            "Индекс": "RUONIA",
            "1 месяц": "1 мес",
            "3 месяца": "3 мес",
            "6 месяцев": "6 мес"
        })
        df["Дата"] = pd.to_datetime(df["Дата"], dayfirst=True)
        df = df.dropna(subset=["RUONIA", "1 мес", "3 мес", "6 мес"])
        df = df.sort_values("Дата")

        # --- 📈 График со всеми данными ---
        plt.figure(figsize=(14, 7))
        plt.plot(df["Дата"], df["RUONIA"], label="RUONIA (overnight)", linewidth=2)
        plt.plot(df["Дата"], df["1 мес"], label="RUONIA 1 мес", linestyle="--")
        plt.plot(df["Дата"], df["3 мес"], label="RUONIA 3 мес", linestyle="-.")
        plt.plot(df["Дата"], df["6 мес"], label="RUONIA 6 мес", linestyle=":")

        plt.title(f"Динамика индекса RUONIA и срочных ставок до {today_str}", fontsize=14)
        plt.xlabel("Дата")
        plt.ylabel("Ставка (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(f"📈 График (все данные) сохранён: {output_path}")

        # --- 📉 График за последние 90 дней ---
        short_df = df[df["Дата"] >= (datetime.today() - timedelta(days=90))]

        plt.figure(figsize=(14, 7))
        plt.plot(short_df["Дата"], short_df["1 мес"], label="RUONIA 1 мес", linestyle="--")
        plt.plot(short_df["Дата"], short_df["3 мес"], label="RUONIA 3 мес", linestyle="-.")
        plt.plot(short_df["Дата"], short_df["6 мес"], label="RUONIA 6 мес", linestyle=":")

        plt.title(f"RUONIA (последние 90 дней) до {today_str}", fontsize=14)
        plt.xlabel("Дата")
        plt.ylabel("Ставка (%)")
        plt.legend()
        plt.grid(True)

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        plt.xticks(rotation=90)

        plt.tight_layout()

        # Сохраняем второй файл с _last90
        short_filename = f"{base_filename}_last90"
        short_output_path = os.path.join(output_dir, short_filename + ext)
        version = 1
        while os.path.exists(short_output_path):
            version += 1
            short_output_path = os.path.join(output_dir, f"{short_filename}_v{version}{ext}")

        plt.savefig(short_output_path)
        plt.close()

        logger.info(f"📉 График (последние 90 дней) сохранён: {short_output_path}")
        return output_path, short_output_path

    except Exception as e:
        logger.exception(f"❌ Ошибка при построении графиков: {e}")
        return None
    
#проветси анализ РУОНИИ 
def make_analyze_ruonia(filepath="ruonia_data.xlsx"):
    try:
        df = pd.read_excel(filepath)
        df = df.rename(columns={
            "Индекс": "RUONIA",
            "1 месяц": "1 мес",
            "3 месяца": "3 мес",
            "6 месяцев": "6 мес"
        })
        df["Дата"] = pd.to_datetime(df["Дата"], dayfirst=True)
        df = df.sort_values("Дата")

        last_30 = df.tail(30)
        last_15 = last_30.tail(15)
        last_10 = last_30.tail(10)

        latest_date = last_10["Дата"].iloc[-1].strftime("%d.%m.%Y")
        previous_date = last_10["Дата"].iloc[-2].strftime("%d.%m.%Y")

        indicators = ["RUONIA", "1 мес", "3 мес", "6 мес"]
        full_text = f"📅 Последняя дата данных: {latest_date}\n"

        for col in indicators:
            latest = last_10[col].iloc[-1]
            previous = last_10[col].iloc[-2]

            delta_1 = latest - previous
            delta_10 = last_10[col].iloc[-1] - last_10[col].iloc[0]
            delta_15 = last_15[col].iloc[-1] - last_15[col].iloc[0]
            delta_30 = last_30[col].iloc[-1] - last_30[col].iloc[0]

            mean_10 = last_10[col].mean()
            mean_15 = last_15[col].mean()
            mean_30 = last_30[col].mean()

            if delta_10 > 0 and delta_15 > 0 and delta_30 > 0:
                trend = "📈 плавный восходящий тренд"
            elif delta_10 < 0 and delta_15 < 0 and delta_30 < 0:
                trend = "📉 стабильное снижение"
            else:
                trend = "📊 неопределённое поведение"

            full_text += (
                f"\n📌 **{col}**\n"
                f"• Сегодня: {latest:.4f}\n"
                f"• Вчера ({previous_date}): {previous:.4f}\n"
                f"• Δ за день: {delta_1:+.4f}\n"
                f"• Среднее за 10 дней: {mean_10:.4f}\n"
                f"• Среднее за 15 дней: {mean_15:.4f}\n"
                f"• Среднее за 30 дней: {mean_30:.4f}\n"
                f"• Рост за 10 дней: {delta_10:+.4f}\n"
                f"• Рост за 15 дней: {delta_15:+.4f}\n"
                f"• Рост за 30 дней: {delta_30:+.4f}\n"
                f"• Тренд: {trend}\n"
            )

        logger.info("🧾 Анализ RUONIA успешно выполнен.")
        logger.debug(f"\n{full_text}")
        return full_text

    except Exception as e:
        logger.exception(f"❌ Ошибка в аналитике RUONIA: {e}")
        return None


def send_info_ruonia(client, recipients):
    folder_path = os.path.join(os.getcwd(), "src")
    base_name = "ruonia_trend_"
    short_base_name = "ruonia_trend_"
    short_suffix = "_last90"
    extension = ".png"

    # Получаем список подходящих файлов
    matching_files = [
        f for f in os.listdir(folder_path)
        if f.startswith(base_name) and f.endswith(extension) and short_suffix not in f
    ] if os.path.exists(folder_path) else []

    # Получаем список коротких графиков (last90)
    matching_short_files = [
        f for f in os.listdir(folder_path)
        if f.startswith(short_base_name) and short_suffix in f and f.endswith(extension)
    ] if os.path.exists(folder_path) else []

    # if matching_files:
    #     matching_files.sort(reverse=True)
    #     latest_file = os.path.join(folder_path, matching_files[0])
    #     logger.info(f"📂 Найден последний график: {latest_file}")
    # else:
    #     logger.warning("📂 График не найден. Генерируем с помощью analitics()...")
    #     latest_file = analitics()

    #Заменил на всегда генерацию
    logger.warning("📂 Всегда!!!!. Генерируем с помощью analitics()...")
    # latest_file, latest_short_file = analitics()

    # latest_file = analitics()
    result = analitics()
    if not result:
        logger.error("❌ Не удалось создать графики.")

    latest_file, latest_short_file = result

    if not os.path.exists(latest_file):
        logger.error("❌ Файл графика не найден после генерации.")


    # Поиск соответствующего short-файла
    latest_short_file = None
    if matching_short_files:
        matching_short_files.sort(reverse=True)
        latest_short_file = os.path.join(folder_path, matching_short_files[0])
        logger.info(f"📂 Найден короткий график (90 дней): {latest_short_file}")





    if not latest_file or not os.path.exists(latest_file):
        logger.error("❌ Не удалось найти или создать файл графика RUONIA.")
        return

    analysis = make_analyze_ruonia()

    for chat_id in recipients:
        try:
            logger.info(f"📤 Отправка графика и анализа в чат: {chat_id}")
            client.send_photo(
                chat_id,
                photo=latest_file,
                caption="📈 График RUONIA за всё время до " + datetime.today().strftime("%Y-%m-%d")
            )

            # Отправка дополнительного графика (last90), если найден
            if latest_short_file and os.path.exists(latest_short_file):
                client.send_photo(
                    chat_id,
                    photo=latest_short_file,
                    caption="📉 RUONIA за последние 90 дней"
                )

            if analysis:
                client.send_message(chat_id, analysis)
                logger.info(f"✅ Анализ успешно отправлен в {chat_id}")
            else:
                logger.warning(f"⚠️ Анализ не сгенерирован — сообщение не отправлено в {chat_id}")
        except Exception as e:
            logger.exception(f"⚠️ Ошибка при отправке в {chat_id}: {e}")


# https://cbr.ru/Queries/UniDbQuery/DownloadExcel/125022?Posted=True&From=11.01.2010&To=30.04.2025&I1=true&M1=true&M3=true&M6=true&FromDate=01%2F11%2F2010&ToDate=04%2F30%2F2025

###########################################################################################################################AI

# ---------- основная функция с контекст-менеджером ----------

from functions.ai import run_brief


def send_ai(client, recipients):
    import os, json
    from datetime import datetime

    TELEGRAM_LIMIT = 4000

    def normalize_code_fences(text: str) -> str:
        if text.count("```") % 2 == 1:
            text += "\n```"
        return text

    def split_chunks(text: str, limit: int):
        parts, cur = [], ""
        for para in text.split("\n\n"):
            if len(cur) + len(para) < limit:
                cur += para + "\n\n"
            else:
                parts.append(cur.strip())
                cur = para + "\n\n"
        if cur.strip():
            parts.append(cur.strip())
        return parts

    try:
        text, usage, data = run_brief()
    except Exception as e:
        for r in recipients:
            client.send_message(r, f"❌ Ошибка AI-brief: {e}")
        return

    text = normalize_code_fences(text)
    chunks = split_chunks(text, TELEGRAM_LIMIT)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("src", "ai")
    os.makedirs(out_dir, exist_ok=True)

    md_path = os.path.join(out_dir, f"ai_brief_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text + "\n\n" + usage)

    json_path = None
    if isinstance(data, dict):
        json_path = os.path.join(out_dir, f"ai_brief_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    for r in recipients:
        for i, c in enumerate(chunks, 1):
            prefix = f"Часть {i}/{len(chunks)}\n\n" if len(chunks) > 1 else ""
            # client.send_message(r, prefix + c, parse_mode="markdown", disable_web_page_preview=True)
            from pyrogram.enums import ParseMode
            client.send_message(r, prefix + c, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

            # client.send_message(r, prefix + c, disable_web_page_preview=True)


        if usage:
            client.send_message(r, usage)

        client.send_document(r, md_path, caption="📄 AI-brief (.md)")
        if json_path:
            client.send_document(r, json_path, caption="🧾 AI-brief (.json)")



###########################################################################################################################AI


#################################### Вернуть ######
# check_if_need_new_rec()

# analitics()  # Либо переделать 
#################################### Вернуть ######


###### Проверка обновления ####
# import subprocess
# import os
# import sys

# def check_git_update(commit_file="log/current_commit.txt"):
#     try:
#         # Убедимся, что папка log существует
#         os.makedirs(os.path.dirname(commit_file), exist_ok=True)

#         # Получаем текущий коммит с origin
#         subprocess.run(["git", "fetch"], check=True)
#         new_commit = subprocess.check_output(
#             ["git", "rev-parse", "origin/main"], text=True
#         ).strip()

#         # Если файла нет — создаём и записываем текущий коммит
#         if not os.path.exists(commit_file):
#             with open(commit_file, "w") as f:
#                 f.write(new_commit)
#             logger.info(f"📄 Файл {commit_file} создан. Установлен коммит: {new_commit}")
#             return None  # Первый запуск — обновление не требуется

#         # Считываем сохранённый коммит
#         with open(commit_file, "r") as f:
#             last_commit = f.read().strip()

#         if new_commit != last_commit:
#             logger.info(f"🔄 Обнаружен новый коммит: {new_commit}")
#             return new_commit
#         else:
#             logger.info("✅ Версия актуальна. Обновление не требуется.")
#             return None

#     except Exception as e:
#         logger.exception("❌ Ошибка при проверке обновления Git:")
#         return None


# def update_and_restart(new_commit, commit_file="log/current_commit.txt"):
#     try:
#         subprocess.run(["git", "pull"], check=True)

#         with open(commit_file, "w") as f:
#             f.write(new_commit)

#         logger.info("♻️ Проект обновлён. Перезапускаем...")
#         os.execv(sys.executable, ['python'] + sys.argv)

#     except Exception as e:
#         logger.exception("❌ Ошибка при обновлении и перезапуске:")



# commit_file = "log/current_commit.txt"
# new_commit = check_git_update(commit_file)
# if new_commit:
#     update_and_restart(new_commit, commit_file)



################################################################################ВЕРНУТЬ
from functions.auto_update import check_and_restart_if_updated
check_and_restart_if_updated()
#################################################################################ВЕРНУТЬ
# ######

load_dotenv()  

api_hash = os.getenv('api_hash')
for_whom = os.getenv('for_whom')
api_id = os.getenv('api_id')
bot_token = os.getenv('bot_token')

recipients_raw = os.getenv("for_whom_list", "")
recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
if not recipients:
    raise ValueError("❌ Нет получателей. Убедись, что for_whom_list задан в .env")




from pyrogram import Client, idle

#################  вернуть 
client = Client(name='me_client', api_id=api_id, api_hash=api_hash, bot_token = bot_token )
# Запуск клиента
client.start()

        


check_if_need_new_rec()
send_info_ruonia(client, recipients)

time.sleep(10)
send_ai(client, recipients)

time.sleep(10)
send_wordstat_digest(client, recipients)



time.sleep(10)
send_incidents_daily_regions_digest(client, recipients, phrases=["пожар", "взрыв", "бпла"], top_n_regions=25)


# idle()

# Завершение сессии
client.stop()

#################### вернуть 







