"""Small, deterministic daily-series forecasts on the Russian working calendar."""
import pandas as pd

from .analysis import SERIES_KEYS, evaluate_forecasts, normalize_observations
from .russian_calendar import calendar as working_calendar


def working_dates(start, end):
    calendar = working_calendar()
    return [day for day in pd.date_range(start, end) if calendar.is_working_day(day.date())]


def forecast_daily(history, *, created_at=None, origin_date=None, kind="issued", horizon=None):
    frame = normalize_observations(history)
    created = pd.Timestamp(created_at or pd.Timestamp.now(tz="UTC"))
    created = created.tz_localize("UTC") if created.tzinfo is None else created.tz_convert("UTC")
    if frame.empty or frame["count"].notna().sum() == 0:
        return []
    if len(frame[SERIES_KEYS].drop_duplicates()) != 1 or not frame["frequency"].eq("daily").all():
        raise ValueError("One daily series required")
    if kind not in {"issued", "backtest"}:
        raise ValueError("Invalid forecast kind")
    origin = pd.Timestamp(origin_date) if origin_date is not None else frame.loc[frame["count"].notna(), "date"].max()
    origin = (origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")).normalize()
    if kind == "issued" and created.normalize() < origin:
        raise ValueError("Creation precedes training observations")
    frame = frame[frame["date"] <= origin]
    if kind == "issued":
        frame = frame[frame["fetched_at"].isna() | (frame["fetched_at"] <= created)]
    metadata = {key: frame.iloc[-1][key] for key in SERIES_KEYS} if not frame.empty else {}
    if not metadata:
        return []
    from .market_analysis import forecast
    meta = {"frequency": "daily", "unit": metadata["unit"],
            "calendar": "calendar" if metadata["query"] in {"Индекс RUONIA", "1 мес", "3 мес", "6 мес"} else "russia_working"}
    if metadata["unit"] in {"индекс", "млрд руб"}:
        meta["value_bounds"] = [0, float("inf")]
    rows = [{"date": row["date"].date().isoformat(), "value": row["count"],
             **{key: row[key] for key in ('period_end', 'published_at', 'definition')
                if key in row and pd.notna(row[key])}}
            for row in frame.to_dict("records") if pd.notna(row["count"])]
    return [{**metadata, **p, "metric": metadata["query"], "origin_date": p["origin"],
             "target_date": p["target"], "baseline_prediction": p["baseline"],
             "kind": kind, "training_points": len(rows)}
            for p in forecast(rows, meta, created_at=created.isoformat(), horizon=horizon)]


def backtest_daily(history, max_origins=12, *, horizon=None):
    frame = normalize_observations(history)
    if not 1 <= max_origins <= 52:
        raise ValueError("Invalid origin limit")
    origins = frame["date"].iloc[65:-20:5].tolist()[-max_origins:]
    predictions = []
    for origin in origins:
        predictions.extend(forecast_daily(frame[frame["date"] <= origin], origin_date=origin,
                                         created_at=origin + pd.Timedelta(hours=23), kind="backtest", horizon=horizon))
    return {"forecasts": predictions, "metrics": evaluate_forecasts(predictions, frame, issued_only=False)}
