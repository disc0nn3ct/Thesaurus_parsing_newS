"""Deterministic report analysis. Missing observations are never filled with zero.

Functions accept dataframes or row dictionaries; network probes are opt-in.
Forecast evaluation distinguishes issued forecasts from retrospective backtests.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import math
import re
import socket
import ssl
import unicodedata
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SERIES_KEYS = ["source", "unit", "query", "region_id", "device", "frequency"]
TRACKING_KEYS = {"fbclid", "gclid", "yclid", "ymclid", "mc_cid", "mc_eid"}


def json_safe(value):
    """Return plain JSON types, replacing non-finite numbers with null."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _timestamp(value=None) -> pd.Timestamp:
    result = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Timestamp must not be missing")
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def normalize_observations(observations) -> pd.DataFrame:
    """Use canonical columns while retaining metadata and actual missing values."""
    frame = pd.DataFrame(observations).copy()
    aliases = {"phrase": "query", "query_text": "query", "metric": "query", "regionId": "region_id",
               "value": "count", "observed_date": "date", "period_start": "date",
               "collected_at": "fetched_at", "retrieved_at": "fetched_at", "period": "frequency"}
    frame = frame.rename(columns={old: new for old, new in aliases.items()
                                  if old in frame and new not in frame})
    if frame.empty:
        for key in ["date", "count", *SERIES_KEYS, "fetched_at"]:
            if key not in frame:
                frame[key] = pd.Series(dtype="object")
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        return frame
    missing = {"date", "count"} - set(frame)
    if missing:
        raise ValueError(f"Missing observation columns: {', '.join(sorted(missing))}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce", format="mixed").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Observation dates must be valid")
    frame["count"] = pd.to_numeric(frame["count"], errors="coerce")
    frame["count"] = frame["count"].where(np.isfinite(frame["count"]))
    for key, default in (("source", "wordstat"), ("unit", "count"), ("query", ""), ("region_id", 225), ("device", "all"), ("frequency", "daily")):
        if key not in frame:
            frame[key] = default
        frame[key] = frame[key].fillna(default)
    frame.loc[frame["source"].eq("wordstat") & (frame["count"] < 0), "count"] = np.nan
    if "quality" in frame:
        frame.loc[frame["quality"].isin(["missing", "error", "invalid"]), "count"] = np.nan
    frame["region_id"] = frame["region_id"].astype(str)
    if "fetched_at" not in frame:
        frame["fetched_at"] = pd.NaT
    frame["fetched_at"] = pd.to_datetime(frame["fetched_at"], utc=True, errors="coerce", format="mixed")
    keys = [*SERIES_KEYS, "date"]
    return (frame.sort_values("fetched_at", na_position="first", kind="stable")
            .drop_duplicates(keys, keep="last").sort_values("date").reset_index(drop=True))


def filter_observations(observations, start_date=None, end_date=None, queries=None,
                        region_ids=None, sources=None) -> pd.DataFrame:
    frame = normalize_observations(observations)
    if start_date is not None:
        frame = frame[frame["date"] >= _timestamp(start_date).normalize()]
    if end_date is not None:
        frame = frame[frame["date"] <= _timestamp(end_date).normalize()]
    for column, values in (("query", queries), ("region_id", region_ids), ("source", sources)):
        if values is not None:
            values = [values] if isinstance(values, (str, int)) else list(values)
            frame = frame[frame[column].astype(str).isin([str(value) for value in values])]
    return frame.copy()


def quality_summary(observations, start_date, end_date, queries=None, region_ids=None,
                    as_of=None, frequency="daily") -> dict:
    """Coverage uses the requested grid, including wholly absent query/region pairs."""
    start, end = _timestamp(start_date).normalize(), _timestamp(end_date).normalize()
    if end < start or frequency not in {"daily", "weekly"}:
        raise ValueError("Invalid date bounds or frequency")
    frame = filter_observations(observations, start, end, queries, region_ids)
    frame = frame[frame["frequency"].eq(frequency)]
    query_values = list(dict.fromkeys(queries)) if queries is not None and not isinstance(queries, str) else ([queries] if queries is not None else frame["query"].unique().tolist())
    region_values = list(dict.fromkeys(map(str, region_ids))) if region_ids is not None and not isinstance(region_ids, (str, int)) else ([str(region_ids)] if region_ids is not None else frame["region_id"].unique().tolist())
    periods = (end - start).days // (7 if frequency == "weekly" else 1) + 1
    providers = max(1, len(frame[["source", "unit", "device"]].drop_duplicates()))
    expected = periods * len(query_values) * len(region_values) * providers
    valid = frame[frame["count"].notna()]
    observed = len(valid.drop_duplicates(["date", *SERIES_KEYS]))
    newest = valid["date"].max() if not valid.empty else pd.NaT
    fetched = frame["fetched_at"].max() if not frame.empty else pd.NaT
    now = _timestamp(as_of)
    return {
        "start_date": start.date().isoformat(), "end_date": end.date().isoformat(),
        "expected_points": expected, "observed_points": observed,
        "missing_points": max(0, expected - observed),
        "coverage": observed / expected if expected else None,
        "latest_observation": newest.date().isoformat() if pd.notna(newest) else None,
        "last_fetched_at": fetched.isoformat() if pd.notna(fetched) else None,
        "freshness_hours": max(0.0, (now - fetched).total_seconds() / 3600) if pd.notna(fetched) else None,
        "observation_lag_days": max(0, (now.normalize() - newest).days) if pd.notna(newest) else None,
        "missing_is_zero": False,
    }


def compare_periods(observations, current_start, current_end, previous_start, previous_end,
                    queries=None, region_ids=None) -> dict:
    """Compare identical source/unit/query/region/day-offset cells in equal periods."""
    cs, ce = _timestamp(current_start).normalize(), _timestamp(current_end).normalize()
    ps, pe = _timestamp(previous_start).normalize(), _timestamp(previous_end).normalize()
    if ce < cs or pe < ps or ce - cs != pe - ps:
        raise ValueError("Comparison periods must have equal nonnegative lengths")
    current = filter_observations(observations, cs, ce, queries, region_ids)
    previous = filter_observations(observations, ps, pe, queries, region_ids)
    current["offset"] = (current["date"] - cs).dt.days
    previous["offset"] = (previous["date"] - ps).dt.days
    keys = [*SERIES_KEYS, "offset"]
    paired = current.dropna(subset=["count"]).merge(previous.dropna(subset=["count"]), on=keys, suffixes=("_current", "_previous"))
    groups = []
    for identity, group in paired.groupby(SERIES_KEYS, dropna=False):
        before, after = float(group["count_previous"].sum()), float(group["count_current"].sum())
        groups.append({**dict(zip(SERIES_KEYS, identity)), "metric": identity[2], "matched_points": len(group),
                       "current_total": after, "previous_total": before, "difference": after - before,
                       "percent_change": (after / before - 1) * 100 if before else None})
    union = pd.concat([current[keys], previous[keys]], ignore_index=True).drop_duplicates()
    return {"groups": groups, "matched_points": len(paired), "available_union_points": len(union),
            "comparison_coverage": len(paired) / len(union) if len(union) else None,
            "excluded_points": len(union) - len(paired), "comparable_cells_only": True}


def normalize_url(url: str) -> str:
    parts = urlsplit(str(url).strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("Only HTTP(S) URLs without credentials are supported")
    host = parts.hostname.encode("idna").decode("ascii").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port and port != (443 if parts.scheme.lower() == "https" else 80):
        host += f":{port}"
    query = sorted((key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
                   if not key.lower().startswith("utm_") and key.lower() not in TRACKING_KEYS)
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", urlencode(query), ""))


def normalize_title(title: str) -> str:
    title = unicodedata.normalize("NFKC", str(title)).casefold().replace("ё", "е")
    return re.sub(r"[\W_]+", " ", title, flags=re.UNICODE).strip()


def deduplicate_news(records) -> list[dict]:
    """Keep the latest revision of a URL; cluster same-day normalized titles."""
    items, url_index, title_index = [], {}, {}
    records = sorted((dict(record) for record in records), key=lambda row: str(row.get("fetched_at") or row.get("published_at") or ""))
    for record in records:
        raw_url = record.get("url") or record.get("link")
        try:
            url = normalize_url(raw_url) if raw_url else None
        except (ValueError, UnicodeError):
            url = None
        title = str(record.get("title") or "").strip()
        day = str(record.get("published_at") or record.get("date") or "")[:10]
        title_key = (normalize_title(title), day)
        match = url_index.get(url) if url else None
        if match is None and title_key[0] and day:
            match = title_index.get(title_key)
        if match is None:
            item = {**record, "url": url or "", "title": title, "normalized_title": title_key[0],
                    "sources": [record["source"]] if record.get("source") else [],
                    "alternate_urls": [url] if url else [], "corrections": []}
            match = len(items)
            items.append(item)
        else:
            item = items[match]
            same_url = url is not None and url in item["alternate_urls"]
            if same_url and any(str(record.get(key) or "") != str(item.get(key) or "") for key in ("title", "summary", "content", "claim") if key in record):
                item["corrections"].append({key: item.get(key) for key in ("title", "summary", "content", "claim", "fetched_at")})
                for key, value in record.items():
                    if key not in {"sources", "alternate_urls", "corrections"}:
                        item[key] = value
                item["url"] = url
                item["normalized_title"] = title_key[0]
            if url and url not in item["alternate_urls"]:
                item["alternate_urls"].append(url)
            if record.get("source") and record["source"] not in item["sources"]:
                item["sources"].append(record["source"])
        if url:
            url_index[url] = match
        if title_key[0] and day:
            title_index[title_key] = match
    for item in items:
        identity = item.get("url") or (item["normalized_title"] + "|" + str(item.get("published_at") or "")[:10])
        item["fingerprint"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        content = {key: item.get(key) for key in ("title", "summary", "content", "claim")}
        item["content_fingerprint"] = hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return json_safe(items)


def classify_news_changes(current, previous) -> dict:
    """Classify new/repeated/corrected news against a saved report snapshot."""
    old = {item["fingerprint"]: item for item in deduplicate_news(previous)}
    result = {"new": [], "repeated": [], "corrected": []}
    for item in deduplicate_news(current):
        prior = old.get(item["fingerprint"])
        if prior is None:
            result["new"].append(item)
        elif prior["content_fingerprint"] == item["content_fingerprint"]:
            result["repeated"].append(item)
        else:
            result["corrected"].append({**item, "previous_content": {key: prior.get(key) for key in ("title", "summary", "content", "claim")}})
    return result


@dataclass(frozen=True)
class AlertConfig:
    percent_threshold: float = 50.0
    z_threshold: float = 2.5
    baseline_weeks: int = 8
    minimum_coverage: float = 0.90
    minimum_baseline: float = 20.0
    cooldown_hours: float = 24.0
    direction: str = "increase"

    def __post_init__(self):
        if (not math.isfinite(self.percent_threshold) or self.percent_threshold < 0
                or not math.isfinite(self.z_threshold) or self.z_threshold < 0
                or self.baseline_weeks < 2 or not 0 < self.minimum_coverage <= 1
                or not math.isfinite(self.minimum_baseline) or self.minimum_baseline < 0
                or not math.isfinite(self.cooldown_hours) or self.cooldown_hours < 0
                or self.direction not in {"increase", "both"}):
            raise ValueError("Invalid alert configuration")


def cooldown_allowed(last_sent_at, now=None, cooldown_hours=24.0) -> bool:
    return last_sent_at is None or (_timestamp(now) - _timestamp(last_sent_at)).total_seconds() >= cooldown_hours * 3600


def evaluate_alerts(observations, end_date=None, config: AlertConfig | None = None) -> list[dict]:
    config = config or AlertConfig()
    frame = normalize_observations(observations)
    if frame.empty:
        return []
    end = _timestamp(end_date).normalize() if end_date is not None else frame["date"].max()
    results = []
    for identity, group in frame.groupby(SERIES_KEYS, dropna=False):
        weekly = set(group["frequency"].dropna()) <= {"weekly", "week"}
        windows = []
        for index in range(config.baseline_weeks + 1):
            window_end = end - pd.Timedelta(days=7 * index)
            window_start = window_end if weekly else window_end - pd.Timedelta(days=6)
            values = group[(group["date"] >= window_start) & (group["date"] <= window_end)]["count"].dropna()
            coverage = len(values) / (1 if weekly else 7)
            windows.append((float(values.sum()) if len(values) else None, min(1.0, coverage)))
        row = {**dict(zip(SERIES_KEYS, identity)), "metric": identity[2], "end_date": end.date().isoformat(),
               "current_total": windows[0][0], "coverage": windows[0][1], "alert": False,
               "baseline_weeks": config.baseline_weeks, "reason": "insufficient_coverage"}
        if all(value is not None and coverage >= config.minimum_coverage for value, coverage in windows):
            baseline = np.array([value for value, _ in windows[1:]], dtype=float)
            mean, std = float(baseline.mean()), float(baseline.std(ddof=1))
            delta = windows[0][0] - mean
            percent = delta / mean * 100 if mean else (math.inf if delta > 0 else 0.0)
            z_score = delta / std if std > 1e-12 else (math.copysign(math.inf, delta) if delta else 0.0)
            significant = (abs(percent) >= config.percent_threshold and abs(z_score) >= config.z_threshold)
            fires = significant and mean >= config.minimum_baseline and (config.direction == "both" or delta > 0)
            row.update(baseline_mean=mean, baseline_std=std,
                       percent_change=percent if math.isfinite(percent) else None,
                       z_score=z_score if math.isfinite(z_score) else None,
                       constant_baseline=std <= 1e-12, zero_baseline=mean == 0,
                       alert=bool(fires), reason="insufficient_baseline" if mean < config.minimum_baseline else ("threshold_exceeded" if fires else "below_threshold"))
        results.append(row)
    return results


def _weekly_history(history) -> pd.DataFrame:
    frame = normalize_observations(history)
    if len(frame[SERIES_KEYS].drop_duplicates()) > 1:
        raise ValueError("A forecast must contain exactly one source/unit/query/region series")
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate weekly dates")
    if not frame.empty and ((frame["date"] - frame["date"].min()).dt.days % 7 != 0).any():
        raise ValueError("Forecast observations must follow a weekly calendar")
    return frame.sort_values("date").reset_index(drop=True)


def forecast_weekly(history, horizon=4, origin_date=None, created_at=None, min_train_weeks=30,
                    train_weeks=156, kind="issued") -> list[dict]:
    """Ridge/log1p forecast; absent weeks retain their time coordinates.

    Intervals are residual-based approximate 80% prediction intervals, not a
    calibrated guarantee. Only issued records saved before their target can be
    counted by evaluate_forecasts; backtests are explicitly labelled.
    """
    if horizon < 1 or min_train_weeks < 3 or train_weeks < min_train_weeks or kind not in {"issued", "backtest"}:
        raise ValueError("Invalid forecast configuration")
    frame = _weekly_history(history)
    if frame.empty:
        return []
    origin = _timestamp(origin_date).normalize() if origin_date is not None else frame["date"].max()
    created = _timestamp(created_at)
    frame = frame[(frame["date"] <= origin) & (frame["date"] > origin - pd.Timedelta(weeks=train_weeks))]
    if kind == "issued":
        # Revised/future-collected values cannot be used before they were available.
        frame = frame[frame["fetched_at"].isna() | (frame["fetched_at"] <= created)]
    valid = frame.dropna(subset=["count"])
    if len(valid) < min_train_weeks:
        return []
    full_weeks = (origin - frame["date"].min()).days // 7 + 1
    if len(valid) / full_weeks < 0.9 or valid["date"].max() != origin:
        return []
    if kind == "issued" and created.normalize() < origin:
        raise ValueError("Forecast creation cannot precede its training observations")
    anchor = valid["date"].min()
    t = ((valid["date"] - anchor).dt.days.to_numpy() / 7).reshape(-1, 1)
    y = np.log1p(valid["count"].to_numpy(dtype=float))
    model = Ridge(alpha=1.0).fit(t, y)
    # Preserve numerical tolerance for a constant series with an exact fit.
    sigma = max(float(np.std(y - model.predict(t), ddof=1)), 1e-9)
    latest_t = (origin - anchor).days / 7
    predictions = model.predict(np.arange(latest_t + 1, latest_t + horizon + 1).reshape(-1, 1))
    metadata = {key: valid.iloc[-1][key] for key in SERIES_KEYS}
    metadata.update(metric=metadata["query"], frequency="weekly")
    rows = []
    for step, predicted_log in enumerate(predictions, start=1):
        target = origin + pd.Timedelta(weeks=step)
        if kind == "issued" and target <= created:
            continue
        # A bounded exponent avoids returning non-JSON infinities for unstable fits.
        values = [float(max(0.0, np.expm1(np.clip(value, -700, 700)))) for value in
                  (predicted_log, predicted_log - 1.2815515655 * sigma, predicted_log + 1.2815515655 * sigma)]
        rows.append({**metadata, "origin_date": origin.date().isoformat(), "target_date": target.date().isoformat(),
                     "created_at": created.isoformat(), "prediction": values[0], "lower": values[1], "upper": values[2],
                     "baseline_prediction": float(valid.iloc[-1]["count"]), "horizon_weeks": step,
                     "baseline": float(valid.iloc[-1]["count"]), "horizon": step,
                     "model": "ridge_log1p", "kind": kind, "interval_level": 0.8,
                     "interval_method": "training_residuals_approximate", "training_points": len(valid)})
    return rows


def evaluate_forecasts(forecasts, actuals, issued_only=True) -> dict:
    predicted = pd.DataFrame(forecasts).copy()
    if predicted.empty:
        return {"n": 0, "mae": None, "wape": None, "interval_coverage": None, "baseline_mae": None, "baseline_wape": None}
    actual = normalize_observations(actuals).rename(columns={"date": "target_date", "count": "actual"})
    predicted = predicted.rename(columns={old: new for old, new in {"metric": "query", "baseline": "baseline_prediction"}.items() if old in predicted and new not in predicted})
    if "kind" not in predicted:
        predicted["kind"] = "issued"
    for key, default in (("device", "all"), ("frequency", "weekly")):
        if key not in predicted:
            predicted[key] = default
    if "unit" not in predicted:
        unit_keys = [key for key in SERIES_KEYS if key != "unit"]
        predicted["region_id"] = predicted["region_id"].astype(str)
        units = actual[[*unit_keys, "unit"]].drop_duplicates()
        if units.duplicated(unit_keys).any():
            raise ValueError("Ambiguous actual units")
        predicted = predicted.merge(units, on=unit_keys, how="left")
    required = {*SERIES_KEYS, "target_date", "origin_date", "created_at", "prediction", "lower", "upper", "baseline_prediction", "kind"}
    if required - set(predicted):
        raise ValueError("Forecasts must include provenance, dates, bounds and baseline")
    predicted["region_id"] = predicted["region_id"].astype(str)
    for key in ("target_date", "origin_date", "created_at"):
        predicted[key] = pd.to_datetime(predicted[key], utc=True, errors="coerce")
    mask = (predicted["origin_date"] < predicted["target_date"]) & (predicted["created_at"] < predicted["target_date"])
    if issued_only:
        mask &= predicted["kind"].eq("issued")
    predicted = predicted[mask].drop_duplicates([*SERIES_KEYS, "origin_date", "target_date", "kind"], keep="first")
    scored = predicted.merge(actual[[*SERIES_KEYS, "target_date", "actual"]], on=[*SERIES_KEYS, "target_date"])
    for key in ("prediction", "baseline_prediction", "actual", "lower", "upper"):
        scored[key] = pd.to_numeric(scored[key], errors="coerce")
    scored = scored[np.isfinite(scored[["prediction", "baseline_prediction", "actual", "lower", "upper"]]).all(axis=1)]
    if scored.empty:
        return {"n": 0, "mae": None, "wape": None, "interval_coverage": None, "baseline_mae": None, "baseline_wape": None}
    absolute = (scored["prediction"] - scored["actual"]).abs()
    baseline = (scored["baseline_prediction"] - scored["actual"]).abs()
    denominator = float(scored["actual"].abs().sum())
    return {"n": len(scored), "mae": float(absolute.mean()), "wape": float(absolute.sum() / denominator) if denominator else None,
            "interval_coverage": float(scored["actual"].between(scored["lower"], scored["upper"]).mean()),
            "baseline_mae": float(baseline.mean()), "baseline_wape": float(baseline.sum() / denominator) if denominator else None,
            "evaluation": "issued_before_target" if issued_only else "retrospective_backtest"}


def rolling_backtest(history, horizon=4, min_train_weeks=30, train_weeks=156, max_origins=52) -> dict:
    frame = _weekly_history(history)
    if max_origins < 1:
        raise ValueError("max_origins must be positive")
    rows = []
    origins = frame["date"].iloc[min_train_weeks - 1:-1].tolist()[-max_origins:]
    for origin in origins:
        rows.extend(forecast_weekly(frame[frame["date"] <= origin], horizon, origin,
                                   origin + pd.Timedelta(hours=23), min_train_weeks, train_weeks, kind="backtest"))
    return {"forecasts": rows, "metrics": evaluate_forecasts(rows, frame, issued_only=False),
            "note": "Rolling historical holdout; final historical revisions may differ from information available at the time."}


def verify_numeric_claims(claims, observations, tolerance=1e-6) -> list[dict]:
    """Verify structured numeric references; this does not verify causal prose."""
    frame = normalize_observations(observations)
    results = []
    for claim in claims:
        item = dict(claim) if isinstance(claim, dict) else {"text": str(claim)}
        verified, evidence, errors = [], [], []
        references = item.get("numbers") or []
        if not isinstance(references, list):
            references = []
            errors.append("numeric_references_must_be_a_list")
        for reference in references:
            if not isinstance(reference, dict) or not {"value", "metric", "query", "region_id", "start_date", "end_date", "unit"} <= set(reference):
                verified.append(False)
                errors.append("numeric_reference_incomplete")
                continue
            if (any(not isinstance(reference[key], str) for key in ("metric", "query", "unit"))
                    or any(reference.get(key) is not None and not isinstance(reference[key], str) for key in ("source", "device", "frequency"))):
                verified.append(False)
                errors.append("numeric_reference_invalid_types")
                continue
            try:
                if reference["start_date"] is None or reference["end_date"] is None:
                    raise ValueError("Missing reference date")
                start, end = _timestamp(reference["start_date"]), _timestamp(reference["end_date"])
                if end < start:
                    raise ValueError("Reversed reference dates")
                selected = filter_observations(frame, start, end, [reference["query"]], [reference["region_id"]])
            except (ValueError, TypeError, OverflowError):
                verified.append(False)
                errors.append("numeric_reference_invalid_dates")
                continue
            selected = selected[selected["unit"].eq(reference["unit"])]
            if reference.get("source"):
                selected = selected[selected["source"].eq(reference["source"])]
            for key in ("device", "frequency"):
                if reference.get(key):
                    selected = selected[selected[key].eq(reference[key])]
            usable = selected.dropna(subset=["count"])
            expected = None
            expected_points = None
            if not usable.empty and len(usable[SERIES_KEYS].drop_duplicates()) == 1:
                frequency = usable.iloc[0]["frequency"]
                expected_points = (end.normalize() - start.normalize()).days // (7 if frequency == "weekly" else 1) + 1
                if reference["metric"] == "sum":
                    expected = float(usable["count"].sum())
                elif reference["metric"] == "mean":
                    expected = float(usable["count"].mean())
                elif reference["metric"] == "latest":
                    expected = float(usable.sort_values("date").iloc[-1]["count"])
            try:
                complete = reference["metric"] == "latest" or (expected_points is not None and len(usable) == expected_points)
                matches = complete and expected is not None and math.isfinite(float(reference["value"])) and math.isclose(float(reference["value"]), expected, abs_tol=tolerance, rel_tol=tolerance)
            except (TypeError, ValueError):
                matches = False
            verified.append(bool(matches))
            evidence.append({"reference": reference, "observed_value": expected, "observed_points": len(usable),
                             "expected_points": expected_points,
                             "missing_points_in_rows": int(selected["count"].isna().sum()), "verified": bool(matches)})
        results.append({**item, "numbers_verified": bool(verified) and all(verified), "numeric_evidence": evidence,
                        "verification_errors": errors, "claim_verified": False,
                        "provenance": "observations_numeric_check_only", "url_availability_is_claim_verification": False})
    return json_safe(results)


def safe_probe_url(url, *, enabled=False, timeout=5.0, max_redirects=3) -> dict:
    """Opt-in HEAD probe with public-IP checks and connections pinned to checked IPs."""
    result = {"url": url, "availability": "not_checked", "status_code": None, "claim_verified": False}
    if not enabled:
        return result
    try:
        current = normalize_url(url)
        for _ in range(max_redirects + 1):
            parsed = urlsplit(current)
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            if port not in {80, 443}:
                raise ValueError("Nonstandard URL port")
            hostname = parsed.hostname
            addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(address[4][0]).is_global or ipaddress.ip_address(address[4][0]).is_multicast for address in addresses):
                raise ValueError("Non-public URL address")
            address = addresses[0][4][0]
            connection = http.client.HTTPConnection(hostname, port, timeout=timeout)
            try:
                connection.sock = socket.create_connection((address, port), timeout=timeout)
                if parsed.scheme == "https":
                    connection.sock = ssl.create_default_context().wrap_socket(connection.sock, server_hostname=hostname)
                path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
                connection.request("HEAD", path, headers={"Host": parsed.netloc, "User-Agent": "Thesaurus-LinkCheck/1"})
                response = connection.getresponse()
                status, location = response.status, response.getheader("Location")
                result["status_code"] = status
            finally:
                connection.close()
            if status in {301, 302, 303, 307, 308} and location:
                current = normalize_url(urljoin(current, location))
                continue
            return {**result, "final_url": current, "availability": "available" if 200 <= status < 400 else "unavailable"}
        return {**result, "availability": "redirect_limit"}
    except (ValueError, UnicodeError):
        return {**result, "availability": "blocked"}
    except (OSError, http.client.HTTPException):
        return {**result, "availability": "unreachable"}
