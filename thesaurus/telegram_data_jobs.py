"""Offline, scoped Telegram exports and backtests on the sequential worker."""

import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

from .analysis import evaluate_forecasts, json_safe, rolling_backtest
from .delivery import prepare_delivery
from .exporting import export_data
from .reporting import report_filters
from .scheduling import seed_subscriptions, validate_filters
from .store import dumps, utcnow

DATA_JOB_KINDS = {"telegram_export", "telegram_backtest"}


def require_actor(store, actor_id, *, admin=False):
    if type(actor_id) is not int or not store.authorize(actor_id, admin=admin):
        raise PermissionError("Действие недоступно для вашей учётной записи")


def report_identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Нужен ID отчёта из /history")
    return value


def authorized_report(store, actor_id, report_id):
    """Report IDs never grant access without a matching private delivery."""
    require_actor(store, actor_id)
    report_identifier(report_id)
    if not store.authorize(actor_id, admin=True) and not store.one(
        "SELECT id FROM deliveries WHERE report_id=? AND recipient=? LIMIT 1",
        (report_id, str(actor_id)),
    ):
        raise PermissionError("Этот отчёт вам недоступен")
    report = store.one("SELECT * FROM reports WHERE id=?", (report_id,))
    if report is None:
        raise ValueError("Отчёт не найден")
    from .subscription_access import report_section, require
    require(store, actor_id, report_section(report))
    from .feature_access import report_allowed
    if not report_allowed(store, actor_id, report):
        raise PermissionError('Функция этого отчёта больше не доступна для вас')
    return report


def actor_scope(store, actor_id, section):
    """Expand only the actor's own subscription, including its default filters."""
    require_actor(store, actor_id)
    if section not in {"ruonia", "wordstat", "regions"}:
        raise ValueError("Разделы данных: ruonia, wordstat, regions")
    from .subscription_access import require
    require(store, actor_id, section)
    seed_subscriptions(store, str(actor_id), actor_id)
    subscription = store.one(
        "SELECT * FROM subscriptions WHERE user_id=? AND recipient=? AND section=?",
        (actor_id, str(actor_id), section),
    )
    if subscription is None:
        raise PermissionError("Личная подписка недоступна")
    filters = validate_filters(store, section, json.loads(subscription["phrases"]),
                               json.loads(subscription["regions"]))
    phrases, regions = report_filters(store, section, **filters)
    return {"actor_id": actor_id, "recipient": str(actor_id), "section": section,
            "subscription_id": subscription["id"], "phrases": phrases, "regions": regions,
            "source": "ruonia" if section == "ruonia" else "wordstat",
            "frequency": "weekly" if section == "wordstat" else "daily"}


def validate_job_scope(store, payload):
    """Frozen selections may shrink in availability, but must never broaden."""
    current = actor_scope(store, payload.get("actor_id"), payload.get("section"))
    for field in ("recipient", "subscription_id", "source", "frequency"):
        if payload.get(field) != current[field]:
            raise PermissionError("Параметры задания больше не соответствуют вашей подписке")
    filters = validate_filters(store, payload["section"], payload.get("phrases"), payload.get("regions"))
    for field in ("phrases", "regions"):
        if payload.get(field) != filters[field] or not set(filters[field]).issubset(current[field]):
            raise PermissionError("Выбор данных изменился. Повторите команду")
    return current


def scoped_rows(store, table, scope, *, as_of=None):
    if table not in {"observations", "forecasts"}:
        raise ValueError("Unknown data table")
    phrases, regions = scope["phrases"], scope["regions"]
    if not phrases or not regions:
        return []
    where = ("source=? AND frequency=? AND metric IN (" + ",".join("?" for _ in phrases)
             + ") AND region_id IN (" + ",".join("?" for _ in regions) + ")")
    parameters = [scope["source"], scope["frequency"], *phrases, *regions]
    if as_of:
        where += " AND " + ("date" if table == "observations" else "origin_date") + "<=?"
        parameters.append(as_of)
    return store.query(f"SELECT * FROM {table} WHERE {where} ORDER BY "
                       + ("date" if table == "observations" else "origin_date DESC,metric,horizon"), parameters)


def forecast_summary(store, actor_id, section="wordstat"):
    if section not in {"wordstat", "ruonia"}:
        raise ValueError("Forecast sections: wordstat, ruonia")
    scope = actor_scope(store, actor_id, section)
    today = utcnow()[:10]
    forecasts = scoped_rows(store, "forecasts", scope, as_of=today)
    actuals = scoped_rows(store, "observations", scope, as_of=today)
    # Stored actuals are refreshed on collection; evaluation joins the current
    # observations itself, and must not receive a second, stale actual column.
    predictions = [{key: value for key, value in row.items() if key != "actual"} for row in forecasts]
    if section == "ruonia":
        scores = []
        for metric in scope["phrases"]:
            actual = [row for row in actuals if row["metric"] == metric]
            score = evaluate_forecasts([row for row in predictions if row["metric"] == metric], actual)
            scores.append({"metric": metric, "unit": actual[-1]["unit"] if actual else "", "metrics": score})
        metrics = {"series": scores}
    else:
        metrics = evaluate_forecasts(predictions, actuals)
    return forecasts, metrics


def metrics_text(metrics):
    if "series" in metrics:
        return "\n".join(f"{item['metric']} ({item.get('unit', '')}): " + metrics_text(item["metrics"]) for item in metrics["series"])
    if not metrics.get("n"):
        return "Недостаточно сопоставимых фактов для оценки ошибки."
    def number(key):
        value = metrics.get(key)
        return "—" if value is None else f"{value:.3g}"
    coverage = metrics.get("interval_coverage")
    return (f"Проверено: {metrics['n']}; MAE: {number('mae')}; "
            f"MAE базового прогноза: {number('baseline_mae')}; "
            f"WAPE: {number('wape')}; покрытие интервала: "
            + ("—" if coverage is None else f"{coverage:.0%}"))


def export_dates(date_from=None, date_to=None):
    try:
        start = date.fromisoformat(date_from).isoformat() if date_from else None
        end = date.fromisoformat(date_to).isoformat() if date_to else None
    except (TypeError, ValueError):
        raise ValueError("Даты должны быть в формате YYYY-MM-DD") from None
    if start and end and start > end:
        raise ValueError("Начало периода не может быть позже конца")
    return start, end


def _artifact_path(settings, report_id, filename):
    root = Path(settings.reports_dir).resolve()
    directory = (root / report_identifier(report_id)).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Invalid artifact directory")
    directory.mkdir(parents=True, exist_ok=True)
    output = (directory / filename).resolve()
    if not output.is_relative_to(directory):
        raise ValueError("Invalid artifact path")
    return output


def _backtest(store, payload):
    if payload["section"] not in {"wordstat", "ruonia"} or type(payload.get("max_origins")) is not int or not 1 <= payload["max_origins"] <= 52:
        raise ValueError("Invalid backtest selection")
    as_of = export_dates(payload.get("as_of"))[0]
    if as_of is None or as_of > utcnow()[:10]:
        raise ValueError("Backtest cutoff cannot be in the future")
    rows = scoped_rows(store, "observations", payload, as_of=as_of)
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in ("source", "metric", "region_id", "device", "frequency", "unit"))
        groups[key].append(row)
    predictions, series = [], []
    for key, history in sorted(groups.items()):
        from .rate_forecasts import backtest_daily
        runner = backtest_daily if payload["section"] == "ruonia" else rolling_backtest
        result = runner(history, max_origins=payload["max_origins"], **({'horizon': payload.get('forecast')} if payload['section'] == 'ruonia' else {}))
        predictions.extend(result["forecasts"])
        series.append({"metric": key[1], "region_id": key[2], "device": key[3], "unit": key[5], "metrics": result["metrics"]})
    metrics = {"series": series} if payload["section"] == "ruonia" else evaluate_forecasts(predictions, rows, issued_only=False)
    result = {"evaluation": "retrospective_backtest", "as_of": as_of, "metrics": metrics,
              "series": series, "forecasts": predictions,
              "note": "Обучение использует только наблюдения до каждой точки прогноза. "
                      "Это ретроспективная проверка; исторические данные могли быть пересмотрены."}
    return json_safe(result)


def run_data_job(store, settings, job, payload, *, dry_run=False):
    """Build a local artifact and durable delivery; never create a Telegram client."""
    if job["kind"] not in DATA_JOB_KINDS:
        raise ValueError("Unknown data job")
    validate_job_scope(store, payload)
    report_id = report_identifier(payload.get("report_id"))
    fingerprint = f"telegram-data:{job['id']}"
    existing = store.one("SELECT id,fingerprint,quality FROM reports WHERE id=?", (report_id,))
    if existing:
        if (existing["fingerprint"] != fingerprint
                or json.loads(existing["quality"]).get("actor_id") != payload["actor_id"]):
            raise PermissionError("Artifact ownership mismatch")
    else:
        if job["kind"] == "telegram_export":
            output_format = payload.get("format")
            if output_format not in {"csv", "xlsx"}:
                raise ValueError("Invalid export format")
            start, end = export_dates(payload.get("date_from"), payload.get("date_to"))
            path = _artifact_path(settings, report_id, f"export.{output_format}")
            forecast_rows = None
            if payload.get('chart_options', {}).get('forecast') is not None:
                from .charts import chart_series
                groups, _, _ = chart_series(store, settings, payload['actor_id'], payload['section'],
                                             payload['chart_options'].get('identity'), payload['chart_options'])
                forecast_rows = [{'source': payload['source'], 'metric': group['title'], 'date': p[group['target_key']],
                                  'value': p['prediction'], 'unit': group['unit'], 'frequency': group['frequency'],
                                  'record_type': 'forecast', 'quality': p.get('note', 'forecast'), 'model': p['model'],
                                  'lower': p.get('lower'), 'upper': p.get('upper'), 'baseline': p.get('baseline'),
                                  'forecast_created_at': p.get('created_at')}
                                 for group in groups for p in group['forecasts']]
            export_data(store, path, output_format, start, end, payload["phrases"], payload["regions"],
                        payload["source"], payload["frequency"], **({'forecast_rows': forecast_rows} if forecast_rows is not None else {}))
            summary = (f"Экспорт {payload['section']} ({output_format.upper()})\n"
                       f"Период: {start or 'вся история'} — {end or 'последняя дата'}\n"
                       f"Фразы: {', '.join(payload['phrases']) or 'не выбраны'}\n"
                       f"Регионы: {', '.join(map(str, payload['regions'])) or 'не выбраны'}")
        else:
            result = _backtest(store, payload)
            path = _artifact_path(settings, report_id, "backtest.json")
            path.write_text(dumps(result), encoding="utf-8")
            summary = "Ретроспективная проверка ваших рядов\n" + metrics_text(result["metrics"]) + "\n" + result["note"]
        validate_job_scope(store, payload)
        parts = [{"kind": "document", "path": str(path), "caption": "Экспорт данных" if job["kind"] == "telegram_export" else "Результаты backtest"}]
        store.execute(
            """INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,quality,parts,fingerprint)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (report_id, payload["section"], utcnow(), store.version("keyword"), store.version("cohort"),
             summary, summary, dumps({"actor_id": payload["actor_id"], "job_kind": job["kind"]}), dumps(parts), fingerprint),
        )
    validate_job_scope(store, payload)
    if not dry_run:
        prepare_delivery(store, report_id, str(payload["actor_id"]), detail="full")
    return report_id
