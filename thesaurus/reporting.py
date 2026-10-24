"""Build immutable, filtered reports from shared stored observations."""
import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from .analysis import (
    AlertConfig,
    classify_news_changes,
    cooldown_allowed,
    deduplicate_news,
    evaluate_alerts,
    forecast_weekly,
    quality_summary,
    verify_numeric_claims,
)
from .cbr_data import METRICS as CBR_METRICS
from .rate_forecasts import forecast_daily
from .store import INCIDENTS, dumps, utcnow

TITLES = {"ruonia": "RUONIA", "ai": "Обзор новостей", "wordstat": "Поисковые индикаторы", "regions": "Поисковый интерес по регионам"}


def report_filters(store, section, phrases=None, regions=None):
    if section not in TITLES:
        raise ValueError("Unknown report section")
    if section == "ai":
        if phrases or regions:
            raise ValueError("AI reports do not support observation filters")
        return [], []
    allowed_phrases = (CBR_METRICS if section == "ruonia" else
                       [x["phrase"] for x in store.keywords()] if section == "wordstat" else store.get_setting("incident_phrases", INCIDENTS))
    allowed_regions = store.regions() if section == "regions" else [225]
    phrases = list(phrases or allowed_phrases)
    regions = [int(x) for x in (regions or allowed_regions)]
    if set(phrases) - set(allowed_phrases) or set(regions) - set(allowed_regions):
        raise ValueError("Selected phrases/regions are not in the approved watchlist")
    return phrases, regions


def selected_observations(store, section, phrases=None, regions=None):
    phrases, regions = report_filters(store, section, phrases, regions)
    source = "ruonia" if section == "ruonia" else "wordstat"
    frequency = "weekly" if section == "wordstat" else "daily"
    rows = store.query("SELECT * FROM observations WHERE source=? AND frequency=? ORDER BY date", (source, frequency))
    return [r for r in rows if r["metric"] in phrases and r["region_id"] in regions], phrases, regions


class ReportBuilder:
    def __init__(self, store, settings, now=None):
        self.store, self.settings = store, settings
        self.now = now or (lambda: datetime.now(timezone.utc))

    def today(self):
        return self.now().astimezone(ZoneInfo("Europe/Moscow")).date()

    def build(self, section, metadata, *, phrases=None, regions=None, run_id=None, charts=True):
        if section not in TITLES:
            raise ValueError("Unknown report section")
        if section == "ai":
            chosen_phrases, chosen_regions = report_filters(self.store, section, phrases, regions)
            observations = []
        else:
            observations, chosen_phrases, chosen_regions = selected_observations(self.store, section, phrases, regions)
        fingerprint = hashlib.sha256(dumps({"report_version": 2, "section": section, "phrases": chosen_phrases,
            "regions": chosen_regions, "keyword_version": self.store.version("keyword"),
            "cohort_version": self.store.version("cohort"), "metadata": {k: v for k, v in metadata.items() if k != "cached"},
            "latest": max((r["retrieved_at"] for r in observations), default=None), "day": self.today().isoformat()}).encode()).hexdigest()
        existing = self.store.one("SELECT id FROM reports WHERE fingerprint=?", (fingerprint,))
        if existing:
            return existing["id"]
        report_id = uuid4().hex
        directory = self.settings.reports_dir / report_id
        directory.mkdir(parents=True, exist_ok=True)
        news, metrics, parts = [], {}, []
        if section == "ai":
            cached = metadata.get("ai") or self.store.get_setting("collection_latest:ai", {}).get("ai", {})
            full = cached.get("text") or "Нет сохранённого AI-обзора."
            data = cached.get("data", {})
            raw_claims = data.get("claims", []) if isinstance(data, dict) else []
            raw_claims = raw_claims if isinstance(raw_claims, list) else []
            claims = [self._normalize_claim(c) for c in raw_claims if isinstance(c, dict)]
            evidence = self.store.query("SELECT * FROM observations WHERE value IS NOT NULL")
            checked = verify_numeric_claims(claims, evidence)
            news = deduplicate_news(checked)
            previous = self.store.one("SELECT news FROM reports WHERE section='ai' ORDER BY created_at DESC LIMIT 1")
            changes = classify_news_changes(news, json.loads(previous["news"]) if previous else [])
            for change_type, items in changes.items():
                for item in items:
                    for current in news:
                        if current["fingerprint"] == item["fingerprint"]:
                            current["change"] = change_type
                            current["status"] = "numbers_checked" if current.get("numbers_verified") else "unverified"
                            if item.get("previous_content"):
                                current["previous_content"] = item["previous_content"]
            if news:
                full = (f"Новых сообщений: {len(changes['new'])}; уточнений: {len(changes['corrected'])}; "
                        f"повторов: {len(changes['repeated'])}.\n\n" + full)
            full += "\n\nПроверка: источники и выводы AI не считаются подтверждёнными автоматически."
            if news:
                full += "\nЧисловые ссылки проверены: " + str(sum(bool(c.get("numbers_verified")) for c in news)) + f" из {len(news)}."
            quality = {"cached": bool(metadata.get("cached")), "status": metadata.get("status"),
                       "last_fetched_at": metadata.get("data_retrieved_at") or metadata.get("retrieved_at"), "claims": len(news),
                       "errors": metadata.get("errors", [])}
            quality["dated_sources"] = sum(bool(c.get("published_at")) for c in news)
            full += (f"\nОбзор загружен: {quality['last_fetched_at'] or 'нет данных'}; "
                     f"источников с указанной датой: {quality['dated_sources']} из {len(news)}.")
            if quality["cached"]:
                full += "\nИспользован сохранённый обзор."
            if quality["errors"]:
                full += "\nОбновление неполное: " + "; ".join(quality["errors"][:8])
            if news:
                brief_lines = ["Изменения в обзоре новостей:"]
                brief_lines.extend("Новое: " + item["title"] for item in changes["new"])
                brief_lines.extend("Уточнение: " + item["title"] for item in changes["corrected"])
                if not changes["new"] and not changes["corrected"]:
                    brief_lines.append("Новых структурированных сообщений нет.")
                brief_lines.append(f"Повторов свёрнуто: {len(changes['repeated'])}. Полный обзор по кнопке.")
                brief = "\n".join(brief_lines)
            else:
                brief = full[:1800] + ("\nПолный обзор доступен по кнопке." if len(full) > 1800 else "")
            start = end = self.today().isoformat()
        else:
            end = self._expected_end(section)
            start = end - timedelta(days=7 * 51 if section == "wordstat" else 59 if section == "regions" else 89)
            if section == "wordstat":
                # Stored weekly observations are dated by their closing Sunday.
                start = end - timedelta(weeks=51)
            quality = quality_summary(observations, start, end, queries=chosen_phrases,
                                      region_ids=chosen_regions, frequency="weekly" if section == "wordstat" else "daily", as_of=self.now())
            if section == "ruonia":
                self._business_coverage(quality, observations, start, end, chosen_phrases)
            quality.update(cached=bool(metadata.get("cached")), status=metadata.get("status"), errors=metadata.get("errors", []))
            if metadata.get("errors") and observations:
                quality["cached"] = True
            for row in observations:
                if row["value"] is None or row["date"] > end.isoformat():
                    continue
                key = f"{row['metric']}|{row['region_id']}"
                metrics[key] = {"value": row["value"], "date": row["date"], "metric": row["metric"],
                                "region_id": row["region_id"], "unit": row["unit"]}
            lines = [TITLES[section], f"Период: {start} — {end}", self._quality_line(quality)]
            lines.extend(f"{m['metric']} · {m['region_id']}: {m['value']:.4g} {m['unit']} ({m['date']})" for m in metrics.values())
            if not metrics:
                lines.append("Наблюдений пока нет. Пропуски не заменяются нулём.")
            predictions = self._save_forecasts([r for r in observations if r["date"] <= end.isoformat()]) if section in {"wordstat", "ruonia"} else []
            if predictions:
                lines.append("\nПрогноз на 1–4 недели (приблизительный диапазон, не гарантия):")
                if section == "ruonia":
                    lines.append("Горизонты: 5, 10, 15 и 20 рабочих дней РФ; оценка продолжения ряда, без прогноза решений ЦБ.")
                units = {row["metric"]: row["unit"] for row in observations}
                lines.extend(f"{r['metric']} · {r['target_date']}: {r['prediction']:.3g} [{r['lower']:.3g}; {r['upper']:.3g}] {units[r['metric']]}" for r in predictions)
            elif section == "ruonia":
                lines.append("Прогноз недоступен: требуется минимум 40 наблюдений и 90% покрытия последних 90 дней.")
            if section == "regions":
                lines.append("Поисковый интерес отражает запросы пользователей, а не число происшествий.")
            if metadata.get("errors"):
                lines.append("\nЧасть данных недоступна: " + "; ".join(metadata["errors"][:8]))
            full = "\n".join(lines)
            brief = "\n".join(lines[:min(len(lines), 9)])
            if len(lines) > 9:
                if metadata.get("errors"):
                    brief += "\nЧасть данных недоступна: " + "; ".join(metadata["errors"][:2])
                brief += "\nОстальные показатели и графики доступны по кнопке."
            if charts and observations:
                try:
                    parts.extend(self._charts(section, observations, directory, chosen_regions))
                except Exception as exc:  # noqa: BLE001 - Rendering failure must preserve the numeric report.
                    quality["chart_error"] = type(exc).__name__
                    full += "\nГрафик недоступен; числовые данные сохранены."
            start, end = start.isoformat(), end.isoformat()
        text_path = directory / "report.md"
        text_path.write_text(full, encoding="utf-8")
        parts.append({"kind": "document", "path": str(text_path), "caption": "Полный отчёт"})
        with self.store.connection() as conn:
            conn.execute("""INSERT INTO reports(id,section,created_at,period_start,period_end,keyword_version,cohort_version,
              text,full_text,metrics,news,quality,parts,run_id,fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (report_id, section, utcnow(), start, end, self.store.version("keyword"), self.store.version("cohort"),
                 brief, full, dumps(metrics), dumps(news), dumps(quality), dumps(parts), run_id, fingerprint))
            for item in news:
                conn.execute("INSERT INTO news(report_id,title,url,published_at,claim,status,fingerprint,payload) VALUES(?,?,?,?,?,?,?,?)",
                    (report_id, item.get("title", ""), item.get("url", ""), item.get("published_at"), item.get("claim", ""),
                     "numbers_checked" if item.get("numbers_verified") else "unverified", item.get("fingerprint", ""), dumps(item)))
        return report_id

    def _expected_end(self, section):
        today = self.today()
        if section == "wordstat":
            end = today - timedelta(days=2)
            return end - timedelta(days=(end.weekday() + 1) % 7)
        return today - timedelta(days=1)

    @staticmethod
    def _business_coverage(quality, observations, start, end, phrases):
        from .russian_calendar import calendar as working_calendar
        calendar = working_calendar()
        days = {d.date().isoformat() for d in pd.date_range(start, end)
                if calendar.is_working_day(d.date())}
        expected = len(days) * len(phrases)
        observed = len({(r["date"], r["metric"]) for r in observations if r["date"] in days and r["value"] is not None})
        quality.update(expected_points=expected, observed_points=observed, missing_points=max(0, expected - observed),
                       coverage=observed / expected if expected else None, coverage_basis="Russian working days")

    @staticmethod
    def _quality_line(quality):
        coverage = quality.get("coverage")
        return (f"Последние данные: {quality.get('latest_observation') or 'нет'}; "
                f"полнота: {coverage:.0%}" if coverage is not None else "Наблюдений пока нет") + (
                f"; охват: {quality.get('observed_points', 0)}/{quality.get('expected_points', 0)}; "
                f"пропусков: {quality.get('missing_points', 0)}" +
                f"; загружено: {quality.get('last_fetched_at') or 'нет'}" +
                ("; использованы сохранённые данные" if quality.get("cached") else ""))

    def _save_forecasts(self, observations):
        results = []
        frame = pd.DataFrame(observations)
        if frame.empty:
            return results
        for _, series in frame.groupby(["source", "metric", "region_id", "device", "frequency", "unit"]):
            forecast = forecast_daily if series.iloc[0]["source"] == "ruonia" else forecast_weekly
            forecasts = forecast(series, created_at=self.now())
            for row in forecasts:
                if date.fromisoformat(row["target_date"][:10]) <= self.today():
                    continue
                self.store.execute("""INSERT OR IGNORE INTO forecasts(source,metric,region_id,device,frequency,origin_date,
                    target_date,horizon,prediction,lower,upper,baseline,created_at,model) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (row["source"], row["metric"], int(row["region_id"]), row["device"], row["frequency"],
                     row["origin_date"], row["target_date"], row["horizon"], row["prediction"], row["lower"],
                     row["upper"], row["baseline"], row["created_at"], row["model"]))
                # The first issued prediction remains immutable even if source
                # weeks are revised. Show the same numbers that will be scored.
                results.append(self.store.one("""SELECT * FROM forecasts WHERE source=? AND metric=? AND region_id=?
                    AND device=? AND frequency=? AND origin_date=? AND horizon=? AND model=?""",
                    (row["source"], row["metric"], int(row["region_id"]), row["device"], row["frequency"],
                     row["origin_date"], row["horizon"], row["model"])))
        self.store.execute("""UPDATE forecasts SET actual=(SELECT value FROM observations o WHERE o.source=forecasts.source
            AND o.metric=forecasts.metric AND o.region_id=forecasts.region_id AND o.device=forecasts.device
            AND o.frequency=forecasts.frequency AND o.date=substr(forecasts.target_date,1,10))""")
        return results

    @staticmethod
    def _normalize_claim(claim):
        item = dict(claim)
        item["title"] = str(item.get("title") or item.get("claim") or "Утверждение AI")[:300]
        item["claim"] = str(item.get("claim") or "")
        item["url"] = str(item.get("url") or "")
        published = item.get("published_at")
        try:
            if not isinstance(published, str):
                raise ValueError("Missing source date")  # noqa: TRY004 - Treat malformed and absent external dates uniformly.
            datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            item["published_at"] = None
            item["source_date_status"] = "missing_or_invalid"
        else:
            item["source_date_status"] = "provided_unverified"
        if "numbers" not in item:
            references = item.get("numeric_claims")
            references = references if isinstance(references, list) else []
            item["numbers"] = [{"source": ref.get("source"), "query": ref.get("metric"), "metric": "latest",
                "region_id": ref.get("region_id", 225), "start_date": ref.get("date"), "end_date": ref.get("date"),
                "value": ref.get("value"), "unit": ref.get("unit", "%" if ref.get("source") == "ruonia" else "запросов")}
                for ref in references if isinstance(ref, dict)]
        return item

    @staticmethod
    def _charts(section, observations, directory, regions):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        frame = pd.DataFrame(observations)
        frame["date"] = pd.to_datetime(frame["date"])
        parts = []
        if section == "regions":
            for i, (phrase, rows) in enumerate(frame.groupby("metric")):
                pivot = rows.pivot_table(index="region_id", columns="date", values="value", aggfunc="first", dropna=False).reindex(regions)
                fig, ax = plt.subplots(figsize=(12, max(4, len(regions) * .23)))
                try:
                    data = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd")
                    ax.set_title(phrase + ": запросы по регионам; пропуски без цвета")
                    ax.set_yticks(range(len(regions)), [str(r) for r in regions])
                    ticks = list(range(0, len(pivot.columns), max(1, len(pivot.columns) // 8)))
                    ax.set_xticks(ticks, [pivot.columns[x].strftime("%d.%m") for x in ticks])
                    fig.colorbar(data, ax=ax)
                    fig.tight_layout()
                    path = directory / f"regions_{i}.png"
                    fig.savefig(path, dpi=120)
                    parts.append({"kind": "photo", "path": str(path), "caption": phrase})
                finally:
                    plt.close(fig)
            parts.extend(ReportBuilder._maps(frame, directory))
        else:
            for index, (unit, unit_frame) in enumerate(frame.groupby("unit")):
                fig, ax = plt.subplots(figsize=(12, 5))
                try:
                    for metric, rows in unit_frame.groupby("metric"):
                        rows = rows.sort_values("date").tail(156 if section == "wordstat" else 90)
                        ax.plot(rows["date"], rows["value"], label=metric)
                    ax.set_title(TITLES[section] + ": " + unit)
                    ax.set_ylabel(unit)
                    ax.legend()
                    ax.grid(alpha=.3)
                    fig.autofmt_xdate()
                    fig.tight_layout()
                    path = directory / ("history.png" if index == 0 else f"history_{index}.png")
                    fig.savefig(path, dpi=120)
                    parts.append({"kind": "photo", "path": str(path), "caption": TITLES[section] + ": " + unit})
                finally:
                    plt.close(fig)
        return parts

    @staticmethod
    def _maps(frame, directory):
        # Read only the bundled map assets: report rendering never calls an API.
        from pathlib import Path

        import plotly.graph_objects as go
        from plotly.colors import sample_colorscale
        geo = Path(__file__).resolve().parents[1] / "functions" / "src" / "wordstat" / "geo"
        shapes = pd.read_parquet(geo / "russia_regions.parquet")
        mapping = json.loads((geo / "regionid_to_mapregion.json").read_text(encoding="utf-8"))
        parts = []
        for i, (phrase, rows) in enumerate(frame.groupby("metric")):
            end = rows["date"].max()
            recent = rows[rows["date"] >= end - pd.Timedelta(days=6)]
            sums = recent.groupby("region_id")["value"].sum(min_count=7)
            values = {mapping.get(str(r)): v for r, v in sums.items() if pd.notna(v)}
            rates = {r["region"]: values[r["region"]] / float(r["population"]) * 100000
                     for _, r in shapes.iterrows() if r["region"] in values and float(r["population"]) > 0}
            maximum = max(rates.values(), default=1) or 1
            fig = go.Figure()
            for _, row in shapes.iterrows():
                rate = rates.get(row["region"])
                color = sample_colorscale("YlOrRd", [rate / maximum])[0] if rate is not None and math.isfinite(rate) else "#dddddd"
                fig.add_trace(go.Scatter(x=list(row["x"]), y=list(row["y"]), mode="lines", fill="toself",
                    fillcolor=color, line={"width": .3, "color": "#666"}, showlegend=False,
                    text=f"{row['region']}: {rate:.2f} на 100 тыс." if rate is not None else f"{row['region']}: нет полных данных",
                    hoverinfo="text"))
            fig.update_layout(title=f"{phrase}: запросы за 7 дней на 100 тыс. жителей; серый = нет данных",
                              xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x"})
            path = directory / f"map_{i}.html"
            fig.write_html(path, include_plotlyjs=True)
            parts.append({"kind": "document", "path": str(path), "caption": f"Интерактивная карта: {phrase}"})
        return parts


def record_alerts(store, now=None):
    now = now or datetime.now(timezone.utc)
    observations = store.query("SELECT * FROM observations WHERE source='wordstat' AND frequency='daily' ORDER BY date")
    records = []
    for rule in store.query("SELECT * FROM alert_rules WHERE enabled=1"):
        selected = [r for r in observations if (not rule["metric"] or r["metric"] == rule["metric"])
                    and (not rule["region_id"] or r["region_id"] == rule["region_id"])
                    and r["cohort_version"] == store.version("cohort")]
        config = AlertConfig(percent_threshold=rule["pct_threshold"], z_threshold=rule["z_threshold"],
                             minimum_coverage=rule["min_coverage"], minimum_baseline=rule["min_baseline"],
                             cooldown_hours=rule["cooldown_hours"], direction="both")
        for item in evaluate_alerts(selected, end_date=now.astimezone(ZoneInfo("Europe/Moscow")).date() - timedelta(days=1), config=config):
            if not item["alert"]:
                continue
            last = store.one("SELECT created_at FROM alerts WHERE rule_id=? AND metric=? AND region_id=? ORDER BY created_at DESC LIMIT 1",
                             (rule["id"], item["metric"], item["region_id"]))
            if last and not cooldown_allowed(last["created_at"], now, rule["cooldown_hours"]):
                continue
            with store.connection() as conn:
                cursor = conn.execute("INSERT OR IGNORE INTO alerts(rule_id,metric,region_id,observed_date,created_at,payload) VALUES(?,?,?,?,?,?)",
                    (rule["id"], item["metric"], int(item["region_id"]), str(item["end_date"])[:10], now.isoformat(), dumps(item)))
                if cursor.rowcount:
                    records.append({**item, "id": cursor.lastrowid})
    return records
