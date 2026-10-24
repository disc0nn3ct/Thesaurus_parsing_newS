"""Provider adapters and shared incremental collection, independent of delivery."""
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from functions.wordstat_errors import WordstatError

from . import budget
from .store import INCIDENTS, dumps


class Providers:
    def __init__(self, store):
        self.store = store

    def wordstat(self, path, payload):
        from functions.wordstat_api import _wordstat_post
        attempts = []
        try:
            result = _wordstat_post(path, payload, before_attempt=lambda: attempts.append(budget.reserve(self.store, "wordstat")))
        except Exception:
            for usage_id in attempts:
                budget.finish(self.store, usage_id, failed=True)
            raise
        for usage_id in attempts[:-1]:
            budget.finish(self.store, usage_id, failed=True)
        if attempts:
            budget.finish(self.store, attempts[-1])
        if not isinstance(result, dict):
            raise ValueError("Wordstat returned an unexpected response type")  # noqa: TRY004 - Provider schema failures share the validation-error contract.
        return result

    def ruonia(self, now=None):
        from .cbr_incremental import collect_report
        from .market_sources import Transport
        now = now or datetime.now(timezone.utc)
        return collect_report(self.store, Transport(), now.astimezone(ZoneInfo('Europe/Moscow')).date(), now)

    def ai(self, context):
        from .local_ai import backend, overview
        if backend() != "perplexity":
            return overview(context)
        from functions.ai import get_client, run_brief
        get_client()  # Validate credentials before reserving any provider attempts.
        attempts = []
        try:
            result = run_brief(before_attempt=lambda: attempts.append(budget.reserve(self.store, "ai")),
                               max_tokens=self.store.get_setting("budgets")["ai_output_tokens"], context=context)
        except Exception:
            for usage_id in attempts:
                budget.finish(self.store, usage_id, failed=True)
            raise
        for usage_id in attempts[:-1]:
            budget.finish(self.store, usage_id, failed=True)
        # run_brief's existing usage string contains the provider's actual totals.
        usage = result[1]
        prompt = re.search(r"prompt\s*[:=]\s*(\d+)", usage, re.IGNORECASE)
        completion = re.search(r"completion\s*[:=]\s*(\d+)", usage, re.IGNORECASE)
        if attempts:
            budget.finish(self.store, attempts[-1], tokens_in=int(prompt[1]) if prompt else 0,
                          tokens_out=int(completion[1]) if completion else 0)
        return result

    def propose_keywords(self, current):
        from .local_ai import backend, proposals
        if backend() != "perplexity":
            return proposals(current)
        from functions.ai import MODEL, get_client, retry_call
        client = get_client()
        attempts = []
        response = None

        def call():
            attempts.append(budget.reserve(self.store, "ai"))
            return client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content":
                "Предложи до 5 дополнительных поисковых индикаторов, не меняя текущий набор: "
                + dumps(current) + '. Ответ только JSON: [{"phrase":"...","theme":"..."}].'}],
                max_tokens=min(800, self.store.get_setting("budgets")["ai_output_tokens"]),
            )
        try:
            response = retry_call(call)
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            entries = json.loads(content)
            if not isinstance(entries, list) or not 1 <= len(entries) <= 5:
                raise ValueError("Invalid keyword proposals")
            for item in entries:
                if not isinstance(item, dict) or not isinstance(item.get("phrase"), str) or not item["phrase"].strip():
                    raise ValueError("Invalid keyword proposal")
            usage = getattr(response, "usage", None)
            for usage_id in attempts[:-1]:
                budget.finish(self.store, usage_id, failed=True)
            budget.finish(self.store, attempts[-1], tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
                          tokens_out=getattr(usage, "completion_tokens", 0) or 0)
            return entries
        except Exception:
            for usage_id in attempts[:-1]:
                budget.finish(self.store, usage_id, failed=True)
            if attempts:
                # A rejected proposal can still be a paid successful model
                # response. Preserve the reported tokens when parsing fails.
                usage = getattr(response, "usage", None)
                budget.finish(self.store, attempts[-1], failed=True,
                              tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
                              tokens_out=getattr(usage, "completion_tokens", 0) or 0)
            raise


def ruonia_rows(frame):
    from .cbr_data import parse_day
    names = {"Индекс": "Индекс RUONIA", "1 месяц": "1 мес", "3 месяца": "3 мес", "6 месяцев": "6 мес"}
    frame = frame.rename(columns=names).copy()
    required = {"Дата", *names.values()}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("RUONIA workbook has no expected table")
    dates = frame["Дата"].map(parse_day)
    if dates.isna().any():
        raise ValueError("RUONIA workbook contains invalid dates")
    rows = []
    metrics = [*names.values(), *[name for name in ("Ставка RUONIA", "Объём RUONIA") if name in frame]]
    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="coerce")
        for day, value in zip(dates, values):
            unit = "индекс" if metric == "Индекс RUONIA" else "млрд руб" if metric == "Объём RUONIA" else "%"
            rows.append({"source": "ruonia", "metric": "RUONIA" if metric == "Ставка RUONIA" else metric, "frequency": "daily",
                         "date": day.date().isoformat(), "value": None if pd.isna(value) else float(value),
                         "unit": unit, "source_url": ("https://cbr.ru/hd_base/ruonia/ruonia/" if metric in {"Ставка RUONIA", "Объём RUONIA"}
                                                       else "https://cbr.ru/hd_base/ruonia/")})
    return rows


class Collector:
    def __init__(self, store, settings, providers=None, now=None):
        self.store, self.settings = store, settings
        self.providers = providers or Providers(store)
        self.now = now or (lambda: datetime.now(timezone.utc))

    def import_legacy(self):
        if self.store.get_setting("legacy_ruonia_imported", False):
            return
        path = self.settings.root / "ruonia_data.xlsx"
        if path.is_file():
            self.store.save_observations(ruonia_rows(pd.read_excel(path)))
        self.store.set_setting("legacy_ruonia_imported", True)

    def collect(self, section, dry_run=False, force=False):
        from .search_provider import require_section
        require_section(self.store, section)
        from .google_trends import active
        if section in {'wordstat', 'regions'} and active(self.store):
            return {'status': 'cached', 'cached': True, 'retrieved_at': None,
                    'errors': ['Wordstat выключен. Поисковые данные: /trends → импорт CSV Google Trends.']}
        now = self.now()
        today = now.astimezone(ZoneInfo("Europe/Moscow")).date()
        edition = "am" if now.astimezone(ZoneInfo("Europe/Moscow")).hour < 12 else "pm"
        if section == "ai":
            edition += ":" + os.getenv("AI_BACKEND", "perplexity") + ":" + os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
        elif section == "ruonia":
            edition = "cbr-v2"
        snapshot = self._settings_snapshot()
        cache_key = f"collection:{section}:{today}:{edition if section in {'ai', 'ruonia'} else 'day'}:{snapshot['keyword_version']}:{snapshot['cohort_version']}"
        mode = os.getenv('WORDSTAT_BACKEND', 'legacy').strip().lower()
        mode = mode if mode in {'legacy', 'cloud'} else 'invalid'
        if section in {'wordstat', 'regions'}:
            cache_key += ':' + mode
        previous = self.store.get_setting(cache_key)
        if dry_run:
            saved = previous or self.store.get_setting(f"collection_latest:{section}", {})
            return {**saved, "status": "cached", "cached": True, "errors": saved.get("errors", []),
                    "retrieved_at": saved.get("retrieved_at")}
        if section in {'wordstat', 'regions'} and not force:
            pause = self.store.get_setting('wordstat_pause:' + mode, {})
            try:
                paused = datetime.fromisoformat(pause['until']) > now
            except (KeyError, TypeError, ValueError):
                paused = False
            if paused:
                saved = self.store.get_setting(f'collection_latest:{section}', {})
                return {**saved, 'status': 'partial', 'cached': True,
                        'retrieved_at': saved.get('retrieved_at'),
                        'errors': [pause['reason'], 'Повторная загрузка доступна после ' + pause['until']]}
        if previous and not force and self._cache_usable(previous, now, section):
            return {**previous, "cached": True}
        errors = []
        metadata = {"status": "complete", "cached": False, "errors": errors, "retrieved_at": now.isoformat()}
        try:
            if section == "ruonia":
                self._repair_legacy_index()
                frame = self.providers.ruonia(now) if isinstance(self.providers, Providers) else self.providers.ruonia()
                errors.extend(frame.attrs.get("collection_errors", []))
                rows = [] if frame.attrs.get('stored_incrementally') else ruonia_rows(frame)
                if frame.attrs.get('stored_incrementally'):
                    metadata.update(cached=frame.attrs['cached'], new_observations=frame.attrs['new_observations'],
                                    data_retrieved_at=frame.attrs['data_retrieved_at'])
                for row in rows:
                    row.update(keyword_version=snapshot["keyword_version"], cohort_version=snapshot["cohort_version"], retrieved_at=self.now().isoformat())
                self.store.save_observations(rows)
                from .market import sync_ruonia
                sync_ruonia(self.store)
            elif section == "wordstat":
                end = today - timedelta(days=2)
                end -= timedelta(days=(end.weekday() + 1) % 7)
                for entry in snapshot["keywords"]:
                    self._series(entry["phrase"], 225, "weekly", date(2018, 1, 1), end, errors, snapshot)
            elif section == "regions":
                if not snapshot["regions"]:
                    self._initialize_cohort(snapshot)
                end = today - timedelta(days=1)
                for phrase in snapshot["incident_phrases"]:
                    for region_id in snapshot["regions"]:
                        self._series(phrase, region_id, "daily", end - timedelta(days=59), end, errors, snapshot)
            elif section == "ai":
                context = self.store.query("""SELECT source,metric,region_id,date,value,unit FROM observations o
                    WHERE value IS NOT NULL AND date=(SELECT MAX(date) FROM observations x
                    WHERE x.source=o.source AND x.metric=o.metric AND x.region_id=o.region_id AND x.frequency=o.frequency)
                    ORDER BY date DESC LIMIT 100""")
                text, usage, data = self.providers.ai(context)
                metadata["ai"] = {"text": text, "usage": usage, "data": data or {}}
                if data and data.get("fallback"):
                    errors.append("Локальная модель недоступна; использован шаблон")
            else:
                raise ValueError("Unknown section")
        except WordstatError as exc:
            errors.append(exc.safe_message())
            if exc.stops_collection:
                self.store.set_setting('wordstat_pause:' + mode, {
                    'until': (self.now() + timedelta(minutes=5)).isoformat(), 'reason': exc.safe_message()})
        except Exception as exc:  # noqa: BLE001 - Isolate provider failures and preserve other collected sections.
            errors.append(self._safe_error(exc))
        if errors:
            metadata["status"] = "partial"
            if section == "ai" and "ai" not in metadata:
                saved = self.store.get_setting("collection_latest:ai", {})
                if saved.get("ai"):
                    metadata["ai"] = saved["ai"]
                    metadata["cached"] = True
                    metadata["data_retrieved_at"] = saved.get("data_retrieved_at") or saved.get("retrieved_at")
        # Only our own initial-cohort update changes the collection snapshot.
        # A concurrent admin edit must never label old results as a new version.
        cache_key = f"collection:{section}:{today}:{edition if section in {'ai', 'ruonia'} else 'day'}:{snapshot['keyword_version']}:{snapshot['cohort_version']}"
        if section in {'wordstat', 'regions'}:
            cache_key += ':' + mode
        metadata["retrieved_at"] = self.now().isoformat()
        metadata["keyword_version"] = snapshot["keyword_version"]
        metadata["cohort_version"] = snapshot["cohort_version"]
        self.store.set_setting(cache_key, metadata)
        self.store.set_setting(f"collection_latest:{section}", metadata)
        return metadata

    def _repair_legacy_index(self):
        """Preserve old index values under their correct name, with a backup."""
        import sqlite3
        from contextlib import closing
        from uuid import uuid4

        condition = "source='ruonia' AND metric='RUONIA' AND source_url='https://cbr.ru/hd_base/ruonia/'"
        if not self.store.one("SELECT id FROM observations WHERE " + condition + " LIMIT 1"):
            return
        backup = self.store.path.with_name(self.store.path.name + ".index-repair-" + uuid4().hex + ".bak")
        with self.store.connection() as conn, closing(sqlite3.connect(backup)) as target:
            conn.backup(target)
        fields = "source,metric,region_id,device,frequency,date,value,unit,retrieved_at,quality,source_url,keyword_version,cohort_version"
        with self.store.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT OR IGNORE INTO observations(" + fields + ") SELECT source,'Индекс RUONIA',region_id,device,"
                         "frequency,date,value,'индекс',retrieved_at,quality,source_url,keyword_version,cohort_version "
                         "FROM observations WHERE " + condition)
            conn.execute("DELETE FROM observations WHERE " + condition)

    def _settings_snapshot(self):
        keys = ("keywords", "keyword_version", "regions", "cohort_version", "incident_phrases")
        rows = self.store.query("SELECT key,value FROM settings WHERE key IN (?,?,?,?,?)", keys)
        snapshot = {row["key"]: json.loads(row["value"]) for row in rows}
        snapshot.setdefault("incident_phrases", INCIDENTS)
        return snapshot

    @staticmethod
    def _cache_usable(metadata, now, section=None):
        complete = metadata.get("status") == "complete"
        if complete and section not in {"ruonia", "regions"}:
            return True
        ttl = {"ruonia": 1800, "regions": 3600}.get(section, 300) if complete else 300
        try:
            retrieved = datetime.fromisoformat(metadata["retrieved_at"])
            if retrieved.tzinfo is None:
                return False
            return 0 <= (now - retrieved).total_seconds() < ttl
        except (KeyError, TypeError, ValueError):
            return False

    def _initialize_cohort(self, snapshot):
        totals = {}
        for phrase in snapshot["incident_phrases"]:
            response = self.providers.wordstat("/v1/regions", {"phrase": phrase, "regionType": "regions", "devices": ["all"]})
            rows = response.get("regions", [])
            if not rows:
                raise ValueError("Cannot initialize a fixed cohort from incomplete distributions")
            for row in rows:
                region = int(row["regionId"])
                totals[region] = totals.get(region, 0) + float(row.get("count", 0))
        selected = sorted(totals, key=lambda region: (-totals[region], region))[:25]
        if self.store.set_regions(selected, expected_version=snapshot["cohort_version"]):
            snapshot["regions"] = selected
            snapshot["cohort_version"] += 1
        else:
            # Preserve a cohort the administrator chose while discovery ran.
            latest = self._settings_snapshot()
            snapshot["regions"] = latest["regions"]
            snapshot["cohort_version"] = latest["cohort_version"]

    def _series(self, phrase, region, frequency, first, end, errors, snapshot=None):
        snapshot = snapshot or self._settings_snapshot()
        last = self.store.one("SELECT MAX(date) AS date FROM observations WHERE source='wordstat' AND metric=? AND region_id=? AND frequency=?",
                              (phrase, region, frequency))["date"]
        if last:
            first = max(first, date.fromisoformat(last) - timedelta(days=28 if frequency == "weekly" else 7))
        if frequency == "weekly":
            first -= timedelta(days=first.weekday())
        try:
            data = self.providers.wordstat("/v1/dynamics", {"phrase": phrase, "regions": [region], "devices": ["all"],
                "period": frequency, "fromDate": first.isoformat(), "toDate": end.isoformat()})
            if not isinstance(data, dict) or not isinstance(data.get("dynamics"), list):
                raise TypeError("Unexpected Wordstat dynamics schema")
            rows = []
            for item in data.get("dynamics", []):
                day = date.fromisoformat(item["date"][:10])
                # Persist weekly buckets by their closing date. Forecast target
                # dates then mean period end, not an already elapsed Monday.
                if frequency == "weekly":
                    day += timedelta(days=6)
                if not first <= day <= end:
                    continue
                value = item.get("count")
                if value is not None and float(value) < 0:
                    raise ValueError("Negative Wordstat count")
                rows.append({"source": "wordstat", "metric": phrase, "region_id": region,
                             "frequency": frequency, "date": day.isoformat(), "value": value,
                             "unit": "запросов", "source_url": "https://wordstat.yandex.ru/",
                             "keyword_version": snapshot["keyword_version"], "cohort_version": snapshot["cohort_version"],
                             "retrieved_at": self.now().isoformat()})
            if not rows:
                errors.append(f"Нет наблюдений: {phrase}, регион {region}")
            else:
                self.store.save_observations(rows)
        except budget.BudgetExceeded:
            raise
        except WordstatError as exc:
            if exc.stops_collection:
                raise
            errors.append(f"{phrase}, регион {region}: {exc.safe_message()}")
        except Exception as exc:  # noqa: BLE001 - One unavailable series must not discard successful series.
            errors.append(f"{phrase}, регион {region}: {self._safe_error(exc)}")

    @staticmethod
    def _safe_error(exc):
        if isinstance(exc, budget.BudgetExceeded):
            return str(exc)
        if isinstance(exc, WordstatError):
            return exc.safe_message()
        return f"Источник недоступен или ответ некорректен ({type(exc).__name__})"
