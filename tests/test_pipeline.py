"""Offline pipeline integration with real SQLite, pandas, and immutable reports."""

import json
import os
import shutil
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stdout
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pandas as pd

from thesaurus import budget, cli
from thesaurus.collectors import Collector
from thesaurus.config import Settings
from thesaurus.delivery import deliver_pending, prepare_delivery
from thesaurus.reporting import ReportBuilder
from thesaurus.scheduling import seed_subscriptions
from thesaurus.service import Worker, bootstrap
from thesaurus.store import Store, dumps, utcnow

NOW = datetime(2026, 10, 2, 16, 0, tzinfo=timezone.utc)


class FakeProviders:
    def __init__(self):
        self.calls = []
        self.fail_phrase = None

    def ruonia(self):
        self.calls.append(("ruonia",))
        return pd.DataFrame({"Дата": pd.date_range("2026-09-20", "2026-10-01"),
                             "Индекс": [15.5] * 12, "1 месяц": [15.6] * 12,
                             "3 месяца": [15.7] * 12, "6 месяцев": [15.8] * 12})

    def wordstat(self, path, payload):
        self.calls.append(("wordstat", path, dict(payload)))
        if payload.get("phrase") == self.fail_phrase:
            raise TimeoutError("private-provider-response")
        if path == "/v1/regions":
            return {"regions": [{"regionId": 1, "count": 200}, {"regionId": 2, "count": 100}]}
        end = date.fromisoformat(payload["toDate"])
        if payload["period"] == "weekly":
            dates = pd.date_range(end=end - timedelta(days=6), periods=60, freq="W-MON")
        else:
            dates = pd.date_range(end=end, periods=60, freq="D")
        return {"dynamics": [{"date": day.date().isoformat(), "count": 100 + i} for i, day in enumerate(dates)]}

    def ai(self, context):
        self.calls.append(("ai", context))
        return "Offline overview", "Tokens used: prompt=10, completion=20, total=30", {"claims": []}

    def propose_keywords(self, current):
        self.calls.append(("propose", current))
        return [{"phrase": "proposal", "theme": "test"}]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {'WORDSTAT_BACKEND': 'legacy'}))
        self.parent = Path(__file__).resolve().parent
        self.root = self.parent / f"pipeline_test_{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.cleanup_directory)
        self.settings = Settings(root=self.root, admin_ids=(1,), recipients=("2", "@legacy"))
        self.path = self.root / "state.sqlite3"
        self.store = Store(self.path)
        self.store.set_keywords([{"phrase": "обвал", "theme": "экономика"}, {"phrase": "рост цен", "theme": "цены"}])
        self.store.set_regions([1, 2])
        self.store.set_setting("incident_phrases", ["пожар", "взрыв"])
        self.providers = FakeProviders()
        self.worker = Worker(self.store, self.settings, self.providers, charts=False, now=lambda: NOW)
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")))

    def cleanup_directory(self):
        if self.root.resolve().parent != self.parent:
            raise AssertionError("Unexpected test cleanup path")
        shutil.rmtree(self.root)

    def enqueue(self, section, recipient="2", **filters):
        return self.store.enqueue("report", {"section": section, "recipient": recipient, "detail": "full", **filters})

    def subscription(self, section="ai", user=2):
        self.store.add_user(user, subscriptions='all')
        seed_subscriptions(self.store, str(user), user)
        self.store.execute("UPDATE subscriptions SET enabled=1 WHERE recipient=? AND section=?", (str(user), section))
        return self.store.one("SELECT id FROM subscriptions WHERE recipient=? AND section=?", (str(user), section))["id"]

    def test_disabled_schedule_job_is_cancelled_without_api_but_manual_job_is_allowed(self):
        subscription_id = self.subscription()
        scheduled = self.enqueue("ai", subscription_id=subscription_id)
        self.store.execute("UPDATE subscriptions SET enabled=0 WHERE id=?", (subscription_id,))
        self.worker.process_next()
        self.assertEqual(self.providers.calls, [])
        self.assertEqual(self.store.one("SELECT state FROM jobs WHERE id=?", (scheduled,))["state"], "cancelled")
        manual = self.enqueue("ai")
        self.worker.process_next()
        self.assertEqual(len(self.providers.calls), 1)
        self.assertEqual(self.store.one("SELECT state FROM jobs WHERE id=?", (manual,))["state"], "done")
        self.store.execute("UPDATE subscriptions SET enabled=1 WHERE id=?", (subscription_id,))
        self.worker.recover()
        self.assertEqual(self.store.one("SELECT state FROM jobs WHERE id=?", (scheduled,))["state"], "cancelled")

    def test_subscription_disabled_during_running_job_prevents_preparation(self):
        subscription_id = self.subscription()
        job_id = self.enqueue("ai", subscription_id=subscription_id)
        original_build = self.worker.builder.build

        def build_then_disable(*args, **kwargs):
            report_id = original_build(*args, **kwargs)
            self.store.execute("UPDATE subscriptions SET enabled=0 WHERE id=?", (subscription_id,))
            return report_id

        with patch.object(self.worker.builder, "build", side_effect=build_then_disable):
            self.worker.process_next()
        self.assertEqual(len(self.providers.calls), 1)
        self.assertEqual(self.store.one("SELECT state FROM jobs WHERE id=?", (job_id,))["state"], "cancelled")
        self.assertEqual(self.store.query("SELECT * FROM deliveries"), [])

    def test_scheduled_job_ownership_checked_before_collection_and_origin_is_preserved(self):
        subscription_id = self.subscription()
        wrong_user = self.enqueue("ai", "3", subscription_id=subscription_id)
        wrong_section = self.enqueue("wordstat", subscription_id=subscription_id)
        self.worker.drain()
        self.assertEqual(self.providers.calls, [])
        self.assertTrue(all(self.store.one("SELECT state FROM jobs WHERE id=?", (value,))["state"] == "cancelled"
                            for value in (wrong_user, wrong_section)))
        self.enqueue("ai", subscription_id=subscription_id)
        self.worker.drain()
        deliveries = self.store.query("SELECT payload FROM deliveries")
        self.assertTrue(deliveries)
        self.assertTrue(all(json.loads(row["payload"])["subscription_id"] == subscription_id for row in deliveries))

    def test_alert_delivery_keeps_subscription_origin_for_later_cancellation(self):
        subscription_id = self.subscription(section="regions")
        alert = {"metric": "пожар", "region_id": 1, "current_total": 100, "coverage": 1}
        self.store.execute("INSERT INTO alerts(metric,region_id,observed_date,created_at,payload) VALUES(?,?,?,?,?)",
                           ("пожар", 1, "2026-10-01", utcnow(), dumps(alert)))
        with patch("thesaurus.service.record_alerts", return_value=[]):
            self.worker._alerts()
        rows = self.store.query("SELECT * FROM deliveries")
        self.assertTrue(rows)
        self.assertTrue(all(json.loads(row["payload"])["subscription_id"] == subscription_id for row in rows))
        self.store.execute("UPDATE subscriptions SET enabled=0 WHERE id=?", (subscription_id,))
        client = Mock()
        deliver_pending(self.store, client)
        self.assertTrue(all(row["state"] == "cancelled" for row in self.store.query("SELECT state FROM deliveries")))
        client.send_message.assert_not_called()

    def test_all_sections_build_persistent_reports_with_fake_providers(self):
        for section in ("ruonia", "wordstat", "regions", "ai"):
            self.enqueue(section)
        self.assertEqual(self.worker.drain(), 4)
        self.assertTrue(all(row["state"] == "done" for row in self.store.query("SELECT state FROM jobs")))
        reports = self.store.query("SELECT * FROM reports")
        self.assertEqual({report["section"] for report in reports}, {"ruonia", "wordstat", "regions", "ai"})
        for report in reports:
            with self.subTest(section=report["section"]):
                self.assertTrue(report["text"])
                artifacts = json.loads(report["parts"])
                self.assertTrue(artifacts)
                self.assertTrue(all(Path(part["path"]).is_file() for part in artifacts))
        self.assertGreater(self.store.one("SELECT COUNT(*) AS n FROM observations")["n"], 0)
        self.assertGreater(self.store.one("SELECT COUNT(*) AS n FROM deliveries")["n"], 0)

    def test_per_recipient_filters_do_not_change_shared_collection(self):
        self.enqueue("regions", "2", phrases=["пожар"], regions=[1])
        self.enqueue("regions", "3", phrases=["взрыв"], regions=[2])
        self.worker.drain()
        calls = [call for call in self.providers.calls if call[0] == "wordstat"]
        self.assertEqual(len(calls), 4)  # Two approved phrases x two fixed regions, collected once.
        for recipient, key in (("2", "пожар|1"), ("3", "взрыв|2")):
            row = self.store.one("SELECT r.metrics FROM reports r JOIN deliveries d ON d.report_id=r.id WHERE d.recipient=?", (recipient,))
            self.assertEqual(set(json.loads(row["metrics"])), {key})

    def test_same_report_and_collection_reused_for_multiple_recipients(self):
        self.enqueue("wordstat", "2")
        self.enqueue("wordstat", "3")
        self.worker.drain()
        self.assertEqual(len(self.providers.calls), 2)
        self.assertEqual(self.store.one("SELECT COUNT(*) AS n FROM reports")["n"], 1)
        self.assertEqual({row["recipient"] for row in self.store.query("SELECT recipient FROM deliveries")}, {"2", "3"})

    def test_disabled_recipient_job_cannot_call_provider(self):
        self.store.add_user(2, enabled=False, subscriptions='all')
        job_id = self.enqueue("ai")
        self.worker.drain()
        self.assertEqual(self.providers.calls, [])
        self.assertEqual(self.store.one("SELECT state FROM jobs WHERE id=?", (job_id,))["state"], "failed")
        self.assertEqual(self.store.query("SELECT * FROM reports"), [])

    def test_partial_source_failure_preserves_other_metrics_and_hides_raw_error(self):
        self.providers.fail_phrase = "рост цен"
        self.enqueue("wordstat")
        self.worker.drain()
        report = self.store.one("SELECT * FROM reports")
        self.assertEqual(set(json.loads(report["metrics"])), {"обвал|225"})
        self.assertEqual(json.loads(report["quality"])["status"], "partial")
        self.assertNotIn("private-provider-response", report["full_text"] + report["quality"])

    def test_wordstat_auth_failure_is_visible_in_brief_report_without_response_text(self):
        from functions.wordstat_api import WordstatError
        self.store.set_keywords([{'phrase': f'фраза {i}'} for i in range(8)])
        self.providers.wordstat = Mock(side_effect=WordstatError('http', 403))
        self.enqueue('wordstat')
        self.worker.drain()
        report = self.store.one('SELECT * FROM reports')
        self.assertIn('HTTP 403', report['text'])
        self.assertIn('Client ID', report['full_text'])
        self.assertEqual(json.loads(report['quality'])['status'], 'partial')
        self.assertEqual(self.store.query("SELECT * FROM observations WHERE source='wordstat'"), [])
        self.assertNotIn('private-provider-response', report['text'] + report['full_text'])

    def test_changed_wordstat_schema_is_failure_not_empty_success(self):
        self.providers.wordstat = Mock(return_value={'results': [{'date': '2026-09-21', 'count': '100'}]})
        metadata = self.worker.collector.collect('wordstat')
        self.assertEqual(metadata['status'], 'partial')
        self.assertTrue(metadata['errors'])
        self.assertEqual(self.store.query("SELECT * FROM observations WHERE source='wordstat'"), [])

    def test_wordstat_source_failure_stops_remaining_phrases_and_pauses_regions(self):
        from functions.wordstat_api import WordstatError
        self.store.set_keywords([{'phrase': name} for name in ['обвал', 'рост цен', 'безработица']])
        self.providers.wordstat = Mock(side_effect=[{'dynamics': [{'date': '2026-09-21', 'count': 120}]},
                                                     WordstatError('tls')])
        metadata = self.worker.collector.collect('wordstat')
        self.assertEqual(metadata['status'], 'partial')
        self.assertEqual(self.providers.wordstat.call_count, 2)
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM observations WHERE source='wordstat'")['n'], 1)
        for section in ('wordstat', 'regions'):
            cached = self.worker.collector.collect(section)
            self.assertTrue(cached['cached'])
            self.assertEqual(cached['status'], 'partial')
            self.assertIn('SSL/TLS', cached['errors'][0])
        self.assertEqual(self.providers.wordstat.call_count, 2)
        self.assertEqual(self.worker.collector.collect('ruonia')['status'], 'complete')

    def test_wordstat_pause_expires_and_backend_switch_does_not_reuse_old_error(self):
        from functions.wordstat_api import WordstatError
        original = self.providers.wordstat
        self.providers.wordstat = Mock(side_effect=WordstatError('network', attempts=4))
        self.worker.collector.collect('wordstat')
        self.assertEqual(self.providers.wordstat.call_count, 1)
        self.providers.wordstat = Mock(side_effect=original)
        with patch.dict(os.environ, {'WORDSTAT_BACKEND': 'cloud'}):
            self.assertEqual(self.worker.collector.collect('wordstat')['status'], 'complete')
        self.assertEqual(self.providers.wordstat.call_count, 2)
        self.providers.wordstat.reset_mock()
        self.worker.collector.now = lambda: NOW + timedelta(minutes=6)
        self.assertEqual(self.worker.collector.collect('wordstat')['status'], 'complete')
        self.assertEqual(self.providers.wordstat.call_count, 2)

    def test_phrase_validation_error_does_not_stop_other_phrases(self):
        from functions.wordstat_api import WordstatError
        self.providers.wordstat = Mock(side_effect=[WordstatError('http', 400),
            {'dynamics': [{'date': '2026-09-21', 'count': 100}]}])
        metadata = self.worker.collector.collect('wordstat')
        self.assertEqual(metadata['status'], 'partial')
        self.assertEqual(self.providers.wordstat.call_count, 2)
        self.assertEqual(self.store.one("SELECT metric FROM observations WHERE source='wordstat'")['metric'], 'рост цен')
        self.assertFalse(self.store.get_setting('wordstat_pause:legacy'))

    def test_cli_dry_runs_leave_live_queue_and_deliveries_untouched(self):
        self.store.enqueue("report", {"section": "ai", "recipient": "2"})
        report_id = self.worker.builder.build("ruonia", {"status": "cached"}, charts=False)
        delivery_id = prepare_delivery(self.store, report_id, "2")[0]
        self.store.execute("UPDATE deliveries SET state='sending' WHERE id=?", (delivery_id,))
        jobs_before = self.store.query("SELECT * FROM jobs")
        deliveries_before = self.store.query("SELECT * FROM deliveries")
        for command in ("once", "serve"):
            with self.subTest(command=command), redirect_stdout(StringIO()), \
                    patch("thesaurus.cli.telegram_client") as telegram, \
                    patch.object(Worker, "recover") as recover:
                self.assertEqual(cli.run([command, "--dry-run", "--no-charts", "--db", str(self.path), "--section", "ruonia"],
                                         settings=self.settings, providers=self.providers), 0)
                telegram.assert_not_called()
                recover.assert_not_called()
            self.assertEqual(self.store.query("SELECT * FROM jobs"), jobs_before)
            self.assertEqual(self.store.query("SELECT * FROM deliveries"), deliveries_before)
        self.assertEqual(self.providers.calls, [])

    def test_recovery_requeues_running_jobs_but_never_replays_sending(self):
        job_id = self.enqueue("ai")
        self.store.execute("UPDATE jobs SET state='running' WHERE id=?", (job_id,))
        self.store.execute("INSERT INTO runs(id,section,status,started_at) VALUES('interrupted','ai','running',?)", (utcnow(),))
        report_id = self.worker.builder.build("ruonia", {"status": "cached"}, charts=False)
        delivery_id = prepare_delivery(self.store, report_id, "2")[0]
        self.store.execute("UPDATE deliveries SET state='sending' WHERE id=?", (delivery_id,))
        self.worker.recover()
        self.assertEqual(self.store.one("SELECT state FROM jobs WHERE id=?", (job_id,))["state"], "pending")
        self.assertEqual(self.store.one("SELECT status FROM runs WHERE id='interrupted'")["status"], "interrupted")
        self.assertEqual(self.store.one("SELECT state FROM deliveries WHERE id=?", (delivery_id,))["state"], "uncertain")
        self.assertEqual(self.providers.calls, [])

    def test_bootstrap_grants_only_configured_admins_and_preserves_schedules(self):
        bootstrap(self.store, self.settings)
        self.assertTrue(self.store.authorize(1, admin=True))
        self.assertFalse(self.store.authorize(2))
        self.assertTrue(all(row["enabled"] == 0 for row in self.store.query("SELECT enabled FROM subscriptions")))
        self.store.execute("UPDATE subscriptions SET enabled=1 WHERE recipient='2' AND section='ai'")
        bootstrap(self.store, self.settings)
        self.assertEqual(self.store.one("SELECT enabled FROM subscriptions WHERE recipient='2' AND section='ai'")["enabled"], 1)
        self.assertFalse(self.store.authorize(2))

    def test_store_reopen_preserves_data_and_versions_change_only_on_edit(self):
        version = self.store.version("keyword")
        self.store.set_keywords(self.store.keywords())
        self.assertEqual(self.store.version("keyword"), version)
        self.store.set_keywords([{"phrase": "new phrase", "theme": "test"}])
        reopened = Store(self.path)
        self.assertEqual(reopened.version("keyword"), version + 1)
        self.assertEqual(reopened.get_setting(f"keywords_history_{version + 1}"), reopened.keywords())
        with reopened.connection() as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 3)

    def test_unknown_legacy_schema_is_backed_up_without_overwrite(self):
        legacy = self.root / "legacy.sqlite3"
        with closing(sqlite3.connect(legacy)) as conn:
            conn.execute("CREATE TABLE original(value TEXT)")
            conn.execute("INSERT INTO original VALUES('preserve')")
            conn.commit()
        with self.assertRaisesRegex(RuntimeError, "backup"):
            Store(legacy)
        backups = list(self.root.glob("legacy.sqlite3.backup-*.sqlite3"))
        self.assertEqual(len(backups), 1)
        for path in (legacy, backups[0]):
            with closing(sqlite3.connect(path)) as conn:
                self.assertEqual(conn.execute("SELECT value FROM original").fetchone()[0], "preserve")

    def test_newer_database_version_is_rejected_without_reset(self):
        self.store.execute("PRAGMA user_version=99")
        with self.assertRaisesRegex(RuntimeError, "newer"):
            Store(self.path)
        with closing(sqlite3.connect(self.path)) as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 99)
        self.assertTrue(self.store.keywords())

    def test_provider_budget_reservation_is_atomic_between_workers(self):
        budget.set_budgets(self.store, {"ai_calls": 1})

        def reserve():
            try:
                return budget.reserve(self.store, "ai", now=NOW)
            except budget.BudgetExceeded:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _: reserve(), range(2)))
        self.assertEqual(sum(value is not None for value in outcomes), 1)
        self.assertEqual(self.store.one("SELECT SUM(calls) AS n FROM usage")["n"], 1)

    def test_budget_resets_by_moscow_day_and_failed_attempt_still_counts(self):
        budget.set_budgets(self.store, {"ai_calls": 1})
        before = datetime(2026, 10, 2, 20, 59, tzinfo=timezone.utc)
        after = datetime(2026, 10, 2, 21, 1, tzinfo=timezone.utc)
        first = budget.reserve(self.store, "ai", now=before)
        budget.finish(self.store, first, failed=True)
        with self.assertRaises(budget.BudgetExceeded):
            budget.reserve(self.store, "ai", now=before)
        self.assertIsInstance(budget.reserve(self.store, "ai", now=after), int)
        self.assertEqual({row["day"] for row in self.store.query("SELECT day FROM usage")}, {"2026-10-02", "2026-10-03"})

    def test_report_period_agrees_with_collector_moscow_day(self):
        late_utc = datetime(2026, 10, 3, 22, 30, tzinfo=timezone.utc)
        collector = Collector(self.store, self.settings, self.providers, now=lambda: late_utc)
        metadata = collector.collect("regions")
        builder = ReportBuilder(self.store, self.settings, now=lambda: late_utc)
        report_id = builder.build("regions", metadata, charts=False)
        report = self.store.one("SELECT * FROM reports WHERE id=?", (report_id,))
        self.assertEqual(report["period_end"], "2026-10-03")
        self.assertEqual({value["date"] for value in json.loads(report["metrics"]).values()}, {"2026-10-03"})


if __name__ == "__main__":
    unittest.main()
