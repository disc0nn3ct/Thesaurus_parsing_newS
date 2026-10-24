"""Regression cases found by an independent offline pipeline review."""

import shutil
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch
from uuid import uuid4

import pandas as pd

from thesaurus import budget
from thesaurus.collectors import Collector, Providers
from thesaurus.config import Settings
from thesaurus.reporting import ReportBuilder
from thesaurus.store import Store

NOW = datetime(2026, 10, 2, 16, tzinfo=timezone.utc)


class ReviewIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parent / f"review_test_{uuid4().hex}"
        self.folder.mkdir()
        self.addCleanup(self.cleanup)
        self.store = Store(self.folder / "state.sqlite3")
        self.settings = Settings(root=self.folder)
        self.builder = ReportBuilder(self.store, self.settings, now=lambda: NOW)

    def cleanup(self):
        assert self.folder.resolve().parent == Path(__file__).resolve().parent
        shutil.rmtree(self.folder)

    def test_reissued_forecast_matches_immutable_saved_prediction(self):
        history = [{"source": "wordstat", "metric": "test", "region_id": 225, "device": "all", "frequency": "weekly",
                    "date": date.date().isoformat(), "value": 100 + index, "unit": "count", "retrieved_at": date.isoformat()}
                   for index, date in enumerate(pd.date_range(end="2026-09-27", periods=40, freq="W-SUN", tz="UTC"))]
        original = self.builder._save_forecasts(history)
        self.assertEqual(len(original), 4)
        revised = [{**row, "value": row["value"] * 2, "retrieved_at": "2026-10-02T12:00:00Z"} for row in history]
        shown_again = self.builder._save_forecasts(revised)
        stored = self.store.query("SELECT prediction FROM forecasts ORDER BY horizon")
        self.assertEqual([row["prediction"] for row in shown_again], [row["prediction"] for row in stored])
        self.assertEqual([row["prediction"] for row in stored], [row["prediction"] for row in original])

    def mutating_provider(self):
        self.store.set_keywords([{"phrase": "old", "theme": "test"}])
        store = self.store

        class Provider:
            calls: ClassVar[list[str]] = []

            def wordstat(self, path, payload):
                self.calls.append(payload["phrase"])
                if len(self.calls) == 1:
                    store.set_keywords([{"phrase": "new", "theme": "test"}])
                return {"dynamics": [{"date": "2026-09-21", "count": 100}]}

        return Provider()

    def test_mid_collection_watchlist_edit_cannot_cache_old_data_as_new_version(self):
        provider = self.mutating_provider()
        collector = Collector(self.store, self.settings, provider, now=lambda: NOW)
        collector.collect("wordstat")
        collector.collect("wordstat")
        self.assertEqual(provider.calls, ["old", "new"])
        self.assertIsNotNone(self.store.one("SELECT id FROM observations WHERE metric='new'"))

    def test_mid_collection_watchlist_edit_preserves_observation_version(self):
        provider = self.mutating_provider()
        version_before = self.store.version("keyword")
        collector = Collector(self.store, self.settings, provider, now=lambda: NOW)
        collector.collect("wordstat")
        row = self.store.one("SELECT keyword_version FROM observations WHERE metric='old'")
        self.assertEqual(row["keyword_version"], version_before)

    def test_initial_cohort_discovery_preserves_concurrent_admin_selection(self):
        self.store.set_setting("incident_phrases", ["test"])
        store = self.store

        class Provider:
            regions_requested: ClassVar[list[int]] = []

            def wordstat(self, path, payload):
                if path == "/v1/regions":
                    store.set_regions([99])
                    return {"regions": [{"regionId": 1, "count": 1000}]}
                self.regions_requested.extend(payload["regions"])
                return {"dynamics": [{"date": "2026-10-01", "count": 100}]}

        provider = Provider()
        collector = Collector(self.store, self.settings, provider, now=lambda: NOW)
        collector.collect("regions")
        self.assertEqual(self.store.regions(), [99])
        self.assertEqual(provider.regions_requested, [99])
        self.assertEqual(self.store.one("SELECT cohort_version FROM observations")["cohort_version"], self.store.version("cohort"))

    def test_mid_collection_region_edit_does_not_relabel_old_observations(self):
        self.store.set_setting("incident_phrases", ["test"])
        self.store.set_regions([1])
        before = self.store.version("cohort")
        store = self.store

        class Provider:
            def wordstat(self, path, payload):
                store.set_regions([2])
                return {"dynamics": [{"date": "2026-10-01", "count": 100}]}

        collector = Collector(self.store, self.settings, Provider(), now=lambda: NOW)
        collector.collect("regions")
        row = self.store.one("SELECT region_id,cohort_version FROM observations")
        self.assertEqual(row, {"region_id": 1, "cohort_version": before})

    def test_partial_cache_expires_after_five_minutes(self):
        self.store.set_keywords([{"phrase": "test", "theme": "test"}])
        clock = [NOW]

        class Provider:
            calls = 0

            def wordstat(self, path, payload):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("offline failure")
                return {"dynamics": [{"date": "2026-09-21", "count": 100}]}

        provider = Provider()
        collector = Collector(self.store, self.settings, provider, now=lambda: clock[0])
        self.assertEqual(collector.collect("wordstat")["status"], "partial")
        clock[0] = NOW + timedelta(seconds=299)
        self.assertTrue(collector.collect("wordstat")["cached"])
        self.assertEqual(provider.calls, 1)
        clock[0] = NOW + timedelta(seconds=300)
        metadata = collector.collect("wordstat")
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["retrieved_at"], clock[0].isoformat())
        self.assertEqual(provider.calls, 2)

    def test_ruonia_success_cache_expires_after_thirty_minutes(self):
        clock = [NOW]
        workbook = pd.DataFrame({"Дата": ["01.10.2026"], "Индекс": [15], "1 месяц": [15], "3 месяца": [15], "6 месяцев": [15]})
        provider = Mock()
        provider.ruonia.return_value = workbook
        collector = Collector(self.store, self.settings, provider, now=lambda: clock[0])
        self.assertFalse(collector.collect("ruonia")["cached"])
        clock[0] = NOW + timedelta(seconds=1799)
        self.assertTrue(collector.collect("ruonia")["cached"])
        provider.ruonia.assert_called_once()
        clock[0] = NOW + timedelta(seconds=1800)
        self.assertFalse(collector.collect("ruonia")["cached"])
        self.assertEqual(provider.ruonia.call_count, 2)

    def test_regions_success_cache_expires_after_sixty_minutes(self):
        self.store.set_regions([1])
        self.store.set_setting("incident_phrases", ["test"])
        clock = [NOW]
        provider = Mock()
        provider.wordstat.return_value = {"dynamics": [{"date": "2026-10-01", "count": 100}]}
        collector = Collector(self.store, self.settings, provider, now=lambda: clock[0])
        self.assertFalse(collector.collect("regions")["cached"])
        clock[0] = NOW + timedelta(seconds=3599)
        self.assertTrue(collector.collect("regions")["cached"])
        provider.wordstat.assert_called_once()
        clock[0] = NOW + timedelta(seconds=3600)
        self.assertFalse(collector.collect("regions")["cached"])
        self.assertEqual(provider.wordstat.call_count, 2)

    def test_ruonia_provider_date_bounds_use_moscow_day(self):
        timestamp = datetime(2026, 10, 3, 22, 30, tzinfo=timezone.utc)
        expected = pd.DataFrame()
        with patch("thesaurus.collectors.datetime", wraps=datetime) as clock, \
             patch("thesaurus.cbr_incremental.collect_report", return_value=expected) as collect, \
             patch("thesaurus.market_sources.Transport"):
            clock.now.return_value = timestamp
            self.assertIs(Providers(self.store).ruonia(), expected)
        self.assertEqual(collect.call_args.args[2].isoformat(), "2026-10-04")
        self.assertEqual(collect.call_args.args[3], timestamp)

    def test_rejected_ai_proposal_retains_reported_paid_usage(self):
        budget.set_budgets(self.store, {"prices": {"ai": {"input_per_million": 10, "output_per_million": 20}}})
        client = Mock()
        fake_ai = ModuleType("functions.ai")
        fake_ai.get_client = lambda: client
        fake_ai.retry_call = lambda call: call()
        fake_ai.MODEL = "offline-test-model"
        for content in ("invalid JSON", "[]", '{"phrase": "wrong container"}'):
            with self.subTest(content=content):
                client.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
                )
                with patch.dict(sys.modules, {"functions.ai": fake_ai}), self.assertRaises(ValueError):
                    Providers(self.store).propose_keywords([])
                usage = self.store.one("SELECT * FROM usage ORDER BY id DESC LIMIT 1")
                self.assertEqual(usage["status"], "failed")
                self.assertEqual(usage["tokens_in"], 100)
                self.assertEqual(usage["tokens_out"], 50)
                self.assertAlmostEqual(usage["cost"], .002)

    def test_malformed_news_publication_date_does_not_leave_partial_report(self):
        metadata = {"status": "complete", "ai": {"text": "Offline text", "data": {"claims": [
            {"title": "Example claim", "claim": "Unverified claim", "url": "https://example.com/article",
             "published_at": {"year": 2026}, "numbers": []},
        ]}}}
        report_id = self.builder.build("ai", metadata, charts=False)
        self.assertIsNotNone(self.store.one("SELECT id FROM reports WHERE id=?", (report_id,)))
        rows = self.store.query("SELECT published_at FROM news WHERE report_id=?", (report_id,))
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["published_at"])
        self.assertEqual(self.builder.build("ai", metadata, charts=False), report_id)


if __name__ == "__main__":
    unittest.main()
