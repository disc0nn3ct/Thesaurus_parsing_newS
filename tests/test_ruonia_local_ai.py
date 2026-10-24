"""Offline checks of CBR identities, daily forecasts and free local explanations."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from thesaurus.cbr_data import parse_day, parse_rates
from thesaurus.collectors import Collector, Providers, ruonia_rows
from thesaurus.config import Settings
from thesaurus.local_ai import chat, overview
from thesaurus.rate_forecasts import backtest_daily, forecast_daily, working_dates
from thesaurus.reporting import ReportBuilder
from thesaurus.store import Store
from thesaurus.telegram_data import DataControls


class RuoniaLocalTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / ".test-tmp"
        parent.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = Store(self.root / "state.sqlite3")
        self.settings = Settings(root=self.root)

    def history(self):
        days = working_dates("2026-01-01", "2026-09-30")
        return [{"source": "ruonia", "metric": "RUONIA", "frequency": "daily", "date": day.date().isoformat(),
                 "value": 15 + (i % 5) * .01, "unit": "%", "retrieved_at": day.isoformat()}
                for i, day in enumerate(days)]

    def test_index_is_not_rate_and_volume_has_separate_unit(self):
        frame = pd.DataFrame({"Дата": ["2026-09-30"], "Индекс": [200], "1 месяц": [15],
                              "3 месяца": [16], "6 месяцев": [17], "Ставка RUONIA": [14], "Объём RUONIA": [400]})
        rows = {row["metric"]: row for row in ruonia_rows(frame)}
        self.assertEqual((rows["RUONIA"]["value"], rows["RUONIA"]["unit"]), (14, "%"))
        self.assertEqual((rows["Индекс RUONIA"]["value"], rows["Индекс RUONIA"]["unit"]), (200, "индекс"))
        self.assertEqual(rows["Объём RUONIA"]["unit"], "млрд руб")

    def test_public_xml_and_invalid_response(self):
        xml = b'<root><ro><D0>2026-09-01T00:00:00+03:00</D0><ruo>13.84</ruo><vol>404.4</vol></ro></root>'
        self.assertEqual(parse_rates(xml).iloc[0]["Ставка RUONIA"], 13.84)
        for bad in (b'<root/>', b'<!DOCTYPE foo><root/>', b'<broken>'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_rates(bad)

    def test_iso_dates_are_not_month_day_swapped(self):
        for value in ("2026-10-02", "02.10.2026", pd.Timestamp("2026-10-02")):
            self.assertEqual(parse_day(value).date().isoformat(), "2026-10-02")

    def test_daily_forecast_calendar_and_future_cutoff(self):
        history = self.history()
        created = "2026-10-01T10:00:00Z"
        expected = forecast_daily(history, created_at=created)
        self.assertEqual(len(expected), 4)
        future = history + [{**history[-1], "date": "2099-01-01", "value": 1e9}]
        self.assertEqual(expected, forecast_daily(future, created_at=created, origin_date="2026-09-30"))
        for row in expected:
            self.assertEqual(row["frequency"], "daily")
            self.assertLess(row["origin_date"], row["target_date"])
            self.assertLessEqual(row["lower"], row["prediction"])
            self.assertLessEqual(row["prediction"], row["upper"])
            self.assertTrue(working_dates(row["target_date"], row["target_date"]))

    def test_missing_or_future_collected_data_cannot_produce_forecast(self):
        history = self.history()
        self.assertEqual(forecast_daily(history[-12:], created_at="2026-10-01T10:00:00Z"), [])
        revised = [{**row, "retrieved_at": "2099-01-01T00:00:00Z"} for row in history]
        self.assertEqual(forecast_daily(revised, created_at="2026-10-01T10:00:00Z"), [])
        with self.assertRaises(ValueError):
            forecast_daily(history, created_at="2026-01-01T00:00:00Z")

    def test_publication_lag_uses_last_actual_not_null_outer_join(self):
        history = self.history()
        history.append({**history[-1], "date": "2026-10-01", "value": None})
        forecasts = forecast_daily(history, created_at="2026-10-02T10:00:00Z")
        self.assertEqual(len(forecasts), 4)
        self.assertEqual({row["origin_date"] for row in forecasts}, {"2026-09-30"})

    def test_backtest_and_immutable_saved_forecasts(self):
        history = self.history()
        result = backtest_daily(history, max_origins=3)
        self.assertGreater(result["metrics"]["n"], 0)
        self.assertTrue(all(row["kind"] == "backtest" for row in result["forecasts"]))
        self.store.save_observations(history)
        builder = ReportBuilder(self.store, self.settings, now=lambda: datetime(2026, 10, 1, 10, tzinfo=timezone.utc))
        forecasts = builder._save_forecasts(self.store.query("SELECT * FROM observations"))
        self.assertEqual(len(forecasts), 4)
        revised = [{**row, "value": row["value"] + 2} for row in self.store.query("SELECT * FROM observations")]
        self.assertEqual(forecasts, builder._save_forecasts(revised))

    def test_legacy_index_repair_backed_up_and_idempotent(self):
        self.store.save_observations([{**self.history()[-1], "value": 200, "source_url": "https://cbr.ru/hd_base/ruonia/"}])
        collector = Collector(self.store, self.settings, Mock())
        collector._repair_legacy_index()
        collector._repair_legacy_index()
        row = self.store.one("SELECT * FROM observations")
        self.assertEqual((row["metric"], row["unit"], row["value"]), ("Индекс RUONIA", "индекс", 200))
        self.assertEqual(len(list(self.root.glob("*.bak"))), 1)

    def test_template_never_calls_network_or_paid_provider(self):
        with patch.dict(os.environ, {"AI_BACKEND": "template"}), patch("requests.post") as post, \
                patch("functions.ai.get_client") as paid:
            text, _, data = Providers(self.store).ai(self.history()[-2:])
        post.assert_not_called()
        paid.assert_not_called()
        self.assertIn("RUONIA", text)
        self.assertEqual(data["backend"], "template")
        self.assertEqual(self.store.query("SELECT * FROM usage"), [])

    def test_ollama_request_small_and_no_paid_fallback(self):
        response = Mock(status_code=200)
        response.json.return_value = {"message": {"content": "Локальный обзор"}}
        with patch.dict(os.environ, {"AI_BACKEND": "ollama", "OLLAMA_URL": "http://127.0.0.1:11434"}), \
                patch("requests.post", return_value=response) as post:
            _, _, data = overview(self.history()[-2:])
        body = post.call_args.kwargs["json"]
        self.assertFalse(body["think"])
        self.assertEqual(body["options"]["num_ctx"], 2048)
        self.assertEqual(data["backend"], "ollama")
        with patch.dict(os.environ, {"AI_BACKEND": "ollama"}), \
                patch("requests.post", side_effect=requests.Timeout()), patch("functions.ai.get_client") as paid:
            _, _, data = overview([])
        paid.assert_not_called()
        self.assertTrue(data["fallback"])

    def test_local_endpoint_rejects_remote_or_redirect(self):
        with patch.dict(os.environ, {"OLLAMA_URL": "https://remote.example"}), patch("requests.post") as post, \
                self.assertRaises(ValueError):
            chat("test")
        post.assert_not_called()
        response = Mock(status_code=302)
        with patch.dict(os.environ, {"OLLAMA_URL": "http://127.0.0.1:11434"}), \
                patch("requests.post", return_value=response), self.assertRaises(ValueError):
            chat("test")

    def test_bot_ruonia_forecasts_and_scoped_backtest(self):
        self.store.add_user(2, subscriptions='all')
        self.store.save_observations(self.history())
        reply = Mock()
        bot = DataControls(self.store, reply)
        bot.dispatch(Mock(), 2, "forecasts", "ruonia")
        self.assertIn("ruonia", reply.call_args.args[2])
        bot.dispatch(Mock(), 2, "backtest", "ruonia 3", 10)
        payload = json.loads(self.store.one("SELECT payload FROM jobs")["payload"])
        self.assertEqual(payload["source"], "ruonia")
        self.assertEqual(payload["frequency"], "daily")


if __name__ == "__main__":
    unittest.main()
