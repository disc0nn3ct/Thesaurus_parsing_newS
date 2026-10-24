import json
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from thesaurus.analysis import (
    AlertConfig,
    classify_news_changes,
    compare_periods,
    cooldown_allowed,
    deduplicate_news,
    evaluate_alerts,
    evaluate_forecasts,
    filter_observations,
    forecast_weekly,
    quality_summary,
    rolling_backtest,
    safe_probe_url,
    verify_numeric_claims,
)


def rows(values, start="2025-01-06", frequency="daily", **overrides):
    dates = pd.date_range(start, periods=len(values), freq="7D" if frequency == "weekly" else "D")
    return [{"source": "wordstat", "metric": "пожар", "region_id": 225, "device": "all",
             "frequency": frequency, "date": day.date().isoformat(), "value": value,
             "unit": "count", "retrieved_at": (day + pd.Timedelta(hours=18)).isoformat(),
             "quality": "ok" if value is not None else "missing", **overrides}
            for day, value in zip(dates, values)]


class QualityTests(unittest.TestCase):
    def test_missing_and_unrequested_regions_do_not_become_zero(self):
        observations = rows([0, None, 20])
        quality = quality_summary(observations, "2025-01-06", "2025-01-08", queries=["пожар"], region_ids=[225, 1], as_of="2025-01-09T18:00:00Z")
        self.assertEqual(quality["expected_points"], 6)
        self.assertEqual(quality["observed_points"], 2)
        self.assertEqual(quality["missing_points"], 4)
        self.assertAlmostEqual(quality["coverage"], 1 / 3)
        self.assertEqual(quality["freshness_hours"], 24)
        self.assertFalse(quality["missing_is_zero"])
        json.dumps(quality, allow_nan=False)

    def test_latest_revision_replaces_previous_value(self):
        observations = rows([10]) + rows([None], retrieved_at="2025-01-07T00:00:00Z")
        filtered = filter_observations(observations)
        self.assertEqual(len(filtered), 1)
        self.assertTrue(pd.isna(filtered.iloc[0]["count"]))

    def test_comparison_uses_same_regions_and_day_offsets(self):
        observations = rows([10, None], start="2025-01-01") + rows([20, 100], start="2025-01-08")
        observations += rows([1000, 1000], start="2025-01-08", region_id=1)
        result = compare_periods(observations, "2025-01-08", "2025-01-09", "2025-01-01", "2025-01-02")
        self.assertEqual(result["matched_points"], 1)
        self.assertEqual(result["groups"][0]["current_total"], 20)
        self.assertEqual(result["groups"][0]["previous_total"], 10)
        self.assertEqual(result["groups"][0]["percent_change"], 100)

    def test_incomparable_devices_are_not_merged(self):
        observations = rows([10], start="2025-01-01", device="desktop") + rows([20], start="2025-01-08", device="phone")
        result = compare_periods(observations, "2025-01-08", "2025-01-08", "2025-01-01", "2025-01-01")
        self.assertEqual(result["matched_points"], 0)

    def test_unequal_periods_rejected(self):
        with self.assertRaises(ValueError):
            compare_periods(rows([1]), "2025-01-08", "2025-01-09", "2025-01-01", "2025-01-01")


class AlertTests(unittest.TestCase):
    def test_eight_prior_week_sums_and_current_week_trigger(self):
        report = evaluate_alerts(rows([10] * 56 + [20] * 7))[0]
        self.assertTrue(report["alert"])
        self.assertEqual(report["baseline_mean"], 70)
        self.assertEqual(report["current_total"], 140)
        self.assertEqual(report["percent_change"], 100)
        self.assertTrue(report["constant_baseline"])
        self.assertIsNone(report["z_score"])
        json.dumps(report, allow_nan=False)

    def test_missing_day_suppresses_alert_not_counted_zero(self):
        result = evaluate_alerts(rows([10] * 56 + [20] * 6 + [None]))[0]
        self.assertFalse(result["alert"])
        self.assertEqual(result["reason"], "insufficient_coverage")

    def test_insufficient_baseline_and_low_volume_suppressed(self):
        self.assertFalse(evaluate_alerts(rows([10] * 60))[0]["alert"])
        result = evaluate_alerts(rows([1] * 56 + [3] * 7))[0]
        self.assertEqual(result["reason"], "insufficient_baseline")

    def test_configurable_threshold_and_cooldown_boundaries(self):
        self.assertFalse(evaluate_alerts(rows([10] * 56 + [20] * 7), config=AlertConfig(percent_threshold=150))[0]["alert"])
        self.assertFalse(cooldown_allowed("2025-01-01T12:00Z", "2025-01-02T11:59Z"))
        self.assertTrue(cooldown_allowed("2025-01-01T12:00Z", "2025-01-02T12:00Z"))


class ForecastTests(unittest.TestCase):
    def test_four_week_issued_forecast_and_store_fields(self):
        history = rows(list(range(100, 140)), frequency="weekly")
        created = pd.Timestamp(history[-1]["date"]) + pd.Timedelta(hours=23)
        forecasts = forecast_weekly(history, created_at=created)
        self.assertEqual(len(forecasts), 4)
        self.assertEqual(forecasts[0]["metric"], "пожар")
        self.assertEqual(forecasts[0]["baseline"], 139)
        self.assertEqual(forecasts[-1]["horizon"], 4)
        self.assertTrue(all(row["lower"] <= row["prediction"] <= row["upper"] for row in forecasts))
        json.dumps(forecasts, allow_nan=False)

    def test_future_points_cannot_change_prior_prediction(self):
        history = rows(list(range(100, 145)), frequency="weekly")
        origin = history[35]["date"]
        first = forecast_weekly(history, origin_date=origin, created_at=origin + "T23:00:00Z")
        for record in history[36:]:
            record["value"] = 1_000_000
        second = forecast_weekly(history, origin_date=origin, created_at=origin + "T23:00:00Z")
        self.assertEqual(first, second)

    def test_insufficient_or_gapped_history_returns_no_forecast(self):
        self.assertEqual(forecast_weekly(rows([100] * 29, frequency="weekly")), [])
        history = rows([100] * 40, frequency="weekly")
        for record in history[20:25]:
            record["value"] = None
        self.assertEqual(forecast_weekly(history), [])

    def test_backtest_labels_and_metrics(self):
        history = rows([100] * 42, frequency="weekly")
        result = rolling_backtest(history, max_origins=5)
        self.assertGreater(result["metrics"]["n"], 0)
        self.assertAlmostEqual(result["metrics"]["mae"], 0, places=8)
        self.assertEqual(result["metrics"]["baseline_mae"], 0)
        self.assertEqual(result["metrics"]["interval_coverage"], 1)
        self.assertEqual(evaluate_forecasts(result["forecasts"], history)["n"], 0)

    def test_late_saved_forecasts_do_not_count_as_issued(self):
        history = rows([100] * 42, frequency="weekly")
        origin = history[35]["date"]
        forecast = forecast_weekly(history, origin_date=origin, created_at=origin + "T23:00:00Z")
        forecast[0]["created_at"] = forecast[0]["target_date"] + "T01:00:00Z"
        result = evaluate_forecasts(forecast, history)
        self.assertEqual(result["n"], 3)

    def test_evaluate_raw_store_forecasts(self):
        history = rows([100] * 42, frequency="weekly")
        origin = history[35]["date"]
        forecasts = forecast_weekly(history, origin_date=origin, created_at=origin + "T23:00:00Z")
        columns = {"source", "metric", "region_id", "device", "frequency", "origin_date", "target_date", "horizon", "prediction", "lower", "upper", "baseline", "created_at", "model"}
        stored = [{key: value for key, value in row.items() if key in columns} for row in forecasts]
        self.assertEqual(evaluate_forecasts(stored, history)["n"], 4)


class NewsAndProvenanceTests(unittest.TestCase):
    def test_tracking_urls_and_title_duplicates_are_merged(self):
        articles = [
            {"url": "https://EXAMPLE.com/news?utm_source=test&id=1#top", "title": "Пожар: новости!", "published_at": "2025-01-01", "source": "one"},
            {"url": "https://example.com/news?id=1", "title": "Пожар: новости!", "published_at": "2025-01-01", "source": "one"},
            {"url": "https://example.org/article", "title": "ПОЖАР новости", "published_at": "2025-01-01", "source": "two"},
        ]
        result = deduplicate_news(articles)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["sources"], ["one", "two"])
        self.assertEqual(len(result[0]["alternate_urls"]), 2)
        self.assertEqual(len(result[0]["fingerprint"]), 64)

    def test_correction_keeps_old_content(self):
        result = deduplicate_news([
            {"url": "https://example.com/article", "title": "Первый заголовок", "fetched_at": "2025-01-01"},
            {"url": "https://example.com/article", "title": "Исправленный заголовок", "fetched_at": "2025-01-02"},
        ])
        self.assertEqual(result[0]["title"], "Исправленный заголовок")
        self.assertEqual(result[0]["corrections"][0]["title"], "Первый заголовок")

    def test_claim_corrections_and_cross_report_classification(self):
        old = {"url": "https://example.com/article", "title": "Новости", "claim": "Ранее указано 10", "fetched_at": "2025-01-01"}
        new = {**old, "claim": "Исправлено на 12", "fetched_at": "2025-01-02"}
        item = deduplicate_news([old, new])[0]
        self.assertEqual(item["claim"], "Исправлено на 12")
        self.assertEqual(item["corrections"][0]["claim"], "Ранее указано 10")
        result = classify_news_changes([new, {"url": "https://example.com/another", "title": "Другая новость"}], [old])
        self.assertEqual(len(result["corrected"]), 1)
        self.assertEqual(len(result["new"]), 1)
        self.assertEqual(len(classify_news_changes([old], [old])["repeated"]), 1)

    def test_numeric_check_does_not_claim_full_verification(self):
        reference = {"value": 30, "metric": "sum", "query": "пожар", "region_id": 225,
                     "start_date": "2025-01-06", "end_date": "2025-01-07", "unit": "count"}
        result = verify_numeric_claims([{"text": "Сумма 30, причина неизвестна", "numbers": [reference]}], rows([10, 20]))[0]
        self.assertTrue(result["numbers_verified"])
        self.assertFalse(result["claim_verified"])
        self.assertFalse(result["url_availability_is_claim_verification"])
        reference["value"] = 99
        self.assertFalse(verify_numeric_claims([{"numbers": [reference]}], rows([10, 20]))[0]["numbers_verified"])

    def test_numeric_check_rejects_ambiguous_device_frequency(self):
        observations = rows([10, 20]) + rows([20, 40], device="phone")
        reference = {"value": 90, "metric": "sum", "query": "пожар", "region_id": 225,
                     "start_date": "2025-01-06", "end_date": "2025-01-07", "unit": "count"}
        self.assertFalse(verify_numeric_claims([{"numbers": [reference]}], observations)[0]["numbers_verified"])

    def test_malformed_ai_dates_are_unverified_without_crashing(self):
        for value in (None, "not-a-date", {}, float("nan")):
            with self.subTest(value=value):
                reference = {"value": value, "metric": "sum", "query": "пожар", "region_id": 225,
                             "start_date": value, "end_date": "2025-01-07", "unit": "count"}
                result = verify_numeric_claims([{"numbers": [reference]}], rows([10, 20]))
                self.assertFalse(result[0]["numbers_verified"])
                json.dumps(result, allow_nan=False)

    def test_partial_sum_is_not_verified_as_complete(self):
        reference = {"value": 10, "metric": "sum", "query": "пожар", "region_id": 225,
                     "start_date": "2025-01-06", "end_date": "2025-01-07", "unit": "count"}
        result = verify_numeric_claims([{"numbers": [reference]}], rows([10]))[0]
        self.assertFalse(result["numbers_verified"])

    def test_claim_without_url_retains_database_safe_empty_string(self):
        self.assertEqual(deduplicate_news([{"title": "Claim"}])[0]["url"], "")

    def test_probe_is_off_by_default_and_private_addresses_blocked(self):
        with patch("thesaurus.analysis.socket.getaddrinfo") as resolve, patch("thesaurus.analysis.socket.create_connection") as connect:
            self.assertEqual(safe_probe_url("https://example.com")["availability"], "not_checked")
            resolve.assert_not_called()
            resolve.return_value = [(2, 1, 6, "", ("127.0.0.1", 443))]
            self.assertEqual(safe_probe_url("https://example.com", enabled=True)["availability"], "blocked")
            connect.assert_not_called()

    def test_probe_redirect_to_private_network_is_blocked(self):
        public = [(2, 1, 6, "", ("93.184.216.34", 80))]
        private = [(2, 1, 6, "", ("10.0.0.1", 80))]
        response = Mock(status=302)
        response.getheader.return_value = "http://internal.example/"
        with patch("thesaurus.analysis.socket.getaddrinfo", side_effect=[public, private]), \
             patch("thesaurus.analysis.socket.create_connection") as connect, \
             patch("thesaurus.analysis.http.client.HTTPConnection") as connection:
            connection.return_value.getresponse.return_value = response
            result = safe_probe_url("http://example.com", enabled=True)
            self.assertEqual(result["availability"], "blocked")
            connect.assert_called_once_with(("93.184.216.34", 80), timeout=5.0)


if __name__ == "__main__":
    unittest.main()
