"""Dashboard integration tests use temporary SQLite and never open API clients."""

import csv
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from thesaurus.config import Settings
from thesaurus.dashboard import COOKIE_NAME, create_app
from thesaurus.store import Store, dumps, utcnow


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="thesaurus-dashboard-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = Settings(root=self.root, dashboard_enabled=True)
        self.store = Store(self.root / "data" / "test.sqlite3")
        self.password = "test-" + "dashboard-password"
        self.app = create_app(self.store, self.settings, password=self.password)
        self.client = TestClient(self.app, base_url="http://127.0.0.1", client=("127.0.0.1", 44001), follow_redirects=False)
        self.addCleanup(self.client.close)

    def csrf(self, text):
        return re.search(r'name="csrf" value="([^"]+)"', text).group(1)

    def login(self):
        response = self.client.get("/login")
        token = self.csrf(response.text)
        result = self.client.post("/login", data={"csrf": token, "password": self.password}, headers={"Origin": "http://127.0.0.1"})
        self.assertEqual(result.status_code, 303, result.text)
        return self.csrf(self.client.get("/").text)

    def post(self, path, values):
        return self.client.post(path, data={"csrf": self.token, **values}, headers={"Origin": "http://127.0.0.1"})

    def add_report(self, parts=None, text="Текст отчёта"):
        self.store.execute("INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,parts) VALUES(?,?,?,?,?,?,?,?)",
                           ("sample", "wordstat", utcnow(), 1, 0, text, text, dumps(parts or [])))

    def test_missing_password_refuses_to_create_app(self):
        with self.assertRaisesRegex(ValueError, "DASHBOARD_PASSWORD"):
            create_app(self.store, self.settings, password="")

    def test_disabled_dashboard_refuses_factory_even_with_password(self):
        self.settings.dashboard_enabled = False
        with self.assertRaisesRegex(ValueError, "DASHBOARD_ENABLED"):
            create_app(self.store, self.settings, password=self.password)

    def test_reads_require_login_and_cookie_is_rotated(self):
        self.assertEqual(self.client.get("/").headers["location"], "/login")
        response = self.client.get("/login")
        before = self.client.cookies.get(COOKIE_NAME)
        response = self.client.post("/login", data={"csrf": self.csrf(response.text), "password": self.password}, headers={"Origin": "http://127.0.0.1"})
        after = self.client.cookies.get(COOKIE_NAME)
        self.assertNotEqual(before, after)
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertIn("SameSite=strict", response.headers["set-cookie"])
        self.assertNotIn(self.password, after)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_remote_client_and_unexpected_host_are_rejected(self):
        with TestClient(self.app, base_url="http://127.0.0.1", client=("192.0.2.10", 10)) as remote:
            self.assertEqual(remote.get("/login").status_code, 403)
        for host in ("attacker.example", "127.0.0.1.attacker.example", "evil@127.0.0.1", "127.0.0.1/anything"):
            self.assertEqual(self.client.get("/login", headers={"host": host}).status_code, 403)

    def test_csrf_and_origin_required_even_for_login(self):
        token = self.csrf(self.client.get("/login").text)
        for data, origin in (({"password": self.password}, "http://127.0.0.1"),
                             ({"password": self.password, "csrf": token}, "https://evil.invalid")):
            self.assertEqual(self.client.post("/login", data=data, headers={"Origin": origin}).status_code, 403)

    def test_incidental_requests_do_not_invalidate_login_form(self):
        response = self.client.get("/login")
        token = self.csrf(response.text)
        cookie = self.client.cookies.get(COOKIE_NAME)
        self.assertEqual(response.headers["referrer-policy"], "same-origin")
        self.assertEqual(self.client.get("/favicon.ico").status_code, 204)
        self.assertEqual(self.csrf(self.client.get("/login").text), token)
        self.assertEqual(self.client.cookies.get(COOKIE_NAME), cookie)
        result = self.client.post("/login", data={"csrf": token, "password": self.password}, headers={"Origin": "http://127.0.0.1"})
        self.assertEqual(result.status_code, 303)

    def test_all_pages_render_on_empty_database(self):
        self.login()
        for path in ("/", "/history", "/compare", "/reports", "/forecasts", "/jobs", "/subscriptions", "/settings"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn("Thesaurus", response.text)
                self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
                self.assertEqual(response.headers["cache-control"], "no-store")

    def test_queue_action_only_writes_job_and_rejects_csrf(self):
        self.token = self.login()
        denied = self.client.post("/actions/report", data={"section": "ai"}, headers={"Origin": "http://127.0.0.1"})
        self.assertEqual(denied.status_code, 403)
        self.assertIsNone(self.store.one("SELECT id FROM jobs"))
        response = self.post("/actions/report", {"section": "wordstat"})
        self.assertEqual(response.status_code, 303)
        row = self.store.one("SELECT * FROM jobs")
        self.assertEqual(row["state"], "pending")
        self.assertEqual(json.loads(row["payload"]), {"section": "wordstat"})
        self.assertEqual(self.post("/actions/report", {"section": "unknown"}).status_code, 400)

    def test_explicit_broadcast_enqueues_only_active_subscriptions(self):
        self.token = self.login()
        self.assertEqual(self.post("/actions/report", {"section": "wordstat", "recipient": "*"}).status_code, 400)
        for user_id in (123, 456):
            self.post("/actions/user", {"user_id": str(user_id), "role": "subscriber", "enabled": "1"})
        self.store.execute("UPDATE subscriptions SET enabled=1,detail='full',phrases=? WHERE user_id=123 AND section='wordstat'", (dumps(["рост цен"]),))
        self.assertEqual(self.post("/actions/report", {"section": "wordstat", "recipient": "*"}).status_code, 303)
        jobs = self.store.query("SELECT payload FROM jobs")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(json.loads(jobs[0]["payload"])["recipient"], "123")
        self.assertEqual(json.loads(jobs[0]["payload"])["phrases"], ["рост цен"])

    def test_forecast_scores_exclude_late_predictions_and_show_coverage(self):
        self.login()
        for origin, created, predicted in (("2026-01-01", "2026-01-02T10:00:00Z", 12), ("2026-01-02", "2026-02-01T10:00:00Z", 99999)):
            self.store.execute("INSERT INTO forecasts(source,metric,region_id,origin_date,target_date,horizon,prediction,baseline,actual,lower,upper,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                               ("wordstat", "рост цен", 225, origin, "2026-01-11", 1, predicted, 11, 10, 9, 13, created))
        response = self.client.get("/forecasts")
        self.assertEqual(response.status_code, 200)
        self.assertIn("20.0% / 10.0%", response.text)
        self.assertIn("100.0%", response.text)

    def test_source_verification_is_queued_and_status_is_separate(self):
        self.token = self.login()
        self.add_report()
        self.store.execute("UPDATE reports SET news=? WHERE id='sample'", (dumps([{"title": "Example", "url": "https://example.org/report", "status": "unverified", "source_availability": {"availability": "available"}}]),))
        response = self.client.get("/reports/sample")
        self.assertIn("Не подтверждено", response.text)
        self.assertIn("Доступна", response.text)
        response = self.post("/actions/reports/sample/verify-sources", {})
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.store.one("SELECT kind FROM jobs")["kind"], "verify_sources")

    def test_user_access_is_numeric_and_new_subscriptions_are_disabled(self):
        self.token = self.login()
        self.assertEqual(self.post("/actions/user", {"user_id": "@name", "role": "subscriber", "enabled": "1"}).status_code, 400)
        response = self.post("/actions/user", {"user_id": "123", "role": "subscriber", "enabled": "1"})
        self.assertEqual(response.status_code, 303)
        self.assertTrue(self.store.authorize(123))
        rows = self.store.query("SELECT * FROM subscriptions WHERE user_id=123")
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["enabled"] == 0 for row in rows))

    def test_subscription_uses_shared_validation(self):
        self.token = self.login()
        self.store.set_regions([1, 225])
        self.post("/actions/user", {"user_id": "123", "role": "subscriber", "enabled": "1"})
        row = self.store.one("SELECT * FROM subscriptions WHERE user_id=123 AND section='wordstat'")
        values = {"subscription_id": str(row["id"]), "enabled": "1", "detail": "full", "times": "09:00, 18:30",
                  "weekdays": "1, 3, 7", "phrases": "рост цен\nбезработица", "regions": "225"}
        self.assertEqual(self.post("/actions/subscription", values).status_code, 303)
        saved = self.store.one("SELECT * FROM subscriptions WHERE id=?", (row["id"],))
        self.assertEqual(json.loads(saved["weekdays"]), [0, 2, 6])
        self.assertEqual(json.loads(saved["regions"]), [225])
        self.assertEqual(self.post("/actions/subscription", {**values, "regions": "1"}).status_code, 400)
        self.assertEqual(self.post("/actions/subscription", {**values, "times": "25:99"}).status_code, 400)
        self.assertEqual(self.client.get("/subscriptions?edit=" + str(row["id"])).status_code, 200)

    def test_chart_separates_frequency_and_rejects_mixed_units(self):
        self.login()
        self.store.save_observations([
            {"source": "wordstat", "metric": "цены", "frequency": "daily", "date": "2026-01-01", "value": 5, "unit": "count"},
            {"source": "wordstat", "metric": "цены", "frequency": "weekly", "date": "2026-01-01", "value": 35, "unit": "count"},
            {"source": "ruonia", "metric": "RUONIA", "frequency": "daily", "date": "2026-01-01", "value": 20, "unit": "%"},
        ])
        chart = self.client.get("/api/chart?source=wordstat").json()
        self.assertEqual(len(chart["traces"]), 2)
        self.assertEqual(self.client.get("/api/chart").json()["traces"], [])
        self.assertEqual(self.client.get("/history?date_from=not-a-date").status_code, 400)

    def test_download_only_registered_files_inside_reports_root(self):
        self.login()
        self.settings.reports_dir.mkdir(parents=True)
        inside = self.settings.reports_dir / "report.txt"
        inside.write_text("Expected report", encoding="utf-8")
        outside = self.root / "private.txt"
        outside.write_text("Must stay private", encoding="utf-8")
        self.add_report(parts=[{"kind": "document", "path": str(inside)}, {"kind": "document", "path": str(outside)},
                               {"kind": "document", "path": "../private.txt"}])
        response = self.client.get("/reports/sample/files/0")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "Expected report")
        self.assertIn("attachment", response.headers["content-disposition"])
        for index in (1, 2, -1, 20):
            self.assertEqual(self.client.get(f"/reports/sample/files/{index}").status_code, 404)

    def test_report_text_is_escaped_and_exceptions_are_not_exposed(self):
        self.login()
        self.add_report(text='<script>alert("example")</script>')
        response = self.client.get("/reports/sample")
        self.assertIn("&lt;script&gt;", response.text)
        self.assertNotIn('<script>alert("example")', response.text)
        with patch.object(self.store, "query", side_effect=RuntimeError("private-detail-not-for-browser")):
            response = self.client.get("/history")
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private-detail-not-for-browser", response.text)

    def test_keywords_and_regions_are_versioned(self):
        self.token = self.login()
        version = self.store.version("keyword")
        response = self.post("/actions/keywords", {"entries": "тест цены | экономика\nтест работа | рынок труда"})
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.store.version("keyword"), version + 1)
        self.assertEqual(self.post("/actions/regions", {"regions": "2, 1, 2"}).status_code, 303)
        self.assertEqual(self.store.regions(), [1, 2])
        self.assertEqual(self.post("/actions/regions", {"regions": "-1"}).status_code, 400)

    def test_logout_invalidates_session(self):
        self.token = self.login()
        self.assertEqual(self.post("/logout", {}).status_code, 303)
        self.assertEqual(self.client.get("/").headers["location"], "/login")

    def test_oversized_and_duplicate_form_fields_are_rejected(self):
        self.token = self.login()
        self.assertEqual(self.post("/actions/keywords", {"entries": "a" * 40_000}).status_code, 413)
        response = self.client.post("/actions/report", content=f"csrf={self.token}&section=ai&section=ruonia", headers={"Origin": "http://127.0.0.1", "Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(response.status_code, 400)

    def test_exports_match_frequency_filter_and_remove_temporary_file(self):
        self.login()
        self.store.save_observations([
            {"source": "wordstat", "metric": "цены", "frequency": "daily", "date": "2026-01-01", "value": 5},
            {"source": "wordstat", "metric": "цены", "frequency": "weekly", "date": "2026-01-01", "value": 35},
        ])
        response = self.client.get("/export/csv?source=wordstat&frequency=weekly")
        self.assertEqual(response.status_code, 200, response.text)
        rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frequency"], "weekly")
        self.assertEqual(list((self.settings.data_dir / "exports").iterdir()), [])
        self.assertEqual(self.client.get("/export/xlsx?source=wordstat").status_code, 200)
        self.assertEqual(self.client.get("/export/csv?source=unrecognized").status_code, 400)

    def test_uncertain_delivery_requires_explicit_valid_action(self):
        self.token = self.login()
        self.add_report()
        delivery_id = self.store.execute("INSERT INTO deliveries(report_id,recipient,part_index,payload,state,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                                         ("sample", "123", 0, dumps({"kind": "text", "text": "sample"}), "uncertain", utcnow(), utcnow()))
        self.assertEqual(self.client.get("/jobs").status_code, 200)
        self.assertEqual(self.post(f"/actions/delivery/{delivery_id}", {"action": "invalid"}).status_code, 400)
        self.assertEqual(self.post(f"/actions/delivery/{delivery_id}", {"action": "sent"}).status_code, 303)
        self.assertEqual(self.store.one("SELECT state FROM deliveries WHERE id=?", (delivery_id,))["state"], "sent")
        self.assertEqual(self.post(f"/actions/delivery/{delivery_id}", {"action": "retry"}).status_code, 409)

    def test_budget_prices_are_optional_and_nonfinite_values_rejected(self):
        self.token = self.login()
        values = {"wordstat_calls": "100", "ai_calls": "5", "ai_output_tokens": "1000", "currency": "у.е.", "ai_input_per_million": "3.5"}
        self.assertEqual(self.post("/actions/budgets", values).status_code, 303)
        saved = self.store.get_setting("budgets")
        self.assertEqual(saved["prices"], {"ai": {"input_per_million": 3.5}})
        self.assertEqual(self.post("/actions/budgets", {**values, "ai_input_per_million": "nan"}).status_code, 400)
        self.assertEqual(self.store.get_setting("budgets"), saved)

    def test_login_attempts_are_rate_limited(self):
        token = self.csrf(self.client.get("/login").text)
        for _ in range(5):
            self.assertEqual(self.client.post("/login", data={"csrf": token, "password": "wrong"}, headers={"Origin": "http://127.0.0.1"}).status_code, 400)
        self.assertEqual(self.client.post("/login", data={"csrf": token, "password": self.password}, headers={"Origin": "http://127.0.0.1"}).status_code, 429)

    def test_alert_rule_filters_and_thresholds_are_validated(self):
        self.token = self.login()
        self.store.set_regions([225])
        response = self.post("/actions/alerts", {"name": "Региональное правило", "metric": "пожар", "region_id": "225"})
        self.assertEqual(response.status_code, 303)
        rule = self.store.one("SELECT * FROM alert_rules WHERE name='Региональное правило'")
        self.assertEqual((rule["metric"], rule["region_id"], rule["enabled"]), ("пожар", 225, 0))
        fields = {"enabled": "1", "metric": "пожар", "region_id": "225", "pct_threshold": "55", "z_threshold": "3", "min_coverage": "0.9", "min_baseline": "20", "cooldown_hours": "24"}
        self.assertEqual(self.post(f"/actions/alerts/{rule['id']}", fields).status_code, 303)
        self.assertEqual(self.post(f"/actions/alerts/{rule['id']}", {**fields, "region_id": "999"}).status_code, 400)
        self.assertEqual(self.post(f"/actions/alerts/{rule['id']}", {**fields, "pct_threshold": "nan"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
