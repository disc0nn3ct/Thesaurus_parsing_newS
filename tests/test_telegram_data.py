"""Real SQLite and offline providers for personal bot data jobs."""

import csv
import json
import shutil
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from openpyxl import load_workbook

from thesaurus.analysis import forecast_weekly
from thesaurus.config import Settings
from thesaurus.delivery import deliver_pending, prepare_delivery
from thesaurus.exporting import export_data
from thesaurus.scheduling import update_subscription
from thesaurus.service import Worker
from thesaurus.store import Store, dumps, utcnow
from thesaurus.telegram_bot import BotControls
from thesaurus.telegram_data import DataControls
from thesaurus.telegram_data_jobs import forecast_summary


class TelegramDataTests(unittest.TestCase):
    def setUp(self):
        self.test_root = (Path(__file__).resolve().parents[1] / ".test-tmp").resolve()
        self.directory = self.test_root / f"telegram-data-{uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.store = Store(self.directory / "state.sqlite3")
        self.settings = Settings(root=self.directory)
        self.store.add_user(1, "admin", subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.store.add_user(3, subscriptions='all')
        self.store.set_keywords([{"phrase": "mine"}, {"phrase": "other"}])
        self.store.set_regions([213, 2])
        update_subscription(self.store, 2, "wordstat", phrases=["mine"])
        update_subscription(self.store, 2, "regions", phrases=["пожар"], regions=[213])
        self.client = Mock()
        self.client.send_message.return_value = SimpleNamespace(id=31)
        self.client.send_document.return_value = SimpleNamespace(id=32)
        self.reply = Mock()
        self.controls = DataControls(self.store, self.reply)
        self.providers = Mock()
        self.worker = Worker(self.store, self.settings, providers=self.providers, charts=False)

    def cleanup(self):
        target = self.directory.resolve()
        if target.is_relative_to(self.test_root) and target != self.test_root:
            shutil.rmtree(target)

    def command(self, command, argument="", actor=2, event=1):
        return self.controls.dispatch(self.client, actor, command, argument, event)

    def observations(self):
        rows = []
        for index in range(42):
            observed = (date(2024, 1, 1) + timedelta(weeks=index)).isoformat()
            for phrase, region, frequency in (("mine", 225, "weekly"), ("other", 225, "weekly"),
                                               ("mine", 213, "weekly"), ("пожар", 213, "daily"),
                                               ("пожар", 2, "daily"), ("взрыв", 213, "daily")):
                rows.append({"source": "wordstat", "metric": phrase, "region_id": region,
                             "frequency": frequency, "date": observed, "value": 100 + index,
                             "unit": "count", "retrieved_at": observed + "T18:00:00+00:00"})
        self.store.save_observations(rows)
        return rows

    def job(self):
        return self.store.one("SELECT * FROM jobs ORDER BY id DESC LIMIT 1")

    def artifact(self):
        report = self.store.one("SELECT * FROM reports ORDER BY created_at DESC LIMIT 1")
        return report, Path(json.loads(report["parts"])[0]["path"])

    def report(self, recipient=2, state="sent", section="ai"):
        report_id = uuid4().hex
        self.store.execute(
            "INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text) VALUES(?,?,?,?,?,?,?)",
            (report_id, section, utcnow(), 1, 0, "brief", "full"),
        )
        ids = prepare_delivery(self.store, report_id, str(recipient))
        self.store.execute("UPDATE deliveries SET state=? WHERE id=?", (state, ids[0]))
        return report_id

    def test_csv_export_queues_once_filters_own_selection_and_uses_durable_delivery(self):
        self.observations()
        self.command("export", "wordstat csv 2024-01-01 2024-01-22")
        self.command("export", "wordstat csv 2024-01-01 2024-01-22")
        self.assertEqual(len(self.store.query("SELECT * FROM jobs")), 1)
        self.assertEqual(self.store.query("SELECT * FROM reports"), [])
        self.worker.drain()
        self.assertEqual(self.job()["state"], "done")
        report, path = self.artifact()
        self.assertTrue(path.is_relative_to(self.settings.reports_dir))
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["metric"] for row in rows}, {"mine"})
        self.assertEqual({row["region_id"] for row in rows}, {"225"})
        deliveries = self.store.query("SELECT * FROM deliveries WHERE report_id=?", (report["id"],))
        self.assertEqual({row["state"] for row in deliveries}, {"pending"})
        self.assertEqual([json.loads(row["payload"])["kind"] for row in deliveries], ["text", "document"])
        self.client.send_document.assert_not_called()
        deliver_pending(self.store, self.client)
        self.client.send_document.assert_called_once()
        self.assertEqual(self.providers.mock_calls, [])

    def test_xlsx_regional_export_preserves_own_phrase_and_region(self):
        self.observations()
        self.command("export", "regions xlsx")
        self.worker.drain()
        _, path = self.artifact()
        workbook = load_workbook(path, read_only=True)
        self.addCleanup(workbook.close)
        values = list(workbook.active.values)
        self.assertEqual(len(values), 43)
        self.assertEqual({row[1] for row in values[1:]}, {"пожар"})
        self.assertEqual({row[2] for row in values[1:]}, {213})

    def test_disabled_or_reassigned_actor_cannot_run_queued_export(self):
        self.command("export", "wordstat csv")
        self.store.add_user(2, enabled=False, subscriptions='all')
        self.worker.drain()
        self.assertEqual(self.job()["state"], "failed")
        self.assertEqual(self.store.query("SELECT * FROM reports"), [])
        self.assertEqual(self.store.query("SELECT * FROM deliveries"), [])

    def test_current_filter_changes_block_stale_job_and_filter_expansion(self):
        self.command("export", "wordstat csv")
        update_subscription(self.store, 2, "wordstat", phrases=["other"])
        self.worker.drain()
        self.assertEqual(self.job()["state"], "failed")
        self.command("export", "wordstat csv", event=2)
        payload = json.loads(self.job()["payload"])
        payload["phrases"] = ["mine", "other"]
        self.store.execute("UPDATE jobs SET payload=? WHERE id=?", (dumps(payload), self.job()["id"]))
        self.worker.drain()
        self.assertEqual(self.job()["state"], "failed")
        self.assertEqual(self.store.query("SELECT * FROM reports"), [])

    def test_worker_rejects_recipient_and_file_path_tampering(self):
        for field, value in (("recipient", "3"), ("subscription_id", 99999), ("report_id", "../secret")):
            with self.subTest(field=field):
                self.command("export", "wordstat csv", event=field)
                payload = json.loads(self.job()["payload"])
                payload[field] = value
                self.store.execute("UPDATE jobs SET payload=? WHERE id=?", (dumps(payload), self.job()["id"]))
                self.worker.drain()
                self.assertEqual(self.job()["state"], "failed")
        self.assertEqual(self.store.query("SELECT * FROM reports"), [])

    def test_worker_recovery_reuses_artifact_and_delivery(self):
        self.command("export", "wordstat csv")
        self.worker.drain()
        before = self.store.query("SELECT * FROM deliveries")
        self.store.execute("UPDATE jobs SET state='pending'")
        self.worker.drain()
        self.assertEqual(self.job()["state"], "done")
        self.assertEqual(len(self.store.query("SELECT * FROM reports")), 1)
        self.assertEqual(self.store.query("SELECT * FROM deliveries"), before)

    def test_permission_is_rechecked_after_artifact_generation(self):
        self.command("export", "wordstat csv")

        def export_and_revoke(*args, **kwargs):
            output = export_data(*args, **kwargs)
            self.store.add_user(2, enabled=False, subscriptions='all')
            return output

        with patch("thesaurus.telegram_data_jobs.export_data", side_effect=export_and_revoke):
            self.worker.drain()
        self.assertEqual(self.job()["state"], "failed")
        self.assertEqual(self.store.query("SELECT * FROM reports"), [])
        self.assertEqual(self.store.query("SELECT * FROM deliveries"), [])

    def test_history_and_detail_enforce_report_ownership_and_completed_delivery(self):
        own = self.report()
        other = self.report(recipient=3)
        pending = self.report(state="pending")
        self.command("history", "ai")
        text = self.reply.call_args.args[2]
        self.assertIn(own, text)
        self.assertNotIn(other, text)
        self.assertNotIn(pending, text)
        with self.assertRaises(PermissionError):
            self.controls.callback(self.client, 2, f"history:{other}")
        with self.assertRaises(PermissionError):
            self.command("history", other)
        self.controls.callback(self.client, 2, f"history:{own}")
        self.controls.callback(self.client, 2, f"history:{own}")
        full = self.store.query("SELECT * FROM deliveries WHERE report_id=? AND variant LIKE 'full:%'", (own,))
        self.assertEqual(len(full), 1)
        self.command("history", actor=1)
        self.assertIn(other, self.reply.call_args.args[2])
        self.controls.callback(self.client, 1, f"history:{other}")

    def test_every_helper_action_rejects_disabled_user(self):
        own = self.report()
        self.store.add_user(2, enabled=False, subscriptions='all')
        for command, argument in (("history", ""), ("export", "wordstat csv"), ("forecasts", ""),
                                   ("backtest", ""), ("verify", own)):
            with self.subTest(command=command), self.assertRaises(PermissionError):
                self.command(command, argument)
        with self.assertRaises(PermissionError):
            self.controls.callback(self.client, 2, f"history:{own}")

    def test_reopening_delivered_history_shows_full_text_and_preserves_files(self):
        self.command("export", "wordstat csv")
        self.worker.drain()
        report, _ = self.artifact()
        deliver_pending(self.store, self.client)
        self.store.execute("UPDATE reports SET news=? WHERE id=?", (dumps([{
            "title": "Проверенный источник", "numbers_verified": True,
            "source_availability": {"availability": "available"},
        }]), report["id"]))
        before = self.store.query("SELECT * FROM deliveries")
        self.command("history", report["id"])
        text = self.reply.call_args.args[2]
        self.assertIn(report["full_text"], text)
        self.assertIn("уже доставлен", text)
        self.assertIn("available; числа проверены", text)
        self.assertNotIn("поставлены в очередь", text)
        self.assertEqual(self.store.query("SELECT * FROM deliveries"), before)
        deliver_pending(self.store, self.client)
        self.client.send_document.assert_called_once()

    def test_first_full_expansion_only_queues_text_without_duplicate_reply(self):
        report_id = self.report()
        self.command("history", report_id)
        text = self.reply.call_args.args[2]
        self.assertIn("Полный текст поставлен в очередь доставки", text)
        self.assertNotIn("\nfull", text)
        full = self.store.query("SELECT * FROM deliveries WHERE report_id=? AND variant LIKE 'full:%'", (report_id,))
        self.assertEqual(len(full), 1)
        self.assertEqual(json.loads(full[0]["payload"])["text"], "full")
        self.assertEqual(full[0]["state"], "pending")

    def test_history_reports_pending_and_uncertain_files_without_requeue(self):
        self.command("export", "wordstat csv")
        self.worker.drain()
        report, _ = self.artifact()
        for state, expected in (("pending", "ожидает отправки"), ("uncertain", "результат отправки неизвестен")):
            with self.subTest(state=state):
                self.store.execute("UPDATE deliveries SET state=? WHERE report_id=? AND part_index=1", (state, report["id"]))
                before = self.store.query("SELECT * FROM deliveries")
                self.controls.callback(self.client, 2, f"history:{report['id']}")
                self.assertIn(expected, self.reply.call_args.args[2])
                self.assertIn(report["full_text"], self.reply.call_args.args[2])
                self.assertEqual(self.store.query("SELECT * FROM deliveries"), before)

    def test_report_button_reuses_sent_full_variant_with_filter_metadata(self):
        report_id = self.report()
        prepare_delivery(self.store, report_id, "2", detail="full", filters={"phrases": ["mine"]})
        deliver_pending(self.store, self.client)
        before = self.store.query("SELECT * FROM deliveries")
        query = SimpleNamespace(data=f"report:{report_id}", id="details", from_user=SimpleNamespace(id=2),
                                message=SimpleNamespace(chat=SimpleNamespace(id=2, type="private")), answer=Mock())
        BotControls(self.store).handle_callback(self.client, query)
        query.answer.assert_called_once_with()
        self.assertIn("Полный текст:\nfull", self.client.send_message.call_args.args[1])
        self.assertEqual(self.store.query("SELECT * FROM deliveries"), before)

    def test_forecast_scores_show_only_own_weekly_phrases(self):
        observations = self.observations()
        for phrase in ("mine", "other"):
            history = [row for row in observations if row["metric"] == phrase and row["region_id"] == 225]
            origin = history[35]["date"]
            predictions = forecast_weekly(history, origin_date=origin, created_at=origin + "T23:00:00Z")
            for row in predictions:
                fields = ("source", "metric", "region_id", "device", "frequency", "origin_date", "target_date",
                          "horizon", "prediction", "lower", "upper", "baseline", "created_at", "model")
                self.store.execute("INSERT INTO forecasts(" + ",".join(fields) + ") VALUES(" + ",".join("?" for _ in fields) + ")",
                                   tuple(row[field] for field in fields))
        rows, scores = forecast_summary(self.store, 2)
        self.assertEqual({row["metric"] for row in rows}, {"mine"})
        self.assertEqual(scores["n"], 4)
        self.command("forecasts")
        text = self.reply.call_args.args[2]
        self.assertIn("mine", text)
        self.assertNotIn("other", text)
        self.assertIn("MAE", text)

    def test_backtest_is_offline_scoped_and_excludes_future_data(self):
        self.observations()
        self.store.save_observations([{"source": "wordstat", "metric": "mine", "region_id": 225,
                                       "frequency": "weekly", "date": "2099-01-01", "value": 999999}])
        self.command("backtest", "3")
        self.command("backtest", "3")
        self.assertEqual(len(self.store.query("SELECT * FROM jobs")), 1)
        self.worker.drain()
        self.assertEqual(self.job()["state"], "done")
        report, path = self.artifact()
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["evaluation"], "retrospective_backtest")
        self.assertEqual({row["metric"] for row in result["forecasts"]}, {"mine"})
        self.assertTrue(all(row["origin_date"] < "2099" for row in result["forecasts"]))
        self.assertTrue(all(row["origin_date"] < row["target_date"] for row in result["forecasts"]))
        self.assertGreater(result["metrics"]["n"], 0)
        self.assertIn("ретроспективная", report["full_text"])
        self.assertEqual(self.providers.mock_calls, [])
        self.client.send_document.assert_not_called()

    def test_verify_is_admin_only_and_rechecks_role_in_worker(self):
        report_id = self.report()
        with self.assertRaises(PermissionError):
            self.command("verify", report_id)
        self.command("verify", report_id, actor=1)
        self.command("verify", report_id, actor=1)
        self.assertEqual(len(self.store.query("SELECT * FROM jobs")), 1)
        self.store.add_user(1, role="subscriber", subscriptions='all')
        with patch.object(self.worker, "_verify_sources") as verify:
            self.worker.drain()
            verify.assert_not_called()
        self.assertEqual(self.job()["state"], "failed")
        self.store.add_user(1, role="admin", subscriptions='all')
        self.command("verify", report_id, actor=1, event=2)
        with patch.object(self.worker, "_verify_sources") as verify:
            self.worker.drain()
            verify.assert_called_once_with(report_id)

    def test_validation_rejects_malformed_dates_sections_and_work_limits(self):
        cases = [("export", "ai csv"), ("export", "wordstat exe"), ("export", "wordstat csv nope nope"),
                 ("export", "wordstat csv 2025-02-01 2025-01-01"), ("backtest", "53"), ("backtest", "-1"),
                 ("history", "../../secret"), ("forecasts", "other")]
        for command, argument in cases:
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                self.command(command, argument)
        self.assertEqual(self.store.query("SELECT * FROM jobs"), [])


if __name__ == "__main__":
    unittest.main()
