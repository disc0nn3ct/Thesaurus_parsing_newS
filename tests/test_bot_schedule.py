"""Offline authorization and scheduling tests against a real SQLite store."""

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from thesaurus.delivery import prepare_delivery
from thesaurus.scheduling import (
    enqueue_due,
    seed_subscriptions,
    update_subscription,
    validate_filters,
    validate_subscription,
)
from thesaurus.store import Store, dumps, utcnow
from thesaurus.telegram_bot import install_handlers


class BotScheduleTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parent / f"bot_test_{uuid4().hex}.sqlite3"
        self.addCleanup(self.cleanup_database)
        self.store = Store(self.path)
        self.store.add_user(1, "admin", subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.store.add_user(3, subscriptions='all')
        self.client = Mock()
        self.bot = install_handlers(self.client, self.store)

    def cleanup_database(self):
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.path) + suffix).unlink(missing_ok=True)

    def message(self, text, user=2, chat=None, kind="private", message_id=1):
        message = SimpleNamespace(text=text, id=message_id, from_user=SimpleNamespace(id=user),
                                  chat=SimpleNamespace(id=user if chat is None else chat, type=kind))
        self.bot.handle_message(self.client, message)

    def callback(self, data, user=2, chat=None, kind="private", callback_id=None):
        query = SimpleNamespace(data=data, id=callback_id or uuid4().hex, from_user=SimpleNamespace(id=user),
                                message=SimpleNamespace(chat=SimpleNamespace(id=user if chat is None else chat, type=kind)),
                                answer=Mock())
        self.bot.handle_callback(self.client, query)
        return query

    def subscription(self, user, section="ai"):
        return self.store.one("SELECT * FROM subscriptions WHERE user_id=? AND section=?", (user, section))

    def report(self):
        report_id = uuid4().hex
        self.store.execute(
            "INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text) VALUES(?,?,?,?,?,?,?)",
            (report_id, "ai", utcnow(), 1, 0, "Brief", "Full"),
        )
        return report_id

    def test_templates_are_disabled_and_seed_does_not_grant_access(self):
        seed_subscriptions(self.store, "@legacy-destination")
        rows = self.store.query("SELECT * FROM subscriptions")
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["enabled"] == 0 and row["timezone"] == "Europe/Moscow" for row in rows))
        self.assertFalse(self.store.authorize(99))
        ai = next(row for row in rows if row["section"] == "ai")
        self.assertEqual(json.loads(ai["times"]), ["08:30", "18:30"])

    def test_menu_buttons_are_present_and_reuse_authorized_views(self):
        expected = {"menu:report", "menu:settings", "menu:data", "ch:menu", "menu:help"}
        self.message("/start")
        self.callback('pj:market')
        keyboard = self.client.send_message.call_args.kwargs["reply_markup"]
        self.assertTrue(expected.issubset({button.callback_data for row in keyboard.inline_keyboard for button in row}))
        for action in expected:
            with self.subTest(action=action):
                self.client.send_message.reset_mock()
                query = self.callback(action)
                query.answer.assert_called_once_with()
                keyboard = self.client.send_message.call_args.kwargs["reply_markup"]
                self.assertIn('menu:home', {button.callback_data for row in keyboard.inline_keyboard for button in row})
        self.assertEqual(self.store.query("SELECT * FROM jobs"), [])

    def test_report_menu_offers_all_sections_and_queues_one_job_per_callback(self):
        self.callback("menu:report")
        keyboard = self.client.send_message.call_args.kwargs["reply_markup"]
        choices = {button.callback_data for row in keyboard.inline_keyboard for button in row
                   if button.callback_data.startswith("request:")}
        self.assertEqual(choices, {"request:ruonia", "request:ai", "request:wordstat", "request:regions", "request:market"})
        self.callback("request:wordstat", callback_id="same-event")
        self.callback("request:wordstat", callback_id="same-event")
        jobs = self.store.query("SELECT * FROM jobs")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(json.loads(jobs[0]["payload"])["section"], "wordstat")

    def test_every_menu_callback_rechecks_allowlist_and_private_chat(self):
        for action in ("menu:status", "menu:report", "menu:subscriptions", "menu:regions", "menu:keywords", "menu:help", "request:ai"):
            for user, chat, kind in ((99, 99, "private"), (2, -1, "group"), (2, 3, "private")):
                with self.subTest(action=action, user=user, chat=chat):
                    self.client.send_message.reset_mock()
                    query = self.callback(action, user=user, chat=chat, kind=kind)
                    query.answer.assert_called_once_with("Нет доступа", show_alert=True)
                    self.client.send_message.assert_not_called()
        self.assertEqual(self.store.query("SELECT * FROM jobs"), [])
        self.assertEqual(self.store.query("SELECT * FROM subscriptions"), [])

    def test_admin_own_regions_do_not_change_shared_cohort(self):
        self.store.set_regions([1, 2])
        self.message("/regions mine 2", user=1)
        self.assertEqual(json.loads(self.subscription(1, "regions")["regions"]), [2])
        self.assertEqual(self.store.regions(), [1, 2])
        self.message("/regions set 1,2,3", user=1)
        self.assertEqual(self.store.regions(), [1, 2, 3])
        self.assertEqual(json.loads(self.subscription(1, "regions")["regions"]), [2])

    def test_missing_and_spoofed_authorization_cannot_enqueue(self):
        for user, chat, kind in ((99, 99, "private"), (2, -10, "group"), (2, 3, "private")):
            with self.subTest(user=user, chat=chat):
                self.message("/report ai", user=user, chat=chat, kind=kind)
        self.assertEqual(self.store.query("SELECT * FROM jobs"), [])
        self.assertEqual(self.store.query("SELECT * FROM subscriptions"), [])

    def test_each_callback_rechecks_disabled_user(self):
        self.message("/subscriptions ai on")
        self.store.add_user(2, enabled=False, subscriptions='all')
        query = self.callback("sub:ai:off")
        self.assertEqual(self.subscription(2)["enabled"], 1)
        query.answer.assert_called_once_with("Нет доступа", show_alert=True)

    def test_subscriber_controls_only_own_schedule(self):
        self.message("/subscriptions ai on 10:00")
        self.message("/subscriptions ai on 11:00", user=3)
        self.callback("sub:ai:off", user=2)
        self.assertEqual(self.subscription(2)["enabled"], 0)
        self.assertEqual(self.subscription(3)["enabled"], 1)
        self.assertEqual(json.loads(self.subscription(3)["times"]), ["11:00"])

    def test_subscriber_cannot_change_admin_settings_or_resolve_callback(self):
        old_keywords, old_budgets = self.store.keywords(), self.store.get_setting("budgets")
        for command in ("/allow 99", "/deny 1", "/keywords add injected", "/budget ai 999"):
            self.message(command)
        delivery_id = prepare_delivery(self.store, self.report(), 2)[0]
        self.store.execute("UPDATE deliveries SET state='uncertain' WHERE id=?", (delivery_id,))
        self.callback(f"resolve:{delivery_id}:retry")
        self.assertFalse(self.store.authorize(99))
        self.assertTrue(self.store.authorize(1, admin=True))
        self.assertEqual(self.store.keywords(), old_keywords)
        self.assertEqual(self.store.get_setting("budgets"), old_budgets)
        self.assertEqual(self.store.one("SELECT state FROM deliveries WHERE id=?", (delivery_id,))["state"], "uncertain")

    def test_admin_can_allow_budget_and_explicitly_resolve(self):
        self.message("/allow 99", user=1)
        self.message("/budget ai 5", user=1)
        self.assertTrue(self.store.authorize(99))
        self.assertEqual(self.store.get_setting("budgets")["ai_calls"], 5)
        self.assertTrue(all(row["enabled"] == 0 for row in self.store.query("SELECT enabled FROM subscriptions WHERE user_id=99")))
        delivery_id = prepare_delivery(self.store, self.report(), 2)[0]
        self.store.execute("UPDATE deliveries SET state='uncertain' WHERE id=?", (delivery_id,))
        self.message(f"/resolve {delivery_id} sent", user=1)
        self.assertEqual(self.store.one("SELECT state FROM deliveries WHERE id=?", (delivery_id,))["state"], "sent")

    def test_report_requests_are_persistent_and_idempotent_per_message(self):
        self.message("/report ai full", message_id=20)
        self.message("/report ai full", message_id=20)
        jobs = self.store.query("SELECT * FROM jobs")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(json.loads(jobs[0]["payload"])["recipient"], "2")
        self.assertEqual(json.loads(jobs[0]["payload"])["detail"], "full")

    def test_details_callback_cannot_read_another_recipients_report(self):
        report_id = self.report()
        prepare_delivery(self.store, report_id, 3)
        self.callback(f"report:{report_id}", user=2)
        self.assertEqual(self.store.query("SELECT * FROM deliveries WHERE recipient='2'"), [])
        self.callback(f"report:{report_id}", user=3)
        rows = self.store.query("SELECT * FROM deliveries WHERE recipient='3'")
        self.assertEqual(len(rows), 2)
        self.assertTrue(any(row["variant"].startswith("full:") for row in rows))

    def test_whitelist_and_region_cohort_enforced_in_shared_updates(self):
        self.store.set_regions([1, 2])
        update_subscription(self.store, 2, "regions", regions=[2], phrases=["пожар"])
        with self.assertRaises(ValueError):
            update_subscription(self.store, 2, "regions", regions=[999])
        with self.assertRaises(ValueError):
            update_subscription(self.store, 2, "wordstat", phrases=["arbitrary costly query"])
        self.assertEqual(json.loads(self.subscription(2, "regions")["regions"]), [2])

    def test_section_filters_reject_cross_section_queries_and_regions(self):
        self.store.set_regions([1, 2])
        good = [
            ("ruonia", ["RUONIA", "3 мес"], [225]),
            ("wordstat", ["обвал"], [225]),
            ("regions", ["пожар"], [2]),
            ("ai", [], []),
        ]
        for section, phrases, regions in good:
            with self.subTest(section=section):
                update_subscription(self.store, 2, section, phrases=phrases, regions=regions)
                self.assertEqual(json.loads(self.subscription(2, section)["phrases"]), phrases)
        bad = [
            ("ruonia", ["обвал"], [225]), ("ruonia", ["RUONIA"], [1]),
            ("wordstat", ["пожар"], [225]), ("wordstat", ["обвал"], [2]),
            ("regions", ["обвал"], [2]), ("regions", ["пожар"], [225]),
            ("ai", ["обвал"], []), ("ai", [], [225]),
        ]
        for section, phrases, regions in bad:
            with self.subTest(section=section, phrases=phrases, regions=regions), self.assertRaises(ValueError):
                update_subscription(self.store, 2, section, phrases=phrases, regions=regions)

    def test_stale_invalid_filters_cannot_be_enabled_or_scheduled(self):
        seed_subscriptions(self.store, "2", 2)
        self.store.execute("UPDATE subscriptions SET phrases=?,enabled=1,created_at=? WHERE user_id=2 AND section='wordstat'",
                           (dumps(["пожар"]), "2026-01-01T00:00:00+00:00"))
        with self.assertRaises(ValueError):
            update_subscription(self.store, 2, "wordstat", enabled=True)
        self.assertEqual(enqueue_due(self.store, datetime(2026, 10, 2, 16, 0, tzinfo=timezone.utc)), [])
        self.message("/report wordstat")
        self.assertEqual(self.store.query("SELECT * FROM jobs"), [])
        with self.assertRaises(ValueError):
            validate_filters(self.store, "ai", phrases="")

    def test_invalid_schedule_values_rejected(self):
        for values in ({"times": ["25:00"]}, {"weekdays": [7]}, {"timezone": "invalid/timezone"},
                       {"recipient": "somebody-else"}, {"enabled": "yes"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_subscription(values)

    def enable_old_schedule(self, user=2, section="ai", **values):
        update_subscription(self.store, user, section, enabled=True, **values)
        self.store.execute("UPDATE subscriptions SET created_at=? WHERE user_id=?",
                           ("2026-01-01T00:00:00+00:00", user))

    def test_missed_slots_coalesce_and_repeated_scheduler_ticks_do_not_duplicate(self):
        self.enable_old_schedule()
        now = datetime(2026, 10, 2, 16, 0, tzinfo=timezone.utc)  # 19:00 Moscow
        queued = enqueue_due(self.store, now)
        self.assertEqual(len(queued), 1)
        self.assertEqual(enqueue_due(self.store, now), [])
        job = self.store.one("SELECT * FROM jobs")
        self.assertEqual(json.loads(job["payload"])["scheduled_for"], "2026-10-02T15:30:00+00:00")
        next_day = datetime(2026, 10, 3, 16, 0, tzinfo=timezone.utc)
        self.assertEqual(enqueue_due(self.store, next_day), queued)
        self.assertEqual(len(self.store.query("SELECT * FROM jobs")), 1)
        self.assertIn("2026-10-03", self.store.one("SELECT payload FROM jobs")["payload"])

    def test_scheduler_respects_timezone_and_skips_disabled_users(self):
        self.enable_old_schedule(times=["09:00"], timezone="UTC")
        self.store.execute("UPDATE subscriptions SET last_scheduled_at=? WHERE user_id=2 AND section='ai'",
                           ("2026-10-01T09:00:00+00:00",))
        self.assertEqual(enqueue_due(self.store, datetime(2026, 10, 2, 8, 0, tzinfo=timezone.utc)), [])
        self.store.add_user(2, enabled=False, subscriptions='all')
        self.assertEqual(enqueue_due(self.store, datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc)), [])

    def test_weekly_template_selects_latest_wednesday(self):
        self.enable_old_schedule(section="wordstat")
        enqueue_due(self.store, datetime(2026, 10, 2, 16, 0, tzinfo=timezone.utc))
        payload = json.loads(self.store.one("SELECT payload FROM jobs")["payload"])
        self.assertEqual(payload["scheduled_for"], "2026-09-30T06:00:00+00:00")

    def test_malformed_persisted_schedule_does_not_block_other_subscribers(self):
        self.enable_old_schedule(user=2)
        self.enable_old_schedule(user=3)
        self.store.execute("UPDATE subscriptions SET phrases='broken-json' WHERE user_id=2 AND section='ai'")
        enqueue_due(self.store, datetime(2026, 10, 2, 16, 0, tzinfo=timezone.utc))
        jobs = self.store.query("SELECT payload FROM jobs")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(json.loads(jobs[0]["payload"])["recipient"], "3")


if __name__ == "__main__":
    unittest.main()
