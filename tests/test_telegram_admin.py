"""Offline Telegram administration and atomic proposal review regressions."""

import json
import shutil
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4
from zoneinfo import ZoneInfo

from thesaurus.config import Settings
from thesaurus.management import resolve_proposal, save_alert_rule
from thesaurus.service import Worker
from thesaurus.store import Store, dumps, utcnow
from thesaurus.telegram_bot import install_handlers


class TelegramAdminTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / ".test-tmp"
        self.folder = self.root / f"telegram_admin_{uuid4().hex}"
        self.folder.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.store = Store(self.folder / "state.sqlite3")
        self.store.add_user(1, "admin", subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.store.set_regions([213, 2])
        self.client = Mock()
        self.bot = install_handlers(self.client, self.store)

    def cleanup(self):
        assert self.folder.resolve().parent == self.root.resolve()
        shutil.rmtree(self.folder)

    def message(self, text, user=1, chat=None, kind="private", message_id=10):
        event = SimpleNamespace(text=text, id=message_id, from_user=SimpleNamespace(id=user),
                                chat=SimpleNamespace(id=user if chat is None else chat, type=kind))
        self.bot.handle_message(self.client, event)

    def callback(self, data, user=1, chat=None, kind="private", query_id="callback-id"):
        event = SimpleNamespace(data=data, id=query_id, from_user=SimpleNamespace(id=user),
                                message=SimpleNamespace(chat=SimpleNamespace(id=user if chat is None else chat, type=kind)),
                                answer=Mock())
        self.bot.handle_callback(self.client, event)
        return event

    def text(self):
        return "\n".join(call.args[1] for call in self.client.send_message.call_args_list)

    def proposal(self, entries=None):
        return self.store.execute("INSERT INTO keyword_proposals(created_at,entries) VALUES(?,?)",
                                  (utcnow(), dumps(entries if entries is not None else [{"phrase": "новая фраза", "theme": "тема"}])))

    def test_all_admin_commands_reject_subscriber_and_invalid_private_identity(self):
        proposal_id = self.proposal()
        old_budget = self.store.get_setting("budgets")
        old_keywords = self.store.keywords()
        for text in ("/admin", "/budget ai 5", "/budget price ai per_call 5", "/alerts add injected|*|*",
                     "/alerts off 1", "/proposals request", f"/proposals approve {proposal_id}", "/jobs", "/users"):
            for user, chat, kind in ((2, 2, "private"), (99, 99, "private"), (1, -1, "group"),
                                     (1, 2, "private"), ("1", "1", "private"), (True, True, "private")):
                with self.subTest(text=text, user=user, chat=chat):
                    self.message(text, user, chat, kind)
        self.assertEqual(self.store.get_setting("budgets"), old_budget)
        self.assertEqual(self.store.keywords(), old_keywords)
        self.assertEqual(self.store.query("SELECT * FROM jobs"), [])
        self.assertEqual(len(self.store.query("SELECT * FROM alert_rules")), 1)
        self.assertEqual(self.store.one("SELECT enabled FROM alert_rules WHERE id=1")["enabled"], 1)
        self.assertNotIn("Пользователи:", self.text())

    def test_admin_callbacks_recheck_privileges_and_revocation(self):
        proposal_id = self.proposal()
        for data in ("menu:admin", "menu:budget", "menu:alerts", "menu:proposals", "menu:jobs", "menu:users", "menu:verify_help",
                     "alert:off:1", f"proposal:approve:{proposal_id}", "proposal:request:new", "adminpage:users:2"):
            for user, chat, kind in ((2, 2, "private"), (1, 2, "private"), (1, -1, "group"), ("1", "1", "private")):
                with self.subTest(data=data, user=user):
                    self.client.send_message.reset_mock()
                    event = self.callback(data, user, chat, kind)
                    self.assertTrue(event.answer.call_args.kwargs["show_alert"])
                    self.client.send_message.assert_not_called()
        self.store.add_user(1, role="admin", enabled=False, subscriptions='all')
        event = self.callback(f"proposal:approve:{proposal_id}")
        event.answer.assert_called_once_with("Нет доступа", show_alert=True)
        self.assertEqual(self.store.one("SELECT state FROM keyword_proposals")["state"], "pending")
        self.assertEqual(self.store.query("SELECT * FROM jobs"), [])

    def test_admin_and_subscriber_menus_show_appropriate_discovery(self):
        self.message("/start")
        keyboard = self.client.send_message.call_args.kwargs["reply_markup"]
        self.assertIn("menu:admin", {button.callback_data for row in keyboard.inline_keyboard for button in row})
        self.callback("menu:admin")
        keyboard = self.client.send_message.call_args.kwargs["reply_markup"]
        items = {button.callback_data for row in keyboard.inline_keyboard for button in row}
        self.assertTrue({"menu:budget", "menu:jobs", "menu:users", "pj:market:manage"}.issubset(items))
        self.assertNotIn('menu:alerts', items)
        self.callback('pj:market:manage')
        keyboard = self.client.send_message.call_args.kwargs['reply_markup']
        self.assertTrue({'menu:alerts', 'menu:proposals', 'menu:sources', 'menu:autocollect'}.issubset(
            {button.callback_data for row in keyboard.inline_keyboard for button in row}))
        self.message("/start", user=2)
        keyboard = self.client.send_message.call_args.kwargs["reply_markup"]
        self.assertNotIn("menu:admin", {button.callback_data for row in keyboard.inline_keyboard for button in row})

    def test_data_and_verification_menu_help_sends_copyable_commands_without_queueing(self):
        self.callback("menu:data", user=2).answer.assert_called_once_with()
        self.callback("menu:export_help", user=2).answer.assert_called_once_with()
        self.callback("menu:backtest_help", user=2).answer.assert_called_once_with()
        self.callback("menu:verify_help").answer.assert_called_once_with()
        self.assertIn("/export wordstat xlsx", self.text())
        self.assertIn("/verify", self.text())
        self.assertEqual(self.store.query("SELECT * FROM jobs"), [])

    def test_budget_view_prices_currency_preserve_other_provider_and_use_shared_setter(self):
        from thesaurus.budget import set_budgets

        with patch("thesaurus.telegram_admin.set_budgets", wraps=set_budgets) as setter:
            self.message("/budget ai 5")
            setter.assert_called_once_with(self.store, {"ai_calls": 5})
        self.message("/budget currency RUB")
        self.message("/budget price wordstat per_call 2")
        self.message("/budget price ai input_per_million 30")
        config = self.store.get_setting("budgets")
        self.assertEqual(config["prices"], {"wordstat": {"per_call": 2}, "ai": {"input_per_million": 30}})
        self.message("/budget")
        self.assertIn("RUB", self.text())
        self.assertIn("Wordstat 300, AI 5", self.text())
        self.message("/budget price ai clear")
        self.assertEqual(self.store.get_setting("budgets")["prices"], {"wordstat": {"per_call": 2}})

    def test_bad_budget_values_do_not_mutate_state_or_relabel_existing_costs(self):
        previous = self.store.get_setting("budgets")
        for command in ("/budget ai -1", "/budget ai 10001", "/budget ai ²", "/budget ai_tokens 0",
                        "/budget price ai per_call nan", "/budget price ai per_call inf", "/budget price ai per_call -2",
                        "/budget price unknown per_call 5", "/budget currency TOOLONGCURRENCY"):
            self.message(command)
            self.assertEqual(previous, self.store.get_setting("budgets"), command)
        self.store.execute("INSERT INTO usage(provider,day,cost,created_at) VALUES('ai','2026-01-01',1,?)", (utcnow(),))
        self.message("/budget currency USD")
        self.assertEqual(previous, self.store.get_setting("budgets"))

    def test_alert_rule_create_edit_thresholds_toggle_and_filter_validation(self):
        self.message("/alerts add Пожары в Москве|пожар|213")
        row = self.store.one("SELECT * FROM alert_rules WHERE id=2")
        self.assertEqual((row["name"], row["metric"], row["region_id"], row["enabled"]), ("Пожары в Москве", "пожар", 213, 0))
        self.message("/alerts set 2 pct=25 z=3 coverage=0.8 baseline=15 cooldown=6")
        self.callback("alert:on:2").answer.assert_called_once_with()
        row = self.store.one("SELECT * FROM alert_rules WHERE id=2")
        self.assertEqual((row["pct_threshold"], row["z_threshold"], row["min_coverage"], row["min_baseline"], row["cooldown_hours"]),
                         (25, 3, .8, 15, 6))
        self.assertEqual(row["enabled"], 1)
        self.message("/alerts edit 2 Все взрывы|взрыв|*")
        row = self.store.one("SELECT * FROM alert_rules WHERE id=2")
        self.assertEqual((row["name"], row["metric"], row["region_id"], row["enabled"]), ("Все взрывы", "взрыв", None, 1))
        self.message("/alerts off 2")
        self.assertEqual(self.store.one("SELECT enabled FROM alert_rules WHERE id=2")["enabled"], 0)

    def test_invalid_alert_commands_preserve_rule_and_stale_rule_can_be_disabled(self):
        old = self.store.query("SELECT * FROM alert_rules")
        for command in ("/alerts add Wrong|arbitrary|213", "/alerts add Wrong|пожар|999", "/alerts set 1 pct=inf",
                        "/alerts set 1 z=-1", "/alerts set 1 coverage=1.1", "/alerts set 1 baseline=nan",
                        "/alerts set 1 pct=2 pct=3", "/alerts set 1 enabled=1", "/alerts on 999", "/alerts edit 1 |*|*"):
            self.message(command)
            self.assertEqual(old, self.store.query("SELECT * FROM alert_rules"), command)
        rule_id = save_alert_rule(self.store, {"name": "Old region", "metric": "пожар", "region_id": 213, "enabled": 1})
        self.store.set_regions([2])
        self.message(f"/alerts off {rule_id}")
        self.assertEqual(self.store.one("SELECT enabled FROM alert_rules WHERE id=?", (rule_id,))["enabled"], 0)
        self.message(f"/alerts on {rule_id}")
        self.assertEqual(self.store.one("SELECT enabled FROM alert_rules WHERE id=?", (rule_id,))["enabled"], 0)

    def test_proposal_requests_queue_once_per_event_and_callbacks_apply(self):
        self.message("/proposals request")
        self.message("/proposals request")
        self.callback("proposal:request:new")
        self.callback("proposal:request:new")
        jobs = self.store.query("SELECT kind FROM jobs")
        self.assertEqual(jobs, [{"kind": "propose_keywords"}, {"kind": "propose_keywords"}])
        self.assertTrue(all(json.loads(row["payload"])["actor_id"] == 1
                            for row in self.store.query("SELECT payload FROM jobs")))
        approved, rejected = self.proposal(), self.proposal([{"phrase": "не нужна"}])
        before = self.store.version("keyword")
        self.callback(f"proposal:approve:{approved}").answer.assert_called_once_with()
        self.message(f"/proposals reject {rejected}")
        self.assertEqual(self.store.version("keyword"), before + 1)
        self.assertIn("новая фраза", [row["phrase"] for row in self.store.keywords()])
        self.assertNotIn("не нужна", [row["phrase"] for row in self.store.keywords()])
        self.assertEqual([row["state"] for row in self.store.query("SELECT state FROM keyword_proposals ORDER BY id")],
                         ["approved", "rejected"])
        self.callback(f"proposal:approve:{approved}")
        self.assertEqual(self.store.version("keyword"), before + 1)

    def test_queued_proposal_rechecks_revoked_or_demoted_admin_before_provider_call(self):
        providers = Mock()
        worker = Worker(self.store, Settings(root=self.folder), providers=providers, charts=False)
        for enabled, role in ((False, "admin"), (True, "subscriber")):
            with self.subTest(enabled=enabled, role=role):
                self.store.add_user(1, role="admin", subscriptions='all')
                self.message("/proposals request", message_id=uuid4().hex)
                self.store.add_user(1, role=role, enabled=enabled, subscriptions='all')
                self.assertTrue(worker.process_next())
                job = self.store.one("SELECT state,error FROM jobs ORDER BY id DESC LIMIT 1")
                self.assertEqual(job["state"], "failed")
                self.assertIn("PermissionError", job["error"])
                providers.propose_keywords.assert_not_called()
                self.assertEqual(self.store.query("SELECT * FROM keyword_proposals"), [])

    def test_proposal_worker_keeps_internal_jobs_without_actor_compatible(self):
        providers = Mock()
        providers.propose_keywords.return_value = [{"phrase": "новое предложение", "theme": "тема"}]
        worker = Worker(self.store, Settings(root=self.folder), providers=providers, charts=False)
        self.store.enqueue("propose_keywords")
        self.assertTrue(worker.process_next())
        providers.propose_keywords.assert_called_once_with(self.store.keywords())
        self.assertEqual(self.store.one("SELECT state FROM jobs")["state"], "done")
        self.assertEqual(json.loads(self.store.one("SELECT entries FROM keyword_proposals")["entries"]),
                         [{"phrase": "новое предложение", "theme": "тема"}])

    def test_proposal_commit_failure_rolls_back_keywords_and_version_history(self):
        identifier = self.proposal()
        before, version = self.store.keywords(), self.store.version("keyword")
        self.store.execute("CREATE TRIGGER reject_proposal_update BEFORE UPDATE ON keyword_proposals "
                           "BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            resolve_proposal(self.store, identifier, "approve")
        self.assertEqual(self.store.keywords(), before)
        self.assertEqual(self.store.version("keyword"), version)
        self.assertIsNone(self.store.get_setting(f"keywords_history_{version + 1}"))
        self.assertEqual(self.store.one("SELECT state FROM keyword_proposals")["state"], "pending")

    def test_concurrent_proposal_approval_only_updates_once(self):
        identifier = self.proposal()
        before = self.store.version("keyword")

        def approve():
            try:
                return resolve_proposal(self.store, identifier, "approve")
            except ValueError:
                return "already-reviewed"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _: approve(), range(2)))
        self.assertCountEqual(outcomes, ["approved", "already-reviewed"])
        self.assertEqual(self.store.version("keyword"), before + 1)

    def test_invalid_proposal_can_be_rejected_without_changing_keywords(self):
        identifier = self.proposal([{"phrase": ""}])
        before = self.store.keywords()
        self.message(f"/proposals approve {identifier}")
        self.assertEqual(self.store.one("SELECT state FROM keyword_proposals")["state"], "pending")
        self.message(f"/proposals reject {identifier}")
        self.assertEqual(self.store.one("SELECT state FROM keyword_proposals")["state"], "rejected")
        self.assertEqual(self.store.keywords(), before)

    def test_status_and_jobs_show_failures_usage_without_raw_errors_or_payload(self):
        day = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/Moscow")).date().isoformat()
        self.store.execute("INSERT INTO usage(provider,day,tokens_in,tokens_out,cost,created_at) VALUES('ai',?,12,5,0.2,?)",
                           (day, utcnow()))
        self.store.enqueue("report", {"secret": "PROVIDER_SECRET"})
        failed = self.store.enqueue("propose_keywords")
        self.store.execute("UPDATE jobs SET state='failed',error='API_KEY_SECRET' WHERE id=?", (failed,))
        self.message("/status")
        self.message("/jobs failed")
        self.assertIn("в очереди: 1", self.text())
        self.assertIn("ошибки: 1", self.text())
        self.assertIn(f"#{failed} предложения фраз — ошибки", self.text())
        self.assertIn("12/5", self.text())
        self.assertNotIn("API_KEY_SECRET", self.text())
        self.assertNotIn("PROVIDER_SECRET", self.text())

    def test_users_and_rule_pagination_are_reachable(self):
        for number in range(3, 26):
            self.store.add_user(number, subscriptions='all')
        self.callback("adminpage:users:2").answer.assert_called_once_with()
        self.assertIn("25 — подписчик", self.text())
        self.message("/alerts add Any|*|*")
        self.message("/alerts 2")
        self.assertIn("#2 Any", self.text())
        self.assertEqual(json.loads(self.store.one("SELECT value FROM settings WHERE key='regions'")["value"]), [2, 213])

    def test_user_page_resolves_only_missing_names_with_six_call_limit(self):
        from thesaurus.user_management import remember
        remember(self.store, SimpleNamespace(id=2, first_name='Анна', username='anna'), interacted=False)
        for identity in range(3, 11):
            self.store.add_user(identity, subscriptions='all')
        self.client.get_users.side_effect = lambda target: SimpleNamespace(id=target, first_name=f'Имя {target}', username=None)
        self.bot.admin_controls.users(self.client, 1, '')
        self.assertEqual([call.args[0] for call in self.client.get_users.call_args_list], [1, 3, 4, 5, 6, 7])
        text = self.client.send_message.call_args.args[1]
        self.assertIn('Анна · @anna · ID 2', text)
        self.assertIn('Имя 3 · ID 3', text)
        self.assertIn('Имя неизвестно · ID 8', text)
        self.assertIn('Telegram пока не передал имя', text)
        self.assertTrue(all(row['last_seen'] is None for row in self.store.query('SELECT last_seen FROM user_profiles')))
        keyboard = self.client.send_message.call_args.kwargs['reply_markup']
        self.assertIn('ua:refresh:1', {button.callback_data for row in keyboard.inline_keyboard for button in row})

    def test_explicit_page_refresh_reaches_remaining_users_after_lookup_failures(self):
        for identity in range(3, 27):
            self.store.add_user(identity, subscriptions='all')
        self.client.get_users.side_effect = TimeoutError('private-api-error')
        self.bot.admin_controls.users(self.client, 1, '')
        self.assertEqual([call.args[0] for call in self.client.get_users.call_args_list], list(range(1, 7)))
        self.client.get_users.reset_mock()
        self.callback('ua:refresh:1').answer.assert_called_once_with()
        self.assertEqual([call.args[0] for call in self.client.get_users.call_args_list], list(range(7, 13)))
        self.assertNotIn('private-api-error', self.text())
        self.client.get_users.reset_mock()
        self.callback('ua:refresh:2').answer.assert_called_once_with()
        self.assertEqual([call.args[0] for call in self.client.get_users.call_args_list], list(range(21, 27)))
        keyboard = self.client.send_message.call_args.kwargs['reply_markup']
        self.assertIn('ua:refresh:2', {button.callback_data for row in keyboard.inline_keyboard for button in row})

    def test_name_refresh_rechecks_admin_and_page_without_looking_up_forbidden_peers(self):
        for identity, chat, kind in ((2, 2, 'private'), (99, 99, 'private'), (1, -1, 'group')):
            self.callback('ua:refresh:1', user=identity, chat=chat, kind=kind)
        for invalid in ('0', '-1', 'x', '²'):
            self.callback('ua:refresh:'+invalid).answer.assert_called_once()
        self.client.get_users.assert_not_called()
        self.assertFalse(self.store.query('SELECT * FROM access_audit'))

    def test_unknown_profile_is_explicit_and_wrong_returned_identity_is_never_saved(self):
        self.client.get_users.return_value = SimpleNamespace(id=999, first_name='Чужое имя', username='wrong_user')
        self.bot.admin_controls.users(self.client, 1, '')
        self.assertIn('Имя неизвестно · ID 2', self.client.send_message.call_args.args[1])
        self.assertFalse(self.store.query('SELECT * FROM user_profiles'))
        self.bot.admin_controls.user_subscriptions(self.client, 1, 2)
        text = self.client.send_message.call_args.args[1]
        self.assertTrue(text.startswith('Имя неизвестно · ID 2'))
        self.assertIn('Telegram пока не передал имя', text)
        self.assertNotIn('wrong_user', text)
        self.assertEqual(self.store.one('SELECT role FROM users WHERE user_id=2')['role'], 'subscriber')


if __name__ == "__main__":
    unittest.main()
