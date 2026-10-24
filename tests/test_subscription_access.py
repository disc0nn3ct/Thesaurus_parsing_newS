"""Admin-issued project access, independent from personal delivery settings."""
import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from thesaurus import subscription_access as access
from thesaurus.config import Settings
from thesaurus.delivery import deliver_pending, prepare_delivery
from thesaurus.market_store import change_access
from thesaurus.scheduling import seed_subscriptions, update_subscription
from thesaurus.service import Worker
from thesaurus.store import Store, dumps, utcnow
from thesaurus.telegram_bot import BotControls
from thesaurus.telegram_commands import configure_chat
from thesaurus.telegram_data_jobs import actor_scope, authorized_report


class SubscriptionAccessTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]/'.test-tmp'/('access-'+uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root/'state.sqlite3')
        self.store.add_user(1, role='admin')
        self.store.add_user(2)
        self.store.add_user(3, subscriptions=['ruonia'])
        self.store.add_user(4, subscriptions=['vivino'])
        self.bot = BotControls(self.store)
        self.client = Mock()
        self.client.send_message.return_value = SimpleNamespace(id=123)
        self.client.edit_message_text.return_value = SimpleNamespace(id=123)
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No network')))

    def message(self, text, actor=2):
        self.bot.handle_message(self.client, SimpleNamespace(text=text, id=99,
            from_user=SimpleNamespace(id=actor), chat=SimpleNamespace(id=actor, type='private')))

    def click(self, data, actor=2, kind='private'):
        event = SimpleNamespace(data=data, id=uuid4().hex, from_user=SimpleNamespace(id=actor),
            message=SimpleNamespace(id=900, chat=SimpleNamespace(id=actor, type=kind)), answer=Mock())
        self.bot.handle_callback(self.client, event)
        return event

    def view(self):
        call = next(c for c in reversed(self.client.mock_calls) if c[0] in {'send_message', 'edit_message_text'})
        text = call.args[2] if call[0] == 'edit_message_text' else call.args[1]
        markup = call.kwargs.get('reply_markup')
        return text, {b.callback_data: b.text for row in markup.inline_keyboard for b in row} if markup else {}

    def report(self, section='ruonia', parts=None):
        identity = uuid4().hex
        self.store.execute('INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,parts) VALUES(?,?,?,?,?,?,?,?)',
                           (identity, section, utcnow(), 1, 0, 'Brief', 'Full', dumps(parts or [])))
        return identity

    def test_new_users_have_no_projects_even_after_allow_start_and_seeding(self):
        change_access(self.store, 1, 9, True)
        self.message('/start', 9)
        seed_subscriptions(self.store, '9', 9)
        self.assertEqual(access.assigned(self.store, 9), [])
        text, buttons = self.view()
        self.assertIn('Пока нет подключённых проектов', text)
        self.assertFalse({'pj:market', 'pj:life', 'pj:vivino'} & buttons.keys())
        self.assertEqual(self.store.one('SELECT role FROM users WHERE user_id=9')['role'], 'subscriber')

    def test_project_and_nested_buttons_only_show_assigned_sections(self):
        self.message('/start', 4)
        _, buttons = self.view()
        self.assertIn('pj:life', buttons)
        self.click('pj:life', 4)
        self.assertIn('pj:vivino', self.view()[1])
        self.assertNotIn('lf:home', self.view()[1])
        self.assertNotIn('pj:market', buttons)
        self.message('/start', 3)
        self.click('pj:market', 3)
        _, buttons = self.view()
        self.assertIn('ch:menu', buttons)
        self.assertNotIn('usd:menu', buttons)
        self.assertNotIn('cb:menu', buttons)
        self.click('ch:menu', 3)
        _, buttons = self.view()
        self.assertIn('ch:menu:ruonia', buttons)
        self.assertNotIn('ch:menu:market', buttons)
        self.assertNotIn('ch:menu:wordstat', buttons)

    def test_commands_and_forged_callbacks_cannot_open_unassigned_sections(self):
        for command in ['/vivino', '/dollar', '/market', '/report ruonia', '/chart ruonia', '/subscriptions ruonia on', '/export ruonia csv']:
            self.message(command)
            self.assertIn('не подключ', self.view()[0], command)
        for route in ['pj:market', 'pj:vivino', 'vi:upload', 'cb:menu', 'usd:menu', 'request:ruonia',
                      'ch:menu:ruonia', 'cs:ruonia:on', 'sub:ruonia:on', 'ui:subscription:ruonia:enabled:on']:
            event = self.click(route)
            self.assertTrue(event.answer.call_args.kwargs.get('show_alert'), route)
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.assertIsNone(self.store.get_setting('project:vivino:2:dataset'))

    def test_admin_can_assign_by_button_subscriber_cannot_assign_self_or_others(self):
        self.click('ua:open:2', 1)
        self.assertIn('ua:set:2:vivino:on', self.view()[1])
        self.click('ua:set:2:vivino:on', 1)
        self.assertEqual(access.assigned(self.store, 2), ['vivino'])
        self.assertTrue(self.view()[1]['ua:set:2:vivino:off'].startswith('✓'))
        self.click('ua:set:2:market:on', 2).answer.assert_called_once_with('Действие доступно администратору', show_alert=True)
        self.click('ua:set:3:market:on', 2).answer.assert_called_once_with('Действие доступно администратору', show_alert=True)
        self.click('ua:set:2:market:on', 1, kind='group').answer.assert_called_once_with('Нет доступа', show_alert=True)
        self.assertFalse(access.allowed(self.store, 2, 'market'))
        self.assertEqual(self.store.one('SELECT role FROM users WHERE user_id=2')['role'], 'subscriber')

    def test_admin_has_all_projects_and_cannot_be_restricted_via_user_card(self):
        self.assertEqual(set(access.assigned(self.store, 1)), set(access.CATALOG))
        with self.assertRaises(ValueError):
            access.change(self.store, 1, 1, 'market', False)
        with self.assertRaises(PermissionError):
            access.change(self.store, 2, 3, 'market', True)

    def test_grants_do_not_enable_schedules_and_existing_settings_remain_personal(self):
        seed_subscriptions(self.store, '2', 2)
        self.store.set_setting('chart:2:ruonia', {'period': 'year', 'horizon': 8})
        with self.assertRaises(PermissionError):
            update_subscription(self.store, 2, 'ruonia', enabled=True)
        access.change(self.store, 1, 2, 'ruonia', True)
        self.assertEqual(self.store.one("SELECT enabled FROM subscriptions WHERE user_id=2 AND section='ruonia'")['enabled'], 0)
        update_subscription(self.store, 2, 'ruonia', times=['09:00'], weekdays=[4], enabled=True)
        self.assertEqual(self.store.get_setting('chart:2:ruonia')['period'], 'year')
        with self.assertRaises(PermissionError):
            actor_scope(self.store, 2, 'wordstat')
        self.assertEqual(actor_scope(self.store, 2, 'ruonia')['actor_id'], 2)

    def test_subscription_list_hides_templates_without_grants(self):
        self.message('/subscriptions', 3)
        _, buttons = self.view()
        self.assertIn('ui:subscription:ruonia', buttons)
        self.assertNotIn('ui:subscription:market', buttons)
        self.assertNotIn('ui:subscription:ai', buttons)

    def test_duplicate_allow_and_restore_preserve_grants_without_starting_delivery(self):
        change_access(self.store, 1, 3, True)
        update_subscription(self.store, 3, 'ruonia', enabled=True, times=['09:15'])
        self.store.set_setting('chart_schedule:3:ruonia', {'enabled': True, 'times': ['09:15']})
        change_access(self.store, 1, 3, False)
        change_access(self.store, 1, 3, True)
        self.assertEqual(access.assigned(self.store, 3), ['ruonia'])
        row = self.store.one("SELECT enabled,times FROM subscriptions WHERE user_id=3 AND section='ruonia'")
        self.assertFalse(row['enabled'])
        self.assertEqual(json.loads(row['times']), ['09:15'])
        self.assertFalse(self.store.get_setting('chart_schedule:3:ruonia')['enabled'])

    def test_revocation_cancels_only_personal_section_and_regrant_keeps_schedules_off(self):
        seed_subscriptions(self.store, '3', 3)
        update_subscription(self.store, 3, 'ruonia', enabled=True)
        job = self.store.enqueue('telegram_chart', {'actor_id': 3, 'section': 'ruonia'})
        shared = self.store.enqueue('report', {'section': 'ruonia'})
        other = self.store.enqueue('vivino_map', {'actor_id': 4})
        self.store.set_setting('chart_schedule:3:ruonia', {'enabled': True, 'times': ['08:00'], 'generation': 'old'})
        access.change(self.store, 1, 3, 'ruonia', False)
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (job,))['state'], 'cancelled')
        for identity in [shared, other]:
            self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (identity,))['state'], 'pending')
        access.change(self.store, 1, 3, 'ruonia', True)
        self.assertFalse(self.store.get_setting('chart_schedule:3:ruonia')['enabled'])
        self.assertFalse(self.store.one("SELECT enabled FROM subscriptions WHERE user_id=3 AND section='ruonia'")['enabled'])
        self.assertEqual(len(self.store.query("SELECT * FROM access_audit WHERE action LIKE '%:ruonia'")), 2)

    def test_revocation_after_first_delivery_blocks_remaining_parts_and_history(self):
        identity = self.report()
        self.store.execute('UPDATE reports SET full_text=? WHERE id=?', ('a'*5000, identity))
        prepare_delivery(self.store, identity, '3', detail='full')
        def sent(*args, **kwargs):
            access.change(self.store, 1, 3, 'ruonia', False)
            return SimpleNamespace(id=321)
        self.client.send_message.side_effect = sent
        deliver_pending(self.store, self.client)
        self.assertEqual(self.client.send_message.call_count, 1)
        rows = self.store.query('SELECT state FROM deliveries WHERE report_id=? ORDER BY part_index', (identity,))
        self.assertEqual([row['state'] for row in rows], ['sent', 'cancelled'])
        self.assertEqual(prepare_delivery(self.store, identity, '3'), [])
        with self.assertRaises(PermissionError):
            authorized_report(self.store, 3, identity)

    def test_worker_rechecks_scope_before_collection_even_for_forged_job(self):
        identity = self.store.enqueue('report', {'actor_id': 2, 'recipient': '2', 'section': 'ruonia'})
        worker = Worker(self.store, Settings(root=self.root), dry_run=True, charts=False)
        with patch.object(worker.collector, 'collect', side_effect=AssertionError('Must not collect')):
            self.assertTrue(worker.process_next())
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (identity,))['state'], 'cancelled')

    def test_legacy_rights_are_frozen_once_with_backup_no_future_blanket_grant(self):
        self.store.execute("DELETE FROM settings WHERE key='subscription_access_version' OR key LIKE 'user_subscriptions:%'")
        upgraded = Store(self.store.path)
        self.assertTrue(access.allowed(upgraded, 2, 'market'))
        self.assertEqual(len(list(self.root.glob('*.access-backup-*'))), 1)
        access.change(upgraded, 1, 2, 'market', False)
        Store(self.store.path)
        self.assertFalse(access.allowed(upgraded, 2, 'market'))
        self.assertEqual(len(list(self.root.glob('*.access-backup-*'))), 1)
        change_access(upgraded, 1, 8, True)
        self.assertEqual(access.assigned(upgraded, 8), [])

    def test_native_command_menu_is_personal(self):
        configure_chat(self.client, self.store, 4)
        commands = {command.command for command in self.client.set_bot_commands.call_args.args[0]}
        self.assertIn('vivino', commands)
        self.assertFalse({'dollar', 'chart', 'graphs', 'subscriptions'} & commands)
        configure_chat(self.client, self.store, 3)
        commands = {command.command for command in self.client.set_bot_commands.call_args.args[0]}
        self.assertIn('chart', commands)
        self.assertNotIn('vivino', commands)
        self.assertNotIn('dollar', commands)

    def test_migration_rolls_back_partial_grants_on_failure(self):
        self.store.execute("DELETE FROM settings WHERE key='subscription_access_version' OR key LIKE 'user_subscriptions:%'")
        with patch('thesaurus.subscription_access.dumps', side_effect=[dumps(list(access.CATALOG)), ValueError('failure')]), self.assertRaises(ValueError):
            access.migrate(self.store)
        self.assertIsNone(self.store.get_setting('subscription_access_version'))
        self.assertEqual(self.store.query("SELECT * FROM settings WHERE key LIKE 'user_subscriptions:%'"), [])
        access.migrate(self.store)
        self.assertTrue(access.allowed(self.store, 2, 'market'))

    def test_revocation_between_form_validation_and_write_cannot_enable_schedule(self):
        seed_subscriptions(self.store, '3', 3)
        def revoke_then_seed(*args):
            access.change(self.store, 1, 3, 'ruonia', False)
            seed_subscriptions(*args)
        with patch('thesaurus.scheduling.seed_subscriptions', side_effect=revoke_then_seed), self.assertRaises(PermissionError):
            update_subscription(self.store, 3, 'ruonia', enabled=True)
        self.assertFalse(self.store.one("SELECT enabled FROM subscriptions WHERE user_id=3 AND section='ruonia'")['enabled'])


if __name__ == '__main__':
    unittest.main()
