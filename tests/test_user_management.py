"""Offline granular rights, identities, durable notices and administrator changes."""
import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_subscription_access as fixtures

from thesaurus import feature_access as features
from thesaurus import subscription_access as sections
from thesaurus import user_management as users
from thesaurus.config import Settings
from thesaurus.delivery import deliver_pending, prepare_delivery
from thesaurus.market_store import change_access
from thesaurus.service import Worker, bootstrap
from thesaurus.store import Store, dumps
from thesaurus.telegram_data_jobs import authorized_report


class UserManagementTests(unittest.TestCase):
    setUp = fixtures.SubscriptionAccessTests.setUp
    message = fixtures.SubscriptionAccessTests.message
    click = fixtures.SubscriptionAccessTests.click
    view = fixtures.SubscriptionAccessTests.view
    report = fixtures.SubscriptionAccessTests.report

    def notice(self):
        return self.store.one('SELECT * FROM access_notices ORDER BY id DESC LIMIT 1')

    def ready_notices(self):
        self.store.execute("UPDATE access_notices SET ready_at='2000-01-01' WHERE state='pending'")

    def test_section_revocation_names_removed_section_when_functions_overlap(self):
        sections.change(self.store, 1, 2, 'market', True)
        sections.change(self.store, 1, 2, 'ruonia', True)
        users.remember(self.store, SimpleNamespace(id=2))
        users.send_notices(self.store, self.client, force=True)
        before = features.assigned(self.store, 2)
        sections.change(self.store, 1, 2, 'ruonia', False)
        self.assertEqual(features.assigned(self.store, 2), before)
        self.assertEqual(users.send_notices(self.store, self.client), 0)
        self.ready_notices()
        self.assertEqual(users.send_notices(self.store, self.client), 1)
        text = self.client.send_message.call_args.args[1]
        removed, remaining = text.split('Осталось доступно · разделы:', 1)
        self.assertIn('Отключено:', removed)
        self.assertIn('Раздел: '+sections.CATALOG['ruonia'], removed)
        self.assertNotIn(sections.CATALOG['ruonia'], remaining)
        self.assertIn(sections.CATALOG['market'], remaining)
        self.assertIn(features.CATALOG['charts'][0], remaining)

    def test_rapid_changes_merge_exact_deltas_and_send_once_after_quiet_period(self):
        sections.change(self.store, 1, 2, 'vivino', True)
        sections.change(self.store, 1, 2, 'life', True)
        users.remember(self.store, SimpleNamespace(id=2))
        users.send_notices(self.store, self.client, force=True)
        self.client.send_message.reset_mock()
        features.change(self.store, 1, 2, 'vivino_plan', False)
        self.assertEqual(users.send_notices(self.store, self.client), 0)
        sections.change(self.store, 1, 2, 'life', False)
        self.assertEqual(users.send_notices(self.store, self.client), 0)
        features.change(self.store, 1, 2, 'dollar', True)
        self.assertEqual(users.send_notices(self.store, self.client), 0)
        self.assertEqual(len(self.store.query("SELECT id FROM access_notices WHERE state='pending'")), 1)
        self.ready_notices()
        self.assertEqual(users.send_notices(self.store, self.client), 1)
        self.assertEqual(users.send_notices(self.store, self.client), 0)
        self.assertEqual(self.client.send_message.call_count, 1)
        text = self.client.send_message.call_args.args[1]
        changes, remaining = text.split('Осталось доступно · разделы:', 1)
        self.assertIn('Отключено:', changes)
        self.assertIn(features.CATALOG['vivino_plan'][0], changes)
        self.assertIn(sections.CATALOG['life'], changes)
        self.assertIn(features.CATALOG['life_finance'][0], changes)
        self.assertIn('Добавлено:', changes)
        self.assertIn(features.CATALOG['dollar'][0], changes)
        self.assertIn(features.CATALOG['wine_cellar'][0], remaining)
        self.assertNotIn(features.CATALOG['vivino_plan'][0], remaining)
        self.assertNotIn(features.CATALOG['life_finance'][0], remaining)

    def test_reverting_quick_changes_sends_no_identical_summary(self):
        sections.change(self.store, 1, 2, 'market', True)
        users.remember(self.store, SimpleNamespace(id=2))
        users.send_notices(self.store, self.client, force=True)
        self.client.send_message.reset_mock()
        features.change(self.store, 1, 2, 'charts', False)
        features.change(self.store, 1, 2, 'charts', True)
        self.ready_notices()
        self.assertEqual(users.send_notices(self.store, self.client), 0)
        self.client.send_message.assert_not_called()
        self.assertFalse(self.store.query("SELECT id FROM access_notices WHERE state='pending'"))

    def test_sender_race_keeps_each_notice_its_own_snapshot(self):
        features.change(self.store, 1, 2, 'dollar', True)
        users.remember(self.store, SimpleNamespace(id=2))
        self.ready_notices()
        original = users.notice_text
        def change_during_send(store, target, **kwargs):
            features.change(store, 1, 2, 'charts', True)
            return original(store, target, **kwargs)
        with patch.object(users, 'notice_text', side_effect=change_during_send):
            self.assertEqual(users.send_notices(self.store, self.client), 1)
        text = self.client.send_message.call_args.args[1]
        self.assertIn(features.CATALOG['dollar'][0], text)
        self.assertNotIn(features.CATALOG['charts'][0], text)
        self.ready_notices()
        self.assertEqual(users.send_notices(self.store, self.client), 1)
        changes = self.client.send_message.call_args.args[1].split('Осталось доступно · разделы:', 1)[0]
        self.assertIn(features.CATALOG['charts'][0], changes)
        self.assertNotIn(features.CATALOG['dollar'][0], changes)

    def test_coalescing_during_flood_wait_preserves_telegram_retry_deadline(self):
        from pyrogram.errors import FloodWait
        users.remember(self.store, SimpleNamespace(id=2))
        features.change(self.store, 1, 2, 'dollar', True)
        self.client.send_message.side_effect = FloodWait(60)
        self.assertEqual(users.send_notices(self.store, self.client, force=True), 0)
        retry = self.notice()['retry_at']
        features.change(self.store, 1, 2, 'charts', True)
        self.assertEqual(self.notice()['retry_at'], retry)
        self.assertEqual(self.notice()['attempts'], 1)
        self.client.send_message.reset_mock()
        self.client.send_message.side_effect = None
        self.assertEqual(users.send_notices(self.store, self.client, force=True), 0)
        self.client.send_message.assert_not_called()
        self.store.execute("UPDATE access_notices SET retry_at='2000-01-01',ready_at='2000-01-01' WHERE state='pending'")
        self.assertEqual(users.send_notices(self.store, self.client), 1)

    def test_flood_wait_race_merges_new_changes_without_two_pending_messages(self):
        from pyrogram.errors import FloodWait
        users.remember(self.store, SimpleNamespace(id=2))
        features.change(self.store, 1, 2, 'dollar', True)
        def change_and_fail(*args, **kwargs):
            features.change(self.store, 1, 2, 'charts', True)
            raise FloodWait(60)
        self.client.send_message.side_effect = change_and_fail
        self.assertEqual(users.send_notices(self.store, self.client, force=True), 0)
        self.assertEqual(len(self.store.query("SELECT id FROM access_notices WHERE state='pending'")), 1)
        self.assertTrue(self.notice()['retry_at'])
        self.client.send_message.side_effect = None
        self.store.execute("UPDATE access_notices SET retry_at='2000-01-01',ready_at='2000-01-01' WHERE state='pending'")
        self.assertEqual(users.send_notices(self.store, self.client), 1)
        text = self.client.send_message.call_args.args[1]
        added = text.split('Осталось доступно · разделы:', 1)[0]
        self.assertIn(features.CATALOG['dollar'][0], added)
        self.assertIn(features.CATALOG['charts'][0], added)
        self.assertEqual(users.send_notices(self.store, self.client), 0)

    def test_wine_entitlement_is_independent_of_finance_and_vivino_upload(self):
        for feature in ('vivino_map', 'vivino_plan', 'wine_cellar'):
            features.change(self.store, 1, 2, feature, True)
        sections.change(self.store, 1, 2, 'life', True)
        sections.change(self.store, 1, 2, 'life', False)
        self.message('/cellar')
        self.assertIn('Моя винотека', self.view()[0])
        self.click('wc:add')
        self.assertIn('wc:manual', self.view()[1])
        self.assertIn('wc:photo', self.view()[1])
        self.assertNotIn('lf:receipt', self.view()[1])
        self.assertTrue(sections.route_allowed(self.store, 2, 'vi:upload'))
        self.assertTrue(sections.route_allowed(self.store, 2, 'wc:photo'))
        self.assertFalse(features.allowed(self.store, 2, 'life_finance'))
        self.assertFalse(sections.route_allowed(self.store, 2, 'lf:receipt'))
        self.click('wc:manual')
        for value in ('Мальбек', 'Винодельня', '2021', '2', '750', '1000', '01.01.2024', 'Шкаф'):
            self.message(value)
        self.click(next(data for data in self.view()[1] if data.startswith('wc:save:')))
        from thesaurus import wine_cellar
        lot = wine_cellar.lots(self.store, 2)[0]
        self.assertEqual((lot['name'], lot['quantity']), ('Мальбек', 2))
        self.click(f'wc:stock:{lot["id"]}:drink')
        self.click(next(data for data in self.view()[1] if data.startswith('wc:apply:')))
        self.assertEqual(wine_cellar.lots(self.store, 2)[0]['quantity'], 1)
        self.assertFalse(self.store.query('SELECT id FROM life_ledger_entries'))
        self.message('/expenses')
        self.assertIn('не подключ', self.view()[0])

    def test_version_one_upgrade_preserves_users_notices_and_granular_entitlements(self):
        features.change(self.store, 1, 2, 'vivino_map', True)
        notice_id = self.notice()['id']
        self.store.execute("UPDATE access_notices SET state='sent',message_id=456 WHERE id=?", (notice_id,))
        configured_before = features.configured(self.store, 2)
        with self.store.connection() as conn:
            for column in ('ready_at', 'after_state', 'before_state'):
                conn.execute(f'ALTER TABLE access_notices DROP COLUMN {column}')
            conn.execute("UPDATE settings SET value='1' WHERE key='user_management_version'")
        backups = len(list(self.root.glob('*.users-backup-*.sqlite3')))
        self.store = Store(self.store.path)
        self.assertEqual(self.store.get_setting('user_management_version'), users.VERSION)
        self.assertEqual(features.configured(self.store, 2), configured_before)
        self.assertFalse(features.allowed(self.store, 2, 'wine_cellar'))
        self.assertEqual((self.notice()['state'], self.notice()['message_id']), ('sent', 456))
        self.assertIsNone(self.notice()['before_state'])
        self.assertEqual(len(list(self.root.glob('*.users-backup-*.sqlite3'))), backups+1)
        Store(self.store.path)
        self.assertEqual(len(list(self.root.glob('*.users-backup-*.sqlite3'))), backups+1)

    def test_single_function_grant_does_not_open_other_market_functions(self):
        features.change(self.store, 1, 2, 'dollar', True)
        self.assertEqual(sections.assigned(self.store, 2), ['market'])
        self.assertEqual(features.assigned(self.store, 2), ['dollar'])
        self.assertTrue(sections.route_allowed(self.store, 2, 'usd:menu'))
        for route in ('cb:menu', 'ch:menu:market', 'request:market', 'ui:backtest:market',
                      'mk:open:correlations', 'ui:export:market', 'cs:market:on', 'mk:open:alerts'):
            self.assertFalse(sections.route_allowed(self.store, 2, route), route)
            self.assertTrue(self.click(route).answer.call_args.kwargs.get('show_alert'), route)
        self.message('/dollar')
        self.assertIn('Доллар для поездки', self.view()[0])
        self.click('pj:market')
        self.assertIn('usd:menu', self.view()[1])
        self.assertNotIn('ch:menu', self.view()[1])
        self.assertNotIn('cb:menu', self.view()[1])
        self.assertFalse(self.store.query('SELECT * FROM jobs'))

    def test_exact_toggle_is_idempotent_and_does_not_change_other_users(self):
        self.click('ua:open:2', 1)
        self.click('ua:functions:2', 1)
        self.click('ua:feature:2:dollar:on', 1)
        self.assertTrue(self.view()[1]['ua:feature:2:dollar:off'].startswith('✓'))
        self.assertEqual(features.assigned(self.store, 2), ['dollar'])
        self.assertFalse(features.change(self.store, 1, 2, 'dollar', True))
        self.assertEqual(len(self.store.query("SELECT * FROM access_audit WHERE action='feature_grant:dollar'")), 1)
        self.assertFalse(features.allowed(self.store, 3, 'dollar'))
        self.click('ua:feature:2:dollar:off', 1)
        self.assertFalse(features.allowed(self.store, 2, 'dollar'))
        self.assertFalse(sections.project_allowed(self.store, 2, 'market'))

    def test_commands_require_their_function_without_requiring_reports(self):
        features.change(self.store, 1, 2, 'dollar', True)
        for command in ('export market csv', 'backtest market', 'forecasts market', 'correlations'):
            with self.subTest(command=command):
                self.message('/'+command)
                self.assertIn('не подключ', self.view()[0])
        self.assertFalse(self.store.query('SELECT * FROM jobs'))
        features.change(self.store, 1, 2, 'exports', True)
        self.message('/export market csv')
        self.assertEqual(self.store.one('SELECT kind FROM jobs')['kind'], 'market_export')
        self.assertFalse(features.allowed(self.store, 2, 'reports'))
        with self.assertRaises(PermissionError):
            self.bot.market_controls.dispatch(self.client, 2, 'correlations')
        with self.assertRaises(PermissionError):
            self.bot.data_controls.dispatch(self.client, 2, 'backtest', 'ruonia')

    def test_scheduled_job_checks_schedule_function_before_provider(self):
        features.change(self.store, 1, 2, 'charts', True)
        self.store.enqueue('telegram_chart', {'actor_id': 2, 'section': 'market', 'chart_schedule': {}})
        with patch('thesaurus.charts.run_chart_job') as run:
            Worker(self.store, Settings(root=self.root), dry_run=True, charts=False).process_next()
        run.assert_not_called()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'cancelled')

    def test_export_only_user_can_use_buttons_without_chart_access(self):
        features.change(self.store, 1, 2, 'exports', True)
        self.message('/projects')
        self.click('pj:market')
        self.click('menu:data')
        self.assertIn('ui:export:menu', self.view()[1])
        self.assertNotIn('ch:menu:market', self.view()[1])
        self.click('ui:export:menu')
        self.click('ui:export:market')
        self.click('ui:export:market:csv')
        self.assertEqual(self.store.one('SELECT kind FROM jobs')['kind'], 'market_export')
        self.assertFalse(features.allowed(self.store, 2, 'charts'))

    def test_disabled_existing_user_loses_native_privileged_command_menu(self):
        from thesaurus.telegram_commands import configure_chat
        features.change(self.store, 1, 2, 'dollar', True)
        configure_chat(self.client, self.store, 2)
        self.assertIn('dollar', [c.command for c in self.client.set_bot_commands.call_args.args[0]])
        change_access(self.store, 1, 2, False)
        configure_chat(self.client, self.store, 2)
        self.assertEqual({c.command for c in self.client.set_bot_commands.call_args.args[0]}, {'start', 'projects', 'status', 'help'})

    def test_partial_event_keeps_previously_known_identity(self):
        users.remember(self.store, SimpleNamespace(id=2, first_name='Иван', username='ivan_test'))
        users.remember(self.store, SimpleNamespace(id=2))
        self.assertIn('Иван · @ivan_test', users.label(self.store, 2))

    def test_forged_permissions_roles_and_group_callbacks_are_denied(self):
        for route in ('ua:feature:2:dollar:on', 'ua:roleconfirmed:2:admin', 'ua:notify:3', 'ua:identity:3'):
            self.assertTrue(self.click(route).answer.call_args.kwargs.get('show_alert'), route)
            self.click(route, 1, kind='group').answer.assert_called_once_with('Нет доступа', show_alert=True)
        self.assertEqual(self.store.one('SELECT role FROM users WHERE user_id=2')['role'], 'subscriber')
        self.assertFalse(features.assigned(self.store, 2))
        self.assertFalse(self.store.query('SELECT * FROM access_notices'))

    def test_revoking_feature_cancels_jobs_but_preserves_preferences_and_data(self):
        features.change(self.store, 1, 2, 'dollar', True)
        features.change(self.store, 1, 2, 'reports', True)
        self.store.set_setting('dollar:2', {'forecast_date': '2027-01-01'})
        own = self.store.enqueue('dollar_analysis', {'actor_id': 2})
        other = self.store.enqueue('market_report', {'actor_id': 2})
        common = self.store.enqueue('market_collect', {})
        features.change(self.store, 1, 2, 'dollar', False)
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (own,))['state'], 'cancelled')
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (other,))['state'], 'pending')
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (common,))['state'], 'pending')
        self.assertEqual(self.store.get_setting('dollar:2'), {'forecast_date': '2027-01-01'})
        features.change(self.store, 1, 2, 'dollar', True)
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (own,))['state'], 'cancelled')

    def test_worker_rechecks_function_after_job_was_queued(self):
        features.change(self.store, 1, 2, 'dollar', True)
        features.change(self.store, 1, 2, 'dollar', False)
        job = self.store.enqueue('dollar_analysis', {'actor_id': 2})
        with patch('thesaurus.dollar_jobs.run_job') as run:
            Worker(self.store, Settings(root=self.root), dry_run=True, charts=False).process_next()
        run.assert_not_called()
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (job,))['state'], 'cancelled')

    def test_revocation_between_parts_blocks_remaining_delivery_and_old_report(self):
        features.change(self.store, 1, 2, 'dollar', True)
        identity = self.report('market')
        self.store.execute('UPDATE reports SET quality=?,full_text=? WHERE id=?',
                           (dumps({'tool': 'dollar'}), 'A'*4500, identity))
        prepare_delivery(self.store, identity, '2', 'full')
        def revoke(*args, **kwargs):
            features.change(self.store, 1, 2, 'dollar', False)
            return SimpleNamespace(id=55)
        self.client.send_message.side_effect = revoke
        self.assertEqual(deliver_pending(self.store, self.client), 1)
        self.assertEqual(self.client.send_message.call_count, 1)
        states = [r['state'] for r in self.store.query('SELECT state FROM deliveries ORDER BY part_index')]
        self.assertEqual(states[0], 'sent')
        self.assertTrue(all(state == 'cancelled' for state in states[1:]))
        with self.assertRaises(PermissionError):
            authorized_report(self.store, 2, identity)

    def test_profile_is_saved_from_private_events_and_username_changes(self):
        event = SimpleNamespace(text='/start', id=10, from_user=SimpleNamespace(id=2, first_name='Анна', last_name='Иванова', username='anna_test'),
                                chat=SimpleNamespace(id=2, type='private'))
        self.bot.handle_message(self.client, event)
        self.bot.admin_controls.users(self.client, 1, '')
        self.assertIn('Анна Иванова · @anna_test · ID 2', self.view()[0])
        event.from_user.username = 'anna_new'
        self.bot.handle_message(self.client, event)
        self.assertIn('@anna_new', users.label(self.store, 2))
        event.from_user.username = None
        self.bot.handle_message(self.client, event)
        self.assertNotIn('@', users.label(self.store, 2))
        self.assertEqual(self.store.one('SELECT role FROM users WHERE user_id=2')['role'], 'subscriber')

    def test_identity_refresh_cannot_invent_interaction_or_use_username_as_key(self):
        self.client.get_users.return_value = SimpleNamespace(id=2, first_name='Борис', last_name='', username='boris_test')
        self.bot.admin_controls.callback(self.client, 1, 'ua:identity:2')
        self.assertIn('@boris_test', self.view()[0])
        self.assertIsNone(self.store.one('SELECT last_seen FROM user_profiles WHERE user_id=2')['last_seen'])
        features.change(self.store, 1, 2, 'dollar', True)
        self.client.send_message.reset_mock()
        self.assertEqual(users.send_notices(self.store, self.client), 0)
        self.client.send_message.assert_not_called()

    def test_unstarted_user_receives_latest_welcome_once_on_start(self):
        features.change(self.store, 1, 2, 'dollar', True)
        features.change(self.store, 1, 2, 'charts', True)
        features.change(self.store, 1, 2, 'charts', False)
        self.assertEqual(users.send_notices(self.store, self.client), 0)
        self.client.send_message.reset_mock()
        self.message('/start')
        welcome = [c.args[1] for c in self.client.send_message.call_args_list if 'Здравствуйте!' in c.args[1]]
        self.assertEqual(len(welcome), 1)
        self.assertIn(features.CATALOG['dollar'][0], welcome[0])
        self.assertNotIn(features.CATALOG['charts'][0], welcome[0])
        self.assertEqual(self.notice()['state'], 'sent')
        self.message('/start')
        self.assertEqual(len([c for c in self.client.send_message.call_args_list if 'Здравствуйте!' in c.args[1]]), 1)

    def test_notification_failure_never_rolls_back_rights_or_exposes_exception_text(self):
        users.remember(self.store, SimpleNamespace(id=2, first_name='Пользователь'))
        features.change(self.store, 1, 2, 'dollar', True)
        self.client.send_message.side_effect = TimeoutError('secret-in-provider-error')
        self.assertEqual(users.send_notices(self.store, self.client, force=True), 0)
        self.assertTrue(features.allowed(self.store, 2, 'dollar'))
        self.assertEqual(self.notice()['state'], 'uncertain')
        self.assertEqual(self.notice()['error'], 'TimeoutError')
        self.assertEqual(users.send_notices(self.store, self.client), 0)
        self.assertEqual(self.client.send_message.call_count, 1)
        self.assertNotIn('secret-in-provider-error', users.status(self.store, 2))

    def test_disabled_user_still_receives_access_revocation_notice(self):
        users.remember(self.store, SimpleNamespace(id=2))
        change_access(self.store, 1, 2, False)
        self.assertFalse(self.store.authorize(2))
        self.assertEqual(users.send_notices(self.store, self.client), 1)
        self.assertIn('отключил ваш доступ', self.client.send_message.call_args.args[1])
        self.assertEqual(self.client.send_message.call_args.args[0], 2)

    def test_blocked_peer_can_retry_on_new_interaction_but_timeout_is_not_replayed(self):
        from pyrogram.errors import UserIsBlocked
        users.remember(self.store, SimpleNamespace(id=2))
        features.change(self.store, 1, 2, 'dollar', True)
        self.client.send_message.side_effect = UserIsBlocked()
        users.send_notices(self.store, self.client, force=True)
        self.assertEqual(self.notice()['state'], 'failed')
        self.client.send_message.side_effect = None
        users.release_notices(self.store, 2)
        self.assertEqual(users.send_notices(self.store, self.client, force=True), 1)
        features.change(self.store, 1, 2, 'dollar', False)
        self.store.execute("UPDATE access_notices SET state='sending' WHERE id=?", (self.notice()['id'],))
        users.recover_notices(self.store)
        users.release_notices(self.store, 2)
        self.assertEqual(self.notice()['state'], 'uncertain')

    def test_admin_role_confirmation_and_demote_preserve_personal_grants(self):
        features.change(self.store, 1, 2, 'dollar', True)
        self.bot.admin_controls.callback(self.client, 1, 'ua:role:2:admin')
        self.assertEqual(self.store.one('SELECT role FROM users WHERE user_id=2')['role'], 'subscriber')
        self.assertIn('ua:roleconfirmed:2:admin', self.view()[1])
        self.bot.admin_controls.callback(self.client, 1, 'ua:roleconfirmed:2:admin')
        self.assertTrue(self.store.authorize(2, admin=True))
        self.assertEqual(features.assigned(self.store, 2), list(features.CATALOG))
        users.change_role(self.store, 1, 2, 'subscriber')
        self.assertFalse(self.store.authorize(2, admin=True))
        self.assertEqual(features.assigned(self.store, 2), ['dollar'])
        self.assertFalse(features.allowed(self.store, 2, 'exports'))
        self.assertEqual(len(self.store.query("SELECT * FROM access_audit WHERE action LIKE 'role:%'")), 2)

    def test_protected_self_disabled_and_unknown_role_changes_rejected(self):
        users.change_role(self.store, 1, 2, 'admin')
        self.store.set_setting('protected_admin_ids', [2])
        for actor, target, role in ((1, 2, 'subscriber'), (1, 1, 'subscriber'), (3, 2, 'subscriber'), (1, 999, 'admin'), (1, 3, 'owner')):
            with self.subTest(actor=actor, target=target, role=role), self.assertRaises((PermissionError, ValueError)):
                users.change_role(self.store, actor, target, role)
        with self.assertRaises(ValueError):
            change_access(self.store, 1, 2, False)
        change_access(self.store, 1, 3, False)
        with self.assertRaises(ValueError):
            users.change_role(self.store, 1, 3, 'admin')
        bootstrap(self.store, Settings(root=self.root, admin_ids=(1,)))
        self.assertEqual(self.store.get_setting('protected_admin_ids'), [1])
        self.assertTrue(self.store.authorize(2, admin=True))

    def test_migration_preserves_prior_grants_has_backup_and_is_idempotent(self):
        result = self.report('market')
        self.store.enqueue('market_backtest', {'actor_id': 2, 'report_id': result})
        with self.store.connection() as conn:
            conn.execute('DROP TABLE access_notices')
            conn.execute('DROP TABLE user_profiles')
            conn.execute("DELETE FROM settings WHERE key='user_management_version' OR key LIKE 'user_features:%'")
        before = len(list(self.root.glob('*.users-backup-*.sqlite3')))
        restored = Store(self.store.path)
        self.assertEqual(features.assigned(restored, 3), features.for_sections(['ruonia']))
        self.assertEqual(features.report_feature(restored.one('SELECT * FROM reports WHERE id=?', (result,))), 'backtests')
        self.assertEqual(len(list(self.root.glob('*.users-backup-*.sqlite3'))), before+1)
        Store(self.store.path)
        self.assertEqual(len(list(self.root.glob('*.users-backup-*.sqlite3'))), before+1)
        self.assertFalse(restored.query('SELECT * FROM access_notices'))

    def test_migration_failure_rolls_back_tables_and_marker(self):
        with self.store.connection() as conn:
            conn.execute('DROP TABLE access_notices')
            conn.execute('DROP TABLE user_profiles')
            conn.execute("DELETE FROM settings WHERE key='user_management_version'")
        with patch.object(users, 'SCHEMA', users.SCHEMA+'invalid SQL;'), self.assertRaises(sqlite3.DatabaseError):
            users.migrate(self.store)
        self.assertIsNone(self.store.get_setting('user_management_version'))
        self.assertFalse(self.store.one("SELECT name FROM sqlite_master WHERE name='user_profiles'"))
        self.assertTrue(self.store.authorize(1, admin=True))


if __name__ == '__main__':
    unittest.main()
