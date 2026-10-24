"""Offline acceptance of native menus, Friday charts, revocation and alerts."""
import asyncio
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from thesaurus import chart_schedules, market
from thesaurus.charts import bounds, run_chart_job
from thesaurus.config import Settings
from thesaurus.delivery import deliver_pending
from thesaurus.scheduling import update_subscription
from thesaurus.store import Store, utcnow
from thesaurus.telegram_bot import BotControls
from thesaurus.telegram_commands import configure, configure_chat

NOW = datetime(2026, 10, 9, 13, tzinfo=timezone.utc)  # Friday, 16:00 Moscow.


class ChartScheduleTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / '.test-tmp'
        root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.store.add_user(3, subscriptions='all')
        self.settings = Settings(root=self.root)
        last = date.fromisoformat(utcnow()[:10]) - timedelta(days=1)
        self.store.save_observations([{'source': 'ruonia', 'metric': 'RUONIA', 'frequency': 'daily',
            'date': (last - timedelta(days=119-i)).isoformat(), 'value': 14 + i*.001, 'unit': '%', 'retrieved_at': utcnow()}
            for i in range(120)])
        self.client = Mock()
        self.client.send_message.return_value = SimpleNamespace(id=100)
        self.bot = BotControls(self.store)
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No network in tests')))

    def config(self, actor=2):
        options = {'identity': 'RUONIA', 'period': 'month', 'horizon': 2}
        config = chart_schedules.defaults(actor, 'ruonia', options)
        config.update(enabled=True, refresh=False, created_at=(NOW-timedelta(days=8)).isoformat())
        self.store.set_setting(chart_schedules.key(actor, 'ruonia'), config)
        return config

    def event(self, text, actor=2):
        return SimpleNamespace(text=text, id=1, from_user=SimpleNamespace(id=actor), chat=SimpleNamespace(id=actor, type='private'))

    def click(self, data, actor=2):
        event = SimpleNamespace(data=data, id='callback', from_user=SimpleNamespace(id=actor),
                                message=SimpleNamespace(id=900, chat=SimpleNamespace(id=actor, type='private')), answer=Mock())
        self.bot.handle_callback(self.client, event)
        return event

    def test_native_commands_and_menu_button_are_configured(self):
        client = SimpleNamespace(set_bot_commands=AsyncMock(), invoke=AsyncMock())
        asyncio.run(configure(client))
        commands = client.set_bot_commands.call_args.args[0]
        self.assertEqual({command.command for command in commands}, {'start', 'projects', 'status', 'help'})
        configure_chat(self.client, self.store, 2)
        self.assertIn('chart', [command.command for command in self.client.set_bot_commands.call_args.args[0]])
        self.assertIn('graphs', [command.command for command in self.client.set_bot_commands.call_args.args[0]])
        self.assertNotIn('admin', [command.command for command in commands])
        self.assertEqual(type(client.invoke.call_args.args[0].button).__name__, 'BotMenuButtonCommands')

    def test_private_admin_command_scope_and_unauthorized_user(self):
        configure_chat(self.client, self.store, 1)
        self.assertIn('admin', [command.command for command in self.client.set_bot_commands.call_args.args[0]])
        configure_chat(self.client, self.store, 2)
        self.assertNotIn('admin', [command.command for command in self.client.set_bot_commands.call_args.args[0]])
        self.client.reset_mock()
        configure_chat(self.client, self.store, 999)
        self.client.set_bot_commands.assert_not_called()

    def test_native_menu_failure_does_not_hide_start(self):
        self.client.set_bot_commands.side_effect = RuntimeError('not exposed')
        self.bot.market_controls.pending(2, 'chart:market')
        self.bot.handle_message(self.client, self.event('/start'))
        self.assertIn('Главное меню', self.client.send_message.call_args.args[1])
        self.assertNotIn('not exposed', self.client.send_message.call_args.args[1])

    def test_saved_ruonia_makes_graph_button_available_without_external_sources(self):
        self.bot.handle_message(self.client, self.event('/chart market'))
        buttons = [button.callback_data for row in self.client.send_message.call_args.kwargs['reply_markup'].inline_keyboard for button in row]
        self.assertIn('ch:build:market', buttons)
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM market_series WHERE source IN ('cbr','moex') AND enabled=1")['n'], 0)

    def test_friday_schedule_ui_keeps_period_horizon_and_is_off_by_default(self):
        self.store.set_setting('chart:2:ruonia', {'identity': 'RUONIA', 'period': 'quarter', 'horizon': 3})
        self.click('cs:ruonia:menu')
        config = self.store.get_setting('chart_schedule:2:ruonia')
        self.assertFalse(config['enabled'])
        self.assertEqual(config['weekdays'], [4])
        self.assertEqual(config['options']['period'], 'quarter')
        self.click('cs:ruonia:on')
        self.assertTrue(self.store.get_setting('chart_schedule:2:ruonia')['enabled'])
        self.click('cs:ruonia:times')
        self.bot.handle_message(self.client, self.event('18:40'))
        self.assertEqual(self.store.get_setting('chart_schedule:2:ruonia')['times'], ['18:40'])
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_weekly_sleep_coalesces_to_one_latest_chart(self):
        config = self.config()
        chart_schedules.enqueue_due(self.store, NOW)
        chart_schedules.enqueue_due(Store(self.store.path), NOW+timedelta(days=14))
        jobs = self.store.query('SELECT * FROM jobs')
        self.assertEqual(len(jobs), 1)
        payload = json.loads(jobs[0]['payload'])
        self.assertEqual(payload['options']['horizon'], 2)
        self.assertEqual(payload['options']['anchor'], 'today')
        self.assertEqual(payload['chart_schedule']['generation'], config['generation'])
        self.assertEqual(payload['scheduled_for'][:10], '2026-10-23')

    def test_rolling_period_ends_today_instead_of_stale_latest_fact(self):
        with patch('thesaurus.charts.today', return_value='2026-10-09'):
            self.assertEqual(bounds({'period': 'month', 'horizon': 2, 'anchor': 'today'}, '2026-10-02'),
                             ('2026-09-10', '2026-10-09'))

    def test_disabled_and_revoked_schedules_enqueue_nothing(self):
        config = self.config()
        chart_schedules.save(self.store, 2, 'ruonia', config, enabled=False)
        chart_schedules.enqueue_due(self.store, NOW)
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.config()
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=2')
        chart_schedules.enqueue_due(self.store, NOW)
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_disable_cancels_scheduled_queue_but_preserves_manual_chart(self):
        config = self.config()
        chart_schedules.enqueue_due(self.store, NOW)
        manual = self.store.enqueue('telegram_chart', {'actor_id': 2, 'section': 'ruonia'})
        chart_schedules.save(self.store, 2, 'ruonia', config, enabled=False)
        self.assertEqual(self.store.one("SELECT state FROM jobs WHERE id<>?", (manual,))['state'], 'cancelled')
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (manual,))['state'], 'pending')

    def test_job_restores_same_chart_after_crash_without_duplicate_parts(self):
        self.config()
        chart_schedules.enqueue_due(self.store, NOW)
        job = self.store.one('SELECT * FROM jobs')
        payload = json.loads(job['payload'])
        identifier = run_chart_job(self.store, self.settings, job, payload)
        parts = self.store.query('SELECT * FROM deliveries')
        self.assertTrue(any(json.loads(part['payload'])['kind'] == 'photo' for part in parts))
        self.assertEqual(run_chart_job(Store(self.store.path), self.settings, job, payload), identifier)
        self.assertEqual(len(self.store.query('SELECT * FROM deliveries')), len(parts))

    def test_disabling_during_delivery_blocks_later_parts(self):
        config = self.config()
        chart_schedules.enqueue_due(self.store, NOW)
        job = self.store.one('SELECT * FROM jobs')
        run_chart_job(self.store, self.settings, job, json.loads(job['payload']))
        client = Mock()
        def send(*args, **kwargs):
            chart_schedules.save(self.store, 2, 'ruonia', config, enabled=False)
            return SimpleNamespace(id=123)
        client.send_message.side_effect = send
        deliver_pending(self.store, client)
        client.send_photo.assert_not_called()
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM deliveries WHERE state='sent'")['n'], 1)

    def test_disabled_job_does_not_refresh_or_render(self):
        config = self.config()
        chart_schedules.enqueue_due(self.store, NOW)
        job = self.store.one('SELECT * FROM jobs')
        chart_schedules.save(self.store, 2, 'ruonia', config, enabled=False)
        with patch('thesaurus.collectors.Collector.collect') as collect, patch('thesaurus.charts.chart_series') as series:
            self.assertIsNone(run_chart_job(self.store, self.settings, job, json.loads(job['payload'])))
        collect.assert_not_called()
        series.assert_not_called()

    def test_malformed_schedule_does_not_block_valid_schedule(self):
        self.config()
        self.store.set_setting('chart_schedule:3:ruonia', {'enabled': True, 'actor_id': 3})
        chart_schedules.enqueue_due(self.store, NOW)
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)

    def test_fixed_dates_rejected_for_recurring_charts(self):
        config = self.config()
        with self.assertRaises(ValueError):
            chart_schedules.save(self.store, 2, 'ruonia', config, options={'period': 'custom', 'horizon': 0,
                'date_from': '2026-01-01', 'date_to': '2026-09-01', 'identity': 'RUONIA'})

    def test_unavailable_indicator_can_still_be_disabled(self):
        config = self.config()
        update_subscription(self.store, 2, 'ruonia', phrases=['1 мес'])
        chart_schedules.save(self.store, 2, 'ruonia', config, enabled=False)
        self.assertFalse(self.store.get_setting('chart_schedule:2:ruonia')['enabled'])

    def test_private_ruonia_alert_is_independent_of_market_chart_selection(self):
        from thesaurus.market_alerts import add_rule, allowed_series
        identity = allowed_series(self.store, 2)[0]['id']
        self.store.set_setting('market_selection:2', ['unavailable.market.series'])
        self.assertEqual(market.selected(self.store, 2), [])
        self.assertIsInstance(add_rule(self.store, 2, identity, 'above', 16.), int)

    def test_alert_condition_threshold_quality_and_cooldown_are_configurable(self):
        self.store.set_setting('chart:2:ruonia', {'identity': 'RUONIA', 'period': 'quarter', 'horizon': 2})
        self.click('ch:alert:ruonia')
        rule = self.store.one('SELECT * FROM market_rules WHERE user_id=2')
        self.assertFalse(rule['enabled'])
        identifier = rule['id']
        self.click(f'ui:rule:{identifier}')
        self.click(f'ui:rulekinds:{identifier}')
        self.click(f'ui:rulekind:{identifier}:above')
        self.click(f'ui:rulethreshold:{identifier}')
        self.bot.handle_message(self.client, self.event('16,5'))
        for field, value in [('window', '120'), ('coverage', '95'), ('cooldown', '48')]:
            self.click(f'ui:ruleparams:{identifier}')
            self.click(f'ui:ruleparam:{identifier}:{field}')
            self.bot.handle_message(self.client, self.event(value))
        rule = self.store.one('SELECT * FROM market_rules WHERE id=?', (identifier,))
        self.assertEqual(rule['kind'], 'above')
        self.assertEqual(json.loads(rule['config'])['threshold'], 16.5)
        self.assertEqual(json.loads(rule['config'])['coverage'], .95)
        self.assertEqual(rule['cooldown'], 48.)
        self.click(f'ui:rulemode:{identifier}:immediate')
        self.assertTrue(self.store.one('SELECT enabled FROM market_rules WHERE id=?', (identifier,))['enabled'])

    def test_other_subscriber_cannot_change_alert(self):
        from thesaurus.market_alerts import add_rule
        row = market.selected(self.store, 2)[0]
        identifier = add_rule(self.store, 2, row['id'], 'deviation')
        event = self.click(f'ui:rulekind:{identifier}:above', actor=3)
        self.assertTrue(event.answer.call_args.kwargs['show_alert'])
        self.assertEqual(self.store.one('SELECT kind FROM market_rules WHERE id=?', (identifier,))['kind'], 'deviation')

    def test_local_ruonia_respects_personal_indicator_filters(self):
        update_subscription(self.store, 2, 'ruonia', phrases=['1 мес'])
        self.assertEqual(market.selected(self.store, 2), [])
        self.assertTrue(market.selected(self.store, 3))

    def test_dry_run_never_refreshes_or_sends(self):
        config = self.config()
        config['refresh'] = True
        self.store.set_setting('chart_schedule:2:ruonia', config)
        chart_schedules.enqueue_due(self.store, NOW)
        job = self.store.one('SELECT * FROM jobs')
        with patch('thesaurus.collectors.Collector.collect') as collect:
            run_chart_job(self.store, self.settings, job, json.loads(job['payload']), dry_run=True)
        collect.assert_not_called()
        self.assertEqual(self.store.query('SELECT * FROM deliveries'), [])


if __name__ == '__main__':
    unittest.main()
