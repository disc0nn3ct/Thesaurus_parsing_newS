"""Whole-screen bot flows using an offline Telegram client."""
import json
import shutil
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from pyrogram.errors import MessageIdInvalid, MessageNotModified

from thesaurus import autocollect, market, market_store, trends_api
from thesaurus.config import Settings
from thesaurus.service import Worker
from thesaurus.store import Store, dumps
from thesaurus.telegram_bot import BotControls
from thesaurus.telegram_navigation import CURRENT


class TelegramUXTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('ux-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.store.add_user(3, subscriptions='all')
        self.store.set_regions([213, 2])
        self.store.set_keywords([{'phrase': 'рост цен'}, {'phrase': 'инфляция'}])
        self.store.set_setting('region_names', {'213': 'Москва', '2': 'Санкт-Петербург'})
        self.client = Mock()
        sequence = count(100)
        self.client.send_message.side_effect = lambda *args, **kwargs: SimpleNamespace(id=next(sequence))
        self.bot = BotControls(self.store)
        market_store.register(self.store, [{'id': f'ux.{i}', 'source': 'cbr', 'dataset': 'ux', 'title': f'Показатель {i}',
            'kind': 'series', 'frequency': 'daily', 'calendar': 'calendar', 'unit': '%', 'transform': 'difference',
            'aggregation': 'last', 'url': 'https://cbr.ru/hd_base/keyrate/'} for i in range(10)])
        for i in range(10):
            market.enable(self.store, f'ux.{i}', True)

    def message(self, text, actor=2):
        event = SimpleNamespace(text=text, id=1, from_user=SimpleNamespace(id=actor), chat=SimpleNamespace(id=actor, type='private'))
        self.bot.handle_message(self.client, event)

    def click(self, data, actor=2, message_id=None):
        record = self.store.get_setting(f'ui_screen:{actor}', {})
        event = SimpleNamespace(data=data, id=uuid4().hex, from_user=SimpleNamespace(id=actor),
            message=SimpleNamespace(id=message_id or record.get('message_id', 100), text='экран', chat=SimpleNamespace(id=actor, type='private')), answer=Mock())
        self.bot.handle_callback(self.client, event)
        return event

    def view(self):
        # Compare all RPCs in their actual order; edits are the visible screen.
        rpc = next(call for call in reversed(self.client.mock_calls) if call[0] in {'send_message', 'edit_message_text'})
        text = rpc.args[2] if rpc[0] == 'edit_message_text' else rpc.args[1]
        markup = rpc.kwargs.get('reply_markup')
        buttons = {b.callback_data: b.text for row in markup.inline_keyboard for b in row} if markup else {}
        return text, buttons

    def charts(self):
        self.message('/chart market')

    def test_empty_wordstat_offers_explicit_loading_and_return_to_graph(self):
        self.message('/chart wordstat')
        text, buttons = self.view()
        self.assertIn('история ещё не загружена', text)
        self.assertIn('request:wordstat', buttons)
        self.assertNotIn('ch:build:wordstat', buttons)
        self.assertNotIn('ch:view:wordstat:export', buttons)
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.assertEqual(self.store.query('SELECT * FROM usage'), [])
        self.click('request:wordstat')
        text, buttons = self.view()
        self.assertIn('Wordstat API', text)
        self.assertIn('ch:menu:wordstat', buttons)
        jobs = self.store.query('SELECT * FROM jobs')
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['kind'], 'report')
        self.assertEqual(json.loads(jobs[0]['payload'])['section'], 'wordstat')
        self.click('ch:menu:wordstat')
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)

    def test_wordstat_availability_uses_only_own_nonmissing_weekly_observations(self):
        rows = [
            {'metric': 'рост цен', 'frequency': 'weekly', 'date': '2026-09-20', 'value': None},
            {'metric': 'посторонняя фраза', 'frequency': 'weekly', 'date': '2026-09-27', 'value': 100},
            {'metric': 'рост цен', 'frequency': 'daily', 'date': '2026-09-28', 'value': 100},
        ]
        self.store.save_observations([{'source': 'wordstat', 'region_id': 225, 'unit': 'запросов', **row} for row in rows])
        self.message('/chart wordstat')
        self.assertNotIn('ch:build:wordstat', self.view()[1])
        self.store.save_observations([{'source': 'wordstat', 'metric': 'рост цен', 'region_id': 225,
            'frequency': 'weekly', 'date': '2026-09-27', 'value': 100, 'unit': 'запросов'}])
        self.message('/chart wordstat')
        text, buttons = self.view()
        self.assertIn('2026-09-27', text)
        self.assertIn('ch:build:wordstat', buttons)
        self.assertIn('ch:view:wordstat:export', buttons)
        self.assertEqual(buttons['request:wordstat'], 'Обновить данные из Яндекса')
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_home_groups_tasks_and_does_not_repeat_navigation(self):
        self.message('/start')
        text, buttons = self.view()
        self.assertIn('Главное меню', text)
        self.assertLessEqual(len(buttons), 7)
        self.assertNotIn('menu:admin', buttons)
        self.click('pj:market')
        self.click('menu:settings')
        _, buttons = self.view()
        self.assertIn('menu:home', buttons)
        self.assertNotIn('menu:report', buttons)
        self.assertNotIn('menu:help', buttons)
        self.assertEqual(self.client.send_message.call_count, 1)
        self.assertEqual(self.client.edit_message_text.call_count, 2)

    def test_chart_summary_has_no_large_flat_picker(self):
        self.charts()
        text, buttons = self.view()
        self.assertIn('Период: 3 месяца', text)
        self.assertIn('ch:view:market:period', buttons)
        self.assertNotIn('ch:period:market:month', buttons)
        self.assertNotIn('menu:subscriptions', buttons)
        self.assertLessEqual(len(buttons), 11)

    def test_period_selection_edits_and_marks_current_choice(self):
        self.charts()
        self.click('ch:view:market:period')
        _, buttons = self.view()
        self.assertEqual(buttons['ch:period:market:quarter'], '✓ 3 месяца')
        self.click('ch:period:market:month')
        text, _ = self.view()
        self.assertIn('Период: Месяц', text)
        self.click('ch:view:market:period')
        _, buttons = self.view()
        self.assertEqual(buttons['ch:period:market:month'], '✓ Месяц')
        self.assertEqual(sum(label.startswith('✓') for label in buttons.values()), 1)
        self.assertEqual(self.client.send_message.call_count, 1)

    def test_indicator_pick_and_pagination_are_distinct_from_back(self):
        self.charts()
        self.click('ch:view:market:indicator')
        _, buttons = self.view()
        self.assertIn('ch:page:market:1', buttons)
        self.assertEqual(buttons['ch:menu:market'], '← Назад')
        token = next(k for k in buttons if k.startswith('ch:select:market:') and not k.endswith(':all'))
        self.click(token)
        self.click('ch:view:market:indicator')
        _, buttons = self.view()
        self.assertTrue(buttons[token].startswith('✓'))

    def test_horizon_labels_have_units_and_off_state(self):
        self.charts()
        self.click('ch:view:market:horizon')
        _, buttons = self.view()
        self.assertEqual(buttons['ch:horizon:market:4'], '✓ До 4 нед.')
        self.click('ch:horizon:market:0')
        text, _ = self.view()
        self.assertIn('Прогноз: выключен', text)

    def test_custom_dates_cancel_returns_to_picker_without_changes(self):
        self.charts()
        self.click('ch:view:market:period')
        self.click('ch:custom:market')
        _, buttons = self.view()
        self.assertIn('ui:cancel', buttons)
        self.click('ui:cancel')
        text, _ = self.view()
        self.assertIn('Период графика', text)
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.assertEqual(self.bot.chart_controls.options(2, 'market')['period'], 'quarter')

    def test_invalid_input_keeps_back_and_can_retry_in_same_message(self):
        self.charts()
        self.click('ch:view:market:period')
        self.click('ch:custom:market')
        self.message('2025-02-01 2025-01-01')
        _, buttons = self.view()
        self.assertIn('ch:view:market:period', buttons)
        self.assertIn('ui:cancel', buttons)
        self.message('2025-01-01 2025-02-01')
        text, buttons = self.view()
        self.assertIn('2025-01-01 — 2025-02-01', text)
        self.assertNotIn('ui:cancel', buttons)
        self.assertEqual(self.client.send_message.call_count, 1)

    def test_leaving_input_clears_hidden_pending_state(self):
        self.charts()
        self.click('ch:view:market:period')
        self.click('ch:custom:market')
        self.click('menu:home')
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.message('random text')
        text, _ = self.view()
        self.assertNotIn('Начало периода', text)

    def test_double_build_does_not_create_duplicate_jobs(self):
        self.charts()
        self.click('ch:build:market')
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)
        event = self.click('ch:build:market')
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)
        self.assertTrue(event.answer.call_args.kwargs['show_alert'])

    def test_subscription_card_changes_format_days_and_time_without_enabling(self):
        self.message('/subscriptions')
        self.click('ui:subscription:ruonia')
        self.click('ui:subscription:ruonia:detail:full')
        self.click('ui:subscription:ruonia:days:work')
        self.click('ui:subscription:ruonia:times')
        self.message('08:30,18:30')
        row = self.store.one("SELECT * FROM subscriptions WHERE user_id=2 AND section='ruonia'")
        self.assertEqual(row['detail'], 'full')
        self.assertEqual(json.loads(row['weekdays']), list(range(5)))
        self.assertEqual(json.loads(row['times']), ['08:30', '18:30'])
        self.assertFalse(row['enabled'])
        self.assertIn('ui:subscription:ruonia:enabled:on', self.view()[1])

    def test_regions_and_keywords_toggle_visibly_and_only_own_filters(self):
        self.message('/regions')
        _, buttons = self.view()
        self.assertIn('Москва', buttons['rg:toggle:213'])
        self.click('rg:toggle:213')
        _, buttons = self.view()
        self.assertFalse(buttons['rg:toggle:213'].startswith('✓'))
        self.assertTrue(buttons['rg:toggle:2'].startswith('✓'))
        self.assertEqual(self.store.regions(), [2, 213])
        self.message('/keywords')
        self.click('kw:open:wordstat')
        token = next(k for k in self.view()[1] if k.startswith('kw:toggle:'))
        self.click(token)
        self.assertFalse(self.view()[1][token].startswith('✓'))
        self.assertEqual(len(self.store.keywords()), 2)

    def test_market_selection_keeps_page_and_checkmarks(self):
        self.message('/indicators')
        token = next(k for k in self.view()[1] if k.startswith('mi:toggle:'))
        self.assertTrue(self.view()[1][token].startswith('✓'))
        self.click(token)
        self.assertFalse(self.view()[1][token].startswith('✓'))
        self.click('mi:none:0')
        self.assertIn('Выбрано 0', self.view()[0])
        self.click('mi:all:0')
        self.assertIn('Выбрано 10', self.view()[0])

    def test_pair_wizard_uses_names_and_validates_both_scopes(self):
        self.message('/correlations')
        self.click('ui:pair:left:0')
        first = next(k for k in self.view()[1] if k.startswith('ui:pair:pickleft:'))
        self.click(first)
        second = next(k for k in self.view()[1] if k.startswith('ui:pair:pickright:'))
        self.click(second)
        pairs = self.store.get_setting('market_pairs:2')
        self.assertEqual(len(pairs), 1)
        self.assertNotEqual(pairs[0]['left'], pairs[0]['right'])
        self.assertIn('Показатель', self.view()[0])
        self.assertNotIn('"left"', self.view()[0])

    def test_rule_card_changes_mode_threshold_and_rejects_other_owner(self):
        self.message('/alerts market')
        rule = next(k for k in self.view()[1] if k.startswith('ui:rule:'))
        self.click(rule)
        identifier = rule.split(':')[2]
        self.click(f'ui:rulemode:{identifier}:digest')
        self.assertTrue(self.view()[1][f'ui:rulemode:{identifier}:digest'].startswith('✓'))
        self.click(f'ui:rulethreshold:{identifier}')
        self.message('2,5')
        row = self.store.one('SELECT * FROM market_rules WHERE id=?', (identifier,))
        self.assertEqual(json.loads(row['config'])['threshold'], 2.5)
        # An old or forged keyboard never gives access to another person's rule.
        self.message('/start', actor=3)
        event = self.click(rule, actor=3, message_id=900)
        self.assertTrue(event.answer.call_args.kwargs['show_alert'])

    def test_admin_sources_search_toggle_and_return_to_catalog(self):
        self.message('/sources', 1)
        self.click('ui:sources:search', 1)
        self.message('Показатель 0', 1)
        _, buttons = self.view()
        self.assertIn('ms:reset:new', buttons)
        token = next(k for k in buttons if k.startswith('ms:toggle:'))
        self.click(token, 1)
        self.assertFalse(self.view()[1][token].startswith('✓'))
        self.assertIn('ms:reset:new', self.view()[1])
        self.click('ms:reset:new', 1)
        self.assertNotIn('ms:reset:new', self.view()[1])

    def test_disabling_user_requires_product_confirmation_and_preserves_admins(self):
        self.message('/users', 1)
        self.click('ua:open:2', 1)
        self.click('user:deny:2', 1)
        self.assertTrue(self.store.authorize(2))
        self.click('user:denyconfirmed:2', 1)
        self.assertFalse(self.store.authorize(2))
        event = self.click('user:deny:1', 1, message_id=900)
        self.assertTrue(event.answer.call_args.kwargs['show_alert'])
        self.assertTrue(self.store.authorize(1, admin=True))

    def test_edit_not_modified_and_missing_message_fallback(self):
        self.charts()
        self.client.edit_message_text.side_effect = MessageNotModified()
        self.click('ch:view:market:period')
        self.assertEqual(self.client.send_message.call_count, 1)
        self.client.edit_message_text.side_effect = MessageIdInvalid()
        self.click('ch:period:market:month')
        self.assertEqual(self.client.send_message.call_count, 2)
        self.assertEqual(self.store.get_setting('ui_screen:2')['message_id'], 101)

    def test_forecast_cards_are_paged_and_open_selected_chart(self):
        self.message('/menu')
        self.click('pj:market')
        self.click('menu:data')
        self.click('ui:forecasts:market')
        self.assertIn('ui:forecastpage:1', self.view()[1])
        token = next(k for k in self.view()[1] if k.startswith('ui:forecast:'))
        self.click(token)
        text, buttons = self.view()
        self.assertIn('Нет сохранённого прогноза', text)
        graph = next(k for k in buttons if k.startswith('ui:chartfor:'))
        self.click(graph)
        self.assertIsNotNone(self.bot.chart_controls.options(2, 'market')['identity'])
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_collection_schedule_and_history_settings_are_guided_and_off_by_default(self):
        self.message('/sources', 1)
        self.click('ui:schedule:overview', 1)
        self.assertIn('выключено', self.view()[0])
        self.click('ui:schedule:times', 1)
        self.message('10:00,18:30', 1)
        self.assertFalse(self.store.get_setting('market_schedule')['enabled'])
        self.click('ui:schedule:on', 1)
        self.assertTrue(self.store.get_setting('market_schedule')['enabled'])
        self.click('menu:sources', 1)
        self.click('ui:history:menu', 1)
        self.click('ui:history:daily', 1)
        self.click('ui:history:daily:10', 1)
        self.assertEqual(self.store.get_setting('market_history_years')['daily'], 10)
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_autocollect_admin_menu_updates_visible_state_and_preserves_delivery(self):
        self.store.set_setting('search_provider', 'google_trends')
        self.message('/admin', 1)
        self.click('pj:market:manage', 1)
        self.assertIn('menu:autocollect', self.view()[1])
        self.click('menu:autocollect', 1)
        text, buttons = self.view()
        self.assertIn('Google Trends: выключен', text)
        self.assertIn('ЦБ и Мосбиржа: выключен', text)
        self.assertIn('pj:market:manage', buttons)
        subscriptions = self.store.query('SELECT * FROM subscriptions')
        sends = self.client.send_message.call_count
        self.click('ui:autocollect:market:on', 1)
        self.assertIn('ЦБ и Мосбиржа: включён', self.view()[0])
        self.assertIn('ui:autocollect:market:off', self.view()[1])
        self.assertEqual(self.client.send_message.call_count, sends)
        self.assertEqual(self.store.query('SELECT * FROM subscriptions'), subscriptions)
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.click('ui:autocollect:market:off', 1)
        self.assertFalse(self.store.get_setting('market_schedule')['enabled'])

    def test_autocollect_command_and_direct_google_toggle(self):
        self.store.set_setting('search_provider', 'google_trends')
        self.message('/autocollect', 1)
        self.click('ui:autocollect:trends:on', 1)
        self.assertTrue(Store(self.store.path).get_setting('trends_schedule')['enabled'])
        self.message('/trends', 1)
        self.assertIn('gt:off', self.view()[1])
        self.click('gt:off', 1)
        self.assertIn('Состояние: выключен', self.view()[0])
        self.assertIn('gt:on', self.view()[1])

    def test_autocollect_subscriber_and_group_cannot_change_global_schedule(self):
        self.message('/autocollect', 2)
        self.assertIn('администратору', self.view()[0])
        for callback in ('ui:autocollect:trends:on', 'ui:autocollect:market:on', 'menu:autocollect'):
            event = self.click(callback, 2, message_id=900)
            self.assertTrue(event.answer.call_args.kwargs['show_alert'])
        group = SimpleNamespace(data='ui:autocollect:market:on', id='group-callback', from_user=SimpleNamespace(id=1),
            message=SimpleNamespace(id=900, chat=SimpleNamespace(id=-1, type='group')), answer=Mock())
        self.bot.handle_callback(self.client, group)
        self.assertIsNone(self.store.get_setting('market_schedule'))

    def test_disabling_cancels_only_queued_automatic_jobs(self):
        for source in ('trends', 'market'):
            kind = autocollect.SOURCES[source][1]
            autocollect.update_schedule(self.store, source, 1, enabled=True)
            automatic = self.store.enqueue(kind, {'actor_id': 1, **({'scheduled': True} if source == 'trends' else {'scheduled_for': '2026-10-01T00:00:00+00:00'})})
            manual = self.store.enqueue(kind, {'actor_id': 1})
            running = self.store.enqueue(kind, {'actor_id': 1, 'scheduled': True})
            self.store.execute("UPDATE jobs SET state='running' WHERE id=?", (running,))
            autocollect.update_schedule(self.store, source, 1, enabled=False)
            self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (automatic,))['state'], 'cancelled')
            self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (manual,))['state'], 'pending')
            self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (running,))['state'], 'running')

    def test_toggles_preserve_options_and_repeated_enable_does_not_reset_schedule(self):
        original = {'enabled': False, 'times': ['09:10', '19:30'], 'geo': 'RU-MOW', 'window': 'today 12-m', 'phrases': ['рост цен']}
        self.store.set_setting('trends_schedule', original)
        enabled = autocollect.update_schedule(self.store, 'trends', 1, enabled=True)
        repeated = autocollect.update_schedule(self.store, 'trends', 1, enabled=True)
        self.assertEqual(enabled, repeated)
        disabled = autocollect.update_schedule(self.store, 'trends', 1, enabled=False)
        for key in ('times', 'geo', 'window', 'phrases'):
            self.assertEqual(disabled[key], original[key])

    def test_reenable_skips_missed_slots_and_expired_retry(self):
        self.store.set_setting('search_provider', 'google_trends')
        now = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
        self.store.set_setting('trends_schedule', {'enabled': False, 'times': ['08:30'], 'actor_id': 1,
            'created_at': (now - timedelta(days=3)).isoformat()})
        self.store.set_setting('trends_pause_until', (now - timedelta(hours=1)).isoformat())
        with patch('thesaurus.autocollect.utcnow', return_value=now.isoformat()):
            autocollect.update_schedule(self.store, 'trends', 1, enabled=True)
        self.assertIsNone(trends_api.enqueue_due(self.store, now))
        self.assertIsNotNone(trends_api.enqueue_due(self.store, now + timedelta(days=1)))

    def test_reenable_keeps_active_google_pause(self):
        now = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
        pause = (now + timedelta(hours=1)).isoformat()
        self.store.set_setting('trends_pause_until', pause)
        with patch('thesaurus.autocollect.utcnow', return_value=now.isoformat()):
            autocollect.update_schedule(self.store, 'trends', 1, enabled=True)
        self.assertEqual(self.store.get_setting('trends_pause_until'), pause)

    def test_disabled_schedule_jobs_cannot_collect_even_if_enqueued_after_disable(self):
        for source in ('market', 'trends'):
            autocollect.update_schedule(self.store, source, 1, enabled=False)
            self.store.enqueue(autocollect.SOURCES[source][1], {'actor_id': 1, 'scheduled': True,
                               'scheduled_for': '2026-10-05T05:30:00+00:00'})
        with patch('thesaurus.market.collect') as market_collect, patch('thesaurus.trends_api.collect') as trends_collect:
            Worker(self.store, Settings(root=self.root), charts=False).drain()
        market_collect.assert_not_called()
        trends_collect.assert_not_called()

    def test_legacy_market_schedule_off_preserves_time_and_cancels_queue(self):
        autocollect.update_schedule(self.store, 'market', 1, enabled=True, changes={'times': ['10:15']})
        job = self.store.enqueue('market_collect', {'actor_id': 1, 'scheduled_for': '2026-10-05T07:15:00+00:00'})
        self.message('/sources schedule off', 1)
        self.assertEqual(self.store.get_setting('market_schedule')['times'], ['10:15'])
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (job,))['state'], 'cancelled')

    def test_budget_form_and_forged_admin_buttons_recheck_access(self):
        self.message('/budget', 1)
        self.click('ui:budget:ai', 1)
        self.message('4', 1)
        self.assertEqual(self.store.get_setting('budgets')['ai_calls'], 4)
        self.message('/start', 2)
        for data in ('ui:budget:ai', 'ui:schedule:on', 'ui:history:daily:50', 'ui:jobs:failed'):
            event = self.click(data, 2, message_id=900)
            self.assertTrue(event.answer.call_args.kwargs['show_alert'])
        self.assertEqual(self.store.get_setting('budgets')['ai_calls'], 4)
        self.assertIsNone(self.store.get_setting('market_schedule'))

    def test_monthly_horizon_and_unavailable_selected_series(self):
        row = self.store.one("SELECT * FROM market_series WHERE id='ux.0'")
        meta = json.loads(row['metadata'])
        meta['frequency'] = 'monthly'
        self.store.execute('UPDATE market_series SET metadata=? WHERE id=?', (dumps(meta), 'ux.0'))
        self.store.set_setting('chart:2:market', {'period': 'month', 'horizon': 4, 'identity': 'ux.0'})
        self.charts()
        self.click('ch:view:market:horizon')
        _, buttons = self.view()
        self.assertIn('ch:horizon:market:3', buttons)
        self.assertNotIn('ch:horizon:market:4', buttons)
        market.enable(self.store, 'ux.0', False)
        self.message('/chart market')
        self.assertIn('недоступен', self.view()[0])
        self.assertNotIn('ch:build:market', self.view()[1])

    def test_all_menu_buttons_are_short_and_each_view_has_exit(self):
        for command in ('/start', '/help', '/settings', '/subscriptions', '/regions', '/keywords', '/chart market', '/indicators', '/correlations', '/alerts market'):
            self.message(command)
            text, buttons = self.view()
            self.assertLessEqual(len(text), 4000)
            self.assertTrue(all(len(data.encode('utf-8')) <= 64 for data in buttons))
            if command != '/start':
                self.assertIn('menu:home', buttons)

    def test_uncertain_edit_error_does_not_send_duplicate_message(self):
        self.charts()
        self.client.edit_message_text.side_effect = TimeoutError()
        event = self.click('ch:view:market:period')
        self.assertEqual(self.client.send_message.call_count, 1)
        self.assertTrue(event.answer.call_args.kwargs['show_alert'])

    def test_delivered_report_content_is_not_edited(self):
        self.message('/start')
        identifier = uuid4().hex
        self.store.execute('INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,quality,parts) VALUES(?,?,?,?,?,?,?,?,?)',
            (identifier, 'ruonia', '2025-01-01', 1, 1, 'immutable', 'immutable', '{}', '[]'))
        self.store.execute('INSERT INTO deliveries(report_id,recipient,variant,part_index,payload,state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)',
            (identifier, '2', 'brief:test', 0, dumps({'kind': 'text', 'text': 'immutable'}), 'sent', '2025-01-01', '2025-01-01'))
        self.click('report:' + identifier, message_id=900)
        self.assertTrue(all(call.args[1] != 900 for call in self.client.edit_message_text.call_args_list))
        self.assertEqual(self.store.one('SELECT text FROM reports WHERE id=?', (identifier,))['text'], 'immutable')

    def test_context_resets_and_different_users_do_not_share_targets(self):
        self.message('/start', 2)
        self.message('/start', 3)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda actor: self.click('pj:market', actor), [2, 3]))
            list(pool.map(lambda actor: self.click('menu:settings', actor), [2, 3]))
        edits = {call.args[0]: call.args[1] for call in self.client.edit_message_text.call_args_list}
        self.assertEqual(edits, {2: 100, 3: 101})
        self.assertIsNone(CURRENT.get())


if __name__ == '__main__':
    unittest.main()
