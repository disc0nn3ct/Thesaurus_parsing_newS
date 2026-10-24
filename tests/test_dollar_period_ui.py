"""Offline personal USD period controls, navigation and authorization."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import test_dollar as fixtures

from thesaurus.telegram_dollar import form_parent, years_before

NOW = '2026-10-11T09:00:00+00:00'


class DollarPeriodUITests(unittest.TestCase):
    def setUp(self):
        fixtures.DollarBotTests.setUp(self)
        self.enterContext(patch('thesaurus.telegram_dollar.utcnow', return_value=NOW))
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No network')))
        self.client.edit_message_text.return_value = SimpleNamespace(id=42)
        self.controls = self.bot.dollar_controls

    def view(self):
        call = next(call for call in reversed(self.client.mock_calls)
                    if call[0] in {'send_message', 'edit_message_text'})
        text = call.args[2] if call[0] == 'edit_message_text' else call.args[1]
        markup = call.kwargs.get('reply_markup')
        buttons = {button.callback_data: button.text for row in markup.inline_keyboard for button in row} if markup else {}
        return text, buttons

    def click(self, route, *, actor=2, kind='private'):
        query = SimpleNamespace(data=route, id=uuid4().hex, from_user=SimpleNamespace(id=actor),
                                message=SimpleNamespace(id=900, chat=SimpleNamespace(id=actor, type=kind)), answer=Mock())
        self.bot.handle_callback(self.client, query)
        return query

    def test_main_menu_separates_input_history_forecast_and_model(self):
        self.controls.menu(self.client, 2)
        _, buttons = self.view()
        self.assertTrue(buttons['usd:period'].startswith('История для расчёта:'))
        self.assertIn('начала истории', buttons['usd:period'])
        self.assertIn('сегодня', buttons['usd:period'])
        self.assertEqual(buttons['usd:horizon'], 'Срок прогноза: 7 дн.')
        self.assertIn('окно обучения', buttons['usd:model'])
        self.assertNotIn('usd:update', buttons)

    def test_period_presets_checked_personal_and_today_stays_dynamic(self):
        self.controls.callback(self.client, 2, 'usd:period:years:3')
        cfg = self.controls.config(2)
        self.assertEqual(cfg['history_start'], '2023-10-11')
        self.assertIsNone(cfg['history_end'])
        self.assertTrue(self.view()[1]['usd:period:years:3'].startswith('✓'))
        self.assertTrue(self.view()[1]['usd:period:today'].startswith('✓'))
        self.assertIsNone(self.controls.config(3)['history_start'])
        with patch('thesaurus.telegram_dollar.utcnow', return_value='2026-10-12T09:00:00+00:00'):
            self.controls.callback(self.client, 2, 'usd:period')
            self.assertIn('сегодня · обновляется', self.view()[1]['usd:set:history_end'])
            self.assertEqual(self.controls.config(2)['history_start'], '2023-10-11')
        self.assertIsNone(self.controls.config(2)['history_end'])

    def test_invalid_dates_keep_pending_and_settings_then_return_to_period(self):
        self.controls.callback(self.client, 2, 'usd:set:history_start')
        before = self.controls.config(2)
        for text in ('31.02.2020', '2030-01-01'):
            with self.assertRaises(ValueError):
                self.controls.input(self.client, 2, text)
            self.assertEqual(self.controls.config(2), before)
            self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'], 'dollar:history_start')
        self.assertTrue(self.controls.input(self.client, 2, '01.01.2020'))
        self.assertEqual(self.controls.config(2)['history_start'], '2020-01-01')
        self.assertIn('История для расчёта USD/RUB', self.view()[0])
        self.controls.callback(self.client, 2, 'usd:set:history_end')
        with self.assertRaises(ValueError):
            self.controls.input(self.client, 2, '2019-12-31')
        self.assertIsNone(self.controls.config(2)['history_end'])
        self.controls.input(self.client, 2, '01.10.2026')
        self.assertEqual(self.controls.config(2)['history_end'], '2026-10-01')
        self.assertIn('совет о покупке сегодня отключён', self.view()[0])
        self.controls.callback(self.client, 2, 'usd:period:reset')
        self.assertIsNone(self.controls.config(2)['history_start'])
        self.assertIsNone(self.controls.config(2)['history_end'])

    def test_days_months_custom_horizon_and_legacy_reset_are_exclusive(self):
        self.controls.callback(self.client, 2, 'usd:horizon:12')
        self.assertTrue(self.view()[1]['usd:horizon:12'].startswith('✓'))
        self.controls.callback(self.client, 2, 'usd:days:30')
        self.assertEqual(self.controls.config(2)['analysis_months'], 0)
        self.assertEqual(self.controls.config(2)['forecast_days'], 30)
        self.assertTrue(self.view()[1]['usd:days:30'].startswith('✓'))
        self.assertFalse(self.view()[1]['usd:horizon:12'].startswith('✓'))
        self.controls.callback(self.client, 2, 'usd:horizon:24')
        self.controls.callback(self.client, 2, 'usd:set:forecast_days')
        for text in ('0', '91', '12.5'):
            with self.assertRaises(ValueError):
                self.controls.input(self.client, 2, text)
        self.controls.input(self.client, 2, '45')
        self.assertEqual(self.controls.config(2)['analysis_months'], 0)
        self.assertEqual(self.controls.config(2)['forecast_days'], 45)
        self.assertIn('45 дн.', self.view()[0])
        self.controls.callback(self.client, 2, 'usd:horizon:0')
        self.assertEqual(self.controls.config(2)['forecast_days'], 7)
        self.assertEqual(self.controls.config(3)['forecast_days'], 7)
        for route in ('usd:days:91', 'usd:days:15', 'usd:period:years:100', 'usd:period:unknown'):
            with self.subTest(route=route), self.assertRaises(ValueError):
                self.controls.callback(self.client, 2, route)

    def test_boundaries_override_model_window_until_reset_without_losing_preference(self):
        self.controls.callback(self.client, 2, 'usd:model:history:recent')
        self.controls.callback(self.client, 2, 'usd:period:years:5')
        self.controls.callback(self.client, 2, 'usd:model')
        text, buttons = self.view()
        self.assertIn('весь выбранный интервал', text)
        self.assertFalse(any(route.startswith('usd:model:history:') for route in buttons))
        self.assertTrue(any(route.startswith('usd:model:engine:') for route in buttons))
        self.assertEqual(self.controls.config(2)['history_mode'], 'recent')
        self.controls.callback(self.client, 2, 'usd:period:reset')
        self.controls.callback(self.client, 2, 'usd:model')
        self.assertTrue(self.view()[1]['usd:model:history:recent'].startswith('✓'))

    def test_historical_run_bypasses_current_quote_inputs_and_freezes_requested_bounds(self):
        self.controls.callback(self.client, 2, 'usd:mode:purchase')
        self.controls.callback(self.client, 2, 'usd:set:history_end')
        self.controls.input(self.client, 2, '2025-12-31')
        self.controls.callback(self.client, 2, 'usd:period:years:3')
        self.controls.callback(self.client, 2, 'usd:days:60')
        self.controls.callback(self.client, 2, 'usd:run', request_id='historical')
        payload = json.loads(self.store.one('SELECT payload FROM jobs')['payload'])
        self.assertEqual(payload['config']['history_start'], '2022-12-31')
        self.assertEqual(payload['config']['history_end'], '2025-12-31')
        self.assertEqual(payload['config']['forecast_days'], 60)
        self.assertIsNone(payload['config']['trip'])
        self.assertIsNone(payload['config']['quote'])
        self.assertIn('Совет о покупке сегодня отключён', self.view()[0])
        self.controls.callback(self.client, 2, 'usd:period:reset')
        self.assertEqual(json.loads(self.store.one('SELECT payload FROM jobs')['payload']), payload)

    def test_private_chat_permissions_and_cancel_return_to_correct_parent(self):
        self.click('usd:period', kind='group').answer.assert_called_once_with('Нет доступа', show_alert=True)
        self.assertIsNone(self.store.get_setting('dollar:2'))
        self.click('usd:period')
        self.assertEqual(self.store.get_setting('ui_screen:2')['back'], 'usd:menu')
        self.click('usd:set:history_start')
        self.assertEqual(self.store.get_setting('ui_screen:2')['back'], 'usd:period')
        event = SimpleNamespace(text='/cancel', id=12, from_user=SimpleNamespace(id=2), chat=SimpleNamespace(id=2, type='private'))
        self.bot.handle_message(self.client, event)
        self.assertIn('История для расчёта USD/RUB', self.view()[0])
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=2')
        self.click('usd:days:30').answer.assert_called_once_with('Нет доступа', show_alert=True)
        with self.assertRaises(PermissionError):
            self.controls.callback(self.client, 2, 'usd:period')
        self.store.add_user(4, subscriptions=['vivino'])
        self.click('usd:period', actor=4)
        with self.assertRaises(PermissionError):
            self.controls.callback(self.client, 4, 'usd:set:history_start')
        self.assertIsNone(self.store.get_setting('dollar:4'))

    def test_leap_year_preset_and_form_parents(self):
        from datetime import date
        self.assertEqual(years_before(date(2024, 2, 29), 1), date(2023, 2, 28))
        self.assertEqual(form_parent('history_start'), 'usd:period')
        self.assertEqual(form_parent('history_end'), 'usd:period')
        self.assertEqual(form_parent('forecast_days'), 'usd:horizon')
        self.assertEqual(form_parent('analysis_months'), 'usd:horizon')


if __name__ == '__main__':
    unittest.main()
