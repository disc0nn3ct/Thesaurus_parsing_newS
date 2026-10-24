"""Cross-screen navigation, saved choices and task feedback through real handlers."""
import unittest
from unittest.mock import patch

import test_telegram_ux as fixtures

from thesaurus import market_store


class NavigationJourneyTests(unittest.TestCase):
    message = fixtures.TelegramUXTests.message
    click = fixtures.TelegramUXTests.click
    view = fixtures.TelegramUXTests.view

    def setUp(self):
        fixtures.TelegramUXTests.setUp(self)
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('Offline UX acceptance')))

    def test_cancel_cbr_search_returns_to_catalog_without_resetting_project(self):
        self.message('/start')
        self.click('pj:market')
        self.click('menu:data')
        self.click('cb:menu')
        self.click('cb:search')
        self.click('ui:cancel')
        self.assertIn('Базы Банка России', self.view()[0])
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.assertEqual(self.client.send_message.call_count, 1)

    def test_cbr_card_back_preserves_search_and_page(self):
        self.message('/start')
        self.click('pj:market')
        self.click('menu:data')
        self.click('cb:menu')
        self.click('cb:search')
        self.message('Показатель')
        self.click('cb:results:1')
        card = next(k for k in self.view()[1] if k.startswith('cb:card:'))
        self.click(card)
        self.assertEqual(self.view()[1]['cb:results:1'], '← Назад')
        self.click('cb:results:1')
        self.assertIn('Поиск: Показатель', self.view()[0])
        self.assertIn('Страница 2 из 2', self.view()[0])
        self.assertIn(card, self.view()[1])

    def purchase(self):
        self.message('/dollar')
        self.click('usd:mode:purchase')

    def test_advanced_dollar_is_compact_and_keeps_form_after_invalid_input(self):
        self.purchase()
        self.click('usd:advanced')
        self.assertLessEqual(len(self.view()[1]), 6)
        self.assertNotIn('usd:run', self.view()[1])
        self.click('usd:set:fee_pct')
        self.message('nan')
        self.assertIn('ui:cancel', self.view()[1])
        self.assertEqual(self.view()[1]['usd:advanced'], '← Назад')
        self.message('1,5')
        self.assertIn('комиссии и риск', self.view()[0])
        self.assertIn('1.5', self.view()[1]['usd:set:fee_pct'])
        self.assertEqual(self.bot.dollar_controls.config(2)['fee_pct'], 1.5)
        self.assertEqual(self.bot.dollar_controls.config(3)['fee_pct'], 0)
        self.click('usd:menu')
        self.assertTrue(self.view()[1]['usd:mode:purchase'].startswith('✓'))

    def test_cancel_commission_keeps_advanced_menu_and_saved_value(self):
        self.purchase()
        self.click('usd:advanced')
        self.click('usd:set:risk')
        before = self.bot.dollar_controls.config(2)['risk']
        self.message('/cancel')
        self.assertIn('комиссии и риск', self.view()[0])
        self.assertIn('usd:set:risk', self.view()[1])
        self.assertEqual(self.bot.dollar_controls.config(2)['risk'], before)

    def test_custom_trip_cancel_returns_to_date_picker(self):
        self.purchase()
        self.click('usd:dates')
        self.click('usd:set:trip')
        self.click('ui:cancel')
        self.assertIn('Когда валюта должна быть куплена', self.view()[0])
        self.assertIn('usd:date:7', self.view()[1])

    def test_double_dollar_request_reuses_active_job_and_can_recalculate_after_done(self):
        self.message('/dollar')
        self.click('usd:run')
        self.click('usd:run')
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)
        self.assertIn('повторный запрос не создан', self.view()[0])
        self.assertIn('usd:status', self.view()[1])
        self.store.execute("UPDATE jobs SET state='done'")
        self.click('usd:run')
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 2)

    def test_task_status_is_private_refreshable_and_returns_to_dollar(self):
        self.message('/dollar')
        self.click('usd:run')
        self.store.enqueue('dollar_analysis', {'actor_id': 3})
        self.store.execute("UPDATE jobs SET state='failed' WHERE json_extract(payload,'$.actor_id')=3")
        self.store.enqueue('report', {'recipient': '3'})
        self.click('usd:status')
        text, buttons = self.view()
        self.assertIn('Ваши задания: в очереди: 1', text)
        self.assertNotIn('ошибки:', text)
        self.assertEqual(buttons['usd:status'], 'Обновить состояние')
        self.assertEqual(buttons['usd:menu'], '← Назад')
        self.assertNotIn('menu:jobs', buttons)
        self.click('usd:status')
        self.click('usd:menu')
        self.assertIn('Доллар для поездки', self.view()[0])

    def test_schedule_warns_before_applying_chart_changes_without_enabling(self):
        self.message('/chart ruonia')
        self.click('cs:ruonia:menu')
        self.assertTrue(self.view()[1]['cs:ruonia:apply'].startswith('✓'))
        self.click('ch:menu:ruonia')
        self.click('ch:view:ruonia:period')
        self.click('ch:period:ruonia:month')
        self.click('cs:ruonia:menu')
        self.assertIn('ещё не сохранённые в расписание', self.view()[0])
        self.assertFalse(self.view()[1]['cs:ruonia:apply'].startswith('✓'))
        self.click('cs:ruonia:apply')
        self.assertNotIn('ещё не сохранённые в расписание', self.view()[0])
        self.assertTrue(self.view()[1]['cs:ruonia:apply'].startswith('✓'))
        self.assertFalse(self.store.get_setting('chart_schedule:2:ruonia')['enabled'])

    def test_custom_dates_are_marked_fixed_and_schedule_not_enabled(self):
        self.store.set_setting('chart:2:ruonia', {'period': 'custom', 'date_from': '2025-01-01', 'date_to': '2025-03-01', 'horizon': 1})
        self.message('/graphs')
        self.click('cs:ruonia:menu')
        self.assertIn('2025-01-01 — 2025-03-01 (фиксированные даты)', self.view()[0])
        self.assertIn('Для включения расписания нужен скользящий период', self.view()[0])
        event = self.click('cs:ruonia:on')
        self.assertTrue(event.answer.call_args.kwargs['show_alert'])
        self.assertFalse(self.store.get_setting('chart_schedule:2:ruonia')['enabled'])

    def test_currency_parameter_shows_selected_value_and_can_be_changed(self):
        market_store.register(self.store, [{'id': 'cbr.hd_base.seldomc', 'source': 'cbr', 'dataset': 'cbr.hd_base.seldomc',
            'title': 'Редкие валюты', 'kind': 'dataset', 'frequency': 'irregular', 'unit': 'руб',
            'url': 'https://cbr.ru/hd_base/seldomc/', 'default_params': {'UniDbQuery.ValID': '1'},
            'parameter_choices': {'UniDbQuery.ValID': [{'value': '1', 'label': 'Валюта A'}, {'value': '2', 'label': 'Валюта B'}]}}])
        token = self.store.one("SELECT rowid token FROM market_series WHERE id='cbr.hd_base.seldomc'")['token']
        self.message('/start', 1)
        self.click(f'cb:card:{token}', 1, message_id=900)
        self.click(f'cb:params:{token}:0', 1)
        self.assertEqual(self.view()[1][f'cb:param:{token}:1'], '✓ Валюта A')
        self.click(f'cb:param:{token}:2', 1)
        self.click(f'cb:params:{token}:0', 1)
        self.assertEqual(self.view()[1][f'cb:param:{token}:2'], '✓ Валюта B')
        self.assertFalse(self.view()[1][f'cb:param:{token}:1'].startswith('✓'))


if __name__ == '__main__':
    unittest.main()
