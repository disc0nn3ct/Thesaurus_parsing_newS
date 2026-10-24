"""Real bot journeys and dated portfolio math, with mocked Telegram and no network."""
import unittest
from unittest.mock import patch

import test_telegram_wines as fixtures

from thesaurus import wine_cellar as cellar
from thesaurus.store import Store


class CellarUXTests(unittest.TestCase):
    setUp = fixtures.WineBotIntegrationTests.setUp
    message = fixtures.WineBotIntegrationTests.message
    click = fixtures.WineBotIntegrationTests.click
    view = fixtures.WineBotIntegrationTests.view

    def test_overview_hides_bottles_storage_archive_and_repeat_confirmation_are_safe(self):
        lot = cellar.add_bottle(self.store,2,{'name':'Merlot','winery':'Example','vintage':2021,
            'country':'Чили','region':'Центральная Долина','grapes':['Мерло'],
            'quantity':3,'purchase_price':1000,'purchase_date':'2024-01-01'})
        cellar.record_price(self.store,2,lot['id'],1200)
        self.message('/cellar')
        text,buttons = self.view()
        self.assertIn('Открытый результат: +600.00 ₽',text)
        self.assertFalse(any(route.startswith('wc:lot:') for route in buttons))
        self.assertIn('wc:incomplete:0',buttons)
        self.click('wc:list:2:0')
        self.assertIn('Винный погреб',self.view()[0])
        self.assertIn('wc:consumed:2:0',self.view()[1])
        self.click(f"wc:lot:2:{lot['id']}")
        self.assertEqual(self.view()[1]['wc:list:2:0'],'← Назад')
        self.click(f"wc:stock:{lot['id']}:drink")
        token = self.store.get_setting('wine_draft:2')['token']
        self.assertEqual(cellar.get_lot(self.store,2,lot['id'])['quantity'],3)
        self.click('wc:apply:'+token)
        self.click('wc:apply:'+token)
        self.assertEqual(cellar.get_lot(self.store,2,lot['id'])['quantity'],2)
        self.assertEqual(len(cellar.consumptions(self.store,2)),1)
        self.message('/cellar')
        self.click('wc:list:2:0')
        self.click('wc:consumed:2:0')
        self.assertIn('Архив',self.view()[0])
        self.assertIn('+200.00 ₽',self.view()[0])
        cellar.record_price(self.store,2,lot['id'],900)
        self.assertEqual(cellar.consumptions(self.store,2)[0]['result'],200)
        self.message('/cellar')
        self.assertIn('Открытый результат: -200.00 ₽',self.view()[0])
        self.assertIn('Зафиксировано по выпитым: +200.00 ₽',self.view()[0])
        cellar.consume_bottles(self.store,2,lot['id'],2,operation_id='final-drink')
        self.message('/cellar')
        self.click('wc:list:2:0')
        self.assertNotIn(f"wc:lot:2:{lot['id']}",self.view()[1])
        self.click('wc:consumed:2:0')
        self.assertIn('-200.00 ₽',self.view()[0])
        self.assertEqual(cellar.attribute_statistics(self.store,2)['groups']['country']['Чили'],
                         {'purchased':3,'remaining':0,'consumed':3})
        self.assertFalse(self.store.query('SELECT * FROM life_ledger_entries'))

    def test_dated_prices_daily_result_history_and_restart_preserve_realized_results(self):
        with patch('thesaurus.wine_cellar.utcnow',return_value='2024-01-01T12:00:00+00:00'):
            lot = cellar.add_bottle(self.store,2,{'name':'Test','quantity':3,'purchase_price':1000,'purchase_date':'2024-01-01'})
        with patch('thesaurus.wine_cellar.utcnow',return_value='2024-01-02T12:00:00+00:00'):
            cellar.record_price(self.store,2,lot['id'],1200,price_date='2024-01-02')
        with patch('thesaurus.wine_cellar.utcnow',return_value='2024-01-03T12:00:00+00:00'):
            cellar.consume_bottles(self.store,2,lot['id'],1,consumed_on='2024-01-03',operation_id='first')
        with patch('thesaurus.wine_cellar.utcnow',return_value='2024-01-04T12:00:00+00:00'):
            cellar.record_price(self.store,2,lot['id'],900,price_date='2024-01-04')
        with patch('thesaurus.wine_cellar.utcnow',return_value='2024-01-05T12:00:00+00:00'):
            cellar.consume_bottles(self.store,2,lot['id'],2,consumed_on='2024-01-05',operation_id='last')
        with patch('thesaurus.wine_cellar.utcnow',return_value='2024-01-06T12:00:00+00:00'):
            cellar.record_price(self.store,2,lot['id'],3000,price_date='2024-01-06')
        history = cellar.daily_result_history(self.store,2)
        self.assertEqual([(row['open_result'],row['closed_result'],row['combined_result']) for row in history],
                         [(None,0,None),(600,0,600),(400,200,600),(-200,200,0),(0,0,0),(0,0,0)])
        self.store = Store(self.root/'state.sqlite3')
        self.assertEqual(cellar.daily_result_history(self.store,2),history)
        self.assertEqual([item['result'] for item in cellar.consumptions(self.store,2)],[-200,200])
        self.assertEqual(cellar.portfolio(self.store,2)['combined_result'],0)
        self.assertEqual(len(self.store.query('SELECT * FROM wine_lots')),1)

    def test_unknown_valuation_stays_unknown_in_result_history(self):
        lot = cellar.add_bottle(self.store,2,{'name':'Unknown','quantity':1,'purchase_price':1000})
        self.assertIsNone(cellar.daily_result_history(self.store,2)[0]['open_result'])
        cellar.consume_bottles(self.store,2,lot['id'],unpriced=True,operation_id='unpriced')
        self.assertIsNone(cellar.daily_result_history(self.store,2)[-1]['closed_result'])
        self.assertIsNone(cellar.daily_result_history(self.store,2)[-1]['combined_result'])
