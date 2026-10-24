"""Backfill saved wines without repeating stock, expenses or valuation events."""
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

import test_telegram_wines as fixtures

from thesaurus import wine_cellar as cellar
from thesaurus import wine_details as details
from thesaurus import wine_jobs, wine_retail
from thesaurus.config import Settings
from thesaurus.store import Store


class WineDetailsTests(unittest.TestCase):
    setUp = fixtures.WineControlTests.setUp
    lot = fixtures.WineControlTests.lot
    view = fixtures.WineControlTests.view
    click = fixtures.WineControlTests.click
    input = fixtures.WineControlTests.input

    def test_manual_backfill_updates_statistics_only_after_review_and_keeps_financial_history(self):
        lot = self.lot(country='',region='',grapes=[])
        cellar.consume_bottles(self.store,2,lot['id'],1,price=1400,operation_id='drink-details')
        before = {table:self.store.query('SELECT * FROM '+table) for table in
                  ('wine_events','wine_prices','wine_snapshots','wine_consumptions','life_ledger_entries')}
        self.click('wc:stats:2:country')
        self.assertIn('wc:incomplete:0',self.view()[1])
        self.click('wc:incomplete:0')
        self.assertIn('Страна',self.view()[0])
        self.click(f"wc:details:{lot['id']}")
        current = details.draft(self.store,2,lot['id'])
        self.click(f"wc:df:{lot['id']}:c:{current['token']}")
        self.assertIn('Сейчас: не указано',self.view()[0])
        self.input('Chile')
        self.assertEqual(cellar.get_lot(self.store,2,lot['id'])['country'],'')
        self.assertIn('Страна: не указано → Чили',self.view()[0])
        self.click(f"wc:du:{lot['id']}:{current['token']}")
        saved = cellar.get_lot(self.store,2,lot['id'])
        self.assertEqual((saved['country'],saved['quantity'],saved['purchase_price']),('Чили',1,1000))
        self.assertEqual(cellar.attribute_statistics(self.store,2)['groups']['country']['Чили']['purchased'],2)
        for table,rows in before.items():
            self.assertEqual(rows,self.store.query('SELECT * FROM '+table),table)
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM monitor_events WHERE action='details_updated'")['n'],1)

    def test_cancel_other_owner_and_optimistic_conflict_never_overwrite_saved_wine(self):
        lot = self.lot()
        current = details.draft(self.store,2,lot['id'],create=True)
        details.set_field(self.store,2,lot['id'],current['token'],'country','Spain')
        self.click(f"wc:dd:{lot['id']}:{current['token']}")
        self.assertEqual(cellar.get_lot(self.store,2,lot['id'])['country'],'')
        with self.assertRaises(ValueError):
            details.draft(self.store,3,lot['id'],create=True)
        fields = {field:lot[field] for field in cellar.DETAIL_FIELDS}
        cellar.update_details(self.store,2,lot['id'],{**fields,'country':'Чили'},cellar.details_signature(lot))
        with self.assertRaises(ValueError):
            cellar.update_details(self.store,2,lot['id'],{**fields,'country':'Испания'},cellar.details_signature(lot))
        self.assertEqual(cellar.get_lot(self.store,2,lot['id'])['country'],'Чили')

    def test_lookup_job_and_field_choices_preserve_known_values_and_require_save(self):
        wine_retail.configure_shop(self.store,1,'decanter',True)
        lot = self.lot(vintage=None)
        current = details.draft(self.store,2,lot['id'],create=True)
        payload = {'actor_id':2,'report_id':uuid4().hex,'lot_id':lot['id'],'details_token':current['token']}
        candidate = {'id':'test','name':'Merlot','winery':'Another winery','published_vintage':2021,
                     'country':'Chile','source_url':'https://decanter.ru/product/test','fetched_at':'2026-10-17T00:00:00+00:00'}
        with patch('thesaurus.wine_enrichment.lookup',return_value={'candidates':[candidate],'warnings':[]}):
            wine_jobs.run_job(self.store,Settings(root=self.root),{'kind':'wine_details_lookup','id':123},payload)
        self.assertEqual(cellar.get_lot(self.store,2,lot['id'])['vintage'],'')
        current = details.draft(self.store,2,lot['id'])
        choice = details.choices(self.store,2,current,'vintage')[0]
        self.click(f"wc:df:{lot['id']}:y:{current['token']}")
        self.assertIn('2021',self.view()[1].values())
        self.click(f"wc:dp:{lot['id']}:y:{current['token']}:{choice['fingerprint']}")
        self.assertEqual(details.draft(self.store,2,lot['id'])['fields']['winery'],'Example')
        self.assertEqual(cellar.get_lot(self.store,2,lot['id'])['vintage'],'')
        self.click(f"wc:du:{lot['id']}:{current['token']}")
        self.assertEqual(cellar.get_lot(self.store,2,lot['id'])['vintage'],'2021')
        report = self.store.one('SELECT quality FROM reports WHERE id=?',(payload['report_id'],))
        self.assertEqual(json.loads(report['quality'])['feature'],'wine_cellar')

    def test_lookup_does_not_overwrite_edit_made_during_search(self):
        lot = self.lot()
        current = details.draft(self.store,2,lot['id'],create=True)
        def change(*args,**kwargs):
            details.set_field(self.store,2,lot['id'],current['token'],'country','Chile')
            return {'candidates':[],'warnings':[]}
        with patch('thesaurus.wine_enrichment.lookup',side_effect=change),self.assertRaises(ValueError):
            details.lookup(self.store,2,lot['id'],current['token'])
        self.assertEqual(details.draft(self.store,2,lot['id'])['fields']['country'],'Чили')
        self.assertNotIn('lookup',details.draft(self.store,2,lot['id']))


class WineDetailsBotTests(unittest.TestCase):
    setUp = fixtures.WineBotIntegrationTests.setUp
    message = fixtures.WineBotIntegrationTests.message
    click = fixtures.WineBotIntegrationTests.click
    view = fixtures.WineBotIntegrationTests.view

    def test_completion_from_statistics_survives_restart_and_updates_existing_lot(self):
        lot = cellar.add_bottle(self.store,2,{'name':'Bariloche Malbec','quantity':2,'purchase_price':817.51})
        self.message('/cellar')
        self.click('wc:stats:2:country')
        self.click('wc:incomplete:0')
        self.click(f"wc:details:{lot['id']}")
        current = details.draft(self.store,2,lot['id'])
        self.click(f"wc:df:{lot['id']}:c:{current['token']}")
        self.message('Argentina')
        self.assertIn('Страна: не указано → Аргентина',self.view()[0])
        self.store = Store(self.root/'state.sqlite3')
        from thesaurus.telegram_bot import BotControls
        self.bot = BotControls(self.store)
        self.message('/cellar')
        self.click('wc:list:2:0')
        self.click(f"wc:lot:2:{lot['id']}")
        self.click(f"wc:details:{lot['id']}")
        self.click(f"wc:du:{lot['id']}:{current['token']}")
        self.assertIn('Уточнения сохранены',self.view()[0])
        self.assertEqual(len(cellar.lots(self.store,2)),1)
        self.assertEqual(cellar.get_lot(self.store,2,lot['id'])['quantity'],2)
        self.message('/cellar')
        self.click('wc:stats:2:country')
        self.assertIn('Аргентина: 2 / 2 / 0',self.view()[0])
