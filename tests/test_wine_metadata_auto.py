"""Source-backed attributes survive receipt confirmation, depletion and restart."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thesaurus import life_ledger as ledger
from thesaurus import receipt_wines
from thesaurus import wine_cellar as cellar
from thesaurus import wine_enrichment as enrich
from thesaurus import wine_retail as retail
from thesaurus.store import Store
from thesaurus.wine_reference_sources import ROUTES, parse_reference

BAHETLE_URL = ROUTES['bariloche malbec'][0][1]
VINUM_URL = ROUTES['bariloche malbec'][1][1]
# Minimal structural fixtures from public pages verified on 12 October 2026.
BAHETLE = '''<article class="product-page__content" itemscope itemtype="https://schema.org/Product">
<h1 class="product-about__title" itemprop="name">Вино Барилоче Мальбек кр п/сух 12-13% ст/б 0,75л</h1>
<div class="product-characteristics-details__tech-parameter">
<div class="product-characteristics-details__tech-parameter-name">Производитель</div>
<div class="product-characteristics-details__tech-parameter-value">ФАМИЛИЯ ШРЕДЕР</div></div>
<div class="product-characteristics-details__tech-parameter">
<div class="product-characteristics-details__tech-parameter-name">Страна</div>
<div class="product-characteristics-details__tech-parameter-value">Аргентина</div></div></article>'''
VINUM = '''<h2 class="ficha-tecnica__title">Schroeder Malbec Bariloche 2024</h2>''' + ''.join(
    '<div class="informacoes-tecnicas__linha"><div class="linha_spec_type">'+key+'</div><p>'+value+'</p></div>'
    for key, value in [('País', 'Argentina'), ('Região', 'Patagônia'), ('Vinícola', 'Familia Schroeder'),
                       ('Castas', 'Malbec'), ('Safra', '2024'), ('Guarda', '2030')])


class WineAutomaticMetadataTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1]/'.test-tmp'
        directory = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name)/'metadata.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, subscriptions=['life', 'vivino'])
        self.store.add_user(3, subscriptions=['life'])
        self.store.set_setting('wine_metadata_auto:2', True)
        for source in ('bahetle_metadata', 'vinumday_metadata'):
            retail.configure_shop(self.store, 1, source, True)
        self.token = 'automatic-metadata'
        self.receipt = {'status': 'candidate', 'fingerprint': 'e'*64, 'date': '2024-01-02', 'total': 817.51,
            'merchant': 'Test', 'items': [{'name': 'Вино Барилоче Мальбек кр п/сух 0,75', 'quantity': 1,
                'amount': 817.51, 'is_wine': True, 'volume_ml': 750}], 'warnings': []}
        ledger.save_receipt_draft(self.store, 2, self.receipt, self.token)

    def download(self, url, **kwargs):
        if url == BAHETLE_URL:
            return BAHETLE, url
        if url == VINUM_URL:
            return VINUM, url
        raise AssertionError('Unexpected external request: '+url)

    def run_lookup(self):
        self.assertEqual(receipt_wines.queue_automatic(self.store, 2, self.token), 1)
        self.assertEqual(receipt_wines.queue_automatic(self.store, 2, self.token), 0)
        job = self.store.one("SELECT * FROM jobs WHERE kind='life_wine_enrich'")
        payload = json.loads(job['payload'])
        self.assertTrue(payload['automatic'])
        with patch.object(enrich, '_download', side_effect=self.download) as fetch:
            state = receipt_wines.run_lookup(self.store, 2, payload)
        self.assertEqual(fetch.call_count, 2)
        return state

    def test_real_receipt_identity_merges_sources_and_preserves_unknown_bottle_year(self):
        state = self.run_lookup()
        self.assertEqual(state['status'], 'filled')
        ui = self.store.get_setting('receipt_ui:2:'+self.token)
        values = ui['overrides']['0']
        self.assertEqual((values['country'], values['region'], values['grapes']), ('Аргентина', 'Патагония', ['Мальбек']))
        self.assertEqual(values['winery'], 'ФАМИЛИЯ ШРЕДЕР')
        self.assertNotIn('vintage', values)
        self.assertEqual(values['metadata_provenance']['region'][0]['url'], VINUM_URL)
        self.assertIn(2024, [row['published_vintage'] for row in state['candidates']])
        self.assertNotIn(2030, [row['published_vintage'] for row in state['candidates']])
        self.assertFalse(cellar.lots(self.store, 2))
        self.assertEqual(ledger.summary(self.store, 2)['count'], 0)
        self.assertEqual(receipt_wines.queue_automatic(self.store, 2, self.token), 0)
        ledger.confirm_receipt(self.store, 2, self.token, selected_wine_items=[0], overrides=ui['overrides'])
        lot = cellar.lots(self.store, 2)[0]
        self.assertEqual((lot['country'], lot['region'], lot['grapes'], lot['vintage']), ('Аргентина', 'Патагония', ['Мальбек'], ''))
        cellar.consume_bottles(self.store, 2, lot['id'], price=1000)
        reopened = Store(self.store.path)
        self.assertEqual(cellar.get_lot(reopened, 2, lot['id'])['metadata_provenance'], lot['metadata_provenance'])
        stats = cellar.attribute_statistics(reopened, 2)
        self.assertEqual(stats['groups']['country']['Аргентина'], {'purchased': 1, 'remaining': 0, 'consumed': 1})
        self.assertEqual(stats['groups']['vintage']['Не указано']['purchased'], 1)
        self.assertEqual(cellar.export_rows(reopened, 2)[0]['grapes'], 'Мальбек')
        with self.assertRaises(PermissionError):
            cellar.attribute_statistics(reopened, 3, owner=2)

    def test_source_disconnection_during_lookup_does_not_apply_metadata(self):
        receipt_wines.queue_automatic(self.store, 2, self.token)
        payload = json.loads(self.store.one("SELECT payload FROM jobs WHERE kind='life_wine_enrich'")['payload'])
        def disconnect(url, **kwargs):
            retail.configure_shop(self.store, 1, 'vinumday_metadata', False)
            return self.download(url, **kwargs)
        with patch.object(enrich, '_download', side_effect=disconnect), self.assertRaises(PermissionError):
            receipt_wines.run_lookup(self.store, 2, payload)
        self.assertFalse(self.store.get_setting('receipt_ui:2:'+self.token).get('wine_info'))

    def test_automatic_jobs_are_private_and_do_not_send_one_message_per_bottle(self):
        from thesaurus import life_jobs
        receipt_wines.queue_automatic(self.store, 2, self.token)
        job = self.store.one("SELECT * FROM jobs WHERE kind='life_wine_enrich'")
        payload = json.loads(job['payload'])
        with patch.object(enrich, '_download', side_effect=self.download):
            life_jobs.run_job(self.store, None, job, payload)
        self.assertFalse(self.store.query('SELECT * FROM deliveries'))
        self.assertEqual(receipt_wines.queue_automatic(self.store, 3, self.token), 0)
        with self.assertRaises(PermissionError):
            receipt_wines.queue_lookup(self.store, 3, self.token, 0)

    def test_ambiguous_producer_does_not_autofill_and_region_conflict_stays_unknown(self):
        inputs = enrich._inputs(self.receipt['items'][0])
        first = enrich._match(parse_reference(BAHETLE, BAHETLE_URL), inputs)
        second = enrich._match(parse_reference(VINUM, VINUM_URL), inputs)
        self.assertIsNone(enrich.automatic_candidate([first, {**second, 'winery': 'Different winery'}]))
        merged = enrich.automatic_candidate([{**first, 'region': 'Мендоса'}, second])
        self.assertEqual(merged['region'], '')
        self.assertEqual(merged['country'], 'Аргентина')
        self.assertIsNone(enrich._match(parse_reference(BAHETLE.replace('Мальбек', 'Шардоне'), BAHETLE_URL), inputs))
        with self.assertRaises(ValueError):
            parse_reference(VINUM+VINUM, VINUM_URL)

    def test_schema_two_upgrade_preserves_stock_and_closed_results_and_backfills_old_notes(self):
        lot = cellar.add_bottle(self.store, 2, {'name': 'Old wine', 'quantity': 2, 'purchase_price': 1000,
            'notes': 'Страна: Chile; Регион: Central Valley; Сорта: Merlot; Источник: https://example.com/wine'})
        cellar.consume_bottles(self.store, 2, lot['id'], price=1200)
        before = cellar.consumptions(self.store, 2)
        self.store.execute('DROP TABLE wine_attributes')
        self.store.set_setting('wine_cellar_version', 2)
        cellar.migrate(self.store)
        after = cellar.get_lot(self.store, 2, lot['id'])
        self.assertEqual((after['quantity'], after['country'], after['region'], after['grapes']),
                         (1, 'Чили', 'Центральная Долина', ['Мерло']))
        self.assertEqual(cellar.consumptions(self.store, 2), before)
        self.assertEqual(self.store.get_setting('wine_cellar_version'), cellar.VERSION)

    def test_vivino_exact_identity_supplies_attributes_without_sending_private_catalogue(self):
        fields = {'name': 'Frontera Cabernet Sauvignon', 'winery': 'Frontera'}
        self.store.set_setting('project:vivino:2:dataset', {'generation': 'test', 'rows': [{
            **fields, 'country': 'Chile', 'region': 'Central Valley', 'grapes': ['Cabernet Sauvignon'], 'vintage': 2020}]})
        for source in ('bahetle_metadata', 'vinumday_metadata'):
            retail.configure_shop(self.store, 1, source, False)
        with patch.object(enrich, '_download') as fetch:
            result = enrich.lookup(self.store, 2, fields)
        fetch.assert_not_called()
        self.assertEqual(len(result['candidates']), 1)
        self.assertEqual(result['candidates'][0]['country'], 'Chile')
        self.assertIsNone(result['candidates'][0]['vintage'])
        automatic = enrich.automatic_candidate(result['candidates'])
        self.assertEqual(automatic['region'], 'Central Valley')
        self.assertEqual(automatic['grapes'], ['Cabernet Sauvignon'])
        self.assertIsNone(automatic['vintage'])

    def test_owned_vivino_varietal_title_and_multisource_attributes_are_compared_completely(self):
        fields = {'name': 'Вино Сан Валентин Гарнача кр.сух.0,75'}
        self.store.set_setting('project:vivino:2:dataset', {'generation': 'owned', 'rows': [{
            'name': 'San Valentín Garnacha', 'winery': 'Familia Torres', 'country': 'Spain',
            'region': 'Catalunya', 'grapes': [], 'vintage': 2019}]})
        for source in ('bahetle_metadata', 'vinumday_metadata'):
            retail.configure_shop(self.store, 1, source, False)
        with patch.object(enrich, '_download') as fetch:
            result = enrich.lookup(self.store, 2, fields)
        fetch.assert_not_called()
        automatic = enrich.automatic_candidate(result['candidates'])
        self.assertEqual(automatic['grapes'], ['garnacha'])
        self.assertIsNone(automatic['vintage'])
        first = result['candidates'][0]
        second = {**first, 'country': 'Испания', 'region': 'Каталония', 'grapes': ['Гарнача']}
        self.assertIsNotNone(enrich.automatic_candidate([first, second]))
        second['grapes'] = ['Гренаш/гарнача']
        self.assertEqual(enrich.automatic_candidate([first, second])['grapes'], ['garnacha'])
        self.assertEqual(cellar._attributes(second)['grapes'], ['Гарнача'])
        # Unknown varieties in a blend cannot disappear during canonicalization.
        second['grapes'] = ['Гарнача', 'Carignan']
        self.assertEqual(enrich.automatic_candidate([first, second])['grapes'], [])
