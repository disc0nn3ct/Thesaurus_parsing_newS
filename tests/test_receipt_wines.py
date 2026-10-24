"""Requested receipt enrichment keeps identity, ownership and purchase accounting separate."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from thesaurus import feature_access, receipt_wines, wine_cellar, wine_retail
from thesaurus import life_ledger as ledger
from thesaurus.store import Store


def candidate(**changes):
    return {'id': 'b'*32, 'name': 'Merlot Reserva', 'winery': 'Bodega Ejemplo', 'volume_ml': 750,
            'vintage': None, 'published_vintage': 2020, 'country': 'Испания', 'region': 'Риоха',
            'grapes': ['Мерло'], 'description': 'Сухое красное вино.',
            'source_url': 'https://decanter.ru/wine/merlot-reserva', 'source_title': 'Decanter',
            'fetched_at': '2026-10-01T09:00:00+00:00', 'match_status': 'exact', 'match_notes': [], **changes}


def lookup_response(*values, **changes):
    return {'status': 'candidates', 'candidates': list(values), 'warnings': [], 'cached': False, **changes}


class ReceiptWineTests(unittest.TestCase):
    def test_other_field_choices_are_normalized_and_never_overwrite_another_field(self):
        self.run_lookup(lookup_response(candidate(),candidate(id='c'*32,winery='Familia Schroeder',
            country='Argentina',region='Mendoza',grapes=['Malbec'],volume_ml=375)))
        for field,expected in [('winery','Familia Schroeder'),('country','Аргентина'),('region','Мендоса'),
                               ('grapes',['Мальбек']),('volume_ml',375)]:
            choices = receipt_wines.field_choices(self.store,2,self.token,0,field)
            choice = next(row for row in choices if row['value']==expected)
            before = self.ui()['overrides']['0'].copy()
            receipt_wines.apply_field_choice(self.store,2,self.token,0,field,choice['candidate'],choice['fingerprint'])
            after = self.ui()['overrides']['0']
            self.assertEqual(after[field],expected)
            self.assertEqual({key:value for key,value in after.items() if key!=field},
                             {key:value for key,value in before.items() if key!=field})
        self.assert_no_financial_changes()

    def test_field_choices_deduplicate_years_and_only_change_the_confirmed_field(self):
        self.result['items'][0]['volume_ml'] = 750
        ledger.save_receipt_draft(self.store,2,self.result,self.token)
        self.run_lookup(lookup_response(candidate(published_vintage=2015),
            candidate(id='c'*32,published_vintage=2024,winery='Другой производитель'),
            candidate(id='d'*32,published_vintage=2024)))
        before = self.ui()
        choices = receipt_wines.field_choices(self.store,2,self.token,0,'vintage')
        self.assertEqual([row['label'] for row in choices],['2015','2024'])
        self.assertNotIn('vintage',before['overrides'].get('0',{}))
        choice = choices[1]
        receipt_wines.apply_field_choice(self.store,2,self.token,0,'vintage',choice['candidate'],choice['fingerprint'])
        after = self.ui()['overrides']['0']
        self.assertEqual(after['vintage'],'2024')
        self.assertEqual({key:value for key,value in after.items() if key!='vintage'},before['overrides'].get('0',{}))
        self.assert_no_financial_changes()
        with self.assertRaises(ValueError):
            receipt_wines.apply_field_choice(self.store,2,self.token,0,'vintage',choice['candidate'],choice['fingerprint'])
        renewed = receipt_wines.field_choices(self.store,2,self.token,0,'vintage')[0]
        self.assertNotEqual(renewed['fingerprint'],choices[0]['fingerprint'])
        ledger.confirm_receipt(self.store,2,self.token,selected_wine_items=[0],overrides={0:after})
        self.assertEqual(wine_cellar.lots(self.store,2)[0]['vintage'],'2024')

    def test_field_choice_respects_expiry_ownership_and_disabled_source(self):
        self.run_lookup(lookup_response(candidate()))
        choice = receipt_wines.field_choices(self.store,2,self.token,0,'vintage')[0]
        with self.assertRaises((PermissionError,ValueError)):
            receipt_wines.apply_field_choice(self.store,3,self.token,0,'vintage',choice['candidate'],choice['fingerprint'])
        ui = self.ui()
        ui['wine_lookups']['0']['expires_at'] = '2000-01-01T00:00:00+00:00'
        self.save_ui(ui)
        with self.assertRaises(ValueError):
            receipt_wines.apply_field_choice(self.store,2,self.token,0,'vintage',choice['candidate'],choice['fingerprint'])
        ui['wine_lookups']['0']['expires_at'] = '2099-01-01T00:00:00+00:00'
        self.save_ui(ui)
        wine_retail.configure_shop(self.store,1,'decanter',False)
        self.assertEqual(receipt_wines.field_choices(self.store,2,self.token,0,'vintage'),[])
        with self.assertRaises(ValueError):
            receipt_wines.apply_field_choice(self.store,2,self.token,0,'vintage',choice['candidate'],choice['fingerprint'])
        self.assert_no_financial_changes()

    def test_source_name_becomes_final_wine_name_without_rewriting_receipt(self):
        final_name = 'Bodega Ejemplo Merlot Reserva'
        self.run_lookup(lookup_response(candidate(name=final_name)))
        ui = self.ui()
        self.assertEqual(ui['overrides']['0']['name'], final_name)
        self.assertEqual(ui['original_items']['0']['name'], 'Merlot Reserva')
        self.assertEqual(ledger.get_receipt_draft(self.store, 2, self.token)['result']['items'][0]['name'], 'Merlot Reserva')
        self.assert_no_financial_changes()
        ledger.confirm_receipt(self.store, 2, self.token, selected_wine_items=[0], overrides={0: ui['overrides']['0']})
        lot = wine_cellar.lots(self.store, 2)[0]
        self.assertEqual((lot['name'], lot['quantity'], lot['purchase_price']), (final_name, 2, 1000))
        imported = self.store.one('SELECT items FROM life_receipt_imports WHERE user_id=2')
        self.assertEqual(json.loads(imported['items'])[0]['name'], 'Merlot Reserva')
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 2200)

    def test_switching_source_names_keeps_original_and_saves_last_choice(self):
        first = candidate(name='Первое название', match_status='possible')
        second = candidate(id='c'*32, name='Итоговое название', match_status='possible')
        self.run_lookup(lookup_response(first, second))
        self.assertNotIn('name', self.ui()['overrides'].get('0', {}))
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0)
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 1)
        ui = self.ui()
        self.assertEqual(ui['original_items']['0']['name'], 'Merlot Reserva')
        self.assertEqual(ui['overrides']['0']['name'], 'Итоговое название')
        self.assertEqual(ui['wine_info']['0']['previous_values']['name'], 'Merlot Reserva')
        ledger.confirm_receipt(self.store, 2, self.token, selected_wine_items=[0], overrides={0: ui['overrides']['0']})
        self.assertEqual(wine_cellar.lots(self.store, 2)[0]['name'], 'Итоговое название')

    def test_source_name_too_long_for_inventory_is_not_silently_truncated(self):
        self.run_lookup(lookup_response(candidate(name='В'*201)))
        self.assertNotIn('name', self.ui()['overrides']['0'])
        self.assertEqual(self.ui()['overrides']['0']['winery'], 'Bodega Ejemplo')
        self.assert_no_financial_changes()

    def test_refresh_fills_legacy_empty_grapes_without_duplicate_jobs_or_purchase_changes(self):
        self.store.set_setting('wine_metadata_auto:2', True)
        self.save_ui({'selected': [0], 'overrides': {'0': {'grapes': [], 'notes': 'Моя пометка'}}})
        self.run_lookup(lookup_response(candidate(grapes=[])))
        first = self.store.one('SELECT id FROM jobs ORDER BY id LIMIT 1')['id']
        self.assertNotIn('grapes', self.ui()['wine_info']['0']['auto_values'])
        self.store.execute("UPDATE jobs SET state='done' WHERE id=?", (first,))
        refreshed = receipt_wines.queue_lookup(self.store, 2, self.token, 0, automatic=True, refresh=True)
        self.assertNotEqual(first, refreshed)
        repeated = receipt_wines.queue_lookup(self.store, 2, self.token, 0, automatic=True, refresh=True)
        self.assertEqual(refreshed, repeated)
        payload = json.loads(self.store.one('SELECT payload FROM jobs WHERE id=?', (refreshed,))['payload'])
        self.assertTrue(payload['automatic'])
        self.run_lookup(lookup_response(candidate(grapes=['Merlot'])), payload=payload)
        self.assertEqual(self.ui()['overrides']['0']['grapes'], ['Мерло'])
        self.assertTrue(self.ui()['overrides']['0']['notes'].startswith('Моя пометка'))
        self.assertEqual(self.ui()['selected'], [0])
        self.assert_no_financial_changes()

    def test_published_year_requires_separate_bottle_confirmation(self):
        self.run_lookup(lookup_response(candidate(match_status='possible', published_vintage=2020)))
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0)
        self.assertNotIn('vintage', self.ui()['overrides']['0'])
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0, confirm_vintage=True)
        self.assertEqual(self.ui()['overrides']['0']['vintage'], '2020')
        self.assertEqual(receipt_wines.lookup_result(self.store, 2, self.token, 0)['status'], 'filled')
        receipt_wines.invalidate_item(self.store, 2, self.token, 0, 'name')
        self.assertEqual(self.ui()['overrides']['0']['vintage'], '2020')
        self.assert_no_financial_changes()

    def test_owned_vivino_candidate_can_fill_receipt_and_is_invalid_after_replacement(self):
        from thesaurus import wine_retail
        wine_retail.configure_shop(self.store, 1, 'decanter', False)
        self.store.set_setting('project:vivino:2:dataset', {'generation': 'owned', 'rows': [{
            'name': 'Merlot Reserva', 'winery': 'Bodega Ejemplo', 'country': 'Spain', 'region': 'Rioja',
            'vintage': 2020, 'link': 'https://www.vivino.com/wines/12345'}]})
        _, payload = self.queue()
        result = receipt_wines.run_lookup(self.store, 2, payload)
        self.assertEqual(len(result['candidates']), 1)
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0)
        override = self.ui()['overrides']['0']
        self.assertEqual(override['winery'], 'Bodega Ejemplo')
        self.assertIn('Spain', override['notes'])
        self.assertIn('Ваша загрузка Vivino', override['notes'])
        self.assertNotIn('vintage', override)
        self.store.set_setting('project:vivino:2:dataset', {'generation': 'new', 'rows': []})
        with self.assertRaises(ValueError):
            receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0)
        self.assert_no_financial_changes()

    def setUp(self):
        parent = Path(__file__).resolve().parents[1]/'.test-tmp'
        parent.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temp.cleanup)
        self.store = Store(Path(temp.name)/'receipt-wines.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        for actor in (2, 3):
            self.store.add_user(actor, subscriptions=['life', 'vivino'])
        self.store.add_user(4, subscriptions=['life'])
        self.store.add_user(5, subscriptions=['vivino'])
        wine_retail.configure_shop(self.store, 1, 'decanter', True)
        self.token = uuid4().hex
        self.result = {'status': 'candidate', 'fingerprint': 'a'*64, 'date': '2026-10-01',
                       'merchant': 'Тестовый магазин', 'currency': 'RUB', 'total': 2200,
                       'items': [{'name': 'Merlot Reserva', 'quantity': 2, 'amount': 2000, 'unit_price': 1000,
                                  'winery': '', 'volume_ml': None, 'vintage': None, 'is_wine': True},
                                 {'name': 'Сыр', 'quantity': 0.2, 'amount': 200, 'is_wine': False}], 'warnings': []}
        ledger.save_receipt_draft(self.store, 2, self.result, self.token)
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No network in receipt tests')))

    def ui(self):
        return self.store.get_setting(f'receipt_ui:2:{self.token}', {'selected': [], 'overrides': {}})

    def save_ui(self, values):
        self.store.set_setting(f'receipt_ui:2:{self.token}', values)

    def queue(self):
        identifier = receipt_wines.queue_lookup(self.store, 2, self.token, 0)
        return identifier, json.loads(self.store.one('SELECT payload FROM jobs WHERE id=?', (identifier,))['payload'])

    def run_lookup(self, values, *, payload=None, dry_run=False):
        if payload is None:
            _, payload = self.queue()
        with patch('thesaurus.wine_enrichment.lookup', return_value=values) as lookup:
            state = receipt_wines.run_lookup(self.store, 2, payload, dry_run=dry_run)
        return state, lookup

    def assert_no_financial_changes(self):
        self.assertEqual(self.store.query('SELECT * FROM life_ledger_entries'), [])
        self.assertEqual(self.store.query('SELECT * FROM life_receipt_imports'), [])
        self.assertEqual(self.store.query('SELECT * FROM wine_lots'), [])

    def test_queue_repeated_taps_reuse_one_private_job_without_http_or_purchase_writes(self):
        first, payload = self.queue()
        second, repeated = self.queue()
        self.assertEqual(first, second)
        self.assertEqual(payload, repeated)
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)
        self.assertEqual((payload['actor_id'], payload['owner_id'], payload['receipt_token'], payload['index']),
                         (2, 2, self.token, 0))
        self.assertTrue(payload['signature'])
        self.assertIn('wine_cellar', payload['required_features'])
        self.assertNotIn('Тестовый магазин', str(payload))
        self.assertNotIn('Сыр', str(payload))
        self.assertEqual(ledger.get_receipt_draft(self.store, 2, self.token)['result']['total'], 2200)
        self.assert_no_financial_changes()

    def test_only_private_owner_with_both_functions_can_queue(self):
        for actor in (1, 3, 4, 5, 99):
            with self.subTest(actor=actor), self.assertRaises((ValueError, PermissionError)):
                receipt_wines.queue_lookup(self.store, actor, self.token, 0)
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.assert_no_financial_changes()

    def test_confirmed_expired_and_invalid_receipt_rows_cannot_be_enriched(self):
        for index in (-1, 2, True, '0'):
            with self.subTest(index=index), self.assertRaises(ValueError):
                receipt_wines.queue_lookup(self.store, 2, self.token, index)
        self.store.execute("UPDATE life_receipt_drafts SET expires_at='2000-01-01' WHERE token=?", (self.token,))
        with self.assertRaises(ValueError):
            self.queue()
        self.store.execute("UPDATE life_receipt_drafts SET expires_at='2999-01-01' WHERE token=?", (self.token,))
        ledger.confirm_receipt(self.store, 2, self.token, selected_wine_items=[])
        with self.assertRaises(ValueError):
            self.queue()
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_unique_exact_adds_only_missing_identity_and_provenance_not_guessed_year(self):
        original = ledger.get_receipt_draft(self.store, 2, self.token)['result']
        state, lookup = self.run_lookup(lookup_response(candidate(vintage=2020)))
        self.assertEqual(state['status'], 'filled')
        self.assertTrue({'winery', 'volume_ml', 'notes'} <= set(state['fields_added']))
        override = self.ui()['overrides']['0']
        self.assertEqual((override['winery'], override['volume_ml']), ('Bodega Ejemplo', 750))
        self.assertNotIn('vintage', override)
        self.assertNotIn('name', override)
        self.assertIn('Испания', override['notes'])
        self.assertIn('Риоха', override['notes'])
        self.assertIn('Мерло', override['notes'])
        self.assertIn(candidate()['source_url'], override['notes'])
        self.assertEqual(ledger.get_receipt_draft(self.store, 2, self.token)['result'], original)
        self.assertEqual(self.ui()['selected'], [])
        self.assertTrue(callable(lookup.call_args.kwargs['guard']))
        self.assertEqual(receipt_wines.lookup_result(self.store, 2, self.token, 0)['status'], 'filled')
        self.assert_no_financial_changes()

    def test_existing_manual_fields_and_notes_survive_exact_lookup(self):
        self.save_ui({'selected': [], 'overrides': {'0': {'name': 'Моё название', 'winery': 'Мой производитель',
                     'vintage': 2019, 'volume_ml': 1500, 'notes': 'Подарок'}}})
        self.run_lookup(lookup_response(candidate(vintage=2020)))
        override = self.ui()['overrides']['0']
        self.assertEqual((override['name'], override['winery'], override['vintage'], override['volume_ml']),
                         ('Моё название', 'Мой производитель', 2019, 1500))
        self.assertTrue(override['notes'].startswith('Подарок'))
        self.assert_no_financial_changes()

    def test_ambiguous_lookup_requires_explicit_choice_and_choice_is_idempotent(self):
        values = lookup_response(candidate(match_status='possible'), candidate(id='c'*32, winery='Второй производитель',
                                 source_url='https://decanter.ru/wine/second', match_status='possible'))
        state, _ = self.run_lookup(values)
        self.assertEqual(state['fields_added'], [])
        self.assertFalse(any(self.ui()['overrides'].values()))
        added = receipt_wines.apply_candidate(self.store, 2, self.token, 0, 1)
        self.assertIn('winery', added)
        once = self.ui()['overrides']
        self.assertEqual(once['0']['winery'], 'Второй производитель')
        self.assertNotIn('vintage', once['0'])
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 1)
        self.assertEqual(self.ui()['overrides'], once)
        self.assert_no_financial_changes()

    def test_single_possible_match_is_not_automatically_applied(self):
        state, _ = self.run_lookup(lookup_response(candidate(match_status='possible')))
        self.assertEqual(state['fields_added'], [])
        self.assertFalse(any(self.ui()['overrides'].values()))
        self.assert_no_financial_changes()

    def test_reapplying_same_information_keeps_provenance_for_later_name_correction(self):
        self.save_ui({'selected': [], 'overrides': {'0': {'notes': 'Моя пометка'}}})
        self.run_lookup(lookup_response(candidate()))
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0)
        receipt_wines.invalidate_item(self.store, 2, self.token, 0, 'name')
        override = self.ui()['overrides']['0']
        self.assertNotIn('winery', override)
        self.assertNotIn('volume_ml', override)
        self.assertEqual(override['notes'], 'Моя пометка')
        self.assertFalse(self.ui().get('wine_info', {}).get('0'))
        self.assert_no_financial_changes()

    def test_switching_candidate_replaces_only_previous_automatic_information(self):
        purchase_record = ledger.get_receipt_draft(self.store, 2, self.token)['result']
        self.save_ui({'selected': [0], 'overrides': {'0': {'notes': 'Моя пометка'}}})
        first = candidate(match_status='possible')
        second = candidate(id='c'*32, winery='Другая винодельня', volume_ml=1500, country='Чили',
                           region='Долина Мауле', grapes=['Каберне Совиньон'],
                           source_url='https://decanter.ru/wine/second', match_status='possible')
        self.run_lookup(lookup_response(first, second))
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0)
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 1)
        override = self.ui()['overrides']['0']
        self.assertEqual((override['winery'], override['volume_ml']), ('Другая винодельня', 1500))
        self.assertIn('Моя пометка', override['notes'])
        self.assertIn(second['source_url'], override['notes'])
        self.assertIn('Долина Мауле', override['notes'])
        self.assertNotIn(first['source_url'], override['notes'])
        self.assertNotIn('Риоха', override['notes'])
        self.assertEqual(self.ui()['selected'], [0])
        self.assertEqual(ledger.get_receipt_draft(self.store, 2, self.token)['result'], purchase_record)
        self.assert_no_financial_changes()

    def test_switching_candidate_preserves_fields_manually_corrected_after_first_choice(self):
        first = candidate(match_status='possible')
        second = candidate(id='c'*32, winery='Из магазина', volume_ml=1500,
                           source_url='https://decanter.ru/wine/second', match_status='possible')
        self.run_lookup(lookup_response(first, second))
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0)
        receipt_wines.invalidate_item(self.store, 2, self.token, 0, 'winery')
        ui = self.ui()
        ui['overrides']['0']['winery'] = 'Проверил на этикетке'
        self.save_ui(ui)
        # A changed identity requires a fresh lookup before any further choice.
        self.run_lookup(lookup_response(first, second))
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 1)
        override = self.ui()['overrides']['0']
        self.assertEqual(override['winery'], 'Проверил на этикетке')
        self.assertEqual(override['volume_ml'], 1500)
        self.assertIn(second['source_url'], override['notes'])
        self.assertNotIn(first['source_url'], override['notes'])
        self.assert_no_financial_changes()

    def test_candidate_without_previous_automatic_identity_keeps_current_result_signature(self):
        self.run_lookup(lookup_response(candidate()))
        self.assertEqual(self.ui()['overrides']['0']['volume_ml'], 750)
        refreshed = candidate(winery='', volume_ml=None, match_status='possible',
                              fetched_at='2026-10-11T10:30:00+00:00')
        self.run_lookup(lookup_response(refreshed))
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0)
        override = self.ui()['overrides']['0']
        self.assertFalse(override.get('winery'))
        self.assertFalse(override.get('volume_ml'))
        state = receipt_wines.lookup_result(self.store, 2, self.token, 0)
        self.assertEqual(state['status'], 'filled')
        self.assertEqual(state['candidates'][0]['fetched_at'], refreshed['fetched_at'])
        receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0)
        self.assertEqual(receipt_wines.lookup_result(self.store, 2, self.token, 0)['signature'], state['signature'])
        self.assert_no_financial_changes()

    def test_manual_volume_correction_survives_later_name_change(self):
        self.run_lookup(lookup_response(candidate()))
        receipt_wines.invalidate_item(self.store, 2, self.token, 0, 'volume_ml')
        ui = self.ui()
        ui['overrides']['0']['volume_ml'] = 1500
        self.save_ui(ui)
        receipt_wines.invalidate_item(self.store, 2, self.token, 0, 'name')
        override = self.ui()['overrides']['0']
        self.assertEqual(override['volume_ml'], 1500)
        self.assertNotIn('winery', override)
        self.assertNotIn('notes', override)
        self.assert_no_financial_changes()

    def test_each_manual_identity_edit_invalidates_source_metadata_and_keeps_manual_fields(self):
        for field, value in (('winery', 'С этикетки'), ('vintage', '2021'), ('volume_ml', 375)):
            with self.subTest(field=field):
                self.save_ui({'selected': [0], 'overrides': {'0': {'notes': 'Мои заметки',
                                                                 'location': 'Винный шкаф', 'vintage': '2019'}}})
                self.run_lookup(lookup_response(candidate()))
                receipt_wines.invalidate_item(self.store, 2, self.token, 0, field)
                ui = self.ui()
                self.assertFalse(ui.get('wine_info', {}).get('0'))
                self.assertFalse(ui.get('wine_lookups', {}).get('0'))
                override = ui['overrides']['0']
                self.assertEqual((override['notes'], override['location']), ('Мои заметки', 'Винный шкаф'))
                if field != 'vintage':
                    self.assertEqual(override['vintage'], '2019')
                override[field] = value
                self.save_ui(ui)
                receipt_wines.invalidate_item(self.store, 2, self.token, 0, 'name')
                self.assertEqual(self.ui()['overrides']['0'][field], value)
                self.assertEqual(self.ui()['selected'], [0])
        self.assert_no_financial_changes()

    def test_refresh_same_source_replaces_previous_automatic_notes_without_duplicates(self):
        self.save_ui({'selected': [], 'overrides': {'0': {'notes': 'Мои заметки'}}})
        first = candidate(fetched_at='2026-10-01T09:00:00+00:00', description='Старое описание.')
        updated = candidate(fetched_at='2026-10-11T10:30:00+00:00', description='Уточнённое описание.', region='Новый регион')
        self.run_lookup(lookup_response(first))
        self.run_lookup(lookup_response(updated))
        notes = self.ui()['overrides']['0']['notes']
        self.assertEqual(notes.count(first['source_url']), 1)
        self.assertEqual(notes.count('Мои заметки'), 1)
        self.assertNotIn(first['fetched_at'], notes)
        self.assertNotIn('Старое описание', notes)
        self.assertNotIn('Риоха', notes)
        self.assertIn(updated['fetched_at'], notes)
        self.assertIn('Уточнённое описание', notes)
        self.assertIn('Новый регион', notes)
        receipt_wines.invalidate_item(self.store, 2, self.token, 0, 'name')
        self.assertEqual(self.ui()['overrides']['0']['notes'], 'Мои заметки')
        self.assert_no_financial_changes()

    def test_disabling_source_during_lookup_prevents_final_automatic_apply(self):
        self.save_ui({'selected': [], 'overrides': {'0': {'notes': 'Мои заметки'}}})
        _, payload = self.queue()
        def disable_during_lookup(*args, **kwargs):
            wine_retail.configure_shop(self.store, 1, 'decanter', False)
            return lookup_response(candidate())
        with patch('thesaurus.wine_enrichment.lookup', side_effect=disable_during_lookup):
            state = receipt_wines.run_lookup(self.store, 2, payload)
        self.assertEqual(state['status'], 'disabled')
        self.assertFalse(state['candidates'])
        self.assertFalse(state['fields_added'])
        self.assertEqual(self.ui()['overrides'], {'0': {'notes': 'Мои заметки'}})
        self.assertFalse(self.ui().get('wine_info', {}).get('0'))
        self.assert_no_financial_changes()

    def test_manual_name_or_volume_change_during_lookup_makes_result_stale(self):
        for field, value in (('name', 'Другое вино'), ('volume_ml', 1500)):
            with self.subTest(field=field):
                self.save_ui({'selected': [], 'overrides': {}})
                _, payload = self.queue()
                def edit_during_lookup(*args, field=field, value=value, **kwargs):
                    ui = self.ui()
                    ui['overrides']['0'] = {field: value}
                    self.save_ui(ui)
                    return lookup_response(candidate())
                with patch('thesaurus.wine_enrichment.lookup', side_effect=edit_during_lookup):
                    state = receipt_wines.run_lookup(self.store, 2, payload)
                self.assertEqual(state['status'], 'stale')
                self.assertEqual(self.ui()['overrides']['0'], {field: value})
        self.assert_no_financial_changes()

    def test_changed_identity_or_expired_result_cannot_be_applied(self):
        self.run_lookup(lookup_response(candidate(match_status='possible')))
        ui = self.ui()
        ui['overrides']['0'] = {'volume_ml': 1500}
        self.save_ui(ui)
        with self.assertRaises(ValueError):
            receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0)
        ui['overrides'] = {}
        ui['wine_lookups']['0']['expires_at'] = '2000-01-01'
        self.save_ui(ui)
        with self.assertRaises(ValueError):
            receipt_wines.apply_candidate(self.store, 2, self.token, 0, 0)
        self.assert_no_financial_changes()

    def test_result_and_choice_cannot_cross_users_or_missing_permissions(self):
        self.run_lookup(lookup_response(candidate(match_status='possible')))
        for actor in (1, 3, 4, 5):
            with self.subTest(actor=actor), self.assertRaises((ValueError, PermissionError)):
                receipt_wines.lookup_result(self.store, actor, self.token, 0)
            with self.subTest(actor=actor), self.assertRaises((ValueError, PermissionError)):
                receipt_wines.apply_candidate(self.store, actor, self.token, 0, 0)
        for index in (-1, True, '0', 1):
            with self.subTest(index=index), self.assertRaises(ValueError):
                receipt_wines.apply_candidate(self.store, 2, self.token, 0, index)
        self.assert_no_financial_changes()

    def test_revocation_after_queue_is_checked_before_provider(self):
        _, payload = self.queue()
        feature_access.change(self.store, 1, 2, 'wine_cellar', False)
        with patch('thesaurus.wine_enrichment.lookup') as lookup, self.assertRaises(PermissionError):
            receipt_wines.run_lookup(self.store, 2, payload)
        lookup.assert_not_called()
        self.assert_no_financial_changes()

    def test_revocation_during_lookup_prevents_overrides_and_confirmation(self):
        _, payload = self.queue()
        def revoke(*args, **kwargs):
            feature_access.change(self.store, 1, 2, 'wine_cellar', False)
            return lookup_response(candidate())
        with patch('thesaurus.wine_enrichment.lookup', side_effect=revoke), self.assertRaises(PermissionError):
            receipt_wines.run_lookup(self.store, 2, payload)
        self.assertEqual(self.ui()['overrides'], {})
        self.assert_no_financial_changes()

    def test_dry_run_never_calls_provider_or_changes_review_state(self):
        _, payload = self.queue()
        before = self.ui()
        state, lookup = self.run_lookup(lookup_response(candidate()), payload=payload, dry_run=True)
        self.assertEqual(state['status'], 'dry_run')
        lookup.assert_not_called()
        self.assertEqual(self.ui(), before)
        self.assert_no_financial_changes()

    def test_cached_source_result_still_waits_for_explicit_receipt_confirmation(self):
        state, _ = self.run_lookup(lookup_response(candidate(), cached=True))
        self.assertTrue(state['cached'])
        self.assert_no_financial_changes()
        original = ledger.get_receipt_draft(self.store, 2, self.token)['result']
        ui = self.ui()
        ledger.confirm_receipt(self.store, 2, self.token, selected_wine_items=[0], overrides={0: ui['overrides']['0']})
        lot = wine_cellar.lots(self.store, 2)[0]
        self.assertEqual((lot['name'], lot['quantity'], lot['purchase_price'], lot['purchase_date']),
                         ('Merlot Reserva', 2, 1000, original['date']))
        self.assertEqual(lot['volume_ml'], 750)
        self.assertFalse(lot['vintage'])
        self.assertIn(candidate()['source_url'], lot['notes'])
        self.assertIn('Риоха', lot['notes'])
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], original['total'])
        self.assertEqual(ledger.summary(self.store, 2)['count'], 1)


if __name__ == '__main__':
    unittest.main()
