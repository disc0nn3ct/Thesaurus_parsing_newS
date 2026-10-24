"""Moscow retail discovery stays private, opt-in and separate from valuations."""
import json
import shutil
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from thesaurus import wine_cellar as cellar
from thesaurus import wine_jobs
from thesaurus import wine_retail as retail
from thesaurus.config import Settings
from thesaurus.feature_access import key
from thesaurus.store import Store, utcnow


class RetailTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]/'.test-tmp'/('retail-'+uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root/'state.sqlite3')
        for actor, role in ((1, 'admin'), (2, 'subscriber'), (3, 'subscriber')):
            self.store.add_user(actor, role, subscriptions='all')
        self.store.set_setting('wine_price_allowed_hosts', [])
        self.lot = cellar.add_bottle(self.store, 2, {'name': 'Chateau Minuty M Rose',
                    'winery': 'Chateau Minuty', 'vintage': 2025, 'quantity': 2, 'volume_ml': 750})
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No network')))

    def quote(self, **changes):
        return {'status': 'ok', 'title': 'Chateau Minuty M Rose 2025, 0.75 л',
                'name': 'Chateau Minuty M Rose', 'winery': 'Chateau Minuty', 'vintage': 2025,
                'volume_ml': 750, 'amount': 3670, 'currency': 'RUB', 'availability': 'InStock',
                'city': 'Москва', 'price_kind': 'public', 'availability_scope': 'В московском магазине, сегодня',
                'as_of': utcnow(), 'shop_id': 'decanter', 'url': 'https://decanter.ru/product/minuty-id1',
                'match': True, **changes}

    def enable(self):
        retail.configure_shop(self.store, 1, 'decanter', True)

    def save(self, quote=None, actor=2, **changes):
        token = uuid4().hex
        result = {'actor_id': actor, 'lot_id': self.lot['id'], 'created_at': utcnow(),
                  'expires': (datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat(),
                  'candidates': [quote or self.quote()], 'sources': [], **changes}
        self.store.set_setting(f'wine_retail_search:{actor}:{token}', result)
        return token

    def test_profiles_start_disabled_and_separate_supported_from_blocked(self):
        profiles = {row['id']: row for row in retail.shops(self.store)}
        self.assertFalse(any(row['enabled'] for row in profiles.values()))
        self.assertTrue(profiles['decanter']['automatic'])
        self.assertFalse(profiles['winelab']['automatic'])
        self.assertFalse(profiles['perekrestok']['automatic'])
        self.assertTrue(all(row['city'] == 'Москва' for row in profiles.values()))

    def test_only_admin_configures_presets_disable_retains_custom_domains(self):
        with self.assertRaises(PermissionError):
            retail.configure_shop(self.store, 2, 'decanter', True)
        self.store.set_setting('wine_price_allowed_hosts', ['custom.example', 'www.decanter.ru'])
        self.enable()
        self.assertEqual(retail.allowed_hosts(self.store), ['custom.example', 'decanter.ru'])
        retail.configure_shop(self.store, 1, 'decanter', False)
        self.assertEqual(retail.allowed_hosts(self.store), ['custom.example'])
        self.assertEqual(self.store.get_setting('wine_retail_audit')[-1]['enabled'], False)
        self.assertFalse(next(row for row in retail.shops(self.store) if row['id'] == 'decanter')['enabled'])

    def test_disable_overrides_inherited_environment_domain(self):
        self.store.set_setting('wine_price_allowed_hosts', None)
        with patch.dict('os.environ', {'WINE_PRICE_ALLOWED_HOSTS': 'decanter.ru, custom.example'}):
            self.assertIn('decanter.ru', retail.allowed_hosts(self.store))
            retail.configure_shop(self.store, 1, 'decanter', False)
            self.assertEqual(retail.allowed_hosts(self.store), ['custom.example'])

    def test_search_disabled_manual_and_dry_run_never_request_sources(self):
        with patch.object(retail, '_source_search') as search:
            self.assertEqual(retail.search(self.store, 2, self.lot['id'])['sources'], [])
            retail.configure_shop(self.store, 1, 'winelab', True)
            result = retail.search(self.store, 2, self.lot['id'])
            self.assertEqual(result['sources'][0]['status'], 'manual')
            self.enable()
            retail.search(self.store, 2, self.lot['id'], dry_run=True)
        search.assert_not_called()

    def test_search_cache_keeps_actual_observation_date_and_never_changes_prices(self):
        self.enable()
        quote = self.quote()
        with patch.object(retail, '_source_search', return_value=([quote], {'status': 'ok', 'reason': 'Найдено'})) as search:
            one = retail.search(self.store, 2, self.lot['id'])
            two = retail.search(self.store, 2, self.lot['id'])
        search.assert_called_once()
        self.assertEqual(two['candidates'][0]['as_of'], quote['as_of'])
        self.assertTrue(two['sources'][0]['cached'])
        self.assertTrue(one['candidates'][0]['match'])
        self.assertEqual(self.store.one('SELECT count(*) n FROM wine_prices')['n'], 0)
        self.assertFalse(cellar.price_links(self.store, 2))

    def test_search_failure_is_partial_and_safe(self):
        self.enable()
        retail.configure_shop(self.store, 1, 'winelab', True)
        with patch.object(retail, '_source_search', side_effect=OSError('sensitive response')):
            result = retail.search(self.store, 2, self.lot['id'])
        self.assertEqual(len(result['sources']), 2)
        self.assertNotIn('sensitive', str(result))
        self.assertEqual(result['candidates'], [])

    def test_unique_search_redirect_reuses_verified_card_without_second_download(self):
        quote = self.quote()
        with patch('thesaurus.wine_tools._download', return_value=('single card', quote['url'])), \
                patch('thesaurus.wine_retail_sources.parse_decanter_product', return_value=quote), \
                patch('thesaurus.wine_tools.fetch_price') as fetch:
            offers, status = retail._source_search('decanter', self.lot, ['decanter.ru'])
        fetch.assert_not_called()
        self.assertTrue(offers[0]['match'])
        self.assertEqual(status['status'], 'ok')

    def test_source_checks_access_between_product_requests(self):
        urls = [{'url': f'https://decanter.ru/product/wine-id{number}'} for number in (1, 2, 3)]
        calls = []
        def guard():
            calls.append(True)
            if len(calls) == 3:  # initial search, first card, then revoked before second card.
                raise PermissionError('Access revoked')
        with patch('thesaurus.wine_tools._download', return_value=('search page', 'https://decanter.ru/search/?q=Test')), \
                patch('thesaurus.wine_retail_sources.decanter_search_status', return_value={'status': 'ok', 'candidates': urls}), \
                patch('thesaurus.wine_tools.fetch_price', return_value=self.quote()) as fetch, self.assertRaises(PermissionError):
            retail._source_search('decanter', self.lot, ['decanter.ru'], guard=guard)
        self.assertEqual(fetch.call_count, 1)

    def test_search_does_not_swallow_access_revocation_as_source_error(self):
        self.enable()
        with patch.object(retail, '_source_search', side_effect=PermissionError('Access revoked')), self.assertRaises(PermissionError):
            retail.search(self.store, 2, self.lot['id'])

    def test_confirmed_quote_cache_is_reused_by_refresh_with_observation_date(self):
        self.enable()
        quote = self.quote()
        token = self.save(quote)
        retail.confirm_candidate(self.store, 2, token, 0)
        payload = {'actor_id': 2, 'report_id': uuid4().hex, 'lot_id': self.lot['id']}
        with patch('thesaurus.wine_tools.fetch_price') as fetch:
            wine_jobs.run_job(self.store, Settings(root=self.root), {'kind': 'wine_price_refresh', 'id': 102}, payload)
        fetch.assert_not_called()
        latest = cellar.get_lot(self.store, 2, self.lot['id'])['latest_price']
        self.assertEqual(latest['metadata']['observed_at'], quote['as_of'])
        self.assertEqual(latest['amount'], quote['amount'])

    def test_unknown_region_cannot_undercut_verified_moscow_quote(self):
        self.enable()
        self.store.set_setting('wine_price_allowed_hosts', ['custom.example'])
        cellar.add_price_link(self.store, 2, self.lot['id'], self.quote()['url'])
        cellar.add_price_link(self.store, 2, self.lot['id'], 'https://custom.example/product/test')
        unscoped = self.quote(amount=1000, url='https://custom.example/product/test')
        for field in ('city', 'price_kind', 'availability_scope', 'shop_id'):
            unscoped.pop(field, None)
        with patch('thesaurus.wine_tools.fetch_price', side_effect=[self.quote(), unscoped]):
            wine_jobs.run_job(self.store, Settings(root=self.root), {'kind': 'wine_price_refresh', 'id': 103},
                              {'actor_id': 2, 'report_id': uuid4().hex, 'lot_id': self.lot['id']})
        latest = cellar.get_lot(self.store, 2, self.lot['id'])['latest_price']
        self.assertEqual(latest['amount'], 3670)
        self.assertEqual(latest['metadata']['checked_offer_count'], 1)
        self.assertEqual(latest['metadata']['city'], 'Москва')

    def test_refresh_preserves_final_custom_source_and_rechecks_its_permission(self):
        self.store.set_setting('wine_price_allowed_hosts', ['first.example', 'final.example'])
        cellar.add_price_link(self.store, 2, self.lot['id'], 'https://first.example/product/a')
        quote = self.quote(url='https://final.example/product/b')
        for field in ('city', 'price_kind', 'availability_scope', 'shop_id'):
            quote.pop(field)
        with patch('thesaurus.wine_tools.fetch_price', return_value=quote):
            wine_jobs.run_job(self.store, Settings(root=self.root), {'kind': 'wine_price_refresh', 'id': 104},
                              {'actor_id': 2, 'report_id': uuid4().hex, 'lot_id': self.lot['id']})
        self.assertEqual(cellar.get_lot(self.store, 2, self.lot['id'])['latest_price']['url'], quote['url'])
        self.store.set_setting('wine_price_allowed_hosts', ['first.example'])
        payload = {'actor_id': 2, 'report_id': uuid4().hex, 'lot_id': self.lot['id']}
        with patch('thesaurus.wine_tools.fetch_price') as fetch:
            wine_jobs.run_job(self.store, Settings(root=self.root), {'kind': 'wine_price_refresh', 'id': 105}, payload)
        fetch.assert_not_called()  # Cached result also needs the final shop's current permission.
        self.assertEqual(self.store.one('SELECT count(*) n FROM wine_prices')['n'], 1)
        self.assertIn('магазин отключён', self.store.one('SELECT text FROM reports WHERE id=?', (payload['report_id'],))['text'])

    def test_review_is_private_and_expires(self):
        token = self.save()
        with self.assertRaises(ValueError):
            retail.candidates(self.store, 3, token)
        expired = self.save(expires='2000-01-01T00:00:00+00:00')
        with self.assertRaises(ValueError):
            retail.candidates(self.store, 2, expired)
        self.store.set_setting(key(2), [])
        with self.assertRaises(PermissionError):
            retail.candidates(self.store, 2, token)

    def test_confirm_adds_exact_link_and_one_idempotent_moscow_quote(self):
        self.enable()
        token = self.save()
        retail.confirm_candidate(self.store, 2, token, 0)
        retail.confirm_candidate(self.store, 2, token, 0)
        self.assertEqual(len(cellar.price_links(self.store, 2)), 1)
        self.assertEqual(self.store.one('SELECT count(*) n FROM wine_prices')['n'], 1)
        latest = cellar.get_lot(self.store, 2, self.lot['id'])['latest_price']
        self.assertEqual(latest['metadata']['city'], 'Москва')
        self.assertEqual(latest['metadata']['price_kind'], 'public')
        self.assertEqual(cellar.get_lot(self.store, 2, self.lot['id'])['quantity'], 2)

    def test_confirmation_revalidates_identity_city_price_and_current_configuration(self):
        self.enable()
        for change in ({'vintage': 2024}, {'volume_ml': 1500}, {'city': 'Санкт-Петербург'},
                       {'price_kind': 'club'}, {'availability': 'OutOfStock'}, {'amount': 0},
                       {'amount': float('nan')}, {'price_valid_until': '2000-01-01'},
                       {'as_of': '2000-01-01T00:00:00+00:00'}, {'winery': 'Other producer'}):
            # NaN cannot be persisted by Store; eligibility still rejects it directly.
            if change.get('amount') != change.get('amount'):
                self.assertFalse(retail.quote_eligible(self.quote(**change), self.lot, moscow=True)[0])
                continue
            token = self.save(self.quote(**change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                retail.confirm_candidate(self.store, 2, token, 0)
        token = self.save()
        retail.configure_shop(self.store, 1, 'decanter', False)
        with self.assertRaises(ValueError):
            retail.confirm_candidate(self.store, 2, token, 0)
        self.assertFalse(cellar.price_links(self.store, 2))

    def test_job_search_delivers_private_review_without_mutating_inventory(self):
        self.enable()
        token = uuid4().hex
        job = {'id': 101, 'kind': 'wine_price_search'}
        payload = {'actor_id': 2, 'owner_id': 2, 'lot_id': self.lot['id'], 'report_id': token}
        with patch.object(retail, '_source_search', return_value=([self.quote()], {'status': 'ok', 'reason': 'Найдено'})):
            wine_jobs.run_job(self.store, Settings(root=self.root), job, payload)
            wine_jobs.run_job(self.store, Settings(root=self.root), job, payload)
        result = retail.candidates(self.store, 2, token)
        self.assertEqual(len(result['candidates']), 1)
        report = self.store.one('SELECT * FROM reports WHERE id=?', (token,))
        self.assertIn('wc:offers:'+token, json.loads(report['quality'])['wine_actions'][0][0]['callback_data'])
        self.assertEqual([row['recipient'] for row in self.store.query('SELECT DISTINCT recipient FROM deliveries')], ['2'])
        self.assertEqual(self.store.one('SELECT count(*) n FROM wine_prices')['n'], 0)
