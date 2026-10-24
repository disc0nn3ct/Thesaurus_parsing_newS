"""Public suggestions never silently identify a receipt bottle or create stock."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from thesaurus import wine_enrichment as enrichment
from thesaurus import wine_retail as retail
from thesaurus.feature_access import key
from thesaurus.store import Store

CITY = '<a class="head-menu__link--city"><span class="head-menu__link-border">Москва</span></a>'
URL = 'https://decanter.ru/product/chateau-test-merlot-2020-id12345'
SEARCH = 'https://decanter.ru/search/?q=test'


def prop(name, value):
    return ('<li itemprop="additionalProperty" itemscope itemtype="https://schema.org/PropertyValue">'
            f'<span itemprop="name">{name}</span><span itemprop="value">{value}</span></li>')


def product(*, name='Вино Chateau Test Merlot 2020 0.75 л', producer='Chateau Test', brand='Chateau Test',
            grapes='Мерло', country='Чили', region='Центральная Долина', volume='0.75', year='2020', identifier=12345, extra=''):
    return (CITY + f'<div class="prod-page" itemscope itemtype="https://schema.org/Product" data-prod-id="{identifier}">'
            f'<h1 itemprop="name">{name}</h1><meta itemprop="brand" content="{brand}">'
            f'<meta itemprop="manufacturer" content="{producer}">'
            '<div itemprop="description">Описание <b>Фруктовый аромат.</b><script>PRIVATE SCRIPT</script></div>'
            + prop('Страна', country) + prop('Регион', region) + prop('Виноград', grapes)
            + (prop('Объем (л)', volume) if volume else '') + (prop('Винтаж', year) if year else '')
            + extra + '</div>')


def search(rows):
    cards = []
    for index, (url, title) in enumerate(rows, 1):
        cards.append('<div class="prod__item" data-type="product_card" itemprop="itemListElement" '
                     f'itemscope itemtype="https://schema.org/ListItem"><meta itemprop="position" content="{index}">'
                     '<div itemprop="item" itemscope itemtype="https://schema.org/Product">'
                     f'<meta itemprop="name" content="{title}"></div>'
                     f'<a class="prod__link" itemprop="url" href="{url}"></a></div>')
    return CITY + '<div class="prod-list"><div id="CATALOG_CONTAINER">' + ''.join(cards) + '</div></div>'


EMPTY = CITY + '<div class="prod-list"><div class="not-found not-found--search">Ничего не найдено</div></div>'


class WineMetadataTests(unittest.TestCase):
    def test_main_wine_metadata_does_not_require_price_or_stock(self):
        row = enrichment.parse_decanter_metadata(product(), URL)
        self.assertEqual((row['name'], row['winery']), ('Chateau Test Merlot', 'Chateau Test'))
        self.assertEqual((row['volume_ml'], row['published_vintage'], row['vintage']), (750, 2020, None))
        self.assertEqual((row['country'], row['region'], row['grapes']), ('Чили', 'Центральная Долина', ['Мерло']))
        self.assertNotIn('PRIVATE SCRIPT', row['description'])
        self.assertNotIn('<', row['description'])
        self.assertLessEqual(len(row['description']), 600)
        self.assertNotIn('amount', row)
        self.assertEqual(row['id'], enrichment.parse_decanter_metadata(product(), URL)['id'])

    def test_property_links_use_visible_values_not_category_urls(self):
        html = product(country='<a href="/wine-chile">Чили</a>', region='<a href="/wine/central-valley">Центральная Долина</a>')
        row = enrichment.parse_decanter_metadata(html, URL)
        self.assertEqual(row['country'], 'Чили')
        self.assertEqual(row['region'], 'Центральная Долина')

    def test_manufacturer_organization_reads_name_without_addresses_or_contacts(self):
        manufacturer = ('<div itemprop="manufacturer" itemscope itemtype="https://schema.org/Organization">'
                        '<span itemprop="name">Chateau Test</span><span itemprop="address">PUBLIC ADDRESS</span>'
                        '<span itemprop="email">contact@example.com</span></div>')
        row = enrichment.parse_decanter_metadata(product(producer='', extra=manufacturer), URL)
        self.assertEqual(row['winery'], 'Chateau Test')
        self.assertNotIn('PUBLIC ADDRESS', str(row))

    def test_nested_recommendations_do_not_supply_attributes(self):
        nested = ('<div itemscope itemtype="https://schema.org/Product"><meta itemprop="manufacturer" content="WRONG">'
                  + prop('Страна', 'WRONG COUNTRY') + prop('Виноград', 'Шардоне') + '</div>')
        row = enrichment.parse_decanter_metadata(product(extra=nested), URL)
        self.assertEqual(row['winery'], 'Chateau Test')
        self.assertEqual(row['country'], 'Чили')
        self.assertEqual(row['grapes'], ['Мерло'])

    def test_producer_from_title_is_explicitly_marked_as_suggestion(self):
        row = enrichment.parse_decanter_metadata(product(name='Вино Concha y Toro Casillero del Diablo Merlot 0.75 л',
                 producer='', brand='Casillero del Diablo', year=''), URL)
        self.assertEqual(row['winery'], 'Concha y Toro')
        self.assertIn('префиксу названия', str(row['match_notes']))
        fields = enrichment._inputs({'name': 'Вино КазильДьябло Мерло кр.сух.0.75'})
        candidate = enrichment._match(row, fields)
        self.assertEqual(candidate['match_status'], 'possible')
        self.assertIsNone(candidate['vintage'])

    def test_contradictory_volume_vintage_or_product_structure_is_rejected(self):
        bad = [product(volume='1.5'), product(year='2021'), product(identifier=54321),
               product(extra=prop('Страна', 'Франция')), product().removesuffix('</div>'),
               product(name='Виски Chateau Test Merlot 2020 0.75 л')]
        for html in bad:
            with self.subTest(html=html[:100]), self.assertRaises(ValueError):
                enrichment.parse_decanter_metadata(html, URL)
        for url in (URL + '?city=moscow', URL.replace('decanter.ru', 'spb.decanter.ru'), URL.replace('https:', 'http:')):
            with self.assertRaises(ValueError):
                enrichment.parse_decanter_metadata(product(), url)

    def test_archived_internal_id_alias_requires_one_exact_public_canonical(self):
        exact = '<link rel="canonical" href="' + URL + '">'
        row = enrichment.parse_decanter_metadata(exact + product(identifier=37961), URL)
        self.assertEqual(row['source_url'], URL)
        for head in (exact + exact, exact.replace('id12345', 'id99999'), ''):
            with self.assertRaises(ValueError):
                enrichment.parse_decanter_metadata(head + product(identifier=37961), URL)

    def test_explicit_title_grape_and_property_grape_conflict_is_rejected(self):
        with self.assertRaises(ValueError):
            enrichment.parse_decanter_metadata(product(grapes='Шардоне'), URL)

    def test_explicit_millilitre_size_in_title_does_not_create_vintage(self):
        row = enrichment.parse_decanter_metadata(product(name='Вино Chateau Test Merlot 2020 мл', volume='2.02', year=''), URL)
        self.assertIsNone(row['published_vintage'])
        self.assertEqual(row['volume_ml'], 2020)


class WineEnrichmentTests(unittest.TestCase):
    def test_actual_receipt_spelling_and_abbreviations_have_clean_queries(self):
        names = {
            'Вино КазилДьябло Мерло кр.сух.0.75': 'casillero del diablo merlot',
            'Вино Барилоче Мальбек кр п/сух 0,75': 'bariloche malbec',
            'Вино Сан Валентин Гарнача кр.сух.0,75': 'san valentin garnacha',
            'Вино Фронтера Шардоне бел. 0,75': 'frontera chardonnay',
            'Вино Гран Кастильо Мальбек кр.сух.0,75': 'gran castillo malbec',
        }
        for name, query in names.items():
            with self.subTest(name=name):
                fields = enrichment._inputs({'name': name})
                self.assertEqual(fields['core'], query)
                self.assertEqual(fields['volume_ml'], 750)

    def test_varietal_only_vivino_name_uses_winery_for_search_and_identity(self):
        html = product(name='Вино Manu Sauvignon Blanc 2020 0.75 л', producer='Manu', brand='Manu', grapes='Совиньон Блан')
        with patch.object(enrichment, '_download', return_value=(html, URL)) as fetch:
            result = self.lookup({'name': 'Sauvignon Blanc', 'winery': 'Manu'})
        self.assertEqual(parse_qs(urlsplit(fetch.call_args.args[0]).query)['q'][0], 'sauvignon blanc manu')
        self.assertEqual(result['candidates'][0]['winery'], 'Manu')

    def test_owned_vivino_fallback_does_not_request_http_and_does_not_leak_between_users(self):
        retail.configure_shop(self.store, 1, 'decanter', False)
        rows = [
            {'name': 'Gran Castillo Monte de Argentina Malbec', 'winery': 'Finca La Celia', 'country': 'Argentina', 'region': 'Mendoza',
             'vintage': '', 'link': 'https://www.vivino.com/wines/179790348'},
            {'name': 'San Valentín Garnacha', 'winery': 'Familia Torres', 'country': 'Spain', 'region': 'Catalunya',
             'vintage': '2019', 'link': 'https://www.vivino.com/wines/159371593'},
            {'name': 'Merlot (Reserva)', 'winery': 'Casillero del Diablo', 'country': 'Chile', 'region': 'Central Valley',
             'vintage': '2022', 'link': 'https://www.vivino.com/wines/170471409'},
        ]
        self.store.set_setting('project:vivino:2:dataset', {'generation': 'owned', 'rows': rows, 'imported_at': '2026-10-11'})
        for name in ('Вино Гран Кастильо Мальбек кр.сух.0,75', 'Вино Сан Валентин Гарнача кр.сух.0,75', 'Вино КазилДьябло Мерло кр.сух.0.75'):
            with self.subTest(name=name), patch.object(enrichment, '_download') as download:
                result = self.lookup({'name': name})
                self.assertEqual(result['status'], 'candidates')
                self.assertEqual(len(result['candidates']), 1)
                candidate = result['candidates'][0]
                self.assertTrue(candidate['country'])
                self.assertTrue(candidate['winery'])
                self.assertIsNone(candidate['vintage'])
                self.assertTrue(enrichment.candidate_allowed(self.store, 2, candidate))
                self.assertFalse(enrichment.candidate_allowed(self.store, 3, candidate))
                self.assertEqual(enrichment.lookup(self.store, 3, {'name': name})['status'], 'disabled')
                download.assert_not_called()
        self.store.set_setting('project:vivino:2:dataset', {'generation': 'replaced', 'rows': []})
        self.assertFalse(enrichment.candidate_allowed(self.store, 2, candidate))

    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / '.test-tmp'
        parent.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / 'state.sqlite3')
        self.store.add_user(1, 'admin')
        self.store.add_user(2, subscriptions=['life', 'vivino'])
        self.store.add_user(3, subscriptions=['life'])
        self.store.add_user(4)
        self.store.set_setting('wine_price_allowed_hosts', [])
        retail.configure_shop(self.store, 1, 'decanter', True)
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No real network')))
        self.fields = {'name': 'Chateau Test Merlot', 'winery': 'Chateau Test', 'volume_ml': 750}

    def download(self, rows=None, cards=None):
        rows = rows if rows is not None else [(URL, 'Вино Chateau Test Merlot 2020 0.75 л')]
        cards = cards if cards is not None else {URL: product()}
        def fetch(url, **kwargs):
            if kwargs.get('search_shop'):
                return search(rows), url
            if isinstance(cards[url], Exception):
                raise cards[url]
            return cards[url], url
        return fetch

    def lookup(self, fields=None, **kwargs):
        return enrichment.lookup(self.store, 2, fields or self.fields, **kwargs)

    def test_unique_exact_candidate_is_read_only_and_keeps_unknown_vintage_unknown(self):
        with patch.object(enrichment, '_download', side_effect=self.download()) as fetch:
            result = self.lookup()
        self.assertEqual(result['status'], 'candidates')
        self.assertEqual(len(result['candidates']), 1)
        row = result['candidates'][0]
        self.assertEqual(row['match_status'], 'exact')
        self.assertIsNone(row['vintage'])
        self.assertEqual(row['published_vintage'], 2020)
        self.assertEqual(row['source_url'], URL)
        self.assertIn('не переносится', str(row['match_notes']))
        self.assertEqual(fetch.call_count, 2)
        for table in ('wine_lots', 'wine_prices', 'life_receipt_drafts', 'life_ledger_entries', 'life_receipt_imports'):
            self.assertFalse(self.store.query('SELECT * FROM ' + table))

    def test_receipt_alias_cleanup_is_a_search_hint_not_hardcoded_wine_data(self):
        html = product(name='Вино Concha y Toro Casillero del Diablo Merlot 2023 0.75 л',
                       producer='Concha y Toro', brand='Casillero del Diablo', year='2023')
        with patch.object(enrichment, '_download', return_value=(html, URL)) as fetch:
            result = self.lookup({'name': 'Вино КазильДьябло Мерло кр.сух.0.75'})
        query = parse_qs(urlsplit(fetch.call_args.args[0]).query)['q'][0]
        self.assertEqual(query, 'casillero del diablo merlot')
        row = result['candidates'][0]
        self.assertEqual(row['winery'], 'Concha y Toro')
        self.assertIsNone(row['vintage'])
        self.assertEqual(row['volume_ml'], 750)
        self.assertEqual(row['match_status'], 'exact')
        self.assertEqual(row['volume_ml'], 750)

    def test_multiple_vintages_remain_distinct_choices_without_invented_bottle_year(self):
        second = URL.replace('2020-id12345', '2021-id12346')
        rows = [(URL, 'Вино Chateau Test Merlot 2020 0.75 л'), (second, 'Вино Chateau Test Merlot 2021 0.75 л')]
        cards = {URL: product(), second: product(name='Вино Chateau Test Merlot 2021 0.75 л', year='2021', identifier=12346)}
        with patch.object(enrichment, '_download', side_effect=self.download(rows, cards)):
            result = self.lookup()
        self.assertEqual(len(result['candidates']), 2)
        self.assertEqual({row['published_vintage'] for row in result['candidates']}, {2020, 2021})
        self.assertTrue(all(row['vintage'] is None for row in result['candidates']))
        self.assertEqual(len({row['id'] for row in result['candidates']}), 2)

    def test_wrong_grape_volume_vintage_or_producer_does_not_match(self):
        alternatives = [product(name='Вино Chateau Test Chardonnay 2020 0.75 л', grapes='Шардоне'),
                        product(name='Вино Chateau Test Merlot 2020 1.5 л', volume='1.5'),
                        product(producer='Wrong Producer')]
        for html in alternatives:
            with patch.object(enrichment, '_download', side_effect=self.download(cards={URL: html})):
                self.assertEqual(self.lookup(force=True)['status'], 'empty')
        with patch.object(enrichment, '_download', side_effect=self.download()):
            self.assertEqual(self.lookup({**self.fields, 'vintage': 2021}, force=True)['status'], 'empty')

    def test_explicit_matching_input_year_can_be_confirmed(self):
        with patch.object(enrichment, '_download', side_effect=self.download()):
            row = self.lookup({**self.fields, 'vintage': 2020})['candidates'][0]
        self.assertEqual(row['vintage'], 2020)

    def test_other_series_is_possible_and_unrelated_brands_are_excluded(self):
        for name, expected in [('Вино Chateau Test Merlot Reserva 2020 0.75 л', 'possible'),
                               ('Вино Unrelated Merlot 2020 0.75 л', None)]:
            with patch.object(enrichment, '_download', side_effect=self.download(cards={URL: product(name=name)})):
                result = self.lookup(force=True)
            self.assertEqual(result['candidates'][0]['match_status'] if result['candidates'] else None, expected)

    def test_fresh_cache_avoids_requests_force_refreshes_and_other_actor_has_own_scope(self):
        with patch.object(enrichment, '_download', side_effect=self.download()) as fetch:
            first = self.lookup()
            second = self.lookup()
            self.assertEqual(fetch.call_count, 2)
            self.assertTrue(second['cached'])
            self.assertEqual(second['candidates'][0]['id'], first['candidates'][0]['id'])
            self.lookup(force=True)
            self.assertEqual(fetch.call_count, 4)
            enrichment.lookup(self.store, 3, self.fields)
            self.assertEqual(fetch.call_count, 6)
        cache = self.store.query("SELECT key,value FROM settings WHERE key LIKE 'wine_enrichment_cache:%'")
        self.assertEqual(len(cache), 2)
        self.assertNotIn('receipt_id', str(cache))

    def test_expired_cache_is_refetched(self):
        with patch.object(enrichment, '_download', side_effect=self.download()) as fetch:
            self.lookup()
            row = self.store.one("SELECT key FROM settings WHERE key LIKE 'wine_enrichment_cache:%'")
            cache = self.store.get_setting(row['key'])
            cache['expires'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            self.store.set_setting(row['key'], cache)
            self.assertFalse(self.lookup()['cached'])
            self.assertEqual(fetch.call_count, 4)

    def test_disabling_source_blocks_cached_results_without_requests_or_new_enablement(self):
        with patch.object(enrichment, '_download', side_effect=self.download()) as fetch:
            self.lookup()
            retail.configure_shop(self.store, 1, 'decanter', False)
            result = self.lookup()
        self.assertEqual(result['status'], 'disabled')
        self.assertFalse(result['candidates'])
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(self.store.get_setting('wine_retail_shops'), [])

    def test_dry_run_and_unsupported_shops_do_not_read_cache_or_query_network(self):
        with patch.object(enrichment, '_download') as fetch:
            self.assertEqual(self.lookup(dry_run=True)['status'], 'dry_run')
            retail.configure_shop(self.store, 1, 'decanter', False)
            retail.configure_shop(self.store, 1, 'winelab', True)
            self.assertEqual(self.lookup()['status'], 'disabled')
        fetch.assert_not_called()
        self.assertFalse(self.store.query("SELECT key FROM settings WHERE key LIKE 'wine_enrichment_cache:%'"))

    def test_disabled_and_unassigned_actors_cannot_read_even_cached_public_results(self):
        with patch.object(enrichment, '_download', side_effect=self.download()) as fetch:
            self.lookup()
            self.store.set_setting(key(2), [])
            with self.assertRaises(PermissionError):
                self.lookup()
            with self.assertRaises(PermissionError):
                enrichment.lookup(self.store, 4, self.fields)
        self.assertEqual(fetch.call_count, 2)

    def test_revocation_after_search_aborts_before_card_or_cache_write(self):
        def revoke(url, **kwargs):
            self.store.set_setting(key(2), [])
            return search([(URL, 'Вино Chateau Test Merlot 2020 0.75 л')]), url
        with patch.object(enrichment, '_download', side_effect=revoke) as fetch, self.assertRaises(PermissionError):
            self.lookup()
        self.assertEqual(fetch.call_count, 1)
        self.assertFalse(self.store.query("SELECT key FROM settings WHERE key LIKE 'wine_enrichment_cache:%'"))

    def test_supplied_guard_errors_are_not_swallowed_as_source_errors(self):
        calls = 0
        def guard():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise ValueError('Receipt changed during lookup')
        with patch.object(enrichment, '_download', side_effect=self.download()) as fetch, self.assertRaisesRegex(ValueError, 'Receipt changed'):
            self.lookup(guard=guard)
        self.assertEqual(fetch.call_count, 1)
        self.assertFalse(self.store.query("SELECT key FROM settings WHERE key LIKE 'wine_enrichment_cache:%'"))

    def test_cached_result_guard_error_aborts_without_fresh_network(self):
        with patch.object(enrichment, '_download', side_effect=self.download()):
            self.lookup()
        calls = 0
        def guard():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError('Receipt changed during cached lookup')
        with patch.object(enrichment, '_download') as fetch, self.assertRaisesRegex(ValueError, 'Receipt changed'):
            self.lookup(guard=guard)
        fetch.assert_not_called()

    def test_bare_receipt_bottle_volume_does_not_match_magnum(self):
        html = product(name='Вино Concha y Toro Casillero del Diablo Merlot 2023 1.5 л',
                       producer='Concha y Toro', brand='Casillero del Diablo', year='2023', volume='1.5')
        with patch.object(enrichment, '_download', return_value=(html, URL)):
            result = self.lookup({'name': 'Вино КазильДьябло Мерло кр.сух.0.75'})
        self.assertEqual(result['status'], 'empty')

    def test_unique_search_redirect_reuses_downloaded_card(self):
        with patch.object(enrichment, '_download', return_value=(product(), URL)) as fetch:
            result = self.lookup()
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(result['candidates'][0]['source_url'], URL)
        self.assertEqual(fetch.call_args.kwargs['search_shop'], 'decanter')

    def test_source_errors_are_private_and_partial_card_failure_keeps_other_results(self):
        second = URL.replace('12345', '12346')
        rows = [(URL, 'Вино Chateau Test Merlot 2020 0.75 л'), (second, 'Вино Chateau Test Merlot 2020 0.75 л')]
        with patch.object(enrichment, '_download', side_effect=self.download(rows, {URL: product(), second: OSError('PRIVATE SOURCE DIAGNOSTICS')})):
            result = self.lookup()
        self.assertEqual(result['status'], 'candidates')
        self.assertEqual(len(result['candidates']), 1)
        self.assertTrue(result['warnings'])
        self.assertNotIn('PRIVATE SOURCE', str(result))
        with patch.object(enrichment, '_download', side_effect=OSError('PRIVATE SOURCE DIAGNOSTICS')):
            result = self.lookup(force=True)
        self.assertEqual(result['status'], 'unavailable')
        self.assertNotIn('PRIVATE SOURCE', str(result))

    def test_all_changed_or_unavailable_cards_are_source_failure_not_empty_match(self):
        with patch.object(enrichment, '_download', side_effect=self.download(cards={URL: 'captcha'})):
            self.assertEqual(self.lookup()['status'], 'unavailable')

    def test_verified_empty_search_is_not_reported_as_transport_failure(self):
        with patch.object(enrichment, '_download', return_value=(EMPTY, SEARCH)):
            self.assertEqual(self.lookup()['status'], 'empty')

    def test_secret_like_or_invalid_input_is_rejected_before_requests(self):
        bad = [{'name': 'email@example.com'}, {'name': 'Wine https://private.example/token'},
               {'name': '<script>Wine</script>'}, {'name': 'ФД 1234567890123456'}, {'name': 'ФД 12345'},
               {'name': 'Chateau Test 2020', 'vintage': 2021},
               {'name': 'Chateau Test 750 ml', 'volume_ml': 1500}, {'name': 'Wine', 'volume_ml': True},
               {'name': 'Chateau Test', 'volume_ml': 0}, {'name': 'Chateau Test', 'volume_ml': False}]
        with patch.object(enrichment, '_download') as fetch:
            for fields in bad:
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    self.lookup(fields)
        fetch.assert_not_called()

    def test_unusual_explicit_size_is_not_a_harvest_year(self):
        fields = enrichment._inputs({'name': 'Wine Chateau Test 2020 ml'})
        self.assertEqual(fields['volume_ml'], 2020)
        self.assertIsNone(fields['vintage'])


if __name__ == '__main__':
    unittest.main()
