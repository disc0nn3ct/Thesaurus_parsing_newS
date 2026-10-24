"""Curated structural AMWine fixtures; no network and no imported customer data."""
import json
import unittest

from thesaurus.wine_retail_amwine import parse_amwine_product, parse_amwine_search


class AMWineTests(unittest.TestCase):
    url = 'https://amwine.ru/catalog/vino/inkerman_muskat_winemaker_s_selection/'
    name = 'Вино Inkerman Muskat Winemaker`s Selection 0.75 л'

    def fixture(self, *, price='749.99', visible='749,99 ₽', stock='Осталось 8 шт',
                address='Бакунинская ул., д.<br>23-41, г. Москва', vintage=None, extra='', **changes):
        product = {'@context': 'https://schema.org', '@type': 'Product', 'name': self.name, 'sku': '20240412',
                   'offers': {'@type': 'Offer', 'url': self.url, 'price': price,
                              'priceCurrency': 'RUB', 'availability': 'https://schema.org/InStock'}}
        product.update(changes)
        year = self.field('Год урожая', vintage) if vintage is not None else ''
        return ('<html><script type="application/ld+json">' + json.dumps(product) + '</script>'
                '<div class="address-picker"><span v-if="!isMounted" class="address-picker__text">' + address + '</span>'
                '<template v-else><span class="address-picker__text">{{ currentShopTitle }}</span></template></div>'
                '<h1 class="product-hero__title">' + self.name + '</h1>'
                '<div class="product-cartblock"><div class="ui-price-tag__value">' + visible + '</div>'
                '<p class="product-cartblock__wherehouse">' + stock + '</p></div>'
                '<div class="product-stats"><ul>' + self.field('Бренд', 'Inkerman')
                + self.field('Производитель', 'Инкерманский завод марочных вин') + self.field('Объем', '0.75 л')
                + year + '</ul></div><div class="reviews">В 2020 понравилось; завод основан в 1961 году.</div>'
                + extra + '</html>')

    def field(self, key, value):
        return ('<li class="stats-list__item"><span class="stats-list__item-title">' + key
                + '</span><span class="stats-list__item-value">' + str(value) + '</span></li>')

    def test_public_price_has_specific_moscow_shop_and_stock_provenance(self):
        quote = parse_amwine_product(self.fixture(), self.url)
        self.assertEqual(quote['status'], 'ok')
        self.assertEqual(quote['amount'], 749.99)
        self.assertEqual(quote['currency'], 'RUB')
        self.assertEqual(quote['availability'], 'InStock')
        self.assertEqual(quote['shop_id'], 'amwine')
        self.assertEqual(quote['city'], 'Москва')
        self.assertEqual(quote['price_kind'], 'public')
        self.assertEqual(quote['availability_scope'], 'selected_moscow_shop')
        self.assertEqual(quote['available_shops'], ['Бакунинская ул., д. 23-41, г. Москва'])
        self.assertEqual(quote['stock_count'], 8)
        self.assertEqual(quote['source'], self.url)
        self.assertTrue(quote['as_of'])
        self.assertEqual(quote['volume_ml'], 750)
        self.assertEqual(quote['winery'], 'Inkerman')
        self.assertIn('не перепродажи', quote['warnings'][0])

    def test_unknown_vintage_not_guessed_from_sku_reviews_or_founding_year(self):
        quote = parse_amwine_product(self.fixture(), self.url)
        self.assertIsNone(quote['vintage'])
        self.assertEqual(quote['sku'], '20240412')

    def test_explicit_harvest_year_or_nv_is_read_from_product_characteristic(self):
        for value, expected in (('2020', 2020), ('NV', 'NV'), ('Без года', 'NV')):
            with self.subTest(value=value):
                quote = parse_amwine_product(self.fixture(vintage=value), self.url)
                self.assertEqual(quote['status'], 'ok')
                self.assertEqual(quote['vintage'], expected)

    def test_jsonld_price_requires_matching_visible_ordinary_price(self):
        for visible in ('562,49 ₽', 'от 749,99 ₽', '749,99 ₽ по карте', '749,99 ₽ с бонусами', '0 ₽'):
            with self.subTest(visible=visible):
                self.assertEqual(parse_amwine_product(self.fixture(visible=visible), self.url)['status'], 'unavailable')
        # An unrelated bonus advert cannot replace the main ordinary price.
        quote = parse_amwine_product(self.fixture(extra='<div class="bonus-ad">562,49 ₽ с бонусами</div>'), self.url)
        self.assertEqual(quote['amount'], 749.99)

    def test_stock_requires_positive_published_quantity_at_the_selected_shop(self):
        for stock in ('Осталось 0 шт', 'Нет в наличии', 'Под заказ', 'Осталось -1 шт', 'Уточните наличие', ''):
            with self.subTest(stock=stock):
                self.assertEqual(parse_amwine_product(self.fixture(stock=stock), self.url)['status'], 'unavailable')

    def test_organization_moscow_address_cannot_override_selected_other_city(self):
        page = self.fixture(address='ул. Тестовая, 10, г. Санкт-Петербург', extra='<footer>Адрес компании: г. Москва</footer>')
        self.assertEqual(parse_amwine_product(page, self.url)['status'], 'unavailable')
        page = self.fixture().replace('v-if="!isMounted"', 'v-if="isMounted"')
        self.assertEqual(parse_amwine_product(page, self.url)['status'], 'unavailable')

    def test_missing_or_ambiguous_main_price_title_characteristics_fail_closed(self):
        pages = [self.fixture().replace('product-cartblock"', 'changed-cartblock"'),
                 self.fixture().replace('product-hero__title', 'changed-title'),
                 self.fixture().replace('product-stats', 'changed-stats'),
                 self.fixture(extra='<div class="product-cartblock"><div class="ui-price-tag__value">1 ₽</div></div>'),
                 self.fixture().replace('ui-price-tag__value', 'changed-price')]
        for page in pages:
            with self.subTest(page=page[-150:]):
                self.assertEqual(parse_amwine_product(page, self.url)['status'], 'unavailable')

    def test_offer_for_another_product_or_an_aggregate_is_not_a_quote(self):
        for changes in ({'url': self.url.replace('inkerman_', 'other_')},
                        {'offers': {'@type': 'AggregateOffer', 'lowPrice': 100, 'url': self.url}},
                        {'offers': {'@type': 'Offer', 'url': self.url, 'price': 749.99,
                                    'priceCurrency': 'USD', 'availability': 'https://schema.org/InStock'}},
                        {'offers': {'@type': 'Offer', 'url': self.url, 'price': 749.99,
                                    'priceCurrency': 'RUB', 'availability': 'https://schema.org/OutOfStock'}}):
            with self.subTest(changes=changes):
                self.assertEqual(parse_amwine_product(self.fixture(**changes), self.url)['status'], 'unavailable')
        page = self.fixture().replace(self.url, self.url.replace('inkerman_', 'other_'))
        self.assertEqual(parse_amwine_product(page, self.url)['status'], 'unavailable')

    def test_conflicting_vintage_or_volume_never_selects_cheaper_variant(self):
        page = self.fixture(vintage='2020').replace('0.75 л</span>', '1.5 л</span>')
        self.assertEqual(parse_amwine_product(page, self.url)['status'], 'unavailable')
        page = self.fixture(vintage='2020', name=self.name + ' 2021')
        self.assertEqual(parse_amwine_product(page, self.url)['status'], 'unavailable')
        page = self.fixture(vintage='2020').replace('</ul>', self.field('Год урожая', '2021') + '</ul>')
        self.assertEqual(parse_amwine_product(page, self.url)['status'], 'unavailable')

    def test_invalid_product_routes_and_unverified_search_are_rejected(self):
        for url in ('https://amwine.ru/catalog/vino/', 'https://amwine.ru/catalog/vino/filter/color-is-red/',
                    self.url + '?q=wine', self.url + '#section', self.url.replace('amwine.ru', 'evil.example'),
                    self.url.replace('https:', 'http:')):
            with self.subTest(url=url):
                self.assertEqual(parse_amwine_product(self.fixture(), url)['status'], 'unavailable')
        self.assertEqual(parse_amwine_search(self.fixture(), 'https://amwine.ru/search/?q=wine'), [])
