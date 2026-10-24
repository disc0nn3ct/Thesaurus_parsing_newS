"""Curated synthetic Decanter structures; no network, cookies or real purchases."""
import unittest
from datetime import datetime, timedelta, timezone

from thesaurus.wine_retail_sources import (
    decanter_search_status,
    parse_decanter_product,
    parse_decanter_search,
)

URL = "https://decanter.ru/product/chateau-test-2020-id12345"
SEARCH = "https://decanter.ru/search/?q=Chateau+Test"
CITY = '<a class="head-menu__link head-menu__link--city"><span class="head-menu__link-border">Москва</span></a>'
STOCK = ('<div class="prod-card__available-item"><div class="prod-card__available-street">'
         '<a data-shop="presnya"><span class="store__head">Пресненский Вал, 44</span></a></div>'
         '<div class="prod-card__available-text">в наличии</div></div>')


def product(*, name="Вино Chateau Test 2020 0.75 л", amount="3670", currency="RUB",
            availability="InStock", expiry=None, stock=STOCK, city=CITY, extra="", offer_url=URL):
    expiry = expiry if expiry is not None else (datetime.now(timezone.utc).date() + timedelta(days=30)).isoformat()
    return (f'{city}<div class="prod-page" itemscope itemtype="https://schema.org/Product" data-prod-id="12345">'
            f'<h1 itemprop="name" data-prod-name>{name}</h1><meta itemprop="brand" content="Chateau Test">'
            '<meta itemprop="sku" content="09-12345">'
            '<div class="prod-card__price" itemprop="offers" itemscope itemtype="http://schema.org/Offer">'
            f'<link itemprop="url" href="{offer_url}"><meta itemprop="price" content="{amount}">'
            f'<meta itemprop="priceCurrency" content="{currency}">'
            f'<link itemprop="availability" href="https://schema.org/{availability}">'
            f'<meta itemprop="priceValidUntil" content="{expiry}"><div class="prod-card__price-label">Цена</div>'
            f'{extra}</div>{stock}</div>')


def search_item(index, name="Вино Chateau Test 2020 0,75 л", path=None):
    path = path or f"/product/chateau-test-2020-id{12345 + index}"
    return ('<div class="prod__item" data-type="product_card" itemprop="itemListElement" itemscope '
            f'itemtype="https://schema.org/ListItem"><meta itemprop="position" content="{index}">'
            '<div itemprop="item" itemscope itemtype="https://schema.org/Product">'
            f'<meta itemprop="name" content="{name}"><meta itemprop="brand" content="Chateau Test">'
            '<div itemprop="offers" itemscope itemtype="https://schema.org/Offer">'
            '<meta itemprop="price" content="1"><meta itemprop="priceCurrency" content="RUB"></div></div>'
            f'<a class="prod__link" itemprop="url" href="{path}"></a>'
            f'<div class="prod__text"><a data-prod-name href="{path}">{name}</a></div></div>')


def search_page(items, city=CITY):
    return city + '<div class="prod-list"><div id="CATALOG_CONTAINER">' + items + '</div></div>'


class DecanterProductTests(unittest.TestCase):
    def test_main_offer_with_moscow_stock_and_identity(self):
        row = parse_decanter_product(product(), URL)
        self.assertEqual(row["status"], "ok")
        self.assertEqual((row["amount"], row["currency"], row["vintage"], row["volume_ml"]), (3670, "RUB", 2020, 750))
        self.assertEqual(row["winery"], "Chateau Test")
        self.assertEqual(row["sku"], "09-12345")
        self.assertEqual((row["shop_id"], row["city"], row["price_kind"]), ("decanter", "Москва", "public"))
        self.assertEqual(row["available_shops"], ["Пресненский Вал, 44"])
        self.assertEqual(row["availability_scope"], "moscow_branches")

    def test_nested_strikethrough_price_does_not_replace_current_price(self):
        old = ('<div class="prod-card__price-old" itemprop="priceSpecification" itemscope '
               'itemtype="https://schema.org/UnitPriceSpecification"><meta itemprop="price" content="8888">'
               '<meta itemprop="priceCurrency" content="RUB">'
               '<meta itemprop="priceType" content="https://schema.org/StrikethroughPrice">8 888</div>')
        self.assertEqual(parse_decanter_product(product(extra=old), URL)["amount"], 3670)

    def test_absent_or_unknown_vintage_is_not_guessed(self):
        row = parse_decanter_product(product(name="Вино Chateau Test 0,75 л"), URL)
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["vintage"])
        row = parse_decanter_product(product(name="Вино Chateau Test NV 750 ml"), URL)
        self.assertEqual(row["vintage"], "NV")

    def test_recommendation_offer_cannot_supply_main_price(self):
        recommendation = ('<div itemscope itemtype="https://schema.org/Product"><meta itemprop="name" content="Other">'
                          '<div itemscope itemtype="https://schema.org/Offer"><meta itemprop="price" content="22">'
                          '<meta itemprop="priceCurrency" content="RUB"></div></div>')
        self.assertEqual(parse_decanter_product(product(extra=recommendation), URL)["amount"], 3670)
        self.assertEqual(parse_decanter_product(recommendation + CITY, URL)["status"], "unavailable")

    def test_other_city_or_no_explicit_selected_city_is_rejected(self):
        for city in ("", CITY.replace("Москва", "Санкт-Петербург"), CITY + CITY.replace("Москва", "Иваново"),
                     '<script type="application/ld+json">{"addressLocality":"Москва"}</script>'):
            with self.subTest(city=city):
                self.assertEqual(parse_decanter_product(product(city=city), URL)["status"], "unavailable")

    def test_root_host_only_and_exact_product_identity(self):
        for url in (URL.replace("decanter.ru", "spb.decanter.ru"), URL + "/reviews", URL + "?city=moscow",
                    URL.replace("https:", "http:"), URL.replace("decanter.ru", "decanter.ru.evil.example"),
                    URL.replace("decanter.ru", "user@decanter.ru"), URL + "#prices"):
            with self.subTest(url=url):
                self.assertEqual(parse_decanter_product(product(), url)["status"], "unavailable")
        self.assertEqual(parse_decanter_product(product().replace('data-prod-id="12345"', 'data-prod-id="54321"'), URL)["status"], "unavailable")
        self.assertEqual(parse_decanter_product(product(offer_url=URL.replace("id12345", "id99999")), URL)["status"], "unavailable")

    def test_out_of_stock_under_order_and_missing_branch_are_not_prices(self):
        for stock in ("", STOCK.replace("в наличии", "под заказ"), STOCK.replace("в наличии", "нет в наличии"),
                      STOCK.replace("в наличии", "в наличии в другом городе"),
                      STOCK.replace('data-shop="presnya"', 'data-shop="spb"'),
                      STOCK.replace("Пресненский Вал, 44", "Санкт-Петербург, Невский, 1")):
            with self.subTest(stock=stock):
                row = parse_decanter_product(product(stock=stock), URL)
                self.assertEqual(row["status"], "unavailable")
                self.assertNotIn("amount", row)
        self.assertEqual(parse_decanter_product(product(availability="OutOfStock"), URL)["status"], "unavailable")

    def test_invalid_price_currency_and_conditional_club_prices_rejected(self):
        for amount in ("0", "-10", "nan", "inf", "1e4", "100000001", "от 2500"):
            with self.subTest(amount=amount):
                self.assertEqual(parse_decanter_product(product(amount=amount), URL)["status"], "unavailable")
        for currency in ("USD", "RUR", "", "руб"):
            self.assertEqual(parse_decanter_product(product(currency=currency), URL)["status"], "unavailable")
        for extra in ("<span>Цена по карте</span>", "<span>Клубная цена</span>", "<span>от 3000 ₽</span>",
                      '<span class="prod-price-club">3000 ₽</span>'):
            self.assertEqual(parse_decanter_product(product(extra=extra), URL)["status"], "unavailable")

    def test_expired_invalid_or_ambiguous_price_is_not_accepted(self):
        for expiry in ((datetime.now(timezone.utc).date() - timedelta(days=3)).isoformat(), "tomorrow", "2026-15-32"):
            self.assertEqual(parse_decanter_product(product(expiry=expiry), URL)["status"], "unavailable")
        extra = '<meta itemprop="price" content="4000">'
        self.assertEqual(parse_decanter_product(product(extra=extra), URL)["status"], "unavailable")

    def test_drift_empty_oversized_and_truncated_page_fail_closed(self):
        for html in ("", "captcha", "x" * (2 * 1024 * 1024 + 1), product().replace('class="prod-page"', 'class="new-card"'),
                     product().removesuffix("</div>"), product() + product()):
            with self.subTest(length=len(html)):
                self.assertEqual(parse_decanter_product(html, URL)["status"], "unavailable")


class DecanterSearchTests(unittest.TestCase):
    def test_genuine_search_candidates_have_identity_but_no_snippet_price(self):
        result = parse_decanter_search(search_page(search_item(1)), SEARCH)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["url"], "https://decanter.ru/product/chateau-test-2020-id12346")
        self.assertEqual((result[0]["vintage"], result[0]["volume_ml"]), (2020, 750))
        self.assertNotIn("price", result[0])
        self.assertNotIn("amount", result[0])

    def test_no_results_recommendations_are_not_search_results(self):
        recommendation = '<div class="prod-rec__info"><a data-prod-name href="/product/test-id12345">Test 2020 750 ml</a></div>'
        self.assertEqual(parse_decanter_search(CITY + recommendation, SEARCH), [])
        self.assertEqual(parse_decanter_search(CITY + product(), SEARCH), [])
        self.assertEqual(parse_decanter_search(search_page(search_item(1)).replace('id="CATALOG_CONTAINER"', 'id="recommendations"'), SEARCH), [])

    def test_ambiguous_product_ownership_in_one_result_is_rejected(self):
        other = '<div itemscope itemtype="https://schema.org/Product"><meta itemprop="name" content="Other 2020 750 ml"></div>'
        html = search_item(1).replace('<div class="prod__text">', other + '<div class="prod__text">')
        self.assertEqual(parse_decanter_search(search_page(html), SEARCH), [])

    def test_result_limit_deduplication_and_foreign_links(self):
        html = search_item(1) * 2 + search_item(2, path="https://spb.decanter.ru/product/test-id99999")
        html += search_item(3, path="/product/test-id11111/reviews")
        html += "".join(search_item(index) for index in range(4, 14))
        result = parse_decanter_search(search_page(html), SEARCH)
        self.assertEqual(len(result), 6)
        self.assertEqual(len({row["url"] for row in result}), 6)
        self.assertTrue(all(row["url"].startswith("https://decanter.ru/product/") for row in result))

    def test_invalid_search_context_and_structure_return_no_guesses(self):
        for url in (URL, SEARCH.replace("decanter.ru", "spb.decanter.ru"), SEARCH + "&city=spb", "https://decanter.ru/search/"):
            self.assertEqual(parse_decanter_search(search_page(search_item(1)), url), [])
        for html in ("", CITY, CITY.replace("Москва", "Иваново") + search_item(1),
                     CITY + search_item(1).replace('itemprop="position"', 'itemprop="new-position"')):
            self.assertEqual(parse_decanter_search(html, SEARCH), [])

    def test_status_distinguishes_results_genuine_empty_and_changed_markup(self):
        result = decanter_search_status(search_page(search_item(1)), SEARCH)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["candidates"]), 1)
        empty = (CITY + '<div class="prod-list"><div class="not-found not-found--search">'
                 '<div class="not-found__box"><h2>Ничего не найдено</h2>'
                 '<div class="not-found__text">Проверьте правильность ввода.</div></div></div></div>')
        recommendation = '<div class="prod-rec__info"><a href="/product/test-id12345" data-prod-name>Other</a></div>'
        result = decanter_search_status(empty + recommendation, SEARCH)
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["candidates"], [])
        self.assertEqual(parse_decanter_search(empty, SEARCH), [])
        for html in (CITY, search_page(""), empty.replace("Ничего не найдено", "Ошибка"),
                     empty.replace('class="prod-list"', 'class="recommendations"'),
                     empty.removesuffix("</div>"), search_page(search_item(1)).removesuffix("</div></div>"),
                     search_page(search_item(1)).replace('itemprop="name"', 'itemprop="new-name"')):
            with self.subTest(html=html[:150]):
                result = decanter_search_status(html, SEARCH)
                self.assertEqual(result["status"], "unavailable")
                self.assertEqual(result["candidates"], [])

    def test_status_rejects_challenge_foreign_city_and_unsupported_url(self):
        for html, url in ((search_page(search_item(1), city=CITY.replace("Москва", "Иваново")), SEARCH),
                          (search_page(search_item(1)), URL),
                          ('<title>Just a moment...</title>' + search_page(search_item(1)), SEARCH),
                          (search_page(search_item(1)) + '<form id="challenge-form"></form>', SEARCH)):
            result = decanter_search_status(html, url)
            self.assertEqual(result["status"], "unavailable")
            self.assertTrue(result["reason"])
            self.assertEqual(result["candidates"], [])


if __name__ == "__main__":
    unittest.main()
