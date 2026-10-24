"""Pure AMWine adapter for published prices at the explicitly selected Moscow shop.

No city cookies, private APIs, account/bonus prices or age-confirmation actions are
used. Unknown harvest years stay unknown. Search is not verified and is disabled.
"""
import json
import re
from urllib.parse import urljoin, urlsplit

from .wine_retail_sources import _document, _nodes, _visible
from .wine_tools import (
    _JSONLD,
    _kind,
    _normal,
    _number,
    _objects,
    _text,
    _vintage,
    _volume,
    parse_price,
)

_HOST = "amwine.ru"
_ROOT = "https://amwine.ru"
_PRODUCT_PATH = re.compile(r"/catalog/(?:vino|shampanskoe_i_igristye_vina)/[a-z0-9][a-z0-9_-]*/?\Z")
_SHOP = {"shop_id": "amwine", "shop_name": "Ароматный мир", "price_kind": "public"}
_SCOPE = "selected_moscow_shop"


def _product_url(url):
    if not isinstance(url, str) or len(url) > 2048:
        raise ValueError("Некорректная карточка Ароматного мира")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != _HOST or parsed.query or parsed.fragment
            or not _PRODUCT_PATH.fullmatch(parsed.path)):
        raise ValueError("Нужна точная публичная карточка вина Ароматного мира")
    return _ROOT + parsed.path.rstrip("/") + "/"


def _one(nodes, class_name):
    found = [node for node in nodes if node.has_class(class_name)]
    if len(found) != 1 or not found[0].closed:
        raise ValueError("Изменилась структура основной карточки")
    return found[0]


def _selected_shop(document):
    # Organization addresses and nearby-shop suggestions do not establish city.
    addresses = {_text(" ".join(_visible(node).split()), 300) for node in _nodes(document)
                 if node.has_class("address-picker__text") and node.attrs.get("v-if") == "!isMounted"
                 and node.closed}
    if len(addresses) != 1:
        raise ValueError("Выбранный магазин не опубликован однозначно")
    address = addresses.pop()
    if not re.search(r"\bг\.?\s*Москва\b", address, re.IGNORECASE) or "{{" in address:
        raise ValueError("Не подтверждён выбранный магазин в Москве")
    return address


def _product(html, url):
    parser = _JSONLD()
    parser.feed(html)
    products = []
    for raw in parser.scripts[:50]:
        try:
            products.extend(node for node in _objects(json.loads(raw)) if _kind(node, "Product"))
        except (ValueError, RecursionError):
            continue
    if len(products) != 1:
        raise ValueError("Основной товар JSON-LD неоднозначен")
    product = products[0]
    offers = product.get("offers")
    offers = offers if isinstance(offers, list) else [offers]
    if len(offers) != 1 or not isinstance(offers[0], dict):
        raise ValueError("Основное предложение неоднозначно")
    offer = offers[0]
    if not _kind(offer, "Offer") or _kind(offer, "AggregateOffer"):
        raise ValueError("Нужна точная цена предложения")
    # The public page currently puts its identity in offers.url, not Product.url.
    if not offer.get("url") or _product_url(urljoin(url, offer["url"])) != url:
        raise ValueError("Предложение относится к другой карточке")
    if product.get("url") and _product_url(urljoin(url, product["url"])) != url:
        raise ValueError("Товар относится к другой карточке")
    return product


def _characteristics(document):
    main = _one(_nodes(document), "product-stats")
    fields = {}
    for row in _nodes(main):
        if not row.has_class("stats-list__item"):
            continue
        nodes = list(_nodes(row))
        title = _normal(_visible(_one(nodes, "stats-list__item-title")))
        value = _text(" ".join(_visible(_one(nodes, "stats-list__item-value")).split()), 500)
        if title in fields and fields[title] != value:
            raise ValueError("Опубликованы конфликтующие характеристики")
        fields[title] = value
    return fields


def parse_amwine_product(html, url):
    """Require matching JSON-LD/visible RUB price, positive stock and Moscow address."""
    try:
        canonical = _product_url(url)
        document = _document(html)
        address = _selected_shop(document)
        _product(html, canonical)
        quote = parse_price(html, canonical)
        if quote.get("status") != "ok":
            raise ValueError("Нет однозначной опубликованной цены RUB")
        title = _one(_nodes(document), "product-hero__title")
        if title.tag != "h1" or _normal(_visible(title)) != _normal(quote["title"]):
            raise ValueError("Название основной карточки не совпало")
        cart = _one(_nodes(document), "product-cartblock")
        price = _text(" ".join(_visible(_one(_nodes(cart), "ui-price-tag__value")).split()), 200)
        # Only the current ordinary price: discounts conditional on points/cards fail closed.
        price_match = re.fullmatch(r"(\d[\d\s]*(?:[.,]\d{1,2})?)\s*₽", price)
        visible_amount = _number(re.sub(r"\s", "", price_match.group(1)).replace(",", ".")) if price_match else None
        if visible_amount is None or abs(visible_amount - quote["amount"]) > .005:
            raise ValueError("Текущая публичная цена не совпала с JSON-LD")
        stock = " ".join(_visible(_one(_nodes(cart), "product-cartblock__wherehouse")).split())
        stock_match = re.fullmatch(r"Осталось\s+(\d+)\s+шт\.?", stock, re.IGNORECASE)
        if not stock_match or not 0 < int(stock_match.group(1)) <= 1_000_000:
            raise ValueError("Не подтверждено наличие в выбранном магазине")
        fields = _characteristics(document)
        brand = fields.get("бренд", "")
        producer = fields.get("производитель", "")
        if brand:
            quote["winery"] = brand
        elif producer:
            quote["winery"] = producer
        volume = _volume(fields.get("объем", fields.get("объём", "")))
        if volume:
            if quote.get("volume_ml") and volume != quote["volume_ml"]:
                raise ValueError("Опубликованы конфликтующие объёмы бутылки")
            quote["volume_ml"] = volume
        vintage_fields = [fields[key] for key in ("год урожая", "урожай", "винтаж", "vintage") if key in fields]
        years = {_vintage(value) for value in vintage_fields}
        if None in years or len(years) > 1:
            raise ValueError("Год урожая не удалось однозначно прочитать")
        if years:
            vintage = years.pop()
            if quote.get("vintage") is not None and quote["vintage"] != vintage:
                raise ValueError("Опубликованы конфликтующие годы урожая")
            quote["vintage"] = vintage
        # Reviews, producer founding dates and SKU digits never supply vintage.
        return {**quote, **_SHOP, "city": "Москва", "availability_scope": _SCOPE,
                "available_shops": [address], "shop_address": address,
                "stock_count": int(stock_match.group(1)), "producer": producer, "brand": brand,
                "available": True, "source": canonical, "url": canonical,
                "method": "AMWine Product/Offer, current ordinary price and selected Moscow shop stock",
                "warnings": [("Публичная цена и остаток выбранного московского магазина на момент проверки; "
                              "перед покупкой уточните в магазине. Это цена замещения, не перепродажи.")]}
    except (ValueError, TypeError, KeyError, RecursionError):
        return {**_SHOP, "status": "unavailable", "source": url, "url": url,
                "warnings": [("Не подтверждены обычная цена RUB и остаток нужной карточки в выбранном магазине Москвы. "
                              "Возможны изменение страницы, неизвестное наличие или другой город.")]}


def parse_amwine_search(html, url):
    """The public search route is not verified; never treat recommendations as hits."""
    return []
