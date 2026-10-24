"""Pure adapters for verified public retail markup; these functions never fetch URLs.

Decanter's main product microdata and branch list are deliberately required. Search
snippets and recommendation cards are identities only, never price observations.
"""
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urljoin, urlsplit

from .wine_tools import MAX_PAGE_BYTES, _identity, _normal, _number, _text

_HOST = "decanter.ru"
_ROOT = "https://decanter.ru"
_PRODUCT_PATH = re.compile(r"/product/[a-z0-9]+(?:-[a-z0-9]+)*-id([1-9]\d*)/?\Z")
_VOID = frozenset(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"])
_SHOP = {"shop_id": "decanter", "shop_name": "Декантер", "city": "Москва", "price_kind": "public"}
_MOSCOW = timezone(timedelta(hours=3))
# Verified root-host Moscow stores. New branches require a reviewed adapter update.
_MOSCOW_SHOPS = {"presnya": "пресненский вал 44", "masterkova": "мастеркова 1",
                 "basmannaya": "новая басманная 31с1", "leninskiy": "ленинский пр т 66"}


@dataclass(eq=False)
class _Node:
    tag: str
    attrs: dict = field(default_factory=dict)
    parent: object = None
    children: list = field(default_factory=list)
    text: list = field(default_factory=list)
    closed: bool = False

    def has_class(self, name):
        return name in self.attrs.get("class", "").split()

    def is_kind(self, kind):
        return "itemscope" in self.attrs and any(
            value.rstrip("/").rsplit("/", 1)[-1] == kind
            for value in self.attrs.get("itemtype", "").split()
        )


class _Document(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("document", closed=True)
        self.stack = [self.root]
        self.count = 0

    def handle_starttag(self, tag, attrs):
        self.count += 1
        if self.count > 60000 or len(self.stack) > 256:
            raise ValueError("Слишком сложная структура карточки")
        node = _Node(tag, dict(attrs), self.stack[-1], closed=tag in _VOID)
        self.stack[-1].children.append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                self.stack[index].closed = True
                del self.stack[index:]
                return

    def handle_data(self, text):
        if self.stack[-1].tag not in {"script", "style"}:
            self.stack[-1].text.append(text)


def _nodes(node):
    yield node
    for child in node.children:
        yield from _nodes(child)


def _visible(node):
    if node.tag in {"script", "style"}:
        return ""
    return " ".join([*node.text, *(_visible(child) for child in node.children)])


def _enclosing(node, kind=None):
    parent = node.parent
    while parent is not None:
        if (parent.is_kind(kind) if kind else "itemscope" in parent.attrs):
            return parent
        parent = parent.parent
    return None


def _in_catalog(node):
    parent = node.parent
    while parent is not None:
        if parent.attrs.get("id") == "CATALOG_CONTAINER":
            return True
        parent = parent.parent
    return False


def _properties(scope, name):
    """Ignore properties owned by nested scopes, including old price specs."""
    result = []
    for node in _nodes(scope):
        if name in node.attrs.get("itemprop", "").split() and _enclosing(node) is scope:
            value = node.attrs.get("content", node.attrs.get("href", _visible(node)))
            value = _text(value, 1000)
            if value:
                result.append(value)
    return list(dict.fromkeys(result))


def _one(scope, name):
    values = _properties(scope, name)
    return values[0] if len(values) == 1 else None


def _document(html):
    if not isinstance(html, str) or not html.strip() or len(html.encode("utf-8")) > MAX_PAGE_BYTES:
        raise ValueError("Пустая или слишком большая страница")
    parser = _Document()
    parser.feed(html)
    parser.close()
    return parser.root


def _product_url(url):
    if not isinstance(url, str) or len(url) > 2000 or any(ord(c) < 33 for c in url):
        raise ValueError("Некорректная ссылка на товар")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != _HOST or parsed.username or parsed.password
            or parsed.query or parsed.fragment or not _PRODUCT_PATH.fullmatch(parsed.path)):
        raise ValueError("Нужна карточка decanter.ru для Москвы")
    return _ROOT + parsed.path.rstrip("/")


def _search_url(url):
    if not isinstance(url, str) or len(url) > 2000:
        return False
    parsed = urlsplit(url)
    try:
        pairs = parse_qsl(parsed.query, strict_parsing=True, max_num_fields=4)
    except ValueError:
        return False
    return (parsed.scheme == "https" and parsed.netloc == _HOST and parsed.path in {"/search", "/search/"}
            and not parsed.fragment and len(pairs) == 1 and pairs[0][0] == "q" and bool(pairs[0][1].strip()))


def _moscow(document):
    labels = []
    for node in _nodes(document):
        if node.has_class("head-menu__link--city"):
            city = [item for item in _nodes(node) if item.has_class("head-menu__link-border")]
            labels.extend(_normal(_visible(item)) for item in city)
    # The active city must be explicit; Organization/address metadata alone is not enough.
    return bool(labels) and set(labels) == {"москва"}


def _unavailable(url, reason):
    return {**_SHOP, "status": "unavailable", "source": url, "url": url,
            "availability_scope": "moscow_branches", "warnings": [reason]}


def parse_decanter_product(html, url):
    """Return one current public RUB offer, only with verified Moscow stock."""
    try:
        canonical = _product_url(url)
        document = _document(html)
        if not _moscow(document):
            raise ValueError("Не подтверждён выбранный город Москва")
        main = [node for node in _nodes(document)
                if node.is_kind("Product") and node.has_class("prod-page")]
        if len(main) != 1 or not main[0].closed:
            raise ValueError("Изменилась структура основной карточки товара")
        product = main[0]
        page_id = _PRODUCT_PATH.fullmatch(urlsplit(canonical).path).group(1)
        if product.attrs.get("data-prod-id") != page_id:
            raise ValueError("Карточка не соответствует запрошенному товару")
        headings = [node for node in _nodes(product)
                    if node.tag == "h1" and "name" in node.attrs.get("itemprop", "").split()
                    and _enclosing(node) is product]
        name = _one(product, "name")
        if len(headings) != 1 or not name or _text(_visible(headings[0]), 1000) != name:
            raise ValueError("Название основной карточки неоднозначно")
        identity = _identity({"name": name, "brand": _one(product, "brand") or "",
                              "sku": _one(product, "sku") or ""})
        available_shops = []
        for row in _nodes(product):
            if not row.has_class("prod-card__available-item") or _enclosing(row, "Product") is not product:
                continue
            addresses = [item for item in _nodes(row) if item.has_class("prod-card__available-street")]
            statuses = [item for item in _nodes(row) if item.has_class("prod-card__available-text")]
            if (len(addresses) == 1 and len(statuses) == 1
                    and _normal(_visible(statuses[0])) == "в наличии"):
                links = [item for item in _nodes(addresses[0]) if item.tag == "a" and item.attrs.get("data-shop")]
                heads = [item for item in _nodes(addresses[0]) if item.has_class("store__head")]
                address = _text(_visible(heads[0]), 300) if len(heads) == 1 else ""
                if (len(links) == 1 and _normal(address) == _MOSCOW_SHOPS.get(links[0].attrs["data-shop"])):
                    available_shops.append(address)
        if not available_shops:
            raise ValueError("Нет подтверждённого наличия в магазинах Москвы")
        offers = [node for node in _nodes(product)
                  if node.is_kind("Offer") and node.has_class("prod-card__price")
                  and "offers" in node.attrs.get("itemprop", "").split()
                  and _enclosing(node, "Product") is product]
        if len(offers) != 1 or not offers[0].closed:
            raise ValueError("Текущая публичная цена неоднозначна")
        offer = offers[0]
        offer_url = _one(offer, "url")
        if not offer_url or _product_url(urljoin(canonical, offer_url)) != canonical:
            raise ValueError("Цена относится к другому товару")
        # A crossed-out nested UnitPriceSpecification is deliberately ignored.
        labels = [node for node in _nodes(offer) if node.has_class("prod-card__price-label")]
        if len(labels) != 1 or _normal(_visible(labels[0])) != "цена" or _properties(offer, "priceType"):
            raise ValueError("Не найдена однозначная публичная цена карточки")
        qualifiers = " ".join(_visible(node) for node in _nodes(offer)
                              if _enclosing(node) is offer and not node.is_kind("UnitPriceSpecification"))
        conditional_class = any(re.search(r"club|member|loyal|клуб|card-price|price-card", node.attrs.get("class", ""), re.IGNORECASE)
                                for node in _nodes(offer) if (node is offer or _enclosing(node) is offer)
                                and not node.is_kind("UnitPriceSpecification"))
        if conditional_class or re.search(r"\b(?:по\s+карте|клубн\w*|club|member\w*|от\s+\d)", qualifiers, re.IGNORECASE):
            raise ValueError("Условная или клубная цена не является публичной ценой")
        amount = _number(_one(offer, "price"))
        currency = _one(offer, "priceCurrency")
        availability = (_one(offer, "availability") or "").rstrip("/").rsplit("/", 1)[-1]
        if amount is None or currency != "RUB" or availability not in {"InStock", "LimitedAvailability"}:
            raise ValueError("Нет текущего доступного предложения с положительной ценой RUB")
        expiry = _one(offer, "priceValidUntil")
        if expiry and (not re.fullmatch(r"\d{4}-\d{2}-\d{2}", expiry)
                       or date.fromisoformat(expiry) < datetime.now(_MOSCOW).date()):
            raise ValueError("Срок опубликованной цены истёк или некорректен")
        return {**identity, **_SHOP, "status": "ok", "amount": amount, "price": amount,
                "currency": "RUB", "availability": availability, "available": True,
                "availability_scope": "moscow_branches", "available_shops": list(dict.fromkeys(available_shops)),
                "source": canonical, "url": canonical, "as_of": datetime.now(timezone.utc).isoformat(),
                "price_valid_until": expiry, "method": "Decanter main Product/Offer microdata and Moscow branches",
                "warnings": ["Розничная цена замещения в Москве на момент проверки; не стоимость перепродажи."]}
    except (ValueError, TypeError, RecursionError) as exc:
        return _unavailable(url, "Декантер: " + _text(str(exc), 200) + ". Цена не записана; доступен ручной ввод.")


def _search_candidates(document):
    result = []
    seen = set()
    for product in _nodes(document):
        if (not product.is_kind("Product") or not product.closed
                or _enclosing(product, "ListItem") is None):
            continue
        list_item = _enclosing(product, "ListItem")
        if (not list_item.closed or not list_item.has_class("prod__item")
                or not _in_catalog(list_item)
                or list_item.attrs.get("data-type") != "product_card"
                or "itemListElement" not in list_item.attrs.get("itemprop", "").split()
                or not str(_one(list_item, "position") or "").isdigit()):
            continue
        owned_products = [node for node in _nodes(list_item)
                          if node.is_kind("Product") and _enclosing(node, "ListItem") is list_item]
        if len(owned_products) != 1:
            continue
        name = _one(product, "name")
        # The visible card is a sibling of the Product microdata in this ListItem.
        links = [node for node in _nodes(list_item)
                 if node.tag == "a" and "url" in node.attrs.get("itemprop", "").split()
                 and node.has_class("prod__link") and _enclosing(node, "ListItem") is list_item]
        if not name or len(links) != 1:
            continue
        try:
            target = _product_url(urljoin(_ROOT, links[0].attrs.get("href", "")))
        except ValueError:
            continue
        if target in seen:
            continue
        identity = _identity({"name": name, "brand": _one(product, "brand") or ""})
        result.append({"url": target, **identity})
        seen.add(target)
        if len(result) == 6:
            break
    return result


def _search_empty(node):
    if (not node.has_class("not-found--search") or not node.has_class("not-found")
            or not node.closed or "ничего не найдено" not in _normal(_visible(node))):
        return False
    parent = node.parent
    while parent is not None:
        if parent.has_class("prod-list") and parent.closed:
            return True
        parent = parent.parent
    return False


def decanter_search_status(html, url):
    """Distinguish a verified empty search from inaccessible or changed markup."""
    unavailable = {"status": "unavailable", "reason": "Не подтверждена структура поиска Декантера; повторите позже или добавьте ссылку вручную.",
                   "candidates": []}
    try:
        if not _search_url(url):
            return {**unavailable, "reason": "Нужен публичный поиск decanter.ru для Москвы."}
        document = _document(html)
        if not _moscow(document):
            return {**unavailable, "reason": "Поиск Декантера не подтвердил выбранный город Москва."}
        for node in _nodes(document):
            if (node.attrs.get("id") in {"challenge-form", "cf-challenge-running"}
                    or node.has_class("cf-browser-verification")
                    or (node.tag == "title" and re.search(
                        r"just a moment|attention required|captcha|verify you are human|проверка браузера",
                        _visible(node), re.IGNORECASE))):
                return {**unavailable, "reason": "Магазин вернул проверку доступа вместо результатов поиска; обход не выполняется."}
        catalogs = [node for node in _nodes(document) if node.attrs.get("id") == "CATALOG_CONTAINER"]
        empty = [node for node in _nodes(document) if _search_empty(node)]
        if catalogs:
            if len(catalogs) != 1 or not catalogs[0].closed or empty:
                return unavailable
            candidates = _search_candidates(catalogs[0])
            if candidates:
                return {"status": "ok", "reason": "Получены подтверждённые результаты поиска для Москвы.", "candidates": candidates}
            return unavailable
        if len(empty) == 1:
            return {"status": "empty", "reason": "Публичный поиск Декантера не нашёл товаров по этому запросу.", "candidates": []}
        return unavailable
    except (ValueError, TypeError, RecursionError):
        return unavailable


def parse_decanter_search(html, url):
    """At most six genuine search identities; suggestions and prices are excluded."""
    return decanter_search_status(html, url)["candidates"]
