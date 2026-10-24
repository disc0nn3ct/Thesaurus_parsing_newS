"""Opt-in, read-only wine identity suggestions from verified public retail cards.

Receipt abbreviations are search hints, never evidence of producer or vintage.
No inventory, expense, receipt or price observations are written by this module.
"""
import hashlib
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from http.client import HTTPException
from urllib.parse import urlencode, urljoin, urlsplit

from . import wine_retail as retail
from .feature_access import allowed
from .store import dumps, utcnow
from .wine_retail_sources import (
    _document,
    _enclosing,
    _moscow,
    _nodes,
    _one,
    _product_url,
    _visible,
    decanter_search_status,
)
from .wine_tools import (
    _download,
    _identity,
    _normal,
    _text,
    _vintage,
    _volume,
    validate_product_url,
)

CACHE_MINUTES = 30
MAX_CARDS = 4
# Spelling equivalence only, verified against the public retail title and the
# official brand page. These aliases contain no hard-coded wine attributes.
# https://decanter.ru/product/concha-y-toro-casillero-del-diablo-merlot-id8159
# https://www.casillerodeldiablo.com/wine/reserva-merlot/
_ALIASES = (
    (r"\bка[зс]ил[ь]?(?:еро)?\s*(?:дель\s*)?дьябло\b", "Casillero del Diablo"),
    (r"\bбарилоче\b", "Bariloche"),
    (r"\bсан\s+валентин\b", "San Valentin"),
    (r"\bфронтера\b", "Frontera"),
    (r"\bгран\s+кастильо\b", "Gran Castillo"),
    (r"\bfamilia\s+torres\b", "Torres"),
    (r"\b(?:фамилия\s+шредер|familia\s+schroeder)\b", "Familia Schroeder"),
)
_GRAPES = {
    'merlot': ('мерло', 'merlot'), 'cabernet sauvignon': ('каберне совиньон', 'cabernet sauvignon'),
    'carmenere': ('карменер', 'карменере', 'carmenere', 'carménère'),
    'chardonnay': ('шардоне', 'chardonnay'), 'sauvignon blanc': ('совиньон блан', 'sauvignon blanc'),
    'syrah': ('сира', 'шираз', 'syrah', 'shiraz'), 'pinot noir': ('пино нуар', 'pinot noir'),
    'riesling': ('рислинг', 'riesling'), 'malbec': ('мальбек', 'malbec'),
    'pinot grigio': ('пино гриджио', 'pinot grigio'),
    'garnacha': ('гарнача', 'гренаш', 'garnacha', 'grenache'),
    'tempranillo': ('темпранильо', 'tempranillo'),
    'saperavi': ('саперави', 'saperavi'),
}
_STOP = frozenset(['вино', 'wine', 'кр', 'красное', 'красный', 'бел', 'белое', 'белый', 'роз', 'розовое', 'розовый', 'сух', 'сухое', 'сухой', 'полусухое', 'полусладкое', 'сладкое', 'dry', 'red', 'white', 'rose', 'п', 'сл', 'полусух', 'полуслад'])
_VOLUME = r"(?<!\d)\d+(?:[.,]\d+)?\s*(?:ml|мл|cl|сл|l|л)(?!\w)"


class _SourceUnavailable(ValueError):
    """A source failure, distinct from caller access/review guard exceptions."""


def _plain(value, limit=300):
    if not isinstance(value, str) or len(value) > 10000:
        return ''
    try:
        return _text(' '.join(_visible(_document(value)).split()), limit)
    except (ValueError, RecursionError):
        return ''


def _aliases(text):
    # Accent marks do not distinguish wine identities (Valentín / Valentin).
    # Preserve Cyrillic letters, including й, when folding Latin diacritics.
    text = ''.join(''.join(char for char in unicodedata.normalize('NFKD', letter) if not unicodedata.combining(char))
                   if 'LATIN' in unicodedata.name(letter, '') else letter for letter in text)
    for pattern, replacement in _ALIASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    for canonical, variants in _GRAPES.items():
        for variant in variants:
            text = re.sub(r'(?<!\w)' + re.escape(variant) + r'(?!\w)', canonical, text, flags=re.IGNORECASE)
    return text


def _core(text):
    text = _aliases(text)
    text = re.sub(r'\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?\s*%', ' ', text)
    text = re.sub(r'\bст\s*/\s*б\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(_VOLUME, ' ', text, flags=re.IGNORECASE)
    # A bare decimal in a receipt is excluded from search, but is not a volume.
    text = re.sub(r'(?<!\d)0[.,](?:375|75|5)(?!\d)', ' ', text)
    text = re.sub(r'\b(?:18|19|20)\d{2}\b', ' ', text)
    return ' '.join(word for word in _normal(text).split() if word not in _STOP)


def _grape_names(value):
    text = ' ' + _normal(_aliases(value)) + ' '
    return {name for name in _GRAPES if ' ' + name + ' ' in text}


def _inputs(fields):
    if not isinstance(fields, dict):
        raise TypeError('Нужны поля вина')
    name, winery = fields.get('name', ''), fields.get('winery', '')
    if (not isinstance(name, str) or not 2 <= len(name.strip()) <= 500 or not isinstance(winery, str)
            or len(winery) > 300 or re.search(r'https?://|www\.|@|[<>]|\b\d{7,}\b|\b(?:ФН|ФД|ФПД?|ИНН|телефон|пароль|api_key|token)\b',
                                             name + ' ' + winery, re.IGNORECASE)):
        raise ValueError('Укажите только название вина и производителя, без ссылок или данных чека')
    # A millilitre size such as 2020 ml must not become a harvest year.
    identity = _identity({'name': re.sub(_VOLUME, ' ', name, flags=re.IGNORECASE)})
    vintage = _vintage(fields.get('vintage')) if fields.get('vintage') not in {None, ''} else identity['vintage']
    if fields.get('vintage') not in {None, ''} and vintage is None:
        raise ValueError('Неверный год урожая')
    if vintage and identity['vintage'] and str(vintage) != str(identity['vintage']):
        raise ValueError('Название и поле года противоречат друг другу')
    from .receipts import receipt_volume
    volume_hint = receipt_volume(name, is_wine=True)
    explicit_volume = fields.get('volume_ml')
    if explicit_volume is not None and (type(explicit_volume) is not int or not 20 <= explicit_volume <= 30000):
        raise ValueError('Укажите объём бутылки в миллилитрах')
    volume = explicit_volume if explicit_volume is not None else volume_hint['volume_ml']
    if explicit_volume and volume_hint['volume_ml'] and explicit_volume != volume_hint['volume_ml']:
        raise ValueError('Название и поле объёма противоречат друг другу')
    core = _core(name + ' ' + winery)
    if not core or not any(char.isalpha() for char in core):
        raise ValueError('Уточните название или марку вина')
    return {'name': _plain(name, 500), 'winery': _plain(winery), 'vintage': vintage,
            'volume_ml': volume, 'core': core, 'grapes': sorted(_grape_names(name)),
            'volume_requires_confirmation': not fields.get('volume_ml') and volume_hint['volume_requires_confirmation']}


def _property_value(scope):
    values = []
    for node in _nodes(scope):
        if 'value' in node.attrs.get('itemprop', '').split() and _enclosing(node) is scope:
            value = _plain(node.attrs.get('content', _visible(node)), 600)
            if value and value not in values:
                values.append(value)
    return values[0] if len(values) == 1 else ''


def _manufacturer(product):
    values = []
    for node in _nodes(product):
        if 'manufacturer' in node.attrs.get('itemprop', '').split() and _enclosing(node) is product:
            value = _one(node, 'name') if 'itemscope' in node.attrs else node.attrs.get('content', _visible(node))
            value = _plain(value or '')
            if value and value not in values:
                values.append(value)
    if len(values) > 1:
        raise ValueError('Производитель основной карточки неоднозначен')
    return values[0] if values else ''


def parse_decanter_metadata(html, url):
    """Read the main Product identity even if its price/stock is unavailable."""
    canonical = _product_url(url)
    document = _document(html)
    if not _moscow(document):
        raise ValueError('Не подтверждён выбранный город магазина')
    main = [node for node in _nodes(document) if node.is_kind('Product') and node.has_class('prod-page')]
    if len(main) != 1 or not main[0].closed:
        raise ValueError('Изменилась структура основной карточки вина')
    product = main[0]
    declared = [_product_url(urljoin(canonical, node.attrs.get('href', ''))) for node in _nodes(document)
                if node.tag == 'link' and node.attrs.get('rel', '').casefold() == 'canonical'
                and _enclosing(node, 'Product') is None]
    if declared and declared != [canonical]:
        raise ValueError('Канонический адрес карточки относится к другому товару')
    # Archived Decanter URLs can retain a public ID while the internal Product ID
    # changes. The exact canonical declaration verifies that reviewed alias.
    if product.attrs.get('data-prod-id') != re.search(r'-id(\d+)/?$', canonical).group(1) and declared != [canonical]:
        raise ValueError('Карточка относится к другому товару')
    headings = [node for node in _nodes(product) if node.tag == 'h1'
                and 'name' in node.attrs.get('itemprop', '').split() and _enclosing(node) is product]
    title = _plain(_one(product, 'name') or '', 500)
    if len(headings) != 1 or _plain(_visible(headings[0]), 500) != title or not title:
        raise ValueError('Название основной карточки неоднозначно')
    if not re.search(r'\b(?:вино|wine|шампанское|игристое)\b', title, re.IGNORECASE):
        raise ValueError('Карточка не подтверждает, что товар является вином')
    props = {}
    for node in _nodes(product):
        if (node.is_kind('PropertyValue') and 'additionalProperty' in node.attrs.get('itemprop', '').split()
                and _enclosing(node, 'Product') is product):
            key, value = _normal(_one(node, 'name') or ''), _property_value(node)
            if key and value:
                if key in props and props[key] != value:
                    raise ValueError('Характеристики карточки противоречат друг другу')
                props[key] = value
    brand = _plain(_one(product, 'brand') or '')
    manufacturer = _manufacturer(product)
    if manufacturer and props.get('производитель') and _normal(manufacturer) != _normal(props['производитель']):
        raise ValueError('Опубликованные производители противоречат друг другу')
    manufacturer = manufacturer or props.get('производитель', '')
    producer_note = ''
    if not manufacturer and brand:
        # This is explicitly shown in the title; the attribution is a suggestion,
        # not an invented brand-to-producer mapping or an automatic exact match.
        stripped = re.sub(r'^\s*(?:Вино|Wine)\s+', '', title, flags=re.IGNORECASE)
        position = stripped.casefold().find(brand.casefold())
        if position > 0:
            prefix = stripped[:position].strip(' ,·–-')
            if len(prefix) <= 200 and not re.search(r'\d|резерв|reserva|reserve|special', prefix, re.IGNORECASE):
                manufacturer = prefix
                producer_note = 'Производитель предложен по префиксу названия карточки; отдельное поле отсутствует.'
    identity = _identity({'name': re.sub(_VOLUME, ' ', title, flags=re.IGNORECASE), 'manufacturer': manufacturer})
    volume = _volume(title)
    stated_volume = props.get('объем л', props.get('объём л', ''))
    property_volume = _volume(stated_volume + ' л') if stated_volume else None
    if volume and property_volume and volume != property_volume:
        raise ValueError('Объём в названии и характеристиках не совпадает')
    volume = volume or property_volume
    year = _vintage(props.get('винтаж', props.get('год урожая')))
    if year and identity['vintage'] and str(year) != str(identity['vintage']):
        raise ValueError('Годы в названии и характеристиках не совпадают')
    year = year or identity['vintage']
    name = re.sub(r'^\s*(?:Вино|Wine)\s+', '', title, flags=re.IGNORECASE)
    name = re.sub(_VOLUME, '', name, flags=re.IGNORECASE)
    if year and year != 'NV':
        name = re.sub(r'\b' + str(year) + r'\b', '', name)
    name = ' '.join(name.split()).strip(' ,·–-')
    grapes = [_plain(value, 100) for value in re.split(r'\s*[,;+]\s*', props.get('виноград', '')) if value.strip()][:10]
    named_grapes, stated_grapes = _grape_names(title), _grape_names(' '.join(grapes))
    if named_grapes and stated_grapes and not named_grapes.issubset(stated_grapes):
        raise ValueError('Название и опубликованные сорта винограда противоречат друг другу')
    return {'id': hashlib.sha256(canonical.encode()).hexdigest()[:32], 'name': name, 'winery': manufacturer,
            'volume_ml': volume, 'vintage': None, 'published_vintage': year,
            'country': props.get('страна', '')[:150], 'region': props.get('регион', '')[:200],
            'grapes': grapes, 'description': _plain(_one(product, 'description') or '', 600),
            'source_url': canonical, 'source_title': title, 'fetched_at': utcnow(),
            'match_notes': [producer_note] if producer_note else []}


def _match(candidate, fields):
    notes = list(candidate['match_notes'])
    certain = not notes
    # A different bottle/year in the owned upload does not by itself make the
    # stable wine identity uncertain. Names, series and grapes are checked below.
    identity_certain = not notes or candidate.get('source_kind') == 'vivino_upload'
    for field, published in (('volume_ml', candidate['volume_ml']), ('vintage', candidate['published_vintage'])):
        expected = fields[field]
        if expected is not None:
            if published is not None and str(expected) != str(published):
                return None
            if published is None:
                certain = False
                notes.append('Карточка не подтверждает ' + ('объём бутылки.' if field == 'volume_ml' else 'год урожая.'))
    expected_grapes = set(fields['grapes'])
    found_grapes = _grape_names(candidate['name'] + ' ' + candidate['source_title'] + ' ' + ' '.join(candidate['grapes']))
    if expected_grapes and found_grapes and not expected_grapes.issubset(found_grapes):
        return None
    if expected_grapes and not found_grapes:
        certain = False
        identity_certain = False
        notes.append('Сорт винограда не подтверждён карточкой.')
    wanted = set(fields['core'].split())
    core = _core(candidate['name'] + ' ' + candidate['winery'])
    actual = set(core.split())
    meaningful = wanted - set(' '.join(fields['grapes']).split()) - {'del', 'de', 'di', 'la', 'le', 'y'}
    if not meaningful or not meaningful.issubset(actual):
        return None
    if fields['winery'] and _normal(_aliases(fields['winery'])) != _normal(_aliases(candidate['winery'])):
        if candidate['winery']:
            return None
        certain = False
        notes.append('Производитель не опубликован отдельно.')
    producer_tokens = set(_normal(_aliases(candidate['winery'])).split())
    named = set(_core(candidate['name']).split())
    # A shared producer cannot turn a different named wine into a match. A
    # varietal-only Vivino title may identify its brand in the winery field.
    identity_words = named - set(' '.join(found_grapes).split()) - {'reserva', 'reserve', 'del', 'de', 'di', 'la', 'le', 'y'}
    if not meaningful.issubset(named) and identity_words - wanted - producer_tokens:
        return None
    if wanted != actual and not (wanted.issubset(actual) and actual - wanted <= producer_tokens):
        certain = False
        identity_certain = False
        notes.append('Название или серия отличаются; сравните с этикеткой.')
    if fields['vintage'] is None:
        notes.append('Год вашей бутылки неизвестен; год текущей карточки не переносится.')
    if fields['volume_ml'] is None and candidate['volume_ml']:
        notes.append('Объём опубликован магазином; подтвердите его по бутылке.')
    elif fields.get('volume_requires_confirmation'):
        notes.append('Объём сопоставлен по десятичному числу в конце названия чека; подтвердите единицы по бутылке.')
    return {**candidate, 'vintage': candidate['published_vintage'] if fields['vintage'] is not None else None,
            'match_status': 'exact' if certain else 'possible',
            'attribute_match_status': 'exact' if identity_certain else 'possible', 'match_notes': notes}


def automatic_candidate(candidates):
    """Agree on stable attributes across exact identities; vintage stays separate."""
    from .wine_cellar import _attributes
    if len(candidates) == 1 and candidates[0].get('match_status') == 'exact':
        return candidates[0]
    if not candidates or any(row.get('attribute_match_status') != 'exact' for row in candidates):
        return None
    result = {**candidates[0], 'vintage': None, 'published_vintage': None, 'attribute_sources': {}}
    for field in ('winery', 'country', 'region', 'grapes', 'volume_ml'):
        groups = {}
        for row in candidates:
            value = row.get(field)
            if value in (None, '', []):
                continue
            if field in {'country', 'region', 'grapes'}:
                value = _attributes({field: value})[field]
            canonical = (sorted(_normal(_aliases(item)) for item in value)
                         if field == 'grapes' else _normal(_aliases(str(value))))
            key = dumps(canonical)
            groups.setdefault(key, []).append(row)
        if len(groups) > 1:
            # A conflict in producer/country indicates potentially different wine.
            if field in {'winery', 'country'}:
                return None
            result[field] = [] if field == 'grapes' else None if field == 'volume_ml' else ''
        elif groups:
            rows = next(iter(groups.values()))
            result[field] = rows[0][field]
            result['attribute_sources'][field] = [{'url': row['source_url'], 'fetched_at': row['fetched_at']} for row in rows]
    result['match_status'] = 'exact'
    return result


def _vivino_candidates(store, actor):
    """Read only this user's uploaded catalogue; do not request Vivino over HTTP."""
    if not allowed(store, actor, 'wine_cellar'):
        return []
    dataset = store.get_setting(f'project:vivino:{actor}:dataset', {})
    if not isinstance(dataset, dict) or not isinstance(dataset.get('rows'), list):
        return []
    found = {}
    for row in dataset['rows'][:3000]:
        if not isinstance(row, dict) or not row.get('name'):
            continue
        data = {key: _plain(row.get(key) or '', limit) for key, limit in
                (('name', 500), ('winery', 200), ('country', 150), ('region', 200))}
        grapes = row.get('grapes', [])
        if not isinstance(grapes, list):
            grapes = []
        data['grapes'] = [_plain(value, 100) for value in grapes[:10] if isinstance(value, str)]
        if not data['grapes']:
            # An explicit varietal in the uploaded title is source evidence;
            # never derive a grape from a brand or producer name.
            data['grapes'] = sorted(_grape_names(data['name']))
        year = _vintage(row.get('vintage'))
        volume = row.get('volume_ml')
        volume = volume if type(volume) is int and 20 <= volume <= 30000 else None
        link = row.get('link', '')
        try:
            parsed = urlsplit(link) if isinstance(link, str) else urlsplit('')
        except ValueError:
            parsed = urlsplit('')
        if (parsed.scheme != 'https' or parsed.netloc not in {'vivino.com', 'www.vivino.com'}
                or parsed.query or parsed.fragment or not re.fullmatch(r'/(?:wines/\d+|(?:[\w-]+/)*w/\d+)', parsed.path)):
            link = ''
        identity = hashlib.sha256(dumps({**data, 'year': year, 'volume': volume, 'link': link}).encode()).hexdigest()[:32]
        found[identity] = {**data, 'id': identity, 'source_kind': 'vivino_upload',
            'dataset_generation': dataset.get('generation'), 'source_url': link, 'source_title': 'Ваша загрузка Vivino',
            'volume_ml': volume, 'vintage': None, 'published_vintage': year, 'description': '',
            'fetched_at': dataset.get('imported_at') or '',
            'match_notes': ['Сведения из вашей загрузки Vivino; это может быть другая бутылка или другой год.']}
    return list(found.values())


def candidate_allowed(store, actor, candidate):
    """Recheck source consent or the current owned upload before applying data."""
    if candidate.get('source_kind') == 'vivino_upload':
        return any(row['id'] == candidate.get('id') and row['dataset_generation'] == candidate.get('dataset_generation')
                   for row in _vivino_candidates(store, actor))
    try:
        validate_product_url(candidate['source_url'], allowed_hosts=retail.allowed_hosts(store))
    except (ValueError, KeyError):
        return False
    return True


def _require(store, actor, guard):
    if (type(actor) is not int or not store.authorize(actor)
            or not any(allowed(store, actor, feature) for feature in ('life_finance', 'wine_cellar'))):
        raise PermissionError('Поиск сведений о вине недоступен для вашей учётной записи')
    if guard:
        guard()


def _search(query, hosts, guard):
    guard()
    try:
        page, final = _download('https://decanter.ru/search/?' + urlencode({'q': query}), allowed_hosts=hosts, search_shop='decanter')
    except (OSError, ValueError, RuntimeError, LookupError, HTTPException):
        raise _SourceUnavailable('Магазин не предоставил поиск') from None
    guard()
    if urlsplit(final).path.startswith('/product/'):
        try:
            return [parse_decanter_metadata(page, final)], []
        except (ValueError, RuntimeError, LookupError):
            raise _SourceUnavailable('Не подтверждена карточка') from None
    observed = decanter_search_status(page, final)
    if observed['status'] == 'unavailable':
        raise _SourceUnavailable('Магазин не предоставил проверяемые результаты поиска')
    candidates, warnings = [], []
    links = observed['candidates']
    if len(links) > MAX_CARDS:
        warnings.append('Проверены первые четыре карточки. Уточните название для более точного поиска.')
    for row in links[:MAX_CARDS]:
        guard()
        url = validate_product_url(_product_url(row['url']), allowed_hosts=hosts)
        try:
            html, final = _download(url, allowed_hosts=hosts)
        except PermissionError:
            raise
        except (OSError, ValueError, RuntimeError, LookupError, HTTPException):
            warnings.append('Одну из карточек магазина не удалось проверить; остальные результаты сохранены.')
            continue
        guard()
        try:
            candidates.append(parse_decanter_metadata(html, final))
        except (ValueError, RuntimeError, LookupError):
            warnings.append('Структура одной из карточек не подтверждена; остальные результаты сохранены.')
    if links and not candidates:
        raise _SourceUnavailable('Ни одна карточка не предоставила проверяемые сведения')
    return candidates, warnings


def lookup(store, actor, fields, *, guard=None, force=False, dry_run=False):
    """Suggest public identities; the caller owns review and missing-field updates."""
    _require(store, actor, guard)
    if type(force) is not bool or type(dry_run) is not bool:
        raise TypeError('Неверные параметры поиска')
    inputs = _inputs(fields)
    empty = {'candidates': [], 'warnings': [], 'cached': False}
    if dry_run:
        return {**empty, 'status': 'dry_run', 'warnings': ['Поиск не запускался; сеть и кэш не использовались.']}
    hosts = retail.allowed_hosts(store)
    local = [matched for candidate in _vivino_candidates(store, actor) if (matched := _match(candidate, inputs))]
    supported = [row for row in retail.shops(store) if row['enabled'] and row.get('search_supported') and row['id'] == 'decanter']
    references = [row for row in retail.shops(store) if row['enabled'] and row.get('metadata_supported')]
    if not supported and not references:
        if local:
            return {**empty, 'status': 'candidates', 'candidates': local[:8]}
        return {**empty, 'status': 'disabled', 'warnings': ['Администратор должен подключить магазин с поддержкой поиска сведений о вине.']}
    signature = hashlib.sha256(dumps({'fields': inputs, 'hosts': hosts, 'sources': [row['id'] for row in supported+references],
                                      'local': [(row['id'], row['dataset_generation']) for row in local]}).encode()).hexdigest()
    key = f'wine_enrichment_cache:v3:{actor}:{signature}'
    cache = store.get_setting(key)
    now = datetime.now(timezone.utc)
    cached_result = None
    if not force and isinstance(cache, dict):
        try:
            expiry = datetime.fromisoformat(cache.get('expires', ''))
            if now < expiry <= now + timedelta(minutes=CACHE_MINUTES):
                cached_result = cache['result'] if isinstance(cache['result'], dict) else None
        except (ValueError, TypeError, KeyError):
            pass
    if cached_result is not None:
        _require(store, actor, guard)
        if set(hosts).issubset(retail.allowed_hosts(store)):
            return {**cached_result, 'cached': True}

    def check():
        _require(store, actor, guard)
        if not {row['host'] for row in supported+references}.issubset(retail.allowed_hosts(store)):
            raise PermissionError('Магазин отключён администратором во время поиска')

    query = inputs['core']
    if inputs['vintage'] and inputs['vintage'] != 'NV':
        query += ' ' + str(inputs['vintage'])
    query = query[:150]
    result = {**empty, 'candidates': local[:4], 'status': 'candidates' if local else 'empty'}
    try:
        found, warnings = [], []
        if references:
            from .wine_reference_sources import discover
            found, warnings = discover(inputs['core'], hosts, _download, check)
        if supported:
            try:
                searched, notices = _search(query, hosts, check)
                found += searched
                warnings += notices
            except _SourceUnavailable:
                warnings.append('Поиск магазина недоступен; использованы другие подключённые источники.')
                if not found and not local:
                    raise
        result['warnings'] += warnings
        for candidate in found:
            matched = _match(candidate, inputs)
            if matched:
                result['candidates'].append(matched)
        result['candidates'].sort(key=lambda row: (
            row['match_status'] != 'exact', any('серия отличаются' in note for note in row['match_notes']),
            inputs['vintage'] is None and row['published_vintage'] is not None, row['source_url']))
        result['status'] = 'candidates' if result['candidates'] else 'empty'
        if not result['candidates']:
            result['warnings'].append('Подходящих карточек не найдено; уточните марку, сорт или добавьте сведения вручную.')
    except _SourceUnavailable:
        result.update(status='candidates' if local else 'unavailable', warnings=['Подключённый магазин не предоставил проверяемые сведения. Попробуйте позже или заполните поля вручную.'])
    check()
    ttl = CACHE_MINUTES if result['status'] != 'unavailable' else 5
    store.set_setting(key, {'expires': (now + timedelta(minutes=ttl)).isoformat(), 'result': result})
    return result
