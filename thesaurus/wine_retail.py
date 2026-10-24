"""Opt-in Moscow retail profiles, bounded discovery and reviewed bottle links.

Search results are suggestions, never automatic inventory or valuation changes.
The worker performs network requests; Telegram callbacks only review saved results.
"""
import hashlib
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from http.client import HTTPException
from urllib.parse import urlencode, urlsplit

from . import wine_cellar as cellar
from .store import dumps, utcnow
from .wine_tools import _normal, canonical_hosts, match_offer, validate_product_url

CITY = 'Москва'
CACHE_MINUTES = 30
CATALOG = {
    'bahetle_metadata': {
        'title': 'Бахетле · сведения о вине', 'host': 'bahetle-sib.ru', 'homepage': 'https://bahetle-sib.ru/',
        'automatic': False, 'search_supported': False, 'price_supported': False, 'metadata_supported': True,
        'status_label': 'характеристики вина; цены не используются',
        'reason': 'Публичные карточки: производитель, страна и состав. Это справочник, а не источник московских цен.',
    },
    'vinumday_metadata': {
        'title': 'VinumDay · сведения о вине', 'host': 'vinumday.com.br', 'homepage': 'https://www.vinumday.com.br/',
        'automatic': False, 'search_supported': False, 'price_supported': False, 'metadata_supported': True,
        'status_label': 'регион и сорта из технических карточек; цены не используются',
        'reason': 'Публичные технические характеристики. Указанный год — год карточки, его нужно сверить с бутылкой.',
    },
    'decanter': {
        'title': 'Декантер', 'host': 'decanter.ru', 'homepage': 'https://decanter.ru/',
        'automatic': True, 'search_supported': True, 'price_supported': True,
        'status_label': 'поддерживаются поиск и проверка цены',
        'reason': 'Публичные карточки и наличие в московских магазинах. Проверка выполняется при запросе.',
    },
    'amwine': {
        'title': 'Ароматный мир', 'host': 'amwine.ru', 'homepage': 'https://amwine.ru/',
        'automatic': True, 'search_supported': False, 'price_supported': True,
        'status_label': 'проверка цены по сохранённой ссылке',
        'reason': 'Сохраните ссылку на конкретную карточку. Без опубликованного года урожая точная оценка бутылки недоступна; автоматический поиск не подключён.',
    },
    'winelab': {
        'title': 'ВинЛаб', 'host': 'winelab.ru', 'homepage': 'https://www.winelab.ru/',
        'automatic': False, 'status_label': 'автоматическая проверка недоступна',
        'reason': 'Проверенный публичный каталог не предоставил товарные данные; требуется взаимодействие с сайтом.',
    },
    'perekrestok': {
        'title': 'Перекрёсток', 'host': 'perekrestok.ru', 'homepage': 'https://www.perekrestok.ru/',
        'automatic': False, 'status_label': 'автоматическая проверка недоступна',
        'reason': 'Вместо каталога получена защитная страница без цен и наличия.',
    },
    'simplewine': {
        'title': 'SimpleWine', 'host': 'simplewine.ru', 'homepage': 'https://simplewine.ru/',
        'automatic': False, 'status_label': 'автоматическая проверка недоступна',
        'reason': 'Проверенный публичный запрос отклонён магазином (HTTP 403).',
    },
}


def _raw_hosts(store):
    values = store.get_setting('wine_price_allowed_hosts')
    return canonical_hosts(os.getenv('WINE_PRICE_ALLOWED_HOSTS', '') if values is None else values)


def _selected(store):
    values = store.get_setting('wine_retail_shops', [])
    if not isinstance(values, list):
        return []
    return [key for key in CATALOG if key in values]


def allowed_hosts(store):
    """Retain custom opt-in domains and add only administrator-selected profiles."""
    return sorted(set(_raw_hosts(store)) | {CATALOG[key]['host'] for key in _selected(store)})


def shops(store):
    hosts = allowed_hosts(store)
    result = []
    for identifier, profile in CATALOG.items():
        state = store.get_setting('wine_retail_status:'+identifier, {})
        result.append({**profile, 'id': identifier, 'enabled': profile['host'] in hosts,
                       'city': CITY, 'last_check': state if isinstance(state, dict) else {}})
    return result


def configure_shop(store, actor, shop_id, enabled):
    if not store.authorize(actor, admin=True):
        raise PermissionError('Магазины подключает администратор.')
    if shop_id not in CATALOG or not isinstance(enabled, bool):
        raise ValueError('Неизвестный магазин.')
    selected = set(_selected(store))
    if enabled:
        selected.add(shop_id)
    else:
        selected.discard(shop_id)
    # Disabling a preset also removes the same domain from the advanced allowlist,
    # including an inherited environment value. Other custom domains are retained.
    hosts = _raw_hosts(store)
    if not enabled:
        hosts = [host for host in hosts if host != CATALOG[shop_id]['host']]
    audit = store.get_setting('wine_retail_audit', [])
    if not isinstance(audit, list):
        audit = []
    audit = (audit + [{'actor_id': actor, 'shop_id': shop_id, 'enabled': enabled, 'at': utcnow()}])[-100:]
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not store.authorize(actor, admin=True):
            raise PermissionError('Магазины подключает администратор.')
        for key, value in (('wine_retail_shops', sorted(selected)), ('wine_retail_city', CITY),
                           ('wine_retail_audit', audit)):
            conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                         (key, dumps(value)))
        if not enabled:
            conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                         ('wine_price_allowed_hosts', dumps(hosts)))


def profile_for_url(url):
    host = (urlsplit(url).hostname or '').lower().removeprefix('www.')
    return next(({'id': identifier, **row} for identifier, row in CATALOG.items() if row['host'] == host), None)


def availability_label(quote):
    scope = quote.get('availability_scope')
    return {'moscow_branches': 'в наличии в московских магазинах',
            'selected_moscow_shop': 'в наличии в выбранном московском магазине'}.get(scope, scope or 'место наличия не подтверждено')


def quote_eligible(quote, lot, *, moscow=False):
    """Check again after caches/review; missing identity or availability is not zero."""
    matched, reasons = match_offer(quote, lot)
    if (quote.get('winery') and lot.get('winery')
            and _normal(quote['winery']) != _normal(lot['winery'])):
        reasons.append('Опубликованный производитель отличается от указанного у бутылки.')
    amount = quote.get('amount')
    if (isinstance(amount, bool) or not isinstance(amount, (int, float))
            or not math.isfinite(amount) or not 0 < amount <= 1_000_000_000):
        reasons.append('Нет действительной положительной цены.')
    if quote.get('status') != 'ok' or quote.get('currency') != 'RUB':
        reasons.append('Не подтверждена цена в рублях.')
    if quote.get('availability') not in {'InStock', 'LimitedAvailability'}:
        reasons.append('Нет подтверждённого наличия.')
    expiry = quote.get('price_valid_until')
    if expiry:
        try:
            if date.fromisoformat(str(expiry)[:10]) < datetime.now(timezone(timedelta(hours=3))).date():
                reasons.append('Срок действия цены истёк.')
        except ValueError:
            reasons.append('Не удалось проверить срок цены.')
    if moscow and (quote.get('city') != CITY or quote.get('price_kind') != 'public'
                   or not quote.get('availability_scope')):
        reasons.append('Не подтверждены Москва, обычная публичная цена и место наличия.')
    return matched and not reasons, reasons


def _query(lot):
    # Product name is the strongest search term; producer is verified on each card.
    text = ''.join(c if c.isalnum() or c in " -.'’`&/(),+" else ' ' for c in str(lot.get('name', '')))
    text = ' '.join(text.split())[:120]
    year = str(lot.get('vintage') or '')
    if year and year != 'NV' and year not in text:
        text += ' ' + year
    return text


def _source_search(shop_id, lot, hosts, *, guard=None):
    from .wine_tools import _download, fetch_price
    direct_quote = None
    if guard:
        guard()
    if shop_id == 'decanter':
        from .wine_retail_sources import decanter_search_status, parse_decanter_product
        url = 'https://decanter.ru/search/?' + urlencode({'q': _query(lot)})
        page, final_url = _download(url, allowed_hosts=hosts, search_shop='decanter')
        # A unique genuine search result redirects to its exact product card.
        # The transport permits only the reviewed same-host product route.
        if urlsplit(final_url).path.startswith('/product/'):
            direct_quote = parse_decanter_product(page, final_url)
            if direct_quote.get('status') != 'ok':
                return [], {'status': 'unavailable', 'reason': '; '.join(direct_quote.get('warnings', []))[:500]}
            links = [{'url': final_url}]
        else:
            observed = decanter_search_status(page, final_url)
            if observed['status'] != 'ok':
                return [], {'status': observed['status'], 'reason': observed['reason']}
            links = observed['candidates']
    else:
        return [], {'status': 'manual', 'reason': CATALOG[shop_id]['reason']}
    offers, failures = [], 0
    for item in links[:3]:
        if guard:
            guard()
        # Never use a search snippet as a price: open and verify the exact card.
        url = validate_product_url(item['url'], allowed_hosts=hosts)
        profile = profile_for_url(url)
        if not profile or profile['id'] != shop_id:
            raise ValueError('Поиск вернул карточку другого магазина.')
        quote = direct_quote or fetch_price(url, allowed_hosts=hosts)
        if quote.get('status') != 'ok':
            failures += 1
            continue
        matched, reasons = quote_eligible(quote, lot, moscow=True)
        offers.append({**quote, 'shop_id': shop_id, 'shop_title': CATALOG[shop_id]['title'],
                       'match': matched, 'match_reasons': reasons})
    note = f'Проверено карточек: {min(3, len(links))}; доступных цен: {len(offers)}.'
    if len(links) > 3:
        note += ' Проверены первые 3 результата; можно уточнить название вина.'
    if not links:
        note = 'Поиск не нашёл это вино. Рекомендации магазина не считаются результатами поиска.'
    if failures:
        note += f' Без проверяемой цены: {failures}.'
    return offers, {'status': 'ok' if offers else 'empty', 'reason': note}


def search(store, actor, lot_id, *, dry_run=False):
    cellar.require_owner(store, actor, write=True)
    lot = cellar.get_lot(store, actor, lot_id)
    if not lot['quantity']:
        raise ValueError('Для поиска выберите вино в запасах.')
    offers, statuses = [], []
    hosts = allowed_hosts(store)
    signature = hashlib.sha256(dumps({key: lot.get(key) for key in ('name', 'winery', 'vintage', 'volume_ml')}).encode()).hexdigest()[:24]
    for shop in shops(store):
        if not shop['enabled']:
            continue
        cellar.require_owner(store, actor, write=True)
        state = {'shop_id': shop['id'], 'title': shop['title'], 'city': CITY, 'checked_at': utcnow()}
        if dry_run:
            state.update(status='dry_run', reason='Магазин не запрашивался.')
        elif not shop.get('search_supported'):
            state.update(status='manual', reason=shop['reason'])
        else:
            key = f"wine_retail_cache:{actor}:{lot_id}:{shop['id']}:{signature}"
            cache = store.get_setting(key)
            threshold = (datetime.now(timezone.utc)-timedelta(minutes=CACHE_MINUTES)).isoformat()
            if isinstance(cache, dict) and cache.get('checked_at', '') > threshold:
                found, observed = cache.get('candidates', []), cache.get('state', {})
                state.update(observed, cached=True)
            else:
                try:
                    def guard(shop_host=shop['host']):
                        cellar.require_owner(store, actor, write=True)
                        if shop_host not in allowed_hosts(store):
                            raise ValueError('Магазин отключён во время поиска.')
                    found, observed = _source_search(shop['id'], lot, hosts, guard=guard)
                except PermissionError:
                    raise
                except (OSError, ValueError, LookupError, RuntimeError, HTTPException):
                    found, observed = [], {'status': 'unavailable', 'reason': 'Магазин не предоставил проверяемые данные; сохранённые цены не изменены.'}
                state.update(observed)
                store.set_setting(key, {'checked_at': utcnow(), 'candidates': found, 'state': state})
            offers.extend(found)
            store.set_setting('wine_retail_status:'+shop['id'], state)
        statuses.append(state)
    # Recheck revoked configuration and access after all requests, including caches.
    cellar.require_owner(store, actor, write=True)
    current_hosts = allowed_hosts(store)
    offers = [row for row in offers if CATALOG[row['shop_id']]['host'] in current_hosts]
    for row in offers:
        row['match'], row['match_reasons'] = quote_eligible(row, lot, moscow=True)
    offers.sort(key=lambda row: (not row['match'], row['amount'], row['url']))
    return {'actor_id': actor, 'lot_id': lot_id, 'city': CITY, 'created_at': utcnow(),
            'expires': (datetime.now(timezone.utc)+timedelta(minutes=CACHE_MINUTES)).isoformat(),
            'candidates': offers[:12], 'sources': statuses}


def candidates(store, actor, token):
    cellar.require_owner(store, actor, write=True)
    if not isinstance(token, str) or not re.fullmatch(r'[0-9a-f]{32}', token):
        raise ValueError('Некорректный результат поиска.')
    result = store.get_setting(f'wine_retail_search:{actor}:{token}')
    if (not isinstance(result, dict) or result.get('actor_id') != actor
            or result.get('expires', '') < utcnow()):
        raise ValueError('Поиск устарел. Найдите вино заново.')
    cellar.get_lot(store, actor, result['lot_id'])
    return result


def confirm_candidate(store, actor, token, index):
    result = candidates(store, actor, token)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(result['candidates']):
        raise ValueError('Предложение не найдено.')
    quote = result['candidates'][index]
    lot = cellar.get_lot(store, actor, result['lot_id'])
    profile = profile_for_url(quote['url'])
    if not profile or profile['id'] != quote.get('shop_id') or not profile['automatic']:
        raise ValueError('Магазин не поддерживает проверяемые предложения.')
    validate_product_url(quote['url'], allowed_hosts=allowed_hosts(store))
    matches, reasons = quote_eligible(quote, lot, moscow=True)
    if not matches or not lot['quantity']:
        raise ValueError('Цена не подходит этой бутылке. ' + '; '.join(reasons))
    if not quote.get('as_of') or quote['as_of'] < (datetime.now(timezone.utc)-timedelta(minutes=CACHE_MINUTES)).isoformat():
        raise ValueError('Цена устарела. Выполните поиск повторно.')
    link = cellar.add_price_link(store, actor, lot['id'], quote['url'])
    cellar.record_price(store, actor, lot['id'], quote['amount'], price_date=quote['as_of'][:10],
                        source='shop_confirmed', url=quote['url'], operation_id=f'wine-offer:{token}:{index}',
                        vintage=quote.get('vintage'), volume_ml=quote.get('volume_ml'),
                        metadata={**quote, 'selection': 'user_confirmed_exact_moscow_offer'})
    store.set_setting(f"wine_quote_cache:{actor}:{link['id']}", {'checked_at': quote['as_of'], 'quote': quote})
    return quote
