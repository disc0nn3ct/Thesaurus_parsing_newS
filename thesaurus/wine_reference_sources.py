"""Public reference routes and strict parsers; no purchase records or prices.

Routes are discovery hints only. All attributes are read from the current page,
checked against the requested identity and retained with their source URLs.
"""
import hashlib
import re
from urllib.parse import urlsplit

from .store import utcnow
from .wine_retail_sources import _document, _nodes, _visible
from .wine_tools import _normal, _vintage

ROUTES = {
    'bariloche malbec': (
        ('bahetle-sib.ru', 'https://bahetle-sib.ru/goods/vino-bariloche-malbek-kr-p-suh-12-13-st-b-0-75l--307240'),
        ('vinumday.com.br', 'https://www.vinumday.com.br/oferta/schroeder-malbec-bariloche-2024'),
    ),
}


def _one(nodes, class_name):
    rows = [node for node in nodes if node.has_class(class_name)]
    if len(rows) != 1 or not rows[0].closed:
        raise ValueError('Характеристики основной карточки неоднозначны')
    return rows[0]


def _text(node):
    return ' '.join(_visible(node).split())[:500]


def parse_reference(html, url):
    from .receipts import receipt_volume
    from .wine_enrichment import _plain
    doc, host = _document(html), urlsplit(url).hostname
    if host == 'bahetle-sib.ru':
        main = [n for n in _nodes(doc) if n.is_kind('Product') and n.has_class('product-page__content')]
        if len(main) != 1 or not main[0].closed:
            raise ValueError('Основной товар не найден')
        nodes = list(_nodes(main[0]))
        title = _text(_one(nodes, 'product-about__title'))
        fields = {}
        for row in nodes:
            if row.has_class('product-characteristics-details__tech-parameter'):
                children = list(_nodes(row))
                key = _normal(_text(_one(children, 'product-characteristics-details__tech-parameter-name')))
                value = _text(_one(children, 'product-characteristics-details__tech-parameter-value'))
                if key in fields and fields[key] != value:
                    raise ValueError('Характеристики противоречат друг другу')
                fields[key] = value
        composition = [n for n in nodes if n.has_class('product-characteristics-composition')]
        # This shop's composition contains grape names plus regulatory suffixes.
        from .wine_enrichment import _grape_names
        grapes = sorted(_grape_names(' '.join(_text(n) for n in composition)))
        if not grapes:
            # The product title is itself published varietal evidence.
            grapes = sorted(_grape_names(title))
        data = {'winery': fields.get('производитель', ''), 'country': fields.get('страна', ''),
                'region': fields.get('регион', ''), 'grapes': grapes, 'published_vintage': None}
        volume = receipt_volume(title, is_wine=True)['volume_ml']
    elif host == 'www.vinumday.com.br':
        nodes = list(_nodes(doc))
        title = _text(_one(nodes, 'ficha-tecnica__title'))
        fields = {}
        for row in nodes:
            if row.has_class('informacoes-tecnicas__linha'):
                children = [n for n in row.children if not isinstance(n, str)]
                if len(children) != 2:
                    raise ValueError('Структура технической таблицы изменилась')
                key, value = _normal(_text(children[0])), _text(children[1])
                if key in fields and fields[key] != value:
                    raise ValueError('Технические характеристики противоречат друг другу')
                fields[key] = value
        data = {'winery': fields.get('vinícola', ''), 'country': fields.get('país', ''),
                'region': fields.get('região', ''), 'grapes': [s.strip() for s in re.split(r'[,;]', fields.get('castas', '')) if s.strip()],
                'published_vintage': _vintage(fields.get('safra'))}
        volume = None
    else:
        raise ValueError('Неизвестный справочник')
    if not title or not data['winery'] or not data['country']:
        raise ValueError('Не подтверждены название, производитель и страна')
    # Translate controlled geographic names, never infer geography from a brand.
    data['country'] = {'Argentina': 'Аргентина'}.get(data['country'], data['country'])
    data['region'] = {'Patagônia': 'Патагония'}.get(data['region'], data['region'])
    return {**data, 'name': re.sub(r'\b20\d{2}\b', '', title).strip(), 'volume_ml': volume,
            'id': hashlib.sha256(url.encode()).hexdigest()[:32], 'vintage': None,
            'source_kind': 'public_reference', 'source_url': url, 'source_title': _plain(title, 500),
            'fetched_at': utcnow(), 'description': '', 'match_notes': []}


def discover(core, hosts, download, guard):
    results, warnings = [], []
    for key, routes in ROUTES.items():
        if not set(key.split()).issubset(set(core.split())):
            continue
        for host, url in routes:
            if host not in hosts:
                continue
            guard()
            try:
                html, final = download(url, allowed_hosts=hosts)
                if final.rstrip('/') != url.rstrip('/'):
                    raise ValueError('Карточка перенаправлена на другой товар')
                results.append(parse_reference(html, final))
            except (OSError, ValueError, RuntimeError):
                warnings.append('Один справочник недоступен; проверены остальные источники.')
            guard()
    return results, warnings
