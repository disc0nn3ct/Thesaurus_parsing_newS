"""Refresh reviewed public source contracts. Run manually, never on service startup."""
import json
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesaurus.market_sources import CATALOG, Transport, contracts, day, number, page


def build():
    transport = Transport()
    root = page(transport.get('https://cbr.ru/hd_base/').text)
    roots = [(urljoin('https://cbr.ru', url), title) for url, title in root.links if title and (
        url.startswith('/hd_base/') or url in {'/currency_base/', '/pr/',
        '/oper_br/o_dkp/reserve_requirements/reserv_pr4/', '/statistics/flikvid/', '/statistics/pffl/'})]
    queue = [(url, title, urlsplit(url).path, None, 0) for url, title in roots]
    visited, items = set(), []
    while queue:
        url, title, prefix, parent, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        response = transport.get(url)
        parsed = page(response.text)
        identity = 'cbr.' + urlsplit(url).path.strip('/').replace('/', '.').lower()
        contract = contracts(response.text)
        frequency = 'monthly' if any('Инфляция' in title for _ in [0]) else 'daily'
        if '/mrrf/' in url or '/mb_nd/' in url:
            frequency = 'weekly'
        if '/mrrf/mrrf_m/' in url or '/mb_nd/mb_nd_month/' in url or 'month' in url.lower() or 'мнт' in title.lower():
            frequency = 'monthly'
        if any(part in url for part in ('/tg/', '/svs_db/')):
            frequency = 'quarterly'
        unit = '%' if any(part in url for part in ('/infl/', '/mkr/', '/ProcStav/', '/zcyc_params/', '/ImpliedRateCNY/')) else 'source_units'
        if any(part in url for part in ('/bliquidity/', '/ostat_base/', '/mb_nd/')):
            unit = 'млрд руб'
        links = [(urljoin(url, u), t) for u, t in parsed.links if t and u and not any(c.isspace() for c in u) and (
            urlsplit(urljoin(url, u)).hostname in {'cbr.ru', 'www.cbr.ru'}) and
            '#' not in u]
        children = [(u, t) for u, t in links if u != url and urlsplit(u).path != '/hd_base/' and
                    (urlsplit(u).path.startswith(prefix) or urlsplit(u).path.startswith('/hd_base/')) and
                    ('?' not in u or 'tab.current=' in u) and not Path(urlsplit(u).path).suffix]
        downloads = list(dict.fromkeys(u for u, _ in links if Path(urlsplit(u).path).suffix.lower() in {'.xlsx', '.xls', '.csv', '.pdf'} or 'DownloadExcel' in u))
        item = {'id': identity, 'source': 'cbr', 'dataset': identity, 'title': title,
                'url': url, 'parent': parent, 'kind': 'dataset' if contract else 'reference',
                'frequency': frequency, 'calendar': 'russia_working' if frequency == 'daily' else 'periodic',
                'aggregation': 'last', 'unit': unit, 'transform': 'difference', 'contracts': contract,
                'downloads': downloads, 'methodology': url, 'available_period': 'source_available',
                'definition': 'pre2023' if '/bliquidity/' in url else '1',
                'administrative': any(part in url for part in ('/infl/', '/ProcStav/', '/pr/'))}
        if '/bliquidity/' in url:
            item['definition'] = 'liquidity_nov2023_break'
            item['break_date'] = '2023-11-01'
        if '?' in url:
            item['id'] += '.' + urlsplit(url).query.replace('=', '_').replace('.', '_')
            item['dataset'] = item['id']
        for table_contract in contract:
            if table_contract.get('layout') == 'wide':
                item['frequency'] = 'annual'
                continue
            table = parsed.tables[table_contract['table']]
            numeric, dimensions = [], []
            for index, label in enumerate(table_contract['headers'][1:], 1):
                sample = [r[index] for r in table if len(r) > index and day(r[0])][:12]
                try:
                    for value in sample:
                        number(value)
                    numeric.append(index)
                except ValueError:
                    dimensions.append(index)
            table_contract['numeric'] = numeric
            table_contract['dimensions'] = dimensions
        if '/metall/' in url:
            item['unit'], item['transform'] = 'руб за грамм', 'returns'
        if '/mrrf/' in url:
            item['unit'] = 'млн долларов США'
        if '/valintbr/' in url:
            item['unit'] = 'млн долларов США'
        if '/swap' in url.lower() and 'ставк' not in title.lower():
            item['unit'] = 'млрд руб'
        if '/keyrate/' in url or 'ставк' in title.lower():
            item['unit'], item['aggregation'] = '%', 'mean'
        if '/infl/' in url:
            item['period_semantics'] = 'flow'
        if '/mrrf/' in url or '/mb_nd/' in url:
            item['period_semantics'] = 'stock'
        if 'mnth' in url:
            item['frequency'] = 'monthly'
        items.append(item)
        if depth < 4:
            queue.extend((u, t, prefix, identity, depth + 1) for u, t in children if u not in visited)
        print(identity, len(contract), flush=True)
    for title, code in [('USD', 'R01235'), ('EUR', 'R01239'), ('CNY', 'R01375')]:
        items.append({'id': 'cbr.fx.' + title.lower(), 'source': 'cbr', 'dataset': 'cbr.currency_base',
                      'title': 'Курс ' + title, 'unit': 'руб за 1 ' + title, 'frequency': 'daily',
                      'calendar': 'calendar', 'transform': 'returns', 'aggregation': 'last', 'kind': 'series',
                      'adapter': 'currency', 'code': code, 'url': 'https://cbr.ru/currency_base/', 'definition': '1'})
    for code in ('IMOEX', 'RTSI', 'RGBI'):
        items.append({'id': 'moex.' + code.lower() + '.close', 'source': 'moex', 'dataset': 'moex.indices',
                      'title': code, 'unit': 'индекс', 'frequency': 'daily', 'calendar': 'moex',
                      'transform': 'returns', 'aggregation': 'last', 'kind': 'series', 'adapter': 'moex',
                      'field': 'CLOSE', 'security': code, 'market': 'index',
                      'url': f'https://iss.moex.com/iss/history/engines/stock/markets/index/securities/{code}.json',
                      'delay': 'public ISS; access/history limits apply', 'definition': '1'})
    CATALOG.write_text(json.dumps(items, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Sources:', len(items), 'Root datasets:', len(roots))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    started = time.monotonic()
    build()
    print('Seconds:', round(time.monotonic() - started))
