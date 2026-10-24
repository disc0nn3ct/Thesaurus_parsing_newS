"""Public CBR/ISS adapters with bounded transport and explicit table contracts."""
import hashlib
import json
import re
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree

import pandas as pd
import requests

CATALOG = Path(__file__).with_name('market_catalog.json')


class Page(HTMLParser):
    """Extract links and tables, expanding HTML spans without executing content."""
    def __init__(self):
        super().__init__()
        self.links, self.tables, self.headings, self.table_titles = [], [], [], []
        self.href = None
        self.text = ''
        self.heading = False
        self.table = None
        self.row = None
        self.cell = None
        self.spans = {}
        self.col = 0
        self.attrs = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a':
            self.href, self.text = attrs.get('href'), ''
        if tag in {'h1', 'h2', 'h3'}:
            self.heading, self.text = True, ''
        if tag == 'table' and self.table is None:
            self.table, self.spans = [], {}
            self.table_titles.append(self.headings[-1] if self.headings else '')
        if tag in {'thead', 'tbody', 'tfoot'} and self.table is not None:
            # HTML row spans stop at their row group, even if the source declares an oversized span.
            self.spans = {}
        if tag == 'tr' and self.table is not None:
            self.row, self.col = [], 0
        if tag in {'td', 'th'} and self.row is not None:
            self.cell, self.attrs = '', attrs
        if tag == 'br' and self.cell is not None:
            self.cell += ' '

    def handle_data(self, text):
        if self.cell is not None:
            self.cell += text
        if self.href or self.heading:
            self.text += text

    def _span(self):
        while self.col in self.spans:
            value, remaining = self.spans[self.col]
            self.row.append(value)
            if remaining <= 1:
                del self.spans[self.col]
            else:
                self.spans[self.col] = (value, remaining - 1)
            self.col += 1

    def handle_endtag(self, tag):
        if tag == 'a' and self.href:
            self.links.append((self.href, ' '.join(self.text.split())))
            self.href = None
        if tag in {'h1', 'h2', 'h3'}:
            self.headings.append(' '.join(self.text.split()))
            self.heading = False
        if tag in {'td', 'th'} and self.cell is not None:
            self._span()
            value = ' '.join(self.cell.split())
            for _ in range(min(100, int(self.attrs.get('colspan', '1')))):
                self.row.append(value)
                height = min(100, int(self.attrs.get('rowspan', '1')))
                if height > 1:
                    self.spans[self.col] = (value, height - 1)
                self.col += 1
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self._span()
            self.table.append(self.row)
            self.row = None
        if tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def page(text):
    parser = Page()
    parser.feed(text)
    return parser


def catalogue():
    from .dollar_sources import catalogue as fx_context_catalogue
    items = json.loads(CATALOG.read_text(encoding='utf-8'))
    reviewed = json.loads(Path(__file__).with_name('resources').joinpath('cbr_adapters.json').read_text(encoding='utf-8'))
    for item in items:
        item.update(reviewed.get('datasets', {}).get(item['id'], {}))
    known = {item['id'] for item in items}
    items.extend(item for item in reviewed.get('series', []) if item['id'] not in known)
    transfer_fields = [('CO_total_sum', 'Сумма переводов', 0),
                       ('CO_total_asum', 'Средняя сумма переводов', 0),
                       ('CO_total_res_sum', 'Сумма переводов резидентов', 0),
                       ('CO_total_res_asum', 'Средняя сумма переводов резидентов', 0),
                       ('CO_total_nres_sum', 'Сумма переводов нерезидентов', 0),
                       ('CO_total_nres_asum', 'Средняя сумма переводов нерезидентов', 0),
                       ('Syst_sum', 'Сумма денежных переводов', 1),
                       ('Syst_asum', 'Средняя сумма денежного перевода', 1)]
    for direction, direction_title in [('1', 'Из России'), ('2', 'В Россию')]:
        for field, title, group in transfer_fields:
            identity = f'cbr.hd_base.tg.{direction}.{field}'
            items.append({'id': identity, 'dataset': identity, 'parent': 'cbr.hd_base.tg', 'source': 'cbr',
                          'title': f'{direction_title}: {title} (архив 2006–2021)', 'kind': 'dataset',
                          'adapter': 'transfers', 'frequency': 'quarterly', 'calendar': 'calendar',
                          'aggregation': 'mean' if 'asum' in field else 'sum', 'transform': 'difference',
                          'unit': 'долларов США' if 'asum' in field else 'млн долларов США',
                          'url': 'https://www.cbr.ru/hd_base/tg/', 'methodology': 'https://www.cbr.ru/hd_base/tg/',
                          'available_period': '2006–2021', 'definition': '1', 'full_snapshot': True,
                          'available_end': '2021-12-31', 'data_type': 'archive',
                          'direction': direction, 'transfer_field': field, 'transfer_group': group,
                          'description': 'Архив. По умолчанию: всего по странам. Полные опубликованные периоды; повторная загрузка только через сверку.'})
    items.extend([{'id': 'cbr.ruonia.' + field, 'source': 'cbr', 'dataset': 'cbr.hd_base.ruonia.ruonia',
        'parent': 'cbr.hd_base.ruonia.ruonia', 'title': title, 'frequency': 'daily', 'calendar': 'russia_working',
        'unit': unit, 'kind': 'series', 'adapter': 'ruonia', 'field': field, 'transform': 'difference',
        'aggregation': 'mean' if field == 'rate' else 'sum', 'definition': '1',
        'url': 'https://cbr.ru/hd_base/ruonia/ruonia/'} for field, title, unit in
        [('rate', 'Ставка RUONIA', '%'), ('volume', 'Объём RUONIA', 'млрд руб')]])
    for item in items:
        title = item['title'].lower()
        item.setdefault('data_type', 'reference' if item['kind'] == 'reference' else 'actual')
        if any(word in title for word in ('ретроспектив', 'архив', 'до 01.07.1992', 'ссср', 'по 04.07.2003', 'по 30.12.2016')):
            item['data_type'] = 'archive'
        if item['id'] == 'cbr.statistics.pffl':
            item.update(kind='dataset', adapter='official_forecast', frequency='weekly', unit='млрд руб', data_type='official_forecast')
        if any(part in item['id'] for part in ('ir_chg', '.icl', '.overnight', '.fixedstav', '.depomaxproc', '.repo_proc', '.reserv_pr4', 'cbr.pr')):
            item['calendar'] = 'events'
            item['administrative'] = True
        if item['id'] == 'cbr.hd_base.ruonia.dynamic':
            item['calendar'] = 'calendar'
        if item.get('adapter') == 'currency':
            item['calendar'] = 'cbr_fx'
        if item['id'] in {'cbr.hd_base.bsliquidityma', 'cbr.hd_base.bsliquidityma.full'}:
            item.update(frequency='irregular', calendar='events')
        if item['id'] == 'cbr.hd_base.mkr.mkr_monthes':
            headers = ['Дата', '1 день', 'от 2 до 7 дн.', 'от 8 до 30 дн.', 'от 31 до 90 дн.', 'от 91 до 180 дн.', 'от 181 дн. до 1 года']
            item.update(kind='dataset', data_type='actual', frequency='monthly', unit='%',
                        date_format='%m.%Y', default_params={'UniDbQuery.Currency': '-1',
                            'UniDbQuery.st': ['SF', 'HR', 'MB'],
                            'UniDbQuery.sk': ['Dd1_', 'Dd7', 'Dd30', 'Dd90', 'Dd180', 'Dd360']},
                        table_labels=['MIACR RUB', 'MIACR USD', 'MIACR-IG RUB', 'MIACR-B RUB'],
                        table_display_labels=['MIACR RUB · средняя ставка', 'MIACR USD · средняя ставка', 'MIACR-IG RUB · средняя ставка', 'MIACR-B RUB · средняя ставка'],
                        contracts=[{'table': i, 'headers': headers, 'numeric': list(range(1, 7)), 'dimensions': [], 'unit': '%', 'aggregation': 'mean'} for i in range(4)])
        if item['id'] == 'cbr.oper_br.o_dkp.reserve_requirements.reserv_pr4':
            item.update(kind='dataset', data_type='actual', adapter='reserve_coefficients', unit='коэффициент', full_snapshot=True)
    items.extend(fx_context_catalogue())
    return items


def number(text):
    value = str(text).replace('\xa0', '').replace(' ', '').replace(',', '.').replace('−', '-')
    if value in {'', '-', '—', '–', 'nan', 'None'}:
        return None
    if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?', value):
        raise ValueError('CBR numeric schema changed')
    return float(value)


def day(text):
    text = str(text).strip()
    if re.fullmatch(r'\d{2}\.\d{2}\.\d{4}', text):
        return pd.to_datetime(text, format='%d.%m.%Y').date().isoformat()
    if re.fullmatch(r'\d{2}\.\d{4}', text):
        return pd.to_datetime(text, format='%m.%Y').date().isoformat()
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
        return date.fromisoformat(text).isoformat()
    months = {'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6,
              'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12}
    match = re.fullmatch(r'([А-Яа-я]+)\s+(\d{4})(?:\s*г\.?)?', text)
    if match and match[1].lower() in months:
        return date(int(match[2]), months[match[1].lower()], 1).isoformat()
    if re.fullmatch(r'(?:19|20)\d{2}', text):
        return text + '-01-01'
    return None


def contracts(text):
    result = []
    for index, table in enumerate(page(text).tables):
        if table and any(re.fullmatch(r'(?:19|20)\d{2}', label) for label in table[0]):
            years = [i for i, label in enumerate(table[0]) if re.fullmatch(r'(?:19|20)\d{2}', label)]
            result.append({'table': index, 'headers': table[0], 'layout': 'wide', 'numeric': years,
                           'dimensions': [i for i in range(len(table[0])) if i not in years]})
            continue
        first = next((i for i, row in enumerate(table) if row and day(row[0])), None)
        if first is None or first == 0:
            continue
        width = max(len(row) for row in table[:first])
        headers = []
        for col in range(width):
            parts = list(dict.fromkeys(row[col] for row in table[:first] if len(row) > col and row[col]))
            headers.append(' / '.join(parts))
        result.append({'table': index, 'headers': headers})
    return result


def parse_table(text, contract):
    tables = page(text).tables
    index = contract['table']
    if index >= len(tables):
        raise ValueError('CBR table disappeared')
    current = next((item for item in contracts(text) if item['table'] == index), None)
    if current is None and tables[index]:
        # A valid date-filtered response can contain only the original header.
        # Validate every header cell; an arbitrary/changed empty page still fails.
        table = tables[index]
        width = max(len(row) for row in table)
        headers = [' / '.join(dict.fromkeys(row[col] for row in table if len(row) > col and row[col]))
                   for col in range(width)]
        if headers == contract['headers']:
            return []
    if current is not None and contract.get('layout') == 'wide':
        dimensions = [label for label in current['headers'] if not re.fullmatch(r'(?:19|20)\d{2}', label)]
        expected = [contract['headers'][i] for i in contract['dimensions']]
        if dimensions != expected:
            raise ValueError('CBR wide table dimensions changed')
        contract = {**contract, **current}
    if current is None or current['headers'] != contract['headers']:
        raise ValueError('CBR table headers changed')
    output = []
    if contract.get('layout') == 'wide':
        for row in tables[index][1:]:
            if len(row) != len(contract['headers']):
                raise ValueError('CBR wide table width changed')
            dimensions = ' | '.join(contract['headers'][i] + '=' + row[i] for i in contract['dimensions'])
            for col in contract['numeric']:
                output.append({'date': day(contract['headers'][col]), 'column': 'Значение',
                               'dimensions': dimensions, 'value': number(row[col])})
        return output
    for row in tables[index]:
        observed = day(row[0]) if row else None
        if observed is None:
            continue
        if row[1:] and all(cell == row[1] for cell in row[1:]) and 'Аукцион не состоялся' in row[1]:
            # An officially cancelled auction is an event, not a zero-valued transaction.
            continue
        if len(row) != len(contract['headers']):
            raise ValueError('CBR table width changed')
        dimensions = ' | '.join(contract['headers'][i] + '=' + row[i] for i in contract.get('dimensions', []))
        for col in contract.get('numeric', range(1, len(contract['headers']))):
            header = contract['headers'][col]
            record = {'date': observed, 'column': header, 'dimensions': dimensions, 'value': number(row[col])}
            if contract.get('published_column') is not None:
                published = day(row[contract['published_column']])
                record['published_at'] = published + 'T23:59:59+03:00' if published else None
                record['quality'] = 'publication_date_only'
            output.append(record)
    return output


class Transport:
    def __init__(self, *, session=None, sleep=time.sleep, clock=time.monotonic):
        self.session = session or requests.Session()
        self.sleep, self.clock, self.last = sleep, clock, {}
        self.cache = {}

    def get(self, url, params=None):
        return self.request('get', url, params=params)

    def post(self, url, body, headers):
        return self.request('post', url, data=body.encode('utf-8'), headers=headers)

    def request(self, method, url, **kwargs):
        host = urlsplit(url).hostname
        if urlsplit(url).scheme != 'https' or host not in {'cbr.ru', 'www.cbr.ru', 'iss.moex.com', 'fred.stlouisfed.org'}:
            raise ValueError('Unsupported market endpoint')
        key = (method, url, repr(sorted(kwargs.items())))
        host = 'cbr' if host in {'cbr.ru', 'www.cbr.ru'} else host
        if key in self.cache:
            return self.cache[key]
        delay = 1 - (self.clock() - self.last.get(host, -float('inf')))
        if delay > 0:
            self.sleep(delay)
        self.last[host] = self.clock()
        response = getattr(self.session, method)(url, timeout=(10, 45), allow_redirects=False, **kwargs)
        response.raise_for_status()
        if 300 <= response.status_code < 400 or len(response.content) > 15_000_000:
            raise ValueError('Invalid market response')
        if len(self.cache) >= 16:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = response
        return response


def xml_root(content):
    if len(content) > 15_000_000 or b'<!DOCTYPE' in content.upper() or b'<!ENTITY' in content.upper():
        raise ValueError('Invalid CBR XML')
    return ElementTree.fromstring(content)


def parse_currency(content):
    rows = []
    root = xml_root(content)
    if root.tag != 'ValCurs':
        raise ValueError('CBR currency XML root changed')
    for node in root.findall('Record'):
        nominal = number(node.findtext('Nominal'))
        if nominal is None or nominal <= 0:
            raise ValueError('Invalid currency nominal')
        rows.append({'date': day(node.attrib['Date']), 'value': number(node.findtext('Value')) / nominal})
    return rows


def moex_rows(payload, field):
    block = payload.get('history') or payload.get('candles')
    if not isinstance(block, dict) or field not in block.get('columns', []):
        raise ValueError('ISS columns changed or data unavailable')
    names = block['columns']
    out = []
    for values in block['data']:
        if len(values) != len(names):
            raise ValueError('ISS row width changed')
        row = dict(zip(names, values))
        observed = row.get('TRADEDATE') or str(row.get('begin', ''))[:10]
        out.append({'date': date.fromisoformat(observed).isoformat(), 'value': row[field],
                    'quality': 'delayed_publication_unknown'})
    return out


def metric_id(dataset, label):
    return dataset + '.' + hashlib.sha256(label.encode()).hexdigest()[:12]


def reviewed_table_metadata(item, child):
    """Apply declared per-table semantics without changing historical identities."""
    index = child.get('table_identity')
    if index is None or not 0 <= index < len(item.get('contracts', [])):
        return child
    contract = item['contracts'][index]
    result = dict(child)
    for field in ('unit', 'aggregation', 'field'):
        if field in contract:
            result[field] = contract[field]
    labels = item.get('table_display_labels', [])
    if index < len(labels):
        result['title'] = labels[index] + ' · ' + child['column']
        if child.get('dimensions'):
            result['title'] += ' | ' + child['dimensions']
    return result


def fetch(item, transport, start, end):
    kind = item.get('adapter', 'html')
    if kind in {'fred_csv', 'balance_snapshot'}:
        from .dollar_sources import fetch as fx_context_fetch
        return fx_context_fetch(item, transport, start, end)
    if kind == 'transfers':
        if any(not re.fullmatch(r'CC_\d+', key) or value != 'on' for key, value in item.get('params', {}).items()):
            raise ValueError('Параметры архива переводов: CC_КОД=on; направление и показатель выбираются набором')
        first, last = max(start, date(2006, 1, 1)), min(end, date(2021, 12, 31))
        if first > last:
            return []
        params = {'Direction': item['direction'], 'IsYearSum': '0', 'PokGroup': str(item['transfer_group']),
                  'PokName': item['transfer_field'], 'FromDateKv': f'{(first.month-1)//3+1}.{first.year}',
                  'ToDateKv': f'{(last.month-1)//3+1}.{last.year}', 'CC_1000': 'on', **item.get('params', {})}
        text = transport.request('post', item['url'], data=params).text
        parsed = page(text)
        if len(parsed.tables) != 1 or parsed.tables[0][0][0] != 'Страна':
            raise ValueError('CBR transfer table structure changed')
        table = parsed.tables[0]
        dates = []
        quarters = {'I': 1, 'II': 4, 'III': 7, 'IV': 10}
        for header in table[0][1:]:
            match = re.fullmatch(r'(I|II|III|IV) квартал (\d{4}) г\.', header)
            if not match:
                raise ValueError('CBR transfer period changed')
            dates.append(date(int(match[2]), quarters[match[1]], 1))
        output = []
        for row in table[1:]:
            if len(row) != len(dates)+1:
                raise ValueError('CBR transfer row width changed')
            child = {**item, 'id': metric_id(item['id'], row[0]), 'kind': 'series', 'dataset': item['id'],
                     'title': item['title'] + ': ' + row[0], 'column': row[0], 'dimensions': 'Страна=' + row[0],
                     'data_type': 'archive'}
            observations = [{'date': stamp.isoformat(), 'period_end': (pd.Timestamp(stamp)+pd.offsets.QuarterEnd()).date().isoformat(),
                             'value': number(value), 'definition': item.get('definition', '1')}
                            for stamp, value in zip(dates, row[1:])]
            output.append((child, observations))
        return output
    if kind == 'ruonia':
        # The existing official SOAP adapter includes a rate and transaction volume.
        from .cbr_data import parse_rates
        body = ('<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body>'
                '<Ruonia xmlns="http://web.cbr.ru/"><fromDate>' + start.isoformat() + 'T00:00:00</fromDate>'
                '<ToDate>' + end.isoformat() + 'T00:00:00</ToDate></Ruonia></soap:Body></soap:Envelope>')
        response = transport.post('https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx', body,
            {'Content-Type': 'text/xml; charset=utf-8', 'SOAPAction': 'http://web.cbr.ru/Ruonia'})
        frame = parse_rates(response.content)
        field = 'Ставка RUONIA' if item['field'] == 'rate' else 'Объём RUONIA'
        return [(item, [{'date': row['Дата'], 'value': row[field]} for row in frame.to_dict('records')])]
    if kind == 'currency':
        content = transport.get('https://cbr.ru/scripts/XML_dynamic.asp',
            {'date_req1': start.strftime('%d/%m/%Y'), 'date_req2': end.strftime('%d/%m/%Y'), 'VAL_NM_RQ': item['code']}).content
        observations = parse_currency(content)
        if item['id'] == 'cbr.fx.usd':
            for observation in observations:
                if observation['date'] < '1998-01-01':
                    observation['value'] /= 1000
                    observation['quality'] = 'redenominated_1000_to_1_publication_unknown'
        return [(item, observations)]
    if kind == 'moex':
        if not item.get('board') and item.get('market', 'index') != 'index':
            description = transport.get('https://iss.moex.com/iss/securities/' + item['security'] + '.json', {'iss.meta': 'off'}).json()
            boards = description.get('boards', {})
            choices = [dict(zip(boards.get('columns', []), row)) for row in boards.get('data', [])]
            primary = next((row.get('boardid') for row in choices if row.get('is_primary') == 1 and row.get('market') == item['market']), None)
            if not primary:
                raise ValueError('Specify an ISS board explicitly')
            item = {**item, 'board': primary, 'url': item['url'].replace('/securities/', '/boards/' + primary + '/securities/')}
        rows, offset = [], 0
        while True:
            response = transport.get(item['url'], {'from': start.isoformat(), 'till': end.isoformat(), 'start': offset, 'iss.meta': 'off'})
            batch = moex_rows(response.json(), item['field'])
            rows.extend(batch)
            if not batch:
                break
            offset += len(batch)
            if offset > 100_000:
                raise ValueError('ISS pagination limit')
        return [(item, rows)]
    if item.get('available_end'):
        end = min(end, date.fromisoformat(item['available_end']))
    if start > end:
        return []
    date_format = item.get('date_format', '%m.%Y' if '/mrrf/' in item['url'] else '%d.%m.%Y')
    params = {'UniDbQuery.Posted': 'True', 'UniDbQuery.From': start.strftime(date_format), 'UniDbQuery.To': end.strftime(date_format),
              **item.get('default_params', {}), **item.get('params', {})}
    text = transport.get(item['url'], None if item.get('full_snapshot') else params).text
    output = []
    if kind == 'reserve_coefficients':
        parsed = page(text)
        for table_index, table in enumerate(parsed.tables[:2]):
            for row in table:
                if not row:
                    continue
                match = re.match(r'(\d{2}\.\d{2}\.\d{4})\s*[–—-]\s*(\d{2}\.\d{2}\.\d{4})?', row[0])
                if not match:
                    continue
                for column, cell in enumerate(row[1:], 1):
                    if re.fullmatch(r'[\d\s\xa0,\.−+-]+', cell):
                        label = (parsed.table_titles[table_index] or 'Коэффициент') + f' · колонка {column}'
                        child = {**item, 'id': metric_id(item['id'], label), 'dataset': item['id'], 'kind': 'series', 'column': label, 'title': label}
                        observed = day(match[1])
                        values = [{'date': observed, 'period_end': day(match[2]) if match[2] else observed,
                                   'value': number(cell), 'definition': item.get('definition', '1')}]
                        previous = next((values for meta, values in output if meta['id'] == child['id']), None)
                        if previous is not None:
                            previous.extend(values)
                        else:
                            output.append((child, values))
        if not output:
            raise ValueError('CBR reserve coefficient structure changed')
        return output
    if kind == 'official_forecast':
        from html import unescape
        plain = ' '.join(unescape(re.sub('<[^>]+>', ' ', text)).split())
        match = re.search(r'За период с\s+(\d{2}\.\d{2}\.\d{4})\s+по\s+(\d{2}\.\d{2}\.\d{4})', plain)
        if not match:
            raise ValueError('CBR official forecast period changed')
        first, last = day(match[1]), day(match[2])
        tables = page(text).tables
        rows = [row for table in tables for row in table if len(row) == 2 and row[0] and re.fullmatch(r'[\d\s\xa0,\.−+-]+', row[1])]
        if not rows:
            raise ValueError('CBR official forecast table changed')
        for label, value in rows:
            child = {**item, 'id': metric_id(item['id'], label), 'dataset': item['id'],
                     'title': item['title'] + ': ' + label, 'kind': 'series', 'column': label}
            output.append((child, [{'date': first, 'period_end': last, 'value': number(value),
                                   'quality': 'official_forecast_publication_unknown', 'definition': item.get('definition', '1')}]))
        return output
    if item.get('kind') == 'reference':
        parsed = page(text)
        snapshot = {'tables': parsed.tables, 'links': parsed.links, 'headings': parsed.headings, 'url': item['url']}
        return [({**item, '_snapshot': snapshot}, [])]
    if not item.get('contracts'):
        raise ValueError('No verified numeric table contract')
    if (not page(text).tables and
            'За выбранный вами период нет информации.' in text and 'name="UniDbQuery.From"' in text):
        # Exact official date-filtered empty response, not a schema failure.
        return []
    for contract in item['contracts']:
        rows = parse_table(text, contract)
        for column, dimensions in dict.fromkeys((row['column'], row.get('dimensions', '')) for row in rows):
            label = column + (' | ' + dimensions if dimensions else '')
            table_identity = contract['table'] if len(item['contracts']) > 1 else None
            if table_identity is not None:
                # Headings can contain the selected dates; identity must stay stable across requests.
                table_label = item.get('table_labels', [])[table_identity] if table_identity < len(item.get('table_labels', [])) else f'Таблица {table_identity+1}'
                label = table_label + ': ' + label
            if item.get('params'):
                label += ' | ' + json.dumps(item['params'], ensure_ascii=False, sort_keys=True)
            child = {**item, 'id': metric_id(item['id'], label), 'dataset': item['id'],
                     'title': item['title'] + ': ' + label, 'kind': 'series', 'column': column, 'dimensions': dimensions}
            if table_identity is not None:
                child['table_identity'] = table_identity
                child['table_context'] = page(text).table_titles[table_identity]
            if contract.get('layout') == 'wide':
                child['frequency'] = 'annual'
            # Units are declared by the source contract, never inferred from a value.
            child['unit'] = contract.get('units', {}).get(column, item.get('unit', 'source_units'))
            if any(word in column.lower() for word in ('количество', 'число ')):
                child['unit'] = 'единиц'
            if '%' in column or 'ставк' in column.lower():
                child['unit'] = '%'
            if 'б.п.' in column:
                child['unit'] = 'б.п.'
            if 'в календарных днях' in column or column.lower().startswith('срок'):
                child['unit'] = 'дней'
            compact = column.lower().replace(' ', '')
            if 'млрд' in compact and 'руб' in compact:
                child['unit'] = 'млрд руб'
            elif 'млн' in compact and 'руб' in compact:
                child['unit'] = 'млн руб'
            elif 'млн' in compact and ('доллар' in compact or 'usd' in compact) or 'млн' in compact and 'долл.' in compact:
                child['unit'] = 'млн долларов США'
            elif 'млн' in compact and 'юан' in compact:
                child['unit'] = 'млн юаней'
            elif 'млн' in compact and 'евро' in compact:
                child['unit'] = 'млн евро'
            elif 'млрдед.инвалюты' in compact:
                child['unit'] = 'млрд единиц валюты'
            elif 'млнед.инвалюты' in compact:
                child['unit'] = 'млн единиц валюты'
            elif 'курс' in compact and 'руб' in compact:
                child['unit'] = 'руб за единицу валюты'
            elif 'своп-разница(руб.)' in compact or compact.endswith('(руб.)'):
                child['unit'] = 'руб'
            elif 'курс' in compact and 'usd/rur' in compact:
                child['unit'] = 'руб за доллар США'
            elif 'объем' in compact and child['unit'] == '%':
                child['unit'] = 'source_units'
            if item['id'] == 'cbr.hd_base.keyrate':
                child['unit'], child['administrative'] = '%', True
            if item['id'] == 'cbr.hd_base.seldomc.sc_dynamic':
                child['unit'] = 'долларов США за единицу валюты' if 'Обратная' in dimensions else 'единиц валюты за доллар США'
                parameters = {**item.get('default_params', {}), **item.get('params', {})}
                currency = next((choice['label'] for choice in item.get('parameter_choices', {}).get('UniDbQuery.ValID', [])
                                 if choice['value'] == parameters.get('UniDbQuery.ValID')), None)
                if currency:
                    child['title'] = currency + ': ' + label
            if item['id'] == 'cbr.hd_base.infl':
                child['administrative'] = 'ключев' in column.lower() or 'цел' in column.lower()
            if child['unit'] == '%' and 'ruonia' not in item['id']:
                child['aggregation'] = 'mean'
            match = re.search(r'Единица измерения=([^|]+)', dimensions)
            if match:
                child['unit'] = match[1].strip()
            if item['id'] == 'cbr.hd_base.zcyc_params':
                child['maturity'] = float(column.split('/')[-1].strip().replace(',', '.'))
            child = reviewed_table_metadata(item, child)
            output.append((child, [{**row, 'definition': item.get('definition', '1')} for row in rows
                                   if row['column'] == column and row.get('dimensions', '') == dimensions and start.isoformat() <= row['date'] <= end.isoformat()]))
    events = [{'date': day(row[0]), 'description': row[1]} for table in page(text).tables for row in table
              if row and day(row[0]) and len(row) > 1 and 'Аукцион не состоялся' in row[1]]
    if events and output:
        output[0][0]['_snapshot'] = {'events': events, 'url': item['url']}
    return output
