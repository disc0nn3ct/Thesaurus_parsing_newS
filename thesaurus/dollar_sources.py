"""Free official FX context: FRED CSV downloads and CBR quarterly accounts."""
import csv
import io
import math
import re
from datetime import date

import pandas as pd

FRED = 'https://fred.stlouisfed.org/graph/fredgraph.csv'
BALANCE = 'https://www.cbr.ru/eng/statistics/macro_itm/external_sector/pb/p_balance/'
TRADE = {'Current account': ('current_account', 'Текущий счёт'),
         'Trade balance': ('trade_balance', 'Торговый баланс'),
         'Exports of goods': ('exports', 'Экспорт товаров'),
         'Imports of goods': ('imports', 'Импорт товаров'),
         'Balance on services': ('services', 'Баланс услуг')}
PACK = ['cbr.fx.usd', 'cbr.fx.eur', 'cbr.fx.cny', 'cbr.ruonia.rate',
        'cbr.hd_base.infl', 'cbr.hd_base.keyrate', 'cbr.hd_base.bliquidity', 'cbr.hd_base.valintbr.valvol_day',
        'cbr.hd_base.mrrf.mrrf_7d', 'cbr.fx_context.balance',
        'fred.brent', 'fred.us_rate', 'fred.usd_broad']


def catalogue():
    items = []
    for identity, code, title, unit, calendar, transform, lag in (
        ('brent', 'DCOILBRENTEU', 'Нефть Brent · спот EIA', 'USD за баррель', 'weekdays', 'returns', 7),
        ('us_rate', 'DFF', 'Эффективная ставка федеральных фондов США', '%', 'calendar', 'difference', 3),
        ('usd_broad', 'DTWEXBGS', 'Широкий индекс доллара США (не DXY)', 'индекс, январь 2006=100', 'weekdays', 'returns', 10),
    ):
        items.append({'id': 'fred.'+identity, 'source': 'fred', 'dataset': 'fred.'+identity,
                      'title': title, 'kind': 'series', 'adapter': 'fred_csv', 'fred_code': code,
                      'frequency': 'daily', 'calendar': calendar, 'aggregation': 'last',
                      'transform': transform, 'unit': unit, 'definition': '1',
                      'url': 'https://fred.stlouisfed.org/series/'+code,
                      'methodology': 'https://fred.stlouisfed.org/help/data/downloading/using-the-download-data-link/',
                      'release_lag_days': lag, 'description': 'Официальная CSV-выгрузка без ключа. Даты исторических публикаций неизвестны; только контекст, не признаки прогноза USD.'})
    items.append({'id': 'cbr.fx_context.balance', 'source': 'cbr', 'dataset': 'cbr.fx_context.balance',
                  'title': 'Платёжный баланс · текущая публикация ЦБ', 'kind': 'dataset',
                  'adapter': 'balance_snapshot', 'frequency': 'quarterly', 'calendar': 'calendar',
                  'catalog_group': 'external',
                  'aggregation': 'sum', 'transform': 'difference', 'unit': 'млрд USD',
                  'definition': '1', 'url': BALANCE, 'snapshot_current': True,
                  'release_lag_days': 190, 'period_semantics': 'flow', 'description': 'Квартальные потоки; текущая таблица, а не полная историческая база. Годовые итоги исключены.'})
    return items


def parse_fred(content, code):
    text = content.decode('utf-8-sig') if isinstance(content, bytes) else content
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames not in (['observation_date', code], ['DATE', code]):
        raise ValueError('FRED CSV headers changed')
    rows, seen = [], set()
    for record in reader:
        day = date.fromisoformat(record[reader.fieldnames[0]])
        if day in seen or None in record or record[code] is None:
            raise ValueError('FRED CSV duplicate or width changed')
        seen.add(day)
        raw = record[code].strip()
        value = None if raw in {'', '.'} else float(raw)
        if value is not None and not math.isfinite(value):
            raise ValueError('FRED CSV nonfinite value')
        rows.append({'date': day.isoformat(), 'value': value, 'quality': 'publication_unknown'})
    return rows


def parse_balance(item, text):
    from .market_sources import number, page
    parsed = page(text)
    tables = [table for table in parsed.tables if len(table) > 3 and table[1][0] == 'Key Aggregates']
    if len(tables) != 1:
        raise ValueError('CBR balance table changed')
    table = tables[0]
    if len(table[0]) != len(table[1]):
        raise ValueError('CBR balance year headers changed')
    columns = []
    for i, (year, quarter) in enumerate(zip(table[0][1:], table[1][1:]), 1):
        if not re.fullmatch(r'20\d{2}', year):
            raise ValueError('CBR balance year changed')
        if quarter == year:
            continue  # Annual sum must never masquerade as a quarter.
        if not re.fullmatch('Q[1-4]', quarter):
            raise ValueError('CBR balance quarter changed')
        columns.append((i, date(int(year), 3*(int(quarter[1])-1)+1, 1)))
    if len({stamp for _, stamp in columns}) != len(columns) or not columns:
        raise ValueError('CBR balance duplicate periods')
    output = []
    for name, (key, title) in TRADE.items():
        matches = [row for row in table[2:] if row and row[0] == name]
        if len(matches) != 1 or len(matches[0]) != len(table[0]):
            raise ValueError('CBR balance numeric row changed')
        row = matches[0]
        metadata = {**item, 'id': item['id']+'.'+key, 'kind': 'series', 'dataset': item['id'],
                    'title': title, 'column': name, 'data_type': 'actual'}
        observations = [{'date': stamp.isoformat(), 'period_end': (pd.Timestamp(stamp)+pd.offsets.QuarterEnd()).date().isoformat(),
                         'value': number(row[i]), 'quality': 'publication_unknown'} for i, stamp in columns]
        output.append((metadata, observations))
    return output


def fetch(item, transport, start, end):
    if item['adapter'] == 'fred_csv':
        response = transport.get(FRED, {'id': item['fred_code'], 'cosd': start.isoformat(), 'coed': end.isoformat()})
        if not response.content.strip() and response.headers.get('Content-Type', '').split(';')[0] in {'text/csv', 'application/csv'}:
            return [(item, [])]  # Official response to an interval with no publications.
        rows = parse_fred(response.content, item['fred_code'])
        return [(item, [r for r in rows if start.isoformat() <= r['date'] <= end.isoformat()])]
    response = transport.get(item['url'])
    return parse_balance(item, response.text)
