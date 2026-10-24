"""DB-first RUONIA ingestion shared by reports and market jobs."""
from datetime import date, datetime, timedelta, timezone
from io import BytesIO

import pandas as pd

from .cbr_data import RATE_URL, parse_rates
from .market_sources import metric_id
from .rate_forecasts import working_dates
from .store import utcnow

INDEX_METRICS = ('Индекс RUONIA', '1 мес', '3 мес', '6 мес')
RATE_METRICS = ('RUONIA', 'Объём RUONIA')
MARKET_IDS = dict(zip(RATE_METRICS, ('cbr.ruonia.rate', 'cbr.ruonia.volume')))
INDEX_DATASET = 'cbr.hd_base.ruonia.dynamic'
INDEX_COLUMNS = dict(zip(INDEX_METRICS, ('Индекс', '1 месяц', '3 месяца', '6 месяцев')))
MARKET_IDS.update({metric: metric_id(INDEX_DATASET, column) for metric, column in INDEX_COLUMNS.items()})


class IncompleteCollection(RuntimeError):
    """A bounded batch leaves additional missing periods for the next job."""


def unit(metric):
    return 'индекс' if metric == 'Индекс RUONIA' else 'млрд руб' if metric == 'Объём RUONIA' else '%'


def history(store, metrics):
    """Use either existing table, preserving the original retrieval timestamp."""
    placeholders = ','.join('?' for _ in metrics)
    rows = store.query("SELECT * FROM observations WHERE source='ruonia' AND region_id=225 "
                       "AND device='all' AND frequency='daily' AND metric IN (" + placeholders + ')', metrics)
    by_key = {(row['metric'], row['date']): row for row in rows}
    for metric in metrics:
        identity = MARKET_IDS.get(metric)
        if not identity:
            continue
        for row in store.query('SELECT * FROM market_observations WHERE series_id=?', (identity,)):
            key = (metric, row['date'])
            if key not in by_key or by_key[key]['value'] is None:
                by_key[key] = {**row, 'source': 'ruonia', 'metric': metric, 'frequency': 'daily',
                               'unit': unit(metric),
                               'source_url': RATE_URL if metric in RATE_METRICS else 'https://cbr.ru/hd_base/ruonia/'}
    return by_key


def ranges(days, missing):
    """Group only consecutive missing calendar periods, never across stored data."""
    result = []
    first = last = None
    for day in days:
        if day in missing:
            if first is None:
                first = day
            last = day
        elif first is not None:
            result.append((first, last))
            first = last = None
    if first is not None:
        result.append((first, last))
    return result


def expected_dates(metrics, start, end):
    # Index/term series contain calendar days; overnight rate/volume use working days.
    grid = pd.date_range(start, end) if set(metrics) <= set(INDEX_METRICS) else working_dates(start, end)
    return [stamp.date() for stamp in grid]


def missing_ranges(store, metrics, start, end, now):
    days = expected_dates(metrics, start, end)
    stored = history(store, metrics)
    missing = set()
    for metric in metrics:
        state = store.get_setting('ruonia_availability:' + metric, {})
        floor = state.get('from')
        checks = []
        for check in state.get('empty', []):
            try:
                if datetime.fromisoformat(check['until']) > now:
                    checks.append((check['start'], check['end']))
            except (ValueError, KeyError, TypeError):
                continue
        for day in days:
            text = day.isoformat()
            if (stored.get((metric, text), {}).get('value') is not None
                    or stored.get((metric, text), {}).get('quality') == 'source_removed_confirmed'):
                continue
            if floor and text < floor:
                continue
            if any(first <= text <= last for first, last in checks):
                continue
            missing.add(day)
    return ranges(days, missing)


def remember_empty(store, metrics, start, end, now):
    """A valid empty response is publication delay, not a zero or network failure."""
    days = expected_dates(metrics, start, end)
    all_history = history(store, metrics)
    for metric in metrics:
        state = store.get_setting('ruonia_availability:' + metric, {})
        checks = [check for check in state.get('empty', []) if check['until'] > now.isoformat()]
        # Only a request reaching the configured initial boundary can establish
        # the beginning of source history; an internal gap cannot move it.
        if start <= date(2010, 1, 11):
            known = sorted(day for (name, day), row in all_history.items() if name == metric and row['value'] is not None)
            if known:
                state['from'] = known[0]
        missing = {day for day in days if all_history.get((metric, day.isoformat()), {}).get('value') is None
                   and (not state.get('from') or day.isoformat() >= state['from'])}
        for first, last in ranges(days, missing):
            checks.append({'start': first.isoformat(), 'end': last.isoformat(),
                           'until': (now + timedelta(hours=1)).isoformat()})
        state['empty'] = checks
        store.set_setting('ruonia_availability:' + metric, state)


def save_new(store, rows):
    """Never let bundled provider columns overwrite already stored observations."""
    unique = {}
    for row in rows:
        key = (row['metric'], row['date'])
        if key in unique and unique[key]['value'] != row['value']:
            raise ValueError('Ambiguous duplicate CBR observations')
        unique[key] = row
    rows = list(unique.values())
    existing = history(store, tuple({row['metric'] for row in rows})) if rows else {}
    fresh = [row for row in rows if row['value'] is not None
             and existing.get((row['metric'], row['date']), {}).get('value') is None]
    store.save_observations(fresh)
    return len(fresh)


def collect_rates(store, transport, start, end, now=None):
    now = now or datetime.now(timezone.utc)
    count = requests = 0
    batches = missing_ranges(store, RATE_METRICS, start, end, now)
    for first, last in batches[:64]:
        body = ('<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body>'
                '<Ruonia xmlns="http://web.cbr.ru/"><fromDate>' + first.isoformat()
                + 'T00:00:00</fromDate><ToDate>' + last.isoformat()
                + 'T00:00:00</ToDate></Ruonia></soap:Body></soap:Envelope>')
        response = transport.post('https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx', body,
                                  {'Content-Type': 'text/xml; charset=utf-8', 'SOAPAction': 'http://web.cbr.ru/Ruonia'})
        frame = parse_rates(response.content, allow_empty=True)
        rows = []
        for row in frame.to_dict('records'):
            if not first.isoformat() <= row['Дата'] <= last.isoformat():
                continue
            for metric, column in zip(RATE_METRICS, ('Ставка RUONIA', 'Объём RUONIA')):
                rows.append({'source': 'ruonia', 'metric': metric, 'date': row['Дата'], 'frequency': 'daily',
                             'value': row[column], 'unit': unit(metric),
                             'source_url': RATE_URL, 'retrieved_at': now.isoformat()})
        count += save_new(store, rows)
        requests += 1
        remember_empty(store, RATE_METRICS, first, last, now)
    copy_to_ordinary(store, RATE_METRICS)
    if len(batches) > 64:
        raise IncompleteCollection('Остальные пробелы будут догружены при следующем обновлении')
    return {'new_observations': count, 'requests': requests}


def copy_to_ordinary(store, metrics):
    """Import the shared cache without refreshing its age."""
    stored = list(history(store, metrics).values())
    ordinary = {(row['metric'], row['date']) for row in store.query(
        "SELECT metric,date FROM observations WHERE source='ruonia' AND value IS NOT NULL")}
    store.save_observations([row for row in stored if row['value'] is not None and (row['metric'], row['date']) not in ordinary])


def collect_index(store, transport, start, end, now=None):
    from .collectors import ruonia_rows
    now = now or datetime.now(timezone.utc)
    requests = count = 0
    batches = missing_ranges(store, INDEX_METRICS, start, end, now)
    for first, last in batches[:64]:
        response = transport.get('https://cbr.ru/Queries/UniDbQuery/DownloadExcel/125022', {
            'Posted': 'True', 'From': first.strftime('%d.%m.%Y'), 'To': last.strftime('%d.%m.%Y'),
            'I1': 'true', 'M1': 'true', 'M3': 'true', 'M6': 'true',
            'FromDate': first.strftime('%m/%d/%Y'), 'ToDate': last.strftime('%m/%d/%Y')})
        frame = pd.read_excel(BytesIO(response.content))
        # Empty files still need the verified table schema.
        if frame.empty:
            if not {'Дата', 'Индекс', '1 месяц', '3 месяца', '6 месяцев'}.issubset(frame.columns):
                raise ValueError('CBR table headers changed')
            rows = []
        else:
            rows = [row for row in ruonia_rows(frame) if first.isoformat() <= row['date'] <= last.isoformat()]
        for row in rows:
            row['retrieved_at'] = now.isoformat()
        count += save_new(store, rows)
        requests += 1
        remember_empty(store, INDEX_METRICS, first, last, now)
    copy_to_ordinary(store, INDEX_METRICS)
    if len(batches) > 64:
        raise IncompleteCollection('Остальные пробелы будут догружены при следующем обновлении')
    return {'new_observations': count, 'requests': requests}


def collect_report(store, transport, today, now=None):
    now = now or datetime.now(timezone.utc)
    end = today - timedelta(days=1)
    errors = []
    before = {key for key, row in history(store, INDEX_METRICS + RATE_METRICS).items() if row['value'] is not None}
    try:
        collect_index(store, transport, date(2010, 1, 11), end, now)
    except IncompleteCollection as exc:
        errors.append(str(exc))
    except Exception as exc:  # noqa: BLE001 - Never expose provider URLs or credentials.
        errors.append('Индекс и срочные ставки не обновлены (' + type(exc).__name__ + ')')
    try:
        collect_rates(store, transport, today - timedelta(days=365 * 3), end, now)
    except IncompleteCollection as exc:
        errors.append(str(exc))
    except Exception as exc:  # noqa: BLE001 - Keep successful index batches and stored rates.
        errors.append('Ставка и объём не обновлены (' + type(exc).__name__ + ')')
    # The collector consumes metadata, not a rewritten copy of all DB history.
    after = history(store, INDEX_METRICS + RATE_METRICS)
    count = len({key for key, row in after.items() if row['value'] is not None} - before)
    frame = pd.DataFrame()
    frame.attrs.update(stored_incrementally=True, cached=count == 0, new_observations=count,
                       collection_errors=errors, data_retrieved_at=max(
                           (row['retrieved_at'] for row in after.values()
                            if row['value'] is not None), default=None), checked_at=utcnow())
    return frame
