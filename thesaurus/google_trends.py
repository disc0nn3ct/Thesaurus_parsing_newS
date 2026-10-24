"""Offline Google Trends snapshots. Website exports have independent scales."""
import calendar
import csv
import hashlib
import io
import json
import re
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from uuid import uuid4

from .store import dumps, utcnow

MAX_BYTES = 1024 * 1024
FREQUENCIES = {'day': 'daily', 'день': 'daily', 'week': 'weekly', 'неделя': 'weekly',
               'month': 'monthly', 'месяц': 'monthly'}
NOTE = ('Источник: Google Trends. Индекс интереса 0–100, не количество запросов. '
        'Разные выгрузки имеют разные шкалы; их история не склеивается. '
        'Обновление — неофициальным интерфейсом Google или импортом CSV. Автодоступ может ограничиваться Google.')


def active(store):
    return store.get_setting('search_provider') == 'google_trends'


def activate(store):
    """One-time reversible configuration migration, after taking the worker lock."""
    with store.connection() as conn:
        current = conn.execute("SELECT value FROM settings WHERE key='search_provider'").fetchone()
        if current and json.loads(current[0]) == 'google_trends':
            return
        backup = store.path.with_name(store.path.name + f'.trends-backup-{uuid4().hex}.sqlite3')
        with closing(sqlite3.connect(backup)) as target:
            conn.backup(target)
        conn.execute('BEGIN IMMEDIATE')
        conn.execute("UPDATE subscriptions SET enabled=0 WHERE section IN ('wordstat','regions')")
        conn.execute("UPDATE market_series SET enabled=0 WHERE source='wordstat'")
        conn.execute("UPDATE jobs SET state='cancelled',finished_at=?,error='search_provider_changed' "
                     "WHERE state='pending' AND kind='report' AND json_extract(payload,'$.section') IN ('wordstat','regions')", (utcnow(),))
        conn.execute("INSERT INTO settings(key,value) VALUES('search_provider',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (dumps('google_trends'),))


def parse_csv(raw, allowed_phrases, *, today=None):
    """Accept only time-series exports; validate the entire file before any write."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
        raise ValueError('Нужен CSV размером до 1 МБ.')
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeError:
        raise ValueError('CSV должен быть в UTF-8. Загрузите исходный файл Google Trends.') from None
    if '\x00' in text:
        raise ValueError('Неверный CSV.')
    # Export preamble is allowed, regional rankings and related queries are not.
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines[:20])
                   if re.split('[,;]', line, maxsplit=1)[0].strip(' "').lower() in FREQUENCIES), None)
    if header is None:
        raise ValueError('Скачайте CSV диаграммы «Динамика популярности», с колонкой День, Неделя или Месяц.')
    delimiter = ';' if lines[header].count(';') > lines[header].count(',') else ','
    try:
        table = list(csv.reader(io.StringIO('\n'.join(lines[header:])), delimiter=delimiter, strict=True))
    except csv.Error:
        raise ValueError('Неверная структура CSV. Загрузите исходный файл Google Trends.') from None
    names = table[0]
    frequency = FREQUENCIES[names[0].strip().lower()]
    if not 2 <= len(names) <= 9 or len(table) > 10001:
        raise ValueError('Допускается до 8 запросов и 10000 строк в одном CSV.')
    allowed = {p.casefold(): p for p in allowed_phrases}
    columns = []
    for label in names[1:]:
        match = re.fullmatch(r'(.+?):\s*\(([^()]+)\)', label.strip())
        if not match or match[1].strip().casefold() not in allowed:
            raise ValueError('Колонки должны соответствовать разрешённым фразам. Администратор добавляет фразы через /keywords add фраза.')
        region = match[2].strip()
        if len(region) > 100 or any(ord(c) < 32 for c in region):
            raise ValueError('Неверное название региона в CSV.')
        # Keep original localized region labels as provenance; never guess Yandex IDs.
        columns.append({'phrase': allowed[match[1].strip().casefold()], 'region': region, 'rows': []})
    if len({(c['phrase'], c['region']) for c in columns}) != len(columns):
        raise ValueError('В CSV повторяются колонки.')
    seen = set()
    today = today or date.fromisoformat(utcnow()[:10])
    for cells in table[1:]:
        if not cells or all(not c.strip() for c in cells):
            continue
        if len(cells) != len(names):
            raise ValueError('Число колонок в CSV изменилось.')
        try:
            stamp = date.fromisoformat(cells[0].strip() + ('-01' if frequency == 'monthly' and len(cells[0].strip()) == 7 else ''))
        except ValueError:
            raise ValueError('Неверная дата в CSV. Нужны даты YYYY-MM-DD или месяцы YYYY-MM.') from None
        if stamp in seen or stamp > today or (frequency == 'monthly' and stamp.day != 1):
            raise ValueError('В CSV повторяются даты или присутствуют будущие/неверные периоды.')
        seen.add(stamp)
        end = stamp + timedelta(days=6) if frequency == 'weekly' else date(stamp.year, stamp.month, calendar.monthrange(stamp.year, stamp.month)[1]) if frequency == 'monthly' else stamp
        for column, cell in zip(columns, cells[1:], strict=True):
            cell = cell.strip()
            quality, value = 'publication_unknown', None
            if cell in {'', '<1', '< 1'}:
                quality = 'missing' if not cell else 'below_resolution'
            elif re.fullmatch(r'\d{1,3}', cell) and 0 <= int(cell) <= 100:
                value = float(cell)
                if value == 0:
                    value, quality = None, 'low_volume_or_zero'
            else:
                raise ValueError('Индекс Google Trends должен быть целым числом 0–100 или <1.')
            if end >= today:
                value, quality = None, 'incomplete_period'
            column['rows'].append({'date': stamp.isoformat(), 'period_end': end.isoformat(), 'value': value, 'quality': quality})
    if not seen:
        raise ValueError('В CSV нет наблюдений.')
    for column in columns:
        column['rows'].sort(key=lambda r: r['date'])
        if frequency == 'weekly' and len({date.fromisoformat(r['date']).weekday() for r in column['rows']}) != 1:
            raise ValueError('Недельные периоды должны начинаться в один день недели.')
    return {'frequency': frequency, 'columns': columns, 'digest': hashlib.sha256(raw).hexdigest(),
            'export_context': '\n'.join(lines[:header])[:1000], 'export_columns': names[1:]}


def import_csv(store, raw, actor, *, acquisition='csv'):
    from .telegram_data_jobs import require_actor
    require_actor(store, actor, admin=True)
    from .search_provider import require_section
    require_section(store, 'trends')
    if acquisition not in {'csv', 'unofficial_api'}:
        raise ValueError('Unknown Trends acquisition method')
    parsed = parse_csv(raw, [r['phrase'] for r in store.keywords()])
    retrieved = utcnow()
    identities = []
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        user = conn.execute('SELECT enabled,role FROM users WHERE user_id=?', (actor,)).fetchone()
        if not user or not user['enabled'] or user['role'] != 'admin':
            raise PermissionError('Импорт доступен администратору.')
        for column in parsed['columns']:
            scope = hashlib.sha256(dumps([column['phrase'], column['region'], parsed['frequency']]).encode()).hexdigest()[:20]
            identity = f"google_trends.{scope}.{parsed['digest'][:20]}"
            identities.append(identity)
            if conn.execute('SELECT 1 FROM market_series WHERE id=?', (identity,)).fetchone():
                continue  # Replaying an old file never rolls back the active snapshot.
            old = conn.execute("SELECT id FROM market_series WHERE source='google_trends' AND dataset=? AND enabled=1", (scope,)).fetchall()
            conn.execute("UPDATE market_series SET enabled=0,status='archived' WHERE source='google_trends' AND dataset=?", (scope,))
            meta = {'id': identity, 'source': 'google_trends', 'dataset': scope,
                    'title': column['phrase'] + ' · ' + column['region'], 'kind': 'series', 'adapter': 'csv',
                    'frequency': parsed['frequency'], 'calendar': 'calendar', 'aggregation': 'mean',
                    'week_anchor': date.fromisoformat(column['rows'][0]['period_end']).strftime('%a').upper(),
                    'transform': 'difference', 'unit': 'индекс интереса (0–100)', 'value_bounds': [0, 100],
                    'phrase': column['phrase'], 'region_label': column['region'], 'snapshot': parsed['digest'],
                    'export_context': parsed['export_context'], 'export_columns': parsed['export_columns'],
                    'acquisition': acquisition,
                    'methodology': NOTE, 'url': 'https://trends.google.com/trends/', 'publication_known': False}
            conn.execute("INSERT INTO market_series(id,source,dataset,title,metadata,enabled,status,updated_at) VALUES(?,?,?,?,?,1,'loaded',?)",
                         (identity, 'google_trends', scope, meta['title'], dumps(meta), retrieved))
            for row in column['rows']:
                values = (identity, row['date'], row['value'], row['period_end'], None, retrieved, parsed['digest'], row['quality'])
                for table in ('market_observations', 'market_revisions'):
                    conn.execute(f'INSERT INTO {table}(series_id,date,value,period_end,published_at,retrieved_at,definition,quality) VALUES(?,?,?,?,?,?,?,?)', values)
            old_ids = {r['id'] for r in old}
            if old_ids:
                # Saved personal selections follow the new snapshot; old forecasts stay immutable.
                for setting in conn.execute("SELECT key,value FROM settings WHERE key LIKE 'market_selection:%' OR key LIKE 'chart:%' OR key LIKE 'market_pairs:%' OR key LIKE 'market_features:%'").fetchall():
                    value = json.loads(setting['value'])
                    if isinstance(value, list):
                        if setting['key'].startswith('market_pairs:'):
                            value = [{**pair, **{side: identity if pair.get(side) in old_ids else pair.get(side) for side in ('left', 'right')}} for pair in value]
                        else:
                            value = list(dict.fromkeys(identity if x in old_ids else x for x in value))
                    elif isinstance(value, dict) and value.get('identity') in old_ids:
                        value['identity'] = identity
                    conn.execute('UPDATE settings SET value=? WHERE key=?', (dumps(value), setting['key']))
                    if setting['key'].startswith('market_features:') and setting['key'].removeprefix('market_features:') in old_ids:
                        conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', ('market_features:' + identity, dumps(value)))
                # A new normalization must not trigger an old threshold/correlation rule.
                for old_id in old_ids:
                    conn.execute('UPDATE market_rules SET enabled=0 WHERE series_id=?', (old_id,))
        conn.execute("INSERT INTO access_audit(actor_id,target_id,action,created_at) VALUES(?,?,?,?)", (actor, actor, 'google_trends_import', retrieved))
    return identities


def run_import_job(store, job, payload, *, dry_run=False):
    import base64

    from .delivery import prepare_delivery
    from .telegram_data_jobs import require_actor
    actor = payload['actor_id']
    require_actor(store, actor, admin=True)
    fingerprint = f"trends-import:{job['id']}"
    existing = store.one('SELECT id FROM reports WHERE fingerprint=?', (fingerprint,))
    if existing:
        identifier = existing['id']
    else:
        encoded = payload['csv']
        if not isinstance(encoded, str) or len(encoded) > MAX_BYTES * 2:
            raise ValueError('Слишком большой CSV.')
        raw = base64.b64decode(encoded, validate=True)
        if dry_run:
            parse_csv(raw, [r['phrase'] for r in store.keywords()])
            return None
        identities = import_csv(store, raw, actor)
        identifier = uuid4().hex
        text = f'Google Trends: проверено {len(identities)} рядов.\n{NOTE}\nГрафики: /chart trends. Проверка прогноза и корреляции: /indicators, /backtest market, /correlations.'
        store.execute("INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,quality,parts,fingerprint) VALUES(?,'trends',?,?,?,?,?,?,?,?)",
                      (identifier, utcnow(), store.version('keyword'), store.version('cohort'), text, text, dumps({'actor_id': actor}), dumps([]), fingerprint))
    if not dry_run:
        require_actor(store, actor, admin=True)
        prepare_delivery(store, identifier, str(actor), detail='full')
    return identifier
