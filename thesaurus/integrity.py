"""Resumable source audits. Corrections are staged, never silently applied."""
import hashlib
import json
from datetime import date, datetime, timedelta, timezone

from . import market_sources as sources
from . import market_store as repository
from .store import dumps, utcnow
from .telegram_data_jobs import require_actor

FIELDS = ('date', 'value', 'period_end', 'published_at', 'definition', 'quality')


def incident(store, kind, *, check_id=None, series_id=None, day=None, old=None, new=None, severity='warning'):
    fingerprint = hashlib.sha256(dumps([kind, series_id, day, old, new]).encode()).hexdigest()
    store.execute('INSERT OR IGNORE INTO integrity_incidents(fingerprint,check_id,series_id,date,kind,severity,old_payload,new_payload,created_at) VALUES(?,?,?,?,?,?,?,?,?)',
                  (fingerprint, check_id, series_id, day, kind, severity, dumps(old), dumps(new), utcnow()))
    return store.one('SELECT id FROM integrity_incidents WHERE fingerprint=?', (fingerprint,))['id']


def candidates(store, check_id, series_id, rows, *, start, end, complete=False):
    """Compare canonical values; a valid empty response alone proves no deletions."""
    old = {row['date']: row for row in repository.history(store, series_id) if start <= row['date'] <= end}
    received = {}
    for row in rows:
        if not start <= row['date'] <= end:
            continue
        if row['date'] in received and row.get('value') != received[row['date']].get('value'):
            raise ValueError('Ambiguous audit observations')
        received[row['date']] = row
    for day, row in received.items():
        if day not in old:
            continue  # New data belongs to incremental collection.
        previous = {key: old[day].get(key) for key in FIELDS}
        current = {**previous, **{key: row[key] for key in FIELDS if key in row}}
        current['period_end'] = current.get('period_end') or day
        if previous != current:
            incident(store, 'source_revision', check_id=check_id, series_id=series_id, day=day, old=previous, new=current)
    if complete and received:
        for day in old.keys() - received.keys():
            previous = {key: old[day].get(key) for key in FIELDS}
            # Preserve a missing observation explicitly; never replace it by zero.
            incident(store, 'source_missing', check_id=check_id, series_id=series_id, day=day,
                     old=previous, new={**previous, 'value': None, 'quality': 'source_removed_confirmed'})


def enqueue(store, actor, selection=None, *, date_from=None, date_to=None):
    require_actor(store, actor, admin=True)
    start = date.fromisoformat(date_from).isoformat() if date_from else None
    end = date.fromisoformat(date_to).isoformat() if date_to else None
    if start and end and start > end:
        raise ValueError('Начало периода позже конца')
    loaded = {row['series_id'] for row in store.query('SELECT DISTINCT series_id FROM market_observations UNION SELECT dataset AS series_id FROM source_snapshots')}
    selected = list(dict.fromkeys(selection if selection is not None else sorted(loaded)))
    if not selected:
        raise ValueError('Сначала загрузите данные выбранного набора')
    if not set(selected).issubset(loaded):
        raise ValueError('Проверять можно уже загруженные показатели')
    with store.connection() as conn:
        cursor = conn.execute('INSERT INTO integrity_checks(actor_id,created_at,selection,date_from,date_to) VALUES(?,?,?,?,?)',
                              (actor, utcnow(), dumps(selected), start, end))
        check_id = cursor.lastrowid
        conn.execute("INSERT INTO jobs(kind,payload,dedupe_key,created_at) VALUES('integrity_audit',?,?,?)",
                     (dumps({'actor_id': actor, 'check_id': check_id}), f'integrity:{check_id}', utcnow()))
    return check_id


def _physical(store, row):
    meta = json.loads(row['metadata'])
    if meta.get('adapter') == 'stored_ruonia' or meta['dataset'] == 'local.ruonia':
        from .cbr_incremental import INDEX_DATASET, MARKET_IDS
        metric = meta.get('metric', row['title'])
        identity = MARKET_IDS.get(metric)
        if identity is None:
            return None
        parent = store.one('SELECT * FROM market_series WHERE id=?',
                           (identity if metric in {'RUONIA', 'Объём RUONIA'} else INDEX_DATASET,))
        return json.loads(parent['metadata']) if parent else None
    if meta.get('adapter') in {'derived', 'stored'} or meta.get('source') in {'wordstat', 'google_trends'}:
        return None
    if meta.get('kind') == 'series' and meta.get('adapter', 'html') in {'html', 'transfers', 'reserve_coefficients', 'official_forecast'}:
        parent = store.one('SELECT metadata FROM market_series WHERE id=?', (row['dataset'],))
        return {**json.loads(parent['metadata']), 'params': meta.get('params', {})} if parent else None
    return meta


def run(store, payload, *, dry_run=False, transport=None):
    require_actor(store, payload['actor_id'], admin=True)
    check = store.one('SELECT * FROM integrity_checks WHERE id=? AND actor_id=?', (payload['check_id'], payload['actor_id']))
    if not check or check['state'] in {'cancelled', 'done'}:
        return
    if dry_run:
        return
    store.execute("UPDATE integrity_checks SET state='running',error=NULL WHERE id=?", (check['id'],))
    checkpoint = json.loads(check['checkpoint'])
    transport = transport or sources.Transport()
    grouped = {}
    for identity in json.loads(check['selection']):
        row = store.one('SELECT * FROM market_series WHERE id=?', (identity,))
        item = _physical(store, row) if row else None
        if item is None:
            continue
        item = {**item, 'params': item.get('params', store.get_setting('market_params:' + item['id'], {}))}
        key = item['id'] + ':' + hashlib.sha256(dumps(item['params']).encode()).hexdigest()[:12]
        group = grouped.setdefault(key, {'item': item, 'members': []})
        group['members'].append(row)
    errors = []
    for group_key, group in grouped.items():
        if store.one('SELECT state FROM integrity_checks WHERE id=?', (check['id'],))['state'] == 'cancelled':
            return
        item = group['item']
        dataset = item['id']
        if item.get('kind') == 'reference':
            if dataset in checkpoint:
                continue
            try:
                today = datetime.now(timezone.utc).date()
                batches = sources.fetch(item, transport, today, today)
                snapshot = batches[0][0].get('_snapshot')
                accepted = store.get_setting('source_snapshot_current:' + dataset)
                previous = (store.one('SELECT * FROM source_snapshots WHERE dataset=? AND version=?', (dataset, accepted))
                            if accepted else store.one('SELECT * FROM source_snapshots WHERE dataset=? ORDER BY received_at DESC LIMIT 1', (dataset,)))
                if previous and snapshot is not None:
                    new = {'dataset': dataset, 'version': hashlib.sha256(dumps(snapshot).encode()).hexdigest(), 'payload': snapshot}
                    old = {'dataset': dataset, 'version': previous['version'], 'payload': json.loads(previous['payload'])}
                    if old != new:
                        incident(store, 'source_reference_revision', check_id=check['id'], series_id=dataset, old=old, new=new, severity='info')
                checkpoint[dataset] = True
                store.execute('UPDATE integrity_checks SET checkpoint=? WHERE id=?', (dumps(checkpoint), check['id']))
            except Exception as exc:  # noqa: BLE001 - No provider details in journal.
                errors.append(dataset + ': ' + type(exc).__name__)
                incident(store, 'source_unavailable', check_id=check['id'], series_id=dataset, new={'error': type(exc).__name__})
            continue
        dates = [obs['date'] for row in group['members'] for obs in repository.history(store, row['id'])]
        if not dates:
            continue
        first, last = max(min(dates), check['date_from'] or min(dates)), min(max(dates), check['date_to'] or max(dates))
        # Bounded chunks checkpoint independently; restart never re-fetches finished ranges.
        begin, finish = date.fromisoformat(first), date.fromisoformat(last)
        while begin <= finish:
            until = finish if item.get('full_snapshot') else min(finish, begin + timedelta(days=365))
            token = group_key + ':' + begin.isoformat() + ':' + until.isoformat()
            begin = until + timedelta(days=1)
            if token in checkpoint:
                continue
            current = store.one('SELECT state FROM integrity_checks WHERE id=?', (check['id'],))
            if current['state'] == 'cancelled':
                return
            try:
                start = date.fromisoformat(token.split(':')[-2])
                batches = sources.fetch(item, transport, start, until)
                for row in group['members']:
                    meta = json.loads(row['metadata'])
                    matching = next((values for child, values in batches if child['id'] == row['id'] or
                        meta.get('column') and child.get('column') == meta['column'] and child.get('dimensions', '') == meta.get('dimensions', '')
                        and child.get('table_identity') == meta.get('table_identity')), None)
                    if matching is None and meta.get('metric') in {'RUONIA', 'Объём RUONIA'}:
                        field = 'rate' if meta['metric'] == 'RUONIA' else 'volume'
                        matching = next((values for child, values in batches if child.get('field') == field), None)
                    if matching is None and meta.get('metric') in {'Индекс RUONIA', '1 мес', '3 мес', '6 мес'}:
                        from .cbr_incremental import INDEX_COLUMNS
                        matching = next((values for child, values in batches if child.get('column') == INDEX_COLUMNS[meta['metric']]), None)
                    if matching is None:
                        raise ValueError('Audit series disappeared from source contract')
                    candidates(store, check['id'], row['id'], matching, start=start.isoformat(), end=until.isoformat(), complete=True)
                checkpoint[token] = hashlib.sha256(dumps(batches).encode()).hexdigest()
                store.execute('UPDATE integrity_checks SET checkpoint=? WHERE id=?', (dumps(checkpoint), check['id']))
            except Exception as exc:  # noqa: BLE001 - No raw provider error can enter the incident journal.
                safe = 'Проверка набора не завершена (' + type(exc).__name__ + ')'
                errors.append(dataset + ': ' + safe)
                incident(store, 'source_schema' if isinstance(exc, ValueError) else 'source_unavailable',
                         check_id=check['id'], series_id=dataset, new={'error': safe})
                break
    store.execute("UPDATE integrity_checks SET state=?,error=? WHERE id=? AND state<>'cancelled'", ('partial' if errors else 'done', '\n'.join(errors) or None, check['id']))


def resolve(store, actor, identity, decision):
    require_actor(store, actor, admin=True)
    if decision not in {'accept', 'reject'}:
        raise ValueError('Решение: принять или отклонить')
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        user = conn.execute("SELECT enabled,role FROM users WHERE user_id=?", (actor,)).fetchone()
        if not user or not user['enabled'] or user['role'] != 'admin':
            raise PermissionError('Действие доступно администратору')
        entry = conn.execute('SELECT * FROM integrity_incidents WHERE id=?', (identity,)).fetchone()
        if not entry or entry['state'] not in {'new', 'review'}:
            return False
        if decision == 'accept':
            if entry['kind'] not in {'source_revision', 'source_missing', 'source_reference_revision'}:
                raise ValueError('Этот инцидент требует исправления источника или восстановления базы')
            old, new = json.loads(entry['old_payload']), json.loads(entry['new_payload'])
            if entry['kind'] == 'source_reference_revision':
                accepted = conn.execute('SELECT value FROM settings WHERE key=?', ('source_snapshot_current:' + new['dataset'],)).fetchone()
                current = (json.loads(accepted['value']) if accepted else
                           conn.execute('SELECT version FROM source_snapshots WHERE dataset=? ORDER BY received_at DESC LIMIT 1', (new['dataset'],)).fetchone()['version'])
                if current != old['version']:
                    raise ValueError('Снимок уже изменился. Выполните проверку заново')
                conn.execute('INSERT OR IGNORE INTO source_snapshots(dataset,version,received_at,payload) VALUES(?,?,?,?)',
                             (new['dataset'], new['version'], utcnow(), dumps(new['payload'])))
                conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                             ('source_snapshot_current:' + new['dataset'], dumps(new['version'])))
            else:
                current = conn.execute('SELECT * FROM market_observations WHERE series_id=? AND date=?', (entry['series_id'], entry['date'])).fetchone()
                if not current or any(current[key] != old.get(key) for key in FIELDS):
                    raise ValueError('Данные уже изменились. Выполните проверку заново')
                values = (entry['series_id'], new['date'], new.get('value'), new.get('period_end'), new.get('published_at'), utcnow(), new.get('definition', '1'), new.get('quality', 'publication_unknown'))
                conn.execute('INSERT INTO market_revisions(series_id,date,value,period_end,published_at,retrieved_at,definition,quality) VALUES(?,?,?,?,?,?,?,?)', values)
                conn.execute('UPDATE market_observations SET value=?,period_end=?,published_at=?,retrieved_at=?,definition=?,quality=? WHERE series_id=? AND date=?', (*values[2:], *values[:2]))
                # RUONIA has legacy and market aliases; apply one accepted revision everywhere.
                from .cbr_incremental import MARKET_IDS
                meta_row = conn.execute('SELECT metadata FROM market_series WHERE id=?', (entry['series_id'],)).fetchone()
                meta = json.loads(meta_row['metadata'])
                metric = next((name for name, key in MARKET_IDS.items() if key == entry['series_id']), meta.get('metric'))
                if metric and (entry['series_id'] in MARKET_IDS.values() or meta.get('dataset') == 'local.ruonia'):
                    conn.execute("UPDATE observations SET value=?,quality=?,retrieved_at=? WHERE source='ruonia' AND metric=? AND region_id=225 AND device='all' AND frequency='daily' AND date=?",
                                 (new.get('value'), new.get('quality', 'publication_unknown'), values[5], metric, new['date']))
                    aliases = {MARKET_IDS.get(metric), sources.metric_id('local.ruonia', metric)} - {None, entry['series_id']}
                    for alias in aliases:
                        if conn.execute('SELECT id FROM market_observations WHERE series_id=? AND date=?', (alias, new['date'])).fetchone():
                            alias_values = (alias, *values[1:])
                            conn.execute('INSERT OR IGNORE INTO market_revisions(series_id,date,value,period_end,published_at,retrieved_at,definition,quality) VALUES(?,?,?,?,?,?,?,?)', alias_values)
                            conn.execute('UPDATE market_observations SET value=?,period_end=?,published_at=?,retrieved_at=?,definition=?,quality=? WHERE series_id=? AND date=?', (*alias_values[2:], *alias_values[:2]))
        conn.execute('UPDATE integrity_incidents SET state=?,resolved_at=?,resolved_by=? WHERE id=?',
                     ('applied' if decision == 'accept' else 'rejected', utcnow(), actor, identity))
        conn.execute('INSERT INTO access_audit(actor_id,target_id,action,created_at) VALUES(?,?,?,?)',
                     (actor, actor, f'integrity:{identity}:{decision}', utcnow()))
    return True


def enqueue_due(store, now=None):
    from zoneinfo import ZoneInfo
    now = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo('Europe/Moscow'))
    config = store.get_setting('integrity_schedule', {})
    if not isinstance(config, dict) or config.get('enabled') is not True:
        return
    slot = now.replace(day=1, hour=3, minute=0, second=0, microsecond=0)
    if now < slot or (config.get('last_slot') or '') >= slot.isoformat():
        return
    actor = config.get('actor_id')
    if not store.authorize(actor, admin=True):
        return
    if not store.one("SELECT id FROM jobs WHERE kind='integrity_audit' AND state IN ('pending','running') LIMIT 1"):
        enqueue(store, actor)
    store.set_setting('integrity_schedule', {**config, 'last_slot': slot.isoformat()})


def dispatch(store, *, now=None):
    """Use the existing delivery queue; no Telegram connection is opened here."""
    from zoneinfo import ZoneInfo

    from .delivery import prepare_delivery
    now = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo('Europe/Moscow'))
    rows = store.query("SELECT * FROM integrity_incidents WHERE notified_at IS NULL AND state IN ('new','review') AND (severity='critical' OR ?) ORDER BY id",
                       (now.hour >= 18,))
    if not isinstance(rows, list) or not rows:
        return
    admins = store.query("SELECT user_id FROM users WHERE role='admin' AND enabled=1")
    if not admins:
        return
    identifier = 'integrity-' + hashlib.sha256(dumps([row['id'] for row in rows]).encode()).hexdigest()[:24]
    from .integrity_text import digest
    text = digest(store, rows)
    store.execute("INSERT OR IGNORE INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,quality,parts,fingerprint) VALUES(?,'market',?,0,0,?,?,?,?,?)",
                  (identifier, utcnow(), text, text, dumps({'integrity': True}), '[]', identifier))
    for admin in admins:
        prepare_delivery(store, identifier, str(admin['user_id']), detail='full')
    with store.connection() as conn:
        conn.executemany('UPDATE integrity_incidents SET notified_at=? WHERE id=?', [(utcnow(), row['id']) for row in rows])


def health(store):
    try:
        result = store.query('PRAGMA quick_check')
        if not result or any(next(iter(row.values())) != 'ok' for row in result):
            incident(store, 'sqlite_integrity', severity='critical', new={'error': 'SQLite quick_check failed'})
            return False
    except Exception:  # noqa: BLE001 - The database itself may be unreadable.
        store.path.with_suffix('.integrity-emergency.json').write_text(dumps({'kind': 'sqlite_integrity', 'severity': 'critical', 'at': utcnow()}), encoding='utf-8')
        return False
    return True
