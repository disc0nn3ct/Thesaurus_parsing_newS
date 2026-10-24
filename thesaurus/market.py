"""Market collection/report jobs share the existing worker and delivery queue."""
import json
import re
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd

from . import market_analysis as analysis
from . import market_sources as sources
from . import market_store as repository
from .store import dumps, utcnow


def initialize(store):
    catalog = sources.catalogue()
    repository.register(store, catalog)
    # Repair reviewed table semantics locally; values and identities stay intact.
    for item in catalog:
        if not any('unit' in contract for contract in item.get('contracts', [])):
            continue
        for row in store.query("SELECT id,metadata FROM market_series WHERE dataset=? AND json_extract(metadata,'$.kind')='series'", (item['id'],)):
            old = json.loads(row['metadata'])
            corrected = sources.reviewed_table_metadata(item, old)
            unit_override = store.get_setting('market_unit:' + row['id'])
            if unit_override is not None:
                corrected['unit'] = unit_override
            if corrected != old:
                repository.register(store, [corrected])
    # Old multi-table identifiers could mix RUB/USD or different MIACR groups.
    for item in catalog:
        if len(item.get('contracts', [])) > 1:
            for legacy in store.query("SELECT id,metadata FROM market_series WHERE dataset=? AND json_extract(metadata,'$.kind')='series' AND json_extract(metadata,'$.table_identity') IS NULL AND status<>'archived'", (item['id'],)):
                metadata = json.loads(legacy['metadata'])
                metadata['data_type'] = 'archive'
                store.execute("UPDATE market_series SET enabled=0,status='archived',metadata=? WHERE id=?", (dumps(metadata), legacy['id']))
                from .integrity import incident
                incident(store, 'legacy_ambiguous_series', series_id=legacy['id'], new={'reason': 'Не различались таблицы источника; сохранена архивная версия'})
    sync_ruonia(store)
    for source, frequency, phrases, regions in (
        ('wordstat', 'weekly', [r['phrase'] for r in store.keywords()], [225]),
        ('regions', 'daily', store.get_setting('incident_phrases', []), store.regions()),
    ):
        for phrase in phrases:
            for region in regions:
                identity = sources.metric_id('wordstat.' + source + '.' + str(region), phrase)
                repository.register(store, [{'id': identity, 'source': 'wordstat', 'dataset': 'wordstat.' + source,
                    'title': f'{phrase} ({region})', 'kind': 'series', 'adapter': 'stored', 'section': source,
                    'phrase': phrase, 'region': region, 'frequency': frequency, 'week_anchor': 'MON',
                    'calendar': 'calendar', 'aggregation': 'sum', 'transform': 'difference', 'unit': 'запросов',
                    'url': 'https://wordstat.yandex.ru/', 'definition': '1'}])


def enqueue_collection(store, now=None):
    """Coalesce missed global collection slots independently of personal delivery."""
    configured = store.get_setting('market_schedule', {})
    if not isinstance(configured, dict) or configured.get('enabled') is not True:
        return
    from .scheduling import _latest_due
    now = now or datetime.now(timezone.utc)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        setting = conn.execute("SELECT value FROM settings WHERE key='market_schedule'").fetchone()
        config = json.loads(setting[0]) if setting else {}
        if not config.get('enabled'):
            return
        actor = conn.execute('SELECT enabled,role FROM users WHERE user_id=?', (config.get('actor_id'),)).fetchone()
        if not actor or not actor['enabled'] or actor['role'] != 'admin':
            return
        slot = _latest_due({**config, 'times': dumps(config['times']), 'weekdays': dumps(list(range(7))),
                            'timezone': 'Europe/Moscow', 'last_scheduled_at': config.get('last_scheduled_at')}, now)
        if slot is None:
            return
        payload = dumps({'actor_id': config['actor_id'], 'scheduled_for': slot.isoformat()})
        pending = conn.execute("SELECT id FROM jobs WHERE kind='market_collect' AND state='pending' AND dedupe_key LIKE 'market-collection:%' LIMIT 1").fetchone()
        key = 'market-collection:' + slot.isoformat()
        if pending:
            conn.execute('UPDATE jobs SET payload=?,dedupe_key=? WHERE id=?', (payload, key, pending['id']))
        else:
            conn.execute("INSERT OR IGNORE INTO jobs(kind,payload,dedupe_key,created_at) VALUES('market_collect',?,?,?)", (payload, key, utcnow()))
        config['last_scheduled_at'] = slot.isoformat()
        conn.execute("UPDATE settings SET value=? WHERE key='market_schedule'", (dumps(config),))


def sync_ruonia(store):
    """Expose existing RUONIA cache to local analytics without enabling downloads."""
    from .cbr_data import METRICS, RATE_URL
    for metric in METRICS:
        rows = store.query("SELECT date,value,retrieved_at,quality FROM observations WHERE source='ruonia' "
                           "AND metric=? AND region_id=225 AND device='all' AND frequency='daily' ORDER BY date", (metric,))
        if not any(row['value'] is not None for row in rows):
            continue
        identity = sources.metric_id('local.ruonia', metric)
        existed = store.one('SELECT id FROM market_series WHERE id=?', (identity,))
        repository.register(store, [{'id': identity, 'source': 'ruonia', 'dataset': 'local.ruonia',
            'title': metric, 'metric': metric, 'kind': 'series', 'adapter': 'stored_ruonia', 'frequency': 'daily',
            'calendar': 'calendar' if metric in {'Индекс RUONIA', '1 мес', '3 мес', '6 мес'} else 'russia_working',
            'unit': 'индекс' if metric == 'Индекс RUONIA' else 'млрд руб' if metric == 'Объём RUONIA' else '%',
            'aggregation': 'last', 'transform': 'difference',
            'url': RATE_URL if metric in {'RUONIA', 'Объём RUONIA'} else 'https://cbr.ru/hd_base/ruonia/dynamic/', 'definition': '1'}])
        from .market_incremental import only_missing
        repository.save_rows(store, identity, only_missing(store, identity, rows))
        stamp = max((row['retrieved_at'] for row in rows if row['value'] is not None), default=None)
        store.execute("UPDATE market_series SET enabled=CASE WHEN ? THEN enabled ELSE 1 END,status='cached',updated_at=? WHERE id=?", (bool(existed), stamp, identity))


def selected(store, user_id=None):
    rows = store.query("SELECT * FROM market_series WHERE enabled=1 AND json_extract(metadata,'$.kind')='series' ORDER BY id")
    from .search_provider import disabled
    if disabled(store):
        rows = [row for row in rows if row['source'] not in {'wordstat', 'google_trends'}]
    if user_id is None:
        return rows
    from .subscription_access import allowed
    if not allowed(store, user_id, 'market'):
        return []
    chosen = store.get_setting(f'market_selection:{user_id}', [])
    filtered = [row for row in rows if not chosen or row['id'] in chosen]
    result = []
    for row in filtered:
        metadata = json.loads(row['metadata'])
        if metadata.get('adapter') == 'stored_ruonia':
            if not allowed(store, user_id, 'ruonia'):
                continue
            from .telegram_data_jobs import actor_scope
            if metadata['metric'] not in actor_scope(store, user_id, 'ruonia')['phrases']:
                continue
        if row['source'] == 'wordstat':
            from .telegram_data_jobs import actor_scope
            scope = actor_scope(store, user_id, metadata['section'])
            if metadata['phrase'] not in scope['phrases'] or metadata['region'] not in scope['regions']:
                continue
        result.append(row)
    return result


def enable(store, identity, enabled):
    from .google_trends import active
    item = store.one('SELECT source,status FROM market_series WHERE id=?', (identity,))
    from .search_provider import NOTICE, disabled
    if enabled and disabled(store) and item and item['source'] in {'wordstat', 'google_trends'}:
        raise ValueError(NOTICE)
    if enabled and item and (item['source'] == 'wordstat' and active(store) or item['status'] == 'archived'):
        raise ValueError('Источник выключен или версия архивная. Google Trends обновляется через /trends.')
    if not store.one('SELECT id FROM market_series WHERE id=?', (identity,)):
        raise ValueError('Источник не найден; откройте /sources')
    store.execute('UPDATE market_series SET enabled=? WHERE id=? OR dataset=?', (int(enabled), identity, identity))
    store.set_setting('market_last_collection', {})
    for child in store.query("SELECT id FROM market_series WHERE json_extract(metadata,'$.parent')=? AND id<>?", (identity, identity)):
        enable(store, child['id'], enabled)


def add_security(store, market, code, board=None):
    if market not in {'shares', 'bonds', 'index'} or not re.fullmatch('[A-Z0-9_-]{1,30}', code):
        raise ValueError('/sources moex shares|bonds|index КОД [РЕЖИМ]')
    if board and not re.fullmatch('[A-Z0-9_-]{1,16}', board):
        raise ValueError('Неверный режим торгов')
    path = f'boards/{board}/' if board else ''
    fields = [('CLOSE', 'цена'), ('VALUE', 'руб'), ('VOLUME', 'бумаг')]
    if market == 'bonds':
        fields += [('YIELDCLOSE', '%')]
        fields[0] = ('CLOSE', '% номинала')
    if market == 'index':
        fields = [('CLOSE', 'индекс')]
    for field, unit in fields:
        identity = f'moex.{market}.{code}.{board or "main"}.{field}'.lower()
        repository.register(store, [{'id': identity, 'source': 'moex', 'dataset': f'moex.{code.lower()}',
            'title': f'{code}: {field} ({board or "основной режим"})', 'unit': unit, 'frequency': 'daily',
            'calendar': 'moex', 'transform': 'returns' if field == 'CLOSE' else 'difference',
            'aggregation': 'sum' if field in {'VALUE', 'VOLUME'} else 'last', 'kind': 'series',
            'adapter': 'moex', 'security': code, 'market': market, 'field': field, 'board': board,
            'url': f'https://iss.moex.com/iss/history/engines/stock/markets/{market}/{path}securities/{code}.json',
            'definition': '1', 'delay': 'ISS public; history and access limits apply'}])


def collect(store, *, dry_run=False, transport=None, today=None, selection=None):
    from . import market_incremental as incremental
    today = today or datetime.now(timezone.utc).date()
    transport = transport or sources.Transport()
    result = {'status': 'cached' if dry_run else 'ok', 'cached': True, 'errors': [],
              'last_fetched_at': utcnow(), 'checked_at': utcnow()}
    if dry_run:
        return result
    sync_ruonia(store)
    rows = store.query("SELECT * FROM market_series WHERE enabled=1 ORDER BY id")
    if selection is not None:
        if not isinstance(selection, list) or not set(selection).issubset({r['id'] for r in rows}):
            raise PermissionError('Источники больше не доступны для загрузки')
        rows = [r for r in rows if r['id'] in selection or r['dataset'] in selection]
    # A dataset is fetched once for all its numeric columns.
    datasets = {row['id']: row for row in rows if json.loads(row['metadata']).get('kind') != 'series' or
                json.loads(row['metadata']).get('adapter') in {'currency', 'moex', 'ruonia', 'stored', 'fred_csv'}}
    for row in rows:
        metadata = json.loads(row['metadata'])
        if metadata.get('kind') == 'series' and metadata.get('adapter', 'html') == 'html':
            parent = store.one('SELECT * FROM market_series WHERE id=?', (row['dataset'],))
            if parent:
                datasets[parent['id']] = parent
    for row in datasets.values():
        item = {**json.loads(row['metadata']), 'params': store.get_setting('market_params:' + row['id'], {})}
        frequency = item['frequency']
        years = 5 if frequency == 'daily' else 10 if frequency == 'weekly' else 20
        years = store.get_setting('market_history_years', {}).get(frequency, years)
        end = min(today, datetime.fromisoformat(item['available_end']).date()) if item.get('available_end') else today
        start = end - timedelta(days=365 * min(50, max(1, int(years))))
        explicit_start = store.get_setting('market_history_start:' + row['id'])
        if explicit_start:
            start = date_from_setting(explicit_start, end)
        if frequency in {'monthly', 'quarterly', 'annual'}:
            start = incremental.period(start, frequency)
        try:
            from . import cbr_incremental as cbr_cache
            if row['id'] == cbr_cache.INDEX_DATASET:
                update = cbr_cache.collect_index(store, transport, start, today - timedelta(days=1))
                if update['requests']:
                    result['cached'] = False
                shared = cbr_cache.history(store, cbr_cache.INDEX_METRICS)
                for metric, column in cbr_cache.INDEX_COLUMNS.items():
                    identity = cbr_cache.MARKET_IDS[metric]
                    existed = store.one('SELECT id FROM market_series WHERE id=?', (identity,))
                    meta = {**item, 'id': identity, 'dataset': row['id'], 'kind': 'series', 'column': column,
                            'dimensions': '', 'title': item['title'] + ': ' + column, 'unit': cbr_cache.unit(metric)}
                    repository.register(store, [meta])
                    if not existed:
                        enable(store, identity, True)
                    observations = [value for (name, day), value in shared.items() if name == metric]
                    repository.save_rows(store, identity, incremental.only_missing(store, identity, observations))
                    fetched_at = max((obs['retrieved_at'] for obs in observations if obs['value'] is not None), default=None)
                    store.execute("UPDATE market_series SET status='ok',error=NULL,updated_at=COALESCE(?,updated_at) WHERE id=?", (fetched_at, identity))
                store.execute("UPDATE market_series SET status='ok',error=NULL,updated_at=? WHERE id=?", (utcnow(), row['id']))
                continue
            if item.get('adapter') == 'ruonia':
                from .cbr_incremental import (
                    MARKET_IDS,
                    RATE_METRICS,
                    collect_rates,
                    history,
                )
                # One SOAP call supplies both fields; both interfaces share it.
                update = collect_rates(store, transport, start, today - timedelta(days=1))
                if update['requests']:
                    result['cached'] = False
                metric = next(name for name in RATE_METRICS if MARKET_IDS[name] == row['id'])
                observations = [value for (name, day), value in history(store, RATE_METRICS).items() if name == metric]
                repository.save_rows(store, row['id'], incremental.only_missing(store, row['id'], observations))
                fetched_at = max((obs['retrieved_at'] for obs in observations if obs['value'] is not None), default=None)
                store.execute("UPDATE market_series SET status='ok',error=NULL,updated_at=COALESCE(?,updated_at) WHERE id=?", (fetched_at, row['id']))
                continue
            if item.get('adapter') == 'stored':
                observations = store.query('SELECT date,value,retrieved_at,quality FROM observations WHERE source=? AND metric=? AND region_id=? AND frequency=? ORDER BY date',
                    ('wordstat', item['phrase'], item['region'], item['frequency']))
                repository.save_rows(store, row['id'], observations)
                fetched_at = max((obs['retrieved_at'] for obs in observations if obs['value'] is not None), default=None)
                store.execute("UPDATE market_series SET status='cached',updated_at=COALESCE(?,updated_at) WHERE id=?", (fetched_at, row['id']))
                continue
            if item.get('kind') == 'reference':
                batches = [] if store.one('SELECT dataset FROM source_snapshots WHERE dataset=? LIMIT 1', (item['id'],)) else [(start, today)]
            elif item.get('adapter') == 'official_forecast' or item.get('snapshot_current'):
                # The source publishes a current weekly snapshot, not a date-filtered history.
                previous = store.get_setting('market_snapshot_checked:' + item['id'], '')
                batches = [] if previous == today.isoformat() else [(today, today)]
            elif item.get('full_snapshot'):
                batches = [] if store.get_setting('market_history_loaded:' + row['id']) else [(start, today)]
            else:
                batches = incremental.plan(store, item, start, end)
            for first, last in batches:
                result['cached'] = False
                for meta, observations in sources.fetch(item, transport, first, last):
                    snapshot = meta.pop('_snapshot', None)
                    if snapshot is not None:
                        import hashlib
                        content = dumps(snapshot)
                        store.execute('INSERT OR IGNORE INTO source_snapshots(dataset,version,received_at,payload) VALUES(?,?,?,?)',
                                      (item['id'], hashlib.sha256(content.encode()).hexdigest(), utcnow(), content))
                    existed = store.one('SELECT id FROM market_series WHERE id=?', (meta['id'],))
                    repository.register(store, [meta])
                    observations, conflicts = repository.partition_rows(observations)
                    if conflicts:
                        from .integrity import incident
                        for day, candidates in conflicts.items():
                            incident(store, 'source_conflict', series_id=meta['id'], day=day,
                                     new={'candidates': candidates, 'action': 'excluded_from_analysis'})
                    if meta['id'] != row['id'] and (selection is not None or not existed and not meta.get('dimensions')):
                        enable(store, meta['id'], True)
                    if meta.get('snapshot_current'):
                        from .integrity import incident
                        for observation in observations:
                            saved = store.one('SELECT * FROM market_observations WHERE series_id=? AND date=?', (meta['id'], observation['date']))
                            if saved and saved['value'] != observation.get('value'):
                                incident(store, 'source_revision', series_id=meta['id'], day=observation['date'], old=saved, new=observation)
                    if meta.get('adapter') in {'fred_csv', 'currency'}:
                        # Permanent historic absences are stored as NULL, never zero.
                        # Recent unavailable publications remain retryable after the cache TTL.
                        cutoff = today - timedelta(days=30)
                        observed = {obs['date'] for obs in observations}
                        for observation in observations:
                            if observation.get('value') is None and observation['date'] <= cutoff.isoformat():
                                observation['quality'] = 'source_no_observation'
                        observations += [{'date': stamp.date().isoformat(), 'value': None, 'quality': 'source_no_observation'}
                                         for stamp in analysis.grid(first, min(last, cutoff), meta) if stamp.date().isoformat() not in observed]
                    observations = incremental.only_missing(store, meta['id'], [observation for observation in observations
                        if meta.get('data_type') == 'official_forecast' or meta.get('full_snapshot') or meta.get('snapshot_current') or first.isoformat() <= observation['date'] <= last.isoformat()])
                    for observation in observations:
                        if frequency == 'monthly' and meta.get('period_semantics') == 'flow':
                            stamp = pd.Timestamp(observation['date'])
                            observation['period_end'] = (stamp + pd.offsets.MonthEnd(0)).date().isoformat()
                        if meta['unit'] == 'source_units':
                            observation['quality'] = 'unit_not_specified_publication_unknown'
                        observation['retrieved_at'] = result['last_fetched_at']
                    repository.save_rows(store, meta['id'], observations)
                    fetched_at = max((obs['retrieved_at'] for obs in observations if obs['value'] is not None), default=None)
                    store.execute("UPDATE market_series SET status=?,error=NULL,updated_at=COALESCE(?,updated_at) WHERE id=?", ('ok' if observations else 'cached', fetched_at, meta['id']))
                incremental.remember(store, item, first, last, start)
            if (item.get('adapter') == 'official_forecast' or item.get('snapshot_current')) and batches:
                store.set_setting('market_snapshot_checked:' + item['id'], today.isoformat())
            if len(batches) == 64 and incremental.plan(store, item, start, end):
                raise cbr_cache.IncompleteCollection('Остались недостающие периоды')
            conflict_count = store.one("SELECT COUNT(*) n FROM integrity_incidents WHERE kind='source_conflict' "
                "AND state IN ('new','review') AND series_id IN (SELECT id FROM market_series WHERE id=? OR dataset=?)",
                (row['id'], row['id']))['n']
            if conflict_count:
                message = f'Противоречивых наблюдений источника: {conflict_count}. Они исключены из новой загрузки; остальные данные сохранены. Подробности — в журнале целостности.'
                result['errors'].append({'source': row['id'], 'title': row['title'], 'code': 'source_conflict', 'error': message})
                store.execute("UPDATE market_series SET status='partial',error=? WHERE id=?", (message, row['id']))
            else:
                store.execute("UPDATE market_series SET status=?,error=NULL WHERE id=?", ('reference' if item.get('kind') == 'reference' else 'ok', row['id']))
            store.set_setting('market_history_loaded:' + row['id'], {'start': start.isoformat(), 'end': today.isoformat()})
        except Exception as exc:  # noqa: BLE001 - public transport errors may contain proxy credentials.
            from .source_errors import diagnostic
            code, safe = diagnostic(exc)
            result['errors'].append({'source': row['id'], 'title': row['title'], 'code': code, 'error': safe})
            store.execute("UPDATE market_series SET status='partial',error=? WHERE id=?", (safe, row['id']))
    if result['errors']:
        result['status'] = 'partial'
    if not rows:
        result['note'] = 'Выберите источники через /sources; автоматический сбор выключен.'
    saved = store.one('SELECT MAX(retrieved_at) stamp FROM market_observations WHERE series_id IN (SELECT id FROM market_series WHERE enabled=1)')
    result['last_fetched_at'] = saved['stamp'] if saved else None
    derive(store)
    return result


def date_from_setting(value, end):
    from datetime import date
    start = date.fromisoformat(value)
    if start > end or start < date(1900, 1, 1):
        raise ValueError('Invalid history boundary')
    return start


def derive(store):
    """Official-rate spread and yield curve shape, preserving date alignment."""
    active = selected(store)
    tenors = {}
    key_rate = None
    for row in active:
        metadata = json.loads(row['metadata'])
        if metadata.get('maturity'):
            tenors[metadata['maturity']] = row['id']
        if row['dataset'] == 'cbr.hd_base.keyrate':
            key_rate = row['id']
    definitions = []
    if 10 in tenors and 2 in tenors:
        definitions.append(('cbr.derived.yield_slope', 'Наклон кривой 10 лет − 2 года', [(tenors[10], 1), (tenors[2], -1)]))
    if 5 in tenors and 2 in tenors and 10 in tenors:
        definitions.append(('cbr.derived.yield_curvature', 'Кривизна: 5 лет − (2 года + 10 лет)/2', [(tenors[5], 1), (tenors[2], -.5), (tenors[10], -.5)]))
    if key_rate and any(row['id'] == 'cbr.ruonia.rate' for row in active):
        definitions.append(('cbr.derived.ruonia_spread', 'Спред RUONIA − ключевая ставка', [('cbr.ruonia.rate', 1), (key_rate, -1)]))
    for identity, title, components in definitions:
        repository.register(store, [{'id': identity, 'title': title, 'source': 'cbr', 'dataset': 'cbr.derived',
            'kind': 'series', 'adapter': 'derived', 'frequency': 'daily', 'calendar': 'russia_working',
            'unit': 'п.п.', 'transform': 'difference', 'aggregation': 'mean', 'components': components,
            'url': 'https://cbr.ru/hd_base/', 'formula': title, 'definition': '1'}])
        maps = [{r['date']: r['value'] for r in repository.history(store, source)} for source, _ in components]
        dates = set(maps[0]).intersection(*(set(m) for m in maps[1:]))
        observations = [{'date': d, 'value': sum(m[d]*weight for m, (_, weight) in zip(maps, components))}
                        for d in sorted(dates) if all(m[d] is not None for m in maps)]
        repository.save_rows(store, identity, observations)
        # Initially enabled with parent indicators; can be switched off independently later.
        if store.get_setting('market_derived_seen:' + identity) is None:
            enable(store, identity, True)
            store.set_setting('market_derived_seen:' + identity, True)


def save_forecasts(store, series, rows, *, horizon=None):
    import hashlib

    from .forecast_horizons import from_options, validate
    metadata = json.loads(series['metadata'])
    duration = validate(horizon) if horizon is not None else from_options({}, metadata['frequency'])
    created = utcnow()
    known = repository.history(store, series['id'], as_of=created)
    if not known:
        return
    feature_ids = store.get_setting('market_features:' + series['id'], [])
    features = []
    for identity in feature_ids[:10]:
        row = store.one('SELECT metadata FROM market_series WHERE id=? AND enabled=1', (identity,))
        if row:
            meta = json.loads(row['metadata'])
            if meta['frequency'] == metadata['frequency']:
                features.append((repository.history(store, identity, as_of=created), meta))
    inputs = {'series_id': series['id'], 'revisions': [r['id'] for r in known], 'as_of': created,
              'features': feature_ids[:10], 'feature_revisions': [[r['id'] for r in data] for data, _ in features],
              'definition': metadata.get('definition')}
    from .russian_calendar import VERSION as calendar_version
    signature = {**inputs, 'duration': duration, 'metadata': metadata,
                 'model_version': 'duration-5-administrative' if metadata.get('administrative') else 'duration-4',
                 'calendar_version': calendar_version}
    signature.pop('as_of')
    if metadata.get('administrative') and metadata.get('calendar') == 'events':
        signature['scenario_date'] = created[:10]
    fingerprint = hashlib.sha256(dumps(signature).encode()).hexdigest()
    existing = store.one('SELECT id FROM forecast_runs WHERE fingerprint=?', (fingerprint,))
    if existing:
        return [json.loads(p['payload']) for p in store.query('SELECT payload FROM forecast_points WHERE run_id=? ORDER BY target', (existing['id'],))]
    predictions = analysis.forecast(known, metadata, created_at=created, features=features, horizon=duration)
    run_id = uuid4().hex
    with store.connection() as conn:
        conn.execute('INSERT INTO forecast_runs(id,series_id,fingerprint,created_at,duration,inputs) VALUES(?,?,?,?,?,?)',
                     (run_id, series['id'], fingerprint, created, dumps(duration), dumps(inputs)))
        for prediction in predictions:
            conn.execute('INSERT INTO forecast_points(run_id,target,payload) VALUES(?,?,?)',
                         (run_id, prediction['target'], dumps(prediction)))
            if horizon is not None:
                continue
            conn.execute("""INSERT OR IGNORE INTO market_forecasts(series_id,origin,target,horizon,prediction,lower,upper,baseline,model,created_at,inputs)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (series['id'], prediction['origin'], prediction['target'], prediction['horizon'],
                    prediction['prediction'], prediction['lower'], prediction['upper'], prediction['baseline'], prediction['model'], created, dumps(inputs)))
        conn.execute("""UPDATE market_forecasts SET actual=(SELECT value FROM market_observations o WHERE o.series_id=market_forecasts.series_id AND o.date=market_forecasts.target)
            WHERE series_id=?""", (series['id'],))
    return predictions


def pair_results(store, user_id):
    permitted = {r['id'] for r in selected(store, user_id)}
    results = []
    for pair in store.get_setting(f'market_pairs:{user_id}', [])[:50]:
        left, right = pair['left'], pair['right']
        if not {left, right}.issubset(permitted):
            continue
        metas = [json.loads(store.one('SELECT metadata FROM market_series WHERE id=?', (key,))['metadata']) for key in (left, right)]
        result = analysis.correlations(repository.history(store, left), repository.history(store, right), *metas,
            window=pair.get('window'), lag=pair.get('lag', 0), levels=pair.get('levels', False))
        results.append({'left': left, 'right': right, **result})
    return analysis.fdr(results)


def report(store, user_id, profile='all', charts=True, settings=None, forecast_enabled=True, identities=None):
    if profile not in {'all', 'macro', 'investment', 'liquidity'}:
        raise ValueError('Профили: all, macro, investment, liquidity')
    lines = [f'Обзор рынка ({profile}). Расчёты по официальным данным.']
    full, metrics = [], {}
    series = selected(store, user_id)
    if identities is not None:
        series = [row for row in series if row['id'] in identities]
    for row in series:
        meta = json.loads(row['metadata'])
        if profile == 'investment' and meta.get('transform') != 'returns' and row['source'] != 'moex' and not any(part in row['id'] for part in ('zcyc', 'yield_', 'keyrate')):
            continue
        if profile == 'liquidity' and not any(part in row['id'] for part in ('liquidity', 'ruonia', 'ostat', 'depo', 'reserve', 'repodb')):
            continue
        observations = repository.history(store, row['id'])
        summary = analysis.diagnostics(observations, meta)
        metrics[row['id']] = summary
        if summary['status'] == 'missing':
            lines.append(row['title'] + ': наблюдений нет')
            continue
        change_unit = 'п.п.' if meta['unit'] == '%' else meta['unit']
        change = f"; изменение {summary['change']:+.4g} {change_unit} с {summary['previous_date']}" if summary['change'] is not None else ''
        text = f"{row['title']}: {summary['value']:.5g} {meta['unit']}; {summary['date']}{change}"
        lines.append(text)
        coverage = f"{summary['coverage']:.0%}" if summary['coverage'] is not None else 'не определено (отдельные события)'
        state = {'cached': 'использована сохранённая история', 'ok': 'данные сохранены', 'partial': 'обновление неполное', 'archived': 'архив'}.get(row['status'], row['status'])
        full.append(text + f"\nИстория {summary['period_start']} — {summary['date']}. Покрытие внутри этого периода: {coverage}; ожидаемых дат {summary['expected']}, фактов {summary['points']}, пропусков {summary['missing']}."
                    + (f" Вне расчётного календаря: {summary['off_calendar']}; требуется проверка календаря." if summary['off_calendar'] and summary['expected'] is not None else '')
                    + f"\nЗагружено: {row['updated_at'] or 'неизвестно'}; {state}.\n{meta['url']}")
        if row['id'] == 'cbr.derived.yield_slope':
            full.append('Длинная доходность ' + ('ниже' if summary['value'] < 0 else 'выше или равна') + ' короткой: сравнение 10 и 2 лет по одной дате.')
        if row['id'] == 'cbr.derived.ruonia_spread':
            full.append('RUONIA ' + ('ниже' if summary['value'] < 0 else 'выше или равна') + ' ключевой ставке по одной дате.')
        scenarios = store.get_setting(f'market_scenarios:{user_id}', {})
        if row['id'] in scenarios and meta.get('administrative'):
            delta = scenarios[row['id']]
            full.append(f"Сценарий пользователя: {summary['value']:.4g} + ({delta:+.4g}) = {summary['value']+delta:.4g} {meta['unit']}. Это заданное изменение, не предсказание решения ЦБ.")
        if forecast_enabled:
            save_forecasts(store, row, observations)
    # Derived spreads require observations at the same effective date and explicit units.
    rates = [r for r in series if 'ruonia' in r['id'] and 'Ставка' in r['title']]
    keys = [r for r in series if 'Ключевая' in r['title'] or 'ключевая' in r['title']]
    if rates and keys:
        a, b = repository.history(store, rates[0]['id']), repository.history(store, keys[0]['id'])
        matched = {r['date']: r['value'] for r in b}
        matches = [r for r in a if r['date'] in matched and r['value'] is not None and matched[r['date']] is not None]
        if matches:
            last = matches[-1]
            lines.append(f"Спред RUONIA − ключевая: {last['value']-matched[last['date']]:+.3g} п.п.; {last['date']}")
    for pair in pair_results(store, user_id):
        from .market_text import correlation
        full.append(correlation(pair, {row['id']: row['title'] for row in series}))
    for study in store.get_setting(f'market_events:{user_id}', []):
        if study['series_id'] not in {r['id'] for r in series}:
            continue
        from .telegram_data_jobs import actor_scope
        scope = actor_scope(store, user_id, 'regions')
        if study['phrase'] not in scope['phrases'] or study['region'] not in scope['regions']:
            continue
        events = [r['observed_date'] for r in store.query('SELECT observed_date FROM alerts WHERE metric=? AND region_id=?', (study['phrase'], study['region']))]
        row = next(r for r in series if r['id'] == study['series_id'])
        result = analysis.event_study(repository.history(store, row['id']), json.loads(row['metadata']), events)
        full.append(f"События Wordstat: {study['phrase']} ({study['region']}) → {row['title']}\n" + dumps(result))
    if not series:
        lines.append('Источники не выбраны. Администратор: /sources. Личный выбор: /indicators.')
    identifier = uuid4().hex
    parts = []
    if charts and settings is not None and series:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        root = settings.reports_dir / identifier
        root.mkdir(parents=True, exist_ok=True)
        for row in series[:12]:
            meta = json.loads(row['metadata'])
            curve = analysis.values(repository.history(store, row['id']), meta).tail(252)
            if curve.dropna().empty:
                continue
            figure = Figure(figsize=(9, 4), layout='constrained')
            FigureCanvasAgg(figure)
            axes = figure.subplots()
            axes.plot(curve.index, curve.values, label='История')
            predictions = store.query('SELECT * FROM market_forecasts WHERE series_id=? AND origin=(SELECT MAX(origin) FROM market_forecasts WHERE series_id=?) ORDER BY horizon', (row['id'], row['id']))
            if predictions:
                axes.plot(pd.to_datetime([p['target'] for p in predictions]), [p['prediction'] for p in predictions], '--o', label='Прогноз')
                if all(p['lower'] is not None for p in predictions):
                    axes.fill_between(pd.to_datetime([p['target'] for p in predictions]), [p['lower'] for p in predictions], [p['upper'] for p in predictions], alpha=.2)
            axes.set_title(row['title'][:100])
            axes.set_ylabel(meta['unit'])
            axes.legend()
            output = root / (str(len(parts)) + '.png')
            figure.savefig(output)
            parts.append({'kind': 'photo', 'path': str(output), 'caption': row['title'][:100]})
    collection = store.get_setting('market_last_collection', {})
    quality = collection.get('quality', {})
    if quality.get('errors'):
        from .market_text import errors
        lines.append('Обновление неполное; ошибки источников: ' + str(len(quality['errors'])))
        titles = {row['id']: row['title'] for row in store.query('SELECT id,title FROM market_series')}
        full.append('Не удалось обновить:\n' + errors(quality['errors'], titles))
    full.append('Методика: изменение — последнее значение минус предыдущее наблюдение, не обязательно за вчера. Покрытие рассчитано внутри указанной истории и не означает, что последние данные уже опубликованы. Пропуски не заменяются нулями. Волатильность — стандартное отклонение логарифмических доходностей за последние 60 расчётных дат без склеивания пропусков; просадка — отношение к предыдущему максимуму минус 1. Дата получения не считается датой публикации; дата публикации используется только когда её указал источник.')
    store.execute("""INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,metrics,quality,parts)
        VALUES(?,'market',?,?,?,?,?,?,?,?)""", (identifier, utcnow(), store.version('keyword'), store.version('cohort'),
             '\n'.join(lines[:25]), '\n\n'.join(full) or '\n'.join(lines), dumps(metrics),
             dumps({**quality, 'actor_id': user_id, 'profile': profile}), dumps(parts)))
    return identifier
