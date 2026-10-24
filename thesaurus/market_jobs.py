"""Offline exports, backtests, and market reports in the shared worker."""
import csv
import json
from pathlib import Path
from uuid import uuid4

from openpyxl import Workbook

from . import market
from . import market_analysis as analysis
from . import market_store as repository
from .delivery import prepare_delivery
from .exporting import safe_cell
from .store import dumps, utcnow
from .telegram_data_jobs import export_dates, require_actor

COLUMNS = ['series_id', 'title', 'source', 'date', 'period_end', 'value', 'unit', 'frequency', 'quality',
           'published_at', 'retrieved_at', 'definition', 'source_url', 'reference_data',
           'record_type', 'forecast_run', 'model', 'lower', 'upper', 'baseline', 'forecast_created_at']


def export_market(store, path, identities, *, date_from=None, date_to=None, format='csv', include_forecasts=False):
    start, end = export_dates(date_from, date_to)
    if format not in {'csv', 'xlsx'}:
        raise ValueError('Supported export formats: csv, xlsx')
    rows = []
    for identity in identities:
        series = store.one('SELECT * FROM market_series WHERE id=?', (identity,))
        if not series:
            raise ValueError('Unknown market series')
        meta = json.loads(series['metadata'])
        observations = repository.history(store, identity)
        if not observations and meta.get('kind') == 'reference':
            snapshot = store.one('SELECT payload FROM source_snapshots WHERE dataset=? ORDER BY received_at DESC LIMIT 1', (identity,))
            accepted = store.get_setting('source_snapshot_current:' + identity)
            if accepted:
                snapshot = store.one('SELECT payload FROM source_snapshots WHERE dataset=? AND version=?', (identity, accepted))
            rows.append({'series_id': identity, 'title': series['title'], 'source': series['source'], 'unit': meta.get('unit'),
                         'frequency': meta.get('frequency'), 'quality': 'reference', 'source_url': meta['url'],
                         'retrieved_at': series['updated_at'], 'reference_data': snapshot['payload'] if snapshot else None})
        for row in observations:
            if (start and row['date'] < start) or (end and row['date'] > end):
                continue
            rows.append({**row, 'title': series['title'], 'source': series['source'], 'unit': meta['unit'],
                         'frequency': meta['frequency'], 'source_url': meta['url'], 'record_type': 'actual'})
        if include_forecasts:
            for point in repository.forecast_history(store, identity, latest=True):
                if (start and point['origin'] < start) or (end and point['origin'] > end):
                    continue
                rows.append({'series_id': identity, 'title': series['title'], 'source': series['source'],
                             'date': point['target'], 'value': point['prediction'], 'unit': meta['unit'],
                             'frequency': point.get('frequency', meta['frequency']), 'source_url': meta['url'],
                             'record_type': 'scenario' if point.get('model_family') == 'monthly_usd' and point.get('validation_n', 0) < 10 else 'forecast',
                             'quality': 'conditional_scenario' if point.get('model_family') == 'monthly_usd' and point.get('validation_n', 0) < 10 else 'local_forecast',
                             'forecast_run': point['run_id'], 'model': point['model'],
                             'lower': point['lower'], 'upper': point['upper'], 'baseline': point['baseline'],
                             'forecast_created_at': point['created_at']})
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if format == 'csv':
        with output.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for row in rows:
                writer.writerow([safe_cell(row.get(key)) for key in COLUMNS])
    else:
        workbook = Workbook(write_only=True)
        sheet = workbook.create_sheet('Market')
        sheet.append(COLUMNS)
        for row in rows:
            sheet.append([safe_cell(row.get(key)) for key in COLUMNS])
        workbook.save(output)
    return output


def run_market_job(store, settings, job, payload, *, dry_run=False, charts=True):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        return _run(store, settings, job, payload, dry_run=dry_run, charts=charts)


def _run(store, settings, job, payload, *, dry_run=False, charts=True):
    actor = payload.get('actor_id')
    if actor is None:
        user = store.one('SELECT user_id FROM users WHERE chat_id=? AND enabled=1', (str(payload.get('recipient')),))
        if not user:
            raise PermissionError('Рыночные задания требуют разрешённого пользователя')
        actor = user['user_id']
    require_actor(store, actor)
    from .subscription_access import require
    require(store, actor, 'market')
    if job['kind'] == 'market_collect':
        require_actor(store, actor, admin=True)
        if payload.get('scheduled_for') and not store.get_setting('market_schedule', {}).get('enabled'):
            return None
        fingerprint = f'market-job:{job["id"]}'
        existing = store.one('SELECT id FROM reports WHERE fingerprint=?', (fingerprint,)) if payload.get('notify_actor') else None
        if existing:
            if not dry_run:
                prepare_delivery(store, existing['id'], str(actor), detail='full')
            return existing['id']
        selection = payload.get('selection')
        quality = market.collect(store, dry_run=dry_run, selection=selection) if selection is not None else market.collect(store, dry_run=dry_run)
        if not dry_run:
            from datetime import datetime, timedelta, timezone
            if selection is None:
                store.set_setting('market_last_collection', {'at': utcnow(), 'quality': quality,
                    'next_after': (datetime.now(timezone.utc)+timedelta(minutes=15)).isoformat()})
            if selection is None:
                for row in market.selected(store):
                    market.save_forecasts(store, row, repository.history(store, row['id']))
            from .market_alerts import process
            process(store, settings)
        if payload.get('notify_actor'):
            from .telegram_cbr import descendants
            rows = descendants(store, payload['cbr_root'])
            loaded = []
            for row in rows:
                count = store.one('SELECT COUNT(*) n,MIN(date) first,MAX(date) last FROM market_observations WHERE series_id=? AND value IS NOT NULL', (row['id'],))
                if count['n']:
                    loaded.append((row, count))
            personal = store.get_setting(f'market_selection:{actor}', [])
            if personal and loaded and not dry_run:
                store.set_setting(f'market_selection:{actor}', list(dict.fromkeys(
                    [identity for identity in personal if identity != '__none__'] + [row['id'] for row, _ in loaded if row['enabled']])))
            root = store.one('SELECT title,rowid token FROM market_series WHERE id=?', (payload['cbr_root'],))
            text = root['title'] + '\n' + ('Загрузка частичная.' if quality['errors'] else 'Загрузка завершена.')
            text += f'\nЧисловых рядов с данными: {len(loaded)}. Наблюдений: {sum(count["n"] for _, count in loaded)}.'
            if loaded:
                text += '\nПериод: ' + min(count['first'] for _, count in loaded) + ' — ' + max(count['last'] for _, count in loaded) + '.'
            text += '\nОткройте «Аналитика → Показатели Банка России», выберите базу и показатель для графика, прогноза или сравнения.'
            text += '\n' + '\n'.join(error['source'] + ': ' + error['error'] for error in quality['errors']) if quality['errors'] else ''
            report_id = uuid4().hex
            require_actor(store, actor, admin=True)
            store.execute("""INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,quality,parts,fingerprint)
                VALUES(?,'market',?,1,0,?,?,?,?,?)""", (report_id, utcnow(), text, text, dumps(quality), '[]', fingerprint))
            if not dry_run:
                prepare_delivery(store, report_id, str(actor), detail='full')
            return report_id
        return None
    subscription = payload.get('subscription_id')
    if subscription is not None:
        from .service import subscription_active
        if not subscription_active(store, subscription, str(actor), 'market'):
            raise PermissionError('Подписка выключена')
    selected = market.selected(store, actor)
    if job['kind'] == 'market_export_sources':
        require_actor(store, actor, admin=True)
        selected = store.query('SELECT * FROM market_series')
    allowed = {row['id']: row for row in selected}
    identities = payload.get('selection', list(allowed))
    if not set(identities).issubset(allowed):
        raise PermissionError('Выбор показателей изменился; повторите запрос')
    report_id = payload.get('report_id') or uuid4().hex
    existing = store.one('SELECT id FROM reports WHERE fingerprint=?', (f'market-job:{job["id"]}',))
    if existing:
        report_id = existing['id']
    elif job['kind'] in {'market_report', 'report'}:
        # Collection is shared. Per-recipient report preparation uses existing snapshots.
        cached = store.get_setting('market_last_collection', {})
        if dry_run or not cached.get('at') or utcnow() > cached.get('next_after', ''):
            quality = market.collect(store, dry_run=dry_run)
            if not dry_run:
                from datetime import datetime, timedelta, timezone
                store.set_setting('market_last_collection', {'at': utcnow(), 'quality': quality,
                    'next_after': (datetime.now(timezone.utc)+timedelta(minutes=15)).isoformat()})
        else:
            store.set_setting('market_last_collection', {**cached, 'quality': {**cached.get('quality', {}), 'cached': True}})
        restricted = payload.get('phrases') or identities or None
        if restricted is not None and not set(restricted).issubset({r['id'] for r in market.selected(store, actor)}):
            raise PermissionError('Показатели больше не доступны')
        report_id = market.report(store, actor, payload.get('profile', store.get_setting(f'market_profile:{actor}', 'all')),
            charts=charts, settings=settings, identities=restricted)
        if any(row['source'] == 'google_trends' and row['id'] in identities for row in selected):
            from .google_trends import NOTE
            store.execute('UPDATE reports SET text=text || ?,full_text=full_text || ? WHERE id=?', ('\n' + NOTE, '\n' + NOTE, report_id))
        store.execute('UPDATE reports SET fingerprint=? WHERE id=?', (f'market-job:{job["id"]}', report_id))
        if not dry_run:
            from .market_alerts import process
            process(store, settings)
    else:
        root = (settings.reports_dir / report_id).resolve()
        if not root.is_relative_to(settings.reports_dir.resolve()):
            raise ValueError('Invalid artifact path')
        root.mkdir(parents=True, exist_ok=True)
        if job['kind'] in {'market_export', 'market_export_sources'}:
            format = payload.get('format', 'csv')
            options = payload.get('chart_options', {})
            duration = options.get('forecast')
            if duration and duration['value']:
                for identity in identities:
                    market.save_forecasts(store, allowed[identity], repository.history(store, identity), horizon=duration)
            output = export_market(store, root / ('market.' + format), identities, date_from=payload.get('date_from'), date_to=payload.get('date_to'), format=format,
                                   include_forecasts=bool(duration and duration['value']))
            text = 'Экспорт выбранных рыночных данных. Единицы, источники и качество включены.'
        elif job['kind'] == 'market_backtest':
            count = payload.get('max_origins', 12)
            if type(count) is not int or not 1 <= count <= 52:
                raise ValueError('Invalid origin count')
            results = []
            for identity in identities:
                meta = json.loads(allowed[identity]['metadata'])
                results.append({'series_id': identity, 'unit': meta['unit'], **analysis.backtest(repository.history(store, identity), meta, count, horizon=payload.get('forecast'))})
            output = root / 'market-backtest.json'
            output.write_text(dumps(results), encoding='utf-8')
            text = 'Ретроспективная проверка рыночных прогнозов:\n' + '\n'.join(r['series_id'] + ': ' + dumps(r['metrics']) for r in results)
            text += '\nИсторические значения могли быть пересмотрены; будущие наблюдения не используются при обучении.'
        else:
            raise ValueError('Unknown market job')
        require_actor(store, actor)
        store.execute("""INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,quality,parts,fingerprint)
            VALUES(?,'market',?,1,0,?,?,?,?,?)""", (report_id, utcnow(), text, text, dumps({'actor_id': actor, 'job_kind': job['kind']}),
                dumps([{'kind': 'document', 'path': str(output), 'caption': 'Рыночные данные'}]), f'market-job:{job["id"]}'))
    require_actor(store, actor)
    if not dry_run:
        if subscription is not None:
            from .service import subscription_active
            if not subscription_active(store, subscription, str(actor), 'market'):
                return report_id
        prepare_delivery(store, report_id, str(actor), detail=payload.get('detail', 'full'), subscription_id=subscription)
    return report_id
