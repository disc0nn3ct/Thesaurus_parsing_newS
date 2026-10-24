"""Opt-in sequential delivery of the remaining pages of a personal chart request."""
import json
from uuid import uuid4

from .store import dumps, utcnow
from .telegram_data_jobs import report_identifier, require_actor


def owned_report(store, actor, report_id):
    require_actor(store, actor)
    report_identifier(report_id)
    report = store.one('SELECT * FROM reports WHERE id=?', (report_id,))
    quality = json.loads(report['quality']) if report else {}
    if quality.get('actor_id') != actor or not quality.get('chart_request'):
        raise PermissionError('Эти графики недоступны')
    return quality


def active(store, reference, recipient=None):
    if not isinstance(reference, dict):
        return False
    config = store.get_setting('chart_batch:' + str(reference.get('batch_id')), {})
    actor = reference.get('actor_id')
    from .subscription_access import allowed
    if not allowed(store, actor, config.get('request', {}).get('section')):
        return False
    return bool(type(actor) is int and store.authorize(actor) and config.get('enabled')
                and config.get('actor_id') == actor and config.get('generation') == reference.get('generation')
                and (recipient is None or str(actor) == str(recipient)))


def navigation(store, report_id, quality):
    page, pages = quality['chart_page'], quality['chart_pages']
    batch_id = quality.get('chart_batch_id', report_id)
    config = store.get_setting('chart_batch:' + batch_id, {})
    enabled = bool(config.get('enabled') and page+1 < pages)
    buttons = []
    if not enabled:
        if page:
            buttons.append([{'text': '‹ Предыдущие графики', 'callback_data': f'ch:batch:{report_id}:{page-1}'}])
        if page + 1 < pages:
            buttons.append([{'text': 'Следующие графики ›', 'callback_data': f'ch:batch:{report_id}:{page+1}'}])
    if enabled or page+1 < pages:
        buttons.append([{'text': ('✓' if enabled else '☐') + ' Присылать остальные по готовности',
                         'callback_data': f'ch:auto:{report_id}:{"off" if enabled else "on"}'}])
    buttons.append([{'text': 'Настроить графики', 'callback_data': f"ch:menu:{quality['chart_request']['section']}"}])
    note = (f'Графики: страница {page+1}/{pages}. ' + ('Это последняя страница выпуска.' if page+1 == pages else
            ('✓ Остальные страницы придут автоматически. Снимите галочку, чтобы остановить отправку.' if enabled else
             'Выберите страницу или включите отправку остальных по готовности.')))
    if config.get('error'):
        note += '\nАвтоотправка остановлена: ошибка задания или доставки. Проверьте /jobs и /deliveries.'
    return {'kind': 'text', 'text': note, 'buttons': buttons}


def _write(conn, key, config):
    conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                 (key, dumps(config)))


def _cancel(conn, batch_id):
    for table, timestamp in (('jobs', 'finished_at'), ('deliveries', 'updated_at')):
        for row in conn.execute(f"SELECT id,payload FROM {table} WHERE state='pending'").fetchall():
            if json.loads(row['payload']).get('chart_auto', {}).get('batch_id') == batch_id:
                conn.execute(f"UPDATE {table} SET state='cancelled',error='chart_auto_inactive',{timestamp}=? WHERE id=?", (utcnow(), row['id']))


def toggle(store, actor, report_id, enabled=None):
    quality = owned_report(store, actor, report_id)
    from .charts import validate_selection
    request = quality['chart_request']
    allowed = dict(validate_selection(store, actor, request['section'], request.get('identity')))
    if not set(request['selection']).issubset(allowed):
        raise PermissionError('Выбор показателей изменился. Постройте графики заново.')
    batch_id = quality.get('chart_batch_id', report_id)
    key = 'chart_batch:' + batch_id
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        user = conn.execute('SELECT enabled FROM users WHERE user_id=?', (actor,)).fetchone()
        if not user or not user['enabled']:
            raise PermissionError('Нет доступа')
        row = conn.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
        config = json.loads(row['value']) if row else {
            'actor_id': actor, 'cursor': quality['chart_page']+1, 'pages': quality['chart_pages'],
            'request': request, 'batch_id': batch_id, 'enabled': False, 'job_id': None,
        }
        if config['actor_id'] != actor:
            raise PermissionError('Эти графики недоступны')
        if enabled is not None and config['enabled'] == enabled:
            return navigation(store, report_id, quality)
        if not config['enabled'] and config['cursor'] >= config['pages']:
            raise ValueError('Все страницы этого выпуска уже отправлены')
        config.update(enabled=not config['enabled'] if enabled is None else enabled, generation=uuid4().hex)
        if not config['enabled']:
            _cancel(conn, batch_id)
        elif config.get('job_id'):
            # Resume the same report and parts; confirmed message IDs stay intact.
            job = conn.execute('SELECT * FROM jobs WHERE id=?', (config['job_id'],)).fetchone()
            payload = json.loads(job['payload'])
            reference = {'batch_id': batch_id, 'actor_id': actor, 'generation': config['generation']}
            payload['chart_auto'] = reference
            has_parts = conn.execute('SELECT 1 FROM deliveries WHERE report_id=?', (payload['report_id'],)).fetchone()
            state = 'pending' if job['state'] == 'cancelled' or job['state'] == 'done' and not has_parts else job['state']
            conn.execute('UPDATE jobs SET payload=?,state=? WHERE id=?', (dumps(payload), state, job['id']))
            for part in conn.execute('SELECT * FROM deliveries WHERE report_id=?', (payload['report_id'],)).fetchall():
                content = json.loads(part['payload'])
                content['chart_auto'] = reference
                state = 'pending' if part['state'] == 'cancelled' and part['error'] == 'chart_auto_inactive' else part['state']
                conn.execute('UPDATE deliveries SET payload=?,state=? WHERE id=?', (dumps(content), state, part['id']))
        _write(conn, key, config)
    pump(store)
    return navigation(store, report_id, quality)


def pump(store):
    """Recoverable queue pump: next page only after every previous part is confirmed."""
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        for row in conn.execute("SELECT key,value FROM settings WHERE key LIKE 'chart_batch:%'").fetchall():
            config = json.loads(row['value'])
            if not config.get('enabled'):
                continue
            user = conn.execute('SELECT enabled FROM users WHERE user_id=?', (config['actor_id'],)).fetchone()
            if not user or not user['enabled']:
                config['enabled'] = False
                _cancel(conn, config['batch_id'])
                _write(conn, row['key'], config)
                continue
            if config.get('job_id'):
                job = conn.execute('SELECT * FROM jobs WHERE id=?', (config['job_id'],)).fetchone()
                parts = conn.execute('SELECT state FROM deliveries WHERE report_id=?', (json.loads(job['payload'])['report_id'],)).fetchall()
                if job['state'] in {'failed', 'cancelled'} or any(p['state'] == 'failed' for p in parts):
                    config.update(enabled=False, error=True)
                    _write(conn, row['key'], config)
                    continue
                if job['state'] == 'done' and not parts:
                    conn.execute("UPDATE jobs SET state='pending' WHERE id=?", (job['id'],))
                    continue
                if job['state'] != 'done' or any(p['state'] != 'sent' for p in parts):
                    continue
                config.update(cursor=config['cursor']+1, job_id=None)
            if config['cursor'] >= config['pages']:
                config['enabled'] = False
            else:
                payload = {**config['request'], 'actor_id': config['actor_id'], 'recipient': str(config['actor_id']),
                           'report_id': uuid4().hex, 'page': config['cursor'], 'chart_batch_id': config['batch_id'],
                           'chart_auto': {name: config[name] for name in ('batch_id', 'actor_id', 'generation')}}
                result = conn.execute('INSERT INTO jobs(kind,payload,dedupe_key,created_at) VALUES(?,?,?,?)',
                                      ('telegram_chart', dumps(payload), f"chart-auto:{config['batch_id']}:{config['cursor']}", utcnow()))
                config['job_id'] = result.lastrowid
            _write(conn, row['key'], config)
