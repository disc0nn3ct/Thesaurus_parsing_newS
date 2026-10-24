"""Personal chart schedules persisted in SQLite, with revocable deliveries."""
import json
from datetime import datetime, timezone
from uuid import uuid4

from .charts import SECTIONS, validate_options, validate_selection
from .scheduling import _latest_due, validate_subscription
from .store import dumps, utcnow
from .telegram_data_jobs import require_actor


def key(actor, section):
    if type(actor) is not int or actor <= 0 or section not in SECTIONS:
        raise ValueError('Неизвестное расписание графика')
    return f'chart_schedule:{actor}:{section}'


def defaults(actor, section, options):
    return {'actor_id': actor, 'section': section, 'enabled': False, 'times': ['15:30'], 'weekdays': [4],
            'timezone': 'Europe/Moscow', 'options': validate_options(options), 'refresh': section == 'ruonia',
            'generation': uuid4().hex, 'created_at': utcnow(), 'last_scheduled_at': None}


def active(store, reference, recipient=None):
    if not isinstance(reference, dict):
        return False
    try:
        actor, section = reference['actor_id'], reference['section']
        from .search_provider import require_section
        require_section(store, section)
        config = store.get_setting(key(actor, section), {})
    except (KeyError, ValueError, TypeError):
        return False
    from .subscription_access import allowed
    return bool(allowed(store, actor, section) and config.get('enabled') and
                config.get('generation') == reference.get('generation') and
                (recipient is None or str(actor) == str(recipient)))


def save(store, actor, section, config, **changes):
    require_actor(store, actor)
    from .subscription_access import require
    require(store, actor, section)
    from .search_provider import require_section
    require_section(store, section)
    setting_key = key(actor, section)
    updated = {**config, **changes, 'actor_id': actor, 'section': section}
    validate_subscription({field: updated[field] for field in ('enabled', 'times', 'weekdays', 'timezone')})
    updated['options'] = validate_options(updated['options'])
    if updated['enabled']:
        validate_selection(store, actor, section, updated['options'].get('identity'))
        if updated['options'].get('selection_mode') == 'multiple' and not updated['options'].get('series_ids'):
            raise ValueError('Выберите хотя бы один показатель для расписания')
    if updated['enabled'] and updated['options']['period'] == 'custom':
        raise ValueError('Для регулярной отправки выберите скользящий период: месяц, 3/6 месяцев, год или всю историю')
    if updated != config:
        updated.update(generation=uuid4().hex, created_at=utcnow(), last_scheduled_at=None)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        require(store, actor, section)
        user = conn.execute('SELECT enabled FROM users WHERE user_id=?', (actor,)).fetchone()
        if not user or not user['enabled']:
            raise PermissionError('Нет доступа')
        conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (setting_key, dumps(updated)))
        if updated.get('generation') != config.get('generation') or not updated['enabled']:
            for job in conn.execute("SELECT id,payload FROM jobs WHERE kind='telegram_chart' AND state='pending'").fetchall():
                reference = json.loads(job['payload']).get('chart_schedule', {})
                if reference.get('actor_id') == actor and reference.get('section') == section:
                    conn.execute("UPDATE jobs SET state='cancelled',finished_at=? WHERE id=?", (utcnow(), job['id']))
            for delivery in conn.execute("SELECT id,payload FROM deliveries WHERE recipient=? AND state='pending'", (str(actor),)).fetchall():
                reference = json.loads(delivery['payload']).get('chart_schedule', {})
                if reference.get('actor_id') == actor and reference.get('section') == section:
                    conn.execute("UPDATE deliveries SET state='cancelled',updated_at=? WHERE id=?", (utcnow(), delivery['id']))
    return updated


def enqueue_due(store, now=None):
    now = now or datetime.now(timezone.utc)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        for setting in conn.execute("SELECT key,value FROM settings WHERE key LIKE 'chart_schedule:%'").fetchall():
            try:
                config = json.loads(setting['value'])
                from .search_provider import require_section
                require_section(store, config['section'])
                if key(config['actor_id'], config['section']) != setting['key']:
                    continue
                user = conn.execute('SELECT enabled FROM users WHERE user_id=?', (config['actor_id'],)).fetchone()
                validate_options(config['options'])
                slot = _latest_due({**config, 'times': dumps(config['times']), 'weekdays': dumps(config['weekdays'])}, now)
            except (KeyError, TypeError, ValueError):
                continue
            if not config.get('enabled') or not user or not user['enabled']:
                continue
            from .subscription_access import allowed
            if not allowed(store, config['actor_id'], config['section']):
                continue
            from .feature_access import allowed as function_allowed
            if not all(function_allowed(store, config['actor_id'], name) for name in ('charts', 'schedules')):
                continue
            if slot is None:
                continue
            reference = {field: config[field] for field in ('actor_id', 'section', 'generation')}
            options = {**config['options'], 'anchor': 'today'}
            prefix = f"chart-schedule:{config['actor_id']}:{config['section']}:{config['generation']}:"
            pending = conn.execute("SELECT id,payload FROM jobs WHERE state='pending' AND dedupe_key LIKE ?", (prefix+'%',)).fetchone()
            payload = {'actor_id': config['actor_id'], 'recipient': str(config['actor_id']), 'section': config['section'],
                       'identity': options.get('identity'), 'options': options, 'chart_schedule': reference,
                       'refresh': config.get('refresh', False), 'scheduled_for': slot.isoformat(),
                       'report_id': json.loads(pending['payload'])['report_id'] if pending else uuid4().hex}
            if pending:
                conn.execute('UPDATE jobs SET payload=?,dedupe_key=? WHERE id=?', (dumps(payload), prefix+slot.isoformat(), pending['id']))
            else:
                conn.execute("INSERT OR IGNORE INTO jobs(kind,payload,dedupe_key,created_at) VALUES('telegram_chart',?,?,?)",
                             (dumps(payload), prefix+slot.isoformat(), utcnow()))
            config['last_scheduled_at'] = slot.isoformat()
            conn.execute('UPDATE settings SET value=? WHERE key=?', (dumps(config), setting['key']))
