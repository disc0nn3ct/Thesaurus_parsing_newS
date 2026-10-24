"""Durable operational history with bounded, redacted request descriptions."""
import json
import logging
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .store import dumps, utcnow

logger = logging.getLogger(__name__)
VERSION = 1
CATEGORIES = {'all': 'Все события', 'interaction': 'Действия пользователей', 'access': 'Права доступа',
              'identity': 'Имена и профили',
              'job': 'Задания', 'delivery': 'Доставка', 'source': 'Источники',
              'wine': 'Винотека', 'finance': 'Доходы и расходы', 'schedule': 'Расписания',
              'service': 'Сервер', 'errors': 'Ошибки и предупреждения'}
SCHEMA = [
    '''CREATE TABLE monitor_events(id INTEGER PRIMARY KEY,occurred_at TEXT NOT NULL,
       category TEXT NOT NULL,action TEXT NOT NULL,outcome TEXT NOT NULL,level TEXT NOT NULL DEFAULT 'info',
       actor_id INTEGER,subject_id INTEGER,entity_type TEXT,entity_id TEXT,duration_ms REAL,
       details TEXT NOT NULL DEFAULT '{}',origin TEXT NOT NULL DEFAULT 'live',source_key TEXT UNIQUE)''',
    'CREATE INDEX monitor_event_time ON monitor_events(occurred_at DESC,id DESC)',
    'CREATE INDEX monitor_event_actor ON monitor_events(actor_id,occurred_at DESC,id DESC)',
    'CREATE INDEX monitor_event_subject ON monitor_events(subject_id,occurred_at DESC,id DESC)',
    'CREATE INDEX monitor_event_category ON monitor_events(category,occurred_at DESC,id DESC)',
    '''CREATE TABLE monitor_runtime(id TEXT PRIMARY KEY,pid INTEGER NOT NULL,host TEXT NOT NULL,
       started_at TEXT NOT NULL,heartbeat_at TEXT NOT NULL,finished_at TEXT,state TEXT NOT NULL,
       telegram_connected INTEGER NOT NULL DEFAULT 0)''',
    '''CREATE TABLE monitor_samples(id INTEGER PRIMARY KEY,runtime_id TEXT NOT NULL REFERENCES monitor_runtime(id),
       created_at TEXT NOT NULL,metrics TEXT NOT NULL)''',
    'CREATE INDEX monitor_sample_time ON monitor_samples(created_at DESC,id DESC)',
]


def _job_actor(alias):
    payload = f'{alias}.payload'
    return (f"CASE WHEN json_valid({payload}) THEN CASE WHEN json_type({payload},'$.actor_id')='integer' "
            f"AND json_extract({payload},'$.actor_id')>0 THEN json_extract({payload},'$.actor_id') "
            f"ELSE (SELECT user_id FROM users WHERE chat_id=CAST(json_extract({payload},'$.recipient') AS TEXT)) END END")


def _definitions():
    """SQL expressions reference only whitelisted metadata columns, never raw payloads."""
    actor = _job_actor('NEW')
    recipient = '(SELECT user_id FROM users WHERE chat_id=NEW.recipient)'
    return [
        {'table': 'jobs', 'category': 'job', 'action': 'NEW.kind', 'state': 'NEW.state',
         'actor': actor, 'subject': actor, 'time': 'COALESCE(NEW.finished_at,NEW.started_at,NEW.created_at)',
         'changed': 'OLD.state IS NOT NEW.state',
         'duration': "CASE WHEN NEW.finished_at IS NOT NULL AND NEW.started_at IS NOT NULL THEN MAX(0,(julianday(NEW.finished_at)-julianday(NEW.started_at))*86400000) END"},
        {'table': 'deliveries', 'category': 'delivery', 'action': "'report'", 'state': 'NEW.state',
         'subject': recipient, 'time': 'NEW.updated_at', 'changed': 'OLD.state IS NOT NEW.state'},
        {'table': 'access_audit', 'category': 'access', 'action': 'NEW.action', 'state': "'ok'",
         'actor': 'NEW.actor_id', 'subject': 'NEW.target_id', 'time': 'NEW.created_at'},
        {'table': 'access_notices', 'category': 'delivery', 'action': "'access_notice'", 'state': 'NEW.state',
         'subject': 'NEW.user_id', 'time': 'NEW.updated_at', 'changed': 'OLD.state IS NOT NEW.state'},
        {'table': 'user_profiles', 'category': 'identity', 'action': "'profile_updated'", 'state': "'ok'",
         'subject': 'NEW.user_id', 'time': 'NEW.updated_at', 'entity': 'CAST(NEW.user_id AS TEXT)',
         'changed': 'OLD.first_name IS NOT NEW.first_name OR OLD.last_name IS NOT NEW.last_name OR OLD.username IS NOT NEW.username',
         'details': "json_object('username',NEW.username)",
         'update_details': "json_object('old_username',OLD.username,'username',NEW.username)"},
        {'table': 'runs', 'category': 'source', 'action': 'NEW.section', 'state': 'NEW.status',
         'time': 'COALESCE(NEW.completed_at,NEW.started_at)', 'changed': 'OLD.status IS NOT NEW.status'},
        {'table': 'market_series', 'category': 'source', 'action': 'NEW.id', 'state': 'NEW.status',
         'time': "COALESCE(NEW.updated_at,strftime('%Y-%m-%dT%H:%M:%f+00:00','now'))",
         'level': "CASE WHEN NEW.status IN ('error','failed') THEN 'error' WHEN NEW.error IS NOT NULL AND NEW.error<>'' THEN 'warning' ELSE 'info' END",
         'changed': 'OLD.status IS NOT NEW.status OR OLD.error IS NOT NEW.error',
         'insert_when': "NEW.updated_at IS NOT NULL OR NEW.status<>'not_loaded'",
         'snapshot_where': "NEW.updated_at IS NOT NULL OR NEW.status<>'not_loaded'"},
        {'table': 'wine_events', 'category': 'wine',
         'action': "CASE WHEN NEW.delta>0 THEN 'stock_added' ELSE 'stock_removed' END", 'state': "'ok'",
         'subject': 'NEW.user_id', 'actor': 'NEW.user_id', 'time': 'NEW.created_at'},
        {'table': 'life_ledger_entries', 'category': 'finance', 'action': 'NEW.kind',
         'state': "CASE WHEN NEW.voided_at IS NOT NULL THEN 'cancelled' ELSE 'ok' END",
         'subject': 'NEW.user_id', 'actor': 'NEW.user_id', 'time': 'COALESCE(NEW.voided_at,NEW.created_at)',
         'changed': 'OLD.voided_at IS NOT NEW.voided_at'},
        {'table': 'life_receipt_imports', 'category': 'finance', 'action': "'receipt_confirmed'",
         'state': "'ok'", 'subject': 'NEW.user_id', 'actor': 'NEW.user_id', 'time': 'NEW.confirmed_at'},
        {'table': 'subscriptions', 'category': 'schedule', 'action': 'NEW.section',
         'state': "CASE WHEN NEW.enabled=1 THEN 'enabled' ELSE 'disabled' END", 'subject': 'NEW.user_id',
         'time': "strftime('%Y-%m-%dT%H:%M:%f+00:00','now')",
         'details': "json_object('times',NEW.times,'enabled',NEW.enabled)",
         'update_details': "json_object('old_times',OLD.times,'times',NEW.times,'old_enabled',OLD.enabled,'enabled',NEW.enabled)",
         'changed': 'OLD.enabled IS NOT NEW.enabled OR OLD.times IS NOT NEW.times OR OLD.weekdays IS NOT NEW.weekdays OR OLD.timezone IS NOT NEW.timezone OR OLD.detail IS NOT NEW.detail'},
        {'table': 'integrity_incidents', 'category': 'source', 'action': 'NEW.kind', 'state': 'NEW.state',
         'level': "CASE WHEN NEW.state IN ('resolved','dismissed','accepted') THEN 'info' WHEN NEW.severity IN ('error','critical') THEN 'error' ELSE 'warning' END",
         'time': 'COALESCE(NEW.resolved_at,NEW.created_at)', 'changed': 'OLD.state IS NOT NEW.state'},
    ]


def migrate(store):
    with store.connection() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key='monitoring_version'").fetchone()
        if row and int(row['value']) >= VERSION:
            conn.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('monitoring_details_started_at',?)",(dumps(utcnow()),))
            return
        backup = store.path.with_name(store.path.name + f'.monitor-backup-{uuid4().hex}.sqlite3')
        with closing(sqlite3.connect(backup)) as target:
            conn.backup(target)
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute("SELECT 1 FROM settings WHERE key='monitoring_version'").fetchone():
            return
        for statement in SCHEMA:
            conn.execute(statement)
        for spec in _definitions():
            table = spec['table']
            state = spec['state']
            level = spec.get('level', f"CASE WHEN {state}='failed' THEN 'error' WHEN {state} IN ('uncertain','retry','partial') THEN 'warning' ELSE 'info' END")
            columns = 'occurred_at,category,action,outcome,level,actor_id,subject_id,entity_type,entity_id,duration_ms,details,origin,source_key'
            values = [spec['time'], f"'{spec['category']}'", spec['action'], state, level,
                      spec.get('actor', 'NULL'), spec.get('subject', 'NULL'), f"'{table}'", spec.get('entity','CAST(NEW.id AS TEXT)'),
                      spec.get('duration', 'NULL'), spec.get('details',"'{}'")]
            # Existing rows are snapshots of the known state, not invented intermediate transitions.
            historical_time = 'NEW.created_at' if table == 'subscriptions' else spec['time']
            historical = [historical_time, *values[1:], "'history'", f"'{table}:'||{spec.get('entity','NEW.id')}||':snapshot'"]
            historical_where = ' WHERE '+spec['snapshot_where'] if spec.get('snapshot_where') else ''
            conn.execute(f"INSERT INTO monitor_events({columns}) SELECT {','.join(historical)} FROM {table} AS NEW{historical_where}")
            for operation in ('INSERT', 'UPDATE') if 'changed' in spec else ('INSERT',):
                condition = spec['changed'] if operation == 'UPDATE' else spec.get('insert_when')
                when = ' WHEN '+condition if condition else ''
                live_values = list(values)
                if operation == 'UPDATE':
                    live_values[0] = "strftime('%Y-%m-%dT%H:%M:%f+00:00','now')"
                if operation == 'UPDATE' and spec.get('update_details'):
                    live_values[-1] = spec['update_details']
                conn.execute(f'''CREATE TRIGGER monitor_{table}_{operation.lower()} AFTER {operation} ON {table}{when}
                    BEGIN INSERT INTO monitor_events({columns}) VALUES({','.join(live_values)},'live',NULL); END''')
        conn.execute("INSERT INTO settings(key,value) VALUES('monitoring_version',?)", (dumps(VERSION),))
        conn.execute("INSERT INTO settings(key,value) VALUES('monitoring_started_at',?)", (dumps(utcnow()),))
        conn.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('monitoring_details_started_at',?)",(dumps(utcnow()),))


def safe_record(store, category, action, *, outcome='ok', level='info', actor=None, subject=None,
                entity_type=None, entity_id=None, duration_ms=None, details=None):
    """Monitoring outages must not interrupt an authorized user operation."""
    try:
        return store.execute('''INSERT INTO monitor_events(occurred_at,category,action,outcome,level,
            actor_id,subject_id,entity_type,entity_id,duration_ms,details) VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
            (utcnow(), category, action, outcome, level, actor, subject, entity_type,
             str(entity_id) if entity_id is not None else None, duration_ms, dumps(details or {})))
    except sqlite3.Error as exc:
        logger.warning('Operational history write failed (%s)', type(exc).__name__)
        return None


def interaction_action(function_name, event, pending=None, *, callback=False):
    if callback:
        data = getattr(event, 'data', '')
        parts = data.split(':') if isinstance(data, str) else []
        allowed = {'menu','pj','ua','user','wc','wn','lf','rc','vi','va','usd','cb','ch','cs','mk','ms','mi','ma',
                   'ig','gt','ui','kw','rg','sub','su','alert','proposal','adminpage','report','history','request','resolve','mon'}
        if len(parts) >= 2 and parts[0] in allowed and re.fullmatch('[a-z_]{1,32}', parts[1]):
            return 'button:'+':'.join(parts[:2])
        return 'button:other'
    if function_name == 'handle_document':
        return 'upload:document'
    if function_name == 'handle_wine_photo':
        return 'upload:photo'
    if function_name == 'handle_other_message':
        from .monitor_content import request_details
        kind = request_details(function_name,event).get('kind','other')
        return 'upload:'+kind
    text = getattr(event, 'text', '')
    if isinstance(text, str) and text.startswith('/'):
        command = text.split(maxsplit=1)[0][1:].split('@')[0].lower()
        commands = {'start','projects','wines','life','expenses','receipt','cellar','vivino','dollar','chart','graphs',
                    'subscriptions','notifications','report','settings','status','help','admin','autocollect','integrity',
                    'sources','users','allow','deny','jobs','budget','alerts','proposals','resolve','export','backtest',
                    'forecasts','indicators','correlations','market','regions','keywords','cancel','menu','monitor','trends'}
        return 'command:'+command if command in commands else 'command:other'
    prefix = pending.split(':')[0] if isinstance(pending, str) else ''
    return 'input:'+prefix if prefix in {'add_user','sourceunit','command','dollar','usd','life','wine','wc','rc','lf',
                                       'vivino','chart','charts','cbr','trends','regions','keywords','schedule'} else 'input:form'


def record_interaction(store, actor, action, state, duration_ms):
    error = state.get('monitor_error')
    outcome = state.get('monitor_outcome', 'ok')
    level = 'error' if outcome == 'failed' else 'warning' if outcome in {'denied','invalid','stale','unsupported'} else 'info'
    details = {'error_type': error} if error else {}
    for key in ('request','response','job_ids'):
        if state.get('monitor_'+key):
            details[key] = state['monitor_'+key]
    safe_record(store, 'interaction', action, actor=actor, subject=actor, outcome=outcome,
                level=level, duration_ms=round(duration_ms, 1), details=details)


def mark_error(error):
    from .telegram_navigation import CURRENT
    state = CURRENT.get()
    if state is not None:
        state['monitor_outcome'] = 'denied' if isinstance(error, PermissionError) else 'invalid' if isinstance(error, ValueError) else 'failed'
        state['monitor_error'] = type(error).__name__


def response_jobs(store, actor, text):
    """Link an existing queued request only when its saved owner matches the actor."""
    from .telegram_navigation import CURRENT
    state = CURRENT.get()
    if state is None or not isinstance(text,str):
        return
    identities = re.findall(r'(?:[Зз]адани[ея]|[Рр]аспознавание)\s*№\s*(\d{1,18})',text)
    for value in identities[:10]:
        identity = int(value)
        if store.one(f'SELECT id FROM jobs WHERE id=? AND {_job_actor("jobs") }=?',(identity,actor)):
            jobs = state.setdefault('monitor_job_ids',[])
            if identity not in jobs and len(jobs)<20:
                jobs.append(identity)


def filters(scope='all', days=7, category='all', *, until=None):
    if scope not in {'all','server'} and not re.fullmatch(r'u[1-9][0-9]{0,18}', scope):
        raise ValueError('Неверный пользователь журнала')
    if days not in {0,1,7,30} or category not in CATEGORIES:
        raise ValueError('Неверный фильтр журнала')
    end = datetime.fromisoformat(until) if until else datetime.now(timezone.utc)
    if end.tzinfo is None:
        raise ValueError('Нужен часовой пояс времени журнала')
    clauses, values = ['occurred_at<=?'], [end.astimezone(timezone.utc).isoformat()]
    if days:
        clauses.append('occurred_at>=?')
        values.append((end-timedelta(days=days)).astimezone(timezone.utc).isoformat())
    if scope == 'server':
        clauses.append('actor_id IS NULL AND subject_id IS NULL')
    elif scope.startswith('u'):
        clauses.append('(actor_id=? OR subject_id=?)')
        values.extend([int(scope[1:])] * 2)
    if category == 'errors':
        clauses.append("level IN ('error','warning')")
    elif category != 'all':
        clauses.append('category=?')
        values.append(category)
    return ' AND '.join(clauses), values


def events(store, scope='all', days=7, category='all', *, page=0, limit=8, until=None):
    if type(page) is not int or not 0 <= page <= 100000 or not 1 <= limit <= 10001:
        raise ValueError('Неверная страница журнала')
    where, values = filters(scope, days, category, until=until)
    total = store.one(f'SELECT COUNT(*) n FROM monitor_events WHERE {where}', values)['n']
    rows = store.query(f'SELECT * FROM monitor_events WHERE {where} ORDER BY occurred_at DESC,id DESC LIMIT ? OFFSET ?',
                       [*values, limit, page*limit])
    return rows, total


def event_summary(store, scope='all', days=7):
    where, values = filters(scope, days)
    return store.query(f'SELECT category,level,COUNT(*) n FROM monitor_events WHERE {where} GROUP BY category,level', values)


def user_list(store, page=0):
    if type(page) is not int or not 0 <= page <= 100000:
        raise ValueError('Неверная страница пользователей')
    return store.query('''WITH identities AS (SELECT user_id FROM users UNION SELECT user_id FROM user_profiles)
        SELECT i.user_id,u.role,COALESCE(u.enabled,0) enabled,p.first_name,p.last_name,p.username,p.last_seen,
        (SELECT MAX(occurred_at) FROM monitor_events e WHERE e.actor_id=i.user_id OR e.subject_id=i.user_id) last_event
        FROM identities i LEFT JOIN users u ON u.user_id=i.user_id LEFT JOIN user_profiles p ON p.user_id=i.user_id
        ORDER BY i.user_id LIMIT 9 OFFSET ?''', (page*8,))


def server_snapshot(store, *, now=None):
    now = now or datetime.now(timezone.utc)
    runtime = store.one('SELECT * FROM monitor_runtime ORDER BY started_at DESC LIMIT 1')
    sample = store.one('SELECT * FROM monitor_samples WHERE runtime_id=? ORDER BY id DESC LIMIT 1', (runtime['id'],)) if runtime else None
    age = max(0,(now-datetime.fromisoformat(runtime['heartbeat_at'])).total_seconds()) if runtime else None
    active = bool(runtime and runtime['state'] in {'starting','running'} and age <= 45)
    queues = store.query('SELECT state,COUNT(*) n FROM jobs GROUP BY state')
    deliveries = store.query('SELECT state,COUNT(*) n FROM deliveries GROUP BY state')
    sources = store.query('SELECT status,COUNT(*) n FROM market_series WHERE enabled=1 GROUP BY status')
    notices = store.query('SELECT state,COUNT(*) n FROM access_notices GROUP BY state')
    running_jobs = store.query("SELECT id,kind,started_at FROM jobs WHERE state='running' ORDER BY started_at LIMIT 3")
    oldest_pending = store.one("SELECT created_at FROM jobs WHERE state='pending' ORDER BY created_at LIMIT 1")
    latest_run = store.one("SELECT kind,finished_at FROM jobs WHERE state='done' ORDER BY finished_at DESC LIMIT 1")
    return {'runtime': runtime, 'heartbeat_age': age, 'active': active,
            'sample_at': sample['created_at'] if sample else None, 'metrics': json.loads(sample['metrics']) if sample else {},
            'queues': {row['state']:row['n'] for row in queues},
            'deliveries': {row['state']:row['n'] for row in deliveries},
            'sources': {row['status']:row['n'] for row in sources}, 'latest_done_job': latest_run,
            'notices': {row['state']:row['n'] for row in notices},
            'running_jobs': running_jobs, 'oldest_pending': oldest_pending}
