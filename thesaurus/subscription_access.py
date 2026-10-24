"""Admin-assigned project/section entitlements, separate from personal schedules."""
import json
import sqlite3
from contextlib import closing
from uuid import uuid4

from .store import dumps, utcnow

CATALOG = {'market': 'Рынок и ЦБ · доллар, аналитика, сигналы', 'ruonia': 'RUONIA · графики и отчёты',
           'ai': 'Новости и AI', 'wordstat': 'Wordstat', 'regions': 'Региональные индикаторы',
           'vivino': 'Жизнь → Вино · Vivino и винотека',
           'life': 'Жизнь → Доходы, расходы и чеки'}
MARKET = set(CATALOG)-{'vivino', 'life'}


def key(actor):
    return f'user_subscriptions:{actor}'


def normalize(values):
    if values == 'all':
        return list(CATALOG)
    if not isinstance(values, (list, tuple)) or any(value not in CATALOG for value in values):
        raise ValueError('Неизвестный проект или раздел')
    return list(dict.fromkeys(values))


def migrate(store):
    """Freeze pre-existing rights once; new users start with no subscriptions."""
    with store.connection() as conn:
        if conn.execute("SELECT 1 FROM settings WHERE key='subscription_access_version'").fetchone():
            return
        users = conn.execute('SELECT user_id FROM users').fetchall()
        if users:
            backup = store.path.with_name(store.path.name+f'.access-backup-{uuid4().hex}.sqlite3')
            with closing(sqlite3.connect(backup)) as target:
                conn.backup(target)
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute("SELECT 1 FROM settings WHERE key='subscription_access_version'").fetchone():
            return
        for user in conn.execute('SELECT user_id FROM users').fetchall():
            conn.execute('INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)',
                         (key(user['user_id']), dumps(list(CATALOG))))
        conn.execute("INSERT INTO settings(key,value) VALUES('subscription_access_version','1')")


def assigned(store, actor):
    if not store.authorize(actor):
        return []
    if store.authorize(actor, admin=True):
        return list(CATALOG)
    return normalize(store.get_setting(key(actor), []))


def allowed(store, actor, section):
    if store.authorize(actor, admin=True):
        return True
    if section == 'trends':
        section = 'market'
    return section in assigned(store, actor)


def require(store, actor, section):
    if not allowed(store, actor, section):
        raise PermissionError('Этот раздел не подключён для вас. Обратитесь к администратору.')


def project_allowed(store, actor, identifier):
    values = set(assigned(store, actor))
    from .feature_access import assigned as functions
    from .feature_access import for_sections
    parents = values & MARKET if identifier == 'market' else values & ({'vivino', 'life'} if identifier == 'life' else {identifier})
    return bool(set(functions(store, actor)) & set(for_sections(parents)))


def report_section(report):
    quality = json.loads(report['quality'])
    if quality.get('section') in CATALOG:
        return quality['section']
    return 'market' if report['section'] == 'trends' else report['section']


def job_section(kind, payload):
    if kind.startswith('life_'):
        return 'life'
    if kind.startswith(('vivino', 'wine_')):
        return 'vivino'
    if kind.startswith(('market_', 'dollar_')):
        return 'market'
    return payload.get('section')


def change(store, actor, target, section, enabled):
    """Serialize grants with revocation; never enable a schedule or change roles."""
    if section not in CATALOG or type(target) is not int or target <= 0 or type(enabled) is not bool:
        raise ValueError('Неверная настройка доступа')
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        admin = conn.execute("SELECT 1 FROM users WHERE user_id=? AND role='admin' AND enabled=1", (actor,)).fetchone()
        user = conn.execute('SELECT role FROM users WHERE user_id=?', (target,)).fetchone()
        if not admin:
            raise PermissionError('Действие доступно администратору')
        if not user:
            raise ValueError('Сначала добавьте пользователя')
        if user['role'] == 'admin':
            raise ValueError('Администраторы имеют доступ ко всем разделам')
        record = conn.execute('SELECT value FROM settings WHERE key=?', (key(target),)).fetchone()
        values = normalize(json.loads(record['value']) if record else [])
        if (section in values) == enabled:
            return
        from .user_management import access_state, queue_notice
        before = access_state(conn, target)
        values = [*values, section] if enabled else [s for s in values if s != section]
        conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (key(target), dumps(values)))
        if enabled:
            from .feature_access import for_sections
            from .feature_access import key as feature_key
            record = conn.execute('SELECT value FROM settings WHERE key=?', (feature_key(target),)).fetchone()
            functions = json.loads(record['value']) if record else []
            functions = list(dict.fromkeys([*functions, *for_sections([section])]))
            conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                         (feature_key(target), dumps(functions)))
        if not enabled:
            conn.execute('UPDATE subscriptions SET enabled=0,last_scheduled_at=? WHERE user_id=? AND section=?',
                         (utcnow(), target, section))
            schedule_key = f'chart_schedule:{target}:{section}'
            schedule = conn.execute('SELECT value FROM settings WHERE key=?', (schedule_key,)).fetchone()
            if schedule:
                config = json.loads(schedule['value'])
                config.update(enabled=False, generation=uuid4().hex, last_scheduled_at=utcnow())
                conn.execute('UPDATE settings SET value=? WHERE key=?', (dumps(config), schedule_key))
            conn.execute('UPDATE bot_sessions SET pending=NULL WHERE user_id=?', (target,))
            if section in {'market', 'ruonia'}:
                for rule in conn.execute('SELECT r.id,s.metadata FROM market_rules r LEFT JOIN market_series s ON s.id=r.series_id WHERE r.user_id=?', (target,)).fetchall():
                    ruonia = bool(rule['metadata'] and json.loads(rule['metadata']).get('adapter') == 'stored_ruonia')
                    if section == 'market' or ruonia:
                        conn.execute("UPDATE market_rules SET enabled=0,mode='off' WHERE id=?", (rule['id'],))
            for batch in conn.execute("SELECT key,value FROM settings WHERE key LIKE 'chart_batch:%'").fetchall():
                config = json.loads(batch['value'])
                if config.get('actor_id') == target and config.get('request', {}).get('section') == section:
                    config.update(enabled=False, generation=uuid4().hex)
                    conn.execute('UPDATE settings SET value=? WHERE key=?', (dumps(config), batch['key']))
            for job in conn.execute("SELECT id,kind,payload FROM jobs WHERE state='pending'").fetchall():
                payload = json.loads(job['payload'])
                if (payload.get('actor_id') == target or str(payload.get('recipient')) == str(target)) and job_section(job['kind'], payload) == section:
                    conn.execute("UPDATE jobs SET state='cancelled',finished_at=? WHERE id=?", (utcnow(), job['id']))
            for row in conn.execute("SELECT d.id,r.section,r.quality FROM deliveries d JOIN reports r ON r.id=d.report_id WHERE d.recipient=? AND d.state IN ('pending','retry')", (str(target),)).fetchall():
                if report_section(row) == section:
                    conn.execute("UPDATE deliveries SET state='cancelled',updated_at=? WHERE id=?", (utcnow(), row['id']))
        conn.execute('INSERT INTO access_audit(actor_id,target_id,action,created_at) VALUES(?,?,?,?)',
                     (actor, target, ('grant:' if enabled else 'revoke:')+section, utcnow()))
        queue_notice(conn, target, 'grant' if enabled else 'revoke', before=before)


def route_requirement(data):
    """Map UI actions to entitlements, including forged and old callbacks."""
    parts = data.split(':')
    prefix = parts[0]
    if prefix in {'lf', 'rc'}:
        return 'life'
    if prefix in {'vi', 'va', 'wc', 'wn'} or data == 'pj:vivino':
        return 'vivino'
    if prefix in {'usd', 'cb', 'mk', 'mi', 'ma', 'gt'}:
        return 'market'
    if prefix == 'rg' or data == 'menu:regions':
        return 'regions'
    if prefix in {'request', 'sub'} and len(parts) >= 2:
        return parts[1]
    if prefix == 'cs' and len(parts) >= 2:
        return parts[1]
    if prefix == 'kw' and len(parts) >= 3:
        return parts[2]
    if prefix == 'ch':
        return next((p for p in parts[1:] if p in {*CATALOG, 'trends'}), 'chart_project')
    if prefix == 'ui' and len(parts) >= 2:
        if parts[1] in {'subscription', 'subscriptioninput', 'backtest', 'export', 'forecasts'}:
            return parts[2] if len(parts) >= 3 and parts[2] in CATALOG else 'chart_project'
        if parts[1].startswith(('rule', 'digest')):
            return 'market'
        if parts[1].startswith(('pair', 'forecast')):
            return 'market'
    if data in {'menu:indicators'}:
        return 'market'
    if data == 'menu:ruonia_forecasts':
        return 'ruonia'
    if data == 'menu:forecasts':
        return 'wordstat'
    if data in {'menu:graphs', 'menu:data', 'menu:backtest_help', 'menu:export_help'}:
        return 'chart_project'
    if data in {'pj:market', 'menu:settings', 'menu:graphs', 'menu:report', 'menu:data', 'menu:subscriptions',
                'menu:help', 'menu:keywords', 'menu:backtest_help', 'menu:export_help'}:
        return 'market_project'
    return None


def route_allowed(store, actor, data):
    from .feature_access import route_allowed as function_allowed
    if not function_allowed(store, actor, data):
        return False
    if data in {'pj:market', 'pj:vivino', 'pj:life'} and not project_allowed(store, actor, data.split(':')[1]):
        return False
    requirement = route_requirement(data)
    if requirement == 'chart_project':
        return bool(set(assigned(store, actor)) & {'ruonia', 'market', 'wordstat'})
    if requirement == 'market_project':
        return project_allowed(store, actor, 'market')
    return requirement is None or allowed(store, actor, requirement)


def require_route(store, actor, data):
    if not route_allowed(store, actor, data):
        raise PermissionError('Этот раздел не подключён для вас. Обратитесь к администратору.')


def command_route(command, argument=''):
    if command == 'monitor':
        return 'mon:home'
    section = argument.split()[0] if argument.split() else None
    if command == 'life':
        return 'pj:life'
    if command in {'expenses', 'receipt'}:
        return 'lf:receipt' if command == 'receipt' else 'lf:home'
    if command == 'vivino':
        return 'vi:menu'
    if command == 'wines':
        return 'pj:vivino'
    if command == 'cellar':
        return 'wc:menu'
    if command == 'dollar':
        return 'usd:menu'
    if command == 'chart':
        return 'ch:menu'+(':'+section if section else '')
    if command in {'export', 'backtest', 'forecasts'}:
        default = 'wordstat' if command in {'backtest', 'forecasts'} else 'market'
        return 'ui:'+command+':'+(section if section in CATALOG else default)
    if command == 'report':
        return 'request:'+section if section in CATALOG else 'menu:report'
    if command == 'notifications':
        return 'mk:open:alerts'
    if command in {'indicators', 'correlations'}:
        return 'mk:open:'+command
    if command in {'market', 'trends'}:
        return 'mk:menu:all'
    if command in {'subscriptions', 'graphs', 'settings', 'regions', 'keywords'}:
        if command == 'subscriptions' and section in CATALOG:
            return 'sub:'+section
        return 'menu:'+command
    return None
