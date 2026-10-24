"""Function entitlements layered over projects; enforced again by worker/delivery."""
import json

from .store import dumps, utcnow

CATALOG = {
    'dollar': ('Доллар · обзор, прогноз и моя покупка', ('market',)),
    'cbr': ('Каталог показателей Банка России', ('market',)),
    'charts': ('Графики с прогнозом', ('market', 'ruonia', 'wordstat')),
    'reports': ('Отчёты и обзор рынка', ('market', 'ruonia', 'ai', 'wordstat', 'regions')),
    'correlations': ('Корреляции показателей', ('market',)),
    'backtests': ('Проверка и оценка прогнозов', ('market', 'ruonia', 'wordstat')),
    'exports': ('Экспорт CSV и Excel', ('market', 'ruonia', 'wordstat', 'regions')),
    'schedules': ('Личные расписания доставки', ('market', 'ruonia', 'ai', 'wordstat', 'regions')),
    'alerts': ('Настройка и получение уведомлений', ('market', 'ruonia', 'wordstat', 'regions')),
    'vivino_map': ('Vivino · загрузка данных и карта', ('vivino',)),
    'vivino_plan': ('Vivino · каталог и план дегустаций', ('vivino',)),
    'wine_cellar': ('Винотека · добавление вин, запасы и стоимость', ('vivino',)),
    'life_finance': ('Жизнь · доходы, расходы и распознавание чеков', ('life',)),
}


def key(actor):
    return f'user_features:{actor}'


def for_sections(sections):
    return [name for name, (_, parents) in CATALOG.items() if set(parents) & set(sections)]


def assigned(store, actor):
    if not store.authorize(actor):
        return []
    if store.authorize(actor, admin=True):
        return list(CATALOG)
    return configured(store, actor)


def configured(store, actor):
    from .subscription_access import key as section_key
    from .subscription_access import normalize
    parents = normalize(store.get_setting(section_key(actor), []))
    values = store.get_setting(key(actor), [])
    return [name for name in for_sections(parents) if name in values]


def allowed(store, actor, feature):
    return feature in assigned(store, actor)


def require(store, actor, feature):
    if not allowed(store, actor, feature):
        raise PermissionError('Эта функция не подключена для вас. Обратитесь к администратору.')


def route_features(data):
    parts = data.split(':')
    if parts[0] in {'lf', 'rc'}:
        if parts[0] == 'rc' and len(parts) > 1 and parts[1] in {'enrich', 'info', 'useinfo', 'useyear', 'cellar', 'pick'}:
            return ['life_finance', 'wine_cellar']
        if parts[0] == 'rc' and len(parts) > 3 and parts[1] == 'edit' and parts[3] in {'winery', 'vintage', 'volume_ml'}:
            return ['life_finance', 'wine_cellar']
        return ['life_finance']
    if parts[0] == 'wc':
        return ['wine_cellar']
    if parts[0] == 'usd':
        return ['dollar']
    if parts[0] == 'cb':
        return ['cbr']
    if parts[0] == 'ch':
        return ['exports'] if 'export' in parts else ['charts']
    if parts[0] == 'cs' or data == 'menu:graphs':
        return ['charts', 'schedules']
    if parts[0] == 'sub' or data == 'menu:subscriptions':
        return ['reports', 'schedules']
    if parts[0] == 'request' or data == 'menu:report':
        return ['reports']
    if parts[0] == 'va':
        return ['vivino_plan']
    if parts[0] == 'vi' and data not in {'vi:menu', 'vi:help'} or data == 'pj:vivino:data':
        return ['vivino_map']
    if parts[0] == 'ma' or data in {'mk:open:alerts', 'mk:input:alerts'}:
        return ['alerts']
    if parts[0] == 'mk':
        if 'correlations' in parts:
            return ['correlations']
        if 'indicators' in parts:
            return ['cbr']
        if 'forecasts' in parts or 'backtest' in parts:
            return ['backtests']
        if 'export' in parts:
            return ['exports']
        if len(parts) > 1 and parts[1] in {'menu', 'profile', 'report'}:
            return ['reports']
    if parts[0] == 'ui' and len(parts) > 1:
        action = parts[1]
        if action in {'export', 'exportinput'}:
            return ['exports']
        if action in {'backtest', 'forecasts'}:
            return ['backtests']
        if action.startswith(('rule', 'digest')):
            return ['alerts']
        if action.startswith('pair'):
            return ['correlations']
        if action.startswith('forecast'):
            return ['backtests']
        if action.startswith('subscription'):
            return ['reports', 'schedules']
    if data in {'menu:forecasts', 'menu:ruonia_forecasts', 'menu:backtest_help'}:
        return ['backtests']
    if data == 'menu:export_help':
        return ['exports']
    return []


def route_allowed(store, actor, data):
    if data.startswith('mon:'):
        return store.authorize(actor, admin=True)
    if data in {'vi:menu', 'vi:help'}:
        return any(allowed(store, actor, feature) for feature in ('vivino_map', 'vivino_plan'))
    return all(allowed(store, actor, name) for name in route_features(data))


def job_feature(kind, payload):
    if kind.startswith('life_'):
        return 'life_finance'
    if kind.startswith('wine_'):
        return 'wine_cellar'
    if kind.startswith('dollar_'):
        return 'dollar'
    return {'telegram_chart': 'charts', 'telegram_export': 'exports', 'market_export': 'exports',
            'market_export_sources': 'exports',
            'telegram_backtest': 'backtests', 'market_backtest': 'backtests',
            'vivino_map': 'vivino_map', 'report': 'reports', 'market_report': 'reports'}.get(kind)


def report_feature(report):
    quality = json.loads(report['quality'])
    if quality.get('feature') in CATALOG:
        return quality['feature']
    if quality.get('tool') == 'dollar':
        return 'dollar'
    if 'signal_ids' in quality or 'alert_id' in quality:
        return 'alerts'
    if report['section'] == 'vivino':
        return 'vivino_map'
    if 'chart_request' in quality or 'options' in quality:
        return 'charts'
    return job_feature(quality.get('job_kind', ''), {}) or 'reports'


def report_allowed(store, actor, report):
    quality = json.loads(report['quality'])
    dependencies = quality.get('required_features', [])
    return (isinstance(dependencies, list) and all(name in CATALOG and allowed(store, actor, name) for name in dependencies)
            and allowed(store, actor, report_feature(report)))


def _cancel(conn, target, feature):
    """Keep preferences/data, stop work that can no longer be delivered."""
    conn.execute('UPDATE bot_sessions SET pending=NULL WHERE user_id=?', (target,))
    if feature in {'schedules', 'reports'}:
        conn.execute('UPDATE subscriptions SET enabled=0,last_scheduled_at=? WHERE user_id=?', (utcnow(), target))
    if feature in {'schedules', 'charts'}:
        for row in conn.execute("SELECT key,value FROM settings WHERE key LIKE ? OR key LIKE 'chart_batch:%'", (f'chart_schedule:{target}:%',)).fetchall():
            value = json.loads(row['value'])
            if value.get('actor_id') == target or row['key'].startswith(f'chart_schedule:{target}:'):
                from uuid import uuid4
                value.update(enabled=False, generation=uuid4().hex, last_scheduled_at=utcnow())
                conn.execute('UPDATE settings SET value=? WHERE key=?', (dumps(value), row['key']))
    if feature == 'alerts':
        conn.execute("UPDATE market_rules SET enabled=0,mode='off' WHERE user_id=?", (target,))
    for row in conn.execute("SELECT id,kind,payload FROM jobs WHERE state='pending'").fetchall():
        value = json.loads(row['payload'])
        owns = value.get('actor_id') == target or str(value.get('recipient')) == str(target)
        scheduled = feature == 'schedules' and ('subscription_id' in value or 'chart_schedule' in value)
        if owns and (job_feature(row['kind'], value) == feature or feature in value.get('required_features', []) or scheduled):
            conn.execute("UPDATE jobs SET state='cancelled',finished_at=?,error='feature_access_revoked' WHERE id=?", (utcnow(), row['id']))
    for row in conn.execute("SELECT d.id,r.section,r.quality,d.payload FROM deliveries d JOIN reports r ON r.id=d.report_id WHERE d.recipient=? AND d.state IN ('pending','retry')", (str(target),)).fetchall():
        value = json.loads(row['payload'])
        scheduled = feature == 'schedules' and ('subscription_id' in value or 'chart_schedule' in value)
        if report_feature(row) == feature or feature in json.loads(row['quality']).get('required_features', []) or scheduled:
            conn.execute("UPDATE deliveries SET state='cancelled',updated_at=?,error='feature_access_revoked' WHERE id=?", (utcnow(), row['id']))


def change(store, actor, target, feature, enabled):
    if feature not in CATALOG or type(target) is not int or target <= 0 or type(enabled) is not bool:
        raise ValueError('Неизвестная функция или пользователь')
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute("SELECT 1 FROM users WHERE user_id=? AND role='admin' AND enabled=1", (actor,)).fetchone():
            raise PermissionError('Действие доступно администратору')
        user = conn.execute('SELECT role FROM users WHERE user_id=?', (target,)).fetchone()
        if not user:
            raise ValueError('Пользователь не найден')
        if user['role'] == 'admin':
            raise ValueError('Администратору доступны все функции. Сначала измените его роль.')
        from .user_management import access_state, queue_notice
        before = access_state(conn, target)
        row = conn.execute('SELECT value FROM settings WHERE key=?', (key(target),)).fetchone()
        values = json.loads(row['value']) if row else []
        from .subscription_access import key as section_key
        record = conn.execute('SELECT value FROM settings WHERE key=?', (section_key(target),)).fetchone()
        parents = json.loads(record['value']) if record else []
        if parents == 'all':
            from .subscription_access import CATALOG as SECTIONS
            parents = list(SECTIONS)
        parent_changed = enabled and not set(parents) & set(CATALOG[feature][1])
        if parent_changed:
            parents.append(CATALOG[feature][1][0])
            conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', (section_key(target), dumps(parents)))
        if (feature in values) == enabled and not parent_changed:
            return False
        values = list(dict.fromkeys([*values, feature])) if enabled else [f for f in values if f != feature]
        conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key(target), dumps(values)))
        if not enabled:
            _cancel(conn, target, feature)
        conn.execute('INSERT INTO access_audit(actor_id,target_id,action,created_at) VALUES(?,?,?,?)',
                     (actor, target, ('feature_grant:' if enabled else 'feature_revoke:')+feature, utcnow()))
        queue_notice(conn, target, 'grant' if enabled else 'revoke', before=before)
    return True
