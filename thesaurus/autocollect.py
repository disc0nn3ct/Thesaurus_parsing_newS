"""Shared administrative controls for global collection, separate from delivery."""
import json

from .store import dumps, utcnow

SOURCES = {
    'trends': ('trends_schedule', 'trends_collect', {'times': ['08:30'], 'geo': 'RU', 'window': 'today 5-y'}),
    'market': ('market_schedule', 'market_collect', {'times': ['18:30']}),
}


def update_schedule(store, source, actor, *, enabled, changes=None):
    if source not in SOURCES or type(enabled) is not bool:
        raise ValueError('Неизвестная настройка автосбора')
    key, kind, defaults = SOURCES[source]
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if type(actor) is not int:
            raise PermissionError('Автосбором управляет администратор')
        user = conn.execute('SELECT enabled,role FROM users WHERE user_id=?', (actor,)).fetchone()
        if not user or not user['enabled'] or user['role'] != 'admin':
            raise PermissionError('Автосбором управляет администратор')
        if source == 'trends' and enabled:
            from .search_provider import require_section
            require_section(store, 'trends')
        row = conn.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
        config = {'enabled': False, **defaults, **(json.loads(row[0]) if row else {})}
        if enabled and (not config['enabled'] or changes and changes.get('times', config['times']) != config['times']):
            # Enabling starts a fresh schedule; do not replay time spent disabled.
            config.update(created_at=utcnow(), last_scheduled_at=None)
            if source == 'trends':
                pause = conn.execute("SELECT value FROM settings WHERE key='trends_pause_until'").fetchone()
                if pause and json.loads(pause[0]) and json.loads(pause[0]) <= config['created_at']:
                    conn.execute("UPDATE settings SET value=? WHERE key='trends_pause_until'", (dumps(''),))
        config.update(changes or {})
        config.update(enabled=enabled, actor_id=actor)
        conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (key, dumps(config)))
        if not enabled:
            for job in conn.execute("SELECT id,payload FROM jobs WHERE kind=? AND state='pending'", (kind,)).fetchall():
                payload = json.loads(job['payload'])
                if payload.get('scheduled') or source == 'market' and payload.get('scheduled_for'):
                    conn.execute("UPDATE jobs SET state='cancelled',finished_at=? WHERE id=? AND state='pending'",
                                 (utcnow(), job['id']))
    return config


def menu(store, reply, client, actor):
    from .google_trends import active
    from .telegram_charts import keyboard
    from .telegram_data_jobs import require_actor
    from .telegram_navigation import screen
    require_actor(store, actor, admin=True)
    screen('menu:autocollect', 'pj:market:manage')
    text = ('Автосбор данных\nОбновляет общую базу по расписанию. '
            'Личная доставка отчётов настраивается отдельно в «Мои подписки».\n')
    buttons = []
    for source, label, route in (('trends', 'Google Trends', 'gt:auto'), ('market', 'ЦБ и Мосбиржа', 'ui:schedule:overview')):
        from .search_provider import disabled
        if source == 'trends' and disabled(store):
            continue
        key, _, defaults = SOURCES[source]
        config = {**defaults, **store.get_setting(key, {})}
        enabled = bool(config.get('enabled'))
        available = source != 'trends' or active(store)
        text += '\n' + label + ': ' + ('включён' if enabled and available else 'выключен')
        text += '\nВремя: ' + ', '.join(config['times']) + ' · Москва\n'
        if not available:
            text += 'Google Trends не выбран как поисковый источник.\n'
            continue
        buttons.append([(('Выключить ' if enabled else 'Включить ') + label,
                         'ui:autocollect:' + source + ':' + ('off' if enabled else 'on'))])
        buttons.append([('Настройки · ' + label, route)])
        if source == 'market':
            buttons.append([('Выбрать источники ЦБ и Мосбиржи', 'menu:sources')])
            text += 'Собираются источники, выбранные администратором.\n'
    text += '\nВыключение отменяет ожидающие автоматические задания. Начатый сбор завершится; ручное обновление остаётся доступным.'
    reply(client, actor, text, keyboard(buttons))
