"""User identities, administrator roles and durable access-change notifications."""
import json
import logging
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .store import dumps, utcnow

logger = logging.getLogger(__name__)
VERSION = 2
NOTICE_DELAY_SECONDS = 15
SCHEMA = """
CREATE TABLE user_profiles(user_id INTEGER PRIMARY KEY,first_name TEXT NOT NULL DEFAULT '',
 last_name TEXT NOT NULL DEFAULT '',username TEXT,updated_at TEXT NOT NULL,last_seen TEXT);
CREATE TABLE access_notices(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(user_id),
 reason TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending',attempts INTEGER NOT NULL DEFAULT 0,
 message_id INTEGER,error TEXT,retry_at TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
 before_state TEXT,after_state TEXT,ready_at TEXT);
CREATE INDEX access_notice_pending ON access_notices(state,user_id,id);
"""


def migrate(store):
    with store.connection() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key='user_management_version'").fetchone()
        if row and int(row['value']) >= VERSION:
            return
        backup = store.path.with_name(store.path.name+f'.users-backup-{uuid4().hex}.sqlite3')
        with closing(sqlite3.connect(backup)) as target:
            conn.backup(target)
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute("SELECT value FROM settings WHERE key='user_management_version'").fetchone()
        version = int(row['value']) if row else 0
        if version >= VERSION:
            return
        if version == 0:
            for statement in SCHEMA.split(';'):
                if statement.strip():
                    conn.execute(statement)
            from .feature_access import CATALOG, key
            for user in conn.execute('SELECT user_id FROM users').fetchall():
                conn.execute('INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)', (key(user['user_id']), dumps(list(CATALOG))))
            # Older reports retain their function boundary when reopened.
            for job in conn.execute("SELECT kind,payload FROM jobs WHERE kind IN ('market_export','market_export_sources','market_backtest')").fetchall():
                identity = json.loads(job['payload']).get('report_id')
                report = conn.execute('SELECT quality FROM reports WHERE id=?', (identity,)).fetchone() if isinstance(identity, str) else None
                if report:
                    quality = json.loads(report['quality'])
                    quality.setdefault('job_kind', job['kind'])
                    conn.execute('UPDATE reports SET quality=? WHERE id=?', (dumps(quality), identity))
        else:
            for column in ('before_state', 'after_state', 'ready_at'):
                conn.execute(f'ALTER TABLE access_notices ADD COLUMN {column} TEXT')
        conn.execute("INSERT INTO settings(key,value) VALUES('user_management_version',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(VERSION),))


def remember(store, user, *, interacted=True):
    identity = getattr(user, 'id', None)
    if type(identity) is not int or identity <= 0:
        return
    if not any(hasattr(user, attr) for attr in ('first_name', 'last_name', 'username')):
        existing = store.one('SELECT 1 FROM user_profiles WHERE user_id=?', (identity,))
        if existing:
            if interacted:
                store.execute('UPDATE user_profiles SET last_seen=? WHERE user_id=?', (utcnow(), identity))
            return
    def clean(value):
        return ''.join(c for c in str(value or '') if c.isprintable() and c not in '\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')[:100]
    username = getattr(user, 'username', None)
    username = username if isinstance(username, str) and re.fullmatch(r'[A-Za-z0-9_]{1,32}', username) else None
    store.execute('''INSERT INTO user_profiles(user_id,first_name,last_name,username,updated_at,last_seen)
        VALUES(?,?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET first_name=excluded.first_name,
        last_name=excluded.last_name,username=excluded.username,updated_at=excluded.updated_at,
        last_seen=COALESCE(excluded.last_seen,user_profiles.last_seen)''',
                  (identity, clean(getattr(user, 'first_name', '')), clean(getattr(user, 'last_name', '')),
                   username, utcnow(), utcnow() if interacted else None))


def resolve_profile(store, client, target):
    try:
        user = client.get_users(target)
        if getattr(user, 'id', None) == target:
            remember(store, user, interacted=False)
    except Exception as exc:  # noqa: BLE001 - Telegram may not know the user's peer yet.
        logger.info('Telegram identity lookup unavailable (%s)', type(exc).__name__)


def resolve_user_reference(store, client, value):
    """Resolve a current Telegram username; persist access by immutable numeric ID."""
    value = value.strip()
    if value.isascii() and value.isdigit() and int(value) > 0:
        return int(value)
    if not re.fullmatch(r'@[A-Za-z0-9_]{1,32}', value):
        raise ValueError('Введите @username, например @ivan_test, или положительный числовой Telegram ID.')
    from pyrogram.errors import FloodWait, UsernameInvalid, UsernameNotOccupied
    try:
        user = client.get_users(value)
    except (UsernameInvalid, UsernameNotOccupied):
        raise ValueError('Такой @username не найден в Telegram. Проверьте написание или введите числовой ID.') from None
    except FloodWait as exc:
        raise ValueError(f'Telegram временно ограничил поиск. Повторите через {max(1, int(exc.value))} с или введите числовой ID.') from None
    except Exception as exc:  # noqa: BLE001 - Do not grant access using a stale cached username.
        logger.info('Telegram username lookup unavailable (%s)', type(exc).__name__)
        raise ValueError('Не удалось найти пользователя по @username. Попросите его открыть этот бот и нажать /start, '
                         'затем повторите поиск или введите Telegram ID из ответа бота.') from None
    target = getattr(user, 'id', None)
    username = getattr(user, 'username', None)
    if (type(target) is not int or target <= 0 or not isinstance(username, str)
            or username.casefold() != value[1:].casefold()):
        raise ValueError('Telegram не подтвердил пользователя с этим @username. Проверьте имя или введите числовой ID.')
    if getattr(user, 'is_bot', False) is True or getattr(user, 'is_deleted', False) is True:
        raise ValueError('Выберите личный аккаунт пользователя. Боты и удалённые аккаунты не подходят.')
    remember(store, user, interacted=False)
    return target


def label(store, target):
    profile = store.one('SELECT first_name,last_name,username FROM user_profiles WHERE user_id=?', (target,))
    name = ' '.join(p for p in (profile['first_name'], profile['last_name']) if p) if profile else ''
    username = '@'+profile['username'] if profile and profile['username'] else ''
    return ' · '.join(p for p in (name, username, f'ID {target}') if p)


def access_state(conn, target):
    """Snapshot effective rights in the same transaction that changes them."""
    from .feature_access import CATALOG, for_sections
    from .subscription_access import CATALOG as SECTIONS
    from .subscription_access import normalize
    user = conn.execute('SELECT role,enabled FROM users WHERE user_id=?', (target,)).fetchone()
    state = {'role': user['role'] if user else None, 'enabled': bool(user and user['enabled']), 'sections': [], 'features': []}
    if not state['enabled']:
        return state
    if user['role'] == 'admin':
        return {**state, 'sections': list(SECTIONS), 'features': list(CATALOG)}
    row = conn.execute('SELECT value FROM settings WHERE key=?', (f'user_subscriptions:{target}',)).fetchone()
    parents = normalize(json.loads(row['value']) if row else [])
    row = conn.execute('SELECT value FROM settings WHERE key=?', (f'user_features:{target}',)).fetchone()
    functions = json.loads(row['value']) if row else []
    return {**state, 'sections': [name for name in SECTIONS if name in parents],
            'features': [name for name in for_sections(parents) if name in functions]}


def queue_notice(conn, target, reason, *, before=None):
    after = access_state(conn, target)
    queued = conn.execute("SELECT before_state,retry_at,attempts FROM access_notices WHERE user_id=? AND state IN ('pending','failed') ORDER BY id", (target,)).fetchall()
    pending = queued[0] if queued else None
    if pending and pending['before_state']:
        before = json.loads(pending['before_state'])
    conn.execute("UPDATE access_notices SET state='superseded',updated_at=? WHERE user_id=? AND state IN ('pending','failed')", (utcnow(), target))
    if before == after and reason != 'manual':
        return False
    delay = 0 if reason in {'deny', 'manual'} else NOTICE_DELAY_SECONDS
    ready = (datetime.now(timezone.utc)+timedelta(seconds=delay)).isoformat()
    retry_at = max((row['retry_at'] for row in queued if row['retry_at']), default=None)
    attempts = max((row['attempts'] for row in queued), default=0)
    conn.execute('INSERT INTO access_notices(user_id,reason,before_state,after_state,ready_at,retry_at,attempts,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
                 (target, reason, dumps(before) if before is not None else None, dumps(after), ready, retry_at, attempts, utcnow(), utcnow()))
    return True


def change_role(store, actor, target, role):
    if type(target) is not int or target <= 0 or role not in {'admin', 'subscriber'}:
        raise ValueError('Неверная роль или пользователь')
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute("SELECT 1 FROM users WHERE user_id=? AND role='admin' AND enabled=1", (actor,)).fetchone():
            raise PermissionError('Действие доступно администратору')
        user = conn.execute('SELECT role,enabled FROM users WHERE user_id=?', (target,)).fetchone()
        if not user:
            raise ValueError('Пользователь не найден')
        if user['role'] == role:
            return False
        pins = conn.execute("SELECT value FROM settings WHERE key='protected_admin_ids'").fetchone()
        if role != 'admin':
            if target == actor:
                raise ValueError('Нельзя понизить собственную роль')
            if pins and target in json.loads(pins['value']):
                raise ValueError('Администратор задан в ADMIN_TELEGRAM_IDS; его роль защищена настройками сервера')
            if conn.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND enabled=1 AND user_id<>?", (target,)).fetchone()[0] == 0:
                raise ValueError('Нельзя удалить последнего администратора')
        elif not user['enabled']:
            raise ValueError('Сначала восстановите доступ пользователя')
        before = access_state(conn, target)
        conn.execute('UPDATE users SET role=? WHERE user_id=?', (role, target))
        conn.execute('UPDATE bot_sessions SET pending=NULL WHERE user_id=?', (target,))
        conn.execute('INSERT INTO access_audit(actor_id,target_id,action,created_at) VALUES(?,?,?,?)', (actor, target, 'role:'+role, utcnow()))
        queue_notice(conn, target, 'role', before=before)
    return True


def notice_text(store, target, *, before=None, after=None):
    from .feature_access import CATALOG, for_sections
    from .subscription_access import CATALOG as SECTIONS
    if after is None:
        with store.connection() as conn:
            after = access_state(conn, target)
    lines = ['Здравствуйте! Ваш доступ к боту обновлён.']
    roles = {'admin': 'администратор', 'subscriber': 'пользователь', None: 'не был подключён'}
    if before is not None:
        for heading, left, right in (('Отключено:', before, after), ('Добавлено:', after, before)):
            values = ['• Раздел: '+SECTIONS[name] for name in left['sections'] if name not in right['sections']]
            values += ['• Функция: '+CATALOG[name][0] for name in left['features'] if name not in right['features']]
            if values:
                lines.append(heading+'\n'+'\n'.join(values))
        if before['role'] != after['role']:
            lines.append(f'Роль: {roles[before["role"]]} → {roles[after["role"]]}')
    if not after['enabled']:
        lines += ['Администратор отключил ваш доступ к боту.', 'Осталось доступно: нет подключённых функций.']
    else:
        lines.append('Роль: '+roles[after['role']]+('. Доступны управление пользователями, источниками и всеми функциями.' if after['role'] == 'admin' else '. Личные настройки и выданные функции.'))
        parents = [name for name in after['sections'] if set(for_sections([name])) & set(after['features'])]
        lines.append('Осталось доступно · разделы:\n'+('\n'.join('• '+SECTIONS[name] for name in parents) or 'Нет подключённых разделов.'))
        lines.append('Осталось доступно · функции:\n'+('\n'.join('• '+CATALOG[name][0] for name in after['features']) or 'Пока ни одна функция не подключена.'))
        lines.append('Откройте /start или /projects, чтобы увидеть доступные инструменты.')
    lines.append('Личные данные и настройки сохранены. Выдача прав не включает рассылки. Расписания, отключённые при отзыве прав, нужно включить заново после восстановления доступа.')
    return '\n\n'.join(lines)


def status(store, target):
    row = store.one('SELECT state,error FROM access_notices WHERE user_id=? ORDER BY id DESC LIMIT 1', (target,))
    if not row:
        return 'Уведомлений о правах пока не было.'
    labels = {'sent': 'Уведомление о правах отправлено.', 'sending': 'Уведомление отправляется.',
              'superseded': 'Быстрые изменения вернули прежний набор прав; повторное уведомление не требуется.',
              'uncertain': 'Отправка неоднозначна; автоматического повтора нет.', 'failed': 'Уведомление не доставлено.',
              'pending': 'Быстрые изменения объединяются в одно уведомление: отправка через 15 секунд после последнего изменения. Если пользователь ещё не открывал бот, оно придёт при /start.'}
    return labels.get(row['state'], 'Состояние уведомления: '+row['state'])


def recover_notices(store):
    store.execute("UPDATE access_notices SET state='uncertain',error='interrupted_send',updated_at=? WHERE state='sending'", (utcnow(),))


def release_notices(store, target):
    # A new private interaction resolves blocked/unknown peers. Ambiguous sends
    # still require an explicit administrator retry, never a blind replay.
    store.execute("UPDATE access_notices SET state='pending',error=NULL,retry_at=NULL,updated_at=? WHERE id=(SELECT MAX(id) FROM access_notices WHERE user_id=?) AND state='failed'", (utcnow(), target))


def send_notices(store, client, target=None, *, force=False):
    from pyrogram.errors import FloodWait, RPCError

    from .delivery import _send_part
    notices = store.query("SELECT n.id,n.user_id FROM access_notices n LEFT JOIN user_profiles p ON p.user_id=n.user_id "
                          "LEFT JOIN bot_sessions s ON s.user_id=n.user_id WHERE n.state='pending' "
                          "AND (p.last_seen IS NOT NULL OR s.started_at IS NOT NULL) AND (n.retry_at IS NULL OR n.retry_at<=?) "
                          + ('' if force else "AND (n.ready_at IS NULL OR n.ready_at<=?) ")
                          + ('AND n.user_id=? ' if target is not None else '')+'ORDER BY n.id LIMIT 20',
                          (utcnow(),)+(() if force else (utcnow(),))+(() if target is None else (target,)))
    sent = 0
    for notice in notices:
        with store.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            now = utcnow()
            claimed = conn.execute("UPDATE access_notices SET state='sending',attempts=attempts+1,updated_at=? WHERE id=? AND state='pending' AND (retry_at IS NULL OR retry_at<=?)"+
                                   ('' if force else ' AND (ready_at IS NULL OR ready_at<=?)'),
                                   (now, notice['id'], now)+(() if force else (now,))).rowcount
            snapshot = conn.execute('SELECT before_state,after_state FROM access_notices WHERE id=?', (notice['id'],)).fetchone() if claimed else None
        if not claimed:
            continue
        try:
            before = json.loads(snapshot['before_state']) if snapshot['before_state'] else None
            after = json.loads(snapshot['after_state']) if snapshot['after_state'] else None
            message_id = _send_part(client, str(notice['user_id']), {'kind': 'text', 'text': notice_text(store, notice['user_id'], before=before, after=after)})
        except Exception as exc:  # noqa: BLE001 - Message failures must not roll back successfully changed rights.
            retry_at = None
            state = 'uncertain'
            if isinstance(exc, FloodWait):
                attempts = store.one('SELECT attempts FROM access_notices WHERE id=?', (notice['id'],))['attempts']
                state = 'pending' if attempts < 5 else 'failed'
                retry_at = (datetime.now(timezone.utc)+timedelta(seconds=max(1, int(exc.value or 1)))).isoformat()
            elif isinstance(exc, RPCError) and exc.CODE in {400, 401, 403, 406}:
                state = 'failed'
            with store.connection() as conn:
                conn.execute('BEGIN IMMEDIATE')
                conn.execute('UPDATE access_notices SET state=?,error=?,retry_at=?,updated_at=? WHERE id=?', (state, type(exc).__name__, retry_at, utcnow(), notice['id']))
                newer = conn.execute("SELECT 1 FROM access_notices WHERE user_id=? AND id>? AND state='pending'", (notice['user_id'], notice['id'])).fetchone()
                if newer and state in {'pending', 'failed'}:
                    # Definite failures can be folded into changes made during
                    # I/O. Ambiguous sends must never be blindly replayed.
                    queue_notice(conn, notice['user_id'], 'retry', before=before)
            continue
        store.execute("UPDATE access_notices SET state='sent',message_id=?,error=NULL,retry_at=NULL,updated_at=? WHERE id=?", (message_id, utcnow(), notice['id']))
        sent += 1
    return sent
