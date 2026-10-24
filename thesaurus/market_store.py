"""Versioned market data and access audit; no provider I/O inside transactions."""
import json
import math
import sqlite3
from contextlib import closing
from uuid import uuid4

from .store import dumps, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_series(id TEXT PRIMARY KEY, source TEXT NOT NULL,
 dataset TEXT NOT NULL, title TEXT NOT NULL, metadata TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 0,
 status TEXT NOT NULL DEFAULT 'not_loaded', error TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS market_observations(id INTEGER PRIMARY KEY, series_id TEXT NOT NULL REFERENCES market_series(id),
 date TEXT NOT NULL, value REAL, period_end TEXT, published_at TEXT, retrieved_at TEXT NOT NULL,
 definition TEXT NOT NULL, quality TEXT NOT NULL, UNIQUE(series_id,date));
CREATE TABLE IF NOT EXISTS market_revisions(id INTEGER PRIMARY KEY, series_id TEXT NOT NULL REFERENCES market_series(id),
 date TEXT NOT NULL, value REAL, period_end TEXT, published_at TEXT, retrieved_at TEXT NOT NULL,
 definition TEXT NOT NULL, quality TEXT NOT NULL, UNIQUE(series_id,date,retrieved_at));
CREATE INDEX IF NOT EXISTS market_revision_asof ON market_revisions(series_id,retrieved_at,date);
CREATE TABLE IF NOT EXISTS market_forecasts(id INTEGER PRIMARY KEY, series_id TEXT NOT NULL REFERENCES market_series(id),
 origin TEXT NOT NULL, target TEXT NOT NULL, horizon INTEGER NOT NULL, prediction REAL NOT NULL,
 lower REAL, upper REAL, baseline REAL NOT NULL, model TEXT NOT NULL, created_at TEXT NOT NULL,
 inputs TEXT NOT NULL, actual REAL, UNIQUE(series_id,origin,horizon));
CREATE TABLE IF NOT EXISTS market_rules(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(user_id),
 series_id TEXT NOT NULL, kind TEXT NOT NULL, config TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 0,
 mode TEXT NOT NULL DEFAULT 'off', cooldown REAL NOT NULL DEFAULT 24);
CREATE TABLE IF NOT EXISTS market_signals(id INTEGER PRIMARY KEY, rule_id INTEGER REFERENCES market_rules(id),
 observed_date TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL, report_id TEXT,
 UNIQUE(rule_id,observed_date));
CREATE TABLE IF NOT EXISTS access_audit(id INTEGER PRIMARY KEY,actor_id INTEGER NOT NULL,target_id INTEGER NOT NULL,
 action TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS bot_sessions(user_id INTEGER PRIMARY KEY REFERENCES users(user_id),started_at TEXT,
 last_seen TEXT NOT NULL,pending TEXT);
CREATE TABLE IF NOT EXISTS forecast_runs(id TEXT PRIMARY KEY,series_id TEXT NOT NULL REFERENCES market_series(id),
 fingerprint TEXT NOT NULL UNIQUE,created_at TEXT NOT NULL,duration TEXT NOT NULL,inputs TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS forecast_points(id INTEGER PRIMARY KEY,run_id TEXT NOT NULL REFERENCES forecast_runs(id),
 target TEXT NOT NULL,payload TEXT NOT NULL,UNIQUE(run_id,target));
CREATE TABLE IF NOT EXISTS integrity_checks(id INTEGER PRIMARY KEY,actor_id INTEGER NOT NULL,created_at TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'pending',selection TEXT NOT NULL,date_from TEXT,date_to TEXT,checkpoint TEXT NOT NULL DEFAULT '{}',error TEXT);
CREATE TABLE IF NOT EXISTS integrity_incidents(id INTEGER PRIMARY KEY,fingerprint TEXT NOT NULL UNIQUE,
 check_id INTEGER REFERENCES integrity_checks(id),series_id TEXT,date TEXT,kind TEXT NOT NULL,severity TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'new',old_payload TEXT,new_payload TEXT,created_at TEXT NOT NULL,
 resolved_at TEXT,resolved_by INTEGER,notified_at TEXT);
CREATE INDEX IF NOT EXISTS integrity_open ON integrity_incidents(state,severity,id);
CREATE TABLE IF NOT EXISTS source_snapshots(dataset TEXT NOT NULL,version TEXT NOT NULL,received_at TEXT NOT NULL,
 payload TEXT NOT NULL,PRIMARY KEY(dataset,version));
"""


def migrate(store):
    with store.connection() as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= 3:
            return
        backup = store.path.with_name(store.path.name + f".market-backup-{uuid4().hex}.sqlite3")
        with closing(sqlite3.connect(backup)) as target:
            conn.backup(target)
        conn.execute("BEGIN IMMEDIATE")
        for statement in SCHEMA.split(";"):
            if statement.strip():
                conn.execute(statement)
        conn.execute("PRAGMA user_version=3")


def register(store, series):
    with store.connection() as conn:
        for item in series:
            override = conn.execute('SELECT value FROM settings WHERE key=?', ('market_unit:' + item['id'],)).fetchone()
            if override:
                item = {**item, 'unit': json.loads(override[0])}
            conn.execute("""INSERT INTO market_series(id,source,dataset,title,metadata) VALUES(?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title,metadata=excluded.metadata""",
                         (item['id'], item['source'], item['dataset'], item['title'], dumps(item)))


def partition_rows(rows):
    """Quarantine contradictory dates, never choose a source row by its order."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['date'], []).append(row)
    clean, conflicts = [], {}
    fields = ('value', 'period_end', 'published_at', 'definition')
    for day, candidates in grouped.items():
        versions = {tuple(row.get(key, day if key == 'period_end' else '1' if key == 'definition' else None)
                          for key in fields) for row in candidates}
        if len(versions) > 1:
            conflicts[day] = [{key: row.get(key) for key in ('date', *fields)} for row in candidates]
            clean.append({'date': day, 'value': None, 'quality': 'source_conflicting_values'})
        else:
            clean.append(candidates[0])
    return clean, conflicts


def save_rows(store, series_id, rows):
    seen = {}
    for row in rows:
        if row['date'] in seen and seen[row['date']] != row.get('value'):
            raise ValueError('Ambiguous duplicate market observations')
        seen[row['date']] = row.get('value')
    with store.connection() as conn:
        for row in rows:
            value = row.get('value')
            if value is not None and not math.isfinite(float(value)):
                raise ValueError('Nonfinite market observation')
            values = (series_id, row['date'], value, row.get('period_end', row['date']),
                      row.get('published_at'), row.get('retrieved_at', utcnow()),
                      row.get('definition', '1'), row.get('quality', 'publication_unknown'))
            old = conn.execute("SELECT * FROM market_observations WHERE series_id=? AND date=?", values[:2]).fetchone()
            if old and all(old[k] == values[i] for i, k in ((2, 'value'), (3, 'period_end'), (4, 'published_at'), (6, 'definition'), (7, 'quality'))):
                continue
            conn.execute("INSERT OR IGNORE INTO market_revisions(series_id,date,value,period_end,published_at,retrieved_at,definition,quality) VALUES(?,?,?,?,?,?,?,?)", values)
            conn.execute("""INSERT INTO market_observations(series_id,date,value,period_end,published_at,retrieved_at,definition,quality)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(series_id,date) DO UPDATE SET
                value=excluded.value,period_end=excluded.period_end,published_at=excluded.published_at,
                retrieved_at=excluded.retrieved_at,definition=excluded.definition,quality=excluded.quality""", values)


def history(store, series_id, *, as_of=None, effective_through=None):
    if as_of is None:
        return store.query("SELECT * FROM market_observations WHERE series_id=? ORDER BY date", (series_id,))
    return store.query("""SELECT * FROM (SELECT *,ROW_NUMBER() OVER(PARTITION BY date ORDER BY retrieved_at DESC,id DESC) n
        FROM market_revisions WHERE series_id=? AND retrieved_at<=? AND (published_at IS NULL OR published_at<=?))
        WHERE n=1 AND date<=? ORDER BY date""", (series_id, as_of, as_of, effective_through or as_of[:10]))


def forecast_history(store, series_id, *, latest=False):
    runs = store.query('SELECT * FROM forecast_runs WHERE series_id=? ORDER BY created_at DESC,rowid DESC', (series_id,))
    if latest:
        runs = runs[:1]
    result = []
    from .market_analysis import values
    series = store.one('SELECT metadata FROM market_series WHERE id=?', (series_id,))
    now = utcnow()
    known = values(history(store, series_id, as_of=now), json.loads(series['metadata']), cutoff=now) if series else {}
    actuals = {stamp.date().isoformat(): value for stamp, value in known.items()}
    monthly_actuals = None
    for run in runs:
        for point in store.query('SELECT payload FROM forecast_points WHERE run_id=? ORDER BY target', (run['id'],)):
            row = json.loads(point['payload'])
            # Series values use native period starts, matching forecast target dates.
            if row.get('model_family') == 'monthly_usd' and series:
                if monthly_actuals is None:
                    from .dollar import today_moscow
                    from .dollar_long import monthly_closes
                    closes = monthly_closes(history(store, series_id, as_of=now), json.loads(series['metadata']),
                                             now=now, today=today_moscow(now))
                    monthly_actuals = {stamp.date().isoformat(): value for stamp, value in closes.items()}
                actual = monthly_actuals.get(row['target'])
            else:
                actual = actuals.get(row['target'])
            row.update(run_id=run['id'], series_id=series_id, actual=None)
            if actual is not None and row['created_at'][:10] < row['target'] <= now[:10]:
                row['actual'] = float(actual) if math.isfinite(actual) else None
            if row.get('distribution') and row['actual'] is not None:
                from .dollar_distribution import score_actual
                row['distribution_evaluation'] = score_actual(row['distribution'], row['actual'])
            result.append(row)
    return result


def change_access(store, actor_id, target_id, enabled, *, authenticated_dashboard=False):
    if type(target_id) is not int or target_id <= 0:
        raise ValueError('Нужен положительный числовой Telegram ID')
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        actor = conn.execute("SELECT role,enabled FROM users WHERE user_id=?", (actor_id,)).fetchone()
        if not authenticated_dashboard and (not actor or not actor['enabled'] or actor['role'] != 'admin'):
            raise PermissionError('Действие доступно администратору')
        target = conn.execute("SELECT role,enabled FROM users WHERE user_id=?", (target_id,)).fetchone()
        if target and target['role'] == 'admin' and not enabled:
            raise ValueError('Сначала снимите роль администратора в карточке пользователя. Администраторы из ADMIN_TELEGRAM_IDS защищены.')
        from .user_management import access_state, queue_notice
        before = access_state(conn, target_id)
        conn.execute("""INSERT INTO users(user_id,role,enabled,chat_id,created_at) VALUES(?,'subscriber',?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET enabled=excluded.enabled""", (target_id, int(enabled), str(target_id), utcnow()))
        from .subscription_access import key
        conn.execute('INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)', (key(target_id), '[]'))
        from .feature_access import key as feature_key
        conn.execute('INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)', (feature_key(target_id), '[]'))
        if not enabled or (target and not target['enabled']):
            conn.execute("UPDATE subscriptions SET enabled=0 WHERE user_id=?", (target_id,))
            conn.execute("UPDATE market_rules SET enabled=0,mode='off' WHERE user_id=?", (target_id,))
            for setting in conn.execute("SELECT key,value FROM settings WHERE key LIKE ? OR key LIKE 'chart_batch:%'", (f'chart_schedule:{target_id}:%',)).fetchall():
                config = json.loads(setting['value'])
                if config.get('actor_id') == target_id or setting['key'].startswith(f'chart_schedule:{target_id}:'):
                    config.update(enabled=False, generation=uuid4().hex, last_scheduled_at=utcnow())
                    conn.execute('UPDATE settings SET value=? WHERE key=?', (dumps(config), setting['key']))
            conn.execute("UPDATE deliveries SET state='cancelled',updated_at=? WHERE recipient=? AND state IN ('pending','retry')", (utcnow(), str(target_id)))
            for job in conn.execute("SELECT id,payload FROM jobs WHERE state='pending'").fetchall():
                payload = json.loads(job['payload'])
                if payload.get('actor_id') == target_id or payload.get('recipient') == str(target_id):
                    conn.execute("UPDATE jobs SET state='cancelled',finished_at=? WHERE id=?", (utcnow(), job['id']))
        conn.execute("INSERT INTO access_audit(actor_id,target_id,action,created_at) VALUES(?,?,?,?)", (actor_id, target_id, 'allow' if enabled else 'deny', utcnow()))
        if target is None or bool(target['enabled']) != enabled:
            queue_notice(conn, target_id, 'allow' if enabled else 'deny', before=before)
    from .scheduling import seed_subscriptions
    if enabled:
        seed_subscriptions(store, str(target_id), target_id)
