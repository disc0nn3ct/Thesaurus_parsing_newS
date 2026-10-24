"""SQLite state. Connections are short lived and transactions contain no I/O."""
import json
import math
import sqlite3
import time
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEYWORDS = [
    {"phrase": "обвал", "theme": "экономика"},
    {"phrase": "рост цен", "theme": "цены"},
    {"phrase": "дефолт", "theme": "экономика"},
    {"phrase": "безработица", "theme": "работа"},
    {"phrase": "кредитные каникулы", "theme": "кредиты"},
    {"phrase": "закрытие банков", "theme": "банки"},
    {"phrase": "дефицит лекарств", "theme": "здоровье"},
    {"phrase": "отключение электричества", "theme": "услуги"},
]
INCIDENTS = ["пожар", "взрыв", "бпла"]
SECTIONS = ("ruonia", "ai", "wordstat", "regions", "market")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


SCHEMA = """
CREATE TABLE settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE users(user_id INTEGER PRIMARY KEY,role TEXT NOT NULL DEFAULT 'subscriber',
 enabled INTEGER NOT NULL DEFAULT 1,chat_id TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE subscriptions(id INTEGER PRIMARY KEY,user_id INTEGER REFERENCES users(user_id),
 recipient TEXT NOT NULL,section TEXT NOT NULL,phrases TEXT NOT NULL DEFAULT '[]',
 regions TEXT NOT NULL DEFAULT '[]',detail TEXT NOT NULL DEFAULT 'brief',
 timezone TEXT NOT NULL DEFAULT 'Europe/Moscow',times TEXT NOT NULL DEFAULT '[]',
 weekdays TEXT NOT NULL DEFAULT '[0,1,2,3,4,5,6]',enabled INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL,last_scheduled_at TEXT,
 UNIQUE(recipient,section));
CREATE TABLE observations(id INTEGER PRIMARY KEY,source TEXT NOT NULL,metric TEXT NOT NULL,
 region_id INTEGER NOT NULL DEFAULT 225,device TEXT NOT NULL DEFAULT 'all',frequency TEXT NOT NULL,
 date TEXT NOT NULL,value REAL,unit TEXT NOT NULL,retrieved_at TEXT NOT NULL,
 quality TEXT NOT NULL DEFAULT 'ok',source_url TEXT NOT NULL DEFAULT '',
 keyword_version INTEGER NOT NULL DEFAULT 1,cohort_version INTEGER NOT NULL DEFAULT 0,
 UNIQUE(source,metric,region_id,device,frequency,date));
CREATE INDEX observation_dates ON observations(source,frequency,date);
CREATE TABLE jobs(id INTEGER PRIMARY KEY,kind TEXT NOT NULL,payload TEXT NOT NULL,
 dedupe_key TEXT UNIQUE,state TEXT NOT NULL DEFAULT 'pending',created_at TEXT NOT NULL,
 started_at TEXT,finished_at TEXT,error TEXT);
CREATE TABLE runs(id TEXT PRIMARY KEY,section TEXT NOT NULL,status TEXT NOT NULL,
 started_at TEXT NOT NULL,completed_at TEXT,quality TEXT NOT NULL DEFAULT '{}',
 parameters TEXT NOT NULL DEFAULT '{}');
CREATE TABLE reports(id TEXT PRIMARY KEY,section TEXT NOT NULL,created_at TEXT NOT NULL,
 period_start TEXT,period_end TEXT,keyword_version INTEGER NOT NULL,cohort_version INTEGER NOT NULL,
 text TEXT NOT NULL,full_text TEXT NOT NULL,metrics TEXT NOT NULL DEFAULT '{}',
 news TEXT NOT NULL DEFAULT '[]',quality TEXT NOT NULL DEFAULT '{}',parts TEXT NOT NULL DEFAULT '[]',
 run_id TEXT REFERENCES runs(id),fingerprint TEXT UNIQUE);
CREATE TABLE deliveries(id INTEGER PRIMARY KEY,report_id TEXT NOT NULL REFERENCES reports(id),
 recipient TEXT NOT NULL,variant TEXT NOT NULL DEFAULT 'brief',part_index INTEGER NOT NULL,payload TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'pending',message_id INTEGER,attempts INTEGER NOT NULL DEFAULT 0,
 error TEXT,retry_at TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
 UNIQUE(report_id,recipient,variant,part_index));
CREATE TABLE forecasts(id INTEGER PRIMARY KEY,source TEXT NOT NULL,metric TEXT NOT NULL,
 region_id INTEGER NOT NULL,device TEXT NOT NULL DEFAULT 'all',frequency TEXT NOT NULL DEFAULT 'weekly',
 origin_date TEXT NOT NULL,target_date TEXT NOT NULL,horizon INTEGER NOT NULL,
 prediction REAL NOT NULL,lower REAL,upper REAL,baseline REAL,actual REAL,
 created_at TEXT NOT NULL,model TEXT NOT NULL DEFAULT 'ridge-log1p',
 UNIQUE(source,metric,region_id,device,frequency,origin_date,horizon,model));
CREATE TABLE usage(id INTEGER PRIMARY KEY,provider TEXT NOT NULL,day TEXT NOT NULL,
 calls INTEGER NOT NULL DEFAULT 1,tokens_in INTEGER NOT NULL DEFAULT 0,
 tokens_out INTEGER NOT NULL DEFAULT 0,cost REAL,status TEXT NOT NULL DEFAULT 'reserved',
 created_at TEXT NOT NULL);
CREATE INDEX usage_day ON usage(provider,day);
CREATE TABLE alert_rules(id INTEGER PRIMARY KEY,name TEXT NOT NULL,metric TEXT,
 region_id INTEGER,enabled INTEGER NOT NULL DEFAULT 1,pct_threshold REAL NOT NULL DEFAULT 50,
 z_threshold REAL NOT NULL DEFAULT 2.5,min_coverage REAL NOT NULL DEFAULT .9,
 min_baseline REAL NOT NULL DEFAULT 20,cooldown_hours REAL NOT NULL DEFAULT 24);
CREATE TABLE alerts(id INTEGER PRIMARY KEY,rule_id INTEGER REFERENCES alert_rules(id),
 metric TEXT NOT NULL,region_id INTEGER NOT NULL,observed_date TEXT NOT NULL,
 created_at TEXT NOT NULL,payload TEXT NOT NULL,report_id TEXT REFERENCES reports(id),dispatched INTEGER NOT NULL DEFAULT 0,
 UNIQUE(rule_id,metric,region_id,observed_date));
CREATE TABLE news(id INTEGER PRIMARY KEY,report_id TEXT NOT NULL REFERENCES reports(id),
 title TEXT NOT NULL,url TEXT NOT NULL DEFAULT '',published_at TEXT,claim TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'unverified',fingerprint TEXT NOT NULL,payload TEXT NOT NULL DEFAULT '{}');
CREATE TABLE keyword_proposals(id INTEGER PRIMARY KEY,created_at TEXT NOT NULL,
 entries TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending');
"""


class Store:
    def __init__(self, path=None):
        self.path = Path(path or ROOT / "data" / "thesaurus.sqlite3").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.migrate()
        except (sqlite3.DatabaseError, RuntimeError):
            self.path.with_suffix('.integrity-emergency.json').write_text(
                dumps({'kind': 'sqlite_or_schema_integrity', 'severity': 'critical', 'at': utcnow()}), encoding='utf-8')
            raise

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def migrate(self):
        with self.connection() as conn:
            # Simultaneous first launches can race while changing the journal
            # mode. This PRAGMA can return BUSY immediately despite busy_timeout.
            deadline = time.monotonic() + 15
            while True:
                try:
                    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                    if mode != "wal":
                        raise RuntimeError("SQLite WAL mode is unavailable")
                    break
                except sqlite3.OperationalError as exc:
                    if (getattr(exc, "sqlite_errorcode", None) not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
                            or time.monotonic() >= deadline):
                        raise
                    time.sleep(.05)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > 3:
                raise RuntimeError("Database schema is newer than this application")
            if version == 0:
                conn.execute("BEGIN IMMEDIATE")
                # Another process may have initialized the fresh database while
                # this connection waited for the write lock.
                if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
                    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    if tables:
                        # sqlite backup must run outside our own write transaction.
                        conn.rollback()
                        backup = self.path.with_name(self.path.name + f".backup-{uuid4().hex}.sqlite3")
                        with closing(sqlite3.connect(backup)) as target:
                            conn.backup(target)
                        raise RuntimeError("Unrecognized database schema; a backup was created")
                    for statement in SCHEMA.split(";"):
                        if statement.strip():
                            conn.execute(statement)
                    conn.execute("PRAGMA user_version=1")
        from .market_store import migrate
        migrate(self)
        defaults = {
            "keywords": DEFAULT_KEYWORDS, "keywords_history_1": DEFAULT_KEYWORDS, "keyword_version": 1, "regions": [],
            "cohort_version": 0, "incident_phrases": INCIDENTS,
            "budgets": {"wordstat_calls": 300, "ai_calls": 8, "ai_output_tokens": 2200, "prices": {}},
        }
        with self.connection() as conn:
            conn.executemany("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)",
                             [(k, dumps(v)) for k, v in defaults.items()])
            if not conn.execute("SELECT 1 FROM alert_rules LIMIT 1").fetchone():
                conn.execute("INSERT INTO alert_rules(name) VALUES(?)", ("Изменение поискового интереса",))
        from .subscription_access import migrate as migrate_access
        migrate_access(self)
        from .user_management import migrate as migrate_users
        migrate_users(self)
        from .wine_cellar import migrate as migrate_cellar
        migrate_cellar(self)
        from .life_ledger import migrate as migrate_life
        migrate_life(self)
        from .monitoring import migrate as migrate_monitoring
        migrate_monitoring(self)

    def query(self, sql, params=()):
        with self.connection() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def one(self, sql, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql, params=()):
        with self.connection() as conn:
            return conn.execute(sql, params).lastrowid

    def get_setting(self, key, default=None):
        row = self.one("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set_setting(self, key, value):
        self.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, dumps(value)))

    def keywords(self):
        return self.get_setting("keywords", [])

    def regions(self):
        return self.get_setting("regions", [])

    def version(self, name):
        return int(self.get_setting(name + "_version", 0))

    def set_keywords(self, entries):
        normalized = []
        for entry in entries:
            phrase = str(entry.get("phrase", "")).strip()
            if not phrase or len(phrase) > 120:
                raise ValueError("Each phrase must contain 1 to 120 characters")
            if phrase not in [x["phrase"] for x in normalized]:
                normalized.append({"phrase": phrase, "theme": str(entry.get("theme", "другое"))[:80]})
        if not normalized or len(normalized) > 100:
            raise ValueError("The watchlist must contain 1 to 100 phrases")
        self._versioned_setting("keywords", "keyword_version", normalized)

    def set_regions(self, regions, expected_version=None):
        ids = sorted({int(x) for x in regions})
        if not ids or len(ids) > 100 or min(ids) <= 0:
            raise ValueError("Choose 1 to 100 valid region IDs")
        return self._versioned_setting("regions", "cohort_version", ids, expected_version=expected_version)

    def _versioned_setting(self, key, version_key, value, expected_version=None):
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("SELECT value FROM settings WHERE key=?", (version_key,)).fetchone()
            current_version = int(json.loads(version[0])) if version else 0
            if expected_version is not None and current_version != expected_version:
                return False
            current = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            if current and json.loads(current[0]) == value:
                return True
            number = current_version + 1
            for k, v in ((key, value), (version_key, number)):
                conn.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                             (k, dumps(v)))
            conn.execute("INSERT INTO settings(key,value) VALUES(?,?)", (f"{key}_history_{number}", dumps(value)))
            return True

    def enqueue(self, kind, payload=None, dedupe_key=None):
        with self.connection() as conn:
            cursor = conn.execute("INSERT OR IGNORE INTO jobs(kind,payload,dedupe_key,created_at) VALUES(?,?,?,?)",
                                  (kind, dumps(payload or {}), dedupe_key, utcnow()))
            if cursor.rowcount:
                identity = cursor.lastrowid
            else:
                identity = conn.execute("SELECT id FROM jobs WHERE dedupe_key=?", (dedupe_key,)).fetchone()[0]
        from .telegram_navigation import CURRENT
        state = CURRENT.get()
        if state is not None:
            jobs = state.setdefault('monitor_job_ids',[])
            if identity not in jobs and len(jobs)<20:
                jobs.append(identity)
        return identity

    def save_observations(self, rows):
        timestamp = utcnow()
        keyword_version, cohort_version = self.version("keyword"), self.version("cohort")
        with self.connection() as conn:
            for row in rows:
                value = row.get("value")
                if value is not None and not math.isfinite(float(value)):
                    value = None
                values = (row["source"], row["metric"], int(row.get("region_id", 225)),
                          row.get("device", "all"), row["frequency"], row["date"], value,
                          row.get("unit", "count"), row.get("retrieved_at", timestamp),
                          row.get("quality", "ok" if value is not None else "missing"),
                          row.get("source_url", ""), row.get("keyword_version", keyword_version),
                          row.get("cohort_version", cohort_version))
                conn.execute("""INSERT INTO observations(source,metric,region_id,device,frequency,date,value,unit,
                 retrieved_at,quality,source_url,keyword_version,cohort_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                 ON CONFLICT(source,metric,region_id,device,frequency,date) DO UPDATE SET value=excluded.value,
                 unit=excluded.unit,retrieved_at=excluded.retrieved_at,quality=excluded.quality,
                 source_url=excluded.source_url,keyword_version=excluded.keyword_version,cohort_version=excluded.cohort_version""", values)

    def add_user(self, user_id, role="subscriber", enabled=True, *, subscriptions=None):
        user_id = int(user_id)
        if user_id <= 0 or role not in {"admin", "subscriber"}:
            raise ValueError("Invalid user or role")
        from .subscription_access import key, normalize
        granted = normalize(subscriptions) if subscriptions is not None else []
        from .feature_access import for_sections
        from .feature_access import key as feature_key
        with self.connection() as conn:
            conn.execute("""INSERT INTO users(user_id,role,enabled,chat_id,created_at) VALUES(?,?,?,?,?)
          ON CONFLICT(user_id) DO UPDATE SET role=excluded.role,enabled=excluded.enabled""",
                     (user_id, role, int(enabled), str(user_id), utcnow()))
            conn.execute('INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)', (key(user_id), dumps(granted)))
            conn.execute('INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)', (feature_key(user_id), dumps(for_sections(granted))))

    def authorize(self, user_id, admin=False):
        row = self.one("SELECT * FROM users WHERE user_id=? AND enabled=1", (user_id,))
        return bool(row and (not admin or row["role"] == "admin"))
