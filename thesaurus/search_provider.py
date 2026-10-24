"""Explicit search shutdown, without enabling a replacement provider."""
import json
import sqlite3
from contextlib import closing
from uuid import uuid4

from .store import dumps, utcnow

SEARCH_SECTIONS = {'trends', 'wordstat', 'regions'}
NOTICE = 'Поисковые индикаторы временно отключены. Выберите показатели Банка России.'


def disabled(store):
    return store.get_setting('search_provider') == 'disabled'


def require_section(store, section):
    if section in SEARCH_SECTIONS and disabled(store):
        raise ValueError(NOTICE)


def search_job(kind, payload):
    return (kind in {'trends_collect', 'trends_import', 'propose_keywords'}
            or payload.get('section') in SEARCH_SECTIONS)


def hidden_route(route):
    parts = (route or '').split(':')
    return (parts[0] in {'gt', 'kw', 'rg'}
            or route in {'menu:keywords', 'menu:regions', 'menu:forecasts', 'menu:alerts', 'menu:proposals'}
            or parts[0] in {'ch', 'cs', 'request', 'menu', 'ui'}
            and any(part in SEARCH_SECTIONS for part in parts))


def deactivate(store):
    """Called under the worker lock. Keep history and settings; stop future work."""
    with store.connection() as conn:
        current = conn.execute("SELECT value FROM settings WHERE key='search_provider'").fetchone()
        if not current or json.loads(current[0]) != 'disabled':
            backup = store.path.with_name(store.path.name + f'.search-backup-{uuid4().hex}.sqlite3')
            with closing(sqlite3.connect(backup)) as target:
                conn.backup(target)
        conn.execute('BEGIN IMMEDIATE')
        conn.execute("INSERT INTO settings(key,value) VALUES('search_provider',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (dumps('disabled'),))
        conn.execute("UPDATE subscriptions SET enabled=0 WHERE section IN ('wordstat','regions')")
        conn.execute("UPDATE market_series SET enabled=0 WHERE source IN ('google_trends','wordstat')")
        conn.execute("UPDATE alert_rules SET enabled=0")
        conn.execute("UPDATE market_rules SET enabled=0 WHERE series_id IN (SELECT id FROM market_series WHERE source IN ('google_trends','wordstat'))")
        for row in conn.execute("SELECT key,value FROM settings WHERE key='trends_schedule' OR key LIKE 'chart_schedule:%'").fetchall():
            config = json.loads(row['value'])
            if row['key'] == 'trends_schedule' or config.get('section') in SEARCH_SECTIONS:
                config['enabled'] = False
                conn.execute('UPDATE settings SET value=? WHERE key=?', (dumps(config), row['key']))
        for row in conn.execute("SELECT id,kind,payload FROM jobs WHERE state='pending'").fetchall():
            if search_job(row['kind'], json.loads(row['payload'])):
                conn.execute("UPDATE jobs SET state='cancelled',finished_at=?,error='search_disabled' WHERE id=?", (utcnow(), row['id']))
        conn.execute("UPDATE deliveries SET state='cancelled',updated_at=? WHERE state='pending' AND (report_id IN (SELECT id FROM reports WHERE section IN ('trends','wordstat','regions')) OR json_extract(payload,'$.chart_schedule.section') IN ('trends','wordstat','regions'))", (utcnow(),))
        conn.execute("UPDATE deliveries SET state='cancelled',updated_at=? WHERE state='pending' AND report_id IN (SELECT s.report_id FROM market_signals s JOIN market_rules r ON r.id=s.rule_id JOIN market_series m ON m.id=r.series_id WHERE m.source IN ('google_trends','wordstat'))", (utcnow(),))
        conn.execute("UPDATE bot_sessions SET pending=NULL WHERE pending LIKE 'trends%' OR pending LIKE 'chartdelivery:trends' OR pending LIKE 'chartdelivery:wordstat'")
