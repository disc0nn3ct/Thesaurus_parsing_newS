"""Experimental public website JSON interface, not the gated official Trends API."""
import csv
import io
import json
import re
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from itertools import pairwise
from zoneinfo import ZoneInfo

import requests

from . import google_trends
from .store import dumps, utcnow

BASE = 'https://trends.google.com/trends/api/'
USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')
WINDOWS = {'today 12-m': 'Год', 'today 5-y': '5 лет'}
LIMIT = 60


class TrendsError(ValueError):
    def __init__(self, kind, cooldown=0):
        self.kind, self.cooldown = kind, cooldown
        super().__init__({
            'rate_limit': 'Google ограничил обращения (429). Автосбор поставлен на паузу; сохранённые данные доступны.',
            'blocked': 'Google временно закрыл доступ. Сохранённые данные доступны; используйте CSV.',
            'network': 'Не удалось подключиться к Google. Сохранённые данные доступны.',
            'schema': 'Формат ответа Google изменился. Данные не записаны; используйте CSV.',
            'budget': 'Достигнут дневной лимит бесплатного автосбора (60 обращений).',
            'http': 'Google не вернул данные. Сохранённая история доступна.',
        }[kind])


def validate_geo(value):
    if value != '' and (not isinstance(value, str) or not re.fullmatch(r'[A-Z]{2}(?:-[A-Z0-9]{1,5})?', value)):
        raise ValueError('Регион Google: RU, US, RU-MOW или WORLD (весь мир).')
    return value


def region_label(geo):
    return {'RU': 'Россия', '': 'Весь мир'}.get(geo, geo)


class TrendsClient:
    def __init__(self, store, *, session=None, sleep=time.sleep, clock=time.monotonic):
        self.store = store
        self.session = session or requests.Session()
        self.sleep, self.clock, self.last = sleep, clock, None

    def close(self):
        self.session.close()

    def _get(self, path, params):
        if path not in {'explore', 'widgetdata/multiline'}:
            raise ValueError('Unknown Trends endpoint')
        day = datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat()
        # Single worker, plus a transactional reservation before every actual request.
        with self.store.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute("SELECT value FROM settings WHERE key='trends_usage'").fetchone()
            usage = json.loads(row[0]) if row else {}
            calls = usage.get('calls', 0) if usage.get('day') == day else 0
            if calls >= LIMIT:
                raise TrendsError('budget')
            conn.execute("INSERT INTO settings(key,value) VALUES('trends_usage',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (dumps({'day': day, 'calls': calls+1}),))
        if self.last is not None:
            self.sleep(max(0, 1.0 - (self.clock() - self.last)))
        self.last = self.clock()
        response = None
        try:
            response = self.session.get(BASE + path, params=params, timeout=(10, 30), stream=True,
                allow_redirects=False, headers={'User-Agent': USER_AGENT, 'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.7'})
            if response.status_code == 429:
                retry = response.headers.get('Retry-After', '')
                pause = max(3600, min(86400, int(retry))) if re.fullmatch(r'\d{1,8}', retry) else 3600
                raise TrendsError('rate_limit', pause)
            if response.status_code in {401, 403}:
                raise TrendsError('blocked', 3600)
            if response.status_code != 200:
                raise TrendsError('http', 900)
            parts, size = [], 0
            for part in response.iter_content(65536):
                size += len(part)
                if size > google_trends.MAX_BYTES:
                    raise TrendsError('schema', 3600)
                parts.append(part)
            body = b''.join(parts).decode('utf-8').lstrip()
            if body.startswith(")]}'"):
                body = body[4:].lstrip(',\n ')
            result = json.loads(body)
            if not isinstance(result, dict):
                raise TrendsError('schema', 3600)
            return result
        except requests.RequestException:
            raise TrendsError('network', 900) from None
        except (UnicodeError, json.JSONDecodeError):
            raise TrendsError('schema', 3600) from None
        finally:
            if response is not None:
                response.close()

    def fetch(self, phrase, *, geo='RU', window='today 5-y'):
        validate_geo(geo)
        if window not in WINDOWS or not isinstance(phrase, str) or not 0 < len(phrase) <= 200:
            raise ValueError('Неверные параметры Google Trends.')
        query = {'comparisonItem': [{'keyword': phrase, 'geo': geo, 'time': window}], 'category': 0, 'property': ''}
        explore = self._get('explore', {'hl': 'ru', 'tz': '-180', 'req': dumps(query)})
        widgets = explore.get('widgets')
        widget = next((w for w in widgets if isinstance(w, dict) and w.get('id') == 'TIMESERIES'), None) if isinstance(widgets, list) else None
        if not widget or not isinstance(widget.get('request'), dict) or not isinstance(widget.get('token'), str) or not 0 < len(widget['token']) <= 4096:
            raise TrendsError('schema', 3600)
        result = self._get('widgetdata/multiline', {'hl': 'ru', 'tz': '-180', 'req': dumps(widget['request']), 'token': widget['token']})
        return to_csv(result, phrase, geo, window)


def to_csv(result, phrase, geo, window):
    """Strict schema validation; zeros/missing/partial periods stay distinguished."""
    try:
        rows = result['default']['timelineData']
        if not isinstance(rows, list) or not 2 <= len(rows) <= 10000:
            raise ValueError
        values = []
        for row in rows:
            stamp = row['time']
            if not isinstance(stamp, str) or not re.fullmatch(r'\d{1,12}', stamp):
                raise ValueError
            stamp = datetime.fromtimestamp(int(stamp), timezone.utc).date()
            value, has_data = row['value'], row.get('hasData', [True])
            partial = row.get('isPartial', False)
            if (not isinstance(value, list) or len(value) != 1 or type(value[0]) is not int or not 0 <= value[0] <= 100
                    or not isinstance(has_data, list) or len(has_data) != 1 or type(has_data[0]) is not bool or type(partial) is not bool):
                raise ValueError
            values.append((stamp, '' if partial or not has_data[0] else value[0]))
        dates = [r[0] for r in values]
        if dates != sorted(set(dates)) or dates[-1] > date.fromisoformat(utcnow()[:10]):
            raise ValueError
        gaps = Counter((b-a).days for a, b in pairwise(dates))
        step = gaps.most_common(1)[0][0]
        frequency = 'monthly' if all(d.day == 1 for d in dates) and step >= 28 else 'weekly' if step == 7 else 'daily' if step == 1 else None
        if frequency is None:
            raise ValueError
        output = io.StringIO(newline='')
        output.write(f'Google Trends: Web Search; category=0; geo={geo or "WORLD"}; window={window}\n\n')
        writer = csv.writer(output)
        writer.writerow([{'daily': 'Day', 'weekly': 'Week', 'monthly': 'Month'}[frequency], phrase + ': (' + region_label(geo) + ')'])
        for stamp, value in values:
            writer.writerow([stamp.strftime('%Y-%m') if frequency == 'monthly' else stamp.isoformat(), value])
        return output.getvalue().encode('utf-8')
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        raise TrendsError('schema', 3600) from None


def collect(store, actor, *, dry_run=False, client_factory=TrendsClient):
    from .telegram_data_jobs import require_actor
    require_actor(store, actor, admin=True)
    from .search_provider import require_section
    require_section(store, 'trends')
    config = store.get_setting('trends_schedule', {})
    geo = validate_geo(config.get('geo', 'RU'))
    window = config.get('window', 'today 5-y')
    if window not in WINDOWS:
        raise ValueError('Неверный период Google Trends.')
    if dry_run:
        return {'status': 'cached', 'dry_run': True}
    pause = store.get_setting('trends_pause_until', '')
    if pause and utcnow() < pause:
        return {'status': 'paused', 'pause_until': pause, 'errors': ['Google Trends временно на паузе. Сохранённые данные доступны.']}
    result = {'status': 'ok', 'retrieved_at': utcnow(), 'series': [], 'errors': [], 'geo': geo, 'window': window}
    phrases = config.get('phrases') or [r['phrase'] for r in store.keywords()]
    allowed = {r['phrase'] for r in store.keywords()}
    if not isinstance(phrases, list) or not phrases or len(phrases) > 50 or not set(phrases).issubset(allowed):
        raise ValueError('Выберите до 50 разрешённых фраз Google Trends.')
    client = client_factory(store)
    try:
        for phrase in dict.fromkeys(phrases):
            require_actor(store, actor, admin=True)
            try:
                raw = client.fetch(phrase, geo=geo, window=window)
                ids = google_trends.import_csv(store, raw, actor, acquisition='unofficial_api')
                result['series'].extend(ids)
            except TrendsError as exc:
                result['status'] = 'partial'
                result['errors'].append(str(exc))
                if exc.cooldown:
                    result['pause_until'] = (datetime.now(timezone.utc) + timedelta(seconds=exc.cooldown)).isoformat()
                    store.set_setting('trends_pause_until', result['pause_until'])
                break  # Shared source failure must not repeat for every keyword.
    finally:
        client.close()
    if result['status'] == 'ok' or not result.get('pause_until'):
        store.set_setting('trends_pause_until', '')
    store.set_setting('trends_last_collection', result)
    return result


def enqueue_due(store, now=None):
    """Coalesce sleep/missed slots; one shared worker performs the actual requests."""
    from .scheduling import _latest_due
    now = now or datetime.now(timezone.utc)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute("SELECT value FROM settings WHERE key='trends_schedule'").fetchone()
        config = json.loads(row[0]) if row else {}
        if not config.get('enabled') or not google_trends.active(store):
            return None
        actor = conn.execute('SELECT enabled,role FROM users WHERE user_id=?', (config.get('actor_id'),)).fetchone()
        if not actor or not actor['enabled'] or actor['role'] != 'admin':
            return None
        pause = conn.execute("SELECT value FROM settings WHERE key='trends_pause_until'").fetchone()
        pause_until = json.loads(pause[0]) if pause else ''
        if pause_until and pause_until > now.isoformat():
            return None
        slot = _latest_due({**config, 'times': dumps(config.get('times', ['08:30'])), 'weekdays': dumps(list(range(7))),
                            'timezone': 'Europe/Moscow', 'last_scheduled_at': config.get('last_scheduled_at')}, now)
        if slot is None and not pause_until:
            return None
        # A paused source resumes automatically, even before the next daily slot.
        # Consume the expired pause transactionally so each recovery is enqueued once.
        if pause_until:
            conn.execute("UPDATE settings SET value=? WHERE key='trends_pause_until'", (dumps(''),))
        payload = dumps({'actor_id': config['actor_id'], 'scheduled': True})
        pending = conn.execute("SELECT id FROM jobs WHERE kind='trends_collect' AND state='pending' LIMIT 1").fetchone()
        key = 'trends-retry:' + pause_until if pause_until else 'trends-collection:' + slot.isoformat()
        if pending:
            identifier = pending['id']
            # Preserve a manual request's receipt and actor when sharing its collection.
            manual = conn.execute('SELECT payload FROM jobs WHERE id=?', (identifier,)).fetchone()
            if json.loads(manual[0]).get('scheduled'):
                conn.execute('UPDATE jobs SET payload=?,dedupe_key=? WHERE id=?', (payload, key, identifier))
        else:
            cursor = conn.execute("INSERT OR IGNORE INTO jobs(kind,payload,dedupe_key,created_at) VALUES('trends_collect',?,?,?)", (payload, key, utcnow()))
            identifier = cursor.lastrowid if cursor.rowcount else None
        if slot is not None:
            config['last_scheduled_at'] = slot.isoformat()
        conn.execute("UPDATE settings SET value=? WHERE key='trends_schedule'", (dumps(config),))
        return identifier


def run_job(store, job, payload, *, dry_run=False):
    from uuid import uuid4

    from .delivery import prepare_delivery
    from .telegram_data_jobs import require_actor
    actor = payload['actor_id']
    require_actor(store, actor, admin=True)
    fingerprint = f"trends-collect:{job['id']}"
    existing = store.one('SELECT id FROM reports WHERE fingerprint=?', (fingerprint,))
    if existing and not dry_run:
        prepare_delivery(store, existing['id'], str(actor), detail='full')
        return {'status': 'cached'}
    if payload.get('scheduled') and not store.get_setting('trends_schedule', {}).get('enabled'):
        return None
    result = collect(store, actor, dry_run=dry_run)
    if dry_run or payload.get('scheduled'):
        return result
    existing = store.one('SELECT id FROM reports WHERE fingerprint=?', (fingerprint,))
    if existing:
        identifier = existing['id']
    else:
        identifier = uuid4().hex
        text = f"Google Trends: обновлено рядов {len(result.get('series', []))}; состояние {result['status']}.\n" + '\n'.join(result.get('errors', [])) + '\n' + google_trends.NOTE + '\nГрафики: /chart trends.'
        store.execute("INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,quality,parts,fingerprint) VALUES(?,'trends',?,?,?,?,?,?,?,?)",
                      (identifier, utcnow(), store.version('keyword'), store.version('cohort'), text, text, dumps({'actor_id': actor}), dumps([]), fingerprint))
    require_actor(store, actor, admin=True)
    prepare_delivery(store, identifier, str(actor), detail='full')
    return result
