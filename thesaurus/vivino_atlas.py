"""Offline geography, exact identity matching and per-owner tasting progress."""
import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from .telegram_data_jobs import require_actor
from .telegram_projects import setting_key


@lru_cache(maxsize=1)
def catalog():
    return json.loads((Path(__file__).parent / 'resources' / 'vivino_catalog.json').read_text(encoding='utf-8'))


def normalize(value):
    value = unicodedata.normalize('NFKD', str(value)).casefold()
    return ' '.join(re.sub(r'[^\w\s]', ' ', ''.join(c for c in value if not unicodedata.combining(c))).split())


def wine_id(row):
    return hashlib.sha256('|'.join(str(row.get(k, '')) for k in ('name', 'winery', 'vintage', 'country', 'region')).encode()).hexdigest()[:16]


def subject(store, viewer, owner=None):
    require_actor(store, viewer)
    from .subscription_access import require
    require(store, viewer, 'vivino')
    owner = viewer if owner is None else owner
    if type(owner) is not int or owner <= 0 or (owner != viewer and not store.authorize(viewer, admin=True)):
        raise PermissionError('Данные другого пользователя доступны только администратору')
    return owner


@lru_cache(maxsize=1)
def indexes():
    reference = catalog()
    country_names, region_names, grape_names = {}, {}, {}
    for country in reference['countries']:
        for name in [country['id'], country['name'], country['ru'], *country['aliases']]:
            if name:
                country_names[normalize(name)] = country['id']
    country_names.update({normalize(k): v for k, v in {'United States': 'US', 'USA': 'US', 'Viet Nam': 'VN',
        'Russia': 'RU', 'South Korea': 'KR', 'Czech Republic': 'CZ'}.items()})
    for kind, target in [('regions', region_names), ('grapes', grape_names)]:
        for row in reference[kind]:
            for name in {row['id'], row['name'], row.get('ru', ''), *row['aliases']}:
                if name:
                    target.setdefault(normalize(name), set()).add(row['id'])
    return country_names, region_names, grape_names


def match_grape(name):
    matches = indexes()[2].get(normalize(name), set())
    return next(iter(matches)) if len(matches) == 1 else None


def model(store, viewer, owner=None):
    owner = subject(store, viewer, owner)
    dataset = store.get_setting(setting_key(owner, 'vivino', 'dataset'))
    reference = catalog()
    annotations = store.get_setting(setting_key(owner, 'vivino', 'annotations'), {})
    manual = store.get_setting(setting_key(owner, 'vivino', 'progress'), [])
    plan = store.get_setting(setting_key(owner, 'vivino', 'plan'), [])
    countries = {r['id']: dict(r, tasted=0) for r in reference['countries']}
    regions = {r['id']: dict(r, tasted=0, wines=[], planned='region:' + r['id'] in plan, coordinate_quality='region' if r['lon'] is not None else 'country') for r in reference['regions']}
    grapes = {r['id']: dict(r, tasted=0, wines=[], planned='grape:' + r['id'] in plan) for r in reference['grapes']}
    custom = store.get_setting(setting_key(owner, 'vivino', 'custom'), [])
    for row in custom:
        collection = regions if row['kind'] == 'region' else grapes
        collection[row['id']] = dict(row, tasted=0, wines=[], planned=row['kind'] + ':' + row['id'] in plan,
                                     ru='', aliases=[], coordinate_quality='user' if row.get('lon') is not None else 'country')
    wines, unknown_grapes = [], 0
    for original in dataset['rows'] if dataset else []:
        row = dict(original)
        identifier = wine_id(row)
        note = annotations.get(identifier, {})
        row.update(id=identifier, tasted=note.get('tasted', bool(row.get('rating') is not None or row.get('review'))))
        country = indexes()[0].get(normalize(row.get('country', '')))
        matches = indexes()[1].get(normalize(row.get('region', '')), set())
        candidates = [regions[i] for i in matches if not country or regions[i]['country'] == country]
        personal = [regions[r['id']] for r in custom if r['kind'] == 'region'
            and normalize(r['name']) == normalize(row.get('region', '')) and (not country or r['country'] == country)]
        if personal:
            candidates = personal
        region = candidates[0] if len(candidates) == 1 else None
        if region is None:
            country_identity = country or normalize(row.get('country', ''))
            key = 'u' + hashlib.sha256((country_identity + '|' + normalize(row.get('region', ''))).encode()).hexdigest()[:16]
            region = regions.setdefault(key, {'id': key, 'name': row.get('region') or 'Регион не указан', 'ru': '',
                'country': country or '', 'description': 'Регион из вашего экспорта; точная привязка не найдена в справочнике.',
                'lat': None, 'lon': None, 'tasted': 0, 'wines': [], 'source': '', 'planned': 'region:' + key in plan,
                'coordinate_quality': 'country' if country else 'unknown'})
        region['wines'].append(len(wines))
        region['tasted'] += bool(row['tasted'])
        row.update(region_id=region['id'], country_id=country, grapes=[], unknown_grapes=[])
        grape_values = note.get('grapes', original.get('grapes', []))
        for value in grape_values:
            grape = match_grape(value)
            if not grape:
                matches = [r['id'] for r in custom if r['kind'] == 'grape' and normalize(r['name']) == normalize(value)]
                grape = matches[0] if len(matches) == 1 else None
            if grape:
                if grape in row['grapes']:
                    continue
                row['grapes'].append(grape)
                grapes[grape]['wines'].append(len(wines))
                grapes[grape]['tasted'] += bool(row['tasted'])
            else:
                row['unknown_grapes'].append(value)
        if row['tasted'] and (not grape_values or row['unknown_grapes']):
            unknown_grapes += 1
        if country and row['tasted']:
            countries[country]['tasted'] += 1
        wines.append(row)
    for kind, collection in [('region', regions), ('grape', grapes)]:
        for row in collection.values():
            row['manual'] = kind + ':' + row['id'] in manual
            row['tried'] = row['tasted'] > 0 or row['manual']
    result = {'owner': owner, 'version': reference['version'], 'complete': False,
        'regions': list(regions.values()), 'grapes': list(grapes.values()), 'countries': list(countries.values()),
        'wines': wines, 'unknown_grapes': unknown_grapes,
        'stats': {'tasted': sum(r['tasted'] for r in wines), 'regions': sum(r['tried'] for r in regions.values()),
                  'grapes': sum(r['tried'] for r in grapes.values()), 'planned': len(plan),
                  'located': sum(regions[r['region_id']]['lon'] is not None for r in wines)},
        'quality': dataset['quality'] if dataset else {}, 'sources': reference['sources']}
    from .vivino_analytics import analyze
    result['analytics'] = analyze(result)
    return result


def revise(store, actor, name, value):
    """Owner-only edits invalidate stale queued maps and delivery snapshots."""
    subject(store, actor)
    if name not in {'annotations', 'progress', 'plan', 'custom'}:
        raise ValueError('Неизвестная настройка Vivino')
    from uuid import uuid4

    from .store import dumps, utcnow
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        user = conn.execute('SELECT enabled FROM users WHERE user_id=?', (actor,)).fetchone()
        if not user or not user[0]:
            raise PermissionError('Нет доступа')
        conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (setting_key(actor, 'vivino', name), dumps(value)))
        row = conn.execute('SELECT value FROM settings WHERE key=?', (setting_key(actor, 'vivino', 'dataset'),)).fetchone()
        if row:
            dataset = json.loads(row[0])
            dataset['generation'] = uuid4().hex
            conn.execute('UPDATE settings SET value=? WHERE key=?', (dumps(dataset), setting_key(actor, 'vivino', 'dataset')))
        for job in conn.execute("SELECT id,payload FROM jobs WHERE kind='vivino_map' AND state='pending'").fetchall():
            payload = json.loads(job['payload'])
            if payload.get('owner_id', payload.get('actor_id')) == actor:
                conn.execute("UPDATE jobs SET state='cancelled',finished_at=? WHERE id=?", (utcnow(), job['id']))
