"""Reviewable backfill of saved wine lots, independent from receipts and purchases."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from . import wine_cellar as cellar
from .store import dumps, utcnow

LABELS = {'name':'Название','winery':'Производитель','vintage':'Год урожая','volume_ml':'Объём бутылки',
          'country':'Страна','region':'Регион','grapes':'Сорта винограда'}
CODES = dict(zip('nwyvcrg',LABELS,strict=True))


def display(field,value):
    if value in (None,'',[]):
        return 'не указано'
    return ', '.join(value) if isinstance(value,list) else str(value)+(' мл' if field=='volume_ml' else '')


def missing(lot):
    return [key for key in LABELS if not lot.get(key)]


def parse(field,value):
    if field not in LABELS:
        raise ValueError('Неизвестное поле')
    if value.strip().casefold() in {'не знаю','не указано','-'} and field not in {'name','volume_ml'}:
        return [] if field=='grapes' else ''
    if field=='vintage':
        return cellar._vintage(value)
    if field=='volume_ml':
        if not value.strip().isascii() or not value.strip().isdigit():
            raise ValueError('Введите объём целым числом миллилитров, например 750.')
        return cellar._integer(int(value),20,30000,'Объём')
    if field in {'country','region','grapes'}:
        return cellar._attributes({field:value})[field]
    return cellar._text(value,required=field=='name')


def key(actor,lot):
    return f'wine_details:{actor}:{lot}'


def draft(store,actor,lot,*,create=False):
    cellar.require_owner(store,actor,write=True)
    saved = cellar.get_lot(store,actor,lot)
    value = store.get_setting(key(actor,lot))
    if not isinstance(value,dict) or value.get('expires','')<utcnow():
        if not create:
            raise ValueError('Форма устарела. Откройте уточнение вина заново.')
        value = {'token':uuid4().hex[:12],'base':cellar.details_signature(saved),
                 'original':{field:saved[field] for field in LABELS},'fields':{field:saved[field] for field in LABELS}}
        save(store,actor,lot,value)
    if value['base']!=cellar.details_signature(saved):
        raise ValueError('Характеристики уже изменились. Отмените этот черновик и откройте уточнение заново.')
    return value


def save(store,actor,lot,value):
    value['expires'] = (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
    store.set_setting(key(actor,lot),value)


def set_field(store,actor,lot,token,field,value):
    current = draft(store,actor,lot)
    if current['token']!=token:
        raise ValueError('Эта форма устарела.')
    current['fields'][field] = parse(field,value)
    if field in {'name','winery','vintage','volume_ml'}:
        current.pop('lookup',None)
    save(store,actor,lot,current)


def choices(store,actor,current,field):
    from .wine_enrichment import candidate_allowed
    result,seen = [],set()
    found = current.get('lookup',{})
    if found.get('expires','')<utcnow():
        return []
    for candidate in found.get('candidates',[])[:8]:
        if not candidate_allowed(store,actor,candidate):
            continue
        raw = candidate.get('published_vintage' if field=='vintage' else field)
        if raw in (None,'',[]):
            continue
        try:
            value = parse(field,', '.join(raw) if isinstance(raw,list) else str(raw))
        except (ValueError,TypeError):
            continue
        fingerprint = hashlib.sha256(dumps([found.get('request'),field,value]).encode()).hexdigest()[:10]
        if fingerprint not in seen:
            seen.add(fingerprint)
            result.append({'value':value,'fingerprint':fingerprint,'label':display(field,value)})
    return sorted(result,key=lambda row:row['label'])


def apply_choice(store,actor,lot,token,field,fingerprint):
    current = draft(store,actor,lot)
    if current['token']!=token:
        raise ValueError('Эта форма устарела.')
    choice = next((row for row in choices(store,actor,current,field) if row['fingerprint']==fingerprint),None)
    if not choice:
        raise ValueError('Вариант устарел. Обновите поиск.')
    current['fields'][field] = choice['value']
    # Preserve other choices from this result; applying one field never fills another.
    save(store,actor,lot,current)


def lookup(store,actor,lot,token,*,dry_run=False):
    from .wine_enrichment import lookup as find
    current = draft(store,actor,lot)
    signature = cellar.details_signature(current['fields'])
    def guard():
        latest = draft(store,actor,lot)
        if latest['token']!=token or cellar.details_signature(latest['fields'])!=signature:
            raise ValueError('Карточка изменилась во время поиска. Запустите поиск заново.')
    guard()
    result = find(store,actor,current['fields'],guard=guard,dry_run=dry_run)
    guard()
    if not dry_run:
        with store.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT value FROM settings WHERE key=?',(key(actor,lot),)).fetchone()
            latest = json.loads(row['value']) if row else None
            if (not isinstance(latest,dict) or latest.get('token')!=token or latest.get('expires','')<utcnow()
                    or cellar.details_signature(latest['fields'])!=signature
                    or cellar.details_signature(cellar._lot(conn,actor,lot))!=latest['base']):
                raise ValueError('Карточка изменилась во время поиска. Запустите поиск заново.')
            latest['lookup'] = {'request':uuid4().hex,'candidates':result.get('candidates',[])[:8],
                                'warnings':result.get('warnings',[])[:3],
                                'expires':(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat()}
            conn.execute('UPDATE settings SET value=? WHERE key=?',(dumps(latest),key(actor,lot)))
    return result
