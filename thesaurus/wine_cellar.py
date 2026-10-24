"""Personal wine lots, stock ledger, dated quotes and confirmed tasting proposals.

Money is stored in kopecks. No network or LLM calls occur in this module.
"""
import hashlib
import ipaddress
import json
import re
import sqlite3
import unicodedata
from contextlib import closing
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4
from zoneinfo import ZoneInfo

from .feature_access import require
from .store import dumps, utcnow

VERSION = 3
ATTRIBUTE_SCHEMA = """
CREATE TABLE IF NOT EXISTS wine_attributes(lot_id INTEGER PRIMARY KEY REFERENCES wine_lots(id),
 user_id INTEGER NOT NULL REFERENCES users(user_id),country TEXT NOT NULL DEFAULT '',
 region TEXT NOT NULL DEFAULT '',grapes TEXT NOT NULL DEFAULT '[]',provenance TEXT NOT NULL DEFAULT '{}');
"""
CONSUMPTION_SCHEMA = """
CREATE TABLE wine_consumptions(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(user_id),
 event_id INTEGER NOT NULL UNIQUE REFERENCES wine_events(id),lot_id INTEGER NOT NULL REFERENCES wine_lots(id),
 quantity INTEGER NOT NULL CHECK(quantity>0),consumed_on TEXT NOT NULL,purchase_cents INTEGER,
 valuation_cents INTEGER,price_date TEXT,price_source TEXT,price_url TEXT,price_id INTEGER REFERENCES wine_prices(id),
 created_at TEXT NOT NULL);
CREATE INDEX wine_consumption_owner ON wine_consumptions(user_id,id);
"""
CONSUMPTION_REASONS = {'drink', 'consumed', 'consume', 'vivino_confirmed'}
SCHEMA = """
CREATE TABLE wine_lots(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(user_id),
 name TEXT NOT NULL,winery TEXT NOT NULL DEFAULT '',vintage TEXT NOT NULL DEFAULT '',
 quantity INTEGER NOT NULL CHECK(quantity>=0),volume_ml INTEGER NOT NULL,
 purchase_cents INTEGER,purchase_date TEXT,location TEXT NOT NULL DEFAULT '',
 notes TEXT NOT NULL DEFAULT '',operation_id TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
 UNIQUE(user_id,operation_id));
CREATE INDEX wine_owner ON wine_lots(user_id,id);
CREATE TABLE wine_events(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(user_id),
 lot_id INTEGER NOT NULL REFERENCES wine_lots(id),delta INTEGER NOT NULL,reason TEXT NOT NULL,
 flow_cents INTEGER,operation_id TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(user_id,operation_id));
CREATE TABLE wine_prices(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(user_id),
 lot_id INTEGER NOT NULL REFERENCES wine_lots(id),amount_cents INTEGER NOT NULL CHECK(amount_cents>=0),
 price_date TEXT NOT NULL,source TEXT NOT NULL,url TEXT,operation_id TEXT NOT NULL,
 retrieved_at TEXT NOT NULL,metadata TEXT NOT NULL DEFAULT '{}',UNIQUE(user_id,operation_id));
CREATE INDEX wine_quote_dates ON wine_prices(lot_id,price_date,id);
CREATE TABLE wine_price_links(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(user_id),
 lot_id INTEGER NOT NULL REFERENCES wine_lots(id),url TEXT NOT NULL,created_at TEXT NOT NULL,
 UNIQUE(lot_id,url));
CREATE TABLE wine_snapshots(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(user_id),
 created_at TEXT NOT NULL,reason TEXT NOT NULL,quantity INTEGER NOT NULL,priced_quantity INTEGER NOT NULL,
 value_cents INTEGER NOT NULL,cost_cents INTEGER NOT NULL,unknown_cost_quantity INTEGER NOT NULL,
 net_flow_cents INTEGER NOT NULL,unknown_flow_events INTEGER NOT NULL,positions TEXT NOT NULL);
CREATE INDEX wine_snapshot_dates ON wine_snapshots(user_id,id);
CREATE TABLE wine_review_state(user_id INTEGER PRIMARY KEY REFERENCES users(user_id),generation TEXT NOT NULL,
 initialized_at TEXT NOT NULL);
CREATE TABLE wine_seen_reviews(user_id INTEGER NOT NULL REFERENCES users(user_id),review_key TEXT NOT NULL,
 first_generation TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(user_id,review_key));
CREATE TABLE wine_review_proposals(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(user_id),
 review_key TEXT NOT NULL,review_name TEXT NOT NULL,winery TEXT NOT NULL,vintage TEXT NOT NULL,
 lot_ids TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending',generation TEXT NOT NULL,
 selected_lot_id INTEGER REFERENCES wine_lots(id),created_at TEXT NOT NULL,resolved_at TEXT,
 UNIQUE(user_id,review_key));
"""


def _today():
    return datetime.now(ZoneInfo('Europe/Moscow')).date()


def migrate(store):
    """Transactional, idempotent extension; preserve existing personal imports."""
    with store.connection() as conn:
        current = conn.execute("SELECT value FROM settings WHERE key='wine_cellar_version'").fetchone()
        if current:
            version = int(json.loads(current['value']))
            if version not in {1, 2, VERSION}:
                raise RuntimeError('Unsupported wine cellar schema')
            if version == VERSION:
                return
        backup = store.path.with_name(store.path.name+f'.wine-backup-{uuid4().hex}.sqlite3')
        with closing(sqlite3.connect(backup)) as target:
            conn.backup(target)
        conn.execute('BEGIN IMMEDIATE')
        current = conn.execute("SELECT value FROM settings WHERE key='wine_cellar_version'").fetchone()
        if current and int(json.loads(current['value'])) == VERSION:
            return
        if current:
            if int(json.loads(current['value'])) == 1:
                for statement in CONSUMPTION_SCHEMA.split(';'):
                    if statement.strip():
                        conn.execute(statement)
            # Only quotes actually recorded by the time of the old consumption
            # may be used. Later downloads must never backfill a fictitious gain.
            for event in conn.execute('''SELECT e.* FROM wine_events e LEFT JOIN wine_consumptions c ON c.event_id=e.id
                                      WHERE e.delta<0 AND c.id IS NULL ORDER BY e.id''').fetchall():
                if event['reason'] in CONSUMPTION_REASONS:
                    day = datetime.fromisoformat(event['created_at']).astimezone(ZoneInfo('Europe/Moscow')).date().isoformat()
                    lot = _lot(conn, event['user_id'], event['lot_id'])
                    quote = _consumption_quote(conn, event['user_id'], event['lot_id'], day, known_at=event['created_at'])
                    _freeze_consumption(conn, event['user_id'], event['id'], lot, -event['delta'], day, quote, event['created_at'])
            conn.execute(ATTRIBUTE_SCHEMA)
            for row in conn.execute('SELECT id,user_id,notes FROM wine_lots').fetchall():
                values = _attributes_from_notes(row['notes'])
                _save_attributes(conn, row['user_id'], row['id'], values)
            conn.execute("UPDATE settings SET value=? WHERE key='wine_cellar_version'", (dumps(VERSION),))
            return
        for statement in SCHEMA.split(';'):
            if statement.strip():
                conn.execute(statement)
        for statement in CONSUMPTION_SCHEMA.split(';'):
            if statement.strip():
                conn.execute(statement)
        conn.execute(ATTRIBUTE_SCHEMA)
        for setting in conn.execute("SELECT key,value FROM settings WHERE key LIKE 'project:vivino:%:dataset'").fetchall():
            parts = setting['key'].split(':')
            if len(parts) != 4 or not parts[2].isdigit():
                continue
            owner = int(parts[2])
            if not conn.execute('SELECT 1 FROM users WHERE user_id=?', (owner,)).fetchone():
                continue
            dataset = json.loads(setting['value'])
            if isinstance(dataset, dict) and isinstance(dataset.get('rows'), list):
                _baseline(conn, owner, dataset['rows'], str(dataset.get('generation') or 'migration'))
        conn.execute("INSERT INTO settings(key,value) VALUES('wine_cellar_version',?)", (dumps(VERSION),))


def require_owner(store, actor, owner=None, write=False):
    if type(actor) is not int or actor <= 0:
        raise PermissionError('Нет доступа')
    require(store, actor, 'wine_cellar')
    owner = actor if owner is None else owner
    if type(owner) is not int or owner <= 0:
        raise ValueError('Неверный пользователь')
    if owner != actor and (write or not store.authorize(actor, admin=True)):
        raise PermissionError('Можно изменять только свою винотеку')
    if not store.one('SELECT 1 FROM users WHERE user_id=?', (owner,)):
        raise ValueError('Пользователь не найден')
    return owner


def _text(value, maximum=200, required=False):
    if value is None:
        value = ''
    if not isinstance(value, str):
        raise ValueError('Ожидается текст')  # noqa: TRY004 - user form validation.
    value = ''.join(c for c in value if c.isprintable() and unicodedata.category(c) != 'Cf').strip()
    if len(value) > maximum or required and not value:
        raise ValueError(f'Введите текст длиной от {1 if required else 0} до {maximum} символов')
    return value


def _integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{label}: целое число от {minimum} до {maximum}')
    return value


def _vintage(value):
    if value is None or value == '':
        return ''
    if isinstance(value, bool):
        raise ValueError('Неверный год вина')  # noqa: TRY004 - user form validation.
    value = str(value).strip().upper()
    if value in {'NV', 'N/V', 'Н/В', 'Б/Г', 'БЕЗ ГОДА'}:
        return 'NV'
    if not re.fullmatch(r'\d{4}', value) or not 1700 <= int(value) <= _today().year+1:
        raise ValueError('Год вина: четыре цифры или NV (без указания года)')
    return value


def _date(value=None):
    try:
        parsed = date.fromisoformat(str(value)) if value is not None and value != '' else _today()
    except ValueError:
        raise ValueError('Дата должна быть в формате ГГГГ-ММ-ДД') from None
    if parsed > _today() or parsed.year < 1900:
        raise ValueError('Дата должна быть от 1900 года до сегодня')
    return parsed.isoformat()


def _cents(value, optional=False):
    if value is None or value == '':
        if optional:
            return None
        raise ValueError('Укажите цену в рублях')
    try:
        if isinstance(value, bool):
            raise InvalidOperation
        amount = Decimal(str(value).replace(',', '.'))
        if not amount.is_finite() or not 0 <= amount <= 1_000_000_000:
            raise InvalidOperation
        return int((amount*100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    except (ValueError, TypeError, InvalidOperation, OverflowError):
        raise ValueError('Цена должна быть неотрицательным конечным числом в рублях') from None


def _operation(value):
    return _text(value or uuid4().hex, maximum=160, required=True)


def validate_price_url(url):
    """Only public HTTPS destinations; fetchers must additionally validate DNS."""
    url = _text(url, maximum=2000, required=True)
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if (parsed.scheme != 'https' or not host or parsed.username or parsed.password
                or parsed.port not in {None, 443} or '.' not in host
                or host.lower().endswith(('.local', '.localhost', '.internal', '.test'))):
            raise ValueError
        try:
            if not ipaddress.ip_address(host).is_global:
                raise ValueError('Непубличный адрес')
        except ValueError as exc:
            if str(exc) == 'Непубличный адрес' or re.fullmatch(r'[0-9.]+', host):
                raise ValueError from None
        if host.lower() in {'localhost', 'metadata.google.internal'} or not re.fullmatch(r'[A-Za-z0-9.-]+', host):
            raise ValueError
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or '/', parsed.query, ''))
    except ValueError:
        raise ValueError('Нужна публичная ссылка HTTPS на страницу конкретного вина без логина и пароля') from None


def _attributes(fields):
    grapes = fields.get('grapes') or []
    if isinstance(grapes, str):
        grapes = re.split(r'[,;]', grapes)
    if not isinstance(grapes, list) or len(grapes) > 20:
        raise ValueError('Сорта винограда: список до 20 наименований')
    grape_names = {'malbec': 'Мальбек', 'merlot': 'Мерло', 'chardonnay': 'Шардоне', 'garnacha': 'Гарнача',
                   'гренаш': 'Гарнача', 'гренаш/гарнача': 'Гарнача', 'гарнача/гренаш': 'Гарнача',
                   'grenache/garnacha': 'Гарнача', 'garnacha/grenache': 'Гарнача',
                   'grenache': 'Гарнача', 'sauvignon blanc': 'Совиньон Блан', 'cabernet sauvignon': 'Каберне Совиньон',
                   'pinot noir': 'Пино Нуар', 'syrah': 'Сира', 'saperavi': 'Саперави', 'riesling': 'Рислинг'}
    grapes = list(dict.fromkeys(grape_names.get(_text(value, 100, required=True).casefold(), _text(value, 100)) for value in grapes))
    provenance = fields.get('metadata_provenance') or {}
    if not isinstance(provenance, dict) or len(dumps(provenance)) > 12000:
        raise ValueError('Некорректные источники сведений о вине')
    country = _text(fields.get('country'), 150)
    country = {'argentina': 'Аргентина', 'chile': 'Чили', 'spain': 'Испания', 'france': 'Франция',
               'italy': 'Италия', 'georgia': 'Грузия', 'russia': 'Россия', 'portugal': 'Португалия'}.get(country.casefold(), country)
    region = _text(fields.get('region'), 200)
    region = {'central valley': 'Центральная Долина', 'mendoza': 'Мендоса', 'catalunya': 'Каталония',
              'patagônia': 'Патагония', 'patagonia': 'Патагония'}.get(region.casefold(), region)
    return {'country': country, 'region': region,
            'grapes': grapes, 'metadata_provenance': provenance}


def _attributes_from_notes(notes):
    values = {}
    for key, label in (('country', 'Страна'), ('region', 'Регион'), ('grapes', 'Сорта')):
        match = re.search(r'(?:^|;\s*)'+label+r':\s*([^;]+)', notes or '')
        if match:
            values[key] = match.group(1).strip()
    if values:
        values['metadata_provenance'] = {'origin': 'legacy_notes'}
    return _attributes(values)


def _save_attributes(conn, owner, lot_id, values):
    conn.execute('''INSERT INTO wine_attributes(lot_id,user_id,country,region,grapes,provenance)
        VALUES(?,?,?,?,?,?) ON CONFLICT(lot_id) DO NOTHING''',
        (lot_id, owner, values['country'], values['region'], dumps(values['grapes']), dumps(values['metadata_provenance'])))


def attribute_statistics(store, actor, owner=None):
    owner = require_owner(store, actor, owner)
    rows = purchase_history(store, actor, owner=owner)
    result = {'purchased': sum(row['purchased_quantity'] for row in rows),
              'remaining': sum(row['quantity'] for row in rows), 'groups': {}}
    for key in ('country', 'region', 'grapes', 'vintage'):
        groups = {}
        for row in rows:
            labels = row[key] if key == 'grapes' else [row.get(key)]
            for label in labels or ['Не указано']:
                label = str(label or 'Не указано')
                group = groups.setdefault(label, {'purchased': 0, 'remaining': 0, 'consumed': 0})
                for field, count in (('purchased', row['purchased_quantity']), ('remaining', row['quantity']), ('consumed', row['consumed_quantity'])):
                    group[field] += count
        result['groups'][key] = dict(sorted(groups.items(), key=lambda item: (-item[1]['purchased'], item[0])))
    return result


def _lot(conn, owner, lot_id):
    _integer(lot_id, 1, 2**63-1, 'Вино')
    row = conn.execute('SELECT * FROM wine_lots WHERE id=? AND user_id=?', (lot_id, owner)).fetchone()
    if not row:
        raise ValueError('Вино не найдено в вашей винотеке')
    result = dict(row)
    # During a v1 migration the attributes table does not exist yet.
    present = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='wine_attributes'").fetchone()
    attributes = conn.execute('SELECT * FROM wine_attributes WHERE lot_id=? AND user_id=?', (lot_id, owner)).fetchone() if present else None
    result.update(country=attributes['country'] if attributes else '', region=attributes['region'] if attributes else '',
                  grapes=json.loads(attributes['grapes']) if attributes else [],
                  metadata_provenance=json.loads(attributes['provenance']) if attributes else {})
    result['purchase_price'] = result.pop('purchase_cents')
    if result['purchase_price'] is not None:
        result['purchase_price'] /= 100
    price = conn.execute('SELECT * FROM wine_prices WHERE lot_id=? AND price_date<=? ORDER BY price_date DESC,id DESC LIMIT 1',
                         (lot_id, _today().isoformat())).fetchone()
    result['latest_price'] = _price(price) if price else None
    result['current_value'] = round(result['quantity']*result['latest_price']['amount'], 2) if price else None
    return result


def _price(row):
    result = dict(row)
    result['amount'] = result.pop('amount_cents')/100
    result['currency'] = 'RUB'
    result['metadata'] = json.loads(result['metadata'])
    return result


def add_bottle(store, actor, fields, operation_id=None):
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        return add_bottle_in_transaction(store, actor, fields, conn, operation_id)


def add_bottle_in_transaction(store, actor, fields, conn, operation_id=None):
    """Add a lot within a caller-owned transaction; never commit partial receipt work."""
    if not conn.in_transaction:
        raise RuntimeError('Wine addition requires an active transaction')
    if not isinstance(fields, dict):
        raise ValueError('Нужны сведения о вине')  # noqa: TRY004 - user form validation.
    if fields.get('currency', 'RUB') != 'RUB':
        raise ValueError('Пока поддерживаются цены только в RUB (рублях)')
    name = _text(fields.get('name'), required=True)
    winery = _text(fields.get('winery', fields.get('producer', '')))
    vintage = _vintage(fields.get('vintage'))
    quantity = _integer(fields.get('quantity', 1), 1, 100_000, 'Количество бутылок')
    volume = _integer(fields.get('volume_ml', 750), 1, 30_000, 'Объём, мл')
    cents = _cents(fields.get('purchase_price'), optional=True)
    purchased = _date(fields['purchase_date']) if fields.get('purchase_date') not in {None, ''} else None
    location = _text(fields.get('location'), 200)
    notes = _text(fields.get('notes'), 2000)
    attributes = _attributes(fields)
    operation = _operation(operation_id)
    owner = require_owner(store, actor, write=True)
    existing = conn.execute('SELECT id FROM wine_lots WHERE user_id=? AND operation_id=?', (owner, operation)).fetchone()
    if existing:
        lot = _lot(conn, owner, existing['id'])
        initial = conn.execute('SELECT delta FROM wine_events WHERE user_id=? AND operation_id=?', (owner, operation)).fetchone()
        if any(lot[field] != expected for field, expected in {
                'name': name, 'winery': winery, 'vintage': vintage, 'volume_ml': volume,
                'purchase_price': None if cents is None else cents/100, 'purchase_date': purchased,
                'location': location, 'notes': notes, **attributes}.items()) or not initial or initial['delta'] != quantity:
            raise ValueError('Идентификатор операции уже использован для другого вина')
        return lot
    if conn.execute('SELECT 1 FROM wine_events WHERE user_id=? AND operation_id=?', (owner, operation)).fetchone():
        raise ValueError('Идентификатор операции уже использован')
    now = utcnow()
    cursor = conn.execute('''INSERT INTO wine_lots(user_id,name,winery,vintage,quantity,volume_ml,purchase_cents,
        purchase_date,location,notes,operation_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (owner, name, winery, vintage, quantity, volume, cents, purchased, location, notes, operation, now, now))
    identity = cursor.lastrowid
    _save_attributes(conn, owner, identity, attributes)
    conn.execute('INSERT INTO wine_events(user_id,lot_id,delta,reason,flow_cents,operation_id,created_at) VALUES(?,?,?,?,?,?,?)',
                 (owner, identity, quantity, 'purchase', None if cents is None else quantity*cents, operation, now))
    if cents is not None:
        conn.execute('''INSERT INTO wine_prices(user_id,lot_id,amount_cents,price_date,source,operation_id,retrieved_at,metadata)
                     VALUES(?,?,?,?,?,?,?,?)''',
                     (owner, identity, cents, purchased or _today().isoformat(),
                      'purchase' if purchased else 'purchase_undated', f'purchase:{operation}', now,
                      dumps({'purchase_date_unknown': not bool(purchased)})))
    _snapshot(conn, owner, 'purchase')
    return _lot(conn, owner, identity)


def lots(store, actor, owner=None, include_empty=False):
    owner = require_owner(store, actor, owner)
    with store.connection() as conn:
        rows = conn.execute('SELECT id FROM wine_lots WHERE user_id=? '+('' if include_empty else 'AND quantity>0 ')+
                            'ORDER BY name COLLATE NOCASE,winery,id', (owner,)).fetchall()
        return [_lot(conn, owner, row['id']) for row in rows]


def get_lot(store, actor, lot_id, owner=None):
    owner = require_owner(store, actor, owner)
    with store.connection() as conn:
        return _lot(conn, owner, lot_id)


DETAIL_FIELDS = ('name','winery','vintage','volume_ml','country','region','grapes')


def details_signature(lot):
    return hashlib.sha256(dumps({key:lot.get(key) for key in DETAIL_FIELDS}).encode()).hexdigest()


def update_details(store, actor, lot_id, fields, expected):
    """Edit a saved lot's identity atomically, without changing purchase or stock."""
    require_owner(store,actor,write=True)
    if not isinstance(fields,dict) or set(fields)!=set(DETAIL_FIELDS):
        raise ValueError('Нужны характеристики вина')
    normalized = {**fields,**_attributes(fields)}
    normalized.update(name=_text(fields['name'],required=True),winery=_text(fields['winery']),
                      vintage=_vintage(fields['vintage']),volume_ml=_integer(fields['volume_ml'],20,30000,'Объём'))
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        old = _lot(conn,actor,lot_id)
        if details_signature(old)!=expected:
            raise ValueError('Характеристики уже изменились. Откройте карточку заново.')
        changes = {key:{'before':old[key],'after':normalized[key]} for key in DETAIL_FIELDS if old[key]!=normalized[key]}
        if not changes:
            return old
        provenance = dict(old['metadata_provenance'])
        for key in changes:
            provenance[key] = [{'origin':'user_confirmed','confirmed_at':utcnow()}]
        conn.execute('UPDATE wine_lots SET name=?,winery=?,vintage=?,volume_ml=?,updated_at=? WHERE id=? AND user_id=?',
                     (normalized['name'],normalized['winery'],normalized['vintage'],normalized['volume_ml'],utcnow(),lot_id,actor))
        conn.execute('''INSERT INTO wine_attributes(lot_id,user_id,country,region,grapes,provenance) VALUES(?,?,?,?,?,?)
            ON CONFLICT(lot_id) DO UPDATE SET country=excluded.country,region=excluded.region,
            grapes=excluded.grapes,provenance=excluded.provenance''',
            (lot_id,actor,normalized['country'],normalized['region'],dumps(normalized['grapes']),dumps(provenance)))
        result = _lot(conn,actor,lot_id)
    from .monitoring import safe_record
    safe_record(store,'wine','details_updated',actor=actor,subject=actor,entity_type='wine_lots',entity_id=lot_id,details={'changes':changes})
    return result


def _adjust(conn, owner, lot_id, delta, reason, operation, consumption=None):
    previous = conn.execute('SELECT * FROM wine_events WHERE user_id=? AND operation_id=?', (owner, operation)).fetchone()
    if previous:
        if previous['lot_id'] != lot_id or previous['delta'] != delta or previous['reason'] != reason:
            raise ValueError('Идентификатор операции уже использован для другого действия')
        if consumption is not None:
            saved = conn.execute('SELECT * FROM wine_consumptions WHERE event_id=?', (previous['id'],)).fetchone()
            day, quote = consumption
            if not saved or any(saved[key] != value for key, value in {
                    'consumed_on': day, 'valuation_cents': quote.get('amount_cents'),
                    'price_id': quote.get('id')}.items()):
                raise ValueError('Идентификатор списания уже использован с другой датой или ценой')
        return _lot(conn, owner, lot_id)
    lot = _lot(conn, owner, lot_id)
    if not 0 <= lot['quantity']+delta <= 100_000:
        raise ValueError('Недостаточно бутылок или превышен лимит количества')
    cents = _cents(lot['purchase_price'], optional=True)
    flow = delta*cents if cents is not None else None
    now = utcnow()
    event = conn.execute('INSERT INTO wine_events(user_id,lot_id,delta,reason,flow_cents,operation_id,created_at) VALUES(?,?,?,?,?,?,?)',
                         (owner, lot_id, delta, reason, flow, operation, now))
    if delta < 0 and reason in CONSUMPTION_REASONS:
        day, quote = consumption or (_today().isoformat(), _consumption_quote(conn, owner, lot_id, _today().isoformat()))
        _freeze_consumption(conn, owner, event.lastrowid, lot, -delta, day, quote, now)
    conn.execute('UPDATE wine_lots SET quantity=quantity+?,updated_at=? WHERE id=? AND user_id=?', (delta, now, lot_id, owner))
    _snapshot(conn, owner, reason)
    return _lot(conn, owner, lot_id)


def adjust_stock(store, actor, lot_id, delta, reason='adjustment', operation_id=None):
    _integer(delta, -100_000, 100_000, 'Изменение количества')
    if delta == 0:
        raise ValueError('Количество должно измениться')
    reason = _text(reason, maximum=100, required=True)
    operation = _operation(operation_id)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner = require_owner(store, actor, write=True)
        return _adjust(conn, owner, lot_id, delta, reason, operation)


def _consumption_quote(conn, owner, lot_id, day, *, price=None, price_id=None, known_at=None):
    if price is not None and price_id is not None:
        raise ValueError('Выберите ручную цену или сохранённую оценку')
    if price is not None:
        return {'amount_cents': _cents(price), 'price_date': day, 'source': 'manual_consumption'}
    sql = "SELECT * FROM wine_prices WHERE user_id=? AND lot_id=? AND price_date<=? AND source NOT LIKE 'purchase%'"
    params = [owner, lot_id, day]
    if price_id is not None:
        _integer(price_id, 1, 2**63-1, 'Цена')
        sql += ' AND id=?'
        params.append(price_id)
    if known_at:
        sql += ' AND retrieved_at<=?'
        params.append(known_at)
    quote = conn.execute(sql+' ORDER BY price_date DESC,id DESC LIMIT 1', params).fetchone()
    if price_id is not None and not quote:
        raise ValueError('Сохранённая оценка не подходит к этому вину или дате списания')
    return dict(quote) if quote else {}


def _freeze_consumption(conn, owner, event_id, lot, quantity, day, quote, now):
    conn.execute('''INSERT INTO wine_consumptions(user_id,event_id,lot_id,quantity,consumed_on,purchase_cents,
        valuation_cents,price_date,price_source,price_url,price_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
        (owner, event_id, lot['id'], quantity, day, _cents(lot['purchase_price'], optional=True),
         quote.get('amount_cents'), quote.get('price_date'), quote.get('source'), quote.get('url'), quote.get('id'), now))


def consumption_preview(store, actor, lot_id, quantity=1, consumed_on=None, *, price=None, price_id=None, unpriced=False):
    owner = require_owner(store, actor, write=True)
    quantity = _integer(quantity, 1, 100_000, 'Количество выпитых бутылок')
    day = _date(consumed_on)
    with store.connection() as conn:
        lot = _lot(conn, owner, lot_id)
        if quantity > lot['quantity']:
            raise ValueError('В запасе меньше бутылок')
        if lot.get('purchase_date') and day < lot['purchase_date']:
            raise ValueError('Дата списания не может быть раньше покупки')
        if type(unpriced) is not bool or unpriced and (price is not None or price_id is not None):
            raise ValueError('Неверный режим оценки списания')
        quote = {} if unpriced else _consumption_quote(conn, owner, lot_id, day, price=price, price_id=price_id)
        cost = _cents(lot['purchase_price'], optional=True)
        value = quote.get('amount_cents')
        return {'lot_id': lot_id, 'quantity': quantity, 'consumed_on': day,
                'purchase_price': None if cost is None else cost/100,
                'valuation_price': None if value is None else value/100,
                'price_id': quote.get('id'), 'price_date': quote.get('price_date'),
                'price_source': quote.get('source'), 'price_url': quote.get('url'),
                'result': None if cost is None or value is None else quantity*(value-cost)/100}


def consume_bottles(store, actor, lot_id, quantity=1, consumed_on=None, *, price=None, price_id=None, unpriced=False, operation_id=None):
    quantity = _integer(quantity, 1, 100_000, 'Количество выпитых бутылок')
    day, operation = _date(consumed_on), _operation(operation_id)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner = require_owner(store, actor, write=True)
        lot = _lot(conn, owner, lot_id)
        if lot.get('purchase_date') and day < lot['purchase_date']:
            raise ValueError('Дата списания не может быть раньше покупки')
        if type(unpriced) is not bool or unpriced and (price is not None or price_id is not None):
            raise ValueError('Неверный режим оценки списания')
        quote = {} if unpriced else _consumption_quote(conn, owner, lot_id, day, price=price, price_id=price_id)
        return _adjust(conn, owner, lot_id, -quantity, 'consumed', operation, (day, quote))


def consumptions(store, actor, owner=None, lot_id=None):
    owner = require_owner(store, actor, owner)
    if lot_id is not None:
        get_lot(store, actor, lot_id, owner=owner)
    rows = store.query('''SELECT c.*,l.name,l.winery,l.vintage,l.volume_ml FROM wine_consumptions c
        JOIN wine_lots l ON l.id=c.lot_id WHERE c.user_id=? '''+
        ('AND c.lot_id=? ' if lot_id is not None else '')+'ORDER BY c.consumed_on DESC,c.id DESC',
        (owner, lot_id) if lot_id is not None else (owner,))
    result = []
    for row in rows:
        item = dict(row)
        cost, value, quantity = item.pop('purchase_cents'), item.pop('valuation_cents'), item['quantity']
        item.update(purchase_price=None if cost is None else cost/100, valuation_price=None if value is None else value/100,
                    result=None if cost is None or value is None else quantity*(value-cost)/100,
                    result_pct=None if not cost or value is None else round((value/cost-1)*100, 2))
        result.append(item)
    return result


def purchase_history(store, actor, owner=None):
    owner = require_owner(store, actor, owner)
    incoming = {row['lot_id']: row for row in store.query('''SELECT lot_id,SUM(delta) AS quantity,
        SUM(flow_cents) AS cost,SUM(CASE WHEN flow_cents IS NULL THEN delta ELSE 0 END) AS unknown
        FROM wine_events WHERE user_id=? AND delta>0 GROUP BY lot_id''', (owner,))}
    consumed = {}
    for item in consumptions(store, actor, owner=owner):
        totals = consumed.setdefault(item['lot_id'], {'quantity': 0, 'result': 0, 'unknown': 0})
        totals['quantity'] += item['quantity']
        if item['result'] is None:
            totals['unknown'] += item['quantity']
        else:
            totals['result'] += item['result']
    result = []
    for lot in lots(store, actor, owner=owner, include_empty=True):
        bought = incoming.get(lot['id'], {})
        closed = consumed.get(lot['id'], {'quantity': 0, 'result': 0, 'unknown': 0})
        lot.update(purchased_quantity=bought.get('quantity', 0), consumed_quantity=closed['quantity'],
                   total_purchase_cost=None if bought.get('unknown') else (bought.get('cost') or 0)/100,
                   known_closed_result=round(closed['result'], 2), closed_unknown_quantity=closed['unknown'],
                   closed_result=None if closed['unknown'] else round(closed['result'], 2))
        result.append(lot)
    return result


def record_price(store, actor, lot_id, amount, price_date=None, source='manual', url=None, operation_id=None, **kwargs):
    if kwargs.get('currency', 'RUB') != 'RUB':
        raise ValueError('Пока поддерживаются цены только в RUB (рублях)')
    cents, priced = _cents(amount), _date(price_date)
    if kwargs.get('available', True) is not True:
        raise ValueError('Цена недоступного товара не обновляет стоимость винотеки')
    source = _text(source, maximum=120, required=True)
    url = validate_price_url(url) if url else None
    operation = _operation(operation_id)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner = require_owner(store, actor, write=True)
        lot = _lot(conn, owner, lot_id)
        if kwargs.get('vintage') not in {None, ''} and _vintage(kwargs['vintage']) != lot['vintage']:
            raise ValueError('Год вина в предложении магазина не совпадает с вашей бутылкой')
        if kwargs.get('volume_ml') is not None and kwargs['volume_ml'] != lot['volume_ml']:
            raise ValueError('Объём вина в предложении магазина не совпадает с вашей бутылкой')
        metadata = _quote_metadata(kwargs.get('metadata', {}))
        metadata.update(_quote_metadata({name: kwargs[name] for name in (
            'title', 'vintage', 'volume_ml', 'observed_at', 'availability', 'currency', 'price_status') if name in kwargs}))
        if metadata.get('currency', 'RUB') != 'RUB':
            raise ValueError('Цена магазина должна быть указана в RUB')
        if metadata.get('availability', 'InStock') not in {'InStock', 'LimitedAvailability'}:
            raise ValueError('Недоступное предложение магазина не обновляет стоимость')
        previous = conn.execute('SELECT * FROM wine_prices WHERE user_id=? AND operation_id=?', (owner, operation)).fetchone()
        if previous:
            if previous['lot_id'] != lot_id or previous['amount_cents'] != cents or previous['price_date'] != priced:
                raise ValueError('Идентификатор операции уже использован для другой цены')
            return _price(previous)
        cursor = conn.execute('''INSERT INTO wine_prices(user_id,lot_id,amount_cents,price_date,source,url,operation_id,retrieved_at,metadata)
                       VALUES(?,?,?,?,?,?,?,?,?)''', (owner, lot_id, cents, priced, source, url, operation, utcnow(), dumps(metadata)))
        _snapshot(conn, owner, 'price_update')
        return _price(conn.execute('SELECT * FROM wine_prices WHERE id=?', (cursor.lastrowid,)).fetchone())


def _quote_metadata(value):
    if not isinstance(value, dict):
        raise ValueError('Некорректные сведения об источнике цены')  # noqa: TRY004 - external data validation.
    permitted = {'title', 'name', 'winery', 'vintage', 'volume_ml', 'amount', 'currency', 'availability',
                 'as_of', 'url', 'source', 'price_valid_until', 'method', 'status', 'warnings', 'selection',
                 'checked_offer_count', 'observed_at', 'price_status', 'purchase_date_unknown',
                 'shop_id', 'shop_name', 'shop_title', 'city', 'price_kind', 'availability_scope',
                 'available_shops', 'shop_address', 'selected_shop_address'}
    result = {name: value[name] for name in permitted & value.keys()}
    if 'candidate_offers' in value:
        offers = value['candidate_offers']
        if not isinstance(offers, list) or len(offers) > 100 or any(not isinstance(row, dict) for row in offers):
            raise ValueError('Некорректный список предложений магазинов')
        result['candidate_offers'] = [{name: row[name] for name in permitted & row.keys()} for row in offers]
    try:
        encoded = dumps(result)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Некорректные сведения об источнике цены') from None
    if len(encoded.encode('utf-8')) > 32_768:
        raise ValueError('Сведения об источнике цены слишком большие')
    return result


def add_price_link(store, actor, lot_id, url):
    from .wine_retail import allowed_hosts
    from .wine_tools import validate_product_url
    url = validate_product_url(url, allowed_hosts=allowed_hosts(store))
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner = require_owner(store, actor, write=True)
        _lot(conn, owner, lot_id)
        conn.execute('INSERT OR IGNORE INTO wine_price_links(user_id,lot_id,url,created_at) VALUES(?,?,?,?)',
                     (owner, lot_id, url, utcnow()))
        return dict(conn.execute('SELECT * FROM wine_price_links WHERE lot_id=? AND url=?', (lot_id, url)).fetchone())


def price_links(store, actor, lot_id=None, owner=None):
    owner = require_owner(store, actor, owner)
    if lot_id is not None:
        get_lot(store, actor, lot_id, owner=owner)
    return store.query('SELECT * FROM wine_price_links WHERE user_id=? '+('AND lot_id=? ' if lot_id is not None else '')+
                       'ORDER BY id', (owner, lot_id) if lot_id is not None else (owner,))


def _positions(conn, owner):
    positions = []
    for row in conn.execute('SELECT id FROM wine_lots WHERE user_id=? AND quantity>0 ORDER BY id', (owner,)).fetchall():
        lot = _lot(conn, owner, row['id'])
        quote = lot['latest_price']
        positions.append({'lot_id': lot['id'], 'quantity': lot['quantity'],
                          'price_cents': _cents(quote['amount']) if quote else None,
                          'price_date': quote['price_date'] if quote else None,
                          'price_url': quote.get('url') if quote else None,
                          'price_source': quote.get('source') if quote else None,
                          'purchase_cents': _cents(lot['purchase_price'], optional=True)})
    return positions


def _snapshot(conn, owner, reason):
    positions = _positions(conn, owner)
    quantity = sum(p['quantity'] for p in positions)
    priced = sum(p['quantity'] for p in positions if p['price_cents'] is not None)
    value = sum(p['quantity']*p['price_cents'] for p in positions if p['price_cents'] is not None)
    cost = sum(p['quantity']*p['purchase_cents'] for p in positions if p['purchase_cents'] is not None)
    unknown = sum(p['quantity'] for p in positions if p['purchase_cents'] is None)
    flows = conn.execute('SELECT COALESCE(SUM(flow_cents),0) AS total,SUM(flow_cents IS NULL) AS missing FROM wine_events WHERE user_id=?',
                         (owner,)).fetchone()
    cursor = conn.execute('''INSERT INTO wine_snapshots(user_id,created_at,reason,quantity,priced_quantity,value_cents,
        cost_cents,unknown_cost_quantity,net_flow_cents,unknown_flow_events,positions) VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
        (owner, utcnow(), reason, quantity, priced, value, cost, unknown, flows['total'], flows['missing'] or 0, dumps(positions)))
    return _snapshot_dict(conn.execute('SELECT * FROM wine_snapshots WHERE id=?', (cursor.lastrowid,)).fetchone())


def _snapshot_dict(row):
    result = dict(row)
    result['positions'] = json.loads(result['positions'])
    result['known_current_value'] = result.pop('value_cents')/100
    result['known_purchase_cost'] = result.pop('cost_cents')/100
    result['known_net_flows'] = result.pop('net_flow_cents')/100
    result['current_value'] = result['known_current_value'] if result['quantity'] == result['priced_quantity'] else None
    result['purchase_cost'] = result['known_purchase_cost'] if not result['unknown_cost_quantity'] else None
    result['net_flows'] = result['known_net_flows'] if not result['unknown_flow_events'] else None
    result['unpriced_quantity'] = result['quantity']-result['priced_quantity']
    result['currency'] = 'RUB'
    return result


def snapshot(store, actor, reason='manual'):
    reason = _text(reason, maximum=100, required=True)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner = require_owner(store, actor, write=True)
        return _snapshot(conn, owner, reason)


def portfolio(store, actor, owner=None):
    owner = require_owner(store, actor, owner)
    with store.connection() as conn:
        positions = _positions(conn, owner)
        current = {
            'quantity': sum(p['quantity'] for p in positions), 'lot_count': len(positions), 'currency': 'RUB',
            'priced_quantity': sum(p['quantity'] for p in positions if p['price_cents'] is not None),
            'known_current_value': sum(p['quantity']*p['price_cents'] for p in positions if p['price_cents'] is not None)/100,
            'known_purchase_cost': sum(p['quantity']*p['purchase_cents'] for p in positions if p['purchase_cents'] is not None)/100,
            'unknown_cost_quantity': sum(p['quantity'] for p in positions if p['purchase_cents'] is None),
            'positions': positions,
        }
        current['unpriced_quantity'] = current['quantity']-current['priced_quantity']
        current['current_value'] = current['known_current_value'] if not current['unpriced_quantity'] else None
        current['purchase_cost'] = current['known_purchase_cost'] if not current['unknown_cost_quantity'] else None
        current['price_change'] = (round(current['current_value']-current['purchase_cost'], 2)
                                   if current['current_value'] is not None and current['purchase_cost'] is not None else None)
        comparable = [p for p in positions if p['price_cents'] is not None and p['purchase_cents'] is not None
                      and p['price_url'] and not str(p['price_source']).startswith('purchase')]
        shop_cost = sum(p['quantity']*p['purchase_cents'] for p in comparable)
        shop_value = sum(p['quantity']*p['price_cents'] for p in comparable)
        current.update(shop_compared_quantity=sum(p['quantity'] for p in comparable),
                       shop_price_change=round((shop_value-shop_cost)/100, 2) if comparable else None,
                       shop_price_change_pct=round((shop_value/shop_cost-1)*100, 2) if shop_cost else None,
                       shop_price_dates=sorted({p['price_date'] for p in comparable if p['price_date']}))
        comparable_open = [p for p in positions if p['price_cents'] is not None and p['purchase_cents'] is not None
                           and not str(p['price_source']).startswith('purchase')]
        open_quantity = sum(p['quantity'] for p in comparable_open)
        open_cost = sum(p['quantity']*p['purchase_cents'] for p in comparable_open)
        open_result = sum(p['quantity']*(p['price_cents']-p['purchase_cents']) for p in comparable_open)
        current.update(open_compared_quantity=open_quantity, open_unknown_quantity=current['quantity']-open_quantity,
                       known_open_result=open_result/100,
                       open_result=open_result/100 if open_quantity == current['quantity'] else None,
                       open_result_pct=round(open_result/open_cost*100, 2) if open_cost else None)
        closed = consumptions(store, actor, owner=owner)
        closed_unknown = sum(c['quantity'] for c in closed if c['result'] is None)
        closed_result = round(sum(c['result'] for c in closed if c['result'] is not None), 2)
        bought = conn.execute('''SELECT COALESCE(SUM(delta),0) AS quantity,COALESCE(SUM(flow_cents),0) AS cost,
            COALESCE(SUM(CASE WHEN flow_cents IS NULL THEN delta ELSE 0 END),0) AS unknown
            FROM wine_events WHERE user_id=? AND delta>0''', (owner,)).fetchone()
        current.update(purchased_quantity=bought['quantity'], known_total_purchase_cost=bought['cost']/100,
                       total_purchase_cost=None if bought['unknown'] else bought['cost']/100,
                       consumed_quantity=sum(c['quantity'] for c in closed), closed_unknown_quantity=closed_unknown,
                       known_closed_result=closed_result, closed_result=None if closed_unknown else closed_result,
                       known_combined_result=round(open_result/100+closed_result, 2),
                       combined_result=None if closed_unknown or current['open_unknown_quantity'] else round(open_result/100+closed_result, 2))
        current['snapshots'] = [_snapshot_dict(row) for row in conn.execute('SELECT * FROM wine_snapshots WHERE user_id=? ORDER BY id', (owner,))]
        flows = conn.execute('SELECT COALESCE(SUM(flow_cents),0) AS total,SUM(flow_cents IS NULL) AS missing FROM wine_events WHERE user_id=?', (owner,)).fetchone()
        current['known_net_flows'] = flows['total']/100
        current['net_flows'] = current['known_net_flows'] if not flows['missing'] else None
        current['events'] = [dict(row) for row in conn.execute('SELECT id,lot_id,delta,reason,created_at,flow_cents FROM wine_events WHERE user_id=? ORDER BY id', (owner,))]
        for event in current['events']:
            event['flow'] = event.pop('flow_cents')
            if event['flow'] is not None:
                event['flow'] /= 100
        # Revalue only positions common to consecutive snapshots: stock purchases
        # and consumption cannot masquerade as appreciation of the collection.
        current['price_movement'] = []
        for before, after in zip(current['snapshots'], current['snapshots'][1:]):
            old = {p['lot_id']: p for p in before['positions']}
            movement, compared, unknown = 0, 0, 0
            for new in after['positions']:
                if new['lot_id'] not in old:
                    continue
                previous = old[new['lot_id']]
                common = min(previous['quantity'], new['quantity'])
                if previous['price_cents'] is None or new['price_cents'] is None:
                    unknown += common
                else:
                    compared += common
                    movement += common*(new['price_cents']-previous['price_cents'])
            current['price_movement'].append({'from_id': before['id'], 'to_id': after['id'], 'amount': movement/100,
                                              'compared_quantity': compared, 'unknown_quantity': unknown})
        return current


def _normalized(value):
    return ' '.join(re.sub(r'[^\w\s]', ' ', unicodedata.normalize('NFKC', str(value or '')).casefold()).split())


def _identity(row):
    return (_normalized(row.get('name')), _normalized(row.get('winery')),
            _normalized(_vintage(row.get('vintage'))))


def _review_key(row):
    if not isinstance(row, dict) or not row.get('name') or not (row.get('review') or row.get('rating') is not None):
        return None
    try:
        identity = _identity(row)
    except ValueError:
        return None
    return hashlib.sha256(dumps(identity).encode('utf-8')).hexdigest()


def _baseline(conn, owner, rows, generation):
    now = utcnow()
    conn.execute('INSERT OR IGNORE INTO wine_review_state(user_id,generation,initialized_at) VALUES(?,?,?)', (owner, generation, now))
    for row in rows:
        key = _review_key(row)
        if key:
            conn.execute('INSERT OR IGNORE INTO wine_seen_reviews(user_id,review_key,first_generation,created_at) VALUES(?,?,?,?)',
                         (owner, key, generation, now))


def _proposal(row):
    result = dict(row)
    result['lot_ids'] = json.loads(result['lot_ids'])
    return result


def sync_reviews(store, actor, rows, generation):
    if not isinstance(rows, list) or len(rows) > 3000:
        raise ValueError('Ожидается до 3000 записей Vivino')
    generation = _text(generation, 160, required=True)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner = require_owner(store, actor, write=True)
        if not conn.execute('SELECT 1 FROM wine_review_state WHERE user_id=?', (owner,)).fetchone():
            _baseline(conn, owner, rows, generation)
            return []
        stock = [dict(row) for row in conn.execute('SELECT * FROM wine_lots WHERE user_id=? AND quantity>0', (owner,))]
        for row in rows:
            key = _review_key(row)
            if not key:
                continue
            inserted = conn.execute('INSERT OR IGNORE INTO wine_seen_reviews(user_id,review_key,first_generation,created_at) VALUES(?,?,?,?)',
                                    (owner, key, generation, utcnow())).rowcount
            if not inserted:
                continue
            matches = [lot['id'] for lot in stock if _identity(lot) == _identity(row)]
            if matches:
                conn.execute('''INSERT OR IGNORE INTO wine_review_proposals(user_id,review_key,review_name,winery,vintage,
                    lot_ids,generation,created_at) VALUES(?,?,?,?,?,?,?,?)''',
                    (owner, key, _text(row['name']), _text(row.get('winery')), _vintage(row.get('vintage')), dumps(matches), generation, utcnow()))
        conn.execute('UPDATE wine_review_state SET generation=? WHERE user_id=?', (generation, owner))
        return [_proposal(row) for row in conn.execute("SELECT * FROM wine_review_proposals WHERE user_id=? AND state='pending' ORDER BY id", (owner,))]


def proposals(store, actor):
    owner = require_owner(store, actor)
    return [_proposal(row) for row in store.query("SELECT * FROM wine_review_proposals WHERE user_id=? AND state='pending' ORDER BY id", (owner,))]


def resolve_proposal(store, actor, proposal_id, consume, operation_id=None, *, consumed_on=None, price=None, price_id=None, unpriced=False):
    _integer(proposal_id, 1, 2**63-1, 'Предложение')
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner = require_owner(store, actor, write=True)
        row = conn.execute('SELECT * FROM wine_review_proposals WHERE id=? AND user_id=?', (proposal_id, owner)).fetchone()
        if not row:
            raise ValueError('Предложение не найдено')
        if row['state'] != 'pending':
            return _proposal(row)
        candidate_ids = json.loads(row['lot_ids'])
        if consume is False:
            state, chosen = 'dismissed', None
        else:
            if consume is True:
                if len(candidate_ids) != 1:
                    raise ValueError('Найдено несколько партий. Выберите конкретное вино для списания')
                chosen = candidate_ids[0]
            elif type(consume) is int and consume in candidate_ids:
                chosen = consume
            else:
                raise ValueError('Выберите подходящую партию или оставьте бутылки в винотеке')
            lot = _lot(conn, owner, chosen)
            day = _date(consumed_on)
            if lot.get('purchase_date') and day < lot['purchase_date']:
                raise ValueError('Дата списания не может быть раньше покупки')
            if type(unpriced) is not bool or unpriced and (price is not None or price_id is not None):
                raise ValueError('Неверный режим оценки списания')
            quote = {} if unpriced else _consumption_quote(conn, owner, chosen, day, price=price, price_id=price_id)
            _adjust(conn, owner, chosen, -1, 'vivino_confirmed', _operation(operation_id or f'vivino-review:{proposal_id}'), (day, quote))
            state = 'consumed'
        conn.execute('UPDATE wine_review_proposals SET state=?,selected_lot_id=?,resolved_at=? WHERE id=?',
                     (state, chosen, utcnow(), proposal_id))
        return _proposal(conn.execute('SELECT * FROM wine_review_proposals WHERE id=?', (proposal_id,)).fetchone())


def export_rows(store, actor, owner=None):
    return [{'name': lot['name'], 'winery': lot['winery'], 'vintage': lot['vintage'], 'quantity': lot['quantity'],
             'country': lot['country'], 'region': lot['region'], 'grapes': ', '.join(lot['grapes']),
             'metadata_sources': dumps(lot['metadata_provenance']) if lot['metadata_provenance'] else '',
             'volume_ml': lot['volume_ml'], 'purchase_price_rub': lot['purchase_price'], 'purchase_date': lot['purchase_date'],
             'latest_price_rub': lot['latest_price']['amount'] if lot['latest_price'] else None,
             'price_date': lot['latest_price']['price_date'] if lot['latest_price'] else None,
             'price_source': lot['latest_price']['source'] if lot['latest_price'] else None,
             'price_url': lot['latest_price']['url'] if lot['latest_price'] else None,
             'value_rub': lot['current_value'], 'location': lot['location'], 'notes': lot['notes'], 'currency': 'RUB',
             'purchased_quantity': lot['purchased_quantity'], 'consumed_quantity': lot['consumed_quantity'],
             'closed_result': lot['closed_result'], 'closed_unknown_quantity': lot['closed_unknown_quantity']}
            for lot in purchase_history(store, actor, owner=owner)]


def owners(store, actor):
    require_owner(store, actor)
    if not store.authorize(actor, admin=True):
        raise PermissionError('Список винотек доступен администратору')
    return store.query('''SELECT user_id,COALESCE(SUM(quantity),0) AS quantity,
                         SUM(quantity>0) AS lot_count FROM wine_lots GROUP BY user_id ORDER BY user_id''')


def valuation_history(store, actor, owner=None):
    owner = require_owner(store, actor, owner)
    rows = store.query('SELECT * FROM wine_snapshots WHERE user_id=? ORDER BY id', (owner,))
    result = []
    for row in rows:
        item = _snapshot_dict(row)
        item.update(cost=item['purchase_cost'], value=item['current_value'], flows=item['net_flows'])
        result.append(item)
    return result


def daily_result_history(store, actor, owner=None):
    """Use observed snapshots and frozen write-offs; never apply later prices to the past."""
    owner = require_owner(store,actor,owner)
    days = {}
    for row in valuation_history(store,actor,owner=owner):
        day = datetime.fromisoformat(row['created_at']).astimezone(ZoneInfo('Europe/Moscow')).date().isoformat()
        days[day] = row
    closed = consumptions(store,actor,owner=owner)
    result = []
    for day,row in sorted(days.items()):
        positions = row.get('positions',[])
        comparable = [p for p in positions if p.get('price_cents') is not None and p.get('purchase_cents') is not None
                      and p.get('price_source') and not str(p['price_source']).startswith('purchase')]
        unknown = row['quantity']-sum(p['quantity'] for p in comparable)
        opened = None if unknown else sum(p['quantity']*(p['price_cents']-p['purchase_cents']) for p in comparable)/100
        known_closed = [item for item in closed if item['created_at']<=row['created_at']]
        frozen = None if any(item['result'] is None for item in known_closed) else round(sum(item['result'] for item in known_closed),2)
        result.append({'day':day,'created_at':row['created_at'],'open_result':opened,'closed_result':frozen,
                       'combined_result':None if opened is None or frozen is None else round(opened+frozen,2)})
    return result
