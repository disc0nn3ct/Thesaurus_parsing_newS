"""Private income/expenses and explicitly confirmed receipt imports, without I/O.

Receipt files, fiscal identifiers and raw OCR text are never retained here. Money
uses integer kopecks; adding wine stock and the whole receipt expense is atomic.
"""
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from uuid import uuid4

from . import wine_cellar
from .feature_access import require
from .store import dumps, utcnow
from .subscription_access import require as require_section

VERSION = 2
SCHEMA = """
CREATE TABLE life_receipt_imports(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(user_id),
 fingerprint TEXT NOT NULL,entry_id INTEGER REFERENCES life_ledger_entries(id),receipt_date TEXT NOT NULL,
 total_cents INTEGER NOT NULL CHECK(total_cents>0),merchant TEXT NOT NULL,confirmed_at TEXT NOT NULL,
 items TEXT NOT NULL,wine_lot_ids TEXT NOT NULL,rounding_difference_cents INTEGER NOT NULL DEFAULT 0,
 operation_id TEXT NOT NULL,UNIQUE(user_id,fingerprint),UNIQUE(user_id,operation_id));
CREATE TABLE life_ledger_entries(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(user_id),
 kind TEXT NOT NULL CHECK(kind IN ('income','expense')),amount_cents INTEGER NOT NULL CHECK(amount_cents>0),
 date TEXT NOT NULL,category TEXT NOT NULL,note TEXT NOT NULL,merchant TEXT NOT NULL DEFAULT '',
 receipt_id INTEGER UNIQUE REFERENCES life_receipt_imports(id),operation_id TEXT NOT NULL,created_at TEXT NOT NULL,voided_at TEXT,
 UNIQUE(user_id,operation_id));
CREATE INDEX life_ledger_owner_date ON life_ledger_entries(user_id,date,id);
CREATE TABLE life_receipt_drafts(user_id INTEGER NOT NULL REFERENCES users(user_id),token TEXT NOT NULL,
 fingerprint TEXT NOT NULL,result TEXT NOT NULL,status TEXT NOT NULL CHECK(status IN ('draft','confirmed')),
 created_at TEXT NOT NULL,updated_at TEXT NOT NULL,expires_at TEXT NOT NULL,
 receipt_id INTEGER REFERENCES life_receipt_imports(id),PRIMARY KEY(user_id,token));
CREATE INDEX life_receipt_draft_expiry ON life_receipt_drafts(expires_at,status);
"""
VOID_SCHEMA = """
CREATE TABLE IF NOT EXISTS life_ledger_actions(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(user_id),
 entry_id INTEGER NOT NULL REFERENCES life_ledger_entries(id),action TEXT NOT NULL,created_at TEXT NOT NULL,
 UNIQUE(user_id,entry_id,action));
"""


def migrate(store):
    with store.connection() as conn:
        current = conn.execute("SELECT value FROM settings WHERE key='life_ledger_version'").fetchone()
        previous_version = json.loads(current['value']) if current else None
        if current:
            if type(previous_version) is not int or previous_version not in {1, VERSION}:
                raise RuntimeError('Unsupported life ledger schema')
            if previous_version == VERSION:
                return
        backup = store.path.with_name(store.path.name+f'.life-backup-{uuid4().hex}.sqlite3')
        with closing(sqlite3.connect(backup)) as target:
            conn.backup(target)
        conn.execute('BEGIN IMMEDIATE')
        locked = conn.execute("SELECT value FROM settings WHERE key='life_ledger_version'").fetchone()
        locked_version = json.loads(locked['value']) if locked else None
        if locked_version == VERSION:
            return
        if locked_version not in {None, 1}:
            raise RuntimeError('Unsupported life ledger schema')
        schema = SCHEMA if locked_version is None else 'ALTER TABLE life_ledger_entries ADD COLUMN voided_at TEXT;'
        for statement in (schema+VOID_SCHEMA).split(';'):
            if statement.strip():
                conn.execute(statement)
        conn.execute("INSERT INTO settings(key,value) VALUES('life_ledger_version',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (dumps(VERSION),))


def require_owner(store, actor, owner=None, write=False):
    if type(actor) is not int or actor <= 0:
        raise PermissionError('Нет доступа')
    require_section(store, actor, 'life')
    require(store, actor, 'life_finance')
    owner = actor if owner is None else owner
    if type(owner) is not int or owner <= 0:
        raise ValueError('Неверный пользователь')
    if owner != actor and (write or not store.authorize(actor, admin=True)):
        raise PermissionError('Можно изменять только свои доходы и расходы')
    if not store.one('SELECT 1 FROM users WHERE user_id=?', (owner,)):
        raise ValueError('Пользователь не найден')
    return owner


def _amount(value, optional=False):
    cents = wine_cellar._cents(value, optional=optional)
    if cents is not None and cents <= 0:
        raise ValueError('Сумма должна быть больше нуля')
    return cents


def _kind(value):
    if not isinstance(value, str) or value not in {'income', 'expense'}:
        raise ValueError('Выберите доход или расход')
    return value


def _entry(row):
    result = dict(row)
    result['amount'] = result['amount_cents']/100
    result['currency'] = 'RUB'
    result['voided'] = bool(result.get('voided_at'))
    return result


def _add_entry(conn, owner, kind, cents, day, category, note, operation, merchant='', receipt_id=None):
    previous = conn.execute('SELECT * FROM life_ledger_entries WHERE user_id=? AND operation_id=?', (owner, operation)).fetchone()
    expected = {'kind': kind, 'amount_cents': cents, 'date': day, 'category': category, 'note': note,
                'merchant': merchant, 'receipt_id': receipt_id}
    if previous:
        if any(previous[key] != value for key, value in expected.items()):
            raise ValueError('Идентификатор операции уже использован для другой записи')
        return _entry(previous)
    cursor = conn.execute('''INSERT INTO life_ledger_entries(user_id,kind,amount_cents,date,category,note,merchant,
        receipt_id,operation_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)''',
        (owner, kind, cents, day, category, note, merchant, receipt_id, operation, utcnow()))
    return _entry(conn.execute('SELECT * FROM life_ledger_entries WHERE id=?', (cursor.lastrowid,)).fetchone())


def add_entry(store, actor, kind, amount, date=None, category='Прочее', note='', operation_id=None):
    kind, cents, day = _kind(kind), _amount(amount), wine_cellar._date(date)
    category = wine_cellar._text(category, 100, required=True)
    note = wine_cellar._text(note, 2000)
    operation = wine_cellar._operation(operation_id)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner = require_owner(store, actor, write=True)
        return _add_entry(conn, owner, kind, cents, day, category, note, operation)


def _filters(owner, start, end, kind=None, include_void=False):
    start = wine_cellar._date(start) if start is not None else None
    end = wine_cellar._date(end) if end is not None else None
    if start and end and start > end:
        raise ValueError('Начало периода позже окончания')
    where, params = ['user_id=?'], [owner]
    if not include_void:
        where.append('voided_at IS NULL')
    if start:
        where.append('date>=?')
        params.append(start)
    if end:
        where.append('date<=?')
        params.append(end)
    if kind is not None:
        where.append('kind=?')
        params.append(_kind(kind))
    return ' AND '.join(where), params, {'start': start, 'end': end}


def entries(store, actor, owner=None, start=None, end=None, kind=None, limit=50, include_void=False, offset=0):
    owner = require_owner(store, actor, owner)
    wine_cellar._integer(limit, 1, 1000, 'Количество записей')
    wine_cellar._integer(offset, 0, 1_000_000, 'Смещение списка')
    if type(include_void) is not bool:
        raise ValueError('Неверный фильтр отменённых записей')
    where, params, _ = _filters(owner, start, end, kind, include_void)
    rows = store.query('SELECT * FROM life_ledger_entries WHERE '+where+' ORDER BY date DESC,id DESC LIMIT ? OFFSET ?',
                       (*params, limit, offset))
    return [_entry(row) for row in rows]


def get_entry(store, actor, entry_id, owner=None):
    owner = require_owner(store, actor, owner)
    wine_cellar._integer(entry_id, 1, 2**63-1, 'Запись')
    row = store.one('SELECT * FROM life_ledger_entries WHERE id=? AND user_id=?', (entry_id, owner))
    if not row:
        raise ValueError('Запись не найдена в вашем учёте')
    return _entry(row)


def void_entry(store, actor, entry_id):
    """Cancel one own manual entry with an audit trail; receipt stock is immutable here."""
    wine_cellar._integer(entry_id, 1, 2**63-1, 'Запись')
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner = require_owner(store, actor, write=True)
        row = conn.execute('SELECT * FROM life_ledger_entries WHERE id=? AND user_id=?', (entry_id, owner)).fetchone()
        if not row:
            raise ValueError('Запись не найдена в вашем учёте')
        if row['receipt_id'] is not None:
            raise ValueError('Запись из чека связана с винотекой; отдельная отмена расхода недоступна')
        already_voided = bool(row['voided_at'])
        if not already_voided:
            now = utcnow()
            conn.execute('UPDATE life_ledger_entries SET voided_at=? WHERE id=? AND user_id=?', (now, entry_id, owner))
            conn.execute("INSERT INTO life_ledger_actions(user_id,entry_id,action,created_at) VALUES(?,?,'void_manual',?)",
                         (owner, entry_id, now))
        return {**_entry(conn.execute('SELECT * FROM life_ledger_entries WHERE id=?', (entry_id,)).fetchone()),
                'already_voided': already_voided}


def summary(store, actor, owner=None, start=None, end=None):
    owner = require_owner(store, actor, owner)
    where, params, period = _filters(owner, start, end)
    rows = store.query('SELECT kind,category,SUM(amount_cents) AS cents,COUNT(*) AS count FROM life_ledger_entries WHERE '+
                       where+' GROUP BY kind,category ORDER BY kind,category', params)
    income = sum(row['cents'] for row in rows if row['kind'] == 'income')
    expenses = sum(row['cents'] for row in rows if row['kind'] == 'expense')
    return {'income': income/100, 'expenses': expenses/100, 'balance': (income-expenses)/100,
            'count': sum(row['count'] for row in rows), 'currency': 'RUB', 'period': period,
            'by_category': [{'kind': row['kind'], 'category': row['category'], 'amount': row['cents']/100,
                             'count': row['count']} for row in rows]}


def monthly_summary(store, actor, owner=None, start=None, end=None):
    owner = require_owner(store, actor, owner)
    where, params, _ = _filters(owner, start, end)
    rows = store.query('''SELECT substr(date,1,7) AS month,
        SUM(CASE WHEN kind='income' THEN amount_cents ELSE 0 END) AS income_cents,
        SUM(CASE WHEN kind='expense' THEN amount_cents ELSE 0 END) AS expense_cents,COUNT(*) AS count
        FROM life_ledger_entries WHERE '''+where+' GROUP BY month ORDER BY month', params)
    return [{'month': row['month'], 'income': row['income_cents']/100, 'expenses': row['expense_cents']/100,
             'balance': (row['income_cents']-row['expense_cents'])/100, 'count': row['count']} for row in rows]


def users(store, actor):
    require_owner(store, actor)
    if not store.authorize(actor, admin=True):
        raise PermissionError('Список доступен администратору')
    return store.query('''SELECT e.user_id,COUNT(*) AS entry_count,p.first_name,p.last_name,p.username
        FROM life_ledger_entries e LEFT JOIN user_profiles p ON p.user_id=e.user_id WHERE e.voided_at IS NULL
        GROUP BY e.user_id ORDER BY e.user_id''')


def _quantity(value):
    try:
        if isinstance(value, bool):
            raise InvalidOperation
        number = Decimal(str(value).replace(',', '.'))
        if not number.is_finite() or not 0 < number <= 100_000:
            raise InvalidOperation
        return number
    except (ValueError, TypeError, InvalidOperation):
        raise ValueError('Количество в строке чека должно быть положительным числом') from None


def _receipt_result(result):
    if not isinstance(result, dict):
        raise ValueError('Нужны распознанные данные чека')  # noqa: TRY004 - input validation.
    if result.get('currency', 'RUB') != 'RUB':
        raise ValueError('Пока поддерживаются чеки только в RUB')
    fingerprint = result.get('fingerprint', '')
    if not isinstance(fingerprint, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', fingerprint):
        raise ValueError('Не найден безопасный идентификатор чека')
    items = result.get('items', [])
    if not isinstance(items, list) or len(items) > 200:
        raise ValueError('В чеке должно быть не более 200 строк')
    cleaned = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Некорректная строка чека')  # noqa: TRY004 - input validation.
        name = wine_cellar._text(item.get('name'), 500, required=True)
        quantity = _quantity(item.get('quantity', 1))
        unit = wine_cellar._cents(item.get('unit_price'), optional=True)
        amount = wine_cellar._cents(item.get('amount', item.get('line_total')), optional=True)
        volume = item.get('volume_ml')
        if volume is not None:
            volume = wine_cellar._integer(volume, 1, 30_000, 'Объём, мл')
        vintage = wine_cellar._vintage(item.get('vintage')) or None
        from .receipts import receipt_identity
        inferred = receipt_identity(name, is_wine=item.get('is_wine') is True)
        if volume is None:
            volume = inferred['volume_ml']
            item = {**item, **{key: inferred[key] for key in ('volume_source', 'volume_requires_confirmation')}}
        cleaned.append({'name': name, 'quantity': float(quantity),
                        'unit_price': None if unit is None else unit/100, 'amount': None if amount is None else amount/100,
                        'line_total': None if amount is None else amount/100, 'is_wine': item.get('is_wine') is True,
                        'winery': wine_cellar._text(item.get('winery'), 200), 'vintage': vintage, 'volume_ml': volume,
                        'volume_source': item.get('volume_source') if item.get('volume_source') in {'name_explicit_unit', 'name_decimal_suffix'} else None,
                        'volume_requires_confirmation': item.get('volume_requires_confirmation') is True})
    total = _amount(result.get('total'), optional=True)
    day = wine_cellar._date(result['date']) if result.get('date') else None
    warnings = result.get('warnings', [])
    warnings = [wine_cellar._text(item, 500) for item in warnings[:20] if isinstance(item, str)] if isinstance(warnings, list) else []
    return {'fingerprint': fingerprint.lower(), 'date': day, 'total': None if total is None else total/100,
            'merchant': wine_cellar._text(result.get('merchant'), 300), 'currency': 'RUB', 'items': cleaned,
            'warnings': warnings, 'requires_confirmation': True}


def _draft(row):
    result = dict(row)
    result['result'] = _receipt_result(json.loads(result['result'])) if result['status'] == 'draft' else json.loads(result['result'])
    return result


def save_receipt_draft(store, actor, result, token):
    result = _receipt_result(result)
    token = wine_cellar._text(token, 160, required=True)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner = require_owner(store, actor, write=True)
        conn.execute("DELETE FROM life_receipt_drafts WHERE user_id=? AND status='draft' AND expires_at<?", (owner, utcnow()))
        previous = conn.execute('SELECT * FROM life_receipt_drafts WHERE user_id=? AND token=?', (owner, token)).fetchone()
        if previous:
            if previous['fingerprint'] != result['fingerprint']:
                raise ValueError('Токен уже принадлежит другому чеку')
            if previous['status'] == 'confirmed':
                return _draft(previous)
        now = utcnow()
        expiry = (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
        conn.execute('''INSERT INTO life_receipt_drafts(user_id,token,fingerprint,result,status,created_at,updated_at,expires_at)
            VALUES(?,?,?,?,'draft',?,?,?) ON CONFLICT(user_id,token) DO UPDATE SET result=excluded.result,
            updated_at=excluded.updated_at,expires_at=excluded.expires_at''',
            (owner, token, result['fingerprint'], dumps(result), now, now, expiry))
        return _draft(conn.execute('SELECT * FROM life_receipt_drafts WHERE user_id=? AND token=?', (owner, token)).fetchone())


def get_receipt_draft(store, actor, token):
    owner = require_owner(store, actor)
    token = wine_cellar._text(token, 160, required=True)
    row = store.one('SELECT * FROM life_receipt_drafts WHERE user_id=? AND token=?', (owner, token))
    if not row or row['status'] == 'draft' and row['expires_at'] < utcnow():
        raise ValueError('Черновик не найден или истёк; загрузите чек заново')
    return _draft(row)


def _confirmed(row, already_imported):
    return {'receipt_id': row['id'], 'entry_id': row['entry_id'], 'wine_lot_ids': json.loads(row['wine_lot_ids']),
            'already_imported': already_imported, 'total': row['total_cents']/100, 'fingerprint': row['fingerprint'],
            'rounding_difference_cents': row['rounding_difference_cents'], 'currency': 'RUB'}


def _wine_fields(item, override, day, line_cents):
    if not isinstance(override, dict):
        raise ValueError('Некорректные исправления строки вина')  # noqa: TRY004 - input validation.
    allowed = {'name', 'winery', 'vintage', 'volume_ml', 'quantity', 'location', 'notes',
               'country', 'region', 'grapes', 'metadata_provenance'}
    if set(override)-allowed:
        raise ValueError('Можно исправлять только сведения о выбранном вине')
    merged = {**item, **override}
    quantity = _quantity(merged['quantity'])
    if quantity != quantity.to_integral_value():
        raise ValueError('Для вина укажите целое число бутылок')
    count = int(quantity)
    volume = merged.get('volume_ml')
    if volume is None:
        raise ValueError('Укажите объём бутылки выбранного вина; по чеку он неизвестен')
    unit_cents = int((Decimal(line_cents)/count).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    fields = {'name': merged['name'], 'winery': merged.get('winery', ''), 'vintage': merged.get('vintage'),
              'quantity': count, 'volume_ml': volume, 'purchase_price': str(Decimal(unit_cents)/100),
              'purchase_date': day, 'location': merged.get('location', ''), 'notes': merged.get('notes', ''), 'currency': 'RUB'}
    fields.update(wine_cellar._attributes(merged))
    return fields, unit_cents*count-line_cents


def prepare_receipt_wines(store, actor, token):
    """Open an explicit stock review for a receipt whose expense already exists."""
    owner = require_owner(store, actor, write=True)
    wine_cellar.require_owner(store, actor, write=True)
    draft = get_receipt_draft(store, actor, token)
    imported = store.one('SELECT * FROM life_receipt_imports WHERE user_id=? AND fingerprint=?',
                         (owner, draft['fingerprint']))
    if not imported:
        raise ValueError('Сначала сохраните чек или выберите вина в его обычном черновике')
    # Reopening an accounted receipt must resume the user's corrections and selection.
    for row in store.query("SELECT token FROM life_receipt_drafts WHERE user_id=? AND fingerprint=? AND status='draft' AND expires_at>=? ORDER BY updated_at DESC",
                           (owner, draft['fingerprint'], utcnow())):
        ui = store.get_setting(f'receipt_ui:{owner}:{row["token"]}', {})
        if ui.get('stock_only') == imported['id']:
            return row['token']
    result = {**draft['result'], 'items': json.loads(imported['items']),
              'date': imported['receipt_date'], 'total': imported['total_cents']/100, 'merchant': imported['merchant']}
    new_token = uuid4().hex
    save_receipt_draft(store, actor, result, new_token)
    existing = {row['operation_id'] for row in store.query(
        'SELECT operation_id FROM wine_lots WHERE user_id=? AND operation_id LIKE ?', (owner, f'receipt:{imported["id"]}:line:%'))}
    selected = [index for index, item in enumerate(result['items'])
                if item.get('is_wine') and f'receipt:{imported["id"]}:line:{index}' not in existing]
    store.set_setting(f'receipt_ui:{owner}:{new_token}', {'selected': selected, 'overrides': {},
                      'stock_only': imported['id'], 'return_to': 'wc:menu'})
    return new_token


def receipt_wine_sources(store, actor):
    """Owned, resumable receipts with wine lines that have not entered stock."""
    owner = require_owner(store, actor)
    imported = {row['fingerprint']: row for row in store.query(
        'SELECT * FROM life_receipt_imports WHERE user_id=?', (owner,))}
    added = {row['operation_id'] for row in store.query(
        "SELECT operation_id FROM wine_lots WHERE user_id=? AND operation_id LIKE 'receipt:%'", (owner,))}
    sources, seen = [], set()
    for row in store.query("SELECT token,fingerprint,status,result FROM life_receipt_drafts WHERE user_id=? AND (status='confirmed' OR expires_at>=?) ORDER BY updated_at DESC",
                           (owner, utcnow())):
        fingerprint = row['fingerprint']
        if fingerprint in seen:
            continue
        saved = imported.get(fingerprint)
        items = json.loads(saved['items']) if saved else json.loads(row['result'])['items']
        remaining = [index for index, item in enumerate(items) if item.get('is_wine')
                     and (not saved or f'receipt:{saved["id"]}:line:{index}' not in added)]
        if not remaining:
            continue
        seen.add(fingerprint)
        ui = store.get_setting(f'receipt_ui:{owner}:{row["token"]}', {})
        kind = 'draft' if row['status'] == 'draft' and (not saved or ui.get('stock_only') == saved['id']) else 'saved'
        sources.append({'token': row['token'], 'kind': kind, 'wine_count': len(remaining)})
    return sources


def confirm_receipt(store, actor, token, *, selected_wine_items, overrides=None, operation_id=None, add_wines_to_existing=False):
    token = wine_cellar._text(token, 160, required=True)
    operation = wine_cellar._operation(operation_id or token)
    if type(add_wines_to_existing) is not bool:
        raise ValueError('Неверный режим подтверждения чека')
    if (not isinstance(selected_wine_items, list) or len(selected_wine_items) > 200
            or any(type(index) is not int or index < 0 for index in selected_wine_items)
            or len(set(selected_wine_items)) != len(selected_wine_items)):
        raise ValueError('Выберите строки вина без повторов')
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError('Некорректные исправления чека')  # noqa: TRY004 - input validation.
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner = require_owner(store, actor, write=True)
        if selected_wine_items:
            wine_cellar.require_owner(store, actor, write=True)
        draft = conn.execute('SELECT * FROM life_receipt_drafts WHERE user_id=? AND token=?', (owner, token)).fetchone()
        if not draft:
            raise ValueError('Черновик чека не найден')
        previous = conn.execute('SELECT * FROM life_receipt_imports WHERE user_id=? AND fingerprint=?',
                                (owner, draft['fingerprint'])).fetchone()
        if previous and not add_wines_to_existing:
            conn.execute("UPDATE life_receipt_drafts SET status='confirmed',receipt_id=?,updated_at=? WHERE user_id=? AND token=?",
                         (previous['id'], utcnow(), owner, token))
            return _confirmed(previous, True)
        if draft['status'] == 'draft' and draft['expires_at'] < utcnow():
            raise ValueError('Черновик истёк; загрузите чек заново')
        if add_wines_to_existing and not previous:
            raise ValueError('Учтённый чек не найден; повторный расход не создан')
        if not previous and conn.execute('SELECT 1 FROM life_receipt_imports WHERE user_id=? AND operation_id=?', (owner, operation)).fetchone():
            raise ValueError('Идентификатор операции уже использован для другого чека')
        result = _receipt_result(json.loads(draft['result']))
        if not result['date'] or result['total'] is None:
            raise ValueError('Перед подтверждением укажите дату и итоговую сумму чека')
        total, items = _amount(result['total']), result['items']
        if any(index >= len(items) for index in selected_wine_items):
            raise ValueError('Строка чека не найдена')
        if any(str(key) not in {str(index) for index in selected_wine_items} for key in overrides):
            raise ValueError('Исправления относятся к невыбранной строке')
        if previous:
            recorded = json.loads(previous['items'])
            if any(not isinstance(value, dict) for value in overrides.values()):
                raise ValueError('Некорректные исправления чека')
            if (total != previous['total_cents'] or result['date'] != previous['receipt_date'] or len(items) != len(recorded)
                    or any(any(item.get(key) != old.get(key) for key in ('quantity', 'amount', 'unit_price'))
                           for item, old in zip(items, recorded))
                    or any('quantity' in value and _quantity(value['quantity']) != _quantity(items[int(index)]['quantity'])
                           for index, value in overrides.items())):
                raise ValueError('Суммы, дата и количество учтённого чека не меняются; расход уже записан')
        wine_inputs = []
        if selected_wine_items:
            amounts = [wine_cellar._cents(item['amount'], optional=True) for item in items]
            if any(amount is None for amount in amounts) or sum(amounts) != total:
                raise ValueError('Проверьте суммы всех строк после скидок: их итог должен совпадать с суммой чека')
            for index in selected_wine_items:
                fields, rounding = _wine_fields(items[index], overrides.get(index, overrides.get(str(index), {})),
                                                result['date'], amounts[index])
                wine_inputs.append((index, fields, rounding))
        if previous:
            lot_ids, added_ids = json.loads(previous['wine_lot_ids']), []
            rounding = previous['rounding_difference_cents']
            for index, fields, difference in wine_inputs:
                line_operation = f'receipt:{previous["id"]}:line:{index}'
                if conn.execute('SELECT 1 FROM wine_lots WHERE user_id=? AND operation_id=?', (owner, line_operation)).fetchone():
                    continue
                lot = wine_cellar.add_bottle_in_transaction(store, actor, fields, conn, operation_id=line_operation)
                lot_ids.append(lot['id'])
                added_ids.append(lot['id'])
                rounding += difference
            conn.execute('UPDATE life_receipt_imports SET wine_lot_ids=?,rounding_difference_cents=? WHERE id=?',
                         (dumps(lot_ids), rounding, previous['id']))
            conn.execute("UPDATE life_receipt_drafts SET status='confirmed',receipt_id=?,updated_at=? WHERE user_id=? AND token=?",
                         (previous['id'], utcnow(), owner, token))
            return {**_confirmed(conn.execute('SELECT * FROM life_receipt_imports WHERE id=?', (previous['id'],)).fetchone(), True),
                    'added_wine_lot_ids': added_ids}
        cursor = conn.execute('''INSERT INTO life_receipt_imports(user_id,fingerprint,receipt_date,total_cents,merchant,
            confirmed_at,items,wine_lot_ids,operation_id) VALUES(?,?,?,?,?,?,?,'[]',?)''',
            (owner, result['fingerprint'], result['date'], total, result['merchant'], utcnow(), dumps(items), operation))
        receipt_id = cursor.lastrowid
        entry = _add_entry(conn, owner, 'expense', total, result['date'], 'Покупки по чеку', '',
                           'receipt:'+str(receipt_id), result['merchant'], receipt_id)
        lot_ids, rounding = [], 0
        for index, fields, difference in wine_inputs:
            lot = wine_cellar.add_bottle_in_transaction(store, actor, fields, conn, operation_id=f'receipt:{receipt_id}:line:{index}')
            lot_ids.append(lot['id'])
            rounding += difference
        conn.execute('UPDATE life_receipt_imports SET entry_id=?,wine_lot_ids=?,rounding_difference_cents=? WHERE id=?',
                     (entry['id'], dumps(lot_ids), rounding, receipt_id))
        conn.execute("UPDATE life_receipt_drafts SET status='confirmed',receipt_id=?,updated_at=? WHERE user_id=? AND token=?",
                     (receipt_id, utcnow(), owner, token))
        return _confirmed(conn.execute('SELECT * FROM life_receipt_imports WHERE id=?', (receipt_id,)).fetchone(), False)
