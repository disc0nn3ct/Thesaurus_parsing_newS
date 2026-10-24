"""Requested wine identity lookups for private, unconfirmed receipt drafts."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from . import life_ledger as ledger
from . import wine_cellar
from .store import dumps, utcnow

FIELD_CODES = {'y': 'vintage', 'w': 'winery', 'v': 'volume_ml', 'c': 'country', 'r': 'region', 'g': 'grapes'}


def field_choices(store, actor, token, index, field):
    """Offer normalized, deduplicated source values without selecting one."""
    from .wine_enrichment import candidate_allowed
    if field not in FIELD_CODES.values():
        return []
    state = lookup_result(store, actor, token, index)
    choices, seen = [], set()
    for position,candidate in enumerate(state.get('candidates', [])[:8]):
        if not candidate_allowed(store, actor, candidate):
            continue
        raw = candidate.get('published_vintage' if field == 'vintage' else field)
        try:
            if field == 'vintage':
                value = wine_cellar._vintage(raw)
            elif field == 'volume_ml':
                value = wine_cellar._integer(raw,20,30000,'Объём')
            elif field == 'winery':
                value = wine_cellar._text(raw,160)
            else:
                value = wine_cellar._attributes({field:raw})[field]
        except (ValueError, TypeError):
            continue
        if value in (None,'',[]):
            continue
        key = dumps(sorted(value) if isinstance(value,list) else value)
        if key in seen:
            continue
        seen.add(key)
        fingerprint = hashlib.sha256(dumps([state.get('request_id'),state.get('signature'),field,
                                           position,candidate.get('id'),candidate.get('source_url'),value]).encode()).hexdigest()[:8]
        label = ', '.join(value) if isinstance(value,list) else str(value)+(' мл' if field == 'volume_ml' else '')
        choices.append({'candidate':position,'fingerprint':fingerprint,'value':value,'label':label})
    return sorted(choices,key=lambda row:row['label'])


def apply_field_choice(store, actor, token, index, field, position, fingerprint):
    """An explicit source choice changes only the selected field in the draft."""
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        ui, fields, _ = _context(store, actor, token, index, conn)
        choices = field_choices(store, actor, token, index, field)
        choice = next((row for row in choices if row['candidate']==position and row['fingerprint']==fingerprint),None)
        if not choice:
            raise ValueError('Этот вариант устарел. Откройте изменение поля заново.')
        ui.setdefault('overrides', {}).setdefault(str(index), {})[field] = choice['value']
        # The selected value is now confirmed by the user, rather than owned by autofill.
        info = ui.get('wine_info', {}).get(str(index), {})
        info.get('auto_values', {}).pop(field,None)
        state = ui['wine_lookups'][str(index)]
        state.setdefault('confirmed_fields', {})[field] = {'value':choice['value'],
            'source_url':state['candidates'][position].get('source_url')}
        merged = {**fields, **ui['overrides'][str(index)]}
        state['signature'] = hashlib.sha256(dumps({key:merged.get(key) for key in
            ('name','winery','vintage','volume_ml')}).encode()).hexdigest()
        _save(conn, actor, token, ui)
        return choice['value']


class _StaleLookup(ValueError):
    pass


def original_item(store, actor, ui, index, item, conn=None):
    """Keep the receipt wording available separately from the final wine card."""
    saved = ui.get('original_items', {}).get(str(index))
    if saved is not None:
        return saved
    if ui.get('stock_only'):
        query = 'SELECT items FROM life_receipt_imports WHERE id=? AND user_id=?'
        params = (ui['stock_only'], actor)
        row = conn.execute(query, params).fetchone() if conn else store.one(query, params)
        if row:
            items = json.loads(row['items'])
            if 0 <= index < len(items):
                return items[index]
    return item


def _context(store, actor, token, index, conn=None, *, require_wine=True):
    ledger.require_owner(store, actor, write=True)
    if require_wine:
        wine_cellar.require_owner(store, actor, write=True)
    draft = ledger.get_receipt_draft(store, actor, token)
    if draft['status'] != 'draft':
        raise ValueError('Чек уже сохранён. Дополнение доступно до подтверждения.')
    items = draft['result']['items']
    if type(index) is not int or not 0 <= index < len(items):
        raise ValueError('Товар не найден.')
    key = f'receipt_ui:{actor}:{token}'
    row = conn.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone() if conn else store.one('SELECT value FROM settings WHERE key=?', (key,))
    ui = json.loads(row['value']) if row else {'selected': [], 'overrides': {}}
    original = original_item(store, actor, ui, index, items[index], conn)
    ui.setdefault('original_items', {}).setdefault(str(index), dict(original))
    fields = {**items[index], **ui.get('overrides', {}).get(str(index), {})}
    identity = {key: fields.get(key) for key in ('name', 'winery', 'vintage', 'volume_ml')}
    signature = hashlib.sha256(dumps(identity).encode()).hexdigest()
    return ui, fields, signature


def _save(conn, actor, token, ui):
    conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                 (f'receipt_ui:{actor}:{token}', dumps(ui)))


def remember_item(store, actor, token, index):
    """Capture the initial row before a manual edit, including non-wine rows."""
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        ui, _, _ = _context(store, actor, token, index, conn, require_wine=False)
        _save(conn, actor, token, ui)


def queue_lookup(store, actor, token, index, *, automatic=False, refresh=False):
    """Only queue I/O; repeated taps reuse a pending job for the same identity."""
    if type(automatic) is not bool or type(refresh) is not bool:
        raise TypeError('Неверные параметры поиска')
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        ui, _fields, signature = _context(store, actor, token, index, conn)
        previous = ui.get('wine_lookups', {}).get(str(index), {})
        if automatic and not refresh and previous.get('signature') == signature and previous.get('status'):
            return previous.get('job_id')
        if previous.get('signature') == signature and previous.get('job_id'):
            job = conn.execute('SELECT state FROM jobs WHERE id=?', (previous['job_id'],)).fetchone()
            if job and job['state'] in {'pending', 'running'}:
                return previous['job_id']
        request_id = uuid4().hex
        payload = {'actor_id': actor, 'owner_id': actor, 'report_id': uuid4().hex, 'request_id': request_id,
                   'receipt_token': token, 'index': index, 'signature': signature, 'required_features': ['wine_cellar']}
        payload['automatic'] = automatic
        cursor = conn.execute('INSERT INTO jobs(kind,payload,dedupe_key,created_at) VALUES(?,?,?,?)',
                              ('life_wine_enrich', dumps(payload), 'receipt-wine:'+request_id, utcnow()))
        ui.setdefault('wine_lookups', {})[str(index)] = {'signature': signature, 'request_id': request_id,
            'status': 'pending', 'job_id': cursor.lastrowid, 'created_at': utcnow(), 'candidates': [], 'warnings': []}
        _save(conn, actor, token, ui)
        return cursor.lastrowid


def queue_automatic(store, actor, token, index=None):
    """Queue local/public enrichment once per wine, without changing inventory."""
    from .feature_access import allowed
    if not allowed(store, actor, 'wine_cellar') or not allowed(store, actor, 'life_finance'):
        return 0
    if not store.get_setting(f'wine_metadata_auto:{actor}', False):
        return 0
    draft = ledger.get_receipt_draft(store, actor, token)
    if draft['status'] != 'draft':
        return 0
    count = 0
    for position, item in enumerate(draft['result']['items']):
        if index is not None and position != index:
            continue
        if not item.get('is_wine'):
            continue
        ui, fields, signature = _context(store, actor, token, position)
        if all(fields.get(key) for key in ('winery', 'country', 'region', 'grapes')):
            continue
        previous = ui.get('wine_lookups', {}).get(str(position), {})
        if previous.get('signature') == signature and previous.get('status'):
            continue
        queue_lookup(store, actor, token, position, automatic=True)
        count += 1
        if count >= 20:
            break
    return count


def _check_request(ui, signature, payload):
    state = ui.get('wine_lookups', {}).get(str(payload['index']), {})
    if signature != payload.get('signature') or state.get('request_id') != payload.get('request_id'):
        raise _StaleLookup('Сведения о товаре изменились. Найдите вино заново.')


def _clear_automatic(ui, index):
    """Restore only values we still own, including the previous source's notes."""
    info = ui.get('wine_info', {}).pop(str(index), {})
    override = ui.get('overrides', {}).get(str(index), {})
    restored = {}
    for key, value in info.get('auto_values', {}).items():
        if override.get(key) == value:
            restored[key] = info.get('previous_values', {}).get(key)
            if restored[key] in (None, ''):
                override.pop(key, None)
            else:
                override[key] = restored[key]
    return restored


def _apply(ui, fields, index, candidate):
    from .receipts import receipt_volume
    prior_info = ui.get('wine_info', {}).get(str(index), {})
    prior_override = ui.get('overrides', {}).get(str(index), {})
    initial_name = ui.get('original_items', {}).get(str(index), {}).get('name', fields.get('name'))
    automatic_name = prior_info.get('auto_values', {}).get('name')
    manual_name = (str(index) in ui.get('manual_names', []) or
                   fields.get('name') != initial_name and fields.get('name') != automatic_name or
                   'name' in prior_override and prior_override['name'] not in (initial_name, automatic_name))
    if candidate:
        # A refreshed page replaces its earlier metadata too; it must not append
        # another full description just because the retrieval time has changed.
        # Callers also use this mapping to sign the resulting identity.
        fields.update(_clear_automatic(ui, index))
    override = ui.setdefault('overrides', {}).setdefault(str(index), {})
    previous = ui.get('wine_info', {}).get(str(index), {})
    original = dict(fields)
    added = []
    # Normalize the wine card's name; preserve receipt wording and manual edits.
    if candidate.get('name') and not manual_name:
        try:
            name = wine_cellar._text(candidate['name'], required=True)
        except ValueError:
            name = None
        if name and name != fields.get('name'):
            override['name'] = name
            added.append('name')
    # Quantities and paid amounts remain the purchase record.
    for key in ('winery', 'volume_ml', 'country', 'region', 'grapes'):
        value = candidate.get(key)
        if fields.get(key) in (None, '', []) and value not in (None, '', []):
            override[key] = value
            added.append(key)
    normalized = wine_cellar._attributes({**fields, **override})
    for key in ('country', 'region', 'grapes'):
        if key in added:
            override[key] = normalized[key]
    if candidate and any(key in added for key in ('country', 'region', 'grapes', 'winery')):
        override['metadata_provenance'] = candidate.get('attribute_sources') or {
            key: [{'url': candidate.get('source_url'), 'fetched_at': candidate.get('fetched_at')}]
            for key in added if key in {'winery', 'country', 'region', 'grapes', 'volume_ml'}}
        added.append('metadata_provenance')
    inferred = receipt_volume(fields['name'], is_wine=True)
    if not fields.get('volume_ml') and not override.get('volume_ml') and inferred['volume_ml']:
        override['volume_ml'] = inferred['volume_ml']
        added.append('volume_ml')
    details = [('Вино', candidate.get('name')), ('Страна', candidate.get('country')), ('Регион', candidate.get('region')),
               ('Сорта', candidate.get('grapes')), ('Описание', candidate.get('description')),
               ('Каталог', candidate.get('source_title') if candidate.get('source_kind') == 'vivino_upload' else None),
               ('Источник', candidate.get('source_url')), ('Проверено', candidate.get('fetched_at'))]
    notes = '; '.join(f'{label}: {(", ".join(map(str, value)) if isinstance(value, list) else str(value))[:450]}' for label, value in details if value)
    existing = str(fields.get('notes') or '')
    if notes and notes not in existing:
        override['notes'] = (existing+'; '+notes).strip('; ')[:2000]
        added.append('notes')
    if candidate or added:
        automatic = {key: value for key, value in previous.get('auto_values', {}).items() if override.get(key) == value}
        automatic.update({key: override[key] for key in added})
        prior_values = {**{key: original.get(key) for key in added}, **previous.get('previous_values', {})}
        ui.setdefault('wine_info', {})[str(index)] = {**candidate, 'fields_added': added,
            'volume_from_name': inferred['volume_source'] if 'volume_ml' in added and not candidate.get('volume_ml') else None,
            'auto_values': automatic, 'previous_values': prior_values}
    return added


def run_lookup(store, actor, payload, *, dry_run=False):
    from .wine_enrichment import lookup
    token, index = payload['receipt_token'], payload['index']

    def guard():
        if payload.get('automatic') and not store.get_setting(f'wine_metadata_auto:{actor}', False):
            raise PermissionError('Автоматический поиск отключён пользователем')
        ui, fields, signature = _context(store, actor, token, index)
        _check_request(ui, signature, payload)
        return fields

    try:
        fields = guard()
        if dry_run:
            return {'status': 'dry_run', 'candidates': [], 'warnings': [], 'fields_added': []}
        try:
            result = lookup(store, actor, fields, guard=guard)
        except _StaleLookup:
            raise
        except ValueError:
            result = {'status': 'invalid', 'candidates': [], 'warnings': [
                'Уточните название, год и объём вина: без ссылок, фискальных номеров и противоречий между названием и полями.']}
        with store.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            ui, current, signature = _context(store, actor, token, index, conn)
            _check_request(ui, signature, payload)
            from .wine_enrichment import automatic_candidate, candidate_allowed
            candidates = []
            for candidate in result.get('candidates', [])[:8]:
                if candidate_allowed(store, actor, candidate):
                    candidates.append(candidate)
            if result.get('candidates') and not candidates:
                result = {**result, 'status': 'disabled', 'warnings': ['Источник отключён или больше не разрешён. Сведения о вине не применены.']}
            automatic = automatic_candidate(candidates)
            unique = automatic is not None
            added = _apply(ui, current, index, automatic or {})
            # Filling missing fields changes the signature; choices now refer to this version.
            merged = {**current, **ui.get('overrides', {}).get(str(index), {})}
            signature = hashlib.sha256(dumps({key: merged.get(key) for key in ('name', 'winery', 'vintage', 'volume_ml')}).encode()).hexdigest()
            state = {**result, 'candidates': candidates, 'fields_added': added, 'request_id': payload['request_id'],
                     'signature': signature, 'status': 'filled' if unique else result['status'],
                     'expires_at': (datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat()}
            ui.setdefault('wine_lookups', {})[str(index)] = state
            _save(conn, actor, token, ui)
            return state
    except _StaleLookup:
        return {'status': 'stale', 'candidates': [], 'warnings': ['Карточка товара изменена. Запустите поиск заново.'], 'fields_added': []}


def lookup_result(store, actor, token, index):
    ui, _, signature = _context(store, actor, token, index)
    state = ui.get('wine_lookups', {}).get(str(index), {})
    if not state:
        raise ValueError('Сначала нажмите «Найти сведения о вине».')
    if state.get('signature') != signature or state.get('expires_at', utcnow()) < utcnow() and state.get('status') != 'pending':
        raise ValueError('Поиск устарел или карточка изменена. Найдите вино заново.')
    return state


def apply_candidate(store, actor, token, index, candidate_index, *, confirm_vintage=False):
    from .wine_enrichment import candidate_allowed
    if type(confirm_vintage) is not bool:
        raise TypeError('Неверное подтверждение года бутылки')
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        ui, fields, _signature = _context(store, actor, token, index, conn)
        state = lookup_result(store, actor, token, index)
        candidates = state.get('candidates', [])
        if type(candidate_index) is not int or not 0 <= candidate_index < len(candidates):
            raise ValueError('Вариант не найден. Найдите вино заново.')
        candidate = candidates[candidate_index]
        if not candidate_allowed(store, actor, candidate):
            raise ValueError('Источник отключён или загрузка Vivino изменилась. Найдите вино заново.')
        added = _apply(ui, fields, index, candidate)
        if confirm_vintage:
            year = wine_cellar._vintage(candidate.get('published_vintage'))
            current_year = wine_cellar._vintage(fields.get('vintage'))
            if not year or current_year and current_year != year:
                raise ValueError('Год не опубликован или отличается от уже указанного. Проверьте этикетку.')
            ui.setdefault('overrides', {}).setdefault(str(index), {})['vintage'] = year
            added.append('vintage')
        merged = {**fields, **ui.get('overrides', {}).get(str(index), {})}
        state.update(status='filled', fields_added=added,
            signature=hashlib.sha256(dumps({key: merged.get(key) for key in ('name', 'winery', 'vintage', 'volume_ml')}).encode()).hexdigest())
        ui['wine_lookups'][str(index)] = state
        _save(conn, actor, token, ui)
        return added


def invalidate_item(store, actor, token, index, field):
    """A changed identity must not retain attributes copied for a different wine."""
    if field not in {'name', 'winery', 'vintage', 'volume_ml'}:
        return
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        ui, _fields, _signature = _context(store, actor, token, index, conn, require_wine=False)
        ui.get('wine_lookups', {}).pop(str(index), None)
        _clear_automatic(ui, index)
        _save(conn, actor, token, ui)
