"""Sequential receipt recognition and private ledger exports; no paid APIs."""
import csv
import json
from pathlib import Path

from . import life_ledger as ledger
from .delivery import prepare_delivery
from .exporting import safe_cell
from .store import dumps, utcnow
from .telegram_data_jobs import _artifact_path, report_identifier


def _recognize(store, settings, payload, dry_run):
    from .receipts import recognize_receipt
    actor, token = payload['actor_id'], payload['request_id']
    report_identifier(token)
    root = (settings.data_dir/'receipts'/str(actor)).resolve()
    path = Path(payload.get('path', '')).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Чек не найден в ваших загрузках. Пришлите файл заново.')
    result = recognize_receipt(path, dry_run=dry_run)
    ledger.require_owner(store, actor, write=True)
    warnings = '\n'.join(str(value)[:300] for value in result.get('warnings', [])[:4])
    if result.get('status') == 'candidate' and not dry_run:
        ledger.save_receipt_draft(store, actor, result, token)
        from .receipt_wines import queue_automatic
        lookups = queue_automatic(store, actor, token)
        text = ('Чек распознан · нужна ваша проверка\n'
                f'Товаров: {len(result.get("items", []))}. Дата: {result.get("date") or "нужно указать"}.\n'
                f'Итог: {str(result["total"])+" ₽" if result.get("total") is not None else "нужно указать"}.\n'
                'Расходы и запас ещё не изменены. Проверьте товары и выберите вина для винотеки.\n'+warnings)
        if lookups:
            text += '\nСведения о винах ищутся автоматически. Страна, регион, сорта и источники появятся в карточках товаров.'
        buttons = [[{'text': 'Проверить чек', 'callback_data': 'rc:review:'+token}]]
    else:
        text = ('Чек не распознан. Учёт не изменён. Можно ввести расход вручную или отправить более чёткий файл.\n'+warnings)
        buttons = [[{'text': 'Расход вручную', 'callback_data': 'lf:add:expense'}],
                   [{'text': 'Отправить другой чек', 'callback_data': 'lf:receipt'}]]
    return text, [], buttons


def _export(store, settings, payload):
    actor = payload['actor_id']
    format_ = payload.get('format')
    if format_ not in {'csv', 'xlsx'}:
        raise ValueError('Выберите CSV или Excel.')
    ledger.require_owner(store, actor)
    where, params, _ = ledger._filters(actor, payload.get('start'), payload.get('end'))
    path = _artifact_path(settings, payload['report_id'], 'life-ledger.'+format_)
    columns = ['date', 'kind', 'amount', 'currency', 'category', 'note', 'merchant', 'receipt_id']
    headings = ['Дата', 'Вид', 'Сумма', 'Валюта', 'Категория', 'Описание', 'Магазин', 'Чек']
    count = 0

    def values(cursor):
        nonlocal count
        while batch := cursor.fetchmany(1000):
            ledger.require_owner(store, actor)
            for source in batch:
                row = {**dict(source), 'amount': source['amount_cents']/100, 'currency': 'RUB'}
                count += 1
                yield [safe_cell(('Расход' if row[key] == 'expense' else 'Доход') if key == 'kind' else row.get(key)) for key in columns]

    try:
        # A read snapshot prevents inserts/cancellations from shifting paginated rows.
        # fetchmany and write-only workbooks bound memory independently of ledger size.
        with store.connection() as conn:
            conn.execute('BEGIN')
            cursor = conn.execute('SELECT * FROM life_ledger_entries WHERE '+where+' ORDER BY date DESC,id DESC', params)
            if format_ == 'csv':
                with path.open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(headings)
                    writer.writerows(values(cursor))
            else:
                from openpyxl import Workbook
                book = Workbook(write_only=True)
                sheet = book.create_sheet('Доходы и расходы')
                sheet.append(headings)
                for row in values(cursor):
                    sheet.append(row)
                book.save(path)
        ledger.require_owner(store, actor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return f'Доходы и расходы · экспорт\nОпераций: {count}. Валюта: RUB.', [
        {'kind': 'document', 'path': str(path), 'caption': 'Личный учёт · '+format_.upper()}], []


def _enrich(store, payload, dry_run=False):
    from .receipt_wines import run_lookup
    state = run_lookup(store, payload['actor_id'], payload, dry_run=dry_run)
    labels = {'name': 'название', 'winery': 'производитель', 'volume_ml': 'объём', 'country': 'страна', 'region': 'регион',
              'grapes': 'сорта винограда', 'notes': 'описание', 'metadata_provenance': 'источники'}
    if state['status'] == 'filled':
        text = ('Сведения о вине найдены. В черновик добавлены: '+', '.join(labels.get(key, key) for key in state['fields_added'])+'.'
                if state['fields_added'] else 'Сведения о вине проверены. Заполненные поля сохранены.')
        from .telegram_life import receipt_changes
        actor, token, index = payload['actor_id'], payload['receipt_token'], payload['index']
        draft = ledger.get_receipt_draft(store, actor, token)
        ui = store.get_setting(f'receipt_ui:{actor}:{token}', {'overrides': {}})
        changes = receipt_changes(store, actor, ui, index, draft['result']['items'][index])
        if changes:
            text += '\n\nИзменения: было → стало\n'+'\n'.join(changes)
    elif state.get('candidates'):
        text = f'Найдено вариантов: {len(state["candidates"])}. Выберите карточку вашего вина; неоднозначные сведения не подставлены.'
    else:
        text = {'disabled': 'Нет подключённых источников для поиска вина. Их подключает администратор.',
                'stale': 'Карточка товара изменилась во время поиска. Найдите вино заново.',
                'unavailable': 'Источник временно не предоставил сведения о вине. Попробуйте позже или уточните название.',
                'empty': 'По названию из чека вино не найдено. Уточните название с этикетки и повторите поиск.'}.get(state['status'], 'Поиск завершён.')
        if 'volume_ml' in state.get('fields_added', []):
            text += '\nОбъём определён из названия чека; проверьте его.'
    text += '\n\nПроверьте карточку товара перед сохранением чека. Год урожая, отсутствующий в чеке, не определяется по текущему ассортименту магазина.'
    warnings = state.get('warnings', [])
    if warnings:
        text += '\n'+'; '.join(str(value)[:250] for value in warnings[:3])
    token, index = payload['receipt_token'], payload['index']
    buttons = [[{'text': 'Проверить товар', 'callback_data': f'rc:item:{token}:{index}'}]]
    if state.get('candidates'):
        buttons.append([{'text': 'Найденные сведения и источник', 'callback_data': f'rc:info:{token}:{index}'}])
    return text, [], buttons


def run_job(store, settings, job, payload, *, dry_run=False):
    if job['kind'] not in {'life_receipt', 'life_export', 'life_wine_enrich'}:
        raise ValueError('Неизвестное задание раздела «Жизнь».')
    actor = payload.get('actor_id')
    owner = ledger.require_owner(store, actor, payload.get('owner_id'), write=job['kind'] != 'life_export')
    if job['kind'] == 'life_wine_enrich':
        from .wine_cellar import require_owner
        require_owner(store, actor, write=True)
    if owner != actor:
        raise PermissionError('Загрузка чеков и экспорт выполняются только для себя.')
    report_id = report_identifier(payload['report_id'])
    if dry_run:
        if job['kind'] == 'life_receipt':
            _recognize(store, settings, payload, True)
        elif job['kind'] == 'life_wine_enrich':
            _enrich(store, payload, True)
        elif payload.get('format') not in {'csv', 'xlsx'}:
            raise ValueError('Выберите CSV или Excel.')
        return report_id
    existing = store.one('SELECT quality FROM reports WHERE id=?', (report_id,))
    if existing:
        quality = json.loads(existing['quality'])
        if quality.get('feature') != 'life_finance' or quality.get('actor_id') != actor or quality.get('owner_id') != owner:
            raise PermissionError('Результат принадлежит другому пользователю или инструменту.')
    else:
        if job['kind'] == 'life_receipt':
            text, parts, buttons = _recognize(store, settings, payload, dry_run)
        elif job['kind'] == 'life_export':
            text, parts, buttons = _export(store, settings, payload)
        elif job['kind'] == 'life_wine_enrich':
            text, parts, buttons = _enrich(store, payload)
        else:
            raise ValueError('Неизвестное задание раздела «Жизнь».')
        ledger.require_owner(store, actor)
        wine_context = payload.get('return_to') == 'wc:menu'
        if not wine_context and payload.get('receipt_token'):
            wine_context = store.get_setting(f'receipt_ui:{actor}:{payload["receipt_token"]}', {}).get('return_to') == 'wc:menu'
        return_action = {'text': 'Моя винотека', 'callback_data': 'wc:menu'} if wine_context else {'text': 'Доходы и расходы', 'callback_data': 'lf:home'}
        quality = {'feature': 'life_finance', 'actor_id': actor, 'owner_id': owner,
                   'life_actions': buttons+[[return_action]]}
        if job['kind'] == 'life_wine_enrich':
            quality['required_features'] = ['wine_cellar']
        store.execute('''INSERT OR IGNORE INTO reports(id,section,created_at,keyword_version,cohort_version,
            text,full_text,quality,parts) VALUES(?,'life',?,1,0,?,?,?,?)''',
                      (report_id, utcnow(), text, text, dumps(quality), dumps(parts)))
    if not dry_run and not payload.get('automatic'):
        prepare_delivery(store, report_id, str(actor), detail='full')
    return report_id
