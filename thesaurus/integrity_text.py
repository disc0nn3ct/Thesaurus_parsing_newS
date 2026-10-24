"""Compact integrity notices; individual evidence stays in the incident journal."""
import json

KINDS = {'source_conflict': 'Противоречивые значения источника', 'source_revision': 'Пересмотр значений',
         'source_missing': 'Наблюдения исчезли из источника', 'source_reference_revision': 'Изменение справочника',
         'source_schema': 'Изменилась структура источника', 'source_unavailable': 'Источник недоступен',
         'sqlite_integrity': 'Ошибка целостности SQLite', 'legacy_ambiguous_series': 'Неоднозначная старая история'}
SEVERITIES = {'critical': 'Критическая', 'warning': 'Требует проверки', 'info': 'Информационная'}


def digest(store, rows):
    series = {row['id']: row for row in store.query('SELECT id,dataset,title,metadata FROM market_series')}
    groups = {}
    for row in rows:
        child = series.get(row['series_id'])
        dataset = child['dataset'] if child else row['series_id']
        groups.setdefault((row['severity'], row['kind'], dataset), []).append(row)
    ordered = sorted(groups.items(), key=lambda pair: (
        {'critical': 0, 'warning': 1, 'info': 2}.get(pair[0][0], 3), -len(pair[1]), str(pair[0][2])))
    lines = [f'Целостность данных · новых записей: {len(rows)}']
    for (severity, kind, dataset), entries in ordered[:6]:
        parent = series.get(dataset)
        title = (parent['title'] if parent else 'Локальная база SQLite' if kind == 'sqlite_integrity' else 'Источник данных')[:180]
        dates = sorted({row['date'] for row in entries if row['date']})
        count = len({row['series_id'] for row in entries if row['series_id']})
        lines += ['', f"{SEVERITIES.get(severity, 'Требует проверки')} · {title}",
                  f"{KINDS.get(kind, 'Проблема качества данных')}: {len(entries)}; показателей: {count}."]
        if dates:
            lines.append(f'Период: {dates[0]} — {dates[-1]}; дат: {len(dates)}.')
        if kind == 'source_conflict':
            lines.append('Источник указал разные значения на одну дату. Спорные новые наблюдения исключены из расчётов; остальные данные сохранены.')
        elif kind in {'source_revision', 'source_missing', 'source_reference_revision'}:
            lines.append('Сохранённые данные не заменены автоматически. Проверьте изменения перед принятием.')
        elif kind == 'sqlite_integrity':
            lines.append('Проверка SQLite не пройдена. Сбор остановлен; требуется проверка базы и резервной копии.')
        else:
            lines.append('Причина и исходные сведения сохранены в журнале.')
    if len(ordered) > 6:
        lines.append(f'\nЕщё групп: {len(ordered)-6}. Все записи доступны в журнале.')
    lines.append('\nОткрыть журнал: /integrity. Уже уведомлённые записи повторно не рассылаются.')
    return '\n'.join(lines)


def conflict_details(payload):
    candidates = payload.get('candidates', []) if isinstance(payload, dict) else []
    values = list(dict.fromkeys(str(row.get('value', '—')) for row in candidates))
    return ('Источник указал разные значения: ' + ', '.join(values[:8]) + '.\n'
            'Спорное новое наблюдение исключено из расчётов. Автоматический выбор значения не выполняется.')


def indicator_title(store, identity):
    row = store.one('SELECT title,metadata FROM market_series WHERE id=?', (identity,)) if identity else None
    if row:
        unit = json.loads(row['metadata']).get('unit')
        return row['title'] + (f' ({unit})' if unit else '')
    return identity or 'Локальная база SQLite'
