"""Allowlisted diagnostics: never display arbitrary transport exception text."""
from requests import HTTPError, RequestException

MESSAGES = {
    'Ambiguous duplicate market observations': ('source_conflict', 'Источник вернул разные значения на одну дату; требуется проверка целостности.'),
    'CBR table disappeared': ('table_missing', 'На странице нет ожидаемой таблицы; сохранённая история доступна.'),
    'CBR table headers changed': ('schema_changed', 'Изменились заголовки таблицы; требуется проверка парсера.'),
    'CBR wide table dimensions changed': ('schema_changed', 'Изменились измерения таблицы; требуется проверка парсера.'),
    'CBR table width changed': ('schema_changed', 'Изменилось число колонок таблицы; требуется проверка парсера.'),
    'CBR wide table width changed': ('schema_changed', 'Изменилось число колонок таблицы; требуется проверка парсера.'),
    'CBR numeric schema changed': ('numeric_format', 'В числовой колонке встретился неизвестный формат; данные не подменены нулями.'),
    'No verified numeric table contract': ('adapter_missing', 'Для набора ещё нет проверенного числового парсера.'),
    'Invalid market response': ('invalid_response', 'Источник вернул перенаправление или слишком большой ответ.'),
}


def diagnostic(exc):
    if isinstance(exc, HTTPError):
        status = getattr(exc.response, 'status_code', None)
        return 'http_error', f'Источник вернул HTTP {status}.' if type(status) is int else 'Источник вернул ошибку HTTP.'
    if isinstance(exc, RequestException):
        return 'network_error', 'Не удалось связаться с источником; сохранённая история доступна.'
    if type(exc) is ValueError and str(exc) in MESSAGES:
        return MESSAGES[str(exc)]
    from .cbr_incremental import IncompleteCollection
    if isinstance(exc, IncompleteCollection):
        return 'backfill_pending', 'Часть истории сохранена; остались недостающие периоды для следующей дозагрузки.'
    return 'collection_error', 'Обновление не выполнено (' + type(exc).__name__ + '); сохранённая история доступна.'
