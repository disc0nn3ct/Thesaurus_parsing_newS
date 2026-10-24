"""Personal USD history windows and forecast horizons, independent of storage."""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

MOSCOW = ZoneInfo('Europe/Moscow')
FIRST_DATE = date(1992, 7, 1)
FIELDS = {
    'forecast_date': ('Дата прогноза',
                      ('Введите дату прогноза: ДД.ММ.ГГГГ или ГГГГ-ММ-ДД. Она должна быть позже конца выбранной истории, '
                      'не более чем на 5 лет. Это дата оценки курса ЦБ, а не срок вашей покупки. '
                      '«Сбросить» — вернуться к прогнозу на заданное число дней.'), None, None),
    'history_start': ('Начало истории',
                      'Введите дату начала: ГГГГ-ММ-ДД или ДД.ММ.ГГГГ. «Вся история» — с первого доступного значения.',
                      None, None),
    'history_end': ('Конец истории',
                    'Введите дату окончания: ГГГГ-ММ-ДД или ДД.ММ.ГГГГ. «Сегодня» — обновлять конец периода автоматически.',
                    None, None),
    'forecast_days': ('Срок прогноза, дней',
                      'Введите целое число дней от 1 до 90. Прогноз отсчитывается от конца выбранного периода.',
                      1, 90),
}
_START_CLEAR = {'', '-', 'all', 'все', 'всё', 'вся', 'вся история', 'с начала', 'начало'}
_END_CLEAR = {'', '-', 'today', 'сегодня', 'до сегодня', 'до сегодняшнего дня'}
_MONTH_LABELS = {1: '1 месяц', 3: '3 месяца', 6: '6 месяцев', 12: '1 год',
                 24: '2 года', 36: '3 года', 60: '5 лет'}


def date_target(config, anchor):
    """A fixed calendar endpoint; never round to a month or silently roll it forward."""
    anchor = _date(anchor)
    target = _date(config['forecast_date'])
    try:
        last = anchor.replace(year=anchor.year+5)
    except ValueError:
        last = anchor.replace(year=anchor.year+5, day=28)
    if not anchor < target <= last:
        raise ValueError('Дата прогноза должна быть позже конца истории и не далее 5 лет. '
                         'Выберите новую дату через «Курс на дату» или другой срок прогноза.')
    if config.get('analysis_months', 0):
        raise ValueError('Выберите один режим: конкретная дата либо срок в месяцах')
    return (target-anchor).days, target


def _date(value):
    if isinstance(value, datetime):
        raise ValueError('Нужна дата без времени: ГГГГ-ММ-ДД или ДД.ММ.ГГГГ')  # noqa: TRY004 - form validation uses ValueError
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError('Нужна дата ГГГГ-ММ-ДД или ДД.ММ.ГГГГ')  # noqa: TRY004 - form validation uses ValueError
    value = value.strip()
    for fmt in ('%Y-%m-%d', '%d.%m.%Y'):
        try:
            parsed = datetime.strptime(value, fmt).replace(tzinfo=MOSCOW).date()
        except ValueError:
            continue
        if parsed.strftime(fmt) == value:
            return parsed
    raise ValueError('Нужна дата ГГГГ-ММ-ДД или ДД.ММ.ГГГГ')


def _now(value):
    try:
        stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError
        return stamp.astimezone(MOSCOW)
    except (TypeError, ValueError):
        raise ValueError('Время расчёта должно содержать часовой пояс') from None


def validate(config, *, now):
    """Validate persisted settings too, so forged/stale callbacks cannot bypass bounds."""
    today = _now(now).date()
    start = _date(config['history_start']) if config.get('history_start') is not None else None
    end = _date(config['history_end']) if config.get('history_end') is not None else today
    if any(day < FIRST_DATE or day > today for day in (start, end) if day is not None):
        raise ValueError('История должна быть в пределах от 01.07.1992 до сегодня (Москва)')
    if start is not None and start > end:
        raise ValueError('Начало истории не может быть позже окончания')
    days = config.get('forecast_days', 7)
    if type(days) is not int or not 1 <= days <= 90:
        raise ValueError('Введите целое число дней от 1 до 90')
    months = config.get('analysis_months', 0)
    if type(months) is not int or not 0 <= months <= 60:
        raise ValueError('Выберите срок прогноза от 1 до 60 месяцев')
    if config.get('forecast_date') is not None:
        date_target(config, end)


def change(config, field, text, *, now):
    """Return validated settings without mutating the caller's personal configuration."""
    if field not in FIELDS:
        raise ValueError('Неизвестная настройка периода')
    if not isinstance(text, str):
        raise ValueError('Введите дату или целое число дней')  # noqa: TRY004 - form validation uses ValueError
    result = dict(config)
    value = text.strip().casefold()
    if field == 'forecast_date':
        result[field] = None if value in {'', '-', 'сбросить', 'сброс', 'нет'} else _date(text).isoformat()
        result['analysis_months'] = 0
    elif field == 'forecast_days':
        if not value.isascii() or not value.isdigit():
            raise ValueError('Введите целое число дней от 1 до 90')
        result[field] = int(value)
        result['analysis_months'] = 0
        result['forecast_date'] = None
    else:
        clear = _START_CLEAR if field == 'history_start' else _END_CLEAR
        result[field] = None if value in clear else _date(text).isoformat()
    validate(result, now=now)
    return result


def resolve(config, *, now):
    validate(config, now=now)
    stamp = _now(now)
    start = _date(config['history_start']) if config.get('history_start') is not None else None
    end = _date(config['history_end']) if config.get('history_end') is not None else stamp.date()
    retrospective = end < stamp.date()
    cutoff = datetime.combine(end, time.max, tzinfo=MOSCOW) if retrospective else stamp
    return {'start': start.isoformat() if start else None, 'end': end.isoformat(),
            'cutoff': cutoff.isoformat(), 'retrospective': retrospective}


def clip(rows, window):
    """Keep available observations, including explicit gaps; no fill or interpolation.

    The source's publication time, when known, bounds information availability.
    Retrieval timestamps cannot reconstruct historical revisions and are not used
    as publication dates; callers must label such calculations retrospective.
    """
    start = _date(window['start']) if window.get('start') is not None else None
    end = _date(window['end'])
    cutoff = _now(window['cutoff'])
    kept = []
    for row in rows:
        day = _date(row['date'])
        period_end = _date(row.get('period_end') or row['date'])
        if day > end or (start is not None and day < start) or period_end > end:
            continue
        published = row.get('published_at')
        if published:
            try:
                # Date-only source publications are available from local midnight.
                stamp = datetime.fromisoformat(published) if isinstance(published, str) else published
                if isinstance(stamp, date) and not isinstance(stamp, datetime):
                    stamp = datetime.combine(stamp, time.min, tzinfo=MOSCOW)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=MOSCOW)
                if stamp > cutoff:
                    continue
            except (AttributeError, TypeError, ValueError):
                raise ValueError('Некорректная дата публикации наблюдения') from None
        kept.append(row)
    return kept


def history_label(config):
    start = _date(config['history_start']).strftime('%d.%m.%Y') if config.get('history_start') else 'начала истории'
    end = _date(config['history_end']).strftime('%d.%m.%Y') if config.get('history_end') else 'сегодня'
    return f'С {start} до {end}'


def horizon_label(config):
    if config.get('forecast_date'):
        return 'на '+_date(config['forecast_date']).strftime('%d.%m.%Y')
    months = config.get('analysis_months', 0)
    return _MONTH_LABELS.get(months, f'{months} мес.') if months else f"{config.get('forecast_days', 7)} дн."


def targets(config, today):
    """Return (days ahead, calendar date) pairs, including the exact endpoint."""
    anchor = _date(today)
    if config.get('forecast_date') is not None:
        return [date_target(config, anchor)]
    days = config.get('forecast_days', 7)
    if type(days) is not int or not 1 <= days <= 90:
        raise ValueError('Введите целое число дней от 1 до 90')
    months = config.get('analysis_months', 0)
    if type(months) is not int or not 0 <= months <= 60:
        raise ValueError('Выберите срок прогноза от 1 до 60 месяцев')
    offsets = [1, 2, 3, 7] if months else sorted({d for d in (1, 2, 3, 7, 14, 30, 60, 90, days) if d <= days})
    return [(offset, anchor + timedelta(days=offset)) for offset in offsets]
