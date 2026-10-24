"""Explain point estimates separately from historical forecast errors."""
from math import isfinite

POINT_EXPLANATION = (
    'Точка — результат выбранного метода, а не наиболее вероятный курс. '
    'Положение внутри полосы не показывает вероятность отдельных значений.'
)
INTERVAL_EXPLANATION = (
    'Полоса: оценка ± 80-й процентиль прошлых абсолютных ошибок. '
    'Это не гарантия попадания будущего курса в диапазон.'
)


def _style(point):
    if point.get('known_published'):
        return 'Опубликованный курс ЦБ', 'purple', 'D', True
    if point['model'] == 'calendar_carry':
        return 'Действующий курс · календарь', 'dimgray', 'x', True
    if point['model'] == 'last':
        return 'Базовый ориентир · последний курс', 'dimgray', 's', False
    if point.get('validation_n', 0) >= 10:
        return 'Точечная оценка модели', 'tab:orange', 'o', True
    return 'Оценка · недостаточно проверок', 'dimgray', 'D', False


def _has_interval(point):
    lower, upper = point.get('lower'), point.get('upper')
    return (not point.get('known_published') and point['model'] != 'calendar_carry'
            and point.get('validation_n', 0) >= 10 and lower is not None and upper is not None
            and isfinite(lower) and isfinite(upper) and lower <= upper)


def _horizons(series, points):
    import pandas as pd
    return [int(point.get('days_ahead', (pd.Timestamp(point['target'])-series.index[-1]).days))
            for point in points]


def _title(horizons, points, period):
    title = 'USD/RUB · '
    if period and period.get('retrospective'):
        title += 'ретроспектива · '
    if len(points) == 1:
        from datetime import date
        title += 'курс на '+date.fromisoformat(points[0]['target']).strftime('%d.%m.%Y')
    elif horizons:
        low, high = min(horizons), max(horizons)
        title += f'прогноз на {low}–{high} дн.' if low != high else f'прогноз на {high} дн.'
    else:
        title += 'выбранные даты прогноза'
    modeled = [p for p in points if not p.get('known_published') and p['model'] != 'calendar_carry']
    if modeled and all(p['model'] == 'last' for p in modeled):
        title += '\nБазовый ориентир; направление не определено'
    return title


def _observed(series, period):
    import pandas as pd
    observed = series
    if period:
        if period.get('start'):
            observed = observed.loc[observed.index >= pd.Timestamp(period['start'])]
        end = period.get('history_end') or period.get('end')
        if end:
            observed = observed.loc[observed.index < pd.Timestamp(end)+pd.Timedelta(days=1)]
    explicit = period and (period.get('explicit') or period.get('start') or period.get('history_end') or period.get('retrospective'))
    if not explicit and not observed.empty:
        observed = observed.loc[observed.index >= observed.index[-1]-pd.Timedelta(days=30)]
    label = 'Официальный курс · выбранная история' if explicit else 'Официальный курс · 30 дней'
    return observed, label


def _footer(footer, axis, labels, points):
    footer.set_axis_off()
    handles, names = axis.get_legend_handles_labels()
    footer.legend(handles, names, loc='upper left', frameon=False, ncol=2, fontsize=10)
    modeled = any(not p.get('known_published') and p['model'] != 'calendar_carry' for p in points)
    note = ('Расчётная отметка — оценка метода, не «самый вероятный курс».\n' if modeled else
            'Показаны опубликованный курс или его действие по календарю.\n')
    if 'interval' in labels:
        note += 'Полоса — ± 80-й процентиль прошлых абсолютных ошибок;\nэто не гарантия будущего курса.'
    elif modeled:
        note += 'Диапазон не показан: недостаточно проверочных ошибок.'
    footer.text(.015, .02, note, transform=footer.transAxes, fontsize=10, va='bottom')


def _part(path, points, *, long=False, retrospective=False):
    start = 'Исторический расчёт: ' if retrospective else 'Выбранные даты: '
    descriptions = {
        ('purple', 'D', True): 'фиолетовый ромб — уже опубликованный курс ЦБ',
        ('dimgray', 'x', True): 'серый крестик — действующий курс по календарю',
        ('dimgray', 's', False): 'серый пустой квадрат — базовый ориентир «последний курс»',
        ('tab:orange', 'o', True): 'оранжевая точка — оценка модели с достаточной проверкой',
        ('dimgray', 'D', False): 'серый пустой ромб — оценка с недостаточной проверкой',
    }
    # Use the same resolved styles as the plotted marks, including published-rate
    # precedence, so the caption never advertises marks absent from this figure.
    styles = dict.fromkeys(_style(point)[1:] for point in points)
    caption = start+'; '.join(descriptions[style] for style in styles)+'. '
    if long:
        caption += 'Каждая строка — отдельная опорная дата, без ежедневной траектории. '
    modeled = any(not p.get('known_published') and p['model'] != 'calendar_carry' for p in points)
    if modeled:
        caption += POINT_EXPLANATION+' '
    if any(_has_interval(point) for point in points):
        caption += INTERVAL_EXPLANATION+' '
    caption += 'Значения относятся к курсу ЦБ, не к цене банка.'
    return {'kind': 'photo', 'path': str(path), 'caption': caption}


def _horizon_chart(series, points, path, *, period=None):
    """Separate long forecast dates so dense endpoints cannot hide one another."""
    from matplotlib import dates
    from matplotlib.figure import Figure

    single = len(points) == 1
    figure = Figure(figsize=(10, 7.5 if single else 10), dpi=140, layout='constrained')
    history, axis, footer = figure.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1.5 if single else 4, 2]})
    observed, label = _observed(series, period)
    history.plot(observed.index, observed, color='tab:blue', label=label)
    history.set(title=label, ylabel='Рублей за 1 USD')
    history.xaxis.set_major_formatter(dates.ConciseDateFormatter(history.xaxis.get_major_locator()))
    history.grid(alpha=.2)
    ordered = sorted(zip(points, _horizons(series, points)), key=lambda item: item[0]['target'])
    labels, ticks = set(), []
    for index, (point, days) in enumerate(ordered):
        value = point['prediction']
        label, color, marker, filled = _style(point)
        style = {'color': color} if filled else {'facecolors': 'white', 'edgecolors': color}
        axis.scatter([value], [index], marker=marker, s=55, linewidths=1.5, zorder=4,
                     label=label if label not in labels else None, **style)
        labels.add(label)
        if _has_interval(point):
            axis.hlines(index, point['lower'], point['upper'], color=color, alpha=.35, linewidth=5,
                        label='Диапазон по прошлым ошибкам · 80%' if 'interval' not in labels else None)
            labels.add('interval')
        axis.annotate(f'{value:.2f}', (value, index), xytext=(7, 7), textcoords='offset points',
                      fontsize=9, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8, 'pad': 1})
        ticks.append(f"{point['target']} · +{days} дн.")
    axis.set_yticks(range(len(ticks)), ticks)
    axis.set_ylim(len(ticks)-.5, -.5)
    axis.set(title='Оценки на опорные даты · не ежедневная траектория', xlabel='Рублей за 1 USD')
    axis.margins(x=.12)
    axis.grid(axis='x', alpha=.2)
    _footer(footer, axis, labels, points)
    figure.suptitle(_title([days for _, days in ordered], points, period), fontsize=12)
    figure.savefig(path, dpi=140)
    return _part(path, points, long=True, retrospective=bool(period and period.get('retrospective')))


def near_term(series, points, path, *, period=None):
    """Render frozen estimates without changing their values or implied method."""
    import pandas as pd
    from matplotlib import dates
    from matplotlib.figure import Figure

    horizons = _horizons(series, points)
    observed, history_label = _observed(series, period)
    extended_history = len(observed) > 1 and (observed.index[-1]-observed.index[0]).days > 90
    if (horizons and max(horizons) > 7) or extended_history:
        return _horizon_chart(series, points, path, period=period)
    figure = Figure(figsize=(10, 7), dpi=140, layout='constrained')
    axis, footer = figure.subplots(2, 1, gridspec_kw={'height_ratios': [5, 2]})
    footer.set_axis_off()
    axis.plot(observed.index, observed, color='tab:blue', label=history_label)
    labels = set()
    for i, point in enumerate(points):
        stamp, value = pd.Timestamp(point['target']), point['prediction']
        label, color, marker, filled = _style(point)
        style = {'color': color} if filled else {'facecolors': 'white', 'edgecolors': color}
        axis.scatter([stamp], [value], marker=marker, s=55, linewidths=1.5, zorder=4,
                     label=label if label not in labels else None, **style)
        labels.add(label)
        axis.annotate(f"{stamp:%d.%m}\n{value:.2f}", (stamp, value),
                      xytext=(0, 14 if i % 2 == 0 else -35), textcoords='offset points',
                      ha='center', fontsize=9,
                      bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 1})
        # Published rates and calendar carry are facts, not uncertain model outputs.
        # Fail closed on stale/unvalidated/non-finite bounds; never invent a range.
        if _has_interval(point):
            axis.vlines(stamp, point['lower'], point['upper'], color=color, alpha=.35, linewidth=4,
                        label='Диапазон по прошлым ошибкам · 80%' if 'interval' not in labels else None)
            labels.add('interval')
    title = _title(horizons, points, period)
    axis.set(title=title, ylabel='Рублей за 1 USD')
    axis.xaxis.set_major_formatter(dates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    axis.margins(x=.07, y=.18)
    axis.grid(alpha=.2)
    _footer(footer, axis, labels, points)
    figure.savefig(path, dpi=140)
    return _part(path, points, retrospective=bool(period and period.get('retrospective')))
