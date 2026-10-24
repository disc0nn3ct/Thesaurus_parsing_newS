"""Bounded monthly FX forecasts with nested, horizon-specific past validation.

Monthly closes are the last valid fixing in a sufficiently covered completed
month, not monthly averages. No daily interpolation or unknown future factors.
"""
import numpy as np
import pandas as pd

from . import market_analysis as analysis

PRESETS = {0: 'Только ближайшие дни', 1: '1 месяц', 3: '3 месяца', 6: '6 месяцев',
           12: '1 год', 24: '2 года', 36: '3 года', 60: '5 лет'}
METHODS = {'auto': 'Авто: сравнить методы', 'last': 'Последний курс',
           'damped': 'Затухающий тренд', 'ridge': 'Ridge с лагами USD'}
LABELS = {'last': 'Последний курс', 'damped': 'Затухающий тренд', 'ridge': 'Ridge с лагами USD'}


def validate(months, method):
    if type(months) is not int or not 0 <= months <= 60 or method not in METHODS:
        raise ValueError('Выберите срок 1–60 месяцев и метод кнопками')


def monthly_closes(rows, metadata, *, now, today):
    native = analysis.values(rows, {k: v for k, v in metadata.items() if k != 'break_date'}, cutoff=now)
    native = native.where(native > 0)
    good = native.dropna()
    if good.empty:
        return pd.Series(dtype=float)
    # A current incomplete month cannot masquerade as a historical monthly close.
    end = pd.Timestamp(today).to_period('M').start_time-pd.Timedelta(days=1)
    if end < good.index[0]:
        return pd.Series(dtype=float)
    native = native.reindex(analysis.grid(good.index[0].to_period('M').start_time, end, metadata))
    closes = {}
    for period, group in native.groupby(native.index.to_period('M')):
        valid = group.dropna()
        closes[period.to_timestamp('M')] = (float(valid.iloc[-1]) if len(valid) and
            group.notna().mean() >= .9 and (period.end_time.normalize()-valid.index[-1]).days <= 7 else np.nan)
    return pd.Series(closes, dtype=float).sort_index()


def design(series):
    log = np.log(series)
    return pd.DataFrame({f'USD:изменение_{lag}м': (log-log.shift(lag)).where(
        series.notna().rolling(lag+1).sum().eq(lag+1)) for lag in (1, 3, 12)})


def fit(series, x, horizon, candidate):
    """All labels finish within this prefix; scaling is fitted here too."""
    model, window, alpha = candidate
    start = max(0, len(series)-120) if window == 'ten_years' else 0
    if window == 'recent':
        positions = np.flatnonzero(series.index >= '2024-07-01')
        start = int(positions[0]) if len(positions) else len(series)
    training = series.iloc[start:]
    base = {'model': model, 'window': window, 'alpha': alpha, 'features': [],
            'training_start': training.index[0].date().isoformat() if len(training) else None,
            'training_end': series.index[-1].date().isoformat(), 'training_n': int(training.notna().sum())}
    if model == 'last':
        return {**base, 'prediction': float(series.iloc[-1])}
    if len(training) < 36 or training.iloc[-13:].isna().any():
        return None
    if model == 'damped':
        changes = np.log(training).diff().dropna()
        if len(changes) < 24:
            return None
        exponent = float(changes.mean()*sum(.9**i for i in range(1, horizon+1)))
        prediction = float(series.iloc[-1]*np.exp(exponent))
    else:
        target = np.log(series.shift(-horizon)/series)
        path = series.notna().rolling(horizon+1).sum().shift(-horizon).eq(horizon+1)
        eligible = path & target.notna() & x.notna().all(axis=1) & (np.arange(len(series)) >= start)
        matrix = x.loc[eligible].to_numpy()
        y = target.loc[eligible].to_numpy()
        if len(y) < 36 or x.iloc[-1].isna().any():
            return None
        center, scale = matrix.mean(axis=0), matrix.std(axis=0)
        scale[scale < 1e-12] = 1.
        z = (matrix-center)/scale
        weights = np.linalg.solve(z.T@z+alpha*np.eye(len(scale)), z.T@(y-y.mean()))
        exponent = float(y.mean()+((x.iloc[-1].to_numpy()-center)/scale)@weights)
        prediction = float(series.iloc[-1]*np.exp(exponent)) if abs(exponent) < 700 else np.inf
        base.update(training_n=len(y), training_end=series.index[np.flatnonzero(eligible)[-1]+horizon].date().isoformat(),
                    features=[{'name': name, 'coefficient': float(weight)} for name, weight in zip(x, weights)])
    return {**base, 'prediction': prediction} if np.isfinite(prediction) and prediction > 0 else None


def origins(series, horizon, limit):
    # Disjoint target intervals (not a claim of statistical independence).
    return sorted(range(len(series)-horizon, 47, -horizon))[-limit:]


def available(series, stop, horizon):
    return series.iloc[stop-1:stop+horizon].notna().all()


def select(series, x, horizon, history, method, cached):
    windows = ['full'] if history == 'full' else ['recent'] if history == 'recent' else ['full', 'ten_years', 'recent']
    candidates = [('last', windows[0], 0.)]
    if method != 'last':
        for window in windows:
            if method in {'auto', 'damped'}:
                candidates.append(('damped', window, 0.))
            if method in {'auto', 'ridge'}:
                candidates.extend(('ridge', window, alpha) for alpha in (10., 100.))
    def fitted(stop, candidate):
        key = (stop, horizon, candidate)
        if key not in cached:
            cached[key] = fit(series.iloc[:stop], x.iloc[:stop], horizon, candidate)
        return cached[key]
    stops = [s for s in origins(series, horizon, 6) if available(series, s, horizon)]
    errors = {c: [] for c in candidates}
    for stop in stops:
        actual = series.iloc[stop+horizon-1]
        for candidate in candidates:
            point = fitted(stop, candidate)
            errors[candidate].append(abs(actual-point['prediction']) if point else np.inf)
    baseline = fitted(len(series), candidates[0])
    if len(stops) < 5:
        return {**baseline, 'selection_n': len(stops)}
    best = min(candidates, key=lambda c: np.mean(errors[c]))
    if np.mean(errors[best]) >= .95*np.mean(errors[candidates[0]]):
        best = candidates[0]
    return {**(fitted(len(series), best) or baseline), 'selection_n': len(stops)}


def forecast(rows, metadata, *, now, today, months, history='auto', method='auto', distribution=True):
    validate(months, method)
    if history not in {'auto', 'full', 'recent'}:
        raise ValueError('Неизвестное окно истории')
    series = monthly_closes(rows, metadata, now=now, today=today)
    if history == 'recent':
        series = series.loc[series.index >= '2024-07-01']
    if not months or series.empty or pd.isna(series.iloc[-1]) or series.iloc[-12:].notna().mean() < .9:
        return []
    x, cached, result = design(series), {}, []
    steps = sorted({m for m in PRESETS if 0 < m <= months} | {months})
    for horizon in steps:
        errors, baseline_errors, folds = [], [], []
        stops = origins(series, horizon, 12)
        for stop in stops:
            if not available(series, stop, horizon):
                continue
            chosen = select(series.iloc[:stop], x.iloc[:stop], horizon, history, method, cached)
            actual = float(series.iloc[stop+horizon-1])
            errors.append(actual-chosen['prediction'])
            baseline_errors.append(actual-float(series.iloc[stop-1]))
            folds.append({'origin': series.index[stop-1].date().isoformat(),
                          'target': series.index[stop+horizon-1].date().isoformat(),
                          'training_end': chosen['training_end'], 'model': chosen['model']})
        selected = select(series, x, horizon, history, method, cached)
        pipeline_mae = float(np.mean(np.abs(errors))) if errors else None
        baseline_mae = float(np.mean(np.abs(baseline_errors))) if baseline_errors else None
        if selected['model'] != 'last' and (len(errors) < 10 or pipeline_mae >= .95*baseline_mae):
            selected = {**fit(series, x, horizon, ('last', 'recent' if history == 'recent' else 'full', 0.)),
                        'selection_n': selected['selection_n'],
                        'rejected': 'Недостаточно проверок или нет улучшения MAE на 5%'}
        if selected['model'] == 'last':
            errors = baseline_errors
        mae = float(np.mean(np.abs(errors))) if errors else None
        width = float(np.quantile(np.abs(errors), .8)) if len(errors) >= 10 else None
        # Fixed reference variants are inspectable even if the selection rejects
        # them. These are research scenarios, never substituted for the winner.
        alternatives = []
        window = 'recent' if history == 'recent' else 'full'
        for model, alpha in (('last', 0.), ('damped', 0.), ('ridge', 10.)):
            candidate = (model, window, alpha)
            reference = fit(series, x, horizon, candidate)
            if not reference:
                continue
            reference_errors = []
            for stop in stops:
                if not available(series, stop, horizon):
                    continue
                key = (stop, horizon, candidate)
                if key not in cached:
                    cached[key] = fit(series.iloc[:stop], x.iloc[:stop], horizon, candidate)
                past = cached[key]
                if past:
                    reference_errors.append(abs(series.iloc[stop+horizon-1]-past['prediction']))
            alternatives.append({'model': model, 'prediction': reference['prediction'], 'window': window,
                                 'alpha': alpha, 'validation_n': len(reference_errors),
                                 'mae': float(np.mean(reference_errors)) if reference_errors else None,
                                 'research_only': True})
        from .dollar_distribution import forecast as distribution_forecast
        scenarios = distribution_forecast(series, horizon, history) if distribution else None
        result.append({**selected, 'origin': series.index[-1].date().isoformat(),
            'target': (series.index[-1]+pd.offsets.MonthEnd(horizon)).date().isoformat(),
            'prediction': selected['prediction'], 'baseline': float(series.iloc[-1]),
            'lower': max(.01, selected['prediction']-width) if width is not None else None,
            'upper': selected['prediction']+width if width is not None else None,
            'validation_n': len(errors), 'validation_mae': mae, 'baseline_mae': baseline_mae,
            'selection_pipeline_mae': pipeline_mae, 'folds': folds, 'alternatives': alternatives, 'distribution': scenarios,
            'horizon': horizon, 'duration': {'unit': 'months', 'value': horizon},
            'created_at': now, 'frequency': 'monthly', 'requested_method': method,
            'history_mode': history, 'model_family': 'monthly_usd', 'retrospective': True,
            'interval_level': .8 if width is not None else None,
            'note': 'Последние курсы полных месяцев; внешние проверки на неперекрывающихся периодах. '
                    'Выбор окна и метода внутри прошлого; диапазон по прошлым ошибкам, не гарантия. '
                    'Разные валютные режимы ограничивают сопоставимость всей истории.'})
    return result


def describe(points, months):
    lines = [f'Долгосрочный анализ · до {months} мес.']
    if not points:
        return lines+['Расчёт недоступен: нужна актуальная история полных месяцев с охватом от 90%; обновление данных выполняет администратор.']
    lines.append(f"Отправная точка: {points[0]['origin']} — последний полный месяц. Значения на конец месяца; это не курс на каждый день.")
    for p in points:
        text = f"{p['horizon']} мес. · {p['target']}: {p['prediction']:.2f} ₽/USD · {LABELS[p['model']]}"
        if p['validation_mae'] is not None:
            text += f"; MAE {p['validation_mae']:.2f} ₽, базовый MAE {p['baseline_mae']:.2f} ₽; проверок {p['validation_n']}"
        if p['lower'] is not None:
            text += f"\nДиапазон 80% по прошлым ошибкам: {p['lower']:.2f}–{p['upper']:.2f} ₽/USD"
        if p['validation_n'] < 10:
            text += '\nНедостаточно проверок: условный сценарий без диапазона.'
        elif p['model'] == 'last':
            text += ('\nБазовый метод выбран вами; сохранение последнего курса не обещает его неизменность.'
                     if p['requested_method'] == 'last' else
                     '\nВыбранные сложные методы не доказали преимущество; одинаковые ориентиры не обещают неизменный курс.')
        lines.append(text)
        if p.get('distribution'):
            from .dollar_distribution import describe as distribution_description
            lines.extend(distribution_description(p['distribution']))
    alternatives = points[-1].get('alternatives', [])
    if alternatives:
        lines.append(f"Другие расчётные варианты на {points[-1]['horizon']} мес. (исследование, не рекомендация):")
        for variant in alternatives:
            error = f"; MAE {variant['mae']:.2f} ₽, проверок {variant['validation_n']}" if variant['mae'] is not None else '; нет прошлых проверок'
            lines.append(f"{LABELS[variant['model']]}: {variant['prediction']:.2f} ₽/USD{error}.")
        window = 'периоде после 13.06.2024' if points[-1]['history_mode'] == 'recent' else 'всей загруженной истории'
        lines.append(f'Это фиксированные варианты на {window}, без автоматического сравнения окон. '
                     'Число проверок может различаться; ошибки на разных выборках нельзя напрямую ранжировать. '
                     'Основной результат выше сохраняет проверку преимущества перед последним курсом.')
    lines.append('MAE сложного варианта оценивает процедуру выбора на прошлом; методы внутри прошлых окон могли различаться. '
                 'Сравниваются последний курс, затухающий тренд и прямая ridge-регрессия с месячными лагами USD. '
                 'Будущие значения экономических показателей не подставляются. Их связи остаются в краткосрочной модели и контексте. '
                 'На годах особенно важны смены валютного режима; прежняя ошибка не ограничивает будущие потери. '
                 'Проверки ретроспективные, периоды не перекрываются, но не считаются независимыми.')
    if any(p.get('distribution') for p in points):
        lines.append('Медиана делит сценарии пополам и не является самым вероятным точным курсом. '
                     'Центральные 80% сценариев истории — не проверенный прогнозный интервал. '
                     'Доли при повторении прошлого могут не сохраниться в будущем. '
                     'Для сценариев авто берёт 10 лет при достаточной выборке, иначе всю историю; '
                     'это правило применяется и внутри каждой проверки.')
    return lines


def chart(series, points, path):
    """Discrete checked horizons, not an invented daily forecast trajectory."""
    from matplotlib import dates
    from matplotlib.figure import Figure
    figure = Figure(figsize=(11, 6), layout='constrained', dpi=140)
    ax = figure.subplots()
    origin = pd.Timestamp(points[0]['origin'])
    recent = series.loc[series.index >= origin-pd.DateOffset(years=5)]
    ax.plot(recent.index, recent, color='tab:blue', linewidth=1, label='Факты ЦБ · до 5 лет')
    ax.axvline(origin, color='grey', linestyle=':', alpha=.6)
    labels = set()
    for i, point in enumerate(points):
        stamp = pd.Timestamp(point['target'])
        checked = point['validation_n'] >= 10
        baseline = point['model'] == 'last'
        label = ('Базовый ориентир · последний курс' if checked else 'Базовый сценарий · мало проверок') if baseline else (
            'Модель · проверка от 10 периодов' if checked else 'Условный сценарий · мало проверок')
        color = '#707070' if baseline or not checked else 'tab:orange'
        ax.scatter([stamp], [point['prediction']], facecolors='none' if baseline else color,
                   edgecolors=color, marker=('s' if checked else 'D') if baseline else 'o' if checked else 'D',
                   label=label if label not in labels else None, zorder=4)
        labels.add(label)
        ax.annotate(f"{point['horizon']}м", (stamp, point['prediction']), xytext=(0, 12 if i % 2 == 0 else -20),
                    textcoords='offset points', fontsize=9, ha='center')
        if checked and point['lower'] is not None and point['upper'] is not None:
            interval_label = 'Ошибки базового метода · диапазон 80%' if baseline else 'Диапазон 80% · прошлые ошибки модели'
            ax.vlines(stamp, point['lower'], point['upper'], color=color, linewidth=3, alpha=.4,
                      label=interval_label if interval_label not in labels else None)
            labels.add(interval_label)
    title = f"USD/RUB · долгосрочный анализ до {points[-1]['horizon']} мес."
    if all(point['model'] == 'last' for point in points):
        manual = all(point.get('requested_method') == 'last' for point in points)
        title += ('\nБазовый метод выбран вами; неизменность курса не предсказана' if manual else
                  '\nНаправление не определено — показан последний курс полного месяца')
    ax.set(title=title, ylabel='Рублей за 1 USD',
           xlabel=f"Полный месяц: {origin:%m.%Y} · ориентиры на конец месяца; не ежедневная траектория")
    ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(alpha=.2)
    ax.legend(loc='best', fontsize=9)
    figure.savefig(path, dpi=140)
    return {'kind': 'photo', 'path': str(path), 'caption': 'Долгосрочный USD/RUB: серые пустые маркеры — последний курс полного месяца, а не предсказание неизменности. Ромбы означают нехватку проверок. Оранжевые точки — сложная модель, прошедшая проверки. Полосы отражают прошлые ошибки соответствующего метода; при нехватке проверок полос нет. Это ориентиры ЦБ, не цена банка.'}
