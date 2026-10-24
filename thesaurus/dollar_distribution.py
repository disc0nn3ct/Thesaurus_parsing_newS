"""Empirical FX return distributions, separately from selected point forecasts.

Historical disjoint horizon returns are scenarios, not independent experiments.
No Gaussian extrapolation, invented future factors or simulated sample inflation.
"""
import numpy as np
import pandas as pd

WINDOWS = {'full': 'Вся история', 'ten_years': 'Последние 10 лет', 'recent': 'После 13.06.2024'}
THRESHOLD = .02
MINIMUM = 10


def sample(series, horizon):
    """Return ratios from complete disjoint intervals, newest first."""
    ratios, periods = [], []
    for end in range(len(series)-1, horizon-1, -horizon):
        block = series.iloc[end-horizon:end+1]
        if not np.isfinite(block).all() or (block <= 0).any():
            continue
        ratios.append(float(block.iloc[-1]/block.iloc[0]))
        periods.append({'start': block.index[0].date().isoformat(), 'end': block.index[-1].date().isoformat()})
    return np.asarray(ratios), periods


def windowed(series, window):
    if window == 'ten_years':
        return series.iloc[-121:]
    if window == 'recent':
        return series.loc[series.index >= '2024-07-01']
    return series


def classify(ratio):
    return 0 if ratio < 1-THRESHOLD else 2 if ratio > 1+THRESHOLD else 1


def estimate(series, horizon, window):
    if window == 'auto':
        return estimate(series, horizon, 'ten_years') or estimate(series, horizon, 'full')
    training = windowed(series, window)
    ratios, periods = sample(training, horizon)
    if len(ratios) < MINIMUM or not np.isfinite(series.iloc[-1]) or series.iloc[-1] <= 0:
        return None
    counts = np.bincount([classify(r) for r in ratios], minlength=3)
    # Dirichlet(1,1,1) prevents spurious 0%/100% from small samples.
    probability = (counts+1)/(len(ratios)+3)
    quantiles = np.quantile(ratios, [.1, .5, .9])
    if not np.isfinite(quantiles*series.iloc[-1]).all():
        return None
    return {'window': window, 'sample_n': len(ratios), 'sample_periods': periods,
            'training_start': training.index[0].date().isoformat(), 'training_end': training.index[-1].date().isoformat(),
            'base': float(series.iloc[-1]), 'median': float(series.iloc[-1]*quantiles[1]),
            'lower': float(series.iloc[-1]*quantiles[0]), 'upper': float(series.iloc[-1]*quantiles[2]),
            'ratio_quantiles': quantiles.tolist(), 'probabilities': {'down': float(probability[0]),
                'flat': float(probability[1]), 'up': float(probability[2])}, 'counts': counts.tolist()}


def pinball(actual, quantiles):
    errors = actual-np.asarray(quantiles)
    levels = np.array([.1, .5, .9])
    return float(np.maximum(levels*errors, (levels-1)*errors).mean())


def display_shares(probabilities):
    keys = ('down', 'flat', 'up')
    exact = np.array([probabilities[key]*100 for key in keys])
    rounded = np.floor(exact).astype(int)
    for i in np.argsort(-(exact-rounded), kind='stable')[:100-int(rounded.sum())]:
        rounded[i] += 1
    return {key: '<1%' if value == 0 else '>99%' if value == 100 else f'{value}%' for key, value in zip(keys, rounded)}


def evaluate(series, horizon, window):
    folds, brier, quantile, baseline_quantile, errors, baseline_errors, covered = [], [], [], [], [], [], []
    for stop in sorted(range(len(series)-horizon, 35, -horizon))[-20:]:
        target_path = series.iloc[stop-1:stop+horizon]
        if not np.isfinite(target_path).all() or (target_path <= 0).any():
            continue
        predicted = estimate(series.iloc[:stop], horizon, window)
        if predicted is None:
            continue
        actual = float(series.iloc[stop+horizon-1])
        ratio = actual/predicted['base']
        observed = np.eye(3)[classify(ratio)]
        probabilities = np.array([predicted['probabilities'][k] for k in ('down', 'flat', 'up')])
        brier.append(float(np.sum((probabilities-observed)**2)))
        quantile.append(pinball(ratio, predicted['ratio_quantiles']))
        baseline_quantile.append(pinball(ratio, [1., 1., 1.]))
        errors.append(abs(actual-predicted['median']))
        baseline_errors.append(abs(actual-predicted['base']))
        tolerance = abs(actual)*1e-10
        covered.append(predicted['lower']-tolerance <= actual <= predicted['upper']+tolerance)
        folds.append({'origin': series.index[stop-1].date().isoformat(), 'window': predicted['window'],
                      'target': series.index[stop+horizon-1].date().isoformat(),
                      'training_end': predicted['training_end'], 'sample_n': predicted['sample_n']})
    return {'n': len(folds), 'brier': float(np.mean(brier)) if brier else None,
            'uniform_brier': 2/3, 'quantile_score': float(np.mean(quantile)) if quantile else None,
            'point_baseline_quantile_score': float(np.mean(baseline_quantile)) if quantile else None,
            'median_mae': float(np.mean(errors)) if errors else None,
            'baseline_mae': float(np.mean(baseline_errors)) if errors else None,
            'range_coverage': float(np.mean(covered)) if covered else None, 'folds': folds}


def forecast(series, horizon, history):
    if type(horizon) is not int or not 1 <= horizon <= 60 or history not in {'auto', 'full', 'recent'}:
        raise ValueError('Неверный срок или окно сценариев')
    window = 'recent' if history == 'recent' else 'full'
    # Auto's sample rule is fixed before validation, never chosen using held-out scores.
    if history == 'auto':
        window = 'auto'
    selected = estimate(series, horizon, window)
    if selected is None:
        return {'status': 'insufficient', 'minimum_n': MINIMUM, 'method': 'empirical_horizon_returns'}
    # Auto evaluates the *same sample-size rule*, including its past fallbacks.
    validation = evaluate(series, horizon, 'auto' if window == 'auto' else window)
    enough = selected['sample_n'] >= 20 and validation['n'] >= 10
    evidence = (enough and validation['brier'] < validation['uniform_brier'] and validation['range_coverage'] >= .65
                and validation['quantile_score'] < validation['point_baseline_quantile_score'])
    return {**selected, 'status': 'evaluated' if evidence else 'research', 'validation': validation,
            'threshold': THRESHOLD, 'smoothing': 'Dirichlet(1,1,1)', 'method': 'empirical_horizon_returns',
            'history_mode': history, 'sample_rule': 'ten_years_else_full' if window == 'auto' else window,
            'retrospective': True, 'not_independent': True,
            'note': 'Оценка при переносимости прошлых изменений. Медиана не является самым вероятным точным курсом. '
                    'Диапазон 10–90% описывает исторические сценарии, а не проверенный прогнозный интервал. '
                    'Разные валютные режимы ограничивают переносимость частот на будущее.'}


def describe(distribution):
    if distribution['status'] == 'insufficient':
        return ['Распределение сценариев не оценено: меньше 10 полных неперекрывающихся исторических периодов такой длины.']
    d, v = distribution, distribution['validation']
    p = display_shares(d['probabilities'])
    lines = [(f"Сценарии истории · {WINDOWS[d['window']]}: {d['sample_n']} периодов; медиана {d['median']:.2f} ₽/USD; "
              f"центральные 80% сценариев {d['lower']:.2f}–{d['upper']:.2f} ₽/USD.")]
    lines.append(f"При повторении прошлой динамики: рост >2% — {p['up']}; в пределах ±2% — {p['flat']}; "
                 f"снижение >2% — {p['down']}. Относительно {d['base']:.2f} ₽/USD; частоты сглажены и округлены.")
    if v['n']:
        lines.append(f"Проверка распределения на прошлом: {v['n']} периодов; MAE медианы {v['median_mae']:.2f} ₽, "
                     f"MAE последнего курса {v['baseline_mae']:.2f} ₽; факт попал в диапазон в {v['range_coverage']:.0%} случаев. "
                     f"Brier {v['brier']:.3f} (меньше лучше; равные шансы трёх исходов: {v['uniform_brier']:.3f}).")
    if d['status'] == 'research':
        lines.append('Исследовательская оценка: данных или качества проверки недостаточно для подтверждения вероятностей будущего. '
                     'Проценты описывают модель частот истории, не уверенность в будущем курсе.')
    else:
        lines.append('Оценка прошла минимальную проверку частот на прошлом; это не доказательство точной калибровки '
                     'и не преимущество перед всеми другими моделями. Валютный режим может измениться.')
    return lines


def score_actual(distribution, actual):
    """Evaluate the frozen scenario, never refit it when the fact arrives."""
    if distribution.get('status') not in {'research', 'evaluated'} or not np.isfinite(actual) or actual <= 0:
        return None
    keys = ('down', 'flat', 'up')
    observed = classify(actual/distribution['base'])
    probabilities = np.array([distribution['probabilities'][key] for key in keys])
    return {'actual': float(actual), 'outcome': keys[observed],
            'brier': float(np.sum((probabilities-np.eye(3)[observed])**2)),
            'median_absolute_error': abs(actual-distribution['median']),
            'inside_scenario_range': bool(distribution['lower'] <= actual <= distribution['upper'])}


def chart(series, points, path):
    from matplotlib import dates
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter
    usable = [p for p in points if p.get('distribution', {}).get('status') in {'research', 'evaluated'}]
    figure = Figure(figsize=(11, 9), dpi=140, layout='constrained')
    price, probability = figure.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    origin = pd.Timestamp(points[0]['origin'])
    recent = series.loc[series.index >= origin-pd.DateOffset(years=2)]
    price.plot(recent.index, recent, color='#3979a6', linewidth=1, label='Факты ЦБ · до 2 лет')
    price.hlines(points[0]['baseline'], origin, pd.Timestamp(usable[-1]['target']), color='#707070',
                 linestyle='--', linewidth=1, label='Последний курс полного месяца · ориентир')
    stamps = pd.to_datetime([p['target'] for p in usable])
    median = [p['distribution']['median'] for p in usable]
    price.scatter(stamps, median, color='#875b92', marker='^', label='Медиана исторических сценариев', zorder=3)
    price.vlines(stamps, [p['distribution']['lower'] for p in usable], [p['distribution']['upper'] for p in usable],
                 color='#875b92', linewidth=3, alpha=.4, label='Центральные 80% исторических сценариев')
    for i, point in enumerate(usable):
        price.annotate(f"{point['horizon']}м", (pd.Timestamp(point['target']), point['distribution']['median']),
                       xytext=(0, 12 if i % 2 == 0 else -20), textcoords='offset points', fontsize=9, ha='center')
    price.set(title='Сценарии при повторении прошлой динамики · не подтверждённая траектория курса',
              ylabel='Рублей за 1 USD', xlabel=f"Отправная точка — полный месяц {origin:%m.%Y}; цели — концы месяцев")
    price.xaxis.set_major_formatter(dates.ConciseDateFormatter(price.xaxis.get_major_locator()))
    price.grid(alpha=.2)
    price.legend(fontsize=8, loc='best')
    positions = np.arange(len(usable))
    left = np.zeros(len(usable))
    for key, label, color in (('down', 'Снижение >2%', '#3979a6'), ('flat', 'В пределах ±2%', '#b6b6b6'),
                              ('up', 'Рост >2%', '#cda04d')):
        widths = np.array([p['distribution']['probabilities'][key]*100 for p in usable])
        probability.barh(positions, widths, left=left, color=color, label=label, height=.65)
        for row, (start, width) in enumerate(zip(left, widths)):
            if width >= 6:
                text = display_shares(usable[row]['distribution']['probabilities'])[key]
                probability.text(start+width/2, row, text, ha='center', va='center', fontsize=9)
        left += widths
    probability.set_yticks(positions, [f"{p['horizon']}м · {p['distribution']['sample_n']} периодов\n"
        f"{p['distribution']['validation']['n']} проверок\n"
        + ('минимальная проверка' if p['distribution']['status'] == 'evaluated' else 'не подтверждено') for p in usable], fontsize=9)
    probability.invert_yaxis()
    probability.set_xlim(0, 100)
    probability.xaxis.set_major_formatter(PercentFormatter(100))
    probability.set(xlabel='Относительно курса полного месяца')
    probability.set_title('Модель частот истории · сглаженные доли, не гарантии будущего', pad=40)
    probability.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncols=3, fontsize=9)
    scope = f"до {usable[-1]['horizon']} мес."
    if usable[-1]['horizon'] < points[-1]['horizon']:
        scope += f" · для {points[-1]['horizon']} мес. недостаточно истории"
    figure.suptitle('USD/RUB · исторические сценарии '+scope, fontsize=14)
    figure.savefig(path, dpi=140)
    return {'kind': 'photo', 'path': str(path), 'caption': 'USD/RUB: фиолетовые треугольники — медиана исторических сценариев, полосы — их центральные 80%. Это не проверенная траектория будущего курса. Внизу — модель частот роста/снижения относительно последнего полного месяца, с числом периодов и проверок. При недостатке истории срок не показан. Медиана не означает «самый вероятный точный курс».'}
