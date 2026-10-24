"""Short FX horizons with nested chronological validation and retail cost assumptions."""
from datetime import timedelta

import numpy as np
import pandas as pd

from . import market_analysis as analysis


def choose(training, step):
    scores = {name: [] for name in ('last', 'drift', 'ridge')}
    stops = list(range(max(60, len(training)-step-60), len(training)-step+1, 5))[-12:]
    for stop in stops:
        for name, errors in scores.items():
            predicted = analysis.model_prediction(training[:stop], step, name)
            errors.append(abs(float(training[stop+step-1]-predicted)))
    if len(scores['last']) < 10 or np.mean(scores['last']) == 0:
        return 'last'
    best = min(scores, key=lambda name: np.mean(scores[name]))
    return best if np.mean(scores[best]) <= .95*np.mean(scores['last']) else 'last'


def forecast(rows, metadata, *, now, today):
    series = analysis.values(rows, metadata, cutoff=now)
    good = series.dropna()
    if good.empty:
        return []
    series = series.reindex(analysis.grid(good.index.min(), series.index.max(), metadata))
    if series.notna().mean() < .9 or pd.isna(series.iloc[-1]):
        return []
    gaps = np.flatnonzero(series.isna())
    if len(gaps):
        series = series.iloc[gaps[-1]+1:]
    if len(series) < 60:
        return []
    y = series.to_numpy()
    points = []
    for days in (1, 2, 3, 7):
        target = today+timedelta(days=days)
        step = len(analysis.grid(series.index[-1]+pd.Timedelta(days=1), pd.Timestamp(target), metadata))
        errors, baseline = [], []
        if step:
            # Outer origins are never used for model selection at their own origin.
            stops = list(range(max(120, len(y)-step-80), len(y)-step+1, 5))[-16:]
            for stop in stops:
                name = choose(y[:stop], step)
                predicted = analysis.model_prediction(y[:stop], step, name)
                actual = float(y[stop+step-1])
                errors.append(actual-predicted)
                baseline.append(actual-float(y[stop-1]))
            name = choose(y, step)
            prediction = max(.01, analysis.model_prediction(y, step, name))
        else:
            name, prediction = 'calendar_carry', float(y[-1])
        width = float(np.quantile(np.abs(errors), .8)) if len(errors) >= 10 else None
        points.append({'origin': series.index[-1].date().isoformat(), 'target': target.isoformat(),
                       'prediction': prediction, 'baseline': float(y[-1]), 'model': name,
                       'lower': max(.01, prediction-width) if width is not None else None,
                       'upper': prediction+width if width is not None else None,
                       'validation_mae': float(np.mean(np.abs(errors))) if errors else None,
                       'baseline_mae': float(np.mean(np.abs(baseline))) if baseline else None,
                       'validation_n': len(errors), 'native_step': step, 'days_ahead': days,
                       'interval_level': .8 if width is not None else None, 'created_at': now,
                       'duration': {'unit': 'weeks', 'value': 1}, 'horizon': len(points)+1,
                       'note': 'Внешняя последовательная проверка; выбор модели внутри каждого обучающего окна'})
    return points


def comparison(points, config, *, spot, days, ready):
    amount = max(0., config['amount']-config['owned'])
    result = []
    for point in points:
        row = {**point, 'cost': None, 'saving': None, 'error_rub': None, 'assessment': 'Нужен курс банка для расчёта в рублях.'}
        if config['quote'] is not None and spot > 0:
            rate = config['quote']*(1+config['fee_pct']/100)
            current = amount*rate+(config['fee_fixed'] if amount else 0)
            # Assumed relative bank markup; retail spread volatility is not measured by CBR MAE.
            later_rate = rate*point['prediction']/spot
            interest = amount*rate*config['rub_yield']/100*point['days_ahead']/365
            row['cost'] = amount*later_rate+(config['fee_fixed'] if amount else 0)-interest
            row['saving'] = current-row['cost']
            mae = point['validation_mae']
            row['error_rub'] = amount*rate/spot*mae if mae is not None else None
            if days is not None and point['days_ahead'] > days:
                row['assessment'] = 'Позже вашего срока покупки.'
            elif not ready:
                row['assessment'] = 'Нужны срок покупки и свежие данные/цена банка.'
            elif point.get('known_published'):
                row['assessment'] = 'Курс ЦБ уже опубликован; будущую цену обмена подтвердите у банка.'
            elif point['native_step'] == 0:
                row['assessment'] = 'Нового курса ЦБ по календарю не ожидается; цена банка может меняться.'
            elif point['validation_n'] < 10:
                row['assessment'] = 'Недостаточно проверочных ошибок для решения.'
            elif abs(row['saving']) <= row['error_rub']:
                row['assessment'] = 'Разница меньше типичной ошибки: преимущества ожидания не видно.'
            elif row['saving'] > 0:
                row['assessment'] = 'Модель допускает экономию при ожидании; нужна проверка цены банка.'
            else:
                row['assessment'] = 'Модель не показывает экономии от ожидания.'
        result.append(row)
    return result
