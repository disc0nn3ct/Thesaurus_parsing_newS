"""Lagged FX models: window/feature/penalty selection happens inside each past fold.

Unknown publication vintages are explicitly retrospective. Conservative release
delays prevent same-day joins; they do not reconstruct historical data vintages.
"""
import json

import numpy as np
import pandas as pd

from . import market_analysis as analysis
from . import market_store as repository

HISTORY = {'auto': 'Авто: сравнить окна', 'full': 'Вся загруженная история', 'recent': 'После 13.06.2024'}
ENGINES = {'multivariate': 'Лаги показателей + ridge', 'basic': 'Только история USD'}
BREAKS = ('1998-01-01', '2014-11-10', '2022-02-24', '2024-06-13')
LAGS = {'fred.brent': 7, 'fred.us_rate': 3, 'fred.usd_broad': 10}


def factors(store, identities, *, now):
    result, excluded = [], []
    for identity in identities:
        row = store.one('SELECT * FROM market_series WHERE id=? AND enabled=1', (identity,))
        if not row or identity == 'cbr.fx.usd':
            continue
        meta = json.loads(row['metadata'])
        if meta['frequency'] != 'daily' or meta.get('data_type') in {'archive', 'official_forecast'}:
            excluded.append({'id': identity, 'reason': 'Редкая частота: используется только в контексте'})
            continue
        observations = repository.history(store, identity, as_of=now)
        result.append({'id': identity, 'title': row['title'], 'metadata': meta, 'rows': observations,
                       'assumed_lag_days': LAGS.get(identity, 2),
                       'retrospective': any(not r.get('published_at') for r in observations)})
    return result, excluded


def align_features(index, inputs, now):
    """Native changes become usable only on their publication/assumed-release date."""
    columns = {}
    for item in inputs:
        meta = item['metadata']
        native = analysis.values(item['rows'], meta, cutoff=now)
        good = native.dropna()
        if good.empty:
            continue
        native = native.reindex(analysis.grid(good.index[0], native.index[-1], meta))
        changes = analysis.transformed(native, meta)
        by_date = {row['date']: row for row in item['rows']}
        releases = []
        for stamp, change in changes.dropna().items():
            row = by_date.get(stamp.date().isoformat(), {})
            if row.get('published_at'):
                published = pd.Timestamp(row['published_at'])
                published = published.tz_localize('UTC') if published.tzinfo is None else published.tz_convert('UTC')
                usable = published.tz_localize(None).ceil('D')
            else:
                usable = pd.Timestamp(row.get('period_end') or stamp) + pd.Timedelta(days=item['assumed_lag_days'])
            usable = max(usable, stamp)
            releases.append({'usable': usable, 'change': change})
        if not releases:
            continue
        events = pd.DataFrame(releases).sort_values('usable').drop_duplicates('usable', keep='last')
        known = pd.merge_asof(pd.DataFrame({'usable': index}), events, on='usable', direction='backward',
                              tolerance=pd.Timedelta(days=14))['change']
        known.index = index
        for lag in (1, 2, 5):
            columns[item['id']+':lag'+str(lag)] = known.shift(lag-1)
    return pd.DataFrame(columns, index=index)


def design(series, external):
    positive = series.where(series > 0)
    log = np.log(positive)
    x = pd.DataFrame(index=series.index)
    for lag in (1, 2, 5):
        x['usd:lag'+str(lag)] = log-log.shift(lag)
    for boundary in BREAKS:
        x['regime:'+boundary] = (x.index >= pd.Timestamp(boundary)).astype(float)
    return pd.concat([x, external], axis=1)


def fit_predict(series, x, step, window, alpha):
    """Direct log-return regression. Gaps in the target are never joined."""
    n = len(series)
    positive = series.where(series > 0)
    target = np.log(positive.shift(-step)/positive)
    valid_path = positive.notna().rolling(step+1).sum().shift(-step).eq(step+1)
    own = [name for name in x if name.startswith('usd:')]
    eligible = valid_path & target.notna() & x[own].notna().all(axis=1)
    if window == 'recent':
        eligible &= series.index >= pd.Timestamp('2024-06-13')
    elif window == 'three_years':
        eligible &= np.arange(n) >= max(0, n-756)
    train = x.loc[eligible]
    y = target.loc[eligible].to_numpy()
    if len(train) < 120 or x[own].iloc[-1].isna().any():
        return None
    optional = [name for name in x if name not in own and train[name].notna().mean() >= .9
                and np.isfinite(x[name].iloc[-1]) and train[name].std() > 1e-12]
    # Correlation screening is fitted on this training window, never the held-out target.
    correlation = {name: float(train[name].corr(target.loc[eligible])) for name in optional}
    ranked = sorted(optional, key=lambda name: (-abs(correlation[name]) if np.isfinite(correlation[name]) else 0, name))[:6]
    selected = own+ranked
    medians = train[selected].median()
    matrix = train[selected].fillna(medians).to_numpy()
    center, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale < 1e-12] = 1
    z = (matrix-center)/scale
    intercept = float(y.mean())
    coefficient = np.linalg.solve(z.T@z+alpha*np.eye(len(selected)), z.T@(y-intercept))
    last = (x[selected].iloc[-1].fillna(medians).to_numpy()-center)/scale
    predicted = float(series.iloc[-1]*np.exp(intercept+last@coefficient))
    if not np.isfinite(predicted) or predicted <= 0:
        return None
    return {'prediction': predicted, 'model': 'lagged_ridge', 'window': window, 'alpha': alpha,
            'features': [{'name': name, 'training_correlation': correlation.get(name),
                          'standardized_coefficient': float(weight)} for name, weight in zip(selected, coefficient)],
            'training_start': train.index[0].date().isoformat(), 'training_end': train.index[-1].date().isoformat(),
            'training_n': len(train)}


def select(series, x, step, history, cache=None):
    cache = {} if cache is None else cache
    def fitted(prefix, candidate):
        key = (len(prefix), step, *candidate)
        if key not in cache:
            cache[key] = fit_predict(prefix, x.iloc[:len(prefix)], step, *candidate)
        return cache[key]
    windows = ['full'] if history == 'full' else ['recent'] if history == 'recent' else ['recent', 'three_years', 'full']
    candidates = [(window, alpha) for window in windows for alpha in (1., 10., 100.)]
    # One group of non-overlapping past targets; all tuning stays inside this prefix.
    spacing = max(step, 7)
    stops = list(range(max(200, len(series)-step-8*spacing), len(series)-step+1, spacing))[-8:]
    errors = {candidate: [] for candidate in candidates}
    naive = []
    for stop in stops:
        actual = series.iloc[stop+step-1]
        if not np.isfinite(actual) or series.iloc[stop-1:stop+step].isna().any():
            continue
        naive.append(abs(actual-series.iloc[stop-1]))
        for candidate in candidates:
            trained = fitted(series.iloc[:stop], candidate)
            errors[candidate].append(abs(actual-trained['prediction']) if trained else np.inf)
    baseline = {'prediction': float(series.iloc[-1]), 'model': 'last', 'window': history,
                'training_start': series.index[0].date().isoformat(), 'training_end': series.index[-1].date().isoformat(),
                'training_n': int(series.notna().sum()), 'features': [], 'alpha': None}
    if len(naive) < 5 or np.mean(naive) == 0:
        return {**baseline, 'selection_reason': 'Недостаточно проверок или нулевая ошибка базового варианта',
                'selection_n': len(naive)}
    best = min(candidates, key=lambda candidate: np.mean(errors[candidate]) if errors[candidate] else np.inf)
    diagnostics = {'selection_n': len(naive), 'inner_baseline_mae': float(np.mean(naive)),
                   'inner_candidate_mae': float(np.mean(errors[best])) if np.isfinite(np.mean(errors[best])) else None,
                   'candidate_window': best[0], 'candidate_alpha': best[1]}
    if np.mean(errors[best]) > .95*np.mean(naive):
        return {**baseline, **diagnostics, 'selection_reason': 'Нет улучшения MAE на 5% во внутренней проверке'}
    trained = fitted(series, best)
    return {**(trained or baseline), **diagnostics,
            'selection_reason': 'Пройдена внутренняя проверка' if trained else 'Недостаточно данных для итогового обучения'}


def forecast(rows, metadata, *, now, today, inputs=(), history='auto', targets=None):
    series = analysis.values(rows, {k: v for k, v in metadata.items() if k != 'break_date'}, cutoff=now)
    series = series.where(series > 0)
    good = series.dropna()
    if len(good) < 60:
        return []
    series = series.reindex(analysis.grid(good.index[0], series.index[-1], metadata))
    if pd.isna(series.iloc[-1]):
        return []
    # Historic calendar gaps exclude individual samples; they don't discard all pre-gap history.
    if history == 'recent':
        series = series.loc[series.index >= '2024-06-13']
    if len(series) < 60 or series.iloc[-60:].notna().mean() < .9:
        return []
    external = align_features(series.index, inputs, now)
    x = design(series, external)
    result = []
    cache = {}
    fitted_cache = {}
    for days, target in (targets or [(days, today+pd.Timedelta(days=days)) for days in (1, 2, 3, 7)]):
        step = len(analysis.grid(series.index[-1]+pd.Timedelta(days=1), pd.Timestamp(target), metadata))
        if step not in cache:
            errors, naive, models = [], [], []
            spacing = max(step, 7)
            stops = list(range(max(260, len(series)-step-10*spacing), len(series)-step+1, spacing))[-10:] if step else []
            for stop in stops:
                actual = series.iloc[stop+step-1]
                if not np.isfinite(actual) or series.iloc[stop-1:stop+step].isna().any():
                    continue
                fitted = select(series.iloc[:stop], x.iloc[:stop], step, history, fitted_cache) if step else {'prediction': series.iloc[stop-1], 'model': 'calendar_carry'}
                errors.append(actual-fitted['prediction'])
                naive.append(actual-series.iloc[stop-1])
                models.append(fitted['model'])
            selected = select(series, x, step, history, fitted_cache) if step else {'prediction': float(series.iloc[-1]), 'model': 'calendar_carry', 'features': []}
            # The outer evaluation governs publication too; inner gains alone don't suffice.
            outer_mae = float(np.mean(np.abs(errors))) if errors else None
            naive_mae = float(np.mean(np.abs(naive))) if naive else None
            if selected['model'] == 'lagged_ridge' and (len(errors) < 10 or outer_mae > .95*naive_mae):
                selected = {**selected, 'prediction': float(series.iloc[-1]), 'model': 'last', 'features': [], 'rejected': 'Нет улучшения MAE на 5% во внешней проверке'}
                selected['candidate_mae'] = outer_mae
            if selected['model'] == 'last':
                selected['selection_pipeline_mae'] = outer_mae
                errors, outer_mae = naive, naive_mae
            width = float(np.quantile(np.abs(errors), .8)) if len(errors) >= 10 else None
            cache[step] = (selected, errors, outer_mae, naive_mae, width, models)
        selected, errors, mae, baseline_mae, width, models = cache[step]
        predicted = selected['prediction']
        result.append({**selected, 'origin': series.index[-1].date().isoformat(), 'target': target.isoformat(),
                       'baseline': float(series.iloc[-1]), 'lower': max(.01, predicted-width) if width is not None else None,
                       'upper': predicted+width if width is not None else None, 'validation_mae': mae,
                       'baseline_mae': baseline_mae, 'validation_n': len(errors), 'native_step': step, 'days_ahead': days,
                       'interval_level': .8 if width is not None else None, 'created_at': now,
                       'duration': {'unit': 'days', 'value': days}, 'horizon': len(result)+1,
                       'outer_models': models, 'history_mode': history, 'model_family': 'lagged_ridge_windows',
                       'retrospective': any(item['retrospective'] for item in inputs),
                       'note': 'Внешняя последовательная проверка; лаги, корреляции, окно и alpha выбираются только на прошлом'})
    return result
