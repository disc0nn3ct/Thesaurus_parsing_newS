"""Small local models and explicit frequency alignment, never LLM calculations."""
import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .russian_calendar import calendar as working_calendar

FREQUENCIES = {'daily': 0, 'weekly': 1, 'monthly': 2, 'quarterly': 3, 'annual': 4}
MINIMUM = {'daily': 60, 'weekly': 26, 'monthly': 24, 'quarterly': 8, 'annual': 5}
WINDOWS = {'daily': (60, 120, 252), 'weekly': (26, 52), 'monthly': (24, 60), 'quarterly': (8, 20), 'annual': (5, 10)}


def grid(start, end, metadata):
    frequency = metadata['frequency']
    if frequency not in FREQUENCIES:
        return pd.DatetimeIndex([])
    if frequency == 'daily':
        days = pd.date_range(start, end, freq='D')
        if metadata.get('calendar') == 'calendar':
            return days
        if metadata.get('calendar') == 'weekdays':
            return days[days.dayofweek < 5]
        calendar = working_calendar()
        if metadata.get('calendar') == 'cbr_fx':
            return pd.DatetimeIndex([d for d in days if calendar.is_working_day((d-pd.Timedelta(days=1)).date())])
        return pd.DatetimeIndex([d for d in days if calendar.is_working_day(d.date())])
    return pd.date_range(start, end, freq={'weekly': 'W-' + metadata.get('week_anchor', 'FRI'), 'monthly': 'MS', 'quarterly': 'QS', 'annual': 'YS'}[frequency])


def values(rows, metadata, *, cutoff=None):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    frame['date'] = pd.to_datetime(frame['date'])
    if cutoff is not None:
        timestamp = pd.Timestamp(cutoff)
        timestamp = timestamp.tz_localize('UTC') if timestamp.tzinfo is None else timestamp.tz_convert('UTC')
        effective_cutoff = timestamp.tz_convert(metadata.get('date_timezone', 'UTC')).tz_localize(None)
        frame = frame[frame['date'] <= effective_cutoff]
        if 'period_end' in frame:
            frame = frame[pd.to_datetime(frame['period_end'].fillna(frame['date'])) <= effective_cutoff]
        if 'published_at' in frame:
            published = pd.to_datetime(frame['published_at'], utc=True)
            frame = frame[published.isna() | (published <= timestamp)]
    if metadata.get('break_date'):
        frame = frame[frame['date'] >= pd.Timestamp(metadata['break_date'])]
    if 'definition' in frame and not frame.empty:
        frame = frame[frame['definition'] == frame.sort_values('date').iloc[-1]['definition']]
    if metadata.get('source') == 'google_trends' and metadata['frequency'] == 'weekly' and 'period_end' in frame:
        # Google labels a week by its beginning; its value covers the following six days.
        frame['date'] = pd.to_datetime(frame['period_end'])
    if metadata['frequency'] == 'monthly':
        frame['date'] = frame['date'].dt.to_period('M').dt.start_time
    elif metadata['frequency'] in {'quarterly', 'annual'}:
        frame['date'] = frame['date'].dt.to_period('Q' if metadata['frequency'] == 'quarterly' else 'Y').dt.start_time
    return frame.drop_duplicates('date', keep='last').set_index('date')['value'].astype(float).sort_index()


def aggregate(series, frequency, metadata, *, week_anchor='FRI'):
    if frequency == metadata['frequency']:
        return series
    resampler = series.resample({'weekly': 'W-' + week_anchor, 'monthly': 'MS', 'quarterly': 'QS', 'annual': 'YS'}[frequency])
    method = metadata.get('aggregation', 'last')
    # Resample only observed values. A period with inadequate native coverage is missing.
    result = resampler.sum(min_count=1) if method == 'sum' else getattr(resampler, method)()
    for period in result.index:
        end = period if frequency == 'weekly' else period + {'monthly': pd.offsets.MonthEnd(1), 'quarterly': pd.offsets.QuarterEnd(1), 'annual': pd.offsets.YearEnd(1)}[frequency]
        start = end - pd.Timedelta(days=6) if frequency == 'weekly' else period
        expected = len(grid(start, end, metadata))
        observed = series.loc[(series.index >= start) & (series.index <= end)].notna().sum()
        if expected and observed / expected < .9:
            result.loc[period] = np.nan
    return result


def transformed(series, metadata, levels=False):
    if levels:
        return series
    if metadata.get('transform') == 'returns':
        positive = series.where(series > 0)
        return np.log(positive).diff()
    return series.diff()


def correlations(left, right, left_meta, right_meta, *, window=None, lag=0, levels=False):
    if left_meta['frequency'] not in FREQUENCIES or right_meta['frequency'] not in FREQUENCIES:
        return {'status': 'unsupported_frequency'}
    frequency = max((left_meta['frequency'], right_meta['frequency']), key=FREQUENCIES.get)
    if frequency not in WINDOWS:
        return {'status': 'unsupported_frequency'}
    window = window or WINDOWS[frequency][0]
    if window not in WINDOWS[frequency] or not -4 <= lag <= 4:
        raise ValueError('Invalid correlation window or lag')
    rare_meta = left_meta if left_meta['frequency'] == frequency else right_meta
    anchor = rare_meta.get('week_anchor', 'FRI') if frequency == 'weekly' else 'FRI'
    a = aggregate(values(left, left_meta), frequency, left_meta, week_anchor=anchor)
    b = aggregate(values(right, right_meta), frequency, right_meta, week_anchor=anchor)
    if a.empty or b.empty:
        return {'status': 'insufficient', 'n': 0}
    if frequency == 'weekly':
        a.index = a.index.to_period('W-' + anchor).end_time.normalize()
        b.index = b.index.to_period('W-' + anchor).end_time.normalize()
    start, end = max(a.index.min(), b.index.min()), min(a.index.max(), b.index.max())
    common_meta = {**left_meta, 'frequency': frequency, 'week_anchor': anchor}
    expected = grid(start, end, common_meta)
    if frequency == 'daily':
        left_grid, right_grid = grid(start, end, left_meta), grid(start, end, right_meta)
        expected = left_grid.intersection(right_grid)
    # Compare changes over the same observed endpoints, never different native spans.
    a = transformed(a.reindex(expected), left_meta, levels)
    b = transformed(b.reindex(expected), right_meta, levels)
    paired = pd.concat([a.shift(lag).rename('a'), b.rename('b')], axis=1).tail(window)
    clean = paired.dropna()
    coverage = len(clean) / window
    result = {'status': 'insufficient', 'n': len(clean), 'coverage': coverage, 'frequency': frequency,
              'window': window, 'lag': lag, 'direction': 'left precedes right' if lag > 0 else 'right precedes left' if lag < 0 else 'simultaneous',
              'levels': levels, 'note': 'Association does not establish causality'}
    if len(clean) < MINIMUM[frequency] or coverage < .9 or clean['a'].nunique() < 2 or clean['b'].nunique() < 2:
        return result
    result['status'] = 'ok'
    for method in ('pearson', 'spearman'):
        r = float(clean['a'].corr(clean['b'], method=method))
        statistic = abs(r) * math.sqrt((len(clean) - 2) / max(1e-15, 1 - r*r))
        result[method] = r
        result[method + '_p'] = float(2 * student_t.sf(statistic, len(clean) - 2))
    return result


def fdr(results):
    tests = [(r, key, r[key + '_p']) for r in results if r.get('status') == 'ok' for key in ('pearson', 'spearman')]
    ordered = sorted(tests, key=lambda item: item[2])
    last = 1.
    for index in range(len(ordered) - 1, -1, -1):
        row, key, p = ordered[index]
        last = min(last, p * len(ordered) / (index + 1))
        row[key + '_q'] = last
    return results


def diagnostics(rows, metadata):
    series = values(rows, metadata).dropna()
    if series.empty:
        return {'status': 'missing'}
    expected = grid(series.index.min(), series.index.max(), metadata)
    result = {'status': 'ok', 'date': series.index[-1].date().isoformat(), 'value': float(series.iloc[-1]),
              'period_start': series.index[0].date().isoformat(),
              'previous_date': series.index[-2].date().isoformat() if len(series) > 1 else None,
              'unit': metadata['unit'], 'points': len(series), 'change': float(series.diff().iloc[-1]) if len(series) > 1 else None,
              'coverage': len(expected.intersection(series.index)) / max(1, len(expected)),
              'expected': len(expected), 'off_calendar': len(series.index.difference(expected)),
              'missing': len(expected.difference(series.index))}
    if metadata.get('transform') == 'returns' and (series > 0).all():
        returns = np.log(series.reindex(expected)).diff().tail(60).dropna()
        result['volatility'] = float(returns.std()) if len(returns) >= 2 else None
        result['drawdown'] = float((series / series.cummax() - 1).iloc[-1])
        result['formula'] = 'volatility=std(log(Pt/Pt-1)); drawdown=Pt/max(P0..Pt)-1'
    else:
        result['formula'] = 'change=last-previous'
    if metadata['frequency'] not in FREQUENCIES or metadata.get('calendar') == 'events':
        result.update(coverage=None, expected=None, missing=None, note='Наблюдения по отдельным событиям; календарный охват не определён')
    return result


def model_prediction(training, step, name, features=None):
    y = np.asarray(training, dtype=float)
    if name == 'last' or len(y) < 10:
        return float(y[-1])
    if name == 'drift':
        return float(y[-1] + .25 * np.median(np.diff(y[-21:])) * step)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    if len(y) - step < 8:
        return float(y[-1])
    x, target = [], []
    for i in range(4, len(y) - step + 1):
        row = list(y[i-4:i][::-1] - y[i-1])
        if features is not None:
            row += list(features[i-1])
        x.append(row)
        target.append(y[i+step-1] - y[i-1])
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    model = Ridge(alpha=10.).fit(x, target)
    last = list(y[-4:][::-1] - y[-1])
    if features is not None:
        last += list(features[-1])
    return float(y[-1] + model.predict(scaler.transform([last]))[0])


def forecast(rows, metadata, *, created_at, features=None, horizon=None):
    def bounded(value):
        limits = metadata.get('value_bounds')
        return float(np.clip(value, *limits)) if limits else value

    created = pd.Timestamp(created_at)
    created = created.tz_localize('UTC') if created.tzinfo is None else created.tz_convert('UTC')
    origin = created.tz_localize(None).normalize()
    series = values(rows, metadata, cutoff=created.isoformat())
    if (metadata['frequency'] not in MINIMUM or metadata.get('unit') == 'source_units'
            or metadata.get('data_type') in {'reference', 'official_forecast', 'archive'} or series.dropna().empty):
        return []
    if metadata.get('administrative'):
        from .forecast_horizons import from_options, targets, validate
        requested = validate(horizon) if horizon is not None else from_options({}, metadata['frequency'])
        last = series.dropna().iloc[-1]
        base_date = origin if metadata.get('calendar') == 'events' else series.dropna().index[-1]
        return [{'origin': series.dropna().index[-1].date().isoformat(), 'target': stamp.date().isoformat(),
                 'horizon': i, 'native_step': step, 'duration': requested, 'prediction': float(last),
                 'baseline': float(last), 'lower': None, 'upper': None, 'model': 'last', 'created_at': created_at,
                 'validation_n': 0, 'validation_mae': None, 'baseline_mae': None, 'interval_level': None,
                 'note': 'Сценарий сохранения административного значения'}
                for i, (step, stamp) in enumerate(targets(base_date, {**metadata, 'calendar': 'calendar'}, requested), 1)
                if stamp > origin]
    # Unavailable prehistory is not a hole inside the observed training span.
    # Keep trailing missing periods: they prevent issuing a forecast from stale facts.
    series = series.reindex(grid(series.dropna().index.min(), series.index.max(), metadata))
    minimum = MINIMUM[metadata['frequency']]
    if len(series.dropna()) < minimum or series.notna().mean() < .9 or pd.isna(series.iloc[-1]):
        return []
    # Never bridge holes in model training or treat future-filled regressors as observations.
    gaps = np.flatnonzero(series.isna().to_numpy())
    if len(gaps):
        series = series.iloc[gaps[-1]+1:]
    if len(series) < minimum:
        return []
    external = None
    if features:
        aligned = []
        for data, meta in features:
            # Regressors with known delayed publication cannot masquerade as same-day inputs.
            eligible = [row for row in data if not row.get('published_at') or
                        pd.Timestamp(row['published_at']).date() <= pd.Timestamp(row['date']).date()]
            aligned.append(values(eligible, meta, cutoff=created.isoformat()).reindex(series.index))
        if all(data.notna().all() for data in aligned):
            external = np.column_stack([data.to_numpy() for data in aligned])
    from .forecast_horizons import from_options, targets, validate
    requested = validate(horizon) if horizon is not None else from_options({}, metadata['frequency'])
    future = targets(series.index[-1], metadata, requested)
    y = series.to_numpy()
    predictions = []
    for position, (step, target_stamp) in enumerate(future, 1):
        scores = {name: [] for name in ('last', 'drift', 'ridge')}
        validation_points = 120 if metadata['frequency'] == 'daily' else 52
        spacing = 5 if metadata['frequency'] == 'daily' else 1
        first_stop = max(minimum, len(y)-step-validation_points)
        stops = list(range(first_stop, len(y)-step+1, spacing))[-24:]
        for stop in stops:
            for name, errors in scores.items():
                predicted = bounded(model_prediction(y[:stop], step, name, external[:stop] if external is not None else None))
                errors.append(float(y[stop+step-1] - predicted))
        baseline_mae = np.mean(np.abs(scores['last'])) if scores['last'] else None
        chosen = 'last'
        if len(scores['last']) >= 10 and baseline_mae is not None and baseline_mae > 0 and not metadata.get('administrative'):
            best = min(scores, key=lambda n: np.mean(np.abs(scores[n])))
            if np.mean(np.abs(scores[best])) <= .95 * baseline_mae:
                chosen = best
        predicted = bounded(model_prediction(y, step, chosen, external))
        if metadata.get('transform') == 'returns' or metadata.get('field') in {'VOLUME', 'VALUE', 'volume'}:
            predicted = max(0., predicted)
        errors = scores[chosen]
        width = float(np.quantile(np.abs(errors), .8)) if len(errors) >= 10 else None
        target = target_stamp.date().isoformat()
        if target <= origin.date().isoformat():
            continue
        predictions.append({'origin': series.index[-1].date().isoformat(), 'target': target, 'horizon': position,
            'native_step': step, 'duration': requested, 'validation_n': len(errors),
            'note': 'Сценарий сохранения значения' if metadata.get('administrative') else
                    'Базовый прогноз: недостаточно проверочных ошибок' if len(errors) < 10 else 'Последовательная проверка на истории',
            'prediction': predicted, 'baseline': float(y[-1]), 'lower': max(0., predicted-width) if width is not None and metadata.get('transform') == 'returns' else predicted-width if width is not None else None,
            'upper': predicted+width if width is not None else None, 'model': chosen, 'created_at': created_at,
            'validation_mae': float(np.mean(np.abs(errors))) if errors else None,
            'baseline_mae': float(baseline_mae) if baseline_mae is not None else None,
            'interval_level': .8 if width is not None else None})
        if metadata.get('value_bounds'):
            for key in ('lower', 'upper'):
                if predictions[-1][key] is not None:
                    predictions[-1][key] = bounded(predictions[-1][key])
    return predictions


def backtest(rows, metadata, max_origins=12, *, horizon=None):
    series = values(rows, metadata).dropna()
    results = []
    spacing = 5 if metadata['frequency'] == 'daily' else 1
    for origin in series.index[MINIMUM.get(metadata['frequency'], 24)::spacing][-max_origins:]:
        subset = [row for row in rows if row['date'] <= origin.date().isoformat()]
        for prediction in forecast(subset, metadata, created_at=origin.isoformat(), horizon=horizon):
            target = pd.Timestamp(prediction['target'])
            if target in series:
                results.append({**prediction, 'actual': float(series.loc[target])})
    return {'evaluation': 'retrospective_backtest', 'forecasts': results, 'metrics': evaluate(results)}


def evaluate(rows):
    actual = [r for r in rows if r.get('actual') is not None]
    if not actual:
        return {'n': 0, 'mae': None, 'wape': None, 'interval_coverage': None}
    errors = [abs(r['actual']-r['prediction']) for r in actual]
    denominator = sum(abs(r['actual']) for r in actual)
    intervals = [r for r in actual if r['lower'] is not None and r['upper'] is not None]
    return {'n': len(actual), 'mae': float(np.mean(errors)),
            'baseline_mae': float(np.mean([abs(r['actual']-r['baseline']) for r in actual])),
            'wape': sum(errors)/denominator if denominator and all(r['actual'] >= 0 for r in actual) else None,
            'interval_coverage': sum(r['lower'] <= r['actual'] <= r['upper'] for r in intervals)/len(intervals) if intervals else None}


def event_study(rows, metadata, events):
    series = values(rows, metadata).dropna()
    selected, last = [], None
    for event in sorted(set(events)):
        stamp = pd.Timestamp(event)
        if last is not None and (stamp-last).days < 14:
            continue
        i = series.index.searchsorted(stamp)
        if i < 5 or i+5 >= len(series):
            continue
        selected.append({'date': event, 'before': float(series.iloc[i]-series.iloc[i-5]), 'after': float(series.iloc[i+5]-series.iloc[i])})
        last = stamp
    return {'status': 'ok' if len(selected) >= 10 else 'insufficient', 'n': len(selected), 'events': selected,
            'note': 'Exploratory association, not causal effect'}
