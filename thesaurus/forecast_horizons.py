"""Calendar durations shared by Telegram, jobs, backtests and stored forecasts."""
import pandas as pd

UNITS = {'weeks': 'нед.', 'months': 'мес.', 'quarters': 'кв.', 'years': 'лет'}


def validate(spec):
    if not isinstance(spec, dict) or spec.get('unit') not in UNITS or type(spec.get('value')) is not int:
        raise ValueError('Срок прогноза: целое число и единица времени')
    limits = {'weeks': 104, 'months': 24, 'quarters': 8, 'years': 2}
    if not 0 <= spec['value'] <= limits[spec['unit']]:
        raise ValueError('Срок прогноза: максимум 24 месяца (104 недели)')
    return {'value': spec['value'], 'unit': spec['unit']}


def end_date(origin, spec):
    spec = validate(spec)
    offset = {'weeks': pd.DateOffset(weeks=spec['value']), 'months': pd.DateOffset(months=spec['value']),
              'quarters': pd.DateOffset(months=3*spec['value']), 'years': pd.DateOffset(years=spec['value'])}
    return pd.Timestamp(origin) + offset[spec['unit']]


def from_options(options, frequency='daily'):
    if options.get('forecast') is not None:
        return validate(options['forecast'])
    unit = {'monthly': 'months', 'quarterly': 'quarters', 'annual': 'years'}.get(frequency, 'weeks')
    maximum = {'months': 3, 'quarters': 8, 'years': 2}.get(unit, 4)
    return {'value': min(options.get('horizon', 4), maximum), 'unit': unit}


def label(spec):
    spec = validate(spec)
    return f"до {spec['value']} {UNITS[spec['unit']]}" if spec['value'] else 'выключен'


def targets(origin, metadata, spec):
    """Native-calendar step counts; daily output becomes sparser beyond 12 weeks."""
    from .market_analysis import grid
    origin = pd.Timestamp(origin)
    spec = validate(spec)
    if not spec['value']:
        return []
    end = end_date(origin, spec)
    frequency = metadata['frequency']
    native = grid(origin + pd.Timedelta(days=1), end, metadata)
    if native.empty:
        return []
    if frequency != 'daily':
        return [(i+1, stamp) for i, stamp in enumerate(native)]
    marks = [origin + pd.Timedelta(weeks=i) for i in range(1, 13)]
    marks += [origin + pd.DateOffset(months=i) for i in range(4, 25)]
    marks.append(end)
    indices = {min(len(native)-1, native.searchsorted(mark, side='right')-1) for mark in marks if mark <= end}
    return [(int(i)+1, native[i]) for i in sorted(indices) if i >= 0]
