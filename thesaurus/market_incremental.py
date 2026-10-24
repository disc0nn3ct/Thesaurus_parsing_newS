"""Plan missing market periods; successful values are immutable during refresh."""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import market_analysis as analysis
from .cbr_incremental import ranges


def period(day, frequency):
    stamp = pd.Timestamp(day)
    if frequency == 'monthly':
        stamp = stamp.to_period('M').start_time
    elif frequency == 'quarterly':
        stamp = stamp.to_period('Q').start_time
    elif frequency == 'annual':
        stamp = stamp.to_period('Y').start_time
    return stamp.date()


def members(store, item):
    rows = store.query("SELECT id,metadata FROM market_series WHERE status<>'archived' AND (id=? OR dataset=?) "
                       "AND json_extract(metadata,'$.kind')='series'", (item['id'], item['id']))
    return [row for row in rows if json.loads(row['metadata']).get('params', {}) == item.get('params', {})]


def plan(store, item, start, end, now=None):
    now = now or datetime.now(timezone.utc)
    children = members(store, item)
    state = store.get_setting('market_incremental:' + item['id'], {})
    if state.get('params', {}) != item.get('params', {}):
        state = {}
    if item.get('calendar') == 'events' or item['frequency'] not in analysis.FREQUENCIES:
        missing_days = [stamp.date() for stamp in pd.date_range(start, end, freq='D') if not any(
            interval[0] <= stamp.date().isoformat() <= interval[1] for interval in state.get('covered', []))]
        return ranges([stamp.date() for stamp in pd.date_range(start, end, freq='D')], set(missing_days))[:64]
    if not children:
        try:
            if datetime.fromisoformat(state['empty_until']) > now:
                return []
        except (KeyError, ValueError, TypeError):
            pass
        return [(start, end)]
    missing = set()
    all_days = set()
    for child in children:
        metadata = json.loads(child['metadata'])
        frequency = metadata['frequency']
        if frequency not in {'daily', 'weekly', 'monthly', 'quarterly', 'annual'}:
            return [(start, end)]
        days = [day.date() for day in analysis.grid(start, end, metadata)]
        all_days.update(days)
        known = {period(row['date'], frequency) for row in store.query(
            "SELECT date FROM market_observations WHERE series_id=? AND (value IS NOT NULL OR quality IN ('source_removed_confirmed','source_no_observation'))", (child['id'],))}
        floor = state.get('floors', {}).get(child['id']) if start.isoformat() >= state.get('window_start', start.isoformat()) else None
        checks = [check for check in state.get('empty', {}).get(child['id'], []) if check['until'] > now.isoformat()]
        confirmed = state.get('confirmed_absent', {}).get(child['id'], [])
        for day in days:
            if day in known or floor and day.isoformat() < floor:
                continue
            if any(check['start'] <= day.isoformat() <= check['end'] for check in checks):
                continue
            if any(first <= day.isoformat() <= last for first, last in confirmed):
                continue
            missing.add(day)
    batches = ranges(sorted(all_days), missing)[:64]
    offset = {'monthly': pd.offsets.MonthEnd(0), 'quarterly': pd.offsets.QuarterEnd(0),
              'annual': pd.offsets.YearEnd(0)}.get(item['frequency'])
    if offset is not None:
        batches = [(first, min(end, (pd.Timestamp(last) + offset).date())) for first, last in batches]
    return batches


def remember(store, item, first, last, window_start, now=None):
    now = now or datetime.now(timezone.utc)
    state = store.get_setting('market_incremental:' + item['id'], {})
    if state.get('params', {}) != item.get('params', {}):
        state = {}
    state['params'] = item.get('params', {})
    if item.get('calendar') == 'events' or item['frequency'] not in analysis.FREQUENCIES:
        state.setdefault('covered', []).append([first.isoformat(), last.isoformat()])
    children = members(store, item)
    if not children:
        state['empty_until'] = (now + timedelta(hours=1)).isoformat()
    for child in children:
        metadata = json.loads(child['metadata'])
        frequency = metadata['frequency']
        if frequency not in {'daily', 'weekly', 'monthly', 'quarterly', 'annual'}:
            continue
        days = [day.date() for day in analysis.grid(first, last, metadata)]
        known = {period(row['date'], frequency) for row in store.query(
            'SELECT date FROM market_observations WHERE series_id=? AND value IS NOT NULL', (child['id'],))}
        if first <= window_start and known:
            state.setdefault('floors', {})[child['id']] = min(known).isoformat()
        floor = state.get('floors', {}).get(child['id'])
        missing = {day for day in days if day not in known and (not floor or day.isoformat() >= floor)}
        # Only validated successful responses reach remember(). Old official gaps
        # remain gaps, but do not trigger a historic re-download every hour.
        historical = {day for day in missing if day < (now.date() - timedelta(days=30))}
        intervals = state.setdefault('confirmed_absent', {}).get(child['id'], [])
        intervals.extend([begin.isoformat(), finish.isoformat()] for begin, finish in ranges(days, historical))
        merged = []
        for begin, finish in sorted(intervals):
            if merged and begin <= (pd.Timestamp(merged[-1][1]) + pd.Timedelta(days=1)).date().isoformat():
                merged[-1][1] = max(merged[-1][1], finish)
            else:
                merged.append([begin, finish])
        state['confirmed_absent'][child['id']] = merged
        missing -= historical
        checks = [check for check in state.get('empty', {}).get(child['id'], []) if check['until'] > now.isoformat()]
        checks.extend({'start': begin.isoformat(), 'end': finish.isoformat(),
                       'until': (now + timedelta(hours=1)).isoformat()} for begin, finish in ranges(days, missing))
        state.setdefault('empty', {})[child['id']] = checks
    state['window_start'] = min(state.get('window_start', window_start.isoformat()), window_start.isoformat())
    store.set_setting('market_incremental:' + item['id'], state)


def only_missing(store, identity, observations):
    known = {row['date'] for row in store.query(
        "SELECT date FROM market_observations WHERE series_id=? AND (value IS NOT NULL OR quality IN ('source_removed_confirmed','source_no_observation'))", (identity,))}
    return [row for row in observations if row['date'] not in known]
