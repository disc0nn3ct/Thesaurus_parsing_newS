"""Personal market rules, persistent cooldowns and durable signal delivery."""
import json
import math
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from . import market_analysis as analysis
from . import market_store as repository
from .delivery import prepare_delivery
from .market import pair_results, selected
from .store import dumps, utcnow

KINDS = {'above', 'below', 'change', 'deviation', 'cross', 'correlation'}
DEFAULTS = {'threshold': 3., 'window': 60, 'correlation': .6, 'delta': .2, 'confirm': 2, 'coverage': .9, 'cooldown': 24}


def allowed_series(store, actor):
    """RUONIA alerts follow the actor's RUONIA filter, independently of market charts."""
    rows = {row['id']: row for row in selected(store, actor)}
    from .subscription_access import allowed
    if not allowed(store, actor, 'market'):
        return []
    if not allowed(store, actor, 'ruonia'):
        return list(rows.values())
    from .telegram_data_jobs import actor_scope
    phrases = actor_scope(store, actor, 'ruonia')['phrases']
    for row in store.query("SELECT * FROM market_series WHERE enabled=1 AND json_extract(metadata,'$.adapter')='stored_ruonia'"):
        if json.loads(row['metadata'])['metric'] in phrases:
            rows[row['id']] = row
    return list(rows.values())


def add_rule(store, user_id, identity, kind, threshold=3.):
    if kind not in KINDS or not math.isfinite(float(threshold)):
        raise ValueError('Правила: above, below, change, deviation, cross, correlation; порог — конечное число')
    if identity not in {r['id'] for r in allowed_series(store, user_id)}:
        raise ValueError('Выберите доступный показатель через /indicators')
    return store.execute('INSERT INTO market_rules(user_id,series_id,kind,config) VALUES(?,?,?,?)',
        (user_id, identity, kind, dumps({**DEFAULTS, 'threshold': float(threshold)})))


def seed(store, user_id):
    key = f'market_rules_seeded:{user_id}'
    if store.get_setting(key):
        return
    for row in allowed_series(store, user_id):
        add_rule(store, user_id, row['id'], 'deviation')
        if row['id'] in {'cbr.derived.yield_slope', 'cbr.derived.ruonia_spread'}:
            add_rule(store, user_id, row['id'], 'cross', 0)
        if row['source'] == 'moex':
            add_rule(store, user_id, row['id'], 'change', 5)
    store.set_setting(key, True)


def evaluate_rule(rows, metadata, rule, pairs=()):
    config = json.loads(rule['config'])
    window = int(config.get('window', 60))
    series = analysis.values(rows, metadata)
    if metadata['frequency'] not in analysis.FREQUENCIES:
        return None
    if len(series.dropna()) < window + 1:
        return None
    grid = analysis.grid(series.index.min(), series.index.max(), metadata)
    tail = series.reindex(grid).tail(window+1)
    if tail.notna().mean() < config.get('coverage', .9) or tail.iloc[-2:].isna().any():
        return None
    current, previous = float(tail.iloc[-1]), float(tail.iloc[-2])
    threshold, kind = config['threshold'], rule['kind']
    statistic, fired = current, False
    if kind == 'above':
        fired = current >= threshold
    elif kind == 'below':
        fired = current <= threshold
    elif kind == 'cross':
        fired = (previous < threshold <= current) or (previous > threshold >= current)
    elif kind == 'change' and previous != 0:
        statistic = (current-previous)/abs(previous)*100
        fired = abs(statistic) >= abs(threshold)
    elif kind == 'deviation':
        baseline = tail.iloc[:-1].dropna()
        std = float(baseline.std())
        if std > 0:
            statistic = (current-float(baseline.mean()))/std
            fired = abs(statistic) >= abs(threshold)
    elif kind == 'correlation':
        for pair in pairs:
            if rule['series_id'] not in {pair['left'], pair['right']} or pair.get('status') != 'ok':
                continue
            coefficient = pair['spearman']
            if abs(coefficient) >= config.get('correlation', .6) and pair.get('spearman_q', 1) <= .05:
                return {'date': tail.index[-1].date().isoformat(), 'value': current, 'statistic': coefficient,
                        'pair': pair, 'coverage': float(tail.notna().mean()), 'unit': metadata['unit'], 'kind': kind}
    if fired:
        return {'date': tail.index[-1].date().isoformat(), 'value': current, 'previous': previous, 'statistic': statistic,
                'coverage': float(tail.notna().mean()), 'unit': metadata['unit'], 'kind': kind}
    return None


def process(store, settings=None):
    now = datetime.now(timezone.utc)
    for rule in store.query("SELECT r.* FROM market_rules r JOIN users u ON u.user_id=r.user_id WHERE r.enabled=1 AND r.mode<>'off' AND u.enabled=1"):
        allowed = {r['id']: r for r in allowed_series(store, rule['user_id'])}
        if rule['series_id'] not in allowed:
            continue
        row = allowed[rule['series_id']]
        result = evaluate_rule(repository.history(store, row['id']), json.loads(row['metadata']), rule, pair_results(store, rule['user_id']))
        if not result:
            if rule['kind'] == 'correlation':
                key = f"market_confirmation:{rule['id']}"
                state = store.get_setting(key, {})
                store.set_setting(key, {**state, 'count': 0})
            continue
        if rule['kind'] == 'correlation':
            key = f"market_confirmation:{rule['id']}"
            state = store.get_setting(key, {})
            config = json.loads(rule['config'])
            pair_date = result['date']
            if state.get('date') == pair_date:
                continue
            baseline = state.get('baseline', result['statistic'])
            changed = abs(result['statistic']-baseline) >= config.get('delta', .2)
            count = state.get('count', 0)+1 if changed else 0
            store.set_setting(key, {'date': pair_date, 'baseline': baseline, 'count': count})
            if count < config.get('confirm', 2):
                continue
            store.set_setting(key, {'date': pair_date, 'baseline': result['statistic'], 'count': 0})
        previous = store.one('SELECT created_at FROM market_signals WHERE rule_id=? ORDER BY id DESC LIMIT 1', (rule['id'],))
        if previous and now-datetime.fromisoformat(previous['created_at']) < timedelta(hours=rule['cooldown']):
            continue
        store.execute('INSERT OR IGNORE INTO market_signals(rule_id,observed_date,created_at,payload) VALUES(?,?,?,?)',
                      (rule['id'], result['date'], utcnow(), dumps(result)))
    dispatch(store, settings)


def dispatch(store, settings=None):
    # Recover a crash after persisting the signal report but before queuing its delivery.
    for report in store.query("""SELECT DISTINCT s.report_id,r.user_id FROM market_signals s JOIN market_rules r ON r.id=s.rule_id
        JOIN users u ON u.user_id=r.user_id WHERE s.report_id IS NOT NULL AND u.enabled=1 AND r.enabled=1 AND r.mode<>'off'
        AND NOT EXISTS(SELECT 1 FROM deliveries d WHERE d.report_id=s.report_id AND d.recipient=CAST(r.user_id AS TEXT))"""):
        prepare_delivery(store, report['report_id'], str(report['user_id']))
    for user in store.query("SELECT DISTINCT r.user_id FROM market_signals s JOIN market_rules r ON r.id=s.rule_id JOIN users u ON u.user_id=r.user_id WHERE s.report_id IS NULL AND u.enabled=1 AND r.enabled=1 AND r.mode<>'off'"):
        pending = store.query("SELECT s.*,r.mode,r.series_id FROM market_signals s JOIN market_rules r ON r.id=s.rule_id WHERE s.report_id IS NULL AND r.user_id=? AND r.enabled=1 AND r.mode<>'off'", (user['user_id'],))
        preference = store.get_setting(f"market_digest:{user['user_id']}", '18:30')
        from zoneinfo import ZoneInfo
        now = datetime.now(timezone.utc).astimezone(ZoneInfo('Europe/Moscow'))
        daykey = f"market_digest_sent:{user['user_id']}"
        digest_due = now.strftime('%H:%M') >= preference and store.get_setting(daykey) != now.date().isoformat()
        ready = [s for s in pending if s['mode'] == 'immediate' or digest_due]
        if not ready:
            continue
        identifier = uuid4().hex
        text = 'Рыночные сигналы:\n' + '\n'.join(f"{s['series_id']}: {s['payload']}" for s in ready)
        parts = []
        if settings is not None:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            directory = settings.reports_dir / identifier
            directory.mkdir(parents=True, exist_ok=True)
            for identity in dict.fromkeys(signal['series_id'] for signal in ready):
                row = store.one('SELECT * FROM market_series WHERE id=?', (identity,))
                if not row or len(parts) >= 12:
                    continue
                metadata = json.loads(row['metadata'])
                curve = analysis.values(repository.history(store, identity), metadata).tail(120)
                if curve.dropna().empty:
                    continue
                figure = Figure(figsize=(9, 4), layout='constrained')
                FigureCanvasAgg(figure)
                axes = figure.subplots()
                axes.plot(curve.index, curve.values)
                last = curve.dropna()
                axes.scatter(last.index[-1], last.iloc[-1], color='red', label='Сигнал')
                axes.set_title(row['title'][:100])
                axes.set_ylabel(metadata['unit'])
                axes.legend()
                path = directory / f'{len(parts)}.png'
                figure.savefig(path)
                parts.append({'kind': 'photo', 'path': str(path), 'caption': row['title'][:100]})
        with store.connection() as conn:
            conn.execute("INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,quality,parts) VALUES(?,'market',?,1,0,?,?,?,?)",
                (identifier, utcnow(), text, text, dumps({'actor_id': user['user_id'], 'signal_ids': [s['id'] for s in ready], 'feature': 'alerts'}), dumps(parts)))
            for signal in ready:
                conn.execute('UPDATE market_signals SET report_id=? WHERE id=?', (identifier, signal['id']))
        prepare_delivery(store, identifier, str(user['user_id']))
        if any(s['mode'] == 'digest' for s in ready):
            store.set_setting(daykey, now.date().isoformat())
