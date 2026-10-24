"""Travel FX decisions: observed facts, validated models and explicit cost scenarios."""
import json
import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import market_analysis as analysis
from . import market_store as repository
from .dollar_periods import FIELDS as PERIOD_FIELDS

USD = 'cbr.fx.usd'
DEFAULTS = {'amount': 1000., 'owned': 0., 'trip': None, 'quote': None,
            'quote_at': None, 'fee_pct': 0., 'fee_fixed': 0., 'rub_yield': 0., 'risk': 5000., 'mode': 'overview',
            'history_mode': 'auto', 'forecast_engine': 'multivariate', 'analysis_months': 0, 'long_method': 'auto', 'distribution': True,
            'history_start': None, 'history_end': None, 'forecast_days': 7, 'forecast_date': None}
FIELDS = {
    'amount': ('Нужно долларов, всего', 'Введите общую сумму в USD (например 1500).', 1, 1_000_000),
    'owned': ('Уже есть долларов', 'Введите имеющуюся сумму USD. Она вычитается из покупки.', 0, 1_000_000),
    'quote': ('Курс продажи банка', 'Введите цену покупки вами 1 USD в рублях, например 95,50. Комиссии задаются отдельно.', .01, 10000),
    'fee_pct': ('Комиссия, %', 'Введите комиссию в процентах сверх курса (0, если уже включена).', 0, 100),
    'fee_fixed': ('Комиссия за обмен, ₽', 'Введите фиксированную комиссию за каждую операцию, в рублях (0, если нет).', 0, 1_000_000),
    'rub_yield': ('Доходность рублей, % годовых', 'Введите вашу доступную чистую годовую доходность рублей, после налогов/расходов. 0 — не учитывать. Ключевая ставка не подставляется.', 0, 100),
    'risk': ('Допустимое удорожание, ₽', 'Введите, сколько дополнительных рублей готовы потратить, если доллар подорожает.', 0, 10_000_000),
    'trip': ('Крайний срок покупки', 'Введите крайний срок покупки валюты: ГГГГ-ММ-ДД. Учтите время до поездки.', None, None),
    'analysis_months': ('Срок анализа, месяцев', 'Введите целое число месяцев от 1 до 60. Это срок анализа, а не дата поездки.', 0, 60),
}
FIELDS.update(PERIOD_FIELDS)
SOURCE = 'https://www.cbr.ru/faq/foreign_exchange_market/'


def today_moscow(now):
    return datetime.fromisoformat(now).astimezone(ZoneInfo('Europe/Moscow')).date()


def change_settings(current, field, text, *, now):
    if field not in FIELDS:
        raise ValueError('Неизвестная настройка')
    result = {**DEFAULTS, **current}
    if field in PERIOD_FIELDS:
        from .dollar_periods import change
        return change(result, field, text, now=now)
    if field == 'analysis_months':
        if not text.strip().isascii() or not text.strip().isdigit() or not 0 <= int(text.strip()) <= 60:
            raise ValueError('Введите целое число месяцев от 1 до 60 (0 — только ближайшие дни)')
        result[field] = int(text.strip())
        result['forecast_date'] = None
    elif field == 'trip':
        try:
            day = date.fromisoformat(text.strip())
        except ValueError:
            raise ValueError('Нужна дата ГГГГ-ММ-ДД') from None
        if not today_moscow(now) <= day <= today_moscow(now) + timedelta(days=730):
            raise ValueError('Дата должна быть от сегодня до двух лет вперёд')
        result[field] = day.isoformat()
    else:
        try:
            value = float(text.strip().replace(',', '.'))
        except ValueError:
            raise ValueError('Введите число без обозначения валюты') from None
        low, high = FIELDS[field][2:]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Допустимый диапазон: {low:g} — {high:g}')
        result[field] = value
        if field == 'quote':
            result['quote_at'] = now
    return result


def scenarios(config, *, days):
    """Same quoted all-in variable cost and fixed fee per transaction; simple net interest."""
    amount = max(0., config['amount'] - config['owned'])
    rate = config['quote'] * (1 + config['fee_pct']/100)
    fee = config['fee_fixed']
    now_cost = amount * rate + (fee if amount else 0)
    result = []
    for change in (-.10, -.05, 0., .05, .10):
        later = amount * rate * (1 + change) + (fee if amount else 0)
        interest = amount * rate * config['rub_yield']/100 * days/365
        staged = amount * rate/2 + amount * rate/2*(1 + change) + (2*fee if amount else 0) - interest/2
        result.append({'change': change, 'now': now_cost, 'staged': staged,
                       'wait': later - interest, 'interest': interest})
    return result


def history_stats(rows, meta, *, now):
    series = analysis.values(rows, meta, cutoff=now)
    positive = series.where(np.isfinite(series) & (series > 0))
    good = positive.dropna()
    if good.empty:
        return None, positive
    end = good.index[-1]
    expected = analysis.grid(good.index[0], end, meta)
    aligned = positive.reindex(expected)
    returns = np.log(aligned).diff().dropna()
    recent = aligned.loc[aligned.index >= end - pd.Timedelta(days=365)].dropna()
    metrics = {'date': end.date().isoformat(), 'last': float(good.iloc[-1]), 'n': len(good),
               'first': good.index[0].date().isoformat(), 'coverage': float(aligned.notna().mean()),
               'expected': len(expected), 'missing': int(aligned.isna().sum()),
               'percentile_365': float((recent <= good.iloc[-1]).mean()*100),
               'range_365': [float(recent.min()), float(recent.max())],
               'median_365': float(recent.median()),
               'vol_annual': float(returns.std(ddof=1)*np.sqrt(252)*100) if len(returns) >= 60 else None,
               'changes': {}}
    for days in (30, 90, 365):
        before = good.loc[good.index <= end - pd.Timedelta(days=days)]
        metrics['changes'][str(days)] = float((good.iloc[-1]/before.iloc[-1] - 1)*100) if not before.empty else None
    # Distinct source publications only. Effective dates in gaps never turn into zeros.
    return metrics, positive


def context(store, usd_rows, usd_meta, *, now, period=None):
    """Known enabled context only; selection deliberately bounded, never invented data."""
    rows = store.query("SELECT * FROM market_series WHERE enabled=1 ORDER BY id")
    chosen, groups = [], set()
    for row in sorted(rows, key=lambda r: (not r['id'].startswith('cbr.hd_base.keyrate.'), r['id'])):
        meta = json.loads(row['metadata'])
        title = row['title'].rsplit(':', 1)[-1].lower()
        group = None
        if row['id'] in {'cbr.fx.eur', 'cbr.fx.cny', 'cbr.ruonia.rate'}:
            group = row['id']
        elif meta.get('kind') == 'series' and row['dataset'] == 'cbr.hd_base.keyrate':
            group = 'key'
        elif meta.get('kind') == 'series' and 'infl' in row['id']:
            group = 'key' if 'ключев' in title else 'inflation' if 'инфляц' in title and 'цель' not in title else None
        elif meta.get('kind') == 'series' and 'bliquidity' in row['id']:
            group = 'liquidity'
        elif meta.get('kind') == 'series' and 'valintbr' in row['id']:
            group = 'fx_operations'
        elif row['id'].startswith('fred.') or row['id'].startswith('cbr.fx_context.balance.'):
            group = row['id']
        elif meta.get('kind') == 'series' and 'mrrf' in row['id']:
            group = 'reserves'
        if group is None or group in groups or meta.get('unit') == 'source_units':
            continue
        if row['source'] == 'cbr':
            meta = {**meta, 'date_timezone': 'Europe/Moscow'}
        data = repository.history(store, row['id'], as_of=now, effective_through=today_moscow(now).isoformat())
        if period:
            from .dollar_periods import clip
            data = clip(data, period)
        values = analysis.values(data, meta, cutoff=period['cutoff'] if period else now).dropna()
        if values.empty:
            continue
        groups.add(group)
        previous = values.loc[values.index <= values.index[-1] - pd.Timedelta(days=30)]
        record = {'id': row['id'], 'title': row['title'], 'value': float(values.iloc[-1]),
                  'date': values.index[-1].date().isoformat(), 'unit': meta['unit'], 'url': meta.get('url'),
                  'change_unit': 'п.п.' if '%' in meta['unit'] else meta['unit'],
                  'change_30': float(values.iloc[-1]-previous.iloc[-1]) if not previous.empty else None}
        last_period = next((r for r in reversed(data) if r['date'] == values.index[-1].date().isoformat()), {})
        record['period_end'] = last_period.get('period_end') or record['date']
        recent_age = ((date.fromisoformat(period['end']) if period else today_moscow(now))-date.fromisoformat(record['period_end'])).days
        record['date_label'] = (f"{(values.index[-1].month-1)//3+1}-й квартал {values.index[-1].year}, до {record['period_end']}"
                                if meta['frequency'] == 'quarterly' and meta.get('period_semantics') == 'flow' else record['date'])
        record['age_days'] = recent_age
        record['stale'] = recent_age > meta.get('release_lag_days', 100 if meta['frequency'] == 'quarterly' else 65 if meta['frequency'] == 'monthly' else 10)
        record['frequency'] = meta['frequency']
        record['retrieved_at'] = max((r['retrieved_at'] for r in data), default=None)
        record['revisions'] = [r['id'] for r in data]
        record['publication_unknown'] = any(not r.get('published_at') for r in data)
        try:
            record['correlation'] = analysis.correlations(usd_rows, data, usd_meta, meta)
        except ValueError:
            record['correlation'] = {'status': 'unsupported_frequency'}
        chosen.append(record)
    analysis.fdr([r['correlation'] for r in chosen])
    return chosen


def build(store, config, *, now):
    from . import dollar_long, dollar_models, dollar_periods, dollar_timing
    config = {**DEFAULTS, **config}
    if config['history_mode'] not in dollar_models.HISTORY or config['forecast_engine'] not in dollar_models.ENGINES:
        raise ValueError('Выберите настройки модели кнопками')
    dollar_long.validate(config['analysis_months'], config['long_method'])
    if type(config['distribution']) is not bool:
        raise ValueError('Выберите отображение сценариев кнопкой')
    # Revalidate persisted inputs, not only Telegram form values.
    for field in FIELDS:
        if field != 'trip' and config[field] is not None:
            change_settings(config, field, str(config[field]), now=now)
    dollar_periods.validate(config, now=now)
    period = dollar_periods.resolve(config, now=now)
    period['explicit'] = bool(config['history_start'] or config['history_end'])
    today = today_moscow(now)
    origin_day = date.fromisoformat(period['end'])
    model_now = period['cutoff']
    # Explicit date bounds take priority over automatic window reduction.
    model_history = 'full' if config['history_start'] or config['history_end'] else config['history_mode']
    period['effective_history_mode'] = model_history
    series_row = store.one('SELECT * FROM market_series WHERE id=?', (USD,))
    meta = {**(json.loads(series_row['metadata']) if series_row else {'frequency': 'daily', 'calendar': 'cbr_fx'}), 'date_timezone': 'Europe/Moscow'}
    # Full-history candidates use log changes and explicit regime indicators.
    # A recent-only model remains available as a distinct user choice/candidate.
    model_meta = {**meta, **({'break_date': '2024-06-13'} if model_history == 'recent' else {})}
    rows = repository.history(store, USD, as_of=now, effective_through=today.isoformat()) if series_row else []
    rows = dollar_periods.clip(rows, period)
    stats, series = history_stats(rows, meta, now=model_now)
    period.update(available_start=stats['first'] if stats else None, available_end=stats['date'] if stats else None,
                  observations=stats['n'] if stats else 0)
    deadline = date.fromisoformat(config['trip']) if config['trip'] else None
    days = (deadline-today).days if deadline else None
    quote_fresh = False
    if config['quote_at']:
        stamp = datetime.fromisoformat(config['quote_at'])
        age = datetime.fromisoformat(now).astimezone(timezone.utc)-stamp.astimezone(timezone.utc)
        quote_fresh = timedelta(0) <= age <= timedelta(hours=24)
    model_ready = bool(stats and 0 <= (origin_day-date.fromisoformat(stats['date'])).days <= 5)
    data_fresh = model_ready and not period['retrospective']
    quote_fresh = quote_fresh and not period['retrospective']
    amount = max(0., config['amount']-config['owned'])
    current_quote = config['quote'] is not None and not period['retrospective']
    costs = scenarios(config, days=max(0, days or 0)) if current_quote else []
    premium = ((config['quote']*(1+config['fee_pct']/100) + config['fee_fixed']/amount)/stats['last']-1)*100 if stats and current_quote and amount else None
    forecast = []
    economic_context = context(store, rows, meta, now=now, period=period) if stats else []
    inputs, excluded = dollar_models.factors(store, [r['id'] for r in economic_context], now=now) if config['forecast_engine'] == 'multivariate' else ([], [])
    inputs = [{**item, 'rows': dollar_periods.clip(item['rows'], period)} for item in inputs]
    short = dollar_models.forecast(rows, model_meta, now=model_now, today=origin_day, inputs=inputs,
                                  history=model_history, targets=dollar_periods.targets(config, origin_day)) if model_ready else []
    long_term = dollar_long.forecast(rows, model_meta, now=model_now, today=origin_day, months=config['analysis_months'],
                    history=model_history, method=config['long_method'], distribution=config['distribution']) if model_ready and config['analysis_months'] else []
    # An effective future rate retrieved from the official source is already public,
    # but is never passed into numerical model training.
    published = store.one("SELECT id,value,date FROM market_revisions WHERE series_id=? AND date=? AND retrieved_at<=? AND value IS NOT NULL ORDER BY retrieved_at DESC,id DESC LIMIT 1",
                          (USD, (today+timedelta(days=1)).isoformat(), now)) if not period['retrospective'] else None
    for point in short:
        if (published and point['target'] >= published['date'] and
                len(analysis.grid(pd.Timestamp(published['date'])+pd.Timedelta(days=1), pd.Timestamp(point['target']), model_meta)) == 0):
            point.update(prediction=published['value'], model='published_cbr', known_published=True,
                         known_published_revision=published['id'], lower=None, upper=None,
                         validation_mae=None, baseline_mae=None, validation_n=0,
                         note='Официальный курс уже опубликован; это не результат числовой модели')
    purchase_forecast = bool(config['mode'] == 'purchase' and stats and data_fresh and days is not None and days > 0)
    if purchase_forecast:
        months = min(24, max(1, math.ceil(days/30)))
        long_meta = {**model_meta, **({'break_date': '2024-06-13'} if model_history != 'full' else {})}
        if model_history == 'full':
            forecast = dollar_models.forecast(rows, long_meta, now=now, today=today, inputs=inputs,
                                             history='full', targets=[(days, deadline)])
            for point in forecast:
                point['duration'] = {'unit': 'months', 'value': months}
        else:
            forecast = analysis.forecast(rows, long_meta, created_at=now, horizon={'unit': 'months', 'value': months})
    combined = {point['target']: point for point in forecast}
    combined.update({point['target']: point for point in short})
    forecast = [combined[key] for key in sorted(combined)]
    for point in forecast + long_term:
        point.update(created_at=now, analysis_as_of=period['end'], retrospective_period=period['retrospective'])
    comparison_config = config if not period['retrospective'] else {**config, 'quote': None}
    timing = dollar_timing.comparison(short, comparison_config, spot=stats['last'], days=days,
              ready=bool(data_fresh and quote_fresh and days is not None and days >= 0 and stats['coverage'] >= .9)) if stats else []
    # Monthly endpoints participate in the same cost comparison; never interpolate
    # uncalculated days or call a historical date a recommendation for today.
    candidates = {p['target']: p for p in timing}
    monthly = [{**p, 'days_ahead': (date.fromisoformat(p['target'])-origin_day).days,
                'native_step': p['horizon']} for p in long_term if p['target'] > period['end']]
    if stats:
        for point in dollar_timing.comparison(monthly, comparison_config, spot=stats['last'], days=days,
                ready=bool(data_fresh and quote_fresh and days is not None and days >= 0 and stats['coverage'] >= .9)):
            candidates.setdefault(point['target'], point)
    missing = []
    if not config['trip']:
        missing.append('срок покупки')
    if config['quote'] is None:
        missing.append('цена продажи USD банком вам')
    result = {'created_at': now, 'config': config, 'amount_remaining': amount, 'days': days, 'missing_inputs': missing,
            'period': period, 'model_data_fresh': model_ready,
            'stats': stats, 'quote_fresh': quote_fresh, 'data_fresh': data_fresh, 'premium': premium,
            'scenarios': costs, 'forecast': forecast, 'timing': timing, 'long_term': long_term, 'model_metadata': model_meta,
            'purchase_candidates': [candidates[key] for key in sorted(candidates)],
            'forecast_duration': {'unit': 'months', 'value': months} if purchase_forecast else {
                'unit': 'days', 'value': dollar_periods.date_target(config, origin_day)[0] if config.get('forecast_date')
                else 7 if config['analysis_months'] else config['forecast_days']},
            'published_revision': published['id'] if published else None,
            'context': economic_context, 'model_inputs': [{'id': i['id'], 'title': i['title'],
                'assumed_lag_days': i['assumed_lag_days'], 'retrospective': i['retrospective'],
                'revisions': [r['id'] for r in i['rows']]} for i in inputs], 'excluded_features': excluded,
            'input_revisions': [r['id'] for r in rows], 'retrieved_at': max((r['retrieved_at'] for r in rows), default=None)}
    from . import dollar_purchase_guide
    result['purchase_guide'] = dollar_purchase_guide.build(result)
    result['decision'] = result['purchase_guide']['reason']
    return result, series


def render_details(result):
    from .dollar_models import HISTORY
    cfg, stats = result['config'], result['stats']
    lines = ['Доллар для поездки · USD/RUB', result['decision'],
             f"Осталось купить: {result['amount_remaining']:,.2f} USD. Срок: {cfg['trip'] or 'не задан'}."]
    if stats:
        lines += [f"ЦБ: {stats['last']:.4f} ₽ за USD на {stats['date']}; загрузка {result['retrieved_at']}.",
                  f"История: {stats['first']} — {stats['date']}, {stats['n']} наблюдений; покрытие {stats['coverage']:.1%}, пропуски {stats['missing']} из {stats['expected']}."]
        for days, value in stats['changes'].items():
            lines.append(f"Изменение за {days} дней: {value:+.2f}%." if value is not None else f'За {days} дней недостаточно истории.')
        lines.append(f"За доступную часть года: диапазон {stats['range_365'][0]:.2f}–{stats['range_365'][1]:.2f}, медиана {stats['median_365']:.2f}; текущий курс выше или равен {stats['percentile_365']:.0f}% наблюдений. Это не сигнал разворота.")
        if stats['vol_annual'] is not None:
            lines.append(f"Волатильность: {stats['vol_annual']:.1f}% годовых = std(log доходностей) × √252. Пропуски не склеиваются.")
    else:
        lines.append('Нет сохранённой истории USD. Администратор может загрузить её кнопкой «Обновить USD».')
    if cfg['quote'] and not result.get('period', {}).get('retrospective'):
        lines.append(f"Цена банка: {cfg['quote']:.4f} ₽/USD; комиссия {cfg['fee_pct']:g}% + {cfg['fee_fixed']:g} ₽ за операцию. Введена: {cfg['quote_at']}. {'Свежая (до 24 ч).' if result['quote_fresh'] else 'Нужно обновить.'}")
    if result['premium'] is not None:
        lines.append(f"Вся стоимость покупки относительно курса ЦБ: {result['premium']:+.2f}%. Это разница цен, не оценка выгодности банка.")
    if result['scenarios']:
        lines += ['Сценарии затрат, ₽: купить сейчас / 50% сейчас + 50% в срок / всё в срок']
        for row in result['scenarios']:
            lines.append(f"Курс {row['change']:+.0%}: {row['now']:,.0f} / {row['staged']:,.0f} / {row['wait']:,.0f}")
        lines.append(f"Предположения: изменение цены банка, одинаковые комиссии; две операции при покупке частями. Из будущих затрат вычтен простой доход на отложенные рубли по вашей ставке {cfg['rub_yield']:g}% годовых за {max(0, result['days'] or 0)} дней. Это стресс-сценарии, не вероятности; ±10% не предел риска.")
        if result['days'] is not None and result['days'] > 0 and result['amount_remaining']:
            lines.append(f"Порог безубыточности ожидания при тех же комиссиях: рост цены банка не выше {cfg['rub_yield']*result['days']/365:.2f}% компенсируется доходом на рубли. Выше этого уровня покупка сейчас дешевле по модели затрат.")
            midpoint = today_moscow(result['created_at']) + timedelta(days=max(1, result['days']//2))
            lines.append(f"Пример покупки частями: {result['amount_remaining']/2:,.2f} USD сейчас и столько же к {cfg['trip']}. Проверить цену и наличие валюты ещё раз {midpoint.isoformat()}. Это план исполнения, не прогноз курса.")
    predictions = result['forecast']
    if predictions:
        target = date.fromisoformat(cfg['trip']) if cfg['trip'] else today_moscow(result['created_at'])+timedelta(days=7)
        point = min(predictions, key=lambda p: abs((date.fromisoformat(p['target'])-target).days))
        checked = point.get('validation_n', 0)
        lines.append(f"Локальная модель на {point['target']}: {point['prediction']:.2f} ₽/USD; базовый вариант {point['baseline']:.2f}; модель {point['model']}.")
        if point.get('known_published'):
            lines.append('Курс уже опубликован ЦБ; он показан отдельно от прогноза модели. Цена обмена банка может отличаться.')
        elif checked >= 10:
            lines.append(f"Проверок: {checked}; MAE {point['validation_mae']:.2f} ₽, базовый MAE {point['baseline_mae']:.2f} ₽. Проверка ретроспективная, история могла пересматриваться.")
            if point['lower'] is not None:
                lines.append(f"Исторически калиброванный диапазон 80%: {point['lower']:.2f}–{point['upper']:.2f} ₽; не гарантия и не розничная цена.")
        else:
            lines.append('Недостаточно прошлых ошибок для проверки: показан базовый сценарий, прогнозный диапазон не публикуется.')
        lines.append('Короткие горизонты сравнивают окна обучения, лаги и ridge-регрессию. Отбор по корреляциям и настройка выполняются внутри обучающего окна. Смена режима 13.06.2024 отмечается отдельным признаком; доступен режим обучения только после этой даты. Длинные горизонты используют модели истории USD.')
    else:
        lines.append('Прогноз недоступен: нужны свежие данные и достаточная сопоставимая история. Для расчёта покупки дополнительно укажите срок и цену банка.')
    for point in result['timing']:
        if point.get('known_published'):
            continue
        mae = 'недостаточно проверок' if point['validation_mae'] is None else f"{point['validation_mae']:.3f} ₽/USD"
        baseline = 'недостаточно проверок' if point['baseline_mae'] is None else f"{point['baseline_mae']:.3f} ₽/USD"
        window = {**HISTORY, 'three_years': 'Последние 3 года'}.get(point.get('window'), '—')
        lines.append(f"На {point['target']}: модель {point['model']}, окно {window}; история обучения/сравнения {point.get('training_start', '—')} — {point.get('training_end', '—')}, примеров {point.get('training_n', 0)}; MAE {mae}, базовый MAE {baseline}.")
        if point.get('selection_reason'):
            lines.append('Выбор модели: '+point.get('rejected', point['selection_reason'])+'.')
        if point.get('inner_candidate_mae') is not None:
            lines.append(f"Внутренняя проверка: лучший кандидат MAE {point['inner_candidate_mae']:.3f} ₽/USD; последнее значение {point['inner_baseline_mae']:.3f} ₽/USD; точек {point['selection_n']}. Это настройка на прошлом, не итоговая оценка ошибки.")
        if point.get('candidate_mae') is not None:
            lines.append(f"Внешняя проверка выбора модели: MAE {point['candidate_mae']:.3f} ₽/USD; последнее значение {point['baseline_mae']:.3f} ₽/USD.")
        if point.get('features'):
            lines.append('Признаки (корреляция только на обучении): '+', '.join(f"{f['name']} ({f['training_correlation']:+.3f})" if f['training_correlation'] is not None else f['name'] for f in point['features']))
    lines.append('Экономический контекст · доступные сохранённые показатели')
    for row in result['context']:
        change = f"; изменение за 30 дней {row['change_30']:+.4g} {row['change_unit']}" if row['change_30'] is not None else ''
        lines.append(f"{row['title']}: {row['value']:.5g} {row['unit']}; период {row['date_label']}{change}.\n{row['url']}")
        corr = row['correlation']
        if corr.get('status') == 'ok':
            lines.append(f"Связь изменений (без лага): Pearson {corr['pearson']:.2f}, Spearman {corr['spearman']:.2f}, пар {corr['n']}; частота {corr['frequency']}; q после множественных проверок {corr['pearson_q']:.3f}. Исследовательская оценка, не причина и не торговый сигнал.")
    if not result['context']:
        lines.append('Другие показатели пока не загружены или не включены. Их можно подключить в «Показателях Банка России».')
    lines.append('Что оценить перед обменом: экспорт и продажа валютной выручки влияют на предложение долларов; импорт и платежи — на спрос. Ставки, инфляция и решения регуляторов меняют стимулы держать рубли или валюту. Эти факторы могут действовать одновременно и в разные стороны; один показатель не определяет курс.')
    lines.append('При включённой модели показателей ежедневные ряды участвуют в прогнозе с лагами публикации; месячные и квартальные ряды остаются в контексте. Исторические версии и моменты публикации части источников неизвестны: проверка ретроспективная, задержки публикации для них предполагаются, а не доказываются. Новости, наличие наличных и курс в стране поездки не загружаются.')
    lines.append('Официальный курс ЦБ — ориентир, банк устанавливает цену обмена. Экономика курса и методика: ' + SOURCE)
    return '\n\n'.join(lines)


def summary(result):
    stats, cfg = result['stats'], result['config']
    lines = ['Доллар · что делать дальше']
    period = result.get('period')
    if period:
        from .dollar_periods import horizon_label
        lines.append(f"История для расчёта: {period['start'] or 'начало доступных данных'} — {period['end']}. Срок прогноза: {horizon_label(cfg).rstrip('.')}.")
        if period['retrospective']:
            lines.append('Ретроспективный расчёт от '+period['end']+'. Наблюдения после этой даты исключены. Использованы сохранённые версии истории; это не точное воспроизведение доступных тогда публикаций.')
    if cfg.get('forecast_date'):
        point = next((p for p in result['timing'] if p['target'] == cfg['forecast_date']), None)
        lines.append('Курс на '+date.fromisoformat(cfg['forecast_date']).strftime('%d.%m.%Y'))
        if point:
            lines.append(f"Оценка: {point['prediction']:.2f} ₽ за USD. "+(
                'Курс уже опубликован ЦБ; это известное значение.' if point.get('known_published') else
                'Действующий курс по календарю; это не прогноз движения.' if point['model'] == 'calendar_carry' else
                'Метод: последнее известное значение. Это базовый ориентир, направление не определено.' if point['model'] == 'last' else
                'Метод: регрессия с лагами; пройдена проверка против последнего значения.'))
            if point.get('lower') is not None and point.get('upper') is not None and point.get('validation_n', 0) >= 10:
                lines.append(f"Диапазон по прошлым ошибкам (80%): {point['lower']:.2f}–{point['upper']:.2f} ₽/USD.")
            elif not point.get('known_published') and point['model'] != 'calendar_carry':
                lines.append('Диапазон не рассчитан: недостаточно проверочных ошибок на этот срок.')
            if point['days_ahead'] > 90:
                lines.append('Дальний срок: расчёт выполнен непосредственно на выбранную дату, без интерполяции месячных оценок. '
                             'Доступных независимых проверок может быть мало; структурные изменения экономики не прогнозируются.')
        else:
            lines.append('Оценка недоступна: нужны свежие данные и достаточная история в выбранных границах.')
    if cfg.get('mode') == 'purchase':
        if not result.get('purchase_guide'):
            lines.append(result['decision'])
        if result['missing_inputs']:
            lines.append('Сначала укажите: '+', '.join(result['missing_inputs'])+'. Кнопка «Настроить покупку» откроет настройки. Эти сведения нельзя получить из истории ЦБ.')
    elif stats and not (period and period['retrospective']):
        lines.append('Обзор рынка готов. Для решения о вашей покупке нажмите «Настроить покупку» и укажите срок и реальную цену банка.')
    if not stats:
        lines.append('Нет сохранённой истории USD в выбранном периоде. Измените границы истории или запросите загрузку у администратора; прогноз пока недоступен.')
    if result.get('purchase_guide'):
        from .dollar_purchase_guide import describe
        lines.extend(describe(result['purchase_guide']))
    if stats:
        lines.append(f"ЦБ: {stats['last']:.4f} ₽/USD на {stats['date']}; покрытие {stats['coverage']:.0%}.")
        lines.append(f"История в расчёте: {stats['first']} — {stats['date']}, {stats['n']} наблюдений. Настройки обучения и источников — в меню модели.")
        modeled = [point for point in result['timing'] if not point.get('known_published') and point['validation_mae'] is not None]
        if modeled and all(point['model'] == 'last' for point in modeled):
            lines.append('Направление движения надёжно не определено. Выбран базовый вариант «последний известный курс»: одинаковые числа получаются по его формуле, а не означают, что доллар не изменится. Более сложная модель не улучшила ошибку минимум на 5% на требуемых проверках.')
        elif modeled and all(abs(point['prediction']-point['baseline']) <= point['validation_mae'] for point in modeled):
            lines.append('По курсу ЦБ ожидаемые изменения на ближайшие дни меньше типичной ошибки. Проверенного преимущества ожидания модель не показывает.')
        change = stats['changes']['30']
        if change is not None:
            lines.append(f"За месяц доллар {'подорожал' if change > 0 else 'подешевел' if change < 0 else 'не изменился'} {abs(change):.2f}%. Уровень выше или равен {stats['percentile_365']:.0f}% наблюдений за год; это не обещает падения завтра.")
    if cfg.get('analysis_months'):
        from .dollar_long import describe
        lines.extend(describe(result.get('long_term', []), cfg['analysis_months']))
    lines.append('Сравниваемые даты от '+period['end'] if period else 'Сегодня / завтра / через 2, 3 и 7 дней')
    for point in result['timing']:
        label = ('Завтра' if point['days_ahead'] == 1 else f"Через {point['days_ahead']} дн.") if not (period and period['retrospective']) else f"+{point['days_ahead']} дн."
        text = f"{label} ({point['target']}): ориентир ЦБ {point['prediction']:.2f} ₽/USD"
        if point.get('known_published'):
            text += ' · уже опубликован'
        elif point['model'] == 'last':
            text += ' · базовый вариант, направление не определено'
        elif point['model'] == 'calendar_carry':
            text += ' · сохранение действующего курса по календарю'
        if point['validation_mae'] is not None:
            text += f"; средняя абсолютная ошибка {point['validation_mae']:.2f} ₽/USD ({point['validation_n']} проверок)"
            if point.get('lower') is not None and point.get('upper') is not None:
                text += f"\nОриентировочный диапазон 80% по прошлым ошибкам: {point['lower']:.2f}–{point['upper']:.2f} ₽/USD"
        elif not point.get('known_published'):
            text += '; проверенного преимущества ожидания нет'
        if cfg.get('mode') == 'purchase' and point['cost'] is not None:
            text += f"\nЗатраты {point['cost']:,.0f} ₽; {'экономия' if point['saving'] >= 0 else 'удорожание'} относительно сегодня {abs(point['saving']):,.0f} ₽"
            if point['error_rub'] is not None:
                text += f"; ошибка модели ≈{point['error_rub']:,.0f} ₽"
            text += '\n'+point['assessment']
        lines.append(text+'.')
    if not result['timing']:
        lines.append('Краткосрочная оценка недоступна: нужны свежие данные и минимум 60 сопоставимых наблюдений без разрыва в конце ряда.')
    chosen = [p for p in result['timing'] if not p.get('known_published')]
    if any(p['model'] != 'calendar_carry' for p in chosen):
        from .dollar_charts import INTERVAL_EXPLANATION, POINT_EXPLANATION
        lines.append(POINT_EXPLANATION)
        if any(p.get('lower') is not None and p.get('validation_n', 0) >= 10 for p in chosen):
            lines.append(INTERVAL_EXPLANATION)
    if any(p['validation_n'] == 10 for p in chosen):
        lines.append('10 проверок — небольшая выборка. Средняя ошибка и диапазон описывают прошлые ошибки; возможное движение не ограничено ими. Это не уверенность в отсутствии изменений.')
    same_course = {}
    for point in result['timing']:
        if not point.get('known_published'):
            same_course.setdefault((point.get('origin'), point['native_step']), []).append(point['target'])
    for (_, step), dates in same_course.items():
        if step > 0 and len(dates) > 1:
            lines.append('Даты '+', '.join(dates)+f' соответствуют одному горизонту: {step} обновл. курса ЦБ от последнего факта. Это один прогноз с учётом выходных, а не независимые оценки для каждого дня.')
    used = sorted({f['name'].split(':lag')[0] for p in chosen for f in p.get('features', []) if not f['name'].startswith(('usd:', 'regime:'))})
    if used:
        lines.append('В прогнозе используются лаги показателей: '+', '.join(used)+'. Подробности обучения — в расчёте.')
    elif chosen and cfg['forecast_engine'] == 'basic':
        lines.append('В настройках выбран расчёт только по истории USD. Включить лаги других показателей можно в меню модели.')
    elif chosen:
        lines.append('Дополнительные показатели не дали подтверждённого улучшения на этой проверке; выбран прогноз истории USD или последнего значения.')
    if any(p.get('retrospective') for p in chosen):
        lines.append('Проверка ретроспективная: исторические версии публикаций неизвестны, использованы предполагаемые задержки. Это не точное воспроизведение доступной тогда информации.')
    if cfg.get('mode') == 'purchase' and result['scenarios']:
        lines.insert(2, f"Покупка сегодня: {result['scenarios'][2]['now']:,.0f} ₽ за оставшиеся {result['amount_remaining']:,.2f} USD, включая комиссии.")
    if result['context']:
        loaded = len(result['context'])
        stale = sum(row['stale'] for row in result['context'])
        lines.append(f'Экономический контекст: {loaded} рядов; с заметной задержкой — {stale}. Даты, исходные значения и корреляции — в подробностях.')
        by_id = {row['id']: row for row in result['context']}
        for identity in ('fred.brent', 'fred.us_rate', 'fred.usd_broad', 'cbr.fx_context.balance.trade_balance'):
            row = by_id.get(identity)
            if row:
                lines.append(f"{row['title']}: {row['value']:.4g} {row['unit']}; период {row['date_label']}{' · данные с задержкой' if row['stale'] else ''}.")
    if cfg.get('mode') == 'purchase' and cfg['quote'] is not None and not result.get('purchase_guide'):
        lines.append('Цена банка в будущем оценена при неизменной относительной наценке и комиссии. Изменение банковского спреда не включено в ошибку модели. Политические события и новости не прогнозируются.')
    else:
        lines.append('Здесь ориентиры официального курса ЦБ. Для сравнения реальных затрат нужен курс банка в режиме «Моя покупка». Политические события и новости не прогнозируются.')
    return '\n\n'.join(lines)


def render(result):
    return summary(result)+'\n\nПодробности расчёта\n\n'+render_details(result)
