"""Explain compared purchase dates without turning a baseline into a trading signal."""
import math
from datetime import date, datetime
from zoneinfo import ZoneInfo


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _day(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _today(result):
    if result.get('created_at'):
        try:
            stamp = datetime.fromisoformat(result['created_at'])
            if stamp.tzinfo:
                stamp = stamp.astimezone(ZoneInfo('Europe/Moscow'))
            return stamp.date()
        except (TypeError, ValueError):
            pass
    return _day(result.get('today')) or _day((result.get('stats') or {}).get('date'))


def build(result):
    """Return JSON-safe guidance; prices and model outputs are never modified here."""
    cfg, stats = result.get('config', {}), result.get('stats') or {}
    period = result.get('period') or {}
    purchase = cfg.get('mode') == 'purchase'
    amount = max(0., (_number(cfg.get('amount')) or 0.) - (_number(cfg.get('owned')) or 0.))
    today, end = _today(result), _day(period.get('end'))
    trip, quote = _day(cfg.get('trip')), _number(cfg.get('quote'))
    spot = _number(stats.get('last'))
    guide = {'status': 'insufficient', 'reason': '', 'actionable': False,
             'purchase': purchase, 'amount_remaining': amount, 'dates': [], 'evidence': [],
             'horizon_count': 0, 'missing_inputs': [], 'assumptions': []}
    if period.get('retrospective') or (today and end and end < today):
        guide.update(status='retrospective', reason='Выбран прошлый конец истории: это исторический анализ, без совета о покупке сегодня.')
        return guide
    if today and end and end > today:
        guide['reason'] = 'Конец истории находится в будущем. Выберите дату не позже сегодня.'
        return guide
    if purchase and amount == 0:
        guide.update(status='covered', reason='Нужная сумма уже есть: дополнительная покупка для этой цели не требуется.')
        return guide
    if purchase and trip and today and trip < today:
        guide['reason'] = 'Срок покупки прошёл. Укажите новый срок для текущего сравнения.'
        return guide
    data_day = _day(stats.get('date'))
    fresh = result.get('data_fresh', bool(today and data_day and 0 <= (today-data_day).days <= 5))
    if not today or not spot or spot <= 0 or not fresh or (_number(stats.get('coverage')) or 0.) < .9:
        guide['reason'] = 'Для выбора времени покупки нужны свежий курс ЦБ и покрытие истории не ниже 90%.'
        return guide
    if purchase:
        if not trip:
            guide['missing_inputs'].append('крайний срок покупки')
        if quote is None or quote <= 0:
            guide['missing_inputs'].append('реальная цена продажи USD банком')
        elif not result.get('quote_fresh', False):
            guide['missing_inputs'].append('свежая цена банка (до 24 часов)')
        if guide['missing_inputs']:
            guide['reason'] = 'Для решения по расходам задайте: '+', '.join(guide['missing_inputs'])+'.'
            return guide
        guide['actionable'] = True
        guide['assumptions'].append('Будущая цена банка рассчитана при неизменной относительной наценке и комиссии; изменение банковского спреда в ошибке курса ЦБ не учтено.')

    candidates = result.get('purchase_candidates', result.get('timing', [])) or []
    groups = {}
    for row in candidates:
        target, prediction = _day(row.get('target')), _number(row.get('prediction'))
        if not target or target <= today or prediction is None or prediction <= 0:
            continue
        if purchase and trip and target > trip:
            continue
        # Published fixing and carry on non-fixing days are facts/calendar, not a prediction edge.
        if row.get('known_published') or row.get('model') in ('calendar_carry', 'published_cbr'):
            continue
        mae, baseline = _number(row.get('validation_mae')), _number(row.get('baseline_mae'))
        n = max(0, int(_number(row.get('validation_n')) or 0))
        saving, error = _number(row.get('saving')), _number(row.get('error_rub'))
        evidence = {'date': target.isoformat(), 'dates': [target.isoformat()], 'prediction': prediction,
                    'delta': prediction-spot, 'mae': mae, 'validation_n': n, 'model': row.get('model'),
                    'baseline_mae': baseline, 'saving': saving, 'error_rub': error}
        evidence['validated'] = bool(row.get('model') not in (None, 'last') and n >= 10 and
                                     mae is not None and mae >= 0 and baseline is not None and
                                     baseline > 0 and mae < .95*baseline)
        # Preserve different monetary comparisons (interest may vary over a weekend).
        key = (row.get('origin'), row.get('native_step'), row.get('model'), prediction, mae,
               baseline, n, saving if purchase else None, error if purchase else None)
        if row.get('native_step') is None:
            key += (target.isoformat(),)
        if key in groups:
            groups[key]['dates'].append(target.isoformat())
        else:
            groups[key] = evidence
    evidence = sorted(groups.values(), key=lambda item: item['date'])
    guide['horizon_count'] = len(evidence)
    guide['evidence'] = evidence
    if evidence:
        guide['assumptions'].append('Сравниваются только перечисленные опорные даты. Точный минимум между ними не найден; средняя ошибка не ограничивает возможный убыток.')

    # A deadline concerns readiness, even when the model cannot distinguish purchase prices.
    if purchase and trip and (trip-today).days <= 7:
        guide.update(status='near_deadline', reason='До срока не больше недели. Сравните конечную цену банков и обеспечьте нужную сумму к сроку; ожидание добавляет риск не успеть с обменом.')
        return guide
    if not evidence:
        guide['reason'] = 'Нет сопоставимых прогнозов на будущие даты в выбранном сроке. Расширьте срок или историю обучения.'
        return guide

    falling, rising = [], []
    for row in evidence:
        if not row['validated']:
            continue
        if row['delta'] < -row['mae'] and (not purchase or (
                row['saving'] is not None and row['error_rub'] is not None and row['saving'] > row['error_rub'])):
            falling.append(row)
        if row['delta'] > row['mae'] and (not purchase or (
                row['saving'] is not None and row['error_rub'] is not None and -row['saving'] > row['error_rub'])):
            rising.append(row)

    # A user's risk budget takes precedence over tentative savings from waiting.
    risk = _number(cfg.get('risk'))
    if purchase and risk is not None:
        rate = quote*(1+(_number(cfg.get('fee_pct')) or 0.)/100)
        stress = amount*rate*.1
        guide['stress_extra_rub'] = stress
        if stress > risk:
            guide.update(status='staged', reason='В условном сценарии роста курса на 10% удорожание превышает ваш лимит. Рассмотрите покупку частями до срока, сравнив комиссии за несколько операций.')
            guide['assumptions'].append('Рост на 10% — проверка устойчивости бюджета, без оценки вероятности; будущий доход по рублям здесь не уменьшает риск удорожания валюты.')
            return guide
    if falling:
        best = max(falling, key=lambda row: row['saving']-row['error_rub'] if purchase else -row['delta']-row['mae'])
        guide.update(status='consider_wait', dates=best['dates'], selected=best,
                     reason='Есть основание повторно проверить цену к выбранной опорной дате: модель допускает снижение больше своей типичной ошибки.')
    elif rising:
        best = max(rising, key=lambda row: -row['saving']-row['error_rub'] if purchase else row['delta']-row['mae'])
        guide.update(status='consider_now', dates=best['dates'], selected=best,
                     reason='Есть аргумент сравнить цены банков сейчас: к выбранной опорной дате модель допускает рост больше своей типичной ошибки.')
    else:
        guide.update(status='no_edge', reason='Проверенного преимущества покупки позже не видно. Выбирайте дату по сроку поездки, доступной цене банка и допустимому расходу.')
    if not purchase:
        guide['assumptions'].append('Это предварительное сравнение курса ЦБ в ₽/USD. Для решения по реальным расходам откройте «Моя покупка» и задайте цену банка и крайний срок.')
    return guide


def describe(guide):
    """Compact Russian paragraphs intended for both Telegram summary and full report."""
    lines = ['Когда рассмотреть покупку\n'+guide['reason']]
    row = guide.get('selected')
    if row:
        lines.append('Опорные даты: '+', '.join(row['dates'])+'. '
                     f"Оценка {row['prediction']:.2f} ₽/USD; изменение {row['delta']:+.2f}; "
                     f"типичная ошибка {row['mae']:.2f} ₽/USD, проверок {row['validation_n']}. "
                     f"MAE базового прогноза {row['baseline_mae']:.2f} ₽/USD.")
        if guide['purchase'] and row['saving'] is not None:
            lines.append(f"Расчётная экономия ожидания {row['saving']:+,.0f} ₽; типичная ошибка в деньгах {row['error_rub']:,.0f} ₽. Это условное сравнение, без гарантии экономии.")
    if guide['status'] == 'staged':
        lines.append(f"При +10% дополнительные затраты на оставшуюся сумму: {guide['stress_extra_rub']:,.0f} ₽. Доли покупки выбирайте с учётом доступных денег и комиссий.")
    lines.extend(guide['assumptions'])
    return lines
