"""Personal, offline charts; display limits never truncate forecast training."""
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from . import market, market_analysis, market_store
from .delivery import prepare_delivery
from .forecast_horizons import end_date, from_options
from .forecast_horizons import validate as validate_duration
from .reporting import ReportBuilder
from .store import dumps, utcnow
from .telegram_data_jobs import actor_scope, export_dates, require_actor, scoped_rows

PERIODS = {'month': 30, 'quarter': 90, 'half': 180, 'year': 365, 'all': None}
SECTIONS = {'ruonia', 'market', 'wordstat', 'trends'}
PAGE_SIZE = 12


def ordered_available(series, order=None):
    """Dense, current series first; freeze this order for subsequent pages."""
    available = [item for item in series if not item['history'].dropna().empty]
    if order is not None:
        if not isinstance(order, list) or len(order) != len(set(order)) or any(not isinstance(key, str) for key in order):
            raise ValueError('Неверный порядок графиков')
        ranks = {key: rank for rank, key in enumerate(order)}
        available = [item for item in available if item['identity'] in ranks]
        return sorted(available, key=lambda item: ranks[item['identity']])
    def priority(item):
        curve = item['history']
        return (-int(curve.dropna().index.max().value), -float(curve.notna().mean()),
                -int(curve.notna().sum()), item['title'].casefold(), item['identity'])
    return sorted(available, key=priority)


def chart_page(page):
    if type(page) is not int or not 0 <= page <= 10000:
        raise ValueError('Неверная страница графиков')
    return page


def page_items(series, page, order=None):
    available = ordered_available(series, order)
    if order is None:
        return available[page*PAGE_SIZE:(page+1)*PAGE_SIZE]
    # A correction that removes a previous page's value must not shift or skip later rows.
    identities = set(order[page*PAGE_SIZE:(page+1)*PAGE_SIZE])
    return [item for item in available if item['identity'] in identities]


def prediction_kind(point):
    if point.get('note', '').startswith('Сценарий сохранения'):
        return 'administrative'
    if 'Сценарий' in point.get('note', ''):
        return 'scenario'
    if point.get('validation_n', 0) < 10:
        return 'scenario'
    return 'forecast'


def plot_item(axes, item, *, overlay=False, intervals=True, color='#246b9b'):
    """Show isolated observations and distinguish checked forecasts from scenarios."""
    curve, predictions = item['history'], item['forecasts']
    valid = curve.dropna()
    label = item['title'] if overlay else 'Наблюдения'
    axes.plot(curve.index, curve.values, label=label, color=color, linewidth=1.2,
              marker='o' if len(valid) <= 60 else None, markersize=4,
              drawstyle='steps-post' if item.get('administrative') and item.get('frequency') == 'daily' else 'default')
    if len(valid) > 60:
        isolated = curve.notna() & curve.shift().isna() & curve.shift(-1).isna()
        axes.scatter(curve.index[isolated], curve[isolated], color=color, s=15)
    if predictions:
        axes.axvline(valid.index[-1], color='#666666', linewidth=.8, linestyle=':')
        for kind, name, forecast_color, style in (
            ('forecast', 'Прогноз', color if overlay else '#b56e13', '--o'),
            ('scenario', 'Сценарий без достаточной проверки', '#777777', ':s'),
            ('administrative', 'Условие: без изменений (не прогноз)', '#777777', '--'),
        ):
            points = [p for p in predictions if prediction_kind(p) == kind]
            if not points:
                continue
            if kind == 'administrative' and len(points) > 1:
                points = [points[0], points[-1]]
            targets = pd.to_datetime([p[item['target_key']] for p in points])
            line, = axes.plot(targets, [p['prediction'] for p in points], style, color=forecast_color,
                             label=name + (' · ' + item['title'] if overlay else ''), markersize=3)
            if kind == 'administrative' and len(points) == 1:
                line.set_marker('o')
            if kind == 'administrative' and not overlay:
                axes.annotate(f"{points[-1]['prediction']:.4g} {item.get('unit', '')} · без изменений",
                              (targets[-1], points[-1]['prediction']), xytext=(-4, 9),
                              textcoords='offset points', ha='right', color=forecast_color)
            calibrated = [p for p in points if kind == 'forecast' and p['lower'] is not None and p['upper'] is not None]
            if intervals and calibrated:
                axes.fill_between(pd.to_datetime([p[item['target_key']] for p in calibrated]),
                                  [p['lower'] for p in calibrated], [p['upper'] for p in calibrated],
                                  alpha=.15, color=forecast_color, label='Диапазон 80%' if not overlay else None)
    return valid


def focus_panel(panel):
    """Keep the requested full view and add a separate recent-history view."""
    dates = [stamp for item in panel for stamp in item['history'].dropna().index]
    if not dates or not any(item['forecasts'] for item in panel) or max(dates)-min(dates) <= pd.Timedelta(days=5*365):
        return []
    first = max(dates) - pd.DateOffset(years=2)
    return [{**item, 'history': item['history'].loc[first:]} for item in panel
            if not item['history'].loc[first:].dropna().empty]


def format_axes(axes, panel, title):
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    axes.set_title(title)
    axes.set_ylabel(panel[0]['unit'])
    points = [stamp for item in panel for stamp in item['history'].dropna().index]
    if points:
        dates = '%m.%Y' if all(item['frequency'] == 'monthly' for item in panel) else '%d.%m.%Y'
        scope = f'Факты: {min(points).strftime(dates)} — {max(points).strftime(dates)}'
        future = [pd.Timestamp(p[item['target_key']]) for item in panel for p in item['forecasts']]
        if future:
            scope += f' · Будущий участок до {max(future).strftime(dates)}'
        axes.set_xlabel(scope)
    points.extend(pd.Timestamp(p[item['target_key']]) for item in panel for p in item['forecasts'])
    if points and min(points) == max(points):
        axes.set_xlim(min(points)-pd.Timedelta(days=7), max(points)+pd.Timedelta(days=7))
    elif points:
        margin = max(pd.Timedelta(days=2), (max(points)-min(points))*.025)
        axes.set_xlim(min(points)-margin, max(points)+margin)
    locator = AutoDateLocator(minticks=3, maxticks=7)
    axes.xaxis.set_major_locator(locator)
    axes.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    axes.grid(alpha=.25)
    axes.legend()


def today():
    return datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat()


def validate_options(options):
    result = dict(options)
    if result.get('period') not in {*PERIODS, 'custom'}:
        raise ValueError('Неизвестный период графика')
    if result.get('anchor', 'latest') not in {'latest', 'today'}:
        raise ValueError('Неизвестная граница графика')
    if type(result.get('horizon', 4)) is not int or not 0 <= result.get('horizon', 4) <= 4:
        raise ValueError('Горизонт: 0–4; 0 выключает прогноз')
    result.setdefault('horizon', 4)
    if result.get('forecast') is not None:
        result['forecast'] = validate_duration(result['forecast'])
    if result.get('layout', 'separate') not in {'separate', 'overlay'}:
        raise ValueError('Неизвестный вид графика')
    if result.get('series_ids') is not None:
        identities = result['series_ids']
        minimum = 0 if result.get('selection_mode') == 'multiple' else 1
        if (not isinstance(identities, list) or not minimum <= len(identities) <= 6
                or any(not isinstance(identity, str) for identity in identities) or len(set(identities)) != len(identities)):
            raise ValueError('Выберите от 1 до 6 разных показателей')
    if result['period'] == 'custom':
        start, end = export_dates(result.get('date_from'), result.get('date_to'))
        if not start or not end or end > today():
            raise ValueError('Нужны две даты, конец не позже сегодняшнего дня')
        result.update(date_from=start, date_to=end)
    return result


def bounds(options, latest):
    options = validate_options(options)
    if options['period'] == 'custom':
        return options['date_from'], options['date_to']
    days = PERIODS[options['period']]
    end = today() if options.get('anchor') == 'today' else latest or today()
    return ((date.fromisoformat(end) - timedelta(days=days - 1)).isoformat() if days else None), end


def selections(store, actor, section):
    from .subscription_access import require
    require(store, actor, section)
    from .search_provider import require_section
    require_section(store, section)
    require_actor(store, actor)
    if section not in SECTIONS:
        raise ValueError('Графики: ruonia, market, trends, wordstat')
    if section in {'market', 'trends'}:
        return [(r['id'], r['title']) for r in market.selected(store, actor)
                if section != 'trends' or r['source'] == 'google_trends']
    scope = actor_scope(store, actor, section)
    return [(phrase, phrase) for phrase in scope['phrases']]


def validate_selection(store, actor, section, identity):
    choices = selections(store, actor, section)
    if identity is not None and identity not in {key for key, _ in choices}:
        raise PermissionError('Показатель больше не доступен. Выберите его заново.')
    return choices


def latest_date(store, actor, section, identity=None, identities=None):
    choices = validate_selection(store, actor, section, identity)
    if section in {'market', 'trends'}:
        dates = [store.one('SELECT MAX(date) d FROM market_observations WHERE series_id=?', (key,))['d']
                 for key, _ in choices if (identity is None or key == identity) and (identities is None or key in identities)]
    else:
        scope = actor_scope(store, actor, section)
        dates = [r['date'] for r in scoped_rows(store, 'observations', scope, as_of=utcnow()[:10])
                 if (identity is None or r['metric'] == identity) and (identities is None or r['metric'] in identities) and r['value'] is not None and r['retrieved_at'] <= utcnow()]
    return max((d for d in dates if d and d <= utcnow()[:10]), default=None)


def chart_series(store, settings, actor, section, identity, options, selection=None, *, page=0, order=None):
    """Return history and immutable forecasts separately, suitable for plotting."""
    if options.get('selection_mode') == 'multiple' and not options.get('series_ids'):
        raise ValueError('Выберите хотя бы один показатель для сравнения')
    choices = validate_selection(store, actor, section, identity)
    if selection is not None:
        if not isinstance(selection, list) or not set(selection).issubset({key for key, _ in choices}):
            raise PermissionError('Выбор показателей изменился. Повторите запрос.')
        choices = [(key, title) for key, title in choices if key in selection]
    options = validate_options(options)
    if options.get('series_ids'):
        if not set(options['series_ids']).issubset(dict(choices)):
            raise PermissionError('Набор показателей больше не доступен')
        choices = [(key, title) for key, title in choices if key in options['series_ids']]
    page = chart_page(page)
    groups, market_data = [], {}
    if section in {'market', 'trends'}:
        for key, title in choices:
            if identity is not None and key != identity:
                continue
            row = store.one('SELECT * FROM market_series WHERE id=?', (key,))
            meta = json.loads(row['metadata'])
            history = market_store.history(store, key, as_of=utcnow())
            curve = market_analysis.values(history, meta, cutoff=utcnow())
            if not curve.empty:
                # Nulls on non-publication days are not expected observations.
                # Preserve real facts outside the usual calendar (e.g. exceptional Saturdays).
                curve = curve.reindex(curve.dropna().index.union(market_analysis.grid(curve.index.min(), curve.index.max(), meta)))
            market_data[key] = (row, history, meta, curve)
            origin = curve.dropna().index.max() if not curve.dropna().empty else None
            predictions = store.query('SELECT * FROM market_forecasts WHERE series_id=? AND origin=? ORDER BY horizon',
                                      (key, origin.date().isoformat() if origin is not None else ''))
            groups.append((title, meta['unit'], curve, predictions, 'target', meta['frequency'], key))
    else:
        scope = actor_scope(store, actor, section)
        rows = [r for r in scoped_rows(store, 'observations', scope, as_of=utcnow()[:10])
                if (identity is None or r['metric'] == identity) and r['metric'] in dict(choices)
                and r['retrieved_at'] <= utcnow()]
        if options['horizon'] and options.get('forecast') is None:
            ReportBuilder(store, settings)._save_forecasts(rows)
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row['metric'], row['unit'], row['region_id'], row['device'])].append(row)
        for (metric, unit, region, device), history in grouped.items():
            curve = pd.Series([r['value'] for r in history], index=pd.to_datetime([r['date'] for r in history]), dtype=float)
            if not curve.empty:
                if section == 'ruonia':
                    from .cbr_incremental import INDEX_METRICS
                    expected = market_analysis.grid(curve.index.min(), curve.index.max(),
                        {'frequency': 'daily', 'calendar': 'calendar' if metric in INDEX_METRICS else 'russia_working'})
                else:
                    expected = pd.date_range(curve.index.min(), curve.index.max(), freq='7D')
                curve = curve.reindex(curve.index.union(expected))
            origin = curve.dropna().index.max() if not curve.dropna().empty else None
            predictions = store.query('SELECT * FROM forecasts WHERE source=? AND metric=? AND region_id=? AND device=? AND frequency=? AND origin_date=? ORDER BY horizon',
                (scope['source'], metric, region, device, scope['frequency'], origin.date().isoformat() if origin is not None else ''))
            if options.get('forecast') is not None:
                from .market_sources import metric_id
                meta = {'id': metric_id('local.' + section, metric + ':' + str(region) + ':' + device),
                        'source': scope['source'], 'dataset': 'local.' + section, 'title': metric, 'kind': 'series',
                        'adapter': 'stored', 'frequency': scope['frequency'], 'unit': unit,
                        'calendar': 'calendar' if section != 'ruonia' or metric in {'Индекс RUONIA', '1 мес', '3 мес', '6 мес'} else 'russia_working',
                        'week_anchor': curve.dropna().index.max().strftime('%a').upper() if not curve.dropna().empty else 'MON', 'definition': '1'}
                meta['metric'] = metric
                if section == 'ruonia':
                    meta['id'] = metric_id('local.ruonia', metric)
                market_store.register(store, [meta])
                from .market_incremental import only_missing
                market_store.save_rows(store, meta['id'], only_missing(store, meta['id'], history))
                row = store.one('SELECT * FROM market_series WHERE id=?', (meta['id'],))
                predictions = market.save_forecasts(store, row, history, horizon=from_options(options, scope['frequency'])) or []
                for prediction in predictions:
                    prediction['target_date'] = prediction['target']
            groups.append((metric, unit, curve, predictions, 'target_date', scope['frequency'], dumps([metric, unit, region, device])))
    latest = max((curve.dropna().index.max().date().isoformat() for _, _, curve, _, _, _, _ in groups if not curve.dropna().empty), default=None)
    start, end = bounds(options, latest)
    result = []
    for title, unit, curve, predictions, target_key, frequency, key in groups:
        valid = curve.dropna()
        visible = curve[(curve.index <= pd.Timestamp(end)) & (curve.index >= pd.Timestamp(start) if start else True)]
        if options['period'] == 'all' and section in {'market', 'trends'} and not visible.dropna().empty:
            visible = visible.loc[visible.dropna().index.min():visible.dropna().index.max()]
        # A historical display must never overlay a forecast trained on later observations.
        duration = from_options(options, frequency)
        final_target = end_date(valid.index.max(), duration).date().isoformat() if not valid.empty else end
        predictions = [p for p in predictions if duration['value'] and p[target_key] <= (
            end_date(p['created_at'][:10], duration).date().isoformat()
            if p.get('note') == 'Сценарий сохранения административного значения' else final_target
        ) and p[target_key] > utcnow()[:10]] if not valid.empty and end >= valid.index.max().date().isoformat() else []
        result.append({'identity': key, 'title': title, 'unit': unit, 'history': visible, 'forecasts': predictions,
                       'target_key': target_key, 'frequency': frequency,
                       'administrative': bool(market_data.get(key) and market_data[key][2].get('administrative'))})
    # Calculate only the page that will actually be delivered, on full history.
    for item in page_items(result, page, order):
        if item['identity'] not in market_data:
            continue
        row, history, meta, curve = market_data[item['identity']]
        duration = from_options(options, meta['frequency'])
        valid = curve.dropna()
        if not duration['value'] or end < valid.index.max().date().isoformat():
            continue
        calculated = (market.save_forecasts(store, row, history, horizon=duration)
                      if options.get('forecast') is not None else market.save_forecasts(store, row, history))
        if isinstance(calculated, list):
            final_target = end_date(valid.index.max(), duration).date().isoformat()
            item['forecasts'] = [p for p in calculated if p['target'] > utcnow()[:10] and p['target'] <= (
                end_date(p['created_at'][:10], duration).date().isoformat()
                if p.get('note') == 'Сценарий сохранения административного значения' else final_target)]
    return result, start, end


def run_chart_job(store, settings, job, payload, *, dry_run=False):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from threadpoolctl import threadpool_limits
    actor, section = payload['actor_id'], payload['section']
    if payload.get('chart_auto'):
        from .chart_batches import active as batch_active
        if not batch_active(store, payload['chart_auto'], str(actor)):
            return None
    reference = payload.get('chart_schedule')
    if reference:
        from .chart_schedules import active
        if not active(store, reference, str(actor)):
            return None
    validate_selection(store, actor, section, payload.get('identity'))
    identifier = payload['report_id']
    from .telegram_data_jobs import report_identifier
    report_identifier(identifier)
    fingerprint = f"telegram-chart:{job['id']}"
    existing = store.one('SELECT * FROM reports WHERE id=?', (identifier,))
    if existing:
        if existing['fingerprint'] != fingerprint or json.loads(existing['quality']).get('actor_id') != actor:
            raise PermissionError('Этот график принадлежит другому заданию')
    else:
        collection = {}
        if reference and payload.get('refresh') and not dry_run:
            if section in {'ruonia', 'wordstat'}:
                from .collectors import Collector
                collection = Collector(store, settings).collect(section)
                if section == 'ruonia':
                    from .market_alerts import process
                    process(store, settings)
            elif section == 'market':
                collection = market.collect(store)
        with threadpool_limits(limits=4):
            series, start, end = chart_series(store, settings, actor, section, payload.get('identity'), payload['options'], payload.get('selection'),
                                              page=payload.get('page', 0), order=payload.get('order'))
        root = (settings.reports_dir / identifier).resolve()
        if not root.is_relative_to(settings.reports_dir.resolve()):
            raise ValueError('Invalid chart path')
        root.mkdir(parents=True, exist_ok=True)
        parts, notes = [], []
        if not series:
            notes.append({
                'wordstat': 'Нет сохранённой недельной истории для выбранных фраз. Откройте «Графики → Wordstat → Загрузить данные из Яндекса» или выполните /report wordstat. Это отдельный запрос к Wordstat API; после получения отчёта постройте график снова.',
                'ruonia': 'Нет сохранённой истории RUONIA для выбранных показателей. Получите отчёт RUONIA (/report ruonia), затем постройте график снова.',
                'market': 'Нет сохранённых данных рынка для выбранных показателей. Источники включает администратор в «Источниках»; затем получите отчёт рынка (/report market).',
                'trends': 'Нет данных Google Trends. Администратор загружает CSV через /trends; личный набор выбирается через /indicators.',
            }[section])
        available = ordered_available(series, payload.get('order'))
        page = chart_page(payload.get('page', 0))
        order = payload.get('order')
        pages = max(1, ((len(order) if order is not None else len(available)) + PAGE_SIZE - 1) // PAGE_SIZE)
        if page >= pages:
            raise ValueError('Эта страница больше недоступна. Постройте графики заново.')
        displayed = page_items(series, page, order)
        missing = [item for item in series if item['history'].dropna().empty]
        notes.extend(item['title'] + ': нет данных в выбранном периоде.' for item in missing[:6])
        if len(missing) > 6:
            notes.append(f'Ещё {len(missing)-6} рядов без данных в выбранном периоде. Они не занимают места среди графиков.')
        panels = [[item] for item in displayed]
        if payload['options'].get('layout') == 'overlay':
            compatible = defaultdict(list)
            for item in displayed:
                compatible[item['unit']].append(item)
            panels = list(compatible.values())
        for panel in panels:
            nonempty = [item for item in panel if not item['history'].dropna().empty]
            for item in panel:
                if item['history'].dropna().empty:
                    notes.append(item['title'] + ': нет данных в выбранном периоде.')
            if not nonempty:
                continue
            focused = focus_panel(nonempty)
            figure = Figure(figsize=(10, 7 if focused else 4), dpi=120, layout='constrained')
            FigureCanvasAgg(figure)
            if focused:
                axes, focus_axes = figure.subplots(2, 1)
            else:
                axes = figure.subplots()
            palette = ('#246b9b', '#b56e13', '#8b4d86', '#607831', '#c65e76', '#4c7777')
            for position, item in enumerate(nonempty):
                curve, predictions = item['history'], item['forecasts']
                valid = plot_item(axes, item, overlay=len(panel) > 1, intervals=payload['options'].get('intervals', True),
                                  color=palette[position % len(palette)])
                notes.append(f"{item['title']}: {valid.index.min().date()} — {valid.index.max().date()}, наблюдений {len(valid)}, охват {curve.notna().mean():.1%}, дат без значения {curve.isna().sum()}.")
                if predictions:
                    last = predictions[-1]
                    checked = [p for p in predictions if prediction_kind(p) == 'forecast']
                    if checked:
                        point = checked[-1]
                        metrics = ', '.join(f'{label} {point[key]:.4g}' for label, key in (
                            ('MAE', 'validation_mae'), ('базовый MAE', 'baseline_mae')) if point.get(key) is not None)
                        notes.append(f"{item['title']}: проверенный горизонт до {point[item['target_key']]}; проверок {point['validation_n']}" + (', ' + metrics if metrics else '') + '.')
                    if prediction_kind(last) == 'scenario':
                        notes.append(item['title'] + ': дальняя линия — сценарий сохранения последнего значения; достаточной проверки и диапазона нет.')
                    elif prediction_kind(last) == 'administrative':
                        notes.append(f"{item['title']}: условие «без изменений» — {last['prediction']:.4g} {item['unit']} до {last[item['target_key']]}. "
                                     'Это не предсказание решения ЦБ. Вероятность, ошибка и прогнозный диапазон не рассчитаны; будущая ставка может измениться.')
                elif from_options(payload['options'], item['frequency'])['value']:
                    reason = ('недостаточно наблюдений или охвата для прогноза' if len(valid) < market_analysis.MINIMUM.get(item['frequency'], 24) or curve.notna().mean() < .9
                              else 'текущий прогноз недоступен: ряд архивный, устаревший, выбран исторический период либо не хватает непрерывной истории')
                    notes.append(item['title'] + ': ' + reason + '.')
            title = ' / '.join(item['title'] for item in nonempty)[:100]
            format_axes(axes, nonempty, title)
            if focused:
                for position, item in enumerate(focused):
                    plot_item(focus_axes, item, overlay=len(focused) > 1, intervals=payload['options'].get('intervals', True),
                              color=palette[position % len(palette)])
                format_axes(focus_axes, focused, 'Увеличенный участок: последние два года наблюдений и будущие значения')
                notes.append('Нижняя панель увеличена для чтения будущего участка; верхняя сохраняет весь выбранный период. Шкалы по вертикали подобраны отдельно.')
            path = root / f'{len(parts)}.png'
            figure.savefig(path, dpi=120)
            parts.append({'kind': 'photo', 'path': str(path), 'caption': title})
        labels = {'market': 'рынка и ЦБ', 'ruonia': 'RUONIA', 'trends': 'Google Trends', 'wordstat': 'Wordstat'}
        period_text = f'{start} — {end}' if start else f'вся сохранённая история; последняя дата данных — {end}'
        summary = f"Графики {labels[section]}. История: {period_text}.\nСтраница {page+1}/{pages}: показателей {len(displayed)} из {len(available)} с данными.\nИспользована сохранённая история. Пропуски не соединяются линией; отдельные наблюдения показаны точками. Прогноз отсчитывается от последнего факта. Условный сценарий не предсказывает решений регулятора."
        if any(item['frequency'] == 'monthly' for item in displayed):
            summary += '\nМесячные наблюдения подписаны первым числом соответствующего месяца. Это не ежедневный ряд и не подтверждение значения на сегодня.'
        if section == 'trends':
            from .google_trends import NOTE
            summary += '\n' + NOTE
        if pages > 1:
            notes.append('Остальные графики доступны по кнопкам страниц под выпуском. Сначала показаны свежие ряды с большим охватом; редкие и архивные тоже доступны.')
        summary += '\n' + '\n'.join(notes)
        if collection.get('errors'):
            summary += '\nОбновление неполное: ' + '; '.join(str(error) for error in collection['errors'])
        validate_selection(store, actor, section, payload.get('identity'))
        store.execute('INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,quality,parts,fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?)',
            (identifier, section, utcnow(), store.version('keyword'), store.version('cohort'), summary, summary,
             dumps({'actor_id': actor, 'options': payload['options'], 'date_from': start, 'date_to': end, 'collection': collection,
                    'chart_page': page, 'chart_pages': pages, 'chart_total': len(available),
                    'chart_batch_id': payload.get('chart_batch_id', identifier),
                    'chart_request': {'section': section, 'identity': payload.get('identity'), 'options': payload['options'],
                                      'selection': payload.get('selection', [key for key, _ in selections(store, actor, section)]),
                                      'order': order if order is not None else [item['identity'] for item in available]}}), dumps(parts), fingerprint))
    allowed = {key for key, _ in validate_selection(store, actor, section, payload.get('identity'))}
    if not set(payload.get('selection', [])).issubset(allowed):
        raise PermissionError('Выбор показателей изменился. Повторите запрос.')
    if not dry_run:
        prepare_delivery(store, identifier, str(actor), detail='full', chart_schedule=reference, chart_auto=payload.get('chart_auto'))
    return identifier
