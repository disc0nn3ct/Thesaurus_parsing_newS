"""FX reports use the existing sequential worker and resumable delivery."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from . import dollar, market
from .delivery import prepare_delivery
from .store import dumps, utcnow
from .telegram_data_jobs import require_actor


def run_job(store, settings, job, payload, *, dry_run=False, charts=True):
    actor = payload.get('actor_id')
    require_actor(store, actor, admin=job['kind'] == 'dollar_collect')
    from .subscription_access import require
    require(store, actor, 'market')
    fingerprint = f'dollar-job:{job["id"]}'
    existing = store.one('SELECT id FROM reports WHERE fingerprint=?', (fingerprint,))
    if existing:
        if not dry_run and not payload.get('silent'):
            prepare_delivery(store, existing['id'], str(actor), detail='full')
        return existing['id']
    if job['kind'] == 'dollar_collect':
        if payload.get('history') not in {None, 'full'}:
            raise ValueError('Invalid history selection')
        if payload.get('scope') == 'context':
            from .dollar_sources import PACK
            from .market_sources import Transport
            selection = payload.get('selection', PACK)
            if not selection or not isinstance(selection, list) or not set(selection).issubset(PACK):
                raise ValueError('Invalid USD source selection')
            if not dry_run:
                for identity in selection:
                    if payload.get('history') == 'full' and (identity == dollar.USD or identity.startswith('fred.')):
                        store.set_setting('market_history_start:' + identity, '1992-07-01')
                    market.enable(store, identity, True)
            transport = Transport()
            today = dollar.today_moscow(utcnow())
            quality = {'errors': [], 'sources': selection, 'status': 'ok'}
            if dollar.USD in selection:
                update = market.collect(store, selection=[dollar.USD], transport=transport, today=today+timedelta(days=1), dry_run=dry_run)
                quality['errors'].extend(update['errors'])
            other = [i for i in selection if i != dollar.USD]
            if other:
                update = market.collect(store, selection=other, transport=transport, today=today, dry_run=dry_run)
                quality['errors'].extend(update['errors'])
            if quality['errors']:
                quality['status'] = 'partial'
            store.set_setting('dollar_context_collection', {'at': utcnow(), **quality})
        else:
            if not dry_run:
                if payload.get('history') == 'full':
                    store.set_setting('market_history_start:' + dollar.USD, '1992-07-01')
                market.enable(store, dollar.USD, True)
            quality = market.collect(store, selection=[dollar.USD], dry_run=dry_run)
        text = 'USD: обновление завершено. История сохранена; следующие загрузки используют общий механизм дозагрузки.'
        if payload.get('scope') == 'context':
            text = f"Данные для доллара обновлены. Выбрано источников: {len(selection)}.\nОткройте /dollar → «Показать обзор» либо «Моя покупка»."
        if quality.get('errors'):
            text += '\nОбновление неполное: '+str(len(quality['errors']))+' ошибок. Остальные источники сохранены; подробности — в «Источниках и свежести». '
            for error in quality['errors']:
                source = store.one('SELECT title FROM market_series WHERE id=?', (error['source'],))
                text += '\n'+(source['title'] if source else error['source'])+': '+error['error']
        if payload.get('history') == 'full':
            span = store.one('SELECT MIN(date) first,MAX(date) last,COUNT(*) n FROM market_observations WHERE series_id=? AND value IS NOT NULL', (dollar.USD,))
            text += f"\nИстория USD: {span['first'] or 'нет данных'} — {span['last'] or 'нет данных'}, наблюдений {span['n']}. Курс до 1998 года пересчитан 1000:1 в современные рубли."
        parts, metrics = [], {'collection': quality}
    elif job['kind'] == 'dollar_analysis':
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):
            metrics, series = dollar.build(store, payload.get('config', {}), now=utcnow())
        text = dollar.summary(metrics)
        full_text = dollar.render(metrics)
        identifier = uuid4().hex
        root = (settings.reports_dir / identifier).resolve()
        if not root.is_relative_to(settings.reports_dir.resolve()):
            raise ValueError('Invalid artifact path')
        root.mkdir(parents=True, exist_ok=True)
        output = root / 'dollar-analysis.json'
        output.write_text(dumps(metrics), encoding='utf-8')
        (root / 'dollar-analysis.txt').write_text(full_text, encoding='utf-8')
        parts = [{'kind': 'document', 'path': str(output), 'caption': 'USD/RUB: расчёты, сценарии и версии исходных данных'}]
        if charts and not series.dropna().empty:
            import pandas as pd
            from matplotlib import dates
            from matplotlib.figure import Figure
            figure = Figure(figsize=(10, 5), layout='constrained')
            ax = figure.subplots()
            explicit_window = metrics['config'].get('history_start') or metrics['config'].get('history_end')
            recent = series if explicit_window else series.loc[series.index >= series.index[-1]-pd.Timedelta(days=365)]
            recent = recent.reindex(market.analysis.grid(recent.index[0], recent.index[-1], metrics['model_metadata']))
            ax.plot(recent.index, recent, label='Официальный курс ЦБ')
            predictions = metrics['forecast']
            if predictions:
                validated = all(p.get('validation_n', 0) >= 10 for p in predictions)
                ax.plot(pd.to_datetime([p['target'] for p in predictions]), [p['prediction'] for p in predictions],
                        '--', color='tab:orange' if validated else 'grey', label='Локальный прогноз' if validated else 'Базовый сценарий (не проверен)')
                interval = [p for p in predictions if p['lower'] is not None and p.get('validation_n', 0) >= 10]
                if interval:
                    ax.fill_between(pd.to_datetime([p['target'] for p in interval]), [p['lower'] for p in interval],
                                    [p['upper'] for p in interval], alpha=.2, color='tab:orange', label='Диапазон 80%')
            if metrics['quote_fresh']:
                ax.scatter([datetime.fromisoformat(metrics['created_at']).astimezone(timezone.utc).replace(tzinfo=None)],
                           [metrics['config']['quote']], color='tab:red', label='Ваша цена банка (без комиссии)', zorder=5)
            ax.set(title='USD/RUB · цена доллара для поездки', ylabel='Рублей за 1 USD')
            ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.grid(alpha=.2)
            ax.legend()
            image = root / 'dollar.png'
            figure.savefig(image, dpi=140)
            parts.insert(0, {'kind': 'photo', 'path': str(image), 'caption': 'USD/RUB: факты, ваша цена и локальные сценарии'})
            if metrics['timing']:
                from .dollar_charts import near_term
                parts.insert(0, near_term(series, metrics['timing'], root/'dollar-near-term.png', period=metrics.get('period')))
            if metrics['long_term']:
                if any(p.get('distribution', {}).get('status') in {'research', 'evaluated'} for p in metrics['long_term'] if p.get('distribution')):
                    from .dollar_distribution import chart
                else:
                    from .dollar_long import chart
                parts.insert(0, chart(series, metrics['long_term'], root/'dollar-long-term.png'))
    else:
        raise ValueError('Unknown dollar job')
    require_actor(store, actor)
    report_id = uuid4().hex
    with store.connection() as conn:
        # Forecasts and the report commit together: recovery cannot publish a different model run.
        if job['kind'] == 'dollar_analysis' and metrics['forecast'] and not metrics.get('period', {}).get('retrospective'):
            run = uuid4().hex
            conn.execute('INSERT INTO forecast_runs(id,series_id,fingerprint,created_at,duration,inputs) VALUES(?,?,?,?,?,?)',
                         (run, dollar.USD, fingerprint, metrics['created_at'], dumps(metrics['forecast_duration']),
                          dumps({'revisions': metrics['input_revisions'], 'metadata': metrics['model_metadata'],
                                 'published_revision': metrics['published_revision'],
                                 'features': metrics['model_inputs'], 'history_mode': metrics['config']['history_mode'],
                                 'period': metrics.get('period'), 'as_of': metrics['created_at'], 'model_version': 'dollar-4-periods'})))
            for point in metrics['forecast']:
                conn.execute('INSERT INTO forecast_points(run_id,target,payload) VALUES(?,?,?)', (run, point['target'], dumps(point)))
        if job['kind'] == 'dollar_analysis' and metrics['long_term'] and not metrics.get('period', {}).get('retrospective'):
            run = uuid4().hex
            conn.execute('INSERT INTO forecast_runs(id,series_id,fingerprint,created_at,duration,inputs) VALUES(?,?,?,?,?,?)',
                         (run, dollar.USD, fingerprint+':long', metrics['created_at'],
                          dumps({'unit': 'months', 'value': metrics['config']['analysis_months']}),
                          dumps({'revisions': metrics['input_revisions'], 'metadata': metrics['model_metadata'],
                                 'history_mode': metrics['config']['history_mode'], 'method': metrics['config']['long_method'],
                                 'distribution': metrics['config']['distribution'],
                                 'period': metrics.get('period'), 'as_of': metrics['created_at'], 'model_version': 'dollar-monthly-3-periods'})))
            for point in metrics['long_term']:
                conn.execute('INSERT INTO forecast_points(run_id,target,payload) VALUES(?,?,?)', (run, point['target'], dumps(point)))
        conn.execute("INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,metrics,quality,parts,fingerprint) VALUES(?,'market',?,1,0,?,?,?,?,?,?)",
                     (report_id, utcnow(), text, full_text if job['kind'] == 'dollar_analysis' else text, dumps(metrics), dumps({'actor_id': actor, 'tool': 'dollar'}), dumps(parts), fingerprint))
    if not dry_run and not payload.get('silent'):
        prepare_delivery(store, report_id, str(actor), detail='full')
    return report_id
