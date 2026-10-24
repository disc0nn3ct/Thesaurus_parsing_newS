"""Charts of observed operational history, with no fabricated past activity."""
import io
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import monitoring

MOSCOW = ZoneInfo('Europe/Moscow')


def activity_data(store, scope, days, *, now=None):
    now = now or datetime.now(timezone.utc)
    where, values = monitoring.filters(scope, days, until=now.isoformat())
    archive = store.one(f"SELECT COUNT(*) n FROM monitor_events WHERE {where} AND origin='history'", values)['n']
    first = store.one(f"SELECT MIN(occurred_at) first FROM monitor_events WHERE {where} AND origin='live'", values)['first']
    step = 3600 if days == 1 else 86400
    rows = store.query(f'''SELECT CAST((strftime('%s',occurred_at)+10800)/? AS INTEGER)*?-10800 bucket,
        SUM(category='interaction') interactions,
        COUNT(DISTINCT CASE WHEN category='interaction' THEN actor_id END) users,
        SUM(category='job' AND outcome='done') done,
        SUM(category='job' AND outcome='failed') failed,
        SUM(category='job' AND outcome='cancelled') cancelled,
        SUM(level='error') errors,SUM(level='warning') warnings,
        AVG(CASE WHEN category='interaction' THEN duration_ms END)/1000 latency,
        MAX(CASE WHEN category='interaction' THEN duration_ms END)/1000 slowest
        FROM monitor_events WHERE {where} AND origin='live' GROUP BY bucket ORDER BY bucket''', [step,step,*values])
    return {'rows':rows,'step':step,'archive':archive,'first':first,'end':now.isoformat()}


def resource_data(store, days, *, now=None):
    if days not in {0,1,7,30}:
        raise ValueError('Неверный период графика')
    now = now or datetime.now(timezone.utc)
    cutoff = (now-timedelta(days=days or 90)).isoformat()
    step = {1:120,7:900,30:3600,0:10800}[days]
    expressions = ','.join(f"{operation}(json_extract(metrics,'$.{key}')) {alias}" for alias,key,operation in (
        ('cpu','process_cpu_percent','AVG'),('cpu_peak','process_cpu_percent','MAX'),
        ('rss','process_rss_mb','AVG'),('rss_peak','process_rss_mb','MAX'),
        ('disk','disk_free_gb','MIN'),('database','database_mb','MAX')))
    rows = store.query(f'''SELECT runtime_id,CAST(strftime('%s',created_at)/? AS INTEGER)*? bucket,
        MIN(created_at) first,MAX(created_at) last,COUNT(*) samples,{expressions}
        FROM monitor_samples WHERE created_at>=? AND created_at<=?
        GROUP BY runtime_id,bucket ORDER BY bucket,runtime_id''',(step,step,cutoff,now.isoformat()))
    starts = store.query('SELECT started_at FROM monitor_runtime WHERE started_at>=? AND started_at<=? ORDER BY started_at',
                         (cutoff,now.isoformat()))
    return {'rows':rows,'step':step,'starts':starts}


def _figure(title):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(12,8.5),layout='constrained')
    figure.suptitle(title,fontsize=15)
    axes = figure.subplots(2,2,sharex=True)
    return figure,list(axes.flat)


def _finish(figure, axes):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    for axis in axes:
        if not axis.has_data():
            axis.set_axis_off()
            continue
        locator = AutoDateLocator(tz=MOSCOW,minticks=3,maxticks=6)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(ConciseDateFormatter(locator,tz=MOSCOW,
            formats=['%Y','%m.%Y','%d.%m','%H:%M','%H:%M','%S'],
            zero_formats=['','%Y','%m.%Y','%d.%m','%H:%M','%H:%M'],
            offset_formats=['','%Y','%m.%Y','%d.%m.%Y','%d.%m.%Y','%d.%m.%Y %H:%M']))
        axis.grid(alpha=.2)
        axis.tick_params(labelsize=9,labelbottom=True)
        axis.ticklabel_format(axis='y',style='plain',useOffset=False)
    output = io.BytesIO()
    FigureCanvasAgg(figure).print_png(output)
    output.seek(0)
    output.name = 'monitoring-history.png'
    return output


def render_activity(data, title):
    figure,axes = _figure('История системы · '+title)
    titles = ['Действия пользователей и активные пользователи','Завершения заданий · события',
              'Ошибки и предупреждения · все события','Время обработки действия ботом']
    for axis,label in zip(axes,titles,strict=True):
        axis.set_title(label,fontsize=11)
    rows = {row['bucket']:row for row in data['rows']}
    if not rows:
        for axis in axes:
            axis.text(.5,.5,'Нет наблюдаемых событий за этот период',ha='center',transform=axis.transAxes)
        return _finish(figure,axes)
    step = data['step']
    last = int(datetime.fromisoformat(data['end']).timestamp())
    last = (last+10800)//step*step-10800
    buckets = list(range(min(rows),last+1,step))
    dates = [datetime.fromtimestamp(value,timezone.utc) for value in buckets]
    def counts(key):
        return [rows[value][key] if value in rows else 0 for value in buckets]
    width = step/86400*.75
    axes[0].bar(dates,counts('interactions'),width=width,color='#377eb8',label='Действия')
    axes[0].plot(dates,counts('users'),color='#6a3d9a',marker='.',label='Уникальные пользователи за интервал')
    bottom = [0]*len(buckets)
    for key,label,color in [('done','Готово','#328450'),('failed','Ошибка','#d1495b'),('cancelled','Отмена','#999999')]:
        values = counts(key)
        axes[1].bar(dates,values,width=width,bottom=bottom,label=label,color=color)
        bottom = [a+b for a,b in zip(bottom,values,strict=True)]
    for key,label,color in [('errors','Ошибки','#d1495b'),('warnings','Предупреждения','#e69731')]:
        axes[2].plot(dates,counts(key),marker='.',label=label,color=color)
    for key,label,color in [('latency','Среднее','#377eb8'),('slowest','Максимум','#e69731')]:
        values = [rows[value][key] if value in rows and rows[value][key] is not None else math.nan for value in buckets]
        axes[3].plot(dates,values,marker='.',label=label,color=color)
    for axis in axes[:3]:
        from matplotlib.ticker import MaxNLocator
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_ylabel('Количество')
    axes[3].set_ylabel('Секунды')
    for axis in axes:
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8)
    figure.supxlabel(('По часам' if step==3600 else 'По дням')+' · время Москвы · текущий интервал неполный\n'
                    'Число записанных событий; отсутствие записей не подтверждает доступность бота.',fontsize=10)
    return _finish(figure,axes)


def render_resources(data, period=''):
    figure,axes = _figure('Ресурсы процесса бота и диска сервера'+(' · '+period if period else ''))
    specifications = [('cpu','cpu_peak','CPU процесса · % одного ядра','%'),
                      ('rss','rss_peak','Память процесса','МБ'),('disk',None,'Свободное место на диске','ГБ'),
                      ('database',None,'Размер базы с WAL','МБ')]
    for axis,(key,peak,title,unit) in zip(axes,specifications,strict=True):
        axis.set_title(title,fontsize=11)
        axis.set_ylabel(unit)
        segments = []
        for row in data['rows']:
            if not segments or row['runtime_id']!=segments[-1][-1]['runtime_id'] or row['bucket']-segments[-1][-1]['bucket']>data['step']+90:
                segments.append([])
            segments[-1].append(row)
        drawn = False
        for field,color,label in [(key,'#377eb8','Среднее' if peak else 'Минимум за интервал' if key=='disk' else 'Максимум за интервал'),
                                  (peak,'#e69731','Максимум за интервал')]:
            if field is None:
                continue
            labelled = False
            for segment in segments:
                dates = [datetime.fromtimestamp(row['bucket'],timezone.utc) for row in segment]
                values = [row[field] if row[field] is not None else math.nan for row in segment]
                if not any(math.isfinite(value) for value in values):
                    continue
                axis.plot(dates,values,marker='.',markersize=3,color=color,label=None if labelled else label)
                labelled,drawn = True,True
        for index,start in enumerate(data['starts'][-20:]):
            axis.axvline(datetime.fromisoformat(start['started_at']),color='#777777',alpha=.4,linestyle='--',
                         label='Запуск процесса' if index==0 else None)
        if drawn:
            if key=='cpu':
                axis.set_ylim(bottom=0)
            axis.legend(fontsize=8)
        else:
            axis.text(.5,.5,'Нет измерений',ha='center',transform=axis.transAxes)
    figure.supxlabel(f"Время Москвы · интервал агрегации {data['step']//60} мин.\n"
                     'Разрывы и разные запуски не соединяются. Нет измерения — нет оценки нагрузки.',fontsize=10)
    return _finish(figure,axes)
