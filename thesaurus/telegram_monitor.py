"""Administrator-only operational dashboards and paginated history."""
import csv
import io
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import monitoring
from .monitor_content import redact_text, selected_parameters
from .store import utcnow
from .telegram_navigation import checked, screen

STATES = {'ok':'выполнено','pending':'в очереди','running':'выполняется','done':'готово','failed':'ошибка',
          'cancelled':'отменено','sending':'отправляется','sent':'отправлено','retry':'ожидает повтора',
          'uncertain':'доставка не подтверждена','superseded':'заменено новым уведомлением',
          'denied':'доступ запрещён','invalid':'неверный ввод','stale':'устаревшая кнопка','unsupported':'тип сообщения пока не поддерживается',
          'interrupted':'прерывание','partial':'частично','enabled':'включено','disabled':'выключено',
          'disconnected':'нет связи','cached':'из сохранённых данных','new':'новое','resolved':'решено'}
STATES.update(not_loaded='данные не загружены',error='ошибка источника',unavailable='источник недоступен')
NAMES = {'report':'Отчёт','access_notice':'Уведомление о правах','stock_added':'Пополнение винотеки',
         'profile_updated':'Обновлён профиль пользователя',
         'stock_removed':'Списание из винотеки','income':'Записан доход','expense':'Записан расход',
         'receipt_confirmed':'Подтверждён чек','started':'Запуск сервера','stopped':'Остановка сервера',
         'failed':'Ошибка сервера','telegram_connected':'Связь с Telegram установлена',
         'telegram_disconnected':'Связь с Telegram потеряна','previous_run_interrupted':'Предыдущий запуск прерван',
         'allow':'Доступ к боту разрешён','deny':'Доступ к боту отключён',
         'dollar_analysis':'Анализ USD/RUB','dollar_collect':'Обновление данных USD/RUB',
         'telegram_chart':'Построение графика','telegram_export':'Экспорт данных','telegram_backtest':'Проверка прогноза',
         'market_collect':'Обновление источников рынка','market_report':'Обзор рынка',
         'market_export':'Экспорт рынка','market_backtest':'Проверка модели рынка',
         'vivino_map':'Карта Vivino','wine_metadata':'Поиск сведений о вине',
         'wine_photo':'Распознавание этикетки','life_receipt':'Распознавание чека'}
NAMES.update(wine_details_lookup='Поиск характеристик сохранённого вина',details_updated='Уточнены характеристики вина')
PREFIXES = {'menu':'Меню','pj':'Проекты','ua':'Права пользователя','user':'Пользователи','wc':'Винотека',
            'wn':'Вина','lf':'Доходы и расходы','rc':'Чек','vi':'Vivino','va':'Каталог Vivino','usd':'Доллар',
            'cb':'Показатели ЦБ','ch':'График','cs':'Расписание графиков','mk':'Рынок','ms':'Источники',
            'mi':'Показатели','ma':'Уведомления','ig':'Проверка данных','gt':'Поисковые данные',
            'ui':'Настройки','kw':'Ключевые слова','rg':'Регионы','sub':'Подписки','su':'Подписки',
            'alert':'Оповещения','proposal':'Предложения','adminpage':'Страница управления',
            'history':'История отчётов','request':'Запрос отчёта','resolve':'Решение о доставке','mon':'Мониторинг'}
OPERATIONS = {'home':'Обзор','menu':'Меню','open':'Открытие','add':'Добавление','edit':'Изменение',
              'set':'Настройка','on':'Включение','off':'Выключение','save':'Сохранение','confirm':'Подтверждение',
              'cancel':'Отмена','back':'Назад','page':'Страница','users':'Пользователи','admin':'Управление',
              'feed':'Журнал','server':'Сервер','metrics':'Нагрузка','export':'Экспорт','types':'Фильтр событий',
              'user':'История пользователя','identity':'Обновление имени','feature':'Изменение функции',
              'functions':'Функции','role':'Изменение роли','roleconfirmed':'Подтверждение роли','notify':'Уведомление',
              'refresh':'Обновление','report':'Отчёт','receipt':'Загрузка чека','consume':'Списание',
              'drink':'Бутылка выпита','history':'История','prices':'Цены','enrich':'Поиск характеристик',
              'pick':'Выбор найденного значения'}
MEDIA = {'text':'Текст','button':'Нажатие кнопки','document':'Документ','photo':'Фото','video':'Видео',
         'audio':'Аудиофайл','voice':'Голосовое сообщение','video_note':'Видеосообщение','animation':'Анимация',
         'sticker':'Стикер','contact':'Контакт','location':'Геопозиция','other':'Другой тип сообщения'}
PARAMETER_LABELS = {'config':'Настройки','options':'Настройки','mode':'Режим','section':'Раздел','selection':'Выбор показателей',
                    'trip':'Срок покупки','quote':'Курс банка','forecast_date':'Дата прогноза','analysis_months':'Месяцев анализа',
                    'train_start':'История от','train_end':'История до','date_from':'Период от','date_to':'Период до',
                    'horizon':'Горизонт','horizon_days':'Дней прогноза','period':'Период','total':'Нужно всего',
                    'owned':'Уже есть','currency':'Валюта','name':'Название','winery':'Производитель','vintage':'Год',
                    'volume_ml':'Объём, мл','format':'Формат','detail':'Подробность','query':'Поиск','identity':'Показатель'}


def at(value):
    if not value:
        return 'нет данных'
    try:
        return datetime.fromisoformat(value).astimezone(ZoneInfo('Europe/Moscow')).strftime('%d.%m.%Y %H:%M:%S')
    except (ValueError,TypeError):
        return 'нет данных'


def action_name(row):
    action = row['action']
    pieces = action.split(':')
    if row['category'] == 'access':
        from .feature_access import CATALOG as FEATURES
        from .subscription_access import CATALOG as SECTIONS
        if len(pieces) == 2 and pieces[0] in {'grant','revoke','feature_grant','feature_revoke'}:
            enabled = pieces[0] in {'grant','feature_grant'}
            title = FEATURES.get(pieces[1],(pieces[1],))[0] if pieces[0].startswith('feature') else SECTIONS.get(pieces[1],pieces[1])
            return ('Разрешено: ' if enabled else 'Отключено: ')+title
        if action == 'role:admin':
            return 'Назначен администратором'
        if action == 'role:subscriber':
            return 'Сняты права администратора'
    if pieces[0] == 'command':
        return 'Команда /'+pieces[-1]
    if pieces[0] == 'button':
        return PREFIXES.get(pieces[1],'Кнопка')+(' · '+OPERATIONS.get(pieces[2],pieces[2]) if len(pieces)>2 else '')
    if pieces[0] == 'input':
        return 'Ввод в форме'+(' · добавление пользователя' if pieces[-1]=='add_user' else '')
    if action == 'upload:document':
        return 'Загрузка документа'
    if action == 'upload:photo':
        return 'Загрузка фотографии'
    if pieces[0]=='upload':
        return 'Получено: '+MEDIA.get(pieces[-1],pieces[-1])
    return NAMES.get(action,action.replace('_',' '))


class MonitoringControls:
    def __init__(self, store, reply):
        self.store, self.reply = store, reply

    def _authorize(self, actor):
        if not self.store.authorize(actor, admin=True):
            raise PermissionError('Мониторинг и история доступны администратору')

    def _reply(self, client, actor, text, buttons):
        self._authorize(actor)
        from .telegram_admin import _keyboard
        return self.reply(client,actor,text,_keyboard(buttons))

    def dispatch(self, client, actor, argument=''):
        self._authorize(actor)
        if not argument:
            return self.home(client,actor)
        if argument == 'server':
            return self.server(client,actor)
        from .user_management import resolve_user_reference
        target = resolve_user_reference(self.store,client,argument)
        return self.user(client,actor,target)

    def callback(self, client, actor, data):
        self._authorize(actor)
        args = data.split(':')
        if data == 'mon:home':
            return self.home(client,actor)
        if data == 'mon:server':
            return self.server(client,actor)
        if len(args)==3 and args[1]=='users':
            return self.users(client,actor,int(args[2]))
        if len(args)==3 and args[1]=='user':
            return self.user(client,actor,int(args[2]))
        if len(args)==3 and args[1]=='event':
            return self.event(client,actor,int(args[2]))
        if len(args)==6 and args[1]=='feed':
            return self.feed(client,actor,args[2],int(args[3]),args[4],int(args[5]))
        if len(args)==4 and args[1]=='types':
            scope,days=args[2],int(args[3])
            monitoring.filters(scope,days)
            screen(data, f'mon:feed:{scope}:{days}:all:0')
            return self._reply(client,actor,'Тип событий · выберите фильтр',[
                [(title,f'mon:feed:{scope}:{days}:{category}:0')] for category,title in monitoring.CATEGORIES.items()])
        if len(args)==5 and args[1]=='export':
            return self.export(client,actor,args[2],int(args[3]),args[4])
        if len(args)==4 and args[1]=='metrics':
            return self.metrics(client,actor,int(args[2]),int(args[3]))
        if len(args)==4 and args[1]=='charts':
            return self.charts(client,actor,args[2],int(args[3]))
        if len(args)==5 and args[1]=='plot':
            return self.plot(client,actor,args[2],int(args[3]),args[4])
        raise ValueError('Неизвестный экран мониторинга')

    def home(self, client, actor):
        self._authorize(actor)
        screen('mon:home','menu:admin')
        current = monitoring.server_snapshot(self.store)
        connected = bool(current['active'] and current['runtime']['telegram_connected'])
        bot_status = 'запускается' if current['active'] and current['runtime']['state']=='starting' else 'активен' if current['active'] else 'нет свежего сигнала активности'
        errors = sum(row['n'] for row in monitoring.event_summary(self.store,days=1) if row['level']=='error')
        text = ('Мониторинг и история\n'
                f"Бот: {bot_status}. "
                f"Telegram: {'подключён' if connected else 'связь не подтверждена свежим сигналом'}.\n"
                f"Очередь: {current['queues'].get('pending',0)} ожидают, {current['queues'].get('running',0)} выполняются.\n"
                f'Ошибок за 24 часа: {errors}.\n\n'
                f"Подробный журнал действий ведётся с {at(self.store.get_setting('monitoring_started_at'))} (Москва). "
                'Ранее сохранённые задания, отправки, права и операции также доступны. '
                'Архивные строки показывают последнее сохранённое состояние.\n'
                'Выберите пользователя, сервер или общий журнал.')
        text += '\nПодробности запросов и отправленных файлов сохраняются с '+at(self.store.get_setting('monitoring_details_started_at'))+'.'
        return self._reply(client,actor,text,[
            [('Сервер · состояние','mon:server')], [('По пользователям','mon:users:0')],
            [('Графики истории','mon:charts:all:7')],
            [('Общий журнал','mon:feed:all:7:all:0')], [('Ошибки и предупреждения','mon:feed:all:7:errors:0')]])

    def server(self, client, actor):
        self._authorize(actor)
        screen('mon:server','mon:home')
        snapshot = monitoring.server_snapshot(self.store)
        runtime,metrics = snapshot['runtime'],snapshot['metrics']
        lines = ['Сервер · состояние на '+at(utcnow())+' (Москва)']
        if runtime:
            end = runtime['finished_at'] or runtime['heartbeat_at']
            seconds = max(0,(datetime.fromisoformat(end)-datetime.fromisoformat(runtime['started_at'])).total_seconds())
            lines += [f"Хост: {runtime['host']}; процесс: {runtime['pid']}.",
                      f"Запуск: {at(runtime['started_at'])}; наблюдаемое время работы: {int(seconds//3600)} ч {int(seconds%3600//60)} мин.",
                      f"Последний сигнал: {at(runtime['heartbeat_at'])}; {int(snapshot['heartbeat_age'])} с назад.",
                      'Сервис: '+('активен' if snapshot['active'] else 'нет свежего сигнала; проверьте запуск и сон компьютера')+'.',
                      'Telegram по последнему сигналу: '+('подключён' if runtime['telegram_connected'] and snapshot['active'] else 'соединение не подтверждено')+'.']
        else:
            lines.append('Служба ещё не запускалась с мониторингом.')
        def number(key,unit):
            value = metrics.get(key)
            return f'{value:,.1f} {unit}' if isinstance(value,(int,float)) else 'нет измерения'
        lines += ['', 'Измерения: '+at(snapshot['sample_at']),
                  'Память процесса: '+number('process_rss_mb','МБ')+'.',
                  'CPU процесса: '+number('process_cpu_percent','% одного ядра')+'.',
                  'Доступно памяти сервера: '+number('system_memory_available_mb','МБ')+'.',
                  'Свободно на диске базы: '+number('disk_free_gb','ГБ')+'.',
                  'База с WAL: '+number('database_mb','МБ')+'.', '',
                  'Задания в базе: '+', '.join(f'{STATES.get(state,state)} — {count}' for state,count in snapshot['queues'].items())+'.',
                  'Доставки в базе: '+', '.join(f'{STATES.get(state,state)} — {count}' for state,count in snapshot['deliveries'].items())+'.',
                  'Уведомления о правах: '+', '.join(f'{STATES.get(state,state)} — {count}' for state,count in snapshot['notices'].items())+'.',
                  'Включённые ряды источников: '+', '.join(f'{STATES.get(state,state)} — {count}' for state,count in snapshot['sources'].items())+'.',
                  ('\nСигнал активности обновляется каждые 10 с; ресурсы — раз в минуту. '
                   'Без внешнего наблюдателя бот не может сообщить о собственном полном отключении.')]
        if snapshot['oldest_pending']:
            lines.append('Самое раннее ожидание в очереди: '+at(snapshot['oldest_pending']['created_at'])+'.')
        for job in snapshot['running_jobs']:
            lines.append(f"Выполняется №{job['id']}: {NAMES.get(job['kind'],job['kind'])}; начало {at(job['started_at'])}.")
        return self._reply(client,actor,'\n'.join(lines),[
            [('Обновить состояние','mon:server')], [('История сервера','mon:feed:server:7:all:0')],
            [('Графики сервера','mon:charts:server:7')],
            [('История нагрузки','mon:metrics:7:0')], [('Очередь заданий','menu:jobs')]])

    def users(self, client, actor, page=0):
        self._authorize(actor)
        rows = monitoring.user_list(self.store,page)
        screen(f'mon:users:{page}','mon:home')
        from .user_management import label
        lines = [f'Мониторинг пользователей · страница {page+1}\nВыберите карточку. В списке есть и пользователи без доступа.']
        buttons = []
        for row in rows[:8]:
            title = label(self.store,row['user_id'])
            lines.append(f"\n{title}\nДоступ: {'есть' if row['enabled'] else 'закрыт'}; последнее событие: {at(row['last_event'])}.")
            buttons.append([(title[:110],f"mon:user:{row['user_id']}")])
        if not rows:
            lines.append('Пользователей пока нет.')
        if page:
            buttons.append([('‹ Предыдущие',f'mon:users:{page-1}')])
        if len(rows)>8:
            buttons.append([('Следующие ›',f'mon:users:{page+1}')])
        return self._reply(client,actor,'\n'.join(lines),buttons)

    def user(self, client, actor, target):
        self._authorize(actor)
        if type(target) is not int or target<=0:
            raise ValueError('Нужен положительный ID пользователя')
        from .user_management import label
        profile = self.store.one('SELECT * FROM user_profiles WHERE user_id=?',(target,))
        user = self.store.one('SELECT role,enabled FROM users WHERE user_id=?',(target,))
        if not profile and not user:
            raise ValueError('Этот пользователь ещё не известен боту')
        scope = f'u{target}'
        screen(f'mon:user:{target}','mon:users:0')
        summary = monitoring.event_summary(self.store,scope)
        lines = ['История пользователя\n'+label(self.store,target),
                 'Доступ: '+('есть' if user and user['enabled'] else 'закрыт')+'.',
                 'Последнее обращение: '+at(profile['last_seen'] if profile else None)+'.',
                 '\nСобытия за 7 дней:']
        counts = {}
        for row in summary:
            counts[row['category']] = counts.get(row['category'],0)+row['n']
        lines.extend(f"• {monitoring.CATEGORIES.get(key,key)}: {count}" for key,count in counts.items())
        if not summary:
            lines.append('Событий за этот период нет.')
        buttons = [[('Вся история',f'mon:feed:{scope}:0:all:0')],
                   [('Графики пользователя',f'mon:charts:{scope}:7')],
                   [('Действия',f'mon:feed:{scope}:7:interaction:0'),('Ошибки',f'mon:feed:{scope}:7:errors:0')],
                   [('Изменения прав',f'mon:feed:{scope}:0:access:0')],
                   [('Задания и результаты',f'mon:feed:{scope}:7:job:0')],
                   [('Доставка',f'mon:feed:{scope}:7:delivery:0')]]
        if user:
            buttons.append([('Управление доступом',f'ua:open:{target}')])
        return self._reply(client,actor,'\n'.join(lines),buttons)

    def _anchor(self, actor, scope, days, category, *, reset=False):
        key = f'monitor_view:{actor}'
        value = self.store.get_setting(key,{})
        selection = [scope,days,category]
        if reset or value.get('selection') != selection or not value.get('until'):
            value = {'selection':selection,'until':utcnow()}
            self.store.set_setting(key,value)
        return value['until']

    def feed(self, client, actor, scope='all', days=7, category='all', page=0):
        self._authorize(actor)
        monitoring.filters(scope,days,category)
        until = self._anchor(actor,scope,days,category,reset=page==0)
        rows,total = monitoring.events(self.store,scope,days,category,page=page,until=until)
        if page and not rows:
            return self.feed(client,actor,scope,days,category,max(0,(total-1)//8))
        route = f'mon:feed:{scope}:{days}:{category}:{page}'
        back = f'mon:user:{scope[1:]}' if scope.startswith('u') else 'mon:server' if scope=='server' else 'mon:home'
        screen(route,back)
        from .user_management import label
        who = label(self.store,int(scope[1:])) if scope.startswith('u') else 'Сервер' if scope=='server' else 'Все пользователи и сервер'
        lines = [f'Журнал · {who}', f"{monitoring.CATEGORIES[category]} · {'вся история' if days==0 else str(days)+' дн.'}",
                 f'Страница {page+1}/{max(1,(total+7)//8)}; событий: {total}. Время по Москве.']
        for row in rows:
            mark = '🔴' if row['level']=='error' else '⚠' if row['level']=='warning' else '•'
            lines.append(f"\n{mark} №{row['id']} · {at(row['occurred_at'])} · {action_name(row)}\n{STATES.get(row['outcome'],row['outcome'])}")
            if row['actor_id']:
                lines.append('Кто: '+label(self.store,row['actor_id']))
            if row['subject_id'] and row['subject_id']!=row['actor_id']:
                lines.append('Пользователь: '+label(self.store,row['subject_id']))
            if row['entity_type'] in {'jobs','deliveries'}:
                lines.append(('Задание' if row['entity_type']=='jobs' else 'Доставка')+' №'+row['entity_id'])
            if row['entity_type']=='market_series':
                source = self.store.one('SELECT title FROM market_series WHERE id=?',(row['entity_id'],))
                if source:
                    lines.append('Ряд: '+source['title'])
            if row['duration_ms'] is not None:
                lines.append(f"Время выполнения: {row['duration_ms']/1000:.2f} с")
            details = json.loads(row['details'])
            request = details.get('request',{})
            if request.get('text'):
                lines.append('Запрос: '+request['text'].replace('\n',' ')[:180])
            if request.get('button_text'):
                lines.append('Кнопка: '+request['button_text'][:180])
            if request.get('kind') not in {None,'text','button'}:
                lines.append('Отправлено: '+MEDIA.get(request['kind'],request['kind'])+(' · '+request['file_name'] if request.get('file_name') else ''))
            if details.get('error_type'):
                lines.append('Тип ошибки: '+details['error_type'])
            if 'old_username' in details:
                old = '@'+details['old_username'] if details['old_username'] else 'без @username'
                new = '@'+details['username'] if details['username'] else 'без @username'
                lines.append(f'Имя в Telegram: {old} → {new}')
            if 'old_times' in details and details['old_times'] != details['times']:
                lines.append('Время доставки: '+details['old_times']+' → '+details['times'])
            if row['origin']=='history':
                lines.append('Архив: последнее сохранённое состояние.')
        if not rows:
            lines.append('\nСобытий по выбранному фильтру нет.')
        buttons = [[(checked(title,days==value),f'mon:feed:{scope}:{value}:{category}:0')
                    for value,title in [(1,'24 ч'),(7,'7 дн.'),(30,'30 дн.'),(0,'Всё')]],
                   [('Тип: '+monitoring.CATEGORIES[category],f'mon:types:{scope}:{days}')]]
        nav = []
        if page:
            nav.append(('‹ Предыдущие',f'mon:feed:{scope}:{days}:{category}:{page-1}'))
        if (page+1)*8<total:
            nav.append(('Следующие ›',f'mon:feed:{scope}:{days}:{category}:{page+1}'))
        if nav:
            buttons.append(nav)
        if rows:
            buttons.extend([[(f"Подробнее №{row['id']}",f"mon:event:{row['id']}") for row in rows[index:index+2]] for index in range(0,len(rows),2)])
        anchor = self.store.get_setting(f'monitor_view:{actor}',{})
        anchor['page'] = page
        self.store.set_setting(f'monitor_view:{actor}',anchor)
        buttons += [[('Обновить журнал',f'mon:feed:{scope}:{days}:{category}:0')],
                    [('CSV · до 10 000 событий',f'mon:export:{scope}:{days}:{category}')]]
        return self._reply(client,actor,'\n'.join(lines),buttons)

    def _parameters(self, payload, prefix=''):
        lines = []
        for key,value in selected_parameters(payload).items():
            title = prefix+PARAMETER_LABELS.get(key,key)
            if isinstance(value,dict):
                lines.extend(self._parameters(value,title+' · '))
            else:
                text = json.dumps(value,ensure_ascii=False) if isinstance(value,list) else str(value)
                lines.append(f'• {title}: {text}')
        return lines

    def event(self, client, actor, identity):
        self._authorize(actor)
        if type(identity) is not int or identity<=0:
            raise ValueError('Неверное событие')
        row = self.store.one('SELECT * FROM monitor_events WHERE id=?',(identity,))
        if not row:
            raise ValueError('Событие не найдено')
        view = self.store.get_setting(f'monitor_view:{actor}',{})
        selection = view.get('selection',['all',7,'all'])
        back = f'mon:feed:{selection[0]}:{selection[1]}:{selection[2]}:{view.get("page",0)}'
        screen(f'mon:event:{identity}',back)
        from .user_management import label
        lines = [f"Событие №{identity} · {at(row['occurred_at'])} (Москва)",action_name(row),
                 'Обработка действия: '+STATES.get(row['outcome'],row['outcome'])+'.']
        if row['actor_id']:
            lines.append('Кто: '+label(self.store,row['actor_id']))
        if row['subject_id'] and row['subject_id']!=row['actor_id']:
            lines.append('Пользователь: '+label(self.store,row['subject_id']))
        details = json.loads(row['details'])
        request = details.get('request')
        if request:
            lines += ['', 'Получено от пользователя: '+MEDIA.get(request['kind'],request['kind'])]
            if request.get('form_prompt'):
                lines += ['В какой форме:',request['form_prompt']]
            for key,title in [('text','Текст запроса / введённое значение'),('button_text','Нажатая кнопка'),
                              ('file_name','Имя файла'),('mime_type','Формат'),('caption','Подпись')]:
                if request.get(key):
                    lines += [title+':',request[key]]
            if 'file_size' in request:
                lines.append(f"Размер: {request['file_size']:,} байт")
            if request.get('width') is not None and request.get('height') is not None:
                lines.append(f"Изображение: {request['width']} × {request['height']}")
            if 'duration' in request:
                lines.append(f"Длительность: {request['duration']} с")
            if request.get('message_id'):
                lines.append(f"Входящее сообщение №{request['message_id']}")
        elif row['category']=='interaction':
            lines.append('\nСодержимое запроса в этой старой записи не сохранялось.')
        response = details.get('response',{})
        if response.get('text'):
            lines += ['', 'Ответ бота:',response['text']]
        if details.get('error_type'):
            lines.append('Тип ошибки: '+details['error_type'])
        job_ids = details.get('job_ids',[])
        if row['entity_type']=='jobs':
            job_ids = [int(row['entity_id'])]
        for job_id in job_ids[:5]:
            job = self.store.one('SELECT * FROM jobs WHERE id=?',(job_id,))
            if not job:
                continue
            lines += ['', f"Связанное задание №{job['id']}: {NAMES.get(job['kind'],job['kind'])}",
                      'Текущий результат: '+STATES.get(job['state'],job['state']),
                      'Поставлено: '+at(job['created_at'])+'; завершено: '+at(job['finished_at'])]
            try:
                lines.extend(self._parameters(json.loads(job['payload'])))
            except json.JSONDecodeError:
                pass
        if row['entity_type']=='deliveries':
            delivery = self.store.one('SELECT state,payload FROM deliveries WHERE id=?',(row['entity_id'],))
            if delivery:
                lines += ['', 'Отправка ботом · текущий итог: '+STATES.get(delivery['state'],delivery['state'])]
                payload = json.loads(delivery['payload'])
                lines.append('Тип: '+MEDIA.get(payload.get('kind'),payload.get('kind','сообщение')))
                content = payload.get('text') or payload.get('caption')
                if content:
                    lines.append(redact_text(content))
        if row['origin']=='history':
            lines.append('\nАрхивная запись: последнее сохранённое состояние на указанную дату.')
        return self._reply(client,actor,'\n'.join(lines),[[('Обновить результат',f'mon:event:{identity}')]])

    def export(self, client, actor, scope, days, category):
        self._authorize(actor)
        until = self._anchor(actor,scope,days,category)
        rows,total = monitoring.events(self.store,scope,days,category,limit=10000,until=until)
        text = io.StringIO()
        writer = csv.writer(text,delimiter=';')
        writer.writerow(['Время (Москва)','Тип','Действие','Результат','Уровень','Кто (ID)','Пользователь (ID)',
                         'Объект','ID объекта','Длительность, мс','Происхождение','Сведения',
                         'Текст запроса','Нажатая кнопка','Имя файла','Тип файла','Размер, байт','Подпись',
                         'Контекст ввода','Ответ бота','Связанные задания','Параметры задания'])
        def cell(value):
            value = '' if value is None else str(value)
            return "'"+value if value.startswith(('=','+','-','@')) else value
        job_parameters = {}
        for row in rows:
            details = json.loads(row['details'])
            request,response = details.get('request',{}),details.get('response',{})
            parameters = ''
            if row['entity_type']=='jobs':
                identity = row['entity_id']
                if identity not in job_parameters:
                    job = self.store.one('SELECT payload FROM jobs WHERE id=?',(identity,))
                    job_parameters[identity] = '\n'.join(self._parameters(json.loads(job['payload']))) if job else ''
                parameters = job_parameters[identity]
            writer.writerow([cell(value) for value in [at(row['occurred_at']),monitoring.CATEGORIES[row['category']],
                action_name(row),STATES.get(row['outcome'],row['outcome']),row['level'],row['actor_id'],row['subject_id'],
                row['entity_type'],row['entity_id'],row['duration_ms'],row['origin'],row['details'],
                request.get('text'),request.get('button_text'),request.get('file_name'),request.get('mime_type'),
                request.get('file_size'),request.get('caption'),request.get('form_prompt'),response.get('text'),
                ', '.join(str(value) for value in details.get('job_ids',[])),parameters]])
        output = io.BytesIO(text.getvalue().encode('utf-8-sig'))
        output.name = 'monitoring-history.csv'
        self._authorize(actor)
        client.send_document(actor,output,caption=f'История: {len(rows)} из {total} событий по выбранному фильтру. '
                             'При превышении лимита выгружены последние 10 000.')

    def charts(self, client, actor, scope='all', days=7):
        self._authorize(actor)
        monitoring.filters(scope,days)
        back = f'mon:user:{scope[1:]}' if scope.startswith('u') else 'mon:server' if scope=='server' else 'mon:home'
        screen(f'mon:charts:{scope}:{days}',back)
        from .user_management import label
        title = label(self.store,int(scope[1:])) if scope.startswith('u') else 'Сервер' if scope=='server' else 'Вся система'
        rows = [[(checked(title,days==value),f'mon:charts:{scope}:{value}') for value,title in
                 [(1,'24 ч'),(7,'7 дн.'),(30,'30 дн.'),(0,'Всё')]],
                [('Активность, задания и ошибки',f'mon:plot:{scope}:{days}:activity')]]
        if not scope.startswith('u'):
            rows.append([('CPU, память, диск и база',f'mon:plot:{scope}:{days}:resources')])
        return self._reply(client,actor,'Графики истории · '+title+'\n'
            'Выберите период и график. Каждый запрос строится по сохранённым данным.\n\n'
            'События: действия и активные пользователи, завершения заданий, ошибки, время ответа. '
            'Архивные снимки не выдаются за наблюдавшуюся активность.\n'
            'Ресурсы: CPU и память процесса бота, свободный диск и размер базы; измерения хранятся 90 дней. '
            'Пустой участок не подтверждает работу или простой сервера.',rows)

    def plot(self, client, actor, scope, days, kind):
        self._authorize(actor)
        monitoring.filters(scope,days)
        if kind not in {'activity','resources'} or kind=='resources' and scope.startswith('u'):
            raise ValueError('Неверный график мониторинга')
        from .monitor_charts import (
            activity_data,
            render_activity,
            render_resources,
            resource_data,
        )
        from .user_management import label
        title = label(self.store,int(scope[1:])) if scope.startswith('u') else 'Сервер' if scope=='server' else 'Вся система'
        period = '24 часа' if days==1 else f'{days} дней' if days else 'вся сохранённая история'
        if kind=='activity':
            data = activity_data(self.store,scope,days)
            output = render_activity(data,title+' · '+period)
            caption = ('История · '+title+'. Только события, записанные во время работы мониторинга. '
                       f"Архивных снимков исключено: {data['archive']}. "
                       'Задания — переходы в готово, ошибку или отмену; это не размер очереди. '
                       'Время ответа — обработка действия, без ожидания фонового задания. '
                       'Нет записей не означает, что бот был доступен.')
        else:
            data = resource_data(self.store,days)
            output = render_resources(data,period)
            caption = (f"Ресурсы сервера · {sum(row['samples'] for row in data['rows'])} измерений; "
                       f"агрегация {data['step']//60} мин. CPU относится к процессу бота, 100% — одно ядро. "
                       'Пунктир — запуск процесса (последние 20 за период). Разные запуски и пропуски не соединяются. '
                       'Ресурсы хранятся 90 дней.')
        self._authorize(actor)
        client.send_photo(actor,output,caption=caption)
        return self.charts(client,actor,scope,days)

    def metrics(self, client, actor, days=7, page=0):
        self._authorize(actor)
        if days not in {1,7,30} or not 0<=page<=100000:
            raise ValueError('Неверный период нагрузки')
        cutoff = (datetime.now(timezone.utc)-timedelta(days=days)).isoformat()
        rows = self.store.query('SELECT * FROM monitor_samples WHERE created_at>=? ORDER BY id DESC LIMIT 9 OFFSET ?', (cutoff,page*8))
        summary = self.store.one('''SELECT COUNT(*) n,MAX(json_extract(metrics,'$.process_rss_mb')) peak_memory,
            MIN(json_extract(metrics,'$.disk_free_gb')) least_disk FROM monitor_samples WHERE created_at>=?''',(cutoff,))
        screen(f'mon:metrics:{days}:{page}','mon:server')
        lines = [f"История нагрузки · {days} дн. · {summary['n']} измерений", 'Время по Москве. Измерения хранятся 90 дней.']
        if summary['peak_memory'] is not None:
            lines.append(f"Максимум памяти процесса: {summary['peak_memory']:.1f} МБ; минимум свободного диска: {summary['least_disk']:.1f} ГБ.")
        for row in rows[:8]:
            values = json.loads(row['metrics'])
            cpu,rss,disk = values.get('process_cpu_percent'),values.get('process_rss_mb'),values.get('disk_free_gb')
            parts = [at(row['created_at'])]
            if cpu is not None:
                parts.append(f'CPU {cpu:.1f}%')
            if rss is not None:
                parts.append(f'память {rss:.1f} МБ')
            if disk is not None:
                parts.append(f'диск свободен {disk:.1f} ГБ')
            lines.append('\n'+' · '.join(parts))
        if not rows:
            lines.append('\nИзмерений пока нет.')
        buttons = [[(checked(f'{value} дн.',days==value),f'mon:metrics:{value}:0') for value in (1,7,30)]]
        if page:
            buttons.append([('‹ Предыдущие',f'mon:metrics:{days}:{page-1}')])
        if len(rows)>8:
            buttons.append([('Следующие ›',f'mon:metrics:{days}:{page+1}')])
        buttons.append([('Показать график ресурсов',f'mon:plot:server:{days}:resources')])
        buttons.append([('Обновить','mon:metrics:'+str(days)+':0')])
        return self._reply(client,actor,'\n'.join(lines),buttons)
