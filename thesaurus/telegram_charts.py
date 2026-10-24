"""Button-based chart selection, personal periods and forecast display horizons."""
import hashlib
import json
from uuid import uuid4

from .charts import (
    bounds,
    chart_page,
    latest_date,
    selections,
    validate_options,
    validate_selection,
)
from .forecast_horizons import UNITS, from_options
from .forecast_horizons import label as duration_label
from .forecast_horizons import validate as validate_duration
from .store import utcnow
from .telegram_data_jobs import actor_scope, require_actor
from .telegram_navigation import checked, screen

LABELS = {'ruonia': 'RUONIA', 'market': 'Показатели рынка и ЦБ', 'wordstat': 'Wordstat'}
LABELS['trends'] = 'Google Trends'
PRESETS = {'month': 'Месяц', 'quarter': '3 месяца', 'half': '6 месяцев', 'year': 'Год', 'all': 'Вся история'}


def keyboard(rows):
    from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup
    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in row] for row in rows])


class ChartControls:
    def __init__(self, store, reply):
        self.store, self.reply = store, reply

    def options(self, actor, section):
        return validate_options(self.store.get_setting(f'chart:{actor}:{section}', {'period': 'quarter', 'horizon': 4, 'identity': None}))

    def pending(self, actor, value):
        self.store.execute('INSERT INTO bot_sessions(user_id,last_seen,pending) VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET pending=excluded.pending,last_seen=excluded.last_seen', (actor, utcnow(), value))

    def input(self, client, actor, text):
        state = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        if not state or not (state['pending'] or '').startswith(('chart:', 'charthorizon:', 'chartsearch:')):
            return False
        require_actor(self.store, actor)
        if text == '/cancel':
            self.pending(actor, None)
            self.reply(client, actor, 'Ввод периода отменён.')
            return True
        if text.startswith('/'):
            return False
        if state['pending'].startswith('chartsearch:'):
            section = state['pending'].split(':')[1]
            self.store.set_setting(f'chart_search:{actor}:{section}', text.strip()[:100])
            self.store.set_setting(f'chart_filter:{actor}:{section}', 'all')
            self.pending(actor, None)
            self.menu(client, actor, section, view='indicator')
            return True
        if state['pending'].startswith('charthorizon:'):
            _, section, unit = state['pending'].split(':')
            if not text.strip().isascii() or not text.strip().isdigit() or int(text.strip()) <= 0:
                raise ValueError('Введите положительное целое число. Отмена: /cancel')
            options = self.options(actor, section)
            options['forecast'] = validate_duration({'value': int(text.strip()), 'unit': unit})
            self.store.set_setting(f'chart:{actor}:{section}', options)
            self.pending(actor, None)
            self.menu(client, actor, section)
            return True
        args = text.split()
        if len(args) != 2:
            raise ValueError('Введите две даты: YYYY-MM-DD YYYY-MM-DD. Отмена: /cancel')
        section = state['pending'].split(':')[1]
        options = self.options(actor, section)
        options.update(period='custom', date_from=args[0], date_to=args[1])
        options = validate_options(options)
        self.store.set_setting(f'chart:{actor}:{section}', options)
        self.pending(actor, None)
        self.menu(client, actor, section)
        return True

    def dispatch(self, client, actor, argument=''):
        if argument.strip() and argument.strip() not in LABELS:
            raise ValueError('/chart [ruonia|market|trends|wordstat] — выбор периода и показателя кнопками')
        self.menu(client, actor, argument.strip() or None)

    def menu(self, client, actor, section=None, page=0, view='summary'):
        require_actor(self.store, actor)
        if section is None:
            screen('ch:menu')
            self.reply(client, actor, 'Выберите данные для графика:', keyboard([
                [(label, f'ch:menu:{key}')] for key, label in LABELS.items()
                if not (key in {'trends', 'wordstat'} and self.store.get_setting('search_provider') == 'disabled')
                and (key != 'wordstat' or self.store.get_setting('search_provider') != 'google_trends')]))
            return
        choices = selections(self.store, actor, section)
        options = self.options(actor, section)
        unavailable = options.get('identity') is not None and options['identity'] not in {key for key, _ in choices}
        chosen = 'Выберите доступный показатель' if unavailable else dict(choices).get(options.get('identity'), 'Все мои показатели (по 12 на странице)')
        if options.get('selection_mode') == 'multiple' and not options.get('series_ids'):
            chosen, unavailable = 'Ничего не выбрано', True
        if options.get('series_ids'):
            chosen = ', '.join(dict(choices).get(key, 'Недоступен') for key in options['series_ids'])
            unavailable = not set(options['series_ids']).issubset(dict(choices))
        period = (f"{options['date_from']} — {options['date_to']}" if options['period'] == 'custom' else PRESETS[options['period']])
        frequency = 'daily' if section == 'ruonia' else 'weekly'
        administrative = False
        if section in {'market', 'trends'} and options.get('identity'):
            import json
            row = self.store.one('SELECT metadata FROM market_series WHERE id=?', (options['identity'],))
            frequency = json.loads(row['metadata'])['frequency'] if row else frequency
            administrative = bool(row and json.loads(row['metadata']).get('administrative'))
        duration = from_options(options, frequency)
        forecast = duration_label(duration)
        forecast_label = 'Условный сценарий' if administrative else 'Прогноз'
        summary = f"Графики · {LABELS[section]}\nПоказатель: {chosen}\nПериод: {period}\n{forecast_label}: {forecast}"
        if administrative:
            summary += '\n\nЗначение устанавливает регулятор. Линия «без изменений» показывает только условие сохранения последнего значения, а не будущее решение ЦБ.'
        screen(f'ch:menu:{section}', 'ch:menu')
        if view != 'summary':
            screen(f'ch:view:{section}:{view}', f'ch:menu:{section}')
        rows = []
        if view == 'summary':
            last = latest_date(self.store, actor, section, options.get('identity'), options.get('series_ids')) if not unavailable else None
            rows = [[('Показатель: ' + chosen[:36], f'ch:view:{section}:indicator')],
                    [('Период: ' + period, f'ch:view:{section}:period')],
                    [(forecast_label + ': ' + forecast, f'ch:view:{section}:horizon')]]
            if section == 'ruonia':
                rows += [[('RUONIA 1 + 3 + 6 месяцев на одном графике', 'ch:terms:ruonia')]]
            if options.get('series_ids'):
                rows += [[(checked('На одном графике', options.get('layout') == 'overlay'), f'ch:layout:{section}')],
                         [(checked('Диапазоны прогноза', options.get('intervals', False)), f'ch:intervals:{section}')]]
            if choices and not unavailable:
                if section != 'wordstat' or last:
                    rows += [[('Построить график', f'ch:build:{section}')], [('Скачать данные', f'ch:view:{section}:export')]]
                    rows += [[('Отправлять по расписанию', f'cs:{section}:menu')], [('Уведомления по показателю', f'ch:alert:{section}')]]
                if section == 'wordstat':
                    summary += ('\n\nПоследние сохранённые данные: ' + last if last else
                                '\n\nДля выбранных фраз недельная история ещё не загружена. Сначала получите данные из Яндекса, затем вернитесь сюда и постройте график.')
                    summary += '\nЗагрузка использует Wordstat API и дневной лимит обращений. Построение графика к API не обращается.'
                    rows += [[('Обновить данные из Яндекса' if last else 'Загрузить данные из Яндекса', 'request:wordstat')]]
            else:
                summary += ('\n\nОткройте «Показатель» → «Несколько показателей» и отметьте хотя бы один ряд для сравнения.'
                            if options.get('selection_mode') == 'multiple' and not options.get('series_ids') else
                            '\n\nВыбранный показатель недоступен или источники ещё не включены. Выберите доступный показатель; личный набор — в «Моих настройках». ')
                rows += [[('График RUONIA', 'ch:menu:ruonia')]] if section != 'ruonia' else []
                if section in {'market', 'trends'}:
                    rows += [[('Выбрать мои показатели', 'menu:indicators')]]
                    if self.store.authorize(actor, admin=True):
                        rows += [[('Выбрать источники', 'menu:sources')]]
            if section == 'ruonia' and not last:
                summary += '\n\nСохранённой истории пока нет. Сначала запросите данные RUONIA.'
                rows += [[('Загрузить RUONIA', 'request:ruonia')]]
            rows += [[('Как это работает', f'ch:view:{section}:info')]]
            if section == 'trends':
                from .google_trends import NOTE
                summary += '\n\n' + NOTE
                rows += [[('Данные Google Trends', 'gt:menu')]]
            if choices and not unavailable and (section != 'wordstat' or last):
                summary += '\n\nИзмените нужную настройку и нажмите «Построить график». Выбор сохраняется для вас.'
        elif view == 'period':
            summary = 'Период графика\n✓ — ваш текущий выбор. Готовые периоды отсчитываются от последней загруженной даты.'
            rows = [[(checked(PRESETS[key], options['period'] == key), f'ch:period:{section}:{key}')] for key in PRESETS]
            rows += [[(checked('Свои даты', options['period'] == 'custom'), f'ch:custom:{section}')]]
        elif view == 'horizon':
            summary = 'Срок прогноза\nМаксимум 24 месяца. Срок начинается от последнего факта и не меняет период истории. Длинные прогнозы могут быть только базовыми, без оценённого диапазона.'
            rows = [[(checked('Без прогноза', not duration['value']), f'ch:horizon:{section}:0')]]
            units = [('weeks', [1, 2, 4, 8, 12]), ('months', [1, 3, 6, 12, 24])]
            if frequency in {'monthly', 'quarterly', 'annual'}:
                unit = {'monthly': 'months', 'quarterly': 'quarters', 'annual': 'years'}[frequency]
                units = [(unit, {'months': [1, 3, 6, 12, 24], 'quarters': [1, 2, 4, 8], 'years': [1, 2]}[unit])]
            for unit, values in units:
                for i in values:
                    legacy = i <= 4 and ((unit == 'weeks' and frequency in {'daily', 'weekly'}) or (unit == 'months' and frequency == 'monthly'))
                    callback = f'ch:horizon:{section}:{i}' if legacy else f'ch:duration:{section}:{unit}:{i}'
                    rows += [[(checked(f'До {i} {UNITS[unit]}', duration == {'value': i, 'unit': unit}), callback)]]
                rows += [[('Свой срок · ' + UNITS[unit], f'ch:ownhorizon:{section}:{unit}')]]
        elif view == 'indicator':
            filter_name = self.store.get_setting(f'chart_filter:{actor}:{section}', 'all')
            if filter_name in {'favorites', 'recent'}:
                keys = self.store.get_setting(f'chart_{filter_name}:{actor}:{section}', [])
                lookup = dict(choices)
                choices = [(key, lookup[key]) for key in keys if key in lookup]
            search = self.store.get_setting(f'chart_search:{actor}:{section}', '').casefold()
            choices = [(key, title) for key, title in choices if not search or search in title.casefold() or search in key.casefold()]
            size = 8
            page = min(max(page, 0), max(0, (len(choices)-1)//size))
            summary = f'Показатель · {LABELS[section]}\nСтраница {page+1} из {max(1, (len(choices)+size-1)//size)}. ✓ — текущий выбор.'
            rows = [[(checked('Все мои показатели', options.get('identity') is None and not options.get('series_ids')), f'ch:select:{section}:all')],
                    [('Выбрать несколько (до 6)', f'ch:view:{section}:multiple')],
                    [('Поиск показателя', f'ch:search:{section}')]]
            rows += [[(checked('Все', filter_name == 'all'), f'ch:filter:{section}:all'),
                      (checked('Избранное', filter_name == 'favorites'), f'ch:filter:{section}:favorites'),
                      (checked('Недавние', filter_name == 'recent'), f'ch:filter:{section}:recent')]]
            if options.get('identity'):
                favorites = self.store.get_setting(f'chart_favorites:{actor}:{section}', [])
                rows += [[('Убрать из избранного' if options['identity'] in favorites else 'Добавить выбранный в избранное', f'ch:favorite:{section}')]]
            if search:
                summary += '\nПоиск: ' + search
                rows += [[('Сбросить поиск', f'ch:resetsearch:{section}')]]
            if not choices:
                summary += '\nПоказателей в этом списке пока нет. Выберите «Все» или измените поиск.'
            for identity, title in choices[page*size:(page+1)*size]:
                token = hashlib.sha256((section + ':' + identity).encode()).hexdigest()[:16]
                self.store.set_setting(f'chart_token:{actor}:{token}', {'section': section, 'identity': identity})
                rows += [[(checked(title[:47], identity == options.get('identity')), f'ch:select:{section}:{token}')]]
            navigation = []
            if page:
                navigation.append(('‹ Предыдущая', f'ch:page:{section}:{page-1}'))
            if (page+1)*size < len(choices):
                navigation.append(('Следующая ›', f'ch:page:{section}:{page+1}'))
            if navigation:
                rows.append(navigation)
        elif view == 'multiple':
            size = 8
            page = min(max(page, 0), max(0, (len(choices)-1)//size))
            summary = 'Сравнение на графике\nВыберите до 6 показателей. ✓ — выбран. Разные единицы будут показаны отдельно.'
            for identity, title in choices[page*size:(page+1)*size]:
                token = hashlib.sha256((section + ':' + identity).encode()).hexdigest()[:16]
                self.store.set_setting(f'chart_token:{actor}:{token}', {'section': section, 'identity': identity})
                rows.append([(checked(title[:47], identity in options.get('series_ids', [])), f'ch:multi:{section}:{token}')])
            if page:
                rows.append([('‹ Предыдущая', f'ch:mpage:{section}:{page-1}')])
            if (page+1)*size < len(choices):
                rows.append([('Следующая ›', f'ch:mpage:{section}:{page+1}')])
            rows.append([('Готово', f'ch:menu:{section}')])
        elif view == 'export':
            summary += '\n\nФайл содержит выбранные данные за указанный период. Формат:'
            rows = [[('CSV', f'ch:export:{section}:csv'), ('Excel', f'ch:export:{section}:xlsx')]]
        elif view == 'info':
            summary = 'Как устроены графики\n\nГрафик использует уже загруженную историю. Обновить её можно через «Отчёты».\nПериод меняет только отображение, история обучения прогноза не сокращается.\nВ историческом периоде текущий прогноз скрыт. Пропуски не заполняются, редкие наблюдения показаны точками.\nПроверенный прогноз и серый сценарий без достаточной проверки различаются. Для административных ставок линия «без изменений» — условие, а не предсказание решения ЦБ; вероятности и диапазона у неё нет.\nПри длинной истории добавляется увеличенный участок последних двух лет и будущих значений. Верхняя панель сохраняет выбранный период.\nВсе показатели выдаются страницами по 12: кнопки следующих графиков находятся под выпуском. Можно выбрать конкретный ряд.'
        else:
            raise ValueError('Неизвестный экран графика')
        # Also usable by isolated controls without the shared navigator.
        rows.append([('← Назад', 'ch:menu' if view == 'summary' else f'ch:menu:{section}')])
        if section == 'market' and view in {'summary', 'indicator'}:
            rows.insert(0, [('Выбрать данные из баз ЦБ', 'cb:menu')])
        self.reply(client, actor, summary, keyboard(rows))

    def callback(self, client, actor, data, event_id):
        require_actor(self.store, actor)
        args = data.split(':')
        if len(args) < 2 or args[0] != 'ch':
            raise ValueError('Неизвестная настройка графика')
        action = args[1]
        section = args[2] if len(args) > 2 else None
        if action == 'auto' and (len(args) == 3 or len(args) == 4 and args[3] in {'on', 'off'}):
            from .chart_batches import toggle
            result = toggle(self.store, actor, section, enabled=args[3] == 'on' if len(args) == 4 else None)
            rows = [[(button['text'], button['callback_data']) for button in row] for row in result['buttons']]
            screen('ch:result', 'ch:menu')
            self.reply(client, actor, result['text'], keyboard(rows))
            return
        if action == 'batch' and len(args) == 4 and args[3].isascii() and args[3].isdigit() and len(args[3]) <= 5:
            self.queue_page(client, actor, section, chart_page(int(args[3])), event_id)
            return
        if action == 'menu':
            if len(args) > 3:
                raise ValueError('Неверная команда')
            self.menu(client, actor, section)
            return
        selections(self.store, actor, section)
        if action == 'alert' and len(args) == 3:
            options = self.options(actor, section)
            identity = options.get('identity')
            if identity is None:
                screen(f'ch:alert:{section}', f'ch:menu:{section}')
                self.reply(client, actor, 'Для уведомления выберите один показатель, затем нажмите «Уведомления по показателю».',
                           keyboard([[('Выбрать показатель', f'ch:view:{section}:indicator')]]))
                return
            if section == 'ruonia':
                from .market import sync_ruonia
                from .market_sources import metric_id
                sync_ruonia(self.store)
                identity = metric_id('local.ruonia', identity)
            elif section == 'wordstat':
                from .market_sources import metric_id
                identity = metric_id('wordstat.wordstat.225', identity)
            from .market_alerts import add_rule
            rule = self.store.one('SELECT id FROM market_rules WHERE user_id=? AND series_id=? ORDER BY id LIMIT 1', (actor, identity))
            identifier = rule['id'] if rule else add_rule(self.store, actor, identity, 'deviation')
            screen(f'ch:alert:{section}', f'ch:menu:{section}')
            self.reply(client, actor, 'Уведомление для выбранного показателя\nНастройте условие, порог и доставку. Новое правило выключено.',
                       keyboard([[('Настроить уведомление', f'ui:rule:{identifier}')]]))
            return
        if action == 'alerts' and len(args) == 3:
            from .telegram_market import MarketControls
            MarketControls(self.store, self.reply).alerts(client, actor, '')
            return
        if action == 'view' and len(args) == 4:
            if args[3] == 'multiple':
                options = self.options(actor, section)
                if options.get('selection_mode') != 'multiple':
                    options.update(series_ids=[options['identity']] if options.get('identity') else [],
                                   selection_mode='multiple', identity=None, layout='overlay', intervals=False)
                    self.store.set_setting(f'chart:{actor}:{section}', validate_options(options))
            self.menu(client, actor, section, view=args[3])
            return
        options = self.options(actor, section)
        if action == 'filter' and len(args) == 4 and args[3] in {'all', 'favorites', 'recent'}:
            self.store.set_setting(f'chart_filter:{actor}:{section}', args[3])
            self.menu(client, actor, section, view='indicator')
            return
        if action == 'favorite' and len(args) == 3 and options.get('identity'):
            choices = dict(validate_selection(self.store, actor, section, options['identity']))
            favorites = [key for key in self.store.get_setting(f'chart_favorites:{actor}:{section}', []) if key in choices]
            if options['identity'] in favorites:
                favorites.remove(options['identity'])
            else:
                favorites.append(options['identity'])
            self.store.set_setting(f'chart_favorites:{actor}:{section}', favorites)
            self.menu(client, actor, section, view='indicator')
            return
        if action == 'duration' and len(args) == 5 and args[4].isascii() and args[4].isdigit():
            options['forecast'] = validate_duration({'unit': args[3], 'value': int(args[4])})
            self.store.set_setting(f'chart:{actor}:{section}', validate_options(options))
            self.menu(client, actor, section)
            return
        if action == 'ownhorizon' and len(args) == 4 and args[3] in UNITS:
            self.pending(actor, f'charthorizon:{section}:{args[3]}')
            screen(f'ch:ownhorizon:{section}:{args[3]}', f'ch:view:{section}:horizon')
            self.reply(client, actor, f'Введите количество {UNITS[args[3]]}: положительное целое число, максимум 24 месяца. /cancel — отмена.')
            return
        if action == 'mpage' and len(args) == 4 and args[3].isdigit():
            self.menu(client, actor, section, int(args[3]), view='multiple')
            return
        if action == 'search' and len(args) == 3:
            self.pending(actor, f'chartsearch:{section}')
            screen(data, f'ch:view:{section}:indicator')
            self.reply(client, actor, 'Введите часть названия показателя. /cancel — отмена.')
            return
        if action == 'resetsearch' and len(args) == 3:
            self.store.set_setting(f'chart_search:{actor}:{section}', '')
            self.menu(client, actor, section, view='indicator')
            return
        value = args[3] if len(args) == 4 else None
        if action == 'page' and value and value.isdigit():
            self.menu(client, actor, section, int(value), view='indicator')
            return
        if action == 'custom' and len(args) == 3:
            screen(f'ch:input:{section}', f'ch:view:{section}:period')
            self.pending(actor, f'chart:{section}')
            self.reply(client, actor, 'Введите начало и конец периода, например: 2025-01-01 2026-10-01. Отмена: /cancel')
            return
        if action == 'period':
            options.update(period=value, date_from=None, date_to=None)
        elif action == 'horizon' and value and value.isdigit():
            options['horizon'] = int(value)
            options.pop('forecast', None)
        elif action == 'terms' and section == 'ruonia' and len(args) == 3:
            validate_selection(self.store, actor, section, None)
            if not {'1 мес', '3 мес', '6 мес'}.issubset(dict(selections(self.store, actor, section))):
                raise ValueError('Включите срочную RUONIA 1/3/6 месяцев в личных показателях RUONIA')
            options.update(identity=None, series_ids=['1 мес', '3 мес', '6 мес'], selection_mode='multiple', layout='overlay', intervals=False)
        elif action in {'layout', 'intervals'} and len(args) == 3:
            if action == 'layout':
                options['layout'] = 'separate' if options.get('layout') == 'overlay' else 'overlay'
            else:
                options['intervals'] = not options.get('intervals', False)
        elif action == 'multi' and value:
            token = self.store.get_setting(f'chart_token:{actor}:{value}')
            if not token or token['section'] != section:
                raise PermissionError('Кнопка недоступна')
            validate_selection(self.store, actor, section, token['identity'])
            selected = options.get('series_ids', []).copy()
            if token['identity'] in selected:
                selected.remove(token['identity'])
            else:
                selected.append(token['identity'])
            if len(selected) > 6:
                raise ValueError('Можно выбрать максимум 6 показателей')
            options.update(identity=None, layout='overlay', intervals=False, selection_mode='multiple', series_ids=selected)
            self.store.set_setting(f'chart:{actor}:{section}', validate_options(options))
            position = [key for key, _ in selections(self.store, actor, section)].index(token['identity'])
            self.menu(client, actor, section, page=position//8, view='multiple')
            return
        elif action == 'select' and value:
            token = self.store.get_setting(f'chart_token:{actor}:{value}') if value != 'all' else None
            if value != 'all' and (not token or token['section'] != section):
                raise PermissionError('Кнопка недоступна. Откройте /chart заново.')
            identity = token['identity'] if token else None
            validate_selection(self.store, actor, section, identity)
            options['identity'] = identity
            options.pop('series_ids', None)
            options.pop('selection_mode', None)
            if identity:
                recent = self.store.get_setting(f'chart_recent:{actor}:{section}', [])
                self.store.set_setting(f'chart_recent:{actor}:{section}', [identity] + [key for key in recent if key != identity][:7])
        elif action in {'build', 'export'}:
            if (action == 'build' and len(args) != 3) or (action == 'export' and value not in {'csv', 'xlsx'}):
                raise ValueError('Неверная команда графика')
            self.queue(client, actor, section, options, event_id, value if action == 'export' else None)
            return
        else:
            raise ValueError('Неверная настройка графика')
        self.store.set_setting(f'chart:{actor}:{section}', validate_options(options))
        self.menu(client, actor, section)

    def queue(self, client, actor, section, options, event_id, format=None):
        screen(f'ch:result:{section}', f'ch:menu:{section}')
        options = validate_options(options)
        if options.get('selection_mode') == 'multiple' and not options.get('series_ids'):
            raise ValueError('Выберите хотя бы один показатель')
        choices = validate_selection(self.store, actor, section, options.get('identity'))
        if options.get('series_ids'):
            if not set(options['series_ids']).issubset(dict(choices)):
                raise PermissionError('Набор больше не доступен')
            choices = [(key, title) for key, title in choices if key in options['series_ids']]
        payload = {'actor_id': actor, 'recipient': str(actor), 'section': section, 'report_id': uuid4().hex}
        kind = 'telegram_chart'
        if format:
            start, end = bounds(options, latest_date(self.store, actor, section, options.get('identity'), options.get('series_ids')))
            payload.update(format=format, date_from=start, date_to=end)
            payload['chart_options'] = options
            if section in {'market', 'trends'}:
                kind = 'market_export'
                payload['selection'] = [key for key, _ in choices if options.get('identity') is None or key == options['identity']]
            else:
                kind = 'telegram_export'
                payload.update(actor_scope(self.store, actor, section))
                if options.get('identity'):
                    payload['phrases'] = [options['identity']]
        else:
            payload.update(identity=options.get('identity'), options=options)
            payload['selection'] = [key for key, _ in choices if options.get('identity') is None or key == options['identity']]
        identifier = self.store.enqueue(kind, payload, f'chart:{actor}:{event_id}' if event_id is not None else None)
        self.reply(client, actor, f'Задание {identifier} в очереди. График или файл придёт отдельным сообщением. Для обновления данных используйте отчёт соответствующего раздела.')

    def queue_page(self, client, actor, report_id, page, event_id):
        require_actor(self.store, actor)
        page = chart_page(page)
        from .telegram_data_jobs import report_identifier
        report_identifier(report_id)
        report = self.store.one('SELECT quality FROM reports WHERE id=?', (report_id,))
        quality = json.loads(report['quality']) if report else {}
        request = quality.get('chart_request')
        if quality.get('actor_id') != actor or not request:
            raise PermissionError('Эти графики недоступны')
        batch_id = quality.get('chart_batch_id', report_id)
        if self.store.get_setting('chart_batch:' + batch_id, {}).get('enabled'):
            raise ValueError('Снимите галочку автоотправки перед ручным переходом')
        if page >= quality['chart_pages']:
            raise ValueError('Неверная страница графиков')
        choices = validate_selection(self.store, actor, request['section'], request.get('identity'))
        if not set(request['selection']).issubset(dict(choices)):
            raise PermissionError('Выбор показателей изменился. Постройте графики заново.')
        payload = {**request, 'actor_id': actor, 'recipient': str(actor), 'report_id': uuid4().hex, 'page': page, 'chart_batch_id': batch_id}
        identifier = self.store.enqueue('telegram_chart', payload, f'chart:{actor}:{event_id}' if event_id is not None else None)
        screen(f"ch:result:{request['section']}", f"ch:menu:{request['section']}")
        self.reply(client, actor, f'Страница {page+1}/{quality["chart_pages"]}: задание {identifier} в очереди. Используется сохранённая история.')
