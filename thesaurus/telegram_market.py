"""Private market controls. Collection and model work run only in the worker."""
import json
import math
import re
from uuid import uuid4

from . import market, market_alerts
from .market_store import change_access
from .store import dumps, utcnow
from .telegram_admin import _keyboard, positive_id
from .telegram_data_jobs import require_actor
from .telegram_navigation import checked, screen

COMMANDS = {'market', 'sources', 'indicators', 'correlations'}
HELP = '\n/market [all|macro|investment|liquidity] — обзор рынка\n/sources — каталог и включение источников (администратор)\n/indicators — личный выбор\n/correlations — пары и лаги\n/alerts market — личные рыночные правила\n/forecasts market; /backtest market; /export market csv|xlsx'


class MarketControls:
    def __init__(self, store, reply):
        self.store, self.reply = store, reply
        market.initialize(store)

    def pending(self, user_id, action):
        self.store.execute("INSERT INTO bot_sessions(user_id,last_seen,pending) VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET pending=excluded.pending", (user_id, utcnow(), action))

    def input(self, client, user_id, text):
        require_actor(self.store, user_id)
        state = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (user_id,))
        if not state or not state['pending']:
            return False
        action = state['pending']
        if text == '/cancel':
            self.pending(user_id, None)
            self.reply(client, user_id, 'Ввод отменён.')
            return True
        if text.startswith('/'):
            return False
        if action == 'add_user':
            require_actor(self.store, user_id, admin=True)
            from .user_management import (
                label,
                resolve_profile,
                resolve_user_reference,
                send_notices,
            )
            target = resolve_user_reference(self.store, client, text)
            change_access(self.store, user_id, target, True)
            if not text.strip().startswith('@'):
                resolve_profile(self.store, client, target)
            send_notices(self.store, client, target)
            self.pending(user_id, None)
            self.reply(client, user_id, 'Доступ разрешён: '+label(self.store, target)+'.\n'
                       'Подключите нужные проекты и разделы. Пользователю нужно открыть бот и нажать /start.',
                       _keyboard([[('Выбрать проекты и разделы', f'ua:open:{target}')]]))
        elif action.startswith('sourceunit:'):
            require_actor(self.store, user_id, admin=True)
            identity = action.split(':', 1)[1]
            row = self.store.one('SELECT metadata FROM market_series WHERE id=?', (identity,))
            unit = text.strip()
            if not row or not unit or len(unit) > 80:
                raise ValueError('Укажите единицы из описания источника, до 80 символов')
            metadata = json.loads(row['metadata'])
            metadata['unit'] = unit
            self.store.set_setting('market_unit:' + identity, unit)
            self.store.execute('UPDATE market_series SET metadata=? WHERE id=?', (dumps(metadata), identity))
            self.pending(user_id, None)
            self.sources(client, user_id, 'show ' + identity)
        elif action.startswith('command:'):
            command = action.split(':', 1)[1]
            self.pending(user_id, None)
            try:
                self.dispatch(client, user_id, command, text.strip())
            except (ValueError, PermissionError):
                self.pending(user_id, action)
                raise
        else:
            raise ValueError('Неизвестный режим ввода; /cancel')
        return True

    def dispatch(self, client, user_id, command, argument='', message_id=None):
        require_actor(self.store, user_id)
        from .feature_access import require as require_feature
        feature = {'market': 'reports', 'indicators': 'cbr', 'correlations': 'correlations',
                   'alerts': 'alerts', 'forecasts': 'backtests', 'backtest': 'backtests', 'export': 'exports'}.get(command)
        if feature:
            require_feature(self.store, user_id, feature)
        if command != 'sources':
            from .subscription_access import require
            require(self.store, user_id, 'market')
        if command == 'market' and argument.startswith('scenario '):
            args = argument.split()
            if len(args) != 3:
                raise ValueError('/market scenario ID ИЗМЕНЕНИЕ_В_ПРОЦЕНТНЫХ_ПУНКТАХ')
            row = next((r for r in market.selected(self.store, user_id) if r['id'] == args[1]), None)
            delta = float(args[2])
            if not row or not json.loads(row['metadata']).get('administrative') or not math.isfinite(delta):
                raise ValueError('Нужна административная ставка из вашего выбора и конечное изменение')
            scenarios = self.store.get_setting(f'market_scenarios:{user_id}', {})
            scenarios[args[1]] = delta
            self.store.set_setting(f'market_scenarios:{user_id}', scenarios)
            self.reply(client, user_id, 'Сценарий сохранён. Результат будет в полном /market.')
            return
        if command == 'sources':
            self.sources(client, user_id, argument)
        elif command == 'indicators':
            self.indicators(client, user_id, argument)
        elif command == 'correlations':
            self.correlations(client, user_id, argument)
        elif command == 'alerts':
            self.alerts(client, user_id, argument.removeprefix('market').strip())
        elif command == 'forecasts':
            allowed = {r['id']: r for r in market.selected(self.store, user_id)}
            lines = ['Прогнозы рынка:']
            for identity, row in allowed.items():
                predictions = self.store.query('SELECT * FROM market_forecasts WHERE series_id=? AND origin=(SELECT MAX(origin) FROM market_forecasts WHERE series_id=?) ORDER BY horizon', (identity, identity))
                from .market_store import forecast_history
                predictions = forecast_history(self.store, identity, latest=True) or predictions
                unit = json.loads(row['metadata'])['unit']
                if not predictions:
                    meta = json.loads(row['metadata'])
                    reason = 'единицы требуют проверки' if unit == 'source_units' else 'частота не поддерживает прогнозы' if meta['frequency'] not in {'daily', 'weekly', 'monthly', 'quarterly', 'annual'} else 'недостаточно свежей истории или покрытия'
                    lines.append(row['title'] + ': нет прогноза — ' + reason)
                for p in predictions[-12:]:
                    bounds = f"[{p['lower']:.4g}; {p['upper']:.4g}]" if p['lower'] is not None else 'диапазон не оценён'
                    lines.append(f"{row['title']}: {p['target']} → {p['prediction']:.5g} {unit}; {p['model']}; {bounds}")
                from .market_analysis import evaluate
                lines.append(identity + ': ' + dumps(evaluate(forecast_history(self.store, identity) or self.store.query('SELECT * FROM market_forecasts WHERE series_id=?', (identity,)))))
            self.reply(client, user_id, '\n'.join(lines) if allowed else 'Сначала выберите /sources и запросите /market.')
        else:
            payload = {'actor_id': user_id, 'recipient': str(user_id), 'report_id': uuid4().hex}
            if command == 'market':
                profile = argument or self.store.get_setting(f'market_profile:{user_id}', 'all')
                if profile not in {'all', 'macro', 'investment', 'liquidity'}:
                    raise ValueError('Профили: all, macro, investment, liquidity')
                self.store.set_setting(f'market_profile:{user_id}', profile)
                payload['profile'] = profile
                payload['selection'] = [r['id'] for r in market.selected(self.store, user_id)]
            elif command == 'export':
                args = argument.removeprefix('market').strip().split()
                if len(args) not in {1, 3} or args[0] not in {'csv', 'xlsx'}:
                    raise ValueError('/export market csv|xlsx [YYYY-MM-DD YYYY-MM-DD]')
                from .telegram_data_jobs import export_dates
                payload['format'] = args[0]
                payload['date_from'], payload['date_to'] = export_dates(*(args[1:] or [None, None]))
                payload['selection'] = [r['id'] for r in market.selected(self.store, user_id)]
            elif command == 'backtest':
                args = argument.removeprefix('market').strip()
                count = positive_id(args) if args else 12
                if count > 52:
                    raise ValueError('От 1 до 52 точек')
                payload['max_origins'] = count
                payload['forecast'] = self.store.get_setting(f'chart:{user_id}:market', {}).get('forecast')
                payload['selection'] = [r['id'] for r in market.selected(self.store, user_id)]
            else:
                raise ValueError('Неизвестная рыночная команда')
            key = f'market:{command}:{user_id}:{message_id}' if message_id is not None else None
            identifier = self.store.enqueue('market_' + ('report' if command == 'market' else command), payload, key)
            screen('mk:result:' + command, 'mk:menu:all' if command == 'market' else 'menu:data')
            self.reply(client, user_id, f'Задание {identifier} в очереди. Результат придёт через обычную доставку.')

    def sources(self, client, user_id, argument):
        require_actor(self.store, user_id, admin=True)
        screen('menu:sources', 'pj:market:manage')
        args = argument.split()
        if args == ['collect']:
            identifier = self.store.enqueue('market_collect', {'actor_id': user_id})
            self.reply(client, user_id, f'Общий сбор в очереди: {identifier}. Статус источников: /sources.')
            return
        if args and args[0] == 'schedule':
            if len(args) not in {2, 3} or args[1] not in {'on', 'off'}:
                raise ValueError('/sources schedule on|off [10:00,18:30]')
            from .autocollect import update_schedule
            from .scheduling import validate_subscription
            previous = self.store.get_setting('market_schedule', {})
            times = validate_subscription({'times': args[2].split(',') if len(args) == 3 else previous.get('times', ['18:30'])})['times']
            update_schedule(self.store, 'market', user_id, enabled=args[1] == 'on', changes={'times': times})
        elif args and args[0] in {'on', 'off'} and len(args) == 2:
            market.enable(self.store, args[1], args[0] == 'on')
        elif args and args[0] == 'show' and len(args) == 2:
            row = self.store.one('SELECT * FROM market_series WHERE id=?', (args[1],))
            if not row:
                raise ValueError('Источник не найден')
            actual = self.store.one('SELECT MIN(date) start,MAX(date) end,COUNT(*) n FROM market_observations WHERE series_id=?', (row['id'],))
            meta = json.loads(row['metadata'])
            screen('ms:show:' + str(self.store.one('SELECT rowid token FROM market_series WHERE id=?', (row['id'],))['token']), 'menu:sources')
            self.reply(client, user_id, row['title'] + '\n' + meta.get('description', '') +
                f"\nСбор: {'включён' if row['enabled'] else 'выключен'}\nЧастота: {meta.get('frequency', '—')} · единицы: {meta.get('unit', '—')}\nСохранённые данные: {actual['start'] or 'нет'} — {actual['end'] or 'нет'} ({actual['n']} наблюдений)\nСтатус обновления: {row['status']}\n{meta['url']}",
                _keyboard([[(('Выключить сбор' if row['enabled'] else 'Включить сбор'), 'ms:toggle:' + str(self.store.one('SELECT rowid token FROM market_series WHERE id=?', (row['id'],))['token']))],
                    [('Показатели и вложенные наборы', 'ms:open:' + str(self.store.one('SELECT rowid token FROM market_series WHERE id=?', (row['id'],))['token']))],
                    [('Проверить загруженную историю', 'ms:audit:' + str(self.store.one('SELECT rowid token FROM market_series WHERE id=?', (row['id'],))['token']))],
                    [('Экспорт CSV', 'ms:csv:' + str(self.store.one('SELECT rowid token FROM market_series WHERE id=?', (row['id'],))['token']))],
                    [('Указать единицы по источнику', 'ms:unit:' + str(self.store.one('SELECT rowid token FROM market_series WHERE id=?', (row['id'],))['token']))],
                    [('← К каталогу', 'ms:catalog:0')]]))
            return
        elif args and args[0] == 'unit' and len(args) >= 3:
            row = self.store.one('SELECT * FROM market_series WHERE id=?', (args[1],))
            if not row:
                raise ValueError('Источник не найден')
            unit = ' '.join(args[2:])
            if len(unit) > 80:
                raise ValueError('Единица до 80 символов')
            self.store.set_setting('market_unit:' + args[1], unit)
            metadata = json.loads(row['metadata'])
            metadata['unit'] = unit
            self.store.execute('UPDATE market_series SET metadata=? WHERE id=?', (dumps(metadata), row['id']))
        elif args and args[0] == 'moex':
            if len(args) not in {3, 4}:
                raise ValueError('/sources moex shares|bonds|index КОД [РЕЖИМ]')
            market.add_security(self.store, args[1], args[2].upper(), args[3].upper() if len(args) == 4 else None)
        elif args and args[0] == 'history':
            if len(args) != 3 or args[1] not in {'daily', 'weekly', 'monthly', 'quarterly', 'annual'} or not 1 <= positive_id(args[2]) <= 50:
                raise ValueError('/sources history daily|weekly|monthly|quarterly|annual 1..50')
            config = self.store.get_setting('market_history_years', {})
            config[args[1]] = int(args[2])
            self.store.set_setting('market_history_years', config)
        elif args and args[0] == 'features':
            if len(args) != 3:
                raise ValueError('/sources features ЦЕЛЕВОЙ_ID ID,ID или -')
            selected = {r['id'] for r in market.selected(self.store)}
            features = [] if args[2] == '-' else args[2].split(',')
            if len(features) > 10 or not {args[1], *features}.issubset(selected):
                raise ValueError('Выберите до 10 включённых числовых рядов')
            self.store.set_setting('market_features:' + args[1], features)
        elif args and args[0] == 'params' and len(args) == 3:
            if not self.store.one('SELECT id FROM market_series WHERE id=?', (args[1],)):
                raise ValueError('Источник не найден')
            params = {}
            for entry in args[2].split(','):
                key, value = entry.split('=', 1)
                if not re.fullmatch(r'(?:UniDbQuery\.)?[A-Za-z0-9_.]{1,50}', key) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', value):
                    raise ValueError('Недопустимый параметр источника')
                params[key] = value
            self.store.set_setting('market_params:' + args[1], params)
            self.store.execute("UPDATE market_series SET enabled=0,status='archived' WHERE dataset=? AND id<>? AND json_extract(metadata,'$.kind')='series'", (args[1], args[1]))
            self.store.set_setting('market_last_collection', {})
        elif args and args[0] not in {'page'}:
            raise ValueError('/sources on|off ID; /sources moex shares|bonds|index КОД [РЕЖИМ]; /sources history ЧАСТОТА ЛЕТ')
        page = positive_id(args[1]) if len(args) == 2 and args[0] == 'page' else 1
        self.catalog(client, user_id, page-1)

    def catalog(self, client, actor, page=0):
        from .source_catalog import GROUPS, browse, type_label
        require_actor(self.store, actor, admin=True)
        state = self.store.get_setting(f'source_catalog:{actor}', {})
        search = self.store.get_setting(f'source_search:{actor}', '')
        if not state and not search:
            screen('menu:sources', 'pj:market:manage')
            rows = [[(title, 'ms:group:' + key)] for key, (title, _) in GROUPS.items()]
            rows += [[('Поиск источника', 'ui:sources:search')], [('Автосбор и расписания', 'menu:autocollect')],
                     [('Расписание сбора рынка', 'ui:schedule:overview')],
                     [('Глубина первой загрузки', 'ui:history:menu')], [('Собрать недостающее', 'ms:collect:new')],
                     [('Целостность данных', 'ig:menu')], [('Добавить бумагу', 'mk:input:sources')]]
            self.reply(client, actor, 'Каталог источников\nВыберите группу → набор → показатель. ✓ — сбор включён. Выбор источника не включает расписание или доставку.', _keyboard(rows))
            return
        rows, page, total, entries = browse(self.store, group=state.get('group') if not search else None,
                                           parent_id=state.get('parent'), search=search, page=page)
        route = 'ms:catalog:' + str(page)
        back = 'ms:up:0' if state.get('parent') else 'ms:root:0'
        screen(route, back)
        title = entries.get(state.get('parent'), {}).get('title', GROUPS.get(state.get('group'), ('Поиск', []))[0])
        buttons = []
        for row in rows:
            action = 'toggle' if entries[row['id']].get('kind') == 'series' else 'open'
            buttons.append([(checked(row['title'][:40], row['enabled']), f"ms:{action}:{row['token']}"), ('ℹ', f"ms:show:{row['token']}")])
        if page:
            buttons.append([('‹ Предыдущая', f'ms:catalog:{page-1}')])
        if (page+1)*8 < total:
            buttons.append([('Следующая ›', f'ms:catalog:{page+1}')])
        buttons += [[('Поиск', 'ui:sources:search')], [('Все группы', 'ms:root:0')]]
        if search:
            buttons.append([('Сбросить поиск', 'ms:reset:new')])
        lines = [f'{title}\nСтраница {page+1} из {max(1, (total+7)//8)}. Открыть набор — нажать название.']
        lines += [f"{row['title'][:65]} · {type_label(entries[row['id']])} · {row['status']}" for row in rows]
        if not rows:
            lines.append('Вложенных показателей пока нет. Откройте сведения о наборе и включите сбор; числовые ряды появятся после первой загрузки. Справочники доступны по ссылке и в экспорте.')
        self.reply(client, actor, '\n'.join(lines), _keyboard(buttons))

    def indicators(self, client, user_id, argument, page=0):
        rows = market.selected(self.store)
        # Reuse the existing scope checks even for the list of potential choices.
        rows = [r for r in rows if r['source'] != 'wordstat' or self._wordstat_allowed(user_id, r)]
        allowed = {r['id'] for r in rows}
        if argument:
            chosen = [] if argument == 'all' else argument.split(',')
            if not set(chosen).issubset(allowed):
                raise ValueError('Выберите ID включённых числовых рядов через запятую или all')
            self.store.set_setting(f'market_selection:{user_id}', chosen)
        screen('menu:indicators', 'menu:settings')
        chosen = {r['id'] for r in market.selected(self.store, user_id)}
        size = 8
        page = min(max(page, 0), max(0, (len(rows)-1)//size))
        lines = [f'Мои показатели рынка\nВыбрано {len(chosen)} из {len(rows)}. ✓ — включено в ваши обзоры.\nСтраница {page+1} из {max(1, (len(rows)+size-1)//size)}.']
        buttons = []
        for row in rows[page*size:(page+1)*size]:
            token = self.store.one('SELECT rowid token FROM market_series WHERE id=?', (row['id'],))['token']
            buttons.append([(checked(row['title'][:45], row['id'] in chosen), f'mi:toggle:{token}')])
        if page:
            buttons.append([('‹ Предыдущая', f'mi:page:{page-1}')])
        if (page+1)*size < len(rows):
            buttons.append([('Следующая ›', f'mi:page:{page+1}')])
        buttons += [[('Выбрать все', 'mi:all:0'), ('Снять все', 'mi:none:0')]]
        if not rows:
            lines.append('Администратор ещё не включил доступные числовые ряды.')
        self.reply(client, user_id, '\n'.join(lines), _keyboard(buttons))

    def _wordstat_allowed(self, user_id, row):
        from .telegram_data_jobs import actor_scope
        meta = json.loads(row['metadata'])
        scope = actor_scope(self.store, user_id, meta['section'])
        return meta['phrase'] in scope['phrases'] and meta['region'] in scope['regions']

    def correlations(self, client, user_id, argument):
        if argument.startswith('events '):
            from .telegram_data_jobs import actor_scope
            parts = argument[7:].split('|')
            if len(parts) != 3:
                raise ValueError('/correlations events ID|фраза|регион')
            identity, phrase, region = parts[0].strip(), parts[1].strip(), positive_id(parts[2].strip())
            scope = actor_scope(self.store, user_id, 'regions')
            if identity not in {r['id'] for r in market.selected(self.store, user_id)} or phrase not in scope['phrases'] or region not in scope['regions']:
                raise PermissionError('Выберите свой доступный показатель, фразу и регион')
            studies = self.store.get_setting(f'market_events:{user_id}', [])
            item = {'series_id': identity, 'phrase': phrase, 'region': region}
            if item not in studies:
                if len(studies) >= 50:
                    raise ValueError('Максимум 50 исследований событий')
                studies.append(item)
            self.store.set_setting(f'market_events:{user_id}', studies)
            self.reply(client, user_id, 'Исследование событий добавлено; результат в /market. Минимум 10 независимых событий.')
            return
        args = argument.split()
        pairs = self.store.get_setting(f'market_pairs:{user_id}', [])
        if args:
            if args == ['clear']:
                pairs = []
                self.store.set_setting(f'market_events:{user_id}', [])
            elif len(args) in {3, 4, 5, 6} and args[0] == 'add':
                allowed = {r['id'] for r in market.selected(self.store, user_id)}
                if not {args[1], args[2]}.issubset(allowed) or args[1] == args[2] or len(pairs) >= 50:
                    raise ValueError('Выберите два разных доступных ряда; максимум 50 пар')
                pair = {'left': args[1], 'right': args[2], 'lag': int(args[3]) if len(args) > 3 else 0,
                        'window': int(args[4]) if len(args) > 4 else None, 'levels': len(args) == 6 and args[5] == 'levels'}
                from .market_analysis import WINDOWS
                if not -4 <= pair['lag'] <= 4 or pair['window'] not in {None, *(value for windows in WINDOWS.values() for value in windows)}:
                    raise ValueError('Лаг −4..4; окно зависит от частоты ряда: дни 60/120/252, недели 26/52, месяцы 24/60, кварталы 8/20, годы 5/10')
                if pair not in pairs:
                    pairs.append(pair)
            else:
                raise ValueError('/correlations add ID ID [ЛАГ] [ОКНО] [levels]; /correlations clear')
            self.store.set_setting(f'market_pairs:{user_id}', pairs)
        screen('mk:open:correlations', 'menu:data')
        titles = {r['id']: r['title'] for r in market_alerts.allowed_series(self.store, user_id)}
        text = 'Корреляции\n' + ('\n'.join(f"{titles.get(p['left'], p['left'])} ↔ {titles.get(p['right'], p['right'])} · лаг {p['lag']}" for p in pairs[:12]) or 'Пары ещё не выбраны.')
        self.reply(client, user_id, text + '\nРасчёт появится в обзоре рынка. Для связи нужно достаточно сопоставимых данных.', _keyboard([[('Добавить пару', 'ui:pair:left:0')], [('Рассчитать обзор', 'mk:report:all')], [('Дополнительные параметры', 'mk:input:correlations')]]))

    def alerts(self, client, user_id, argument):
        market_alerts.seed(self.store, user_id)
        args = argument.split()
        if args:
            if args[0] == 'add' and len(args) == 4:
                market_alerts.add_rule(self.store, user_id, args[1], args[2], float(args[3]))
            elif args[0] == 'digest' and len(args) == 2 and re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', args[1]):
                self.store.set_setting(f'market_digest:{user_id}', args[1])
            elif args[0] in {'on', 'off', 'immediate', 'digest', 'set'} and len(args) >= 2:
                rule = self.store.one('SELECT * FROM market_rules WHERE id=? AND user_id=?', (positive_id(args[1]), user_id))
                if not rule:
                    raise ValueError('Правило недоступно')
                if args[0] == 'set':
                    config = json.loads(rule['config'])
                    for item in args[2:]:
                        key, value = item.split('=', 1)
                        if key not in market_alerts.DEFAULTS:
                            raise ValueError('Параметры: threshold, window, correlation, delta, confirm, coverage')
                        config[key] = float(value)
                    if (not all(math.isfinite(v) for v in config.values()) or not 2 <= config['window'] <= 252 or
                            not 1 <= config['confirm'] <= 10 or not .9 <= config['coverage'] <= 1 or
                            not 0 <= config['correlation'] <= 1 or not 0 <= config['delta'] <= 2 or not 1 <= config['cooldown'] <= 720):
                        raise ValueError('Недопустимые пороги или покрытие')
                    self.store.execute('UPDATE market_rules SET config=?,cooldown=? WHERE id=?', (dumps(config), config['cooldown'], rule['id']))
                else:
                    mode = 'off' if args[0] == 'off' else args[0] if args[0] in {'immediate', 'digest'} else 'immediate'
                    self.store.execute('UPDATE market_rules SET enabled=?,mode=? WHERE id=?', (int(mode != 'off'), mode, rule['id']))
            else:
                raise ValueError('/alerts market add ID above|below|change|deviation|cross|correlation ПОРОГ; on|off|immediate|digest ID; set ID key=value')
        screen('mk:open:alerts', 'menu:settings')
        page = self.store.get_setting(f'alert_page:{user_id}', 0)
        rows = self.store.query('SELECT * FROM market_rules WHERE user_id=? ORDER BY id', (user_id,))
        page = min(max(0, page), max(0, (len(rows)-1)//8))
        kinds = {'above': 'выше порога', 'below': 'ниже порога', 'change': 'изменение', 'deviation': 'аномалия', 'cross': 'пересечение', 'correlation': 'корреляция'}
        titles = {r['id']: r['title'] for r in market.selected(self.store, user_id)}
        text = f'Мои рыночные уведомления\n✓ — правило включено. Выберите правило для настройки порога и доставки.\nСтраница {page+1} из {max(1, (len(rows)+7)//8)}.'
        buttons = [[(checked(f"{titles.get(r['series_id'], r['series_id'])[:26]} · {kinds[r['kind']]}", r['enabled']), f"ui:rule:{r['id']}")] for r in rows[page*8:(page+1)*8]]
        if page:
            buttons += [[('‹ Предыдущая', f'ui:rulepage:{page-1}')]]
        if (page+1)*8 < len(rows):
            buttons += [[('Следующая ›', f'ui:rulepage:{page+1}')]]
        buttons += [[('Дополнительные правила', 'mk:input:alerts')]]
        self.reply(client, user_id, text, _keyboard(buttons))

    def callback(self, client, user_id, data):
        require_actor(self.store, user_id)
        from .subscription_access import require_route
        require_route(self.store, user_id, data)
        prefix, action, value = data.split(':', 2)
        if prefix == 'user':
            require_actor(self.store, user_id, admin=True)
            if action == 'add':
                screen('user:input', 'menu:users')
                self.pending(user_id, 'add_user')
                self.reply(client, user_id, 'Введите @username или положительный числовой Telegram ID нового пользователя.\n'
                           'Пример: @ivan_test или 123456789.\n'
                           'Если поиск по имени не сработает, попросите пользователя открыть этот бот и нажать /start: '
                           'бот покажет его ID. Отмена: /cancel')
            elif action == 'deny':
                if self.store.one('SELECT role FROM users WHERE user_id=?', (positive_id(value),)) == {'role': 'admin'}:
                    raise ValueError('Нельзя отключить администратора через бот')
                screen('user:confirm', 'menu:users')
                self.reply(client, user_id, f'Отключить доступ пользователя {positive_id(value)}? Его незапущенные задания и следующие части доставки будут отменены. Настройки сохранятся.', _keyboard([[('Отключить доступ', f'user:denyconfirmed:{value}')]]))
            elif action in {'allow', 'denyconfirmed'}:
                change_access(self.store, user_id, positive_id(value), action == 'allow')
                from .telegram_commands import configure_chat
                configure_chat(client, self.store, positive_id(value))
                from .user_management import send_notices
                send_notices(self.store, client, positive_id(value))
                from .telegram_admin import AdminControls
                AdminControls(self.store, self.reply).user_subscriptions(client, user_id, positive_id(value))
            else:
                raise ValueError('Неизвестное действие')
        elif prefix == 'mk':
            if action == 'input' and value in COMMANDS | {'alerts'}:
                if value == 'sources':
                    require_actor(self.store, user_id, admin=True)
                self.pending(user_id, 'command:' + value)
                screen('mk:input:' + value, 'menu:sources' if value == 'sources' else 'mk:open:' + value)
                examples = {'sources': 'moex shares SBER TQBR', 'indicators': 'ID,ID', 'correlations': 'add ID ID 0 60', 'alerts': 'set ID threshold=3 window=60'}
                self.reply(client, user_id, 'Введите параметры: ' + examples.get(value, '') + '. Отмена: /cancel')
            elif action == 'report':
                self.dispatch(client, user_id, 'market', value)
            elif action == 'menu':
                self.market_menu(client, user_id)
            elif action == 'profile' and value in {'all', 'macro', 'investment', 'liquidity'}:
                self.store.set_setting(f'market_profile:{user_id}', value)
                self.market_menu(client, user_id)
            elif action == 'open' and value in COMMANDS | {'alerts'}:
                self.dispatch(client, user_id, value)
            else:
                raise ValueError('Неизвестная команда')
        elif prefix == 'ms':
            require_actor(self.store, user_id, admin=True)
            if action == 'unit':
                row = self.store.one('SELECT id FROM market_series WHERE rowid=?', (positive_id(value),))
                if not row:
                    raise ValueError('Источник не найден')
                self.pending(user_id, 'sourceunit:' + row['id'])
                screen('ms:unit:' + value, 'ms:show:' + value)
                self.reply(client, user_id, 'Введите единицы из методики или заголовка ЦБ, например «млрд руб», «%», «коэффициент». Прогноз заблокирован, пока единицы не проверены. /cancel — отмена.')
                return
            elif action in {'group', 'root', 'up', 'catalog', 'open'}:
                from .source_catalog import GROUPS, parent
                state = self.store.get_setting(f'source_catalog:{user_id}', {})
                if action == 'group':
                    if value not in GROUPS:
                        raise ValueError('Группа не найдена')
                    state = {'group': value}
                    self.store.set_setting(f'source_search:{user_id}', '')
                elif action == 'root':
                    state = {}
                    self.store.set_setting(f'source_search:{user_id}', '')
                elif action == 'up':
                    entries = {r['id']: json.loads(r['metadata']) for r in self.store.query('SELECT id,metadata FROM market_series')}
                    previous = parent(entries.get(state.get('parent'), {}), entries)
                    state = {**state, 'parent': previous}
                elif action == 'open':
                    row = self.store.one('SELECT * FROM market_series WHERE rowid=?', (positive_id(value),))
                    if not row:
                        raise ValueError('Набор не найден')
                    if json.loads(row['metadata']).get('kind') == 'series':
                        return self.sources(client, user_id, 'show ' + row['id'])
                    state = {**state, 'parent': row['id']}
                self.store.set_setting(f'source_catalog:{user_id}', state)
                return self.catalog(client, user_id, int(value) if action == 'catalog' and value.isdigit() else 0)
            elif action in {'audit', 'csv'}:
                from .integrity import enqueue
                from .source_catalog import parent
                target = self.store.one('SELECT id FROM market_series WHERE rowid=?', (positive_id(value),))
                if not target:
                    raise ValueError('Набор не найден')
                entries = {r['id']: json.loads(r['metadata']) for r in self.store.query('SELECT id,metadata FROM market_series')}
                selected = []
                for identity in entries:
                    ancestor, visited = identity, set()
                    while ancestor and ancestor not in visited:
                        visited.add(ancestor)
                        if ancestor == target['id']:
                            selected.append(identity)
                            break
                        ancestor = parent(entries[ancestor], entries)
                if action == 'audit':
                    loaded = {r['series_id'] for r in self.store.query('SELECT DISTINCT series_id FROM market_observations')}
                    identifier = enqueue(self.store, user_id, [identity for identity in selected if identity in loaded])
                    return self.reply(client, user_id, f'Проверка #{identifier} в очереди.', _keyboard([[('Ход проверок', 'ig:checks:0')]]))
                identifier = self.store.enqueue('market_export_sources', {'actor_id': user_id, 'selection': selected, 'format': 'csv', 'report_id': uuid4().hex})
                return self.reply(client, user_id, f'Экспорт #{identifier} в очереди.')
            elif action == 'schedule':
                screen('ms:input:schedule', 'menu:sources')
                self.pending(user_id, 'command:sources')
                self.reply(client, user_id, 'Введите schedule on 10:00,18:30 или schedule off. Отмена: /cancel')
            elif action == 'collect':
                self.sources(client, user_id, 'collect')
            elif action == 'page':
                self.sources(client, user_id, 'page ' + value)
            elif action == 'reset':
                self.store.set_setting(f'source_search:{user_id}', '')
                self.sources(client, user_id, '')
            elif action == 'toggle':
                row = self.store.one('SELECT * FROM market_series WHERE rowid=?', (positive_id(value),))
                if not row:
                    raise ValueError('Источник не найден')
                market.enable(self.store, row['id'], not row['enabled'])
                page = self.store.one('SELECT COUNT(*) n FROM market_series WHERE id<?', (row['id'],))['n']//12+1
                self.sources(client, user_id, 'page ' + str(page))
            elif action == 'show':
                row = self.store.one('SELECT id FROM market_series WHERE rowid=?', (positive_id(value),))
                if not row:
                    raise ValueError('Источник не найден')
                self.sources(client, user_id, 'show ' + row['id'])
        elif prefix == 'mi' and action in {'page', 'all', 'none'}:
            if action != 'page':
                self.store.set_setting(f'market_selection:{user_id}', [] if action == 'all' else ['__none__'])
            self.indicators(client, user_id, '', int(value) if action == 'page' and value.isdigit() else 0)
        elif prefix == 'mi' and action == 'toggle':
            row = self.store.one("SELECT * FROM market_series WHERE rowid=? AND enabled=1 AND json_extract(metadata,'$.kind')='series'", (positive_id(value),))
            if not row:
                raise ValueError('Показатель недоступен')
            if row['source'] == 'wordstat' and not self._wordstat_allowed(user_id, row):
                raise PermissionError('Показатель недоступен для вашей подписки')
            chosen = self.store.get_setting(f'market_selection:{user_id}', []) or [r['id'] for r in market.selected(self.store)]
            chosen = [s for s in chosen if s != row['id']] if row['id'] in chosen else [*chosen, row['id']]
            # An explicit empty selection differs from the default "all".
            self.store.set_setting(f'market_selection:{user_id}', chosen or ['__none__'])
            self.indicators(client, user_id, '', self.store.one("SELECT COUNT(*) n FROM market_series WHERE enabled=1 AND json_extract(metadata,'$.kind')='series' AND id<?", (row['id'],))['n']//8)
        elif prefix == 'ma' and action == 'toggle':
            rule = self.store.one('SELECT * FROM market_rules WHERE id=? AND user_id=?', (positive_id(value), user_id))
            if not rule:
                raise ValueError('Правило недоступно')
            self.alerts(client, user_id, ('off ' if rule['enabled'] else 'on ') + value)
        else:
            raise ValueError('Неизвестная команда')

    def market_menu(self, client, actor):
        screen('mk:menu:all', 'menu:data')
        profile = self.store.get_setting(f'market_profile:{actor}', 'all')
        labels = {'macro': 'Макроэкономика', 'investment': 'Инвестиции', 'liquidity': 'Ликвидность', 'all': 'Все разделы'}
        self.reply(client, actor, f"Обзор рынка\nПрофиль: {labels[profile]}\nВыберите профиль, затем запросите обзор. Личные показатели настраиваются в «Моих настройках».",
            _keyboard([[(checked(label, key == profile), f'mk:profile:{key}')] for key, label in labels.items()] +
                [[('Обновить данные и получить обзор', f'mk:report:{profile}')], [('Графики рынка', 'ch:menu:market')], [('Корреляции', 'mk:open:correlations')]]))
