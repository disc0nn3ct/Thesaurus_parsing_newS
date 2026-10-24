"""Guided analytics and administration forms, with private actor scopes."""
import json
import math

from . import market
from .market_analysis import evaluate
from .store import dumps, utcnow
from .telegram_admin import positive_id
from .telegram_charts import keyboard
from .telegram_data_jobs import require_actor
from .telegram_navigation import checked, screen


class DialogControls:
    def __init__(self, bot):
        self.bot, self.store, self.reply = bot, bot.store, bot._reply

    def forecast_menu(self, client, actor, page=0):
        rows = market.selected(self.store, actor)
        page = min(max(0, page), max(0, (len(rows)-1)//8))
        screen('ui:forecasts:market', 'menu:data')
        buttons = [[('Все базы Банка России · подключение и выбор', 'cb:menu')]]
        for row in rows[page*8:(page+1)*8]:
            token = self.store.one('SELECT rowid token FROM market_series WHERE id=?', (row['id'],))['token']
            buttons += [[(row['title'][:47], f'ui:forecast:{token}')]]
        if page:
            buttons += [[('‹ Предыдущая', f'ui:forecastpage:{page-1}')]]
        if (page+1)*8 < len(rows):
            buttons += [[('Следующая ›', f'ui:forecastpage:{page+1}')]]
        self.reply(client, actor, f'Прогнозы рынка\nВыберите показатель. Страница {page+1} из {max(1, (len(rows)+7)//8)}.' +
                   ('\nДоступных показателей пока нет. Выберите источники и получите отчёт.' if not rows else ''), keyboard(buttons))

    def _series(self, actor, token):
        row = self.store.one('SELECT * FROM market_series WHERE rowid=?', (positive_id(token),))
        if not row or row['id'] not in {r['id'] for r in market.selected(self.store, actor)}:
            raise PermissionError('Показатель недоступен. Выберите его заново.')
        return row

    def pair_menu(self, client, actor, side, page=0):
        rows = market.selected(self.store, actor)
        pair = self.store.get_setting(f'ui_pair:{actor}', {})
        if side == 'right':
            if pair.get('left') not in {r['id'] for r in rows}:
                raise ValueError('Сначала выберите первый показатель')
            rows = [r for r in rows if r['id'] != pair['left']]
        page = min(max(0, page), max(0, (len(rows)-1)//8))
        screen(f'ui:pair:{side}:{page}', 'mk:open:correlations' if side == 'left' else 'ui:pair:left:0')
        buttons = []
        for row in rows[page*8:(page+1)*8]:
            token = self.store.one('SELECT rowid token FROM market_series WHERE id=?', (row['id'],))['token']
            buttons += [[(row['title'][:47], f'ui:pair:pick{side}:{token}')]]
        if page:
            buttons += [[('‹ Предыдущая', f'ui:pair:{side}:{page-1}')]]
        if (page+1)*8 < len(rows):
            buttons += [[('Следующая ›', f'ui:pair:{side}:{page+1}')]]
        text = ('Шаг 1 из 2 · первый показатель' if side == 'left' else 'Шаг 2 из 2 · второй показатель')
        if side == 'right':
            text += '\nПервый: ' + pair.get('title', pair['left'])
        text += '\nБазовая пара сравнивает изменения без лага. Дополнительные параметры можно настроить из списка корреляций.'
        if not rows:
            text += '\nНужны как минимум два доступных числовых ряда.'
        self.reply(client, actor, text, keyboard(buttons))

    def rule_card(self, client, actor, identifier):
        rule = self.store.one('SELECT * FROM market_rules WHERE id=? AND user_id=?', (positive_id(identifier), actor))
        if not rule:
            raise PermissionError('Правило недоступно')
        row = self.store.one('SELECT * FROM market_series WHERE id=?', (rule['series_id'],))
        title = row['title'] if row else rule['series_id']
        config = json.loads(rule['config'])
        kinds = {'above': 'значение выше порога', 'below': 'значение ниже порога', 'change': 'изменение в %',
                 'deviation': 'отклонение от истории в σ', 'cross': 'пересечение порога', 'correlation': 'изменение корреляции'}
        modes = {'off': 'Выключено', 'immediate': 'Сразу', 'digest': 'В сводке'}
        screen('ui:rule:' + identifier, 'mk:open:alerts')
        buttons = [[(checked(label, rule['mode'] == key), f'ui:rulemode:{identifier}:{key}')] for key, label in modes.items()]
        buttons += [[('Условие срабатывания', f'ui:rulekinds:{identifier}')],
                    [('Изменить порог', f'ui:rulethreshold:{identifier}')], [('Параметры проверки', f'ui:ruleparams:{identifier}')],
                    [('Время сводки', 'ui:digest:time')]]
        threshold = config['correlation'] if rule['kind'] == 'correlation' else config['threshold']
        self.reply(client, actor, f"Уведомление #{identifier}\n{title}\nУсловие: {kinds[rule['kind']]}\nПорог: {threshold:g}\nДоставка: {modes[rule['mode']]}\nМинимальный охват: {config['coverage']:.0%}; повтор не чаще чем раз в {rule['cooldown']:g} ч.\nСводка: {self.store.get_setting(f'market_digest:{actor}', '18:30')} · Москва", keyboard(buttons))

    def callback(self, client, actor, data):
        require_actor(self.store, actor)
        args = data.split(':')
        if len(args) in {3, 4} and args[1] in {'rulekinds', 'rulekind', 'ruleparams', 'ruleparam'}:
            identifier = positive_id(args[2])
            rule = self.store.one('SELECT * FROM market_rules WHERE id=? AND user_id=?', (identifier, actor))
            if not rule:
                raise PermissionError('Правило недоступно')
            kinds = {'above': 'Значение выше порога', 'below': 'Значение ниже порога', 'change': 'Изменение в %',
                     'deviation': 'Отклонение от истории в σ', 'cross': 'Пересечение порога'}
            if args[1] == 'rulekind' and len(args) == 4 and args[3] in kinds:
                self.store.execute("UPDATE market_rules SET kind=?,enabled=0,mode='off' WHERE id=? AND user_id=?", (args[3], identifier, actor))
                self.rule_card(client, actor, str(identifier))
            elif args[1] == 'rulekinds' and len(args) == 3:
                screen(data, f'ui:rule:{identifier}')
                self.reply(client, actor, 'Выберите условие. Изменение условия выключает правило: настройте порог и включите доставку.',
                           keyboard([[(checked(label, kind == rule['kind']), f'ui:rulekind:{identifier}:{kind}')] for kind, label in kinds.items()]))
            elif args[1] == 'ruleparams' and len(args) == 3:
                screen(data, f'ui:rule:{identifier}')
                config = json.loads(rule['config'])
                self.reply(client, actor, f"Окно: {config['window']:g} наблюдений\nОхват: {config['coverage']:.0%}\nПовтор: не чаще {config['cooldown']:g} ч.",
                           keyboard([[(label, f'ui:ruleparam:{identifier}:{field}')] for field, label in
                                     [('window', 'Размер окна'), ('coverage', 'Минимальный охват'), ('cooldown', 'Интервал повторов')]]))
            elif args[1] == 'ruleparam' and len(args) == 4 and args[3] in {'window', 'coverage', 'cooldown'}:
                screen(data, f'ui:ruleparams:{identifier}')
                self.bot.market_controls.pending(actor, f'dialog:ruleparam:{identifier}|{args[3]}')
                prompt = {'window': 'Введите число наблюдений от 2 до 252.', 'coverage': 'Введите охват в процентах от 90 до 100.',
                          'cooldown': 'Введите интервал повторов в часах от 1 до 720.'}
                self.reply(client, actor, prompt[args[3]])
            else:
                raise ValueError('Неизвестная настройка правила')
            return True
        if len(args) == 3 and args[1] in {'forecast', 'forecastpage'}:
            if args[1] == 'forecastpage':
                if not args[2].isdigit():
                    raise ValueError('Неверная страница')
                self.forecast_menu(client, actor, int(args[2]))
            else:
                row = self._series(actor, args[2])
                meta = json.loads(row['metadata'])
                predictions = self.store.query('SELECT * FROM market_forecasts WHERE series_id=? AND origin=(SELECT MAX(origin) FROM market_forecasts WHERE series_id=?) ORDER BY horizon', (row['id'], row['id']))
                lines = ['Прогноз · ' + row['title']]
                for p in predictions:
                    bounds = f"[{p['lower']:.4g}; {p['upper']:.4g}]" if p['lower'] is not None else 'диапазон пока не оценён'
                    lines.append(f"{p['target']}: {p['prediction']:.5g} {meta['unit']} · {bounds}")
                if predictions:
                    lines += [f"Выпуск: {predictions[0]['origin']}; модель: {predictions[0]['model']}"]
                else:
                    lines += ['Нет сохранённого прогноза: нужна подходящая свежая история. Получите обзор рынка, чтобы обновить расчёт.']
                metrics = evaluate(self.store.query('SELECT * FROM market_forecasts WHERE series_id=?', (row['id'],)))
                if metrics.get('n'):
                    lines += [f"Проверено: {metrics['n']}; MAE: {metrics.get('mae')}; ошибка базового прогноза: {metrics.get('baseline_mae')}"]
                screen(data, 'ui:forecasts:market')
                self.reply(client, actor, '\n'.join(lines), keyboard([[('График этого показателя', f'ui:chartfor:{args[2]}')]]))
        elif len(args) == 3 and args[1] == 'chartfor':
            row = self._series(actor, args[2])
            options = self.bot.chart_controls.options(actor, 'market')
            options['identity'] = row['id']
            options.pop('series_ids', None)
            options.pop('selection_mode', None)
            self.store.set_setting(f'chart:{actor}:market', options)
            self.bot.chart_controls.menu(client, actor, 'market')
        elif len(args) == 4 and args[1] == 'pair':
            action, value = args[2:]
            if action in {'left', 'right'} and value.isdigit():
                self.pair_menu(client, actor, action, int(value))
            elif action == 'pickleft':
                row = self._series(actor, value)
                self.store.set_setting(f'ui_pair:{actor}', {'left': row['id'], 'title': row['title']})
                self.pair_menu(client, actor, 'right')
            elif action == 'pickright':
                right = self._series(actor, value)
                pair = self.store.get_setting(f'ui_pair:{actor}', {})
                if pair.get('left') not in {r['id'] for r in market.selected(self.store, actor)}:
                    raise PermissionError('Первый показатель больше не доступен')
                self.bot.market_controls.correlations(client, actor, f"add {pair['left']} {right['id']} 0")
                self.store.set_setting(f'ui_pair:{actor}', {})
            else:
                raise ValueError('Неверная настройка пары')
        elif len(args) == 3 and args[1] in {'rule', 'rulepage', 'rulethreshold'}:
            if args[1] == 'rulepage' and args[2].isdigit():
                self.store.set_setting(f'alert_page:{actor}', int(args[2]))
                self.bot.market_controls.alerts(client, actor, '')
            elif args[1] == 'rule':
                self.rule_card(client, actor, args[2])
            elif args[1] == 'rulethreshold':
                if not self.store.one('SELECT id FROM market_rules WHERE id=? AND user_id=?', (positive_id(args[2]), actor)):
                    raise PermissionError('Правило недоступно')
                screen('ui:ruleinput:' + args[2], 'ui:rule:' + args[2])
                self.bot.market_controls.pending(actor, 'dialog:threshold:' + args[2])
                self.reply(client, actor, 'Введите числовой порог. Можно использовать точку или запятую, например 2,5.')
            else:
                raise ValueError('Неверная страница')
        elif len(args) == 4 and args[1] == 'rulemode' and args[3] in {'off', 'immediate', 'digest'}:
            rule = self.store.one('SELECT * FROM market_rules WHERE id=? AND user_id=?', (positive_id(args[2]), actor))
            if not rule:
                raise PermissionError('Правило недоступно')
            from .market_alerts import allowed_series
            if args[3] != 'off' and rule['series_id'] not in {r['id'] for r in allowed_series(self.store, actor)}:
                raise PermissionError('Сначала включите показатель в своих настройках')
            self.store.execute('UPDATE market_rules SET enabled=?,mode=? WHERE id=? AND user_id=?', (int(args[3] != 'off'), args[3], rule['id'], actor))
            self.rule_card(client, actor, args[2])
        elif data == 'ui:digest:time':
            screen(data, 'mk:open:alerts')
            self.bot.market_controls.pending(actor, 'dialog:digest:market')
            self.reply(client, actor, 'Введите время сводки по Москве, например 18:30. Это не включает уведомления автоматически.')
        elif len(args) == 3 and args[1] == 'budget' and args[2] in {'wordstat', 'ai', 'ai_tokens'}:
            require_actor(self.store, actor, admin=True)
            screen(data, 'menu:budget')
            self.bot.market_controls.pending(actor, 'dialog:budget:' + args[2])
            self.reply(client, actor, 'Введите новый лимит целым числом. Для отключения обращений к провайдеру — 0; лимит выходных токенов должен быть положительным.')
        elif data == 'ui:sources:search':
            require_actor(self.store, actor, admin=True)
            screen(data, 'menu:sources')
            self.bot.market_controls.pending(actor, 'dialog:search:sources')
            self.reply(client, actor, 'Введите часть названия источника, например RUONIA, инфляция или moex. Поиск не меняет включённые источники.')
        elif args[1:2] == ['autocollect']:
            from .autocollect import menu, update_schedule
            require_actor(self.store, actor, admin=True)
            if len(args) == 4 and args[2] in {'trends', 'market'} and args[3] in {'on', 'off'}:
                if args[2] == 'trends':
                    from .google_trends import active
                    if not active(self.store):
                        raise ValueError('Google Trends не выбран как поисковый источник')
                update_schedule(self.store, args[2], actor, enabled=args[3] == 'on')
            else:
                raise ValueError('Неизвестная настройка автосбора')
            menu(self.store, self.reply, client, actor)
        elif len(args) == 3 and args[1] == 'schedule' and args[2] in {'overview', 'on', 'off', 'times'}:
            require_actor(self.store, actor, admin=True)
            config = self.store.get_setting('market_schedule', {'enabled': False, 'times': ['18:30']})
            if args[2] in {'on', 'off'}:
                from .autocollect import update_schedule
                config = update_schedule(self.store, 'market', actor, enabled=args[2] == 'on')
            if args[2] == 'times':
                screen(data, 'ui:schedule:overview')
                self.bot.market_controls.pending(actor, 'dialog:schedule:times')
                self.reply(client, actor, 'Введите время сбора по Москве через запятую, например 10:00,18:30. Это не включает расписание или рассылку автоматически.')
            else:
                screen('ui:schedule:overview', 'menu:sources')
                self.reply(client, actor, f"Сбор данных рынка\nРасписание: {'включено' if config.get('enabled') else 'выключено'}\nВремя: {', '.join(config.get('times', ['18:30']))} · Москва\nЭтот сбор общий для пользователей и не отправляет отчёты. Личная доставка включается отдельно в подписках.", keyboard([
                    [('Выключить расписание' if config.get('enabled') else 'Включить расписание', 'ui:schedule:' + ('off' if config.get('enabled') else 'on'))],
                    [('Изменить время', 'ui:schedule:times')], [('Все источники автосбора', 'menu:autocollect')]]))
        elif len(args) in {3, 4} and args[1] == 'history':
            require_actor(self.store, actor, admin=True)
            config = self.store.get_setting('market_history_years', {})
            if args[2] == 'menu':
                screen(data, 'menu:sources')
                self.reply(client, actor, 'Глубина загрузки\nВыберите частоту данных. Расширение истории догружает только недостающие периоды; сохранённые значения берутся из базы.', keyboard([
                    [('Ежедневные', 'ui:history:daily')], [('Недельные', 'ui:history:weekly')], [('Месячные', 'ui:history:monthly')],
                    [('Квартальные', 'ui:history:quarterly')], [('Годовые', 'ui:history:annual')]]))
            elif args[2] in {'daily', 'weekly', 'monthly', 'quarterly', 'annual'}:
                frequency = args[2]
                if len(args) == 4:
                    years = positive_id(args[3])
                    if years > 50:
                        raise ValueError('Глубина истории: от 1 до 50 лет')
                    config[frequency] = years
                    self.store.set_setting('market_history_years', config)
                years = config.get(frequency, {'daily': 5, 'weekly': 10}.get(frequency, 20))
                screen('ui:history:' + frequency, 'ui:history:menu')
                self.reply(client, actor, f'История · {frequency}\nПервая загрузка: до {years} лет. Фактическая доступность зависит от источника. Изменение не запускает сбор.', keyboard([
                    [(checked(f'{n} лет', n == years), f'ui:history:{frequency}:{n}')] for n in (1, 3, 5, 10, 20, 50)]))
            else:
                raise ValueError('Неизвестная частота')
        elif len(args) == 3 and args[1] == 'jobs' and args[2] in {'all', 'pending', 'running', 'failed', 'done'}:
            require_actor(self.store, actor, admin=True)
            self.bot.admin_controls.jobs(client, actor, '' if args[2] == 'all' else args[2])
        else:
            return False
        return True

    def input(self, client, actor, text):
        state = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        if not state or not (state['pending'] or '').startswith('dialog:') or text.startswith('/'):
            return False
        require_actor(self.store, actor)
        _, action, value = state['pending'].split(':', 2)
        if action == 'ruleparam':
            identifier, field = value.split('|', 1)
            rule = self.store.one('SELECT * FROM market_rules WHERE id=? AND user_id=?', (positive_id(identifier), actor))
            if not rule or field not in {'window', 'coverage', 'cooldown'}:
                raise PermissionError('Правило недоступно')
            try:
                number = float(text.strip().replace(',', '.'))
            except ValueError:
                raise ValueError('Введите число') from None
            limits = {'window': (2, 252), 'coverage': (90, 100), 'cooldown': (1, 720)}
            low, high = limits[field]
            if not math.isfinite(number) or not low <= number <= high or field == 'window' and not number.is_integer():
                raise ValueError(f'Допустимо от {low} до {high}; размер окна — целое число')
            config = json.loads(rule['config'])
            config[field] = number / 100 if field == 'coverage' else int(number) if field == 'window' else number
            self.store.execute('UPDATE market_rules SET config=?,cooldown=? WHERE id=? AND user_id=?',
                               (dumps(config), config['cooldown'], rule['id'], actor))
            self.bot.market_controls.pending(actor, None)
            self.rule_card(client, actor, identifier)
        elif action == 'threshold':
            rule = self.store.one('SELECT * FROM market_rules WHERE id=? AND user_id=?', (positive_id(value), actor))
            if not rule:
                raise PermissionError('Правило недоступно')
            try:
                number = float(text.strip().replace(',', '.'))
            except ValueError:
                raise ValueError('Нужно число, например 2,5.') from None
            if not math.isfinite(number) or rule['kind'] == 'correlation' and not 0 <= number <= 1:
                raise ValueError('Нужно конечное число; для корреляции — от 0 до 1.')
            config = json.loads(rule['config'])
            config['correlation' if rule['kind'] == 'correlation' else 'threshold'] = number
            self.store.execute('UPDATE market_rules SET config=? WHERE id=? AND user_id=?', (dumps(config), rule['id'], actor))
            self.bot.market_controls.pending(actor, None)
            self.rule_card(client, actor, value)
        elif action == 'budget':
            require_actor(self.store, actor, admin=True)
            self.bot.market_controls.pending(actor, None)
            try:
                self.bot.admin_controls.budget(client, actor, value + ' ' + text.strip())
            except (ValueError, PermissionError):
                self.bot.market_controls.pending(actor, state['pending'])
                raise
        elif action == 'search':
            require_actor(self.store, actor, admin=True)
            if not 1 <= len(text.strip()) <= 80:
                raise ValueError('Введите от 1 до 80 символов')
            self.store.set_setting(f'source_search:{actor}', text.strip())
            self.bot.market_controls.pending(actor, None)
            self.bot.market_controls.sources(client, actor, '')
        elif action == 'schedule':
            require_actor(self.store, actor, admin=True)
            from .scheduling import validate_subscription
            times = validate_subscription({'times': text.split(',')})['times']
            config = self.store.get_setting('market_schedule', {'enabled': False})
            config.update(times=times, actor_id=actor)
            config.setdefault('created_at', utcnow())
            self.store.set_setting('market_schedule', config)
            self.bot.market_controls.pending(actor, None)
            self.callback(client, actor, 'ui:schedule:overview')
        elif action == 'digest':
            from .scheduling import validate_subscription
            time = validate_subscription({'times': [text.strip()]})['times'][0]
            self.store.set_setting(f'market_digest:{actor}', time)
            self.bot.market_controls.pending(actor, None)
            self.bot.market_controls.alerts(client, actor, '')
        else:
            raise ValueError('Неизвестная форма ввода')
        return True
