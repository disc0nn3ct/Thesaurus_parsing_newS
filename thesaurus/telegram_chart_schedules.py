"""Button-based personal chart deliveries, independent of report subscriptions."""
import json

from . import chart_schedules
from .charts import validate_options
from .scheduling import validate_subscription
from .telegram_charts import LABELS, PRESETS, keyboard
from .telegram_data_jobs import require_actor
from .telegram_navigation import checked, screen

DAYS = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']


class ChartScheduleControls:
    def __init__(self, bot):
        self.bot, self.store, self.reply = bot, bot.store, bot._reply

    def menu(self, client, actor, section=None):
        require_actor(self.store, actor)
        if section is None:
            from .subscription_access import allowed
            screen('menu:graphs', 'menu:settings')
            self.reply(client, actor, 'Графики по расписанию\nВыберите данные. Каждое расписание действует только для вас.',
                       keyboard([[(label, f'cs:{value}:menu')] for value, label in LABELS.items()
                                 if allowed(self.store, actor, value) and not (value in {'trends', 'wordstat'} and self.store.get_setting('search_provider') == 'disabled')
                                 and (value != 'wordstat' or self.store.get_setting('search_provider') != 'google_trends')]))
            return
        from .subscription_access import require
        require(self.store, actor, section)
        config = self.store.get_setting(chart_schedules.key(actor, section))
        if config is None:
            config = chart_schedules.defaults(actor, section, self.bot.chart_controls.options(actor, section))
            self.store.set_setting(chart_schedules.key(actor, section), config)
        options = config['options']
        from .charts import selections
        from .forecast_horizons import from_options, label
        title = dict(selections(self.store, actor, section)).get(options.get('identity'), 'Все мои показатели')
        if options.get('series_ids'):
            title = ', '.join(dict(selections(self.store, actor, section)).get(key, key) for key in options['series_ids'])
        frequency = 'weekly' if section in {'wordstat', 'trends'} else 'daily'
        if section in {'market', 'trends'} and options.get('identity'):
            meta = self.store.one('SELECT metadata FROM market_series WHERE id=?', (options['identity'],))
            if meta:
                frequency = json.loads(meta['metadata'])['frequency']
        screen(f'cs:{section}:menu', f'ch:menu:{section}')
        synchronized = validate_options(self.bot.chart_controls.options(actor, section)) == validate_options(options)
        period = (f"{options['date_from']} — {options['date_to']} (фиксированные даты)" if options['period'] == 'custom'
                  else PRESETS[options['period']]+' до дня отправки')
        text = (f"График по расписанию · {LABELS[section]}\nОтправка: {'включена' if config['enabled'] else 'выключена'}\n"
                f"Показатель: {title}\nПериод: {period}\n"
                f"Прогноз: {label(from_options(options, frequency))}\n"
                f"Дни: {', '.join(DAYS[day] for day in config['weekdays'])}\nВремя: {', '.join(config['times'])} · {config['timezone']}\n"
                f"Данные: {'догрузить недостающие перед отправкой' if config.get('refresh') else 'из своей базы'}\n\n"
                'Настройте показатель, период и прогноз на экране графика; затем нажмите «Сохранить текущий график». '
                'Настройка сама не включает отправку. Скользящие периоды отсчитываются до дня каждой отправки.')
        if not synchronized:
            text += '\n\nЕсть изменения на экране графика, ещё не сохранённые в расписание. Нажмите «Сохранить текущий график», чтобы применять их к будущим отправкам.'
        if options['period'] == 'custom':
            text += '\nДля включения расписания нужен скользящий период. Выберите его на экране графика и сохраните в расписание.'
        self.reply(client, actor, text, keyboard([
            [('Выключить отправку' if config['enabled'] else 'Включить отправку', f"cs:{section}:{'off' if config['enabled'] else 'on'}")],
            [('Настроить график', f'ch:menu:{section}')], [(checked('Сохранить текущий график', synchronized), f'cs:{section}:apply')],
            [(checked(day, i in config['weekdays']), f'cs:{section}:day:{i}') for i, day in enumerate(DAYS[:4])],
            [(checked(day, i in config['weekdays']), f'cs:{section}:day:{i}') for i, day in enumerate(DAYS[4:], 4)],
            [('Время отправки', f'cs:{section}:times')],
            [('Использовать только базу' if config.get('refresh') else 'Догружать новые данные', f'cs:{section}:refresh')]
            if section != 'trends' else [('Обновление Google настраивает администратор', 'gt:menu')],
            [('Все расписания графиков', 'menu:graphs')]]))

    def callback(self, client, actor, data):
        require_actor(self.store, actor)
        args = data.split(':')
        if len(args) not in {3, 4} or args[0] != 'cs':
            raise ValueError('Неизвестное расписание')
        section, action = args[1:3]
        setting_key = chart_schedules.key(actor, section)
        config = self.store.get_setting(setting_key) or chart_schedules.defaults(actor, section, self.bot.chart_controls.options(actor, section))
        changes = {}
        if action == 'times' and len(args) == 3:
            screen(data, f'cs:{section}:menu')
            self.bot.market_controls.pending(actor, 'chartdelivery:' + section)
            self.reply(client, actor, 'Введите время HH:MM по Москве, например 15:30. Можно несколько через запятую.')
            return
        if action in {'on', 'off'} and len(args) == 3:
            changes['enabled'] = action == 'on'
        elif action == 'apply' and len(args) == 3:
            changes['options'] = validate_options(self.bot.chart_controls.options(actor, section))
        elif action == 'day' and len(args) == 4 and args[3] in {str(i) for i in range(7)}:
            days = set(config['weekdays'])
            days.symmetric_difference_update({int(args[3])})
            if not days:
                raise ValueError('Оставьте хотя бы один день; для остановки выключите отправку')
            changes['weekdays'] = sorted(days)
        elif action == 'refresh' and len(args) == 3 and section != 'trends':
            changes['refresh'] = not config.get('refresh', False)
        elif action != 'menu' or len(args) != 3:
            raise ValueError('Неизвестная настройка')
        if changes:
            chart_schedules.save(self.store, actor, section, config, **changes)
        self.menu(client, actor, section)

    def input(self, client, actor, text):
        pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        if not pending or not (pending['pending'] or '').startswith('chartdelivery:') or text.startswith('/'):
            return False
        require_actor(self.store, actor)
        section = pending['pending'].split(':')[1]
        config = self.store.get_setting(chart_schedules.key(actor, section))
        changes = validate_subscription({'times': text.strip().split(',')})
        chart_schedules.save(self.store, actor, section, config, **changes)
        self.bot.market_controls.pending(actor, None)
        self.menu(client, actor, section)
        return True
