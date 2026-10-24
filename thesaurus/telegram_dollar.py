"""Personal travel dollar tool; callback parameters never contain another actor's ID."""
import json
from datetime import date, timedelta
from uuid import uuid4

from . import dollar
from .store import utcnow
from .telegram_charts import keyboard
from .telegram_data_jobs import require_actor
from .telegram_navigation import checked, screen

ADVANCED_FIELDS = ('fee_pct', 'fee_fixed', 'rub_yield', 'risk')
HISTORY_FIELDS = ('history_start', 'history_end')
DAY_PRESETS = (7, 14, 30, 60, 90)


def form_parent(field):
    if field in HISTORY_FIELDS:
        return 'usd:period'
    return 'usd:horizon' if field in {'analysis_months', 'forecast_days', 'forecast_date'} else 'usd:advanced' if field in ADVANCED_FIELDS else 'usd:dates' if field == 'trip' else 'usd:menu'


def years_before(stamp, years):
    """Calendar-year presets keep February 29 in a valid preceding year."""
    try:
        return stamp.replace(year=stamp.year-years)
    except ValueError:
        return stamp.replace(year=stamp.year-years, day=28)


def historical(cfg):
    return bool(cfg.get('history_end') and date.fromisoformat(cfg['history_end']) < dollar.today_moscow(utcnow()))


class DollarControls:
    def __init__(self, bot):
        self.bot, self.store, self.reply = bot, bot.store, bot._reply

    def config(self, actor):
        from .subscription_access import require
        require(self.store, actor, 'market')
        from .feature_access import require as require_feature
        require_feature(self.store, actor, 'dollar')
        return {**dollar.DEFAULTS, **self.store.get_setting(f'dollar:{actor}', {})}

    def menu(self, client, actor, notice='', advanced=False):
        require_actor(self.store, actor)
        screen('usd:advanced' if advanced else 'usd:menu', 'usd:menu' if advanced else 'pj:market')
        cfg = self.config(actor)
        purchase = cfg.get('mode') == 'purchase'
        rows = [] if advanced else [[(checked('Обзор рынка', not purchase), 'usd:mode:overview'),
                 (checked('Моя покупка', purchase), 'usd:mode:purchase')]]
        fields = ADVANCED_FIELDS if advanced else ('trip', 'quote', 'amount', 'owned') if purchase else ()
        for field in fields:
            title = dollar.FIELDS[field][0]
            value = cfg[field]
            display = 'не задан' if value is None else f'{value:g}' if isinstance(value, (float, int)) else value
            rows.append([(f'{"✓" if value is not None else "○"} {title}: {display}', 'usd:dates' if field == 'trip' else 'usd:set:' + field)])
        if advanced:
            self.reply(client, actor, (notice+'\n\n' if notice else '')+'Моя покупка · комиссии и риск\n'
                       'Выберите параметр и введите значение. Изменения сохраняются сразу только для вас. '
                       'Затем вернитесь к покупке для сравнения затрат.', keyboard(rows))
            return
        if purchase:
            rows += [[('Комиссии, доходность и риск', 'usd:advanced')]]
        needed = 'Указать срок покупки' if not cfg['trip'] else 'Указать цену банка' if cfg['quote'] is None else 'Сравнить дни покупки'
        from .dollar_periods import history_label, horizon_label
        action = 'Построить исторический обзор' if historical(cfg) else needed if purchase else 'Показать обзор и прогноз'
        rows += [[(action, 'usd:run')],
                 [('Курс на дату: '+(date.fromisoformat(cfg['forecast_date']).strftime('%d.%m.%Y') if cfg.get('forecast_date') else 'выбрать'), 'usd:set:forecast_date')],
                 [('История для расчёта: '+history_label(cfg), 'usd:period')],
                 [('Срок прогноза: '+horizon_label(cfg), 'usd:horizon')],
                 [('Модель и окно обучения', 'usd:model')], [('Методика и ограничения', 'usd:help')]]
        if self.store.authorize(actor, admin=True):
            rows.append([('Обновить USD · только недостающие данные', 'usd:update')])
            rows.append([('Обновить все данные для USD', 'usd:update_all')])
        rows.append([('Источники и свежесть данных', 'usd:sources')])
        if self.store.one("SELECT id FROM jobs WHERE kind IN ('dollar_analysis','dollar_collect') AND state IN ('pending','running') AND json_extract(payload,'$.actor_id')=? LIMIT 1", (actor,)):
            rows.append([('Проверить выполнение задания', 'usd:status')])
        instructions = ('Настройте срок и цену продажи USD банком вам; затем сравните затраты по дням. '
                        'Комиссии и риск — в дополнительных настройках.') if purchase else (
                        'Выберите «Курс на дату» для оценки на конкретный день либо срок прогноза: дни, месяцы и годы. Дата поездки и цена банка не обязательны. '
                        'Для расчёта покупки в рублях выберите «Моя покупка».')
        if historical(cfg):
            instructions += '\nВыбрана прошлая дата: расчёт от неё ретроспективный. Совет о покупке сегодня отключён.'
        self.reply(client, actor, (notice+'\n\n' if notice else '') + 'Доллар для поездки\n'
                   +instructions+'\nНастройки личные. Расчёт использует БД; обновление отдельно.', keyboard(rows))

    def prompt(self, client, actor, field):
        self.config(actor)
        self.bot.market_controls.pending(actor, 'dollar:' + field)
        screen('usd:set:'+field, form_parent(field))
        self.reply(client, actor, dollar.FIELDS[field][1])

    def period(self, client, actor, notice=''):
        from .dollar_periods import history_label
        cfg = self.config(actor)
        screen('usd:period', 'usd:menu')
        end = date.fromisoformat(cfg['history_end']) if cfg.get('history_end') else dollar.today_moscow(utcnow())
        rows = [[('Начало: '+(cfg.get('history_start') or 'вся доступная история'), 'usd:set:history_start')],
                [('Конец: '+(cfg.get('history_end') or 'сегодня · обновляется'), 'usd:set:history_end')],
                [(checked('С самого начала', not cfg.get('history_start')), 'usd:period:all'),
                 (checked('По сегодня', not cfg.get('history_end')), 'usd:period:today')]]
        for years, title in ((1, '1 год истории'), (3, '3 года истории'), (5, '5 лет истории')):
            start = years_before(end, years).isoformat()
            rows.append([(checked(title, cfg.get('history_start') == start), f'usd:period:years:{years}')])
        rows.append([('Сбросить: вся история → сегодня', 'usd:period:reset')])
        rows.append([('Далее: срок прогноза', 'usd:horizon')])
        text = ('История для расчёта USD/RUB\nСейчас: '+history_label(cfg)+'\n\n'
                'Начало и конец ограничивают данные для анализа и обучения. «По сегодня» сдвигает конец '
                'при каждом расчёте; выбранная дата начала сохраняется. Доступны только загруженные наблюдения.\n'
                'При конце в прошлом прогноз строится от этой даты; совет о покупке сегодня отключён. '
                'Срок прогноза задаётся отдельно. Заданные даты имеют приоритет над окном модели: '
                'обучение использует весь выбранный интервал. Сброс обеих дат возвращает настройку окна модели. '
                'Настройки сохраняются сразу только для вас.')
        self.reply(client, actor, (notice+'\n\n' if notice else '')+text, keyboard(rows))

    def horizon(self, client, actor, notice=''):
        from .dollar_long import METHODS, PRESETS
        from .dollar_periods import horizon_label
        require_actor(self.store, actor)
        screen('usd:horizon', 'usd:menu')
        cfg = self.config(actor)
        rows = [[(checked(f'{days} дн.', not cfg.get('forecast_date') and cfg['analysis_months'] == 0 and cfg['forecast_days'] == days), f'usd:days:{days}')
                 for days in DAY_PRESETS[:3]],
                [(checked(f'{days} дн.', not cfg.get('forecast_date') and cfg['analysis_months'] == 0 and cfg['forecast_days'] == days), f'usd:days:{days}')
                 for days in DAY_PRESETS[3:]]]
        rows.insert(0, [(checked('Курс на дату: '+(date.fromisoformat(cfg['forecast_date']).strftime('%d.%m.%Y') if cfg.get('forecast_date') else 'выбрать'), bool(cfg.get('forecast_date'))), 'usd:set:forecast_date')])
        rows.append([('Свой срок: 1–90 дней', 'usd:set:forecast_days')])
        rows.extend([(checked(title, cfg['analysis_months'] == value), f'usd:horizon:{value}')]
                    for value, title in PRESETS.items() if value)
        rows.append([('Свой срок в месяцах', 'usd:set:analysis_months')])
        if cfg.get('forecast_date'):
            rows.append([('Модель на выбранную дату', 'usd:model')])
        else:
            rows.append([('Метод: '+METHODS[cfg['long_method']], 'usd:longmethods')])
            rows.append([(checked('Сценарии и частоты истории', cfg['distribution']), 'usd:distribution')])
        rows.append([('Изменить историю для расчёта', 'usd:period')])
        rows.append([('Построить обзор и прогноз', 'usd:longrun')])
        method_help = ('Для конкретной даты используется непосредственная регрессия с лагами и проверкой против последнего курса. '
                       'Показатели и окно обучения выбираются в «Модель на выбранную дату». Месячные сценарии здесь не используются.'
                       if cfg.get('forecast_date') else
                       'Выбор метода ограничивает кандидатов; «Авто» сравнивает все. Окно обучения — в меню модели.\n'
                       '«Сценарии и частоты истории» добавляют медиану, исторический разброс и условные доли '
                       'роста/снижения >2%. Это отдельная оценка при повторении прошлого; она не отменяет '
                       'результат проверки основной модели.')
        self.reply(client, actor, (notice+'\n\n' if notice else '')+'Срок прогноза · USD/RUB\n'
                   'Срок анализа USD/RUB: '+horizon_label(cfg)+'\n'
                   'Выберите конкретную дату до 5 лет вперёд, срок до 90 дней либо до 60 месяцев. '
                   'Дата сохраняется фиксированной и не сдвигается при следующем расчёте; для неё рассчитывается один прогноз непосредственно на этот день. '
                   'Если дата уже прошла, выберите новую. Дни отсчитываются от конца выбранной истории, '
                   'месяцы — от последнего полного месяца в ней. Дата поездки не меняется. Настройка сохраняется только для вас. '
                   'На месяцы и годы используются последние курсы полных месяцев, а не ежедневная линия. '
                   'Каждый горизонт проверяется отдельно на прошлых периодах. Сложный метод применяется только '
                   'при улучшении MAE на 5% и от 10 внешних проверок. Иначе — последний курс как базовый вариант; '
                   'при нехватке проверок — условный сценарий без диапазона.\n'+method_help, keyboard(rows))

    def long_methods(self, client, actor):
        from .dollar_long import METHODS
        require_actor(self.store, actor)
        screen('usd:longmethods', 'usd:horizon')
        cfg = self.config(actor)
        rows = [[(checked(title, cfg['long_method'] == value), 'usd:longmethod:'+value)] for value, title in METHODS.items()]
        self.reply(client, actor, 'Метод долгосрочного анализа\nАвто сравнивает все методы на прошлом. '
                   'Отдельный метод ограничивает кандидатов, но сохраняет проверку против последнего курса. '
                   'Если преимущества нет, бот оставит базовый вариант и объяснит причину. '
                   'Краткосрочная модель настраивается отдельно.', keyboard(rows))

    def dates(self, client, actor, notice=''):
        cfg = self.config(actor)
        screen('usd:dates', 'usd:menu')
        rows = []
        for days, title in ((0, 'Сегодня'), (1, 'Завтра'), (3, 'Через 3 дня'), (7, 'Через неделю'), (30, 'Через месяц')):
            stamp = dollar.today_moscow(utcnow())+timedelta(days=days)
            rows.append([(checked(title+' · '+stamp.isoformat(), cfg['trip'] == stamp.isoformat()), f'usd:date:{days}')])
        rows.append([('Своя дата', 'usd:set:trip')])
        self.reply(client, actor, (notice+'\n\n' if notice else '')+'Когда валюта должна быть куплена? Это ваш крайний срок, а не желаемая дата минимального курса.', keyboard(rows))

    def sources(self, client, actor):
        from .dollar_sources import PACK
        selected = self.store.get_setting('dollar_source_pack', PACK)
        screen('usd:sources', 'usd:menu')
        rows, lines = [], ['Данные для анализа доллара\nВыбор определяет дополнительный сбор; общие данные других проектов сохраняются.']
        for identity in PACK:
            row = self.store.one('SELECT rowid token,* FROM market_series WHERE id=?', (identity,))
            if not row:
                continue
            count = self.store.one('SELECT COUNT(*) n,MAX(date) latest FROM market_observations WHERE value IS NOT NULL AND (series_id=? OR series_id IN (SELECT id FROM market_series WHERE dataset=?))', (identity, identity))
            status = {'not_loaded': 'не загружен', 'ok': 'готов', 'partial': 'есть ошибки', 'cached': 'сохранён', 'reference': 'справочник'}.get(row['status'], 'проверьте источник')
            lines.append(f"{row['title']}: {count['n']} наблюдений, последняя дата {count['latest'] or 'нет'}; состояние {status}.")
            if row.get('error'):
                lines.append(row['error'])
            if self.store.authorize(actor, admin=True):
                rows.append([(checked(row['title'][:48], identity in selected), f"usd:source:{row['token']}")])
        if self.store.authorize(actor, admin=True):
            rows.append([('Загрузить выбранные источники', 'usd:update_all')])
        self.reply(client, actor, '\n\n'.join(lines), keyboard(rows))

    def models(self, client, actor):
        from .dollar_models import ENGINES, HISTORY
        require_actor(self.store, actor)
        screen('usd:model', 'usd:menu')
        cfg = self.config(actor)
        bounded = bool(cfg.get('history_start') or cfg.get('history_end'))
        rows = ([] if bounded else [[(checked(title, cfg['history_mode'] == key), 'usd:model:history:'+key)]
                                    for key, title in HISTORY.items()])
        rows += [[(checked(title, cfg['forecast_engine'] == key), 'usd:model:engine:'+key)] for key, title in ENGINES.items()]
        if self.store.authorize(actor, admin=True):
            rows.append([('Дозагрузить всю историю USD и показателей', 'usd:update_history')])
        rows.append([('К расчёту', 'usd:menu')])
        rows.insert(0, [('Границы истории для расчёта', 'usd:period')])
        notice = ('Заданные даты имеют приоритет: обучение использует весь выбранный интервал. '
                  'Чтобы снова выбирать окно модели, сбросьте обе границы истории.\n\n') if bounded else ''
        self.reply(client, actor, 'Модель и окно обучения\n\n'+notice+'Все методы работают внутри выбранных границ истории. '
                   'Авто сравнивает доступную историю, последние 3 года и '
                   'период после 13.06.2024. В режиме «Вся история» выбранный интервал обучения не сокращается. '
                   'Для старых рублей до 1998 года применяется пересчёт 1000:1.\n\n'
                   'Ridge использует изменения USD и лаги ежедневных показателей. Корреляции, признаки и '
                   'параметры выбираются внутри обучения; сложная модель допускается при улучшении MAE на 5%. '
                   'Месячные/квартальные ряды остаются в контексте. Проверка ретроспективная: исторические '
                   'версии публикаций неизвестны, задержки их выхода предполагаются. Настройки личные.', keyboard(rows))

    def callback(self, client, actor, data, request_id=None):
        require_actor(self.store, actor)
        self.config(actor)
        if data == 'usd:menu':
            self.menu(client, actor)
        elif data == 'usd:period':
            self.period(client, actor)
        elif data.startswith('usd:period:'):
            cfg = self.config(actor)
            action = data.removeprefix('usd:period:')
            now = utcnow()
            if action in {'all', 'reset'}:
                cfg = dollar.change_settings(cfg, 'history_start', 'all', now=now)
            if action in {'today', 'reset'}:
                cfg = dollar.change_settings(cfg, 'history_end', 'today', now=now)
            elif action.startswith('years:') and action[6:] in {'1', '3', '5'}:
                end = date.fromisoformat(cfg['history_end']) if cfg.get('history_end') else dollar.today_moscow(now)
                cfg = dollar.change_settings(cfg, 'history_start', years_before(end, int(action[6:])).isoformat(), now=now)
            elif action != 'all':
                raise ValueError('Выберите период кнопками')
            self.store.set_setting(f'dollar:{actor}', cfg)
            self.period(client, actor, '✓ Границы истории сохранены.')
        elif data == 'usd:horizon':
            self.horizon(client, actor)
        elif data.startswith('usd:days:'):
            value = data.rsplit(':', 1)[-1]
            if not value.isascii() or not value.isdigit() or int(value) not in DAY_PRESETS:
                raise ValueError('Выберите срок кнопками')
            cfg = dollar.change_settings(self.config(actor), 'forecast_days', value, now=utcnow())
            self.store.set_setting(f'dollar:{actor}', {**cfg, 'analysis_months': 0})
            self.horizon(client, actor, '✓ Срок прогноза сохранён.')
        elif data == 'usd:longmethods':
            self.long_methods(client, actor)
        elif data == 'usd:distribution':
            self.store.set_setting(f'dollar:{actor}', {**self.config(actor), 'distribution': not self.config(actor)['distribution']})
            self.horizon(client, actor, '✓ Отображение сценариев сохранено.')
        elif data == 'usd:longrun':
            self.store.set_setting(f'dollar:{actor}', {**self.config(actor), 'mode': 'overview'})
            self.callback(client, actor, 'usd:run', request_id=request_id)
        elif data.startswith('usd:horizon:'):
            from .dollar_long import PRESETS
            value = data.rsplit(':', 1)[-1]
            if not value.isascii() or not value.isdigit() or int(value) not in PRESETS:
                raise ValueError('Выберите срок кнопками')
            cfg = {**self.config(actor), 'analysis_months': int(value), 'forecast_date': None}
            if int(value) == 0:
                cfg['forecast_days'] = 7
            self.store.set_setting(f'dollar:{actor}', cfg)
            self.horizon(client, actor, '✓ Срок сохранён. Нажмите «Построить обзор и прогноз».')
        elif data.startswith('usd:longmethod:'):
            from .dollar_long import METHODS
            choice = data.rsplit(':', 1)[-1]
            if choice not in METHODS:
                raise ValueError('Выберите метод кнопками')
            self.store.set_setting(f'dollar:{actor}', {**self.config(actor), 'long_method': choice})
            self.horizon(client, actor, '✓ Метод сохранён.')
        elif data == 'usd:model':
            self.models(client, actor)
        elif data.startswith('usd:model:'):
            from .dollar_models import ENGINES, HISTORY
            _, _, setting, choice = data.split(':')
            field, allowed = ('history_mode', HISTORY) if setting == 'history' else ('forecast_engine', ENGINES) if setting == 'engine' else (None, {})
            if choice not in allowed:
                raise ValueError('Выберите настройку кнопками')
            self.store.set_setting(f'dollar:{actor}', {**self.config(actor), field: choice})
            self.models(client, actor)
        elif data == 'usd:advanced':
            self.menu(client, actor, advanced=True)
        elif data == 'usd:status':
            screen(data, 'usd:menu')
            self.bot._status(client, actor)
        elif data == 'usd:dates':
            self.dates(client, actor)
        elif data.startswith('usd:date:'):
            days = int(data.rsplit(':', 1)[-1])
            if days not in {0, 1, 3, 7, 30}:
                raise ValueError('Выберите срок кнопками')
            now = utcnow()
            cfg = dollar.change_settings(self.config(actor), 'trip', (dollar.today_moscow(now)+timedelta(days=days)).isoformat(), now=now)
            self.store.set_setting(f'dollar:{actor}', cfg)
            if cfg['quote'] is None:
                self.prompt(client, actor, 'quote')
            else:
                self.menu(client, actor, '✓ Срок сохранён. Нажмите «Сравнить дни покупки».')
        elif data.startswith('usd:mode:'):
            mode = data.rsplit(':', 1)[-1]
            if mode not in {'overview', 'purchase'}:
                raise ValueError('Выберите режим кнопками')
            self.store.set_setting(f'dollar:{actor}', {**self.config(actor), 'mode': mode})
            self.menu(client, actor)
        elif data == 'usd:sources':
            self.sources(client, actor)
        elif data.startswith('usd:source:'):
            from .dollar_sources import PACK
            require_actor(self.store, actor, admin=True)
            token = data.rsplit(':', 1)[-1]
            if not token.isdigit():
                raise ValueError('Неверный источник')
            row = self.store.one('SELECT id FROM market_series WHERE rowid=?', (int(token),))
            if not row or row['id'] not in PACK:
                raise ValueError('Источник недоступен в этом наборе')
            selected = self.store.get_setting('dollar_source_pack', PACK)
            selected = [i for i in selected if i != row['id']] if row['id'] in selected else selected+[row['id']]
            self.store.set_setting('dollar_source_pack', selected)
            self.sources(client, actor)
        elif data == 'usd:help':
            screen(data, 'usd:menu')
            self.reply(client, actor, 'Методика\nИзменения курса за 30/90/365 дней, текущий уровень в годовой истории, '
                       'волатильность, локальная проверяемая модель и доступный экономический контекст. '
                       '«История для расчёта» задаёт начало и конец исходных данных. Конец «сегодня» обновляется автоматически; '
                       'прошлая дата включает ретроспективный расчёт без совета о покупке сегодня. '
                       'Окна обучения выбираются только внутри этих границ. '
                       'Лаги показателей и ridge выбираются внутри обучающих окон; сложная модель нужна только при улучшении MAE на 5%. '
                       'Диапазон 80% — только при 10 прошлых ошибках.\n\n'
                       'Срок прогноза: 1–90 дней либо 1–60 месяцев, отдельно от истории и поездки. Дальние оценки используют '
                       'последние курсы полных месяцев и проверки на таком же горизонте; будущие факторы не подставляются. '
                       'При нехватке проверок диапазон не рисуется. Полная история включает разные валютные режимы; '
                       'совпадение старых закономерностей с будущим не гарантируется.\n\n'
                       'Сценарии ±5/10% не прогнозируют вероятность. Результат «сейчас/частями/позже» '
                       'объясняет риск бюджета и срока, не гарантирует лучший курс. '
                       'Учитывайте наличную/безналичную цену и возможную двойную конвертацию в стране поездки. '
                       'При устаревшем курсе банка (>24 ч) или неполной истории текущий совет отключён.\n\n' + dollar.SOURCE)
        elif data.startswith('usd:set:'):
            field = data.split(':')[-1]
            if field not in dollar.FIELDS:
                raise ValueError('Неизвестная настройка')
            self.prompt(client, actor, field)
        elif data in {'usd:run', 'usd:update', 'usd:update_all', 'usd:update_history'}:
            update = data in {'usd:update', 'usd:update_all', 'usd:update_history'}
            require_actor(self.store, actor, admin=update)
            payload = {'actor_id': actor, 'config': self.config(actor)}
            if not update and payload['config'].get('mode') == 'purchase' and not historical(payload['config']):
                if not payload['config']['trip']:
                    self.dates(client, actor)
                    return
                if payload['config']['quote'] is None:
                    self.prompt(client, actor, 'quote')
                    return
            if data in {'usd:update_all', 'usd:update_history'}:
                from .dollar_sources import PACK
                selected = self.store.get_setting('dollar_source_pack', PACK)
                if not selected or not set(selected).issubset(PACK):
                    raise ValueError('Выберите хотя бы один источник')
                payload.update(scope='context', selection=selected)
                if data == 'usd:update_history':
                    payload['history'] = 'full'
            kind = 'dollar_collect' if update else 'dollar_analysis'
            active = self.store.query("SELECT id,payload FROM jobs WHERE kind=? AND state IN ('pending','running') AND json_extract(payload,'$.actor_id')=?", (kind, actor))
            existing = next((job for job in active if json.loads(job['payload']) == payload), None)
            if existing:
                self.menu(client, actor, f"Задание №{existing['id']} уже в очереди или выполняется. Результат придёт сюда; повторный запрос не создан.")
                return
            # Retransmissions reuse the job; after completion a new press recomputes.
            key = f'dollar:{actor}:{data}:{request_id or uuid4().hex}'
            job = self.store.enqueue(kind, payload, dedupe_key=key)
            self.menu(client, actor, f'Задание №{job} в очереди. Результат придёт сюда по готовности.')
        else:
            raise ValueError('Неизвестное действие')

    def input(self, client, actor, text):
        require_actor(self.store, actor)
        state = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        if not state or not (state['pending'] or '').startswith('dollar:') or text.startswith('/'):
            return False
        field = state['pending'].split(':')[-1]
        screen('usd:set:' + field, form_parent(field))
        cfg = dollar.change_settings(self.config(actor), field, text, now=utcnow())
        self.store.set_setting(f'dollar:{actor}', cfg)
        self.bot.market_controls.pending(actor, None)
        if field == 'trip' and cfg.get('mode') == 'purchase' and cfg['quote'] is None:
            self.prompt(client, actor, 'quote')
        elif field in HISTORY_FIELDS:
            self.period(client, actor, '✓ Границы истории сохранены.')
        elif field in {'analysis_months', 'forecast_days', 'forecast_date'}:
            if field == 'forecast_days':
                self.store.set_setting(f'dollar:{actor}', {**cfg, 'analysis_months': 0})
            self.horizon(client, actor, '✓ Срок сохранён.')
        else:
            self.menu(client, actor, '✓ Настройка сохранена.', advanced=field in ADVANCED_FIELDS)
        return True
