"""Discover all CBR bases, connect shared sources, then use personal analytics."""
import json
from uuid import uuid4

from .source_catalog import GROUPS, browse, group_for, parent, type_label
from .telegram_admin import positive_id
from .telegram_charts import keyboard
from .telegram_data_jobs import require_actor
from .telegram_navigation import screen


def descendants(store, identity):
    rows = store.query('SELECT rowid token,* FROM market_series WHERE source=?', ('cbr',))
    entries = {row['id']: json.loads(row['metadata']) for row in rows}
    found = {identity}
    while True:
        added = {key for key, meta in entries.items() if parent(meta, entries) in found}
        if added.issubset(found):
            break
        found.update(added)
    return [row for row in rows if row['id'] in found]


class CbrControls:
    def __init__(self, bot):
        self.bot, self.store, self.reply = bot, bot.store, bot._reply

    def _row(self, token):
        row = self.store.one('SELECT rowid token,* FROM market_series WHERE rowid=? AND source=?',
                             (positive_id(token), 'cbr'))
        if not row or row['status'] == 'archived':
            raise ValueError('Набор недоступен. Откройте каталог заново.')
        return row

    def menu(self, client, actor):
        require_actor(self.store, actor)
        from .subscription_access import require
        require(self.store, actor, 'market')
        screen('cb:menu', 'menu:data')
        loaded = self.store.one("SELECT COUNT(DISTINCT series_id) n FROM market_observations WHERE value IS NOT NULL AND series_id IN (SELECT id FROM market_series WHERE source='cbr')")['n']
        self.reply(client, actor, f'Базы Банка России · 29 разделов\nЗагружено числовых рядов: {loaded}.\nВыберите группу, затем базу и показатель. Администратор подключает источники кнопкой «Подключить и загрузить». Графики, прогнозы и сравнения используют сохранённую историю. Расписания включаются отдельно.',
                   keyboard([[('С чего начать', 'cb:guide')], [('Основные базы · быстрый выбор', 'cb:basics')],
                             [('Уже загруженные показатели', 'cb:loaded:0')]] +
                            [[(title, f'cb:group:{key}:0')] for key, (title, roots) in GROUPS.items() if roots] + [[('Найти базу или показатель', 'cb:search')]]))

    def guide(self, client, actor):
        require_actor(self.store, actor)
        screen('cb:guide', 'cb:menu')
        text = ('Показатели ЦБ · с чего начать\n\n'
                'Цены и ставки: инфляция, ключевая ставка, RUONIA.\n'
                'Валюты: официальный курс и его история.\n'
                'Ликвидность: средства банков, дефицит/профицит.\n'
                'Внешний сектор: международные резервы, переводы и услуги.\n\n'
                '1. Откройте базу. В «Наборах и показателях» выберите конкретный ряд: валюту, срок, ставку или объём.\n'
                '2. В карточке проверьте единицы и даты истории. Если данных нет, администратор нажимает «Подключить и загрузить». Сбор берёт недостающие данные.\n'
                '3. Нажмите «График и прогноз», выберите период и срок прогноза, затем «Построить график». Для начала удобен один показатель за год.\n\n'
                '✓ в каталоге означает подключение общего источника. «Мои показатели» — ваш личный набор для обзоров. Подключение не включает рассылку.\n'
                'Срок ряда (например RUONIA 1/3/6 месяцев), период графика и срок прогноза — три разные настройки.\n'
                'Прогноз доступен при достаточной истории; справочники и опубликованные прогнозы ЦБ не являются фактическими рядами для обучения.')
        self.reply(client, actor, text, keyboard([
            [('Выбрать основную базу', 'cb:basics')],
            [('Посмотреть загруженные ряды', 'cb:loaded:0')],
            [('Мои показатели', 'menu:indicators')]]))

    def basics(self, client, actor):
        require_actor(self.store, actor)
        screen('cb:basics', 'cb:menu')
        buttons = []
        for identity, title in (
            ('cbr.hd_base.infl', 'Инфляция и ключевая ставка'),
            ('cbr.hd_base.ruonia', 'RUONIA · ставки и объём'),
            ('cbr.currency_base', 'Курсы валют'),
            ('cbr.hd_base.bliquidity', 'Дефицит / профицит ликвидности'),
            ('cbr.hd_base.mrrf', 'Международные резервы'),
            ('cbr.hd_base.zcyc_params', 'Кривая доходности облигаций'),
        ):
            row = self.store.one("SELECT rowid token FROM market_series WHERE id=? AND source='cbr'", (identity,))
            if row:
                self.store.set_setting(f'cbr_return:{actor}:{row["token"]}', 'cb:basics')
                buttons.append([(title, f'cb:card:{row["token"]}')])
        buttons.append([('С чего начать', 'cb:guide')])
        self.reply(client, actor, 'Основные базы ЦБ\nНачните с одной задачи и одного ряда. Каждая база содержит отдельные показатели; подключать весь каталог сразу не требуется. Остальные базы доступны в полном каталоге.', keyboard(buttons))

    def listing(self, client, actor, *, group=None, token=None, page=0, search='', loaded=False):
        row = self._row(token) if token is not None else None
        rows, page, total, entries = browse(self.store, group=group, parent_id=row['id'] if row else None, page=page, search=search, source='cbr', loaded=loaded)
        route = f'cb:children:{token}:{page}' if row else f'cb:group:{group}:{page}'
        back = f'cb:card:{token}' if row else 'cb:menu'
        if search:
            route = f'cb:results:{page}'
        if loaded:
            route = f'cb:loaded:{page}'
        screen(route, back)
        title = 'Уже загруженные показатели ЦБ' if loaded else 'Поиск: ' + search if search else row['title'] if row else GROUPS[group][0]
        lines = [title, f'Страница {page+1} из {max(1, (total+7)//8)}', '✓ — источник подключён; загрузка и наличие данных показаны в карточке.']
        buttons = []
        for item in rows:
            meta = entries[item['id']]
            self.store.set_setting(f'cbr_return:{actor}:{item["token"]}', route)
            lines.append(('✓ ' if item['enabled'] else '○ ') + item['title'] + ' · ' + type_label(meta))
            buttons.append([(('✓ ' if item['enabled'] else '') + item['title'][:55], f'cb:card:{item["token"]}')])
        for number, label in [(page-1, '‹ Предыдущая'), (page+1, 'Следующая ›')]:
            if 0 <= number < (total+7)//8:
                buttons.append([(label, f'cb:loaded:{number}' if loaded else f'cb:results:{number}' if search else f'cb:children:{token}:{number}' if row else f'cb:group:{group}:{number}')])
        if not rows:
            lines.append('Пока нет показателей. Подключите и загрузите базу либо уточните поиск.')
        self.reply(client, actor, '\n'.join(lines), keyboard(buttons))

    def card(self, client, actor, token):
        row = self._row(token)
        meta = json.loads(row['metadata'])
        rows = descendants(self.store, row['id'])
        entries = {item['id']: json.loads(item['metadata']) for item in self.store.query('SELECT id,metadata FROM market_series')}
        ancestor = parent(meta, entries)
        parent_row = self.store.one('SELECT rowid token FROM market_series WHERE id=?', (ancestor,)) if ancestor else None
        back = f'cb:card:{parent_row["token"]}' if parent_row else f'cb:group:{group_for(row["id"], entries)}:0'
        back = self.store.get_setting(f'cbr_return:{actor}:{token}', back)
        screen(f'cb:card:{token}', back)
        coverage = self.store.one('SELECT COUNT(*) n,MIN(date) first,MAX(date) last FROM market_observations WHERE series_id=? AND value IS NOT NULL', (row['id'],))
        frequency = {'daily': 'ежедневно', 'weekly': 'еженедельно', 'monthly': 'ежемесячно',
                     'quarterly': 'ежеквартально', 'annual': 'ежегодно', 'irregular': 'по событиям'}.get(meta['frequency'], meta['frequency'])
        lines = [row['title'], f"Тип: {type_label(meta)}. Единицы: {meta['unit']}. Частота: {frequency}.",
                 'Источник: ' + meta['url'], 'Подключён' if row['enabled'] else 'Источник не подключён']
        parameters = {**meta.get('default_params', {}), **self.store.get_setting('market_params:' + row['id'], {})}
        for field, choices in meta.get('parameter_choices', {}).items():
            choice = next((item['label'] for item in choices if item['value'] == parameters.get(field)), None)
            if choice:
                lines.append('Выбрана валюта: ' + choice)
        if coverage['n']:
            lines.append(f"История: {coverage['first']} — {coverage['last']}; наблюдений: {coverage['n']}.")
        else:
            lines.append('История этого ряда ещё не загружена.' if meta['kind'] == 'series' else 'Числовые показатели — в списке ниже после загрузки.')
        if row['error']:
            lines.append(row['error'])
        if meta.get('data_type') == 'archive':
            lines.append('Архив: доступен анализ истории; прогноз на сегодня не строится.')
        if meta.get('data_type') == 'official_forecast':
            lines.append('Опубликованный прогноз ЦБ хранится отдельно от фактических данных и не используется как факт для обучения.')
        if meta.get('calendar') == 'events' or meta['frequency'] == 'irregular':
            lines.append('Нерегулярные наблюдения: график и экспорт доступны; обычный временной прогноз и корреляции требуют подходящей частоты.')
        buttons = []
        if len(rows) > 1:
            buttons += [[('Наборы и показатели', f'cb:children:{token}:0')]]
        if self.store.authorize(actor, admin=True):
            buttons += [[('Подключить и загрузить' if not row['enabled'] else 'Загрузить недостающие данные', f'cb:load:{token}')]]
            if meta.get('parameter_choices'):
                buttons += [[('Выбрать валюту', f'cb:params:{token}:0')]]
        elif not row['enabled']:
            lines.append('Подключение выполняет администратор.')
        if row['enabled'] and meta['kind'] == 'series':
            buttons += [[('График истории' if meta.get('data_type') in {'archive', 'official_forecast'} or meta['frequency'] == 'irregular' else 'График и прогноз', f'cb:chart:{token}')]]
            if meta['frequency'] != 'irregular' and meta.get('data_type') != 'official_forecast':
                buttons += [[('Сравнить · корреляция', f'cb:pair:{token}')]]
                if meta.get('data_type') != 'archive':
                    buttons += [[('Настроить уведомление', f'cb:alert:{token}')]]
            buttons += [[('Экспорт CSV / Excel', f'cb:export:{token}')]]
        if meta['kind'] == 'reference':
            lines.append('Справочные документы доступны по ссылке источника. Числовые наборы этого раздела выбираются отдельно.')
        self.reply(client, actor, '\n'.join(lines), keyboard(buttons))

    def input(self, client, actor, text):
        state = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        if not state or state['pending'] != 'cbrsearch' or text.startswith('/'):
            return False
        require_actor(self.store, actor)
        search = text.strip()[:100]
        if not search:
            raise ValueError('Введите название показателя или базы.')
        self.store.set_setting(f'cbr_search:{actor}', search)
        self.bot.market_controls.pending(actor, None)
        self.listing(client, actor, search=search)
        return True

    def _personal(self, actor, row):
        if not row['enabled'] or json.loads(row['metadata'])['kind'] != 'series':
            raise PermissionError('Сначала подключите числовой показатель.')
        selection = self.store.get_setting(f'market_selection:{actor}', [])
        if selection and row['id'] not in selection:
            self.store.set_setting(f'market_selection:{actor}', [item for item in selection if item != '__none__'] + [row['id']])

    def callback(self, client, actor, data):
        from .feature_access import require as require_feature
        require_feature(self.store, actor, 'cbr')
        from .subscription_access import require
        require(self.store, actor, 'market')
        require_actor(self.store, actor)
        args = data.split(':')
        if data == 'cb:menu':
            self.menu(client, actor)
        elif data == 'cb:guide':
            self.guide(client, actor)
        elif data == 'cb:basics':
            self.basics(client, actor)
        elif len(args) == 3 and args[1] == 'loaded' and args[2].isdigit():
            self.listing(client, actor, loaded=True, page=int(args[2]))
        elif data == 'cb:search':
            screen(data, 'cb:menu')
            self.bot.market_controls.pending(actor, 'cbrsearch')
            self.reply(client, actor, 'Введите часть названия, например «резервы», «золото» или «ликвидность». Отмена: /cancel.')
        elif len(args) == 3 and args[1] == 'results' and args[2].isdigit():
            self.listing(client, actor, search=self.store.get_setting(f'cbr_search:{actor}', ''), page=int(args[2]))
        elif len(args) == 4 and args[1] in {'params', 'param'}:
            require_actor(self.store, actor, admin=True)
            row = self._row(args[2])
            choices = json.loads(row['metadata']).get('parameter_choices', {}).get('UniDbQuery.ValID', [])
            if not choices or not args[3].isdigit():
                raise ValueError('Выберите валюту из списка.')
            if args[1] == 'param':
                chosen = next((item for item in choices if item['value'] == args[3]), None)
                if not chosen:
                    raise ValueError('Недоступная валюта')
                current = self.store.get_setting('market_params:' + row['id'], {})
                initial = json.loads(row['metadata']).get('default_params', {})
                if {**initial, **current}.get('UniDbQuery.ValID') != chosen['value']:
                    # Re-selecting a known variant reuses its observations and cache.
                    value = {} if initial.get('UniDbQuery.ValID') == chosen['value'] else {'UniDbQuery.ValID': chosen['value']}
                    self.store.set_setting('market_params:' + row['id'], value)
                self.card(client, actor, args[2])
            else:
                number = min(int(args[3]), (len(choices)-1)//8)
                screen(data, f'cb:card:{args[2]}')
                parameters = {**json.loads(row['metadata']).get('default_params', {}),
                              **self.store.get_setting('market_params:' + row['id'], {})}
                buttons = [[(('✓ ' if parameters.get('UniDbQuery.ValID') == item['value'] else '')+item['label'],
                             f'cb:param:{args[2]}:{item["value"]}')] for item in choices[number*8:(number+1)*8]]
                if number:
                    buttons.append([('‹ Предыдущая', f'cb:params:{args[2]}:{number-1}')])
                if (number+1)*8 < len(choices):
                    buttons.append([('Следующая ›', f'cb:params:{args[2]}:{number+1}')])
                self.reply(client, actor, 'Архив редких валют. Выберите валюту, затем «Подключить и загрузить». Истории разных валют хранятся раздельно.', keyboard(buttons))
        elif len(args) == 4 and args[1] == 'group' and args[2] in GROUPS and args[3].isdigit():
            self.listing(client, actor, group=args[2], page=int(args[3]))
        elif len(args) == 4 and args[1] == 'children' and args[3].isdigit():
            self.listing(client, actor, token=args[2], page=int(args[3]))
        elif len(args) == 3:
            row = self._row(args[2])
            if args[1] == 'card':
                self.card(client, actor, args[2])
            elif args[1] == 'load':
                require_actor(self.store, actor, admin=True)
                targets = [item for item in descendants(self.store, row['id']) if json.loads(item['metadata'])['kind'] in {'dataset', 'series'} and item['status'] != 'archived']
                if not targets:
                    raise ValueError('Это справочник. Выберите числовой набор внутри раздела.')
                for item in targets:
                    self.store.execute('UPDATE market_series SET enabled=1 WHERE id=?', (item['id'],))
                self.store.execute('UPDATE market_series SET enabled=1 WHERE id=?', (row['id'],))
                pending = self.store.one("SELECT id FROM jobs WHERE kind='market_collect' AND state IN ('pending','running') AND json_extract(payload,'$.cbr_root')=? AND json_extract(payload,'$.actor_id')=?", (row['id'], actor))
                identifier = pending['id'] if pending else self.store.enqueue('market_collect', {'actor_id': actor,
                    'selection': [item['id'] for item in targets], 'notify_actor': True, 'cbr_root': row['id']},
                    'cbr-connect:' + uuid4().hex)
                screen(data, f'cb:card:{args[2]}')
                self.reply(client, actor, f"{row['title']}\nЗагрузка в очереди: #{identifier}. Результат придёт отдельным сообщением. Повторное нажатие не создаёт дубликат; сохранённая история повторно не скачивается.",
                           keyboard([[('К показателям базы', f'cb:children:{args[2]}:0')], [('Состояние задания', 'menu:status')]]))
            elif args[1] in {'chart', 'pair', 'alert', 'export'}:
                self._personal(actor, row)
                if args[1] in {'chart', 'export'}:
                    self.bot.dialog_controls.callback(client, actor, 'ui:chartfor:' + args[2])
                    if args[1] == 'export':
                        self.bot.chart_controls.menu(client, actor, 'market', view='export')
                elif args[1] == 'pair':
                    self.bot.dialog_controls.callback(client, actor, 'ui:pair:pickleft:' + args[2])
                else:
                    from .market_alerts import add_rule
                    existing = self.store.one('SELECT id FROM market_rules WHERE user_id=? AND series_id=? AND kind=?', (actor, row['id'], 'deviation'))
                    identifier = existing['id'] if existing else add_rule(self.store, actor, row['id'], 'deviation', 3)
                    self.bot.dialog_controls.rule_card(client, actor, str(identifier))
            else:
                raise ValueError('Неизвестное действие каталога ЦБ')
        else:
            raise ValueError('Неизвестный экран каталога ЦБ')
        return True
