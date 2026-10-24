"""Geographic progress, personal targets and administrator read-only profiles."""
import json

from . import vivino_atlas as atlas
from .telegram_charts import keyboard
from .telegram_navigation import screen


class AtlasControls:
    def __init__(self, controls):
        self.controls, self.store, self.reply = controls, controls.store, controls.reply

    def callback(self, client, actor, data):
        from .feature_access import require
        require(self.store, actor, 'vivino_plan')
        parts = data.split(':')
        if data.startswith('va:users:'):
            return self.users(client, actor, int(parts[2]))
        if len(parts) < 3 or not parts[2].isascii() or not parts[2].isdigit():
            raise ValueError('Неверная команда атласа')
        owner = atlas.subject(self.store, actor, int(parts[2]))
        view = atlas.model(self.store, actor, owner)
        screen(data, 'vi:menu' if owner == actor else 'va:users:0')
        action = parts[1]
        if action == 'view' and len(parts) == 3:
            text = f"Vivino · пользователь {owner}\nПодтверждённых записей: {view['stats']['tasted']}; регионов: {view['stats']['regions']}; сортов: {view['stats']['grapes']}.\nПросмотр администратора; настройки владельца не меняются."
            self.reply(client, actor, text, keyboard([
                [('Географическая карта', f'va:map:{owner}')],
                [('Регионы', f'va:list:{owner}:regions:0:all'), ('Сорта', f'va:list:{owner}:grapes:0:all')],
                [('Вина и отзывы', f'va:wines:{owner}:0')],
                [('Аналитика и что попробовать', f'va:analysis:{owner}')]]))
        elif action == 'map' and len(parts) == 3:
            self.controls.queue(client, actor, owner)
        elif action == 'analysis' and len(parts) == 3:
            self.analysis(client, actor, owner, view)
        elif action == 'list' and len(parts) == 6:
            self.list(client, actor, owner, parts[3], int(parts[4]), parts[5])
        elif action == 'item' and len(parts) == 5:
            self.item(client, actor, owner, parts[3], parts[4])
        elif action in {'plan', 'done'} and len(parts) == 5:
            if owner != actor:
                raise PermissionError('Просмотр администратора не изменяет чужие данные')
            self.target(view, parts[3], parts[4])
            name = 'plan' if action == 'plan' else 'progress'
            values = self.store.get_setting(self.controls.key(actor, name), [])
            token = parts[3] + ':' + parts[4]
            values = [v for v in values if v != token] if token in values else [*values, token]
            atlas.revise(self.store, actor, name, values)
            self.item(client, actor, owner, parts[3], parts[4])
        elif action == 'wines' and len(parts) == 4:
            self.wines(client, actor, owner, int(parts[3]))
        elif action == 'wine' and len(parts) == 4:
            self.wine(client, actor, owner, parts[3])
        elif action in {'tasted', 'editgrapes'} and len(parts) == 4:
            if owner != actor:
                raise PermissionError('Чужие вина доступны только для просмотра')
            row = next((r for r in view['wines'] if r['id'] == parts[3]), None)
            if not row:
                raise ValueError('Вино не найдено')
            if action == 'editgrapes':
                self.controls.bot.market_controls.pending(actor, 'vivino_grapes:' + row['id'])
                screen(data, f"va:wine:{owner}:{row['id']}")
                self.reply(client, actor, 'Введите сорта через ; например Cabernet Sauvignon; Merlot. '
                    'Берите состав с этикетки или проверенной карточки вина. Для неизвестного состава отправьте «неизвестно».')
            else:
                notes = self.store.get_setting(self.controls.key(actor, 'annotations'), {})
                notes.setdefault(row['id'], {})['tasted'] = not row['tasted']
                atlas.revise(self.store, actor, 'annotations', notes)
                self.wine(client, actor, owner, row['id'])
        elif action == 'search' and len(parts) == 4:
            if parts[3] not in {'regions', 'grapes'}:
                raise ValueError('Неверный раздел')
            self.controls.bot.market_controls.pending(actor, f'vivino_search:{owner}:{parts[3]}')
            screen(data, f'va:list:{owner}:{parts[3]}:0:all')
            self.reply(client, actor, 'Введите название или часть названия. «все» очищает поиск.')
        elif action == 'new' and len(parts) == 4:
            if owner != actor or parts[3] not in {'region', 'grape'}:
                raise PermissionError('Добавляйте цели только в собственный план')
            self.controls.bot.market_controls.pending(actor, 'vivino_custom:' + parts[3])
            screen(data, f"va:list:{owner}:{'regions' if parts[3] == 'region' else 'grapes'}:0:all")
            self.reply(client, actor, 'Личная цель · новый регион\nВведите через ; название; код страны; широту; долготу; описание. '
                'Координаты можно оставить пустыми, например Новый регион; FR; ; ; описание. '
                'Точка с вашими координатами будет отмечена как личное уточнение.' if parts[3] == 'region' else
                'Личная цель · новый сорт\nВведите название; описание. Цель добавится в ваш справочник и план, не в общий каталог.')
        else:
            raise ValueError('Неизвестное действие атласа')

    def analysis(self, client, actor, owner, view):
        screen(f'va:analysis:{owner}', 'vi:menu' if owner == actor else f'va:view:{owner}')
        analysis = view['analytics']
        summary = analysis['summary']
        lines = [f'Vivino · аналитика пользователя {owner}',
            f"Разных вин попробовано: {summary['tasted_wines']}; стран: {summary['countries']}; сортов в винах: {summary['grapes']}.",
            f"Полный известный состав: {summary['known_composition']}; неизвестный/неполный: {summary['unknown_composition']}.",
            '\nСорта в ваших винах:']
        popular = sorted([g for g in analysis['grapes'] if g['wine_count']], key=lambda g: (-g['wine_count'], g['name']))
        for grape in popular[:5]:
            average = f"{grape['mean_rating']:.2f}" if grape['mean_rating'] is not None else 'нет'
            lines.append(f"• {grape['name'][:60]}: {grape['wine_count']} вин; средняя {average}; оценено {grape['rated_count']}.")
        if not popular:
            lines.append('Сорта ещё не указаны. Уточните состав в «Моих винах», чтобы получить предложения по сортам.')
        lines.append('\nПредпочтения (от трёх оценённых вин на сорт):')
        for grape in analysis['preferences'][:3]:
            lines.append(f"• {grape['name'][:60]}: {grape['mean_rating']:.2f}/5, {grape['rated_count']} вин.")
        if not analysis['preferences']:
            lines.append('Недостаточно данных для сравнения.')
        lines.append('Оценки купажей относятся к винам, не к отдельному сорту.')
        lines.append('\nЧто попробовать:')
        buttons = []
        for rec in analysis['recommendations'][:6]:
            lines.append(f"• {rec['name'][:65]} — {rec['reason'][:260]}")
            route = f"va:wine:{owner}:{rec['id']}" if rec['kind'] == 'wine' else f"va:item:{owner}:{rec['kind']}:{rec['id']}"
            buttons.append([(rec['name'][:55], route)])
        lines.append('Это исследовательские цели, без прогноза вкуса или проверки наличия в продаже.')
        buttons.append([('Сорта и план', f'va:list:{owner}:grapes:0:all'), ('Мои вина', f'va:wines:{owner}:0')])
        buttons.append([('Географическая карта', 'vi:build' if owner == actor else f'va:map:{owner}')])
        self.reply(client, actor, '\n'.join(lines), keyboard(buttons))

    def users(self, client, actor, page):
        self.controls.bot._admin(actor)
        if page < 0:
            raise ValueError('Неверная страница')
        rows = self.store.query("""SELECT u.user_id,u.enabled,s.value FROM users u JOIN settings s
            ON s.key='project:vivino:'||u.user_id||':dataset' WHERE s.value<>'null'
            ORDER BY u.user_id LIMIT 8 OFFSET ?""", (page * 7,))
        screen(f'va:users:{page}', 'vi:menu')
        buttons = []
        lines = ['Vivino · пользователи, загрузившие данные']
        for row in rows[:7]:
            dataset = json.loads(row['value'])
            lines.append(f"ID {row['user_id']}: {len(dataset['rows'])} записей; доступ " + ('включён' if row['enabled'] else 'выключен'))
            buttons.append([(f"Посмотреть {row['user_id']}", f"va:view:{row['user_id']}")])
        nav = []
        if page:
            nav.append(('← Страница', f'va:users:{page-1}'))
        if len(rows) > 7:
            nav.append(('Страница →', f'va:users:{page+1}'))
        if nav:
            buttons.append(nav)
        self.reply(client, actor, '\n'.join(lines) if rows else lines[0] + '\nЗагрузок пока нет.', keyboard(buttons))

    def list(self, client, actor, owner, kind, page, mode):
        if kind not in {'regions', 'grapes'} or mode not in {'all', 'new', 'tried', 'plan'} or page < 0:
            raise ValueError('Неверная настройка списка')
        view = atlas.model(self.store, actor, owner)
        query = self.store.get_setting(f'vivino_search:{actor}:{owner}:{kind}', '')
        rows = [r for r in view[kind] if (mode == 'all' or mode == 'new' and not r['tried']
                or mode == 'tried' and r['tried'] or mode == 'plan' and r['planned'])
                and (not query or atlas.normalize(query) in atlas.normalize(' '.join(
                    [r['name'], r.get('ru', ''), r.get('description', ''), *r.get('aliases', [])])))]
        rows.sort(key=lambda r: r['name'])
        page = min(page, max(0, (len(rows)-1)//7))
        screen(f'va:list:{owner}:{kind}:{page}:{mode}', 'vi:menu' if owner == actor else f'va:view:{owner}')
        singular = 'region' if kind == 'regions' else 'grape'
        buttons = [[(('✓ ' if r['tried'] else '＋ ' if r['planned'] else '') + r['name'][:55], f"va:item:{owner}:{singular}:{r['id']}")]
                   for r in rows[page*7:(page+1)*7]]
        buttons += [[(('✓ ' if mode == value else '') + label, f'va:list:{owner}:{kind}:0:{value}')
                     for label, value in [('Все', 'all'), ('Пробовал', 'tried')]],
                    [(('✓ ' if mode == value else '') + label, f'va:list:{owner}:{kind}:0:{value}')
                     for label, value in [('Не отмечено', 'new'), ('План', 'plan')]],
                    [('Поиск', f'va:search:{owner}:{kind}')]]
        if owner == actor:
            buttons.append([('Добавить свою цель', f'va:new:{owner}:{singular}')])
        nav = []
        if page:
            nav.append(('← Страница', f'va:list:{owner}:{kind}:{page-1}:{mode}'))
        if (page+1)*7 < len(rows):
            nav.append(('Страница →', f'va:list:{owner}:{kind}:{page+1}:{mode}'))
        if nav:
            buttons.append(nav)
        text = ('Регионы' if kind == 'regions' else 'Сорта') + f' · пользователь {owner}\nНайдено: {len(rows)}; страница {page+1}.'
        text += '\n«Не отмечено» означает отсутствие подтверждения, а не доказательство, что вы никогда не пробовали.'
        text += '\nСправочник неполный; неизвестные сорта и регионы показываются отдельно в карте.'
        if query:
            text += '\nПоиск: ' + query[:100]
        self.reply(client, actor, text, keyboard(buttons))

    @staticmethod
    def target(view, kind, identifier):
        if kind not in {'region', 'grape'}:
            raise ValueError('Неверный тип цели')
        row = next((r for r in view['regions' if kind == 'region' else 'grapes'] if r['id'] == identifier), None)
        if not row:
            raise ValueError('Цель не найдена')
        return row

    def item(self, client, actor, owner, kind, identifier):
        view = atlas.model(self.store, actor, owner)
        row = self.target(view, kind, identifier)
        collection = 'regions' if kind == 'region' else 'grapes'
        screen(f'va:item:{owner}:{kind}:{identifier}', f'va:list:{owner}:{collection}:0:all')
        text = row['name'] + '\n' + (row['description'] or 'Подробное описание в справочнике отсутствует.')
        text += '\nДегустация: ' + ('подтверждена' if row['tried'] else 'не отмечена') + '; в плане: ' + ('да' if row['planned'] else 'нет')
        text += f"\nСвязанных записей о винах: {len(row['wines'])}."
        text += '\nИсточник: ' + (row['source'] or 'личные данные; привязка не подтверждена внешним справочником')
        if kind == 'grape':
            text += '\nПроисхождение сорта: ' + (row['origin'] or 'не указано') + '; это не регион происхождения конкретного вина.'
        buttons = []
        if owner == actor:
            buttons += [[('Убрать из плана' if row['planned'] else 'Добавить в план', f'va:plan:{owner}:{kind}:{identifier}')],
                        [('Убрать мою отметку' if row['manual'] else 'Я пробовал — отметить', f'va:done:{owner}:{kind}:{identifier}')]]
            if row['tasted']:
                text += '\nОтметки из вин меняются в карточке конкретного вина.'
        for index in row['wines'][:5]:
            wine = view['wines'][index]
            buttons.append([(wine['name'][:55], f"va:wine:{owner}:{wine['id']}")])
        self.reply(client, actor, text, keyboard(buttons))

    def wines(self, client, actor, owner, page):
        if page < 0:
            raise ValueError('Неверная страница')
        view = atlas.model(self.store, actor, owner)
        rows = view['wines']
        screen(f'va:wines:{owner}:{page}', 'vi:menu' if owner == actor else f'va:view:{owner}')
        buttons = [[(('✓ ' if r['tasted'] else '') + r['name'][:55], f"va:wine:{owner}:{r['id']}")] for r in rows[page*7:(page+1)*7]]
        nav = []
        if page:
            nav.append(('← Страница', f'va:wines:{owner}:{page-1}'))
        if (page+1)*7 < len(rows):
            nav.append(('Страница →', f'va:wines:{owner}:{page+1}'))
        if nav:
            buttons.append(nav)
        self.reply(client, actor, f'Вина · пользователь {owner}\nЗаписей: {len(rows)}. Выберите вино для описания и сортов.', keyboard(buttons))

    def wine(self, client, actor, owner, identifier):
        view = atlas.model(self.store, actor, owner)
        row = next((r for r in view['wines'] if r['id'] == identifier), None)
        if not row:
            raise ValueError('Вино не найдено')
        screen(f'va:wine:{owner}:{identifier}', f'va:wines:{owner}:0')
        names = {r['id']: r['name'] for r in view['grapes']}
        varieties = [names[i] for i in row['grapes']] + row['unknown_grapes']
        text = f"{row['name']} · {row['vintage']}\n{row['winery']}\n{row['country']} · {row['region']}\n"
        text += 'Пробовал: ' + ('да' if row['tasted'] else 'не подтверждено') + '; моя оценка: ' + str(row['rating'] if row['rating'] is not None else 'нет')
        text += '\nСорта: ' + (', '.join(varieties) if varieties else 'не указаны — уточните состав')
        text += '\nОтзыв: ' + (row['review'][:1800] or 'нет')
        buttons = []
        if owner == actor:
            buttons += [[('Дегустация: не пробовал' if row['tasted'] else 'Отметить: пробовал', f'va:tasted:{owner}:{identifier}')],
                        [('Уточнить сорта', f'va:editgrapes:{owner}:{identifier}')]]
        self.reply(client, actor, text, keyboard(buttons))

    def input(self, client, actor, text):
        pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        value = pending['pending'] if pending else ''
        if not value or not value.startswith(('vivino_grapes:', 'vivino_search:', 'vivino_custom:')):
            return False
        if value.startswith('vivino_search:'):
            _, owner, kind = value.split(':')
            atlas.subject(self.store, actor, int(owner))
            if len(text) > 100:
                raise ValueError('Поисковый запрос — до 100 символов')
            self.store.set_setting(f'vivino_search:{actor}:{owner}:{kind}', '' if text.casefold() == 'все' else text)
            self.controls.bot.market_controls.pending(actor, None)
            self.list(client, actor, int(owner), kind, 0, 'all')
        elif value.startswith('vivino_custom:'):
            import hashlib
            import math
            kind = value.split(':')[1]
            fields = [v.strip() for v in text.split(';')]
            if len(text) > 1000 or not fields[0] or len(fields[0]) > 100:
                raise ValueError('Название до 100 символов; вся запись до 1000')
            row = {'name': fields[0], 'kind': kind, 'description': fields[-1] if len(fields) > 1 else '', 'source': ''}
            if kind == 'region':
                if len(fields) != 5:
                    raise ValueError('Нужны 5 полей: название; код страны; широта; долгота; описание')
                country = atlas.indexes()[0].get(atlas.normalize(fields[1]))
                if not country:
                    raise ValueError('Страна не найдена. Укажите известный код, например FR, IT, RU')
                try:
                    lat, lon = (float(v.replace(',', '.')) if v else None for v in fields[2:4])
                except ValueError:
                    raise ValueError('Координаты — десятичные числа либо оба поля пустые') from None
                if (lat is None) != (lon is None) or lat is not None and (
                        not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180):
                    raise ValueError('Широта −90…90, долгота −180…180, либо оба поля пустые')
                row.update(country=country, lat=lat, lon=lon)
            else:
                if len(fields) > 2:
                    raise ValueError('Формат: название; описание')
                row.update(origin='', vivc='')
            row['id'] = 'c' + hashlib.sha256((kind + '|' + row.get('country', '') + '|' + atlas.normalize(row['name'])).encode()).hexdigest()[:16]
            custom = self.store.get_setting(self.controls.key(actor, 'custom'), [])
            if len(custom) >= 1000 and not any(r['id'] == row['id'] for r in custom):
                raise ValueError('До 1000 личных целей')
            atlas.revise(self.store, actor, 'custom', [r for r in custom if r['id'] != row['id']] + [row])
            plan = self.store.get_setting(self.controls.key(actor, 'plan'), [])
            token = kind + ':' + row['id']
            if token not in plan:
                atlas.revise(self.store, actor, 'plan', [*plan, token])
            self.controls.bot.market_controls.pending(actor, None)
            self.item(client, actor, actor, kind, row['id'])
        else:
            identifier = value.split(':')[1]
            view = atlas.model(self.store, actor)
            if not any(r['id'] == identifier for r in view['wines']):
                raise ValueError('Вино больше не доступно')
            varieties = [] if text.casefold() == 'неизвестно' else [v.strip() for v in text.split(';') if v.strip()]
            if len(text) > 500 or len(varieties) > 20:
                raise ValueError('До 20 сортов и 500 символов')
            notes = self.store.get_setting(self.controls.key(actor, 'annotations'), {})
            notes.setdefault(identifier, {})['grapes'] = varieties
            atlas.revise(self.store, actor, 'annotations', notes)
            self.controls.bot.market_controls.pending(actor, None)
            self.wine(client, actor, actor, identifier)
        return True
