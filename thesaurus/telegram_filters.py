"""Personal region and phrase selection with visible state and pagination."""
import hashlib
import json

from .config import ROOT
from .scheduling import update_subscription
from .telegram_charts import keyboard
from .telegram_data_jobs import actor_scope, require_actor
from .telegram_navigation import checked, screen


class FilterControls:
    def __init__(self, store, reply):
        self.store, self.reply = store, reply

    def region_names(self):
        names = self.store.get_setting('region_names', {})
        cached = ROOT / 'functions' / 'src' / 'wordstat' / 'geo' / 'wordstat_regions_tree.json'
        try:
            if not cached.is_file() or cached.stat().st_size > 2_000_000:
                return names
            nodes = [json.loads(cached.read_text(encoding='utf-8'))]
            for _ in range(20_000):
                if not nodes:
                    break
                node = nodes.pop()
                if isinstance(node, dict):
                    identifier, label = node.get('value'), node.get('label')
                    if str(identifier).isdigit() and isinstance(label, str):
                        names[str(identifier)] = label[:70]
                    nodes.extend(node.values())
                elif isinstance(node, list):
                    nodes.extend(node)
        except (OSError, ValueError):
            pass
        return names

    def regions(self, client, actor, page=0):
        scope = actor_scope(self.store, actor, 'regions')
        values = self.store.regions()
        size = 10
        page = min(max(0, page), max(0, (len(values)-1)//size))
        screen('menu:regions', 'menu:settings')
        names = self.region_names()
        rows = [[(checked(f"{names.get(str(value), 'Регион')} · {value}", value in scope['regions']), f'rg:toggle:{value}')] for value in values[page*size:(page+1)*size]]
        if page:
            rows.append([('‹ Предыдущая', f'rg:page:{page-1}')])
        if (page+1)*size < len(values):
            rows.append([('Следующая ›', f'rg:page:{page+1}')])
        if values:
            rows.append([('Выбрать все регионы', 'rg:all:0')])
        text = f"Мои регионы\nВыбрано: {len(scope['regions'])} из {len(values)}. ✓ — регион включён в ваши отчёты.\nСтраница {page+1} из {max(1, (len(values)+size-1)//size)}."
        if not values:
            text += '\nНабор регионов ещё не задан администратором.'
        self.reply(client, actor, text, keyboard(rows))

    def keywords(self, client, actor, section=None, page=0):
        require_actor(self.store, actor)
        if section is None:
            screen('menu:keywords', 'menu:settings')
            self.reply(client, actor, 'Мои фразы и показатели\nВыберите раздел. Здесь меняются ваши фильтры; общий набор задаёт администратор.', keyboard([
                [('Wordstat', 'kw:open:wordstat'), ('RUONIA', 'kw:open:ruonia')], [('Региональные запросы', 'kw:open:regions')]]))
            return
        scope = actor_scope(self.store, actor, section)
        from .reporting import report_filters
        choices = report_filters(self.store, section)[0]
        size = 8
        page = min(max(0, page), max(0, (len(choices)-1)//size))
        screen(f'kw:open:{section}', 'menu:keywords')
        rows = []
        for phrase in choices[page*size:(page+1)*size]:
            token = hashlib.sha256(phrase.encode()).hexdigest()[:12]
            rows.append([(checked(phrase[:46], phrase in scope['phrases']), f'kw:toggle:{section}:{token}')])
        if page:
            rows.append([('‹ Предыдущая', f'kw:page:{section}:{page-1}')])
        if (page+1)*size < len(choices):
            rows.append([('Следующая ›', f'kw:page:{section}:{page+1}')])
        if choices:
            rows.append([('Выбрать все', f'kw:all:{section}')])
        self.reply(client, actor, f"Мои фильтры · {section}\nВыбрано {len(scope['phrases'])} из {len(choices)}. ✓ — включено.\nСтраница {page+1} из {max(1, (len(choices)+size-1)//size)}.", keyboard(rows))

    def callback(self, client, actor, data):
        require_actor(self.store, actor)
        args = data.split(':')
        if args[0] == 'rg' and len(args) == 3:
            action, value = args[1:]
            if action == 'page' and value.isdigit():
                self.regions(client, actor, int(value))
                return
            scope = actor_scope(self.store, actor, 'regions')
            if action == 'all':
                chosen = self.store.regions()
            elif action == 'toggle' and value.isdigit() and int(value) in self.store.regions():
                region = int(value)
                chosen = [r for r in scope['regions'] if r != region] if region in scope['regions'] else [*scope['regions'], region]
                if not chosen:
                    raise ValueError('Оставьте хотя бы один регион; доставку можно выключить в подписках.')
            else:
                raise ValueError('Регион недоступен')
            update_subscription(self.store, actor, 'regions', regions=chosen)
            page = self.store.regions().index(int(value))//10 if action == 'toggle' else 0
            self.regions(client, actor, page)
        elif args[0] == 'kw' and len(args) in {3, 4}:
            action, section = args[1:3]
            scope = actor_scope(self.store, actor, section)
            if action == 'open':
                self.keywords(client, actor, section)
                return
            if action == 'page' and len(args) == 4 and args[3].isdigit():
                self.keywords(client, actor, section, int(args[3]))
                return
            from .reporting import report_filters
            choices = report_filters(self.store, section)[0]
            if action == 'all':
                chosen, page = choices, 0
            elif action == 'toggle' and len(args) == 4:
                phrase = next((s for s in choices if hashlib.sha256(s.encode()).hexdigest()[:12] == args[3]), None)
                if phrase is None:
                    raise PermissionError('Фраза больше не доступна')
                chosen = [s for s in scope['phrases'] if s != phrase] if phrase in scope['phrases'] else [*scope['phrases'], phrase]
                if not chosen:
                    raise ValueError('Оставьте хотя бы одну фразу; доставку можно выключить в подписках.')
                page = choices.index(phrase)//8
            else:
                raise ValueError('Неверный фильтр')
            update_subscription(self.store, actor, section, phrases=chosen)
            self.keywords(client, actor, section, page)
        else:
            raise ValueError('Неизвестный фильтр')
