"""Administrator screens for checks and explicit correction decisions."""
import json

from . import integrity
from .integrity_text import KINDS, SEVERITIES, conflict_details, indicator_title
from .telegram_charts import keyboard
from .telegram_data_jobs import export_dates, require_actor
from .telegram_navigation import screen

STATES = {'new': 'новый', 'review': 'на разборе', 'applied': 'принят', 'rejected': 'отклонён',
          'resolved': 'закрыт', 'pending': 'в очереди', 'running': 'выполняется', 'done': 'завершена',
          'partial': 'неполная', 'cancelled': 'отменена'}


class IntegrityControls:
    def __init__(self, bot):
        self.bot, self.store, self.reply = bot, bot.store, bot._reply

    def menu(self, client, actor, page=0):
        require_actor(self.store, actor, admin=True)
        screen('ig:menu', 'pj:market:manage')
        count = self.store.one('SELECT COUNT(*) n FROM integrity_incidents')['n']
        page = min(max(page, 0), max(0, (count-1)//8))
        entries = self.store.query('SELECT * FROM integrity_incidents ORDER BY id DESC LIMIT 8 OFFSET ?', (page*8,))
        rows = [[(f"#{r['id']} · {STATES.get(r['state'], r['state'])} · {KINDS.get(r['kind'], r['kind'])}", f"ig:show:{r['id']}")] for r in entries]
        if page:
            rows.append([('‹ Предыдущая', f'ig:page:{page-1}')])
        if (page+1)*8 < count:
            rows.append([('Следующая ›', f'ig:page:{page+1}')])
        config = self.store.get_setting('integrity_schedule', {})
        rows += [[('Проверить всю загруженную историю', 'ig:check:all')],
                 [('Выбрать показатель и период', 'ig:choose:0')], [('Ход проверок', 'ig:checks:0')],
                 [('Выключить ежемесячную сверку' if config.get('enabled') else 'Включить ежемесячную сверку', 'ig:schedule')]]
        self.reply(client, actor, f'Целостность данных\nИнцидентов: {count}. Страница {page+1}.\n'
                   'Обычный сбор загружает пропуски. Сверка повторно запрашивает указанную историю. '
                   'Исправления применяются после вашего решения.\nЕжемесячная проверка: 1-го числа, 03:00 Москва.', keyboard(rows))

    def input(self, client, actor, text):
        state = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        if not state or not (state['pending'] or '').startswith('integrity:') or text.startswith('/'):
            return False
        require_actor(self.store, actor, admin=True)
        args = text.split()
        if len(args) != 2:
            raise ValueError('Введите YYYY-MM-DD YYYY-MM-DD; /cancel — отмена')
        start, end = export_dates(*args)
        identifier = integrity.enqueue(self.store, actor, self.store.get_setting(f'integrity_selection:{actor}', []), date_from=start, date_to=end)
        self.bot.market_controls.pending(actor, None)
        screen('ig:checks:0', 'ig:menu')
        self.reply(client, actor, f'Проверка #{identifier} в очереди.', keyboard([[('Ход проверок', 'ig:checks:0')]]))
        return True

    def callback(self, client, actor, data):
        require_actor(self.store, actor, admin=True)
        args = data.split(':')
        if len(args) not in {2, 3} or args[0] != 'ig':
            raise ValueError('Неизвестный экран целостности')
        action, value = args[1], args[2] if len(args) == 3 else None
        if action == 'menu':
            return self.menu(client, actor)
        if action == 'page' and value and value.isdigit():
            return self.menu(client, actor, int(value))
        if action == 'schedule':
            config = self.store.get_setting('integrity_schedule', {})
            self.store.set_setting('integrity_schedule', {**config, 'enabled': not config.get('enabled'), 'actor_id': actor})
            return self.menu(client, actor)
        if action == 'check' and value == 'all':
            identifier = integrity.enqueue(self.store, actor)
            screen(data, 'ig:menu')
            self.reply(client, actor, f'Проверка #{identifier} в очереди.', keyboard([[('Ход проверок', 'ig:checks:0')]]))
            return
        if action == 'choose' and value and value.isdigit():
            page = int(value)
            entries = self.store.query('SELECT rowid token,id,title FROM market_series WHERE id IN (SELECT DISTINCT series_id FROM market_observations UNION SELECT dataset FROM source_snapshots) ORDER BY title LIMIT 9 OFFSET ?', (page*8,))
            rows = [[(r['title'][:50], f"ig:period:{r['token']}")] for r in entries[:8]]
            if page:
                rows.append([('‹ Предыдущая', f'ig:choose:{page-1}')])
            if len(entries) > 8:
                rows.append([('Следующая ›', f'ig:choose:{page+1}')])
            screen(data, 'ig:menu')
            self.reply(client, actor, 'Выберите загруженный показатель:', keyboard(rows))
            return
        if action == 'period' and value and value.isdigit():
            row = self.store.one('SELECT id FROM market_series WHERE rowid=?', (int(value),))
            if not row:
                raise ValueError('Показатель не найден')
            self.store.set_setting(f'integrity_selection:{actor}', [row['id']])
            self.bot.market_controls.pending(actor, 'integrity:period')
            screen(data, 'ig:choose:0')
            self.reply(client, actor, 'Введите начало и конец сверки: YYYY-MM-DD YYYY-MM-DD. /cancel — отмена.')
            return
        if action == 'checks' and value and value.isdigit():
            page = int(value)
            entries = self.store.query('SELECT * FROM integrity_checks ORDER BY id DESC LIMIT 9 OFFSET ?', (page*8,))
            rows, lines = [], ['Проверки истории:']
            for check in entries[:8]:
                lines.append(f"#{check['id']}: {STATES.get(check['state'], check['state'])}; проверено блоков {len(json.loads(check['checkpoint']))}")
                if check['error']:
                    lines.append(check['error'])
                if check['state'] in {'pending', 'running'}:
                    rows.append([(f"Отменить #{check['id']}", f"ig:cancel:{check['id']}")])
                if check['state'] in {'partial', 'cancelled'}:
                    rows.append([(f"Продолжить #{check['id']}", f"ig:resume:{check['id']}")])
                rows.append([(f"Решить по исправлениям #{check['id']}", f"ig:group:{check['id']}")])
            if page:
                rows.append([('‹ Предыдущая', f'ig:checks:{page-1}')])
            if len(entries) > 8:
                rows.append([('Следующая ›', f'ig:checks:{page+1}')])
            screen(data, 'ig:menu')
            self.reply(client, actor, '\n'.join(lines), keyboard(rows))
            return
        if action in {'cancel', 'resume'} and value and value.isdigit():
            check = self.store.one('SELECT * FROM integrity_checks WHERE id=?', (int(value),))
            if not check:
                raise ValueError('Проверка не найдена')
            if action == 'resume' and check['state'] not in {'partial', 'cancelled'}:
                raise ValueError('Проверка уже выполняется или завершена')
            self.store.execute('UPDATE integrity_checks SET state=?,actor_id=? WHERE id=?', ('cancelled' if action == 'cancel' else 'pending', actor, check['id']))
            if action == 'resume':
                self.store.enqueue('integrity_audit', {'actor_id': actor, 'check_id': check['id']})
            return self.callback(client, actor, 'ig:checks:0')
        if action == 'group' and value and value.isdigit():
            entries = self.store.query("SELECT id FROM integrity_incidents WHERE check_id=? AND state='new' AND kind IN ('source_revision','source_missing')", (int(value),))
            screen(data, 'ig:checks:0')
            self.reply(client, actor, f'Решение по {len(entries)} исправлениям проверки #{value}. Старые версии сохранятся.',
                       keyboard([[('Принять исправления', f'ig:applygroup:{value}')], [('Отклонить исправления', f'ig:rejectgroup:{value}')]]))
            return
        if action in {'applygroup', 'rejectgroup'} and value and value.isdigit():
            for row in self.store.query("SELECT id FROM integrity_incidents WHERE check_id=? AND state='new' AND kind IN ('source_revision','source_missing')", (int(value),)):
                integrity.resolve(self.store, actor, row['id'], 'accept' if action == 'applygroup' else 'reject')
            return self.menu(client, actor)
        if action == 'details' and value and value.isdigit():
            row = self.store.one('SELECT * FROM integrity_incidents WHERE id=?', (int(value),))
            if not row:
                raise ValueError('Инцидент не найден')
            text = json.dumps({'до': json.loads(row['old_payload']), 'после': json.loads(row['new_payload'])}, ensure_ascii=False, indent=2)
            screen(data, 'ig:show:' + value)
            self.reply(client, actor, f'Подробности инцидента #{value}\n' + text[:3300] + ('\nПолный снимок сохранён в базе; экспорт доступен в карточке источника.' if len(text) > 3300 else ''))
            return
        if action in {'show', 'accept', 'reject'} and value and value.isdigit():
            if action != 'show':
                integrity.resolve(self.store, actor, int(value), action)
            row = self.store.one('SELECT * FROM integrity_incidents WHERE id=?', (int(value),))
            if not row:
                raise ValueError('Инцидент не найден')
            old, new = json.loads(row['old_payload']), json.loads(row['new_payload'])
            if row['kind'] == 'source_conflict':
                comparison = conflict_details(new)
            elif row['kind'] == 'source_reference_revision':
                comparison = f"Версия: {old['version'][:12]} → {new['version'][:12]}\nСохранён полный снимок изменения справочника."
            elif isinstance(old, dict) and isinstance(new, dict):
                comparison = f"Значение: {old.get('value', '—')} → {new.get('value', '—')}\nМетодика: {old.get('definition', '—')} → {new.get('definition', '—')}"
            else:
                comparison = 'Описание: ' + str(new or old or 'Проверка локальной базы')[:400]
            text = (f"Инцидент #{value}\n{KINDS.get(row['kind'], row['kind'])} · {STATES.get(row['state'], row['state'])}\n"
                    f"Важность: {SEVERITIES.get(row['severity'], row['severity'])}\n"
                    f"Показатель: {indicator_title(self.store, row['series_id'])}\nПериод: {row['date'] or '—'}\n"
                    f"{comparison}\n"
                    'Пересмотр источником не означает взлом приложения.')
            rows = [[('Подробнее', 'ig:details:' + value)]]
            if row['state'] == 'new':
                rows.append([('Отклонить / закрыть', 'ig:reject:' + value)])
                if row['kind'] in {'source_revision', 'source_missing', 'source_reference_revision'}:
                    rows.insert(0, [('Принять исправление', 'ig:accept:' + value)])
            screen('ig:show:' + value, 'ig:menu')
            self.reply(client, actor, text, keyboard(rows))
            return
        raise ValueError('Неизвестное действие целостности')
