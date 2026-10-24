"""Personal uploads, expiring previews and queued map generation."""
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from . import vivino
from .store import dumps, utcnow
from .telegram_charts import keyboard
from .telegram_data_jobs import require_actor
from .telegram_navigation import screen
from .telegram_projects import setting_key
from .telegram_vivino_atlas import AtlasControls


class VivinoControls:
    def __init__(self, bot):
        self.bot, self.store, self.reply = bot, bot.store, bot._reply
        self.atlas_controls = AtlasControls(self)

    def key(self, actor, name):
        return setting_key(actor, 'vivino', name)

    def menu(self, client, actor):
        require_actor(self.store, actor)
        from .subscription_access import require
        require(self.store, actor, 'vivino')
        self.store.set_setting(self.key(actor, 'preview'), None)
        screen('vi:menu', 'pj:vivino')
        dataset = self.store.get_setting(self.key(actor, 'dataset'))
        text = ('Vivino · личная карта вин\nЗагрузите ZIP из Vivino или full_wine_list.csv. '
                'После проверки и подтверждения бот пришлёт географический HTML-атлас вин, регионов и сортов. '
                'Расчёт локальный, без API и LLM.\n\n')
        rows = [[('Загрузить ZIP / CSV', 'vi:upload')]]
        if dataset:
            text += f"Записей: {len(dataset['rows'])}; с отзывами: {dataset['quality']['reviewed']}.\nВаши данные доступны вам и администраторам бота."
            rows += [[('Географическая карта', 'vi:build')], [('Мои вина и сорта', f'va:wines:{actor}:0')], [('Удалить мои данные', 'vi:delete')]]
        else:
            text += 'Ваши данные ещё не загружены. Архив из чата Codex не импортируется автоматически.'
        rows += [[('Как получить карту', 'vi:help')]]
        rows += [[('Регионы и план', f'va:list:{actor}:regions:0:all'), ('Сорта и план', f'va:list:{actor}:grapes:0:all')]]
        rows += [[('Аналитика и что попробовать', f'va:analysis:{actor}')]]
        if self.store.authorize(actor, admin=True):
            rows += [[('Пользователи Vivino', 'va:users:0')]]
        self.reply(client, actor, text, keyboard(rows))

    def callback(self, client, actor, data):
        require_actor(self.store, actor)
        from .subscription_access import require
        require(self.store, actor, 'vivino')
        screen(data, 'vi:menu')
        if data == 'vi:menu':
            return self.menu(client, actor)
        if data == 'vi:upload':
            self.store.set_setting(self.key(actor, 'preview'), None)
            self.bot.market_controls.pending(actor, 'vivino_upload')
            self.reply(client, actor, 'Vivino · загрузка\nОтправьте ZIP до 10 МБ с full_wine_list.csv '
                'или сам CSV до 8 МБ. Папка внутри ZIP допустима. RAR нужно перепаковать в ZIP.\n'
                'Профиль, сведения о входах и сканированиях не импортируются. До 3000 записей. '
                'Текущие данные заменяются только после подтверждения.')
        elif data.startswith('vi:confirm:'):
            token = data.removeprefix('vi:confirm:')
            with self.store.connection() as conn:
                conn.execute('BEGIN IMMEDIATE')
                row = conn.execute('SELECT value FROM settings WHERE key=?', (self.key(actor, 'preview'),)).fetchone()
                preview = json.loads(row[0]) if row else None
                if not preview or preview['token'] != token or preview['expires'] < utcnow():
                    raise ValueError('Предпросмотр устарел. Загрузите файл заново.')
                user = conn.execute('SELECT enabled FROM users WHERE user_id=?', (actor,)).fetchone()
                if not user or not user[0]:
                    raise PermissionError('Нет доступа')
                dataset = {**preview['dataset'], 'generation': token, 'imported_at': utcnow()}
                conn.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                             (self.key(actor, 'dataset'), dumps(dataset)))
                conn.execute('DELETE FROM settings WHERE key=?', (self.key(actor, 'preview'),))
                payload = {'actor_id': actor, 'owner_id': actor, 'generation': token, 'report_id': uuid4().hex, 'map_version': vivino.MAP_VERSION}
                cursor = conn.execute('INSERT INTO jobs(kind,payload,dedupe_key,created_at) VALUES(?,?,?,?)',
                    ('vivino_map', dumps(payload), f'vivino:geo{vivino.MAP_VERSION}:{actor}:{actor}:{token}', utcnow()))
                identifier = cursor.lastrowid
            from .feature_access import allowed as function_allowed
            if function_allowed(self.store, actor, 'wine_cellar'):
                from .wine_cellar import sync_reviews
                suggestions = sync_reviews(self.store, actor, dataset['rows'], token)
            else:
                suggestions = []
            self.reply(client, actor, f'Данные сохранены. Построение карты {identifier} в очереди. '
                'HTML-файл придёт отдельным сообщением.' +
                (f' Есть предложений списать бутылку из винотеки: {len(suggestions)}. Проверьте их перед списанием.' if suggestions else ''),
                keyboard([[('Vivino', 'vi:menu')]] + ([[('Проверить запасы после отзывов', 'wc:proposals')]] if suggestions else [])))
        elif data == 'vi:build':
            self.queue(client, actor, actor)
        elif data == 'vi:delete':
            self.reply(client, actor, 'Удалить мои данные Vivino?\nСписок вин, предпросмотр и локальные карты будут удалены. '
                'Рыночные настройки сохранятся. Уже присланные или скачанные файлы нужно удалить самостоятельно.',
                keyboard([[('Удалить данные Vivino', 'vi:delete_confirm')]]))
        elif data == 'vi:delete_confirm':
            self.delete(actor)
            self.menu(client, actor)
        elif data == 'vi:help':
            self.reply(client, actor, 'Vivino · как получить карту\n1. Получите экспорт данных из Vivino.\n'
                '2. Нажмите «Загрузить ZIP / CSV» и отправьте файл документом.\n'
                '3. Проверьте предпросмотр и нажмите «Сохранить и построить карту».\n'
                '4. Скачайте присланный HTML и откройте его в браузере. На телефоне: сохранить файл → открыть в браузере.\n\n'
                'Карта показывает регионы и сорта: пробовал, в плане, не отмечено. Прогресс основан на вашей оценке, отзыве или явной отметке. '
                'В экспорте Vivino сорта часто отсутствуют: уточняйте состав в «Моих винах». '
                'Страны окрашены по числу разных подтверждённых вин; точки точных регионов включаются отдельно. '
                'В HTML и разделе «Аналитика и что попробовать» есть статистика сортов и личные цели. '
                'Справочник Wikidata неполный. '
                'План и отметки меняются в боте; HTML — снимок. Ваши данные могут просматривать администраторы.')
        else:
            raise ValueError('Неизвестное действие Vivino')

    def queue(self, client, actor, owner):
        from .vivino_atlas import subject
        subject(self.store, actor, owner)
        dataset = self.store.get_setting(self.key(owner, 'dataset'))
        if not dataset:
            raise ValueError('Сначала загрузите ZIP или CSV.')
        generation = dataset['generation']
        identifier = self.store.enqueue('vivino_map', {'actor_id': actor, 'owner_id': owner,
            'generation': generation, 'map_version': vivino.MAP_VERSION, 'report_id': uuid4().hex},
            f'vivino:geo{vivino.MAP_VERSION}:{actor}:{owner}:{generation}')
        screen('vi:result' if owner == actor else f'va:result:{owner}', 'vi:menu' if owner == actor else f'va:view:{owner}')
        self.reply(client, actor, f'Географическая карта пользователя {owner}: задание {identifier}. '
            'Повторное нажатие не создаёт копию. Файл отправляется запросившему пользователю, после обработки доступен в /history.')

    def document(self, client, message, actor):
        require_actor(self.store, actor)
        from .subscription_access import require
        require(self.store, actor, 'vivino')
        document = message.document
        filename, size = str(getattr(document, 'file_name', '')), getattr(document, 'file_size', None)
        if type(size) is not int or not 0 < size <= vivino.MAX_UPLOAD or not filename.lower().endswith(('.zip', '.csv')):
            raise ValueError('Нужен ZIP до 10 МБ или CSV до 8 МБ.')
        stream = client.download_media(message, in_memory=True)
        if not stream or not hasattr(stream, 'read'):
            raise ValueError('Не удалось скачать файл. Повторите загрузку.')
        try:
            stream.seek(0)
            raw = stream.read(vivino.MAX_UPLOAD + 1)
        finally:
            stream.close()
        require_actor(self.store, actor)
        dataset = vivino.parse_upload(raw, filename)
        token = uuid4().hex
        self.store.set_setting(self.key(actor, 'preview'), {'token': token,
            'expires': (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(), 'dataset': dataset})
        self.bot.market_controls.pending(actor, None)
        screen('vi:preview', 'vi:menu')
        quality = dataset['quality']
        self.reply(client, actor, f"Vivino · проверьте загрузку\nЗаписей: {len(dataset['rows'])}; с отзывами: {quality['reviewed']}.\n"
            f"Точных дубликатов: {quality['duplicates']}; без названия: {quality['missing_names']}; неверных оценок: {quality['invalid_ratings']}.\n"
            'При подтверждении личный набор заменится и карта будет построена в общей очереди. '
            'Другие файлы ZIP не сохраняются. Подтверждение действует 10 минут.',
            keyboard([[('Сохранить и построить карту', 'vi:confirm:' + token)]]))

    def delete(self, actor):
        require_actor(self.store, actor)
        with self.store.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('DELETE FROM settings WHERE key IN (?,?)', (self.key(actor, 'dataset'), self.key(actor, 'preview')))
            for row in conn.execute("SELECT id,payload FROM jobs WHERE kind='vivino_map' AND state='pending'").fetchall():
                payload = json.loads(row['payload'])
                if payload.get('owner_id', payload.get('actor_id')) == actor:
                    conn.execute("UPDATE jobs SET state='cancelled',finished_at=? WHERE id=?", (utcnow(), row['id']))
            for row in conn.execute("SELECT id,quality FROM reports WHERE section='vivino'").fetchall():
                if json.loads(row['quality']).get('owner') == actor:
                    conn.execute("UPDATE deliveries SET state='cancelled' WHERE report_id=? AND state='pending'", (row['id'],))
                    conn.execute("UPDATE reports SET parts='[]',text='Данные Vivino удалены',full_text='Данные Vivino удалены' WHERE id=?", (row['id'],))
            for name in ('annotations', 'progress', 'plan', 'custom'):
                conn.execute('DELETE FROM settings WHERE key=?', (self.key(actor, name),))
        folder = (self.store.path.parent / 'vivino' / str(actor)).resolve()
        for path in folder.glob('*.html'):
            if path.resolve().parent == folder and not path.is_symlink():
                path.unlink(missing_ok=True)
