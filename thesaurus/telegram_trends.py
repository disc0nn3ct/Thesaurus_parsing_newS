"""Private, administrator-only CSV ingestion with an expiring preview."""
import base64
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from . import google_trends, market
from .store import utcnow
from .telegram_charts import keyboard
from .telegram_data_jobs import require_actor
from .telegram_navigation import screen


class TrendsControls:
    def __init__(self, store, reply):
        self.store, self.reply = store, reply

    def menu(self, client, actor):
        require_actor(self.store, actor)
        self.store.set_setting(f'trends_preview:{actor}', None)
        from .telegram_market import MarketControls
        MarketControls(self.store, self.reply).pending(actor, None)
        screen('gt:menu', 'ch:menu')
        series = [r for r in market.selected(self.store, actor) if r['source'] == 'google_trends']
        text = 'Google Trends\n' + google_trends.NOTE
        text += f'\n\nВаших доступных рядов: {len(series)}.'
        config = self.store.get_setting('trends_schedule', {})
        from .trends_api import WINDOWS, region_label
        text += ('\nАвтосбор: ' + ('включён' if config.get('enabled') else 'выключен') +
                 '; ' + ', '.join(config.get('times', ['08:30'])) + ' (Москва); ' + region_label(config.get('geo', 'RU')) +
                 '; история: ' + WINDOWS.get(config.get('window', 'today 5-y'), '5 лет') + '.')
        last_collection = self.store.get_setting('trends_last_collection', {})
        if last_collection.get('retrieved_at'):
            text += '\nПоследняя попытка: ' + last_collection['retrieved_at'][:19] + '; ' + last_collection['status'] + '.'
        if last_collection.get('errors'):
            text += '\n' + '\n'.join(last_collection['errors'])
        usage = self.store.get_setting('trends_usage', {})
        if usage.get('day'):
            text += f"\nОбращения за {usage['day']} (Москва): {usage.get('calls', 0)}/60."
        for row in series[:8]:
            last = self.store.one('SELECT MAX(date) d,COUNT(value) n FROM market_observations WHERE series_id=?', (row['id'],))
            meta = json.loads(row['metadata'])
            text += f"\n{row['title']}: до {last['d']}, значений {last['n']}; {meta['frequency']}; загружено {row['updated_at'][:10]}."
        rows = [[('Графики и прогноз', 'ch:menu:trends')], [('Мой набор показателей', 'menu:indicators')]]
        if self.store.authorize(actor, admin=True):
            rows += [[('Обновить сейчас', 'gt:collect')],
                     [('Выключить автосбор' if config.get('enabled') else 'Включить автосбор', 'gt:off' if config.get('enabled') else 'gt:on')],
                     [('Настроить автосбор', 'gt:auto')]]
            rows.append([('Загрузить CSV из Google', 'gt:upload')])
        else:
            text += '\nНовые данные загружает администратор.'
        self.reply(client, actor, text, keyboard(rows))

    def callback(self, client, actor, data):
        require_actor(self.store, actor)
        if data == 'gt:menu':
            self.menu(client, actor)
            return
        require_actor(self.store, actor, admin=True)
        if data in {'gt:auto', 'gt:on', 'gt:off', 'gt:time', 'gt:geo'} or data.startswith('gt:window:'):
            from .telegram_market import MarketControls
            from .trends_api import WINDOWS, region_label
            config = {'enabled': False, 'times': ['08:30'], 'geo': 'RU', 'window': 'today 5-y', **self.store.get_setting('trends_schedule', {})}
            if data in {'gt:on', 'gt:off'}:
                from .autocollect import update_schedule
                config = update_schedule(self.store, 'trends', actor, enabled=data == 'gt:on')
            elif data.startswith('gt:window:'):
                config['window'] = {'1': 'today 12-m', '5': 'today 5-y'}.get(data.split(':')[-1])
                if config['window'] is None:
                    raise ValueError('История: 1 или 5 лет.')
            if data.startswith('gt:window:'):
                self.store.set_setting('trends_schedule', config)
            if data in {'gt:time', 'gt:geo'}:
                MarketControls(self.store, self.reply).pending(actor, 'trends_' + data.split(':')[-1])
                screen(data, 'gt:auto')
                self.reply(client, actor, 'Введите время по Москве HH:MM (до двух запусков в день).' if data == 'gt:time' else 'Введите код региона Google: RU (Россия), WORLD (весь мир), US или RU-MOW. Коды отличаются от ID Яндекса.')
                return
            screen('gt:auto', 'gt:menu')
            self.reply(client, actor,
                'Автосбор Google Trends\nНеофициальный бесплатный интерфейс, без ключей. Google может временно ограничить доступ.\n'
                'Состояние: ' + ('включён' if config.get('enabled') else 'выключен') + '\nВремя: ' + ', '.join(config['times']) + ' (Москва)\nРегион: ' + region_label(config['geo']) +
                '\nИстория: ' + WINDOWS[config['window']] + '\nФразы: весь утверждённый набор /keywords. Каждый запрос имеет собственную шкалу.\nЛимит: 60 обращений в сутки; минимум секунда между ними; при 429 пауза не менее часа. '
                'Расписание обновляет общие данные; доставку отчётов включают отдельно.',
                keyboard([[('Выключить' if config.get('enabled') else 'Включить', 'gt:off' if config.get('enabled') else 'gt:on')],
                          [('Изменить время', 'gt:time'), ('Изменить регион', 'gt:geo')],
                          [('✓ 1 год' if config['window'] == 'today 12-m' else '1 год', 'gt:window:1'), ('✓ 5 лет' if config['window'] == 'today 5-y' else '5 лет', 'gt:window:5')],
                          [('Все источники автосбора', 'menu:autocollect')]]))
            return
        if data == 'gt:collect':
            config = self.store.get_setting('trends_schedule', {})
            config['actor_id'] = actor
            self.store.set_setting('trends_schedule', config)
            pending = self.store.one("SELECT id FROM jobs WHERE kind='trends_collect' AND state IN ('pending','running') LIMIT 1")
            identifier = pending['id'] if pending else self.store.enqueue('trends_collect', {'actor_id': actor})
            screen('gt:result', 'gt:menu')
            self.reply(client, actor, f'Разовое обновление {identifier} в очереди. Результат придёт отдельно; платных API нет. Состояние автосбора не изменено.', keyboard([[('Google Trends', 'gt:menu')]]))
            return
        if data == 'gt:upload':
            from .telegram_market import MarketControls
            MarketControls(self.store, self.reply).pending(actor, 'trends_upload')
            self.store.set_setting(f'trends_preview:{actor}', None)
            screen('gt:upload', 'gt:menu')
            phrases = ', '.join(r['phrase'] for r in self.store.keywords())
            self.reply(client, actor,
                'Импорт Google Trends\n1. Откройте https://trends.google.com/trends/explore?geo=RU\n'
                '2. Выберите поисковую фразу, регион и период. Тип поиска: Веб-поиск; категория: Все категории.\n'
                '3. Скачайте CSV именно с графика «Динамика популярности» и отправьте сюда файлом (до 1 МБ).\n'
                'Фраза и название региона будут взяты из заголовка файла. Для сравнения нескольких фраз скачайте их вместе.\n'
                'Разрешённые фразы: ' + phrases + '\nДобавление фразы: /keywords add фраза.\nПосле проверки появится подтверждение.',
                keyboard([[('← Назад', 'gt:menu')]]))
        elif data.startswith('gt:confirm:'):
            token = data.removeprefix('gt:confirm:')
            preview = self.store.get_setting(f'trends_preview:{actor}')
            if not preview or preview['token'] != token or preview['expires'] < utcnow():
                raise ValueError('Подтверждение устарело. Загрузите файл заново через /trends.')
            identifier = self.store.enqueue('trends_import', {'actor_id': actor, 'csv': preview['csv']}, f'trends-import:{actor}:{token}')
            self.store.set_setting(f'trends_preview:{actor}', None)
            screen('gt:result', 'gt:menu')
            self.reply(client, actor, f'Импорт {identifier} в очереди. После сохранения придёт результат. Затем откройте графики.',
                        keyboard([[('Google Trends', 'gt:menu')], [('Графики', 'ch:menu:trends')]]))
        else:
            raise ValueError('Неизвестная команда Google Trends.')

    def input(self, client, actor, text):
        pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        if not pending or pending['pending'] not in {'trends_time', 'trends_geo'}:
            return False
        require_actor(self.store, actor, admin=True)
        config = self.store.get_setting('trends_schedule', {})
        if pending['pending'] == 'trends_time':
            from .scheduling import validate_subscription
            times = text.strip().split(',')
            if not 1 <= len(times) <= 2:
                raise ValueError('До двух запусков в сутки: HH:MM или HH:MM,HH:MM.')
            config.update(validate_subscription({'times': times}))
        else:
            from .trends_api import validate_geo
            config['geo'] = validate_geo('' if text.strip().upper() == 'WORLD' else text.strip().upper())
        config.update(actor_id=actor, last_scheduled_at=None, created_at=utcnow())
        self.store.set_setting('trends_schedule', config)
        from .telegram_market import MarketControls
        MarketControls(self.store, self.reply).pending(actor, None)
        self.callback(client, actor, 'gt:auto')
        return True

    def document(self, client, message, actor):
        require_actor(self.store, actor, admin=True)
        pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        if not pending or pending['pending'] != 'trends_upload':
            self.reply(client, actor, 'Для загрузки CSV откройте /trends → «Загрузить CSV из Google».')
            return
        document = message.document
        size = getattr(document, 'file_size', None)
        if (type(size) is not int or not 0 < size <= google_trends.MAX_BYTES
                or not str(getattr(document, 'file_name', '')).lower().endswith('.csv')):
            raise ValueError('Отправьте CSV-файл размером до 1 МБ.')
        stream = client.download_media(message, in_memory=True)
        if not stream or not hasattr(stream, 'read'):
            raise ValueError('Не удалось скачать файл. Повторите загрузку.')
        try:
            stream.seek(0)
            raw = stream.read(google_trends.MAX_BYTES + 1)
        finally:
            stream.close()
        require_actor(self.store, actor, admin=True)
        parsed = google_trends.parse_csv(raw, [r['phrase'] for r in self.store.keywords()])
        token = uuid4().hex
        self.store.set_setting(f'trends_preview:{actor}', {'csv': base64.b64encode(raw).decode('ascii'),
                               'token': token, 'expires': (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()})
        from .telegram_market import MarketControls
        MarketControls(self.store, self.reply).pending(actor, None)
        screen('gt:preview', 'gt:menu')
        text = 'Проверьте импорт\nЧастота: ' + parsed['frequency']
        for column in parsed['columns']:
            rows = column['rows']
            text += f"\n{column['phrase']} · {column['region']}: {rows[0]['date']} — {rows[-1]['period_end']}; значений {sum(r['value'] is not None for r in rows)}/{len(rows)}."
        text += '\n\nНули, <1 и незаконченные периоды отмечаются как пропуски.\n' + google_trends.NOTE
        text += '\nПрежняя версия будет сохранена. Её правила уведомлений отключатся; выбранные пары корреляций перейдут на новую версию.'
        self.reply(client, actor, text, keyboard([[('Подтвердить импорт', 'gt:confirm:' + token)], [('Отмена', 'gt:menu')]]))
