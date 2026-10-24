"""Private-chat controls; every message and callback rechecks the allowlist."""

import json
import logging
from threading import Lock
from uuid import uuid4

from functions.telegram_utils import split_message

from .delivery import resolve_uncertain
from .monitoring import mark_error
from .scheduling import seed_subscriptions, update_subscription, validate_filters
from .search_provider import NOTICE, SEARCH_SECTIONS, disabled, hidden_route
from .store import SECTIONS
from .subscription_access import (
    allowed,
    command_route,
    project_allowed,
    require,
    require_route,
    route_allowed,
)
from .telegram_admin import BUDGET_HELP, AdminControls, usage_summary
from .telegram_admin import COMMANDS as ADMIN_COMMANDS
from .telegram_admin import HELP as ADMIN_HELP
from .telegram_cbr import CbrControls
from .telegram_chart_schedules import ChartScheduleControls
from .telegram_charts import ChartControls
from .telegram_data import DATA_ADMIN_HELP, DATA_COMMANDS, DATA_HELP, DataControls
from .telegram_dialogs import DialogControls
from .telegram_dollar import DollarControls
from .telegram_filters import FilterControls
from .telegram_integrity import IntegrityControls
from .telegram_life import LifeControls
from .telegram_market import COMMANDS as MARKET_COMMANDS
from .telegram_market import HELP as MARKET_HELP
from .telegram_market import MarketControls
from .telegram_navigation import CURRENT, interaction, normalize_buttons, screen
from .telegram_projects import PROJECTS, ProjectControls
from .telegram_trends import TrendsControls
from .telegram_vivino import VivinoControls
from .telegram_wines import WineControls

logger = logging.getLogger(__name__)
COMMANDS = ("start", "menu", "projects", "life", "expenses", "receipt", "wines", "cellar", "vivino", "dollar", "settings", "cancel", "help", "status", "report", "chart", "graphs", "notifications", "trends", "subscriptions", "regions", "keywords",
            "allow", "deny", "resolve", "integrity", *sorted(ADMIN_COMMANDS), *sorted(DATA_COMMANDS), *sorted(MARKET_COMMANDS))
HELP = (
    "/life — жизнь: вино, доходы, расходы и чеки\n"
    "/expenses — личный учёт доходов и расходов\n"
    "/receipt — распознать чек и проверить покупки\n"
    "/wines — вина: Vivino и личная винотека\n"
    "/cellar — запасы бутылок, покупки и стоимость\n"
    "/dollar — доллар для поездки: сейчас, частями или позже\n"
    "/integrity — сверка истории и исправления (администратор)\n"
    "/trends — Google Trends: данные и импорт CSV администратором\n"
    "/chart — графики: показатель, период и горизонт прогноза кнопками\n"
    "/status — состояние ваших доставок\n"
    "/report [ruonia|ai|wordstat|regions] [brief|full] — запросить отчёт\n"
    "/subscriptions — расписания (по умолчанию Москва)\n"
    "/subscriptions раздел on|off [08:30,18:30] — включить или выключить\n"
    "/subscriptions раздел detail brief|full — объём отчёта\n"
    "/subscriptions раздел weekdays 0,1,2,3,4 — дни (0 — понедельник)\n"
    "/subscriptions раздел timezone Europe/Moscow — часовой пояс\n"
    "/regions — выбранные регионы\n"
    "/regions mine 213,2 — ваши регионы\n"
    "/keywords — список фраз\n"
    "/keywords set wordstat фраза|другая фраза — ваши фразы\n"
    "/keywords set ruonia RUONIA|1 мес — ваши показатели ЦБ\n"
    "/help — эта справка"
)
def _private_user(event, callback=False):
    user = getattr(event, "from_user", None)
    message = getattr(event, "message", None) if callback else event
    chat = getattr(message, "chat", None)
    kind = getattr(chat, "type", None)
    kind = getattr(kind, "value", kind)
    user_id = getattr(user, "id", None)
    if (type(user_id) is not int or user_id <= 0 or kind != "private"
            or getattr(chat, "id", None) != user_id):
        return None
    return user_id


def _menu_rows(admin=False):
    from pyrogram.types import InlineKeyboardButton

    rows = [[InlineKeyboardButton(item.title + ('' if item.ready else ' · подготовка'), callback_data=item.route)]
            for item in PROJECTS]
    rows += [
        [InlineKeyboardButton("Состояние", callback_data="menu:status"),
         InlineKeyboardButton("Как пользоваться", callback_data="menu:guide")],
    ]
    if admin:
        rows.append([InlineKeyboardButton("Управление сервисом", callback_data="menu:admin")])
    return rows


def _reply(client, user_id, text, reply_markup=None, *, menu=True, admin=False):
    from pyrogram.enums import ParseMode
    from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    if menu:
        rows = list(reply_markup.inline_keyboard) if reply_markup else []
        if not any(button.callback_data == 'menu:home' for row in rows for button in row):
            rows.append([InlineKeyboardButton('⌂ Проекты', callback_data='menu:home')])
        reply_markup = InlineKeyboardMarkup(rows)

    chunks = split_message(text)
    for index, chunk in enumerate(chunks):
        options = {"reply_markup": reply_markup} if index == len(chunks) - 1 and reply_markup else {}
        client.send_message(user_id, chunk, parse_mode=ParseMode.DISABLED,
                            disable_web_page_preview=True, **options)


def _ids(text):
    try:
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError:
        raise ValueError("Нужны числовые ID через запятую") from None
    if not values or any(value <= 0 for value in values):
        raise ValueError("Нужны положительные числовые ID")
    return sorted(set(values))


class BotControls:
    def __init__(self, store):
        self.store = store
        self._ui_locks, self._ui_lock_guard = {}, Lock()
        self.admin_controls = AdminControls(store, self._reply)
        self.data_controls = DataControls(store, self._reply)
        self.market_controls = MarketControls(store, self._reply)
        self.chart_controls = ChartControls(store, self._reply)
        self.chart_schedule_controls = ChartScheduleControls(self)
        self.trends_controls = TrendsControls(store, self._reply)
        self.filter_controls = FilterControls(store, self._reply)
        self.dialog_controls = DialogControls(self)
        self.cbr_controls = CbrControls(self)
        self.project_controls = ProjectControls(self)
        self.vivino_controls = VivinoControls(self)
        self.wine_controls = WineControls(self)
        self.life_controls = LifeControls(self)
        self.integrity_controls = IntegrityControls(self)
        self.dollar_controls = DollarControls(self)

    def _reply(self, client, user_id, text, reply_markup=None, *, menu=True):
        from pyrogram.enums import ParseMode
        from pyrogram.errors import (
            MessageAuthorRequired,
            MessageEditTimeExpired,
            MessageIdInvalid,
            MessageNotModified,
        )
        from pyrogram.types import InlineKeyboardMarkup

        state = CURRENT.get()
        rows = list(reply_markup.inline_keyboard) if reply_markup else []
        rows = [[button for button in row if not button.callback_data or route_allowed(self.store, user_id, button.callback_data)] for row in rows]
        rows = [row for row in rows if row]
        if disabled(self.store):
            rows = [[button for button in row if not hidden_route(button.callback_data)] for row in rows]
            rows = [row for row in rows if row]
        route = state['route'] if state else 'menu:home'
        pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (user_id,))
        back = state['back'] if state else 'menu:home'
        rows = normalize_buttons(rows, pending=pending['pending'] if pending else None,
                                 back=back if route_allowed(self.store, user_id, back) else None,
                                 route=route) if menu else normalize_buttons(rows)
        markup = InlineKeyboardMarkup(rows) if rows else None
        chunks = split_message(text)
        # Long immutable content stays separate; editable menus remain short.
        if len(chunks) > 1:
            for chunk in chunks[:-1]:
                client.send_message(user_id, chunk, parse_mode=ParseMode.DISABLED, disable_web_page_preview=True)
        target = state.get('message_id') if state else None
        options = {'parse_mode': ParseMode.DISABLED, 'disable_web_page_preview': True, 'reply_markup': markup}
        if target:
            try:
                client.edit_message_text(user_id, target, chunks[-1], **options)
                identifier = target
            except MessageNotModified:
                identifier = target
            except (MessageIdInvalid, MessageEditTimeExpired, MessageAuthorRequired):
                identifier = getattr(client.send_message(user_id, chunks[-1], **options), 'id', None)
        else:
            identifier = getattr(client.send_message(user_id, chunks[-1], **options), 'id', None)
        if type(identifier) is int and identifier > 0:
            from .monitor_content import redact_text, response_details
            summary = redact_text(text,2000)
            self.store.set_setting(f'ui_screen:{user_id}', {'message_id': identifier, 'route': route,
                                   'back': state['back'] if state else 'menu:home',
                                   'summary':summary,
                                   'button_labels':{button.callback_data:redact_text(button.text,300) for row in rows for button in row if button.callback_data},
                                   'buttons': [button.callback_data for row in rows for button in row if button.callback_data]})
            if state:
                state['message_id'] = identifier
                state['monitor_response'] = response_details(text,identifier)
                from .monitoring import response_jobs
                response_jobs(self.store,user_id,text)

    def _admin(self, user_id):
        if not self.store.authorize(user_id, admin=True):
            raise PermissionError("Действие доступно администратору")

    def _subscription(self, user_id, section):
        require(self.store, user_id, section)
        if section not in SECTIONS:
            raise ValueError("Разделы: ruonia, ai, wordstat, regions")
        seed_subscriptions(self.store, str(user_id), user_id)
        return self.store.one(
            "SELECT * FROM subscriptions WHERE user_id=? AND recipient=? AND section=?",
            (user_id, str(user_id), section),
        )

    @interaction()
    def handle_message(self, client, message):
        user_id = _private_user(message)
        if user_id is None:
            return
        from .user_management import remember
        remember(self.store, getattr(message, 'from_user', None))
        if not self.store.authorize(user_id):
            _reply(client, user_id, f"Нет доступа. Обратитесь к администратору. Ваш Telegram ID: {user_id}", menu=False)
            return
        tokens = getattr(message, "text", "").strip().split(maxsplit=1)
        if not tokens:
            return
        command = tokens[0].lstrip("/").split("@", 1)[0].lower()
        argument = tokens[1].strip() if len(tokens) > 1 else ""
        try:
            if not getattr(message, 'text', '').strip().startswith('/'):
                pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (user_id,))
                if pending and pending['pending']:
                    from .feature_access import require as require_feature
                    value = pending['pending']
                    feature = ('life_finance' if value.startswith('life:') else
                               'wine_cellar' if value.startswith('wine:') else
                               'dollar' if value.startswith('dollar:') else
                               'cbr' if value == 'cbrsearch' else
                               'schedules' if value.startswith(('chartdelivery:', 'dialog:schedule:', 'ui:times:', 'ui:weekdays:', 'ui:timezone:')) else
                               'charts' if value.startswith(('chart:', 'charthorizon:', 'chartsearch:')) else
                               'vivino_plan' if value.startswith(('vivino_grapes:', 'vivino_search:', 'vivino_custom:')) else
                               'alerts' if value.startswith(('dialog:ruleparam:', 'dialog:threshold:', 'dialog:digest:')) else None)
                    if feature:
                        require_feature(self.store, user_id, feature)
            restricted_route = command_route(command, argument)
            if restricted_route:
                require_route(self.store, user_id, restricted_route)
            search_command = (command in {'trends', 'regions', 'keywords', 'propose'}
                or command == 'alerts' and not argument.startswith('market')
                or command in {'chart', 'report', 'export', 'backtest', 'forecasts', 'subscriptions'}
                and argument.split()[:1] and argument.split()[0] in SEARCH_SECTIONS
                or command in {'backtest', 'forecasts'} and not argument)
            if disabled(self.store) and search_command:
                self._search_disabled(client, user_id)
                return
            if getattr(message, 'text', '').strip().startswith('/'):
                self.life_controls.leave(user_id)
                if self.store.get_setting(f'wine_draft:{user_id}') is not None:
                    self.wine_controls._leave(user_id)
                self.market_controls.pending(user_id, None)
            if command == 'cancel':
                self._cancel_form(client, user_id)
                return
            if command in {'start', 'menu', 'projects'}:
                if command == 'start':
                    from .store import utcnow
                    from .telegram_commands import configure_chat
                    self.store.execute("INSERT INTO bot_sessions(user_id,started_at,last_seen) VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET started_at=COALESCE(bot_sessions.started_at,excluded.started_at),last_seen=excluded.last_seen", (user_id, utcnow(), utcnow()))
                    configure_chat(client, self.store, user_id)
                    from .user_management import release_notices, send_notices
                    release_notices(self.store, user_id)
                    send_notices(self.store, client, user_id, force=True)
                self._home(client, user_id)
                return
            if command == 'vivino':
                self.vivino_controls.menu(client, user_id)
                return
            if command in {'life', 'expenses', 'receipt'}:
                if command == 'life':
                    self.project_controls.home(client, user_id, 'life')
                elif command == 'expenses':
                    self.life_controls.finance(client, user_id)
                else:
                    self.life_controls.receipt_prompt(client, user_id)
                return
            if command == 'wines':
                self.project_controls.home(client, user_id, 'vivino')
                return
            if command == 'cellar':
                self.wine_controls.menu(client, user_id)
                return
            if command == 'settings':
                self._settings_menu(client, user_id)
                return
            if self.life_controls.input(client, user_id, getattr(message, 'text', '').strip()):
                return
            if self.wine_controls.input(client, user_id, getattr(message, 'text', '').strip()):
                return
            if self.vivino_controls.atlas_controls.input(client, user_id, getattr(message, 'text', '').strip()):
                return
            pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (user_id,))
            if pending and pending['pending'] == 'vivino_upload':
                screen('vi:upload', 'vi:menu')
                self._reply(client, user_id, 'Отправьте ZIP или CSV файлом. /cancel отменяет загрузку.')
                return
            if self._form_input(client, user_id, getattr(message, 'text', '').strip()):
                return
            if self.dollar_controls.input(client, user_id, getattr(message, 'text', '').strip()):
                return
            if self.chart_schedule_controls.input(client, user_id, getattr(message, 'text', '').strip()):
                return
            if self.dialog_controls.input(client, user_id, getattr(message, 'text', '').strip()):
                return
            if self.cbr_controls.input(client, user_id, getattr(message, 'text', '').strip()):
                return
            if self.chart_controls.input(client, user_id, getattr(message, 'text', '').strip()):
                return
            if self.integrity_controls.input(client, user_id, getattr(message, 'text', '').strip()):
                return
            if command == 'integrity':
                self.integrity_controls.menu(client, user_id)
                return
            if command == 'dollar':
                self.dollar_controls.menu(client, user_id)
                return
            if command == 'chart':
                self.chart_controls.dispatch(client, user_id, argument)
                return
            if command == 'graphs':
                self.chart_schedule_controls.menu(client, user_id)
                return
            if command == 'notifications':
                self.market_controls.alerts(client, user_id, '')
                return
            if command == 'trends':
                self.trends_controls.menu(client, user_id)
                return
            if self.trends_controls.input(client, user_id, getattr(message, 'text', '').strip()):
                return
            pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (user_id,))
            if pending and pending['pending'] == 'trends_upload':
                self._reply(client, user_id, 'Отправьте CSV файлом. Отмена: /cancel.')
                return
            if self.market_controls.input(client, user_id, getattr(message, 'text', '').strip()):
                return
            if command == 'help':
                self._guide(client, user_id)
            elif command == "status":
                self._status(client, user_id)
            elif command == "report":
                self._report(client, user_id, argument, getattr(message, "id", None))
            elif command == "subscriptions":
                self._subscriptions(client, user_id, argument)
            elif command == "regions":
                self._regions(client, user_id, argument)
            elif command == "keywords":
                self._keywords(client, user_id, argument)
            elif command in MARKET_COMMANDS or (command in {'forecasts', 'backtest', 'export', 'alerts'} and argument.split()[:1] == ['market']):
                self.market_controls.dispatch(client, user_id, command, argument, getattr(message, 'id', None))
            elif command in ADMIN_COMMANDS:
                self.admin_controls.dispatch(client, user_id, command, argument, getattr(message, "id", None))
            elif command in DATA_COMMANDS:
                self.data_controls.dispatch(client, user_id, command, argument, getattr(message, "id", None))
            elif command in {"allow", "deny", "resolve"}:
                self._admin_command(client, user_id, command, argument)
            else:
                mark_error(ValueError())
                self._reply(client, user_id, "Неизвестная команда. Справка: /help")
        except (ValueError, PermissionError) as exc:
            mark_error(exc)
            self._reply(client, user_id, str(exc))
        except Exception as exc:  # noqa: BLE001 - Bot command boundary isolates failures and redacts provider details.
            mark_error(exc)
            logger.error("Bot command failed (%s)", type(exc).__name__)
            self._reply(client, user_id, "Не удалось выполнить команду. Попробуйте позже.")

    def _help(self, client, user_id):
        from .telegram_charts import keyboard
        screen('menu:help')
        rows = [[('Показатели ЦБ · с чего начать', 'cb:guide')],
                [('Графики и периоды', 'ui:help:charts')], [('Данные и прогнозы', 'ui:help:data')],
                [('Подписки и настройки', 'ui:help:settings')], [('Список команд', 'ui:help:commands')]]
        if self.store.authorize(user_id, admin=True):
            rows.append([('Администратору', 'ui:help:admin')])
        self._reply(client, user_id, 'Помощь\nВыберите тему. Настройки меняются кнопками; команды остаются доступны для быстрого доступа. /menu открывает главное меню, /cancel отменяет ввод.', keyboard(rows))

    def _home(self, client, user_id, notice=''):
        from pyrogram.types import InlineKeyboardMarkup
        self.wine_controls._leave(user_id)
        self.life_controls.leave(user_id)
        screen('menu:home')
        projects = [item for item in PROJECTS if project_allowed(self.store, user_id, item.id)]
        text = 'Главное меню · проекты\nДоступны проекты и разделы, которые подключил администратор. Внутри них вы настраиваете личные графики и расписания.\n\n'
        text += '\n'.join(item.title + ' — ' + item.summary for item in projects) if projects else 'Пока нет подключённых проектов. Обратитесь к администратору.'
        self._reply(client, user_id, (notice + '\n\n' if notice else '') + text,
                    InlineKeyboardMarkup(_menu_rows(self.store.authorize(user_id, admin=True))))

    def _guide(self, client, user_id):
        from .telegram_charts import keyboard
        screen('menu:guide')
        self._reply(client, user_id, 'Как пользоваться ботом\n\n'
            'Сначала выберите проект, затем задачу. «← Назад» возвращает на уровень выше, '
            '«⌂ Проекты» — к выбору инструментов. /cancel отменяет ввод.\n\n'
            'Рынок и экономика: данные ЦБ, RUONIA, графики, прогнозы и уведомления. '
            'Жизнь: вино, доходы, расходы и покупки по чекам. Вино — личная винотека и Vivino: карта, регионы, сорта и план дегустаций.\n\n'
            '/projects — список проектов; /chart — быстрый переход к рыночным графикам; '
            '/life — жизнь; /expenses — доходы и расходы; /receipt — загрузить чек; '
            '/wines — раздел вин; /cellar — моя винотека; /vivino — карта и отзывы. Старые команды рынка продолжают работать.\n\n'
            'Смена проекта не выключает ваши действующие расписания. '
            'Доступ и состояние сервиса общие для всего бота.', keyboard([
                [('Помощь по рынку', 'menu:help')],
                [('Рынок и экономика', 'pj:market'), ('Жизнь', 'pj:life')]]))

    def _search_disabled(self, client, actor):
        from .telegram_charts import keyboard
        self.market_controls.pending(actor, None)
        screen('search:disabled', 'pj:market')
        self._reply(client, actor, NOTICE, keyboard([[('Показатели Банка России', 'cb:menu')]]))

    def _settings_menu(self, client, user_id):
        from .telegram_charts import keyboard
        screen('menu:settings')
        self._reply(client, user_id, 'Рынок и экономика · мои настройки\nВыбор здесь действует только для вас. Расписание само не включается при выборе показателей.', keyboard([
            [('Подписки и время', 'menu:subscriptions')], [('Показатели рынка', 'menu:indicators')],
            [('Графики по расписанию', 'menu:graphs')],
            [('Регионы', 'menu:regions'), ('Поисковые фразы', 'menu:keywords')],
            [('Рыночные уведомления', 'mk:open:alerts')]]))

    def _data_menu(self, client, user_id):
        from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        screen('menu:data')
        rows = [
            [InlineKeyboardButton('Показатели Банка России · с чего начать', callback_data='cb:guide')],
        ] if disabled(self.store) else []
        self._reply(client, user_id, 'Аналитика и прогнозы\nВыберите задачу. Прогнозы используют загруженную историю; обновление данных — в «Отчётах».', InlineKeyboardMarkup(rows + [
            [InlineKeyboardButton('Доллар для поездки · USD/RUB', callback_data='usd:menu')],
            [InlineKeyboardButton('Показатели Банка России · все базы', callback_data='cb:menu')],
            [InlineKeyboardButton('Графики и сравнение показателей', callback_data='ch:menu:market')],
            [InlineKeyboardButton('Обзор рынка', callback_data='mk:menu:all'),
             InlineKeyboardButton('Корреляции', callback_data='mk:open:correlations')],
            [InlineKeyboardButton("Google Trends" if self.store.get_setting('search_provider') == 'google_trends' else "Прогнозы Wordstat",
                                  callback_data="gt:menu" if self.store.get_setting('search_provider') == 'google_trends' else "menu:forecasts"),
             InlineKeyboardButton("Прогнозы RUONIA", callback_data="menu:ruonia_forecasts")],
            [InlineKeyboardButton('Прогнозы рынка', callback_data='ui:forecasts:market')],
            [InlineKeyboardButton('Проверить прогноз на истории', callback_data='ui:backtest:menu')],
            [InlineKeyboardButton('Экспорт данных', callback_data='ui:export:menu')],
        ]))

    def _verify_help(self, client, user_id):
        self._admin(user_id)
        self._reply(client, user_id, DATA_ADMIN_HELP.strip())

    def _report_menu(self, client, user_id):
        from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        labels = {"ruonia": "RUONIA", "ai": "Новости и AI", "wordstat": "Поисковые индикаторы", "regions": "По регионам", "market": "Рынок"}
        screen('menu:report')
        choices = [[InlineKeyboardButton('Получить: ' + labels[section], callback_data=f"request:{section}")] for section in SECTIONS]
        from .google_trends import active
        if disabled(self.store):
            choices = [row for row in choices if row[0].callback_data not in {'request:wordstat', 'request:regions'}]
        if active(self.store):
            choices = [row for row in choices if row[0].callback_data not in {'request:wordstat', 'request:regions'}]
            choices.append([InlineKeyboardButton('Google Trends (сохранённые данные)', callback_data='ch:menu:trends')])
        choices += [[InlineKeyboardButton('История выпусков', callback_data='menu:history')]]
        self._reply(client, user_id, 'Отчёты\nВыберите раздел для обновления данных и получения выпуска. Результат придёт отдельно. Для графика за свой период откройте «Графики».', InlineKeyboardMarkup(choices))

    def _status(self, client, user_id):
        rows = self.store.query(
            "SELECT state,COUNT(*) AS n FROM deliveries WHERE recipient=? GROUP BY state", (str(user_id),)
        )
        labels = {'pending': 'в очереди', 'running': 'в работе', 'done': 'готово', 'sent': 'доставлено',
                  'failed': 'ошибки', 'sending': 'отправляется', 'uncertain': 'нужна проверка', 'cancelled': 'отменено'}
        text = 'Состояние\nОтправка сообщений: ' + (', '.join(f"{labels.get(r['state'], r['state'])}: {r['n']}" for r in rows) or 'пока сообщений нет')
        if self.store.authorize(user_id, admin=True):
            jobs = self.store.query("SELECT state,COUNT(*) AS n FROM jobs GROUP BY state")
            uncertain = self.store.query("SELECT id,recipient FROM deliveries WHERE state='uncertain' ORDER BY id LIMIT 20")
            text += "\nЗадания: " + (", ".join(f"{labels.get(row['state'], row['state'])}: {row['n']}" for row in jobs) or "пока нет")
            text += '\nЗадание — сбор или расчёт; отправка — отдельные сообщения и файлы.'
            text += "\n\n" + usage_summary(self.store)
            if uncertain:
                text += "\nТребуют решения: " + ", ".join(f"{r['id']} → {r['recipient']}" for r in uncertain)
        else:
            jobs = self.store.query("SELECT state,COUNT(*) AS n FROM jobs WHERE json_extract(payload,'$.actor_id')=? OR json_extract(payload,'$.recipient')=? GROUP BY state", (user_id, str(user_id)))
            text += '\nВаши задания: '+(', '.join(f"{labels.get(row['state'], row['state'])}: {row['n']}" for row in jobs) or 'пока нет')
            text += '\nРезультаты расчётов и файлы придут отдельными сообщениями.'
        from .telegram_charts import keyboard
        route = (CURRENT.get() or {}).get('route', 'menu:status')
        rows = [[('Обновить состояние', route)]]
        if self.store.authorize(user_id, admin=True):
            rows.append([('Очередь и ошибки', 'menu:jobs')])
        self._reply(client, user_id, text, keyboard(rows))

    def _report(self, client, user_id, argument, message_id):
        args = argument.split()
        if not args:
            self._report_menu(client, user_id)
            return
        if len(args) > 2:
            raise ValueError("/report [раздел] [brief|full]")
        section = args[0] if args else "ruonia"
        from .google_trends import active
        if section == 'trends' or (section in {'wordstat', 'regions'} and active(self.store)):
            self.trends_controls.menu(client, user_id)
            return
        subscription = self._subscription(user_id, section)
        detail = args[1] if len(args) > 1 else subscription["detail"]
        if detail not in {"brief", "full"}:
            raise ValueError("Выберите brief или full")
        selected = validate_filters(self.store, section, json.loads(subscription["phrases"]),
                                    json.loads(subscription["regions"]))
        self.store.enqueue("report", {
            "section": section, "recipient": str(user_id), "detail": detail,
            **selected,
        }, dedupe_key=f"telegram:{user_id}:{message_id}" if message_id is not None else None)
        screen('request:result', 'menu:report')
        if section == 'wordstat':
            from .telegram_charts import keyboard
            self._reply(client, user_id, "Запрос принят. Недельная история будет загружена через Wordstat API с учётом дневного лимита и кэша. Отчёт с результатом или причиной ошибки придёт отдельно. После него откройте графики.",
                        keyboard([[('Вернуться к графикам Wordstat', 'ch:menu:wordstat')]]))
        else:
            self._reply(client, user_id, "Запрос принят. Отчёт придёт отдельным сообщением.")

    def _ui_callback(self, client, actor, data):
        from .telegram_charts import keyboard
        from .telegram_navigation import checked
        parts = data.split(':')
        if data == 'ui:cancel':
            self._cancel_form(client, actor)
        elif len(parts) == 3 and parts[1] == 'help':
            topic = parts[2]
            texts = {
                'charts': 'Графики\nВыберите раздел → показатель → период → прогноз → «Построить». ✓ отмечает текущий выбор. «Свои даты» принимает две даты YYYY-MM-DD. Экспорт использует тот же период. Графики не загружают новые данные: сначала получите отчёт.',
                'data': 'Данные и прогнозы\nОтчёт обновляет источник. График отображает сохранённые наблюдения. Прогнозы могут отсутствовать при нехватке истории или устаревших данных. Корреляции показывают связь показателей, а не причину.\n' + DATA_HELP,
                'settings': 'Мои настройки\nПодписки: выберите раздел, время, дни и формат, затем включите доставку.\nРегионы и фразы ограничивают ваши отчёты. Рыночные показатели выбираются отдельно. Настройки других пользователей не меняются.\n' + HELP,
                'commands': HELP + DATA_HELP + MARKET_HELP,
                'admin': ADMIN_HELP + DATA_ADMIN_HELP + BUDGET_HELP,
            }
            if topic not in texts:
                raise ValueError('Неизвестная тема помощи')
            if topic == 'admin':
                self._admin(actor)
            screen(data, 'menu:help')
            text = texts[topic]
            if disabled(self.store):
                text = '\n'.join(line for line in text.splitlines() if not any(word in line for word in ('Wordstat', 'wordstat', '/trends', '/regions', '/keywords', 'Регионы и фразы')))
                text += '\nПоказатели ЦБ: «Рынок и экономика → Показатели Банка России».\n/report ruonia|market|ai — обновление выбранного раздела; /backtest market — проверка рынка.'
            self._reply(client, actor, text)
        elif data == 'ui:forecasts:market':
            self.dialog_controls.forecast_menu(client, actor)
        elif len(parts) == 3 and parts[1] == 'backtest':
            screen(data, 'menu:data' if parts[2] == 'menu' else 'ui:backtest:menu')
            if parts[2] == 'menu':
                self._reply(client, actor, 'Проверка прогнозов на истории\nВыберите данные. Проверка выполняется локально, по последним 12 точкам, без реальной рассылки.', keyboard([
                    [('RUONIA', 'ui:backtest:ruonia'), ('Wordstat', 'ui:backtest:wordstat')], [('Рынок', 'ui:backtest:market')]]))
            elif parts[2] == 'market':
                self.market_controls.dispatch(client, actor, 'backtest', 'market', uuid4().hex)
            elif parts[2] in {'ruonia', 'wordstat'}:
                self.data_controls.dispatch(client, actor, 'backtest', parts[2], uuid4().hex)
            else:
                raise ValueError('Неизвестные данные проверки')
        elif len(parts) in {3, 4} and parts[1] == 'export':
            from .subscription_access import allowed
            choices = {'market': 'Рынок и ЦБ', 'ruonia': 'RUONIA', 'wordstat': 'Wordstat', 'regions': 'Регионы'}
            if len(parts) == 3 and parts[2] == 'menu':
                screen(data, 'menu:data')
                self._reply(client, actor, 'Экспорт данных\nВыберите раздел. Будет использована сохранённая история; новые данные не загружаются.',
                            keyboard([[(title, 'ui:export:'+section)] for section, title in choices.items() if allowed(self.store, actor, section)]))
            elif parts[2] in choices and len(parts) == 3:
                screen(data, 'ui:export:menu')
                self._reply(client, actor, 'Экспорт · '+choices[parts[2]]+'\nВыберите формат для всей доступной истории. Свой период можно указать командой /export '+parts[2]+' csv YYYY-MM-DD YYYY-MM-DD.',
                            keyboard([[('CSV', data+':csv'), ('Excel', data+':xlsx')]]))
            elif parts[2] in choices and len(parts) == 4 and parts[3] in {'csv', 'xlsx'}:
                controls = self.market_controls if parts[2] == 'market' else self.data_controls
                controls.dispatch(client, actor, 'export', parts[2]+' '+parts[3], uuid4().hex)
            else:
                raise ValueError('Неизвестный раздел или формат экспорта')
        elif len(parts) >= 3 and parts[1] == 'subscription':
            section = parts[2]
            row = self._subscription(actor, section)
            if len(parts) == 5 and parts[3] == 'detail' and parts[4] in {'brief', 'full'}:
                update_subscription(self.store, actor, section, detail=parts[4])
                row = self._subscription(actor, section)
            elif len(parts) == 5 and parts[3] == 'enabled' and parts[4] in {'on', 'off'}:
                update_subscription(self.store, actor, section, enabled=parts[4] == 'on')
                row = self._subscription(actor, section)
            elif len(parts) == 5 and parts[3] == 'days' and parts[4] in {'work', 'all'}:
                update_subscription(self.store, actor, section, weekdays=list(range(5 if parts[4] == 'work' else 7)))
                row = self._subscription(actor, section)
            elif len(parts) == 5 and parts[3] == 'day' and parts[4] in {str(i) for i in range(7)}:
                day = int(parts[4])
                selected = set(json.loads(row['weekdays']))
                selected.symmetric_difference_update({day})
                if not selected:
                    raise ValueError('Оставьте хотя бы один день. Для остановки доставки выключите подписку.')
                update_subscription(self.store, actor, section, weekdays=sorted(selected))
                row = self._subscription(actor, section)
            elif len(parts) == 4 and parts[3] in {'times', 'timezone'}:
                screen(f'ui:subscriptioninput:{section}', f'ui:subscription:{section}')
                self.market_controls.pending(actor, f'ui:{parts[3]}:{section}')
                self._reply(client, actor, 'Введите время через запятую, например 08:30,18:30.' if parts[3] == 'times' else 'Введите часовой пояс, например Europe/Moscow.')
                return
            elif len(parts) != 3:
                raise ValueError('Неверная настройка подписки')
            labels = {'ruonia': 'RUONIA', 'market': 'Рынок', 'wordstat': 'Wordstat', 'regions': 'Регионы', 'ai': 'Новости и AI'}
            weekdays = json.loads(row['weekdays'])
            screen(f'ui:subscription:{section}', 'menu:subscriptions')
            self._reply(client, actor, f"Подписка · {labels[section]}\nДоставка: {'включена' if row['enabled'] else 'выключена'}\nВремя: {', '.join(json.loads(row['times']))} · {row['timezone']}\nДни: {', '.join(['Пн','Вт','Ср','Чт','Пт','Сб','Вс'][d] for d in weekdays)}\nФормат: {'краткий' if row['detail'] == 'brief' else 'полный'}", keyboard([
                [('Выключить доставку' if row['enabled'] else 'Включить доставку', f"ui:subscription:{section}:enabled:{'off' if row['enabled'] else 'on'}")],
                [('Время доставки', f'ui:subscription:{section}:times'), ('Часовой пояс', f'ui:subscription:{section}:timezone')],
                [(checked('Будни', weekdays == list(range(5))), f'ui:subscription:{section}:days:work'), (checked('Каждый день', weekdays == list(range(7))), f'ui:subscription:{section}:days:all')],
                [(checked(label, i in weekdays), f'ui:subscription:{section}:day:{i}') for i, label in enumerate(['Пн', 'Вт', 'Ср', 'Чт'])],
                [(checked(label, i in weekdays), f'ui:subscription:{section}:day:{i}') for i, label in enumerate(['Пт', 'Сб', 'Вс'], 4)],
                [(checked('Краткий', row['detail'] == 'brief'), f'ui:subscription:{section}:detail:brief'), (checked('Полный', row['detail'] == 'full'), f'ui:subscription:{section}:detail:full')]]))
        else:
            if not self.dialog_controls.callback(client, actor, data):
                raise ValueError('Неизвестный экран')

    def _form_input(self, client, actor, text):
        state = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        if not state or not (state['pending'] or '').startswith('ui:') or text.startswith('/'):
            return False
        _, field, section = state['pending'].split(':', 2)
        if field == 'times':
            update_subscription(self.store, actor, section, times=text.split(','))
        elif field == 'timezone':
            update_subscription(self.store, actor, section, timezone=text.strip())
        else:
            raise ValueError('Неизвестная форма ввода; /cancel')
        self.market_controls.pending(actor, None)
        self._ui_callback(client, actor, f'ui:subscription:{section}')
        return True

    def _cancel_form(self, client, actor):
        record = self.store.get_setting(f'ui_screen:{actor}', {})
        route = record.get('back') or 'menu:home'
        if not route_allowed(self.store, actor, route):
            in_finance = record.get('route', '').startswith(('lf:', 'rc:'))
            route = 'lf:home' if in_finance and route_allowed(self.store, actor, 'lf:home') else 'menu:home'
        self.life_controls.leave(actor)
        self.wine_controls._leave(actor)
        self.market_controls.pending(actor, None)
        self.store.set_setting(f'trends_preview:{actor}', None)
        self.store.set_setting(self.vivino_controls.key(actor, 'preview'), None)
        if route.startswith(('wc:', 'wn:')):
            self.wine_controls.callback(client, actor, route)
        elif route.startswith('lf:step:') or route == 'lf:entryreview':
            self.life_controls.finance(client, actor, 'Ввод отменён.')
        elif route.startswith(('lf:', 'rc:')):
            self.life_controls.callback(client, actor, route)
        elif route == 'pj:life':
            self.life_controls.home(client, actor)
        elif route == 'pj:vivino':
            self.project_controls.home(client, actor, 'vivino')
        elif route.startswith('va:'):
            self.vivino_controls.atlas_controls.callback(client, actor, route)
        elif route.startswith('usd:'):
            if route == 'usd:horizon':
                self.dollar_controls.horizon(client, actor, 'Ввод отменён.')
            elif route == 'usd:dates':
                self.dollar_controls.dates(client, actor, 'Ввод отменён.')
            elif route == 'usd:period':
                self.dollar_controls.period(client, actor, 'Ввод отменён.')
            else:
                self.dollar_controls.menu(client, actor, 'Ввод отменён.', advanced=route == 'usd:advanced')
        elif route.startswith('cb:'):
            self.cbr_controls.callback(client, actor, route)
        elif route.startswith('vi:'):
            self.vivino_controls.menu(client, actor)
        elif route.startswith('ch:'):
            self.chart_controls.callback(client, actor, route, None)
        elif route.startswith('ig:'):
            self.integrity_controls.callback(client, actor, route)
        elif route.startswith('gt:'):
            self.trends_controls.menu(client, actor)
        elif route.startswith('cs:'):
            self.chart_schedule_controls.callback(client, actor, route)
        elif route.startswith('ui:'):
            self._ui_callback(client, actor, route)
        elif route.startswith('mk:'):
            self.market_controls.callback(client, actor, route)
        elif route == 'menu:subscriptions':
            self._subscriptions(client, actor, '')
        elif route == 'menu:settings':
            self._settings_menu(client, actor)
        elif route == 'menu:sources':
            self.market_controls.sources(client, actor, '')
        elif route in {'menu:users', 'menu:budget'}:
            self.admin_controls.dispatch(client, actor, route.split(':')[1])
        else:
            self._home(client, actor, 'Ввод отменён.')

    def _subscriptions(self, client, user_id, argument):
        from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        if argument:
            args = argument.split()
            if len(args) < 2:
                raise ValueError("/subscriptions раздел on|off [HH:MM,HH:MM]")
            section, action = args[:2]
            if action in {"on", "off"} and len(args) <= 3:
                values = {"enabled": action == "on"}
                if len(args) == 3:
                    values["times"] = args[2].split(",")
            elif action == "detail" and len(args) == 3:
                values = {"detail": args[2]}
            elif action == "weekdays" and len(args) == 3:
                values = {"weekdays": [int(value) for value in args[2].split(",")]}
            elif action == "timezone" and len(args) == 3:
                values = {"timezone": args[2]}
            else:
                raise ValueError("Неверная настройка. Справка: /help")
            update_subscription(self.store, user_id, section, **values)
        seed_subscriptions(self.store, str(user_id), user_id)
        rows = self.store.query("SELECT * FROM subscriptions WHERE user_id=? AND recipient=? ORDER BY section",
                                (user_id, str(user_id)))
        rows = [row for row in rows if allowed(self.store, user_id, row['section'])]
        from .google_trends import active
        if active(self.store) or disabled(self.store):
            rows = [row for row in rows if row['section'] not in {'wordstat', 'regions'}]
        screen('menu:subscriptions', 'menu:settings')
        labels = {'ruonia': 'RUONIA', 'market': 'Рынок и Google Trends' if active(self.store) else 'Рынок', 'wordstat': 'Wordstat', 'regions': 'Регионы', 'ai': 'Новости и AI'}
        text = 'Мои подключённые разделы\nАдминистратор управляет доступом. Здесь настраиваются время, дни и формат ваших отчётов. ✓ — доставка включена. Изменение времени само не включает доставку.'
        if not rows:
            text += '\nРазделы для отчётов пока не подключены.'
        buttons = [[InlineKeyboardButton(
            ('✓ ' if row['enabled'] else '○ ') + labels[row['section']] + ' · ' + ', '.join(json.loads(row['times'])),
            callback_data=f"ui:subscription:{row['section']}",
        )] for row in rows]
        self._reply(client, user_id, text, InlineKeyboardMarkup(buttons))

    def _regions(self, client, user_id, argument):
        from .google_trends import active
        if not argument and active(self.store):
            self.trends_controls.menu(client, user_id)
            return
        if not argument:
            self.filter_controls.regions(client, user_id)
            return
        if argument:
            action, _, values = argument.partition(" ")
            if action not in {"set", "mine"} or not values:
                raise ValueError("/regions mine ID,ID; для общей когорты администратора: /regions set ID,ID")
            ids = _ids(values)
            if action == "set" and self.store.authorize(user_id, admin=True):
                self.store.set_regions(ids)
            else:
                allowed = set(self.store.regions())
                if not allowed or not set(ids).issubset(allowed):
                    raise ValueError("Выберите регионы из утверждённой администратором когорты")
                update_subscription(self.store, user_id, "regions", regions=ids)
        subscription = self._subscription(user_id, "regions")
        chosen = json.loads(subscription["regions"]) or self.store.regions()
        self._reply(client, user_id, "Ваши регионы: " + (", ".join(map(str, chosen)) or "когорта ещё не задана")
               + "\nОбщая когорта: " + (", ".join(map(str, self.store.regions())) or "не задана"))

    def _keywords(self, client, user_id, argument):
        from .google_trends import active
        if not argument and active(self.store):
            self.trends_controls.menu(client, user_id)
            return
        if not argument:
            self.filter_controls.keywords(client, user_id)
            return
        if argument:
            action, _, value = argument.partition(" ")
            if action == "set":
                section, _, raw = value.partition(" ")
                if section not in {"wordstat", "regions", "ruonia"} or not raw:
                    raise ValueError("/keywords set wordstat|regions|ruonia показатель|показатель")
                phrases = [phrase.strip() for phrase in raw.split("|") if phrase.strip()]
                from .cbr_data import METRICS
                allowed = (set(METRICS) if section == "ruonia" else
                           {row["phrase"] for row in self.store.keywords()} if section == "wordstat"
                           else set(self.store.get_setting("incident_phrases", [])))
                if not phrases or not set(phrases).issubset(allowed):
                    raise ValueError("Выберите фразы из утверждённого списка")
                update_subscription(self.store, user_id, section, phrases=phrases)
            else:
                self._admin(user_id)
                entries = self.store.keywords()
                if action == "add" and value:
                    phrase, _, theme = value.partition("|")
                    entries.append({"phrase": phrase.strip(), "theme": theme.strip() or "другое"})
                elif action == "remove" and value:
                    entries = [row for row in entries if row["phrase"] != value.strip()]
                else:
                    raise ValueError("/keywords add фраза|тема или /keywords remove фраза")
                self.store.set_keywords(entries)
        text = "Утверждённые фразы:\n" + "\n".join(
            f"• {row['phrase']} ({row['theme']})" for row in self.store.keywords()
        )
        text += "\nРегиональные: " + ", ".join(self.store.get_setting("incident_phrases", []))
        from .cbr_data import METRICS
        text += "\nПоказатели RUONIA: " + ", ".join(METRICS)
        self._reply(client, user_id, text)

    def _admin_command(self, client, user_id, command, argument):
        self._admin(user_id)
        args = argument.split()
        if command in {"allow", "deny"}:
            if len(args) != 1:
                raise ValueError(f"/{command} ID или /{command} @username")
            from .user_management import (
                label,
                resolve_profile,
                resolve_user_reference,
                send_notices,
            )
            target = resolve_user_reference(self.store, client, args[0])
            if command == "deny" and target == user_id:
                raise ValueError("Нельзя отключить собственный доступ этой командой")
            from .market_store import change_access
            change_access(self.store, user_id, target, command == 'allow')
            from .telegram_commands import configure_chat
            configure_chat(client, self.store, target)
            if not args[0].startswith('@'):
                resolve_profile(self.store, client, target)
            send_notices(self.store, client, target)
            if command == 'allow':
                from .telegram_charts import keyboard
                self._reply(client, user_id, f'Доступ разрешён: {label(self.store, target)}. Подключите нужные проекты и разделы; пользователь настраивает их самостоятельно.',
                            keyboard([[('Проекты и разделы пользователя', f'ua:open:{target}')]]))
                return
            self._reply(client, user_id, 'Доступ отключён: '+label(self.store, target)+'.')
            return
        elif command == "resolve":
            if len(args) != 2 or not args[0].isdigit():
                raise ValueError("/resolve ID retry|sent")
            resolve_uncertain(self.store, int(args[0]), args[1])
        self._reply(client, user_id, "Настройка сохранена.")

    @interaction(callback=True)
    def handle_callback(self, client, query):
        user_id = _private_user(query, callback=True)
        if user_id is not None:
            from .user_management import remember
            remember(self.store, getattr(query, 'from_user', None))
        if user_id is None or not self.store.authorize(user_id):
            query.answer("Нет доступа", show_alert=True)
            return
        data = getattr(query, "data", "")
        if not isinstance(data, str) or len(data.encode("utf-8")) > 64:
            mark_error(ValueError())
            query.answer("Неверная команда", show_alert=True)
            return
        try:
            require_route(self.store, user_id, data)
            if disabled(self.store) and hidden_route(data):
                self._search_disabled(client, user_id)
                query.answer()
                return
            active = self.store.get_setting(f'ui_screen:{user_id}', {})
            if (getattr(query.message, 'id', None) == active.get('message_id') and active.get('buttons') is not None
                    and data not in active['buttons'] and not data.startswith(('history:', 'report:'))):
                if data.startswith('rc:') and len(data.split(':')) >= 3:
                    self.life_controls.resume_receipt(client, user_id, data)
                    query.answer()
                    return
                query.answer('Экран уже обновился. Выберите действие на текущем экране.', show_alert=True)
                if CURRENT.get() is not None:
                    CURRENT.get()['monitor_outcome'] = 'stale'
                return
            if not data.startswith('rc:pick:'):
                self.market_controls.pending(user_id, None)
            if not data.startswith(('lf:', 'rc:')):
                self.life_controls.leave(user_id)
            if not data.startswith(('wc:', 'wn:')) and self.store.get_setting(f'wine_draft:{user_id}') is not None:
                self.wine_controls._leave(user_id)
            if data.startswith(('wc:', 'wn:')):
                self.wine_controls.callback(client, user_id, data, query.id)
                query.answer()
                return
            if data.startswith(('lf:', 'rc:')):
                self.life_controls.callback(client, user_id, data)
                query.answer()
                return
            if data.startswith('ua:'):
                self.admin_controls.callback(client, user_id, data, query.id)
                query.answer()
                return
            if data.startswith('mon:'):
                self.admin_controls.callback(client, user_id, data, query.id)
                query.answer()
                return
            if data.startswith('usd:'):
                self.dollar_controls.callback(client, user_id, data, query.id)
                query.answer()
                return
            if data.startswith('cb:'):
                self.cbr_controls.callback(client, user_id, data)
                query.answer()
                return
            if data.startswith('ig:'):
                self.integrity_controls.callback(client, user_id, data)
                query.answer()
                return
            if data.startswith('va:'):
                self.vivino_controls.atlas_controls.callback(client, user_id, data)
                query.answer()
                return
            if data.startswith('vi:'):
                self.vivino_controls.callback(client, user_id, data)
                query.answer()
                return
            if data.startswith('pj:'):
                self.project_controls.callback(client, user_id, data)
                query.answer()
                return
            if data.startswith('gt:'):
                self.trends_controls.callback(client, user_id, data)
                query.answer()
                return
            if data.startswith('cs:'):
                self.chart_schedule_controls.callback(client, user_id, data)
                query.answer()
                return
            if data.startswith(('rg:', 'kw:')):
                self.filter_controls.callback(client, user_id, data)
                query.answer()
                return
            if data.startswith('ui:'):
                self._ui_callback(client, user_id, data)
                query.answer()
                return
            if data.startswith('ch:'):
                self.chart_controls.callback(client, user_id, data, query.id)
                query.answer()
            elif data.startswith(('user:', 'mk:', 'ms:', 'mi:', 'ma:')):
                self.market_controls.callback(client, user_id, data)
                query.answer()
            elif data.startswith("menu:"):
                action = data.partition(":")[2]
                actions = {
                    'home': lambda: self._home(client, user_id),
                    'guide': lambda: self._guide(client, user_id),
                    'settings': lambda: self._settings_menu(client, user_id),
                    'graphs': lambda: self.chart_schedule_controls.menu(client, user_id),
                    'indicators': lambda: self.market_controls.indicators(client, user_id, ''),
                    'sources': lambda: self.market_controls.sources(client, user_id, ''),
                    "status": lambda: self._status(client, user_id),
                    "report": lambda: self._report_menu(client, user_id),
                    "subscriptions": lambda: self._subscriptions(client, user_id, ""),
                    "regions": lambda: self._regions(client, user_id, ""),
                    "keywords": lambda: self._keywords(client, user_id, ""),
                    "help": lambda: self._help(client, user_id),
                    "data": lambda: self._data_menu(client, user_id),
                    "ruonia_forecasts": lambda: self.data_controls.dispatch(client, user_id, "forecasts", "ruonia"),
                    "backtest_help": lambda: self._reply(client, user_id, DATA_HELP),
                    "export_help": lambda: self._reply(client, user_id,
                        "/export ruonia|wordstat|regions csv|xlsx [YYYY-MM-DD YYYY-MM-DD]\n"
                        "Пример: /export wordstat xlsx\nФайл придёт отдельным сообщением."),
                    "verify_help": lambda: self._verify_help(client, user_id),
                }
                if action in ADMIN_COMMANDS:
                    self.admin_controls.dispatch(client, user_id, action)
                    query.answer()
                    return
                if action in DATA_COMMANDS:
                    self.data_controls.dispatch(client, user_id, action, "", f"callback:{query.id}")
                    query.answer()
                    return
                if action not in actions:
                    raise ValueError("Неизвестный пункт меню")
                actions[action]()
                query.answer()
            elif data.startswith(("proposal:", "alert:", "adminpage:")):
                self.admin_controls.callback(client, user_id, data, query.id)
                query.answer()
            elif data.startswith("history:"):
                self.data_controls.callback(client, user_id, data, query.id)
                query.answer()
            elif data.startswith("request:"):
                section = data.partition(":")[2]
                self._report(client, user_id, section, f"callback:{query.id}")
                query.answer("Запрос принят")
            elif data.startswith("report:"):
                report_id = data.partition(":")[2]
                self.data_controls.callback(client, user_id, f"history:{report_id}", query.id)
                query.answer()
            elif data.startswith("sub:"):
                args = data.split(":")
                if len(args) != 3 or args[2] not in {"on", "off"}:
                    raise ValueError("Неверная настройка")
                update_subscription(self.store, user_id, args[1], enabled=args[2] == "on")
                query.answer("Подписка сохранена")
            elif data.startswith("resolve:"):
                self._admin(user_id)
                args = data.split(":")
                if len(args) != 3 or not args[1].isdigit():
                    raise ValueError("Неверная доставка")
                resolve_uncertain(self.store, int(args[1]), args[2])
                query.answer("Решение сохранено")
            else:
                raise ValueError("Неизвестная команда")
        except (ValueError, PermissionError) as exc:
            mark_error(exc)
            query.answer(str(exc), show_alert=True)
        except Exception as exc:  # noqa: BLE001 - Callback failures must not expose internal exception text.
            mark_error(exc)
            logger.error("Bot callback failed (%s)", type(exc).__name__)
            query.answer("Не удалось выполнить команду", show_alert=True)


    @interaction()
    def handle_wine_photo(self, client, message):
        actor = _private_user(message)
        if actor is None or not self.store.authorize(actor):
            return
        try:
            from .feature_access import require as require_feature
            from .user_management import remember
            remember(self.store, getattr(message, 'from_user', None))
            pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
            route = self.store.get_setting(f'ui_screen:{actor}', {}).get('route', '')
            if (pending and pending['pending'] == 'life:receipt') or route.startswith(('lf:', 'rc:')):
                self._queue_receipt(client, message, actor)
                return
            require(self.store, actor, 'vivino')
            require_feature(self.store, actor, 'wine_cellar')
            remember(self.store, getattr(message, 'from_user', None))
            self._queue_wine_photo(client, message, actor)
        except (ValueError, PermissionError) as exc:
            mark_error(exc)
            self._reply(client, actor, str(exc))
        except Exception as exc:  # noqa: BLE001 - Photo payloads and local paths stay private.
            mark_error(exc)
            logger.error('Wine photo upload failed (%s)', type(exc).__name__)
            self._reply(client, actor, 'Не удалось сохранить фото. Попробуйте ещё раз или добавьте вино по названию.')

    def _queue_wine_photo(self, client, message, actor):
        import hashlib

        from .telegram_charts import keyboard
        media = getattr(message, 'photo', None) or getattr(message, 'document', None)
        size = getattr(media, 'file_size', None)
        if type(size) is not int or not 0 < size <= 8 * 1024 * 1024:
            raise ValueError('Отправьте фотографию этикетки или JPEG/PNG/WebP до 8 МБ.')
        stream = client.download_media(message, in_memory=True)
        if not stream or not hasattr(stream, 'read'):
            raise ValueError('Не удалось загрузить фото из Telegram.')
        try:
            stream.seek(0)
            raw = stream.read(8 * 1024 * 1024+1)
        finally:
            stream.close()
        if not isinstance(raw, bytes) or not raw or len(raw) > 8 * 1024 * 1024:
            raise ValueError('Фото пустое или больше 8 МБ.')
        # Validate before keeping an upload; decoding/resizing/OCR runs in the
        # sequential worker. Never execute a file or trust its extension.
        import io

        from PIL import Image, UnidentifiedImageError

        from .wine_tools import MAX_IMAGE_PIXELS
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if (image.format not in {'JPEG', 'PNG', 'WEBP'} or image.width*image.height > MAX_IMAGE_PIXELS
                        or getattr(image, 'n_frames', 1) != 1):
                    raise ValueError('Поддерживается один снимок JPEG/PNG/WebP до 12 мегапикселей.')
                image.verify()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
            raise ValueError('Файл не является поддерживаемой фотографией.') from None
        from .wine_cellar import require_owner
        require_owner(self.store, actor, write=True)
        self.wine_controls._leave(actor)
        directory = self.store.path.parent/'wine_photos'/str(actor)
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(raw).hexdigest()
        path = directory/(digest+'.jpg')
        if not path.exists():
            path.write_bytes(raw)
        token = uuid4().hex
        identifier = self.store.enqueue('wine_recognize', {
            'actor_id': actor, 'path': str(path), 'request_id': token, 'report_id': uuid4().hex,
        }, f"wine-photo:{actor}:{digest}:{getattr(message, 'id', token)}")
        self.market_controls.pending(actor, None)
        screen('wc:photo:result', 'wc:menu')
        self._reply(client, actor, f'Этикетка сохранена. Распознавание №{identifier} в очереди. '
                    'Результат придёт сюда; проверьте название, год и количество перед добавлением. Запас пока не изменён.',
                    keyboard([[('Моя винотека', 'wc:menu')]]))

    def _queue_receipt(self, client, message, actor):
        import hashlib
        import io

        from PIL import Image, UnidentifiedImageError

        from .life_ledger import require_owner
        from .telegram_charts import keyboard
        require_owner(self.store, actor, write=True)
        media = getattr(message, 'document', None) or getattr(message, 'photo', None)
        size = getattr(media, 'file_size', None)
        limit = 10 * 1024 * 1024
        if type(size) is not int or not 0 < size <= limit:
            raise ValueError('Пришлите PDF до 10 МБ или фото чека до 8 МБ.')
        stream = client.download_media(message, in_memory=True)
        if not stream or not hasattr(stream, 'read'):
            raise ValueError('Не удалось загрузить чек из Telegram.')
        try:
            stream.seek(0)
            raw = stream.read(limit+1)
        finally:
            stream.close()
        if not isinstance(raw, bytes) or not raw or len(raw) > limit:
            raise ValueError('Файл пустой или больше 10 МБ.')
        if raw.startswith(b'%PDF-'):
            extension = '.pdf'
        else:
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError('Фото чека должно быть не больше 8 МБ.')
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format not in {'JPEG', 'PNG', 'WEBP'} or image.width*image.height > 12_000_000 or getattr(image, 'n_frames', 1) != 1:
                        raise ValueError('Пришлите один снимок JPEG/PNG/WebP до 12 мегапикселей.')
                    extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}[image.format]
                    image.verify()
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
                raise ValueError('Поддерживаются PDF и фотографии JPEG/PNG/WebP.') from None
        require_owner(self.store, actor, write=True)
        directory = self.store.path.parent/'receipts'/str(actor)
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(raw).hexdigest()
        path = directory/(digest+extension)
        if not path.exists():
            path.write_bytes(raw)
        token = uuid4().hex
        previous = self.store.get_setting(f'ui_screen:{actor}', {})
        from_wine = str(previous.get('back', '')).startswith('wc:')
        return_to = 'wc:menu' if from_wine else 'lf:home'
        self.store.set_setting(f'receipt_ui:{actor}:{token}', {'selected': [], 'overrides': {}, 'return_to': return_to})
        identifier = self.store.enqueue('life_receipt', {
            'actor_id': actor, 'owner_id': actor, 'path': str(path), 'request_id': token, 'report_id': uuid4().hex, 'return_to': return_to,
        }, f"life-receipt:{actor}:{digest}:{getattr(message, 'id', token)}")
        self.life_controls.leave(actor)
        screen('lf:receipt:queued', return_to)
        self._reply(client, actor, f'Чек принят. Распознавание №{identifier} в очереди; результат придёт сюда по готовности. '
                    'Учёт и винотека пока не изменены.', keyboard([[('Отправить ещё чек', 'lf:receipt')],
                    [('Моя винотека' if from_wine else 'Доходы и расходы', return_to)]]))

    @interaction()
    def handle_document(self, client, message):
        actor = _private_user(message)
        if actor is None or not self.store.authorize(actor):
            return
        try:
            from .user_management import remember
            remember(self.store, getattr(message, 'from_user', None))
            pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
            route = self.store.get_setting(f'ui_screen:{actor}', {}).get('route', '')
            filename = str(getattr(message.document, 'file_name', '')).lower()
            mime = getattr(message.document, 'mime_type', '')
            if filename.endswith('.pdf') or mime == 'application/pdf' or (
                    filename.endswith(('.jpg', '.jpeg', '.png', '.webp')) and (
                        route.startswith(('lf:', 'rc:')) or pending and pending['pending'] == 'life:receipt')):
                self._queue_receipt(client, message, actor)
            elif filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                require(self.store, actor, 'vivino')
                from .feature_access import require as require_feature
                require_feature(self.store, actor, 'wine_cellar')
                self._queue_wine_photo(client, message, actor)
            elif (pending and pending['pending'] == 'vivino_upload') or route.startswith(('vi:', 'pj:vivino')):
                require(self.store, actor, 'vivino')
                from .feature_access import require as require_feature
                require_feature(self.store, actor, 'vivino_map')
                self.vivino_controls.document(client, message, actor)
            elif pending and pending['pending'] == 'trends_upload':
                self.trends_controls.document(client, message, actor)
            else:
                from .telegram_charts import keyboard
                self._reply(client, actor, 'Выберите проект и откройте загрузку файла. ZIP Vivino загружается в его личном разделе.',
                            keyboard([[('Распознать чек', 'lf:receipt')], [('Vivino: загрузить данные', 'vi:menu')], [('Моя винотека', 'wc:menu')]]))
        except (ValueError, PermissionError) as exc:
            mark_error(exc)
            self._reply(client, actor, str(exc))
        except Exception as exc:  # noqa: BLE001 - Uploaded file failures cannot expose payloads.
            mark_error(exc)
            logger.error('Document import failed (%s)', type(exc).__name__)
            self._reply(client, actor, 'Не удалось проверить файл. Повторите загрузку исходного экспорта.')


    @interaction()
    def handle_other_message(self, client, message):
        actor = _private_user(message)
        if actor is None or not self.store.authorize(actor):
            return
        from .user_management import remember
        remember(self.store,getattr(message,'from_user',None))
        mark_error(ValueError())
        if CURRENT.get() is not None:
            CURRENT.get()['monitor_outcome'] = 'unsupported'
        self._reply(client,actor,'Этот тип сообщения пока не обрабатывается. Для чека отправьте PDF или фото, '
                    'для Vivino — ZIP или CSV; запрос можно написать текстом.')


def install_handlers(client, store):
    """Register synchronous callbacks; Pyrogram dispatches them in its executor."""
    from pyrogram import filters
    from pyrogram.handlers import CallbackQueryHandler, MessageHandler

    controls = BotControls(store)
    client.add_handler(MessageHandler(controls.handle_message, filters.private & filters.text))
    client.add_handler(MessageHandler(controls.handle_document, filters.private & filters.document))
    client.add_handler(MessageHandler(controls.handle_wine_photo, filters.private & filters.photo))
    client.add_handler(MessageHandler(controls.handle_other_message, filters.private & (
        filters.video | filters.audio | filters.voice | filters.video_note | filters.animation | filters.sticker | filters.contact | filters.location)))
    client.add_handler(CallbackQueryHandler(controls.handle_callback))
    return controls
