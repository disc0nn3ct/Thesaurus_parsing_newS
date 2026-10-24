"""Private-chat administration; provider work always goes through the queue."""

import json
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .budget import set_budgets
from .management import resolve_proposal, save_alert_rule
from .telegram_navigation import checked, screen

COMMANDS = {"admin", "autocollect", "budget", "alerts", "proposals", "jobs", "users", "monitor"}
HELP = (
    "\n\nАдминистратор:\n/admin — панель управления\n"
    "/autocollect — включить и выключить общий автосбор данных\n"
    "/users — доступ; /allow ID или @username; /deny ID или @username\n"
    "/monitor — мониторинг и история; /monitor ID или @username — пользователь\n"
    "/jobs [pending|running|failed|done] — очередь и ошибки\n"
    "/keywords add фраза|тема; /keywords remove фраза\n"
    "/regions set ID,ID — общая когорта\n"
    "/budget — лимиты, тарифы и расходы\n"
    "/alerts — правила оповещений и команды изменения\n"
    "/proposals — предложения AI; /proposals request — запросить новые\n"
    "/resolve ID retry|sent — решение о неопределённой доставке"
)
ALERT_HELP = (
    "\n\n/alerts add название|фраза или *|регион или * — создать выключенное правило\n"
    "/alerts edit ID название|фраза или *|регион или *\n"
    "/alerts set ID pct=50 z=2.5 coverage=0.9 baseline=20 cooldown=24\n"
    "/alerts on ID; /alerts off ID; /alerts ID — детали\n"
    "* — все фразы или регионы; cooldown — часы"
)
BUDGET_HELP = (
    "\n\n/budget wordstat|ai|ai_tokens число — дневные вызовы / токены ответа\n"
    "/budget price wordstat|ai per_call|input_per_million|output_per_million число\n"
    "/budget price wordstat|ai clear — убрать тариф провайдера\n"
    "/budget currency RUB — единица стоимости (до первого учёта стоимости)"
)


def positive_id(value):
    if not value.isascii() or not value.isdigit() or int(value) <= 0:
        raise ValueError("Нужен положительный числовой ID")
    return int(value)


def _keyboard(rows):
    from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in row]
                                 for row in rows])


def _user_label(store, target):
    from .user_management import label
    profile = store.one('SELECT first_name,last_name,username FROM user_profiles WHERE user_id=?', (target,))
    if not profile or not any(profile.get(field) for field in ('first_name', 'last_name', 'username')):
        return f'Имя неизвестно · ID {target}'
    return label(store, target)


def usage_summary(store):
    day = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/Moscow")).date().isoformat()
    currency = store.get_setting("budgets", {}).get("currency", "у.е.")
    rows = store.query("SELECT provider,SUM(calls) calls,SUM(tokens_in) tokens_in,SUM(tokens_out) tokens_out,"
                       "SUM(cost) cost,COUNT(cost) priced,COUNT(*) attempts FROM usage WHERE day=? GROUP BY provider", (day,))
    lines = [f"Использование за {day} (Москва):"]
    for row in rows:
        cost = f"{row['cost']:.6g} {currency}" if row["priced"] else "не рассчитана"
        suffix = f", оценено {row['priced']}/{row['attempts']} попыток" if row["priced"] else ""
        lines.append(f"{row['provider']}: вызовов {row['calls']}, токены вход/выход "
                     f"{row['tokens_in']}/{row['tokens_out']}; стоимость {cost}{suffix}")
    return "\n".join(lines + ([] if rows else ["Вызовов пока нет"]))


class AdminControls:
    def __init__(self, store, reply):
        self.store = store
        self.reply = reply

    def _authorize(self, user_id):
        if not self.store.authorize(user_id, admin=True):
            raise PermissionError("Действие доступно администратору")

    def dispatch(self, client, user_id, command, argument="", message_id=None):
        self._authorize(user_id)
        if command == "admin":
            screen('menu:admin')
            self.reply(client, user_id, 'Управление сервисом\nОбщий доступ, очередь и лимиты всего бота. Источники и автосбор находятся внутри соответствующего проекта.', _keyboard([
                [("Пользователи", "menu:users")],
                [("Мониторинг и история", "mon:home")],
                [("Задания", "menu:jobs"), ("Лимиты и расходы", "menu:budget")],
                [("Управление проектом рынка", "pj:market:manage")],
            ]))
        elif command == 'autocollect':
            from .autocollect import menu
            menu(self.store, self.reply, client, user_id)
        elif command == "budget":
            self.budget(client, user_id, argument)
        elif command == "alerts":
            self.alerts(client, user_id, argument)
        elif command == "proposals":
            self.proposals(client, user_id, argument, message_id)
        elif command == "jobs":
            self.jobs(client, user_id, argument)
        elif command == "users":
            self.users(client, user_id, argument)
        elif command == 'monitor':
            from .telegram_monitor import MonitoringControls
            MonitoringControls(self.store,self.reply).dispatch(client,user_id,argument)

    def callback(self, client, user_id, data, query_id=None):
        self._authorize(user_id)
        parts = data.split(":")
        if parts[0] == 'mon':
            from .telegram_monitor import MonitoringControls
            return MonitoringControls(self.store,self.reply).callback(client,user_id,data)
        if parts[0] == 'ua':
            if len(parts) == 3 and parts[1] == 'refresh':
                return self.users(client, user_id, f'page {positive_id(parts[2])}', refresh=True)
            if len(parts) == 3 and parts[1] == 'functions':
                return self.user_functions(client, user_id, positive_id(parts[2]))
            if len(parts) == 5 and parts[1] == 'feature' and parts[4] in {'on', 'off'}:
                from .feature_access import change
                target = positive_id(parts[2])
                change(self.store, user_id, target, parts[3], parts[4] == 'on')
                from .telegram_commands import configure_chat
                configure_chat(client, self.store, target)
                from .user_management import send_notices
                send_notices(self.store, client, target)
                return self.user_functions(client, user_id, target)
            if len(parts) == 3 and parts[1] == 'identity':
                from .user_management import resolve_profile
                target = positive_id(parts[2])
                if not self.store.one('SELECT user_id FROM users WHERE user_id=?', (target,)):
                    raise ValueError('Пользователь не найден')
                resolve_profile(self.store, client, target)
                return self.user_subscriptions(client, user_id, target)
            if len(parts) == 4 and parts[1] in {'role', 'roleconfirmed'} and parts[3] in {'admin', 'subscriber'}:
                target = positive_id(parts[2])
                if parts[1] == 'role':
                    from .user_management import label
                    if not self.store.one('SELECT user_id FROM users WHERE user_id=?', (target,)):
                        raise ValueError('Пользователь не найден')
                    screen(f'ua:role:{target}:{parts[3]}', f'ua:open:{target}')
                    text = ('Назначить администратором' if parts[3] == 'admin' else 'Снять права администратора')+'?\n'+label(self.store, target)
                    text += ('\nАдминистратору доступны все функции, пользователи и их данные, источники и настройки сервиса.'
                             if parts[3] == 'admin' else '\nОстанутся только ранее назначенные личные функции. Доступ к управлению будет закрыт.')
                    return self.reply(client, user_id, text, _keyboard([[('Подтвердить изменение роли', f'ua:roleconfirmed:{target}:{parts[3]}')]]))
                from .user_management import change_role
                change_role(self.store, user_id, target, parts[3])
                from .telegram_commands import configure_chat
                configure_chat(client, self.store, target)
                from .user_management import send_notices
                send_notices(self.store, client, target)
                return self.user_subscriptions(client, user_id, target)
            if len(parts) == 3 and parts[1] == 'notify':
                target = positive_id(parts[2])
                if not self.store.one('SELECT user_id FROM users WHERE user_id=?', (target,)):
                    raise ValueError('Пользователь не найден')
                from .user_management import queue_notice
                with self.store.connection() as conn:
                    queue_notice(conn, target, 'manual')
                    conn.execute('INSERT INTO access_audit(actor_id,target_id,action,created_at) VALUES(?,?,?,?)', (user_id, target, 'access_notice:manual', datetime.now(timezone.utc).isoformat()))
                from .user_management import send_notices
                send_notices(self.store, client, target)
                return self.user_subscriptions(client, user_id, target)
            if len(parts) == 3 and parts[1] == 'open':
                return self.user_subscriptions(client, user_id, positive_id(parts[2]))
            if len(parts) == 5 and parts[1] == 'set' and parts[4] in {'on', 'off'}:
                from .subscription_access import change
                target = positive_id(parts[2])
                change(self.store, user_id, target, parts[3], parts[4] == 'on')
                from .telegram_commands import configure_chat
                configure_chat(client, self.store, target)
                from .user_management import send_notices
                send_notices(self.store, client, target)
                return self.user_subscriptions(client, user_id, target)
            raise ValueError('Неизвестная настройка доступа')
        if len(parts) != 3:
            raise ValueError("Неверная команда")
        prefix, action, value = parts
        if prefix == "proposal":
            argument = "request" if action == "request" and value == "new" else f"{action} {value}"
            self.proposals(client, user_id, argument, f"callback:{query_id}")
        elif prefix == "alert":
            self.alerts(client, user_id, f"{action} {value}")
        elif prefix == "adminpage" and action in {"alerts", "proposals", "users"}:
            getattr(self, action)(client, user_id, f"page {value}")
        else:
            raise ValueError("Неизвестная команда")

    def budget(self, client, user_id, argument):
        args = argument.split()
        if args:
            config = self.store.get_setting("budgets")
            limits = {"wordstat": ("wordstat_calls", 0, 1_000_000), "ai": ("ai_calls", 0, 10_000),
                      "ai_tokens": ("ai_output_tokens", 1, 100_000)}
            if len(args) == 2 and args[0] in limits:
                key, minimum, maximum = limits[args[0]]
                if not args[1].isascii() or not args[1].isdigit() or not minimum <= int(args[1]) <= maximum:
                    raise ValueError(f"Лимит должен быть целым числом от {minimum} до {maximum}")
                changes = {key: int(args[1])}
            elif len(args) == 2 and args[0] == "currency":
                if not 1 <= len(args[1]) <= 12 or any(ord(char) < 32 for char in args[1]):
                    raise ValueError("Единица стоимости должна содержать от 1 до 12 символов")
                changes = {"currency": args[1]}
            elif len(args) in {3, 4} and args[0] == "price" and args[1] in {"wordstat", "ai"}:
                prices = config.get("prices", {})
                if len(args) == 3 and args[2] == "clear":
                    prices.pop(args[1], None)
                elif len(args) == 4 and args[2] in {"per_call", "input_per_million", "output_per_million"}:
                    try:
                        number = float(args[3])
                        if not math.isfinite(number) or number < 0:
                            raise ValueError()
                    except ValueError:
                        raise ValueError("Цена должна быть конечным неотрицательным числом") from None
                    prices.setdefault(args[1], {})[args[2]] = number
                else:
                    raise ValueError(BUDGET_HELP.strip())
                changes = {"prices": prices}
            else:
                raise ValueError(BUDGET_HELP.strip())
            set_budgets(self.store, changes)
        config = self.store.get_setting("budgets")
        lines = [(f"Дневные лимиты: Wordstat {config['wordstat_calls']}, AI {config['ai_calls']}; "
                  f"максимум выходных токенов ответа AI {config['ai_output_tokens']}"),
                 f"Единица стоимости: {config.get('currency', 'у.е.')}"]
        for provider in ("wordstat", "ai"):
            prices = config.get("prices", {}).get(provider, {})
            lines.append(f"Тариф {provider}: " + (", ".join(f"{key}={value}" for key, value in prices.items()) or "не задан"))
        screen('menu:budget', 'menu:admin')
        self.reply(client, user_id, "\n".join(lines) + "\n\n" + usage_summary(self.store), _keyboard([
            [('Лимит Wordstat', 'ui:budget:wordstat'), ('Лимит AI', 'ui:budget:ai')],
            [('Токены ответа AI', 'ui:budget:ai_tokens')], [('Тарифы и команды', 'ui:help:admin')]]))

    def _page(self, argument):
        if not argument:
            return 1
        args = argument.split()
        if len(args) != 2 or args[0] != "page":
            raise ValueError("Для следующей страницы: page номер")
        return positive_id(args[1])

    def _navigation(self, kind, page, has_more):
        buttons = []
        if page > 1:
            buttons.append(("‹ Предыдущая страница", f"adminpage:{kind}:{page - 1}"))
        if has_more:
            buttons.append(("Следующая страница ›", f"adminpage:{kind}:{page + 1}"))
        return [buttons] if buttons else []

    def alerts(self, client, user_id, argument):
        args = argument.split(maxsplit=2)
        show_id = None
        if args and args[0] in {"add", "edit"}:
            action, _, raw = argument.partition(" ")
            rule_id = None
            if action == "edit":
                identifier, _, raw = raw.partition(" ")
                rule_id = positive_id(identifier)
            parts = [value.strip() for value in raw.split("|")]
            if len(parts) != 3 or not all(parts):
                raise ValueError(ALERT_HELP.strip())
            show_id = save_alert_rule(self.store, {"name": parts[0], "metric": None if parts[1] == "*" else parts[1],
                                                 "region_id": None if parts[2] == "*" else parts[2]}, rule_id)
        elif args and args[0] == "set":
            if len(args) != 3:
                raise ValueError(ALERT_HELP.strip())
            mapping = {"pct": "pct_threshold", "z": "z_threshold", "coverage": "min_coverage",
                       "baseline": "min_baseline", "cooldown": "cooldown_hours"}
            values = {}
            for setting in args[2].split():
                key, separator, value = setting.partition("=")
                if not separator or key not in mapping or mapping[key] in values:
                    raise ValueError(ALERT_HELP.strip())
                values[mapping[key]] = value
            show_id = save_alert_rule(self.store, values, positive_id(args[1]))
        elif args and args[0] in {"on", "off"}:
            if len(args) != 2:
                raise ValueError(ALERT_HELP.strip())
            show_id = save_alert_rule(self.store, {"enabled": int(args[0] == "on")}, positive_id(args[1]))
        elif len(args) == 1:
            show_id = positive_id(args[0])
        page = 1 if show_id is not None else self._page(argument)
        if show_id is not None:
            rows = self.store.query("SELECT * FROM alert_rules WHERE id=?", (show_id,))
            if not rows:
                raise ValueError("Правило не найдено")
        else:
            rows = self.store.query("SELECT * FROM alert_rules ORDER BY id LIMIT 11 OFFSET ?", ((page - 1) * 10,))
        lines, buttons = ["Правила оповещений:"], []
        for row in rows[:10]:
            lines.append(f"#{row['id']} {row['name']} — {'вкл' if row['enabled'] else 'выкл'}\n"
                         f"Фраза: {row['metric'] or '*'}; регион: {row['region_id'] or '*'}\n"
                         f"pct={row['pct_threshold']:g} z={row['z_threshold']:g} coverage={row['min_coverage']:g} "
                         f"baseline={row['min_baseline']:g} cooldown={row['cooldown_hours']:g} ч")
            action = "off" if row["enabled"] else "on"
            buttons.append([(f"{'Выключить' if row['enabled'] else 'Включить'} #{row['id']}", f"alert:{action}:{row['id']}")])
        buttons += self._navigation("alerts", page, len(rows) > 10)
        self.reply(client, user_id, "\n\n".join(lines) + ALERT_HELP, _keyboard(buttons) if buttons else None)

    def proposals(self, client, user_id, argument, message_id=None):
        args = argument.split()
        prefix = ""
        if args == ["request"]:
            key = f"telegram:proposals:{user_id}:{message_id}" if message_id is not None else None
            job_id = self.store.enqueue("propose_keywords", {"actor_id": user_id}, dedupe_key=key)
            prefix = f"Предложения запрошены, задание #{job_id}. После выполнения откройте /proposals.\n\n"
        elif len(args) == 2 and args[0] in {"approve", "reject"}:
            state = resolve_proposal(self.store, positive_id(args[1]), args[0])
            prefix = ("Предложение применено." if state == "approved" else "Предложение отклонено.") + "\n\n"
        elif args and args[0] != "page":
            raise ValueError("/proposals; /proposals request; /proposals approve|reject ID")
        page = self._page(argument) if args and args[0] == "page" else 1
        rows = self.store.query("SELECT id,created_at,entries FROM keyword_proposals WHERE state='pending' "
                                "ORDER BY id DESC LIMIT 6 OFFSET ?", ((page - 1) * 5,))
        lines, buttons = [prefix + "Предложения AI на рассмотрении:"], []
        for row in rows[:5]:
            try:
                entries = json.loads(row["entries"])
                content = "\n".join(f"• {item['phrase']} ({item.get('theme', 'другое')})" for item in entries)
            except (ValueError, TypeError, KeyError, AttributeError):
                content = "Некорректный список; предложение можно отклонить"
            lines.append(f"#{row['id']} — {row['created_at']}\n{content}")
            buttons.append([("Применить #" + str(row["id"]), f"proposal:approve:{row['id']}"),
                            ("Отклонить #" + str(row["id"]), f"proposal:reject:{row['id']}")])
        if not rows:
            lines.append("Пока нет")
        buttons += self._navigation("proposals", page, len(rows) > 5)
        buttons.append([("Запросить предложения AI", "proposal:request:new")])
        self.reply(client, user_id, "\n\n".join(lines)
                   + "\n\n/proposals request; /proposals approve ID; /proposals reject ID", _keyboard(buttons))

    def jobs(self, client, user_id, argument):
        screen('menu:jobs', 'menu:admin')
        state = argument.strip()
        if state and state not in {"pending", "running", "failed", "done"}:
            raise ValueError("/jobs [pending|running|failed|done]")
        counts = self.store.query("SELECT state,COUNT(*) n FROM jobs GROUP BY state")
        labels = {'pending': 'в очереди', 'running': 'выполняются', 'failed': 'ошибки', 'done': 'готово', 'cancelled': 'отменены'}
        text = "Задания: " + (", ".join(f"{labels.get(row['state'], row['state'])}: {row['n']}" for row in counts) or "пока нет")
        rows = self.store.query("SELECT id,kind,state,created_at FROM jobs " + ("WHERE state=? " if state else "")
                                + "ORDER BY id DESC LIMIT 20", (state,) if state else ())
        kinds = {'report': 'отчёт', 'telegram_chart': 'график', 'telegram_export': 'экспорт', 'telegram_backtest': 'проверка прогноза',
                 'market_report': 'обзор рынка', 'market_collect': 'сбор рынка', 'market_export': 'экспорт рынка',
                 'market_backtest': 'проверка рыночных прогнозов', 'propose_keywords': 'предложения фраз', 'verify_sources': 'проверка ссылок'}
        text += "\nПоследние 20:\n" + "\n".join(f"#{row['id']} {kinds.get(row['kind'], row['kind'])} — {labels.get(row['state'], row['state'])} ({row['created_at'][:16]})"
                                                for row in rows)
        self.reply(client, user_id, text, _keyboard([
            [(checked(label, state == value), 'ui:jobs:' + value) for value, label in [('pending', 'Очередь'), ('running', 'В работе')]],
            [(checked(label, state == value), 'ui:jobs:' + value) for value, label in [('failed', 'Ошибки'), ('done', 'Готово')]], [('Все задания', 'ui:jobs:all')]]))

    def users(self, client, user_id, argument, *, refresh=False):
        self._authorize(user_id)
        screen('menu:users', 'menu:admin')
        page = self._page(argument)
        rows = self.store.query("SELECT u.user_id,u.role,u.enabled,s.started_at,p.first_name,p.last_name,p.username "
                                "FROM users u LEFT JOIN bot_sessions s ON s.user_id=u.user_id "
                                "LEFT JOIN user_profiles p ON p.user_id=u.user_id "
                                "ORDER BY u.user_id LIMIT 21 OFFSET ?", ((page - 1) * 20,))
        candidates = [row['user_id'] for row in rows[:20] if refresh or
                      not any(row.get(field) for field in ('first_name', 'last_name', 'username'))]
        if candidates:
            from .user_management import resolve_profile
            cursor_key = f'identity_refresh_cursor:{user_id}:{page}'
            cursor = self.store.get_setting(cursor_key, 0)
            cursor = cursor if type(cursor) is int else 0
            ordered = [target for target in candidates if target > cursor] + [target for target in candidates if target <= cursor]
            # Telegram may not know a peer yet. Keep one page responsive and let
            # subsequent refreshes reach the remaining users after failed lookups.
            for target in ordered[:6]:
                resolve_profile(self.store, client, target)
            self.store.set_setting(cursor_key, ordered[:6][-1])
        labels = {row['user_id']: _user_label(self.store, row['user_id']) for row in rows[:20]}
        lines = ["Пользователи · откройте карточку для управления:"] + [f"{labels[row['user_id']]} — {'администратор' if row['role'] == 'admin' else 'подписчик'}, {'доступ есть' if row['enabled'] else 'доступ закрыт'}, {'/start выполнен' if row['started_at'] else 'ещё не нажал /start'}" for row in rows[:20]]
        if any(value.startswith('Имя неизвестно') for value in labels.values()):
            lines.append('Telegram пока не передал имя для некоторых ID. Пользователь может открыть бот и нажать /start; '
                         'имя обновится при взаимодействии. Кнопка ниже проверяет до 6 пользователей текущей страницы за раз.')
        buttons = self._navigation("users", page, len(rows) > 20)
        buttons.insert(0, [('Добавить пользователя', 'user:add:new')])
        buttons[1:1] = [[(labels[row['user_id']], f"ua:open:{row['user_id']}")] for row in rows[:20]]
        buttons.append([('Обновить имена · до 6 за раз', f'ua:refresh:{page}')])
        self.reply(client, user_id, "\n".join(lines) + "\n/allow ID или @username — включить доступ; /deny ID или @username — выключить",
                   _keyboard(buttons) if buttons else None)

    def user_subscriptions(self, client, actor, target):
        self._authorize(actor)
        from .subscription_access import CATALOG, key, normalize
        from .user_management import status
        user = self.store.one('SELECT role,enabled FROM users WHERE user_id=?', (target,))
        if not user:
            raise ValueError('Пользователь не найден')
        values = list(CATALOG) if user['role'] == 'admin' else normalize(self.store.get_setting(key(target), []))
        screen(f'ua:open:{target}', 'menu:users')
        text = (f"{_user_label(self.store, target)}\nРоль: {'администратор' if user['role'] == 'admin' else 'пользователь'}.\nДоступ к боту: {'есть' if user['enabled'] else 'закрыт'}.\n"
                '✓ — подключено администратором. Пользователь выбирает настройки только внутри этих разделов. '
                'Ниже подключаются проекты целиком. Для точного выбора откройте «Отдельные функции». '
                'Подключение не включает рассылки; отключение останавливает задания и следующие части доставки раздела. Личные данные сохраняются.\n'+status(self.store, target))
        if _user_label(self.store, target).startswith('Имя неизвестно'):
            text += '\nTelegram пока не передал имя. Обновите его кнопкой ниже или попросите пользователя открыть бот и нажать /start. ID остаётся идентификатором доступа.'
        rows = [[(checked(title, section in values), f"ua:set:{target}:{section}:{'off' if section in values else 'on'}")]
                for section, title in CATALOG.items()] if user['role'] != 'admin' else []
        if user['role'] == 'admin':
            text += '\nАдминистратору доступны все разделы и функции.'
        rows.insert(0, [('Отдельные функции', f'ua:functions:{target}')])
        rows.insert(1, [('История и мониторинг пользователя', f'mon:user:{target}')])
        pins = self.store.get_setting('protected_admin_ids', [])
        if user['role'] != 'admin':
            rows.append([('Назначить администратором', f'ua:role:{target}:admin')])
            rows.append([('Отключить весь доступ' if user['enabled'] else 'Восстановить доступ', f"user:{'deny' if user['enabled'] else 'allow'}:{target}")])
        elif target in pins:
            text += '\nРоль защищена: пользователь указан в ADMIN_TELEGRAM_IDS на сервере.'
        elif target != actor:
            rows.append([('Снять права администратора', f'ua:role:{target}:subscriber')])
        rows.append([('Обновить имя и @username', f'ua:identity:{target}')])
        rows.append([('Повторно отправить уведомление о правах', f'ua:notify:{target}')])
        self.reply(client, actor, text, _keyboard(rows))

    def user_functions(self, client, actor, target):
        self._authorize(actor)
        from .feature_access import CATALOG, configured
        from .user_management import status
        user = self.store.one('SELECT role,enabled FROM users WHERE user_id=?', (target,))
        if not user:
            raise ValueError('Пользователь не найден')
        values = list(CATALOG) if user['role'] == 'admin' else configured(self.store, target)
        screen(f'ua:functions:{target}', f'ua:open:{target}')
        text = _user_label(self.store, target)+'\nОтдельные функции\n✓ — разрешено, ○ — закрыто. Изменения сохраняются сразу. '
        text += 'Включение одной функции подключает нужный проект, но не остальные функции в нём. Отзыв прекращает новые задания и дальнейшую доставку результата.\n'
        text += 'Vivino и винотека — отдельные права. Винотека разрешает добавление вручную и по фото, запасы и списания. Для PDF и фото чеков нужен также финансовый учёт.\n'
        text += status(self.store, target)
        if not user['enabled']:
            text += '\nВесь доступ отключён. Здесь сохраняется набор прав для восстановления.'
        rows = [[(('✓ ' if name in values else '○ ')+title, f"ua:feature:{target}:{name}:{'off' if name in values else 'on'}")]
                for name, (title, _) in CATALOG.items()] if user['role'] != 'admin' else []
        if user['role'] == 'admin':
            text += '\nАдминистратор имеет все права; для ограничения функций сначала снимите роль администратора.'
        self.reply(client, actor, text, _keyboard(rows))
