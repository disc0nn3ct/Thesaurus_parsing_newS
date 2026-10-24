"""Allowlisted personal history, exports, forecast scores and offline backtests."""

import json
from uuid import uuid4

from .delivery import prepare_delivery
from .store import SECTIONS, utcnow
from .telegram_data_jobs import (
    actor_scope,
    authorized_report,
    export_dates,
    forecast_summary,
    metrics_text,
    require_actor,
)

DATA_COMMANDS = ("history", "export", "forecasts", "backtest", "verify")
DATA_HELP = (
    "\n/history [раздел|ID] — история ваших доставленных отчётов\n"
    "/export ruonia|wordstat|regions csv|xlsx [YYYY-MM-DD YYYY-MM-DD] — файл ваших данных\n"
    "/forecasts [ruonia|wordstat] — сохранённые прогнозы и их ошибки\n"
    "/backtest [ruonia|wordstat] [1..52] — проверка на прошлых данных (по умолчанию wordstat, 12 точек)"
)
DATA_ADMIN_HELP = "\n/verify ID_отчёта — проверить доступность источников; итог через /history ID"


class DataControls:
    def __init__(self, store, reply):
        self.store, self.reply = store, reply

    def dispatch(self, client, user_id, command, argument, message_id=None):
        require_actor(self.store, user_id)
        from .feature_access import require as require_feature
        feature = {'export': 'exports', 'backtest': 'backtests', 'forecasts': 'backtests'}.get(command)
        if feature:
            require_feature(self.store, user_id, feature)
        handlers = {"history": self._history, "export": self._export, "forecasts": self._forecasts,
                    "backtest": self._backtest, "verify": self._verify}
        if command not in handlers:
            raise ValueError("Неизвестная команда данных")
        handlers[command](client, user_id, argument, message_id)

    def callback(self, client, user_id, data, query_id=None):
        require_actor(self.store, user_id)
        if not isinstance(data, str) or not data.startswith("history:"):
            raise ValueError("Неверная команда истории")
        self._details(client, user_id, data.partition(":")[2])

    def _history(self, client, user_id, argument, message_id):
        from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        if argument and argument not in SECTIONS:
            return self._details(client, user_id, argument)
        conditions, parameters = [], []
        is_admin = self.store.authorize(user_id, admin=True)
        if not is_admin:
            from .subscription_access import assigned
            granted = assigned(self.store, user_id)
            conditions.append('r.section IN ('+','.join('?' for _ in granted)+')' if granted else '0')
            parameters.extend(granted)
            conditions.append("""EXISTS (SELECT 1 FROM deliveries d WHERE d.report_id=r.id AND d.recipient=?
                GROUP BY d.variant HAVING SUM(CASE WHEN d.state='sent' THEN 0 ELSE 1 END)=0)""")
            parameters.append(str(user_id))
        if argument:
            conditions.append("r.section=?")
            parameters.append(argument)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        rows = self.store.query("SELECT r.id,r.section,r.created_at,r.quality FROM reports r" + where
                                + " ORDER BY r.created_at DESC,r.id DESC LIMIT 12", parameters)
        from .feature_access import report_allowed
        rows = [row for row in rows if report_allowed(self.store, user_id, row)]
        text = ("Последние отчёты:" if is_admin else "Ваши последние доставленные отчёты:")
        text += "\n" + ("\n".join(f"{row['created_at'][:16]} · {row['section']} · {row['id']}" for row in rows)
                        or "Отчётов пока нет.")
        buttons = [[InlineKeyboardButton(f"{row['section']} · {row['created_at'][:16]}",
                                         callback_data=f"history:{row['id']}")] for row in rows]
        self.reply(client, user_id, text, InlineKeyboardMarkup(buttons) if buttons else None)

    def _details(self, client, user_id, report_id):
        report = authorized_report(self.store, user_id, report_id)
        # A report's content is immutable. Different delivery variants may
        # carry schedule/filter metadata, but must not cause history views to
        # resend a document that was already sent or has an uncertain outcome.
        existing = self.store.one(
            """SELECT variant FROM deliveries WHERE report_id=? AND recipient=?
            AND variant LIKE 'full:%' ORDER BY id DESC LIMIT 1""", (report_id, str(user_id)),
        )
        first_expansion = existing is None
        if not existing:
            prepare_delivery(self.store, report_id, str(user_id), detail="full")
            existing = self.store.one(
                "SELECT variant FROM deliveries WHERE report_id=? AND recipient=? AND variant LIKE 'full:%' ORDER BY id DESC LIMIT 1",
                (report_id, str(user_id)),
            )
        deliveries = self.store.query(
            "SELECT id,payload,state,message_id FROM deliveries WHERE report_id=? AND recipient=? AND variant=? ORDER BY part_index",
            (report_id, str(user_id), existing["variant"]),
        )
        lines = [f"Отчёт {report_id}", f"{report['section']} · {report['created_at'][:16]}"]
        if first_expansion:
            lines.append("Полный текст поставлен в очередь доставки.")
        labels = {"sent": "уже доставлен", "pending": "ожидает отправки", "sending": "отправляется",
                  "uncertain": "результат отправки неизвестен; нужна проверка администратора",
                  "failed": "отправка завершилась ошибкой", "cancelled": "доставка отменена"}
        for row in deliveries:
            if json.loads(row["payload"]).get("kind") in {"document", "photo"}:
                reference = f", сообщение {row['message_id']}" if row["state"] == "sent" and row["message_id"] else ""
                lines.append(f"Файл, доставка {row['id']}: {labels.get(row['state'], row['state'])}{reference}.")
        for item in json.loads(report["news"])[:10]:
            availability = item.get("source_availability", {}).get("availability", "not_checked")
            numeric = "проверены" if item.get("numbers_verified") else "не проверены"
            lines.append(f"• {item.get('title') or 'Источник'}: доступность {availability}; числа {numeric}.")
        if json.loads(report["news"]):
            lines.append("Доступность ссылки и проверка чисел не подтверждают содержание публикации.")
        if not first_expansion:
            lines.extend(("", "Полный текст:", report["full_text"]))
        self.reply(client, user_id, "\n".join(lines))

    def _queue(self, kind, user_id, payload, message_id):
        # The same Telegram event cannot schedule a second heavy job, even if
        # its parameters or subscriptions change before a repeated callback.
        key = f"telegram-data:{user_id}:{message_id}" if message_id is not None else None
        return self.store.enqueue(kind, {**payload, "report_id": uuid4().hex}, dedupe_key=key)

    def _export(self, client, user_id, argument, message_id):
        args = argument.split()
        if len(args) not in {2, 4} or args[1] not in {"csv", "xlsx"}:
            raise ValueError("/export ruonia|wordstat|regions csv|xlsx [YYYY-MM-DD YYYY-MM-DD]")
        scope = actor_scope(self.store, user_id, args[0])
        start, end = export_dates(*(args[2:] if len(args) == 4 else (None, None)))
        job_id = self._queue("telegram_export", user_id,
                             {**scope, "format": args[1], "date_from": start, "date_to": end}, message_id)
        self.reply(client, user_id, f"Экспорт в очереди, задание {job_id}. Файл придёт отдельным сообщением.")

    def _forecasts(self, client, user_id, argument, message_id):
        section = argument or "wordstat"
        if section not in {"ruonia", "wordstat"}:
            raise ValueError("/forecasts [ruonia|wordstat]")
        rows, metrics = forecast_summary(self.store, user_id, section)
        if section == 'ruonia':
            from .market_analysis import evaluate
            from .market_sources import metric_id
            from .market_store import forecast_history
            from .telegram_data_jobs import actor_scope
            issued = [p for metric in actor_scope(self.store, user_id, section)['phrases']
                      for p in forecast_history(self.store, metric_id('local.ruonia', metric), latest=True)]
            if issued:
                rows = [{**p, 'metric': self.store.one('SELECT title FROM market_series WHERE id=?', (p['series_id'],))['title'],
                         'region_id': 225, 'device': 'all', 'origin_date': p['origin'], 'target_date': p['target']} for p in issued]
                metrics = evaluate(issued)
        lines = [f"Прогнозы {section}:", metrics_text(metrics),
                 "Ошибка считается только для прогнозов, сохранённых до целевой даты. Базовый прогноз повторяет последнее значение."]
        if not rows:
            lines.append(f"Сохранённых прогнозов пока нет. Запросите /report {section} full.")
        for row in rows[:20]:
            bounds = (f"[{row['lower']:.3g}; {row['upper']:.3g}]" if row["lower"] is not None and row["upper"] is not None else "интервал неизвестен")
            lines.append(f"• {row['metric']}, {row['region_id']}, {row['device']}: {row['target_date']} → "
                         f"{row['prediction']:.3g} {bounds} (выпуск {row['origin_date']})")
        if len(rows) > 20:
            lines.append(f"Показаны последние 20 из {len(rows)} прогнозов; оценка использует все доступные факты.")
        self.reply(client, user_id, "\n".join(lines))

    def _backtest(self, client, user_id, argument, message_id):
        args = argument.split()
        section = args.pop(0) if args and args[0] in {"ruonia", "wordstat"} else "wordstat"
        if len(args) > 1:
            raise ValueError("/backtest [ruonia|wordstat] [1..52]")
        argument = args[0] if args else ""
        if argument and (not argument.isascii() or not argument.isdigit()):
            raise ValueError("/backtest [число точек от 1 до 52]")
        origins = int(argument) if argument else 12
        if not 1 <= origins <= 52:
            raise ValueError("Число точек должно быть от 1 до 52")
        scope = actor_scope(self.store, user_id, section)
        scope['forecast'] = self.store.get_setting(f'chart:{user_id}:{section}', {}).get('forecast')
        job_id = self._queue("telegram_backtest", user_id,
                             {**scope, "max_origins": origins, "as_of": utcnow()[:10]}, message_id)
        self.reply(client, user_id, f"Проверка на сохранённых данных в очереди, задание {job_id}. Итог и JSON придут сообщением.")

    def _verify(self, client, user_id, argument, message_id):
        require_actor(self.store, user_id, admin=True)
        report = authorized_report(self.store, user_id, argument)
        key = f"telegram-verify:{user_id}:{message_id}" if message_id is not None else None
        job_id = self.store.enqueue("verify_sources", {"actor_id": user_id, "report_id": report["id"]}, dedupe_key=key)
        self.reply(client, user_id, f"Проверка источников в очереди, задание {job_id}. Результат: /history {report['id']}")
