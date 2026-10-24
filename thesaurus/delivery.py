"""Durable per-part delivery; ambiguous outcomes require an operator decision."""

import hashlib
import inspect
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from functions.telegram_utils import split_message

from .store import dumps, utcnow

MAX_ATTEMPTS = 3


def _ai_quality_note(report):
    quality = json.loads(report["quality"])
    if not isinstance(quality, dict) or not quality:
        return ""
    try:
        fetched = datetime.fromisoformat(quality.get("last_fetched_at") or "")
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        notes = ["загружено " + fetched.astimezone(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")]
    except (TypeError, ValueError):
        notes = ["время загрузки неизвестно"]
    if quality.get("cached") or quality.get("status") == "cached":
        notes.append("использованы сохранённые данные")
    errors = quality.get("errors")
    if quality.get("status") == "partial" or errors:
        count = f" (ошибок: {len(errors)})" if isinstance(errors, list) and errors else ""
        notes.append("обновление неполное" + count)
    return "\nДанные: " + "; ".join(notes) + "."


def _ai_brief(store, report, recipient):
    """Hide repeated news relative to this recipient's last completed delivery."""
    from .analysis import classify_news_changes
    current = json.loads(report["news"])
    if not current:
        return report["text"]
    previous = store.one("""SELECT r.news FROM reports r JOIN deliveries d ON d.report_id=r.id
        WHERE d.recipient=? AND r.section='ai' AND r.id<>?
        GROUP BY r.id,d.variant HAVING SUM(CASE WHEN d.state='sent' THEN 0 ELSE 1 END)=0
        ORDER BY MAX(d.updated_at) DESC LIMIT 1""", (recipient, report["id"]))
    changes = classify_news_changes(current, json.loads(previous["news"]) if previous else [])
    lines = ["Новости с вашей последней доставки:"]
    lines.extend("Новое: " + item["title"] for item in changes["new"])
    lines.extend("Уточнение: " + item["title"] for item in changes["corrected"])
    if not changes["new"] and not changes["corrected"]:
        lines.append("Новых структурированных сообщений нет.")
    lines.append(f"Повторов свёрнуто: {len(changes['repeated'])}. Источники и полный текст по кнопке.")
    return "\n".join(lines) + _ai_quality_note(report)


def _baseline_prefix(store, report, recipient):
    if json.loads(report['quality']).get('tool') == 'dollar':
        # FX decisions are a separate tool, not comparable to another market report's metrics map.
        return ''
    current = json.loads(report["metrics"])
    if not isinstance(current, dict) or not current:
        return ""
    previous_reports = store.query(
        """SELECT r.* FROM reports r JOIN deliveries d ON d.report_id=r.id
        WHERE d.recipient=? AND r.section=? AND r.id<>?
        GROUP BY r.id,d.variant HAVING SUM(CASE WHEN d.state='sent' THEN 0 ELSE 1 END)=0
        ORDER BY MAX(d.updated_at) DESC""",
        (recipient, report["section"], report["id"]),
    )
    old = None
    previous_regular = False
    for previous in previous_reports:
        candidate = json.loads(previous["metrics"])
        # Alerts have no numerical snapshot and must not hide a regular report.
        if not isinstance(candidate, dict) or not candidate:
            continue
        previous_regular = True
        if (previous["keyword_version"] != report["keyword_version"]
                or previous["cohort_version"] != report["cohort_version"]
                or candidate.keys() != current.keys()):
            continue
        if all(isinstance(entry, dict) and isinstance(candidate[key], dict)
               and entry.get("unit") == candidate[key].get("unit")
               for key, entry in current.items()):
            old = candidate
            break
    if old is None:
        if previous_regular:
            return ("Сравнение с предыдущими доставками недоступно: изменился набор данных "
                    "(персональные фильтры, состав регионов, версии настроек или единицы измерения).\n\n")
        return ""
    changes = []
    for key, entry in current.items():
        before = old.get(key)
        if not isinstance(entry, dict) or not isinstance(before, dict) or entry.get("unit") != before.get("unit"):
            continue
        value, prior = entry.get("value"), before.get("value")
        if (type(value) not in (int, float) or type(prior) not in (int, float)
                or not math.isfinite(value) or not math.isfinite(prior)):
            continue
        delta = value - prior
        if delta == 0:
            continue
        label = str(entry.get("metric", key))[:120]
        region = f", регион {entry['region_id']}" if entry.get("region_id") else ""
        relative = f" ({delta / abs(prior):+.1%})" if prior else ""
        changes.append(f"• {label}{region}: {delta:+.3g} {entry.get('unit', '')}{relative}")
    if not changes:
        return ""
    omitted = f"\nЕщё изменившихся показателей: {len(changes) - 8}." if len(changes) > 8 else ""
    return "Изменения с последней сопоставимой доставкой:\n" + "\n".join(changes[:8]) + omitted + "\n\n"


def subscription_active(store, subscription_id, recipient, section):
    """A scheduled origin must still belong to this enabled destination/section."""
    if type(subscription_id) is not int or subscription_id <= 0:
        return False
    if not recipient_section_enabled(store, recipient, section):
        return False
    return bool(store.one(
        """SELECT s.id FROM subscriptions s LEFT JOIN users u ON u.user_id=s.user_id
        WHERE s.id=? AND s.recipient=? AND s.section=? AND s.enabled=1
        AND (s.user_id IS NULL OR (u.enabled=1 AND u.chat_id=s.recipient))
        AND NOT EXISTS (SELECT 1 FROM users revoked WHERE revoked.chat_id=s.recipient AND revoked.enabled=0)""",
        (subscription_id, str(recipient), section),
    ))


def prepare_delivery(store, report_id, recipient, detail="brief", filters=None, subscription_id=None, chart_schedule=None, project_ref=None, chart_auto=None):
    """Freeze a prefiltered report into idempotent Telegram delivery parts."""
    if detail not in {"brief", "full"}:
        raise ValueError("Choose brief or full")
    recipient = str(recipient).strip()
    if not recipient:
        raise ValueError("Recipient must not be empty")
    report = store.one("SELECT * FROM reports WHERE id=?", (report_id,))
    if report is None:
        raise ValueError("Report not found")
    from .subscription_access import report_section
    if not recipient_section_enabled(store, recipient, report_section(report)):
        return []
    if not recipient_functions_enabled(store, recipient, report):
        return []
    variant = detail + ":" + hashlib.sha256(dumps(filters or {}).encode("utf-8")).hexdigest()[:16]
    if project_ref:
        from .vivino import active
        if not active(store, project_ref, recipient):
            return []
    if chart_schedule:
        from .chart_schedules import active
        if not active(store, chart_schedule, recipient):
            return []
        variant += ':chart:' + chart_schedule['generation']
    if chart_auto:
        from .chart_batches import active
        if not active(store, chart_auto, recipient):
            return []
    if subscription_id is not None:
        if type(subscription_id) is not int or subscription_id <= 0:
            raise ValueError("Invalid subscription ID")
        # A cancelled or already delivered schedule must not suppress a later
        # explicit /report request for the same cached report.
        variant += f":subscription:{subscription_id}"
    body = report["full_text"] if detail == "full" else (_ai_brief(store, report, recipient) if report["section"] == "ai" else report["text"])
    if json.loads(report['quality']).get('tool') == 'dollar':
        body = report['text']
    text = _baseline_prefix(store, report, recipient) + body
    payloads = [{"kind": "text", "text": part} for part in split_message(text)]
    if detail == "brief" and payloads:
        callback = f"report:{report_id}"
        if len(callback.encode("utf-8")) <= 64:
            payloads[-1]["callback_data"] = callback
    if detail == "full":
        for part in json.loads(report["parts"]):
            if part.get("kind") not in {"photo", "document"}:
                continue
            payload = {"kind": part["kind"], "path": part["path"]}
            captions = split_message(str(part.get("caption") or ""), limit=1000)
            if captions:
                payload["caption"] = captions[0]
            payloads.append(payload)
            payloads.extend({"kind": "text", "text": caption} for caption in captions[1:])
        quality = json.loads(report['quality'])
        if quality.get('chart_pages', 1) > 1 and str(quality.get('actor_id')) == recipient:
            from .chart_batches import navigation
            payloads.append(navigation(store, report_id, quality))
        if quality.get('tool') == 'dollar' and str(quality.get('actor_id')) == recipient:
            payloads.append({'kind': 'text', 'text': 'Следующее действие', 'buttons': [
                [{'text': 'Настроить покупку', 'callback_data': 'usd:mode:purchase'}],
                [{'text': 'История для расчёта · от и до', 'callback_data': 'usd:period'}],
                [{'text': 'Срок прогноза · дни, месяцы, годы', 'callback_data': 'usd:horizon'}],
                [{'text': 'Курс на конкретную дату', 'callback_data': 'usd:set:forecast_date'}],
                [{'text': 'Подробности расчёта', 'callback_data': f'report:{report_id}'}],
                [{'text': 'Доллар · меню', 'callback_data': 'usd:menu'}]]})
        if quality.get('feature') == 'wine_cellar' and quality.get('wine_actions') and str(quality.get('actor_id')) == recipient:
            payloads.append({'kind': 'text', 'text': 'Моя винотека · следующее действие', 'buttons': quality['wine_actions']})
        if quality.get('feature') == 'life_finance' and quality.get('life_actions') and str(quality.get('actor_id')) == recipient:
            payloads.append({'kind': 'text', 'text': 'Жизнь · следующее действие', 'buttons': quality['life_actions']})
    if json.loads(report['quality']).get('integrity') and payloads:
        payloads[-1]['buttons'] = [[{'text': 'Открыть журнал целостности', 'callback_data': 'ig:menu'}]]
    if not payloads:
        raise ValueError("The report contains no deliverable content")
    if subscription_id is not None:
        for payload in payloads:
            payload["subscription_id"] = subscription_id
    if chart_schedule:
        for payload in payloads:
            payload['chart_schedule'] = chart_schedule
    if project_ref:
        for payload in payloads:
            payload['project_ref'] = project_ref
    if chart_auto:
        for payload in payloads:
            payload['chart_auto'] = chart_auto
    timestamp = utcnow()
    ids = []
    with store.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT id FROM deliveries WHERE report_id=? AND recipient=? AND variant=? ORDER BY part_index",
            (report_id, recipient, variant),
        ).fetchall()
        if existing:
            return [row["id"] for row in existing]
        for index, payload in enumerate(payloads):
            conn.execute(
                """INSERT OR IGNORE INTO deliveries
                (report_id,recipient,variant,part_index,payload,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?)""",
                (report_id, recipient, variant, index, dumps(payload), timestamp, timestamp),
            )
            ids.append(conn.execute(
                "SELECT id FROM deliveries WHERE report_id=? AND recipient=? AND variant=? AND part_index=?",
                (report_id, recipient, variant, index),
            ).fetchone()["id"])
    return ids


def recover_sending(store):
    """Call once after acquiring the service lock, before starting workers."""
    with store.connection() as conn:
        return conn.execute(
            """UPDATE deliveries SET state='uncertain',error='interrupted_send',updated_at=?
            WHERE state='sending'""", (utcnow(),),
        ).rowcount


def resolve_uncertain(store, delivery_id, action):
    """Admin adapters must authorize the actor before calling this operation."""
    if action not in {"retry", "sent"}:
        raise ValueError("Choose retry or sent")
    with store.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM deliveries WHERE id=?", (int(delivery_id),)).fetchone()
        if row is None or row["state"] != "uncertain":
            raise ValueError("Only an uncertain delivery can be resolved")
        conn.execute(
            """UPDATE deliveries SET state=?,attempts=?,retry_at=NULL,error=?,updated_at=? WHERE id=?""",
            ("pending" if action == "retry" else "sent",
             0 if action == "retry" else row["attempts"],
             "operator_requested_retry" if action == "retry" else "operator_confirmed_sent",
             utcnow(), int(delivery_id)),
        )


def _claim(store, delivery_id, timestamp):
    with store.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
        if row is None or row["state"] != "pending" or (row["retry_at"] and row["retry_at"] > timestamp):
            return None
        previous = conn.execute(
            """SELECT 1 FROM deliveries WHERE report_id=? AND recipient=? AND variant=?
            AND part_index<? AND state IN ('pending','sending','uncertain') LIMIT 1""",
            (row["report_id"], row["recipient"], row["variant"], row["part_index"]),
        ).fetchone()
        if previous:
            return None
        conn.execute(
            "UPDATE deliveries SET state='sending',attempts=attempts+1,updated_at=? WHERE id=?",
            (timestamp, delivery_id),
        )
        claimed = dict(row)
        claimed["attempts"] += 1
        return claimed


def _recipient_enabled(store, recipient):
    user = store.one("SELECT enabled FROM users WHERE chat_id=?", (recipient,))
    if user is not None:
        return bool(user["enabled"])
    # Legacy destinations may be channel usernames or chats with no interactive user.
    return True


def recipient_section_enabled(store, recipient, section):
    user = store.one('SELECT user_id FROM users WHERE chat_id=?', (str(recipient),))
    if user is None:
        return True  # Legacy destinations have no interactive user rights.
    from .subscription_access import allowed
    return allowed(store, user['user_id'], section)


def recipient_functions_enabled(store, recipient, report, payload=None):
    user = store.one('SELECT user_id FROM users WHERE chat_id=?', (str(recipient),))
    if user is None:
        return True
    from .feature_access import allowed, report_allowed
    if not report_allowed(store, user['user_id'], report):
        return False
    return not (payload and ('subscription_id' in payload or 'chart_schedule' in payload)) or allowed(store, user['user_id'], 'schedules')


def _send_part(client, recipient, payload):
    from pyrogram.enums import ParseMode
    from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    destination = int(recipient) if recipient.lstrip("-").isdigit() else recipient
    options = {"parse_mode": ParseMode.DISABLED}
    if payload.get("callback_data"):
        options["reply_markup"] = InlineKeyboardMarkup([
            [InlineKeyboardButton("Подробнее", callback_data=payload["callback_data"])]
        ])
    elif payload.get('buttons'):
        options['reply_markup'] = InlineKeyboardMarkup([
            [InlineKeyboardButton(button['text'], callback_data=button['callback_data']) for button in row]
            for row in payload['buttons']])
    if payload["kind"] == "text":
        result = client.send_message(destination, payload["text"], disable_web_page_preview=True, **options)
    else:
        send = client.send_photo if payload["kind"] == "photo" else client.send_document
        argument = "photo" if payload["kind"] == "photo" else "document"
        result = send(destination, **{argument: payload["path"]}, caption=payload.get("caption", ""), **options)
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise RuntimeError("Synchronous delivery must run outside the event-loop thread")
    if type(getattr(result, "id", None)) is not int:
        raise RuntimeError("Telegram did not return a message ID")
    return result.id


def _validate_payload(payload):
    if payload.get("kind") == "text":
        if not isinstance(payload.get("text"), str) or not payload["text"]:
            raise ValueError("Empty message")
    elif payload.get("kind") in {"photo", "document"}:
        if not isinstance(payload.get("path"), str) or not Path(payload["path"]).is_file():
            raise ValueError("Missing artifact")
    else:
        raise ValueError("Invalid delivery payload")


def _scheduled_part_active(store, row, payload):
    if 'chart_auto' in payload:
        from .chart_batches import active
        if not active(store, payload['chart_auto'], row['recipient']):
            return False
    if 'project_ref' in payload:
        from .vivino import active
        if not active(store, payload['project_ref'], row['recipient']):
            return False
    if 'chart_schedule' in payload:
        from .chart_schedules import active
        if not active(store, payload['chart_schedule'], row['recipient']):
            return False
    if "subscription_id" not in payload:
        return True
    report = store.one("SELECT section FROM reports WHERE id=?", (row["report_id"],))
    return bool(report and subscription_active(store, payload["subscription_id"], row["recipient"], report["section"]))


def _cancel_inactive_scheduled(store):
    # Include future FloodWait retries and later chunks blocked by an uncertain
    # earlier send. Completed and ambiguous parts remain exclusively operator-owned.
    for row in store.query("SELECT id,report_id,recipient,payload FROM deliveries WHERE state='pending'"):
        try:
            payload = json.loads(row["payload"])
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict) or not {'subscription_id', 'chart_schedule', 'project_ref', 'chart_auto'}.intersection(payload):
            continue
        if not _scheduled_part_active(store, row, payload):
            store.execute(
                """UPDATE deliveries SET state='cancelled',error=?,retry_at=NULL,updated_at=?
                WHERE id=? AND state='pending'""",
                ('chart_auto_inactive' if 'chart_auto' in payload else 'subscription_inactive', utcnow(), row["id"]),
            )


def deliver_pending(store, client):
    """Send ready parts synchronously. Run in a worker thread with Pyrogram."""
    from pyrogram.errors import FloodWait, RPCError

    from .chart_batches import pump

    pump(store)
    _cancel_inactive_scheduled(store)
    timestamp = utcnow()
    rows = store.query(
        """SELECT d.id FROM deliveries d WHERE d.state='pending' AND (d.retry_at IS NULL OR d.retry_at<=?)
        AND NOT EXISTS (SELECT 1 FROM deliveries earlier WHERE earlier.report_id=d.report_id
            AND earlier.recipient=d.recipient AND earlier.variant=d.variant AND earlier.part_index<d.part_index
            AND (earlier.state IN ('sending','uncertain') OR (earlier.state='pending' AND earlier.retry_at>?)))
        ORDER BY d.id LIMIT 100""", (timestamp, timestamp),
    )
    completed = 0
    for candidate in rows:
        row = _claim(store, candidate["id"], timestamp)
        if row is None:
            continue
        try:
            payload = json.loads(row["payload"])
            _validate_payload(payload)
            # The subscription can change after the sweep and before claiming.
            if not _scheduled_part_active(store, row, payload):
                store.execute(
                    "UPDATE deliveries SET state='cancelled',error=?,retry_at=NULL,updated_at=? WHERE id=?",
                    ('chart_auto_inactive' if 'chart_auto' in payload else 'subscription_inactive', utcnow(), row["id"]),
                )
                continue
            if not _recipient_enabled(store, row["recipient"]):
                raise ValueError("Recipient was disabled")
            from .subscription_access import report_section
            report = store.one('SELECT section,quality FROM reports WHERE id=?', (row['report_id'],))
            if not report or not recipient_section_enabled(store, row['recipient'], report_section(report)):
                store.execute("UPDATE deliveries SET state='cancelled',updated_at=? WHERE id=?", (utcnow(), row['id']))
                continue
            if not recipient_functions_enabled(store, row['recipient'], report, payload):
                store.execute("UPDATE deliveries SET state='cancelled',error='feature_access_revoked',updated_at=? WHERE id=?", (utcnow(), row['id']))
                continue
        except (ValueError, TypeError, KeyError, OSError):
            store.execute("UPDATE deliveries SET state='failed',error='invalid_payload_or_recipient',updated_at=? WHERE id=?",
                          (utcnow(), row["id"]))
            continue
        try:
            message_id = _send_part(client, row["recipient"], payload)
        except Exception as exc:  # noqa: BLE001 - Unknown send outcomes must be persisted as uncertain, never replayed automatically.
            retry_at = None
            if isinstance(exc, FloodWait):
                state = "pending" if row["attempts"] < MAX_ATTEMPTS else "failed"
                delay = max(1, int(exc.value or 1))
                retry_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
            elif isinstance(exc, RPCError) and exc.CODE in {400, 401, 403, 406}:
                state = "failed"
            else:
                state = "uncertain"
            # Exception messages can contain request data; retain only the class.
            store.execute(
                "UPDATE deliveries SET state=?,error=?,retry_at=?,updated_at=? WHERE id=?",
                (state, type(exc).__name__, retry_at, utcnow(), row["id"]),
            )
            continue
        # If persistence fails after send, keep 'sending'; startup recovery makes
        # this uncertain instead of risking an automatic duplicate.
        store.execute(
            "UPDATE deliveries SET state='sent',message_id=?,error=NULL,retry_at=NULL,updated_at=? WHERE id=?",
            (message_id, utcnow(), row["id"]),
        )
        completed += 1
    pump(store)
    return completed
