"""Disabled schedule templates and durable, coalesced report jobs."""

import json
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .store import SECTIONS, dumps, utcnow

MOSCOW = ZoneInfo("Europe/Moscow")
TEMPLATES = {
    "ai": {"times": ["08:30", "18:30"], "weekdays": list(range(7))},
    "ruonia": {"times": ["15:30"], "weekdays": list(range(7))},
    "regions": {"times": ["10:00"], "weekdays": list(range(7))},
    "wordstat": {"times": ["09:00"], "weekdays": [2]},
    "market": {"times": ["18:30"], "weekdays": list(range(7))},
}


def seed_subscriptions(store, recipient, user_id=None):
    """Add missing templates without enabling delivery or granting access."""
    recipient = str(recipient).strip()
    if not recipient:
        raise ValueError("Recipient must not be empty")
    with store.connection() as conn:
        for section, template in TEMPLATES.items():
            conn.execute(
                """INSERT OR IGNORE INTO subscriptions
                (user_id,recipient,section,times,weekdays,created_at)
                VALUES(?,?,?,?,?,?)""",
                (user_id, recipient, section, dumps(template["times"]),
                 dumps(template["weekdays"]), utcnow()),
            )
        if user_id is not None:
            conn.execute(
                "UPDATE subscriptions SET user_id=? WHERE recipient=? AND user_id IS NULL",
                (user_id, recipient),
            )


def validate_subscription(values):
    """Normalize shared bot/dashboard schedule fields before storing them."""
    allowed = {"enabled", "times", "weekdays", "detail", "phrases", "regions", "timezone"}
    if set(values) - allowed:
        raise ValueError("Unknown subscription setting")
    result = {}
    for key, value in values.items():
        if key == "enabled":
            if value not in (True, False, 0, 1):
                raise ValueError("enabled must be a boolean")
            result[key] = int(value)
        elif key == "times":
            if not isinstance(value, list) or not 1 <= len(value) <= 12:
                raise ValueError("Choose 1 to 12 times")
            if any(not isinstance(t, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", t) for t in value):
                raise ValueError("Use HH:MM in Moscow time")
            result[key] = sorted(set(value))
        elif key == "weekdays":
            if (not isinstance(value, list) or not value
                    or any(type(day) is not int or day not in range(7) for day in value)):
                raise ValueError("Weekdays must be integers from 0 to 6")
            result[key] = sorted(set(value))
        elif key == "detail":
            if value not in {"brief", "full"}:
                raise ValueError("Choose brief or full")
            result[key] = value
        elif key == "timezone":
            try:
                ZoneInfo(value)
            except (ValueError, TypeError, KeyError):
                raise ValueError("Choose a valid IANA timezone") from None
            result[key] = value
        elif key == "phrases":
            if (not isinstance(value, list) or len(value) > 100
                    or any(not isinstance(v, str) or not 1 <= len(v.strip()) <= 120 for v in value)):
                raise ValueError("Choose at most 100 nonempty phrases")
            result[key] = list(dict.fromkeys(v.strip() for v in value))
        elif key == "regions":
            if (not isinstance(value, list) or len(value) > 100
                    or any(type(v) is not int or v <= 0 for v in value)):
                raise ValueError("Region IDs must be positive integers")
            result[key] = sorted(set(value))
    return result


def validate_filters(store, section, phrases=None, regions=None):
    """Validate filters against the selected section's actual source universe."""
    if section not in SECTIONS:
        raise ValueError("Unknown report section")
    values = validate_subscription({"phrases": [] if phrases is None else phrases,
                                    "regions": [] if regions is None else regions})
    if section == 'market':
        allowed_phrases = {row['id'] for row in store.query("SELECT id FROM market_series WHERE enabled=1 AND json_extract(metadata,'$.kind')='series'")}
        allowed_regions = {225}
    elif section == "ruonia":
        from .cbr_data import METRICS
        allowed_phrases, allowed_regions = set(METRICS), {225}
    elif section == "wordstat":
        allowed_phrases, allowed_regions = {entry["phrase"] for entry in store.keywords()}, {225}
    elif section == "regions":
        allowed_phrases = set(store.get_setting("incident_phrases", []))
        allowed_regions = set(store.regions())
    else:
        allowed_phrases, allowed_regions = set(), set()
    if not set(values["phrases"]).issubset(allowed_phrases):
        raise ValueError("Choose phrases supported by this report section")
    if not set(values["regions"]).issubset(allowed_regions):
        raise ValueError("Choose regions supported by this report section")
    return values


def update_subscription(store, user_id, section, **values):
    """Update only the authenticated user's private-chat subscription."""
    from .google_trends import active
    from .search_provider import require_section
    if values.get('enabled'):
        require_section(store, section)
    if section in {'wordstat', 'regions'} and values.get('enabled') and active(store):
        raise ValueError('Wordstat выключен. Для Google Trends загрузите CSV через /trends; рассылка входит в раздел market.')
    if not store.authorize(user_id):
        raise PermissionError("Access denied")
    from .subscription_access import require
    require(store, user_id, section)
    if section not in SECTIONS:
        raise ValueError("Unknown report section")
    values = validate_subscription(values)
    existing = store.one("SELECT phrases,regions FROM subscriptions WHERE user_id=? AND recipient=? AND section=?",
                         (user_id, str(user_id), section))
    phrases = values.get("phrases", json.loads(existing["phrases"]) if existing else [])
    regions = values.get("regions", json.loads(existing["regions"]) if existing else [])
    values.update(validate_filters(store, section, phrases, regions))
    seed_subscriptions(store, str(user_id), user_id)
    if not values:
        return
    encoded = {k: dumps(v) if isinstance(v, list) else v for k, v in values.items()}
    assignments = ",".join(f"{key}=?" for key in encoded)
    with store.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        require(store, user_id, section)
        conn.execute(f"UPDATE subscriptions SET {assignments} WHERE user_id=? AND recipient=? AND section=?",
                     (*encoded.values(), user_id, str(user_id), section))


def _aware(value):
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("Schedule timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _latest_due(subscription, now):
    settings = validate_subscription({
        "times": json.loads(subscription["times"]),
        "weekdays": json.loads(subscription["weekdays"]),
        "timezone": subscription["timezone"],
    })
    zone = ZoneInfo(settings["timezone"])
    local = now.astimezone(zone)
    lower_bound = _aware(subscription["created_at"])
    if subscription["last_scheduled_at"]:
        lower_bound = max(lower_bound, _aware(subscription["last_scheduled_at"]))
    candidates = []
    for offset in range(8):
        day = local.date() - timedelta(days=offset)
        if day.weekday() not in settings["weekdays"]:
            continue
        for clock in settings["times"]:
            hour, minute = map(int, clock.split(":"))
            slot = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
            # A nonexistent local time during a DST jump is skipped.
            if slot.astimezone(timezone.utc).astimezone(zone).replace(fold=0) != slot:
                continue
            slot = slot.astimezone(timezone.utc)
            if lower_bound < slot <= now:
                candidates.append(slot)
    return max(candidates) if candidates else None


def enqueue_due(store, now=None):
    """Queue only the newest missed slot per subscription, atomically."""
    now = _aware(now or datetime.now(timezone.utc))
    queued = []
    with store.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """SELECT s.* FROM subscriptions s LEFT JOIN users u ON u.user_id=s.user_id
            WHERE s.enabled=1 AND (s.user_id IS NULL OR u.enabled=1)"""
        ).fetchall()
        for row in rows:
            subscription = dict(row)
            from .subscription_access import allowed
            if subscription['user_id'] is not None and not allowed(store, subscription['user_id'], subscription['section']):
                continue
            from .feature_access import allowed as function_allowed
            if subscription['user_id'] is not None and not all(function_allowed(store, subscription['user_id'], name) for name in ('reports', 'schedules')):
                continue
            try:
                from .search_provider import require_section
                require_section(store, subscription['section'])
                slot = _latest_due(subscription, now)
                phrases = json.loads(subscription["phrases"])
                regions = json.loads(subscription["regions"])
                validate_subscription({"phrases": phrases, "regions": regions, "detail": subscription["detail"]})
                validate_filters(store, subscription["section"], phrases, regions)
            except (ValueError, TypeError, KeyError):
                # One malformed persisted schedule must not block all recipients.
                continue
            if slot is None or subscription["section"] not in SECTIONS:
                continue
            timestamp = slot.isoformat()
            payload = {
                "section": subscription["section"], "recipient": subscription["recipient"],
                "detail": subscription["detail"], "phrases": phrases,
                "regions": regions,
                "subscription_id": subscription["id"], "scheduled_for": timestamp,
            }
            key = f"schedule:{subscription['id']}:{timestamp}"
            pending = conn.execute(
                "SELECT id FROM jobs WHERE state='pending' AND dedupe_key LIKE ? ORDER BY id LIMIT 1",
                (f"schedule:{subscription['id']}:%",),
            ).fetchone()
            if pending:
                job_id = pending["id"]
                conn.execute("UPDATE jobs SET payload=?,dedupe_key=? WHERE id=? AND state='pending'",
                             (dumps(payload), key, job_id))
            else:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO jobs(kind,payload,dedupe_key,created_at) VALUES('report',?,?,?)",
                    (dumps(payload), key, now.isoformat()),
                )
                job_id = cursor.lastrowid if cursor.rowcount else None
            conn.execute("UPDATE subscriptions SET last_scheduled_at=? WHERE id=?",
                         (timestamp, subscription["id"]))
            if job_id is not None:
                queued.append(job_id)
    return queued
