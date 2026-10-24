"""Shared administrative mutations, independent of web and Telegram transports."""

import json
import math

from .store import dumps


def resolve_proposal(store, proposal_id, action):
    """Apply a pending suggestion and its version history in one transaction."""
    if action not in {"approve", "reject"}:
        raise ValueError("Выберите approve или reject")
    with store.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT entries FROM keyword_proposals WHERE id=? AND state='pending'",
                           (proposal_id,)).fetchone()
        if not row:
            raise ValueError("Предложение уже рассмотрено или не найдено")
        state = "rejected"
        if action == "approve":
            current = json.loads(conn.execute("SELECT value FROM settings WHERE key='keywords'").fetchone()[0])
            try:
                proposed = json.loads(row["entries"])
                if not isinstance(proposed, list):
                    raise TypeError()
                entries = []
                for item in current + proposed:
                    phrase = str(item.get("phrase", "")).strip()
                    if not phrase or len(phrase) > 120:
                        raise ValueError()
                    if phrase not in [entry["phrase"] for entry in entries]:
                        entries.append({"phrase": phrase, "theme": str(item.get("theme", "другое"))[:80]})
                if not 1 <= len(entries) <= 100:
                    raise ValueError()
            except (ValueError, TypeError, AttributeError):
                raise ValueError("Предложение не удалось применить. Проверьте список запросов") from None
            if current != entries:
                version = int(json.loads(conn.execute(
                    "SELECT value FROM settings WHERE key='keyword_version'").fetchone()[0])) + 1
                for key, value in (("keywords", entries), ("keyword_version", version),
                                   (f"keywords_history_{version}", entries)):
                    conn.execute("INSERT INTO settings(key,value) VALUES(?,?) "
                                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, dumps(value)))
            state = "approved"
        conn.execute("UPDATE keyword_proposals SET state=? WHERE id=? AND state='pending'", (state, proposal_id))
    return state


def alert_filters(store, metric, region_id):
    metric = str(metric).strip() if metric is not None else None
    metric = metric or None
    if metric and metric not in set(store.get_setting("incident_phrases", [])):
        raise ValueError("Выберите фразу из списка региональных наблюдений")
    if region_id is not None:
        raw = str(region_id)
        if not raw.isascii() or not raw.isdigit() or int(raw) <= 0:
            raise ValueError("Нужен положительный числовой ID региона")
        region_id = int(raw)
        if region_id not in set(store.regions()):
            raise ValueError("Выберите регион из утверждённой когорты")
    return metric, region_id


def save_alert_rule(store, values, rule_id=None):
    """Validate all merged fields before creating or updating a rule."""
    fields = ("pct_threshold", "z_threshold", "min_coverage", "min_baseline", "cooldown_hours")
    allowed = {"name", "metric", "region_id", "enabled", *fields}
    if set(values) - allowed:
        raise ValueError("Неизвестная настройка правила")
    existing = store.one("SELECT * FROM alert_rules WHERE id=?", (rule_id,)) if rule_id is not None else None
    if rule_id is not None and not existing:
        raise ValueError("Правило не найдено")
    merged = {"name": "", "metric": None, "region_id": None, "enabled": 0,
              "pct_threshold": 50, "z_threshold": 2.5, "min_coverage": .9,
              "min_baseline": 20, "cooldown_hours": 24, **(existing or {}), **values}
    name = str(merged["name"]).strip()
    if not 1 <= len(name) <= 120 or any(ord(char) < 32 for char in name):
        raise ValueError("Название правила должно содержать от 1 до 120 символов")
    if merged["enabled"] not in (0, 1):
        raise ValueError("Выберите on или off")
    try:
        thresholds = [float(merged[key]) for key in fields]
        if any(not math.isfinite(value) or value < 0 for value in thresholds) or thresholds[2] > 1:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("Пороги должны быть неотрицательными, coverage — от 0 до 1") from None
    # An obsolete cohort must never prevent the operator from switching a rule off.
    if existing and values == {"enabled": 0}:
        metric, region_id = merged["metric"], merged["region_id"]
    else:
        metric, region_id = alert_filters(store, merged["metric"], merged["region_id"])
    args = (name, metric, region_id, int(merged["enabled"]), *thresholds)
    if rule_id is None:
        return store.execute("INSERT INTO alert_rules(name,metric,region_id,enabled,pct_threshold,z_threshold,"
                             "min_coverage,min_baseline,cooldown_hours) VALUES(?,?,?,?,?,?,?,?,?)", args)
    store.execute("UPDATE alert_rules SET name=?,metric=?,region_id=?,enabled=?,pct_threshold=?,z_threshold=?,"
                  "min_coverage=?,min_baseline=?,cooldown_hours=? WHERE id=?", (*args, rule_id))
    return rule_id
