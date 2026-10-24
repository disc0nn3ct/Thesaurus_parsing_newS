"""Count each provider attempt before calling it, including retries."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .store import dumps, utcnow


class BudgetExceeded(RuntimeError):
    pass


def reserve(store, provider, now=None):
    if provider not in {"wordstat", "ai"}:
        raise ValueError("Unknown provider")
    day = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("Europe/Moscow")).date().isoformat()
    with store.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        import json
        row = conn.execute("SELECT value FROM settings WHERE key='budgets'").fetchone()
        config = json.loads(row[0])
        limit = int(config[f"{provider}_calls"])
        used = conn.execute("SELECT COALESCE(SUM(calls),0) FROM usage WHERE provider=? AND day=?",
                            (provider, day)).fetchone()[0]
        if used >= limit:
            raise BudgetExceeded(f"Дневной лимит {provider} исчерпан")
        return conn.execute("INSERT INTO usage(provider,day,created_at) VALUES(?,?,?)",
                            (provider, day, utcnow())).lastrowid


def finish(store, usage_id, *, tokens_in=0, tokens_out=0, failed=False):
    row = store.one("SELECT provider FROM usage WHERE id=?", (usage_id,))
    if not row:
        raise ValueError("Unknown usage reservation")
    prices = store.get_setting("budgets", {}).get("prices", {}).get(row["provider"], {})
    cost = None
    if prices:
        cost = (float(prices.get("per_call", 0)) + tokens_in * float(prices.get("input_per_million", 0)) / 1e6
                + tokens_out * float(prices.get("output_per_million", 0)) / 1e6)
    store.execute("UPDATE usage SET tokens_in=?,tokens_out=?,cost=?,status=? WHERE id=?",
                  (tokens_in, tokens_out, cost, "failed" if failed else "complete", usage_id))


def set_budgets(store, config):
    current = store.get_setting("budgets")
    if "currency" in config:
        currency = str(config["currency"]).strip()
        if not 1 <= len(currency) <= 12:
            raise ValueError("Currency label must contain 1 to 12 characters")
        if currency != current.get("currency", "у.е.") and store.one("SELECT 1 FROM usage WHERE cost IS NOT NULL LIMIT 1"):
            raise ValueError("Cannot change currency after costs have been recorded")
        current["currency"] = currency
    for field in ("wordstat_calls", "ai_calls", "ai_output_tokens"):
        if field in config:
            number = int(config[field])
            if number < (1 if field == "ai_output_tokens" else 0):
                raise ValueError("Limits must be nonnegative and output tokens positive")
            current[field] = number
    if "prices" in config:
        prices = config["prices"]
        if not isinstance(prices, dict):
            raise ValueError("Prices must be a dictionary")
        for values in prices.values():
            if any(float(v) < 0 for v in values.values()):
                raise ValueError("Prices cannot be negative")
        current["prices"] = prices
    dumps(current)
    store.set_setting("budgets", current)
