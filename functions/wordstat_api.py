import os
import json
import time
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

WORDSTAT_API = "https://api.wordstat.yandex.net"
WORDSTAT_OAUTH = os.getenv("WORDSTAT_OAUTH")
WORDSTAT_CLIENT_ID = os.getenv("WORDSTAT_CLIENT_ID")


def _wordstat_post(path: str, payload: dict, retries: int = 4, backoff: float = 1.5):
    """
    Универсальный POST к Wordstat API.
    """
    if not WORDSTAT_OAUTH or not WORDSTAT_CLIENT_ID:
        raise RuntimeError("WORDSTAT_OAUTH или WORDSTAT_CLIENT_ID отсутствуют в .env")

    url = f"{WORDSTAT_API}{path}"
    headers = {
        "Authorization": f"Bearer {WORDSTAT_OAUTH}",
        "Content-Type": "application/json; charset=utf-8",
        "X-Client-Id": WORDSTAT_CLIENT_ID,
    }

    for attempt in range(1, retries + 1):
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)

        if r.status_code == 200:
            return r.json()

        if r.status_code in (429, 503):
            wait = backoff ** attempt
            logger.warning(f"Wordstat {path} вернул {r.status_code}. Ретрай через {wait:.1f}s...")
            time.sleep(wait)
            continue

        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Wordstat error {r.status_code}: {err}")

    raise RuntimeError(f"Wordstat {path} не ответил после {retries} попыток")
