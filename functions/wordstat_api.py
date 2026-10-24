import os
import time
import requests
import logging
from pathlib import Path
from dotenv import load_dotenv
from functions.wordstat_errors import WordstatError

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logger = logging.getLogger(__name__)

WORDSTAT_API = "https://api.wordstat.yandex.net"
WORDSTAT_OAUTH = os.getenv("WORDSTAT_OAUTH")
WORDSTAT_CLIENT_ID = os.getenv("WORDSTAT_CLIENT_ID")


def _wordstat_post(path: str, payload: dict, retries: int = 4, backoff: float = 1.5, before_attempt=None):
    """
    Универсальный POST к Wordstat API.
    """
    if retries < 1:
        raise ValueError("retries должен быть не меньше 1")
    if backoff <= 0:
        raise ValueError("backoff должен быть положительным")

    backend = os.getenv('WORDSTAT_BACKEND', 'legacy').strip().lower()
    if backend not in {'legacy', 'cloud'}:
        raise WordstatError('backend')
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    if backend == 'cloud':
        from functions.wordstat_cloud import request_body
        key = os.getenv('WORDSTAT_API_KEY', '').strip()
        token = os.getenv('WORDSTAT_IAM_TOKEN', '').strip()
        folder = os.getenv('WORDSTAT_FOLDER_ID', '').strip()
        if not folder or bool(key) == bool(token):
            raise WordstatError('cloud_configuration')
        url, payload = request_body(path, payload, folder)
        headers['Authorization'] = f'Api-key {key}' if key else f'Bearer {token}'
    else:
        if not WORDSTAT_OAUTH or not WORDSTAT_CLIENT_ID:
            raise WordstatError('configuration')
        url = f"{WORDSTAT_API}{path}"
        headers.update(Authorization=f'Bearer {WORDSTAT_OAUTH}', **{'X-Client-Id': WORDSTAT_CLIENT_ID})

    for attempt in range(1, retries + 1):
        if before_attempt is not None:
            before_attempt()
        try:
            # Не пересылаем заголовки авторизации по редиректам API.
            r = requests.post(url, headers=headers, json=payload, timeout=30, allow_redirects=False)
        except requests.exceptions.SSLError:
            # SSLError inherits ConnectionError. Certificate failures are not transient.
            raise WordstatError('tls') from None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            # Текст сетевого исключения может содержать данные запроса.
            failure = "сетевая ошибка"
            status = None
        else:
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:
                    raise RuntimeError(f"Wordstat {path}: некорректный JSON в ответе") from None
                if isinstance(data, dict) and ("error" in data or "errors" in data):
                    raise RuntimeError(f"Wordstat {path}: API вернул ошибку")
                if backend == 'cloud':
                    from functions.wordstat_cloud import response_body
                    return response_body(path, data)
                return data

            if r.status_code not in (429, 500, 502, 503, 504):
                # Не включаем тело ответа: сервер может отразить в нём секреты.
                raise WordstatError("http", r.status_code)
            failure = f"HTTP {r.status_code}"
            status = r.status_code

        if attempt < retries:
            wait = backoff ** attempt
            logger.warning("Wordstat %s: %s. Повтор через %.1fs", path, failure, wait)
            time.sleep(wait)

    raise WordstatError('network' if status is None else 'http', status, attempts=retries) from None
