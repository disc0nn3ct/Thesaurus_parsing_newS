"""Local data explanations. No web-search capability or implicit paid fallback."""
import json
import os
from urllib.parse import urlsplit

import requests

from .store import dumps


def backend():
    value = os.getenv("AI_BACKEND", "perplexity").strip().lower()
    if value not in {"template", "ollama", "perplexity"}:
        raise ValueError("AI_BACKEND must be template, ollama or perplexity")
    return value


def template(context, note=""):
    lines = ["Обзор сохранённых данных (без языковой модели и платных API)."]
    if note:
        lines.append(note)
    if not context:
        lines.append("Данных пока нет. Сначала запросите /report ruonia full или /report wordstat full.")
    for row in context[:30]:
        lines.append(f"• {row['metric']}: {row['value']:.4g} {row['unit']} ({row['date']}).")
    lines.append("Поиск новостей не выполнялся. Прогнозы доступны в /forecasts ruonia и /forecasts wordstat.")
    return "\n".join(lines), "Без платных токенов", {"claims": [], "backend": "template"}


def chat(prompt):
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
        raise ValueError("OLLAMA_URL must point to a local HTTP server")
    response = requests.post(url + "/api/chat", json={
        "model": os.getenv("OLLAMA_MODEL", "qwen3:1.7b"), "stream": False, "think": False,
        "keep_alive": "0", "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": 2048, "num_predict": 384, "num_thread": 4, "temperature": .1},
    }, timeout=(5, 180), allow_redirects=False, proxies={"http": "", "https": "", "all": ""})
    response.raise_for_status()
    if response.status_code != 200:
        raise ValueError("Unexpected local response")
    data = response.json()
    text = data.get("message", {}).get("content")
    if not isinstance(text, str) or not text.strip() or len(text) > 12000:
        raise ValueError("Invalid local model response")
    return text.strip()


def overview(context):
    if backend() == "template":
        return template(context)
    try:
        prompt = ("Кратко объясни по-русски только приложенные данные. Не придумывай новости, даты, "
                  "ссылки, числа или причины. Не давай финансовых рекомендаций. Ответ до 200 слов. Данные:\n"
                  + dumps(context[:20]))
        text = chat(prompt)
        return ("Локальный комментарий модели, выводы не подтверждены автоматически:\n" + text
                + "\nПоиск новостей не выполнялся.", "Локально, без платы за токены",
                {"claims": [], "backend": "ollama"})
    except (requests.RequestException, ValueError, KeyError, TypeError):
        text, usage, data = template(context, "Локальная модель недоступна; использован обзор по шаблону.")
        return text, usage, {**data, "fallback": True}


def proposals(current):
    if backend() != "ollama":
        raise ValueError("Для предложений фраз включите AI_BACKEND=ollama и запустите модель")
    content = chat("Предложи до 5 дополнительных фраз к набору. Только JSON-массив "
                   '[{"phrase":"...","theme":"..."}], без пояснений. Набор: ' + dumps(current))
    entries = json.loads(content)
    if (not isinstance(entries, list) or not 1 <= len(entries) <= 5
            or any(not isinstance(item, dict) or not isinstance(item.get("phrase"), str)
                   or not 1 <= len(item["phrase"].strip()) <= 120 for item in entries)):
        raise ValueError("Invalid local keyword proposals")
    return entries
