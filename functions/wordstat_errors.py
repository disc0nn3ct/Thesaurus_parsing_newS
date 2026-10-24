"""Safe Wordstat failure codes without configuration or networking side effects."""

class WordstatError(RuntimeError):
    """A provider failure described only by trusted codes, never response text."""

    def __init__(self, kind, status=None, attempts=None):
        self.kind = kind
        self.status = status if type(status) is int and 100 <= status <= 599 else None
        self.attempts = attempts if type(attempts) is int and attempts > 0 else None
        super().__init__(self.safe_message())

    @property
    def stops_collection(self):
        return self.kind in {'configuration', 'cloud_configuration', 'backend', 'tls', 'network'} or (
            self.kind == 'http' and self.status in {401, 403, 404, 429, 500, 502, 503, 504})

    def safe_message(self):
        if self.kind == "configuration":
            return "Wordstat: заполните WORDSTAT_OAUTH и WORDSTAT_CLIENT_ID в .env и перезапустите сервис"
        if self.kind == 'cloud_configuration':
            return 'Wordstat: для cloud задайте WORDSTAT_FOLDER_ID и ровно один WORDSTAT_API_KEY или WORDSTAT_IAM_TOKEN в .env'
        if self.kind == 'backend':
            return 'Wordstat: WORDSTAT_BACKEND должен быть legacy или cloud'
        if self.kind == 'tls':
            return ('Wordstat: ошибка проверки SSL/TLS-сертификата. Проверьте адрес источника и сертификаты сети. '
                    'Для нового Yandex Search API используйте WORDSTAT_BACKEND=cloud с отдельными ключом и ID каталога. '
                    'Проверка HTTPS не отключена; сбор остановлен.')
        if self.kind == 'network':
            return f'Wordstat: соединение не установлено после {self.attempts} попыток; проверьте сеть, DNS и прокси. Сбор остановлен.'
        if self.kind == "http":
            reasons = {
                401: "авторизация отклонена; проверьте OAuth-токен (legacy) или API-ключ/IAM-токен (cloud)",
                403: "доступ запрещён; проверьте разрешение Wordstat API и Client ID (legacy) или роль search-api.webSearch.user и каталог (cloud)",
                404: "метод API не найден; требуется проверить актуальный интерфейс Яндекса",
                429: "лимит Яндекса исчерпан; попробуйте позже",
            }
            reason = reasons.get(self.status, "источник вернул ошибку")
            exhausted = f' после {self.attempts} попыток' if self.attempts else ''
            return f"Wordstat: HTTP {self.status}{exhausted}; {reason}"
        return "Wordstat: источник недоступен или ответ некорректен"
