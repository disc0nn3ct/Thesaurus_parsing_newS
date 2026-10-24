# Запуск и обновление «Тезауруса»

## Быстрый запуск на этом компьютере

Откройте PowerShell и выполните:

```powershell
Set-Location -LiteralPath 'D:\study\project\Thesaurus_parsing_newS'
.\.venv\Scripts\python.exe -X utf8 -u main.py serve
```

Окно оставьте открытым. Не запускайте вторую копию: один исполнитель обслуживает
Telegram, очередь, расписания и пользователей. При закрытии окна, выключении
или перезагрузке компьютера бот перестанет работать. После включения повторите
эти команды. Сон компьютера тоже приостанавливает обработку.

Запуск может занять некоторое время: приложение проверяет базу и каталог источников,
затем подключается к Telegram. Предупреждение `TgCrypto is missing` не означает
ошибку запуска: это необязательное ускорение Pyrogram.

**Используйте именно `serve`.** Запуск `main.py` без команды выполняет прежний
однократный сценарий отчётов, а не постоянную работу бота.

## Как проверить работу

В личном чате с ботом отправьте `/start` или `/projects`. Администратор может открыть
`/monitor`: должен быть свежий сигнал активности и подтверждённое соединение с Telegram.

В другом окне PowerShell можно посмотреть состояние сохранённой очереди:

```powershell
Set-Location -LiteralPath 'D:\study\project\Thesaurus_parsing_newS'
.\.venv\Scripts\python.exe -X utf8 main.py status
```

Эта команда сама по себе не подтверждает, что работающий бот подключён к Telegram;
соединение проверяйте через ответ бота и свежий сигнал в мониторинге.

Проверить, есть ли процесс запуска:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'main\.py\s+serve' } |
    Select-Object ProcessId, ParentProcessId, ExecutablePath, CommandLine
```

Для виртуального окружения Windows могут отображаться два связанных процесса:
запускающий `.venv` и дочерний Python. Это не обязательно две копии бота.

## Остановка и перезапуск

В окне, где запущен `serve`, нажмите **Ctrl+C** и дождитесь завершения. Текущее
задание может потребовать времени для окончания. Затем снова выполните команду
запуска. Для обычного перезапуска ничего удалять или повторно импортировать не нужно.

Если бот запущен в фоне, закрытие нового окна PowerShell его не остановит.
Проверьте процессы командой выше; не запускайте ещё одну копию поверх существующей.
Фоновые запуски, сделанные при сопровождении проекта, пишут вывод в `data/logs/`.

## Первая установка или другой компьютер

Нужны Git и установленный Python 3.12. Команды ниже предназначены для нового клона;
существующий рабочий каталог заново клонировать не нужно.

```powershell
git clone https://github.com/disc0nn3ct/Thesaurus_parsing_newS.git
Set-Location -LiteralPath '.\Thesaurus_parsing_newS'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Создайте локальную конфигурацию, только если её ещё нет:

```powershell
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
notepad .env
```

Заполните `api_id`, `api_hash`, `bot_token` и `ADMIN_TELEGRAM_IDS`. В последней
переменной укажите числовые Telegram ID администраторов через запятую.
Значения доступа к Telegram берутся из вашей регистрации приложения Telegram и
настроек бота; чужой ID или `@username` не заменяет собственные реквизиты.

Для текущего режима оставьте:

```dotenv
SEARCH_PROVIDER=disabled
AUTO_UPDATE=0
DASHBOARD_ENABLED=0
AI_BACKEND=template
```

`for_whom_list` нужен для прежнего однократного сценария рассылки; получатели не
становятся администраторами. Пользователей, доступные разделы и расписания
настраивайте через меню администратора. Для постоянного бота запускайте:

```powershell
.\.venv\Scripts\python.exe -X utf8 -u main.py serve
```

Необязательные OCR и другие дополнительные компоненты описаны в README.
Для запуска обычного Telegram-бота веб-панель не требуется.

## Где лежат данные

- `data/thesaurus.sqlite3` — пользователи, права, вина, покупки, цены, списания,
  статистика, чеки, задания и история мониторинга.
- `data/` — также локальные файлы и служебные данные.
- `src/reports/` — сформированные отчёты и вложения по умолчанию.
- `.env` — локальные настройки и секреты.
- `me_client.session` — локальная сессия Telegram; рядом могут быть служебные файлы SQLite.

Если в `.env` задан `THESAURUS_DATA_DIR`, база находится в этом каталоге вместо
стандартного `data/`. Используйте тот же путь после обновления. Новый клон с пустым
каталогом данных создаёт новую базу и не подтягивает личные данные из GitHub.

Эти данные не публикуются в Git. Обновление кода не должно удалять `.env`, `data/`,
отчёты или сессию. `git reset --hard`, очистка каталога и пересоздание базы для
обычного обновления не нужны.

## Резервная копия

Сначала корректно остановите бот. При остановленном боте можно скопировать
локальные данные и конфигурацию в закрытую папку резервных копий:

```powershell
Set-Location -LiteralPath 'D:\study\project\Thesaurus_parsing_newS'
$backupStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backupFolder = Join-Path (Get-Location).Path "backups\$backupStamp"
New-Item -ItemType Directory -Path $backupFolder -Force | Out-Null
Copy-Item -LiteralPath '.\data' -Destination $backupFolder -Recurse
Copy-Item -LiteralPath '.\.env' -Destination $backupFolder
if (Test-Path -LiteralPath '.\src\reports') {
    Copy-Item -LiteralPath '.\src\reports' -Destination (Join-Path $backupFolder 'reports') -Recurse
}
Get-ChildItem -LiteralPath '.' -Filter 'me_client.session*' -File |
    Copy-Item -Destination $backupFolder
```

При нестандартном `THESAURUS_DATA_DIR` замените `data` фактическим каталогом.
Папка `backups/` исключена из Git. Она содержит личные данные и секреты — храните
её с ограниченным доступом и дополнительно копируйте на надёжный носитель.
Во время работы бота простая копия одного файла SQLite может быть неполной
из-за WAL: остановите бот либо используйте SQLite backup API.

## Как обновить код из GitHub

Остановите бот, сделайте резервную копию и проверьте локальные изменения:

```powershell
git status --short
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 -u main.py serve
```

Если Git сообщает о локальных изменениях или расхождении веток, сначала разберите
их. Не сбрасывайте их принудительно. После запуска проверьте `/monitor`, остаток
вин и доступ пользователей. Расписания включаются отдельно от выдачи прав.

## Linux-сервер

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py serve
```

Для постоянной службы используйте `deploy/thesaurus.service.example`: замените
пути и пользователя службы, сохраните локальные `.env` и данные, затем управляйте
службой средствами systemd. В README есть описание установки службы.
