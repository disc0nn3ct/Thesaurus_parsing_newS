"""CLI entrypoints. A dry run never creates a Telegram client or calls a provider."""
import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
from contextlib import contextmanager, suppress
from pathlib import Path
from uuid import uuid4

from .config import ConfigurationError, Settings
from .locking import worker_lock
from .store import SECTIONS, Store, dumps


@contextmanager
def _termination_signals(stop):
    """Finish the active operation and close Telegram on server shutdown."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda _number, _frame: stop.set())
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def parser():
    result = argparse.ArgumentParser(description="Thesaurus: RUONIA, Wordstat и AI-отчёты")
    result.add_argument("command", choices=["once", "serve", "dashboard", "status", "export", "backtest"], nargs="?", default="once")
    result.add_argument("--dry-run", action="store_true", help="Только сохранённые данные; без API и отправки")
    result.add_argument("--db", type=Path, help="Отдельная SQLite база (удобно для тестов)")
    result.add_argument("--section", choices=SECTIONS, help="Раздел для однократного запуска")
    result.add_argument("--recipient", help="Получатель однократного отчёта")
    result.add_argument("--no-charts", action="store_true")
    result.add_argument("--output", type=Path)
    result.add_argument("--format", choices=["csv", "xlsx"], default="csv")
    result.add_argument("--from", dest="date_from")
    result.add_argument("--to", dest="date_to")
    result.add_argument("--phrase", action="append", dest="phrases")
    result.add_argument("--region", action="append", type=int, dest="regions")
    result.add_argument("--source", choices=["ruonia", "wordstat", "cbr", "moex"])
    result.add_argument("--port", type=int, default=8765)
    result.add_argument('--forecast', type=int, help='Срок прогноза для backtest, максимум 24 месяца')
    result.add_argument('--forecast-unit', choices=['weeks', 'months', 'quarters', 'years'], default='months')
    return result


def telegram_client(settings):
    # Pyrogram captures an asyncio loop at import; initialize it in the main
    # thread, including on Python versions without implicit loop creation.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    from pyrogram import Client
    raw_id = os.getenv("api_id", "")
    if not raw_id.isdecimal() or int(raw_id) <= 0 or not os.getenv("api_hash") or not os.getenv("bot_token"):
        raise ConfigurationError("Задайте положительный числовой api_id, api_hash и bot_token в локальном .env")
    return Client("me_client", api_id=int(raw_id), api_hash=os.getenv("api_hash"),
                  bot_token=os.getenv("bot_token"), workdir=str(settings.root))


def run(argv=None, *, settings=None, providers=None):
    args = parser().parse_args(argv)
    settings = settings or Settings.from_environment()
    if args.command == "dashboard" and not settings.dashboard_enabled:
        print("Веб-панель отключена (DASHBOARD_ENABLED=0). Бот работает через команду serve. "
              "Для отдельного запуска панели задайте DASHBOARD_ENABLED=1.", file=sys.stderr)
        return 2
    if args.command in {"once", "serve"} and not args.dry_run and os.getenv("AUTO_UPDATE", "0").strip().lower() in {"1", "true", "yes"}:
        from functions.auto_update import check_and_restart_if_updated
        check_and_restart_if_updated()
    database_path = args.db or settings.data_dir / 'thesaurus.sqlite3'
    try:
        store = Store(database_path)
    except Exception:
        emergency = Path(database_path).with_suffix('.integrity-emergency.json')
        if emergency.exists() and args.command == 'serve' and not args.dry_run and settings.admin_ids:
            try:
                client = telegram_client(settings)
                async def notify_integrity_failure():
                    await client.start()
                    try:
                        for actor in settings.admin_ids:
                            try:
                                await client.send_message(actor, 'Критический инцидент целостности SQLite или версии схемы. Сервис остановлен; данные не изменяются. Проверьте локальный аварийный журнал и резервную копию.')
                            except Exception:  # noqa: BLE001 - One unreachable admin cannot hide the incident from others.
                                logging.getLogger(__name__).error('Critical integrity notification failed')
                    finally:
                        await client.stop()
                client.run(notify_integrity_failure())
            except Exception:  # noqa: BLE001 - Keep the original startup exception, never print credentials.
                logging.getLogger(__name__).error('Critical integrity notification unavailable; local journal saved')
        raise
    from .service import Worker, bootstrap, status
    bootstrap(store, settings)
    if args.command == "status":
        print(dumps(status(store)))
        return 0
    if args.command == "export":
        from .exporting import export_data
        if not args.output:
            raise ValueError("Укажите --output для экспорта")
        if args.section == 'market' or args.source in {'cbr', 'moex'}:
            from .market_jobs import export_market
            selected = args.phrases if args.phrases is not None else [r['id'] for r in store.query('SELECT id FROM market_series WHERE enabled=1' + (' AND source=?' if args.source else ''), (args.source,) if args.source else ())]
            path = export_market(store, args.output, selected, format=args.format, date_from=args.date_from, date_to=args.date_to)
        else:
            path = export_data(store, args.output, format=args.format, date_from=args.date_from, date_to=args.date_to,
                               phrases=args.phrases, regions=args.regions, source=args.source)
        print(f"Экспорт сохранён: {path}")
        return 0
    if args.command == "backtest":
        import pandas as pd

        from .analysis import rolling_backtest
        from .forecast_horizons import validate
        from .rate_forecasts import backtest_daily
        from .reporting import selected_observations
        duration = validate({'value': args.forecast, 'unit': args.forecast_unit}) if args.forecast is not None else None
        section = args.section or "wordstat"
        if section == 'market':
            import json

            from .market import selected
            from .market_analysis import backtest
            from .market_store import history
            results = [{'series_id': row['id'], **backtest(history(store, row['id']), json.loads(row['metadata']), horizon=duration)}
                       for row in selected(store) if not args.phrases or row['id'] in args.phrases]
            text = dumps(results)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text, encoding='utf-8')
            else:
                print(text)
            return 0
        if section not in {"wordstat", "ruonia"}:
            raise ValueError("Backtest supports wordstat and ruonia")
        rows, _, _ = selected_observations(store, section, args.phrases, args.regions)
        results = []
        if rows:
            runner = backtest_daily if section == "ruonia" else rolling_backtest
            for key, series in pd.DataFrame(rows).groupby(["metric", "region_id", "device", "unit"]):
                results.append({"metric": key[0], "region_id": int(key[1]), "unit": key[3], **runner(series, **({'horizon': duration} if section == 'ruonia' else {}))})
        text = dumps(results)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text)
        return 0
    if args.command == "dashboard":
        if not 1 <= args.port <= 65535:
            raise ValueError("Invalid port")
        if args.dry_run:
            print(dumps({"mode": "dry-run", "dashboard_enabled": True, "url": f"http://127.0.0.1:{args.port}"}))
            return 0
        try:
            import uvicorn

            from .dashboard import create_app
        except ModuleNotFoundError as exc:
            if exc.name not in {"fastapi", "uvicorn", "jinja2", "starlette"}:
                raise
            print("Установите веб-панель: python -m pip install -r requirements-dashboard.txt", file=sys.stderr)
            return 2
        uvicorn.run(create_app(store, settings), host="127.0.0.1", port=args.port, log_level="warning")
        return 0
    if args.command == "serve" and args.dry_run:
        # A dry service run is a finite preview, not a silent background daemon.
        print(dumps({"mode": "dry-run", "enabled_schedules": store.query("SELECT recipient,section,times,weekdays,timezone FROM subscriptions WHERE enabled=1"), "status": status(store)}))
        return 0
    if args.command == "serve" and not settings.admin_ids:
        raise ConfigurationError("В .env не задан ADMIN_TELEGRAM_IDS. Укажите числовой ID своего Telegram-аккаунта и повторите запуск.")
    worker = Worker(store, settings, providers, dry_run=args.dry_run, charts=not args.no_charts)
    if args.dry_run:
        report_ids = []
        # Preview reports without consuming or acknowledging the live queue.
        production_db = (settings.data_dir / 'thesaurus.sqlite3').resolve()
        preview_lock = store.path.with_suffix('.worker.lock') if args.db and store.path != production_db else settings.data_dir / 'worker.lock'
        with worker_lock(preview_lock):
            for section in ([args.section] if args.section else ["ruonia", "wordstat", "regions", "ai"]):
                if section == 'market':
                    from .market import report
                    if not settings.admin_ids:
                        raise ConfigurationError('Для market задайте ADMIN_TELEGRAM_IDS')
                    report_ids.append(report(store, settings.admin_ids[0], charts=not args.no_charts, settings=settings))
                    continue
                metadata = worker.collector.collect(section, dry_run=True)
                report_ids.append(worker.builder.build(section, metadata, phrases=args.phrases,
                                                       regions=args.regions, charts=not args.no_charts))
        print(dumps({"mode": "dry-run", "reports": report_ids, "status": status(store)}))
        return 0
    client = telegram_client(settings)
    with worker_lock(settings.data_dir / "worker.lock"):
        worker.recover()
        from .google_trends import activate
        if settings.search_provider == 'google_trends':
            activate(store)
        elif settings.search_provider == 'disabled':
            from .search_provider import deactivate
            deactivate(store)
        else:
            store.set_setting('search_provider', 'legacy')
        try:
            worker.collector.import_legacy()
        except Exception as exc:  # noqa: BLE001 - Optional legacy import must not prevent service startup.
            logging.getLogger(__name__).warning("Legacy workbook import skipped (%s)", type(exc).__name__)
        if args.command == "once":
            recipients = [args.recipient] if args.recipient else store.get_setting("legacy_recipients", [])
            if not recipients:
                raise ValueError("Укажите --recipient или настройте for_whom_list")
            batch = uuid4().hex
            for section in ([args.section] if args.section else (["ruonia", "ai"] if settings.search_provider != 'legacy' else ["ruonia", "wordstat", "regions", "ai"])):
                from .search_provider import require_section
                require_section(store, section)
                for recipient in recipients or [None]:
                    store.enqueue("report", {"section": section, "recipient": recipient, "detail": "full",
                        "phrases": args.phrases or [], "regions": args.regions or []}, f"once:{batch}:{section}:{recipient}")
            async def once():
                from .delivery import deliver_pending
                await client.start()
                try:
                    await asyncio.to_thread(worker.drain)
                    await asyncio.to_thread(deliver_pending, store, client)
                finally:
                    await client.stop()
            client.run(once())
            return 1 if store.one("SELECT 1 FROM jobs WHERE dedupe_key LIKE ? AND state='failed' LIMIT 1", (f"once:{batch}:%",)) else 0
        from .telegram_bot import install_handlers
        install_handlers(client, store)

        async def serve_monitored(runtime):
            from .delivery import deliver_pending
            from .scheduling import enqueue_due
            stop = asyncio.Event()
            with _termination_signals(stop):
                await client.start()
                try:
                    await asyncio.to_thread(runtime.pulse)
                    from .telegram_commands import configure
                    await configure(client)
                    while not stop.is_set():
                        enqueue_due(store)
                        from .chart_schedules import enqueue_due as enqueue_charts
                        enqueue_charts(store)
                        from .market import enqueue_collection
                        enqueue_collection(store)
                        from .integrity import dispatch as dispatch_integrity
                        from .integrity import enqueue_due as enqueue_integrity
                        enqueue_integrity(store)
                        if settings.search_provider == 'google_trends':
                            from .trends_api import enqueue_due as enqueue_trends
                            enqueue_trends(store)
                        await asyncio.to_thread(worker.process_next)
                        if stop.is_set():
                            break
                        from .market_alerts import dispatch as dispatch_market_signals
                        await asyncio.to_thread(dispatch_market_signals, store, settings)
                        await asyncio.to_thread(dispatch_integrity, store)
                        await asyncio.to_thread(deliver_pending, store, client)
                        from .user_management import send_notices
                        await asyncio.to_thread(send_notices, store, client)
                        with suppress(TimeoutError):
                            await asyncio.wait_for(stop.wait(), timeout=1)
                finally:
                    await client.stop()
        async def serve():
            from .runtime_monitor import RuntimeMonitor
            async with RuntimeMonitor(store,client) as runtime:
                await serve_monitored(runtime)
        client.run(serve())
    return 0


def entrypoint(argv=None):
    # Redirected Windows terminals may default to an ASCII/code-page stream.
    # Reports and paths contain Cyrillic; a successful export must not fail
    # merely while printing its confirmation.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        return run(argv)
    except ConfigurationError as exc:
        print(f"Ошибка настройки: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary suppresses credential-bearing provider details.
        # Never print raw provider errors, credential-bearing URLs or values.
        print(f"Ошибка запуска ({type(exc).__name__}). Проверьте параметры и локальную конфигурацию.", file=sys.stderr)
        return 1
