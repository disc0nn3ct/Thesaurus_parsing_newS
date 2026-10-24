"""Fresh-process checks for a headless Telegram service without web extras."""

import os
import shutil
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[1]
PREAMBLE = """
import importlib.abc
import sys
from pathlib import Path

project = Path(sys.argv[1])
sys.path.insert(0, str(project))
web_modules = {"fastapi", "starlette", "uvicorn", "jinja2"}

class RejectWebImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in web_modules:
            raise AssertionError("Telegram service imported a dashboard dependency: " + fullname)
        return None

sys.meta_path.insert(0, RejectWebImports())
"""


class ServerModeTests(unittest.TestCase):
    def setUp(self):
        self.parent = PROJECT / ".test-tmp"
        self.parent.mkdir(exist_ok=True)
        self.directory = self.parent / f"server_test_{uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if self.directory.resolve().parent != self.parent.resolve():
            raise AssertionError("Unexpected server test cleanup path")
        shutil.rmtree(self.directory)

    def run_script(self, script):
        environment = dict(os.environ, MPLCONFIGDIR=str(self.directory), MPLBACKEND="TkAgg", AUTO_UPDATE="0")
        result = subprocess.run(
            [sys.executable, "-c", PREAMBLE + textwrap.dedent(script), str(PROJECT)],
            cwd=self.directory, env=environment, capture_output=True, text=True,
            encoding="utf-8", timeout=60, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("server-check-ok", result.stdout)

    def test_actual_main_dry_service_is_headless_without_web_dependencies(self):
        self.run_script("""
            import io
            import json
            import runpy
            from contextlib import ExitStack, redirect_stdout
            from unittest.mock import Mock, patch

            from thesaurus import cli
            from thesaurus.config import Settings

            store = Mock()
            store.query.return_value = []
            settings = Settings(root=Path.cwd(), admin_ids=(1,))
            output = io.StringIO()
            with ExitStack() as stack:
                stack.enter_context(patch("dotenv.load_dotenv"))
                stack.enter_context(patch.object(cli.Settings, "from_environment", return_value=settings))
                stack.enter_context(patch.object(cli, "Store", return_value=store))
                stack.enter_context(patch("thesaurus.service.bootstrap"))
                stack.enter_context(patch("thesaurus.service.status", return_value={}))
                telegram = stack.enter_context(patch.object(cli, "telegram_client"))
                stack.enter_context(patch("socket.socket.bind", side_effect=AssertionError("Listener forbidden")))
                stack.enter_context(patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")))
                stack.enter_context(patch.object(sys, "argv", [str(project / "main.py"), "serve", "--dry-run"]))
                with redirect_stdout(output):
                    try:
                        runpy.run_path(str(project / "main.py"), run_name="__main__")
                    except SystemExit as exc:
                        assert exc.code == 0, exc.code
                    else:
                        raise AssertionError("main.py did not exit")
                telegram.assert_not_called()
            import matplotlib
            assert matplotlib.get_backend().lower() == "agg", matplotlib.get_backend()
            assert json.loads(output.getvalue())["mode"] == "dry-run"
            assert not any(name.split(".", 1)[0] in web_modules for name in sys.modules)
            print("server-check-ok")
        """)

    def test_live_service_loop_needs_no_web_dependencies_and_stops_telegram(self):
        self.run_script("""
            import asyncio
            from contextlib import ExitStack, nullcontext
            from unittest.mock import AsyncMock, Mock, patch

            from thesaurus import cli
            from thesaurus.config import Settings

            class EndIteration(Exception):
                pass

            client = Mock()
            client.start = AsyncMock()
            client.stop = AsyncMock()
            client.run.side_effect = asyncio.run
            settings = Settings(root=Path.cwd(), admin_ids=(1,))
            with ExitStack() as stack:
                stack.enter_context(patch.object(cli, "Store"))
                stack.enter_context(patch("thesaurus.service.bootstrap"))
                worker = stack.enter_context(patch("thesaurus.service.Worker")).return_value
                stack.enter_context(patch.object(cli, "telegram_client", return_value=client))
                stack.enter_context(patch.object(cli, "worker_lock", return_value=nullcontext()))
                handlers = stack.enter_context(patch("thesaurus.telegram_bot.install_handlers"))
                schedules = stack.enter_context(patch("thesaurus.scheduling.enqueue_due"))
                deliveries = stack.enter_context(patch("thesaurus.delivery.deliver_pending", side_effect=EndIteration))
                stack.enter_context(patch("asyncio.start_server", side_effect=AssertionError("Listener forbidden")))
                stack.enter_context(patch("socket.create_server", side_effect=AssertionError("Listener forbidden")))
                stack.enter_context(patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")))
                try:
                    cli.run(["serve"], settings=settings)
                except EndIteration:
                    pass
                else:
                    raise AssertionError("Service did not execute its delivery iteration")
                worker.recover.assert_called_once()
                worker.process_next.assert_called_once()
                schedules.assert_called_once()
                deliveries.assert_called_once()
                handlers.assert_called_once()
            client.start.assert_awaited_once()
            client.stop.assert_awaited_once()
            assert not any(name.split(".", 1)[0] in web_modules for name in sys.modules)
            print("server-check-ok")
        """)

    def test_shutdown_signal_finishes_active_job_and_restores_handlers(self):
        self.run_script("""
            import asyncio
            import signal
            from contextlib import ExitStack, nullcontext
            from unittest.mock import AsyncMock, Mock, patch

            from thesaurus import cli
            from thesaurus.config import Settings

            runtime = {}
            previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

            async def start():
                runtime["loop"] = asyncio.get_running_loop()

            def finish_job():
                handler = signal.getsignal(signal.SIGTERM)
                assert callable(handler), "Service must handle systemd's SIGTERM"
                runtime["loop"].call_soon_threadsafe(handler, signal.SIGTERM, None)
                runtime["finished_job"] = True

            client = Mock()
            client.start = AsyncMock(side_effect=start)
            client.stop = AsyncMock()
            client.run.side_effect = asyncio.run
            settings = Settings(root=Path.cwd(), admin_ids=(1,))
            with ExitStack() as stack:
                stack.enter_context(patch.object(cli, "Store"))
                stack.enter_context(patch("thesaurus.service.bootstrap"))
                worker = stack.enter_context(patch("thesaurus.service.Worker")).return_value
                worker.process_next.side_effect = finish_job
                stack.enter_context(patch.object(cli, "telegram_client", return_value=client))
                stack.enter_context(patch.object(cli, "worker_lock", return_value=nullcontext()))
                stack.enter_context(patch("thesaurus.telegram_bot.install_handlers"))
                stack.enter_context(patch("thesaurus.scheduling.enqueue_due"))
                deliveries = stack.enter_context(patch("thesaurus.delivery.deliver_pending"))
                assert cli.run(["serve"], settings=settings) == 0
                worker.process_next.assert_called_once()
                deliveries.assert_not_called()
            assert runtime["finished_job"]
            client.start.assert_awaited_once()
            client.stop.assert_awaited_once()
            assert all(signal.getsignal(number) == handler for number, handler in previous.items())
            print("server-check-ok")
        """)


if __name__ == "__main__":
    unittest.main()
