"""Legacy main regressions without dotenv, network, Git, or Telegram sessions."""

import importlib.util
import os
import shutil
import unittest
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pandas as pd


@contextmanager
def fixture_directory():
    parent = Path(__file__).resolve().parent
    directory = parent / f"main_test_{uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.resolve().parent != parent:
            raise AssertionError("Unexpected test cleanup path")
        shutil.rmtree(directory)


def load_main():
    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("legacy_main_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with fixture_directory() as cache:
        environment = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "USERPROFILE") if key in os.environ}
        environment.update(MPLCONFIGDIR=str(cache), MPLBACKEND="Agg")
        with patch("dotenv.load_dotenv"), patch.dict(os.environ, environment, clear=True), \
                patch("requests.sessions.Session.request") as network, \
                patch("subprocess.run") as git, patch("pyrogram.Client") as client:
            spec.loader.exec_module(module)
    network.assert_not_called()
    git.assert_not_called()
    client.assert_not_called()
    return module


class Morning(datetime):
    @classmethod
    def today(cls):
        return cls(2026, 10, 2, 10, 0)

    @classmethod
    def now(cls, tz=None):
        return cls.today()


def workbook(day, value=15):
    return pd.DataFrame({"Дата": [day], "Индекс": [value], "1 месяц": [value],
                         "3 месяца": [value], "6 месяцев": [value]})


def excel_bytes(frame):
    output = BytesIO()
    frame.to_excel(output, index=False)
    return output.getvalue()


class MainRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main()

    def setUp(self):
        self.enterContext(patch.object(self.main, "logger"))
        self.enterContext(patch.object(self.main, "datetime", Morning))
        self.enterContext(patch.object(self.main, "is_russian_workday", return_value=True))
        self.network = self.enterContext(patch.object(self.main.requests, "get"))

    def test_import_does_not_start_telegram_or_update_git(self):
        self.assertTrue(callable(load_main().main))

    def test_previous_day_workbook_is_fresh_before_1400(self):
        with fixture_directory() as directory:
            path = directory / "ruonia.xlsx"
            path.write_bytes(excel_bytes(workbook("01.10.2026")))
            self.assertEqual(self.main.check_if_need_new_rec(path), 0)
        self.network.assert_not_called()

    def test_invalid_http200_responses_do_not_overwrite_valid_cache(self):
        responses = [b"<html>maintenance</html>", excel_bytes(pd.DataFrame({"wrong": [1]})),
                     excel_bytes(workbook("not-a-date"))]
        with fixture_directory() as directory:
            path = directory / "ruonia.xlsx"
            original = excel_bytes(workbook("01.09.2026"))
            path.write_bytes(original)
            for content in responses:
                with self.subTest(content_length=len(content)):
                    self.network.return_value = Mock(content=content)
                    self.assertEqual(self.main.check_if_need_new_rec(path), -2)
                    self.assertEqual(path.read_bytes(), original)

    def test_valid_download_atomically_replaces_cache(self):
        with fixture_directory() as directory:
            path = directory / "ruonia.xlsx"
            path.write_bytes(excel_bytes(workbook("01.09.2026")))
            downloaded = excel_bytes(workbook("02.10.2026", 16))
            self.network.return_value = Mock(content=downloaded)
            self.assertEqual(self.main.check_if_need_new_rec(path), 1)
            self.assertEqual(path.read_bytes(), downloaded)
            self.assertEqual(list(directory.iterdir()), [path])

    def test_failed_replace_keeps_cache_and_cleans_temporary_file(self):
        with fixture_directory() as directory:
            path = directory / "ruonia.xlsx"
            original = excel_bytes(workbook("01.09.2026"))
            path.write_bytes(original)
            self.network.return_value = Mock(content=excel_bytes(workbook("02.10.2026")))
            with patch.object(self.main.os, "replace", side_effect=PermissionError("locked")):
                self.assertEqual(self.main.check_if_need_new_rec(path), -2)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(directory.iterdir()), [path])

    def test_failed_graph_generation_returns_without_sending(self):
        client = Mock()
        with patch.object(self.main, "analitics", return_value=None):
            self.main.send_info_ruonia(client, [123])
        client.send_photo.assert_not_called()
        client.send_message.assert_not_called()

    def test_fresh_short_chart_from_current_generation_is_used(self):
        with fixture_directory() as directory:
            full, fresh, stale = [directory / name for name in ("full.png", "fresh_short.png", "stale_short.png")]
            for path in (full, fresh, stale):
                path.touch()
            client = Mock()
            with patch.object(self.main, "analitics", return_value=(str(full), str(fresh))), \
                    patch.object(self.main, "make_analyze_ruonia", return_value="Analysis"):
                self.main.send_info_ruonia(client, [123])
            self.assertEqual([call.kwargs["photo"] for call in client.send_photo.call_args_list], [str(full), str(fresh)])

    def test_legacy_main_stops_client_when_a_stage_raises(self):
        client = Mock()
        environment = {"for_whom_list": "123", "api_id": "1", "api_hash": "fixture", "bot_token": "fixture"}
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(self.main, "load_dotenv"), patch.object(self.main, "configure_runtime"), \
                patch("pyrogram.Client", return_value=client), \
                patch("functions.auto_update.check_and_restart_if_updated") as update, \
                patch.object(self.main, "check_if_need_new_rec", side_effect=RuntimeError("stage failed")), \
                patch.object(self.main, "send_info_ruonia") as send, \
                self.assertRaisesRegex(RuntimeError, "stage failed"):
            self.main.main()
        client.start.assert_called_once()
        client.stop.assert_called_once()
        send.assert_not_called()
        update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
