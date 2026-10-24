"""Web UI is an explicit opt-in and never a prerequisite for the bot."""
import io
import os
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from thesaurus.cli import run
from thesaurus.config import Settings


class OptionalWebTests(unittest.TestCase):
    def test_dashboard_defaults_to_disabled_without_password(self):
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv"):
            settings = Settings.from_environment()
        self.assertFalse(settings.dashboard_enabled)
        self.assertEqual(settings.dashboard_password, "")

    def test_dashboard_flag_is_explicit_and_validated(self):
        for value, expected in (("1", True), ("true", True), ("ON", True), ("0", False), ("false", False), ("off", False)):
            with self.subTest(value=value), patch.dict(os.environ, {"DASHBOARD_ENABLED": value}, clear=True), patch("dotenv.load_dotenv"):
                self.assertEqual(Settings.from_environment().dashboard_enabled, expected)
        with patch.dict(os.environ, {"DASHBOARD_ENABLED": "invalid"}, clear=True), patch("dotenv.load_dotenv"), self.assertRaises(ValueError):
            Settings.from_environment()

    def test_disabled_dashboard_stops_before_database_or_optional_imports(self):
        output = io.StringIO()
        with patch("thesaurus.cli.Store") as store, redirect_stderr(output):
            self.assertEqual(run(["dashboard"], settings=Settings()), 2)
        store.assert_not_called()
        self.assertIn("DASHBOARD_ENABLED=0", output.getvalue())


if __name__ == "__main__":
    unittest.main()
