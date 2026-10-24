"""Actionable configuration errors without reflecting secret-bearing errors."""
import io
import os
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from thesaurus.cli import entrypoint
from thesaurus.config import Settings


class CliErrorTests(unittest.TestCase):
    def test_missing_admin_is_explained_before_telegram_connects(self):
        output = io.StringIO()
        with patch("thesaurus.cli.Settings.from_environment", return_value=Settings()), \
                patch("thesaurus.cli.Store"), patch("thesaurus.service.bootstrap"), \
                patch("thesaurus.cli.telegram_client") as telegram, \
                patch.dict(os.environ, {"AUTO_UPDATE": "0"}), redirect_stderr(output):
            self.assertEqual(entrypoint(["serve"]), 1)
        telegram.assert_not_called()
        self.assertIn("ADMIN_TELEGRAM_IDS", output.getvalue())
        self.assertIn("числовой ID", output.getvalue())

    def test_invalid_admin_does_not_echo_supplied_value(self):
        supplied = "invalid-private-value"
        output = io.StringIO()
        with patch.dict(os.environ, {"ADMIN_TELEGRAM_IDS": supplied}, clear=True), \
                patch("dotenv.load_dotenv"), redirect_stderr(output):
            self.assertEqual(entrypoint(["serve"]), 1)
        self.assertIn("ADMIN_TELEGRAM_IDS", output.getvalue())
        self.assertNotIn(supplied, output.getvalue())

    def test_other_value_errors_still_hide_provider_details(self):
        supplied = "private-provider-response"
        output = io.StringIO()
        with patch("thesaurus.cli.run", side_effect=ValueError(supplied)), redirect_stderr(output):
            self.assertEqual(entrypoint(["serve"]), 1)
        self.assertIn("ValueError", output.getvalue())
        self.assertNotIn(supplied, output.getvalue())


if __name__ == "__main__":
    unittest.main()
