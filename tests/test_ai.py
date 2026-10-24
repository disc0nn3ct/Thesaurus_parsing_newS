"""Offline regression tests; neither .env nor a real AI provider is accessed."""

import importlib.util
import os
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import dotenv

try:
    import httpx2 as httpx
except ImportError:
    import httpx


def load_ai_module():
    path = Path(__file__).resolve().parents[1] / "functions" / "ai.py"
    spec = importlib.util.spec_from_file_location("_ai_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(dotenv, "load_dotenv"), \
            patch.dict(os.environ, {}, clear=True):
        spec.loader.exec_module(module)
    return module


class AiTests(unittest.TestCase):
    def setUp(self):
        self.ai = load_ai_module()

    def test_import_does_not_create_logs_or_change_global_logging(self):
        with patch("os.makedirs") as mkdir, \
                patch("logging.basicConfig") as configure, \
                patch("builtins.open") as open_file:
            load_ai_module()
        mkdir.assert_not_called()
        configure.assert_not_called()
        open_file.assert_not_called()

    def test_missing_key_does_not_prevent_import(self):
        self.assertIsNone(self.ai.client)
        with self.assertRaisesRegex(RuntimeError, "ai_api_key is missing"):
            self.ai.get_client()

    def test_client_is_created_once_with_bounded_sdk_retries(self):
        self.ai.AI_API_KEY = "test-key-" + "not-secret"
        # Windows TLS initialization requires SystemRoot even without a request.
        system_env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR")
                      if key in os.environ}
        with patch.object(self.ai, "OpenAI", wraps=self.ai.OpenAI) as factory, \
                patch.dict(os.environ, system_env, clear=True):
            first = self.ai.get_client()
            self.addCleanup(first.close)
            self.assertIs(first, self.ai.get_client())
        factory.assert_called_once_with(
            api_key="test-key-" + "not-secret", base_url=None,
            timeout=60.0, max_retries=0,
        )

    def test_last_fenced_json_is_extracted_without_removing_earlier_block(self):
        prefix = 'Brief\n```json\n{"example": 1}\n```\nMore prose'
        text = prefix + '\n```json\n{"ideas": [{"ticker": "TEST"}]}\n```'
        self.assertEqual(
            self.ai.extract_trailing_json(text),
            (prefix, {"ideas": [{"ticker": "TEST"}]}),
        )

    def test_prose_braces_do_not_hide_bare_trailing_json(self):
        prefix = 'Text with {a placeholder} and {"example": 1}.'
        text = prefix + '\n{"ideas": [{"ticker": "TEST"}]}'
        self.assertEqual(
            self.ai.extract_trailing_json(text),
            (prefix, {"ideas": [{"ticker": "TEST"}]}),
        )

    def test_invalid_json_and_nonobject_json_are_preserved(self):
        for text in (
            'Brief\n```json\n{"bad":}\n```',
            'Brief\n```json\n[{"ticker": "TEST"}]\n```',
            'Brief\n```json\nnull\n```',
            'Brief {"valid": true}\nAdditional prose',
        ):
            with self.subTest(text=text):
                self.assertEqual(self.ai.extract_trailing_json(text), (text, None))

    def test_markdown_table_is_wrapped_once(self):
        table = '| Ticker | Value |\n| --- | ---: |\n| TEST | 1 |'
        expected = '```text\n' + table + '\n```'
        self.assertEqual(self.ai.remove_markdown_tables(table), expected)
        self.assertEqual(self.ai.remove_markdown_tables(expected), expected)

    def test_existing_code_fences_are_preserved(self):
        table = '| A | B |\n| --- | --- |\n| 1 | 2 |'
        for fence in ('```text', '~~~text', '````text'):
            marker = fence[:-4]
            text = fence + '\n' + table + '\n' + marker
            with self.subTest(fence=fence):
                self.assertEqual(self.ai.remove_markdown_tables(text), text)

    def test_normal_prose_with_pipe_and_dashes_is_preserved(self):
        text = 'value | note\ntext | -- a note'
        self.assertEqual(self.ai.remove_markdown_tables(text), text)

    def test_tables_without_outer_pipes_are_converted(self):
        table = 'A | B\n--- | ---\n1 | 2'
        self.assertEqual(
            self.ai.remove_markdown_tables(table), '```text\n' + table + '\n```'
        )

    def test_transient_failure_retries_then_returns_success(self):
        failure = self.ai.APIStatusError(
            "private provider response",
            response=httpx.Response(429, request=httpx.Request("POST", "https://example.invalid")),
            body=None,
        )
        fn = Mock(side_effect=[failure, "ok"])
        with patch.object(self.ai.time, "sleep") as sleep, \
                patch.object(self.ai, "logger") as logger:
            self.assertEqual(self.ai.retry_call(fn), "ok")
        sleep.assert_called_once_with(1.0)
        self.assertEqual(fn.call_count, 2)
        self.assertNotIn("private provider response", str(logger.mock_calls))

    def test_final_failure_does_not_sleep(self):
        failure = self.ai.APIConnectionError(
            request=httpx.Request("POST", "https://example.invalid")
        )
        fn = Mock(side_effect=failure)
        with patch.object(self.ai.time, "sleep") as sleep, \
                patch.object(self.ai, "logger"), self.assertRaises(self.ai.APIConnectionError):
            self.ai.retry_call(fn, tries=3)
        self.assertEqual(fn.call_count, 3)
        self.assertEqual([call.args for call in sleep.call_args_list], [(1.0,), (2.0,)])

    def test_permanent_api_error_is_not_retried(self):
        failure = self.ai.APIStatusError(
            "unauthorized",
            response=httpx.Response(401, request=httpx.Request("POST", "https://example.invalid")),
            body=None,
        )
        fn = Mock(side_effect=failure)
        with patch.object(self.ai.time, "sleep") as sleep, \
                patch.object(self.ai, "logger"), self.assertRaises(self.ai.APIStatusError):
            self.ai.retry_call(fn)
        fn.assert_called_once()
        sleep.assert_not_called()

    def test_programming_error_is_not_retried(self):
        fn = Mock(side_effect=ValueError("invalid input"))
        with patch.object(self.ai.time, "sleep") as sleep, self.assertRaises(ValueError):
            self.ai.retry_call(fn)
        fn.assert_called_once()
        sleep.assert_not_called()

    def test_retry_arguments_are_validated(self):
        for kwargs in ({"tries": 0}, {"base_sleep": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.ai.retry_call(Mock(), **kwargs)

    def test_empty_ai_response_does_not_produce_a_brief(self):
        for choices in ([], [types.SimpleNamespace(
                message=types.SimpleNamespace(content="   "))]):
            provider = Mock()
            provider.chat.completions.create.return_value = types.SimpleNamespace(
                choices=choices
            )
            with self.subTest(choices=choices), \
                    patch.object(self.ai, "get_client", return_value=provider), \
                    patch.object(self.ai.os, "makedirs") as mkdir:
                with self.assertRaisesRegex(ValueError, "contained no text"):
                    self.ai.run_brief()
                mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
