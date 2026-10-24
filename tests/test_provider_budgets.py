"""Exercise attempt quotas through real adapters without provider requests."""
import os
import shutil
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, TextIOWrapper
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import requests

from thesaurus import budget, cli
from thesaurus.collectors import Collector, Providers
from thesaurus.config import Settings
from thesaurus.locking import worker_lock
from thesaurus.service import Worker
from thesaurus.store import Store


class ProviderBudgetTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"AI_BACKEND": "perplexity", 'WORDSTAT_BACKEND': 'legacy'}))
        self.root = Path(__file__).resolve().parent / f"provider_test_{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.cleanup)
        self.store = Store(self.root / "state.sqlite3")
        self.provider = Providers(self.store)

    def cleanup(self):
        assert self.root.resolve().parent == Path(__file__).resolve().parent
        shutil.rmtree(self.root)

    def wordstat_context(self):
        self.enterContext(patch("functions.wordstat_api.WORDSTAT_OAUTH", "synthetic"))
        self.enterContext(patch("functions.wordstat_api.WORDSTAT_CLIENT_ID", "synthetic"))
        self.enterContext(patch("functions.wordstat_api.time.sleep"))

    def test_wordstat_success_after_retry_counts_both_attempts(self):
        self.wordstat_context()
        response = Mock(status_code=200)
        response.json.return_value = {"dynamics": []}
        with patch("functions.wordstat_api.requests.post", side_effect=[requests.Timeout(), response]) as post:
            result = self.provider.wordstat("/v1/dynamics", {})
        self.assertEqual(result, {"dynamics": []})
        self.assertEqual(post.call_count, 2)
        self.assertEqual([r["status"] for r in self.store.query("SELECT status FROM usage ORDER BY id")], ["failed", "complete"])

    def test_wordstat_budget_blocks_third_network_attempt(self):
        self.wordstat_context()
        budget.set_budgets(self.store, {"wordstat_calls": 2})
        with patch("functions.wordstat_api.requests.post", side_effect=requests.Timeout()) as post, \
                self.assertRaises(budget.BudgetExceeded):
            self.provider.wordstat("/v1/dynamics", {})
        self.assertEqual(post.call_count, 2)
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM usage WHERE status='failed'")["n"], 2)

    def test_tls_failure_reserves_one_attempt_and_returns_safe_cause(self):
        from functions.wordstat_api import WordstatError
        self.wordstat_context()
        with patch('functions.wordstat_api.requests.post', side_effect=requests.exceptions.SSLError('synthetic')) as post, \
                self.assertRaises(WordstatError) as caught:
            self.provider.wordstat('/v1/dynamics', {})
        self.assertEqual(caught.exception.kind, 'tls')
        post.assert_called_once()
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM usage WHERE status='failed'")['n'], 1)

    def test_ai_budget_blocks_retry_and_records_failed_attempts(self):
        budget.set_budgets(self.store, {"ai_calls": 2})
        attempts = []

        def brief(**kwargs):
            for _ in range(3):
                kwargs["before_attempt"]()
                attempts.append(1)
            self.fail("The third attempt must be refused")

        with patch("functions.ai.get_client"), patch("functions.ai.run_brief", side_effect=brief), \
                self.assertRaises(budget.BudgetExceeded):
            self.provider.ai([])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM usage WHERE status='failed'")["n"], 2)

    def test_ai_token_cap_and_actual_usage_reach_storage(self):
        captured = {}

        def brief(**kwargs):
            captured.update(kwargs)
            kwargs["before_attempt"]()
            return "text", "Tokens used: prompt=100, completion=40, total=140", {"claims": []}

        with patch("functions.ai.get_client"), patch("functions.ai.run_brief", side_effect=brief):
            self.provider.ai([{"metric": "RUONIA"}])
        self.assertEqual(captured["max_tokens"], 2200)
        row = self.store.one("SELECT * FROM usage")
        self.assertEqual((row["tokens_in"], row["tokens_out"], row["cost"]), (100, 40, None))

    def test_invalid_filters_fail_before_any_provider_attempt(self):
        provider = Mock()
        worker = Worker(self.store, Settings(root=self.root), provider, charts=False)
        for section, filters in [("wordstat", {"phrases": ["unknown"]}), ("ai", {"regions": [225]}),
                                 ("wordstat", {"detail": "invalid"})]:
            self.store.enqueue("report", {"section": section, **filters})
        worker.drain()
        self.assertEqual(provider.mock_calls, [])
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM jobs WHERE state='failed'")["n"], 3)

    def test_fresh_database_initialization_is_concurrent_safe(self):
        for attempt in range(5):
            with self.subTest(attempt=attempt):
                path = self.root / f"fresh-{attempt}.sqlite3"
                with ThreadPoolExecutor(max_workers=4) as pool:
                    stores = list(pool.map(lambda _, database=path: Store(database), range(4)))
                self.assertTrue(all(len(store.keywords()) == 8 for store in stores))
                self.assertEqual(stores[0].one("SELECT COUNT(*) n FROM alert_rules")["n"], 1)

    def test_second_process_cannot_acquire_worker_lock_and_release_is_automatic(self):
        path = self.root / "worker.lock"
        command = [sys.executable, "-c", ("from thesaurus.locking import worker_lock; import sys; "
                   "lock=worker_lock(sys.argv[1]); lock.__enter__(); lock.__exit__(None,None,None)"), str(path)]
        with worker_lock(path):
            second = subprocess.run(command, capture_output=True, timeout=10, check=False,
                                    cwd=Path(__file__).resolve().parents[1])
        self.assertNotEqual(second.returncode, 0)
        after_release = subprocess.run(command, capture_output=True, timeout=10, check=False,
                                       cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(after_release.returncode, 0)

    def test_exception_releases_worker_lock(self):
        path = self.root / "worker.lock"
        with self.assertRaises(ValueError), worker_lock(path):
            raise ValueError("Test interruption")
        with worker_lock(path):
            self.assertTrue(path.is_file())

    def test_relative_paths_anchor_to_project_and_password_stays_out_of_repr(self):
        settings = Settings(root=self.root, data_dir=Path("local-data"), reports_dir=Path("local-reports"),
                            dashboard_password="synthetic" + "-credential")
        self.assertEqual(settings.data_dir, self.root / "local-data")
        self.assertEqual(settings.reports_dir, self.root / "local-reports")
        self.assertNotIn(settings.dashboard_password, repr(settings))

    def test_cli_writes_cyrillic_to_redirected_ascii_terminal(self):
        buffer = BytesIO()
        stream = TextIOWrapper(buffer, encoding="ascii")
        try:
            with patch.object(sys, "stdout", stream), patch.object(sys, "stderr", stream), \
                    patch.object(cli, "run", side_effect=lambda _args: print("Экспорт сохранён")):
                cli.entrypoint([])
            stream.flush()
            self.assertEqual(buffer.getvalue().decode("utf-8").strip(), "Экспорт сохранён")
        finally:
            stream.close()

    def test_ai_failure_preserves_previous_overview_and_data_timestamp(self):
        saved = {"status": "complete", "retrieved_at": "2026-09-01T08:00:00+00:00",
                 "ai": {"text": "previous overview", "data": {"claims": []}}}
        self.store.set_setting("collection_latest:ai", saved)
        provider = Mock()
        provider.ai.side_effect = budget.BudgetExceeded("Daily budget exhausted")
        collector = Collector(self.store, Settings(root=self.root), provider)
        partial = collector.collect("ai")
        self.assertEqual(partial["ai"]["text"], "previous overview")
        self.assertEqual(partial["data_retrieved_at"], saved["retrieved_at"])
        self.assertTrue(partial["cached"])
        self.assertEqual(partial["status"], "partial")
        preview = collector.collect("ai", dry_run=True)
        self.assertEqual(preview["ai"], saved["ai"])
        self.assertEqual(provider.ai.call_count, 1)


if __name__ == "__main__":
    unittest.main()
