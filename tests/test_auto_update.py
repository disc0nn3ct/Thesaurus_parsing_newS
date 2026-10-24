"""Offline safety checks: no test runs Git, fetches, or restarts the process."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from functions import auto_update

CURRENT = "a" * 40
REMOTE = "b" * 40


class AutoUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.branch = "main"
        self.status = ""
        self.head = CURRENT
        self.remote = REMOTE
        self.ancestor = True
        self.fail_command = None
        self.update_head = True
        self.addCleanup(patch.stopall)
        patch.object(auto_update, "PROJECT_ROOT", self.root).start()
        patch.object(auto_update, "logger").start()
        self.run = patch.object(auto_update.subprocess, "run", side_effect=self.fake_git).start()
        self.restart = patch.object(auto_update.os, "execv").start()

    def fake_git(self, command, **kwargs):
        self.assertEqual(kwargs["cwd"], self.root)
        self.assertEqual(kwargs["timeout"], auto_update.GIT_TIMEOUT_SECONDS)
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        args = command[1:]
        if args[0] == self.fail_command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        code = 0
        if args[0] == "symbolic-ref":
            output = self.branch
            code = 0 if self.branch else 1
        elif args[0] == "status":
            self.assertIn("--untracked-files=no", args)
            output = self.status
        elif args[0] == "fetch":
            self.assertEqual(args, ["fetch", "--no-tags", "origin", "refs/heads/main:refs/remotes/origin/main"])
            output = ""
        elif args[0] == "rev-parse":
            output = self.head if args[-1] == "HEAD" else self.remote
        elif args[0] == "merge-base":
            code = 0 if self.ancestor else 1
            output = ""
        elif args[0] == "merge":
            self.assertEqual(args, ["merge", "--ff-only", REMOTE])
            if self.update_head:
                self.head = args[-1]
            output = ""
        else:
            self.fail("Unexpected Git command: " + repr(command))
        return subprocess.CompletedProcess(command, code, output, "")

    def git_commands(self):
        return [call.args[0][1] for call in self.run.call_args_list]

    def test_first_run_detects_update_without_marker(self):
        self.assertEqual(auto_update.check_git_update(), REMOTE)
        self.assertFalse((self.root / "current_commit.txt").exists())
        self.restart.assert_not_called()

    def test_default_marker_and_cwd_are_safe(self):
        self.remote = CURRENT
        cwd = os.getcwd()
        with patch.object(auto_update.sys, "argv", ["outside/project.py"]):
            self.assertIsNone(auto_update.check_git_update())
        self.assertEqual(os.getcwd(), cwd)
        self.assertEqual((self.root / "current_commit.txt").read_text(), CURRENT)

    def test_stale_marker_does_not_trigger_restart(self):
        self.remote = CURRENT
        marker = self.root / "current_commit.txt"
        marker.write_text(REMOTE)
        auto_update.check_and_restart_if_updated("current_commit.txt")
        self.restart.assert_not_called()
        self.assertEqual(marker.read_text(), CURRENT)

    def test_marker_cannot_hide_pending_update(self):
        (self.root / "current_commit.txt").write_text(REMOTE)
        self.assertEqual(auto_update.check_git_update(), REMOTE)

    def test_dirty_checkout_is_not_fetched_or_updated(self):
        self.status = " M config.py"
        auto_update.check_and_restart_if_updated()
        self.assertNotIn("fetch", self.git_commands())
        self.assertNotIn("merge", self.git_commands())
        self.restart.assert_not_called()

    def test_non_main_and_detached_checkouts_are_skipped(self):
        for branch in ("feature", ""):
            with self.subTest(branch=branch):
                self.branch = branch
                self.run.reset_mock()
                self.assertIsNone(auto_update.check_git_update())
                self.assertNotIn("fetch", self.git_commands())

    def test_diverged_or_ahead_checkout_is_not_updated(self):
        self.ancestor = False
        auto_update.check_and_restart_if_updated()
        self.assertNotIn("merge", self.git_commands())
        self.restart.assert_not_called()

    def test_successful_update_writes_nested_marker_before_restart(self):
        def check_restart(*args):
            self.assertEqual((self.root / "log/current_commit.txt").read_text(), REMOTE)

        self.restart.side_effect = check_restart
        auto_update.check_and_restart_if_updated()
        self.assertEqual(self.head, REMOTE)
        self.restart.assert_called_once()
        self.assertNotIn("pull", self.git_commands())

    def test_checkout_is_checked_again_before_update(self):
        self.assertEqual(auto_update.check_git_update(), REMOTE)
        self.status = "M  config.py"
        auto_update.update_and_restart(REMOTE, "current_commit.txt")
        self.assertNotIn("merge", self.git_commands())
        self.restart.assert_not_called()

    def test_changed_remote_is_not_applied(self):
        self.assertEqual(auto_update.check_git_update(), REMOTE)
        self.remote = "c" * 40
        auto_update.update_and_restart(REMOTE, "current_commit.txt")
        self.assertNotIn("merge", self.git_commands())
        self.restart.assert_not_called()

    def test_already_applied_commit_does_not_restart(self):
        self.head = REMOTE
        auto_update.update_and_restart(REMOTE, "current_commit.txt")
        self.assertNotIn("merge", self.git_commands())
        self.restart.assert_not_called()

    def test_failed_fetch_does_not_update(self):
        self.fail_command = "fetch"
        auto_update.check_and_restart_if_updated()
        self.assertNotIn("merge", self.git_commands())
        self.restart.assert_not_called()

    def test_failed_merge_does_not_write_marker_or_restart(self):
        self.fail_command = "merge"
        auto_update.check_and_restart_if_updated()
        self.assertFalse((self.root / "log/current_commit.txt").exists())
        self.restart.assert_not_called()

    def test_unconfirmed_update_does_not_write_marker_or_restart(self):
        self.update_head = False
        auto_update.check_and_restart_if_updated()
        self.assertFalse((self.root / "log/current_commit.txt").exists())
        self.restart.assert_not_called()


if __name__ == "__main__":
    unittest.main()
