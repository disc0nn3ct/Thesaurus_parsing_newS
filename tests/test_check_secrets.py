"""Exercise the guard against disposable local repositories, without any pushes."""

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

# Deliberately assembled: no usable credentials or matched tokens in test source.
SYNTHETIC_TOKEN = "gh" + "p_" + "A" * 36
SOURCE_ROOT = Path(__file__).resolve().parents[1]


class SecretGuardTests(unittest.TestCase):
    def setUp(self):
        self.folder = SOURCE_ROOT / ".test-tmp" / f"guard_test_{uuid4().hex}"
        self.folder.mkdir(parents=True)
        self.addCleanup(self.clean_repository)
        self.git("init", "--quiet", "--initial-branch=main")
        self.git("config", "user.name", "Offline test")
        self.git("config", "user.email", "offline@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "tag.gpgsign", "false")
        self.git("config", "core.hooksPath", ".git/hooks")
        spec = importlib.util.spec_from_file_location("guard_under_test", SOURCE_ROOT / "scripts" / "check_secrets.py")
        self.guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.guard)
        # The guard will only read synthetic files inside this disposable repo.
        self.guard.ROOT = self.folder

    def clean_repository(self):
        assert self.folder.resolve().parent == SOURCE_ROOT / ".test-tmp"

        def remove_readonly(function, path, _error):
            assert self.folder.resolve() in Path(path).resolve().parents or Path(path).resolve() == self.folder.resolve()
            os.chmod(path, stat.S_IWRITE)
            # Git/Windows file watchers can briefly keep a directory handle
            # after the child process exits. Retry only sharing/access errors.
            for attempt in range(50):
                try:
                    function(path)
                    return
                except PermissionError as error:
                    if getattr(error, "winerror", None) not in {5, 32} or attempt == 49:
                        raise
                    time.sleep(.1)

        shutil.rmtree(self.folder, onerror=remove_readonly)

    def git(self, *arguments):
        return subprocess.run(["git", *arguments], cwd=self.folder, check=True,
                              capture_output=True, timeout=20).stdout.decode().strip()

    def write(self, name, contents):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def commit(self, message="test snapshot"):
        self.git("add", "--all")
        self.git("commit", "--quiet", "-m", message)
        return self.git("rev-parse", "HEAD")

    def run_guard(self, mode, stdin=""):
        output, errors = StringIO(), StringIO()
        with patch.object(sys, "argv", ["check_secrets.py", mode]), \
             patch.object(sys, "stdin", StringIO(stdin)), redirect_stdout(output), redirect_stderr(errors):
            status = self.guard.main()
        combined = output.getvalue() + errors.getvalue()
        self.assertNotIn(SYNTHETIC_TOKEN, combined)
        return status, combined

    def test_staged_blob_is_scanned_even_if_worktree_is_clean(self):
        self.write("config.txt", SYNTHETIC_TOKEN)
        self.git("add", "config.txt")
        self.write("config.txt", "credential removed only in working tree")
        self.assertEqual(self.run_guard("--staged")[0], 1)
        self.assertEqual(self.run_guard("--worktree")[0], 0)

    def test_unstaged_secret_is_not_confused_with_index(self):
        self.write("config.txt", "safe value")
        self.git("add", "config.txt")
        self.write("config.txt", SYNTHETIC_TOKEN)
        self.assertEqual(self.run_guard("--staged")[0], 0)
        self.assertEqual(self.run_guard("--worktree")[0], 1)

    def test_deleted_secret_is_still_detected_in_history_and_pre_push(self):
        path = self.write("old.txt", SYNTHETIC_TOKEN)
        self.commit("old snapshot")
        path.unlink()
        self.write("safe.txt", "only safe content at HEAD")
        head = self.commit("remove old file")
        self.assertEqual(self.run_guard("--staged")[0], 0)
        self.assertEqual(self.run_guard("--history")[0], 1)
        stdin = f"refs/heads/main {head} refs/heads/main {'0' * 40}\n"
        self.assertEqual(self.run_guard("--pre-push", stdin)[0], 1)

    def test_deleted_sensitive_filename_remains_blocked(self):
        path = self.write(".env", "# synthetic empty environment fixture")
        self.commit()
        path.unlink()
        self.write("safe.txt", "safe")
        self.commit()
        status, output = self.run_guard("--history")
        self.assertEqual(status, 1)
        self.assertIn("sensitive filename in history", output)

    def test_pre_push_deletion_transfers_no_objects(self):
        self.write("old.txt", SYNTHETIC_TOKEN)
        head = self.commit()
        stdin = f"(delete) {'0' * 40} refs/heads/old {head}\n"
        with patch.object(self.guard, "scan_history") as scan:
            self.assertEqual(self.run_guard("--pre-push", stdin)[0], 0)
            scan.assert_not_called()

    def test_empty_pre_push_input_is_noop(self):
        with patch.object(self.guard, "scan_history") as scan:
            self.assertEqual(self.run_guard("--pre-push")[0], 0)
            scan.assert_not_called()

    def test_malformed_or_unknown_pre_push_ref_fails_closed(self):
        self.write("safe.txt", "safe")
        self.commit()
        for stdin in ("incomplete input\n", "ref not-a-hash remote " + "0" * 40,
                      "ref " + "1" * 40 + " remote " + "0" * 40):
            with self.subTest(stdin=stdin):
                self.assertEqual(self.run_guard("--pre-push", stdin)[0], 2)

    def test_safe_new_branch_passes(self):
        self.write("safe.txt", "safe")
        head = self.commit()
        stdin = f"refs/heads/main {head} refs/heads/main {'0' * 40}\n"
        self.assertEqual(self.run_guard("--pre-push", stdin)[0], 0)

    def test_commit_and_annotated_tag_messages_are_scanned(self):
        self.write("safe.txt", "safe")
        self.commit()
        self.git("tag", "--annotate", "sample", "-m", SYNTHETIC_TOKEN)
        self.assertEqual(self.run_guard("--history")[0], 1)
        self.git("tag", "--delete", "sample")
        self.write("safe.txt", "safe changed")
        self.commit(SYNTHETIC_TOKEN)
        self.assertEqual(self.run_guard("--history")[0], 1)

    def test_ignored_synthetic_env_values_detected_in_regular_file(self):
        value = "synthetic" + "-long-secret-value"
        self.write(".gitignore", ".env\n")
        self.write(".env", "CUSTOM_TOKEN=" + value + "\n")
        self.write("leak.txt", value)
        self.git("add", "leak.txt")
        status, output = self.run_guard("--staged")
        self.assertEqual(status, 1)
        self.assertNotIn(value, output)

    def test_git_and_timeout_errors_fail_closed_without_details(self):
        failures = [OSError(SYNTHETIC_TOKEN), ValueError(SYNTHETIC_TOKEN),
                    subprocess.CalledProcessError(1, ["git"], stderr=SYNTHETIC_TOKEN),
                    subprocess.TimeoutExpired(["git"], 60, output=SYNTHETIC_TOKEN)]
        for error in failures:
            with self.subTest(error=type(error).__name__), patch.object(self.guard, "git", side_effect=error):
                self.assertEqual(self.run_guard("--staged")[0], 2)

    def test_environment_read_failure_fails_closed(self):
        with patch.object(self.guard, "local_credentials", side_effect=OSError(SYNTHETIC_TOKEN)):
            self.assertEqual(self.run_guard("--staged")[0], 2)

    def test_pre_commit_hook_blocks_synthetic_staged_secret(self):
        self.write("config.txt", SYNTHETIC_TOKEN)
        self.git("add", "config.txt")
        self.write("scripts/check_secrets.py", (SOURCE_ROOT / "scripts" / "check_secrets.py").read_text(encoding="utf-8"))
        self.write(".git/hooks/pre-commit", (SOURCE_ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8"))
        (self.folder / ".git/hooks/pre-commit").chmod(0o755)
        process = subprocess.run(["git", "commit", "-m", "blocked test"], cwd=self.folder,
                                 capture_output=True, timeout=30, check=False)
        self.assertNotEqual(process.returncode, 0)
        output = (process.stdout + process.stderr).decode(errors="replace")
        self.assertIn("BLOCKED", output)
        self.assertNotIn(SYNTHETIC_TOKEN, output)

    def test_token_in_filename_is_detected_without_disclosure(self):
        self.write("config-" + SYNTHETIC_TOKEN + ".txt", "nonsecret content")
        self.git("add", "--all")
        self.assertEqual(self.run_guard("--staged")[0], 1)
        self.assertEqual(self.run_guard("--worktree")[0], 1)
        self.commit()
        self.assertEqual(self.run_guard("--history")[0], 1)

    def test_sensitive_filename_never_echoes_known_token(self):
        self.write(".env." + SYNTHETIC_TOKEN, "# synthetic empty fixture")
        self.git("add", "--all")
        self.assertEqual(self.run_guard("--staged")[0], 1)

    def test_pre_push_shell_consumes_stdin_without_a_push(self):
        shell = shutil.which("sh")
        if not shell and os.name == "nt":
            git_path = Path(shutil.which("git")).resolve()
            candidates = [git_path.parent.parent / "usr/bin/sh.exe", git_path.parent.parent / "bin/sh.exe"]
            shell = next((str(path) for path in candidates if path.exists()), None)
        if not shell:
            self.skipTest("Git's POSIX shell is unavailable")
        self.write("old.txt", SYNTHETIC_TOKEN)
        head = self.commit()
        self.write("scripts/check_secrets.py", (SOURCE_ROOT / "scripts" / "check_secrets.py").read_text(encoding="utf-8"))
        hook = self.write(".git/hooks/pre-push", (SOURCE_ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8"))
        for stdin, status in ((f"refs/heads/main {head} refs/heads/main {'0' * 40}\n", 1),
                              (f"(delete) {'0' * 40} refs/heads/main {head}\n", 0)):
            process = subprocess.run([shell, str(hook)], cwd=self.folder, input=stdin, text=True,
                                     capture_output=True, timeout=30, check=False)
            self.assertEqual(process.returncode, status)
            self.assertNotIn(SYNTHETIC_TOKEN, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
