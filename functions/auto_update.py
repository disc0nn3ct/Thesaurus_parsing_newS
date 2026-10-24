"""Optional, fast-forward-only updates from origin/main."""

import logging
import os
from pathlib import Path
import subprocess
import sys

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GIT_TIMEOUT_SECONDS = 60


def _git(*args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )


def _safe_checkout():
    branch = _git("symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    if branch.returncode != 0 or branch.stdout.strip() != "main":
        logger.warning("Автообновление пропущено: требуется ветка main.")
        return False
    if _git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        logger.warning("Автообновление пропущено: есть локальные изменения файлов.")
        return False
    return True


def _is_fast_forward(commit):
    result = _git("merge-base", "--is-ancestor", "HEAD", commit, check=False)
    if result.returncode not in (0, 1):
        result.check_returncode()
    if result.returncode == 1:
        logger.warning("Автообновление пропущено: локальная история опережает удалённую или расходится с ней.")
        return False
    return True


def _write_commit(commit_file, commit):
    path = Path(commit_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(commit, encoding="utf-8")


def check_git_update(commit_file="current_commit.txt"):
    """Return an available fast-forward commit; the marker never controls updates."""
    try:
        if not _safe_checkout():
            return None
        _git("fetch", "--no-tags", "origin", "refs/heads/main:refs/remotes/origin/main")
        current_commit = _git("rev-parse", "--verify", "HEAD").stdout.strip()
        new_commit = _git("rev-parse", "--verify", "origin/main").stdout.strip()
        if current_commit == new_commit:
            _write_commit(commit_file, current_commit)
            logger.info("Локальная версия актуальна.")
            return None
        if not _is_fast_forward(new_commit):
            return None
        logger.info("Обнаружено обновление: %s", new_commit)
        return new_commit
    except (OSError, subprocess.SubprocessError):
        logger.exception("Ошибка при проверке обновления Git.")
        return None


def update_and_restart(new_commit, commit_file):
    """Apply the previously checked origin/main commit without creating a merge."""
    try:
        # Recheck because the checkout may have changed since check_git_update.
        if not _safe_checkout():
            return
        remote_commit = _git("rev-parse", "--verify", "origin/main").stdout.strip()
        if new_commit != remote_commit:
            logger.warning("Автообновление пропущено: удалённая ссылка изменилась после проверки.")
            return
        current_commit = _git("rev-parse", "--verify", "HEAD").stdout.strip()
        if current_commit == new_commit:
            _write_commit(commit_file, current_commit)
            return
        if not _is_fast_forward(new_commit):
            return
        _git("merge", "--ff-only", new_commit)
        if _git("rev-parse", "--verify", "HEAD").stdout.strip() != new_commit:
            logger.error("Автообновление не подтверждено: HEAD отличается от ожидаемого коммита.")
            return
        _write_commit(commit_file, new_commit)
        logger.info("Проект обновлён. Перезапуск...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except (OSError, subprocess.SubprocessError):
        logger.exception("Ошибка при обновлении и перезапуске.")


def check_and_restart_if_updated(commit_file="log/current_commit.txt"):
    new_commit = check_git_update(commit_file)
    if new_commit:
        update_and_restart(new_commit, commit_file)
