"""Install this checkout's local Git hooks without staging or committing files."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    existing = subprocess.run(["git", "config", "--local", "--get", "core.hooksPath"], cwd=ROOT,
                              capture_output=True, text=True, check=False, timeout=30)
    if existing.returncode not in (0, 1):
        raise RuntimeError("Cannot inspect Git hook configuration")
    if existing.stdout.strip() not in ("", ".githooks"):
        raise RuntimeError("Custom hooks already configured; merge the checks manually")
    for name in ("pre-commit", "pre-push"):
        path = ROOT / ".githooks" / name
        if b"\r\n" in path.read_bytes():
            raise RuntimeError("Hooks must use LF line endings")
        path.chmod(path.stat().st_mode | 0o111)
    subprocess.run(["git", "config", "--local", "core.hooksPath", ".githooks"], cwd=ROOT,
                   check=True, timeout=30)
    print("Local pre-commit and pre-push secret checks enabled.")


if __name__ == "__main__":
    main()
