"""Offline Git guard. Never prints matching secret values or source lines.

Uses known credential formats, sensitive filenames, literal assignments and
exact credential values from local .env files. This is a defense in depth, not
a guarantee that every possible secret format can be recognized.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {".env.example", ".env.sample", ".env.template"}
SECRET_NAME = re.compile(r"(?i)(?:secret|password|passwd|token|api[_-]?key|api_hash|oauth)")
PATTERNS = {
    "private key": rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----",
    "Telegram bot token": rb"\b[0-9]{6,12}:[A-Za-z0-9_-]{30,}\b",
    "GitHub token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "API token": rb"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|pplx-[A-Za-z0-9_-]{20,}|y0_[A-Za-z0-9_-]{25,})\b",
    "AWS access key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "literal credential": rb"(?i)\b(?:[a-z][a-z0-9_]*_)?(?:api[_-]?key|api_hash|bot[_-]?token|password|client_secret|wordstat_oauth)\s*[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_./+:=-]{16,}[\"']",
}
RULES = {name: re.compile(pattern) for name, pattern in PATTERNS.items()}


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, timeout=60).stdout


def sensitive_path(path: str) -> bool:
    name = Path(path).name.lower()
    if name in TEMPLATES:
        return False
    return (name == ".env" or name.startswith(".env.")
            or ".session" in name
            or name.endswith((".env", ".pem", ".key", ".p12", ".pfx"))
            or name in {"credentials.json", "secrets.json", "id_rsa", "id_ed25519"})


def local_credentials() -> list[bytes]:
    values = set()
    for path in ROOT.glob(".env*"):
        if not path.is_file() or path.name in TEMPLATES:
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.removeprefix("export ").partition("=")
            if not separator or not SECRET_NAME.search(key.strip()):
                continue
            value = value.strip()
            if value.startswith(('"', "'")):
                value = value[1:].split(value[0], 1)[0]
            else:
                value = value.split(" #", 1)[0].strip()
            if len(value) >= 8 and "${" not in value:
                values.add(value.encode("utf-8"))
    return sorted(values)


def detect(data: bytes, credentials: list[bytes]) -> list[str]:
    findings = [name for name, pattern in RULES.items() if pattern.search(data)]
    if any(value in data for value in credentials):
        findings.append("credential from local environment file")
    return findings


def redact_location(location: str, credentials: list[bytes]) -> str:
    data = location.encode("utf-8", errors="replace")
    for value in credentials:
        data = data.replace(value, b"[redacted]")
    for pattern in RULES.values():
        data = pattern.sub(b"[redacted]", data)
    return data.decode("utf-8", errors="replace")


def scan_index(credentials: list[bytes]) -> list[tuple[str, str]]:
    findings = []
    for record in git("ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode().split()
        path = raw_path.decode("utf-8", errors="replace")
        if stage != "0":
            raise ValueError("Resolve index conflicts before committing")
        if sensitive_path(path):
            findings.append((path, "sensitive filename"))
        for rule in detect(raw_path, credentials):
            findings.append((path, "filename: " + rule))
        if mode == "160000":
            raise ValueError("Submodules require a separate secret review")
        for rule in detect(git("cat-file", "blob", oid), credentials):
            findings.append((path, rule))
    return findings


def scan_worktree(credentials: list[bytes]) -> list[tuple[str, str]]:
    findings = []
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    for raw_path in set(paths.split(b"\0")) - {b""}:
        path = raw_path.decode("utf-8", errors="strict")
        file = ROOT / path
        if not file.exists():
            continue
        if sensitive_path(path):
            findings.append((path, "sensitive filename"))
        for rule in detect(raw_path, credentials):
            findings.append((path, "filename: " + rule))
        if file.is_symlink():
            raise ValueError("Symlinks require a separate secret review")
        for rule in detect(file.read_bytes(), credentials):
            findings.append((path, rule))
    return findings


def scan_history(refs: list[str], credentials: list[bytes]) -> list[tuple[str, str]]:
    findings = []
    # Check every historical filename, including files deleted before HEAD.
    for raw_path in git("log", "--format=", "--name-only", "-z", *refs, "--").split(b"\0"):
        path = raw_path.decode("utf-8", errors="replace").strip("\r\n")
        if path and sensitive_path(path):
            findings.append((path, "sensitive filename in history"))
        for rule in detect(raw_path, credentials):
            findings.append((path, "historical filename: " + rule))
    # --no-object-names avoids ambiguous quoting of names and scans each blob
    # once. Commit/tag objects are scanned too, including their messages.
    objects = git("rev-list", "--objects", "--no-object-names", *refs, "--").splitlines()
    for oid in dict.fromkeys(objects):
        identifier = oid.decode("ascii")
        kind = git("cat-file", "-t", identifier).strip()
        if kind not in {b"blob", b"commit", b"tag"}:
            continue
        data = git("cat-file", kind.decode("ascii"), identifier)
        for rule in detect(data, credentials):
            findings.append((f"Git object {identifier[:12]} ({kind.decode()})", rule))
    return list(dict.fromkeys(findings))


def push_refs(lines) -> list[str]:
    refs = []
    for line in lines:
        fields = line.split()
        if len(fields) != 4 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", fields[1]):
            raise ValueError("Invalid pre-push input")
        oid = fields[1]
        if set(oid) != {"0"}:  # Deleting a remote ref transfers no new objects.
            refs.append(oid)
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--worktree", action="store_true")
    mode.add_argument("--history", action="store_true")
    mode.add_argument("--pre-push", action="store_true")
    args = parser.parse_args()
    try:
        credentials = local_credentials()
        if args.staged:
            findings = scan_index(credentials)
        elif args.worktree:
            findings = scan_worktree(credentials)
        else:
            refs = push_refs(sys.stdin) if args.pre_push else ["--all"]
            findings = scan_history(refs, credentials) if refs else []
        for location, rule in findings:
            location = redact_location(location, credentials)
            print(f"BLOCKED: {location!r}: {rule}", file=sys.stderr)
        if findings:
            print("Remove credentials before committing/pushing; rotate any exposed keys.", file=sys.stderr)
            return 1
        print("Secret check passed (known patterns and local credential values).")
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # Command output and exception details may themselves contain secrets.
        print(f"Secret check could not complete ({type(exc).__name__}); blocked.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
