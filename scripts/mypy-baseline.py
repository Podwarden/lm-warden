#!/usr/bin/env python3
"""Gate `mypy app/` on a committed baseline of accepted errors.

WHY. `mypy --strict` over app/ has been red for a long time (322 errors in 57
files on 2026-09-03, #243) and nothing in CI ran it, so the Makefile advertised
a check nobody could act on and a genuinely new error could land unnoticed
inside the noise. Fixing every error is a large change that touches hot paths
where a wrong annotation is worse than none, so the debt is recorded instead:
`mypy-baseline.txt` lists every accepted error, and this script fails when the
tree drifts from it in EITHER direction.

  * an error mypy reports that is not in the baseline  -> fail ("new")
  * a baseline entry mypy no longer reports             -> fail ("stale")

Failing on the second is deliberate. A baseline that only ratchets up would
silently rot; failing forces `make typecheck-baseline` to shrink it, and the
resulting diff is pure deletions.

HOW ENTRIES ARE KEYED. Line numbers are not part of the key -- they shift
whenever anything above them changes and would make every edit a baseline
edit. An entry is (file, error code, message, source line text): specific
enough to tell one untyped `def` from another in the same file, stable across
unrelated edits. Renaming an untyped function's signature therefore shows up as
one "new" plus one "stale" entry, which is the intended nudge to annotate it
while you are there. mypy 1.13 has no baseline mode of its own; a per-module
`disable_error_code` override would silence NEW `no-untyped-def`s in exactly
the files that have the most, which is the failure mode this replaces.

USAGE (both run inside the project's dependency image via the Makefile):

    make typecheck            # python scripts/mypy-baseline.py
    make typecheck-baseline   # python scripts/mypy-baseline.py --write

Raw `mypy app/` still lists everything, line numbers included, when you want
the full to-do list rather than the delta.

The baseline file is plain text, two lines per entry, in mypy's own format
minus the line number, sorted, with a comment header. Additions to it in a
merge request are a code-review flag: they mean an error was accepted rather
than fixed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "mypy-baseline.txt"
TARGET = "app/"
INDENT = "    "

# One accepted error: (file, error code, message, stripped source line).
Key = tuple[str, str, str, str]

_HEADER_RE = re.compile(r"^(?P<file>[^\s:][^:]*): error: (?P<message>.*)  \[(?P<code>[\w-]+)\]$")


def _fail(msg: str, status: int = 2) -> NoReturn:
    sys.stderr.write(f"mypy-baseline: {msg}\n")
    sys.exit(status)


# --------------------------------------------------------------------------
# Running mypy
# --------------------------------------------------------------------------


class Finding:
    __slots__ = ("file", "line", "code", "message", "source")

    def __init__(self, file: str, line: int, code: str, message: str, source: str) -> None:
        self.file = file
        self.line = line
        self.code = code
        self.message = message
        self.source = source

    @property
    def key(self) -> Key:
        return (self.file, self.code, self.message, self.source)


class _SourceCache:
    def __init__(self) -> None:
        self._files: dict[str, list[str]] = {}

    def line(self, file: str, lineno: int) -> str:
        lines = self._files.get(file)
        if lines is None:
            try:
                lines = (REPO / file).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            self._files[file] = lines
        if 0 < lineno <= len(lines):
            return lines[lineno - 1].strip()
        return ""


def run_mypy() -> tuple[list[Finding], str]:
    """Run mypy over TARGET and return its error findings plus its version."""
    version = subprocess.run(
        [sys.executable, "-m", "mypy", "--version"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--output", "json", "--no-error-summary", TARGET],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    # 0 = clean, 1 = errors reported; anything else is mypy itself failing
    # (bad config, crash, missing package) and must not be mistaken for a
    # result in either direction.
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        _fail(f"mypy exited {proc.returncode}; not a type-check result", proc.returncode)

    sources = _SourceCache()
    findings: list[Finding] = []
    for raw in proc.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            sys.stderr.write(proc.stdout)
            _fail(f"could not parse mypy --output json line: {raw!r}")
        if obj.get("severity") != "error":
            continue  # notes and hints hang off an error; the error is the key
        file = str(obj["file"])
        line = int(obj["line"])
        findings.append(
            Finding(
                file=file,
                line=line,
                code=str(obj.get("code") or "misc"),
                message=str(obj["message"]),
                source=sources.line(file, line),
            )
        )
    return findings, version


# --------------------------------------------------------------------------
# Baseline file
# --------------------------------------------------------------------------


def format_entry(key: Key) -> list[str]:
    file, code, message, source = key
    lines = [f"{file}: error: {message}  [{code}]"]
    if source:
        lines.append(INDENT + source)
    return lines


def render_baseline(keys: Counter[Key], version: str) -> str:
    files = {k[0] for k in keys}
    total = sum(keys.values())
    header = [
        "# mypy baseline -- accepted `mypy app/` errors. See scripts/mypy-baseline.py.",
        "#",
        "# `make typecheck` fails on any error NOT listed here, and on any entry",
        "# listed here that mypy no longer reports. Regenerate with",
        "# `make typecheck-baseline` and commit the diff. Additions to this file in",
        "# a merge request mean an error was accepted, not fixed -- review them.",
        "#",
        "# Format: mypy's own output minus the line number, then the source line.",
        "# Sorted; one entry per accepted error (duplicates are separate entries).",
        f"# {total} error(s) in {len(files)} file(s); generated by {version or 'mypy'}",
        "",
    ]
    body: list[str] = []
    for key in sorted(keys):
        for _ in range(keys[key]):
            body.extend(format_entry(key))
    return "\n".join(header + body) + ("\n" if body else "")


def parse_baseline(text: str) -> Counter[Key]:
    keys: Counter[Key] = Counter()
    pending: tuple[str, str, str] | None = None  # (file, code, message)

    def flush(source: str) -> None:
        nonlocal pending
        if pending is not None:
            file, code, message = pending
            keys[(file, code, message, source)] += 1
            pending = None

    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith(INDENT):
            if pending is None:
                _fail(f"{BASELINE.name}:{lineno}: source line without a preceding error line")
            flush(line.strip())
            continue
        flush("")
        m = _HEADER_RE.match(line)
        if not m:
            _fail(f"{BASELINE.name}:{lineno}: unparseable line: {line!r}")
        pending = (m.group("file"), m.group("code"), m.group("message"))
    flush("")
    return keys


def load_baseline() -> Counter[Key]:
    if not BASELINE.exists():
        _fail(
            f"{BASELINE.name} is missing. Run `make typecheck-baseline` to create it "
            "(it should be committed)."
        )
    return parse_baseline(BASELINE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_write() -> int:
    findings, version = run_mypy()
    current = Counter(f.key for f in findings)
    before = parse_baseline(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else None
    BASELINE.write_text(render_baseline(current, version), encoding="utf-8")
    total = sum(current.values())
    files = len({k[0] for k in current})
    was = f" (was {sum(before.values())})" if before is not None else ""
    print(f"mypy-baseline: wrote {BASELINE.name}: {total} error(s) in {files} file(s){was}")
    if before is not None:
        added = sum((current - before).values())
        removed = sum((before - current).values())
        if added:
            print(
                f"mypy-baseline: WARNING {added} error(s) ADDED to the baseline; "
                "make sure the diff is what you meant to accept"
            )
        if removed:
            print(f"mypy-baseline: {removed} error(s) removed from the baseline")
    return 0


def cmd_check() -> int:
    baseline = load_baseline()
    findings, _ = run_mypy()
    current = Counter(f.key for f in findings)
    by_key: dict[Key, list[Finding]] = defaultdict(list)
    for f in findings:
        by_key[f.key].append(f)

    new = current - baseline
    stale = baseline - current

    if not new and not stale:
        total = sum(current.values())
        files = len({k[0] for k in current})
        if total == 0:
            print(
                "mypy-baseline: OK -- mypy is clean and the baseline is empty. "
                f"Retire {BASELINE.name} and point `make typecheck` at plain `mypy app/`."
            )
        else:
            print(
                f"mypy-baseline: OK -- {total} accepted error(s) in {files} file(s) "
                f"match {BASELINE.name}; 0 new"
            )
        return 0

    out = sys.stdout
    if new:
        n = sum(new.values())
        out.write(f"mypy-baseline: FAIL -- {n} error(s) not in {BASELINE.name}:\n")
        for key in sorted(new):
            # The occurrences we can point at with a line number. When the
            # baseline already accepts some copies of this exact key we cannot
            # tell which copy is the newcomer, so list them all.
            for f in sorted(by_key[key], key=lambda f: f.line):
                out.write(f"  {f.file}:{f.line}: error: {f.message}  [{f.code}]\n")
                if f.source:
                    out.write(f"  {INDENT}{f.source}\n")
        out.write(
            "  Fix them. If an error is deliberate, `make typecheck-baseline` accepts it;\n"
            "  the resulting addition to the baseline is a code-review flag.\n"
        )
    if stale:
        n = sum(stale.values())
        out.write(
            f"mypy-baseline: FAIL -- {n} entr{'y' if n == 1 else 'ies'} in {BASELINE.name} "
            "no longer reported by mypy (baseline is stale):\n"
        )
        for key in sorted(stale):
            for _ in range(stale[key]):
                for line in format_entry(key):
                    out.write(f"  {line}\n")
        out.write(
            "  Good news, but the baseline must shrink with it: run `make typecheck-baseline`\n"
            "  and commit the (deletions-only) diff.\n"
        )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"regenerate {BASELINE.name} from the current tree instead of checking against it",
    )
    args = parser.parse_args(argv)
    return cmd_write() if args.write else cmd_check()


if __name__ == "__main__":
    sys.exit(main())
