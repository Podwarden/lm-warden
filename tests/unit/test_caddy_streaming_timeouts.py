"""Every Caddyfile the product ships must disable ``read_body_idle``.

Caddy 2.11 defaults the server timeout ``read_body_idle`` to 1m and aborts the
client connection 60 s into a streamed response (Claude Code /v1/messages,
SSE, OpenAI streaming). The floating ``caddy:2-alpine`` tag picked that up
silently on a restart, so the image is pinned as well.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Prose and history legitimately mention the old tag; shipped artifacts do not.
SKIP_PREFIXES = ("docs/", "documents/", "frontend/", "tests/", "changelog.md")
TEXT_SUFFIXES = {".yml", ".yaml", ".sh", ".json", ".conf", ""}


_WALK_EXCLUDES = {".git", "node_modules", ".venv", "venv", "__pycache__", ".next", "dist", "build"}


def _walk() -> list[Path]:
    """Filesystem fallback for an export/tarball with no git metadata."""
    return [
        p
        for p in ROOT.rglob("*")
        if p.is_file() and not _WALK_EXCLUDES.intersection(p.relative_to(ROOT).parts)
    ]


def _tracked() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        return _walk()
    files = [ROOT / p for p in out.split("\0") if p]
    return files or _walk()


def _shipped_files() -> list[Path]:
    files = []
    for p in _tracked():
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(SKIP_PREFIXES) or not p.is_file():
            continue
        if p.name.startswith("Caddyfile") or p.suffix in TEXT_SUFFIXES:
            files.append(p)
    return files


def _block(text: str, name: str) -> str:
    """Body of the first ``name { ... }`` block, matched by brace depth."""
    m = re.search(rf"^\s*{name}\s*\{{", text, re.M)
    if not m:
        return ""
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end() : i - 1]


def _global_servers_timeouts(text: str) -> str:
    """Body of the ``timeouts { }`` inside the global ``servers { }`` block."""
    return _block(_block(text, "servers"), "timeouts")


def _caddyfile_sources() -> list[Path]:
    found = []
    for p in _shipped_files():
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        if p.name.startswith("Caddyfile") or ("reverse_proxy" in text and "admin off" in text):
            found.append(p)
    return found


def test_shipped_caddyfiles_are_found() -> None:
    names = {p.relative_to(ROOT).as_posix() for p in _caddyfile_sources()}
    assert "deploy/caddy/Caddyfile" in names


def test_every_shipped_caddyfile_disables_read_body_idle() -> None:
    bad = []
    for p in _caddyfile_sources():
        body = _global_servers_timeouts(p.read_text())
        code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
        if not re.search(r"^\s*read_body_idle\s+-1s\s*$", code, re.M):
            bad.append(p.relative_to(ROOT).as_posix())
    assert not bad, f"read_body_idle -1s missing in global servers timeouts: {bad}"


def test_no_floating_caddy_image_tag() -> None:
    pinned = re.compile(r"^\d+\.\d+\.\d+(-alpine)?$")
    bad = []
    for p in _shipped_files():
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        for m in re.finditer(r"(?<![\w/.-])caddy:([\w][\w.-]*)", text):
            if not pinned.match(m.group(1)):
                bad.append(f"{p.relative_to(ROOT).as_posix()}: caddy:{m.group(1)}")
    assert not bad, f"floating Caddy image tags: {bad}"
