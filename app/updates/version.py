"""CalVer comparison for the update check (spec 2026-11-20).

Release tags are ``vYYYY.MM.DD.N``. They compare as a tuple of four
integers; anything else (``dev``, a short sha, an rc suffix) is not a
release and never takes part in a comparison.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_CALVER = re.compile(r"^v?(\d{4})\.(\d{1,2})\.(\d{1,2})\.(\d{1,3})$")

CalVer = tuple[int, int, int, int]


def parse_calver(value: Any) -> CalVer | None:
    if not isinstance(value, str):
        return None
    m = _CALVER.match(value.strip())
    if m is None:
        return None
    y, mo, d, n = (int(g) for g in m.groups())
    return (y, mo, d, n)


def is_newer(candidate: Any, running: Any) -> bool:
    """True only when both are releases and ``candidate`` is strictly later."""
    a, b = parse_calver(candidate), parse_calver(running)
    return a is not None and b is not None and a > b


def advisory_applies(advisory: Any, running: Any) -> bool:
    """``introduced <= running < fixed_in``; ``introduced`` is optional.
    A malformed advisory, or a running version that is not a release,
    never applies (a false red banner is worse than none)."""
    if not isinstance(advisory, dict):
        return False
    run = parse_calver(running)
    fixed = parse_calver(advisory.get("fixed_in"))
    if run is None or fixed is None:
        return False
    if "introduced" in advisory and advisory["introduced"] is not None:
        intro = parse_calver(advisory["introduced"])
        if intro is None or run < intro:
            return False
    return run < fixed


# --- The running release ---------------------------------------------------------

_BUILD_VERSION = os.environ.get("VW_BUILD_VERSION") or "dev"
_HEADING = re.compile(r"^## \[(v?\d{4}\.\d{1,2}\.\d{1,2}\.\d{1,3})\]", re.MULTILINE)
#: Where a source build finds the changelog: the repo root of a git checkout,
#: and /app/meta in an image built from source (the Dockerfile copies the
#: top-level *.md files there).
_SEARCH = (Path(__file__).resolve().parents[2], Path("/app/meta"))


def release_version(search: Iterable[Path] | None = None) -> str:
    """The release this install runs. Release images carry it as the
    VW_BUILD_VERSION build arg; an image or checkout built from the public
    sources has none, so the newest released heading of its changelog stands
    in. Without either it stays "dev", which never checks for updates."""
    if parse_calver(_BUILD_VERSION) is not None:
        return _BUILD_VERSION
    for d in search if search is not None else _SEARCH:
        try:
            names = [p for p in d.iterdir() if p.name.lower() == "changelog.md"]
        except OSError:
            continue
        for p in names:
            try:
                m = _HEADING.search(p.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
            if m:
                v = m.group(1)
                return v if v.startswith("v") else f"v{v}"
    return _BUILD_VERSION
