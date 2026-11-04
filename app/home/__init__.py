"""What the instance's address (`/`) shows (spec 2026-11-04, "Home page modes").

``home_page_mode`` is ``off`` | ``builtin`` | ``custom``; ``home_page_off_action``
is ``console`` | ``404`` and only matters while the mode is ``off``. The
providers that can answer for ``builtin`` / ``custom`` register themselves in
``app.state.ext.home_providers``; core's own welcome page is the lowest-priority
``builtin`` one. A mode no provider serves behaves as ``off``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from app.db.database import open_db
from app.db.repos.settings import SettingsRepo

if TYPE_CHECKING:
    from app.ext import HomeProvider

MODES = ("off", "builtin", "custom")
OFF_ACTIONS = ("console", "404")
DEFAULT_MODE = "off"
DEFAULT_OFF_ACTION = "console"


async def current_mode(db) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    """(mode, off_action) as stored; a missing or unknown value is the
    conservative one (off / console), so a half-bootstrapped DB never exposes a page."""
    repo = SettingsRepo(db)
    mode = ((await repo.get("home_page_mode")) or "").strip().lower()
    action = ((await repo.get("home_page_off_action")) or "").strip().lower()
    return (
        mode if mode in MODES else DEFAULT_MODE,
        action if action in OFF_ACTIONS else DEFAULT_OFF_ACTION,
    )


def pick_provider(request: Request, mode: str) -> HomeProvider | None:
    """The highest-priority registered provider for ``mode`` (None for ``off``
    or when nothing serves it)."""
    reg = getattr(request.app.state, "ext", None)
    if reg is None or mode not in ("builtin", "custom"):
        return None
    best: HomeProvider | None = None
    for p in reg.home_providers:
        if p.mode == mode and (best is None or p.priority > best.priority):
            best = p
    return best


async def active_provider(request: Request) -> tuple[str, str, HomeProvider | None]:
    """(mode, off_action, provider answering for the mode, or None)."""
    async with open_db(request.app.state.settings.db_path) as db:
        mode, action = await current_mode(db)
    return mode, action, pick_provider(request, mode)
