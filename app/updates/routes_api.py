"""The update banner's API (spec 2026-11-20).

GET  /api/system/update-status          what the console banner shows
POST /api/system/update-status/dismiss  hide the "new release" line for one version
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.settings import SettingsRepo
from app.updates import client as uc
from app.updates.version import advisory_applies, is_newer, parse_calver, release_version

router = APIRouter()

_MAX_DISMISSED = 50


def current_version() -> str:
    return release_version()


def _load_json(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except ValueError:
        return default


def _text(v: Any, limit: int = 300) -> str | None:
    if v is None or isinstance(v, dict | list):
        return None
    return str(v)[:limit]


def _advisory(a: dict[str, Any]) -> dict[str, str | None]:
    """The shape the console renders, whatever the server sent: every value a
    short string or None, and a link only when it is https (a malformed
    advisory must cost one line of the banner, never the console)."""
    url = _text(a.get("url"), 500)
    return {
        "id": _text(a.get("id"), 64) or "advisory",
        "severity": _text(a.get("severity"), 32),
        "title": _text(a.get("title")),
        "fixed_in": _text(a.get("fixed_in"), 32),
        "url": url if url and url.startswith("https://") else None,
    }


@router.get("/api/system/update-status")
async def get_update_status(request: Request, _user: str = Depends(require_jwt)) -> dict[str, Any]:
    db_path = request.app.state.settings.db_path
    edition = request.app.state.ext.edition
    async with open_db(db_path) as db:
        kv = await SettingsRepo(db).get_many([uc.RESULT_KEY, uc.CHECKED_KEY, uc.DISMISSED_KEY])
    result = _load_json(kv.get(uc.RESULT_KEY), {})
    if not isinstance(result, dict):
        result = {}
    dismissed = _load_json(kv.get(uc.DISMISSED_KEY), [])
    running = current_version()
    latest = result.get("latest") if isinstance(result.get("latest"), dict) else None
    newer = bool(latest) and is_newer(latest.get("version"), running)  # type: ignore[union-attr]
    is_dismissed = bool(latest) and latest.get("version") in dismissed  # type: ignore[union-attr]
    raw = result.get("advisories")
    advisories = [
        _advisory(a) for a in (raw if isinstance(raw, list) else []) if advisory_applies(a, running)
    ]
    on = await uc.enabled(db_path, edition)
    return {
        "enabled": on,
        "current": running,
        "last_checked_at": kv.get(uc.CHECKED_KEY),
        "latest": latest,
        # Turned off: no stale "new release" line (advisories still show).
        "update_available": on and newer and not is_dismissed,
        "dismissed": is_dismissed,
        "advisories": advisories,
    }


class DismissBody(BaseModel):
    version: str


@router.post("/api/system/update-status/dismiss")
async def dismiss_update(
    body: DismissBody, request: Request, _user: str = Depends(require_jwt)
) -> dict[str, Any]:
    if parse_calver(body.version) is None:
        raise HTTPException(status_code=422, detail="version must be a release tag (vYYYY.MM.DD.N)")
    db_path = request.app.state.settings.db_path
    async with open_db(db_path) as db:
        repo = SettingsRepo(db)
        dismissed = _load_json(await repo.get(uc.DISMISSED_KEY), [])
        if not isinstance(dismissed, list):
            dismissed = []
        if body.version not in dismissed:
            dismissed = [*dismissed, body.version][-_MAX_DISMISSED:]
            await repo.set(uc.DISMISSED_KEY, json.dumps(dismissed))
    return {"dismissed": body.version}
