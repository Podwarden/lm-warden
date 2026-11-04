"""Admin API for the sensitive data guard: platform settings, packs, the try-it box (stores
nothing) and the action log. Every route needs the admin session."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import closing
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.settings import SettingsRepo
from app.guard.config import compile_guard, validate
from app.guard.events import ACTIONS, query_events
from app.guard.placeholders import guard_secret, placeholder_hex, render
from app.guard.proxy import mask_for
from app.guard.rules import BUILTIN, DEFAULT_PACKS, PACKS
from app.guard.scanner import scan_text
from app.stats.routes_forest import _connect

_BUILTIN_IDS = {r.id for r in BUILTIN}
router = APIRouter(prefix="/api/guard", dependencies=[Depends(require_jwt)])
SETTINGS_KEY = "guard"


def mask(value: str, public_prefix: str | None = None) -> str:
    """For the admin's own pasted text in the try-it box and the client's own block reply only."""
    if public_prefix and value.startswith(public_prefix):
        return f"{public_prefix}… ({len(value)} chars)"
    if len(value) < 12:
        return f"({len(value)} chars)"
    head = min(4, len(value) // 4)
    tail = min(2, len(value) // 8)
    return f"{value[:head]}…{value[-tail:] if tail else ''} ({len(value)} chars)"


async def load_platform(app_state: Any) -> dict[str, Any] | None:
    async with open_db(app_state.settings.db_path) as db:
        raw = await SettingsRepo(db).get(SETTINGS_KEY)
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


class PlatformIn(BaseModel):
    guard: dict[str, Any] | None


class TestIn(BaseModel):
    text: str = Field(max_length=1_000_000)
    guard: dict[str, Any] | None = None


@router.get("/settings")
async def get_settings(request: Request) -> dict[str, Any]:
    return {"guard": getattr(request.app.state, "guard_platform", None)}


@router.put("/settings")
async def put_settings(body: PlatformIn, request: Request) -> dict[str, Any]:
    value = None
    if body.guard is not None:
        try:
            value = validate(body.guard)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
    async with open_db(request.app.state.settings.db_path) as db:
        await SettingsRepo(db).set(SETTINGS_KEY, json.dumps(value) if value is not None else "")
    request.app.state.guard_platform = value
    return {"guard": value}


@router.get("/packs")
async def packs() -> dict[str, Any]:
    return {
        "default": list(DEFAULT_PACKS),
        "packs": [
            {
                "id": pid,
                "label": label,
                "default": pid in DEFAULT_PACKS,
                "rules": list(
                    {
                        r.id: {"id": r.id, "label": r.label, "public_prefix": r.public_prefix}
                        for r in reversed(BUILTIN)
                        if r.pack == pid
                    }.values()
                )[::-1],
            }
            for pid, label in PACKS.items()
        ],
    }


@router.post("/test")
async def test(body: TestIn, request: Request) -> dict[str, Any]:
    raw = (
        body.guard
        if body.guard is not None
        else (getattr(request.app.state, "guard_platform", None) or {"mode": "sanitize"})
    )
    try:
        g = compile_guard(raw)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    assert g is not None
    t0 = time.perf_counter()
    found = await asyncio.to_thread(scan_text, body.text, g.rules)
    ms = (time.perf_counter() - t0) * 1000
    secret = guard_secret(request.app.state.settings.cookie_secret)
    by_id = {r.id: r for r in g.rules}
    preview = body.text
    for f in sorted(found, key=lambda x: x.start, reverse=True):
        preview = (
            preview[: f.start]
            + render(placeholder_hex(secret, "try-it", f.value))
            + preview[f.end :]
        )
    return {
        "ms": round(ms, 2),
        "preview": preview,
        "findings": [
            {
                "rule_id": f.rule_id,
                "label": by_id[f.rule_id].label,
                "start": f.start,
                "end": f.end,
                "length": len(f.value),
                "masked": mask_for(f.value, f.rule_id)
                if f.rule_id in _BUILTIN_IDS
                else f"({len(f.value)} chars)",
                "action": g.action_for(by_id[f.rule_id]),
            }
            for f in found
        ],
    }


@router.get("/events")
async def events(
    request: Request,
    token: str | None = None,
    since: float | None = None,
    action: str | None = None,
    rule: str | None = None,
    limit: int = Query(200, ge=1, le=2000),
) -> dict[str, Any]:
    if action is not None and action not in ACTIONS:
        raise HTTPException(400, f"action must be one of {list(ACTIONS)}")
    t = since if since is not None else time.time() - 7 * 86400
    db_path = request.app.state.settings.db_path

    def work() -> dict[str, Any]:
        with closing(_connect(db_path)) as db:
            return query_events(
                db, token_id=token, since=t, action=action, rule_id=rule, limit=limit
            )

    return await asyncio.to_thread(work)
