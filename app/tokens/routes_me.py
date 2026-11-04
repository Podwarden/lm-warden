"""The key holder's own area (signed in with their API key, app/stats/routes_forest.py forest
login): read the key, its usage and the guard's log; rename, rotate (grace none / 1 h / 24 h) and
revoke it. Every route acts on the signed-in key only; an admin session is refused here (the
admin has /api/tokens)."""

from __future__ import annotations

import asyncio
import time
from contextlib import closing
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from app.auth.jwt import mint_forest
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo
from app.guard.events import query_events
from app.stats.routes_forest import FOREST_TTL_MIN, ForestViewer, _connect, forest_viewer
from app.tokens import routes_api as tokens

router = APIRouter(prefix="/api/me")


async def key_holder(viewer: ForestViewer = Depends(forest_viewer)) -> str:
    if viewer.token_id is None:
        raise HTTPException(403, "sign in with your API key to use this page")
    return viewer.token_id


def _public(detail: dict[str, Any]) -> dict[str, Any]:
    """The key's details as its holder sees them: the admin's custom guard rules are counted,
    not shown (seeing the patterns would tell how to get around them)."""
    out = dict(detail)
    out.pop("guard", None)
    eff = out.get("guard_effective")
    if isinstance(eff, dict):
        eff = dict(eff)
        eff["rules"] = len(eff.get("rules") or [])
        eff.pop("overrides", None)
        out["guard_effective"] = eff
    return out


class MeKeyPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=64)


class MeRotate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grace_hours: Literal[0, 1, 24] = 24


class MeRevoke(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm_name: str


async def _detail(request: Request, tid: str) -> dict[str, Any]:
    async with open_db(request.app.state.settings.db_path) as db:
        d = await tokens._token_detail(db, tid)
    if d is None:
        raise HTTPException(404)
    return _public(tokens._with_effective(d, request))


@router.get("/key")
async def get_key(request: Request, tid: str = Depends(key_holder)) -> dict[str, Any]:
    return await _detail(request, tid)


@router.patch("/key")
async def rename_key(
    body: MeKeyPatch, request: Request, tid: str = Depends(key_holder)
) -> dict[str, Any]:
    async with open_db(request.app.state.settings.db_path) as db:
        await TokenRepo(db).update(tid, name=body.name.strip())
    return await _detail(request, tid)


@router.post("/key/rotate", status_code=status.HTTP_201_CREATED)
async def rotate_key(
    body: MeRotate, request: Request, tid: str = Depends(key_holder)
) -> dict[str, Any]:
    out = await tokens.rotate_token(
        tid, tokens.TokenRotate(grace_hours=body.grace_hours), request, _user="key-holder"
    )
    # rotate_token ended this key's sign-ins, this one included: sign in with the new key.
    sid = await request.app.state.auth_sessions.create("key", out["id"])
    return {
        **out,
        "token": mint_forest(out["id"], request.app.state.jwt_secret, FOREST_TTL_MIN, sid=sid),
        "expires_in": FOREST_TTL_MIN * 60,
    }


@router.post("/key/revoke", status_code=204, response_class=Response)
async def revoke_key(body: MeRevoke, request: Request, tid: str = Depends(key_holder)) -> Response:
    d = await _detail(request, tid)
    if body.confirm_name != d["name"]:
        raise HTTPException(409, "type the key's name exactly to revoke it")
    await tokens.delete_token(tid, request, _user="key-holder")
    return Response(status_code=204)


@router.get("/usage")
async def usage(
    request: Request,
    range: str = Query(default="24h", pattern=r"^(1h|24h|7d)$"),
    tid: str = Depends(key_holder),
) -> Any:
    return await tokens.get_token_usage(tid, request, range=range, _user="key-holder")


@router.get("/cache")
async def cache(
    request: Request,
    range: str = Query(default="24h", pattern=r"^(1h|24h|7d)$"),
    tid: str = Depends(key_holder),
) -> Any:
    return await tokens.get_token_cache(tid, request, range=range, _user="key-holder")


@router.get("/guard/events")
async def guard_events(
    request: Request,
    since: float | None = None,
    limit: int = Query(200, ge=1, le=2000),
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    t = since if since is not None else time.time() - 7 * 86400
    db_path = request.app.state.settings.db_path

    def work() -> dict[str, Any]:
        with closing(_connect(db_path)) as db:
            return query_events(db, token_id=tid, since=t, action=None, rule_id=None, limit=limit)

    return await asyncio.to_thread(work)
