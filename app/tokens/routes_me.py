"""The key holder's own area (signed in with their API key, app/stats/routes_forest.py forest
login): read the key, its usage and the guard's log; rename, rotate (grace none / 1 h / 24 h) and
revoke it; switch it to an isolated KV cache and reset that cache (never back to sharing: O2). Every route acts on the signed-in key only; an admin session is refused here (the
admin has /api/tokens)."""

from __future__ import annotations

import asyncio
import time
from contextlib import closing
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.auth.jwt import mint_forest
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo
from app.guard.events import query_events
from app.kv import policy_sync
from app.kv import usage as kv_usage
from app.kv.schemas import KvKeyReport
from app.proxy.auth import token_allows
from app.proxy.kv_scope import can_isolate_running
from app.proxy.model_cache import all_models
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


#: O2: a holder may make their key more private, never less.
BACK_TO_SHARING = tokens.BACK_TO_SHARING

#: A rotated predecessor in its grace window would change its successor's cache too.
ROTATED = "this key was rotated; change its cache setting on the successor"

#: The audit row's user for a change the key holder made.
KEY_HOLDER = "key-holder"


class MeKeyPatch(BaseModel):
    """Omit a field to leave it alone. ``kv_isolation`` "sharing" on an isolated key is a
    403: only an admin can switch it back."""

    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=64)
    kv_isolation: tokens.KvIsolation | None = None

    @field_validator("name", "kv_isolation", mode="before")
    @classmethod
    def _not_null(cls, v: object) -> object:
        if v is None:
            raise ValueError("cannot be null")
        return v


class MeRotate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grace_hours: Literal[0, 1, 24] = 24


class MeRevoke(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm_name: str


def prompt_lens_view_available(request: Request) -> bool:
    """B11, owner decision 2026-11-22 (replaces D8's two switches): a key holder
    may open captures of their OWN requests whenever Prompt Lens is on for the
    platform (app/proxy/prompt_lens_switch.py: the Settings switch, else
    ``VW_PROMPT_LENS_ENABLED``; the hub present). There is no per-key switch;
    isolation to the caller's own key is enforced by the capture route."""
    from app.proxy import prompt_lens_switch

    return prompt_lens_switch.enabled(request.app)


async def _models(request: Request, tid: str) -> list[dict[str, str]]:
    """The loaded models this key may use, each with ``cache_isolation`` as ``/v1/models``
    tells an isolated key (O4): the settings page marks the ones an isolated key is refused
    on, and the isolate dialog names them before the holder switches."""
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        row = await TokenRepo(db).get(tid)
    if row is None:
        return []
    rows = await all_models(getattr(request.app.state, "model_cache", None), settings.db_path)
    return [
        {
            "id": r.served_model_name,
            "cache_isolation": (
                "isolated" if can_isolate_running(request.app.state, r) else "unsupported"
            ),
        }
        for r in rows
        if r.status == "loaded" and token_allows(row, r.served_model_name)
    ]


async def _detail(request: Request, tid: str) -> dict[str, Any]:
    async with open_db(request.app.state.settings.db_path) as db:
        d = await tokens._token_detail(db, tid)
    if d is None:
        raise HTTPException(404)
    out = _public(tokens._with_effective(d, request))
    out["prompt_lens_view_available"] = prompt_lens_view_available(request)
    # Deprecated name of the same field (the feature was "god mode"): one release.
    out["godmode_view_available"] = out["prompt_lens_view_available"]
    out["models"] = await _models(request, tid)
    return out


@router.get("/key")
async def get_key(request: Request, tid: str = Depends(key_holder)) -> dict[str, Any]:
    return await _detail(request, tid)


@router.patch("/key")
async def rename_key(
    body: MeKeyPatch, request: Request, tid: str = Depends(key_holder)
) -> dict[str, Any]:
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        row = await repo.get(tid)
        if row is None:
            raise HTTPException(404)
        if body.kv_isolation is not None and row.rotated_at is not None:
            raise HTTPException(409, ROTATED)
        # The O2 refusal is decided inside the transaction (allow_to_sharing).
        change = await tokens.kv_change(
            request, repo, row, isolation=body.kv_isolation, reset=False, allow_to_sharing=False
        )
        if body.name is not None:
            await repo.update(tid, name=body.name.strip())
    await tokens.after_kv_change(request, tid, KEY_HOLDER, change, body.kv_isolation)
    if change.family:
        policy_sync.request_sync(request.app.state)
    return await _detail(request, tid)


@router.post("/key/kv-reset")
async def reset_key_kv(request: Request, tid: str = Depends(key_holder)) -> dict[str, Any]:
    """Reset the key's isolated cache (a fresh scope; the old one is retired). 409 for a
    sharing key."""
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        row = await repo.get(tid)
        if row is None:
            raise HTTPException(404)
        if row.rotated_at is not None:
            raise HTTPException(409, ROTATED)
        change = await tokens.kv_change(request, repo, row, isolation=None, reset=True)
    await tokens.after_kv_change(request, tid, KEY_HOLDER, change, None)
    policy_sync.request_sync(request.app.state)
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
    await tokens.delete_token(tid, request, _user=KEY_HOLDER)
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


@router.get("/key/cache", response_model=KvKeyReport)
async def key_cache(request: Request, tid: str = Depends(key_holder)) -> dict[str, Any]:
    """The key's KV report as its holder sees it: the windows and the key's own store
    totals (``store`` null when the store is not answering), its own reuse figures and
    its own most reloaded parts (``top_reused``, null when the store cannot say). Never
    who evicted its blocks, never the isolation cost, never other keys, their reloads or
    tier sizes."""
    stats = await kv_usage.fetch_stats(request.app.state)
    async with open_db(request.app.state.settings.db_path) as db:
        if await tokens._inference_row(TokenRepo(db), tid) is None:
            raise HTTPException(404)
        return await kv_usage.key_cache_report(
            db,
            tid,
            stats,
            kv_usage.network_key(request.app.state),
            admin=False,
            app_state=request.app.state,
        )


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
