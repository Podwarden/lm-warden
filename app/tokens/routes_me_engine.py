"""``GET /api/me/engine``: the engine facts of the models an inference key may use.

Authenticated like ``/v1/chat/completions`` (``require_bearer``: the same key, the same 401 for
a missing, unknown, expired or revoked one, the same 403 for a paused one). An app developer
with only an inference key, a CLI such as ``leanprompt profile detect --lmwarden``, can read it.
The key needs the ``engine_facts`` permission (the admin sets it, migration 0066): 403 without.
Only models the key may use (``token_allows``), whether or not they have served a request and
whether or not they are loaded. Static facts only (app/cache_obs/engine_facts.py).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.cache_obs.engine_facts import model_facts
from app.db.repos.tokens import TokenRow
from app.proxy.auth import require_bearer, token_allows
from app.proxy.model_cache import all_models
from app.updates.version import release_version

router = APIRouter(prefix="/api/me", tags=["engine"])

ENGINE_PATH = "/api/me/engine"
NO_PERMISSION = "this key may not read engine facts; ask the admin to allow it"


class EngineModel(BaseModel):
    #: The served model name, the one a client sends in ``model``.
    model: str
    backend: str | None
    running: bool
    #: Prefix caching on in the running engine; null when unknown.
    prefix_caching: bool | None
    #: The KV block size in tokens (a classic 16 is reported as 16); null when unknown.
    engine_block_tokens: int | None
    #: Tokens per prefix-match step (``--prefix-match-unit``), else the block size.
    engine_match_unit_tokens: int | None
    #: One more block is recomputed on every hit (EAGLE / MTP speculative decoding).
    engine_drops_last_block: bool | None
    #: ``mtp``, ``eagle`` or another method name; null when off or unknown (see ``unknown``).
    speculative: str | None
    max_model_len: int | None
    replicas: int
    #: Conversations are pinned to the replica that holds their cache (more than one replica).
    cache_aware_routing: bool
    #: The fields above that are null because the warden cannot say right now.
    unknown: list[str]


class EngineFacts(BaseModel):
    v: int = 1
    warden_version: str
    models: list[EngineModel]


async def engine_body(state: Any, key: TokenRow) -> dict[str, Any]:
    """The answer for ``key``: only the models it may use, configured or not, loaded or not."""
    rows = await all_models(getattr(state, "model_cache", None), state.settings.db_path)
    models = sorted(
        (r for r in rows if token_allows(key, r.served_model_name)),
        key=lambda r: r.served_model_name,
    )
    return {
        "v": 1,
        "warden_version": release_version(),
        "models": [model_facts(state, r) for r in models],
    }


@router.get("/engine", response_model=EngineFacts)
async def engine_facts(request: Request, key: TokenRow = Depends(require_bearer)) -> dict[str, Any]:
    """Static engine facts for the models this key may use."""
    if not key.engine_facts:
        raise HTTPException(403, NO_PERMISSION)
    return await engine_body(request.app.state, key)
