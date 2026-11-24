"""``GET /api/kv/usage``: the KV store's use per model and per key (admin only).

The figures come from the ``lmw_kv`` daemon and cover the time since it started
(``since``). With no daemon, or one that does not answer, ``available`` is False: the
page says the store is not answering and nothing fails. See ``app.kv.usage``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo
from app.kv import usage
from app.kv.schemas import KvUsage

router = APIRouter(prefix="/api/kv", tags=["kv"])


@router.get("/usage", response_model=KvUsage)
async def get_kv_usage(request: Request, _user: str = Depends(require_jwt)) -> dict[str, Any]:
    """The tiers (capacity, used, dead), then per model its weight, part and use, and per
    key inside it: use, fair share and evictions per tier, and who evicted its blocks.
    With a minor 2 store, ``reuse`` per model and per key (reuse, reloads, disk bytes
    written and read, the histogram of reloads at eviction, regret) and each model's
    ``top_reused`` (null: not available). Byte counts; keys no longer on the server are
    ``unknown``. ``pool_error`` and ``nvme_error`` say why the store keeps its RAM pool or
    its disk tier off (null when fine)."""
    stats = await usage.fetch_stats(request.app.state)
    nk = usage.network_key(request.app.state)
    names: dict[bytes, tuple[str, str]] = {}
    if stats is not None and nk is not None:
        async with open_db(request.app.state.settings.db_path) as db:
            rows = await TokenRepo(db).list_all()
        try:
            names = usage.tenant_names(rows, nk)
        except ImportError:
            names = {}
    top = await usage.fetch_tops(request.app.state, stats)
    out = usage.kv_usage(stats, names, top)
    out.update(usage.store_errors(stats, getattr(request.app.state, "kv_store", None)))
    return out
