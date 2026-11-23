"""KV-cache tiers in the proxy: the engine's ``kv_transfer_params`` and per-turn tier hits.

The engine's ``LmwKVConnector`` (lmw_kv) reads ``kv_transfer_params`` from each request:
``lmw_req`` tags the hits it records in the store daemon, ``lmw_scope`` (hex) salts the
cache keys of an isolated key (empty for a sharing key), ``lmw_tenant`` (hex) names the key
to the daemon for accounting (``app.proxy.kv_scope``), ``lmw_conv`` / ``lmw_session`` are
informational. For a model with ``kv_store`` on, a client's own ``kv_transfer_params`` are
removed: vLLM itself acts on fields such as ``do_remote_prefill``, and nothing a client sends
may steer the connector. With ``kv_store`` off they pass through untouched: the engine's
connector, if any, is one the operator configured (NIXL, disaggregated prefill) and needs them.

After the turn, the api pops the hits the connector recorded under ``lmw_req``
(``pop_hits``: ``{}`` when nothing was recorded, None when the call failed) and puts them
on the history row. The daemon is an optimisation: a missing,
slow or failing daemon leaves the tier columns NULL, never delays or fails the request.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from app.proxy.kv_scope import isolation_cost
from app.proxy.request_registry import LiveRequest, finished_record

if TYPE_CHECKING:
    from lmw_kv.client import Client

logger = logging.getLogger(__name__)

KV_PARAMS = "kv_transfer_params"

#: Pending tier lookups (a reference, so a running task is never garbage collected).
_pending: set[asyncio.Task[None]] = set()


def strip_client_kv_params(body_json: dict[str, Any]) -> bool:
    """Remove a client's ``kv_transfer_params``; True when there were any."""
    if isinstance(body_json, dict) and KV_PARAMS in body_json:
        del body_json[KV_PARAMS]
        return True
    return False


def set_kv_params(
    body_json: dict[str, Any],
    *,
    req_id: str,
    conv: str | None,
    session: str | None,
    tenant: str = "",
    scope: str = "",
) -> None:
    """LM Warden's own ``kv_transfer_params``; only for a model with ``kv_store`` on.

    ``tenant`` and ``scope`` are hex (``""`` for an unattributed request and for a sharing
    key's scope). Both are secrets of the key holder: never log the body this returns.
    """
    body_json[KV_PARAMS] = {
        "lmw_req": req_id,
        "lmw_tenant": tenant,
        "lmw_scope": scope,
        "lmw_conv": conv,
        "lmw_session": session,
    }


def fetch_hits(client: Client | None, req_id: str) -> dict[str, Any] | None:
    """The hits recorded for ``req_id`` (popped), or None. Blocking: run off the loop."""
    if client is None:
        return None
    try:
        hits = client.pop_hits(req_id)
    except Exception:  # noqa: BLE001 -- the daemon is an optimisation
        logger.debug("kv: pop_hits failed", exc_info=True)
        return None
    return hits if isinstance(hits, dict) else None


def _count(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def apply_hits(req: LiveRequest, hits: dict[str, Any] | None) -> None:
    """Set the four tier fields from ``hits`` (rulings R16a, R23, R24).

    Each prefix block is counted once. vLLM's ``cached_tokens`` is local plus external
    (connector-loaded) tokens, so the GPU share is ``max(cached - ram - nvme, 0)``.
    ``{}`` (the daemon answered, nothing recorded) is a GPU-only turn: ram = nvme = 0,
    ``load_s`` None. None (the call failed) or a malformed dict leaves all four None.
    """
    if not isinstance(hits, dict):
        return
    if not hits:
        req.served_gpu_tokens = req.cached_tokens
        req.served_ram_tokens = 0
        req.served_nvme_tokens = 0
        return
    ram, nvme, load = (
        _count(hits.get("ram_tokens")),
        _count(hits.get("nvme_tokens")),
        hits.get("load_s"),
    )
    if ram is None or nvme is None or not isinstance(load, int | float) or isinstance(load, bool):
        return
    cached = req.cached_tokens
    req.served_gpu_tokens = max(cached - ram - nvme, 0) if cached is not None else None
    req.served_ram_tokens = ram
    req.served_nvme_tokens = nvme
    req.load_s = round(float(load), 4)


def kv_client(app_state: Any, model: Any) -> Client | None:
    """The daemon client for a model with ``kv_store`` on while the daemon is healthy.

    None too while the model's engine runs with its tier off (loaded before the daemon was
    up, or the daemon restarted since): its turns then keep the tier columns NULL instead
    of being recorded as GPU-only.
    """
    from app.runtime.kv_store import kv_enabled

    if not kv_enabled(model):
        return None
    tier_of = getattr(getattr(app_state, "supervisor", None), "kv_tier", None)
    if callable(tier_of):
        try:
            tier = tier_of(model.id)
        except Exception:  # noqa: BLE001
            tier = None
        if isinstance(tier, dict) and tier.get("state") == "off_until_reload":
            return None
    sup = getattr(app_state, "kv_store", None)
    try:
        return sup.client() if sup is not None else None
    except Exception:  # noqa: BLE001
        return None


def apply_iso_cost(req: LiveRequest, star_frac: float | None) -> None:
    """``kv_iso_cost_tokens`` of an isolated request, once its tier hits are known."""
    if not req.kv_isolated:
        req.kv_iso_cost_tokens = None
        return
    try:
        req.kv_iso_cost_tokens = isolation_cost(
            star_frac,
            req.prompt_tokens,
            req.served_gpu_tokens,
            req.served_ram_tokens,
            req.served_nvme_tokens,
        )
    except Exception:  # noqa: BLE001 -- one column, never the row
        req.kv_iso_cost_tokens = None


async def _record_with_hits(
    history: Any, req: LiveRequest, client: Client, now: float, star_frac: float | None
) -> None:
    try:
        apply_hits(req, await asyncio.to_thread(fetch_hits, client, req.id))
    except Exception:  # noqa: BLE001 -- bookkeeping must never lose the row
        logger.debug("kv: tier hits failed", exc_info=True)
    apply_iso_cost(req, star_frac)
    try:
        history.record(finished_record(req, now=now))
    except Exception:  # noqa: BLE001
        logger.debug("stats: could not record finished request", exc_info=True)


def record_finished(
    history: Any,
    req: LiveRequest,
    client: Client | None,
    *,
    now: float,
    iso_star_frac: float | None = None,
) -> None:
    """Record ``req``'s history row; with a KV ``client``, after its tier hits arrive.

    The hits are read in a worker thread by a background task, so the response and the
    admission slot never wait on the daemon. ``now`` is the monotonic end of the request.
    ``iso_star_frac`` is the operator lens's matched share of an isolated request's prompt.
    """
    if client is None:
        apply_iso_cost(req, iso_star_frac)
        history.record(finished_record(req, now=now))
        return
    try:
        task = asyncio.get_running_loop().create_task(
            _record_with_hits(history, req, client, now, iso_star_frac)
        )
    except RuntimeError:  # no running loop: record without the tiers
        apply_iso_cost(req, iso_star_frac)
        history.record(finished_record(req, now=now))
        return
    _pending.add(task)
    task.add_done_callback(_pending.discard)
