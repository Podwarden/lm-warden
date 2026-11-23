"""Changing a key's KV isolation: the scope tags to retire, the push to the store, the audit.

Isolation itself works through the key's derived ``cache_salt`` and ``lmw_scope``
(``app.proxy.kv_scope``), for any store. Retiring a scope is bookkeeping on top: the tags
go into ``kv_retired_scopes`` with the key change (``TokenRepo.update_kv``, ``revoke``,
``delete``), and ``push_retired`` tells a running ``lmw_kv`` daemon at once, best effort.
With no daemon the push is skipped; the table is the record either way, and the re-sync
on connect sends it.

Scope tags, scopes and tenant ids are secrets of the key holder: never log one.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from collections.abc import Callable
from typing import Any

from fastapi import Request

from app.auth.admin_audit import STATE_KEY, route_template
from app.utils.client_ip import client_ip

log = logging.getLogger(__name__)

#: ``(kv_tenant_id, kv_epoch) -> scope tag`` (16 bytes).
TagFor = Callable[[str, int], bytes]


def tag_for(network_key: bytes) -> TagFor:
    """``scope_tag(isolation_scope(network_key, tenant, epoch))`` (``lmw_kv.scopes``)."""
    scopes = importlib.import_module("lmw_kv.scopes")

    def tag(tenant_id: str, epoch: int) -> bytes:
        out: bytes = scopes.scope_tag(scopes.isolation_scope(network_key, tenant_id, epoch))
        return out

    return tag


def tag_for_state(app_state: Any) -> TagFor | None:
    """``tag_for`` under the app's network key, or None when there is none (or no
    ``lmw_kv`` package): the change then records no retired tag (fail open)."""
    key_obj = getattr(app_state, "cobs2_key", None)
    nk = getattr(key_obj, "key", None) if key_obj is not None else None
    if not isinstance(nk, bytes):  # the proxy logs the missing key once
        return None
    try:
        return tag_for(nk)
    except ImportError:
        return None


async def push_retired(app_state: Any, tags: list[bytes]) -> None:
    """Send newly retired scope tags to the ``lmw_kv`` daemon, off the event loop.

    Best effort and fail open: no daemon, a daemon that is down, slow or refuses changes
    nothing for the request. The next re-sync from ``kv_retired_scopes`` covers a miss.
    """
    if not tags:
        return
    sup = getattr(app_state, "kv_store", None)
    client_of = getattr(sup, "client", None)
    if not callable(client_of):
        return
    try:
        client = client_of()
        if client is None:
            return
        await asyncio.to_thread(client.retire_scopes, list(tags))
    except Exception:  # noqa: BLE001 -- fail open; never the tags in the log
        log.info("kv: could not send retired scopes to the store; the next sync sends them")


def audit_change(request: Request, *, token_id: str, username: str, change: str) -> None:
    """One audit row for a KV change made by a session admin or a key holder: ``path`` is
    the route template followed by `` kv:<change>`` (isolated, sharing or reset).

    An admin-token request is already audited by the middleware, so it writes nothing.
    """
    if getattr(request.state, STATE_KEY, None) is not None:
        return
    writer = getattr(request.app.state, "admin_audit", None)
    if writer is None:
        return
    writer.record(
        ts=time.time(),
        token_id=token_id,
        username=username,
        method=request.method,
        path=f"{route_template(request.scope)} kv:{change}",
        status=200,
        duration_ms=0,
        client_ip=client_ip(request),
        peer_ip=request.client.host if request.client else None,
    )
