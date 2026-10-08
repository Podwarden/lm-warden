"""One pooled HTTP client per engine run (issue #279 stage 2).

The proxy built a new httpx.AsyncClient per request: no connection reuse, a
TCP (and pool) setup on every call. Clients are now keyed by
(model_id, supervisor generation, host, port): one engine run reuses one
keep-alive pool. The model id is part of the key because freed ports are
reused LIFO on 127.0.0.1 and generations are per model, so (host, port, gen)
alone can name two different engines. Lookup never closes a client of the
requested key's neighbours except one case: a NEWER generation of the same
model retires that model's older-generation clients (that engine process is
gone). A stale lookup (older generation than one already stored) just gets
its own client and never touches the current run's. Clients are also dropped
on model unload (``discard_model``) and at shutdown (``aclose_all``).

Timeouts stay unbounded (long generations are legitimate); the pool is
unbounded too, so admission stays the scheduler's job. The cookie jar accepts
no domains: the client is shared across tenants and must not replay an
engine's Set-Cookie on another caller's request.
"""

from __future__ import annotations

import asyncio
import logging
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_Key = tuple[str, int, str, int]  # (model_id, generation, host, port)


class UpstreamClients:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        # ``transport`` is a test seam (httpx.MockTransport); None in production.
        self._transport = transport
        self._clients: dict[_Key, httpx.AsyncClient] = {}
        self._closing: set[asyncio.Task[Any]] = set()

    def get(self, host: str, port: int, generation: int, *, model_id: str) -> httpx.AsyncClient:
        key = (model_id, generation, host, port)
        hit = self._clients.get(key)
        if hit is not None and not hit.is_closed:
            return hit
        # A newer run of this model supersedes its older ones.
        for k in [k for k in self._clients if k[0] == model_id and k[1] < generation]:
            self._close_later(self._clients.pop(k))
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(None),
            # The engine (uvicorn) closes an idle keep-alive connection after
            # 5 s; expiring ours at 2 s keeps the pool from handing out one
            # the engine is about to close (routes._forward retries once too).
            limits=httpx.Limits(
                max_connections=None, max_keepalive_connections=64, keepalive_expiry=2.0
            ),
            trust_env=False,
            cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
            transport=self._transport,
        )
        self._clients[key] = client
        return client

    def _close_later(self, client: httpx.AsyncClient) -> None:
        try:
            task = asyncio.get_running_loop().create_task(client.aclose())
        except RuntimeError:
            # No running loop (sync caller): nothing can await the close, so
            # leave it to GC rather than fail the lookup.
            logger.debug("upstream client not closed: no running loop")
            return
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def discard_model(self, model_id: str) -> None:
        for k in [k for k in self._clients if k[0] == model_id]:
            try:
                await self._clients.pop(k).aclose()
            except Exception:  # noqa: BLE001
                logger.debug("upstream close", exc_info=True)

    async def aclose_all(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        for c in clients:
            try:
                await c.aclose()
            except Exception:  # noqa: BLE001
                logger.debug("upstream close", exc_info=True)
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)
