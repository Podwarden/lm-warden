"""Model serving state for /healthz, /readyz and /api/admin/readiness (#342).

lmwarden.com, 2026-11-22: for ~3.5 minutes after a deploy /healthz answered
``{"ok": true}`` while every completion 404'd "model not loaded" -- the model
the watchdog was restoring was still loading, and a client that waited on
/healthz sent traffic into a warden that could not serve yet.

* ``/healthz`` is LIVENESS. It stays 200 in every model state: a model load
  takes minutes, and failing liveness meanwhile gets the api restarted mid-load
  (the reload never finishes). It gains ``models: {serving, counts}``, with no
  network call, and a failure to read it never fails the probe.
* ``/readyz`` is READINESS: 200 only when at least one model passes the
  router's own check (``app/proxy/routes.py::_resolve_loaded``: status
  ``loaded`` and the supervisor holds a port) AND its engine answers its health
  path right now. 503 otherwise. The body is ``{ready, reason}`` only.
* ``/api/admin/readiness`` (admin credential) is the same check with the
  per-model detail: id, served name, state, since, engine answer, last error.

Both public routes take no credential, so they never name a model, say when
it changed, or carry exception text (review of !685).

LOAD. Every caller shares one result for ``SHARED_TTL_S``: concurrent callers
await one in-flight check, and a models-table write (the router cache's
generation) starts a new one. The models come from the router's cache when it
is warm, else one plain read (see ``_warm_rows``), so a hot /healthz opens
SQLite at most once a second. Engine probes go through one pooled httpx client.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.auth.deps import require_jwt
from app.proxy.model_cache import MISS, all_models, generation
from app.runtime.watchdog import engine_health_path, probe_health, wants_restart

logger = logging.getLogger(__name__)

#: How long one computed result serves every caller.
SHARED_TTL_S = 1.0

#: Engine health probe timeout for /readyz. Well under a typical readiness
#: probe timeout; an engine that cannot answer /health in this long is not one
#: to route a request to.
ENGINE_PROBE_TIMEOUT_S = 2.0

#: The models read must never hold up liveness.
MODELS_READ_TIMEOUT_S = 2.0

#: Reported states. ``unloaded`` models are counted, never listed.
STATES = ("loading", "ready", "failed", "unloading", "unloaded")


class _Shared:
    """One value for every caller: reused for SHARED_TTL_S while the models
    generation is unchanged, and computed by one task however many callers
    arrive meanwhile. The task is shielded, so a caller that disconnects does
    not cancel it for the others."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._value: Any = None
        self._at = 0.0
        self._gen: int | None = None
        self._task: asyncio.Task[Any] | None = None

    async def get(self, compute: Callable[[], Awaitable[Any]]) -> Any:
        gen = generation()
        if self._gen is not None and self._gen == gen and self._clock() - self._at < SHARED_TTL_S:
            return self._value
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._run(compute, gen))
        return await asyncio.shield(self._task)

    async def _run(self, compute: Callable[[], Awaitable[Any]], gen: int) -> Any:
        value = await compute()
        self._value, self._at, self._gen = value, self._clock(), gen
        return value


@dataclass
class ModelView:
    id: str
    name: str
    state: str
    since: str | None
    error: str | None
    port: int | None = None
    host: str | None = None
    health_path: str = "/health"
    engine: bool | None = None  # set by the readiness check for ready models


@dataclass
class Snapshot:
    error: str | None = None  # admin-only text; public bodies carry a code
    models: list[ModelView] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        c = dict.fromkeys(STATES, 0)
        for m in self.models:
            c[m.state] += 1
        return c


def _will_restore(row: Any, app_state: Any) -> bool:
    """A failed row the watchdog will bring back on its own.

    Same predicate the restore paths use (``wants_restart``, which also honours
    a held stress lease), and only while one of those paths can still run: the
    restart sweep when probing is enabled, or the boot restore before it ran.
    The live config is what the watchdog loop last published; before its first
    tick, the env default."""
    if not wants_restart(row, leases=getattr(app_state, "stress_leases", None)):
        return False
    cfg, boot_done = watchdog_view(app_state)
    if cfg.get("enabled"):
        return True
    return bool(cfg.get("restore_on_boot")) and not boot_done


def watchdog_view(app_state: Any) -> tuple[dict[str, Any], bool]:
    """(live watchdog config, boot restore done) as the watchdog loop last
    published them on app.state; before its first tick, the env default."""
    settings = getattr(app_state, "settings", None)
    cfg = getattr(app_state, "watchdog_cfg", None) or {
        "enabled": getattr(settings, "watchdog_enabled", True),
        "restore_on_boot": True,
    }
    return cfg, bool(getattr(app_state, "watchdog_boot_restore_done", False))


def model_state(row: Any, app_state: Any) -> str:
    """One model's serving state, from its row and the supervisor."""
    status = row.status
    if status == "loaded":
        # Same predicate the router applies before forwarding a request.
        return "ready" if app_state.supervisor.get_port(row.id) is not None else "failed"
    if status == "loading":
        return "loading"
    if status == "failed":
        return "loading" if _will_restore(row, app_state) else "failed"
    if status == "unloading":
        return "unloading"
    return "unloaded"  # registered, pulling, pulled: not meant to be serving


def _since(row: Any) -> str | None:
    # SQLite datetime('now') is UTC "YYYY-MM-DD HH:MM:SS".
    ts = getattr(row, "updated_at", None)
    if not isinstance(ts, str) or not ts:
        return None
    return ts.replace(" ", "T") + ("" if ts.endswith("Z") or "+" in ts else "Z")


def _warm_rows(cache: Any) -> list[Any] | None:
    """The router cache's rows when it is warm, else None.

    A cold cache is read past, not filled. Filling would be race-safe (the
    cache keys a fill on the generation taken before the read), but the unit
    suite boots every app with GET /healthz and then seeds models with direct
    sqlite writes, which never bump the generation: a probe fill hid those
    rows from the router for the cache TTL (290 failures). The 1 s shared
    result already bounds the probes' DB reads to one a second."""
    if cache is None:
        return None
    try:
        hit = cache.rows()
    except Exception:  # noqa: BLE001 -- fail open to a plain read
        return None
    return None if hit is MISS else list(hit)


class ReadinessChecker:
    """Per-app shared state: the snapshot, the readiness result, the client."""

    def __init__(self, app_state: Any) -> None:
        self._state = app_state
        self._snapshot = _Shared()
        self._ready = _Shared()
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=ENGINE_PROBE_TIMEOUT_S)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _read(self) -> Snapshot:
        st = self._state
        try:
            # asyncio.timeout, not wait_for: see app/db/database.py on wait_for
            # swallowing a cancel.
            rows = _warm_rows(getattr(st, "model_cache", None))
            if rows is None:
                async with asyncio.timeout(MODELS_READ_TIMEOUT_S):
                    rows = await all_models(None, st.settings.db_path)
            sup = st.supervisor
            models = []
            for row in rows:
                state = model_state(row, st)
                if state == "unloaded":
                    models.append(ModelView(row.id, row.served_model_name, state, None, None))
                    continue
                view = ModelView(
                    row.id,
                    row.served_model_name,
                    state,
                    _since(row),
                    row.last_error if state == "failed" else None,
                )
                if state == "ready":
                    view.port = sup.get_port(row.id)
                    view.host = sup.get_host(row.id) or "127.0.0.1"
                    view.health_path = engine_health_path(row)
                models.append(view)
            return Snapshot(models=models)
        except Exception as exc:  # noqa: BLE001 -- liveness must not fail over this read
            logger.warning("readiness: model state unavailable: %s", exc)
            return Snapshot(error=f"{type(exc).__name__}: {exc}")

    async def snapshot(self) -> Snapshot:
        return cast("Snapshot", await self._snapshot.get(self._read))

    async def _check(self) -> Snapshot:
        snap = await self.snapshot()
        ready = [m for m in snap.models if m.state == "ready" and m.port is not None]
        client = self._http()

        async def _probe(m: ModelView) -> ModelView:
            ok, _detail = await probe_health(
                m.host or "127.0.0.1", m.port or 0, m.health_path, client=client
            )
            return ModelView(**{**m.__dict__, "engine": ok})

        probed = {m.id: m for m in await asyncio.gather(*(_probe(m) for m in ready))}
        return Snapshot(error=snap.error, models=[probed.get(m.id, m) for m in snap.models])

    async def readiness(self) -> Snapshot:
        return cast("Snapshot", await self._ready.get(self._check))


def checker(app_state: Any) -> ReadinessChecker:
    c = getattr(app_state, "readiness", None)
    if not isinstance(c, ReadinessChecker):
        c = app_state.readiness = ReadinessChecker(app_state)
    return c


def verdict(snap: Snapshot, model: str | None = None) -> tuple[bool, str | None]:
    """(ready, reason code) over every model, or over one by id or served name."""
    if snap.error is not None:
        return False, "unavailable"
    models = snap.models
    if model is not None:
        models = [m for m in models if model in (m.id, m.name)]
    if any(m.engine for m in models):
        return True, None
    states = {m.state for m in models}
    if "loading" in states:
        return False, "loading"
    if "ready" in states:
        return False, "engine_unhealthy"
    if "failed" in states:
        return False, "failed"
    if "unloading" in states:
        return False, "unloading"
    return False, "no_models_configured"


async def models_health(app_state: Any) -> dict[str, Any]:
    """The public /healthz ``models`` block. Never raises, never probes an engine."""
    try:
        snap = await checker(app_state).snapshot()
    except Exception:  # noqa: BLE001 -- liveness must not fail over this
        logger.warning("healthz: model state unavailable", exc_info=True)
        return {"serving": None, "counts": None}
    if snap.error is not None:
        return {"serving": None, "counts": None}
    counts = snap.counts()
    return {"serving": counts["ready"] > 0, "counts": counts}


async def public_readiness(app_state: Any) -> tuple[bool, dict[str, Any]]:
    """The public /readyz body: a verdict and a reason code, nothing else."""
    try:
        ready, reason = verdict(await checker(app_state).readiness())
    except Exception:  # noqa: BLE001
        logger.warning("readyz: check failed", exc_info=True)
        ready, reason = False, "unavailable"
    return ready, {"ready": ready, "reason": reason}


router = APIRouter(tags=["system"])


@router.get(
    "/api/admin/readiness",
    responses={
        503: {"description": "No model (or not the named one) can serve right now"},
        404: {"description": "No such model"},
    },
)
async def admin_readiness(
    request: Request,
    model: str | None = Query(None, description="A model id or served name"),
    _user: str = Depends(require_jwt),
) -> JSONResponse:
    """/readyz with the per-model detail, for an admin."""
    try:
        return await _admin_readiness(request, model)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 -- a clean 503, never a 500
        logger.warning("admin readiness: check failed", exc_info=True)
        return JSONResponse(
            {
                "ready": False,
                "reason": "unavailable",
                "models": {"serving": None, "counts": None, "items": []},
                "error": f"{type(exc).__name__}: {exc}",
            },
            status_code=503,
        )


async def _admin_readiness(request: Request, model: str | None) -> JSONResponse:
    snap = await checker(request.app.state).readiness()
    if model is not None and snap.error is None:
        if not any(model in (m.id, m.name) for m in snap.models):
            raise HTTPException(404, f"model '{model}' not found")
    ready, reason = verdict(snap, model)
    items = [
        {
            "id": m.id,
            "name": m.name,
            "state": m.state,
            "since": m.since,
            **({"error": m.error} if m.error else {}),
            **({"engine": "ok" if m.engine else "unhealthy"} if m.engine is not None else {}),
        }
        for m in snap.models
        if m.state != "unloaded" and (model is None or model in (m.id, m.name))
    ]
    counts = snap.counts() if snap.error is None else None
    body: dict[str, Any] = {
        "ready": ready,
        "reason": reason,
        "models": {
            "serving": None if counts is None else counts["ready"] > 0,
            "counts": counts,
            "items": items,
        },
    }
    if snap.error is not None:
        body["error"] = snap.error
    return JSONResponse(body, status_code=200 if ready else 503)
