"""GET /api/models/{model_id}/dp-routing -- replica routing counters joined with
the engine's own per-rank series (#286).

Read-only and cheap: one ModelRepo.get, one cached /metrics scrape, no DB
write. The scrape is bounded by a short timeout and can never break the
endpoint: when the engine is unloaded, down or slow the routing counters are
still returned and ``engine_metrics.error`` says why the engine fields are null.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.models.parallelism import spill_threshold_for
from app.runtime.backends import registry as backend_registry
from app.runtime.backends.vllm.metrics import RankReading, read_ranks

router = APIRouter(prefix="/api/models", tags=["models"])

#: Cache TTL: absorbs the card's 2 s poll from several tabs into one scrape.
RANK_SCRAPE_TTL_S: float = 1.5
#: The engine is local; a slow scrape surfaces as an error, never a stall.
SCRAPE_TIMEOUT_S: float = 1.5

#: ``fetch(host, port) -> (body, error)``; never needs to raise.
Fetch = Callable[[str, int], Awaitable[tuple[str | None, str | None]]]


async def _default_fetch(host: str, port: int) -> tuple[str | None, str | None]:
    url = f"http://{host}:{port}{backend_registry.get('vllm').capabilities.metrics_path}"
    try:
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT_S) as client:
            r = await client.get(url)
        r.raise_for_status()
        return r.text, None
    except Exception as exc:  # noqa: BLE001 -- any scrape failure is reported, not raised
        return None, str(exc) or exc.__class__.__name__


class RankScrapeCache:
    """Per-model TTL cache over one /metrics scrape. Lives on
    ``app.state.dp_rank_scrape_cache``; tests inject ``fetch`` and ``clock``."""

    def __init__(
        self,
        *,
        ttl: float = RANK_SCRAPE_TTL_S,
        clock: Callable[[], float] = time.monotonic,
        fetch: Fetch = _default_fetch,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._fetch = fetch
        # One lock per model: a slow engine's scrape must not stall the others.
        self._locks: dict[str, asyncio.Lock] = {}
        self._entries: dict[str, tuple[float, str, dict[int, RankReading], str | None]] = {}
        #: the engine run (variant + load generation) each entry was scraped from
        self._epochs: dict[str, str] = {}

    def recent_kv(
        self, model_id: str, max_age_s: float, epoch: str | None = None
    ) -> dict[int, float] | None:
        """KV-cache fill (0..1) per rank from the latest scrape, or None when
        there is none, it is older than ``max_age_s``, or (given ``epoch``) it
        was taken from a previous engine run. A dictionary read: the request
        path never scrapes."""
        hit = self._entries.get(model_id)
        if hit is None or self._clock() - hit[0] > max_age_s or hit[3] is not None:
            return None
        if epoch is not None and self._epochs.get(model_id) != epoch:
            return None
        kv = {
            r: reading.kv_cache_usage_perc
            for r, reading in hit[2].items()
            if reading.kv_cache_usage_perc is not None
        }
        return kv or None

    async def get(
        self, model_id: str, host: str, port: int, epoch: str = ""
    ) -> tuple[dict[int, RankReading], str | None, str]:
        """``(ranks, error, scraped_at_iso)``. A cached entry from another
        ``epoch`` (the engine was reloaded) is not reused."""
        lock = self._locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            now = self._clock()
            hit = self._entries.get(model_id)
            if (
                hit is not None
                and now - hit[0] < self._ttl
                and self._epochs.get(model_id, "") == epoch
            ):
                return hit[2], hit[3], hit[1]
            try:
                body, error = await self._fetch(host, port)
            except Exception as exc:  # noqa: BLE001
                body, error = None, str(exc) or exc.__class__.__name__
            ranks: dict[int, RankReading] = {}
            if body is not None:
                ranks = read_ranks(body)
                if not ranks:
                    error = "no per-replica series in /metrics"
            scraped_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            self._entries[model_id] = (now, scraped_at, ranks, error)
            self._epochs[model_id] = epoch
            return ranks, error, scraped_at


class DpRoutingTotals(BaseModel):
    in_flight: int
    placed: int
    sticky: int
    spilled: int
    client_pinned: int
    balanced: int
    unrouted: int


class DpRankRow(BaseModel):
    rank: int
    in_flight: int
    #: sessions placed on this replica with activity in the last 5 minutes
    assigned_sessions: int
    #: new sessions placed here (least-loaded at first sight)
    placed: int
    sticky: int
    spilled_in: int
    client_pinned: int
    requests_running: float | None
    requests_waiting: float | None
    kv_cache_usage_perc: float | None
    prefix_cache_hits: float | None
    prefix_cache_queries: float | None
    prefix_cache_hit_rate: float | None


class DpEngineMetrics(BaseModel):
    available: bool
    error: str | None
    scraped_at: str | None


class DpRoutingResponse(BaseModel):
    model_id: str
    data_parallel_size: int
    affinity_enabled: bool
    spill_threshold: int
    spill_threshold_source: Literal["setting", "auto"]
    since: str | None
    totals: DpRoutingTotals
    ranks: list[DpRankRow]
    engine_metrics: DpEngineMetrics


def engine_epoch(state: object, model: object) -> str:
    """One engine run: the running variant plus the supervisor's load counter.
    The same string scopes session placements and KV scrapes, so a reload
    cannot inherit the previous engine's placements or readings."""
    from app.runtime.variants import running_variant

    try:
        sup = getattr(state, "supervisor", None)
        gen = getattr(sup, "get_generation", None)
        variant = running_variant(state, model)
        return f"{variant.id}:{gen(model.id) if gen else 0}"  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return ""


def _hit_rate(r: RankReading) -> float | None:
    if not r.prefix_cache_queries or r.prefix_cache_hits is None:
        return None
    return round(r.prefix_cache_hits / r.prefix_cache_queries, 4)


@router.get("/{model_id}/dp-routing", response_model=DpRoutingResponse)
async def dp_routing(
    model_id: str, request: Request, _user: str = Depends(require_jwt)
) -> DpRoutingResponse:
    state = request.app.state
    async with open_db(state.settings.db_path) as db:
        model = await ModelRepo(db).get(model_id)
    if model is None:
        raise HTTPException(404, f"model '{model_id}' not found")

    dp = getattr(model, "data_parallel_size", 1) or 1
    threshold, source = spill_threshold_for(model.extra_args, model.dp_spill_threshold)
    epoch = engine_epoch(state, model)
    snap = state.dp_routing.snapshot(model_id, dp, epoch)

    ranks: dict[int, RankReading] = {}
    error: str | None = None
    scraped_at: str | None = None
    if model.status != "loaded":
        error = "model not loaded"
    else:
        sup = state.supervisor
        port = sup.get_port(model_id)
        if port is None:
            error = "model not running"
        else:
            cache = getattr(state, "dp_rank_scrape_cache", None)
            if cache is None:
                cache = state.dp_rank_scrape_cache = RankScrapeCache()
            host = sup.get_host(model_id) or "127.0.0.1"
            ranks, error, scraped_at = await cache.get(model_id, host, port, epoch)

    rows: list[DpRankRow] = []
    for entry in snap["ranks"]:
        reading = ranks.get(entry["rank"]) or RankReading()
        rows.append(
            DpRankRow(
                **entry,
                requests_running=reading.requests_running,
                requests_waiting=reading.requests_waiting,
                kv_cache_usage_perc=reading.kv_cache_usage_perc,
                prefix_cache_hits=reading.prefix_cache_hits,
                prefix_cache_queries=reading.prefix_cache_queries,
                prefix_cache_hit_rate=_hit_rate(reading),
            )
        )
    return DpRoutingResponse(
        model_id=model_id,
        data_parallel_size=dp,
        affinity_enabled=bool(model.dp_affinity_enabled),
        spill_threshold=threshold,
        spill_threshold_source="setting" if source == "setting" else "auto",
        since=snap["since"],
        totals=DpRoutingTotals(**snap["totals"]),
        ranks=rows,
        engine_metrics=DpEngineMetrics(
            available=bool(ranks) and error is None, error=error, scraped_at=scraped_at
        ),
    )


#: How often the background scraper refreshes the KV readings placement uses.
RANK_SCRAPER_INTERVAL_S: float = 5.0


async def run_rank_scraper(state: object, interval: float = RANK_SCRAPER_INTERVAL_S) -> None:
    """Keep ``dp_rank_scrape_cache`` warm for loaded data-parallel models, so a
    new session can be placed by KV fill without ever scraping on the request
    path. Log-and-continue; a failure only means placement uses in-flight alone."""
    import logging
    import os

    log = logging.getLogger(__name__)
    # VW_DP_RANK_SCRAPER=0 turns the loop off (placement then uses in-flight
    # counts alone); the test suite sets it so no background /metrics call can
    # land inside a test's patched httpx.
    if os.environ.get("VW_DP_RANK_SCRAPER", "1").strip() == "0":
        return
    while True:
        try:
            cache = getattr(state, "dp_rank_scrape_cache", None)
            sup = getattr(state, "supervisor", None)
            if cache is not None and sup is not None:
                async with open_db(state.settings.db_path) as db:  # type: ignore[attr-defined]
                    models = await ModelRepo(db).list_all()
                for m in models:
                    if m.status != "loaded" or (getattr(m, "data_parallel_size", 1) or 1) <= 1:
                        continue
                    port = sup.get_port(m.id)
                    if port is not None:
                        await cache.get(
                            m.id, sup.get_host(m.id) or "127.0.0.1", port, engine_epoch(state, m)
                        )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.debug("dp rank scraper pass failed", exc_info=True)
        await asyncio.sleep(interval)
