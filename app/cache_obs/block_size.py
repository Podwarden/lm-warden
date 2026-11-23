# app/cache_obs/block_size.py
"""The engine's KV block size B, per engine run (#301).

classify() needs B to tell "the engine served every whole block" from
"partial": a hybrid model on vLLM runs 784-token blocks and never caches the
trailing partial one. B comes from the engine's own metrics (vLLM
``cache_config_info``), learned by a background pass once per engine run; the
request path only ever reads the dictionary. Unknown B is None, and None keeps
the block-unaware rules.

The same pass records the prefix-match unit (``--prefix-match-unit``, default
the block) and whether the engine drops the last matched unit on every
hit (vLLM with EAGLE/MTP speculative decoding, confirmed in the v0.32.0 source:
app/runtime/backends/vllm/spec_decode.py). That comes from the launched args,
not the metrics, and is learned with B so both describe the same engine run.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import httpx

from app.runtime.backends import registry as backend_registry

logger = logging.getLogger(__name__)

#: How often the learner looks for engine runs whose B it does not know yet.
LEARN_INTERVAL_S: float = 10.0
#: The engine is local; a slow scrape is retried next pass, never waited on.
SCRAPE_TIMEOUT_S: float = 1.5

#: ``fetch(host, port, path) -> body or None``; may raise, the caller contains it.
Fetch = Callable[[str, int, str], Awaitable[str | None]]


class BlockSizes:
    """model_id -> (engine epoch, B, drops the last unit, match unit). An entry
    from another epoch (the engine was reloaded) reads as unknown, so a reload
    never inherits the old B."""

    def __init__(self) -> None:
        self._by_model: dict[str, tuple[str, int | None, bool, int | None]] = {}

    def known(self, model_id: str, epoch: str) -> bool:
        hit = self._by_model.get(model_id)
        return hit is not None and hit[0] == epoch

    def get(self, model_id: str, epoch: str) -> int | None:
        hit = self._by_model.get(model_id)
        return hit[1] if hit is not None and hit[0] == epoch else None

    def drops_last_block(self, model_id: str, epoch: str) -> bool:
        """Does this engine run drop the last matched block from every hit?
        Unknown reads False (the previous rules)."""
        hit = self._by_model.get(model_id)
        return hit[2] if hit is not None and hit[0] == epoch else False

    def match_unit(self, model_id: str, epoch: str) -> int | None:
        """Tokens per prefix-match step (``--prefix-match-unit``); the block
        when unset; None when B is unknown."""
        hit = self._by_model.get(model_id)
        return hit[3] if hit is not None and hit[0] == epoch else None

    def set(
        self,
        model_id: str,
        epoch: str,
        block: int | None,
        *,
        drops_last_block: bool = False,
        match_unit: int | None = None,
    ) -> None:
        unit = match_unit if match_unit and block else block
        self._by_model[model_id] = (epoch, block, drops_last_block, unit)


def _launched_args(state: Any, model: Any) -> list[str] | None:
    """The args the engine was launched with (the running variant; the row's
    when the supervisor did not launch it)."""
    from app.runtime.variants import running_variant

    try:
        args = running_variant(state, model).descriptor.get("extra_args")
    except Exception:  # noqa: BLE001 -- no variant: fall back to the row
        args = None
    if not isinstance(args, list):
        args = getattr(model, "extra_args", None)
    return args if isinstance(args, list) else None


def _engine_rules(
    state: Any, model: Any, backend: Any, block: int | None
) -> tuple[bool, int | None]:
    """(drops the last matched unit, match unit) from the launched args. A
    backend without the hooks (llama.cpp, MLX) keeps every whole block."""
    drop_hook = getattr(backend, "drops_last_cached_block", None)
    unit_hook = getattr(backend, "prefix_match_unit", None)
    if drop_hook is None and unit_hook is None:
        return False, block
    args = _launched_args(state, model)
    drops = bool(drop_hook(args)) if drop_hook is not None else False
    unit = int(unit_hook(args, block)) if unit_hook is not None and block else block
    return drops, unit


async def _default_fetch(host: str, port: int, path: str) -> str | None:
    async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT_S) as client:
        r = await client.get(f"http://{host}:{port}{path}")
    r.raise_for_status()
    return r.text


async def learn_once(state: Any, models: Iterable[Any], fetch: Fetch = _default_fetch) -> None:
    """One pass: scrape each loaded engine whose B is unknown for its current
    run. A failed scrape stays unknown and is retried next pass; a parsed body
    without the label is known None. Backends that do not advertise B in their
    metrics (llama.cpp) are never scraped."""
    from app.proxy.routes_dp import engine_epoch

    sizes: BlockSizes | None = getattr(state, "engine_block_sizes", None)
    sup = getattr(state, "supervisor", None)
    if sizes is None or sup is None:
        return
    for m in models:
        try:
            if m.status != "loaded":
                continue
            backend = backend_registry.get(getattr(m, "backend", None))
            caps = backend.capabilities
            if not caps.kv_block_size_in_metrics or caps.metrics_path is None:
                continue
            epoch = engine_epoch(state, m)
            port = sup.get_port(m.id)
            if port is None or sizes.known(m.id, epoch):
                continue
            body = await fetch(sup.get_host(m.id) or "127.0.0.1", port, caps.metrics_path)
            reading = backend.parse_metrics(body) if body else None
            if reading is None:
                continue
            block = int(reading.kv_block_size) if reading.kv_block_size else None
            drops, unit = _engine_rules(state, m, backend, block)
            sizes.set(m.id, epoch, block, drops_last_block=drops, match_unit=unit)
        except Exception:  # noqa: BLE001 -- one engine's failure must not stop the pass
            logger.debug("cache-obs: block size scrape failed for %s", m.id, exc_info=True)


async def run_block_size_learner(state: Any, interval: float = LEARN_INTERVAL_S) -> None:
    """Background loop over loaded models. VW_BLOCK_SIZE_LEARNER=0 turns it
    off (classification is then block-unaware); the test suite sets it so no
    background /metrics call lands inside a test's patched httpx."""
    from app.db.database import open_db
    from app.db.repos.models import ModelRepo

    if os.environ.get("VW_BLOCK_SIZE_LEARNER", "1").strip() == "0":
        return
    while True:
        try:
            async with open_db(state.settings.db_path) as db:
                models = await ModelRepo(db).list_all()
            await learn_once(state, models)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("cache-obs: block size learner pass failed", exc_info=True)
        await asyncio.sleep(interval)
