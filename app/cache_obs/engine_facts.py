"""Static engine facts for one model, for the key-scoped ``GET /api/me/engine``.

What an app developer needs to shape a prompt for the engine's prefix cache: the block size
(reuse is in whole blocks), the match unit (``--prefix-match-unit``), whether the last matched
block is dropped (speculative decoding), and what the model is allowed to hold. Nothing here
depends on traffic, so it is safe for any key that may use the model; it carries no traffic
figure, no other key and no host name.

Never a guess (owner decision 2026-11-26): a value the warden cannot determine right now
(the model is not loaded, the engine has not reported it, a backend that does not say) is
``None`` and its field is named in ``unknown``. A classic 16-token page is reported as 16:
the dashboard's ``BLOCK_AWARE_ABOVE`` filter (app/cache_obs/classify.py) is not applied here.
"""

from __future__ import annotations

from typing import Any

from app.runtime.backends import registry as backend_registry

#: Fields that describe the running engine, so they are unknown for a model that is not loaded.
_ENGINE_RUN = (
    "prefix_caching",
    "engine_block_tokens",
    "engine_match_unit_tokens",
    "engine_drops_last_block",
    "speculative",
)
_FIELDS = (
    "model",
    "backend",
    "running",
    *_ENGINE_RUN,
    "max_model_len",
    "replicas",
    "cache_aware_routing",
)


def _engine_run(state: Any, model: Any, backend: Any) -> dict[str, Any]:
    """The engine-run fields of a loaded model; an absent key is unknown."""
    from app.cache_obs.block_size import _launched_args
    from app.proxy.routes_dp import engine_epoch

    out: dict[str, Any] = {}
    sizes = getattr(state, "engine_block_sizes", None)
    if sizes is not None:
        epoch = engine_epoch(state, model)
        if sizes.known(model.id, epoch):
            caching = sizes.prefix_caching(model.id, epoch)
            if caching is not None:
                out["prefix_caching"] = caching
            block = sizes.get(model.id, epoch)
            if block is not None:
                out["engine_block_tokens"] = block
                out["engine_match_unit_tokens"] = sizes.match_unit(model.id, epoch) or block
                out["engine_drops_last_block"] = sizes.drops_last_block(model.id, epoch)
    hook = getattr(backend, "speculative_method", None)
    if hook is not None:
        args = _launched_args(state, model)
        if args is not None:
            out["speculative"] = hook(args)  # None: speculative decoding is off
    return out


def model_facts(state: Any, model: Any) -> dict[str, Any]:
    """One entry of the answer: the fields of the module docstring, ``unknown`` listing the
    ones that are null because the warden cannot say."""
    running = model.status == "loaded"
    try:
        backend = backend_registry.get(getattr(model, "backend", None))
    except Exception:  # noqa: BLE001 -- an unregistered backend: nothing engine-side is known
        backend = None
    known = _engine_run(state, model, backend) if running and backend is not None else {}
    replicas = int(getattr(model, "data_parallel_size", 1) or 1)
    entry: dict[str, Any] = {
        "model": model.served_model_name,
        "backend": getattr(model, "backend", None),
        "running": running,
        "prefix_caching": known.get("prefix_caching"),
        "engine_block_tokens": known.get("engine_block_tokens"),
        "engine_match_unit_tokens": known.get("engine_match_unit_tokens"),
        "engine_drops_last_block": known.get("engine_drops_last_block"),
        "speculative": known.get("speculative"),
        "max_model_len": getattr(model, "max_model_len", None),
        "replicas": replicas,
        # With one replica there is nothing to route between.
        "cache_aware_routing": replicas > 1 and bool(getattr(model, "dp_affinity_enabled", 1)),
    }
    unknown = [f for f in _ENGINE_RUN if f not in known]
    if entry["max_model_len"] is None:
        unknown.append("max_model_len")
    if entry["backend"] is None:
        unknown.append("backend")
    entry["unknown"] = [f for f in _FIELDS if f in unknown]
    return entry
