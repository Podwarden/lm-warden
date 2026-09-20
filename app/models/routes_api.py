from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from app.auth.deps import (
    TRUST_REMOTE_CODE_LOAD_MESSAGE,
    TRUST_REMOTE_CODE_SET_MESSAGE,
    refuse_session_only,
    require_jwt,
    stream_key,
)
from app.auth.stream_guard import guard_stream
from app.db.constants import ACTIVE_STATUSES
from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.runtime import RuntimeRepo
from app.db.repos.setup import SetupRepo
from app.db.repos.stack_attempts import StackAttemptRepo, StackAttemptRow
from app.models.discovery import (
    DiscoveryAuthRequired,
    DiscoveryNotFound,
    discover_repo_files,
    load_hf_token,
)
from app.models.fit import (
    classify_fit,
    dtype_bytes_from_torch_dtype,
    kv_reserve_bytes,
    recommend_max_model_len,
    weights_budget_bytes,
)
from app.models.gpu_capability import capability_warnings
from app.models.load_preflight import decide_preflight
from app.models.pull_task import _snapshot_dir_size, run_pull
from app.models.schemas import (
    ModelCreate,
    ModelList,
    ModelOut,
    TemplateCreate,
    TryStackRequest,
    TryStackResult,
)
from app.models.serialisation import model_detail, model_summary
from app.models.sharding import shard_glob_for
from app.models.stack_classifier import classify
from app.models.suggest import suggest_config
from app.runtime.backends import registry
from app.runtime.backends.paths import ModelFileNotFound, resolve_model_paths
from app.runtime.backends.registry import DEFAULT_BACKEND
from app.runtime.backends.vllm.images import UnsupportedChannelError, resolve_image
from app.runtime.engine.run_marker import tail_since_last_run
from app.runtime.supervisor import UnloadRefused, wait_for_health
from app.runtime.variants import ModelVariantRepo, Variant, mark_recorded
from app.runtime.warmup_probe import warmup_probe
from app.templates import store as template_store
from app.templates.registry import EngineSpec, template_to_dict
from app.utils.sse import sse_headers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models")

# #256 (final-review finding, v2026.09.19.3): trust_remote_code runs the
# target Hugging Face repo's Python inside the warden container -- at
# REGISTER time nothing runs yet (it's just a row), but a model loaded with
# it runs arbitrary code with the warden's own privileges, able to read
# VW_JWT_SECRET, mint a session, or edit api_tokens / admin_audit directly.
# That would make the session-only fence around admin-token management
# (docs/superpowers/specs/2026-09-19-admin-tokens-design.md) an API-level
# guardrail only, not real containment. Every refusal in this module is
# session-only -- an admin token is refused, a signed-in session is not -- and
# reuses app/auth/deps.py::refuse_session_only for the "is this a session?"
# check and the 403 shape; the two messages live there because the settings
# PATCH refuses with the same words.
#
# Three refusals here, not one: the body key, the merged template value, and a
# template create. The first pass at #256 guarded only the literal body key,
# which a template_id walked straight past -- and a builtin template already
# carries the flag (app/templates/registry.py).

# In-process cache for discovery results. 60 s TTL mirrors the rationale in
# ``app.system.routes_gpus._ProbeCache``: short enough that an operator
# re-running Discover after fixing a token sees fresh data, long enough that
# the Add Model modal can refetch on selection changes without thrashing HF.
DISCOVERY_CACHE_TTL_SECONDS = 60.0


class _DiscoveryCache:
    """``(repo_id, revision)``-keyed in-process cache for discovery results.

    Same shape as ``_ProbeCache`` in ``app/system/routes_gpus.py`` — single
    asyncio.Lock guards both the cache read and the underlying fetch, so
    concurrent requests inside the TTL window collapse to one HfApi
    invocation per ``(repo_id, revision)`` pair. Errors are not cached so a
    transient HF outage doesn't poison the cache for 60 s.
    """

    def __init__(
        self,
        *,
        ttl: float = DISCOVERY_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._lock = asyncio.Lock()
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        self.invocations = 0  # exposed for tests

    async def get_or_fetch(
        self,
        key: tuple[str, str],
        fetch: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        async with self._lock:
            now = self._clock()
            # Self-bounding: prune entries older than 5*TTL on every access
            # so a long-lived process polling many distinct repo_ids doesn't
            # accumulate stale tuples forever. No separate sweeper task
            # needed — fresh accesses are the only thing that grows the dict.
            stale_cutoff = 5 * self._ttl
            self._cache = {
                k: v for k, v in self._cache.items() if (now - v[0]) <= stale_cutoff
            }
            entry = self._cache.get(key)
            if entry is not None and (now - entry[0]) < self._ttl:
                return entry[1]
            self.invocations += 1
            value = await fetch()
            self._cache[key] = (now, value)
            return value


def _get_discovery_cache(request: Request) -> _DiscoveryCache:
    cache = getattr(request.app.state, "discovery_cache", None)
    if cache is None:
        cache = _DiscoveryCache()
        request.app.state.discovery_cache = cache
    return cache


def _gen_id() -> str:
    return secrets.token_hex(8)


@router.get("/templates")
async def list_model_templates(request: Request, _user: str = Depends(require_jwt)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        return [template_to_dict(t) for t in await template_store.list_templates(db)]


@router.post("/templates", status_code=201)
async def create_template(
    body: TemplateCreate, request: Request, _user: str = Depends(require_jwt)
):
    from app.templates.registry import ModelTemplate

    # #256 / I1 -- refuse before any write. A template is a register-time
    # prefill, so a saved trust_remote_code template is a stored instruction to
    # persist the flag on every model made from it; ``create_model`` refuses
    # the merged value as well, but a token must not be able to leave the
    # instruction lying around for a session to trip over either.
    if body.trust_remote_code:
        refuse_session_only(request, message=TRUST_REMOTE_CODE_SET_MESSAGE)

    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        # #170: the try-stack "save working combo" flow sends a minimal body
        # (id/label/hf_repo/engine) plus the live ``model_id`` whose combo was
        # just validated. Source the model's ACTUAL tuning from that row so the
        # saved template reproduces what worked — an AWQ single-GPU combo keeps
        # its --enforce-eager + gpu_memory_utilization=0.92 instead of silently
        # falling back to the schema defaults (extra_args=[], gpu_mem=0.9) and
        # OOMing on re-instantiation. Explicit body fields still win over the
        # live row (mirrors create_model's body>row precedence).
        live = (
            await ModelRepo(db).get(body.model_id) if body.model_id else None
        )
        extra_args = (
            list(body.extra_args) if body.extra_args
            else (list(live.extra_args) if live else [])
        )
        gpu_mem = (
            body.gpu_memory_utilization
            if "gpu_memory_utilization" in body.model_fields_set
            else (live.gpu_memory_utilization if live else body.gpu_memory_utilization)
        )

        t = ModelTemplate(
            id=body.id,
            label=body.label,
            hf_repo=body.hf_repo,
            hf_revision=body.hf_revision,
            dtype=body.dtype,
            max_model_len=body.max_model_len,
            tensor_parallel_size=body.tensor_parallel_size,
            gpu_memory_utilization=gpu_mem,
            trust_remote_code=body.trust_remote_code,
            extra_args=extra_args,
            extra_env=dict(body.extra_env),
            engine=None
            if body.engine is None
            else EngineSpec(
                channel=body.engine.channel,
                vllm_version=body.engine.vllm_version,
                image=body.engine.image,
            ),
            source="user",
        )
        await template_store.save_user_template(db, t)
    return {"id": body.id, "source": "user"}


@router.delete("/templates/{template_id}", status_code=204)
async def delete_template(
    template_id: str, request: Request, _user: str = Depends(require_jwt)
):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        try:
            await template_store.delete_user_template(db, template_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    return None


@router.get("/discover")
async def discover_model_repo(
    request: Request,
    repo_id: str,
    revision: str = "main",
    _user: str = Depends(require_jwt),
):
    """List files + config.json summary for an HF Hub model repo (#84).

    Stage 1 of the two-stage Add Model wizard. Returns a stable wire shape
    that ``app.models.discovery.DiscoveryResult`` defines; the FE selects a
    file from ``files`` and posts a normal ``POST /api/models`` after.

    ``config`` is a fixed projection of ``config.json`` — ``hidden_size``,
    ``num_hidden_layers``, ``num_attention_heads``, ``num_key_value_heads``,
    ``max_position_embeddings``, ``torch_dtype`` (the set #82 commits to),
    ``quantization_config`` (#176), and the KV-shape keys ``head_dim``,
    ``sliding_window`` and ``layer_types``. Anything else from ``config.json``
    stays opaque so the VRAM-fit math has a stable contract.

    Gated repos silently reuse ``data_dir/hf-token`` (CTO decision in #84) —
    we do not prompt the user for a second confirmation. Failures map to
    typed envelopes:

    - HF 401/403 → HTTP 401 ``{error_code: "auth_required", ...}``
    - HF 404    → HTTP 404 ``{error_code: "repo_not_found", ...}``
    - other     → HTTP 502 with a sanitized message
    """
    settings = request.app.state.settings
    cache = _get_discovery_cache(request)

    async def _fetch() -> dict[str, Any]:
        token = await load_hf_token(settings)
        result = await discover_repo_files(repo_id, revision, token)
        return result.to_dict()

    try:
        return await cache.get_or_fetch((repo_id, revision), _fetch)
    except DiscoveryAuthRequired as e:
        # Gated/private repo and the operator's stored token (if any) isn't
        # sufficient. The FE will surface "auth required" UI; the operator
        # then updates the token in Settings — no inline prompt.
        raise HTTPException(
            401,
            detail={
                "error_code": "auth_required",
                "message": "HuggingFace Hub requires authentication for this repo "
                "(gated or private). Update the HF token in Settings.",
                "repo_id": repo_id,
                "revision": revision,
            },
        ) from e
    except DiscoveryNotFound as e:
        raise HTTPException(
            404,
            detail={
                "error_code": "repo_not_found",
                "message": f"HuggingFace repo '{repo_id}' (revision '{revision}') not found.",
                "repo_id": repo_id,
                "revision": revision,
            },
        ) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("discover_repo_files failed for %s@%s", repo_id, revision)
        # Sanitised message — never leak the raw HF stack trace to the FE.
        raise HTTPException(
            502,
            detail={
                "error_code": "discovery_failed",
                "message": "HuggingFace Hub request failed; check connectivity and try again.",
                "repo_id": repo_id,
                "revision": revision,
            },
        ) from e


def _effective_file_size(filename: str, files: list[dict[str, Any]]) -> int:
    """Return the byte count we classify against for ``filename``.

    For single-file weights (single safetensors, GGUF, pytorch_model.bin)
    that's the file's own size. For a safetensors shard we sum every shard
    that matches the shard glob — see ``shard_glob_for``. ``files`` is the
    discovery payload's file list.
    """
    glob = shard_glob_for(filename)
    if glob is None:
        for f in files:
            if f["filename"] == filename:
                return int(f["size"] or 0)
        return 0
    total = 0
    for f in files:
        if fnmatch.fnmatch(f["filename"], glob):
            total += int(f["size"] or 0)
    return total


# Weights kinds a repo can offer, in the order we would pick one when the
# caller names no file. safetensors first because that is what vLLM loads by
# default; a shard is a fine stand-in for its whole set, because
# ``_effective_file_size`` re-aggregates the glob anyway.
_WEIGHTS_KIND_PREFERENCE = ("safetensors_single", "safetensors_sharded", "pytorch_bin")


def _default_weights_filename(files: list[dict[str, Any]]) -> str | None:
    """The one file a fit verdict is obviously about, or None if ambiguous.

    A normal HF repo carries exactly one weights format, so asking the caller
    to name a file is asking them to repeat what discovery already knows --
    which is why ``POST /api/models`` makes ``filename`` optional. GGUF repos
    are the exception: they publish many INDEPENDENT quantisations side by
    side (Q4_K_M, Q8_0, ...) that fit very differently, so picking one for the
    caller would answer a question they did not ask. Those still have to name
    a file, and the 422 below says so.
    """
    for kind in _WEIGHTS_KIND_PREFERENCE:
        candidates = [f for f in files if f.get("kind") == kind]
        if candidates:
            # Deterministic: the first shard by name, not "whichever the Hub
            # listed first", so the same repo always yields the same verdict.
            return str(min(candidates, key=lambda f: f["filename"])["filename"])
    return None


class FitPreviewRequest(BaseModel):
    """Ask whether a candidate model fits on the selected GPUs.

    ``hf_repo`` and ``filename`` deliberately match ``POST /api/models``:
    these two endpoints describe the same model moments apart in the Add
    Model flow, and they used to disagree on both -- ``repo_id`` here vs
    ``hf_repo`` there, and ``filename`` required here vs optional there. A
    caller who wrote the create body and then tried the pre-check got a bare
    ``Field required`` naming a field they had never heard of.

    ``repo_id`` remains accepted for the existing frontend and any client
    written against the old shape; exactly one of the two must be given.
    """

    # The create endpoint's spelling, and the one the docs use.
    hf_repo: str | None = Field(default=None, min_length=1)
    # The original spelling. Kept working; not documented for new callers.
    repo_id: str | None = Field(default=None, min_length=1)
    revision: str = "main"
    # Optional, as on POST /api/models. Omit it and the repo's weights file is
    # resolved from discovery; a GGUF repo has no single answer, so naming one
    # is required there (see _default_weights_filename).
    filename: str | None = Field(default=None, min_length=1)
    gpu_indices: list[int] = Field(..., min_length=1)
    max_batch_size: int = Field(default=1, ge=1, le=64)
    gpu_memory_utilization: float = Field(0.9, gt=0, le=1.0)
    # Which engine the verdict is about. Defaults to vLLM so an older client
    # that omits it gets exactly the behaviour it got before. `gpu_memory_
    # utilization` is IGNORED for a backend that declares its own
    # `vram_cap_fraction`, because for that backend the operator's number
    # corresponds to no engine flag -- see BackendCapabilities.
    backend: str = DEFAULT_BACKEND
    # Allow the FE to override max_model_len for the math; when None we fall
    # back to ``config.max_position_embeddings`` (the natural ceiling).
    max_model_len: int | None = Field(default=None, gt=0)

    @property
    def repo(self) -> str:
        """The repo id, from whichever spelling the caller used."""
        return self.hf_repo or self.repo_id or ""

    @model_validator(mode="after")
    def _exactly_one_repo_field(self) -> FitPreviewRequest:
        if self.hf_repo and self.repo_id and self.hf_repo != self.repo_id:
            raise ValueError(
                "give either hf_repo or repo_id, not both with different values"
            )
        if not self.repo:
            raise ValueError("hf_repo is required")
        return self


class FitPreviewBreakdown(BaseModel):
    total_vram: int
    weights_budget: int
    kv_reserve: int
    file_size: int
    ratio: float
    dtype_bytes: int
    max_model_len_used: int


class FitPreviewResponse(BaseModel):
    verdict: Literal["green", "yellow", "orange", "red"]
    breakdown: FitPreviewBreakdown
    recommended_max_model_len: int | None = None
    warnings: list[str] = Field(default_factory=list)


_MIB = 1024 * 1024


@router.post("/fit-preview", response_model=FitPreviewResponse)
async def fit_preview(
    body: FitPreviewRequest,
    request: Request,
    _user: str = Depends(require_jwt),
) -> FitPreviewResponse:
    """Predict whether a candidate weights file will fit on the selected GPUs.

    Calls the shipped ``discover_repo_files`` to grab file sizes + ``config.json``,
    sums VRAM across the selected GPUs from the live ``/api/system/gpus``
    probe, then runs the math in ``app/models/fit.py``. The verdict +
    breakdown drive colour-coded rows in the Add Model wizard (dev-2, #86).

    The shard-aggregation rule (``_effective_file_size``) keeps multi-part
    safetensors honest: pick any shard and we classify against the whole
    shard set, not the 4 GB slice.
    """
    settings = request.app.state.settings
    cache = _get_discovery_cache(request)
    warnings: list[str] = []
    # One name for the repo from here down, whichever spelling arrived.
    repo = body.repo

    async def _fetch() -> dict[str, Any]:
        token = await load_hf_token(settings)
        result = await discover_repo_files(repo, body.revision, token)
        return result.to_dict()

    try:
        discovery = await cache.get_or_fetch((repo, body.revision), _fetch)
    except DiscoveryAuthRequired as e:
        raise HTTPException(
            401,
            detail={
                "error_code": "auth_required",
                "message": "HuggingFace Hub requires authentication for this repo "
                "(gated or private). Update the HF token in Settings.",
                "repo_id": repo,
                "revision": body.revision,
            },
        ) from e
    except DiscoveryNotFound as e:
        raise HTTPException(
            404,
            detail={
                "error_code": "repo_not_found",
                "message": f"HuggingFace repo '{repo}' (revision '{body.revision}') not found.",
                "repo_id": repo,
                "revision": body.revision,
            },
        ) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("fit-preview discovery failed for %s@%s",
                         repo, body.revision)
        raise HTTPException(
            502,
            detail={
                "error_code": "discovery_failed",
                "message": "HuggingFace Hub request failed; check connectivity and try again.",
                "repo_id": repo,
                "revision": body.revision,
            },
        ) from e

    files = discovery["files"]
    # `filename` is optional, matching POST /api/models. Resolve it from
    # discovery when the caller left it out; a GGUF repo publishes several
    # independent quantisations, so there the caller must still choose.
    filename = body.filename or _default_weights_filename(files)
    if filename is None:
        raise HTTPException(
            422,
            detail={
                "error_code": "filename_required",
                "message": (
                    f"repo '{repo}' offers no single weights file to size — "
                    "name one in `filename`. GGUF repos publish several "
                    "independent quantisations that fit very differently, so "
                    "there is no safe default."
                ),
                "repo_id": repo,
                "revision": body.revision,
                "candidates": sorted(
                    f["filename"] for f in files
                    if f.get("kind") in ("gguf", "safetensors_single",
                                         "safetensors_sharded", "pytorch_bin")
                ),
            },
        )
    file_size = _effective_file_size(filename, files)
    if file_size == 0:
        # Either the filename wasn't in the repo or all matched siblings had
        # null size metadata. Either way the verdict isn't meaningful — fail
        # explicit rather than return a misleading green row.
        raise HTTPException(
            422,
            detail={
                "error_code": "filename_not_found",
                "message": f"filename '{filename}' not found in repo "
                f"'{repo}' (or had no size metadata).",
                "repo_id": repo,
                "revision": body.revision,
                "filename": filename,
            },
        )

    config = discovery.get("config") or {}
    # Best-effort defaults when config.json is missing or incomplete. We log
    # a warning so the FE can surface that the math is degraded rather than
    # silently returning a "green" verdict that hides missing inputs.
    hidden_size = int(config.get("hidden_size") or 0)
    num_hidden_layers = int(config.get("num_hidden_layers") or 0)
    num_attention_heads = int(config.get("num_attention_heads") or 0)
    num_kv_heads = int(config.get("num_key_value_heads") or num_attention_heads or 0)
    max_pos_emb = int(config.get("max_position_embeddings") or 0)
    torch_dtype = config.get("torch_dtype")
    dtype_bytes = dtype_bytes_from_torch_dtype(torch_dtype)

    # KV shape. `head_dim` is authoritative when declared; kv_reserve_bytes
    # derives it only when it is absent.
    declared_head_dim = config.get("head_dim")
    head_dim = int(declared_head_dim) if declared_head_dim else None

    # Interleaved sliding-window attention, counted from the EXPLICIT
    # per-layer declaration only. A model that implies sliding attention some
    # other way keeps the all-full-attention estimate, which over-counts --
    # the safe direction.
    layer_types = config.get("layer_types")
    sliding_layers = 0
    if isinstance(layer_types, list):
        sliding_layers = sum(1 for t in layer_types if t == "sliding_attention")
    raw_window = config.get("sliding_window")
    sliding_window = int(raw_window) if raw_window else None
    if sliding_layers and not sliding_window:
        # Layers say they slide but the window is missing: charge them the
        # full context rather than inventing a width.
        sliding_layers = 0

    config_complete = bool(
        hidden_size and num_hidden_layers and num_attention_heads and num_kv_heads
    )
    if not config_complete:
        warnings.append(
            "config_incomplete: missing one of hidden_size, num_hidden_layers, "
            "num_attention_heads, num_key_value_heads — KV math degraded to zero "
            "reserve, verdict may be optimistic"
        )

    # GPU VRAM is reported by nvidia-smi in MiB; multiply to bytes for the
    # math. Re-use the existing live probe cache (same one used by
    # ``/api/system/gpus``) so we don't double-shell nvidia-smi.
    cache_gpu = getattr(request.app.state, "gpu_probe_cache", None)
    if cache_gpu is None:
        from app.system.routes_gpus import _ProbeCache  # noqa: PLC0415
        cache_gpu = _ProbeCache()
        request.app.state.gpu_probe_cache = cache_gpu
    snap = await cache_gpu.get()
    vram_by_index = {g.index: g.memory_total_mib * _MIB for g in snap.gpus}
    missing = [i for i in body.gpu_indices if i not in vram_by_index]
    if missing:
        raise HTTPException(
            422,
            detail={
                "error_code": "gpu_index_missing",
                "message": f"gpu_indices {missing} not present in nvidia-smi probe",
                "available": sorted(vram_by_index.keys()),
                "probe_error": snap.probe_error,
            },
        )
    total_vram = sum(vram_by_index[i] for i in body.gpu_indices)

    # max_model_len falls back to config.max_position_embeddings when the
    # caller didn't override it. Both can be zero in degraded scenarios; in
    # that case KV reserve is also zero and the verdict reflects only
    # weight footprint.
    max_model_len_used = body.max_model_len or max_pos_emb or 0

    kv_reserve = 0
    if config_complete and max_model_len_used > 0:
        kv_reserve = kv_reserve_bytes(
            hidden_size=hidden_size,
            num_layers=num_hidden_layers,
            num_kv_heads=num_kv_heads,
            num_attention_heads=num_attention_heads,
            max_model_len=max_model_len_used,
            dtype_bytes=dtype_bytes,
            max_batch_size=body.max_batch_size,
            head_dim=head_dim,
            sliding_layers=sliding_layers,
            sliding_window=sliding_window,
        )

    # Backend-aware VRAM cap. A backend that declares a `vram_cap_fraction`
    # has no operator-set utilisation flag, so its declared value wins;
    # otherwise the operator's `gpu_memory_utilization` is a real engine cap
    # and is honoured. `registry.get` raises UnknownBackendError rather than
    # falling back, which is the same contract Supervisor.load relies on.
    try:
        caps = registry.get(body.backend).capabilities
    except Exception:
        raise HTTPException(
            422,
            detail={
                "error_code": "unknown_backend",
                "message": f"unknown backend {body.backend!r}",
            },
        ) from None
    cap_fraction = (
        caps.vram_cap_fraction
        if caps.vram_cap_fraction is not None
        else body.gpu_memory_utilization
    )

    budget = weights_budget_bytes(total_vram, cap_fraction, kv_reserve)
    verdict = classify_fit(file_size, budget)
    # Float-zero-denominator safe: classify_fit returns "red" when budget<=0.
    ratio = (file_size / budget) if budget > 0 else float("inf")

    rec: int | None = None
    if verdict in ("orange", "red") and config_complete:
        rec = recommend_max_model_len(
            hidden_size=hidden_size,
            num_layers=num_hidden_layers,
            num_kv_heads=num_kv_heads,
            num_attention_heads=num_attention_heads,
            total_vram=total_vram,
            gpu_memory_utilization=body.gpu_memory_utilization,
            file_size=file_size,
            dtype_bytes=dtype_bytes,
            max_batch_size=body.max_batch_size,
        )
        # Cap the recommendation at the model's own max_position_embeddings —
        # there's no point recommending a context longer than the trained
        # one even if VRAM allows it. ``or None`` keeps the field absent
        # when max_pos_emb is missing.
        if rec is not None and max_pos_emb > 0:
            rec = min(rec, max_pos_emb)

    is_gguf = filename.lower().endswith(".gguf")
    if is_gguf:
        warnings.append(
            "gguf_dequant_peak: actual VRAM during weight load may spike to "
            "~2x file size briefly before settling"
        )

    # --- Capability check (#176) ------------------------------------------
    # Cross the WEAKEST selected GPU's compute capability with the candidate's
    # quant/dtype and append human-readable warnings (emulated FP8, unsupported
    # bf16/fp4, …) to the SAME warnings list. Warn, never block.
    #
    # The min across the tensor-parallel group gates what the whole group can
    # run — one Turing card in a pair of Amperes still blocks bf16. None values
    # (driver didn't report compute_cap) are skipped; if every selected GPU is
    # None the min is None and capability_warnings() returns [].
    cap_by_index = {g.index: g.compute_cap for g in snap.gpus}
    selected_caps: list[float] = [
        cap
        for i in body.gpu_indices
        if (cap := cap_by_index.get(i)) is not None
    ]
    min_compute_cap = min(selected_caps) if selected_caps else None

    quant_method: str | None = None
    quant_cfg = config.get("quantization_config")
    if isinstance(quant_cfg, dict):
        quant_method = str(quant_cfg.get("quant_method") or "").lower() or None

    # GGUF is dequantized to fp16/bf16 at load, so for capability purposes it
    # behaves like a bf16 build regardless of its on-disk INT quant suffix —
    # classify it as bf16-compute (and ignore any stray config quant_method).
    cap_torch_dtype = "bfloat16" if is_gguf else torch_dtype
    cap_quant_method = None if is_gguf else quant_method
    warnings.extend(capability_warnings(
        min_compute_cap,
        torch_dtype=cap_torch_dtype,
        quant_method=cap_quant_method,
    ))

    return FitPreviewResponse(
        verdict=verdict,
        breakdown=FitPreviewBreakdown(
            total_vram=total_vram,
            weights_budget=budget,
            kv_reserve=kv_reserve,
            file_size=file_size,
            # JSON cannot encode inf; map the budget-overflow case to a
            # large-but-finite sentinel that the FE can detect.
            ratio=(ratio if ratio != float("inf") else 1e9),
            dtype_bytes=dtype_bytes,
            max_model_len_used=max_model_len_used,
        ),
        recommended_max_model_len=rec,
        warnings=warnings,
    )


@router.post("", status_code=201)
async def create_model(
    body: ModelCreate, request: Request, _user: str = Depends(require_jwt)
):
    # #256 -- refuse before any write, and before any DB work at all for the
    # obvious case. ModelCreate.trust_remote_code defaults to False, so a True
    # value here came from the body explicitly setting it. The EFFECTIVE value
    # after the template merge is refused further down (I1): that is the check
    # that actually matters, this one just answers the common case without
    # opening the database.
    if body.trust_remote_code:
        refuse_session_only(request, message=TRUST_REMOTE_CODE_SET_MESSAGE)
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        setup_state = await SetupRepo(db).get()
        allowed = set(setup_state.draft.get("allowed_gpu_indices", []))
        if not set(body.gpu_indices).issubset(allowed):
            bad = sorted(set(body.gpu_indices) - allowed)
            raise HTTPException(
                400, f"GPU indices {bad} not in allowed_gpu_indices {sorted(allowed)}"
            )

        repo = ModelRepo(db)
        if await repo.get_by_served_name(body.served_model_name):
            raise HTTPException(409, f"served_model_name '{body.served_model_name}' already exists")

        # Merge template (if any) → effective config. Explicit body fields win.
        tpl = None
        if body.template_id:
            tpl = await template_store.get_template(db, body.template_id)
            if tpl is None:
                raise HTTPException(404, f"template '{body.template_id}' not found")

        hf_repo = body.hf_repo or (tpl.hf_repo if tpl else None)
        if not hf_repo:
            raise HTTPException(400, "hf_repo is required (directly or via template_id)")
        hf_revision = (
            body.hf_revision if body.hf_repo
            else (tpl.hf_revision if tpl else body.hf_revision)
        )
        dtype = body.dtype if body.dtype is not None else (tpl.dtype if tpl else None)
        max_model_len = (
            body.max_model_len if body.max_model_len is not None
            else (tpl.max_model_len if tpl else None)
        )
        gpu_mem = (
            body.gpu_memory_utilization
            if "gpu_memory_utilization" in body.model_fields_set
            else (tpl.gpu_memory_utilization if tpl else body.gpu_memory_utilization)
        )
        trust = (
            body.trust_remote_code
            if "trust_remote_code" in body.model_fields_set
            else (tpl.trust_remote_code if tpl else body.trust_remote_code)
        )
        # #256 / I1 -- the refusal that counts is on the MERGED value, because
        # the row is what a later load reads. Without this a template_id was a
        # one-request bypass of the body-key check above, and no template create
        # was even needed: the builtin `gpt-oss-20b` carries the flag. Still
        # before any write -- everything above this point is a read.
        if trust:
            refuse_session_only(request, message=TRUST_REMOTE_CODE_SET_MESSAGE)
        extra_args = (
            list(body.extra_args) if body.extra_args
            else (list(tpl.extra_args) if tpl else [])
        )
        extra_env = (
            dict(body.extra_env) if body.extra_env
            else (dict(tpl.extra_env) if tpl else {})
        )

        # Backend axis: explicit body > template > the registry default.
        # Templates have no backend field in sub-project B, so getattr keeps
        # this a pass-through until one gains it.
        backend = body.backend or getattr(tpl, "backend", None) or DEFAULT_BACKEND

        # Engine axis: explicit body > template.engine > None (legacy path).
        eng_channel = body.engine_channel or (
            tpl.engine.channel if tpl and tpl.engine else None
        )
        eng_version = body.engine_vllm_version or (
            tpl.engine.vllm_version if tpl and tpl.engine else None
        )
        eng_image = body.engine_image or (
            tpl.engine.image if tpl and tpl.engine else None
        )
        if eng_channel and eng_version:
            try:
                eng_image = resolve_image(eng_channel, eng_version, image=eng_image)
            except UnsupportedChannelError as e:
                raise HTTPException(400, str(e)) from e

        model_id = _gen_id()
        await repo.insert(ModelRow(
            id=model_id,
            served_model_name=body.served_model_name,
            hf_repo=hf_repo,
            hf_revision=hf_revision,
            gpu_indices=sorted(body.gpu_indices),
            tensor_parallel_size=body.tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem,
            trust_remote_code=trust,
            extra_args=extra_args,
            extra_env=extra_env,
            status="registered",
            pulled_bytes=0,
            pulled_total=None,
            last_error=None,
            filename=body.filename,
            parallelism_strategy=body.parallelism_strategy,
            max_batch_size=body.max_batch_size,
            hf_config_repo=body.hf_config_repo,
            tokenizer_repo=body.tokenizer_repo,
            engine_channel=eng_channel,
            engine_vllm_version=eng_version,
            engine_image=eng_image,
            # Sub-project B. Templates carry no backend field yet, so this is a
            # straight pass-through of the body's default; it is merged here
            # anyway, with the same "explicit body wins" rule as hf_repo, so
            # sub-project C does not have to find this site again.
            backend=backend,
            # Sub-project C (migration 0028). llama.cpp only; templates have no
            # field for either, so both are straight body pass-throughs.
            mmproj_filename=body.mmproj_filename,
            n_gpu_layers=body.n_gpu_layers,
        ))
    return {"id": model_id, "served_model_name": body.served_model_name, "status": "registered"}


@router.get("", response_model=ModelList)
async def list_models(request: Request, _user: str = Depends(require_jwt)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()
    # Both read paths serialise through app/models/serialisation.py. They used
    # to hand-build their dicts, which is how migrations 0027/0028 could persist
    # `backend`, `mmproj_filename` and `n_gpu_layers` correctly while both
    # responses dropped all three -- surfacing to the operator as
    # "Engine: vllm" on a model that was running llama-server.
    return {"models": [model_summary(r) for r in rows]}


@router.get("/{model_id}", response_model=ModelOut)
async def get_model(model_id: str, request: Request, _user: str = Depends(require_jwt)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        row = await ModelRepo(db).get(model_id)
    if not row:
        raise HTTPException(404, "not found")
    return model_detail(row)


@router.post("/{model_id}/try-stack", status_code=201)
async def try_stack(
    model_id: str,
    body: TryStackRequest,
    request: Request,
    _user: str = Depends(require_jwt),
):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = ModelRepo(db)
        if not await repo.get(model_id):
            raise HTTPException(404, "not found")
        try:
            image = resolve_image(body.channel, body.vllm_version, image=body.image)
        except UnsupportedChannelError as e:
            raise HTTPException(400, str(e)) from e
        await db.execute(
            "UPDATE models SET engine_channel=?, engine_vllm_version=?, "
            "engine_image=?, updated_at=datetime('now') WHERE id=?",
            (body.channel, body.vllm_version, image, model_id),
        )
        await db.commit()
        attempt_id = _gen_id()
        await StackAttemptRepo(db).insert(StackAttemptRow(
            id=attempt_id,
            model_id=model_id,
            channel=body.channel,
            vllm_version=body.vllm_version,
            image=image,
            result="pending",
            error=None,
            category=None,
            suggested_next=None,
        ))
    return {"attempt_id": attempt_id, "image": image}


@router.get("/{model_id}/try-stack")
async def list_try_stack(
    model_id: str, request: Request, _user: str = Depends(require_jwt)
):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        rows = await StackAttemptRepo(db).list_for_model(model_id)
    return {"attempts": [
        {
            "id": r.id,
            "channel": r.channel,
            "vllm_version": r.vllm_version,
            "image": r.image,
            "result": r.result,
            "error": r.error,
            "category": r.category,
            "suggested_next": r.suggested_next,
            "created_at": r.created_at,
        }
        for r in rows
    ]}


@router.post("/{model_id}/try-stack/{attempt_id}")
async def record_try_stack_result(
    model_id: str,
    attempt_id: str,
    body: TryStackResult,
    request: Request,
    _user: str = Depends(require_jwt),
):
    settings = request.app.state.settings
    category = None
    suggestion = None
    if body.result == "failed":
        c = classify(body.error or "")
        category = c.category
        suggestion = c.suggestion
    async with open_db(settings.db_path) as db:
        repo = StackAttemptRepo(db)
        if not await repo.get(attempt_id):
            raise HTTPException(404, "attempt not found")
        suggested_next = {"suggestion": suggestion} if suggestion else None
        await repo.set_result(
            attempt_id, body.result, body.error, category, suggested_next
        )
    return {"result": body.result, "category": category, "suggestion": suggestion}


@router.get("/{model_id}/suggest-config")
async def suggest_model_config(
    model_id: str,
    request: Request,
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Return a suggested starting-point configuration for ``model_id``.

    Heuristics live in :func:`app.models.suggest.suggest_config` — see
    that module for the full rationale. The endpoint deliberately reads
    from the live discovery cache (config.json + file list) and the
    live GPU probe; both are best-effort and degrade gracefully when
    nvidia-smi is unavailable or the HF Hub is offline (the heuristic
    falls back to ``None`` for any field it can't suggest).

    NEVER auto-applied. The response carries an explicit ``disclaimer``
    field so a downstream consumer that auto-applies the values ignores
    a contract red flag (#113).
    """
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        row = await ModelRepo(db).get(model_id)
    if not row:
        raise HTTPException(404, "not found")

    # Discovery — uses the same cache the fit-preview route built up so
    # repeated visits during a wizard session don't re-hit HF.
    cache = _get_discovery_cache(request)

    async def _fetch() -> dict[str, Any]:
        token = await load_hf_token(settings)
        result = await discover_repo_files(row.hf_repo, row.hf_revision, token)
        return result.to_dict()

    discovery: dict[str, Any] = {}
    try:
        discovery = await cache.get_or_fetch((row.hf_repo, row.hf_revision), _fetch)
    except (DiscoveryAuthRequired, DiscoveryNotFound):
        # Degrade gracefully — return a suggestion with whatever we know
        # (gmu only). The wizard surfaces the discovery failure
        # separately via the existing discovery endpoint.
        discovery = {"files": [], "config": None}
    except Exception:  # noqa: BLE001
        logger.warning(
            "suggest-config discovery failed for %s@%s — degrading to "
            "config-less suggestion",
            row.hf_repo, row.hf_revision,
        )
        discovery = {"files": [], "config": None}

    config = discovery.get("config")
    filenames = [f["filename"] for f in discovery.get("files") or []]

    # Live VRAM probe — same cache the fit-preview route uses so we
    # share a single nvidia-smi shell-out per probe window. When the
    # probe is unavailable we fall through with ``total_vram_bytes=0``;
    # the heuristic tolerates that (gmu is independent of total VRAM).
    cache_gpu = getattr(request.app.state, "gpu_probe_cache", None)
    if cache_gpu is None:
        from app.system.routes_gpus import _ProbeCache  # noqa: PLC0415
        cache_gpu = _ProbeCache()
        request.app.state.gpu_probe_cache = cache_gpu
    snap = await cache_gpu.get()
    vram_by_index = {g.index: g.memory_total_mib * _MIB for g in snap.gpus}
    total_vram_bytes = sum(
        vram_by_index.get(i, 0) for i in (row.gpu_indices or [])
    )

    suggestion = suggest_config(
        hf_repo=row.hf_repo,
        config=config,
        total_vram_bytes=total_vram_bytes,
        filenames=filenames,
    )
    return suggestion.to_dict()


class EffectiveArgvResponse(BaseModel):
    """Envelope for ``GET /api/models/{id}/effective-argv``."""

    argv: list[str] = Field(
        ...,
        description=(
            "The exact argv that the supervisor would invoke for this row at "
            "next load, INCLUDING argv[0] and its subcommand (for the vLLM "
            "backend, ``vllm serve``). Includes overrides resolved from "
            "extra_args, parallelism_strategy, GGUF quant tag, etc."
        ),
    )


# Placeholder port used purely so the backend produces a stable argv for
# display. The real port comes from PortAllocator at load time; surfacing
# 10000 here is a tell to the operator that the argv is for preview only.
_EFFECTIVE_ARGV_PREVIEW_PORT = 10000


@router.get("/{model_id}/effective-argv", response_model=EffectiveArgvResponse)
async def get_effective_argv(
    model_id: str,
    request: Request,
    _user: str = Depends(require_jwt),
) -> EffectiveArgvResponse:
    """Return the argv that ``vllm serve`` would be invoked with for this model.

    Asks the row's backend for a :class:`~app.runtime.backends.LaunchPlan`
    against the persisted row exactly as the supervisor would at load time,
    so the returned argv includes argv[0] and its subcommand, using a
    fixed placeholder port (the real port comes from the supervisor's
    PortAllocator at load time and would jitter on every refresh).
    Includes ``extra_args``, parallelism flag, GGUF quant tag, tokenizer
    override, etc. — i.e. every transformation that turns the curated
    columns + opaque ``extra_args`` list into a final argv.

    This is read-only and side-effect-free; the FE shows it in a
    faux-terminal panel so the operator can verify what the next
    ``Load`` will actually run before clicking. Refreshed by the FE on
    every settings change so the diff highlight shows which args moved.
    """
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        row = await ModelRepo(db).get(model_id)
    if not row:
        raise HTTPException(404, "not found")
    # #211 — pass the configured driver so the preview keeps its promise of
    # showing "exactly as the supervisor would at load time". The bind host is
    # driver-dependent now (loopback for local, 0.0.0.0 for docker), so a
    # preview that omitted this would render --host 127.0.0.1 on a docker
    # deployment and disagree with what actually launches. Display-only.
    backend = registry.get(getattr(row, "backend", None))
    driver_name = getattr(settings, "engine_driver", "local")
    # The preview shows the argv this row would ACTUALLY launch with, which for
    # llama.cpp includes the resolved -m path. If the file is not pulled yet,
    # say so rather than 500-ing: the preview is a diagnostic and an operator
    # will reach for it precisely when a row will not load.
    # Same capability gate as Supervisor.load, and for the same reason: a vLLM
    # preview must not scan the model cache. This route is a read-only
    # diagnostic, so it would be an especially poor place to acquire a
    # filesystem failure mode.
    resolved = None
    if backend.capabilities.needs_local_model_path:
        try:
            resolved = resolve_model_paths(row, hf_cache_dir=settings.hf_cache_dir)
        except ModelFileNotFound as exc:
            raise HTTPException(409, str(exc)) from None
    plan = backend.plan(
        row,
        port=_EFFECTIVE_ARGV_PREVIEW_PORT,
        bind_host=backend.bind_host(driver_name),
        resolved=resolved,
        engine_driver=driver_name,
    )
    return EffectiveArgvResponse(argv=plan.argv)


@router.post("/{model_id}/pull", status_code=202)
async def trigger_pull(
    model_id: str,
    request: Request,
    force: bool = False,
    _user: str = Depends(require_jwt),
):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        row = await ModelRepo(db).get(model_id)
        if not row:
            raise HTTPException(404, "not found")
        if row.status not in ("registered", "failed", "pulled"):
            raise HTTPException(409, f"cannot pull from status '{row.status}'")
    asyncio.create_task(run_pull(model_id, settings, force=force))
    return {"status": "pulling", "force": force}


@router.get("/{model_id}/pull/progress")
async def stream_pull_progress(
    model_id: str, request: Request, _user: str = Depends(require_jwt)
):
    """Server-Sent Events stream of pull progress.

    Emits one JSON event per second with {status, bytes, total, last_error}.
    Terminates when the row leaves 'pulling'/'registered' (sends a final
    event so the FE can react), or when the client disconnects.
    """
    settings = request.app.state.settings
    key = stream_key(request)

    async def gen():
        # Registered for logout / revoke, and an admin token re-checked while
        # the stream is open (app/auth/stream_guard.py).
        async with guard_stream(request, key):
            while True:
                if await request.is_disconnected():
                    return
                async with open_db(settings.db_path) as db:
                    row = await ModelRepo(db).get(model_id)
                if not row:
                    yield 'data: {"status": "missing"}\n\n'
                    return
                payload = json.dumps({
                    "status": row.status,
                    "bytes": row.pulled_bytes,
                    "total": row.pulled_total,
                    "last_error": row.last_error,
                })
                yield f"data: {payload}\n\n"
                if row.status not in ("pulling", "registered"):
                    return
                await asyncio.sleep(1.0)

    # Anti-buffering headers (X-Accel-Buffering, Cache-Control) — see
    # app.utils.sse.sse_headers and #50. Highest-UX-impact SSE endpoint:
    # this drives the live progress bar during the first model setup,
    # which is the operator's first impression of the app.
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=sse_headers(),
    )


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: str, request: Request, _user: str = Depends(require_jwt)
):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        row = await ModelRepo(db).get(model_id)
        if not row:
            raise HTTPException(404, "not found")
        if row.status in ACTIVE_STATUSES:
            raise HTTPException(409, f"cannot delete model in status '{row.status}'")
        await ModelRepo(db).delete(model_id)
    return Response(status_code=204)


def _read_local_config(hf_cache_dir, repo: str) -> dict[str, Any]:
    """Read ``config.json`` from the LOCAL HF snapshot for ``repo``.

    LOAD-PATH RULE: no network. The model is already pulled, so its
    ``config.json`` is on disk. HF stores it under
    ``{hf_cache_dir}/models--{org}--{name}/snapshots/<rev>/config.json``
    (and possibly under a ``hub/`` parent — HF layout drift, mirrored from
    ``app/cache/scanner.py``). We pick the most-recently-modified snapshot
    that has a ``config.json`` (there is normally exactly one). Returns ``{}``
    on any miss / parse error — the preflight then fails OPEN, never blocks.
    """
    from pathlib import Path  # noqa: PLC0415

    safe = "models--" + repo.replace("/", "--")
    candidates = [Path(hf_cache_dir) / safe, Path(hf_cache_dir) / "hub" / safe]
    best: tuple[float, Path] | None = None
    for repo_dir in candidates:
        snaps = repo_dir / "snapshots"
        if not snaps.exists():
            continue
        try:
            for snap in snaps.iterdir():
                cfg = snap / "config.json"
                try:
                    mtime = cfg.stat().st_mtime
                except (FileNotFoundError, OSError):
                    continue
                if best is None or mtime > best[0]:
                    best = (mtime, cfg)
        except OSError:
            continue
    if best is None:
        return {}
    try:
        return json.loads(best[1].read_text())
    except (OSError, ValueError):
        return {}


async def _gather_preflight_inputs(
    settings, model: ModelRow, request: Request
) -> tuple[dict[str, Any], int, int]:
    """Gather the LOCAL inputs the KV-budget preflight needs.

    Returns ``(config, total_vram, weights_size)``. All three come from the
    local box only (on-disk snapshot + live GPU probe) — no HF network call on
    the load path. Any probe failure degrades to a zero/empty value so
    ``decide_preflight`` fails OPEN. Factored out as a single seam so the
    endpoint tests can patch it without standing up a GPU or an HF cache.
    """
    config = await asyncio.to_thread(
        _read_local_config, settings.hf_cache_dir, model.hf_repo
    )
    weights_size = await asyncio.to_thread(
        _snapshot_dir_size, settings.hf_cache_dir, model.hf_repo
    )

    cache_gpu = getattr(request.app.state, "gpu_probe_cache", None)
    if cache_gpu is None:
        from app.system.routes_gpus import _ProbeCache  # noqa: PLC0415
        cache_gpu = _ProbeCache()
        request.app.state.gpu_probe_cache = cache_gpu
    try:
        snap = await cache_gpu.get()
        vram_by_index = {g.index: g.memory_total_mib * _MIB for g in snap.gpus}
        total_vram = sum(vram_by_index.get(i, 0) for i in model.gpu_indices)
    except Exception:  # noqa: BLE001 — fail open, never block on a probe error
        logger.exception("preflight GPU probe failed for %s", model.id)
        total_vram = 0
    return config, total_vram, weights_size


def _read_engine_log_tail(settings, model_id: str, *, max_lines: int = 200) -> str:
    """Read THIS RUN's engine-log tail for ``model_id`` (at most ``max_lines``).

    The log lives at ``{settings.logs_dir}/{model_id}.log``. It is appended
    across every load attempt, so a plain last-``max_lines`` read straddles run
    boundaries and lets a PREVIOUS attempt's traceback be diagnosed as this
    attempt's cause — reported to the operator as a confident, specific and
    wrong answer that replaces the accurate generic message (#234). Drivers
    therefore stamp a run sentinel into the log at spawn and we start from the
    last one; ``tail_since_last_run`` falls back to the whole-tail behaviour for
    logs written before sentinels existed.

    Defensive: returns ``""`` if the file is missing or unreadable — the
    diagnostics parser then returns None and the caller keeps its generic
    ``last_error``.
    """
    log_path = settings.logs_dir / f"{model_id}.log"
    try:
        text = log_path.read_text(errors="replace")
    except (FileNotFoundError, OSError):
        return ""
    return tail_since_last_run(text, max_lines=max_lines)


def _diagnose_or(
    settings, model_id: str, fallback: str, backend_name: str | None = None
) -> str:
    """Return an actionable diagnosis from the engine-log tail, else ``fallback``.

    Used by the three load-failure paths (subprocess exit, health timeout,
    warmup failure) to upgrade the generic ``last_error`` into something the
    operator can act on when the engine log reveals a recognised failure mode
    (KV overflow, OOM, trust_remote_code). Never raises.

    Sub-project C: the grammar comes from the ROW'S BACKEND, because llama.cpp's
    failures are different failures with different strings. Routing every log
    through the vLLM grammar would at best say nothing and at worst produce a
    confident vLLM-flavoured explanation for a llama.cpp fault. ``None`` resolves
    to vLLM through registry.get -- decision D6, and exactly today's behaviour
    for a call site that has no row in scope.
    """
    try:
        tail = _read_engine_log_tail(settings, model_id)
        diag = registry.get(backend_name).diagnose(tail)
    except Exception:  # noqa: BLE001 — diagnostics must never mask the real failure
        logger.exception("engine-log diagnosis failed for %s", model_id)
        return fallback
    return diag.message if diag is not None else fallback


async def start_engine(settings, sup, port_alloc, model, port, overrides=None):
    """Spawn an engine and drive it to `loaded`: health wait, warmup, status.

    Extracted from the load route so the engine watchdog restarts a crashed model
    through the EXACT same path a manual load takes. A copy would drift, and the
    watchdog would silently restore a model differently from how an operator
    loads it -- the two-implementations-one-fix bug this codebase keeps hitting.
    """
    model_id = model.id
    load_overrides = overrides

    async def on_exit(rc: int) -> None:
        async with open_db(settings.db_path) as db:
            row = await ModelRepo(db).get(model_id)
            if row is not None and row.status in ("loaded", "loading"):
                was = row.status
                last_error = _diagnose_or(
                    settings,
                    model_id,
                    f"vllm subprocess exited unexpectedly (rc={rc})",
                    backend_name=getattr(row, "backend", None),
                )
                # Always carry the rc so operators can correlate with the log.
                if not last_error.endswith(f"(rc={rc})"):
                    last_error = f"{last_error} (rc={rc})"
                await ModelRepo(db).update_status(
                    model_id,
                    "failed",
                    last_error=last_error,
                )
                # Hand the watchdog a machine-readable reason to restart this.
                # Only a row that was actually SERVING earns an automatic
                # restart: a 'loading' row that never came up is usually a bad
                # config, and retrying that is a crash loop with a nicer name.
                #
                # This write is what makes recovery survive a diagnosis. The
                # 2026-08-18 outage was exactly this path: the engine died
                # seconds before the warden, on_exit wrote a diagnosed
                # last_error, and boot reconciliation then skipped the
                # already-'failed' row -- so the old string sentinel was never
                # written and nothing ever restarted it.
                if was == "loaded":
                    await ModelRepo(db).set_prior_status(model_id, "loaded")
                await RuntimeRepo(db).clear(model_id)
        port_alloc.release(port)

    async def _run_load():
        try:
            await sup.load(model, port=port, on_exit=on_exit, overrides=load_overrides)
        except Exception as e:
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).update_status(model_id, "failed", last_error=str(e))
            port_alloc.release(port)
            return
        # Record the variant this engine was launched as (0036), so the token
        # page can name it. Bookkeeping: a failure here must not fail a load
        # that succeeded; the proxy then records it on the first request.
        get_variant = getattr(sup, "get_variant", None)
        variant = get_variant(model_id) if get_variant is not None else None
        if isinstance(variant, Variant):
            try:
                async with open_db(settings.db_path) as db:
                    await ModelVariantRepo(db).ensure(variant)
                mark_recorded(settings.db_path, variant.id)
            except Exception:  # noqa: BLE001
                logger.warning("could not record the variant of %s", model_id, exc_info=True)
        # The driver owns where the engine actually listens — loopback for
        # the in-container subprocess, the engine container's DNS name for
        # the docker driver. ``get_host`` returns None only if the engine
        # vanished between spawn and here; loopback is the safe fallback.
        host = sup.get_host(model_id) or "127.0.0.1"
        ok = await wait_for_health(
            port=port, host=host, timeout_s=settings.load_timeout_s
        )
        if not ok:
            # Subprocess is left running; operator must force-unload to
            # release GPUs. This is intentional (spec §"Load-route changes")
            # — auto-SIGTERM here was the original race trigger.
            # If the engine log reveals WHY it never came up (KV overflow, OOM,
            # trust_remote_code), surface that instead of the bare timeout.
            generic = "health timeout; subprocess still holding GPUs — force-unload to release"
            diagnosed = _diagnose_or(
                settings, model_id, generic,
                backend_name=getattr(model, "backend", None),
            )
            last_error = (
                generic
                if diagnosed == generic
                else f"{diagnosed} (subprocess still holding GPUs — force-unload to release)"
            )
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).update_status(
                    model_id,
                    "failed",
                    last_error=last_error,
                )
            return
        await sup.mark_warming(model_id)
        probe_result = await warmup_probe(
            port=port,
            host=host,
            served_model_name=model.served_model_name,
            timeout_s=settings.warmup_probe_timeout_s,
        )
        if not probe_result.ok:
            # The probe failed — but the engine log may explain it crashed
            # mid-warmup (e.g. a deferred KV-cache allocation OOM). Prefer the
            # log diagnosis; fall back to the probe detail.
            generic = (
                f"{probe_result.detail}; subprocess still holding "
                f"GPUs — force-unload to release"
            )
            diagnosed = _diagnose_or(
                settings, model_id, generic,
                backend_name=getattr(model, "backend", None),
            )
            last_error = (
                generic
                if diagnosed == generic
                else f"{diagnosed} (subprocess still holding GPUs — force-unload to release)"
            )
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).update_status(
                    model_id,
                    "failed",
                    last_error=last_error,
                )
            return
        await sup.mark_ready(model_id)
        # #29 — persist health_ok=True now that the warmup probe has
        # actually served a request. Stats/status badges and any future
        # telemetry need to distinguish 'process up but never proved
        # serving' from 'process up and served at least one request'.
        # Both the upsert (creates the row with pid/port/started_at) and
        # the update_health (writes the health columns) MUST happen in
        # this success branch; the legacy code wrote only the upsert and
        # left health_ok stuck at its default 0.
        now_iso = datetime.now(UTC).isoformat()
        async with open_db(settings.db_path) as db:
            await ModelRepo(db).update_status(model_id, "loaded")
            await RuntimeRepo(db).upsert(
                model_id,
                pid=sup.get_pid(model_id),
                port=port,
                started_at=now_iso,
            )
            await RuntimeRepo(db).update_health(model_id, True, now_iso)

    await _run_load()


@router.post("/{model_id}/load", status_code=202)
async def load_model(model_id: str, request: Request, _user: str = Depends(require_jwt)):
    settings = request.app.state.settings
    sup = request.app.state.supervisor
    port_alloc = request.app.state.port_allocator

    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get(model_id)
        if not model:
            raise HTTPException(404, "not found")
        # #256 -- the row may predate this change, so this is checked at
        # load time (when the repo's Python actually runs) rather than
        # trusting that register-time refusal above ever ran for it.
        if model.trust_remote_code:
            refuse_session_only(request, message=TRUST_REMOTE_CODE_LOAD_MESSAGE)
        if model.status not in ("pulled", "failed"):
            raise HTTPException(409, f"cannot load from status '{model.status}'")
        allowed = (await SetupRepo(db).get()).draft.get("allowed_gpu_indices", [])
        if not set(model.gpu_indices).issubset(set(allowed)):
            raise HTTPException(
                422, f"gpu_indices {model.gpu_indices} not subset of allowed {allowed}"
            )
        # Physical-presence pre-flight: a model can be allow-listed yet point
        # at a GPU index that has since vanished (card pulled, driver
        # re-index). Hand vLLM a bad CUDA_VISIBLE_DEVICES and it crashes
        # opaquely; fail fast with the same envelope the fit-preview check
        # uses so the frontend can reuse its handling. Reuse the shared probe
        # cache (same one /api/system/gpus and fit-preview read).
        from app.system.routes_gpus import _get_cache  # noqa: PLC0415
        snap = await _get_cache(request).get()
        # Fail CLOSED: a configured index absent from the probe — OR a
        # probe error that leaves us with no ground truth at all — must
        # block the load (mirrors the fit-preview check). When the probe
        # errored ``snap.gpus`` is empty, so every configured index is
        # "missing" and ``available`` is ``[]``; handing vLLM a
        # CUDA_VISIBLE_DEVICES we cannot confirm only crashes opaquely.
        present = {g.index for g in snap.gpus}
        missing = [i for i in model.gpu_indices if i not in present]
        if missing:
            raise HTTPException(
                422,
                detail={
                    "error_code": "gpu_index_missing",
                    "message": f"gpu_indices {missing} not present in nvidia-smi probe",
                    "available": sorted(present),
                    "probe_error": snap.probe_error,
                },
            )
        await ModelRepo(db).update_status(model_id, "loading")

    # --- KV-budget preflight (runs synchronously, BEFORE spawning the engine).
    # Resolves the effective max_model_len from LOCAL inputs only (on-disk
    # config + live GPU probe + on-disk snapshot size) and decides whether the
    # requested context can fit. Two outcomes change the launch:
    #   - CAP: row.max_model_len is NULL (no user intent) and the model-max
    #     context won't fit → auto-cap to a fittable value via overrides, and
    #     report it in the 202 body's ``context_capped`` field.
    #   - BLOCK: the user EXPLICITLY set a max_model_len that won't fit → 422.
    # Everything else proceeds unchanged. The decision is fail-open: any gap in
    # the local inputs yields PROCEED, never a false block (a real crash is then
    # made actionable by the engine-log diagnostics below).
    config, total_vram, weights_size = await _gather_preflight_inputs(
        settings, model, request
    )
    decision = decide_preflight(
        config=config,
        total_vram=total_vram,
        weights_size=weights_size,
        row_max_model_len=model.max_model_len,
        gpu_memory_utilization=model.gpu_memory_utilization,
        max_batch_size=model.max_batch_size,
    )
    load_overrides: dict[str, Any] | None = None
    context_capped: dict[str, Any] | None = None
    if decision.kind == "block":
        # Restore the row to a re-loadable status — we never spawned anything.
        async with open_db(settings.db_path) as db:
            await ModelRepo(db).update_status(model_id, "pulled")
        raise HTTPException(
            422,
            detail={
                "error_code": "wont_fit",
                "message": (
                    f"max_model_len={model.max_model_len} will not fit on the "
                    f"selected GPU(s): the KV cache plus weights exceed available "
                    f"VRAM. Lower max_model_len"
                    + (
                        f" to <= {decision.recommended_max_model_len}"
                        if decision.recommended_max_model_len
                        else ""
                    )
                    + " or select GPU(s) with more memory."
                ),
                "recommended_max_model_len": decision.recommended_max_model_len,
                "breakdown": decision.breakdown,
            },
        )
    if decision.kind == "cap":
        load_overrides = {"max_model_len": decision.cap_to}
        context_capped = {
            "from": decision.from_len,
            "to": decision.cap_to,
            "reason": "kv_cache_exceeds_vram",
        }
        logger.info(
            "load preflight auto-capped max_model_len for %s: %s -> %s "
            "(kv_cache_exceeds_vram)",
            model_id,
            decision.from_len,
            decision.cap_to,
        )

    port = port_alloc.allocate()

    asyncio.create_task(
        start_engine(settings, sup, port_alloc, model, port, load_overrides)
    )
    body: dict[str, Any] = {"status": "loading", "port": port}
    if context_capped is not None:
        body["context_capped"] = context_capped
    return body


def _unloadable_statuses(force: bool) -> tuple[str, ...]:
    """Which model statuses ``POST /{id}/unload`` accepts.

    #236 — ``force=true`` has to mean force. The transient statuses are
    precisely the ones an operator needs it for: a warden that died mid-load
    leaves a row in 'loading' whose engine no longer exists anywhere, and the
    plain guard below refused exactly that ("cannot unload from status
    'loading'"), so the ONLY way out was editing SQLite by hand. A force
    unload kills whatever the supervisor still holds (nothing, after a
    restart) and writes a terminal status unconditionally.

    'pulling' is deliberately NOT here: no engine is involved in a download,
    and the terminal status this route writes is 'pulled' — a lie for a
    half-finished pull. Boot reconciliation
    (app/runtime/boot_reconcile.py) is what recovers those.
    """
    if force:
        return ("loaded", "failed", "loading", "unloading")
    return ("loaded", "failed")


@router.post("/{model_id}/unload", status_code=202)
async def unload_model(
    model_id: str,
    request: Request,
    force: bool = False,
    _user: str = Depends(require_jwt),
):
    settings = request.app.state.settings
    sup = request.app.state.supervisor
    port_alloc = request.app.state.port_allocator
    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get(model_id)
        if not model:
            raise HTTPException(404, "not found")
        if model.status not in _unloadable_statuses(force):
            raise HTTPException(409, f"cannot unload from status '{model.status}'")
        rt = await RuntimeRepo(db).get(model_id)
        # The runtime row only exists once a load reached 'loaded'. Forcing a
        # model out of 'loading' therefore finds nothing there, while the
        # allocator DOES hold the port this process handed the spawn — ask the
        # supervisor before giving up on releasing it (#236).
        port = (rt.port if rt else None) or sup.get_port(model_id)
    # #166 — surface the refusal SYNCHRONOUSLY (the state check is instant),
    # but run the engine teardown itself in a background task. A large
    # multi-GPU engine can take many seconds to terminate; doing that inline
    # kept the request open until the client/proxy disconnected, at which
    # point the CSRF ``BaseHTTPMiddleware`` raised Starlette's
    # ``RuntimeError("No response returned.")`` → HTTP 500, leaving the row
    # stranded in 'unloading' (the terminal transition below never ran).
    try:
        await sup.ensure_unloadable(model_id, force=force)
    except UnloadRefused as e:
        raise HTTPException(
            409,
            (
                f"refused: supervisor state is {e.state.name}; "
                f"use ?force=true to override"
            ),
        ) from e
    prior_status = model.status
    async with open_db(settings.db_path) as db:
        await ModelRepo(db).update_status(model_id, "unloading")
    # S7 (#124) — flush the proxy's per-repo tokenizer cache so a long-lived
    # warden that loads-and-unloads a rotating set of models doesn't
    # accumulate one cached tokenizer per ever-seen repo (code-review
    # finding #5). ``evict`` is a no-op if the repo was never tokenized.
    tok_cache = getattr(request.app.state, "tokenizers", None)
    hf_repo = model.hf_repo

    async def runner() -> None:
        try:
            await sup.unload(model_id, force=force)
        except UnloadRefused:
            # State changed between the pre-flight check and teardown. Roll
            # back so the row never strands in the transient 'unloading'.
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).update_status(model_id, prior_status)
            return
        except Exception:
            # Teardown raised, but the engine is gone either way. Never leave
            # the row stuck in 'unloading' — fall through to the terminal
            # transition so recovery never requires a control-plane restart.
            logger.exception(
                "unload teardown for %s raised; forcing terminal state",
                model_id,
            )
        if port:
            try:
                port_alloc.release(port)
            except Exception:
                logger.warning(
                    "port release for %s during unload failed",
                    model_id,
                    exc_info=True,
                )
        async with open_db(settings.db_path) as db:
            await RuntimeRepo(db).clear(model_id)
            await ModelRepo(db).update_status(model_id, "pulled")
        if tok_cache is not None and hf_repo:
            await tok_cache.evict(hf_repo)

    asyncio.create_task(runner())
    return {"status": "unloading"}
