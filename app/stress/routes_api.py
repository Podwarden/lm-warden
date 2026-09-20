"""Stress-test API surface (design §7.1) and the publication rules (§7.4, §7.5).

Two endpoints and one shared publication layer:

* ``POST /api/models/{id}/stress`` — start a run. Admission only; the run
  itself is an in-process asyncio task owned by ``app/stress/runner.py``.
* ``GET /api/models/{id}/capabilities`` — the full operator-facing record.
* ``warden_block()`` — what ``GET /v1/models`` adds to each entry. It lives
  here rather than in the proxy so there is exactly one implementation of the
  rule that decides whether a client is told a number at all.

Three things in here are load-bearing and are not plumbing:

**Starting a run is an operator action, never a data-plane one.** A stress
test deliberately crashes a production engine. An API token (``vw_…``) is a
credential handed to a client application so it can send completions; it must
never be able to take a GPU down. The design left this unstated. An admin
token (``vwa_…``) is an operator credential and may start one. ``require_jwt``
would reject a ``vw_`` token with a confusing "invalid token" 401 —
``require_operator`` turns that into a 403 that says why.

**``recommended_config`` never reaches ``/v1/models``.** It describes a
configuration the model is *not currently running* (§6.2). Telling a client it
may send 96k-token prompts to an engine allocated for 32k produces a refusal
the client has no way to interpret. It is returned by ``/capabilities``, which
is operator-facing, and by nothing else.

**On stale, we do not fall back to ``declared``.** ``declared`` is the number
the measurement corrected downward. Falling back silently *widens* the
advertised context back to a value already known not to work, which is the one
outcome worse than serving a slightly out-of-date correction. A stale value
ships as a flagged ceiling and can only ever move down — including down to the
current declared value, when the engine was reloaded smaller than the
configuration that was measured.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from app.auth.bearer import INFERENCE_TOKEN_PREFIX, parse_bearer_header
from app.auth.deps import require_jwt
from app.auth.principal import principal_of
from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.stress_runs import StressRunRepo, StressRunRow
from app.models.suggest import DISCLAIMER_TEXT
from app.models.writer import (
    ModelChangeRefused,
    apply_model_change,
    dry_run_model_change,
)
from app.stress.apply import ApplyRefused, plan_apply
from app.stress.ceiling import resolve_for_model
from app.stress.fingerprint import Gpu, Neighbour, Subject, fingerprint
from app.stress.measured import has_measurement, is_axis
from app.stress.modes import at_least_as_thorough

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["stress"])


# ---------------------------------------------------------------------------
# Harness identity
# ---------------------------------------------------------------------------
#
# Both feed the fingerprint (§7.3, "harness"), and both are hand-bumped rather
# than derived from source: hashing the module text would churn on a comment
# edit and mark every stored measurement stale for no physical reason. Bump
# PROBE_SUITE_HASH when a probe's content changes, ALGORITHM_VERSION when the
# search or the oracle changes.
# They live HERE rather than in the runner because a READER has to compute the
# same fingerprint without importing the runner (which pulls in httpx clients,
# the probe engine and the supervisor). ``app/stress/runner.py`` declares its
# own ``ALGORITHM_VERSION``; the two MUST agree. If they ever drift, every
# stored measurement becomes permanently unmatchable and nothing is published,
# with no error anywhere to say why.
PROBE_SUITE_HASH = "suite-v1"
ALGORITHM_VERSION = 1

#: §7.2 "Cooldown and re-use [R-M12]". A completed, publishable run under the
#: current fingerprint younger than this is returned instead of starting
#: another; nothing else in the design stops a run per minute. Six hours is a
#: starting point, not a measurement — it is short enough that an operator who
#: genuinely wants a fresh number can wait, and `force=true` skips it outright.
DEFAULT_COOLDOWN_S = 6 * 3600
COOLDOWN_ENV_VAR = "VW_STRESS_COOLDOWN_S"

#: §8. Stated before the operator confirms; wrong by a wide margin on unusual
#: hardware, which is why it is labelled an estimate everywhere it surfaces.
MODE_ESTIMATE_MINUTES: dict[str, int] = {
    "conservative": 15,
    "quick": 38,
    "thorough": 195,
}

Mode = Literal["conservative", "quick", "thorough"]

#: Mirrors ``suggest.py``'s convention: every advisory number the warden emits
#: carries the same kind of sentence, in the same field name, so a consumer
#: that auto-applies one of them has to ignore an explicit contract red flag.
STRESS_DISCLAIMER: str = (
    "A stress test deliberately drives this model past its operating limits and "
    "may crash the engine, so it is disruptive to anything currently served by "
    "it. Measured limits describe this exact configuration on this exact "
    "hardware and stop being true when either changes. " + DISCLAIMER_TEXT
)


def cooldown_seconds() -> float:
    """Read at call time, not import time, so a test can monkeypatch the env."""
    try:
        v = float(os.environ.get(COOLDOWN_ENV_VAR, DEFAULT_COOLDOWN_S))
    except ValueError:
        return float(DEFAULT_COOLDOWN_S)
    return v if v >= 0 else float(DEFAULT_COOLDOWN_S)


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


async def require_operator(request: Request) -> str:
    """A session JWT or an admin token. An inference token is refused with an
    explanation.

    ``require_jwt`` alone is already sufficient for security -- a ``vw_`` token
    is neither a JWT nor an admin token. It is not sufficient for the operator
    reading the response, who gets "invalid token" and reasonably concludes
    their token has expired rather than that this endpoint is off-limits to
    every token of that kind.
    """
    token = parse_bearer_header(request.headers.get("authorization"))
    if token is not None and token.startswith(INFERENCE_TOKEN_PREFIX):
        raise HTTPException(
            403,
            detail={
                "error_code": "operator_only",
                "message": (
                    "Inference tokens (vw_...) are data-plane credentials and "
                    "cannot start a stress test: a run deliberately crashes the "
                    "engine. Sign in as an operator, or use an admin token (vwa_...)."
                ),
            },
        )
    return await require_jwt(request)


# ---------------------------------------------------------------------------
# The fingerprint, as the READER can compute it
# ---------------------------------------------------------------------------
#
# The runner stamps a fingerprint onto the run row; every reader recomputes one
# and compares. That only works if every component is derivable by a reader
# holding nothing but app.state and the model row, which forced two documented
# departures from §7.3:
#
#   * ``max_num_batched_tokens`` is read from the CONFIGURED argv, not observed
#     from the live engine. A reader cannot ask an engine that may not be
#     running, and a component only one side can compute makes every
#     measurement permanently stale — which is strictly worse than missing the
#     case where an engine default changes under us.
#   * ``driver_version`` costs a second nvidia-smi invocation, so it is
#     resolved once per process and memoised. Resolving it lazily WITHOUT the
#     memo would flip the fingerprint the first time it succeeded and mark
#     every prior measurement stale.
#
# ``ecc_enabled`` is passed as False for every card, on purpose: the live probe
# does not report ECC state, but ``memory_total_mib`` — which it does report —
# is what actually moves when ECC is toggled (16,384 → 15,360 MiB, measured
# 2026-09-03). The observable proxy is already in the fingerprint.

_MNBT_FLAG = "--max-num-batched-tokens"


def _max_num_batched_tokens(model: ModelRow) -> int | None:
    """Parse ``--max-num-batched-tokens`` out of the configured extra args."""
    args = list(model.extra_args or [])
    for i, arg in enumerate(args):
        if arg == _MNBT_FLAG and i + 1 < len(args):
            raw = args[i + 1]
        elif arg.startswith(_MNBT_FLAG + "="):
            raw = arg.split("=", 1)[1]
        else:
            continue
        try:
            return int(raw)
        except ValueError:
            return None
    return None


async def driver_version(app_state: Any) -> str | None:
    """The NVIDIA driver version, resolved once per process.

    ``app_state.stress_driver_version`` is both the memo and the test seam:
    set it to a string (or to ``None`` via ``stress_driver_version_resolved``)
    and nothing shells out.
    """
    if getattr(app_state, "stress_driver_version_resolved", False):
        return getattr(app_state, "stress_driver_version", None)
    value: str | None = getattr(app_state, "stress_driver_version", None)
    if value is None:
        try:
            from app.system.system_info import collect_gpus

            gpus = await asyncio.to_thread(collect_gpus)
            for g in gpus:
                if g.get("driver_version"):
                    value = str(g["driver_version"])
                    break
        except Exception:  # noqa: BLE001 — a missing driver is not an error here
            logger.debug("stress: could not resolve the NVIDIA driver version")
            value = None
    app_state.stress_driver_version = value
    app_state.stress_driver_version_resolved = True
    return value


def _probe_cache(app_state: Any):
    """The same nvidia-smi snapshot cache ``/api/system/gpus`` uses (TTL 2s).

    Shared deliberately: recomputing a fingerprint must never cost an extra
    subprocess on the ``/v1/models`` path.
    """
    cache = getattr(app_state, "gpu_probe_cache", None)
    if cache is None:
        from app.system.routes_gpus import _ProbeCache

        cache = _ProbeCache()
        app_state.gpu_probe_cache = cache
    return cache


def _neighbours(model: ModelRow, all_models: list[ModelRow]) -> tuple[Neighbour, ...]:
    """Co-resident models, identified by id alone.

    §7.3 specifies ``(model_id, vram_mib)``. Live VRAM is the wrong half of
    that pair: a neighbour's footprint moves every second as its KV pool fills,
    so a fingerprint carrying it would never match itself and no measurement
    would ever be publishable. The concern the design states — "a limit
    measured while a neighbour held several GB is not true once that neighbour
    unloads" — is exactly a change in the SET of co-resident models, which the
    id set captures and the byte count only obscures.
    """
    mine = set(model.gpu_indices or [])
    out = [
        Neighbour(model_id=m.id, vram_mib=0)
        for m in all_models
        if m.id != model.id and m.status == "loaded" and mine & set(m.gpu_indices or [])
    ]
    return tuple(sorted(out, key=lambda n: n.model_id))


@dataclass(frozen=True)
class FingerprintResult:
    value: str
    gpu_uuids: tuple[str, ...]
    probe_error: str | None
    #: True when the probe could not confirm every configured GPU. The
    #: fingerprint is still computed (from what was seen), but it will not
    #: match one taken when the cards were visible — so a measurement goes
    #: stale rather than being served against unknown hardware.
    hardware_incomplete: bool


async def build_subject(
    app_state: Any, model: ModelRow, *, all_models: list[ModelRow], gpus: tuple[Gpu, ...]
) -> Subject:
    """The single subject builder.

    The RUNNER must call this too (via its ``EngineControl.subject()``), not
    assemble its own ``Subject``. Two independent builders that disagree on one
    field produce two fingerprints that never match, so every run would be
    recorded and none would ever be published — and nothing would raise.
    """
    from app.proxy.scheduler import _default_max_inflight
    from app.system.routes_engine import _backend_version

    settings = app_state.settings
    return Subject(
        hf_repo=model.hf_repo,
        hf_revision=model.hf_revision,
        filename=model.filename,
        quantization=None,
        backend=model.backend or "vllm",
        mmproj_filename=model.mmproj_filename,
        max_model_len=model.max_model_len,
        max_num_seqs=model.max_batch_size,
        tensor_parallel_size=model.tensor_parallel_size,
        gpu_memory_utilization=model.gpu_memory_utilization,
        dtype=model.dtype,
        n_gpu_layers=model.n_gpu_layers,
        parallelism_strategy=model.parallelism_strategy,
        extra_args=tuple(model.extra_args or []),
        extra_env=tuple(sorted((model.extra_env or {}).items())),
        engine_version=_backend_version(model.backend or "vllm"),
        engine_image=model.engine_image,
        max_num_batched_tokens=_max_num_batched_tokens(model),
        proxy_max_inflight=_default_max_inflight(),
        request_max_wall_s=float(getattr(settings, "request_max_wall_s", 0.0)),
        gpus=gpus,
        driver_version=await driver_version(app_state),
        neighbours=_neighbours(model, all_models),
        probe_suite_hash=PROBE_SUITE_HASH,
        algorithm_version=ALGORITHM_VERSION,
    )


async def live_gpus(app_state: Any, model: ModelRow) -> tuple[tuple[Gpu, ...], Any, bool]:
    """``(gpus, probe_error, hardware_incomplete)`` from the shared nvidia-smi cache."""
    snap = await _probe_cache(app_state).get()
    by_index = {g.index: g for g in snap.gpus}
    wanted = sorted(model.gpu_indices or [])
    live = [by_index[i] for i in wanted if i in by_index]
    gpus = tuple(
        Gpu(
            uuid=g.uuid,
            name=g.name,
            memory_total_mib=g.memory_total_mib,
            compute_cap=g.compute_cap,
            ecc_enabled=False,
        )
        for g in live
    )
    return gpus, snap.probe_error, len(live) != len(wanted)


async def compute_fingerprint(
    app_state: Any, model: ModelRow, *, all_models: list[ModelRow]
) -> FingerprintResult:
    gpus, probe_error, incomplete = await live_gpus(app_state, model)
    subject = await build_subject(app_state, model, all_models=all_models, gpus=gpus)
    return FingerprintResult(
        value=fingerprint(subject),
        gpu_uuids=tuple(g.uuid for g in gpus),
        probe_error=probe_error,
        hardware_incomplete=incomplete,
    )


# ---------------------------------------------------------------------------
# Publication (§7.4, §7.5)
# ---------------------------------------------------------------------------

#: Every published limit carries all of these, filled with ``None`` where the
#: run did not determine one. A probabilistic edge cannot be published as a
#: scalar, and a client has to be able to tell a value confirmed against a
#: 21% contour from one confirmed against 9% — which it cannot do if the
#: oracle parameters are only sometimes present.
LIMIT_FIELDS: tuple[str, ...] = (
    "value",
    "unit",
    "limited_by",
    "gate_tripped",
    "raw_confirmed",
    "first_observed_failure",
)
ORACLE_FIELDS: tuple[str, ...] = (
    "n_consecutive",
    "contour_p",
    "confirm_m",
    "safety_factor",
)

#: Limits denominated in prompt/context tokens. A stale value in this family is
#: additionally clamped to the CURRENT declared context: a measurement taken at
#: 96k must not be advertised while the engine is loaded at 32k, and clamping
#: only ever moves the published number down.
CONTEXT_LIMIT_KEYS = frozenset({"quality_context", "recommended_max_prompt_tokens"})

#: The limit whose value bounds what a client may actually achieve through the
#: proxy — the run measures the engine directly, but ``PriorityScheduler``
#: admits at most ``VW_PROXY_MAX_INFLIGHT`` per engine (§9.1).
CONCURRENCY_LIMIT_KEY = "recommended_concurrency"


def _normalise_limit(
    name: str,
    raw: Any,
    *,
    row: StressRunRow,
    provenance: str,
    stale_reason: str | None,
    declared_context: int | None,
) -> dict[str, Any] | None:
    """One stored limit, rendered in the published shape.

    Accepts a bare scalar for robustness — a run that recorded ``122880``
    rather than an object still publishes as an object, with every field it did
    not state explicitly null rather than invented.
    """
    if raw is None:
        return None
    src: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {"value": raw}

    out: dict[str, Any] = {f: src.get(f) for f in LIMIT_FIELDS}
    oracle_src = src.get("oracle") if isinstance(src.get("oracle"), dict) else {}
    out["oracle"] = {f: oracle_src.get(f, src.get(f)) for f in ORACLE_FIELDS}
    out["confidence"] = src.get("confidence")
    out["provenance"] = provenance
    out["stale_reason"] = stale_reason
    out["measured_at"] = row.finished_at or row.started_at
    out["run_id"] = row.id
    out["fingerprint"] = row.fingerprint
    out["clamped_to_declared"] = False

    if (
        provenance == "stale"
        and name in CONTEXT_LIMIT_KEYS
        and isinstance(out["value"], int)
        and isinstance(declared_context, int)
        and declared_context < out["value"]
    ):
        # Never widen (§7.4) — and never advertise a ceiling the currently
        # loaded engine cannot honour either.
        out["value"] = declared_context
        out["clamped_to_declared"] = True
    return out


@dataclass(frozen=True)
class PublishedRecord:
    """What a surface is allowed to say about a model right now."""

    #: ``measured`` | ``stale`` | ``none``
    provenance: str
    stale_reason: str | None
    run: StressRunRow | None
    limits: dict[str, dict[str, Any]]

    @property
    def has_measurement(self) -> bool:
        return self.run is not None


_NO_RECORD = PublishedRecord(provenance="none", stale_reason=None, run=None, limits={})


def _publishable(row: StressRunRow) -> bool:
    """The same four conditions ``StressRunRepo.current_for`` enforces.

    Repeated here only because the stale path asks the question across
    fingerprints, which ``current_for`` deliberately cannot express.
    """
    return (
        row.status == "completed"
        and not row.non_monotone
        and not row.traffic_observed
        # ...and the fifth, which current_for applies after its fetch. This
        # function claims to enforce the same rule, so it must move with it or
        # the stale path would publish a run the fresh path rejects.
        and has_measurement(row.limits, row.recommended_config)
    )


async def published_record(
    db, model: ModelRow, fp: str, *, hardware_incomplete: bool = False
) -> PublishedRecord:
    repo = StressRunRepo(db)
    current = await repo.current_for(model.id, fp)
    if current is not None:
        return PublishedRecord(
            provenance="measured",
            stale_reason=None,
            run=current,
            limits=_render(current, "measured", None, model),
        )

    history = await repo.list_for_model(model.id, limit=50)
    fallback = next((r for r in history if _publishable(r)), None)
    if fallback is None:
        return _NO_RECORD
    reason = (
        "the GPU probe could not confirm every configured card, so the "
        "measurement cannot be matched to this hardware"
        if hardware_incomplete
        else "the model, engine, hardware or co-resident set has changed "
        "since this measurement"
    )
    return PublishedRecord(
        provenance="stale",
        stale_reason=reason,
        run=fallback,
        limits=_render(fallback, "stale", reason, model),
    )


def _render(
    row: StressRunRow, provenance: str, stale_reason: str | None, model: ModelRow
) -> dict[str, dict[str, Any]]:
    limits = row.limits if isinstance(row.limits, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for name, raw in limits.items():
        # The runner files bookkeeping in the same dict as its measurements.
        # Rendering those as limits put `neighbours` and `baseline` on screen
        # as MEASURED LIMITS cards with a "measured" badge and no value.
        if not is_axis(name):
            continue
        rendered = _normalise_limit(
            name,
            raw,
            row=row,
            provenance=provenance,
            stale_reason=stale_reason,
            declared_context=model.max_model_len,
        )
        if rendered is not None:
            out[name] = rendered
    return out


def _client_effective_concurrency(record: PublishedRecord) -> int | None:
    """§9.1 — the run probes the engine directly, the client goes through us."""
    limit = record.limits.get(CONCURRENCY_LIMIT_KEY)
    if not limit or not isinstance(limit.get("value"), int):
        return None
    from app.proxy.scheduler import _default_max_inflight

    return min(int(limit["value"]), _default_max_inflight())


def warden_block(record: PublishedRecord) -> dict[str, Any]:
    """The additive ``warden`` sub-object for ``GET /v1/models``.

    Deliberately does NOT contain ``recommended_config``: it describes a
    configuration this engine is not running, and a client acting on it would
    be told to send prompts the engine will correctly refuse (§6.2). The only
    surface that gets it is ``/capabilities``, which an operator reads.
    """
    block: dict[str, Any] = {
        "provenance": record.provenance,
        "stale": record.provenance == "stale",
        "stale_reason": record.stale_reason,
        "limits": record.limits,
        "disclaimer": STRESS_DISCLAIMER,
    }
    if record.run is not None:
        block["run_id"] = record.run.id
        block["measured_at"] = record.run.finished_at or record.run.started_at
        block["fingerprint"] = record.run.fingerprint
        block["loads_since_measurement"] = record.run.loads_since_measurement
    else:
        block["run_id"] = None
        block["measured_at"] = None
        block["fingerprint"] = None
        block["loads_since_measurement"] = 0
    effective = _client_effective_concurrency(record)
    if effective is not None:
        block["client_effective_concurrency"] = effective
    return block


# ---------------------------------------------------------------------------
# POST /api/models/{id}/stress
# ---------------------------------------------------------------------------


class StressRequest(BaseModel):
    mode: Mode = "conservative"
    #: The operator's statement that they know this may take the engine down.
    #: Mirrors the unload-first 409 convention: refuse loudly and synchronously
    #: rather than accepting and surprising someone.
    acknowledge_disruption: bool = False
    #: Skips the cooldown and lets the run continue through conditions that
    #: taint it (foreign traffic). Never skips the acknowledgement, the status
    #: check, or either busy check — those protect something other than the
    #: operator's own patience.
    force: bool = False
    #: "Wipe and run again": discard this model's finished runs before starting.
    #: Offered beside a completed run's results, never taken implicitly. A run
    #: that merely FAILED must not withdraw a good older measurement -- that is
    #: why `current_for` scans past valueless runs -- but an operator who has
    #: read the results and asked to replace them is the opposite case, and
    #: only they can tell the two apart. Implies `force`: having decided to
    #: discard the measurement, waiting out its cooldown protects nothing.
    reset: bool = False


class StressApplyRequest(BaseModel):
    #: Which run's recommendation to apply. Named explicitly rather than
    #: "the latest": the operator applies what they READ, and a run finishing
    #: between the render and the click must not silently change the number.
    run_id: str
    #: Applying unloads and reloads the engine, so the model stops serving.
    #: Same convention as starting a run: refuse loudly and synchronously
    #: rather than accepting and surprising someone.
    acknowledge_disruption: bool = False


class StressAccepted(BaseModel):
    run_id: str
    model_id: str
    mode: Mode
    status: str
    reused: bool
    estimate_minutes: int
    fingerprint: str
    disclaimer: str
    cooldown_expires_at: str | None = None


def _conflict(error_code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(409, detail={"error_code": error_code, "message": message, **extra})


def _parse_sqlite_ts(value: str | None) -> datetime | None:
    """``datetime('now')`` writes ``YYYY-MM-DD HH:MM:SS`` in UTC, unmarked."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def _start_run(
    request: Request, model_id: str, *, mode: str, force: bool, model=None
) -> str:
    """Hand off to the runner.

    Imported lazily, and overridable via ``app.state.stress_start_run``, so
    this module neither imports the runner at boot nor pins its internals into
    a route test.
    """
    override = getattr(request.app.state, "stress_start_run", None)
    if override is not None:
        return await override(request.app.state, model_id, mode=mode, force=force)
    from app.stress.runner import start_run

    # Resolve the model's own ceiling so the sweep can look UPWARD. Without it
    # plan_candidates can only ask whether the current setting is too
    # ambitious, never whether there is headroom -- which is half the question
    # the feature exists to answer. Best-effort by design: see resolve_for_model.
    ceiling = await resolve_for_model(request.app.state, model)
    return await start_run(
        request.app.state, model_id, mode=mode, force=force, model_ceiling=ceiling
    )


@router.post("/{model_id}/stress", status_code=202, response_model=StressAccepted)
async def start_stress(
    model_id: str,
    body: StressRequest,
    request: Request,
    response: Response,
    _user: str = Depends(require_operator),
) -> StressAccepted:
    settings = request.app.state.settings
    if not body.acknowledge_disruption:
        raise _conflict(
            "disruption_not_acknowledged",
            "a stress test may crash the engine serving this model; resend "
            "with acknowledge_disruption=true",
            disclaimer=STRESS_DISCLAIMER,
        )

    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get(model_id)
        if model is None:
            raise HTTPException(404, "not found")
        if model.status != "loaded":
            raise _conflict(
                "model_not_loaded",
                f"cannot stress from status '{model.status}'; the model must be loaded",
            )
        all_models = await ModelRepo(db).list_all()
        fp = await compute_fingerprint(request.app.state, model, all_models=all_models)

        repo = StressRunRepo(db)
        history = await repo.list_for_model(model_id, limit=50)
        active = next((r for r in history if r.status == "running"), None)
        if active is not None:
            raise _conflict(
                "run_already_active",
                f"stress run {active.id} is already running for this model",
                run_id=active.id,
            )

        leases = getattr(request.app.state, "stress_leases", None)
        if leases is not None:
            if leases.is_held(model_id):
                raise _conflict(
                    "run_already_active",
                    "a stress lease is already held for this model",
                    run_id=leases.holder(model_id),
                )
            if fp.gpu_uuids and leases.gpus_busy(fp.gpu_uuids):
                # Two runs sharing a card each attribute the other's memory and
                # scheduling pressure to their own limit, and neither can tell.
                raise _conflict(
                    "gpu_set_busy",
                    "another stress run holds at least one of this model's GPUs; "
                    "a measurement taken alongside it would be attributing that "
                    "run's pressure to this model",
                )

        if body.reset:
            # Before the cooldown is even consulted: there is nothing left to
            # reuse, and asking "is the old measurement still fresh?" about a
            # record the operator has just discarded is incoherent.
            cleared = await repo.clear_for_model(model_id)
            logger.info(
                "stress: reset requested for %s — cleared %d finished run(s)",
                model_id, cleared,
            )

        if not body.force and not body.reset:
            recent = await repo.current_for(model_id, fp.value)
            started = _parse_sqlite_ts(recent.started_at) if recent else None
            ttl = cooldown_seconds()
            # ...and only when that measurement answers THIS request. A
            # conservative run cannot stand in for a thorough one: thorough
            # confirms 7/7 rather than 5/5, probes three needle offsets rather
            # than one, and is the only mode that runs the sweep -- so it is
            # the only mode that can produce a recommended_config at all.
            # Reusing across that boundary withholds exactly what was asked for
            # while reporting success.
            if recent is not None and not at_least_as_thorough(
                recent.mode, body.mode
            ):
                recent = None
            if recent is not None and started is not None and ttl > 0:
                expires = started + timedelta(seconds=ttl)
                if expires > datetime.now(UTC):
                    # 200, not 202: nothing was accepted for processing. The
                    # caller is being handed a measurement that already exists.
                    response.status_code = 200
                    return StressAccepted(
                        run_id=recent.id,
                        model_id=model_id,
                        mode=recent.mode,  # type: ignore[arg-type]
                        status=recent.status,
                        reused=True,
                        estimate_minutes=0,
                        fingerprint=fp.value,
                        disclaimer=STRESS_DISCLAIMER,
                        cooldown_expires_at=expires.isoformat(),
                    )

    run_id = await _start_run(
        request, model_id, mode=body.mode, force=body.force, model=model
    )
    response.status_code = 202
    return StressAccepted(
        run_id=run_id,
        model_id=model_id,
        mode=body.mode,
        status="running",
        reused=False,
        estimate_minutes=MODE_ESTIMATE_MINUTES[body.mode],
        fingerprint=fp.value,
        disclaimer=STRESS_DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# GET /api/models/{id}/capabilities
# ---------------------------------------------------------------------------


def _run_summary(row: StressRunRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "mode": row.mode,
        "status": row.status,
        "fingerprint": row.fingerprint,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "truncated_by": row.truncated_by,
        "non_monotone": row.non_monotone,
        "traffic_observed": row.traffic_observed,
        "loads_since_measurement": row.loads_since_measurement,
        "last_error": row.last_error,
        "publishable": _publishable(row),
        # Mid-run phase, step count and ETA. The one field here a RUNNING run
        # needs: everything else in this summary is about a finished one, and
        # observations are not written until the end.
        "progress": row.progress,
    }


@router.get("/{model_id}/capabilities")
async def model_capabilities(
    model_id: str, request: Request, _user: str = Depends(require_operator)
) -> dict[str, Any]:
    """The full operator-facing record: limits with provenance, history, advice."""
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get(model_id)
        if model is None:
            raise HTTPException(404, "not found")
        all_models = await ModelRepo(db).list_all()
        fp = await compute_fingerprint(request.app.state, model, all_models=all_models)
        record = await published_record(
            db, model, fp.value, hardware_incomplete=fp.hardware_incomplete
        )
        history = await StressRunRepo(db).list_for_model(model_id, limit=20)

    leases = getattr(request.app.state, "stress_leases", None)
    active_run = next((r for r in history if r.status == "running"), None)
    cooldown_until = None
    if record.provenance == "measured" and record.run is not None:
        started = _parse_sqlite_ts(record.run.started_at)
        ttl = cooldown_seconds()
        if started is not None and ttl > 0:
            cooldown_until = (started + timedelta(seconds=ttl)).isoformat()

    return {
        "model_id": model.id,
        "served_model_name": model.served_model_name,
        "status": model.status,
        "fingerprint": fp.value,
        "gpu_probe_error": fp.probe_error,
        "hardware_incomplete": fp.hardware_incomplete,
        "provenance": record.provenance,
        "stale_reason": record.stale_reason,
        "limits": record.limits,
        # WHICH run these limits came from. The published record is keyed on
        # the fingerprint, not on the latest run, so after a run that measured
        # nothing publishable `limits` may still hold an older run's numbers.
        # Without this the UI cannot tell "here is what your run found" from
        # "here is what some earlier run found", and merging them onto the
        # just-finished run would attribute one run's numbers to another.
        "record_run_id": record.run.id if record.run is not None else None,
        "client_effective_concurrency": _client_effective_concurrency(record),
        "declared": {
            "max_model_len": model.max_model_len,
            "max_batch_size": model.max_batch_size,
        },
        # Operator-facing ONLY. §6.2: it names a configuration the model is not
        # currently running, so applying it is an explicit action that reloads
        # the engine and invalidates the live measurement by construction.
        "recommended_config": (
            record.run.recommended_config if record.run is not None else None
        ),
        "active_run_id": active_run.id if active_run is not None else None,
        "lease_held": bool(leases.is_held(model_id)) if leases is not None else False,
        "cooldown_expires_at": cooldown_until,
        "cooldown_ttl_s": cooldown_seconds(),
        "runs": [_run_summary(r) for r in history],
        "modes": {m: MODE_ESTIMATE_MINUTES[m] for m in MODE_ESTIMATE_MINUTES},
        "disclaimer": STRESS_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Used by GET /v1/models
# ---------------------------------------------------------------------------


async def warden_blocks_for(
    app_state: Any, db, models: list[ModelRow], *, all_models: list[ModelRow]
) -> dict[str, dict]:
    """``model_id -> warden block`` for the loaded models the caller will list.

    ``all_models`` is separate from ``models`` on purpose: the neighbour set is
    a property of the machine, not of what this token is allowed to see, so
    filtering it would make the fingerprint depend on the caller's permissions
    and no two clients would agree on whether a measurement was current.

    Fails open per model: a model whose fingerprint cannot be computed simply
    gets no ``warden`` key, because ``/v1/models`` is a data-plane endpoint and
    must not acquire a new failure mode for an advisory field.
    """
    out: dict[str, dict] = {}
    repo = StressRunRepo(db)
    for model in models:
        try:
            # A model that has never been stress-tested cannot produce a
            # warden block, so ask the cheap indexed question BEFORE the
            # fingerprint: computing one costs an nvidia-smi snapshot, and
            # /v1/models must not grow that cost for the overwhelmingly
            # common case of a warden where nobody has run a test.
            if not await repo.list_for_model(model.id, limit=1):
                continue
            fp = await compute_fingerprint(app_state, model, all_models=all_models)
            record = await published_record(
                db, model, fp.value, hardware_incomplete=fp.hardware_incomplete
            )
            out[model.id] = warden_block(record)
        except Exception:  # noqa: BLE001 — advisory metadata never breaks /v1
            logger.exception("stress: could not build the warden block for %s", model.id)
    return out


@router.post("/{model_id}/stress/apply")
async def apply_stress_recommendation(
    model_id: str,
    body: StressApplyRequest,
    request: Request,
    _user: str = Depends(require_operator),
) -> dict[str, Any]:
    """Apply a measured `max_model_len` and reload the engine (design §6.4).

    The only action in this feature that changes the model rather than
    observing it. Four steps, in this order and for this reason: **every
    refusal is raised before anything is torn down**; then the engine is
    unloaded, because the row cannot be written while it serves; then the row is
    written; then the engine is restarted, because a new context takes effect
    only on a fresh start.

    The first of those is not an optimisation, it is the rule this route learned
    the hard way: *never put a destructive step before a validation that can fail
    for reasons unrelated to the request.* `apply_model_change` validates the
    whole merged row, so it can refuse over a column nobody touched here -- an
    `extra_env` key a pre-#261 PATCH wrote, for instance, which loads and serves
    perfectly well because `filter_extra_env` drops it at launch. With the
    validation after the unload, that stale column turned "Apply and reload" into
    an outage: 422 with the engine already down, `status` committed as `pulled`,
    and no restart. The watchdog does not recover such a row -- `pulled` with no
    `prior_status` is exactly what an operator unloading on purpose looks like --
    so the model stayed down until a human noticed.

    Done server-side rather than as three calls from the browser: a client that
    unloads, then patches, then closes its tab leaves the model unloaded and an
    operator wondering why. Here the sequence either completes or reports where
    it stopped.
    """
    if not body.acknowledge_disruption:
        raise _conflict(
            "disruption_not_acknowledged",
            "applying a setting reloads the engine; this model will stop "
            "serving until it comes back. Resend with acknowledge_disruption=true",
        )

    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get(model_id)
        if model is None:
            raise HTTPException(404, "not found")

        run = await StressRunRepo(db).get(body.run_id)
        if run is None or run.model_id != model_id:
            raise HTTPException(404, "run not found for this model")

        all_models = await ModelRepo(db).list_all()
        fp = await compute_fingerprint(request.app.state, model, all_models=all_models)

        try:
            plan = plan_apply(
                run_status=run.status,
                run_fingerprint=run.fingerprint,
                current_fingerprint=fp.value,
                recommended_config=run.recommended_config,
                current_max_model_len=model.max_model_len,
            )
        except ApplyRefused as exc:
            raise _conflict(exc.reason, exc.detail) from exc

        # Every refusal the write can raise, raised now -- while the engine is
        # still serving and a refusal costs the operator nothing but the error
        # message. See this route's docstring for what happened when this ran
        # after the unload instead.
        #
        # `assume_unloaded=True` because the model IS loaded here and the unload
        # below is this route's own next step: the loaded guard is the one check
        # that is legitimately false now and true by the time the write runs.
        # Everything else -- the merged row's own rules, the GPU allowlist, the
        # principal -- would answer the same either way.
        await dry_run_model_change(
            db,
            principal=principal_of(request),
            current=model,
            changes={"max_model_len": plan.max_model_len},
            assume_unloaded=True,
        )

    # A stress run in flight is already driving the engine, and reloading
    # underneath it would corrupt its measurement and race its own reloads.
    leases = getattr(request.app.state, "stress_leases", None)
    if leases is not None and leases.is_held(model_id):
        raise _conflict(
            "run_already_active",
            "a stress run holds this model; applying would reload the engine "
            "underneath it",
            run_id=leases.holder(model_id),
        )

    from app.runtime.watchdog import _restart  # noqa: PLC0415

    sup = request.app.state.supervisor
    try:
        await sup.unload(model_id, force=True)
    finally:
        # Record the unload in a `finally`, because `Supervisor.unload` releases
        # the handle, the port and the GPU claim in a `finally` of its own: the
        # engine is gone whether it returned or raised, so `pulled` is the true
        # status on both paths. Needed at all because `Supervisor.unload` does
        # not touch `models.status` -- the unload ROUTE writes the terminal
        # status itself -- so without this the row went on claiming `loaded`
        # with no process behind it until the restart below fixed it.
        async with open_db(settings.db_path) as db:
            await ModelRepo(db).update_status(model_id, "pulled")

    late_refusal: ModelChangeRefused | None = None
    async with open_db(settings.db_path) as db:
        # Through the one operator-facing writer (#265), not a raw UPDATE. This
        # was the sixth producer of a ModelRow and the only one that then
        # RESTARTED the engine off what it wrote, with no validation of any
        # kind: `plan_apply` bounds the number, but nothing checked that the
        # rest of the row it was about to serve still satisfied its own rules.
        row = await ModelRepo(db).get(model_id)
        if row is None:
            raise HTTPException(404, "not found")
        try:
            await apply_model_change(
                db,
                principal=principal_of(request),
                current=row,
                changes={"max_model_len": plan.max_model_len},
            )
        except ModelChangeRefused as exc:
            # The dry run passed, so reaching here means the row changed
            # underneath this request -- the dry run shrinks that window to a
            # race, it cannot remove it. The engine is down either way, so bring
            # it back on the configuration still in the row rather than leaving
            # the operator with a refusal and a dead model. Handled outside this
            # connection, because `_restart` opens its own.
            late_refusal = exc

    if late_refusal is not None:
        logger.warning(
            "stress: apply refused after the unload for %s; restarting on the "
            "unchanged row", model_id,
        )
        await _restart(settings, request.app.state, model_id, None)
        raise late_refusal

    # overrides=None deliberately: the sweep leaves transient per-load
    # overrides behind, and reloading with them would serve a configuration
    # that is neither the measured one nor the one now in the row.
    outcome = await _restart(settings, request.app.state, model_id, None)
    logger.info(
        "stress: applied max_model_len=%s to %s (was %s) -> %s",
        plan.max_model_len, model_id, plan.previous, outcome,
    )
    return {
        "model_id": model_id,
        "applied": {"max_model_len": plan.max_model_len},
        "previous": {"max_model_len": plan.previous},
        "status": outcome,
        # The measurement described the configuration just replaced, so it no
        # longer describes this model. Saying so is the honest half of applying.
        "note": (
            "the measurement that produced this setting described the previous "
            "configuration; re-run the stress test to measure the new one"
        ),
    }
