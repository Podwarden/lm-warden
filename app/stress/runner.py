"""The stress-test runner (design §9).

One in-process asyncio task that composes the modules around it. There is no
subprocess, no auth token of its own, no PID file, no reattach and no resume.
Those are not omissions: they are the list of things that killed the
predecessor. ``app/bench/`` did limit-finding with a CLI subprocess, SSE replay
with binary index sidecars, PID-file reattach, cooperative pause/resume and
resume-by-matrix-hash, and was deleted in ``01031a5`` — 29,048 lines across 175
files. A run here is minutes to hours and dies with the warden; if it dies, you
re-run it.

Everything in this module is composition. The oracle, the gates, the
classifier, the search, the deadline policy, the lease, the sweep planner and
the fingerprint are each their own module with their own tests, and this file
is the thing that sequences them against a live engine.

Four behaviours here are load-bearing and each exists because getting it wrong
produces a number that is confidently wrong rather than an error:

**The target is re-resolved after every restart.** ``_restart`` allocates a NEW
port (``watchdog.py:579``). A cached port means every probe after the first
crash connects to nothing, is classified CRASHED, and death-spirals through the
crash budget in seconds — turning one real crash into an exhausted budget and a
published limit far below the truth. :meth:`_target` therefore asks the
supervisor every single time and never caches.

**The real ``served_model_name`` goes on the wire.** An unknown model name is
not an error: llama-server b10731 returns 200 with empty content and
``finish_reason: "length"`` (recorded, ``tests/fixtures/stress/llamacpp_recorded.json``).
That trips REASONING_ONLY and ABRUPT, so a harness that sends the model *row id*
publishes its own bug as model degradation.

**Neighbours are watched but never touched.** Co-residency is normal, a
crashing engine can take a co-resident one down with it, and the co-resident
model holds no lease. We poll their health between probes and abort the run if
one goes unhealthy — and we never take a lease on them, because their recovery
is the watchdog's job and standing it down for a model we are not testing is
strictly worse than doing nothing.

**The prefix-cache defeat is verified, not assumed.** vLLM's cache is
block-aligned from position 0 and on by default; concurrent probes sharing a
prefix get their KV deduplicated and the concurrency axis then reports a number
no real traffic can reach. Probes carry unique preambles, and the engine's own
``cached_tokens`` is checked afterwards — if it is high, the axis is measuring
deduplicated KV and its number is withheld rather than published.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import dataclasses
import json
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.db.repos.stress_runs import StressRunRepo, StressRunRow
from app.proxy.runaway import RunawayDetector
from app.stress import search, sweep
from app.stress.deadline import DeadlinePolicy
from app.stress.fingerprint import Subject, fingerprint
from app.stress.gates import Gate, Reference, budget_starved, evaluate
from app.stress.lease import LeaseRegistry
from app.stress.outcomes import Class, Observables, Outcome, RefusalEvidence, classify
from app.stress.probes import (
    BASELINE_TOKENS,
    Probe,
    baseline_tokens_for,
    budget_cap_for,
    build_suite,
    context_from_engine,
    escalate_budget,
    grade_json_equal,
    grade_numeric,
    grade_regex,
    grade_substring,
    make_needle_prompt,
)
from app.stress.progress import Progress
from app.stress.quarantine import confirm_exclusion, ladder_probes, quarantined
from app.stress.stream import StreamAccumulator

log = logging.getLogger(__name__)


def algorithm_version() -> int:
    """The harness identity component of the fingerprint (design §7.3).

    Declared in ``app/stress/routes_api.py`` and read through here rather than
    duplicated, because a READER has to compute the same fingerprint without
    importing this module. Two constants that drift by one would make every
    stored measurement permanently unmatchable — recorded, never published, and
    with nothing anywhere raising to say why.

    Imported lazily so the runtime package does not depend on the API package
    at import time, matching ``watchdog._restart``'s reason for the same trick.
    """
    from app.stress.routes_api import ALGORITHM_VERSION  # noqa: PLC0415

    return ALGORITHM_VERSION


#: Renewed well inside ``lease.DEFAULT_TTL_S`` (60s), so a runner that dies
#: returns the model to the watchdog within a minute rather than within a run.
HEARTBEAT_S = 15.0

#: The length ladder only varies probes whose prompt actually scales with L.
#: Running the fixed-size probes at every rung would triple the cost of the
#: axis and measure the same thing every time. The FULL suite runs at the
#: baseline and again at the confirmed limit (design §8).
LADDER_PROBE_IDS = ("needle", "summary")

#: Above this fraction of prompt tokens served from cache, the concurrency
#: number is meaningless — the engine deduplicated the KV the axis exists to
#: exhaust. Not a tuning knob: any sizeable reuse invalidates the measurement.
CACHED_TOKENS_MAX_RATIO = 0.25

DEADLINE_FLOOR_S = 30.0
DEADLINE_CAP_S = 900.0
DEADLINE_FACTOR = 4.0

#: Design §5.8. Two, then stop — looping here IS the crash loop the restart
#: budget exists to prevent.
RECOVERY_ATTEMPTS = 2

#: Observations are for forensics, not for accounting. A thorough run can emit
#: thousands; keeping the tail bounded keeps the row a row.
MAX_OBSERVATIONS = 400

DEFAULT_SAFETY_FACTOR = 0.9


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeProfile:
    """What a mode costs and how hard it pushes (design §8).

    ``conservative`` has a crash budget of ZERO — it aborts on the first crash.
    That is not the same as promising it will not crash: the 2026-09-03 Quadro
    measurement died at ~2.2k tokens with no refusal and no degradation first,
    so no mode can make that promise and this table does not pretend to.
    """

    name: str
    crash_budget: int
    pass_n: int
    confirm_m: int
    needle_offsets: tuple[float, ...]
    run_sweep: bool
    safety_factor: float = DEFAULT_SAFETY_FACTOR


PROFILES: dict[str, ModeProfile] = {
    "conservative": ModeProfile("conservative", 0, 3, 5, (0.5,), False),
    "quick": ModeProfile("quick", 2, 3, 5, (0.5,), False),
    "thorough": ModeProfile("thorough", 4, 3, 7, (0.1, 0.5, 0.9), True),
}


# ---------------------------------------------------------------------------
# the seam onto the live warden
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    host: str
    port: int

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class Caps:
    """The hard bounds the search may not exceed.

    ``max_model_len`` is the allocated KV pool. Probing past it cannot find a
    runtime limit — both engines size the pool at startup, so an oversized
    prompt is refused, not survived. It is the definitional refusal boundary
    (design §5.3 test 1).
    """

    max_model_len: int | None = None
    max_num_seqs: int | None = None
    model_ceiling: int | None = None
    vram_budget_bytes: int | None = None
    kv_bytes_per_token: int | None = None


class EngineControl(Protocol):
    """Everything the runner is allowed to know about the live system.

    A protocol rather than a direct reach into ``app_state`` because that is
    what makes the whole runner testable with no GPU: the tests drive a fake
    engine through this same surface, so the code path under test is the real
    one rather than a rehearsal of it.
    """

    served_model_name: str

    def target(self) -> Target | None: ...
    def caps(self) -> Caps: ...
    def gpu_uuids(self) -> Sequence[str]: ...
    def subject(self) -> Subject: ...
    def process_alive(self) -> bool | None: ...
    def returncode(self) -> int | None: ...
    def host_oom_counters_moved(self) -> bool: ...
    def foreign_requests(self) -> int: ...
    def reset_budget(self) -> None: ...

    async def health(self) -> bool: ...
    async def row_status(self) -> str: ...
    async def neighbour_health(self) -> dict[str, bool]: ...
    async def restart(self) -> None: ...
    async def wait_for_health(self) -> bool: ...
    async def warmup(self) -> bool: ...
    async def reload_with(self, *, max_model_len: int) -> bool: ...


# ---------------------------------------------------------------------------
# control flow
# ---------------------------------------------------------------------------


class StressAborted(Exception):
    """Class 0. The run lost control of its own conditions; publish nothing.

    ``reason`` is machine-readable and lands in ``limits.aborted_reason``; the
    ROW status is always the schema's ``aborted``. Those are separate on
    purpose: ``stress_runs.status`` has a CHECK constraint over five values
    (``0029_stress_runs.sql``), and widening it to carry a taxonomy would mean
    recreating a table on a live volume every time the taxonomy grew.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


class _AxisTruncated(Exception):
    """This axis stops here, but what it already measured is still true."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def apass_n(trial: Callable[[], Awaitable[bool]], *, n: int) -> bool:
    """Async twin of :func:`app.stress.search.pass_n`, and deliberately tiny.

    ``search.pass_n`` is synchronous, which is exactly why the search module is
    provable without a network. Where the ALGORITHM drives the probes it is run
    in a worker thread and marshalled back onto the loop (:class:`_Bridge`), so
    there is one implementation. That bridge cannot be used from inside a probe
    — the worker thread is blocked waiting on it and would deadlock on itself —
    and the refusal confirmation needs replication from exactly there. Aborts on
    the first failure for the same reason the original does: one failure has
    already decided it.
    """
    for _ in range(n):
        if not await trial():
            return False
    return True


#: How often a parked bridge worker re-checks whether it may still be waiting.
#: Small enough that a cancel is felt within a tick; large enough that a
#: healthy probe -- which resolves on its own long before then -- never pays
#: for the polling.
_BRIDGE_POLL_S = 1.0


class _BridgeStopped(Exception):
    """The run that owned this bridge is gone, so the worker is bailing.

    Raised by ``_Bridge.call`` when the run is cancelled (``stop``), the loop it
    targets is closed, or the call outlives its bound, while a bisect worker is
    parked on a probe. The probe itself may still be running on the loop -- a
    thread cannot be recalled -- so the worker stops waiting for it rather than
    pinning its non-daemon thread for the life of the process.
    """


class _Bridge:
    """Runs the synchronous search algorithms against asynchronous probes.

    ``asyncio.to_thread`` puts ``search.bisect`` on a worker thread; each oracle
    call it makes is handed back to the event loop here and the thread blocks on
    the result. The loop is free for the duration, so the probe runs normally.
    The alternative — an async fork of ``search.py`` — would duplicate the one
    piece of this feature whose correctness argument is a proof about
    ``Pr[PASS_N] = (1-p)**N``, and the two copies would drift.

    The wait is bounded, not blind. A thread cannot be cancelled, so a probe
    that never returns (a wedged engine) would otherwise pin this worker's
    non-daemon thread for the life of the process. The worker therefore polls
    the future one tick at a time and bails -- cancelling the probe it was
    waiting on -- when the run is cancelled (``stop``), the loop closes, or the
    whole call outlives ``timeout_s``.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        stop: threading.Event,
        timeout_s: float,
    ) -> None:
        self._loop = loop
        self._stop = stop
        self._timeout_s = timeout_s

    def call(self, coro: Coroutine[Any, Any, Any]) -> Any:
        if self._stop.is_set():
            coro.close()
            raise _BridgeStopped("the run ended before this probe could start")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        deadline = time.monotonic() + self._timeout_s
        try:
            while True:
                if self._stop.is_set() or self._loop.is_closed():
                    raise _BridgeStopped("the run ended while this probe was in flight")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _BridgeStopped("the probe outlived its bound")
                try:
                    return fut.result(timeout=min(remaining, _BRIDGE_POLL_S))
                except concurrent.futures.TimeoutError:
                    continue
        finally:
            fut.cancel()


@dataclass
class _ProbeRun:
    """One probe execution, with everything the record needs about it."""

    probe_id: str
    outcome: Outcome
    elapsed_s: float
    http_status: int | None
    prompt_tokens: int | None
    cached_tokens: int | None
    output_tokens: int
    content: str
    gates: tuple[Gate, ...]
    #: Kept so a refusal can be re-decided on new evidence without re-probing.
    observables: Observables
    body: object = None
    #: True when OUR max_tokens ended the generation rather than the model
    #: (``gates.budget_starved``). Computed here because it needs the
    #: finish_reason, which the gate tuple cannot carry: ABRUPT fires for both
    #: ``length`` (our ceiling) and ``None`` (a dropped stream), and only the
    #: first is worth re-probing.
    starved: bool = False


#: Worst-first, so a rung with one crash and eleven passes is a crash.
_SEVERITY = {
    Class.INCONCLUSIVE: 0,
    Class.CRASHED: 1,
    Class.TIMEOUT: 2,
    Class.REFUSED: 3,
    Class.BACKPRESSURE: 4,
    Class.ERRORED: 5,
    Class.DEGRADED: 6,
    Class.PASS: 7,
}


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


class StressRunner:
    """One stress run, start to finish.

    Constructed by the API layer and driven by ``asyncio.create_task``. The
    caller keeps the task handle; cancelling it runs the postcondition through
    the ``finally``, which is the only recovery path this feature has.
    """

    def __init__(
        self,
        *,
        control: EngineControl,
        repo: StressRunRepo,
        model_id: str,
        mode: str = "conservative",
        leases: LeaseRegistry | None = None,
        run_id: str | None = None,
        seed: int = 20260903,
        force: bool = False,
        max_concurrency: int | None = None,
        length_floor: int = BASELINE_TOKENS,
        deadline_floor_s: float = DEADLINE_FLOOR_S,
        deadline_cap_s: float = DEADLINE_CAP_S,
    ) -> None:
        self.control = control
        self.repo = repo
        self.model_id = model_id
        self.profile = PROFILES.get(mode) or PROFILES["conservative"]
        self.mode = self.profile.name
        self.leases = leases or LeaseRegistry()
        self.run_id = run_id or uuid.uuid4().hex
        self.seed = seed
        self.force = force
        # Clamped to what this model can actually hold. BASELINE_TOKENS is a
        # harness default, not a fact about any model: a 512-context model
        # cannot have a 1024-token reference condition, and probing there would
        # report our own default as the model's limit.
        self.length_floor = min(
            length_floor, baseline_tokens_for(context_window=control.caps().max_model_len)
        )
        self._max_concurrency_override = max_concurrency

        self.deadline = DeadlinePolicy(
            floor_s=deadline_floor_s, cap_s=deadline_cap_s, factor=DEADLINE_FACTOR
        )

        self._client: httpx.AsyncClient | None = None
        self._inflight: set[asyncio.Task] = set()
        self._references: dict[str, Reference] = {}
        #: probe id -> the max_tokens the baseline settled on. See _calibrate.
        self._budgets: dict[str, int] = {}
        #: probe ids that tripped a gate at the baseline. See app/stress/quarantine.py.
        self._excluded: tuple[str, ...] = ()
        #: Set when the run is cancelled so a parked _Bridge worker can bail. A
        #: thread cannot be cancelled, so the worker polls this instead.
        self._stopped = threading.Event()
        #: Per-probe bound for the bridge. The deadline already caps a single
        #: probe at cap_s, so this is a backstop -- 12x plus slack -- against a
        #: probe that even the deadline fails to recall.
        self._bridge_timeout_s = self.deadline.cap_s * 12 + 60.0
        #: What this run reports about itself while it runs. See progress.py.
        self._progress = Progress()
        #: Context the ENGINE will accept, when the row does not say. See
        #: probes.context_from_engine.
        self._engine_context: int | None = None
        self._observations: deque[dict] = deque(maxlen=MAX_OBSERVATIONS)
        self._limits: dict[str, Any] = {}
        self._recommended: dict | None = None

        self._crashes = 0
        self._slot = 0
        self._truncated_by: str | None = None
        self._non_monotone = False
        self._traffic_observed = False
        self._prefix_cache_suspect = False
        self._last_passing_length: int | None = None
        self._neighbours_at_start: dict[str, bool] = {}
        self._neighbour_down: str | None = None
        self._status_changed = False
        self._limited_by: str | None = None
        self._gate_tripped: str | None = None
        self._confirming = False

    # -- entry point ------------------------------------------------------

    async def run(self) -> StressRunRow:
        subject = self.control.subject()
        fp = fingerprint(subject)
        row = StressRunRow(
            id=self.run_id,
            model_id=self.model_id,
            fingerprint=fp,
            mode=self.mode,
            seed=self.seed,
            probe_suite_hash=subject.probe_suite_hash,
            algorithm_version=subject.algorithm_version,
        )
        await self.repo.insert(row)

        # Taken BEFORE anything that can crash the engine, and released in the
        # finally below even when the body raises. Between those two points the
        # watchdog stands down for this model and recovery is ours.
        self.leases.take(self.model_id, run_id=self.run_id, gpu_uuids=self.control.gpu_uuids())
        heartbeat = asyncio.create_task(self._heartbeat())

        status = "completed"
        last_error: str | None = None
        try:
            self._client = httpx.AsyncClient(timeout=None)
            await self._preflight()
            await self._baseline()
            await self._axis_concurrency()
            await self._axis_length()
            await self._sweep()
        except _AxisTruncated as trunc:
            # Escaped past both axes -- the baseline itself hit the budget or
            # wedged. Publish what there is, which may be nothing, but say WHY
            # rather than filing it as an unexplained failure.
            self._truncated_by = trunc.reason
            last_error = f"run truncated: {trunc.reason}"
        except StressAborted as exc:
            status = "aborted"
            self._limits["aborted_reason"] = exc.reason
            last_error = f"{exc.reason}: {exc.detail}" if exc.detail else exc.reason
            log.info("stress %s: %s", self.run_id, last_error)
        except asyncio.CancelledError:
            status = "interrupted"
            last_error = "run cancelled"
            # Release any bisect worker parked on the bridge: it polls this
            # flag and bails within a tick instead of pinning its thread.
            self._stopped.set()
            raise
        except Exception as exc:  # noqa: BLE001 — recorded, never swallowed
            # A harness exception is class 0 like any other loss of control: the
            # conditions were not what we thought, so nothing here is publishable.
            status = "aborted"
            self._limits["aborted_reason"] = "harness_exception"
            last_error = f"{type(exc).__name__}: {exc}"
            log.exception("stress %s failed", self.run_id)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            recovered, recovery_error = await self._postcondition()
            if not recovered:
                status = "failed_unrecovered"
                last_error = recovery_error or last_error
            self.leases.release(self.model_id)
            self.control.reset_budget()
            await self.repo.finish(
                self.run_id,
                status=status,
                limits=self._limits or None,
                observations=list(self._observations) or None,
                recommended_config=self._recommended,
                truncated_by=self._truncated_by,
                non_monotone=self._non_monotone,
                traffic_observed=self._traffic_observed,
                last_error=last_error,
            )

        got = await self.repo.get(self.run_id)
        if got is None:  # pragma: no cover — we inserted it above
            raise RuntimeError(f"stress run row {self.run_id} vanished")
        return got

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            self.leases.heartbeat(self.model_id)
            # Piggy-backed on the lease tick rather than given a timer of its
            # own: it is the same cadence, and a second timer would be a second
            # thing to cancel correctly on abort. A failure to report progress
            # must never take down a run that is otherwise fine, so this
            # swallows everything.
            try:
                await self.repo.update_progress(self.run_id, self._progress.snapshot())
            except Exception:  # noqa: BLE001 — cosmetic; never fatal
                log.debug("stress: progress write failed", exc_info=True)

    # -- preflight --------------------------------------------------------

    async def _preflight(self) -> None:
        """Refuse to start a run whose conditions are already compromised."""
        if self.control.target() is None:
            raise StressAborted("aborted_operator_action", "no engine is loaded")

        # `models.max_model_len` is optional and commonly unset -- nobody has to
        # fill it in. When it is NULL every budget falls back to BUDGET_FLOOR,
        # which aborted three runs on 2026-09-04 while the engine was happily
        # serving 131,072 tokens. Ask the engine, which is the authority on what
        # it will accept. Best-effort: a run must not fail because a diagnostic
        # endpoint is missing.
        self._engine_context = await self._resolve_engine_context()

        # Re-clamp now that the engine has spoken. The constructor could only
        # see the row, so a model with a small engine context and no
        # max_model_len set would have kept the full default baseline and had
        # every probe refused. Only ever shrinks.
        self.length_floor = min(
            self.length_floor,
            baseline_tokens_for(context_window=self._context_window()),
        )

        # Per-model, never the global count: RequestRegistry.count() spans every
        # model, so a global guard blocks a run on model A because someone is
        # chatting with model B on another GPU.
        if self.control.foreign_requests() > 0:
            if not self.force:
                raise StressAborted("aborted_traffic", "another client is using this model")
            # KV is a shared pool and traffic confounds every latency gate, so
            # there is no partial credit — the whole run is tainted.
            self._traffic_observed = True

        self._neighbours_at_start = await self.control.neighbour_health()
        self._limits["neighbours"] = sorted(self._neighbours_at_start)

    # -- baseline ---------------------------------------------------------

    async def _resolve_engine_context(self) -> int | None:
        """Ask the engine what context it is running with. Never fatal.

        Two endpoints because the two engines answer in different places, and
        llama.cpp's own ``/v1/models`` is Ollama-shaped and carries no context
        at all -- so a miss on one is not evidence of anything.
        """
        target = self.control.target()
        if target is None or self._client is None:
            return None
        for path in ("/props", "/v1/models"):
            try:
                r = await self._client.get(f"{target}{path}", timeout=10.0)
                if r.status_code != 200:
                    continue
                found = context_from_engine(r.json())
            except Exception:  # noqa: BLE001 — diagnostic only
                continue
            if found:
                log.info("stress: engine reports a context of %d tokens (%s)", found, path)
                return found
        return None

    def _context_window(self) -> int | None:
        """The row's setting, else what the engine reports.

        The row wins when set: it is what the operator asked for, and the engine
        may have been given something else entirely.
        """
        return self.control.caps().max_model_len or self._engine_context

    async def _baseline(self) -> None:
        """The reference condition, C=1 and L=1k (design §4.3).

        Every later SHORT verdict is relative to this. A model that answers
        tersely when perfectly healthy is not degraded, and an absolute floor
        would libel it — so a weak baseline is recorded and judged against
        rather than treated as a failure.
        """
        suite = build_suite(run_id=self.run_id, seed=self.seed, approx_tokens=self.length_floor)
        self._progress.enter("baseline", total=len(suite), note="calibrating probe budgets")
        permissive = Reference(output_tokens=0, checks_length=False)
        gates_at_baseline: dict[str, list[str]] = {}
        starved_probes: list[str] = []

        for probe in suite:
            self._progress.step(note=f"baseline: {probe.id}")
            probe, run = await self._calibrate(probe, ref=permissive)
            if run.outcome.cls is Class.CRASHED:
                raise StressAborted(
                    "aborted_inconclusive", "engine crashed at the baseline condition"
                )
            # A failure here retires the probe for the whole run, so confirm
            # it before believing it. `search.py` replicates every ladder
            # verdict because one sample near a threshold is a coin flip; the
            # baseline shapes the entire run and was deciding on one. A dropped
            # stream alone trips ABRUPT via `finish_reason: None`, and that was
            # enough to cost the run an oracle -- with only two ladder probes,
            # two such mistakes abort it outright.
            if run.gates:
                self._progress.step(note=f"baseline: re-checking {probe.id}")
                second = await self._probe_with_recovery(probe, ref=permissive)
                if second.outcome.cls is Class.CRASHED:
                    raise StressAborted(
                        "aborted_inconclusive",
                        "engine crashed at the baseline condition",
                    )
                if not confirm_exclusion(run.gates, second.gates):
                    # The first reading was noise. Judge against the clean one:
                    # a reference taken from a failed probe would libel every
                    # later verdict measured against it.
                    run = second

            if run.starved:
                starved_probes.append(probe.id)
            # The reference is the ANSWER length, not the prompt length.
            self._references[probe.id] = Reference(
                output_tokens=run.output_tokens,
                checks_length=probe.checks_length,
            )
            if run.gates:
                gates_at_baseline[probe.id] = [g.value for g in run.gates]

        # No probe produced an answer even at the ceiling, so there is no
        # reference to judge anything against. Every later verdict would be an
        # artefact of this, and the run would spend hours reloading engines to
        # rediscover it at each candidate. Say so instead: the earlier
        # behaviour was to complete with `no_confirmed_value`, which reads as
        # "the model is bad" when the truth is "the harness never got an
        # answer out of it".
        self._excluded = quarantined(gates_at_baseline)

        if len(starved_probes) == len(suite):
            raise StressAborted(
                "aborted_no_baseline",
                "no probe produced an answer within the budget this model's "
                "context allows, so nothing is measurable",
            )

        self._limits["baseline"] = {
            "approx_tokens": self.length_floor,
            "answer_tokens": {pid: ref.output_tokens for pid, ref in self._references.items()},
            "gates_tripped": gates_at_baseline,
            # What the budget had to be raised to, per probe. Published because
            # it is the single number that explains an otherwise baffling
            # baseline, and because a large value is itself a finding about the
            # model.
            "max_tokens": {pid: mt for pid, mt in sorted(self._budgets.items())},
            "starved_probes": starved_probes,
            # Probes that failed HERE cannot locate a threshold: they read the
            # same at the reference length as at the axis maximum, so they veto
            # every rung without finding anything. Recorded, not silently
            # dropped -- "summary was excluded because it tripped repeating at
            # the baseline" is a finding an operator needs.
            "excluded_probes": list(self._excluded),
        }

        if not ladder_probes(LADDER_PROBE_IDS, self._excluded):
            raise StressAborted(
                "aborted_no_baseline",
                "every ladder probe failed at the baseline condition, so "
                "nothing is left that can tell where the model breaks",
            )

    async def _calibrate(self, probe: Probe, *, ref: Reference) -> tuple[Probe, _ProbeRun]:
        """Find a ``max_tokens`` at which this probe can actually answer.

        The suite budgets ~4x each probe's ANSWER, which is sound for a model
        that answers directly and impossible for one that reasons first: on
        `gpt-oss-20b-gguf` the arithmetic probe needs 56 completion tokens to
        add two numbers, against a budget of 16. Every gate then fires
        truthfully about an artefact of our own configuration, and the first
        production run (2026-09-04) failed all six probes at the baseline on a
        model that was perfectly healthy.

        Escalation happens ONCE, here, and the settled budget is reused for the
        whole run — a per-probe retry in the hot path would make the length
        sweep pay for it at every candidate, and would let the budget drift
        between points that are supposed to be comparable.
        """
        run = await self._probe_with_recovery(probe, ref=ref)
        while run.starved:
            # Derived per model and per probe, never a constant: the answer has
            # to fit in the context alongside the prompt, and that arithmetic
            # is different for a 2k model and a 262k one. `prompt_tokens` is
            # the engine's own count from the probe we just ran, so this is
            # measured rather than estimated.
            cap = budget_cap_for(
                context_window=self._context_window(),
                prompt_tokens=run.prompt_tokens or 0,
            )
            bigger = escalate_budget(probe.max_tokens, cap=cap)
            if bigger is None:
                break
            probe = dataclasses.replace(probe, max_tokens=bigger)
            run = await self._probe_with_recovery(probe, ref=ref)
        self._budgets[probe.id] = probe.max_tokens
        return probe, run

    def _calibrated(self, suite: list[Probe]) -> list[Probe]:
        """Re-apply the baseline's settled budgets to a freshly built suite.

        ``build_suite`` is called again at every length and concurrency point,
        so without this the calibration would be silently discarded after the
        baseline and every later probe would starve again.
        """
        return [
            dataclasses.replace(p, max_tokens=self._budgets[p.id]) if p.id in self._budgets else p
            for p in suite
        ]

    # -- axis A2: concurrency ---------------------------------------------

    async def _axis_concurrency(self) -> None:
        """Capacity only. Quality is NOT comparable across concurrency.

        Continuous batching changes floating-point reduction order, so the same
        probe at ``temperature: 0`` produces different text at C=1 and C=12 for
        reasons that have nothing to do with degradation. Judging this axis by
        comparing answers to a C=1 baseline would measure the batcher.
        """
        cap = self._max_concurrency()
        if cap <= 1:
            return
        rungs = search.ladder(start=1, cap=cap)
        result: dict[str, Any] = {"axis_max": cap, "unit": "concurrent_requests"}

        try:
            best, first_failure = await self._bracket(rungs, self._trial_concurrency)
            if first_failure is None:
                confirmed = best
            else:
                loop = asyncio.get_running_loop()
                bridge = _Bridge(loop, stop=self._stopped, timeout_s=self._bridge_timeout_s)
                bracket = search.Bracket(low=best, high=first_failure)
                confirmed = await asyncio.to_thread(
                    search.bisect,
                    bracket,
                    lambda x: bridge.call(self._trial_concurrency(x)),
                    n=self.profile.pass_n,
                    steps=6,
                )
            result["raw_confirmed"] = confirmed
            result["first_observed_failure"] = first_failure
        except _AxisTruncated as trunc:
            self._truncated_by = trunc.reason
            result["truncated_by"] = trunc.reason
            self._limits["concurrency"] = result
            return

        if self._prefix_cache_suspect:
            # The engine served a large share of these prompts from cache, so
            # this number describes deduplicated KV rather than concurrency.
            # Publishing it would overstate real capacity, which is the exact
            # direction this design refuses to be wrong in.
            result["publishable"] = False
            result["invalid_reason"] = "prefix_cache_not_defeated"
        else:
            result["publishable"] = True
            result["value"] = int(confirmed * self.profile.safety_factor)
            result["provenance"] = "measured"
            result["oracle"] = self._oracle_record()
        self._limits["concurrency"] = result

    def _max_concurrency(self) -> int:
        if self._max_concurrency_override is not None:
            return self._max_concurrency_override
        return self.control.caps().max_num_seqs or 1

    async def _trial_concurrency(self, c: int) -> bool:
        """C simultaneous probes, each with its own high-entropy preamble.

        The preamble differs per slot AND per attempt. Per slot so simultaneous
        probes cannot dedupe against each other; per attempt because a repeated
        trial at the same rung would otherwise replay a prompt the engine has
        already cached, and the second attempt would measure the cache.
        """
        probes = [self._concurrency_probe() for _ in range(c)]
        ref = Reference(output_tokens=0, checks_length=False)
        deadline_s = self.deadline.next_deadline()

        tasks = [
            asyncio.create_task(self._execute(p, ref=ref, deadline_s=deadline_s)) for p in probes
        ]
        self._inflight.update(tasks)
        try:
            runs = await asyncio.gather(*tasks)
        finally:
            self._inflight.difference_update(tasks)

        for run in runs:
            self._record(run, axis="concurrency", value=c)
            if (
                run.prompt_tokens
                and run.cached_tokens
                and run.cached_tokens > run.prompt_tokens * CACHED_TOKENS_MAX_RATIO
            ):
                self._prefix_cache_suspect = True

        worst = min(runs, key=lambda r: _SEVERITY[r.outcome.cls])
        await self._react(worst)
        return all(r.outcome.passes_capacity for r in runs)

    def _concurrency_probe(self) -> Probe:
        self._slot += 1
        np = make_needle_prompt(
            run_id=self.run_id,
            seed=self.seed,
            approx_tokens=self.length_floor,
            offset=0.5,
            slot=self._slot,
        )
        return Probe(
            id="concurrency",
            messages=[{"role": "user", "content": np.text}],
            # Inherits the needle probe's settled budget: same shape of task,
            # same answer, so the same reasoning overhead applies. Without this
            # the concurrency axis would starve on a reasoning model even
            # though the baseline had already measured the fix.
            max_tokens=self._budgets.get("needle", 24),
            checks_length=False,
            allow_repetition_gate=False,
            expected=np.needle_value,
            grader="substring",
        )

    # -- axis A1: length --------------------------------------------------

    async def _axis_length(self) -> None:
        """The quality ladder. This is the number the feature exists to produce.

        Phase A brackets geometrically, B bisects on PASS_N inside the bracket,
        C confirms M/M at the accepted value, D probes the axis maximum to catch
        a failure region that is not upward-closed.
        """
        cap = self.control.caps().max_model_len or self.length_floor * 8
        if cap <= self.length_floor:
            return
        rungs = search.ladder(start=self.length_floor, cap=cap)
        # Rungs, then the bracketed bisection (bounded at 6), then the
        # confirmation samples. Phase D is one more. The plan is an upper
        # bound, and overrunning it simply withdraws the ETA.
        self._progress.enter(
            "length",
            total=len(rungs) + 6 + self.profile.confirm_m + 1,
            note="climbing the context ladder",
        )
        result: dict[str, Any] = {"axis_max": cap, "unit": "tokens"}

        try:
            best, first_failure = await self._bracket(rungs, self._trial_length)
            loop = asyncio.get_running_loop()
            bridge = _Bridge(loop, stop=self._stopped, timeout_s=self._bridge_timeout_s)

            if first_failure is not None:
                bracket = search.Bracket(low=best, high=first_failure)
                best = await asyncio.to_thread(
                    search.bisect,
                    bracket,
                    lambda x: bridge.call(self._trial_length(x)),
                    n=self.profile.pass_n,
                    steps=6,
                )

            # Phase C. On failure step down one quantum and re-confirm, twice at
            # most: this is the value that gets published, so it is confirmed to
            # a stricter contour than the one bisection used.
            quantum = max(1, best // 16)
            confirmed = 0
            for _ in range(3):
                if best <= 0:
                    break
                ok = await asyncio.to_thread(
                    search.confirm,
                    lambda b=best: bridge.call(self._trial_length(b)),
                    m=self.profile.confirm_m,
                )
                if ok:
                    confirmed = best
                    break
                best -= quantum
            outcome = search.SearchOutcome(
                confirmed=confirmed, first_failure=first_failure, axis_max=cap
            )
        except _AxisTruncated as trunc:
            self._truncated_by = trunc.reason
            result["truncated_by"] = trunc.reason
            self._limits["quality_context"] = result
            return

        # Phase D. One probe at the axis maximum. If it PASSES while something
        # below it failed, the failure region has a hole, the bracket was locked
        # by a local dip, and Phase C has confidently converged on a value that
        # may be several times too low.
        self._non_monotone = await asyncio.to_thread(
            outcome.is_non_monotone, lambda x: bridge.call(self._trial_length(x))
        )
        outcome = search.SearchOutcome(
            confirmed=outcome.confirmed,
            first_failure=outcome.first_failure,
            axis_max=cap,
            non_monotone=self._non_monotone,
        )

        result.update(
            raw_confirmed=outcome.confirmed,
            first_observed_failure=outcome.first_failure,
            publishable=outcome.publishable(),
            oracle=self._oracle_record(),
        )
        if outcome.publishable() and outcome.confirmed:
            result["value"] = outcome.recommended(self.profile.safety_factor)
            result["provenance"] = "measured"
            result["confidence"] = "confirmed"
            result["limited_by"] = self._limited_by
            result["gate_tripped"] = self._gate_tripped
            # The full suite at the limit, not just the ladder's two probes: the
            # published number is the one an operator will act on.
            await self._full_suite_at(outcome.confirmed)
        else:
            result["invalid_reason"] = (
                "non_monotone" if self._non_monotone else "no_confirmed_value"
            )
        self._limits["quality_context"] = result

    async def _full_suite_at(self, tokens: int) -> None:
        suite = self._calibrated(
            build_suite(run_id=self.run_id, seed=self.seed, approx_tokens=tokens)
        )
        for probe in suite:
            if probe.id in self._excluded:
                continue
            ref = self._references.get(probe.id, Reference(output_tokens=0, checks_length=False))
            # Same reason as the ladder: a budget settled at the baseline can
            # starve a longer prompt, and recording that as a gate failure at
            # the limit would libel the model with our own configuration.
            probe, run = await self._calibrate(probe, ref=ref)
            self._record(run, axis="limit_suite", value=tokens)

    async def _trial_length(self, tokens: int) -> bool:  # noqa: D401
        """One trial at prompt length ``tokens``: does the suite hold up?

        Quality, not capacity. A DEGRADED probe passed the engine and failed the
        model, and this axis is judged by the model — that asymmetry is what
        yields two numbers from one set of probes.
        """
        suite = [
            p
            for p in self._calibrated(
                build_suite(run_id=self.run_id, seed=self.seed, approx_tokens=tokens)
            )
            if p.id in ladder_probes(LADDER_PROBE_IDS, self._excluded)
        ]
        self._progress.step(note=f"probing {tokens:,} tokens")
        for probe in suite:
            ref = self._references.get(probe.id, Reference(output_tokens=0, checks_length=False))
            # _calibrate, not a bare probe: the budget settled at the 1k
            # baseline can starve the SAME probe at a longer rung, because a
            # reasoning model's chain grows with the input. Measured on
            # gpt-oss-20b-gguf (2026-09-04): needle settled at 96 tokens and
            # every failure from 1664 upward carried the starvation signature
            # `abrupt + reasoning_only + wrong`, so the published limit of
            # 1616 tokens was our own budget rather than the model's. Budgets
            # only ever grow and are carried forward, so a later rung is never
            # judged under a smaller budget than an earlier one.
            probe, run = await self._calibrate(probe, ref=ref)
            self._record(run, axis="length", value=tokens)
            if not run.outcome.passes_quality:
                self._limited_by = _limited_by_for(run.outcome)
                self._gate_tripped = run.gates[0].value if run.gates else None
                return False
        self._last_passing_length = max(self._last_passing_length or 0, tokens)
        return True

    # -- phase A ----------------------------------------------------------

    async def _bracket(
        self, rungs: Sequence[int], trial: Callable[[int], Awaitable[bool]]
    ) -> tuple[int, int | None]:
        """Geometric climb, one sample per rung.

        A single sample is deliberate: an error here only mis-sizes the bracket,
        which Phase B recovers. Replicating it would triple the cost of the
        cheapest phase to improve a number nothing publishes.
        """
        best = rungs[0]
        for rung in rungs:
            if not await trial(rung):
                return best, rung
            best = rung
        return best, None

    # -- probing ----------------------------------------------------------

    async def _probe_with_recovery(self, probe: Probe, *, ref: Reference) -> _ProbeRun:
        """One probe, with the retry and recovery policy the taxonomy dictates.

        BACKPRESSURE is retried and never published as a limit — it is transient
        queue depth, structurally identical to a refusal and semantically its
        opposite. ERRORED is retried once, then failed for capacity. Everything
        else is handled by :meth:`_react`.
        """
        run = await self._execute(probe, ref=ref, deadline_s=self.deadline.next_deadline())
        attempts = 0
        while run.outcome.retry and attempts < 2:
            attempts += 1
            if run.outcome.cls is Class.BACKPRESSURE:
                await asyncio.sleep(0.25 * attempts)
            run = await self._execute(probe, ref=ref, deadline_s=self.deadline.next_deadline())
        if run.outcome.cls is Class.BACKPRESSURE:
            raise StressAborted(
                "aborted_inconclusive", "engine kept returning 429; nothing measurable"
            )
        if run.outcome.cls is Class.ERRORED:
            run = await self._confirm_refusal(run)

        await self._react(run)
        return run

    async def _react(self, run: _ProbeRun) -> None:
        """Apply the consequences of an outcome to the run as a whole."""
        cls = run.outcome.cls

        if cls is Class.INCONCLUSIVE:
            raise StressAborted(*self._abort_reason())

        if cls is Class.CRASHED:
            self._crashes += 1
            if self._crashes > self.profile.crash_budget:
                raise _AxisTruncated("crash_budget")
            await self._recover()
            return

        if cls is Class.TIMEOUT:
            self.deadline.record_timeout(run.elapsed_s)
            if self.deadline.wedged():
                # /health is still 200 — that IS the pathology. TIMEOUT spends
                # no crash budget and does not stop the search, so without this
                # the run burns the full deadline on every remaining rung.
                await self._recover()
                raise _AxisTruncated("wedged")
            return

        self.deadline.record_success(run.elapsed_s)
        await self._check_neighbours()

    def _abort_reason(self) -> tuple[str, str]:
        if self._neighbour_down:
            return ("aborted_neighbour_impact", f"neighbour {self._neighbour_down} went unhealthy")
        if self._status_changed:
            return ("aborted_operator_action", "model status changed underneath the run")
        return ("aborted_traffic", "foreign traffic arrived on this model")

    async def _check_neighbours(self) -> None:
        """Poll co-resident models between probes.

        We hold no lease on them and never will: their recovery belongs to the
        watchdog, and standing it down for a model we are not testing would be
        strictly worse than doing nothing. All we can honestly do is notice and
        stop.
        """
        if not self._neighbours_at_start:
            return
        current = await self.control.neighbour_health()
        for model_id, was_healthy in self._neighbours_at_start.items():
            if was_healthy and current.get(model_id) is False:
                self._neighbour_down = model_id
                raise StressAborted(
                    "aborted_neighbour_impact",
                    f"neighbour {model_id} went unhealthy during the run",
                )

    async def _execute(self, probe: Probe, *, ref: Reference, deadline_s: float) -> _ProbeRun:
        """Send one probe engine-direct and classify what came back.

        Engine-direct because ``PriorityScheduler`` admits VW_PROXY_MAX_INFLIGHT
        (default 16) per engine: a concurrency run through ``/v1`` could never
        test past 16 and would be measuring the warden. The target is resolved
        HERE, on every call, so a restart's new port is picked up immediately.
        """
        target = self.control.target()
        if target is None:
            return self._classified(
                probe,
                ref,
                acc=StreamAccumulator(),
                status=None,
                body=None,
                elapsed=0.0,
                timed_out=False,
                engine_healthy=False,
                row_status=await self.control.row_status(),
                needs_evidence=True,
            )

        detector = None
        if probe.allow_repetition_gate:
            # Budgets scaled to the probe. The production defaults are DELTA
            # counts of 24000/96000; at max_tokens in the tens or hundreds
            # neither could ever fire, so the gate would be dead code.
            detector = RunawayDetector(
                think_budget=max(1, probe.max_tokens // 4),
                repeat_max=3,
                hard_max=max(2, probe.max_tokens * 2),
                shingle_size=48,
                window_chars=4096,
            )
        acc = StreamAccumulator(detector)

        payload = {
            # The REAL served name. An unknown one returns 200 and garbage.
            "model": self.control.served_model_name,
            "messages": probe.messages,
            "max_tokens": probe.max_tokens,
            "temperature": 0,
            "seed": self.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        status: int | None = None
        body: object = None
        timed_out = False
        transport_error: str | None = None
        t0 = time.monotonic()
        try:
            async with asyncio.timeout(deadline_s):
                async with self._client.stream(
                    "POST", f"{target.base}/v1/chat/completions", json=payload
                ) as resp:
                    status = resp.status_code
                    if status >= 400:
                        raw = await resp.aread()
                        body = _maybe_json(raw)
                    else:
                        async for line in resp.aiter_lines():
                            acc.feed_line(line)
        except TimeoutError:
            timed_out = True
        except Exception as exc:  # noqa: BLE001 — any transport failure is evidence
            transport_error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - t0

        needs_evidence = (
            timed_out
            or transport_error is not None
            or status is None
            or status >= 400
            or not acc.saw_done
        )
        engine_healthy = True
        row_status = "loaded"
        if needs_evidence:
            # Both reads are deferred to here on purpose: a healthy 200 answers
            # the liveness question by itself, and row_status is a DB round trip
            # that would otherwise happen on every probe of every rung.
            engine_healthy = await self.control.health()
            row_status = await self.control.row_status()

        return self._classified(
            probe,
            ref,
            acc=acc,
            status=status,
            body=body,
            elapsed=elapsed,
            timed_out=timed_out,
            engine_healthy=engine_healthy,
            row_status=row_status,
            needs_evidence=needs_evidence,
            length_hint=acc.prompt_tokens,
        )

    def _classified(
        self,
        probe: Probe,
        ref: Reference,
        *,
        acc: StreamAccumulator,
        status: int | None,
        body: object,
        elapsed: float,
        timed_out: bool,
        engine_healthy: bool,
        row_status: str,
        needs_evidence: bool,
        length_hint: int | None = None,
    ) -> _ProbeRun:
        graded = self._grade(probe, acc.content)
        observation = acc.observation(graded_ok=graded)
        gates: tuple[Gate, ...] = ()
        if status == 200 and not timed_out:
            gates = evaluate(observation, ref)

        traffic = self.control.foreign_requests() > 0
        if traffic and self.force:
            self._traffic_observed = True

        # An operator unloading mid-run looks exactly like a crash from here,
        # and must NOT trigger recovery: bringing back a model somebody
        # deliberately stopped is worse than abandoning the measurement.
        if row_status in ("unloading", "unloaded", "pulling"):
            self._status_changed = True

        obs = Observables(
            http_status=status,
            body=body,
            engine_healthy=engine_healthy,
            # Only consulted when something already looks wrong: a supervisor
            # read is cheap but a DB read is not, and a healthy 200 answers the
            # liveness question by itself.
            process_alive=self.control.process_alive() if needs_evidence else None,
            returncode=self.control.returncode() if needs_evidence else None,
            row_status=row_status,
            deadline_exceeded=timed_out,
            host_oom_counters_moved=(
                self.control.host_oom_counters_moved() if needs_evidence else False
            ),
            gates_tripped=gates,
            warden_restarted=False,
            foreign_traffic_on_this_model=traffic and not self.force,
            neighbour_unhealthy=self._neighbour_down is not None,
            status_changed_externally=self._status_changed,
        )
        outcome = classify(obs, self._refusal_evidence(status, body, length_hint))
        return _ProbeRun(
            probe_id=probe.id,
            outcome=outcome,
            elapsed_s=elapsed,
            http_status=status,
            prompt_tokens=acc.prompt_tokens,
            cached_tokens=acc.cached_tokens,
            output_tokens=observation.output_tokens,
            content=acc.content,
            gates=gates,
            observables=obs,
            body=body,
            starved=(budget_starved(observation) if status == 200 and not timed_out else False),
        )

    def _refusal_evidence(
        self, status: int | None, body: object, length_hint: int | None
    ) -> RefusalEvidence:
        """The DEFINITIONAL test only: it needs no probe and cannot be wrong.

        An error at a size beyond the row's declared ``max_model_len`` is a
        refusal by construction — the engine had no choice. The second test,
        confirming at the last passing rung, needs to send requests and so lives
        in :meth:`_confirm_refusal`.
        """
        if status is None or status < 400:
            return RefusalEvidence()
        declared = self.control.caps().max_model_len
        if declared is None:
            return RefusalEvidence()
        requested = _requested_tokens(body)
        if length_hint is not None and length_hint > declared:
            return RefusalEvidence(beyond_declared_limit=True)
        if requested is not None and requested > declared:
            return RefusalEvidence(beyond_declared_limit=True)
        return RefusalEvidence()

    async def _confirm_refusal(self, run: _ProbeRun) -> _ProbeRun:
        """Decide an ambiguous error by re-probing the LAST PASSING RUNG.

        Never a fraction of the failing size. v1 confirmed at 0.9x the failure
        against a GEOMETRIC ladder, where the first failing rung lies in (R, 2R]
        — so 0.9x lands below the true threshold only when the rung is within
        1.111R, which is 15.2% of the interval. 85% of genuine refusals were
        therefore filed as errors, reintroducing the defect this feature exists
        to fix inside the mechanism advertised as its fix.

        The last passing rung cannot miss that way: it is not derived from the
        failing value at all, and it is already proven good in this run.
        """
        if self._last_passing_length is None or self._confirming:
            return run
        if not isinstance(run.body, dict) or not isinstance(run.body.get("error"), dict | str):
            return run
        rung = self._last_passing_length
        self._confirming = True
        try:
            ok = await apass_n(lambda: self._trial_length(rung), n=self.profile.pass_n)
        finally:
            self._confirming = False
        if not ok:
            return run
        outcome = classify(run.observables, RefusalEvidence(last_passing_rung_still_passes=True))
        return _ProbeRun(
            probe_id=run.probe_id,
            outcome=outcome,
            elapsed_s=run.elapsed_s,
            http_status=run.http_status,
            prompt_tokens=run.prompt_tokens,
            cached_tokens=run.cached_tokens,
            output_tokens=run.output_tokens,
            content=run.content,
            gates=run.gates,
            observables=run.observables,
            body=run.body,
        )

    def _grade(self, probe: Probe, content: str) -> bool | None:
        if probe.grader == "numeric":
            return grade_numeric(float(probe.expected), content, tol=probe.tol)  # type: ignore[arg-type]
        if probe.grader == "substring":
            return grade_substring(str(probe.expected), content)
        if probe.grader == "regex":
            return grade_regex(str(probe.expected), content)
        if probe.grader == "json":
            return grade_json_equal(probe.expected, content)
        return None

    # -- recovery ---------------------------------------------------------

    async def _recover(self) -> None:
        """Restart, wait for health, warm up — then RE-RESOLVE the target.

        ``_restart`` allocates a new port and preserves the overrides the model
        was loaded with (``sup.get_overrides()``); dropping them would silently
        change the effective ``max_model_len`` mid-run and every measurement
        after this point would describe a different configuration.
        """
        await self.control.restart()
        if not await self.control.wait_for_health():
            raise StressAborted("aborted_recovery_failed", "engine did not come back healthy")
        if not await self.control.warmup():
            raise StressAborted("aborted_recovery_failed", "engine came back but does not serve")
        if self.control.target() is None:
            raise StressAborted("aborted_recovery_failed", "no port after restart")

    async def _postcondition(self) -> tuple[bool, str | None]:
        """Design §5.8. Runs in the ``finally``, on every exit path."""
        for task in list(self._inflight):
            task.cancel()
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
            self._inflight.clear()
        if self._client is not None:
            # Closes the sockets, which is what lets the engine free the KV
            # blocks a timed-out probe is still holding.
            await self._client.aclose()
            self._client = None

        try:
            if await self.control.health():
                return True, None
        except Exception as exc:  # noqa: BLE001
            return False, f"health check failed: {exc}"

        for _ in range(RECOVERY_ATTEMPTS):
            try:
                await self.control.restart()
                if await self.control.wait_for_health() and await self.control.warmup():
                    return True, None
            except Exception as exc:  # noqa: BLE001
                log.warning("stress %s: recovery attempt failed: %s", self.run_id, exc)
        # Two, then stop. Looping here is the crash loop the restart budget
        # exists to prevent.
        return False, "engine did not recover after two restart attempts"

    # -- the sweep --------------------------------------------------------

    async def _sweep(self) -> None:
        """Reload at each candidate ``max_model_len`` and re-measure (design §6).

        This is the PRIMARY output: an in-place run can only say how much of the
        already-allocated context is usable, and the question an operator has is
        what the setting should be. Each candidate is a different configuration
        with a different fingerprint, so the answer is advisory and explicitly
        not what ``/v1/models`` publishes.
        """
        if not self.profile.run_sweep:
            return
        caps = self.control.caps()
        candidates = sweep.plan_candidates(
            current=caps.max_model_len,
            model_ceiling=caps.model_ceiling,
            vram_budget_bytes=caps.vram_budget_bytes,
            kv_bytes_per_token=caps.kv_bytes_per_token,
        )
        results: list[sweep.CandidateResult] = []
        original = caps.max_model_len
        try:
            for candidate in candidates:
                loaded = await self.control.reload_with(max_model_len=candidate)
                if not loaded:
                    # KV-pool OOM at load is cheap and clean: the engine never
                    # served, nothing was at risk, no crash budget is spent.
                    results.append(
                        sweep.CandidateResult(
                            max_model_len=candidate, loaded=False, quality_ok=None
                        )
                    )
                    continue
                try:
                    ok = await self._trial_length(max(self.length_floor, candidate // 2))
                    crashed = False
                except _AxisTruncated:
                    ok, crashed = False, True
                results.append(
                    sweep.CandidateResult(
                        max_model_len=candidate,
                        loaded=True,
                        quality_ok=ok,
                        crashed=crashed,
                        gate_tripped=self._gate_tripped,
                    )
                )
        finally:
            if original is not None:
                # Leave the engine in the configuration we found it in. The
                # recommendation is advisory; applying it is an explicit
                # operator action that invalidates this run's measurement.
                await self.control.reload_with(max_model_len=original)

        rec = sweep.choose_recommendation(results)
        if rec is not None:
            self._recommended = {
                "max_model_len": rec.max_model_len,
                "limited_by": rec.limited_by,
                "next_failed_at": rec.next_failed_at,
                "gate_tripped": rec.gate_tripped,
                "provenance": "measured",
                "applies_to": "a different configuration than the one running",
            }

    # -- record keeping ---------------------------------------------------

    def _oracle_record(self) -> dict:
        """Structured, never prose.

        v1's whole argument for PASS_N was that its bias is quantifiable — and
        then encoded N inside a free-text ``method`` string, so a client could
        not tell a 21%-contour number from a 9%-contour one.
        """
        return {
            "n_consecutive": self.profile.pass_n,
            "contour_p": round(search.CONTOUR_BY_N[self.profile.pass_n], 4),
            "confirm_m": self.profile.confirm_m,
            "safety_factor": self.profile.safety_factor,
        }

    def _record(self, run: _ProbeRun, *, axis: str, value: int) -> None:
        self._observations.append(
            {
                "axis": axis,
                "value": value,
                "probe": run.probe_id,
                "class": run.outcome.cls.value,
                "gates": [g.value for g in run.gates],
                "http_status": run.http_status,
                "prompt_tokens": run.prompt_tokens,
                "cached_tokens": run.cached_tokens,
                "elapsed_s": round(run.elapsed_s, 3),
            }
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _maybe_json(raw: bytes) -> object:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw.decode("utf-8", "replace")[:500]


def _requested_tokens(body: object) -> int | None:
    """``n_prompt_tokens`` from a recorded llama.cpp refusal envelope."""
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if not isinstance(err, dict):
        return None
    n = err.get("n_prompt_tokens")
    return int(n) if isinstance(n, int) else None


def _limited_by_for(outcome: Outcome) -> str:
    return {
        Class.DEGRADED: "quality",
        Class.CRASHED: "crash",
        Class.REFUSED: "refusal",
        Class.TIMEOUT: "timeout",
        Class.ERRORED: "error_alive",
    }.get(outcome.cls, "quality")


# ---------------------------------------------------------------------------
# the real wiring
# ---------------------------------------------------------------------------


class WardenEngineControl:
    """:class:`EngineControl` bound to the live warden.

    Every method here is one call into something that already exists. The
    protocol is deliberately wide and shallow for that reason: the interesting
    code is the runner, and none of it should have to know that a port lives in
    a supervisor dict or that a restart goes through the watchdog.
    """

    def __init__(
        self,
        settings,
        app_state,
        model_row,
        *,
        subject: Subject,
        caps: Caps,
        neighbours: Sequence[str] = (),
        gpu_uuids: Sequence[str] = (),
    ) -> None:
        self._settings = settings
        self._app_state = app_state
        self._row = model_row
        self.served_model_name = model_row.served_model_name
        self._subject = subject
        self._caps = caps
        self._neighbours = tuple(neighbours)
        self._gpu_uuids = tuple(gpu_uuids)
        self._oom_before: int | None = None

    # -- identity ---------------------------------------------------------

    def subject(self) -> Subject:
        return self._subject

    def caps(self) -> Caps:
        return self._caps

    def gpu_uuids(self) -> Sequence[str]:
        return self._gpu_uuids

    # -- target -----------------------------------------------------------

    def target(self) -> Target | None:
        """Straight from the supervisor's in-memory maps, every time.

        Never from ``model_runtime``: that table is WIPED at startup
        (``app/db/repos/runtime.py:52``, called from ``app/main.py:83``), so a
        row read there is either stale or absent. And never cached, because
        ``_restart`` allocates a new port.
        """
        sup = self._app_state.supervisor
        port = sup.get_port(self._row.id)
        host = sup.get_host(self._row.id) or "127.0.0.1"
        if port is None:
            return None
        return Target(host=host, port=port)

    # -- liveness ---------------------------------------------------------

    async def health(self) -> bool:
        from app.runtime.watchdog import probe_health  # noqa: PLC0415

        target = self.target()
        if target is None:
            return False
        ok, _detail = await probe_health(target.host, target.port)
        return ok

    def process_alive(self) -> bool | None:
        sup = self._app_state.supervisor
        if getattr(self._settings, "engine_driver", "subprocess") == "docker":
            # No signal and no /proc under the docker driver, so the honest
            # answer is "unknown" rather than a fabricated True.
            return None
        return sup.is_running(self._row.id)

    def returncode(self) -> int | None:
        sup = self._app_state.supervisor
        handle = getattr(sup, "_handles", {}).get(self._row.id)
        return getattr(handle, "returncode", None)

    def host_oom_counters_moved(self) -> bool:
        """A -9 with host OOM counters moving is the MACHINE running out, not
        the model reaching a limit; publishing it as a context limit would be
        worse than publishing nothing."""
        try:
            with open("/proc/vmstat") as fh:  # noqa: PTH123
                counts = {
                    k: int(v) for k, v in (line.split() for line in fh if line.startswith("oom"))
                }
        except OSError:
            return False
        total = sum(counts.values())
        moved = self._oom_before is not None and total > self._oom_before
        self._oom_before = total
        return moved

    async def row_status(self) -> str:
        from app.db.database import open_db  # noqa: PLC0415
        from app.db.repos.models import ModelRepo  # noqa: PLC0415

        async with open_db(self._settings.db_path) as db:
            row = await ModelRepo(db).get(self._row.id)
        return row.status if row else "missing"

    def foreign_requests(self) -> int:
        """Per-model, from ``snapshot()``, never the global ``count()``.

        ``RequestRegistry.count()`` returns the length across EVERY model, so a
        guard built on it blocks a run on model A because someone is chatting
        with model B on another GPU — a multi-tenancy failure dressed up as a
        busy-ness check. Our own probes are engine-direct and never register
        here, so there is nothing to subtract.
        """
        registry = getattr(self._app_state, "request_registry", None)
        if registry is None:
            return 0
        names = {self._row.id, self.served_model_name}
        return sum(1 for r in registry.snapshot() if r.model in names)

    async def neighbour_health(self) -> dict[str, bool]:
        from app.runtime.watchdog import probe_health  # noqa: PLC0415

        sup = self._app_state.supervisor
        out: dict[str, bool] = {}
        for model_id in self._neighbours:
            port = sup.get_port(model_id)
            if port is None:
                out[model_id] = False
                continue
            ok, _ = await probe_health(sup.get_host(model_id) or "127.0.0.1", port)
            out[model_id] = ok
        return out

    # -- recovery ---------------------------------------------------------

    async def restart(self) -> None:
        from app.runtime.watchdog import _restart  # noqa: PLC0415

        sup = self._app_state.supervisor
        # ``check_once`` passes these and ``restart_crashed_models`` passes
        # None; a recovery that drops them silently changes the effective
        # max_model_len and every later measurement describes a different
        # configuration than the one the run started on.
        overrides = sup.get_overrides(self._row.id)
        await _restart(self._settings, self._app_state, self._row.id, overrides)

    async def wait_for_health(self) -> bool:
        from app.runtime.supervisor import wait_for_health  # noqa: PLC0415

        target = self.target()
        if target is None:
            return False
        return await wait_for_health(
            port=target.port,
            host=target.host,
            timeout_s=float(getattr(self._settings, "load_timeout_s", 600.0)),
        )

    async def warmup(self) -> bool:
        """``/health`` returns 200 before multimodal warmup completes, which is
        why this exists at all: it is the codebase's definition of "actually
        serving"."""
        from app.runtime.warmup_probe import warmup_probe  # noqa: PLC0415

        target = self.target()
        if target is None:
            return False
        result = await warmup_probe(
            port=target.port,
            host=target.host,
            served_model_name=self.served_model_name,
            timeout_s=60.0,
        )
        return result.ok

    def reset_budget(self) -> None:
        budget = getattr(self._app_state, "restart_budget", None)
        if budget is not None:
            budget.reset(self._row.id)

    async def reload_with(self, *, max_model_len: int) -> bool:
        """Reload at a different context size for the sweep.

        Returns False rather than raising when the engine will not load: an
        over-large KV pool fails at startup before serving anything, which is a
        clean upper bound for the sweep and not a crash.
        """
        from app.runtime.watchdog import _restart  # noqa: PLC0415

        sup = self._app_state.supervisor
        overrides = dict(sup.get_overrides(self._row.id) or {})
        overrides["max_model_len"] = max_model_len
        try:
            await _restart(self._settings, self._app_state, self._row.id, overrides)
        except Exception as exc:  # noqa: BLE001 — a failed load IS the answer
            log.info("stress sweep: %s failed to load at %s: %s", self._row.id, max_model_len, exc)
            return False
        return await self.wait_for_health() and await self.warmup()


# ---------------------------------------------------------------------------
# the entry point the API layer calls
# ---------------------------------------------------------------------------


class DbPathRepo:
    """A :class:`StressRunRepo` that opens and CLOSES a connection per call.

    A run is minutes to hours and performs four short writes in that whole
    window. Holding one aiosqlite connection open across it would pin a thread
    and a WAL reader for the duration — and a connection that outlives the code
    that made it is exactly how a test suite ends up with thirteen live threads
    and a CI container that never exits.
    """

    def __init__(self, db_path) -> None:
        self.db_path = db_path

    async def insert(self, row: StressRunRow) -> None:
        from app.db.database import open_db  # noqa: PLC0415

        async with open_db(self.db_path) as db:
            await StressRunRepo(db).insert(row)

    async def get(self, run_id: str) -> StressRunRow | None:
        from app.db.database import open_db  # noqa: PLC0415

        async with open_db(self.db_path) as db:
            return await StressRunRepo(db).get(run_id)

    async def update_progress(self, run_id: str, progress: dict) -> None:
        from app.db.database import open_db  # noqa: PLC0415

        async with open_db(self.db_path) as db:
            await StressRunRepo(db).update_progress(run_id, progress)

    async def finish(self, run_id: str, **kwargs) -> None:
        from app.db.database import open_db  # noqa: PLC0415

        async with open_db(self.db_path) as db:
            await StressRunRepo(db).finish(run_id, **kwargs)


#: Tasks are held so the garbage collector cannot drop a running one — the
#: documented failure mode of a bare ``asyncio.create_task``.
_RUNNING: set[asyncio.Task] = set()


async def start_run(
    app_state,
    model_id: str,
    *,
    mode: str = "conservative",
    force: bool = False,
    model_ceiling: int | None = None,
) -> str:
    """Build the control seam, launch the run, return its id immediately.

    Precedent is ``run_pull`` (``app/models/routes_api.py:1000``): one
    ``asyncio.create_task`` owned by app state, with progress read by polling
    the run row. bench-v2's SSE replay with a binary offset index was pure cost.

    The ``Subject`` comes from ``routes_api.build_subject`` and NOWHERE else.
    The reader side recomputes the fingerprint to decide whether a stored run
    still describes the model as it currently stands, so a second builder that
    disagreed on one field would produce hashes that never match — every run
    recorded, none ever published, and no error anywhere.
    """
    from app.db.database import open_db  # noqa: PLC0415
    from app.db.repos.models import ModelRepo  # noqa: PLC0415
    from app.stress.routes_api import build_subject, live_gpus  # noqa: PLC0415

    settings = app_state.settings
    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get(model_id)
        if model is None:
            raise LookupError(f"model {model_id} not found")
        all_models = await ModelRepo(db).list_all()

    gpus, _probe_error, _incomplete = await live_gpus(app_state, model)
    subject = await build_subject(app_state, model, all_models=all_models, gpus=gpus)

    control = WardenEngineControl(
        settings,
        app_state,
        model,
        subject=subject,
        caps=Caps(
            max_model_len=model.max_model_len,
            max_num_seqs=model.max_batch_size,
            # Left None when the caller does not know it. The model's own
            # trained context lives in its HF config, not in the row, and
            # INVENTING a ceiling is the one thing the sweep must not do —
            # beyond it the engine refuses, so a probe there measures the config
            # file rather than the hardware. Without it the sweep asks only
            # whether the CURRENT setting is too ambitious, which is honest.
            model_ceiling=model_ceiling,
        ),
        neighbours=[n.model_id for n in subject.neighbours],
        gpu_uuids=[g.uuid for g in gpus],
    )

    runner = StressRunner(
        control=control,
        repo=DbPathRepo(settings.db_path),
        model_id=model_id,
        mode=mode,
        leases=getattr(app_state, "stress_leases", None) or LeaseRegistry(),
        force=force,
    )
    task = asyncio.create_task(runner.run())
    _RUNNING.add(task)
    task.add_done_callback(_RUNNING.discard)
    app_state.stress_task = task
    return runner.run_id
