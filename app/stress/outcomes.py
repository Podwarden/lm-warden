"""Outcome classification for one stress probe (design §5.2, §5.3).

Seven classes, evaluated in a fixed order. The order is load-bearing:
**liveness is checked before HTTP status**, because a dying engine can emit a
500 on its way down, and reading the status first would file a crash as an
error and let the search keep climbing into it.

The class that most needed rebuilding is REFUSED. It is the only class that
stops an axis *without* spending crash budget, so a false positive silently
truncates a published limit and a false negative wastes a crash. v1 confirmed
refusals by re-probing at 0.9x the failing size — against a geometric ladder,
where the first failing rung lies in (R, 2R]. That lands below the true
threshold only when rung <= 1.111R: log2(1.111) = 15.2% of the interval. So
85% of genuine refusals were misclassified, which is exactly the defect this
whole feature exists to fix, reintroduced inside the fix.

Everything here is pure. Liveness, health and confirmation are gathered by the
runner and passed in as evidence.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from app.stress.gates import Gate


class Class(str, enum.Enum):
    """Values are published, so they are wire contract."""

    INCONCLUSIVE = "inconclusive"
    CRASHED = "crashed"
    TIMEOUT = "timeout"
    REFUSED = "refused"
    BACKPRESSURE = "backpressure"
    ERRORED = "errored"
    DEGRADED = "degraded"
    PASS = "pass"


@dataclass(frozen=True)
class Observables:
    """Everything the classifier is allowed to look at.

    ``process_alive`` is ``None`` under the docker driver, where there is no
    signal and no ``/proc/vmstat`` to read. ``http_status`` is ``None`` on a
    transport failure.
    """

    http_status: int | None
    body: object
    engine_healthy: bool
    process_alive: bool | None
    returncode: int | None
    row_status: str
    deadline_exceeded: bool
    host_oom_counters_moved: bool
    gates_tripped: tuple[Gate, ...]
    warden_restarted: bool
    foreign_traffic_on_this_model: bool
    neighbour_unhealthy: bool
    status_changed_externally: bool


@dataclass(frozen=True)
class RefusalEvidence:
    """The two non-multiplicative confirmations that a refusal is genuine.

    Either is sufficient. Both avoid v1's mistake of stepping down by a
    fraction against a ladder that steps up by a factor.
    """

    #: The request exceeded the row's declared ``max_model_len``. Definitional
    #: — the engine had no choice, and no probe is needed to establish it.
    beyond_declared_limit: bool = False
    #: A value already proven good in this run still passes. Cannot miss by
    #: construction, because it is not derived from the failing value at all.
    last_passing_rung_still_passes: bool = False

    def confirmed(self) -> bool:
        return self.beyond_declared_limit or self.last_passing_rung_still_passes


@dataclass(frozen=True)
class Outcome:
    cls: Class
    publishable: bool = True
    consumes_crash_budget: bool = False
    passes_capacity: bool = False
    passes_quality: bool = False
    recover: bool = False
    retry: bool = False
    refusal_confidence: str | None = None
    crash_cause: str | None = None
    gates: tuple[Gate, ...] = field(default_factory=tuple)


#: Typed error discriminators that mean "the engine declined an oversized
#: request", as opposed to any other error. These are STRUCTURED fields, not
#: prose: llama-server b10731 returns
#: {"error": {"type": "exceed_context_size_error", "n_prompt_tokens": ..., "n_ctx": ...}}
#: which is machine-readable and stable in a way the message text is not.
#: Matching on it needs no confirmation probe at all.
_CONTEXT_REFUSAL_TYPES = frozenset({"exceed_context_size_error"})


def _typed_context_refusal(body: object) -> bool:
    """True when the engine NAMED the condition as an over-context refusal."""
    if not isinstance(body, dict):
        return False
    err = body.get("error")
    if not isinstance(err, dict):
        return False
    return err.get("type") in _CONTEXT_REFUSAL_TYPES


def _has_error_envelope(body: object) -> bool:
    """An OpenAI-shaped error object, checked structurally.

    Never by message text: vLLM says "request exceeds the available context" at
    400 and llama.cpp "Context size has been exceeded" at 500. Both wordings
    change across versions, and neither string appears anywhere in this repo.
    """
    return isinstance(body, dict) and isinstance(body.get("error"), dict | str)


def classify(
    obs: Observables,
    refusal: RefusalEvidence | None = None,
) -> Outcome:
    refusal = refusal or RefusalEvidence()

    # --- class 0: nothing measured here is trustworthy -------------------
    # Checked first because each of these means the experiment lost control of
    # its own conditions, whatever the engine happened to return.
    if obs.warden_restarted or obs.foreign_traffic_on_this_model or obs.neighbour_unhealthy:
        return Outcome(cls=Class.INCONCLUSIVE, publishable=False)

    # An operator unloading mid-run looks exactly like a crash from here. It
    # must NOT trigger recovery, or the runner brings back a model the operator
    # deliberately stopped.
    if obs.status_changed_externally:
        return Outcome(cls=Class.INCONCLUSIVE, publishable=False, recover=False)

    # --- class 1: the engine is gone -------------------------------------
    # Health is authoritative over process liveness: the `vllm serve` wrapper
    # routinely outlives a dead EngineCore, which is why the watchdog exists.
    engine_gone = not obs.engine_healthy or obs.process_alive is False or obs.row_status == "failed"
    if engine_gone:
        # A host OOM kill is the machine running out of memory, not the model
        # reaching a limit. Publishing it as a context limit would be worse
        # than publishing nothing.
        if obs.returncode == -9 and obs.host_oom_counters_moved:
            return Outcome(cls=Class.INCONCLUSIVE, publishable=False)
        cause = "unknown"
        if obs.returncode is not None and obs.returncode < 0:
            cause = f"signal_{-obs.returncode}"
        elif obs.returncode is not None:
            cause = f"exit_{obs.returncode}"
        return Outcome(
            cls=Class.CRASHED,
            publishable=True,
            consumes_crash_budget=True,
            recover=True,
            crash_cause=cause,
        )

    # --- class 2: our deadline, since nothing else imposes one -----------
    if obs.deadline_exceeded:
        return Outcome(cls=Class.TIMEOUT, publishable=True)

    status = obs.http_status
    is_error = status is None or status >= 400

    if is_error:
        # 429 is structurally indistinguishable from a refusal — envelope,
        # healthy engine, smaller request succeeds — but it is transient
        # backpressure. Baking a momentary queue depth into a published limit
        # would be a durable error from a temporary condition.
        if status == 429:
            return Outcome(cls=Class.BACKPRESSURE, publishable=False, retry=True)

        # A typed discriminator is stronger evidence than anything we could
        # establish with a probe, so it short-circuits the confirmation. It is
        # also always "high" confidence regardless of status code: llama.cpp
        # returns 400 here, but the certainty comes from error.type, not from
        # the status.
        if _typed_context_refusal(obs.body):
            return Outcome(cls=Class.REFUSED, publishable=True, refusal_confidence="high")

        if _has_error_envelope(obs.body) and refusal.confirmed():
            # A 5xx-shaped refusal is still a refusal — llama.cpp emits one —
            # but the shape is weaker evidence than a 4xx, so say so rather
            # than flattening the distinction.
            confidence = "high" if status is not None and status < 500 else "low"
            return Outcome(
                cls=Class.REFUSED,
                publishable=True,
                refusal_confidence=confidence,
            )

        return Outcome(cls=Class.ERRORED, publishable=True, retry=True)

    # --- class 5/6: the engine coped; did the model? ---------------------
    # DEGRADED passes capacity and fails quality. That asymmetry is what
    # yields two numbers from one set of probes: the largest workload the
    # engine survives, and the smaller one where output is still usable.
    if obs.gates_tripped:
        return Outcome(
            cls=Class.DEGRADED,
            publishable=True,
            passes_capacity=True,
            passes_quality=False,
            gates=obs.gates_tripped,
        )

    return Outcome(cls=Class.PASS, publishable=True, passes_capacity=True, passes_quality=True)
