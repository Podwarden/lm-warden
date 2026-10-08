"""Outcome classification (design §5.2, §5.3).

The class that mattered most is REFUSED. v1 of the design confirmed a refusal
by re-probing at 0.9x the failing size — against a GEOMETRIC ladder, where the
first failing rung lies in (R, 2R]. That confirmation succeeds only when
rung <= 1.111R, i.e. log2(1.111) = 15.2% of the interval, so 85% of genuine
engine refusals were misclassified as errors. That is bench-v2's defining
defect reintroduced inside the mechanism advertised as its fix.

v2 replaces it with three non-multiplicative tests: definitional (beyond
max_model_len), last-known-good, and replication.
"""

from __future__ import annotations

from app.stress.gates import Gate
from app.stress.outcomes import Class, Observables, RefusalEvidence, classify


def _obs(**over) -> Observables:
    base = dict(
        http_status=200,
        body={"choices": [{"message": {"content": "ok"}}]},
        engine_healthy=True,
        process_alive=True,
        returncode=None,
        row_status="loaded",
        deadline_exceeded=False,
        host_oom_counters_moved=False,
        gates_tripped=(),
        warden_restarted=False,
        foreign_traffic_on_this_model=False,
        neighbour_unhealthy=False,
        status_changed_externally=False,
    )
    base.update(over)
    return Observables(**base)


# ---- ordering: liveness before status -----------------------------------


def test_a_crash_that_also_returns_500_is_classified_as_a_crash():
    """Liveness is evaluated before HTTP status, deliberately.

    A dying engine can emit a 500 on its way down. Reading the status first
    would file a crash as an error and let the search keep climbing into it.
    """
    out = classify(_obs(http_status=500, engine_healthy=False, process_alive=False))
    assert out.cls is Class.CRASHED


def test_health_is_authoritative_over_process_liveness():
    """The `vllm serve` wrapper routinely outlives a dead EngineCore.

    That is the entire reason app/runtime/watchdog.py exists. A live process
    with a dead engine is a crash.
    """
    out = classify(_obs(http_status=None, engine_healthy=False, process_alive=True))
    assert out.cls is Class.CRASHED


# ---- class 0: inconclusive ----------------------------------------------


def test_a_warden_restart_is_inconclusive_not_a_crash():
    out = classify(_obs(warden_restarted=True, engine_healthy=False))
    assert out.cls is Class.INCONCLUSIVE
    assert out.publishable is False


def test_foreign_traffic_on_this_model_is_inconclusive():
    """KV is a shared pool, so someone else's request moves the measurement."""
    assert classify(_obs(foreign_traffic_on_this_model=True)).cls is Class.INCONCLUSIVE


def test_an_unhealthy_neighbour_is_inconclusive():
    """Crashing our engine can starve a co-resident model.

    Their recovery is the watchdog's job; ours is to stop measuring, because
    the GPU no longer holds what we thought it held.
    """
    assert classify(_obs(neighbour_unhealthy=True)).cls is Class.INCONCLUSIVE


def test_an_operator_unloading_mid_run_is_inconclusive_not_a_crash():
    """Otherwise the runner restarts a model the operator deliberately stopped.

    v1 handled none of this: unload read as ConnectError -> CRASHED ->
    _restart() -> the model comes back up against the operator's intent.
    """
    out = classify(_obs(status_changed_externally=True, engine_healthy=False, http_status=None))
    assert out.cls is Class.INCONCLUSIVE
    assert out.recover is False, "must not restart what the operator stopped"


def test_a_host_oom_kill_is_inconclusive_not_a_model_limit():
    """returncode -9 with host OOM counters moving is the machine, not the model.

    Publishing a host OOM as a model's context limit is worse than publishing
    nothing.
    """
    out = classify(
        _obs(
            http_status=None,
            engine_healthy=False,
            process_alive=False,
            returncode=-9,
            host_oom_counters_moved=True,
        )
    )
    assert out.cls is Class.INCONCLUSIVE


def test_a_cuda_abort_is_a_real_crash():
    """-6 SIGABRT is the CUDA-assert signature seen on the Quadro."""
    out = classify(_obs(http_status=None, engine_healthy=False, process_alive=False, returncode=-6))
    assert out.cls is Class.CRASHED
    assert out.consumes_crash_budget is True


# ---- class 2: timeout ----------------------------------------------------


def test_a_deadline_with_a_healthy_engine_is_a_timeout():
    out = classify(_obs(deadline_exceeded=True, http_status=None))
    assert out.cls is Class.TIMEOUT
    assert out.consumes_crash_budget is False


def test_a_deadline_with_a_dead_engine_is_a_crash():
    out = classify(
        _obs(deadline_exceeded=True, engine_healthy=False, process_alive=False, http_status=None)
    )
    assert out.cls is Class.CRASHED


# ---- class 3: refusal, rebuilt ------------------------------------------

_ENVELOPE = {"error": {"message": "This model's maximum context length is 8192 tokens"}}


def test_an_error_beyond_max_model_len_is_definitionally_a_refusal():
    """No confirmation probe needed — the limit is declared in the row.

    This is the cheapest of the three tests and covers the common case.
    """
    out = classify(
        _obs(http_status=400, body=_ENVELOPE),
        refusal=RefusalEvidence(beyond_declared_limit=True),
    )
    assert out.cls is Class.REFUSED
    assert out.consumes_crash_budget is False


def test_a_refusal_is_confirmed_at_the_last_passing_rung_not_a_fraction():
    """v1 re-probed at 0.9x a FAILING size against a geometric ladder.

    That lands below the true threshold only 15.2% of the time. v2 confirms at
    a value already proven good, which cannot miss by construction.
    """
    out = classify(
        _obs(http_status=400, body=_ENVELOPE),
        refusal=RefusalEvidence(last_passing_rung_still_passes=True),
    )
    assert out.cls is Class.REFUSED


def test_an_unconfirmed_error_envelope_is_not_a_refusal():
    """A flaky failure that happens to return JSON must not stop the climb.

    REFUSED ends the axis and spends no crash budget, so it is the most
    consequential class to get wrong.
    """
    out = classify(
        _obs(http_status=500, body=_ENVELOPE),
        refusal=RefusalEvidence(),  # nothing confirmed
    )
    assert out.cls is Class.ERRORED


def test_a_5xx_shaped_refusal_is_accepted_but_flagged_low_confidence():
    """llama.cpp says 'Context size has been exceeded' at 500, vLLM 400.

    Both strings change across versions and neither appears in this repo, so
    classification is structural — never by message text.
    """
    out = classify(
        _obs(http_status=500, body=_ENVELOPE),
        refusal=RefusalEvidence(beyond_declared_limit=True),
    )
    assert out.cls is Class.REFUSED
    assert out.refusal_confidence == "low"


def test_a_4xx_refusal_is_high_confidence():
    out = classify(
        _obs(http_status=400, body=_ENVELOPE),
        refusal=RefusalEvidence(beyond_declared_limit=True),
    )
    assert out.refusal_confidence == "high"


def test_429_is_backpressure_not_a_refusal():
    """Structurally identical — envelope, healthy engine, smaller request works.

    But it is transient. Publishing it as a capacity limit would bake a
    momentary queue depth into the contract.
    """
    out = classify(
        _obs(http_status=429, body=_ENVELOPE),
        refusal=RefusalEvidence(beyond_declared_limit=True),
    )
    assert out.cls is Class.BACKPRESSURE
    assert out.retry is True
    assert out.publishable is False


# ---- class 5/6: degraded and pass ---------------------------------------


def test_a_tripped_gate_is_degraded():
    out = classify(_obs(gates_tripped=(Gate.ABRUPT,)))
    assert out.cls is Class.DEGRADED


def test_degraded_fails_quality_but_passes_capacity():
    """One set of probes yields two numbers (design §5.2).

    The model answered — the engine coped. The answer was unusable — the model
    did not. Those are different limits and clients need the second one.
    """
    out = classify(_obs(gates_tripped=(Gate.REPEATING,)))
    assert out.passes_capacity is True
    assert out.passes_quality is False


def test_a_clean_answer_passes_both():
    out = classify(_obs())
    assert out.cls is Class.PASS
    assert out.passes_capacity is True
    assert out.passes_quality is True


# ---- docker driver ------------------------------------------------------


def test_the_docker_driver_classifies_without_a_returncode():
    """Under VW_ENGINE_DRIVER=docker there is no signal and no /proc/vmstat.

    Health must still be decisive, and the weaker discrimination is recorded
    rather than silently assumed away.
    """
    out = classify(
        _obs(http_status=None, engine_healthy=False, process_alive=None, returncode=None)
    )
    assert out.cls is Class.CRASHED
    assert out.crash_cause == "unknown"
