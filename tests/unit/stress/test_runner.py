"""The stress runner, driven end to end against a fake engine.

The design's central claim is that every outcome class, the crash/recover path,
the wedge path and the abort paths are testable with **no GPU**. This file is
where that claim is either true or it is not, so the tests below go through the
real HTTP client, the real SSE accumulator, the real gates, the real classifier
and the real search — only the engine and the warden are fakes, and both are
driven through the same seam production uses.

Two mechanical rules, both learned the hard way:

* every aiosqlite connection is opened with ``async with`` and closed. Leaking
  them leaks a thread apiece, and thirteen of them once left the ``unit-tests``
  CI container hanging for half an hour after pytest had finished;
* nothing is decorated with ``@pytest.mark.asyncio`` — ``asyncio_mode = "auto"``.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import aiosqlite
import httpx
import pytest

from app.db.repos.stress_runs import StressRunRepo
from app.stress import routes_api
from app.stress import runner as runner_mod
from app.stress.fingerprint import Gpu, Subject
from app.stress.gates import Gate, Reference
from app.stress.lease import LeaseRegistry
from app.stress.outcomes import Class
from app.stress.probes import Probe, build_suite
from app.stress.runner import (
    Caps,
    StressRunner,
    Target,
    _AxisTruncated,
    _Bridge,
    _BridgeStopped,
)
from tests.conftest import wait_until_async
from tests.fakes.fake_vllm import FakeEngine, Knobs

_SQL_DIR = Path(__file__).resolve().parents[3] / "app" / "db" / "sql"
_DDL = _SQL_DIR / "0029_stress_runs.sql"
#: EVERY stress migration, in order. A fixture pinned to 0029 alone broke 34
#: tests the moment 0030 added a column, because the repo's SELECT list moved
#: with the schema while the test schema did not. Globbing means the next
#: migration is picked up without touching a fixture.
_STRESS_DDL = sorted(_SQL_DIR.glob("00*_stress*.sql"))


def _apply_stress_schema(execute) -> None:
    for f in _STRESS_DDL:
        execute(f.read_text())


SERVED = "fake-model"
MODEL_ID = "m1"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db():
    """In-memory, migrated, and CLOSED afterwards. See the module docstring."""
    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE models (id TEXT PRIMARY KEY)")
        await conn.execute(f"INSERT INTO models (id) VALUES ('{MODEL_ID}')")
        for _f in _STRESS_DDL:
            await conn.executescript(_f.read_text())
        await conn.commit()
        yield conn


@pytest.fixture
async def engine():
    eng = FakeEngine(served_model_name=SERVED, knobs=Knobs())
    await eng.start()
    try:
        yield eng
    finally:
        await eng.stop()


def _subject() -> Subject:
    return Subject(
        hf_repo="org/model",
        hf_revision="main",
        filename=None,
        quantization=None,
        backend="vllm",
        mmproj_filename=None,
        max_model_len=8192,
        max_num_seqs=4,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="auto",
        n_gpu_layers=None,
        parallelism_strategy="auto",
        extra_args=(),
        extra_env=(),
        engine_version="0.1.0",
        engine_image=None,
        max_num_batched_tokens=8192,
        proxy_max_inflight=16,
        request_max_wall_s=0.0,
        gpus=(
            Gpu(
                uuid="GPU-1",
                name="fake",
                memory_total_mib=16384,
                compute_cap=8.6,
                ecc_enabled=False,
            ),
        ),
        driver_version="550.0",
        neighbours=(),
        # From the ONE builder, so a drift between reader and writer shows up
        # here rather than as a measurement that is silently never published.
        probe_suite_hash=routes_api.PROBE_SUITE_HASH,
        algorithm_version=routes_api.ALGORITHM_VERSION,
    )


class FakeControl:
    """:class:`app.stress.runner.EngineControl` over an in-process fake engine.

    Deliberately implements the SAME protocol ``WardenEngineControl`` does, so
    the runner code exercised here is the production path rather than a
    rehearsal of it. What it fakes is only the warden: the port map, the restart,
    the neighbour list and the traffic registry.
    """

    def __init__(
        self,
        engine: FakeEngine,
        *,
        caps: Caps | None = None,
        served_model_name: str = SERVED,
        neighbours: dict[str, bool] | None = None,
    ) -> None:
        self.engine = engine
        self.served_model_name = served_model_name
        self._caps = caps or Caps(max_model_len=8192, max_num_seqs=4)
        self._neighbours = neighbours if neighbours is not None else {}
        self.restarts = 0
        self.restart_raises: Exception | None = None
        self.row = "loaded"
        self.traffic = 0
        self.budget_resets = 0
        self.ports_seen: list[int] = []
        self.reloads: list[int] = []
        self.reload_fails_above: int | None = None
        self.on_target = None

    # -- identity
    def subject(self) -> Subject:
        return _subject()

    def caps(self) -> Caps:
        return self._caps

    def gpu_uuids(self):
        return ("GPU-1",)

    # -- target: RE-RESOLVED every call, never cached
    def target(self) -> Target | None:
        if self.on_target is not None:
            self.on_target()
        if self.engine.port is None:
            return None
        self.ports_seen.append(self.engine.port)
        return Target("127.0.0.1", self.engine.port)

    # -- liveness
    async def health(self) -> bool:
        target = self.target()
        if target is None:
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{target.base}/health")
            return r.status_code == 200
        except Exception:
            return False

    def process_alive(self) -> bool | None:
        return self.engine.port is not None

    def returncode(self) -> int | None:
        return None if self.engine.port is not None else 1

    def host_oom_counters_moved(self) -> bool:
        return False

    async def row_status(self) -> str:
        return self.row

    def foreign_requests(self) -> int:
        return self.traffic

    async def neighbour_health(self) -> dict[str, bool]:
        return dict(self._neighbours)

    def set_neighbour(self, model_id: str, healthy: bool) -> None:
        self._neighbours[model_id] = healthy

    # -- recovery
    async def restart(self) -> None:
        self.restarts += 1
        if self.restart_raises is not None:
            raise self.restart_raises
        await self.engine.restart()

    async def wait_for_health(self) -> bool:
        return await self.health()

    async def warmup(self) -> bool:
        return await self.health()

    def reset_budget(self) -> None:
        self.budget_resets += 1

    async def reload_with(self, *, max_model_len: int) -> bool:
        self.reloads.append(max_model_len)
        if self.reload_fails_above is not None and max_model_len > self.reload_fails_above:
            return False
        await self.engine.restart()
        return True


def make_runner(control, db, **kw) -> StressRunner:
    kw.setdefault("mode", "conservative")
    kw.setdefault("length_floor", 256)
    kw.setdefault("deadline_floor_s", 5.0)
    kw.setdefault("deadline_cap_s", 10.0)
    kw.setdefault("leases", LeaseRegistry())
    return StressRunner(control=control, repo=StressRunRepo(db), model_id=MODEL_ID, **kw)


async def _one(control, db, *, probe=None, ref=None, deadline_s=5.0):
    """Run a single probe through the real transport and classifier."""
    run = make_runner(control, db)
    probe = probe or build_suite(run_id="r", seed=1, approx_tokens=64)[0]
    ref = ref or Reference(output_tokens=0, checks_length=False)
    async with httpx.AsyncClient(timeout=None) as client:
        run._client = client
        return await run._execute(probe, ref=ref, deadline_s=deadline_s)


# ---------------------------------------------------------------------------
# the outcome taxonomy, one class at a time
# ---------------------------------------------------------------------------


async def test_a_healthy_engine_answering_correctly_is_a_pass(engine, db):
    result = await _one(FakeControl(engine), db)
    assert result.outcome.cls is Class.PASS
    assert result.gates == ()


@pytest.mark.parametrize(
    ("mode", "gate"),
    [
        ("empty", Gate.EMPTY),
        ("reasoning_only", Gate.REASONING_ONLY),
        ("wrong", Gate.WRONG),
        ("abrupt", Gate.ABRUPT),
        ("truncated", Gate.ABRUPT),
    ],
)
async def test_each_quality_gate_produces_degraded(engine, db, mode, gate):
    """DEGRADED passes capacity and fails quality — the asymmetry that yields
    two published numbers from one set of probes."""
    engine.set(mode=mode)
    result = await _one(FakeControl(engine), db)
    assert result.outcome.cls is Class.DEGRADED
    assert gate in result.gates
    assert result.outcome.passes_capacity is True
    assert result.outcome.passes_quality is False


async def test_repetition_trips_only_on_the_probe_that_allows_it(engine, db):
    """The detector emits shingles at every character offset, so it must not run
    on the echo probe, which repeats a token on purpose."""
    engine.set(mode="repeat")
    suite = build_suite(run_id="r", seed=1, approx_tokens=64)
    summary = next(p for p in suite if p.id == "summary")
    echo = next(p for p in suite if p.id == "echo")

    looping = await _one(FakeControl(engine), db, probe=summary)
    assert Gate.REPEATING in looping.gates

    quiet = await _one(FakeControl(engine), db, probe=echo)
    assert Gate.REPEATING not in quiet.gates


async def test_short_is_relative_to_the_probes_own_baseline(engine, db):
    suite = build_suite(run_id="r", seed=1, approx_tokens=64)
    summary = next(p for p in suite if p.id == "summary")

    full = await _one(FakeControl(engine), db, probe=summary)
    engine.set(mode="short")
    ref = Reference(output_tokens=full.output_tokens, checks_length=True)
    stunted = await _one(FakeControl(engine), db, probe=summary, ref=ref)

    assert Gate.SHORT in stunted.gates


async def test_a_typed_context_error_is_a_refusal_with_no_confirmation(engine, db):
    """``error.type`` is machine-readable and definitive; nothing else is needed.

    No crash budget is spent, because the engine did exactly what it should."""
    engine.set(refuse_above_tokens=8, refuse_status=400)
    result = await _one(FakeControl(engine), db)
    assert result.outcome.cls is Class.REFUSED
    assert result.outcome.refusal_confidence == "high"
    assert result.outcome.consumes_crash_budget is False


async def test_a_500_shaped_refusal_is_still_a_refusal_but_marked_low(engine, db):
    """llama.cpp emits one. The shape is weaker evidence than a 4xx, so the
    record says so instead of flattening the distinction.

    ``error.type`` here is NOT the context discriminator, so the classifier has
    to reach the definitional test — the request exceeded the row's declared
    max_model_len — rather than short-circuiting on a typed envelope."""
    engine.set(refuse_above_tokens=8, refuse_status=500, refuse_error_type="server_error")
    control = FakeControl(engine, caps=Caps(max_model_len=4, max_num_seqs=4))
    result = await _one(control, db)
    assert result.outcome.cls is Class.REFUSED
    assert result.outcome.refusal_confidence == "low"


async def test_429_is_backpressure_and_is_never_published_as_a_limit(engine, db):
    engine.set(mode="rate_limit")
    result = await _one(FakeControl(engine), db)
    assert result.outcome.cls is Class.BACKPRESSURE
    assert result.outcome.publishable is False
    assert result.outcome.retry is True


async def test_an_unconfirmed_error_is_errored_not_refused(engine, db):
    engine.set(mode="error")
    result = await _one(FakeControl(engine), db)
    assert result.outcome.cls is Class.ERRORED
    assert result.outcome.retry is True


async def test_the_harness_imposes_the_only_deadline_there_is(engine, db):
    """Nothing in the stack times out: httpx runs with ``timeout=None``, the
    scheduler queue wait is unbounded and ``request_max_wall_s`` defaults to 0."""
    engine.set(mode="hang", hang_s=30.0)
    result = await _one(FakeControl(engine), db, deadline_s=0.3)
    assert result.outcome.cls is Class.TIMEOUT
    assert result.outcome.consumes_crash_budget is False


async def test_a_dead_engine_is_crashed_and_consumes_budget(engine, db):
    control = FakeControl(engine)
    await engine.crash()
    result = await _one(control, db)
    assert result.outcome.cls is Class.CRASHED
    assert result.outcome.consumes_crash_budget is True
    assert result.outcome.recover is True


async def test_foreign_traffic_makes_the_probe_inconclusive(engine, db):
    control = FakeControl(engine)
    control.traffic = 1
    result = await _one(control, db)
    assert result.outcome.cls is Class.INCONCLUSIVE
    assert result.outcome.publishable is False


# ---------------------------------------------------------------------------
# the trap that would publish a harness bug as model degradation
# ---------------------------------------------------------------------------


async def test_the_runner_sends_the_real_served_model_name(engine, db):
    await _one(FakeControl(engine), db)
    assert engine.knobs.seen[-1]["model"] == SERVED


async def test_an_unknown_model_name_returns_200_and_garbage(engine, db):
    """Recorded behaviour, not a hypothetical: llama-server serves the loaded
    model anyway and returns empty content with finish_reason=length. A runner
    that sent the model ROW ID would publish that as degradation."""
    control = FakeControl(engine, served_model_name="not-the-loaded-model")
    result = await _one(control, db)
    assert result.http_status == 200
    assert Gate.REASONING_ONLY in result.gates
    assert Gate.ABRUPT in result.gates


# ---------------------------------------------------------------------------
# crash -> recover -> re-resolve
# ---------------------------------------------------------------------------


async def test_recovery_re_resolves_the_port_after_a_restart(engine, db):
    """``_restart`` allocates a NEW port. A cached one means every later probe
    connects to nothing, is classified CRASHED, and burns the crash budget in
    seconds — one real crash becomes an exhausted budget and a limit far below
    the truth."""
    control = FakeControl(engine)
    run = make_runner(control, db, mode="quick")
    old_port = engine.port
    probe = build_suite(run_id="r", seed=1, approx_tokens=64)[0]

    async with httpx.AsyncClient(timeout=None) as client:
        run._client = client
        await engine.crash()
        crashed = await run._probe_with_recovery(
            probe, ref=Reference(output_tokens=0, checks_length=False)
        )
        assert crashed.outcome.cls is Class.CRASHED
        assert control.restarts == 1
        assert engine.port != old_port

        # The proof that the port was re-resolved rather than cached: the very
        # next probe succeeds against the new one.
        after = await run._probe_with_recovery(
            probe, ref=Reference(output_tokens=0, checks_length=False)
        )
        assert after.outcome.cls is Class.PASS
    assert control.ports_seen[-1] == engine.port


async def test_the_crash_budget_stops_the_run_and_says_so(engine, db):
    """A model that dies at a size with no degradation phase first — the
    measured 2026-09-03 Quadro behaviour — must not loop forever restarting."""
    engine.set(crash_above_tokens=600)
    control = FakeControl(engine, caps=Caps(max_model_len=4096, max_num_seqs=1))
    run = make_runner(control, db, mode="quick", length_floor=256)
    row = await run.run()

    assert run._truncated_by == "crash_budget"
    assert row.truncated_by == "crash_budget"
    # Two recoveries inside the budget; the third crash stops the run instead of
    # spending a fourth. The last restart is the postcondition putting the
    # engine back, which happens whatever the reason the run ended.
    assert control.restarts == 3


async def test_conservative_mode_aborts_on_the_first_crash(engine, db):
    engine.set(crash_above_tokens=600)
    control = FakeControl(engine, caps=Caps(max_model_len=4096, max_num_seqs=1))
    run = make_runner(control, db, mode="conservative", length_floor=256)
    await run.run()
    assert run._truncated_by == "crash_budget"
    # No recovery inside the run at all -- only the postcondition's restore.
    assert control.restarts == 1


# ---------------------------------------------------------------------------
# the wedged-but-healthy engine
# ---------------------------------------------------------------------------


async def test_three_timeouts_against_a_healthy_engine_means_wedged(engine, db):
    """/health returns 200 while the scheduler is deadlocked, so every probe
    times out — and TIMEOUT spends no crash budget and does not stop the search.
    Without this the run burns the deadline on every remaining rung."""
    engine.set(mode="wedge")
    control = FakeControl(engine)
    run = make_runner(control, db, mode="quick", deadline_floor_s=0.2, deadline_cap_s=0.2)
    probe = build_suite(run_id="r", seed=1, approx_tokens=64)[0]
    ref = Reference(output_tokens=0, checks_length=False)

    async with httpx.AsyncClient(timeout=None) as client:
        run._client = client
        assert await control.health() is True  # the pathology, precisely
        for _ in range(2):
            await run._probe_with_recovery(probe, ref=ref)
        with pytest.raises(_AxisTruncated) as caught:
            await run._probe_with_recovery(probe, ref=ref)

    assert caught.value.reason == "wedged"
    assert control.restarts == 1


async def test_a_wedged_engine_truncates_the_whole_run(engine, db):
    engine.set(mode="wedge")
    control = FakeControl(engine)
    run = make_runner(control, db, mode="quick", deadline_floor_s=0.2, deadline_cap_s=0.2)
    row = await run.run()
    assert row.truncated_by == "wedged"


# ---------------------------------------------------------------------------
# blast radius
# ---------------------------------------------------------------------------


async def test_a_neighbour_going_unhealthy_aborts_and_publishes_nothing(engine, db):
    """Co-residency is normal and a crashing engine can take a co-resident one
    down with it. The neighbour holds no lease and never will — its recovery is
    the watchdog's job."""
    control = FakeControl(engine, neighbours={"neighbour-a": True})
    leases = LeaseRegistry()
    run = make_runner(control, db, leases=leases)

    calls = {"n": 0}

    original = control.neighbour_health

    async def failing_after_a_few():
        calls["n"] += 1
        if calls["n"] > 2:
            control.set_neighbour("neighbour-a", False)
        return await original()

    control.neighbour_health = failing_after_a_few
    row = await run.run()

    assert row.status == "aborted"
    assert (row.limits or {})["aborted_reason"] == "aborted_neighbour_impact"
    assert "quality_context" not in (row.limits or {})
    assert leases.is_held("neighbour-a") is False


async def test_traffic_on_this_model_refuses_the_run(engine, db):
    control = FakeControl(engine)
    control.traffic = 1
    row = await make_runner(control, db).run()
    assert row.status == "aborted"
    assert (row.limits or {})["aborted_reason"] == "aborted_traffic"


async def test_forcing_past_traffic_taints_the_result(engine, db):
    """No partial credit: KV is a shared pool, so traffic moves the measurement
    by an unknown amount and the whole run is tainted."""
    control = FakeControl(engine, caps=Caps(max_model_len=1024, max_num_seqs=1))
    control.traffic = 1
    row = await make_runner(control, db, force=True, length_floor=256).run()
    assert row.traffic_observed is True


# ---------------------------------------------------------------------------
# the lease
# ---------------------------------------------------------------------------


async def test_the_lease_is_held_during_the_run_and_released_after(engine, db):
    control = FakeControl(engine, caps=Caps(max_model_len=512, max_num_seqs=1))
    leases = LeaseRegistry()
    held: list[bool] = []
    control.on_target = lambda: held.append(leases.is_held(MODEL_ID))

    await make_runner(control, db, leases=leases, length_floor=256).run()

    assert held and all(held)
    assert leases.is_held(MODEL_ID) is False


async def test_the_lease_is_released_even_when_the_run_raises(engine, db):
    """``finally`` is the only recovery path this feature has, so it has to hold
    for an exception nobody anticipated, not just the ones with names."""
    control = FakeControl(engine)
    leases = LeaseRegistry()

    async def boom():
        raise RuntimeError("something nobody planned for")

    control.neighbour_health = boom
    row = await make_runner(control, db, leases=leases).run()

    assert row.status == "aborted"
    assert (row.limits or {})["aborted_reason"] == "harness_exception"
    assert "RuntimeError" in (row.last_error or "")
    assert leases.is_held(MODEL_ID) is False
    assert control.budget_resets == 1


async def test_an_unrecoverable_engine_is_marked_and_not_looped_on(engine, db):
    """Two attempts, then stop. Looping here IS the crash loop the restart
    budget exists to prevent."""
    control = FakeControl(engine)
    run = make_runner(control, db, length_floor=256)
    await engine.crash()
    control.restart_raises = RuntimeError("engine will not come back")

    row = await run.run()
    assert row.status == "failed_unrecovered"
    # Two recovery attempts in the postcondition, and no more.
    assert control.restarts == 2


# ---------------------------------------------------------------------------
# the axes
# ---------------------------------------------------------------------------


async def test_a_quality_wall_becomes_the_published_context_number(engine, db):
    """The number this feature exists to produce: the largest prompt size at
    which the answers still hold up."""
    engine.set(degrade_above_tokens=1200, degrade_mode="empty")
    control = FakeControl(engine, caps=Caps(max_model_len=8192, max_num_seqs=1))
    row = await make_runner(control, db, length_floor=256).run()

    assert row.status == "completed"
    quality = (row.limits or {})["quality_context"]
    assert quality["publishable"] is True
    assert quality["raw_confirmed"] <= 1400
    assert quality["first_observed_failure"] is not None
    assert quality["limited_by"] == "quality"
    assert quality["gate_tripped"] == "empty"
    # Structured, not prose: a client must be able to tell a 21% contour from a
    # 9% one without parsing English.
    assert quality["oracle"]["n_consecutive"] == 3
    assert 0.2 < quality["oracle"]["contour_p"] < 0.22


async def test_the_baseline_records_an_answer_length_per_probe(engine, db):
    control = FakeControl(engine, caps=Caps(max_model_len=256, max_num_seqs=1))
    row = await make_runner(control, db, length_floor=256).run()
    baseline = (row.limits or {})["baseline"]
    # Clamped to a share of THIS model's context rather than taken at the
    # requested 256: a baseline prompt that fills the context leaves nothing to
    # generate into, so the reference answer would be empty by construction and
    # every later SHORT verdict would be measured against nothing. The
    # invariant, not a magic number.
    assert 0 < baseline["approx_tokens"] <= 256 // 2
    assert set(baseline["answer_tokens"]) == {
        p.id for p in build_suite(run_id="r", seed=1, approx_tokens=256)
    }
    assert baseline["answer_tokens"]["summary"] > 0


async def test_the_concurrency_axis_defeats_the_prefix_cache(engine, db):
    """Every probe carries a unique high-entropy preamble at position 0. Without
    it, concurrent probes share a cacheable prefix, the engine deduplicates
    their KV, and the axis reports a number no real traffic can reach."""
    control = FakeControl(engine, caps=Caps(max_model_len=256, max_num_seqs=4))
    run = make_runner(control, db, length_floor=256)
    row = await run.run()

    concurrency = (row.limits or {})["concurrency"]
    assert concurrency["publishable"] is True
    assert run._prefix_cache_suspect is False
    observed = [o for o in (row.observations or []) if o["axis"] == "concurrency"]
    assert observed
    assert all((o["cached_tokens"] or 0) == 0 for o in observed)


async def test_a_shared_prefix_withholds_the_concurrency_number(engine, db):
    """The verification is the point. If the engine served the prompts from
    cache, the axis measured deduplicated KV and its number is meaningless — so
    it is withheld rather than published, which is the safe direction."""
    control = FakeControl(engine, caps=Caps(max_model_len=256, max_num_seqs=2))
    run = make_runner(control, db, length_floor=256)

    fixed = run._concurrency_probe()
    run._concurrency_probe = lambda: Probe(
        id="concurrency",
        messages=fixed.messages,
        max_tokens=24,
        checks_length=False,
        allow_repetition_gate=False,
        expected=fixed.expected,
        grader="substring",
    )
    row = await run.run()

    concurrency = (row.limits or {})["concurrency"]
    assert concurrency["publishable"] is False
    assert concurrency["invalid_reason"] == "prefix_cache_not_defeated"
    assert "value" not in concurrency


async def test_phase_d_withholds_a_non_monotone_axis(engine, db, monkeypatch):
    """Bisection needs the failure region upward-closed, and a CUDA alignment
    fault is notoriously non-monotone in size. One probe at the axis maximum
    converts a silent 4x error into a visible gap."""
    control = FakeControl(engine, caps=Caps(max_model_len=4096, max_num_seqs=1))
    run = make_runner(control, db, length_floor=256)

    # An island of failure: everything between 400 and 1500 fails, the top does
    # not. A bisecting search locks onto the dip and converges, confidently, on
    # a value several times too low.
    async def islanded(tokens: int) -> bool:
        return not (400 < tokens < 1500)

    monkeypatch.setattr(run, "_trial_length", islanded)
    row = await run.run()

    assert row.non_monotone is True
    quality = (row.limits or {})["quality_context"]
    assert quality["publishable"] is False
    assert quality["invalid_reason"] == "non_monotone"


# ---------------------------------------------------------------------------
# the sweep — the primary output
# ---------------------------------------------------------------------------


async def test_the_sweep_recommends_the_largest_context_that_loaded(engine, db):
    """An in-place run can only say how much of the ALREADY ALLOCATED context is
    usable. The question an operator actually has is what the setting should be,
    and only a reload can answer it."""
    control = FakeControl(engine, caps=Caps(max_model_len=512, max_num_seqs=1, model_ceiling=2048))
    control.reload_fails_above = 1024
    row = await make_runner(control, db, mode="thorough", length_floor=256).run()

    assert row.recommended_config is not None
    assert row.recommended_config["max_model_len"] == 1024
    assert row.recommended_config["limited_by"] == "load"
    assert row.recommended_config["next_failed_at"] == 2048
    # The engine is left in the configuration we found it in; the
    # recommendation is advisory and applying it is the operator's call.
    assert control.reloads[-1] == 512


async def test_a_conservative_run_never_sweeps(engine, db):
    control = FakeControl(engine, caps=Caps(max_model_len=512, max_num_seqs=1))
    row = await make_runner(control, db, length_floor=256).run()
    assert row.recommended_config is None
    assert control.reloads == []


# ---------------------------------------------------------------------------
# persistence and identity
# ---------------------------------------------------------------------------


async def test_the_run_row_carries_the_fingerprint_it_was_measured_under(engine, db):
    control = FakeControl(engine, caps=Caps(max_model_len=256, max_num_seqs=1))
    row = await make_runner(control, db, length_floor=256).run()
    assert row.fingerprint.startswith("sha256:")
    assert row.algorithm_version == routes_api.ALGORITHM_VERSION
    assert row.probe_suite_hash == routes_api.PROBE_SUITE_HASH


def test_the_algorithm_version_has_exactly_one_home():
    """The reader computes the fingerprint without importing the runner, so the
    two sides must agree. If they ever drift by one, every stored measurement
    becomes permanently unmatchable — recorded, never published, no error."""
    assert runner_mod.algorithm_version() == routes_api.ALGORITHM_VERSION


class _Req:
    def __init__(self, model: str) -> None:
        self.model = model


class _Registry:
    def __init__(self, *models: str) -> None:
        self._live = [_Req(m) for m in models]

    def count(self) -> int:
        return len(self._live)

    def snapshot(self):
        return list(self._live)


def test_traffic_is_counted_per_model_and_never_globally():
    """``RequestRegistry.count()`` spans EVERY model, so a guard built on it
    blocks a run on model A because somebody is chatting with model B on
    another GPU — a multi-tenancy failure dressed up as a busy-ness check."""
    from types import SimpleNamespace

    from app.stress.runner import WardenEngineControl

    registry = _Registry("someone-elses-model", "another-one")
    app_state = SimpleNamespace(request_registry=registry, supervisor=None)
    model = SimpleNamespace(id=MODEL_ID, served_model_name=SERVED)
    control = WardenEngineControl(
        SimpleNamespace(),
        app_state,
        model,
        subject=_subject(),
        caps=Caps(),
    )

    assert registry.count() == 2
    assert control.foreign_requests() == 0

    registry._live.append(_Req(SERVED))
    assert control.foreign_requests() == 1


async def test_a_completed_run_is_readable_through_current_for(engine, db):
    """Headroom is required, and that is the point rather than test scaffolding.

    `current_for` now also asks whether the run MEASURED anything. With
    max_model_len == length_floor, `_axis_length` returns before writing a
    `quality_context` key at all (there is no rung above the floor to climb),
    and max_num_seqs=1 skips concurrency the same way — so the run completes
    having recorded only `baseline` and `neighbours`, which are bookkeeping.
    Giving it somewhere to climb makes it a real measurement.
    """
    control = FakeControl(engine, caps=Caps(max_model_len=2048, max_num_seqs=1))
    row = await make_runner(control, db, length_floor=256).run()
    got = await StressRunRepo(db).current_for(MODEL_ID, row.fingerprint)
    assert got is not None
    assert got.id == row.id


async def test_a_small_context_model_still_gets_a_real_measurement(engine, db):
    """What clamping the baseline bought.

    This configuration -- a 256-token context on a single sequence -- used to
    measure NOTHING. The requested baseline equalled the context, so
    `_axis_length` returned before writing a key (no rung above the floor) and
    concurrency was skipped for max_num_seqs=1; the run completed honestly
    having found nothing at all.

    Sizing the baseline to the model instead leaves headroom by construction,
    so the ladder has somewhere to climb and the run produces a number. The
    "a run that measured nothing must not be published" rule it used to pin
    lives in test_stress_runs_repo.py, where it can be expressed directly.
    """
    control = FakeControl(engine, caps=Caps(max_model_len=256, max_num_seqs=1))
    row = await make_runner(control, db, length_floor=256).run()
    assert row.status == "completed"
    assert await StressRunRepo(db).current_for(MODEL_ID, row.fingerprint) is not None


async def test_cancelling_the_task_still_runs_the_postcondition(engine, db):
    """The run is in-process and dies with the warden; a cancelled one must not
    leave the lease held or the row stuck at 'running'."""
    engine.set(mode="hang", hang_s=30.0)
    control = FakeControl(engine, caps=Caps(max_model_len=256, max_num_seqs=1))
    leases = LeaseRegistry()
    run = make_runner(control, db, leases=leases, length_floor=256, deadline_floor_s=30.0)

    task = asyncio.create_task(run.run())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert leases.is_held(MODEL_ID) is False
    row = await StressRunRepo(db).get(run.run_id)
    assert row is not None and row.status == "interrupted"


class _RecordingBridge(runner_mod._Bridge):
    """A ``_Bridge`` that records which worker thread ran it and when it returned.

    R37: ``_Bridge.call`` parks the bisect worker on an unbounded
    ``run_coroutine_threadsafe(...).result()``. A run cancelled mid-bisect left
    that non-daemon thread blocked forever. We assert on *this* worker's
    unwind rather than the process thread table, because idle executor threads
    linger in ``threading.enumerate()`` even in the healthy case and a stuck one
    would make ``executor.shutdown(wait=True)`` hang -- the table cannot tell
    "still working" from "leaked".
    """

    instances: list[_RecordingBridge] = []

    def __init__(self, loop: asyncio.AbstractEventLoop, *args, **kwargs) -> None:
        super().__init__(loop, *args, **kwargs)
        self.worker: threading.Thread | None = None
        self.returned = threading.Event()
        type(self).instances.append(self)

    def call(self, coro):
        self.worker = threading.current_thread()
        try:
            return super().call(coro)
        finally:
            self.returned.set()


async def test_cancelling_during_bisect_unwinds_the_thread_bridge(engine, db, monkeypatch):
    """A run cancelled while the length axis is bisecting must not leave the
    bisect's worker thread blocked on the bridge.

    The worker is parked on ``run_coroutine_threadsafe(...).result()`` inside
    ``_Bridge.call``. Threads cannot be cancelled, so the only way it unwinds is
    the bridge noticing the run died and bailing; without that the non-daemon
    ``asyncio_0`` thread outlives the run (conftest's session-end watchdog names
    exactly these) and the next healthy run inherits a leaked thread.
    """
    control = FakeControl(engine, caps=Caps(max_model_len=8192, max_num_seqs=1))
    run = make_runner(control, db, length_floor=256)

    gate = asyncio.Event()

    async def trial(tokens: int) -> bool:
        # Ladder rungs are powers of two: pass below 512, fail at 1024, so the
        # bracket is (512, 1024) and Phase B bisects. The midpoints are not
        # powers of two and hang, parking the bisect worker on the bridge.
        if tokens & (tokens - 1) == 0:
            return tokens <= 512
        await gate.wait()
        return True

    monkeypatch.setattr(run, "_trial_length", trial)
    monkeypatch.setattr(runner_mod, "_Bridge", _RecordingBridge)
    _RecordingBridge.instances.clear()

    task = asyncio.create_task(run.run())
    main = threading.main_thread()
    await wait_until_async(
        lambda: any(
            b.worker is not None and b.worker is not main for b in _RecordingBridge.instances
        ),
        timeout_s=10,
        what="the bisect worker to park on the bridge",
    )

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await wait_until_async(
        lambda: any(b.returned.is_set() for b in _RecordingBridge.instances),
        timeout_s=10,
        what="the bisect worker to unwind after the run was cancelled",
    )

    row = await StressRunRepo(db).get(run.run_id)
    assert row is not None and row.status == "interrupted"


async def test_the_bridge_bails_when_a_probe_outlives_its_bound():
    """The stop flag is the primary mechanism; the per-call bound is the backstop
    for a probe that even the deadline fails to recall. Either way a worker must
    not pin its non-daemon thread for the life of the process, so the bound is
    exercised on its own -- no cancel, no stop flag set."""
    loop = asyncio.get_running_loop()
    gate = asyncio.Event()

    async def never() -> None:
        await gate.wait()

    bridge = _Bridge(loop, stop=threading.Event(), timeout_s=0.5)
    errors: list[BaseException] = []

    def call() -> None:
        try:
            bridge.call(never())
        except BaseException as exc:  # noqa: BLE001 -- captured, asserted below
            errors.append(exc)

    await asyncio.to_thread(call)

    assert len(errors) == 1
    assert isinstance(errors[0], _BridgeStopped)

    gate.set()  # release the orphaned probe so the loop can drain it


# ---------------------------------------------------------------------------
# budget calibration: telling "the model failed" from "we starved it"
#
# The first production run (2026-09-04) failed every probe at the baseline on
# gpt-oss-20b-gguf, a healthy model, because the suite budgets ~4x the ANSWER
# and a reasoning model spends the whole budget thinking. Measured on the live
# engine: max_tokens=16 -> content='', finish=length; max_tokens=64 -> '42'.
# ---------------------------------------------------------------------------


async def test_a_starved_probe_is_marked_starved(engine, db):
    """The signature is recorded on the run, not merely gated.

    The gates fire truthfully -- the output really is abrupt and reasoning-only
    -- so only `starved` can tell the caller the fault was ours.
    """
    engine.set(reasoning_tokens=200)
    result = await _one(FakeControl(engine), db)
    assert result.starved is True
    assert Gate.REASONING_ONLY in result.gates


async def test_a_model_that_answers_is_never_marked_starved(engine, db):
    """A reasoning model with room to answer is just a healthy model."""
    engine.set(reasoning_tokens=2)
    result = await _one(FakeControl(engine), db)
    assert result.starved is False


async def test_the_baseline_escalates_until_the_model_can_answer(engine, db):
    """The whole point: a reasoning model must yield a usable baseline.

    reasoning_tokens=40 starves the 16-token arithmetic probe and is cleared at
    64 -- the exact step measured against the real engine.
    """
    engine.set(reasoning_tokens=40)
    run = make_runner(FakeControl(engine), db)
    async with httpx.AsyncClient(timeout=None) as client:
        run._client = client
        await run._baseline()

    assert run._budgets["arithmetic"] > 16, "budget was never raised"
    assert run._limits["baseline"]["max_tokens"]["arithmetic"] >= 64
    # A reference with a real answer behind it, which is what every later
    # SHORT verdict is measured against.
    assert run._references["arithmetic"].output_tokens > 0


async def test_a_model_that_never_answers_aborts_instead_of_publishing_nothing(engine, db):
    """Distinct from 'the model degraded'.

    Before this, such a run completed with `no_confirmed_value`, which reads as
    a verdict on the model when the truth is that the harness never got an
    answer out of it -- and it spent the whole sweep to say so.
    """
    engine.set(reasoning_tokens=100_000)
    run = make_runner(FakeControl(engine), db)
    with pytest.raises(runner_mod.StressAborted) as exc:
        async with httpx.AsyncClient(timeout=None) as client:
            run._client = client
            await run._baseline()
    assert exc.value.reason == "aborted_no_baseline"


async def test_the_settled_budget_survives_into_the_sweep(engine, db):
    """build_suite is called again at every candidate, so an uncarried
    calibration would starve every probe after the baseline."""
    engine.set(reasoning_tokens=40)
    run = make_runner(FakeControl(engine), db)
    async with httpx.AsyncClient(timeout=None) as client:
        run._client = client
        await run._baseline()

    fresh = build_suite(run_id=run.run_id, seed=run.seed, approx_tokens=1024)
    calibrated = {p.id: p.max_tokens for p in run._calibrated(fresh)}
    assert calibrated["arithmetic"] == run._budgets["arithmetic"]
    assert calibrated["arithmetic"] > 16


async def test_the_concurrency_probe_inherits_the_calibration(engine, db):
    """Same shape of task as the needle probe, so the same overhead applies."""
    engine.set(reasoning_tokens=40)
    run = make_runner(FakeControl(engine), db)
    async with httpx.AsyncClient(timeout=None) as client:
        run._client = client
        await run._baseline()
    assert run._concurrency_probe().max_tokens == run._budgets["needle"]


async def test_probes_that_fail_at_baseline_are_quarantined(engine, db):
    """The second production failure, 2026-09-04.

    `needle` passed at every length up to the axis maximum; `summary` tripped
    `repeating` at every length INCLUDING the baseline, so every rung failed
    and nothing was ever confirmed. A gate that reads the same at the reference
    length as at the ceiling cannot locate a threshold — it only hides one.

    The engine mode here is global, so every probe fails and the ladder empties
    entirely: that is the abort path. Quarantining a SINGLE probe while others
    still discriminate is covered by test_baseline_quarantine.py, which can
    express one failing probe without a per-probe fake.
    """
    engine.set(mode="wrong")
    run = make_runner(FakeControl(engine), db)
    with pytest.raises(runner_mod.StressAborted) as exc:
        async with httpx.AsyncClient(timeout=None) as client:
            run._client = client
            await run._baseline()
    # Not "the model is bad": no probe could discriminate, so there is no
    # oracle. Publishing the axis maximum here would be a number off no
    # evidence at all.
    assert exc.value.reason == "aborted_no_baseline"


async def test_a_clean_baseline_excludes_nothing(engine, db):
    run = make_runner(FakeControl(engine), db)
    async with httpx.AsyncClient(timeout=None) as client:
        run._client = client
        await run._baseline()
    assert run._limits["baseline"]["excluded_probes"] == []
    assert run._excluded == ()


async def test_a_rung_that_starves_the_probe_raises_the_budget_not_the_verdict(engine, db):
    """The published 1,616-token limit that was really our own budget.

    reasoning_tokens=200 clears the baseline's escalation (which reaches 1024)
    but would starve a probe left at its baseline budget. The ladder must raise
    the budget and re-probe rather than record `abrupt + reasoning_only + wrong`
    and call it the model's limit.
    """
    engine.set(reasoning_tokens=200)
    run = make_runner(FakeControl(engine), db)
    async with httpx.AsyncClient(timeout=None) as client:
        run._client = client
        await run._baseline()
        settled = dict(run._budgets)
        ok = await run._trial_length(2048)

    assert ok, "a healthy model was failed by our own token budget"
    # Budgets only ever grow, so a later rung is never judged more harshly
    # than an earlier one.
    for pid, before in settled.items():
        assert run._budgets.get(pid, before) >= before
