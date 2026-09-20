"""The proxy records how long a request waited at the admission gate.

Waiting for a slot in ``PriorityScheduler`` was invisible: ``started_monotonic``
-- the clock ``ttft_s`` and ``duration_s`` are both measured from -- is read
AFTER the slot is held, so a request that queued for ten seconds looked
identical to one that did not. The scheduler's docstring warns that priority-9
traffic can starve lower priorities indefinitely, and nothing could show it.

The scheduler's own queueing is covered by test_scheduler.py. What is pinned
here is the wiring in ``_forward``: that it times the acquire ALONE and puts
the result on the finished record.

Backend-independent by construction -- ``scheduler.acquire`` sits in the single
``_forward`` path ahead of any backend branching, so these assertions hold for
llama.cpp exactly as for vLLM.
"""

import asyncio
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt

from app.db.repos.tokens import hash_token

# Seconds the fake gate is held shut. With the fake clock below this is an
# EXACT figure, not a margin: queued_s must come back as exactly this value.
DELAY = 2.0


def _seed_loaded(db_path):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0, 1, 2, 3]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,'auto',4096,0.9,0,'[]','loaded',0,NULL,NULL)",
            (json.dumps([0]),),
        )
        plaintext = "vw_validtoken1234567890abcdef12345"
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
        return plaintext


class _SlowScheduler:
    """A scheduler whose admission gate takes ``delay`` seconds to open.

    Stands in for a saturated engine: the real PriorityScheduler makes a
    waiter sit in a heap for exactly this reason, and routes.py cannot tell
    the difference between waiting on a heap and waiting on a sleep.

    With a ``clock`` (a ``_FakeClock``), the gate advances that clock by
    ``delay`` instead of actually sleeping -- see
    ``test_a_request_that_waited_for_a_slot_records_the_wait`` below, which
    needs the wait to land in a controlled clock rather than real wall time.
    Without one, it sleeps for real, which is fine for the other tests in
    this file: they only ever use ``delay=0.0``.
    """

    def __init__(self, delay: float, clock: "_FakeClock | None" = None) -> None:
        self.delay = delay
        self.clock = clock
        self.seen: list[tuple[int, str]] = []

    @asynccontextmanager
    async def acquire(self, *, priority: int = 0, engine_key: str = ""):
        self.seen.append((priority, engine_key))
        if self.clock is not None:
            self.clock.advance(self.delay)
            # Yield control without consuming real or fake time, so any
            # other coroutine scheduled to run gets its turn.
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(self.delay)
        yield


class _FakeClock:
    """A monotonic clock that only moves when told to.

    Patches ``app.proxy.routes._monotonic`` -- the module-level indirection
    ``_forward`` reads for queue-wait / duration / TTFT timing -- NOT the
    real ``time.monotonic``. Patching the real one would also freeze
    asyncio's own event-loop clock (``loop.time()`` reads it too), which can
    wedge any coroutine with a real scheduled callback (background writers,
    retries, ...) rather than just this one code path.

    Every read besides the gate's ``advance()`` above sees zero elapsed time,
    which is what turns the assertions below into exact values instead of
    wall-clock margins.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


def _fake_tokenizer():
    cache = MagicMock()
    cache.count = AsyncMock(
        side_effect=lambda repo, text, *, fallback_repo=None: 5
    )
    return cache


def _post(client, plaintext):
    body = {
        "id": "x", "model": "qwen",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(return_value=json.dumps(body).encode())
    fake_resp.aclose = AsyncMock()
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        return client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )


def _rows(db_path, n):
    """Wait for ``n`` rows, oldest first. Generous deadline on purpose: this
    waits on the store's BACKGROUND writer, which is not latency-sensitive."""
    deadline = time.time() + 30
    while time.time() < deadline:
        with sqlite3.connect(db_path) as db:
            rows = db.execute(
                "SELECT queued_s, ttft_s, duration_s FROM request_history "
                "ORDER BY finished_at ASC"
            ).fetchall()
        if len(rows) >= n:
            return rows
        time.sleep(0.05)
    raise AssertionError(f"only {len(rows)} of {n} request_history rows were written")


def test_a_request_that_waited_for_a_slot_records_the_wait(tmp_data_dir, client):
    """Deterministic by construction, not by a bigger margin.

    This test is on its third design; the first two compared wall-clock
    durations across two real requests and both failed in CI on runner
    jitter -- most recently a ~1.4s stall from an unrelated xdist worker
    landing inside the timed section of the second request (pipeline 25468,
    job 286221; vllm-warden#253). Absolute thresholds do not work here
    either: `duration_s < 0.25` assumed the request itself was fast, and
    `duration_s < queued_s` assumed the request was faster than the injected
    delay.

    This design removes the runner from the picture entirely: `_forward`'s
    own clock (`app.proxy.routes._monotonic`) is replaced with a `_FakeClock`
    that only advances when `_SlowScheduler.acquire` tells it to, by exactly
    DELAY. No real sleep happens, so nothing here is sensitive to how fast or
    slow the machine running it is -- the assertions are exact values, not
    margins.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()

    clock = _FakeClock()
    with patch("app.proxy.routes._monotonic", clock):
        # An open gate (0.0) and a gate held shut for DELAY. Both requests run
        # against the same fake clock, which advances only inside `acquire`.
        client.app.state.scheduler = _SlowScheduler(0.0, clock=clock)
        assert _post(client, plaintext).status_code == 200

        client.app.state.scheduler = _SlowScheduler(DELAY, clock=clock)
        assert _post(client, plaintext).status_code == 200

    (base_queued, _, base_duration), (queued, _, duration) = _rows(db_path, 2)

    # The wait itself is measured, exactly -- the fake clock advanced by
    # precisely DELAY inside `acquire`, and by nothing on the open-gate run.
    assert queued == DELAY
    assert base_queued == 0.0

    # ...and it did NOT land in duration_s. `started_monotonic` is read AFTER
    # the acquire, so the timed section spans zero fake seconds on both runs.
    # If `started_monotonic` were read BEFORE the acquire, the delayed run's
    # duration would carry the whole DELAY the gate advanced the clock by,
    # and this would fail with `duration == DELAY` instead.
    assert duration == base_duration, (
        f"duration grew by {duration - base_duration:.3f}s when the gate was "
        f"held {DELAY}s — the queue wait is inside duration_s"
    )


def test_an_unblocked_request_records_zero_rather_than_absent(tmp_data_dir, client):
    # Zero is a real reading -- the engine had a free slot -- and is what the
    # REAL scheduler produces uncontended. Only a build that never measured
    # should store NULL.
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()

    assert _post(client, plaintext).status_code == 200

    (queued_s, _, _), = _rows(db_path, 1)
    assert queued_s is not None
    # Generous on purpose: the claim is "a real, small number rather than
    # absent", not a latency budget for the runner.
    assert queued_s < 1.0


def test_the_admission_gate_is_asked_for_the_tokens_priority_and_the_engine(
    tmp_data_dir, client
):
    # Pins that timing the acquire did not change what it is called with.
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()
    sched = _SlowScheduler(0.0)
    client.app.state.scheduler = sched

    assert _post(client, plaintext).status_code == 200
    # 5 is the api_tokens table default (migration 0018), not a value this
    # test set -- the point is the engine key and that the priority is the
    # token's, not a constant.
    assert sched.seen == [(5, "qwen")]
