"""The proxy's per-engine admission cap scales with the engine's real capacity.

``PriorityScheduler`` admitted a fixed ``VW_PROXY_MAX_INFLIGHT`` (16) requests
per engine no matter what sat behind it. A data-parallel engine (
``--data-parallel-size 7`` x ``--max-num-seqs 64`` per rank) sustains far more
than 16 in flight, so the proxy sat on the engine's capacity: 4,955 tok/s
through the proxy vs 13,711 direct -- a 2.8x loss, the engine running tiny
batches while 96 requests queued at the gate (measured 2026-09-30, 7x RTX 5090;
issue #277, andon A1).

What is pinned here:
  * ``model_admission_cap`` computes the effective per-model cap from the
    engine's flags in the row's ``extra_args`` -- ``--max-num-seqs`` (M) times
    ``--data-parallel-size`` (N, default 1) -- when ``VW_PROXY_MAX_INFLIGHT``
    is unset, and falls back to the documented default (16) otherwise;
  * an explicit ``VW_PROXY_MAX_INFLIGHT`` is a GLOBAL override that wins for
    every model, with or without flags;
  * the scheduler actually USES the per-engine cap as its admission gate (the
    ``cap`` keyword), so a DP engine admits M*N concurrently while a separate
    engine with no override stays on the default;
  * the proxy wires it up: ``_forward`` passes the cap it computed from the
    model row's ``extra_args`` to the admission gate.
"""

import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt

from app.db.repos.tokens import hash_token
from app.proxy.scheduler import PriorityScheduler, model_admission_cap

# The owner's live 7x RTX 5090 shape, as it sits in the model row's extra_args:
# 7 data-parallel ranks, each admitting up to 64 sequences. 64 * 7 == 448, the
# value the owner currently forces host-local with VW_PROXY_MAX_INFLIGHT=448.
DP_EXTRA_ARGS = ["--data-parallel-size", "7", "--max-num-seqs", "64"]


# ---------------------------------------------------------------------------
# model_admission_cap: the pure computation
# ---------------------------------------------------------------------------


def test_admission_cap_is_computed_from_the_dp_flags(monkeypatch):
    """The item's own acceptance assertion: a model whose extra_args set
    ``--data-parallel-size N --max-num-seqs M`` gets a cap of M * N (not the
    hardcoded 16) when VW_PROXY_MAX_INFLIGHT is unset."""
    monkeypatch.delenv("VW_PROXY_MAX_INFLIGHT", raising=False)
    assert model_admission_cap(DP_EXTRA_ARGS) == 64 * 7


def test_admission_cap_env_override_wins_globally(monkeypatch):
    """A control: an explicit VW_PROXY_MAX_INFLIGHT is a GLOBAL override. It
    wins for every model, whether or not the model has DP flags, and it is not
    multiplied by the flag values."""
    monkeypatch.setenv("VW_PROXY_MAX_INFLIGHT", "100")
    assert model_admission_cap(DP_EXTRA_ARGS) == 100
    assert model_admission_cap([]) == 100


def test_admission_cap_single_gpu_keeps_a_sane_cap(monkeypatch):
    """A single-GPU model (max-num-seqs but no data-parallel-size) gets M, not
    M * 7. A model with no relevant flags at all keeps the documented default.
    A DP flag with no max-num-seqs cannot be turned into a product, so it too
    falls back to the default rather than guessing."""
    monkeypatch.delenv("VW_PROXY_MAX_INFLIGHT", raising=False)
    assert model_admission_cap(["--max-num-seqs", "64"]) == 64
    assert model_admission_cap([]) == 16
    assert model_admission_cap(["--data-parallel-size", "7"]) == 16


def test_admission_cap_reads_both_flag_forms(monkeypatch):
    """Both the ``--flag value`` and ``--flag=value`` argv spellings appear in
    the wild; both must parse to the same cap."""
    monkeypatch.delenv("VW_PROXY_MAX_INFLIGHT", raising=False)
    assert model_admission_cap(["--data-parallel-size=7", "--max-num-seqs=64"]) == 448
    # Interleaved with unrelated flags, as a real extra_args list is.
    assert (
        model_admission_cap(
            [
                "--enable-auto-tool-choice",
                "--data-parallel-size",
                "7",
                "--tool-call-parser",
                "hermes",
                "--max-num-seqs",
                "64",
            ]
        )
        == 448
    )


def test_admission_cap_garbage_flags_fall_back_to_the_default(monkeypatch):
    """A flag value that is not a positive integer must not zero-out or go
    negative the cap: it is treated as absent and the default wins."""
    monkeypatch.delenv("VW_PROXY_MAX_INFLIGHT", raising=False)
    assert model_admission_cap(["--max-num-seqs", "not-a-number"]) == 16
    assert model_admission_cap(["--max-num-seqs", "0"]) == 16
    # A bad DP value drops the product to M * 1, not M * 0.
    assert model_admission_cap(["--data-parallel-size", "-1", "--max-num-seqs", "64"]) == 64


# ---------------------------------------------------------------------------
# PriorityScheduler: the per-engine gate honours the cap keyword
# ---------------------------------------------------------------------------


async def test_acquire_cap_kwarg_overrides_the_global_default_per_engine():
    """The ``cap`` keyword is the per-engine admission gate. It raises the gate
    to the engine's real capacity (a DP engine admits M*N) even though the
    scheduler's own default is smaller, and it is independent per engine: the
    DP engine's in-flight requests do not consume a different engine's slots,
    which -- with no override -- stays on the global default.

    Forced manual-release so the nested holders stay admitted while we assert.
    """
    sched = PriorityScheduler(max_inflight=2)  # the global default here is 2
    sched._manual_release = True
    cap = 4  # e.g. --max-num-seqs 2 --data-parallel-size 2
    async with sched.acquire(priority=5, engine_key="dp", cap=cap):
        async with sched.acquire(priority=5, engine_key="dp", cap=cap):
            # Without the per-engine cap the third request would have queued
            # here (the global default is 2). With it, the DP engine admits up
            # to cap=4 concurrently.
            assert sched._inflight_for_test("dp") == 2
            async with sched.acquire(priority=5, engine_key="dp", cap=cap):
                assert sched._inflight_for_test("dp") == 3
                async with sched.acquire(priority=5, engine_key="dp", cap=cap):
                    assert sched._inflight_for_test("dp") == cap
                    # A DIFFERENT engine is on its own gate: the dp engine's
                    # four in-flight do not consume this engine's slots, and
                    # with no override it is still on the global default (2),
                    # so one is admitted fine and its count is independent.
                    async with sched.acquire(priority=5, engine_key="other"):
                        assert sched._inflight_for_test("other") == 1
                        assert sched._inflight_for_test("dp") == cap


async def test_acquire_without_cap_stays_on_the_global_default():
    """A caller that does not pass a cap (the test fakes, the two-model routing
    test) is unaffected: the engine is still gated by the scheduler's own
    max_inflight. This is what keeps the existing single-slot tests valid."""
    sched = PriorityScheduler(max_inflight=2)
    sched._manual_release = True
    async with sched.acquire(priority=5, engine_key="e"):
        async with sched.acquire(priority=5, engine_key="e"):
            assert sched._inflight_for_test("e") == 2


# ---------------------------------------------------------------------------
# The proxy wiring: _forward passes the cap computed from the row's flags
# ---------------------------------------------------------------------------


class _CapSpyScheduler:
    """A stand-in admission gate that records what the proxy asked for.

    ``routes._forward`` acquires the gate exactly once per request; recording
    the ``cap`` it was given is the whole assertion -- that the proxy computed
    the per-model cap from the row's ``extra_args`` and handed it to the gate.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    @asynccontextmanager
    async def acquire(self, *, priority: int = 0, engine_key: str = "", cap: int | None = None):
        self.calls.append((engine_key, cap))
        yield


def _seed_dp_model(db_path: Path) -> str:
    """Setup done, one 'loaded' model whose extra_args carry the DP flags, one
    inference token. Mirrors the seed in test_queue_wait_recorded.py."""
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,'auto',4096,0.9,0,?, 'loaded',0,NULL,NULL)",
            (json.dumps([0]), json.dumps(DP_EXTRA_ARGS)),
        )
        plaintext = "vw_validtoken1234567890abcdef12345"
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
        return plaintext


def _fake_tokenizer():
    cache = MagicMock()
    cache.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: 5)
    return cache


def _post(client, plaintext):
    body = {
        "id": "x",
        "model": "qwen",
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


def test_proxy_passes_the_admission_cap_from_the_model_flags(tmp_data_dir, client, monkeypatch):
    """The proxy computes the per-model cap from the row's ``extra_args`` and
    hands it to the admission gate. With the 7x64 DP flags and no global
    override, the gate is asked for 448, not the hardcoded 16."""
    monkeypatch.delenv("VW_PROXY_MAX_INFLIGHT", raising=False)
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_dp_model(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()
    spy = _CapSpyScheduler()
    client.app.state.scheduler = spy

    assert _post(client, plaintext).status_code == 200
    assert spy.calls == [("qwen", 448)]


def test_proxy_passes_the_global_override_when_set(tmp_data_dir, client, monkeypatch):
    """When VW_PROXY_MAX_INFLIGHT is set, the proxy passes the override for
    every model (the cap function returns it before looking at the flags)."""
    monkeypatch.setenv("VW_PROXY_MAX_INFLIGHT", "32")
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_dp_model(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()
    spy = _CapSpyScheduler()
    client.app.state.scheduler = spy

    assert _post(client, plaintext).status_code == 200
    assert spy.calls == [("qwen", 32)]


# ---------------------------------------------------------------------------
# #286: the rank count comes from the row's data_parallel_size column
# ---------------------------------------------------------------------------


def test_admission_cap_uses_the_data_parallel_column(monkeypatch):
    monkeypatch.delenv("VW_PROXY_MAX_INFLIGHT", raising=False)
    assert model_admission_cap(["--max-num-seqs", "32"], data_parallel_size=7) == 224


def test_admission_cap_takes_the_larger_of_column_and_flag(monkeypatch):
    monkeypatch.delenv("VW_PROXY_MAX_INFLIGHT", raising=False)
    # Column beats a stale smaller flag...
    assert (
        model_admission_cap(
            ["--max-num-seqs", "32", "--data-parallel-size", "2"], data_parallel_size=7
        )
        == 224
    )
    # ...and the flag beats a column that was never promoted.
    assert (
        model_admission_cap(
            ["--max-num-seqs", "32", "--data-parallel-size", "7"], data_parallel_size=1
        )
        == 224
    )
