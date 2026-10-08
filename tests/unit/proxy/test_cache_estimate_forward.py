"""The cache estimate and measured cached tokens through the real forward path."""

import json

import httpx

from app.proxy.dp_affinity import PrefixMemory
from tests.unit.proxy.ledger_helpers import flush_ledger
from tests.unit.proxy.test_dp_routing_forward import (
    OK_BODY,
    _post,
    _ready,
    _resp,
    _sse_resp,
)

UID = "user_ab12_account_aaaaaaaa-1111-4222-8333-999999999999_session_0b1c2d3e-1111-4222-8333-444455556666"


def _body(n_words=1):
    return {
        "model": "qwen",
        "user": "alice-user-1",
        "messages": [{"role": "user", "content": "hi " * n_words}],
    }


def _live_rows(client):
    # the registry forgets a request when it ends, so spy on registration
    return client.app.state._seen_live


def _spy_registry(client):
    reg = client.app.state.request_registry
    seen = []
    orig = reg.register

    async def register(req):
        seen.append(req)
        await orig(req)

    reg.register = register
    client.app.state._seen_live = seen


def test_estimate_uses_routed_rank_and_prompt(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    _post(client, _body(5))
    _post(client, _body(5))
    first, second = _live_rows(client)
    assert first.cache_est_tokens is None
    assert second.cache_est_tokens == first.prompt_tokens > 0


def test_unrouted_dp_request_neither_estimates_nor_remembers(tmp_data_dir, client):
    _ready(client, tmp_data_dir, affinity=0)
    _spy_registry(client)
    _post(client, _body(5))
    _post(client, _body(5))
    assert [r.cache_est_tokens for r in _live_rows(client)] == [None, None]
    assert len(client.app.state.prefix_memory) == 0


def test_prompt_hash_key_is_not_remembered(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    body = {"model": "qwen", "messages": [{"role": "user", "content": "hello there"}]}
    _post(client, body)
    assert len(client.app.state.prefix_memory) == 0


def test_failed_requests_keep_the_previous_entry(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _post(client, _body(5))
    mem = client.app.state.prefix_memory
    assert len(mem) == 1
    before = dict(mem._d)
    # refused connection, upstream 4xx, upstream 5xx: nothing was processed
    _post(client, _body(50), resp=httpx.ConnectError("refused"))
    for status in (400, 503):
        bad = _resp({"error": {"message": "x"}})
        bad.status_code = status
        _post(client, _body(50), resp=bad)
    assert mem._d == before


def test_stream_that_never_started_is_not_remembered(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    body = _body(5)
    body["stream"] = True
    _post(client, body, resp=_sse_resp([]))
    assert len(client.app.state.prefix_memory) == 0


def test_a_raising_prefix_memory_does_not_fail_the_request(tmp_data_dir, client):
    _ready(client, tmp_data_dir)

    class Boom(PrefixMemory):
        def estimate_with_gap(self, *a, **k):
            raise RuntimeError("boom")

        def remember(self, *a, **k):
            raise RuntimeError("boom")

    client.app.state.prefix_memory = Boom()
    r, _ = _post(client, _body(5))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok"


def test_engine_restart_invalidates_the_memory(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    _post(client, _body(5))
    client.app.state.supervisor._generation["qwen"] = 7
    _post(client, _body(5))
    assert _live_rows(client)[1].cache_est_tokens is None


def test_cached_tokens_measured_non_stream(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    body = json.loads(json.dumps(OK_BODY))
    body["usage"]["prompt_tokens"] = 100
    body["usage"]["prompt_tokens_details"] = {"cached_tokens": 42}
    _post(client, _body(), resp=_resp(body))
    assert _live_rows(client)[0].cached_tokens == 42


def test_cached_tokens_measured_stream(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)

    def frame(ev):
        return b"data: " + json.dumps(ev).encode() + b"\n\n"

    chunks = [
        frame({"choices": [{"index": 0, "delta": {"content": "a"}}]}),
        frame(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 7},
                },
            }
        ),
        b"data: [DONE]\n\n",
    ]
    body = _body()
    body["stream"] = True
    r, _ = _post(client, body, resp=_sse_resp(chunks))
    assert r.status_code == 200
    assert _live_rows(client)[0].cached_tokens == 7
    # the stream was processed, so the conversation is remembered
    assert len(client.app.state.prefix_memory) == 1


# ---- the engine's prompt count wins for accounting / display ----------------


def _frame(ev):
    return b"data: " + json.dumps(ev).encode() + b"\n\n"


def _stream_body():
    b = _body(5)
    b["stream"] = True
    return b


def _history(client):
    return client.app.state.request_history


def _spy_history(client):
    recs = []
    client.app.state.request_history = type(
        "H", (), {"record": lambda self, rec, **k: recs.append(rec) or True}
    )()
    return recs


def _counters(client, tmp_data_dir):
    import sqlite3

    flush_ledger(client)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        return db.execute(
            "SELECT prompt_tokens, cached_tokens, cached_measured_requests FROM model_samples"
        ).fetchall()


def test_llamacpp_final_frame_engine_prompt_wins(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    recs = _spy_history(client)
    chunks = [
        _frame({"choices": [{"index": 0, "delta": {"content": "a"}}]}),
        _frame(
            {
                "choices": [],
                "usage": {"prompt_tokens": 17_000, "completion_tokens": 3},
                "timings": {"cache_n": 17_726},
            }
        ),
        b"data: [DONE]\n\n",
    ]
    _post(client, _stream_body(), resp=_sse_resp(chunks))
    live = _live_rows(client)[0]
    assert live.prompt_tokens == 17_000
    assert recs[0]["prompt_tokens"] == 17_000
    # engine-reported both ways, so cached is clamped to the prompt
    assert recs[0]["cached_tokens"] == 17_000
    assert _counters(client, tmp_data_dir) == [(17_000, 17_000, 1)]
    # the next estimate is in engine tokens
    mem = client.app.state.prefix_memory
    assert next(iter(mem._d.values()))[2] == 17_000


def test_continuous_usage_chunk_corrects_live_prompt_mid_stream(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    seen = []

    async def aiter():
        yield _frame(
            {
                "choices": [{"index": 0, "delta": {"content": "a"}}],
                "usage": {"prompt_tokens": 4321, "completion_tokens": 1},
            }
        )
        seen.append(_live_rows(client)[0].prompt_tokens)
        yield _frame({"choices": [{"index": 0, "delta": {"content": "b"}}]})
        yield b"data: [DONE]\n\n"

    resp = _sse_resp([])
    resp.aiter_bytes = aiter
    _post(client, _stream_body(), resp=resp)
    assert seen == [4321]


def test_non_stream_body_engine_prompt_wins(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    recs = _spy_history(client)
    body = json.loads(json.dumps(OK_BODY))
    body["usage"] = {
        "prompt_tokens": 900,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 5000},
    }
    _post(client, _body(5), resp=_resp(body))
    assert recs[0]["prompt_tokens"] == 900
    assert recs[0]["cached_tokens"] == 900
    assert _counters(client, tmp_data_dir) == [(900, 900, 1)]


def test_engine_without_usage_keeps_the_warden_count(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    recs = _spy_history(client)
    body = {k: v for k, v in OK_BODY.items() if k != "usage"}
    _post(client, _body(5), resp=_resp(body))
    live = _live_rows(client)[0]
    assert live.prompt_tokens > 0
    assert recs[0]["prompt_tokens"] == live.prompt_tokens
    assert recs[0]["cached_tokens"] is None


def test_gguf_second_turn_keeps_an_engine_unit_estimate(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    chunks = [
        _frame({"choices": [{"index": 0, "delta": {"content": "a"}}]}),
        _frame({"choices": [], "usage": {"prompt_tokens": 9000, "completion_tokens": 1}}),
        b"data: [DONE]\n\n",
    ]
    _post(client, _stream_body(), resp=_sse_resp(chunks))
    _post(client, _stream_body(), resp=_sse_resp(chunks))
    first, second = _live_rows(client)
    assert first.prompt_tokens == 9000 and second.cache_est_tokens is not None
    assert second.cache_est_tokens > 0


ERR = b'data: {"error":{"code":500,"message":"Context size has been exceeded.","type":"server_error"}}\n\n'


def test_error_only_stream_is_not_remembered_nor_decode(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    recs = _spy_history(client)
    ok = [
        _frame({"choices": [{"index": 0, "delta": {"content": "a"}}]}),
        _frame({"choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 1}}),
        b"data: [DONE]\n\n",
    ]
    _post(client, _stream_body(), resp=_sse_resp(ok))
    mem = client.app.state.prefix_memory
    before = dict(mem._d)
    assert len(before) == 1
    _post(client, _stream_body(), resp=_sse_resp([ERR]))
    assert mem._d == before  # previous entry kept
    live = _live_rows(client)[1]
    assert live.phase == "prefill"
    assert live.first_token_monotonic is None
    assert recs[1]["finish_reason"] == "error"
    assert recs[1]["ttft_s"] is None


def test_error_after_content_is_not_remembered(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    chunks = [_frame({"choices": [{"index": 0, "delta": {"content": "a"}}]}), ERR]
    _post(client, _stream_body(), resp=_sse_resp(chunks))
    assert len(client.app.state.prefix_memory) == 0


def test_normal_stream_still_remembered_with_ttft(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    recs = _spy_history(client)
    chunks = [
        _frame({"choices": [{"index": 0, "delta": {"content": "a"}}]}),
        b"data: [DONE]\n\n",
    ]
    _post(client, _stream_body(), resp=_sse_resp(chunks))
    assert len(client.app.state.prefix_memory) == 1
    assert recs[0]["ttft_s"] is not None
    assert recs[0]["finish_reason"] != "error"


def test_reasoning_only_stream_counts_as_served(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    chunks = [_frame({"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]})]
    _post(client, _stream_body(), resp=_sse_resp(chunks))
    assert len(client.app.state.prefix_memory) == 1


# ---- lifecycle phase --------------------------------------------------------


def _phases_during_stream(client, deltas):
    """Run a streamed request; return the phase seen after each frame."""
    seen = []

    async def aiter():
        for ev in deltas:
            yield _frame(ev)
            seen.append(client.app.state.request_registry.snapshot()[0].phase)
        yield b"data: [DONE]\n\n"

    resp = _sse_resp([])
    resp.aiter_bytes = aiter
    _post(client, _stream_body(), resp=resp)
    return seen


def _d(**delta):
    return {"choices": [{"index": 0, "delta": delta}]}


def test_phase_follows_the_latest_delta_kind(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    seen = _phases_during_stream(
        client,
        [
            _d(role="assistant"),  # role-only: still prefill
            _d(reasoning_content="hm"),
            _d(reasoning="more"),  # newer builds' channel name
            _d(content="Hello"),
            _d(tool_calls=[{"index": 0, "function": {"name": "f", "arguments": "{}"}}]),
            _d(content="done"),
        ],
    )
    assert seen == ["prefill", "thinking", "thinking", "answering", "tool_call", "answering"]


def test_error_frame_leaves_phase_in_prefill(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    seen = _phases_during_stream(client, [{"error": {"code": 500, "message": "x"}}])
    assert seen == ["prefill"]


def test_request_is_visible_as_queued_during_the_admission_wait(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    sched = client.app.state.scheduler
    real = sched.acquire
    seen = []

    def acquire(**kw):
        cm = real(**kw)

        class Gate:
            async def __aenter__(self_inner):
                seen.append(
                    [(r.phase, r.queued_s) for r in client.app.state.request_registry.snapshot()]
                )
                return await cm.__aenter__()

            async def __aexit__(self_inner, *a):
                return await cm.__aexit__(*a)

        return Gate()

    sched.acquire = acquire
    _spy_registry(client)
    r, _ = _post(client, _body(5))
    assert r.status_code == 200
    assert seen == [[("queued", None)]]
    row = _live_rows(client)[0]  # same object, now admitted
    assert row.phase == "waiting" and row.queued_s is not None  # non-stream request


def test_a_cancelled_admission_wait_leaves_no_row(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    sched = client.app.state.scheduler

    def acquire(**kw):
        class Gate:
            async def __aenter__(self):
                raise RuntimeError("refused")

            async def __aexit__(self, *a):
                return None

        return Gate()

    sched.acquire = acquire
    try:
        _post(client, _body(5))
    except Exception:
        pass
    assert client.app.state.request_registry.snapshot() == []


def test_registry_failure_stays_fail_open(tmp_data_dir, client):
    _ready(client, tmp_data_dir)

    async def boom(_req):
        raise RuntimeError("registry down")

    client.app.state.request_registry.register = boom
    r, _ = _post(client, _body(5))
    assert r.status_code == 200


# ---- prefill_slow -----------------------------------------------------------


def _live(**kw):
    from app.proxy.request_registry import LiveRequest

    base = dict(
        id="r",
        token_id="t",
        token_name="n",
        client_ip="1.1.1.1",
        model="m",
        model_row_id="m",
        path="/v1/chat/completions",
        prompt_tokens=20_000,
        max_model_len=40_000,
        started_monotonic=0.0,
        started_iso="x",
    )
    base.update(kw)
    return LiveRequest(**base)


def test_prefill_slow_math():
    from app.stats.live_requests import _serialize

    # 20k fresh tokens at 2000 tok/s = 10 s expected; slow beyond 40 s
    assert _serialize(_live(), 39.0)["prefill_slow"] is False
    assert _serialize(_live(), 41.0)["prefill_slow"] is True
    # cached prefix shrinks the expectation: 1k fresh -> 0.5 s -> floor of 3 s
    est = _live(cache_est_tokens=19_000)
    assert _serialize(est, 2.9)["prefill_slow"] is False
    assert _serialize(est, 3.1)["prefill_slow"] is True
    # a configured rate changes the threshold
    assert _serialize(_live(), 39.0, 10_000.0)["prefill_slow"] is True
    # only prefill can be slow
    assert _serialize(_live(phase="answering"), 999.0)["prefill_slow"] is False
    assert _serialize(_live(phase="queued"), 999.0)["prefill_slow"] is False
    assert _serialize(_live(queued_s=1.234), 1.0)["queued_s"] == 1.234


# ---- non-stream "waiting", construction inside the cleanup try, started_iso ----


def test_non_stream_request_is_waiting_and_never_slow(tmp_data_dir, client):
    from app.stats.live_requests import _serialize

    _ready(client, tmp_data_dir)
    _spy_registry(client)
    r, _ = _post(client, _body(5))  # no "stream": true
    assert r.status_code == 200
    live = _live_rows(client)[0]
    assert live.phase == "waiting" and live.is_stream is False
    row = _serialize(live, live.started_monotonic + 9999)
    assert row["prefill_slow"] is False and row["is_stream"] is False


def test_stream_request_keeps_prefill_and_is_stream(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    seen = _phases_during_stream(client, [_d(content="x")])
    assert seen == ["answering"]
    assert _live_rows(client)[0].is_stream is True


def test_a_synchronous_raise_building_the_gate_leaks_no_queued_row(tmp_data_dir, client):
    _ready(client, tmp_data_dir)

    def acquire(**kw):
        raise RuntimeError("scheduler broke")

    client.app.state.scheduler.acquire = acquire
    try:
        _post(client, _body(5))
    except Exception:
        pass
    assert client.app.state.request_registry.snapshot() == []


def test_started_iso_is_taken_at_admission(tmp_data_dir, client):
    import asyncio as _asyncio

    _ready(client, tmp_data_dir)
    _spy_registry(client)
    recs = _spy_history(client)
    real = client.app.state.scheduler.acquire
    queued_iso = []

    def acquire(**kw):
        cm = real(**kw)

        class Gate:
            async def __aenter__(self_inner):
                queued_iso.append(client.app.state.request_registry.snapshot()[0].started_iso)
                await _asyncio.sleep(0.05)
                return await cm.__aenter__()

            async def __aexit__(self_inner, *a):
                return await cm.__aexit__(*a)

        return Gate()

    client.app.state.scheduler.acquire = acquire
    _post(client, _body(5))
    assert recs[0]["started_iso"] > queued_iso[0]
    assert recs[0]["queued_s"] >= 0.05


# ---- learning inputs recorded on the row --------------------------------------


def _ok_stream():
    return _sse_resp(
        [
            _frame({"choices": [{"index": 0, "delta": {"content": "a"}}]}),
            _frame({"choices": [], "usage": {"prompt_tokens": 9000, "completion_tokens": 1}}),
            b"data: [DONE]\n\n",
        ]
    )


def test_history_records_rank_gap_and_alone_at_start(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    recs = _spy_history(client)
    _post(client, _stream_body(), resp=_ok_stream())
    _post(client, _stream_body(), resp=_ok_stream())
    first, second = recs
    assert first["gap_s"] is None and second["gap_s"] is not None and second["gap_s"] >= 0
    assert first["dp_rank"] == second["dp_rank"] and 0 <= first["dp_rank"] < 7
    assert first["inflight_same_rank_at_start"] == 0


def test_contention_at_start_counts_admitted_rows_on_the_same_rank(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    recs = _spy_history(client)
    _post(client, _stream_body(), resp=_ok_stream())  # places "alice-user-1" on a replica
    rank = recs[0]["dp_rank"]
    recs.clear()
    reg = client.app.state.request_registry

    def fake(rid, **kw):
        base = dict(
            id=rid,
            token_id="t",
            token_name="n",
            client_ip="1",
            model="qwen",
            model_row_id="qwen",
            path="/v1/chat/completions",
            prompt_tokens=10,
            max_model_len=4096,
            started_monotonic=0.0,
            started_iso="x",
            dp_rank=rank,
        )
        base.update(kw)
        return _live_req(**base)

    for r in (
        fake("busy"),  # counts
        fake("other-rank", dp_rank=rank + 1),
        fake("still-queued", phase="queued"),
        fake("other-model", model_row_id="zzz"),
    ):
        client.portal.call(reg.register, r)
    _post(client, _stream_body(), resp=_ok_stream())
    assert recs[0]["inflight_same_rank_at_start"] == 1


def _live_req(**kw):
    from app.proxy.request_registry import LiveRequest

    return LiveRequest(**kw)


def test_learned_gap_decay_scales_the_estimate_and_history_keeps_the_raw_guess(
    tmp_data_dir, client
):
    from app.stats import prefill_model as pm

    _ready(client, tmp_data_dir)
    _spy_registry(client)
    recs = _spy_history(client)
    rows = [
        dict(model_id="qwen", gap_s=1.0, cache_est_tokens=1000, cached_tokens=250)
        for _ in range(12)
    ]  # the 0-30 s bucket is right a quarter of the time... measured >= .5 x est: no
    # a fresh object: the background refresher holds (and rewrites) the old one
    client.app.state.prefill_model = pm.PrefillModelState()
    client.app.state.prefill_model.replace(pm.build(rows))
    _post(client, _stream_body(), resp=_ok_stream())
    _post(client, _stream_body(), resp=_ok_stream())
    second = _live_rows(client)[1]
    assert second.est_decayed is True
    assert second.cache_est_raw_tokens and second.cache_est_tokens == 0
    assert recs[1]["cache_est_tokens"] == second.cache_est_raw_tokens


def test_endpoints_expose_the_learned_model_and_queue_fields(tmp_data_dir, client):
    from app.stats import prefill_model as pm
    from tests.conftest import jwt_login, seed_admin_user

    _ready(client, tmp_data_dir)
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    rows = [
        dict(
            model_id="qwen",
            prompt_tokens=10_000,
            cached_tokens=2_000,
            ttft_s=4.0,
            inflight_same_rank_at_start=0,
        )
        for _ in range(25)
    ]
    client.app.state.prefill_model = pm.PrefillModelState()
    client.app.state.prefill_model.replace(pm.build(rows))
    snap = client.get("/api/stats/v2/prefill-model", headers=auth).json()
    m = snap["models"]["qwen"]
    assert m["rate_tok_s"] == 2000.0 and m["rate_source"] == "learned" and m["samples"] == 25
    assert [b["label"] for b in m["gap_buckets"]] == ["0-30s", "30-120s", "2-5m", "5-15m"]
    live = client.get("/api/stats/requests", headers=auth).json()
    assert live["requests"] == []


# ---- session_source / parent on the row and in history -----------------------------


def test_session_source_and_parent_are_recorded(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    recs = _spy_history(client)
    body = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    _post(
        client,
        body,
        headers={
            "session-id": "0b1c2d3e-1111-4222-8333-444455556666",
            "x-codex-parent-thread-id": "thread-parent-01",
        },
    )
    live = _live_rows(client)[0]
    assert live.session_id == "0b1c2d3e-1111-4222-8333-444455556666"
    assert (live.session_source, live.parent_session_id) == (
        "session_id_header",
        "thread-parent-01",
    )
    assert recs[0]["session_source"] == "session_id_header"
    assert recs[0]["parent_session_id"] == "thread-parent-01"
    from app.stats.live_requests import _serialize

    row = _serialize(live, live.started_monotonic + 1)
    assert (
        row["session_source"] == "session_id_header"
        and row["parent_session_id"] == "thread-parent-01"
    )


def test_prompt_hash_is_a_source_but_not_a_session_id(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    body = {"model": "qwen", "messages": [{"role": "user", "content": "no session here"}]}
    _post(client, body)
    live = _live_rows(client)[0]
    assert live.session_id is None and live.session_source == "prompt_hash"


def test_no_signal_and_no_text_means_no_source(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    _post(client, {"model": "qwen", "messages": []})
    assert _live_rows(client)[0].session_source is None
