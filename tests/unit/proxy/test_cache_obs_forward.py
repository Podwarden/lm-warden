# tests/unit/proxy/test_cache_obs_forward.py
"""Cache observation through the real forward path (spec §2-§3)."""

import json

import app.proxy.routes as routes
from tests.unit.proxy.test_cache_estimate_forward import _live_rows, _spy_registry
from tests.unit.proxy.test_dp_routing_forward import OK_BODY, _post, _ready, _resp, _sse_resp

SYS = "You are a helpful assistant. " * 120  # ~3.5 KB shared prefix


def _body(user, q):
    return {
        "model": "qwen",
        "user": user,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
    }


def _usage(prompt, cached):
    b = json.loads(json.dumps(OK_BODY))
    b["usage"]["prompt_tokens"] = prompt
    if cached is not None:
        b["usage"]["prompt_tokens_details"] = {"cached_tokens": cached}
    return b


def _records(client):
    rows = _live_rows(client)
    return [r.cache_obs for r in rows]


def test_first_request_is_cold_second_is_hit(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    _post(client, _body("alice-user-1", "q1"), resp=_resp(_usage(1000, 0)))
    _post(client, _body("alice-user-1", "q2"), resp=_resp(_usage(1000, 950)))
    first, second = _records(client)
    assert first["cache_outcome"] == "cold"
    assert second["reusable_tokens"] > 0 and second["cache_outcome"] == "hit"
    assert second["cached_source"] == "engine"


def test_lost_when_engine_reports_no_reuse(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    _post(client, _body("alice-user-1", "q1"), resp=_resp(_usage(1000, 0)))
    _post(client, _body("alice-user-1", "q2"), resp=_resp(_usage(1000, 0)))
    assert _records(client)[1]["cache_outcome"] == "lost"


def test_own_lens_does_not_see_other_tokens(tmp_data_dir, client):
    # same token in this harness, so assert the own scope is populated and equal
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    _post(client, _body("alice-user-1", "q1"), resp=_resp(_usage(1000, 0)))
    _post(client, _body("bob-user-22", "q2"), resp=_resp(_usage(1000, 950)))
    rec = _records(client)[1]
    assert rec["reusable_tokens_own"] == rec["reusable_tokens"]  # one token: own == all


def test_unreported_cached_is_null_not_zero(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    _post(client, _body("alice-user-1", "q1"), resp=_resp(_usage(1000, None)))
    rec = _records(client)[0]
    assert rec["cached_source"] is None and rec["cache_outcome"] is None


def test_none_backend_estimates(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    monkeypatch.setattr(routes, "_cached_reporting", lambda model: "none")
    monkeypatch.setattr(client.app.state.prefill_model, "cold_rate_for", lambda mid: 1e9)
    _post(client, _body("alice-user-1", "q1"), resp=_resp(_usage(1000, None)))
    rec = _records(client)[0]
    # non-stream: no ttft, so no estimate -- and still not 0
    assert rec["cached_ttft_est_tokens"] is None and rec["cached_source"] is None


def test_index_failure_is_fail_open(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(client.app.state.cache_index, "match", boom)
    r, _ = _post(client, _body("alice-user-1", "q1"), resp=_resp(_usage(1000, 0)))
    assert r.status_code == 200
    rec = _records(client)[0]
    assert rec is None or rec["cache_outcome"] is None


def test_stream_none_backend_estimates_from_ttft(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    monkeypatch.setattr(routes, "_cached_reporting", lambda model: "none")
    monkeypatch.setattr(client.app.state.prefill_model, "cold_rate_for", lambda mid: 2000.0)

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
    body = _body("alice-user-1", "q1")
    body["stream"] = True
    r, _ = _post(client, body, resp=_sse_resp(chunks))
    assert r.status_code == 200
    rec = _records(client)[0]
    assert rec["cached_source"] == "estimated"
    assert rec["cached_ttft_est_tokens"] is not None


def test_unrouted_dp_requests_feed_the_index_under_rank_none(tmp_data_dir, client):
    # #289: dp=2 with affinity off sends everything unrouted. Those requests
    # are observed under rank None, so a repeat reads hit and a miss reads lost.
    _ready(client, tmp_data_dir, dp=2, affinity=0)
    _spy_registry(client)
    _post(client, _body("alice-user-1", "q1"), resp=_resp(_usage(1000, 0)))
    assert len(client.app.state.cache_index) > 0
    _post(client, _body("alice-user-1", "q2"), resp=_resp(_usage(1000, 950)))
    _post(client, _body("alice-user-1", "q3"), resp=_resp(_usage(1000, 0)))
    first, repeat, miss = _records(client)
    assert first["cache_outcome"] == "cold"
    assert repeat["reusable_tokens"] > 0 and repeat["cache_outcome"] == "hit"
    assert miss["cache_outcome"] == "lost"


def test_unrouted_observation_never_makes_a_routed_miss_misrouted(tmp_data_dir, client):
    # The unrouted request could have landed on either rank: its rank-None
    # entry must not read as "on another replica" for a routed request.
    import sqlite3

    _ready(client, tmp_data_dir, dp=2, affinity=0)
    _spy_registry(client)
    _post(client, _body("alice-user-1", "q1"), resp=_resp(_usage(1000, 0)))
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET dp_affinity_enabled = 1 WHERE id='qwen'")
    # A raw write bypasses ModelRepo, so the model cache (#293) must be told.
    from app.proxy import model_cache

    model_cache.invalidate_all()
    _post(client, _body("alice-user-1", "q2"), resp=_resp(_usage(1000, 0)))
    routed = _records(client)[1]
    assert routed["reusable_tokens_fleet"] == 0
    assert routed["cache_outcome"] == "cold"


def test_usage_backend_without_cached_tokens_logs_once_per_minute(
    tmp_data_dir, client, caplog, monkeypatch
):
    import logging

    monkeypatch.setattr(routes, "_NO_CACHED_LOGGED", {})
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    with caplog.at_level(logging.DEBUG, logger="app.proxy.routes"):
        _post(client, _body("alice-user-1", "q1"), resp=_resp(_usage(1000, None)))
        _post(client, _body("alice-user-1", "q2"), resp=_resp(_usage(1000, None)))
        _post(client, _body("alice-user-1", "q3"), resp=_resp(_usage(1000, 0)))
    hits = [r for r in caplog.records if "no cached_tokens" in r.getMessage()]
    assert len(hits) == 1 and "qwen" in hits[0].getMessage()


def _seed_block_size(client, tmp_data_dir, block):
    """Seed B exactly where the background learner puts it: app.state's
    BlockSizes, keyed by the engine run the proxy is serving."""
    from app.db.database import open_db
    from app.db.repos.models import ModelRepo
    from app.proxy.routes_dp import engine_epoch

    async def go():
        async with open_db(str(tmp_data_dir / "vllm-warden.db")) as db:
            return await ModelRepo(db).get("qwen")

    model = client.portal.call(go)
    client.app.state.engine_block_sizes.set("qwen", engine_epoch(client.app.state, model), block)


def test_block_size_makes_a_whole_block_hit_read_hit(tmp_data_dir, client):
    # #301: one 784-token block cached of a ~1,000-token potential is all a
    # 784-block engine can serve. Block-unaware it reads partial.
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    _post(client, _body("alice-user-1", "q1"), resp=_resp(_usage(1000, 0)))
    _post(client, _body("alice-user-1", "q2"), resp=_resp(_usage(1000, 784)))
    _seed_block_size(client, tmp_data_dir, 784)
    _post(client, _body("alice-user-1", "q3"), resp=_resp(_usage(1000, 784)))
    _post(client, _body("alice-user-1", "q4"), resp=_resp(_usage(500, 0)))
    _, unaware, aware, short = _records(client)
    assert unaware["reusable_tokens"] > 871  # 784 < 0.9 * R
    assert unaware["cache_outcome"] == "partial"
    assert aware["cache_outcome"] == "hit" and aware["cache_outcome_own"] == "hit"
    # under one block: nothing reusable at this engine's granularity
    assert short["reusable_tokens"] >= 256 and short["cache_outcome"] == "cold"


def test_block_size_from_another_engine_run_is_ignored(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    client.app.state.engine_block_sizes.set("qwen", "some-old-run:0", 784)
    _post(client, _body("alice-user-1", "q1"), resp=_resp(_usage(1000, 0)))
    _post(client, _body("alice-user-1", "q2"), resp=_resp(_usage(1000, 784)))
    assert _records(client)[1]["cache_outcome"] == "partial"


# --- #300: cache_salt ---------------------------------------------------------


def _salted(user, q, salt):
    return {**_body(user, q), "cache_salt": salt}


def test_different_salt_reads_cold_not_lost(tmp_data_dir, client):
    # vLLM never shares a block across salts, so C = 0 there is no loss
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    _post(client, _salted("alice-user-1", "q1", "salt-a"), resp=_resp(_usage(1000, 0)))
    _post(client, _salted("alice-user-1", "q2", "salt-b"), resp=_resp(_usage(1000, 0)))
    _post(client, _body("alice-user-1", "q3"), resp=_resp(_usage(1000, 0)))
    _, other_salt, unsalted = _records(client)
    for rec in (other_salt, unsalted):
        assert rec["reusable_tokens"] == 0 and rec["cache_outcome"] == "cold"


def test_same_salt_reads_hit(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    _post(client, _salted("alice-user-1", "q1", "salt-a"), resp=_resp(_usage(1000, 0)))
    _post(client, _salted("alice-user-1", "q2", "salt-a"), resp=_resp(_usage(1000, 950)))
    rec = _records(client)[1]
    assert rec["reusable_tokens"] > 0 and rec["cache_outcome"] == "hit"


def test_salt_is_ignored_for_a_backend_that_ignores_it(tmp_data_dir, client, monkeypatch):
    # llama.cpp reuses a slot's prefix whatever the salt: C = 0 is a real loss
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    monkeypatch.setattr(routes, "_honours_cache_salt", lambda model: False)
    _post(client, _salted("alice-user-1", "q1", "salt-a"), resp=_resp(_usage(1000, 0)))
    _post(client, _salted("alice-user-1", "q2", "salt-b"), resp=_resp(_usage(1000, 0)))
    rec = _records(client)[1]
    assert rec["reusable_tokens"] > 0 and rec["cache_outcome"] == "lost"


def test_honours_cache_salt_follows_the_backend():
    from types import SimpleNamespace

    assert routes._honours_cache_salt(SimpleNamespace(backend="vllm")) is True
    assert routes._honours_cache_salt(SimpleNamespace(backend="llamacpp")) is False
