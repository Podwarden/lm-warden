"""Session id, prefix-memory estimate and measured cached tokens (in-flight table)."""

import json

import pytest

from app.proxy.dp_affinity import PrefixMemory, session_id_of
from app.proxy.request_registry import LiveRequest, finished_record
from app.proxy.routes import _cached_tokens_of, _parse_sse_frame
from app.stats.live_requests import _serialize

SID = "0b1c2d3e-1111-4222-8333-444455556666"
ACCT = "aaaaaaaa-1111-4222-8333-999999999999"


def test_header_wins_over_everything():
    uid = json.dumps({"session_id": "from-user-id"})
    got = session_id_of(
        {"user": "body-user-1"},
        {"x-claude-code-session-id": "hdr-0001", "X-Session-Id": "hdr-0002"},
        uid,
    )
    assert got == "hdr-0001"


def test_json_user_id():
    uid = json.dumps({"device_id": "d", "account_uuid": ACCT, "session_id": SID})
    assert session_id_of({}, {}, uid) == SID


def test_underscore_user_id_never_leaks_account():
    uid = f"user_deadbeef_account_{ACCT}_session_{SID}"
    got = session_id_of({}, {}, uid)
    assert got == SID
    assert ACCT not in got


def test_unparseable_user_id_is_not_shown():
    uid = f"user_deadbeef_account_{ACCT}"
    assert session_id_of({}, {}, uid) is None
    assert session_id_of({}, {}, "{not json") is None


def test_x_session_id_then_body_user():
    assert session_id_of({"user": "user-0001"}, {"X-Session-Id": "sess-0009"}, None) == "sess-0009"
    assert session_id_of({"user": "user-0001"}, {}, None) == "user-0001"


@pytest.mark.parametrize("user", ["has space", "x" * 129, "", 5, None, ["a"]])
def test_garbage_body_user_is_none(user):
    assert session_id_of({"user": user}, {}, None) is None


def test_every_candidate_is_reduced_to_the_session_part():
    uid = f"user_deadbeef_account_{ACCT}_session_{SID}"
    assert session_id_of({}, {"X-Session-Id": uid}, None) == SID
    assert session_id_of({"user": uid}, {}, None) == SID
    assert (
        session_id_of({"user": json.dumps({"session_id": SID, "account_uuid": ACCT})}, {}, None)
        == SID
    )
    # an account-bearing value with no session part shows nothing at all
    assert session_id_of({"user": f"user_x_account_{ACCT}"}, {}, None) is None
    assert session_id_of({}, {"X-Claude-Code-Session-Id": f"user_x_account_{ACCT}"}, None) is None


@pytest.mark.parametrize("user", ["a@b.com", "someone@example.org"])
def test_email_user_is_not_a_session(user):
    assert session_id_of({"user": user}, {}, None) is None


def test_no_sources_and_bad_body():
    assert session_id_of({"messages": [{"role": "user", "content": "hi"}]}, {}, None) is None
    assert session_id_of("junk", {}, None) is None


class _Clock:
    t = 100.0

    def __call__(self) -> float:
        return self.t


def test_prefix_memory_hit_same_rank_miss_other():
    m = PrefixMemory()
    m.remember("m", "k", 2, 5000)
    assert m.estimate("m", "k", 2, 9000) == 5000
    assert m.estimate("m", "k", 3, 9000) is None
    assert m.estimate("m", "other", 2, 9000) is None
    assert m.estimate("m2", "k", 2, 9000) is None


def test_prefix_memory_remembers_the_prompt_and_drops_diverged():
    m = PrefixMemory()
    m.remember("m", "k", None, 5000)
    # min(prev_prompt, prompt) -- a longer prompt reuses at most the old prompt
    assert m.estimate("m", "k", None, 9000) == 5000
    # a shorter prompt means compaction / divergence: no basis
    assert m.estimate("m", "k", None, 3000) is None
    assert m.estimate("m", "k", None, 5000) == 5000


def test_gguf_units_estimate_in_engine_tokens_not_none():
    # warden counts characters (~30% low); the engine reported more
    m = PrefixMemory()
    m.remember("m", "k", None, 12304, 17705)
    est = m.estimate("m", "k", None, 12312)
    assert est == 17705  # min(engine prev, scaled new) -- not None
    row = _serialize(_req(prompt_tokens=12312, cache_est_tokens=est), 2.0)
    assert row["cache_est_pct"] == 1.0  # clamped, engine units over warden prompt


def test_divergence_compares_warden_to_warden():
    m = PrefixMemory()
    m.remember("m", "k", None, 12304, 17705)
    assert m.estimate("m", "k", None, 12303) is None  # genuinely shorter
    m.remember("m", "z", None, 0, 0)
    assert m.estimate("m", "z", None, 10) is None  # no basis, no divide by zero


def test_scaled_estimate_when_the_prompt_grew():
    m = PrefixMemory()
    m.remember("m", "k", None, 1000, 1500)
    assert m.estimate("m", "k", None, 1000) == 1500
    assert m.estimate("m", "k", None, 2000) == 1500  # capped at what was cached


def test_equal_units_unchanged():
    m = PrefixMemory()
    m.remember("m", "k", None, 5000, 5000)
    assert m.estimate("m", "k", None, 9000) == 5000
    assert m.estimate("m", "k", None, 3000) is None


def test_prefix_memory_epoch_isolates_engine_runs():
    m = PrefixMemory()
    m.remember("m", "k", 0, 100, epoch="v1:1")
    assert m.estimate("m", "k", 0, 100, epoch="v1:1") == 100
    assert m.estimate("m", "k", 0, 100, epoch="v1:2") is None
    assert m.estimate("m", "k", 0, 100, epoch="v2:1") is None


def test_prefix_memory_never_stores_raw_keys():
    m = PrefixMemory()
    m.remember("m", "user_secret_account_xyz", 0, 10)
    assert all("secret" not in repr(k) for k in m._d)


def test_prefix_memory_ttl():
    c = _Clock()
    m = PrefixMemory(ttl_s=900, clock=c)
    m.remember("m", "k", 0, 10)
    c.t += 899
    assert m.estimate("m", "k", 0, 10) == 10
    c.t += 902
    assert m.estimate("m", "k", 0, 10) is None
    assert len(m) == 0


def test_prefix_memory_lru_bound():
    m = PrefixMemory(max_entries=3)
    for i in range(3):
        m.remember("m", f"k{i}", 0, 10)
    m.estimate("m", "k0", 0, 10)  # touch: k1 is now the oldest
    m.remember("m", "k3", 0, 10)
    assert len(m) == 3
    assert m.estimate("m", "k1", 0, 10) is None
    assert m.estimate("m", "k0", 0, 10) == 10


def _req(**kw):
    base = dict(
        id="r1",
        token_id="t",
        token_name="dev",
        client_ip="1.2.3.4",
        model="m",
        model_row_id="mid",
        path="/v1/chat/completions",
        prompt_tokens=1000,
        max_model_len=4000,
        started_monotonic=1.0,
        started_iso="x",
    )
    base.update(kw)
    return LiveRequest(**base)


def test_live_serialization_new_fields():
    row = _serialize(_req(session_id=SID, cache_est_tokens=750), 2.0)
    assert row["session_id"] == SID
    assert row["cache_est_tokens"] == 750
    assert row["cache_est_pct"] == 0.75


def test_live_serialization_defaults():
    row = _serialize(_req(prompt_tokens=0), 2.0)
    assert row["session_id"] is None
    assert row["cache_est_tokens"] is None
    assert row["cache_est_pct"] is None
    assert _serialize(_req(prompt_tokens=0, cache_est_tokens=5), 2.0)["cache_est_pct"] is None


def test_finished_record_carries_cache_fields():
    rec = finished_record(_req(cached_tokens=640, cache_est_tokens=700), now=3.0)
    assert rec["cached_tokens"] == 640
    assert rec["cache_est_tokens"] == 700
    rec = finished_record(_req(), now=3.0)
    assert rec["cached_tokens"] is None


def test_cached_tokens_sources_and_precedence():
    assert _cached_tokens_of({"usage": {"prompt_tokens_details": {"cached_tokens": 7}}}) == 7
    assert _cached_tokens_of({"timings": {"cache_n": 11}}) == 11
    assert _cached_tokens_of({"tokens_cached": 13}) == 13
    full = {
        "usage": {"prompt_tokens_details": {"cached_tokens": 1}},
        "timings": {"cache_n": 2},
        "tokens_cached": 3,
    }
    assert _cached_tokens_of(full) == 1
    del full["usage"]
    assert _cached_tokens_of(full) == 2
    del full["timings"]
    assert _cached_tokens_of(full) == 3


@pytest.mark.parametrize(
    "obj",
    [
        {},
        None,
        "x",
        {"usage": None},
        {"usage": {"prompt_tokens_details": None}},
        {"usage": {"prompt_tokens_details": {"cached_tokens": None}}},
        {"usage": {"prompt_tokens_details": {"cached_tokens": True}}},
        {"timings": {"cache_n": -1}},
        {"tokens_cached": "3"},
    ],
)
def test_cached_tokens_missing_is_none_not_zero(obj):
    assert _cached_tokens_of(obj) is None


def test_zero_is_a_real_measurement():
    assert _cached_tokens_of({"usage": {"prompt_tokens_details": {"cached_tokens": 0}}}) == 0


def test_cached_tokens_from_final_sse_frame():
    final = {
        "choices": [],
        "usage": {"completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 321}},
    }
    assert _parse_sse_frame(b"data: " + json.dumps(final).encode()) == (5, "", 321, None, None)
    tok = {"choices": [{"delta": {"content": "hi"}}], "usage": None}
    assert _parse_sse_frame(b"data: " + json.dumps(tok).encode())[2] is None
    assert _parse_sse_frame(b"data: [DONE]") == (None, "", None, None, None)
    assert _parse_sse_frame(b'data: {"timings": {"cache_n": 9}}')[2] == 9
    assert _parse_sse_frame(b"data: {broken cached_tokens")[2] is None
    assert _parse_sse_frame(b'{"timings": {"cache_n": 9}}')[2] is None  # no data: prefix
