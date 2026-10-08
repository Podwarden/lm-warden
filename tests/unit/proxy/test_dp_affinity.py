"""Pure tests for cache-affine replica routing (#286 task 3)."""

import hashlib
import random

import pytest

from app.proxy.dp_affinity import (
    DpRoutingState,
    affinity_key,
    choose_rank,
    client_rank,
    home_rank,
)


def test_home_rank_is_blake2b_and_stable():
    want = int.from_bytes(hashlib.blake2b(b"k", digest_size=8).digest(), "big") % 7
    assert home_rank("k", 7) == want
    assert [home_rank("k", 7) for _ in range(5)] == [want] * 5


def test_home_rank_spreads_keys():
    rng = random.Random(1)
    counts = [0] * 7
    for _ in range(1000):
        counts[home_rank(f"key-{rng.random()}", 7)] += 1
    assert all(c > 50 for c in counts), counts


def test_choose_rank_sticks_below_threshold():
    home = home_rank("a", 4)
    rank, decision = choose_rank("a", 4, {home: 7}, 8)
    assert (rank, decision) == (home, "sticky")


def test_choose_rank_spills_to_least_loaded_with_low_index_ties():
    home = home_rank("a", 4)
    load = {r: 5 for r in range(4)}
    load[home] = 8
    other = [r for r in range(4) if r != home]
    load[other[-1]] = 1
    assert choose_rank("a", 4, load, 8) == (other[-1], "spilled")
    load[other[-1]] = 5  # three-way tie of 5 among the non-home ranks
    assert choose_rank("a", 4, load, 8) == (min(other), "spilled")


def test_choose_rank_sticky_when_home_is_itself_least_loaded():
    home = home_rank("a", 3)
    load = {r: 99 for r in range(3)}
    load[home] = 9
    assert choose_rank("a", 3, load, 8) == (home, "sticky")


def test_choose_rank_balanced_without_key():
    assert choose_rank(None, 3, {0: 2, 1: 0, 2: 1}, 8) == (1, "balanced")
    assert choose_rank(None, 3, {}, 8) == (0, "balanced")


def test_affinity_key_precedence():
    body = {"user": "user-0001", "messages": [{"role": "user", "content": "hi"}]}
    hdr = {"x-session-id": "sess-0001"}
    msgs = {"messages": body["messages"]}
    assert affinity_key(body, hdr, "t", explicit="meta-0001") == ("meta-0001", "anthropic_metadata")
    # headers outrank the body's user; the user only counts when nothing else does
    assert affinity_key(body, hdr, "t") == ("sess-0001", "x_session_id")
    assert affinity_key(body, {}, "t") == ("user-0001", "openai_user")
    assert affinity_key(msgs, hdr, "t") == ("sess-0001", "x_session_id")
    k, src = affinity_key(msgs, {}, "t")
    assert src == "prompt_hash" and k is not None
    assert affinity_key({}, {}, "t") == (None, "none")


def test_affinity_key_header_is_case_insensitive():
    assert affinity_key({}, {"X-SESSION-ID": "sess-0001"}, "t") == ("sess-0001", "x_session_id")


def test_empty_user_and_explicit_are_skipped():
    body = {"user": "", "messages": [{"role": "user", "content": "hi"}]}
    assert affinity_key(body, {}, "t", explicit="")[1] == "prompt_hash"


def test_prompt_hash_ignores_system_and_uses_first_512_chars():
    def key(msgs, tok="t"):
        return affinity_key({"messages": msgs}, {}, tok)[0]

    a = key([{"role": "system", "content": "S1"}, {"role": "user", "content": "x" * 600}])
    b = key([{"role": "system", "content": "S2"}, {"role": "user", "content": "x" * 512 + "y"}])
    assert a == b  # system differs, text differs only after char 512
    assert a != key([{"role": "user", "content": "other"}])
    assert key([{"role": "user", "content": "x"}], "t1") != key(
        [{"role": "user", "content": "x"}], "t2"
    )


def test_prompt_hash_text_parts_and_fallback_and_completions():
    parts = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "cd"}],
        }
    ]
    assert affinity_key({"messages": parts}, {}, "t") == affinity_key(
        {"messages": [{"role": "user", "content": "abcd"}]}, {}, "t"
    )
    only_sys = [{"role": "system", "content": "sys"}]
    assert affinity_key({"messages": only_sys}, {}, "t")[1] == "prompt_hash"
    assert affinity_key({"prompt": "p"}, {}, "t") == affinity_key({"prompt": ["p", "q"]}, {}, "t")
    assert affinity_key({"prompt": "p"}, {}, "t")[1] == "prompt_hash"


@pytest.mark.parametrize(
    ("raw", "want"), [("3", 3), (" 0 ", 0), ("7", None), ("-1", None), ("x", None)]
)
def test_client_rank(raw, want):
    assert client_rank({"X-data-parallel-rank": raw}, 7) == want
    assert client_rank({"x-DATA-parallel-RANK": raw}, 7) == want


def test_client_rank_absent():
    assert client_rank({}, 7) is None


def _route(st, key="a", **kw):
    args = dict(dp=4, key=key, threshold=8, pinned=None, affinity_enabled=True)
    args.update(kw)
    return st.route("m", **args)


def test_state_route_release_consistent_and_never_negative():
    st = DpRoutingState()
    r1, d1 = _route(st)
    r2, d2 = _route(st)
    assert r1 == r2 and (d1, d2) == ("placed", "sticky")
    assert st.snapshot("m", 4)["totals"]["in_flight"] == 2
    st.release("m", r1)
    st.release("m", r2)
    st.release("m", r2)  # extra release
    st.release("m", None)
    st.release("unknown", 0)
    snap = st.snapshot("m", 4)
    assert snap["totals"]["in_flight"] == 0
    assert all(r["in_flight"] >= 0 for r in snap["ranks"])


def test_state_client_pinned_counts_only_valid_rank():
    st = DpRoutingState()
    assert _route(st, pinned=5) == (5, "client_pinned")
    snap = st.snapshot("m", 7)
    assert snap["ranks"][5]["client_pinned"] == 1 and snap["ranks"][5]["in_flight"] == 1
    assert snap["totals"]["client_pinned"] == 1


def test_state_unrouted_when_affinity_off():
    st = DpRoutingState()
    assert _route(st, affinity_enabled=False) == (None, "unrouted")
    snap = st.snapshot("m", 4)
    assert snap["totals"]["unrouted"] == 1 and snap["totals"]["in_flight"] == 0


def test_state_balanced_and_spilled_counters():
    st = DpRoutingState()
    assert _route(st, key=None)[1] == "balanced"
    home, first = _route(st)  # first sight: placed on the least-loaded rank
    assert first == "placed"
    for _ in range(7):
        assert _route(st) == (home, "sticky")  # fills home to the threshold
    rank, decision = _route(st)
    assert decision == "spilled" and rank != home
    t = st.snapshot("m", 4)["totals"]
    assert t["balanced"] == 1 and t["spilled"] == 1 and t["sticky"] == 7 and t["placed"] == 1


def test_snapshot_shape_and_since():
    st = DpRoutingState()
    snap = st.snapshot("m", 3)
    assert snap["since"] is None
    assert [r["rank"] for r in snap["ranks"]] == [0, 1, 2]
    assert set(snap["ranks"][0]) == {
        "rank",
        "in_flight",
        "assigned_sessions",
        "placed",
        "sticky",
        "spilled_in",
        "client_pinned",
    }
    assert set(snap["totals"]) == {
        "in_flight",
        "placed",
        "sticky",
        "spilled",
        "client_pinned",
        "balanced",
        "unrouted",
    }
    _route(st)
    assert st.snapshot("m", 4)["since"] is not None
    st.reset("m")
    assert st.snapshot("m", 4)["since"] is None
