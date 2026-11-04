# tests/unit/cache_obs/test_classify_cobs2.py
"""cobs2 miss causes (plan 3 task 6): compacted and harness_rewrite."""

import json

import pytest

from app.cache_obs.classify import OUTCOMES, classify, divergence_cause
from app.cache_obs.record import observe_fields
from app.proxy.request_registry import LiveRequest, finished_record
from tests.unit.proxy.test_dp_routing_forward import OK_BODY, _post, _ready, _resp

SESSION = {"X-Claude-Code-Session-Id": "0b1c2d3e-1111-4222-8333-444455556666"}

#: (cached, reusable, fleet, diverged_at, allow_misrouted) -> the result
#: WITHOUT cobs2 inputs, one row per existing outcome.
CASES = [
    ((9000, 9000, None, None, True), "hit"),
    ((5000, 9000, None, None, True), "partial"),
    ((0, 9000, None, None, True), "lost"),
    ((0, 0, 8000, None, True), "misrouted"),
    ((0, 100, None, 0, True), "diverged"),
    ((0, 100, None, None, True), "cold"),
]


def _c(args, **kw):
    cached, r, fleet, div, mis = args
    return classify(
        cached=cached,
        reusable=r,
        reusable_fleet=r if fleet is None else fleet,
        diverged_at=div,
        allow_misrouted=mis,
        **kw,
    )


def test_outcomes_appended_after_the_existing_ones():
    assert OUTCOMES == (
        "hit",
        "partial",
        "lost",
        "misrouted",
        "diverged",
        "cold",
        "compacted",
        "harness_rewrite",
    )


@pytest.mark.parametrize("args,expected", CASES)
def test_existing_outcomes_unchanged_without_cobs2_inputs(args, expected):
    assert _c(args) == expected
    # the defaults are off; a non-compaction edge changes nothing either
    assert _c(args, cobs2_edge=None, loose_match_beyond_exact=False) == expected
    for edge in ("new", "continue", "fork", "subagent"):
        assert _c(args, cobs2_edge=edge) == expected


#: Only a miss with nothing reusable is relabelled. A "lost" request still had
#: 256+ reusable tokens (the head before the rewrite): the engine evicted
#: them, and that eviction must stay visible (fix round 1, I1).
EXPLAINED = ("diverged", "cold")


@pytest.mark.parametrize("args,expected", CASES)
def test_compaction_is_compacted_not_lost(args, expected):
    """A compaction relabels diverged / cold; a lost request stays lost."""
    got = _c(args, cobs2_edge="compacted_from")
    assert got == ("compacted" if expected in EXPLAINED else expected)


@pytest.mark.parametrize("args,expected", CASES)
def test_loose_beyond_exact_is_harness_rewrite(args, expected):
    got = _c(args, loose_match_beyond_exact=True)
    assert got == ("harness_rewrite" if expected in EXPLAINED else expected)


def test_lost_is_never_relabelled():
    lost = CASES[2][0]
    assert _c(lost, cobs2_edge="compacted_from", loose_match_beyond_exact=True) == "lost"


def test_compacted_wins_over_harness_rewrite():
    cold = CASES[5][0]
    assert _c(cold, cobs2_edge="compacted_from", loose_match_beyond_exact=True) == "compacted"


@pytest.mark.parametrize(
    "outcome,div,edge,flag,want",
    [
        # relabelled rows carry their cause
        ("compacted", None, "compacted_from", False, "compacted"),
        ("harness_rewrite", 1, None, True, "harness_rewrite"),
        ("harness_rewrite", None, None, True, "harness_rewrite"),
        # a hit / partial with a deep divergence that cobs2 explains
        ("hit", 3, None, True, "harness_rewrite"),
        ("partial", 3, None, True, "harness_rewrite"),
        ("hit", 3, "compacted_from", True, "compacted"),
        # the compacted_from edge is itself the evidence the body was replaced
        # (a collapsed body has no later messages for diverged_at to see)
        ("hit", None, "compacted_from", False, "compacted"),
        ("partial", None, "compacted_from", False, "compacted"),
        # a rewrite flag without the index's divergence evidence explains nothing
        ("hit", None, None, True, None),
        # no cobs2 cause: NULL
        ("hit", 3, None, False, None),
        ("hit", 3, "continue", False, None),
        # lost, misrouted and unknown are never given a cause
        ("lost", 2, "compacted_from", True, None),
        ("misrouted", 2, None, True, None),
        (None, 2, None, True, None),
    ],
)
def test_divergence_cause(outcome, div, edge, flag, want):
    assert divergence_cause(outcome, div, edge, flag) == want


@pytest.mark.parametrize("args", [a for a, e in CASES if e in ("hit", "partial", "misrouted")])
def test_hit_is_never_relabelled(args):
    plain = _c(args)
    for kw in (
        {"cobs2_edge": "compacted_from"},
        {"loose_match_beyond_exact": True},
        {"cobs2_edge": "compacted_from", "loose_match_beyond_exact": True},
    ):
        assert _c(args, **kw) == plain


def test_unknown_stays_unknown():
    assert (
        classify(
            cached=None,
            reusable=5000,
            reusable_fleet=None,
            diverged_at=None,
            cobs2_edge="compacted_from",
            loose_match_beyond_exact=True,
        )
        is None
    )


def test_observe_fields_passes_both_inputs_to_both_lenses():
    f = observe_fields(
        prompt=1000,
        cached=0,
        reporting="usage",
        ttft_s=None,
        cold_rate=None,
        match_all=None,
        match_own=None,
        rank=None,
        rank_known=True,
        cobs2_edge="compacted_from",
    )
    # no index match: R is unknown, so the outcome stays unknown
    assert f["cache_outcome"] is None
    from app.cache_obs.index import Match

    m = Match(best_bytes=0, total_bytes=1000, by_rank={}, diverged_at=None)
    for kw, want in (
        ({"cobs2_edge": "compacted_from"}, "compacted"),
        ({"loose_match_beyond_exact": True}, "harness_rewrite"),
        ({}, "cold"),
    ):
        f = observe_fields(
            prompt=1000,
            cached=0,
            reporting="usage",
            ttft_s=None,
            cold_rate=None,
            match_all=m,
            match_own=m,
            rank=None,
            rank_known=True,
            **kw,
        )
        assert (f["cache_outcome"], f["cache_outcome_own"]) == (want, want)
        cause = want if want != "cold" else None
        assert (f["cobs2_divergence_cause"], f["cobs2_divergence_cause_own"]) == (cause, cause)


def test_observe_fields_cause_per_lens():
    """Each lens gets its own cause, from its own outcome and diverged_at."""
    from app.cache_obs.index import Match

    fleet = Match(best_bytes=900, total_bytes=1000, by_rank={None: 900}, diverged_at=2)
    own = Match(best_bytes=900, total_bytes=1000, by_rank={None: 900}, diverged_at=None)
    f = observe_fields(
        prompt=1000,
        cached=900,
        reporting="usage",
        ttft_s=None,
        cold_rate=None,
        match_all=fleet,
        match_own=own,
        rank=None,
        rank_known=True,
        loose_match_beyond_exact=True,
    )
    assert (f["cache_outcome"], f["cache_outcome_own"]) == ("hit", "hit")
    assert f["cobs2_divergence_cause"] == "harness_rewrite"
    assert f["cobs2_divergence_cause_own"] is None


def test_live_request_and_record_carry_the_flag_in_memory_only():
    req = LiveRequest(
        id="r",
        token_id="t",
        token_name="n",
        client_ip=None,
        model="m",
        model_row_id="mid",
        variant_id=None,
        path="/v1/chat/completions",
        prompt_tokens=1,
        max_model_len=None,
        started_monotonic=0.0,
        started_iso="x",
    )
    assert req.cobs2_loose_beyond_exact is False
    req.cobs2_loose_beyond_exact = True
    assert finished_record(req, now=1.0)["cobs2_loose_beyond_exact"] is True
    from app.stats import request_history as rh

    assert "cobs2_loose_beyond_exact" not in rh._COLUMNS
    assert {"cobs2_divergence_cause", "cobs2_divergence_cause_own"} <= set(rh._COLUMNS)


# ---- end to end through _forward ---------------------------------------------


def _reply(text, prompt=1000, cached=0):
    b = json.loads(json.dumps(OK_BODY))
    b["choices"] = [
        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
    ]
    b["usage"]["prompt_tokens"] = prompt
    b["usage"]["prompt_tokens_details"] = {"cached_tokens": cached}
    return _resp(b)


def _turn(client, recs, messages, reply="ok"):
    r, _ = _post(
        client, {"model": "qwen", "messages": messages}, headers=SESSION, resp=_reply(reply)
    )
    assert r.status_code == 200
    return recs[-1]


def _task(reminder):
    return (
        f"<system-reminder>{reminder}</system-reminder>\n"
        + "Refactor the request parser so it streams. " * 40
    )


def test_reminder_change_is_harness_rewrite(tmp_data_dir, client):
    """Review focus 3: two Claude Code turns where only the system-reminder
    text changed. The exact ids differ, the loose ids match, so the miss is
    the harness's rewrite, not a client prefix change."""
    from tests.unit.proxy.test_cache_estimate_forward import _spy_history

    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    t1 = _turn(client, recs, [{"role": "user", "content": _task("Today is Monday.")}], "done")
    t2 = _turn(
        client,
        recs,
        [
            {"role": "user", "content": _task("Today is Tuesday. 3 files changed.")},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "now add tests"},
        ],
    )
    assert t1["cobs2_loose_beyond_exact"] is False
    assert t2["cobs2_loose_beyond_exact"] is True
    assert t2["cobs2_edge"] == "continue"
    assert t2["cache_outcome"] == "harness_rewrite"
    assert t2["cobs2_divergence_cause"] == "harness_rewrite"


def test_compacted_through_forward(tmp_data_dir, client):
    from tests.unit.proxy.test_cache_estimate_forward import _spy_history

    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    summary = " ".join(f"point{i} of the plan we agreed" for i in range(80))
    a = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "plan it"}]
    _turn(client, recs, a, reply=summary)
    collapsed = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "Context from before: " + summary},
    ]
    t2 = _turn(client, recs, collapsed)
    assert t2["cobs2_edge"] == "compacted_from"
    assert t2["cache_outcome"] == "compacted"


# ---- fix round 1: a real harness head (I1) -----------------------------------

#: ~1,300 tokens of system prompt: a real harness head, reusable on its own.
BIG_SYS = "You are Claude Code, a careful coding agent working in a repository. " * 80


def _big_turn(client, recs, messages, *, reply="ok", prompt=2000, cached=0):
    r, _ = _post(
        client,
        {"model": "qwen", "messages": messages},
        headers=SESSION,
        resp=_reply(reply, prompt=prompt, cached=cached),
    )
    assert r.status_code == 200
    return recs[-1]


def _rewrite_run(client, tmp_data_dir, cached):
    """Three turns; the third resends the first user message with a changed
    reminder after two later messages the index has seen."""
    from tests.unit.proxy.test_cache_estimate_forward import _spy_history

    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    sys = {"role": "system", "content": BIG_SYS}
    # the task first, the reminder after it: the index's break lands inside
    # this message, not floored back into the unchanged head
    task = "Fix the request parser so that it streams large bodies. " * 10
    u1 = {"role": "user", "content": task + "<system-reminder>Monday</system-reminder>"}
    u1b = {"role": "user", "content": task + "<system-reminder>Tuesday 3 files</system-reminder>"}
    a1 = {"role": "assistant", "content": "done"}
    u2 = {"role": "user", "content": "now add tests"}
    a2 = {"role": "assistant", "content": "added"}
    u3 = {"role": "user", "content": "run them"}
    _big_turn(client, recs, [sys, u1], reply="done", cached=0)
    _big_turn(client, recs, [sys, u1, a1, u2], reply="added", cached=1900)
    return _big_turn(client, recs, [sys, u1b, a1, u2, a2, u3], cached=cached)


def test_big_head_reminder_change_with_eviction_stays_lost(tmp_data_dir, client):
    t3 = _rewrite_run(client, tmp_data_dir, cached=0)
    assert t3["cobs2_loose_beyond_exact"] is True
    assert t3["reusable_tokens"] >= 600
    assert t3["cache_outcome"] == "lost"
    assert t3["cobs2_divergence_cause"] is None


def test_big_head_reminder_change_kept_prefix_is_hit_with_cause(tmp_data_dir, client):
    t3 = _rewrite_run(client, tmp_data_dir, cached=1990)
    assert t3["cache_outcome"] == "hit"
    assert t3["diverged_at"] is not None
    assert t3["cobs2_divergence_cause"] == "harness_rewrite"


def _compact_run(client, tmp_data_dir, cached):
    from tests.unit.proxy.test_cache_estimate_forward import _spy_history

    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    summary = " ".join(f"point{i} of the plan we agreed" for i in range(80))
    sys = {"role": "system", "content": BIG_SYS}
    _big_turn(client, recs, [sys, {"role": "user", "content": "plan it"}], reply=summary)
    collapsed = [sys, {"role": "user", "content": "Context from before: " + summary}]
    return _big_turn(client, recs, collapsed, cached=cached)


def test_big_head_compaction_with_eviction_stays_lost(tmp_data_dir, client):
    t2 = _compact_run(client, tmp_data_dir, cached=0)
    assert t2["cobs2_edge"] == "compacted_from"
    assert t2["reusable_tokens"] >= 600
    assert t2["cache_outcome"] == "lost"
    assert t2["cobs2_divergence_cause"] is None


def test_big_head_compaction_kept_head_has_cause(tmp_data_dir, client):
    t2 = _compact_run(client, tmp_data_dir, cached=1990)
    assert t2["cobs2_edge"] == "compacted_from"
    assert t2["cache_outcome"] == "hit"
    assert t2["cobs2_divergence_cause"] == "compacted"
