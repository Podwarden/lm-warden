"""Session lineage and edge detection (plan 3, task 4)."""

import gc
import os
import struct
import time

import pytest

from app.cobs2.canonical import canonicalize
from app.cobs2.ids import Ids, ids_for
from app.cobs2.keys import NetworkKey
from app.cobs2.lineage import COMPACTION_FINGERPRINTS, Lineage, similarity, sketch, strip_analysis

KEY = NetworkKey(epoch=1, key=b"k" * 32)
M = "m"
S = b""


def _ids(*texts: str, system: str = "be brief") -> Ids:
    msgs = [{"role": "system", "content": system}]
    for i, t in enumerate(texts):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    c = canonicalize({"messages": msgs})
    assert c is not None
    return ids_for(c, KEY, S)


def _full(lin: Lineage, ids: Ids, *, sk="s", psk=None, last="hello", first="", **kw):
    return lin.observe(
        model_id=M,
        scope=kw.pop("scope", S),
        session_key=sk,
        parent_session_key=psk,
        ids=ids,
        last_user_text_loose=last,
        first_body_text_loose=first,
        **kw,
    )


def _obs(lin: Lineage, ids: Ids, **kw):
    o = _full(lin, ids, **kw)
    return o.edge, o.parent


def test_new_session() -> None:
    lin = Lineage()
    ids = _ids("a")
    assert _obs(lin, ids) == ("new", None)


def test_continue_chain() -> None:
    lin = Lineage()
    a = _ids("a")
    assert _obs(lin, a)[0] == "new"
    b = _ids("a", "ra", "b")
    assert _obs(lin, b) == ("continue", a.nodes[-1])
    c = _ids("a", "ra", "b", "rb", "c")
    assert _obs(lin, c) == ("continue", b.nodes[-1])


def test_fork_from_older_node() -> None:
    lin = Lineage()
    _obs(lin, _ids("a"))
    b = _ids("a", "ra", "b")
    _obs(lin, b)
    c = _ids("a", "ra", "b", "rb", "c")
    _obs(lin, c)
    f = _ids("a", "ra", "other")
    edge, parent = _obs(lin, f)
    assert edge == "fork"
    assert parent == _ids("a", "ra").nodes[-1]


def test_compaction_by_fingerprint_and_by_overlap() -> None:
    assert COMPACTION_FINGERPRINTS
    # by fingerprint: previous request was a compaction request
    lin = Lineage()
    a = _ids("a", "ra", "write a summary of the conversation so far")
    _obs(lin, a, last="Write a summary of the conversation so far")
    collapsed = _ids("totally different opening")
    assert _obs(lin, collapsed, first="unrelated words here") == (
        "compacted_from",
        a.nodes[-1],
    )
    # by overlap with the last reply
    lin = Lineage()
    a = _ids("a", "ra", "b")
    _obs(lin, a)
    reply = "the quick brown fox jumps over the lazy dog and then runs far away into the hills"
    lin.remember_reply(
        model_id=M,
        scope=S,
        session="s",
        node=a.nodes[-1],
        reply_ids=_ids("x"),
        reply_text_loose=reply,
    )
    c = _ids("summary: " + reply)
    assert _obs(lin, c, first=reply + " plus tail")[0] == "compacted_from"
    # no overlap -> not compaction
    lin2 = Lineage()
    _obs(lin2, a)
    lin2.remember_reply(
        model_id=M,
        scope=S,
        session="s",
        node=a.nodes[-1],
        reply_ids=_ids("x"),
        reply_text_loose=reply,
    )
    assert (
        _obs(lin2, _ids("zzz"), first="nothing in common at all with that other text")[0] == "new"
    )


def test_compaction_by_head_and_token() -> None:
    lin = Lineage()
    a = _ids("a", "ra", "summary of the conversation so far")
    _obs(lin, a, sk=None, last="summary of the conversation so far", token_scope="t1")
    c = _ids("fresh")
    assert _obs(lin, c, sk=None, token_scope="t1")[0] == "compacted_from"
    lin2 = Lineage()
    _obs(lin2, a, sk=None, last="summary of the conversation so far", token_scope="t1")
    assert _obs(lin2, c, sk=None, token_scope="t2")[0] == "new"


def test_subagent_by_parent_header() -> None:
    lin = Lineage()
    a = _ids("a")
    _obs(lin, a, sk="parent")
    sub = _ids("task", system="subagent system")
    assert _obs(lin, sub, sk="child", psk="parent") == ("subagent", a.nodes[-1])
    # new head with another head in flight for the same token
    other = _ids("t2", system="another system")
    assert _obs(lin, other, sk="x", inflight_heads=[a.head])[0] == "subagent"
    # known head is not a subagent
    assert _obs(lin, _ids("zzz"), sk="y", inflight_heads=[a.head])[0] == "new"


def test_ttl_and_lru_bound() -> None:
    now = [0.0]
    lin = Lineage(max_nodes=10, ttl_s=100.0, clock=lambda: now[0])
    for i in range(30):
        _obs(lin, _ids(f"msg{i}", system=f"sys{i}"), sk=f"s{i}")
    assert len(lin._nodes) <= 10 and len(lin._sessions) <= 10
    a = _ids("a", "ra", "b")
    _obs(lin, a, sk="t")
    now[0] = 50.0
    assert _obs(lin, _ids("a", "ra", "b", "rb", "c"), sk="t")[0] == "continue"
    now[0] = 500.0
    assert _obs(lin, _ids("a", "ra", "b", "rb", "c", "rc", "d"), sk="t")[0] == "new"
    assert len(lin._nodes) == 7 and len(lin._sessions) == 1


def test_minhash_text_never_stored() -> None:
    lin = Lineage()
    a = _ids("a", "ra", "b")
    _obs(lin, a)
    secret = "zebra quokka marmot lemur gibbon tapir okapi dugong"
    lin.remember_reply(
        model_id=M,
        scope=S,
        session="s",
        node=a.nodes[-1],
        reply_ids=_ids("x"),
        reply_text_loose=secret,
    )
    blob = repr((lin._nodes, lin._loose, lin._sessions, lin._heads))
    for w in secret.split():
        assert w not in blob
    sig = next(iter(lin._sessions.values())).tips[-1].sig
    assert isinstance(sig, bytes) and len(sig) % 8 == 0 and 0 < len(sig) <= 64 * 8


def test_scopes_are_isolated() -> None:
    lin = Lineage()
    a = _ids("a")
    _obs(lin, a, scope=b"one")
    b = _ids("a", "ra", "b")
    assert _obs(lin, b, scope=b"two")[0] == "new"
    assert _obs(lin, b, scope=b"one")[0] == "continue"


def _pack(vals) -> bytes:
    vals = sorted(vals)
    return struct.pack(f"<{len(vals)}Q", *vals)


def test_similarity_threshold_boundary() -> None:
    a = _pack(range(0, 64))
    assert similarity(a, _pack(range(32, 96))) == 0.5  # exactly at the threshold
    assert similarity(a, _pack(range(33, 97))) < 0.5
    assert similarity(a, _pack(range(31, 95))) > 0.5
    assert similarity(a, a) == 1.0


def test_sketch_overlap_real_text() -> None:
    words = [f"w{i}" for i in range(300)]
    base = sketch(" ".join(words), b"s")
    assert base is not None and len(base) == 64 * 8
    half = sketch(" ".join(words[:150] + [f"x{i}" for i in range(150)]), b"s")
    other = sketch(" ".join(f"y{i}" for i in range(300)), b"s")
    assert half and other
    assert 0.0 < similarity(base, half) < 0.9
    assert similarity(base, other) == 0.0
    # keyed by scope: different scope, different values
    assert sketch(" ".join(words), b"t") != base


def test_interleaved_conversations_one_session_key() -> None:
    lin = Lineage()
    a1, b1 = _ids("a1"), _ids("b1")
    assert _obs(lin, a1)[0] == "new"
    assert _obs(lin, b1)[0] == "new"
    a2 = _ids("a1", "ra1", "a2")
    b2 = _ids("b1", "rb1", "b2")
    a3 = _ids("a1", "ra1", "a2", "ra2", "a3")
    b3 = _ids("b1", "rb1", "b2", "rb2", "b3")
    for ids in (a2, b2, a3, b3):
        assert _obs(lin, ids)[0] == "continue"


def test_true_rewind_is_fork_then_original_continues() -> None:
    lin = Lineage()
    _obs(lin, _ids("a"))
    _obs(lin, _ids("a", "ra", "b"))
    assert _obs(lin, _ids("a", "ra", "b", "rb", "c"))[0] == "continue"
    assert _obs(lin, _ids("a", "ra", "other")) == ("fork", _ids("a", "ra").nodes[-1])
    assert _obs(lin, _ids("a", "ra", "b", "rb", "c", "rc", "d"))[0] == "continue"


def test_identical_retry_reports_own_parent() -> None:
    lin = Lineage()
    b = _ids("a", "ra", "b")
    _obs(lin, b)
    assert _obs(lin, b) == ("continue", b.nodes[-2])
    one = _ids("solo", system="other head")
    _obs(lin, one)
    assert _obs(lin, one) == ("continue", None)


def test_reasoning_drop_is_continue_with_loose_flag() -> None:
    lin = Lineage()

    def conv(*, reasoning: bool, more: bool):
        asst = {"role": "assistant", "content": "ans"}
        if reasoning:
            asst["reasoning_content"] = "let me think about it"
        msgs = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "q"},
            asst,
            {"role": "user", "content": "q2"},
        ]
        if more:
            msgs += [{"role": "assistant", "content": "ans2"}, {"role": "user", "content": "q3"}]
        c = canonicalize({"messages": msgs})
        assert c is not None
        return ids_for(c, KEY, S)

    n = conv(reasoning=True, more=False)
    assert _full(lin, n).edge == "new"
    n1 = conv(reasoning=False, more=True)
    assert n1.nodes[:1] == n.nodes[:1] and n1.nodes[2] != n.nodes[2]
    o = _full(lin, n1)
    assert (o.edge, o.parent, o.loose_match_beyond_exact) == ("continue", n.nodes[2], True)
    # a plain continuation does not set the flag
    assert _full(lin, _ids("zz"), sk="other").loose_match_beyond_exact is False


def test_same_session_new_head_overlap_is_not_compaction() -> None:
    reply = "the quick brown fox jumps over the lazy dog and then runs far away into the hills"
    lin = Lineage()
    a = _ids("a", "ra", "b")
    _obs(lin, a)
    lin.remember_reply(
        model_id=M,
        scope=S,
        session="s",
        node=a.nodes[-1],
        reply_ids=_ids("x"),
        reply_text_loose=reply,
    )
    sub = _ids("task", system="subagent system")
    assert _obs(lin, sub, first=reply, inflight_heads=[a.head])[0] == "subagent"
    lin2 = Lineage()
    _obs(lin2, a)
    lin2.remember_reply(
        model_id=M,
        scope=S,
        session="s",
        node=a.nodes[-1],
        reply_ids=_ids("x"),
        reply_text_loose=reply,
    )
    assert _obs(lin2, sub, first=reply)[0] == "new"
    # a fingerprint-flagged previous request also needs the same head
    lin3 = Lineage()
    _obs(
        lin3,
        _ids("a", "summary of the conversation so far"),
        last="summary of the conversation so far",
    )
    assert _obs(lin3, sub)[0] == "new"


def test_inflight_entry_equal_to_head_is_ignored() -> None:
    lin = Lineage()
    h = _ids("x", system="fresh system")
    assert _obs(lin, h, sk="q", inflight_heads=[h.head])[0] == "new"


def test_no_cross_session_continue_even_same_token() -> None:
    lin = Lineage()
    a = _ids("a", "ra", "b")
    _obs(lin, a, sk="one", token_scope="t1")
    nxt = _ids("a", "ra", "b", "rb", "c")
    assert _obs(lin, nxt, sk="two", token_scope="t2")[0] == "new"
    lin2 = Lineage()
    _obs(lin2, a, sk="one", token_scope="t1")
    assert _obs(lin2, nxt, sk="three", token_scope="t1")[0] == "new"


def test_children_are_per_session() -> None:
    lin = Lineage()
    assert _obs(lin, _ids("hi"), sk="s1", token_scope="t1")[0] == "new"
    assert _obs(lin, _ids("hi", "ra2", "x"), sk="s2", token_scope="t2")[0] == "new"
    # session 1's real continuation is not a fork because of session 2's child
    assert _obs(lin, _ids("hi", "ra1", "y"), sk="s1", token_scope="t1")[0] == "continue"


def test_same_opening_in_two_sessions_each_continues() -> None:
    lin = Lineage()
    assert _obs(lin, _ids("hi"), sk="s1")[0] == "new"
    assert _obs(lin, _ids("hi"), sk="s2")[0] == "new"
    assert _obs(lin, _ids("hi", "r2", "x"), sk="s2")[0] == "continue"
    assert _obs(lin, _ids("hi", "r1", "y"), sk="s1")[0] == "continue"
    # fallback identity (no session key): token + head
    lin = Lineage()
    assert _obs(lin, _ids("hi"), sk=None, token_scope="t1")[0] == "new"
    assert _obs(lin, _ids("hi"), sk=None, token_scope="t2")[0] == "new"
    assert _obs(lin, _ids("hi", "r", "x"), sk=None, token_scope="t2")[0] == "continue"
    assert _obs(lin, _ids("hi", "r", "y"), sk=None, token_scope="t1")[0] == "continue"


def _reasoning_conv(thinking: str | None, more: bool):
    asst = {"role": "assistant", "content": "ans"}
    if thinking:
        asst["reasoning_content"] = thinking
    msgs = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "q"},
        asst,
        {"role": "user", "content": "q2"},
    ]
    if more:
        msgs += [{"role": "assistant", "content": "ans2"}, {"role": "user", "content": "q3"}]
    c = canonicalize({"messages": msgs})
    assert c is not None
    return ids_for(c, KEY, S)


def test_loose_exact_not_overwritten_by_other_session() -> None:
    lin = Lineage()
    mine = _reasoning_conv("my thoughts", False)
    theirs = _reasoning_conv("other thoughts", False)
    assert mine.nodes[2] != theirs.nodes[2] and mine.loose_nodes == theirs.loose_nodes
    _obs(lin, mine, sk="s1")
    _obs(lin, theirs, sk="s2")
    o = _full(lin, _reasoning_conv(None, True), sk="s1")
    assert (o.edge, o.parent, o.loose_match_beyond_exact) == ("continue", mine.nodes[2], True)


def test_compaction_survives_interleaved_subagent_fingerprint() -> None:
    lin = Lineage()
    main = _ids("a", "ra", "b")
    _obs(lin, main)
    comp = _ids("a", "ra", "b", "rb", "summary of the conversation so far")
    _obs(lin, comp, last="summary of the conversation so far")
    sub = _ids("task", system="subagent system")
    assert _obs(lin, sub, psk="s")[0] == "subagent"
    after = _ids("post compaction opening")
    assert _obs(lin, after) == ("compacted_from", comp.nodes[-1])


def test_compaction_survives_interleaved_subagent_overlap() -> None:
    reply = " ".join(f"fact{i}" for i in range(60))
    for before_reply in (True, False):
        lin = Lineage()
        main = _ids("a", "ra", "b")
        _obs(lin, main)
        sub = _ids("task", system="subagent system")

        def remember(lin=lin, main=main, reply=reply) -> None:
            lin.remember_reply(
                model_id=M,
                scope=S,
                session="s",
                node=main.nodes[-1],
                reply_ids=_ids("x"),
                reply_text_loose=reply,
            )

        if before_reply:
            assert _obs(lin, sub, psk="s")[0] == "subagent"
            remember()
        else:
            remember()
            assert _obs(lin, sub, psk="s")[0] == "subagent"
        after = _ids("post compaction")
        assert _obs(lin, after, first="Summary follows. " + reply + " please continue") == (
            "compacted_from",
            main.nodes[-1],
        )


def test_sketch_real_compaction_shape() -> None:
    summary = " ".join(f"s{i}" for i in range(4000))
    pre = "This session is being continued from a previous conversation. Summary:"
    first = pre + " " + summary + " Please continue the conversation."
    base = sketch(strip_analysis(summary), b"s")
    other = sketch(first, b"s")
    assert base and other and similarity(base, other) >= 0.5
    analysis = "<analysis> " + " ".join(f"t{i}" for i in range(4000)) + " </analysis>"
    stripped = strip_analysis(analysis + summary)
    assert stripped.strip() == summary
    withan = sketch(stripped, b"s")
    assert withan and similarity(withan, other) >= 0.5
    assert strip_analysis("no block <analysis>x</analysis>") == "no block <analysis>x</analysis>"


def test_matched_chain_is_refreshed() -> None:
    now = [0.0]
    lin = Lineage(max_nodes=100, ttl_s=100.0, clock=lambda: now[0])
    a = _ids("a", "ra", "b")
    _obs(lin, a)
    now[0] = 80.0
    c = _ids("a", "ra", "b", "rb", "c")
    _obs(lin, c)
    now[0] = 150.0  # the first nodes would have expired without a refresh
    assert _obs(lin, _ids("a", "ra", "b", "rb", "c", "rc", "d"))[0] == "continue"
    assert _obs(lin, _ids("a", "ra", "other"))[0] == "fork"


def _strict(strict: float, default: float) -> float:
    """The original bound under ``LMW_STRICT_PERF=1`` (a quiet box), else a
    generous one: shared runners and a busy Mac run 2-3x slower."""
    return strict if os.environ.get("LMW_STRICT_PERF") == "1" else default


def _best_of_3_mean(fn, n: int = 20) -> float:
    """Best-of-3 mean thread CPU time of ``fn(i)``, collector paused."""
    best = float("inf")
    for r in range(3):
        gc.collect()
        gc.disable()
        try:
            t0 = time.thread_time()
            for i in range(n):
                fn(r * n + i)
            best = min(best, (time.thread_time() - t0) / n)
        finally:
            gc.enable()
    return best


@pytest.mark.perf
def test_remember_reply_is_fast() -> None:
    lin = Lineage()
    a = _ids("a")
    _obs(lin, a)
    text = " ".join(f"word{i % 997}x{i}" for i in range(4000))
    x = _ids("x")

    def run(_i: int) -> None:
        lin.remember_reply(
            model_id=M,
            scope=S,
            session="s",
            node=a.nodes[-1],
            reply_ids=x,
            reply_text_loose=text,
        )

    assert _best_of_3_mean(run) < _strict(3e-3, 15e-3)


@pytest.mark.perf
def test_collapsed_observe_with_signature_is_fast() -> None:
    text = " ".join(f"word{i}" for i in range(4000))
    a = _ids("a", "ra", "b")
    x = _ids("x")
    collapsed = [_ids(f"collapsed{r}") for r in range(60)]
    lins = []
    for _ in range(60):
        lin = Lineage()
        _obs(lin, a)
        lin.remember_reply(
            model_id=M,
            scope=S,
            session="s",
            node=a.nodes[-1],
            reply_ids=x,
            reply_text_loose=text,
        )
        lins.append(lin)

    def run(i: int) -> None:
        edge, _ = _obs(lins[i], collapsed[i], first=text)
        assert edge == "compacted_from"

    assert _best_of_3_mean(run) < _strict(3e-3, 15e-3)


def test_observe_is_fast() -> None:
    lin = Lineage()
    texts = [f"message number {i}" for i in range(50)]
    ids = _ids(*texts)
    _obs(lin, ids)
    t0 = time.perf_counter()
    for _ in range(200):
        _obs(lin, ids)
    assert (time.perf_counter() - t0) / 200 < 1e-3
    # unknown 50-message chain (full backward scan, then insert)
    chains = [_ids(*[f"r{r} {t}" for t in texts]) for r in range(200)]
    t0 = time.perf_counter()
    for r, c in enumerate(chains):
        _obs(lin, c, sk=f"s{r}")
    assert (time.perf_counter() - t0) / 200 < 1e-3


def test_remember_reply_strips_analysis_block() -> None:
    summary = " ".join(f"s{i}" for i in range(500))
    analysis = "<analysis> " + " ".join(f"t{i}" for i in range(2500)) + " </analysis>"
    lin = Lineage()
    a = _ids("a", "ra", "b")
    _obs(lin, a)
    lin.remember_reply(
        model_id=M,
        scope=S,
        session="s",
        node=a.nodes[-1],
        reply_ids=_ids("x"),
        reply_text_loose=analysis + "\n" + summary,
    )
    out = _obs(lin, _ids("post"), first="Summary: " + summary + " continue")
    assert out == ("compacted_from", a.nodes[-1])


def test_dual_sketch_matches_when_the_analysis_block_is_kept() -> None:
    """A harness that keeps the <analysis> text in the post-compaction message
    still matches: the raw sketch is kept next to the stripped one."""
    summary = " ".join(f"s{i}" for i in range(500))
    analysis = "<analysis> " + " ".join(f"t{i}" for i in range(2500)) + " </analysis>"
    reply = analysis + "\n" + summary
    lin = Lineage()
    a = _ids("a", "ra", "b")
    _obs(lin, a)
    lin.remember_reply(
        model_id=M,
        scope=S,
        session="s",
        node=a.nodes[-1],
        reply_ids=_ids("x"),
        reply_text_loose=reply,
    )
    out = _obs(lin, _ids("post"), first="Summary: " + reply + " continue")
    assert out == ("compacted_from", a.nodes[-1])
    # and the stripped form still matches the usual harness
    lin2 = Lineage()
    _obs(lin2, a)
    lin2.remember_reply(
        model_id=M,
        scope=S,
        session="s",
        node=a.nodes[-1],
        reply_ids=_ids("x"),
        reply_text_loose=reply,
    )
    assert _obs(lin2, _ids("post"), first="Summary: " + summary + " go")[0] == "compacted_from"


def test_cross_head_compaction_by_fingerprint() -> None:
    """The summarize request arrives under its own summarizer system prompt (a
    different head): the collapsed request is compacted from that tip."""
    lin = Lineage()
    main = _ids("a", "ra", "b")
    _obs(lin, main)
    summ = _ids("a", "ra", "b", "rb", "summary of the conversation so far", system="summarizer")
    _obs(lin, summ, last="summary of the conversation so far")
    after = _ids("post compaction opening")
    assert _obs(lin, after) == ("compacted_from", summ.nodes[-1])


def test_cross_head_needs_the_fingerprint_not_overlap() -> None:
    reply = " ".join(f"fact{i}" for i in range(60))
    lin = Lineage()
    main = _ids("a", "ra", "b")
    _obs(lin, main)
    other = _ids("a", "ra", "b", "rb", "go on", system="summarizer")
    _obs(lin, other)
    lin.remember_reply(
        model_id=M,
        scope=S,
        session="s",
        node=other.nodes[-1],
        reply_ids=_ids("x"),
        reply_text_loose=reply,
    )
    after = _ids("post compaction opening")
    assert _obs(lin, after, first="Summary: " + reply)[0] == "new"


def test_cross_head_flag_is_bounded_by_the_window() -> None:
    now = [1000.0]
    lin = Lineage(clock=lambda: now[0])
    main = _ids("a", "ra", "b")
    _obs(lin, main)
    summ = _ids("a", "summary of the conversation so far", system="summarizer")
    _obs(lin, summ, last="summary of the conversation so far")
    now[0] += 601.0
    assert _obs(lin, _ids("post compaction opening"))[0] == "new"


# ---- task 5 fix round 1 -------------------------------------------------------

SUMMARY_ASK = "summary of the conversation so far"


def _explore_then_summarize(lin: Lineage, clock=None):
    """A1, C1 (Explore subagent head), A2, B (summarize under its own head,
    flagged), A3 (collapsed): Claude Code's shape."""
    _obs(lin, _ids("a"))
    _obs(lin, _ids("explore the repo", system="explore"))
    a2 = _ids("a", "ra", "b")
    _obs(lin, a2)
    b = _ids("a", "ra", "b", "rb", SUMMARY_ASK, system="summarizer")
    _obs(lin, b, last=SUMMARY_ASK)
    a3 = _ids("post compaction opening")
    return b, _obs(lin, a3)


def test_cross_head_flag_not_reused_by_a_later_subagent() -> None:
    lin = Lineage()
    b, a3 = _explore_then_summarize(lin)
    assert a3 == ("compacted_from", b.nodes[-1])
    c2 = _ids("explore another area", system="explore")
    assert _obs(lin, c2) == ("new", None)


def test_cross_head_flag_is_consumed() -> None:
    lin = Lineage()
    b, a3 = _explore_then_summarize(lin)
    assert a3[0] == "compacted_from"
    tip = next(t for t in lin._sessions[(M, S, "s")].tips if t.node == b.nodes[-1])
    assert tip.compaction is False
    assert _obs(lin, _ids("yet another fresh start"))[0] != "compacted_from"


def test_later_summarize_under_the_summarizer_head_is_not_compacted_from_it() -> None:
    now = [1000.0]
    lin = Lineage(clock=lambda: now[0])
    b, a3 = _explore_then_summarize(lin)
    assert a3[0] == "compacted_from"
    _obs(lin, _ids("post compaction opening", "r", "more work"))
    now[0] += 7200.0
    b2 = _ids("post compaction opening", "r", "more work", "r2", SUMMARY_ASK, system="summarizer")
    edge, parent = _obs(lin, b2, last=SUMMARY_ASK)
    assert edge != "compacted_from" and parent != b.nodes[-1]


def test_first_body_text_is_lazy() -> None:
    calls: list[int] = []

    def first() -> str:
        calls.append(1)
        return "words " * 10

    lin = Lineage()
    _obs(lin, _ids("a"))
    b = _ids("a", "ra", "b")
    assert _obs(lin, b, first=first)[0] == "continue"
    assert calls == []
    lin.remember_reply(
        model_id=M,
        scope=S,
        session="s",
        node=b.nodes[-1],
        reply_ids=_ids("x"),
        reply_text_loose="some reply text that is long enough to sketch here",
    )
    _obs(lin, _ids("collapsed"), first=first)
    assert calls == [1]


# ---- task 6 fix round 1: beyond only when the exact id really differs --------


def _reminder(i: int) -> Ids:
    return _ids(f"<system-reminder>variant {i}</system-reminder> x")


def test_exact_eviction_is_not_a_harness_rewrite() -> None:
    """The LRU drops A's exact nodes (B floods _nodes, but its variants share
    one loose node so A's loose nodes survive). A's identical resend then
    matches only through the loose index: that is not a rewrite."""
    lin = Lineage(max_nodes=6)
    a = _ids("a", "ra", "b")
    _full(lin, a)
    for i in range(6):
        _full(lin, _reminder(i))
    assert not any(k[-1] == a.nodes[0] for k in lin._nodes)
    o = _full(lin, _ids("a", "ra", "b", "rb", "c"))
    assert (o.edge, o.parent, o.loose_match_beyond_exact) == ("continue", a.nodes[-1], False)


def test_exact_ttl_expiry_is_not_a_harness_rewrite() -> None:
    """Whichever bound drops the exact nodes (here the TTL, with the loose
    entries still fresh), an unchanged resend is never flagged."""
    now = [0.0]
    lin = Lineage(ttl_s=100.0, clock=lambda: now[0])
    a = _ids("a", "ra", "b")
    _full(lin, a)
    for e in lin._nodes.values():
        e.ts = -1000.0
    now[0] = 50.0
    o = _full(lin, _ids("a", "ra", "b", "rb", "c"))
    assert (o.edge, o.loose_match_beyond_exact) == ("continue", False)


def test_real_rewrite_after_exact_eviction_is_still_flagged() -> None:
    lin = Lineage(max_nodes=6)
    _full(lin, _ids("<system-reminder>one</system-reminder> a", "ra", "b"))
    for i in range(6):
        _full(lin, _reminder(i))
    o = _full(lin, _ids("<system-reminder>two</system-reminder> a", "ra", "b", "rb", "c"))
    assert (o.edge, o.loose_match_beyond_exact) == ("continue", True)


def test_sessionless_reminder_change_in_the_system_prompt_continues() -> None:
    """No harness session id: the fallback session is keyed by the LOOSE head,
    so a changed reminder in the system prompt keeps the session."""
    lin = Lineage()
    a = _ids("a", system="<system-reminder>Monday</system-reminder> be brief")
    assert _obs(lin, a, sk=None, token_scope="tok")[0] == "new"
    b = _ids("a", "ra", "b", system="<system-reminder>Tuesday</system-reminder> be brief")
    assert a.head != b.head and a.loose_head == b.loose_head
    o = _full(lin, b, sk=None, token_scope="tok")
    assert (o.edge, o.parent, o.loose_match_beyond_exact) == ("continue", a.nodes[0], True)


# -- sessions pages, task 1: request ids on nodes ---------------------------


def test_parent_req_is_the_request_that_produced_the_parent_node() -> None:
    lin = Lineage()
    _full(lin, _ids("a"), request_id="r1")
    o2 = _full(lin, _ids("a", "ra", "b"), request_id="r2")
    o3 = _full(lin, _ids("a", "ra", "c"), request_id="r3")
    assert (o2.edge, o2.parent_req) == ("continue", "r1")
    # interior node "ra" was created by r2 but inherits r1, the request it extends
    assert (o3.edge, o3.parent_req) == ("fork", "r1")
    assert o3.exact_depth == 1


def test_identical_retry_points_at_the_parent_of_the_retried_turn() -> None:
    lin = Lineage()
    _full(lin, _ids("a"), request_id="r1")
    b = _ids("a", "ra", "b")
    _full(lin, b, request_id="r2")
    o = _full(lin, b, request_id="r2b")
    assert (o.edge, o.parent_req) == ("continue", "r1")


def test_compacted_from_carries_parent_req() -> None:
    lin = Lineage()
    _full(lin, _ids("a"), request_id="r1")
    _full(
        lin,
        _ids("a", "ra", "write a summary of the conversation so far"),
        last="write a summary of the conversation so far",
        request_id="r2",
    )
    o = _full(lin, _ids("here is the summary, please continue"), request_id="r3")
    assert (o.edge, o.parent_req) == ("compacted_from", "r2")


def test_loose_beyond_exact_reports_exact_depth() -> None:
    lin = Lineage()
    _full(lin, _ids("x <system-reminder>Mon</system-reminder>"), request_id="r1")
    o = _full(lin, _ids("x <system-reminder>Tue</system-reminder>", "ra", "b"), request_id="r2")
    assert o.loose_match_beyond_exact
    assert o.exact_depth == -1


def test_observe_without_request_id_keeps_parent_req_none() -> None:
    lin = Lineage()
    _full(lin, _ids("a"))
    o = _full(lin, _ids("a", "ra", "b"))
    assert o.edge == "continue" and o.parent_req is None


# -- reply-anchored continuation (a harness rewrote the first message) -------


def _reply_ids(text: str) -> Ids:
    c = canonicalize({"messages": [{"role": "assistant", "content": text}]})
    assert c is not None
    return ids_for(c, KEY, S)


def test_request_carrying_a_known_reply_continues_it_even_if_msg0_changed() -> None:
    """Gleb's session (2026-10-26): a subagent's 2nd request changed its 1st message, so the
    prefix was unknown and it started a new lane. It carries the reply LM Warden saw, so it
    continues that turn, in that conversation."""
    lin = Lineage()
    a = _ids("explore X, agent id 111")
    o1 = _full(lin, a, request_id="r1", conv="CONV-A")
    lin.remember_reply(
        model_id=M,
        scope=S,
        session=o1.session,
        node=a.nodes[-1],
        reply_ids=_reply_ids("calling tools"),
    )
    b = _ids("explore X, agent id 222", "calling tools", "results")
    o2 = _full(lin, b, request_id="r2", conv="CONV-B")
    assert (o2.edge, o2.parent, o2.parent_req) == ("continue", a.nodes[-1], "r1")
    assert o2.loose_match_beyond_exact is True and o2.exact_depth == -1
    assert o2.conv == "CONV-A"
    # and the lane goes on from there: the next turn extends b and keeps the inherited conv
    c = _ids("explore X, agent id 222", "calling tools", "results", "more", "go")
    o3 = _full(lin, c, request_id="r3", conv="CONV-B")
    assert (o3.edge, o3.parent_req, o3.conv) == ("continue", "r2", "CONV-A")


def test_known_reply_from_another_session_does_not_link() -> None:
    lin = Lineage()
    a = _ids("task")
    o1 = _full(lin, a, sk="s1", request_id="r1")
    lin.remember_reply(
        model_id=M, scope=S, session=o1.session, node=a.nodes[-1], reply_ids=_reply_ids("ans")
    )
    o2 = _full(lin, _ids("other", "ans", "x"), sk="s2", request_id="r2")
    assert o2.edge == "new" and o2.conv is None
