# tests/unit/cache_obs/test_index.py
from app.cache_obs.canonical import chain_of
from app.cache_obs.index import PrefixIndex


def _chat(*contents, tools=None):
    body = {"messages": [{"role": "user", "content": c} for c in contents]}
    if tools is not None:
        body["tools"] = tools
    return chain_of(body)


SYS = "S" * 3000


def test_empty_index_matches_nothing():
    m = PrefixIndex().match("m", "e", _chat(SYS, "q"))
    assert m.best_bytes == 0 and m.by_rank == {} and m.diverged_at is None


def test_match_is_per_rank():
    idx = PrefixIndex()
    idx.observe("m", "e", 3, _chat(SYS, "first"), "tokA")
    m = idx.match("m", "e", _chat(SYS, "second"))
    assert m.by_rank.get(3, 0) >= 2816  # whole shared system prompt, chunk-floored
    assert m.by_rank.get(4, 0) == 0
    assert m.best_bytes == m.by_rank[3]


def test_epoch_and_model_isolate():
    idx = PrefixIndex()
    idx.observe("m", "e1", None, _chat(SYS), "t")
    assert idx.match("m", "e2", _chat(SYS)).best_bytes == 0
    assert idx.match("other", "e1", _chat(SYS)).best_bytes == 0


def test_own_scope_sees_only_own_prompts():
    idx = PrefixIndex()
    idx.observe("m", "e", None, _chat(SYS, "a"), "tokA")
    assert idx.match("m", "e", _chat(SYS, "b"), token_id="tokB").best_bytes == 0
    assert idx.match("m", "e", _chat(SYS, "b"), token_id="tokA").best_bytes > 0
    assert idx.match("m", "e", _chat(SYS, "b")).best_bytes > 0  # "*" scope sees all


def test_continuation_is_not_divergence():
    idx = PrefixIndex()
    turn1 = _chat(SYS, "u1" * 200, "a1" * 200)
    idx.observe("m", "e", None, turn1, "t")
    turn2 = _chat(SYS, "u1" * 200, "a1" * 200, "u2" * 200)
    m = idx.match("m", "e", turn2)
    assert m.diverged_at is None
    assert m.best_bytes >= turn1.total_bytes - 256


def test_rewritten_start_reports_divergence_at_message_zero():
    idx = PrefixIndex()
    later = [f"l{i}" * 300 for i in range(4)]
    idx.observe("m", "e", None, _chat("SYSTEM time=10:00 " + "S" * 1000, *later), "t")
    m = idx.match("m", "e", _chat("SYSTEM time=10:01 " + "S" * 1000, *later, "new"))
    assert m.diverged_at == 0


def test_tools_change_reports_minus_one():
    idx = PrefixIndex()
    msgs = [f"x{i}" * 300 for i in range(3)]
    idx.observe("m", "e", None, _chat(*msgs, tools=[{"n": "a"}, {"n": "b"}]), "t")
    m = idx.match("m", "e", _chat(*msgs, tools=[{"n": "b"}, {"n": "a"}]))
    assert m.diverged_at == -1


def test_lru_bound_holds():
    idx = PrefixIndex(max_entries=50)
    for i in range(20):
        idx.observe("m", "e", None, _chat(f"p{i}" * 500), "t")
    assert len(idx) <= 50


def test_continuation_of_short_messages_is_not_divergence():
    idx = PrefixIndex()
    turn1 = ["hi", "ok", "yes", "sure"]
    idx.observe("m", "e", None, _chat(*turn1), "t")
    m = idx.match("m", "e", _chat(*turn1, "more"))
    assert m.diverged_at is None


def test_continuation_after_long_system_and_short_turns_is_not_divergence():
    idx = PrefixIndex()
    turn1 = [SYS] + ["ok"] * 8
    idx.observe("m", "e", None, _chat(*turn1), "t")
    m = idx.match("m", "e", _chat(*turn1, "new"))
    assert m.diverged_at is None


def test_two_ranks_drop_out_at_different_chunks():
    idx = PrefixIndex()
    long_chain = _chat(SYS, "tail" * 300)
    idx.observe("m", "e", 1, long_chain, "t")
    idx.observe("m", "e", 2, _chat(SYS, "other" * 300), "t")
    m = idx.match("m", "e", long_chain)
    assert m.by_rank[1] == long_chain.total_bytes
    assert 2816 <= m.by_rank[2] < m.by_rank[1]
    assert m.best_bytes == m.by_rank[1]


def test_fully_matched_chain_covers_total_bytes():
    idx = PrefixIndex()
    c = _chat(SYS, "q")
    idx.observe("m", "e", None, c, "t")
    m = idx.match("m", "e", c)
    assert m.by_rank[None] == c.total_bytes
    assert m.diverged_at is None


def test_high_ranks_and_unrouted_coexist_on_one_entry():
    idx = PrefixIndex()
    c = _chat(SYS, "q")
    for r in (None, 0, 7, 63):
        idx.observe("m", "e", r, c, None)
    m = idx.match("m", "e", c)
    assert m.by_rank == {r: c.total_bytes for r in (None, 0, 7, 63)}
    assert 5 not in m.by_rank


def test_chunk_entries_hold_a_compact_rank_mask():
    # a set per entry cost ~216 bytes; an int mask keeps 200k entries near 40 MB
    idx = PrefixIndex()
    idx.observe("m", "e", 3, _chat(SYS), None)
    assert all(v is None or isinstance(v, int) for v in idx._d.values())
