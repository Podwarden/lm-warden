"""Prompt Lens hot-path memories (app/lens/hot.py): bounds and packing."""

from __future__ import annotations

import sys

from app.lens.heads import MAX_DIGESTS, RecentHeads
from app.lens.hot import HeadSnap, ParentHeads, close_enough
from app.lens.probes import PackedDigests, first_changed_line


def _d(i: int) -> bytes:
    return i.to_bytes(8, "little")


def _snap(i: int) -> HeadSnap:
    digests = PackedDigests.of([_d(j) for j in range(MAX_DIGESTS)])
    return HeadSnap(f"sys-{i}", "tools", f"ls-{i}", "lt", (), digests)


def test_parent_heads_overfilled_keeps_at_most_max_heads_snapshots_alive() -> None:
    import gc
    import weakref

    p = ParentHeads(max_reqs=64, max_heads=4)
    refs = []
    for i in range(32):
        s = _snap(i)
        refs.append(weakref.ref(s))
        p.put(f"req-{i}", s)
        del s
    gc.collect()
    assert p.snapshots() == 4 and len(p) == 32
    # Only the live snapshots (and their digests) are still referenced.
    assert sum(1 for r in refs if r() is not None) == 4
    # An old request whose snapshot was evicted resolves to None.
    assert p.get("req-0") is None
    assert p.get("req-31") is not None and p.get("req-31").sys == "sys-31"


def test_parent_heads_shares_one_snapshot_per_head() -> None:
    p = ParentHeads(max_reqs=8, max_heads=8)
    a = _snap(1)
    for i in range(8):
        p.put(f"r{i}", a)
    assert p.snapshots() == 1 and len(p) == 8
    p.put("r8", _snap(2))
    assert len(p) == 8 and p.get("r0") is None


def test_packed_digests_index_and_compare_like_the_tuple() -> None:
    t = tuple(_d(i) for i in range(5))
    p = PackedDigests.of(t)
    assert len(p) == 5 and p[0] == t[0] and p[-1] == t[-1] and list(p) == list(t)
    assert p == t and p == PackedDigests.of(list(t))
    assert first_changed_line(p, (*t[:2], _d(99), *t[3:])) == (3, 1)


def test_packed_digests_are_one_object() -> None:
    """2,048 digests: 16 KB plus a header, not ~70 KB of bytes objects."""
    t = tuple(_d(i) for i in range(MAX_DIGESTS))
    p = PackedDigests.of(t)
    as_tuple = sys.getsizeof(t) + sum(sys.getsizeof(x) for x in t)
    packed = sys.getsizeof(p) + sys.getsizeof(p.raw)
    assert packed < 17 * 1024 < as_tuple


def test_recent_heads_store_packed_digests() -> None:
    h = RecentHeads()
    h.remember(b"s", "m", "ls", "t", "sys", tuple(_d(i) for i in range(3)), now=0.0)
    e = h.lookup(b"s", "m", "ls", "t", now=1.0)
    assert e is not None and isinstance(e.digests, PackedDigests) and len(e.digests) == 3


def test_recent_heads_by_tools_fallback_index() -> None:
    h = RecentHeads(ttl_s=10.0)
    h.remember(b"s", "m", "ls-1", "t", "sys-1", (_d(1),), now=0.0)
    f = h.lookup_by_tools(b"s", "m", "t", now=1.0)
    assert f is not None and (f.sys_id, f.loose_sys) == ("sys-1", "ls-1")
    assert h.lookup_by_tools(b"other", "m", "t", now=1.0) is None
    assert h.lookup_by_tools(b"s", "m", "t2", now=1.0) is None
    assert h.lookup_by_tools(b"s", "m", "t", now=11.0) is None


def test_close_enough() -> None:
    base = [_d(i) for i in range(10)]
    three = [*base[:7], _d(97), _d(98), _d(99)]
    four = [*base[:6], _d(96), _d(97), _d(98), _d(99)]
    assert close_enough(base, three) and not close_enough(base, four)
    assert not close_enough(base, base)  # no change: nothing to name
    big = [_d(i) for i in range(200)]
    ten = [*big[:190], *(_d(1000 + i) for i in range(10))]
    eleven = [*big[:189], *(_d(1000 + i) for i in range(11))]
    assert close_enough(big, ten) and not close_enough(big, eleven)
    assert not close_enough(None, base) and not close_enough((), base)


# ---- C1: a first turn has no prior turn of any kind -------------------------------


def test_first_turn_needs_no_prior_turn_of_any_kind() -> None:
    from app.lens.hot import _first_turn

    assert _first_turn(["user"]) is True
    assert _first_turn(["user", "user"]) is True
    assert _first_turn(["user", "assistant", "user"]) is False
    # An agent loop: the only user message is the first one (Review Focus 1).
    assert _first_turn(["user", "assistant", "tool", "assistant", "tool"]) is False
    assert _first_turn(["user", "tool"]) is False
    assert _first_turn(["assistant"]) is None and _first_turn([]) is None


def test_prepare_marks_a_tool_loop_body_as_not_a_first_turn() -> None:
    """Both wire formats reach ``prepare`` as a chat body (an Anthropic body is
    translated first: its ``tool_result`` blocks become ``tool`` messages)."""
    import os

    from app.cobs2.canonical import canonicalize
    from app.cobs2.ids import ids_for
    from app.cobs2.keys import NetworkKey
    from app.lens.hot import prepare
    from app.proxy.messages_schemas import MessagesRequest
    from app.proxy.messages_translate import to_openai_request

    key = NetworkKey(epoch=1, key=os.urandom(32))
    call = {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
    oai = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "list"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "a b"},
        ],
    }
    anthropic = to_openai_request(
        MessagesRequest.model_validate(
            {
                "model": "m",
                "max_tokens": 8,
                "messages": [
                    {"role": "user", "content": "list"},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "t1", "name": "ls", "input": {}}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a"}],
                    },
                ],
            }
        )
    )
    first = {"model": "m", "messages": [{"role": "user", "content": "list"}]}
    for body, want in ((oai, False), (anthropic, False), (first, True)):
        canon = canonicalize(body)
        assert canon is not None
        pre = prepare(body, canon, ids_for(canon, key, b"s"), key, b"s")
        assert pre.first_turn is want, body


# ---- C2: per-tool hashes are capped and packed ------------------------------------


def _key():
    import os

    from app.cobs2.keys import NetworkKey

    return NetworkKey(epoch=1, key=os.urandom(32))


def test_tool_hashes_are_packed_and_capped_at_max_tools() -> None:
    from app.lens import schema_lint
    from app.lens.probes import tool_def_hashes

    key = _key()
    n = schema_lint.MAX_TOOLS
    h = tool_def_hashes([f"t{i}" for i in range(n)], key, b"s")
    # One bytes object (8 bytes a tool), not one object per tool.
    assert isinstance(h, PackedDigests) and len(h) == n and len(h.raw) == 8 * n
    # Above the cap the hashes are unknown: the loose comparison decides.
    assert tool_def_hashes([f"t{i}" for i in range(n + 1)], key, b"s") is None
    assert tool_def_hashes([], key, b"s") == ()


def test_a_50k_tool_body_pins_no_per_tool_memory() -> None:
    """C2: a hostile key holder's 50,000 ``{}`` tools (about 2 bytes each, so
    well under MAX_CHAIN_BYTES) keep nothing per tool in the memos or in
    ParentHeads."""
    import gc
    import tracemalloc

    from app.cobs2.canonical import canonicalize
    from app.cobs2.ids import ids_for
    from app.lens import hot

    hot.clear_memos()
    key = _key()
    parents = ParentHeads()
    body = {
        "model": "m",
        "tools": [{} for _ in range(50_000)],
        "messages": [{"role": "user", "content": "hi"}],
    }
    canon = canonicalize(body)
    assert canon is not None
    ids = ids_for(canon, key, b"s")
    gc.collect()
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        pre = hot.prepare(body, canon, ids, key, b"s")
        unknown = pre.snap.tool_hashes is None
        parents.put("req-1", pre.snap)
        del pre
        gc.collect()
        kept = tracemalloc.get_traced_memory()[0] - base
    finally:
        tracemalloc.stop()
        hot.clear_memos()
    # Per-tool bytes objects would be about 50,000 x 41 bytes = 2 MB.
    assert kept < 256 * 1024, kept
    assert unknown


def test_tool_hash_compare_works_on_packed_hashes() -> None:
    from app.cobs2.lineage import Observation
    from app.lens import causes
    from app.lens.probes import tool_def_hashes

    key = _key()
    a = tool_def_hashes(["a", "b", "c"], key, b"s")
    b = tool_def_hashes(["c", "a", "b"], key, b"s")
    c = tool_def_hashes(["a", "b", "c", "d"], key, b"s")
    obs = Observation("new", None)

    def cause(now, parent):
        ci = causes.CauseInputs(
            edge="new",
            obs=obs,
            tools_now="t-now",
            tools_parent="t-parent",
            loose_tools_now="l1",
            loose_tools_parent="l2",
            tool_hashes_now=now,
            tool_hashes_parent=parent,
        )
        return causes.evaluate(ci, render_order="sys_first").cause

    assert cause(b, a) == "tools_reordered"
    assert cause(c, a) == "tools_added_end"
    assert cause(None, a) == "tools_changed"


# ---- I3: probe and result memos sized like the kvsig loose memo -------------------


def test_probe_memos_match_the_loose_memo_and_pack_each_entry() -> None:
    from app.cobs2 import ids as ids_mod
    from app.cobs2.canonical import canonicalize
    from app.cobs2.ids import ids_for
    from app.lens import hot

    assert hot.PROBES_MAX == hot.RESULTS_MAX == ids_mod._LOOSE_MEMO_MAX == 65_536
    hot.clear_memos()
    key = _key()
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    canon = canonicalize(body)
    assert canon is not None
    ids = ids_for(canon, key, b"s")
    pre = hot.prepare(body, canon, ids, key, b"s")
    (entry,) = hot._PROBES._od.values()
    # One 32-byte bytes per message: the four 8-byte probes back to back.
    assert type(entry) is bytes and len(entry) == 32
    p = pre.probes_at(0)
    assert len(p) == 4 and all(len(x) == 8 for x in p) and b"".join(p) == entry
    hot.clear_memos()


# ---- M1: the span line is read without its line ending ----------------------------


def test_sys_line_numbers_like_the_digests() -> None:
    from app.lens import probes
    from app.lens.hot import sys_line

    text = "a\r\nb\nc"
    assert [sys_line(text, n) for n in (1, 2, 3, 4)] == ["a", "b", "c", None]
    big = "\n".join(f"l{i}" for i in range(probes.MAX_LINES + 3))
    assert sys_line(big, probes.MAX_LINES) == f"l{probes.MAX_LINES - 1}"
    rest = "\n".join(f"l{i}" for i in range(probes.MAX_LINES, probes.MAX_LINES + 3))
    assert sys_line(big, probes.MAX_LINES + 1) == rest
