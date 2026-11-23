"""G3: the per-request kvsig parts event (``build_kvsig``), pure."""

from __future__ import annotations

import json
from typing import Any

from app.cache_obs.canonical import _SEP, CHUNK_BYTES, Chain
from app.cache_obs.index import Match
from app.cobs2.ids import HeadPartIds, Ids
from app.cobs2.lineage import Observation
from app.proxy.prompt_lens_kvsig import MAX_KVSIG_PARTS, build_kvsig


def _hex(tag: str, i: int) -> str:
    return "01" + f"{i:02x}"[-2:] * 15 + tag


def _ids(n: int) -> Ids:
    nodes = tuple("01" + f"{i:02x}"[-2:] * 16 for i in range(n))
    return Ids(
        head=_hex("aa", 0),
        msgs=tuple(_hex("bb", i) for i in range(n)),
        nodes=nodes,
        loose_head=_hex("cc", 0),
        loose_msgs=tuple(_hex("dd", i) for i in range(n)),
        loose_nodes=tuple(_hex("ee", i) for i in range(n)),
    )


_HP = HeadPartIds(
    sys=_hex("01", 1), tools=_hex("02", 2), loose_sys=_hex("03", 3), loose_tools=_hex("04", 4)
)


def _chain(n: int, step: int = 100) -> Chain:
    """Element 0 is tools, element 1 is raw message 0 (the system message),
    body node i is raw message i + 1 (element i + 2); each ``step`` bytes."""
    k = n + 2
    return Chain((), k * step, tuple(j * step for j in range(k)), 1, ())


def _kv(**over: Any) -> dict[str, Any]:
    n = over.pop("n", 4)
    kw: dict[str, Any] = {
        "ids": _ids(n),
        "head_parts": _HP,
        "obs": Observation("new", None),
        "chain": _chain(n, over.pop("step", 100)),
        "match_all": None,
        "match_own": None,
        "prompt_tokens": 1000,
        "raw_body_index": [i + 1 for i in range(n)],
        "parts_chars": [100] * n,
        "sys_chars": 100,
        "tools_chars": 100,
        "roles": ["user" if i % 2 == 0 else "assistant" for i in range(n)],
        "reason": None,
    }
    kw.update(over)
    return build_kvsig(**kw)


def _shape(ev: dict[str, Any]) -> list[tuple[str, int, int, str]]:
    return [(p["p"], p["from"], p["to"], p["state"]) for p in ev["parts"]]


def _match(best: int, total: int = 10_000) -> Match:
    return Match({None: best}, best, total, None)


def test_exact_prefix_merges_and_tail_is_per_node():
    ids = _ids(17)
    ev = _kv(
        n=17,
        ids=ids,
        obs=Observation("continue", "n", False, "s", "req7", exact_depth=13, loose_depth=14),
    )
    assert _shape(ev) == [
        ("sys", -1, -1, "exact"),
        ("tools", -1, -1, "exact"),
        ("nodes", 0, 13, "exact"),
        ("nodes", 14, 14, "loose"),
        ("nodes", 15, 15, "new"),
        ("nodes", 16, 16, "new"),
    ]
    merged = ev["parts"][2]
    # A merged run's ids are its last node's (a node id fixes its prefix).
    assert merged["exact8"] == ids.nodes[13][2:10]
    assert merged["loose8"] == ids.loose_nodes[13][2:10]
    assert merged["chars"] == 1400
    assert merged["roles"] == {"user": 7, "assistant": 7}
    assert ev["parts"][0]["exact8"] == _HP.sys[2:10]
    assert ev["parts"][0]["loose8"] == _HP.loose_sys[2:10]
    assert ev["parts"][1]["exact8"] == _HP.tools[2:10]
    assert ev["parts"][1]["loose8"] == _HP.loose_tools[2:10]
    assert ev["parent_request_id"] == "req7" and ev["profile"] == "loose/1"
    assert ev["available"] is True and "reason" not in ev
    assert (ev["exact_depth"], ev["loose_depth"], ev["n_nodes"]) == (13, 14, 17)
    assert ev["edge"] == "continue"
    assert ev["head8"] == ids.head[2:10]


def test_head_parts_loose_when_only_loose_known():
    ev = _kv(obs=Observation("continue", "n", True, "s", None, exact_depth=-1, loose_depth=1))
    assert [p["state"] for p in ev["parts"]] == ["loose", "loose", "loose", "loose", "new", "new"]


def test_no_sys_or_tools_part_when_empty():
    ev = _kv(n=2, sys_chars=0, tools_chars=0)
    assert _shape(ev) == [("nodes", 0, 0, "new"), ("nodes", 1, 1, "new")]


def test_other_key_span_marks_parts_other():
    # Own lineage knows nothing; another key's request covers tools, sys and
    # node 0 (each element 300 bytes, so each ends a chunk past own's 0).
    ev = _kv(n=4, step=300, match_all=_match(950), match_own=_match(0))
    assert [p["state"] for p in ev["parts"]][:4] == ["other", "other", "other", "new"]


def test_other_needs_an_own_match_to_compare():
    # No own figure: the part might be this key's own, so never "other".
    ev = _kv(n=4, step=300, match_all=_match(950), match_own=None)
    assert {p["state"] for p in ev["parts"]} == {"new"}


def test_other_never_within_the_chunk_own_may_have_sent():
    # Own matched 512 bytes (two chunks). Own's identical prefix may run up to
    # one byte short of the next chunk boundary, so an element ending before
    # 512 + CHUNK_BYTES might be own's: not "other".
    assert CHUNK_BYTES == 256
    ev = _kv(n=4, step=300, match_all=_match(1150), match_own=_match(512))
    # tools ends 299, sys 599 (< 768): new; node 0 ends 899, node 1 1199 > 1150.
    assert [p["state"] for p in ev["parts"]] == ["new", "new", "other", "new", "new", "new"]


def test_other_element_end_excludes_the_separator():
    # Node 0 (element 2) holds bytes [600, 900 - len(SEP)); matched exactly that far.
    end = 900 - len(_SEP)
    ev = _kv(n=4, step=300, match_all=_match(end), match_own=_match(0))
    assert [p["state"] for p in ev["parts"]][:4] == ["other", "other", "other", "new"]
    ev = _kv(n=4, step=300, match_all=_match(end - 1), match_own=_match(0))
    assert [p["state"] for p in ev["parts"]][:4] == ["other", "other", "new", "new"]


def test_own_lineage_wins_over_other():
    ev = _kv(
        n=4,
        step=300,
        match_all=_match(5000),
        match_own=_match(0),
        obs=Observation("continue", "n", False, "s", None, exact_depth=0, loose_depth=1),
    )
    assert [p["state"] for p in ev["parts"]] == [
        "exact",
        "exact",
        "exact",
        "loose",
        "other",
        "other",
    ]


def test_sys_other_needs_every_head_message_covered():
    # A developer message mid-body (raw index 3) is part of the head too.
    n = 4
    raw_body = [1, 2, 4, 5]
    chain = Chain((), 7 * 300, tuple(j * 300 for j in range(7)), 1, ())
    ev = _kv(
        n=n,
        chain=chain,
        raw_body_index=raw_body,
        raw_head_index=[0, 3],
        match_all=_match(1150),
        match_own=_match(0),
    )
    # raw 3 is element 4, ending at 1499 > 1150: sys is not covered.
    assert ev["parts"][0]["p"] == "sys" and ev["parts"][0]["state"] == "new"
    assert ev["parts"][1]["state"] == "other"  # tools
    ev = _kv(
        n=n,
        chain=chain,
        raw_body_index=raw_body,
        raw_head_index=[0, 3],
        match_all=_match(1500),
        match_own=_match(0),
    )
    assert ev["parts"][0]["state"] == "other"


def test_head_roles_counted_on_the_sys_part():
    ev = _kv(head_roles=["system", "developer", "system"])
    assert ev["parts"][0]["roles"] == {"system": 2, "developer": 1}
    assert ev["parts"][1]["roles"] == {}


def test_parts_capped_at_64():
    ev = _kv(n=500)
    parts = ev["parts"]
    assert MAX_KVSIG_PARTS == 64
    assert len(parts) == 64 and parts[-1]["to"] == 499
    nodes = [p for p in parts if p["p"] == "nodes"]
    # Contiguous, oldest merged first, the newest nodes stay single.
    assert nodes[0]["from"] == 0
    assert all(b["from"] == a["to"] + 1 for a, b in zip(nodes, nodes[1:], strict=False))
    assert (nodes[-1]["from"], nodes[-1]["to"]) == (499, 499)
    # The newest half of the parts are single nodes; runs grow toward the front.
    assert all(p["from"] == p["to"] for p in nodes[-len(nodes) // 2 :])
    widths = [p["to"] - p["from"] + 1 for p in nodes]
    assert widths == sorted(widths, reverse=True)
    assert sum(p["chars"] for p in nodes) == 500 * 100
    assert sum(sum(p["roles"].values()) for p in nodes) == 500
    big = nodes[0]
    assert big["exact8"] == _ids(500).nodes[big["to"]][2:10]
    assert ev["n_nodes"] == 500


def test_cap_never_merges_across_states():
    obs = Observation("continue", "n", False, "s", None, exact_depth=99, loose_depth=299)
    ev = _kv(n=500, obs=obs)
    parts = ev["parts"]
    assert len(parts) == 64
    nodes = [p for p in parts if p["p"] == "nodes"]
    assert (nodes[0]["from"], nodes[0]["to"], nodes[0]["state"]) == (0, 99, "exact")
    for p in nodes:
        want = "exact" if p["to"] <= 99 else ("loose" if p["to"] <= 299 else "new")
        assert p["state"] == want
        lo = "exact" if p["from"] <= 99 else ("loose" if p["from"] <= 299 else "new")
        assert lo == want, p


def test_cap_bounds_event_size():
    ev = _kv(n=20_000, roles=["user"] * 20_000)
    assert len(ev["parts"]) == 64
    assert len(json.dumps(ev)) < 16_000


def test_unknown_roles_are_not_echoed():
    ev = _kv(n=2, roles=["user", "x" * 5000])
    assert ev["parts"][-1]["roles"] == {"unknown": 1}


def test_unavailable_has_reason_and_no_parts():
    ev = _kv(ids=None, reason="over_cap")
    assert ev == {"available": False, "reason": "over_cap", "parts": []}
    assert _kv(ids=None) == {"available": False, "reason": "error", "parts": []}
    assert _kv(reason="not_chat") == {"available": False, "reason": "not_chat", "parts": []}


def test_reusable_tokens_from_bytes():
    ev = _kv(
        match_all=Match({None: 1000}, 1000, 2000, None),
        match_own=Match({None: 500}, 500, 2000, None),
        prompt_tokens=400,
    )
    assert (ev["reusable_any"], ev["reusable_own"]) == (200, 100)
    ev = _kv()
    assert (ev["reusable_any"], ev["reusable_own"]) == (None, None)


def test_no_observation_and_no_chain():
    ev = _kv(obs=None, chain=None, match_all=_match(5000), match_own=_match(0))
    assert ev["available"] is True
    assert (ev["edge"], ev["parent_request_id"], ev["conv8"]) == (None, None, None)
    assert (ev["exact_depth"], ev["loose_depth"]) == (-1, -1)
    assert {p["state"] for p in ev["parts"]} == {"new"}


def test_conv8_from_the_observation():
    conv = "01" + "ab" * 16
    ev = _kv(obs=Observation("new", None, conv=conv))
    assert ev["conv8"] == conv[2:10]


def _expected_state(i: int, *, exact: int, loose: int, chain: Chain, lo: int, hi: int) -> str:
    """The per-node rule, written out the slow way."""
    if i <= exact:
        return "exact"
    if i <= loose:
        return "loose"
    k = i + 2  # raw i + 1, msg_offset 1
    end = (
        chain.elem_starts[k + 1] - len(_SEP)
        if k + 1 < len(chain.elem_starts)
        else chain.total_bytes
    )
    return "other" if lo <= end <= hi else "new"


def test_run_states_match_the_per_node_rule_under_the_cap():
    n, step = 3000, 300
    chain = _chain(n, step)
    for exact, loose, own, best in [
        (-1, -1, 0, 900 * step),
        (10, 500, 2000 * step, 2600 * step),
        (-1, 40, 0, 30 * step),
        (2999, 2999, 0, 5000 * step),
        (5, 7, 100 * step, 100 * step + 10),
    ]:
        obs = Observation("continue", "n", False, "s", None, exact_depth=exact, loose_depth=loose)
        ev = _kv(
            n=n,
            chain=chain,
            obs=obs,
            match_all=_match(best, chain.total_bytes),
            match_own=_match(own),
        )
        nodes = [p for p in ev["parts"] if p["p"] == "nodes"]
        assert len(ev["parts"]) <= 64
        assert nodes[0]["from"] == 0 and nodes[-1]["to"] == n - 1
        assert all(b["from"] == a["to"] + 1 for a, b in zip(nodes, nodes[1:], strict=False))
        lo, hi = own + CHUNK_BYTES, best
        for p in nodes:
            want = {
                _expected_state(i, exact=exact, loose=loose, chain=chain, lo=lo, hi=hi)
                for i in range(p["from"], p["to"] + 1)
            }
            assert want == {p["state"]}, (exact, loose, own, best, p)


def test_twenty_thousand_nodes_build_at_most_66_parts():
    """20k nodes cost a bounded number of part builds (runs, not per-node dicts).

    Deterministic stand-in for a wall-clock bound: ``_part`` is the per-part work, and
    it runs at most once per published part plus the two head parts (sys, tools).
    """
    from unittest.mock import patch

    from app.proxy import prompt_lens_kvsig

    n = 20_000
    chain = _chain(n, 300)
    kw: dict[str, Any] = {
        "ids": _ids(n),
        "head_parts": _HP,
        "chain": chain,
        "match_all": _match(9000 * 300, chain.total_bytes),
        "match_own": _match(0),
        "obs": Observation("continue", "n", False, "s", None, exact_depth=100, loose_depth=4000),
        "prompt_tokens": 100_000,
        "roles": ["user", "assistant"] * (n // 2),
        "parts_chars": [100] * n,
        "raw_body_index": [i + 1 for i in range(n)],
        "raw_head_index": [0],
        "head_roles": ["system"],
        "sys_chars": 100,
        "tools_chars": 100,
        "reason": None,
    }
    with patch.object(prompt_lens_kvsig, "_part", wraps=prompt_lens_kvsig._part) as part:
        ev = build_kvsig(**kw)
    assert len(ev["parts"]) == 64
    assert {p["state"] for p in ev["parts"]} == {"exact", "loose", "other", "new"}
    assert 0 < part.call_count <= 66, part.call_count


def test_event_loose_depth_is_never_below_exact_depth():
    ev = _kv(obs=Observation("continue", "n", False, "s", None, exact_depth=2, loose_depth=-1))
    assert (ev["exact_depth"], ev["loose_depth"]) == (2, 2)
    assert [p["state"] for p in ev["parts"]] == ["exact", "exact", "exact", "new"]
