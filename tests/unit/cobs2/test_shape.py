"""Request shape and toolset (sessions pages, task 1): sizes and hashes, never text."""

import json
from typing import Any

import pytest

from app.cobs2.canonical import canonicalize
from app.cobs2.lineage import Observation
from app.cobs2.shape import SHAPE_MAX_BYTES, break_index, shape_json, shape_of, toolset_of
from app.proxy.forest_fields import hkey, tool_family


def _fn(name: str, desc: str = "", params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": params or {}},
    }


def test_shape_counts_sizes_not_text() -> None:
    c = canonicalize(
        {
            "messages": [
                {"role": "system", "content": "x" * 120},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "tool_call_id": "t", "content": "r"},
                {"role": "user", "content": "y" * 23},
            ],
            "tools": [_fn("Read"), _fn("Task")],
        }
    )
    assert c is not None
    s = shape_of(c)
    assert s["sys"] == 120 and s["tools"] == 2 and s["last_user"] == 23
    assert s["msgs"] == {"user": 2, "assistant": 1, "tool": 1}
    assert s["tools_chars"] == sum(len(t) for t in c.tools)
    assert s["chars"] == 120 + 5 + 2 + 1 + 23 + s["tools_chars"]
    assert "hello" not in json.dumps(s)


def test_shape_last_user_none_without_user() -> None:
    c = canonicalize({"messages": [{"role": "system", "content": "s"}]})
    assert c is not None
    assert shape_of(c)["last_user"] is None


def test_shape_json_adds_brk_and_stays_under_200_bytes() -> None:
    c = canonicalize({"messages": [{"role": "user", "content": "hi"}]})
    assert c is not None
    out = shape_json(shape_of(c), 3)
    assert out is not None and len(out.encode()) <= SHAPE_MAX_BYTES and '"brk":3' in out
    huge = {"chars": 1, "pad": "z" * 300}
    assert shape_json(huge, None) is None


def test_break_index_only_on_fork_or_rewrite() -> None:
    assert break_index(None) is None
    assert break_index(Observation("continue", "p", False, "s", exact_depth=4)) is None
    assert break_index(Observation("fork", "p", False, "s", exact_depth=1)) == 2
    assert break_index(Observation("continue", "p", True, "s", exact_depth=2)) == 3


def test_toolset_hashes_names_and_caps_at_128() -> None:
    ts = toolset_of([_fn(f"t{i}") for i in range(130)], [f"x{i}" for i in range(130)], "sec")
    assert ts is not None
    assert len(ts.entries) == 128 and ts.truncated
    assert ts.entries[0][0] == hkey("sec", "t0", 8) and ts.entries[0][1] == "other"
    assert ts.entries[0][2] == 2
    assert all(e[0] != "t0" for e in ts.entries) and ts.names[0] == "t0"


def test_toolset_reads_anthropic_style_name() -> None:
    ts = toolset_of([{"name": "Task", "input_schema": {}}], ["abc"], "s")
    assert ts is not None and ts.entries[0][1] == "agent" and ts.names == ("Task",)


def test_toolset_never_keeps_descriptions_or_schemas() -> None:
    ts = toolset_of([_fn("Read", desc="SECRET-DESC", params={"SECRET_PARAM": {}})], ["..."], "s")
    assert "SECRET" not in repr(ts)


def test_toolset_digest_is_stable_and_order_sensitive() -> None:
    a = toolset_of([_fn("A"), _fn("B")], ["1", "22"], "s")
    b = toolset_of([_fn("A"), _fn("B")], ["1", "22"], "s")
    c = toolset_of([_fn("B"), _fn("A")], ["22", "1"], "s")
    assert a is not None and b is not None and c is not None
    assert a.digest == b.digest != c.digest and len(a.digest) == 32


def test_toolset_none_without_tools() -> None:
    assert toolset_of(None, [], "s") is None
    assert toolset_of([], [], "s") is None


def test_dispatch_agent_is_agent_family() -> None:
    assert tool_family("dispatch_agent") == "agent"


# -- sessions redesign: dead weight -----------------------------------------

from app.cobs2 import canonical as _canonical  # noqa: E402
from app.cobs2 import shape as _shape  # noqa: E402
from app.cobs2.ids import ids_for  # noqa: E402
from app.cobs2.keys import NetworkKey  # noqa: E402
from app.cobs2.shape import dead_weight_of  # noqa: E402
from tests.conftest import cpu_seconds  # noqa: E402

_K = NetworkKey(epoch=1, key=b"d" * 32)


def _sysm(t):
    return {"role": "system", "content": t}


def _u(t):
    return {"role": "user", "content": t}


def _tool(cid, t):
    return {"role": "tool", "tool_call_id": cid, "content": t}


def _dw(body):
    c = canonicalize(body)
    assert c is not None
    return c, ids_for(c, _K, b"")


def test_dead_weight_counts_dup_boil_reason():
    body = {
        "messages": [
            _sysm("S\n2026-10-27"),
            _u("<system-reminder>r</system-reminder>hi"),
            {"role": "assistant", "content": "ok", "reasoning_content": "think" * 10},
            _tool("t1", "FILE"),
            _tool("t1", "FILE"),
            _u("q"),
        ]
    }
    c, ids = _dw(body)
    assert dead_weight_of(c, ids) == {
        "dup": len("FILE"),
        "boil": len("2026-10-27") + len("<system-reminder>r</system-reminder>"),
        "reason": 50,
    }


def test_dead_weight_counts_a_duplicate_once_and_not_its_boilerplate_again():
    m = _u("<system-reminder>rr</system-reminder>same")
    c, ids = _dw({"messages": [m, _u("x"), m]})
    dw = dead_weight_of(c, ids)
    assert dw["dup"] == len("<system-reminder>rr</system-reminder>same")
    assert dw["boil"] == len("<system-reminder>rr</system-reminder>")  # first occurrence only


def test_dead_weight_memo_scans_only_new_messages(monkeypatch):
    calls: list[str] = []
    real = _canonical.boilerplate_chars

    def spy(v):
        calls.append(v)
        return real(v)

    monkeypatch.setattr(_canonical, "boilerplate_chars", spy)
    base = [_sysm("memo test S"), _u("memo one"), {"role": "assistant", "content": "memo two"}]
    c, ids = _dw({"messages": base})
    dead_weight_of(c, ids)
    calls.clear()
    c2, ids2 = _dw({"messages": [*base, _u("memo three")]})
    dead_weight_of(c2, ids2)
    assert calls == ["memo three"]


def test_shape_json_cap_is_256_and_drops_dw_first():
    c = canonicalize({"messages": [_u("hi")]})
    assert c is not None
    big = {
        **shape_of(c),
        "chars": 10**12,
        "sys": 10**12,
        "tools_chars": 10**12,
        "dw": {"dup": 10**30, "boil": 10**30, "reason": 10**30},
    }
    out = shape_json(big, 3)
    assert out is not None and '"dw"' not in out and len(out.encode()) <= 256
    small = shape_json({**shape_of(c), "dw": {"dup": 1, "boil": 2, "reason": 3}}, None)
    assert small is not None and '"dw"' in small
    assert SHAPE_MAX_BYTES == 256


def _big_body(n):
    msgs = [_sysm("You are an agent.\n2026-10-27\n" + "rules " * 2000)]
    for i in range(n):
        if i % 2 == 0:
            extra = "<system-reminder>todo list changed</system-reminder>" if i % 10 == 0 else ""
            msgs.append(_u(extra + f"step {i} " + "w" * 3500))
        else:
            msgs.append({"role": "assistant", "content": f"done {i} " + "z" * 3500})
    return {"messages": msgs}


@pytest.mark.perf
def test_dead_weight_hot_path_cost():
    c, ids = _dw(_big_body(400))
    # thread CPU time, not wall: a loaded CI runner stretches wall time, not the work.
    # Each cold round starts from an empty memo; the last one leaves it filled for the warm pass.
    cold = cpu_seconds(lambda: dead_weight_of(c, ids), reset=_shape._DW_MEMO.clear)
    body2 = _big_body(400)
    body2["messages"].append(_u("one more"))
    c2, ids2 = _dw(body2)
    # One round each: a second round would hit the memo the first one filled and
    # time the fully cached path, hiding a slow incremental (one new message) path.
    # CPU time already ignores a busy runner, so best-of-N buys nothing here.
    warm_ids = cpu_seconds(lambda: ids_for(c2, _K, b""), rounds=1)
    warm = cpu_seconds(lambda: dead_weight_of(c2, ids2), rounds=1)
    assert cold < 0.1, cold
    assert warm < 0.005, warm
    assert warm <= 0.25 * warm_ids or warm < 0.0005, (warm, warm_ids)
