"""G9: the bounded whole-section store (app/proxy/prompt_lens_sections.py).

The hub feeds it every published event; it assembles each request's thinking,
reply and tool calls (by index) per channel, holds the wide structured prompt
capture beside them, and serves one section on expand. Bounded by a per-channel
cap (oldest dropped and counted), a per-key and a total char budget, a request
count, and a TTL after ``request_end``.
"""

from __future__ import annotations

import tracemalloc
from typing import Any

from app.proxy.prompt_lens import PromptLensHub
from app.proxy.prompt_lens_capture import structured_prompt
from app.proxy.prompt_lens_sections import PromptLensSections, kept_view

BODY_WITH_TOOL_RESULT = {
    "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "read the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"p": "a.txt"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file text " * 2000},
        {"role": "user", "content": "thanks"},
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
}


def _start(r: str, t: str = "tok1", seq: int = 1) -> dict[str, Any]:
    return {"type": "request_start", "req_id": r, "token_id": t, "seq": seq, "request_id": f"q-{r}"}


def _d(r: str, ch: str, text: str, seq: int, t: str = "tok1", **kw: Any) -> dict[str, Any]:
    return {
        "type": "delta",
        "req_id": r,
        "token_id": t,
        "channel": ch,
        "text": text,
        "seq": seq,
        **kw,
    }


def _end(r: str, seq: int, t: str = "tok1", **kw: Any) -> dict[str, Any]:
    return {"type": "request_end", "req_id": r, "token_id": t, "seq": seq, "ts": 100.0, **kw}


def test_assembles_channels() -> None:
    s = PromptLensSections()
    s.feed(_start("r"))
    s.feed(_d("r", "reasoning", "ab", 2))
    s.feed(_d("r", "content", "hi", 3))
    s.feed(_d("r", "reasoning", "c", 4))
    v = s.section("r", "thinking")
    assert v is not None
    assert (v.text, v.seq, v.live, v.token_id, v.offset) == ("abc", 4, True, "tok1", 0)
    r = s.section("r", "reply")
    assert r is not None and r.text == "hi"


def test_interleaved_tool_calls_by_index() -> None:
    s = PromptLensSections()
    s.feed(_start("r"))
    s.feed(_d("r", "tool_call", "", 2, index=0, call_id="c0", name="bash"))
    s.feed(_d("r", "tool_call", "", 3, index=1, call_id="c1", name="grep"))
    s.feed(_d("r", "tool_call", '{"a"', 4, index=0))
    s.feed(_d("r", "tool_call", '{"b"', 5, index=1))
    v = s.section("r", "tool_calls")
    assert v is not None
    assert v.calls == [
        {"index": 0, "call_id": "c0", "name": "bash", "args": '{"a"'},
        {"index": 1, "call_id": "c1", "name": "grep", "args": '{"b"'},
    ]


def test_tool_call_elided_and_publisher_drop_counters() -> None:
    s = PromptLensSections()
    s.feed(_start("r"))
    s.feed(_d("r", "tool_call", "{", 2, index=0, call_id="c0", name="bash"))
    s.feed(_d("r", "tool_call", "xy", 3, index=0, elided=True))
    s.feed(_end("r", 4, tool_call_chars_dropped=40, tool_calls_dropped=2))
    v = s.section("r", "tool_calls")
    assert v is not None and v.live is False
    assert v.calls == [{"index": 0, "call_id": "c0", "name": "bash", "args": "{xy", "elided": True}]
    assert (v.dropped_chars, v.dropped_calls) == (40, 2)


def test_tool_call_channel_cap_drops_the_tail_and_counts() -> None:
    s = PromptLensSections(max_channel_chars=10)
    s.feed(_start("r"))
    s.feed(_d("r", "tool_call", "123", 2, index=0, call_id="c", name="n"))  # 5 chars
    s.feed(_d("r", "tool_call", "4567890", 3, index=0))  # 5 fit, 2 dropped
    s.feed(_d("r", "tool_call", "", 4, index=1, call_id="d", name="m"))  # whole call dropped
    v = s.section("r", "tool_calls")
    assert v is not None
    assert v.calls == [
        {"index": 0, "call_id": "c", "name": "n", "args": "12345678", "elided": True}
    ]
    assert v.dropped_chars == 2 and v.dropped_calls == 1


def test_channel_cap_drops_oldest_and_counts() -> None:
    s = PromptLensSections(max_channel_chars=10)
    s.feed(_start("r"))
    s.feed(_d("r", "content", "x" * 8, 2))
    s.feed(_d("r", "content", "y" * 8, 3))
    v = s.section("r", "reply")
    assert v is not None
    assert v.text == "xxyyyyyyyy" and v.dropped_chars == 6 and v.offset == 6


def test_from_offset_is_absolute_and_never_below_dropped() -> None:
    s = PromptLensSections(max_channel_chars=10)
    s.feed(_start("r"))
    s.feed(_d("r", "content", "abcdefgh", 2))
    s.feed(_d("r", "content", "ijklmnop", 3))  # kept: "ghijklmnop" from offset 6
    v = s.section("r", "reply", start=12)
    assert v is not None and (v.offset, v.text) == (12, "mnop")
    v = s.section("r", "reply", start=2)
    assert v is not None and (v.offset, v.text) == (6, "ghijklmnop")
    v = s.section("r", "reply", start=99)
    assert v is not None and (v.offset, v.text) == (16, "")


def test_total_budget_evicts_oldest_request() -> None:
    s = PromptLensSections(total_chars=20)
    s.feed(_start("a"))
    s.feed(_d("a", "content", "a" * 15, 2))
    s.feed(_start("b"))
    s.feed(_d("b", "content", "b" * 15, 4))
    b = s.section("b", "reply")
    assert s.section("a", "reply") is None and b is not None and b.text == "b" * 15
    assert s.size_chars() <= 20


def test_max_requests_bound() -> None:
    s = PromptLensSections(max_requests=2)
    for r in "abc":
        s.feed(_start(r))
    assert s.section("a", "reply") is None
    assert s.section("c", "reply") is not None


def test_newest_always_kept_even_alone_over_budget() -> None:
    s = PromptLensSections(total_chars=5)
    s.feed(_start("r"))
    s.feed(_d("r", "content", "z" * 50, 2))
    v = s.section("r", "reply")
    assert v is not None and v.text == "z" * 50


def test_a_finished_request_goes_before_a_live_older_one() -> None:
    s = PromptLensSections(total_chars=30, key_chars=30)
    s.feed(_start("live"))
    s.feed(_d("live", "content", "L" * 10, 2))
    s.feed(_start("done"))
    s.feed(_d("done", "content", "D" * 10, 4))
    s.feed(_end("done", 5))
    s.feed(_start("new"))
    s.feed(_d("new", "content", "N" * 15, 7))
    assert s.section("done", "reply") is None
    assert s.section("live", "reply") is not None and s.section("new", "reply") is not None


def test_one_keys_traffic_never_evicts_another_keys_sections() -> None:
    """Per-key budgets (isolation): key B's volume cannot push key A's section
    out while both stay under their own budget and the backstop."""
    s = PromptLensSections(total_chars=1000, key_chars=100, max_requests=512, key_requests=4)
    s.feed(_start("a1", t="A"))
    s.feed(_d("a1", "content", "a" * 50, 2, t="A"))
    for n in range(40):
        rid = f"b{n}"
        s.feed(_start(rid, t="B"))
        s.feed(_d(rid, "content", "b" * 90, 3, t="B"))
        s.feed(_end(rid, 4, t="B"))
    a = s.section("a1", "reply")
    assert a is not None and a.text == "a" * 50
    assert s.size_chars() <= 1000
    assert s.section("b0", "reply") is None and s.section("b39", "reply") is not None


def test_released_ttl_after_end() -> None:
    now = [0.0]
    s = PromptLensSections(ttl_s=600, clock=lambda: now[0])
    s.feed(_start("r"))
    s.feed(_end("r", 2))
    now[0] = 599
    assert s.section("r", "reply") is not None
    now[0] = 601
    assert s.section("r", "reply") is None
    assert s.size_chars() == 0


def test_prompt_and_tool_results_from_wide_capture() -> None:
    s = PromptLensSections()
    s.feed(_start("r"))
    parts = structured_prompt(BODY_WITH_TOOL_RESULT, budget_chars=256_000, msg_chars=256_000)
    s.put_prompt("r", "tok1", parts)
    tr = s.section("r", "tool_results")
    assert tr is not None and tr.messages is not None
    assert [m["role"] for m in tr.messages] == ["tool"]
    assert tr.messages[0]["text"] == "file text " * 2000  # whole, not the 6000-char window
    p = s.section("r", "prompt")
    assert p is not None and p.messages is not None
    assert [m["role"] for m in p.messages] == ["sys", "tools", "user", "assistant", "tool", "user"]
    assert p.older is not None and p.older["collapsed"] == 0
    assert s.size_chars() >= 20_000  # the wide prompt counts against the budget


def test_put_prompt_for_another_key_or_unknown_request_is_ignored() -> None:
    s = PromptLensSections()
    s.feed(_start("r"))
    s.put_prompt("r", "tok2", {"messages": [{"role": "tool", "text": "x"}]})
    s.put_prompt("nope", "tok1", {"messages": []})
    v = s.section("r", "tool_results")
    assert v is not None and v.messages == []


def test_unknown_request_and_events_without_start_are_ignored() -> None:
    s = PromptLensSections()
    s.feed(_d("ghost", "content", "boo", 2))
    assert s.section("ghost", "reply") is None
    assert s.section("ghost", "nope") is None  # type: ignore[arg-type]


def test_progress_is_ignored() -> None:
    s = PromptLensSections()
    s.feed(_start("r"))
    s.feed({"type": "progress", "req_id": "r", "token_id": "tok1", "seq": 9, "frames": {}})
    v = s.section("r", "reply")
    assert v is not None and v.seq == 1


def test_feed_never_raises() -> None:
    s = PromptLensSections()
    s.feed({"garbage": object()})
    s.feed({"type": "request_start", "req_id": object()})
    s.feed(None)  # type: ignore[arg-type]
    s.feed(_start("r"))
    s.feed(_d("r", "content", None, 2))  # type: ignore[arg-type]
    s.feed(_d("r", "tool_call", "x", 3, index="nope"))
    assert s.section("r", "reply") is not None


def test_keep_hands_one_record_at_request_end() -> None:
    got: list[dict[str, Any]] = []
    s = PromptLensSections(on_kept=got.append)
    start = {**_start("r"), "prompt": "p", "client_ip": "10.0.0.1"}
    s.feed({**start, "prompt_parts": {"messages": []}})  # not kept: the wide one is
    s.put_prompt("r", "tok1", {"messages": [{"role": "tool", "text": "T"}]}, keep=True)
    kv = {"type": "kvsig", "req_id": "r", "token_id": "tok1", "seq": 2, "parts": []}
    s.feed(kv)
    s.feed(_d("r", "reasoning", "think", 3))
    s.feed(_d("r", "content", "reply", 4))
    s.feed(_end("r", 5, finish_reason="stop"))
    assert len(got) == 1
    rec = got[0]
    assert (rec["req_id"], rec["token_id"], rec["request_id"]) == ("r", "tok1", "q-r")
    assert rec["start"] == start and rec["kvsig"] is kv and rec["end"]["finish_reason"] == "stop"
    assert rec["keep_gen"] == 1
    assert rec["sections"]["thinking"] == {"text": "think", "dropped_chars": 0}
    assert rec["sections"]["reply"] == {"text": "reply", "dropped_chars": 0}
    v = kept_view(rec, "tool_results")
    assert v is not None and v.kept and not v.live and v.seq == 5
    assert v.messages == [{"role": "tool", "text": "T"}]
    tv = kept_view(rec, "reply", start=2)
    assert tv is not None and (tv.offset, tv.text) == (2, "ply")


def test_no_record_without_keep() -> None:
    got: list[dict[str, Any]] = []
    s = PromptLensSections(on_kept=got.append)
    s.feed(_start("r"))
    s.put_prompt("r", "tok1", None)
    s.feed(_end("r", 2))
    assert got == []


def test_a_failing_on_kept_never_raises() -> None:
    def boom(_rec: dict[str, Any]) -> None:
        raise RuntimeError("x")

    s = PromptLensSections(on_kept=boom)
    s.feed(_start("r"))
    s.put_prompt("r", "tok1", None, keep=True)
    s.feed(_end("r", 2))
    v = s.section("r", "reply")
    assert v is not None and v.live is False


# -- the hub hook --------------------------------------------------------


def test_hub_feeds_every_event_after_seq_kept_or_not() -> None:
    s = PromptLensSections()
    hub = PromptLensHub(sections=s)
    hub.publish(_start("r"))
    hub.publish(_d("r", "content", "hi", 0))
    seq = hub.publish(
        {"type": "progress", "req_id": "r", "token_id": "tok1", "frames": {}}, keep=False
    )
    hub.publish(_d("r", "content", "!", 0), keep=False)  # a live-only delta still folds in
    v = s.section("r", "reply")
    assert v is not None and v.text == "hi!" and v.seq == seq + 1


def test_hub_survives_a_failing_store() -> None:
    class Bad:
        def feed(self, _e: dict[str, Any]) -> None:
            raise RuntimeError("x")

    hub = PromptLensHub(sections=Bad())  # type: ignore[arg-type]
    assert hub.publish(_start("r")) == 1


def test_section_store_settings(monkeypatch: Any) -> None:
    from app.config import load_settings

    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    for k in ("CHARS", "MAX_CHARS", "TTL_S"):
        monkeypatch.delenv(f"VW_PROMPT_LENS_SECTION_{k}", raising=False)
    s = load_settings()
    assert (
        s.prompt_lens_section_chars,
        s.prompt_lens_section_max_chars,
        s.prompt_lens_section_ttl_s,
    ) == (
        16_000_000,
        2_000_000,
        600.0,
    )
    monkeypatch.setenv("VW_PROMPT_LENS_SECTION_CHARS", "100")
    monkeypatch.setenv("VW_PROMPT_LENS_SECTION_MAX_CHARS", "10")
    monkeypatch.setenv("VW_PROMPT_LENS_SECTION_TTL_S", "1.5")
    s = load_settings()
    assert (
        s.prompt_lens_section_chars,
        s.prompt_lens_section_max_chars,
        s.prompt_lens_section_ttl_s,
    ) == (
        100,
        10,
        1.5,
    )


async def test_the_app_builds_the_store_and_hands_it_to_the_hub(live_app: Any) -> None:
    sections = live_app.state.prompt_lens_sections
    assert isinstance(sections, PromptLensSections)
    live_app.state.prompt_lens_hub.publish(_start("r"))
    assert sections.section("r", "reply") is not None


# -- fix round 1 ------------------------------------------------------------


def test_backstop_never_takes_a_quiet_keys_section_for_others_volume() -> None:
    """Review C1, with the default bounds: A has one small request; then 8 keys
    send 64 requests each (the request count binds), then 4 keys send ~4M each
    (the chars bind). A keeps its section until its own TTL."""
    now = [0.0]
    s = PromptLensSections(clock=lambda: now[0])
    s.feed(_start("a", t="A"))
    s.feed(_d("a", "content", "mine", 2, t="A"))
    s.feed(_end("a", 3, t="A"))
    seq = 10
    for k in range(8):
        for n in range(64):
            rid = f"k{k}-{n}"
            s.feed(_start(rid, t=f"K{k}", seq=seq))
            s.feed(_d(rid, "content", "x" * 10, seq + 1, t=f"K{k}"))
            s.feed(_end(rid, seq + 2, t=f"K{k}"))
            seq += 3
            assert s.section("a", "reply") is not None, (k, n)
    for k in range(4):
        rid = f"big{k}"
        s.feed(_start(rid, t=f"H{k}", seq=seq))
        s.feed(_d(rid, "content", "r" * 2_000_000, seq + 1, t=f"H{k}"))
        s.feed(_d(rid, "reasoning", "t" * 1_999_000, seq + 2, t=f"H{k}"))
        seq += 3
        assert s.section("a", "reply") is not None, k
    assert s.size_chars() <= 16_000_000 + 4_000_000  # the backstop held (newest kept)
    assert len(s._reqs) <= 512
    now[0] = 601
    assert s.section("a", "reply") is None  # its own TTL


def test_backstop_takes_the_fed_keys_own_older_requests_first() -> None:
    s = PromptLensSections(total_chars=100, key_chars=100, max_requests=10, key_requests=10)
    s.feed(_start("a", t="A"))
    s.feed(_d("a", "content", "a" * 10, 2, t="A"))
    s.feed(_start("b1", t="B"))
    s.feed(_d("b1", "content", "b" * 45, 3, t="B"))
    s.feed(_start("b2", t="B"))
    s.feed(_d("b2", "content", "b" * 50, 4, t="B"))
    assert s.section("a", "reply") is not None and s.section("b1", "reply") is None


def test_real_heap_stays_near_the_counted_chars() -> None:
    """Review I2: a token-per-delta stream is coalesced, not one string per token."""
    s = PromptLensSections()
    s.feed(_start("r"))
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        for i in range(250_000):
            s.feed(_d("r", "content", "abcd", i + 2))
        for i in range(10_000):
            s.feed(
                _d(
                    "r",
                    "tool_call",
                    "efgh",
                    i,
                    index=0,
                    **({"call_id": "c", "name": "n"} if i == 0 else {}),
                )
            )
        held = tracemalloc.get_traced_memory()[0] - base
    finally:
        tracemalloc.stop()
    counted = s.size_chars()
    assert counted >= 1_000_000
    assert held <= 2 * counted, (held, counted)


def test_trimming_a_capped_channel_never_copies_the_whole_text() -> None:
    """Review M12: one big delta is stored in pieces; a read does not store a join."""
    s = PromptLensSections(max_channel_chars=1_000_000)
    s.feed(_start("r"))
    s.feed(_d("r", "content", "z" * 1_000_000, 2))
    assert s.section("r", "reply") is not None  # a read
    buf = s._reqs["r"].texts["reply"]
    for i in range(100):
        s.feed(_d("r", "content", "y", i + 3))
        assert max(len(c) for c in buf.chunks) <= 65536
    v = s.section("r", "reply")
    assert v is not None and len(v.text) == 1_000_000 and v.text.endswith("y" * 100)
    assert v.dropped_chars == 100 and v.offset == 100


def test_the_ttl_queue_is_bounded_by_the_request_bound() -> None:
    """Review M11: evicted requests' TTL entries do not pile up."""
    s = PromptLensSections(max_requests=4)
    for n in range(100):
        s.feed(_start(f"r{n}"))
        s.feed(_end(f"r{n}", 2))
    assert len(s._ends) <= 2 * 4
