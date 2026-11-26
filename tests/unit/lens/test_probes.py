"""Prompt Lens probes: text-free digests and span classes (spec 4.3-4.4)."""

import os

import pytest

from app.cobs2.keys import NetworkKey
from app.lens.probes import (
    PROBE_ORDER,
    SPAN_CLASSES,
    VOLATILE_CLASSES,
    args8,
    content_hash,
    first_changed_line,
    message_probes,
    span_class,
    sys_line_digests,
    tool_def_hashes,
)

NOISE, NO_REASONING, TEXT_ONLY, WS = (PROBE_ORDER.index(p) for p in PROBE_ORDER)


@pytest.fixture
def key() -> NetworkKey:
    return NetworkKey(epoch=1, key=os.urandom(32))


def _img(url: str = "data:image/png;base64,iVBORw0KGgo=") -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def test_probe_order_and_shape(key):
    assert PROBE_ORDER == ("noise", "no_reasoning", "text_only", "ws")
    p = message_probes({"role": "user", "content": "hi"}, key, b"s")
    assert len(p) == 4 and all(isinstance(x, bytes) and len(x) == 8 for x in p)
    assert p == message_probes({"role": "user", "content": "hi"}, key, b"s")
    assert p != message_probes({"role": "user", "content": "hi"}, key, b"other")


def test_line_ending_only_change_is_equal_under_ws_probe(key):
    a = message_probes({"role": "user", "content": "a\r\nb"}, key, b"s")
    b = message_probes({"role": "user", "content": "a\nb"}, key, b"s")
    assert a[PROBE_ORDER.index("ws")] == b[PROBE_ORDER.index("ws")]
    assert a != b


def test_reasoning_dropped_equalises_no_reasoning_probe(key):
    with_r = {"role": "assistant", "content": "ok", "reasoning_content": "thinking hard"}
    without = {"role": "assistant", "content": "ok"}
    a = message_probes(with_r, key, b"s")
    b = message_probes(without, key, b"s")
    assert a[NO_REASONING] == b[NO_REASONING]
    assert a[NOISE] == b[NOISE]
    assert a[TEXT_ONLY] != b[TEXT_ONLY]
    assert a[WS] != b[WS]


def test_media_moved_equalises_text_only_probe(key):
    t = {"type": "text", "text": "look"}
    removed = message_probes({"role": "user", "content": [t]}, key, b"s")
    first = message_probes({"role": "user", "content": [_img(), t]}, key, b"s")
    last = message_probes({"role": "user", "content": [t, _img()]}, key, b"s")
    assert removed[TEXT_ONLY] == first[TEXT_ONLY] == last[TEXT_ONLY]
    assert first[NO_REASONING] != last[NO_REASONING]
    assert first[WS] != removed[WS]


def test_noise_probe_ignores_volatile_boilerplate(key):
    a = message_probes(
        {"role": "user", "content": "x <system-reminder>one</system-reminder> y"}, key, b"s"
    )
    b = message_probes(
        {"role": "user", "content": "x <system-reminder>two</system-reminder> y"}, key, b"s"
    )
    assert a[NOISE] == b[NOISE]
    assert a[NO_REASONING] != b[NO_REASONING]


def test_line_digests_are_bounded(key):
    d = sys_line_digests("x\n" * 5000, key, b"s")
    # The lines past 1,024 are joined with "\n" (no line endings kept) and cut
    # into 4,096-char chunks: 3,976 lines are 7,951 chars, 2 chunks.
    rest = "\n".join(["x"] * (5000 - 1024))
    assert len(rest) == 7951
    assert len(d) == 1024 + (len(rest) + 4095) // 4096 == 1026
    assert all(len(h) == 8 for h in d)
    assert len(sys_line_digests("y" * 1_000_000, key, b"s")) == 1
    assert sys_line_digests("", key, b"s") == ()


def test_line_digests_are_per_line_and_keyed(key):
    a = sys_line_digests("one\ntwo\nthree\n", key, b"s")
    b = sys_line_digests("one\nTWO\nthree\n", key, b"s")
    assert len(a) == 3 and a[0] == b[0] and a[1] != b[1] and a[2] == b[2]
    assert a != sys_line_digests("one\ntwo\nthree\n", key, b"t")
    assert first_changed_line(a, b) == (2, 1)


def test_line_ending_only_change_locates_no_line(key):
    """M1: a line is digested without its ending, so a CRLF/LF flip (or a
    trailing newline) changes no digest and locates no line."""
    lf = "You help.\nline two\nline three\n"
    crlf = lf.replace("\n", "\r\n")
    assert sys_line_digests(crlf, key, b"s") == sys_line_digests(lf, key, b"s")
    assert sys_line_digests(lf.rstrip("\n"), key, b"s") == sys_line_digests(lf, key, b"s")
    assert (
        first_changed_line(sys_line_digests(crlf, key, b"s"), sys_line_digests(lf, key, b"s"))
        is None
    )
    # Past line 1,024 too.
    big = "x\n" * 1100 + "tail\n"
    assert sys_line_digests(big.replace("\n", "\r\n"), key, b"s") == sys_line_digests(
        big, key, b"s"
    )


def test_first_changed_line():
    assert first_changed_line([b"a", b"b", b"c"], [b"a", b"X", b"c"]) == (2, 1)
    assert first_changed_line([b"a"], [b"a"]) is None
    assert first_changed_line([b"a", b"b", b"c"], [b"a", b"X", b"Y"]) == (2, 2)
    assert first_changed_line([b"a"], [b"a", b"b"]) == (2, 1)
    assert first_changed_line([], []) is None


@pytest.mark.parametrize(
    "line,cls",
    [
        ("Today is 2026-11-23", "date"),
        ("Now: 14:02:11", "time"),
        ("Working directory: /home/u/wt/lens-3", "path"),
        ("run 3f2a9c1e-77b0-4c1e-9a55-0d6f1c2b8e10", "uuid_or_hex_id"),
        ("cc_version=2.1.3.a91", "version"),
        ("- Provider ID: lmwarden-chat", "provider_or_model_id"),
        ("Contact: someone@example.com", "user_name_or_email"),
        ("See https://example.com/x", "url"),
        ("Be concise.", "other"),
        ("Retry 42 times", "number"),
    ],
)
def test_span_class(line, cls):
    assert span_class(line) == cls


def test_span_classes_contract():
    assert SPAN_CLASSES[-1] == "other"
    assert VOLATILE_CLASSES == {
        "date",
        "time",
        "path",
        "uuid_or_hex_id",
        "version",
        "provider_or_model_id",
    }
    assert VOLATILE_CLASSES <= set(SPAN_CLASSES)


def test_tool_hashes_detect_reorder_as_same_multiset(key):
    tools = ["tool-a", "tool-b", "tool-c"]
    a = tool_def_hashes(tools, key, b"s")
    b = tool_def_hashes(["tool-c", "tool-a", "tool-b"], key, b"s")
    assert len(a) == 3 and all(len(h) == 8 for h in a)
    assert a != b and sorted(a) == sorted(b)
    assert sorted(a) != sorted(tool_def_hashes(["tool-a", "tool-b", "tool-X"], key, b"s"))
    assert tool_def_hashes([], key, b"s") == ()


def test_content_hash_ignores_tool_call_id(key):
    a = content_hash("result body", key, b"s")
    assert len(a) == 8
    msg = {"role": "tool", "tool_call_id": "call_1", "content": "result body"}
    assert content_hash(msg["content"], key, b"s") == a
    assert content_hash("other body", key, b"s") != a
    assert content_hash([{"type": "text", "text": "result body"}], key, b"s") == a


def test_args8_is_key_order_independent():
    assert args8('{"a":1,"b":2}', "s") == args8('{"b":2,"a":1}', "s") and len(args8("{}", "s")) == 8


def test_args8_details():
    assert args8('{"a": 1}', "s") != args8('{"a": 2}', "s")
    assert args8('{"a":1}', "s") != args8('{"a":1}', "t")
    raw = args8("not json", "s")
    assert len(raw) == 8 and raw == args8("not json", "s")
    int(args8("{}", "s"), 16)
    assert args8(None, "s") == "" == args8("", "s")


def test_result_hash_is_content_hash_over_the_canonical_parts(key):
    from app.cobs2.canonical import _content_parts
    from app.lens.probes import result_hash

    for c in ("x", [{"type": "text", "text": "x"}], [], "CANARY-1d4b7e-result"):
        assert result_hash(_content_parts(c), key, b"s") == content_hash(c, key, b"s")
    assert content_hash("x", key, b"s") == content_hash([{"type": "text", "text": "x"}], key, b"s")
