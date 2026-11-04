import json

from app.cobs2.canonical import canonicalize
from app.cobs2.ids import ids_for
from app.cobs2.keys import NetworkKey
from app.cobs2.reply import MAX_REPLY_BYTES, ReplyCapture

KEY = NetworkKey(epoch=1, key=b"k" * 32)


def _sse(delta, fr=None):
    return (
        b"data: "
        + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": fr}]}).encode()
    )


def _ids(msg):
    c = canonicalize({"messages": [msg]})
    assert c is not None
    return ids_for(c, KEY, b"")


def test_capture_stream_text_and_tool_calls():
    cap = ReplyCapture()
    for d in (
        {"role": "assistant", "content": "he"},
        {"content": "llo"},
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"a"'},
                }
            ]
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]},
    ):
        cap.feed_sse(_sse(d))
    cap.feed_sse(b"data: [DONE]")
    m = cap.message()
    assert m["role"] == "assistant" and m["content"] == "hello"
    assert m["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"a": 1}'}


def test_capture_nonstream():
    cap = ReplyCapture()
    cap.set_json({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
    assert cap.message() == {"role": "assistant", "content": "ok"}


def test_capture_empty_is_none():
    assert ReplyCapture().message() is None


def test_capture_bounded():
    cap = ReplyCapture()
    cap.feed_sse(_sse({"content": "x" * (MAX_REPLY_BYTES + 1)}))
    assert cap.message() is None
    cap = ReplyCapture()
    cap.set_json(
        {"choices": [{"message": {"role": "assistant", "content": "x" * (MAX_REPLY_BYTES + 1)}}]}
    )
    assert cap.message() is None


def test_reasoning_field_variants_hash_alike():
    for field in ("reasoning_content", "reasoning"):
        cap = ReplyCapture()
        cap.feed_sse(_sse({"role": "assistant", field: "think"}))
        cap.feed_sse(_sse({"content": "ans"}))
        m = cap.message()
        assert m["reasoning_content"] == "think"
        client = {"role": "assistant", "content": "ans", "reasoning_content": "think"}
        assert _ids(m).msgs == _ids(client).msgs
        dropped = {"role": "assistant", "content": "ans"}
        assert _ids(m).loose_msgs == _ids(dropped).loose_msgs


def test_reply_id_matches_followup_request_position():
    cap = ReplyCapture()
    cap.feed_sse(_sse({"role": "assistant", "content": "a"}))
    cap.feed_sse(
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"x":1,  "y":2}'},
                    }
                ]
            }
        )
    )
    reply = cap.message()
    follow = canonicalize(
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "a",
                    "tool_calls": [
                        {
                            "id": "zz",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"x":1,  "y":2}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "zz", "content": "r"},
            ]
        }
    )
    assert _ids(reply).msgs[0] == ids_for(follow, KEY, b"").msgs[1]


def test_stream_bound_counts_payload_not_envelope():
    cap = ReplyCapture()
    # many tiny deltas: wire size well over 4 MiB, payload (1 char each) far under
    frame = _sse({"content": "x"}) + b"\n\n"
    n = (MAX_REPLY_BYTES // len(frame)) + 1000
    for _ in range(n):
        cap.feed_sse(frame)
    cap.feed_sse(_sse({}, fr="stop"))
    assert cap.complete
    m = cap.message()
    assert m is not None and len(m["content"]) == n


def test_incomplete_until_finish_reason():
    cap = ReplyCapture()
    cap.feed_sse(_sse({"content": "a"}))
    assert not cap.complete
    cap.feed_sse(_sse({}, fr="stop"))
    assert cap.complete
