import json

import pytest

from app.chat2.events import StreamState, frame, map_finish_reason, normalize, split_sse


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def _chunk(delta, finish=None, usage=None, id_="chatcmpl-1"):
    o = {"id": id_, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        o["usage"] = usage
    return _sse(o)


async def _gen(*chunks: bytes):
    for c in chunks:
        yield c


def test_split_sse_handles_partial_frames() -> None:
    events, left = split_sse(b"data: a\n\ndata: b\n\nda")
    assert events == ["a", "b"] and left == b"da"


def test_map_finish_reason() -> None:
    assert map_finish_reason("stop") == "stop"
    assert map_finish_reason("runaway_think") == "guard"
    assert map_finish_reason(None) == "stop"


@pytest.mark.asyncio
async def test_text_reasoning_usage_and_parts_order() -> None:
    st = StreamState()
    evs = [
        e
        async for e in normalize(
            _gen(
                _chunk({"reasoning_content": "think"}),
                _chunk({"content": "Hel"}),
                _chunk({"content": "lo"}),
                _chunk({}, finish="stop", usage={"prompt_tokens": 5, "completion_tokens": 2}),
                b"data: [DONE]\n\n",
            ),
            st,
            set(),
        )
    ]
    assert [e["type"] for e in evs] == ["reasoning-delta", "text-delta", "text-delta", "usage"]
    assert st.text == "Hello" and st.reasoning == "think" and st.finish_reason == "stop"
    assert st.usage and st.usage.prompt == 5 and st.upstream_id == "chatcmpl-1"
    assert [p["type"] for p in st.parts(set())] == ["reasoning", "text"]


@pytest.mark.asyncio
async def test_tool_calls_reassembled_by_index_and_options_projected() -> None:
    st = StreamState()
    evs = [
        e
        async for e in normalize(
            _gen(
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "present_options", "arguments": '{"opt'},
                            }
                        ]
                    }
                ),
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": 'ions":[{"label":"A","value":"a"},{"label":"B","value":"b"}]}'
                                },
                            }
                        ]
                    }
                ),
                _chunk({}, finish="tool_calls"),
                b"data: [DONE]\n\n",
            ),
            st,
            {"present_options"},
        )
    ]
    types = [e["type"] for e in evs]
    assert types[:2] == ["tool-call-delta", "tool-call-delta"] and types[-1] == "tool-call"
    final = evs[-1]
    assert (
        final["id"] == "call_1"
        and final["client"] is True
        and final["args"]["options"][0]["value"] == "a"
    )
    parts = st.parts({"present_options"})
    assert [p["type"] for p in parts] == ["tool_call", "options"]
    assert parts[1]["callId"] == "call_1" and parts[1]["multi"] is False


@pytest.mark.asyncio
async def test_reasoning_field_alias_and_frame() -> None:
    st = StreamState()
    evs = [e async for e in normalize(_gen(_chunk({"reasoning": "r"})), st, set())]
    assert evs[0] == {"type": "reasoning-delta", "text": "r"}
    assert frame({"type": "done"}) == b'data: {"type": "done"}\n\n'


def _split_mid_separators(raw: bytes) -> list[bytes]:
    """Cut `raw` so every `\\r\\n\\r\\n` boundary lands mid-separator (after
    the first `\\r\\n`, before the second) across two chunks."""
    sep = b"\r\n\r\n"
    chunks: list[bytes] = []
    pos = 0
    search_from = 0
    while True:
        idx = raw.find(sep, search_from)
        if idx == -1:
            break
        cut = idx + 2
        chunks.append(raw[pos:cut])
        pos = cut
        search_from = idx + len(sep)
    chunks.append(raw[pos:])
    return chunks


def test_split_sse_handles_crlf_separators() -> None:
    events, left = split_sse(b"data: a\r\n\r\ndata: b\r\n\r\nda")
    assert events == ["a", "b"] and left == b"da"


@pytest.mark.asyncio
async def test_crlf_framed_stream_split_across_chunks_matches_lf_case() -> None:
    def _sse_crlf(obj: dict) -> bytes:
        return f"data: {json.dumps(obj)}\r\n\r\n".encode()

    def _chunk_crlf(delta, finish=None, usage=None, id_="chatcmpl-1"):
        o = {"id": id_, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage is not None:
            o["usage"] = usage
        return _sse_crlf(o)

    raw = b"".join(
        [
            _chunk_crlf({"reasoning_content": "think"}),
            _chunk_crlf({"content": "Hel"}),
            _chunk_crlf({"content": "lo"}),
            _chunk_crlf({}, finish="stop", usage={"prompt_tokens": 5, "completion_tokens": 2}),
            b"data: [DONE]\r\n\r\n",
        ]
    )
    chunks = _split_mid_separators(raw)
    assert len(chunks) > 5  # sanity: every separator really did get split

    st = StreamState()
    evs = [e async for e in normalize(_gen(*chunks), st, set())]
    assert [e["type"] for e in evs] == ["reasoning-delta", "text-delta", "text-delta", "usage"]
    assert st.text == "Hello" and st.reasoning == "think" and st.finish_reason == "stop"
    assert st.usage and st.usage.prompt == 5 and st.upstream_id == "chatcmpl-1"
    assert [p["type"] for p in st.parts(set())] == ["reasoning", "text"]
