"""Chat reply -> Responses reply: usage, items, errors, and the stream translator.

The stream scenarios are pinned as golden transcripts under
tests/fixtures/responses/ (ids and timestamps normalised). Regenerate with
``UPDATE_GOLDEN=1 pytest tests/unit/proxy/test_responses_stream.py`` and read
the diff: a golden is a contract with Codex's SSE parser.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from app.proxy.responses_schemas import ResponsesRequest
from app.proxy.responses_translate import (
    ResponsesStreamTranslator,
    Translation,
    error_body,
    finish_status,
    responses_usage,
    to_openai_request,
    to_response_object,
)

GOLDEN = Path(__file__).resolve().parents[2] / "fixtures" / "responses"


def translation(**body: Any) -> Translation:
    body.setdefault("model", "qwen")
    body.setdefault("input", "hi")
    return to_openai_request(ResponsesRequest.model_validate(body))


# ---- usage / status / errors ------------------------------------------------------------------


def test_usage_includes_cached_in_input_tokens():
    u = responses_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 64},
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
    )
    assert u == {
        "input_tokens": 100,
        "input_tokens_details": {"cached_tokens": 64},
        "output_tokens": 20,
        "output_tokens_details": {"reasoning_tokens": 5},
        "total_tokens": 120,
    }


def test_usage_defaults_and_cached_is_capped():
    assert responses_usage(None)["total_tokens"] == 0
    u = responses_usage(
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 99},
        }
    )
    assert u["input_tokens_details"]["cached_tokens"] == 10
    assert u["total_tokens"] == 12
    assert u["output_tokens_details"]["reasoning_tokens"] == 0


@pytest.mark.parametrize(
    "reason, want",
    [
        ("stop", ("completed", None)),
        ("tool_calls", ("completed", None)),
        (None, ("completed", None)),
        ("length", ("incomplete", {"reason": "max_output_tokens"})),
        ("runaway_repeat", ("incomplete", {"reason": "max_output_tokens"})),
        ("content_filter", ("incomplete", {"reason": "content_filter"})),
    ],
)
def test_finish_status(reason, want):
    assert finish_status(reason) == want


@pytest.mark.parametrize(
    "status, kind",
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "invalid_request_error"),
        (413, "invalid_request_error"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "server_error"),
        (503, "server_error"),
        (418, "invalid_request_error"),
    ],
)
def test_error_envelope(status, kind):
    assert error_body(status, "m") == {
        "error": {"message": "m", "type": kind, "param": None, "code": None}
    }


# ---- non-stream ---------------------------------------------------------------------------------


def _completion(message, finish="stop", usage=None):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


def test_text_reply():
    tr = translation()
    r = to_response_object(_completion({"role": "assistant", "content": "OK"}), model="qwen", tr=tr)
    assert r["object"] == "response" and r["status"] == "completed" and r["model"] == "qwen"
    assert r["id"].startswith("resp_") and r["store"] is False and r["error"] is None
    (item,) = r["output"]
    assert (
        item["type"] == "message" and item["id"].startswith("msg_") and item["role"] == "assistant"
    )
    assert item["content"] == [{"type": "output_text", "text": "OK", "annotations": []}]
    assert r["usage"]["input_tokens"] == 7 and r["usage"]["total_tokens"] == 10


def test_reasoning_only_when_asked_and_before_the_message():
    msg = {"role": "assistant", "content": "A", "reasoning_content": "think"}
    on = to_response_object(
        _completion(msg), model="q", tr=translation(reasoning={"effort": "low"})
    )
    assert [i["type"] for i in on["output"]] == ["reasoning", "message"]
    assert on["output"][0]["summary"] == [{"type": "summary_text", "text": "think"}]
    assert on["output"][0]["encrypted_content"] is None
    off = to_response_object(_completion(msg), model="q", tr=translation())
    assert [i["type"] for i in off["output"]] == ["message"]


def test_reasoning_field_variant():
    msg = {"role": "assistant", "content": "A", "reasoning": "think"}
    r = to_response_object(_completion(msg), model="q", tr=translation(reasoning={}))
    assert r["output"][0]["type"] == "reasoning"


def test_function_and_custom_calls():
    tr = translation(
        tools=[{"type": "function", "name": "shell"}, {"type": "custom", "name": "apply_patch"}]
    )
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_a",
                "type": "function",
                "function": {"name": "shell", "arguments": '{"c":"ls"}'},
            },
            {
                "id": "call_b",
                "type": "function",
                "function": {"name": "apply_patch", "arguments": '{"input":"*** x"}'},
            },
            {
                "id": None,
                "type": "function",
                "function": {"name": "shell", "arguments": "not json"},
            },
        ],
    }
    r = to_response_object(_completion(msg, "tool_calls"), model="q", tr=tr)
    a, b, c = r["output"]
    assert (a["type"], a["call_id"], a["arguments"], a["status"]) == (
        "function_call",
        "call_a",
        '{"c":"ls"}',
        "completed",
    )
    assert a["id"].startswith("fc_")
    assert (b["type"], b["call_id"], b["input"]) == ("custom_tool_call", "call_b", "*** x")
    assert b["id"].startswith("ctc_")
    assert c["call_id"].startswith("call_") and c["arguments"] == "not json"  # raw, not {}
    assert r["status"] == "completed"


def test_namespaced_call_carries_its_namespace():
    tr = translation(
        tools=[
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "tools": [{"type": "function", "name": "close_agent"}],
            }
        ]
    )
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_a",
                "type": "function",
                "function": {"name": "close_agent", "arguments": "{}"},
            }
        ],
    }
    r = to_response_object(_completion(msg, "tool_calls"), model="q", tr=tr)
    assert (
        r["output"][0]["namespace"] == "multi_agent_v1" and r["output"][0]["name"] == "close_agent"
    )


def test_length_is_incomplete_with_details():
    r = to_response_object(
        _completion({"role": "assistant", "content": "cut"}, "length"), model="q", tr=translation()
    )
    assert r["status"] == "incomplete" and r["incomplete_details"] == {
        "reason": "max_output_tokens"
    }


def test_empty_choices_is_an_empty_output():
    r = to_response_object({"choices": []}, model="q", tr=translation())
    assert r["output"] == [] and r["status"] == "completed"


# ---- stream: helpers ----------------------------------------------------------------------------


def chunk(delta=None, finish=None, usage=None):
    ev: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [],
    }
    if delta is not None or finish is not None:
        ev["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        ev["usage"] = usage
    return ev


def sse(events, done=True, crlf=False):
    frames = [b"data: " + json.dumps(e).encode() + b"\n\n" for e in events]
    if done:
        frames.append(b"data: [DONE]\n\n")
    raw = b"".join(frames)
    return raw.replace(b"\n", b"\r\n") if crlf else raw


def run(raw: bytes, tr: Translation, split: int | None = None, model="qwen"):
    t = ResponsesStreamTranslator(model=model, tr=tr)
    out: list[bytes] = []
    pieces = [raw] if split is None else [raw[i : i + split] for i in range(0, len(raw), split)]
    for piece in pieces:
        out.extend(t.feed_bytes(piece))
        if t.done:
            break
    out.extend(t.finish())
    events = []
    for frame in b"".join(out).decode().split("\n\n"):
        if not frame.strip():
            continue
        lines = dict(line.split(": ", 1) for line in frame.split("\n"))
        data = json.loads(lines["data"])
        assert data["type"] == lines["event"]
        events.append(data)
    return events


def check_invariants(events):
    types = [e["type"] for e in events]
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert types[0] == "response.created" and types[1] == "response.in_progress"
    terminal = [
        t for t in types if t in ("response.completed", "response.failed", "response.incomplete")
    ]
    assert len(terminal) == 1 and types[-1] == terminal[0]
    assert "response.incomplete" not in types
    added = [e for e in events if e["type"] == "response.output_item.added"]
    done = [e for e in events if e["type"] == "response.output_item.done"]
    if types[-1] == "response.completed":
        assert [e["output_index"] for e in added] == list(range(len(added)))
        assert sorted(e["output_index"] for e in done) == list(range(len(added)))
        for a in added:
            d = next(x for x in done if x["output_index"] == a["output_index"])
            assert d["item"]["id"] == a["item"]["id"]
        final = events[-1]["response"]["output"]
        assert sorted(i["id"] for i in final) == sorted(e["item"]["id"] for e in done)


def normalise(events):
    """Stable ids and timestamps, so a transcript can be pinned."""
    text = json.dumps(events, indent=1)
    seen: dict[str, str] = {}

    def sub(m):
        return seen.setdefault(m.group(0), f"{m.group(1)}_{len(seen) + 1:02d}")

    text = re.sub(r"\b(resp|msg|rs|fc|ctc)_[0-9a-f]{24}\b", sub, text)
    text = re.sub(r"\b(call)_[0-9a-f]{24}\b", sub, text)
    text = re.sub(r'"created_at": \d+', '"created_at": 0', text)
    return json.loads(text)


def golden(name, events):
    path = GOLDEN / f"{name}.golden.json"
    got = normalise(events)
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(json.dumps(got, indent=1) + "\n")
    assert got == json.loads(path.read_text()), f"{name} drifted; UPDATE_GOLDEN=1 to accept"


USAGE = {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}


def text_stream():
    return [
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({}, "stop"),
        chunk(usage=USAGE),
    ]


# ---- stream: scenarios ---------------------------------------------------------------------------


def test_text_golden():
    ev = run(sse(text_stream()), translation(stream=True))
    check_invariants(ev)
    assert [e["delta"] for e in ev if e["type"] == "response.output_text.delta"] == ["Hel", "lo"]
    done = next(e for e in ev if e["type"] == "response.output_text.done")
    assert done["text"] == "Hello"
    final = ev[-1]["response"]
    assert final["status"] == "completed" and final["usage"]["input_tokens"] == 12
    assert "[DONE]" not in json.dumps(ev)
    golden("text", ev)


def test_reasoning_then_text_golden():
    events = [
        chunk({"role": "assistant", "reasoning_content": "Let me "}),
        chunk({"reasoning_content": "think."}),
        chunk({"content": "Answer"}),
        chunk({}, "stop"),
        chunk(usage={**USAGE, "completion_tokens_details": {"reasoning_tokens": 3}}),
    ]
    ev = run(sse(events), translation(stream=True, reasoning={"effort": "medium"}))
    check_invariants(ev)
    types = [e["type"] for e in ev]
    assert types.index("response.reasoning_summary_text.delta") < types.index(
        "response.output_text.delta"
    )
    assert ev[-1]["response"]["output"][0]["summary"][0]["text"] == "Let me think."
    assert ev[-1]["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] == 3
    golden("reasoning_text", ev)


def test_reasoning_dropped_when_not_requested():
    events = [
        chunk({"reasoning_content": "secret"}),
        chunk({"content": "A"}),
        chunk({}, "stop"),
        chunk(usage=USAGE),
    ]
    ev = run(sse(events), translation(stream=True))
    check_invariants(ev)
    assert not any("reasoning" in e["type"] for e in ev)
    assert [i["type"] for i in ev[-1]["response"]["output"]] == ["message"]


def tool_delta(index, **fn):
    d: dict[str, Any] = {"index": index, "function": fn}
    if "name" in fn:
        d["id"] = f"call_{index}"
        d["type"] = "function"
    return d


def test_text_plus_tool_golden():
    events = [
        chunk({"content": "Running."}),
        chunk({"tool_calls": [tool_delta(0, name="shell", arguments="")]}),
        chunk({"tool_calls": [tool_delta(0, arguments='{"cmd":')]}),
        chunk({"tool_calls": [tool_delta(0, arguments='"ls"}')]}),
        chunk({}, "tool_calls"),
        chunk(usage=USAGE),
    ]
    tr = translation(stream=True, tools=[{"type": "function", "name": "shell"}])
    ev = run(sse(events), tr)
    check_invariants(ev)
    types = [e["type"] for e in ev]
    assert types.index("response.output_text.done") < types.index(
        "response.function_call_arguments.delta"
    )
    args = next(e for e in ev if e["type"] == "response.function_call_arguments.done")
    assert args["arguments"] == '{"cmd":"ls"}'
    call = ev[-1]["response"]["output"][1]
    assert (call["type"], call["call_id"], call["name"]) == ("function_call", "call_0", "shell")
    golden("text_tool", ev)


def test_interleaved_parallel_tools_golden():
    events = [
        chunk({"tool_calls": [tool_delta(0, name="a", arguments='{"x"')]}),
        chunk({"tool_calls": [tool_delta(1, name="b", arguments='{"y"')]}),
        chunk({"tool_calls": [tool_delta(0, arguments=":1}")]}),
        chunk({"tool_calls": [tool_delta(1, arguments=":2}")]}),
        chunk({}, "tool_calls"),
        chunk(usage=USAGE),
    ]
    tr = translation(
        stream=True, tools=[{"type": "function", "name": "a"}, {"type": "custom", "name": "b"}]
    )
    ev = run(sse(events), tr)
    check_invariants(ev)
    out = ev[-1]["response"]["output"]
    assert [(o["type"], o["call_id"]) for o in out] == [
        ("function_call", "call_0"),
        ("custom_tool_call", "call_1"),
    ]
    assert out[0]["arguments"] == '{"x":1}'
    assert out[1]["input"] == '{"y":2}'  # not an {"input": ...} object: raw
    golden("parallel_tools", ev)


def test_length_ends_with_completed_incomplete_golden():
    events = [chunk({"content": "trunc"}), chunk({}, "length"), chunk(usage=USAGE)]
    ev = run(sse(events), translation(stream=True))
    check_invariants(ev)
    final = ev[-1]["response"]
    assert ev[-1]["type"] == "response.completed"
    assert final["status"] == "incomplete" and final["incomplete_details"] == {
        "reason": "max_output_tokens"
    }
    golden("length", ev)


def test_runaway_terminal_chunk_is_incomplete():
    events = [chunk({"content": "loop loop"}), chunk({}, "runaway_repeat")]
    ev = run(sse(events), translation(stream=True))
    check_invariants(ev)
    assert ev[-1]["response"]["status"] == "incomplete"


def test_engine_error_mid_stream_is_response_failed_golden():
    events = [chunk({"content": "par"}), {"error": {"message": "engine died", "type": "x"}}]
    ev = run(sse(events), translation(stream=True))
    check_invariants(ev)
    assert ev[-1]["type"] == "response.failed"
    resp = ev[-1]["response"]
    assert resp["status"] == "failed"
    assert resp["error"] == {"code": "server_error", "message": "engine died"}
    golden("engine_error", ev)


def test_old_flat_error_shape():
    ev = run(sse([{"object": "error", "message": "boom", "code": 500}]), translation(stream=True))
    assert ev[-1]["type"] == "response.failed" and ev[-1]["response"]["error"]["message"] == "boom"


def test_vllm_continuous_usage_does_not_duplicate_or_reorder():
    u = lambda n: {"prompt_tokens": 12, "completion_tokens": n, "total_tokens": 12 + n}  # noqa: E731
    events = [
        {**chunk({"content": "a"}), "usage": u(1)},
        {**chunk({"content": "b"}), "usage": u(2)},
        {**chunk({}, "stop"), "usage": u(2)},
        chunk(usage=u(2)),
    ]
    ev = run(sse(events), translation(stream=True))
    check_invariants(ev)
    assert [e["type"] for e in ev].count("response.completed") == 1
    assert ev[-1]["response"]["usage"]["output_tokens"] == 2
    golden("vllm_continuous_usage", ev)


def test_llamacpp_style_stream_without_usage_tail_or_roles():
    events = [
        {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]},
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
    ]
    ev = run(sse(events, done=False), translation(stream=True))
    check_invariants(ev)
    assert ev[-1]["response"]["usage"]["total_tokens"] == 4
    golden("llamacpp", ev)


def test_empty_stream_still_completes():
    ev = run(b"", translation(stream=True))
    check_invariants(ev)
    assert ev[-1]["type"] == "response.completed" and ev[-1]["response"]["output"] == []


@pytest.mark.parametrize("split", [1, 2, 7, 64])
@pytest.mark.parametrize("crlf", [False, True])
def test_split_and_crlf_frames_give_the_same_transcript(split, crlf):
    base = normalise(run(sse(text_stream()), translation(stream=True)))
    got = normalise(run(sse(text_stream(), crlf=crlf), translation(stream=True), split=split))
    assert got == base


def test_unterminated_last_frame_is_still_read():
    raw = sse(text_stream(), done=False).rstrip(b"\n")
    ev = run(raw, translation(stream=True))
    check_invariants(ev)
    assert ev[-1]["response"]["usage"]["input_tokens"] == 12


def test_response_model_is_the_requested_name():
    ev = run(sse(text_stream()), translation(stream=True), model="my-served-name")
    assert ev[0]["response"]["model"] == ev[-1]["response"]["model"] == "my-served-name"


def test_nothing_after_the_terminal_event():
    t = ResponsesStreamTranslator(model="q", tr=translation(stream=True))
    t.feed_bytes(sse([{"error": {"message": "x"}}]))
    assert t.done
    assert t.feed_bytes(sse(text_stream())) == []
    assert t.finish() == []


def test_no_finish_reason_with_output_is_incomplete():
    # a wall-clock reaper cut: output, then the stream just ends
    ev = run(sse([chunk({"content": "cut off"})]), translation(stream=True))
    check_invariants(ev)
    final = ev[-1]["response"]
    assert ev[-1]["type"] == "response.completed" and final["status"] == "incomplete"
    assert final["incomplete_details"] == {"reason": "max_output_tokens"}
    r = to_response_object(
        _completion({"role": "assistant", "content": "x"}, None), model="q", tr=translation()
    )
    assert r["status"] == "incomplete"


def test_no_finish_reason_without_output_stays_completed():
    ev = run(sse([chunk({"role": "assistant", "content": ""})]), translation(stream=True))
    assert ev[-1]["response"]["status"] == "completed"


def test_fail_ends_the_stream_once():
    t = ResponsesStreamTranslator(model="q", tr=translation(stream=True))
    frames = t.fail("response translation failed")
    assert t.done and b"response.failed" in b"".join(frames)
    assert t.fail("again") == [] and t.finish() == []
