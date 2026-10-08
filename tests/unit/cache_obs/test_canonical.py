# tests/unit/cache_obs/test_canonical.py
import subprocess
import sys

from app.cache_obs.canonical import CHUNK_BYTES, chain_of, element_of_byte


def _chat(*contents, tools=None):
    body = {"model": "m", "messages": [{"role": "user", "content": c} for c in contents]}
    if tools is not None:
        body["tools"] = tools
    return body


def test_same_body_same_chain():
    a = chain_of(_chat("x" * 1000))
    b = chain_of(_chat("x" * 1000))
    assert a is not None and a.chunks == b.chunks
    assert len(a.chunks) == -(-a.total_bytes // CHUNK_BYTES)


def test_shared_prefix_shares_leading_chunks_only():
    a = chain_of(_chat("A" * 2000, "tail-one"))
    b = chain_of(_chat("A" * 2000, "tail-two"))
    common = 0
    for x, y in zip(a.chunks, b.chunks, strict=False):
        if x != y:
            break
        common += 1
    assert 0 < common < len(a.chunks)


def test_chunk_keys_are_chained_not_positional():
    # identical 256-byte blocks at different positions must not collide
    a = chain_of(_chat("B" * 4000))
    assert len(set(a.chunks)) == len(a.chunks)


def test_model_and_sampling_fields_do_not_matter():
    a = chain_of({**_chat("hi"), "temperature": 0.1, "model": "a"})
    b = chain_of({**_chat("hi"), "temperature": 0.9, "model": "b"})
    assert a.chunks == b.chunks


def test_tools_come_first_and_shift_message_offset():
    c = chain_of(_chat("hi", tools=[{"type": "function", "function": {"name": "f"}}]))
    assert c.msg_offset == 1 and len(c.elem_starts) == 2
    assert element_of_byte(c, 0) == 0


def test_completions_prompt_is_supported():
    assert chain_of({"model": "m", "prompt": "hello"}) is not None


def test_embeddings_and_junk_are_none():
    assert chain_of({"model": "m", "input": "x"}) is None
    assert chain_of("not a dict") is None
    assert chain_of({"messages": []}) is None


def test_oversize_body_is_skipped(monkeypatch):
    import app.cache_obs.canonical as can

    monkeypatch.setattr(can, "MAX_CHAIN_BYTES", 1000)
    assert chain_of(_chat("x" * 2000)) is None


def test_multimodal_parts_hash_without_error():
    parts = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    assert chain_of({"messages": [{"role": "user", "content": parts}]}) is not None


def test_digests_are_stable_across_processes():
    code = (
        "from app.cache_obs.canonical import chain_of;"
        "print(chain_of({'messages':[{'role':'user','content':'z'*600}]}).chunks[-1].hex())"
    )
    out = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    }
    assert len(out) == 1


# --- I5: assistant tool_calls and friends reach the prompt ---------------------


def _break_msg(a, b):
    """Message index where two chains first differ (None when identical)."""
    for i, (x, y) in enumerate(zip(a.chunks, b.chunks, strict=False)):
        if x != y:
            return element_of_byte(a, i * CHUNK_BYTES) - a.msg_offset
    return None


def _tool_turn(call_id, fn="get_weather"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": fn, "arguments": "{}"}}
        ],
    }


def test_tool_call_id_difference_breaks_at_that_message():
    head = [{"role": "system", "content": "S" * 2000}, {"role": "user", "content": "weather?"}]
    tail = [{"role": "tool", "tool_call_id": "x", "content": "sunny"}]
    a = chain_of({"messages": [*head, _tool_turn("call_A"), *tail]})
    b = chain_of({"messages": [*head, _tool_turn("call_B"), *tail]})
    assert a.chunks != b.chunks
    # the chunk break floors into the short message before it; the message
    # digests pin it to the tool-call turn itself
    assert 1 <= _break_msg(a, b) <= 2
    diff = [
        i for i, (x, y) in enumerate(zip(a.message_hashes, b.message_hashes, strict=True)) if x != y
    ]
    assert diff == [2]


def test_null_content_tool_call_messages_hash_differently():
    a = chain_of({"messages": [_tool_turn("c1", "f")]})
    b = chain_of({"messages": [_tool_turn("c1", "g")]})
    assert a.message_hashes != b.message_hashes


def test_non_prompt_message_keys_are_ignored():
    a = chain_of({"messages": [{"role": "user", "content": "hi"}]})
    b = chain_of({"messages": [{"role": "user", "content": "hi", "x_client_meta": 1}]})
    assert a.chunks == b.chunks


# --- I4: large leaves become one digest; oversize bodies bail early ----------


def _img_body(b64, text="describe"):
    return {
        "messages": [
            {"role": "system", "content": "S" * 2000},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                ],
            },
            {"role": "user", "content": "and then?"},
        ]
    }


_3MIB = 3 * 1024 * 1024


def test_different_large_images_break_at_that_message():
    a = chain_of(_img_body("A" * _3MIB))
    b = chain_of(_img_body("A" * (_3MIB - 1) + "B"))
    assert a is not None and b is not None
    assert _break_msg(a, b) == 1


def test_same_large_image_matches():
    a = chain_of(_img_body("Q" * _3MIB))
    b = chain_of(_img_body("Q" * _3MIB))
    assert a.chunks == b.chunks


def test_large_image_body_gives_a_small_chain():
    c = chain_of(_img_body("Z" * _3MIB))
    assert c is not None and len(c.chunks) < 100


def test_audio_and_short_image_urls_are_digested_too():
    def body(data, url):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ]
        }

    a = chain_of(body("aud1", "https://x/1.png"))
    b = chain_of(body("aud2", "https://x/1.png"))
    c = chain_of(body("aud1", "https://x/2.png"))
    assert a.chunks != b.chunks and a.chunks != c.chunks


def test_oversize_text_returns_none_without_serialising(monkeypatch):
    import app.cache_obs.canonical as can

    def boom(*a, **k):
        raise AssertionError("json.dumps must not run for an oversize body")

    monkeypatch.setattr(can.json, "dumps", boom)
    # 2.4 MB of text in leaves under 64 KiB (so none is digested away)
    assert chain_of(_chat(*["w" * 60_000 for _ in range(40)])) is None


def test_max_chain_bytes_is_two_mib():
    import app.cache_obs.canonical as can

    assert can.MAX_CHAIN_BYTES == 2 * 1024 * 1024
