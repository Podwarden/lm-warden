"""God-mode media tap — image_url extraction into the media store.

Covers spec §4: data URIs stored + metadata entry; remote URLs pass through
as url entries; oversized -> too_large placeholder; count cap -> overflow
placeholder; malformed parts skipped; disabled -> byte-identical (no store
calls). Pure-function tests against _extract_godmode_media plus one
gate test mirroring test_godmode_forward.py's disabled-path guard.
"""

import dataclasses
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt

from app.db.repos.tokens import hash_token
from app.proxy.godmode import GodModeMediaStore
from app.proxy.routes import _extract_godmode_media

SETTINGS = SimpleNamespace(godmode_max_image_chars=1000, godmode_max_images_per_req=3)


def _body(*parts):
    return {"messages": [{"role": "user", "content": list(parts)}]}


def _img(url):
    return {"type": "image_url", "image_url": {"url": url}}


def test_data_uri_stored_and_entry_emitted():
    store = GodModeMediaStore()
    body = _body({"type": "text", "text": "what is this?"}, _img("data:image/png;base64,aGVsbG8="))
    media = _extract_godmode_media(body, store, SETTINGS)
    assert len(media) == 1
    e = media[0]
    assert e["kind"] == "image" and e["mime"] == "image/png"
    assert e["chars"] == len("aGVsbG8=")
    assert store.get(e["media_id"]) == ("image/png", "aGVsbG8=")


def test_data_uri_with_params_before_base64_is_captured_and_normalized():
    """A data URI may carry parameters before ';base64,' (e.g. charset).
    The raw slice url[5:sep] used to keep them in the mime string, so
    "image/png;charset=utf-8" both failed the store's strict allowlist AND
    got shown to the operator verbatim in a mislabeled "too_large" chip. The
    mime must be normalized (params stripped, lowercased) before it reaches
    either the allowlist check or the emitted entry."""
    store = GodModeMediaStore()
    body = _body(_img("data:image/png;charset=utf-8;base64,aGVsbG8="))
    media = _extract_godmode_media(body, store, SETTINGS)
    assert len(media) == 1
    e = media[0]
    assert e["kind"] == "image" and e["mime"] == "image/png"
    assert "dropped" not in e
    assert store.get(e["media_id"]) == ("image/png", "aGVsbG8=")


def test_remote_url_passthrough_store_untouched():
    store = MagicMock()
    media = _extract_godmode_media(_body(_img("https://example.com/cat.jpg")), store, SETTINGS)
    assert media == [{"kind": "image", "url": "https://example.com/cat.jpg"}]
    store.put.assert_not_called()


def test_oversized_becomes_too_large_placeholder_not_stored():
    store = GodModeMediaStore()
    huge = "data:image/jpeg;base64," + "A" * 2000  # over the 1000-char test cap
    media = _extract_godmode_media(_body(_img(huge)), store, SETTINGS)
    assert media == [
        {"kind": "image", "mime": "image/jpeg", "chars": 2000, "dropped": "too_large"}
    ]
    assert store.size_chars() == 0


def test_count_cap_emits_overflow_placeholder():
    store = GodModeMediaStore()
    parts = [_img(f"https://example.com/{i}.png") for i in range(5)]  # cap is 3
    media = _extract_godmode_media(_body(*parts), store, SETTINGS)
    assert len(media) == 4  # 3 captured + 1 overflow placeholder
    assert media[3] == {"kind": "image", "dropped": "count", "count": 2}


def test_count_cap_bounds_store_writes_for_data_uris():
    """The remote-URL cap test above never touches the store (no put() call
    for a {"url": ...} entry), so it could not have caught the bug where the
    cap was checked AFTER store.put() ran: every image past max_items still
    got sliced into the store as an orphan blob, evicting legitimate images.
    This drives the cap with data URIs, which do call store.put(), and pins
    that writes stop at max_items."""
    from pathlib import Path

    from app.config import Settings

    settings = Settings(
        data_dir=Path("/data"), hf_cache_dir=Path("/root/.cache/huggingface"),
        cookie_secret="x" * 32, container_gpu_count=0,
    )
    max_items = settings.godmode_max_images_per_req  # 16 by default
    n = max_items + 5
    payload = "aGVsbG8="
    real_store = GodModeMediaStore()
    store = MagicMock(wraps=real_store)
    parts = [_img(f"data:image/png;base64,{payload}") for _ in range(n)]

    media = _extract_godmode_media(_body(*parts), store, settings)

    assert store.put.call_count == max_items
    assert len(media) == max_items + 1  # max_items captured + 1 overflow placeholder
    assert media[-1] == {"kind": "image", "dropped": "count", "count": n - max_items}
    assert all("dropped" not in e for e in media[:max_items])


def test_malformed_parts_skipped():
    store = GodModeMediaStore()
    body = _body(
        {"type": "image_url"},                                  # no image_url key
        {"type": "image_url", "image_url": "not-a-dict-url"},   # tolerated str form -> not data:/http(s) -> skipped
        {"type": "image_url", "image_url": {"url": 42}},        # non-str url
        {"type": "image_url", "image_url": {"url": "data:image/png,notbase64"}},  # no ;base64,
        {"type": "image_url", "image_url": {"url": "data:text/html;base64,PGI+"}},  # non-image mime
        "just a string part",
    )
    assert _extract_godmode_media(body, store, SETTINGS) == []
    assert store.size_chars() == 0


def test_string_image_url_form_tolerated():
    # Some clients send image_url as a bare string instead of {"url": ...}.
    store = GodModeMediaStore()
    media = _extract_godmode_media(
        _body({"type": "image_url", "image_url": "https://example.com/x.png"}),
        store,
        SETTINGS,
    )
    assert media == [{"kind": "image", "url": "https://example.com/x.png"}]


def test_no_messages_or_text_only_returns_empty():
    store = MagicMock()
    assert _extract_godmode_media({"prompt": "hi"}, store, SETTINGS) == []
    assert _extract_godmode_media(_body({"type": "text", "text": "hi"}), store, SETTINGS) == []
    store.put.assert_not_called()


def test_helper_never_raises_on_garbage_body():
    store = MagicMock()
    store.put.side_effect = RuntimeError("boom")
    body = _body(_img("data:image/png;base64,aGVsbG8="))
    media = _extract_godmode_media(body, store, SETTINGS)  # must not raise
    assert isinstance(media, list)
    assert _extract_godmode_media({"messages": "garbage"}, store, SETTINGS) == []
    assert _extract_godmode_media(None, store, SETTINGS) == []


# ---------------------------------------------------------------------------
# Gate tests mirroring test_godmode_forward.py's disabled-path guard, driving
# _forward through the full client + httpx-send mock (same harness).
# ---------------------------------------------------------------------------


def _seed_loaded(db_path):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,'auto',4096,0.9,0,'[]','loaded',0,NULL,NULL)",
            (json.dumps([0]),),
        )
        plaintext = "vw_validtoken1234567890abcdef12345"
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "my-token", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
        return plaintext


def _fake_tokenizer(count_for):
    cache = MagicMock()
    cache.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: count_for(text))
    return cache


def _enable_godmode(client):
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, godmode_enabled=True
    )


def test_disabled_gate_zero_media_store_calls(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)

    # godmode_enabled is false by default, so the gate must never touch the
    # media store — spy in place of the real one, same pattern as the hub spy
    # in test_godmode_forward.py.
    spy = MagicMock()
    client.app.state.godmode_media = spy

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"it is a cat"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    async def aiter():
        for c in sse_chunks:
            yield c

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "text/event-stream"}
    fake_resp.aiter_bytes = aiter
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        with client.stream(
            "POST", "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={
                "model": "qwen", "stream": True,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                ]}],
            },
        ) as r:
            body = b"".join(r.iter_bytes())
            assert r.status_code == 200

    assert body == b"".join(sse_chunks)
    assert spy.method_calls == []


def test_enabled_gate_media_in_request_start_event(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)
    _enable_godmode(client)

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"a cat"},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    async def aiter():
        for c in sse_chunks:
            yield c

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "text/event-stream"}
    fake_resp.aiter_bytes = aiter
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        with client.stream(
            "POST", "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={
                "model": "qwen", "stream": True,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                ]}],
            },
        ) as r:
            for _ in r.iter_bytes():
                pass
            assert r.status_code == 200

    hub = client.app.state.godmode_hub
    _q, snap = hub.subscribe()
    start = snap[0]
    assert start["type"] == "request_start"
    assert "media" in start
    assert len(start["media"]) == 1
    e = start["media"][0]
    assert e["kind"] == "image" and e["mime"] == "image/png" and e["chars"] == len("aGVsbG8=")
    store = client.app.state.godmode_media
    assert store.get(e["media_id"]) == ("image/png", "aGVsbG8=")


def test_missing_media_store_state_does_not_break_forward(tmp_data_dir, client):
    """Mirrors every other god-mode state read (routes.py:290,
    routes_godmode.py:59-60, auth/routes.py:100): the hot-path lookup of
    ``app.state.godmode_media`` must be getattr-defensive. If the attribute is
    simply absent (e.g. app wiring changes, a test harness, a partial
    startup), the proxied request must still complete byte-identically rather
    than 500 inside ``_forward``."""
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)
    _enable_godmode(client)
    # Simulate the attribute never having been set on app.state at all —
    # the exact scenario the review flagged (a plain attribute access at the
    # _forward call site, not a getattr, would raise AttributeError here).
    del client.app.state.godmode_media

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"it is a cat"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    async def aiter():
        for c in sse_chunks:
            yield c

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "text/event-stream"}
    fake_resp.aiter_bytes = aiter
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        with client.stream(
            "POST", "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={
                "model": "qwen", "stream": True,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                ]}],
            },
        ) as r:
            body = b"".join(r.iter_bytes())
            assert r.status_code == 200

    assert body == b"".join(sse_chunks)
