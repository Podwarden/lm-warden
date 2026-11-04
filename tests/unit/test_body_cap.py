"""Request bodies are bounded before any handler or CSRF check reads them (verified findings #5, #6)."""

from __future__ import annotations

import dataclasses

from tests.unit.proxy.test_messages import AUTH, _ready


def _chunks(total: int, size: int = 65536):
    sent = 0
    while sent < total:
        n = min(size, total - sent)
        sent += n
        yield b"x" * n


def test_v1_body_over_the_cap_is_413_before_the_handler(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, max_request_body_mb=1
    )
    big = (
        b'{"model":"qwen","messages":[{"role":"user","content":"'
        + b"a" * (2 * 1024 * 1024)
        + b'"}]}'
    )
    r = client.post(
        "/v1/chat/completions", content=big, headers={**AUTH, "content-type": "application/json"}
    )
    assert r.status_code == 413, r.text


def test_v1_chunked_body_over_the_cap_is_413(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, max_request_body_mb=1
    )
    r = client.post(
        "/v1/chat/completions",
        content=_chunks(3 * 1024 * 1024),
        headers={**AUTH, "content-type": "application/json"},
    )
    assert r.status_code == 413, r.text


def test_anonymous_multipart_without_csrf_is_refused_without_reading_it(client):
    client.get("/healthz")
    read = {"n": 0}

    def body():
        for chunk in _chunks(20 * 1024 * 1024):
            read["n"] += len(chunk)
            yield chunk

    r = client.post(
        "/api/tokens", content=body(), headers={"content-type": "multipart/form-data; boundary=x"}
    )
    assert r.status_code in (403, 413), r.text
    assert read["n"] < 2 * 1024 * 1024, read  # refused without buffering the 20 MB


def test_chunked_attachment_upload_over_the_limit_is_413(tmp_data_dir, client):
    from app.chat2.body_limit import _MAX_DECLARED_BYTES
    from tests.conftest import csrf_header, jwt_login, seed_admin_user

    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    h = {**jwt_login(client), **csrf_header(client)}
    r = client.post(
        "/api/chat2/attachments",
        content=_chunks(_MAX_DECLARED_BYTES + 1024 * 1024),
        headers={**h, "content-type": "multipart/form-data; boundary=x"},
    )
    assert r.status_code == 413, r.text


def test_normal_api_bodies_pass(client):
    client.get("/healthz")
    r = client.post("/api/auth/login", json={"username": "nobody", "password": "x" * 20})
    assert r.status_code == 401
