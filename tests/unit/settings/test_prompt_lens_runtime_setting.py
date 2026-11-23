"""The Prompt Lens switch on the Settings page: runtime setting ``prompt_lens_enabled``.

* Precedence: the setting when set, else the env (``VW_PROMPT_LENS_ENABLED`` or
  the deprecated ``VW_GODMODE_ENABLED``), else off. ``GET
  /api/admin/prompt-lens/status`` says which source is in effect.
* Changed live, no restart. Off stops new captures at once, ends every open
  Prompt Lens stream (admin and key holder) and drops the in-memory ring and
  sections; kept captures on disk stay.
* Admin session only (an admin token gets 403 session_only, like the update
  check); each change is logged with who made it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.proxy import prompt_lens_stream, routes_prompt_lens
from app.proxy.routes_prompt_lens import STREAM_PATH as ADMIN_STREAM_PATH
from app.tokens.routes_me_prompt_lens import ME_STREAM_PATH
from tests.asgi_stream import open_stream
from tests.conftest import bearer, csrf_header, jwt_login, seed_admin_token
from tests.unit.proxy.test_prompt_lens_forward import _fake_tokenizer, _seed_loaded
from tests.unit.tokens.test_me_prompt_lens_stream import A, B, Env, _traffic, env

__all__ = ["env"]  # the fixture, re-used

URL = "/api/settings/runtime"
STATUS = "/api/admin/prompt-lens/status"


def _kv(db: Path) -> dict[str, str]:
    with sqlite3.connect(db) as c:
        return dict(c.execute("SELECT key, value FROM settings").fetchall())


async def _patch(env: Env, on: bool) -> None:
    r = await env.http.patch(URL, json={"prompt_lens_enabled": on}, headers=await env.admin_csrf())
    assert r.status_code == 200, r.text
    assert r.json()["requires_restart_kinds"] == []


async def _status(env: Env) -> dict[str, Any]:
    r = await env.http.get(STATUS, headers=env.admin)
    assert r.status_code == 200, r.text
    return dict(r.json())


# -- precedence ---------------------------------------------------------------


async def test_default_is_off_from_the_default(env: Env) -> None:
    env.prompt_lens(False)
    env.app.state.settings = dataclasses.replace(env.app.state.settings, prompt_lens_env_var=None)
    st = await _status(env)
    assert st == {
        "enabled": False,
        "source": "default",
        "setting": None,
        "env_var": None,
        "env_value": None,
        "last_change": None,
    }
    r = await env.http.get(URL, headers=env.admin)
    assert r.json()["prompt_lens_enabled"] == "false"  # the value in effect


async def test_env_decides_while_the_setting_is_unset(env: Env) -> None:
    env.app.state.settings = dataclasses.replace(
        env.app.state.settings, prompt_lens_enabled=True, prompt_lens_env_var="VW_GODMODE_ENABLED"
    )
    st = await _status(env)
    assert st["enabled"] is True and st["source"] == "env"
    assert st["env_var"] == "VW_GODMODE_ENABLED" and st["env_value"] is True
    r = await env.http.get(URL, headers=env.admin)
    assert r.json()["prompt_lens_enabled"] == "true"


async def test_the_setting_wins_over_the_env(env: Env) -> None:
    env.app.state.settings = dataclasses.replace(
        env.app.state.settings,
        prompt_lens_enabled=True,
        prompt_lens_env_var="VW_PROMPT_LENS_ENABLED",
    )
    await _patch(env, False)
    st = await _status(env)
    assert st["enabled"] is False and st["source"] == "setting" and st["setting"] is False
    assert st["env_var"] == "VW_PROMPT_LENS_ENABLED" and st["env_value"] is True
    assert _kv(env.db)["prompt_lens_enabled"] == "false"
    r = await env.http.post("/api/me/prompt-lens/ticket", headers=env.key[A])
    assert r.status_code == 404

    env.prompt_lens(False)
    await _patch(env, True)
    assert (await _status(env))["enabled"] is True
    r = await env.http.post("/api/me/prompt-lens/ticket", headers=env.key[A])
    assert r.status_code == 200


async def test_the_stored_setting_is_read_at_startup(env: Env) -> None:
    from app.proxy import prompt_lens_switch

    await _patch(env, False)
    env.app.state.prompt_lens_runtime = None  # as a fresh process
    env.prompt_lens(True)
    await prompt_lens_switch.load(env.app)
    assert prompt_lens_switch.enabled(env.app) is False


# -- live toggle --------------------------------------------------------------


async def test_off_ends_open_streams_and_drops_memory(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Re-check and keepalive far away: only the switch itself can end them this fast.
    monkeypatch.setattr(prompt_lens_stream, "RECHECK_INTERVAL_S", 600.0)
    monkeypatch.setattr(routes_prompt_lens, "KEEPALIVE_INTERVAL_S", 600.0)
    await _patch(env, True)
    r = await env.http.post(
        "/api/auth/sse-ticket", json={"path": ADMIN_STREAM_PATH}, headers=env.admin
    )
    assert r.status_code == 200, r.text
    admin_ticket = r.json()["ticket"]
    key_ticket = (await env.http.post("/api/me/prompt-lens/ticket", headers=env.key[A])).json()[
        "ticket"
    ]
    for event, keep in _traffic(A, 1, phase="before"):
        env.hub.publish(event, keep=keep)
    assert env.hub.events()
    async with (
        open_stream(env.app, f"{ADMIN_STREAM_PATH}?ticket={admin_ticket}", {}) as sa,
        open_stream(env.app, f"{ME_STREAM_PATH}?ticket={key_ticket}", {}) as sk,
    ):
        assert sa.status == sk.status == 200
        assert not await sa.ended(within_s=0.2)
        await _patch(env, False)
        assert await sa.ended(within_s=3.0)
        assert await sk.ended(within_s=3.0)
        # A request still running publishes into a hub that keeps nothing.
        for event, keep in _traffic(A, 2, phase="after"):
            env.hub.publish(event, keep=keep)
    assert "after" not in sa.body.decode() + sk.body.decode()
    assert env.hub.events() == [] and env.hub.key_events(A) == []
    assert env.app.state.prompt_lens_sections.size_chars() == 0
    assert env.app.state.stream_registry.count(f"key:{A}") == 0
    # New streams are refused.
    r = await env.http.post(
        "/api/auth/sse-ticket", json={"path": ADMIN_STREAM_PATH}, headers=env.admin
    )
    assert r.status_code == 409


async def test_key_stream_recheck_reads_the_setting(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another worker or a direct KV change: no drop_all reaches this hub, so the
    key stream's own re-check must find the setting off."""
    monkeypatch.setattr(prompt_lens_stream, "RECHECK_INTERVAL_S", 0.05)
    monkeypatch.setattr(routes_prompt_lens, "KEEPALIVE_INTERVAL_S", 600.0)
    t = (await env.http.post("/api/me/prompt-lens/ticket", headers=env.key[B])).json()["ticket"]
    async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", {}) as s:
        assert s.status == 200
        assert not await s.ended(within_s=0.2)
        env.app.state.prompt_lens_runtime = False  # the setting only, no drop
        assert await s.ended(within_s=3.0)


async def test_off_keeps_kept_captures_on_disk(env: Env) -> None:
    def seed() -> None:
        with sqlite3.connect(env.db) as c:
            c.execute(
                "INSERT INTO prompt_lens_kept(req_id, token_id, ended_ts, chars, summary, events)"
                " VALUES ('k1', ?, 1.0, 0, '{}', '[]')",
                (A,),
            )

    await asyncio.to_thread(seed)
    await _patch(env, False)
    with sqlite3.connect(env.db) as c:
        assert c.execute("SELECT req_id FROM prompt_lens_kept").fetchall() == [("k1",)]


async def test_a_change_is_logged_with_who_made_it(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    env.prompt_lens(False)
    with caplog.at_level(logging.INFO, logger="app.settings.routes_api"):
        await _patch(env, True)
        await _patch(env, True)  # no change, no line
    lines = [r.getMessage() for r in caplog.records if "Prompt Lens" in r.getMessage()]
    assert lines == ["Prompt Lens off -> on by admin (session)"]


# -- who may change it --------------------------------------------------------


def test_an_admin_token_cannot_change_it(client: Any, seeded_db: Path) -> None:
    _, secret = seed_admin_token(seeded_db)
    r = client.patch(URL, json={"prompt_lens_enabled": True}, headers=bearer(secret))
    assert r.status_code == 403, r.text
    assert r.json()["detail"]["keys"] == ["prompt_lens_enabled"]
    assert "prompt_lens_enabled" not in _kv(seeded_db)
    # It may still read the status.
    assert client.get(STATUS, headers=bearer(secret)).status_code == 200


def test_bad_values_are_422(client: Any, seeded_db: Path) -> None:
    session = {**jwt_login(client), **csrf_header(client)}
    r = client.patch(URL, json={"prompt_lens_enabled": "maybe"}, headers=session)
    assert r.status_code == 422


# -- the proxy tap follows the setting ------------------------------------------


def _proxy_one(client: Any, plaintext: str) -> None:
    chunks = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]

    async def aiter():
        for c in chunks:
            yield c

    fake = MagicMock()
    fake.status_code = 200
    fake.headers = {"content-type": "text/event-stream"}
    fake.aiter_bytes = aiter
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake)):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "x"}]},
        ) as r:
            for _ in r.iter_bytes():
                pass
            assert r.status_code == 200


def test_capture_follows_the_setting_over_the_env(tmp_data_dir: Path, client: Any) -> None:
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)
    session = {**jwt_login(client), **csrf_header(client)}
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=True
    )

    # Env on, setting off: nothing is captured.
    r = client.patch(URL, json={"prompt_lens_enabled": False}, headers=session)
    assert r.status_code == 200, r.text
    _proxy_one(client, plaintext)
    assert client.app.state.prompt_lens_hub.events() == []

    # Env off, setting on: captured.
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=False
    )
    r = client.patch(URL, json={"prompt_lens_enabled": True}, headers=session)
    assert r.status_code == 200, r.text
    _proxy_one(client, plaintext)
    types = [e["type"] for e in client.app.state.prompt_lens_hub.events()]
    assert types[0] == "request_start" and types[-1] == "request_end"
    assert json.dumps(client.app.state.prompt_lens_hub.events()).count(plaintext) == 0


# -- a stream subscribed just before off --------------------------------------


async def test_a_stream_subscribed_before_off_replays_nothing(env: Env) -> None:
    from app.proxy import prompt_lens_switch
    from app.proxy.prompt_lens_stream import admin_view, prompt_lens_sse

    await _patch(env, True)
    for event, keep in _traffic(A, 1, phase="pre-off"):
        env.hub.publish(event, keep=keep)
    request = MagicMock()
    request.app = env.app
    request.is_disconnected = AsyncMock(return_value=False)
    resp = await prompt_lens_sse(
        request,
        env.hub,
        user="admin",
        wanted=None,
        transform=admin_view,
        ring_meta=True,
        recheck=None,
    )
    # Off between the subscribe (snapshot taken) and the generator's first step.
    prompt_lens_switch.apply(env.app, False)
    body = ""
    async for chunk in resp.body_iterator:
        body += chunk if isinstance(chunk, str) else chunk.decode()
    assert "pre-off" not in body
    assert "request_start" not in body


# -- "Use server default" -----------------------------------------------------


async def test_null_clears_the_setting_so_the_env_applies_again(env: Env) -> None:
    env.app.state.settings = dataclasses.replace(
        env.app.state.settings,
        prompt_lens_enabled=True,
        prompt_lens_env_var="VW_PROMPT_LENS_ENABLED",
    )
    await _patch(env, False)
    assert (await _status(env))["enabled"] is False
    r = await env.http.patch(
        URL, json={"prompt_lens_enabled": None}, headers=await env.admin_csrf()
    )
    assert r.status_code == 200, r.text
    assert "prompt_lens_enabled" not in _kv(env.db)
    st = await _status(env)
    assert st["source"] == "env" and st["enabled"] is True and st["setting"] is None
    r = await env.http.get(URL, headers=env.admin)
    assert r.json()["prompt_lens_enabled"] == "true"


def test_null_is_session_only_too(client: Any, seeded_db: Path) -> None:
    _, secret = seed_admin_token(seeded_db)
    r = client.patch(URL, json={"prompt_lens_enabled": None}, headers=bearer(secret))
    assert r.status_code == 403


def test_null_is_refused_for_other_keys(client: Any, seeded_db: Path) -> None:
    session = {**jwt_login(client), **csrf_header(client)}
    r = client.patch(URL, json={"log_retention_lines": None}, headers=session)
    assert r.status_code == 422


# -- the durable record -------------------------------------------------------


def _audit(db: Path) -> list[tuple[Any, ...]]:
    with sqlite3.connect(db) as c:
        return c.execute(
            "SELECT actor, via, old_enabled, new_enabled, old_source, new_source"
            " FROM prompt_lens_switch_audit ORDER BY id"
        ).fetchall()


async def test_every_change_is_recorded(env: Env) -> None:
    env.app.state.settings = dataclasses.replace(
        env.app.state.settings, prompt_lens_enabled=False, prompt_lens_env_var=None
    )
    await _patch(env, True)
    await _patch(env, True)  # saved again: same value, same source, still recorded
    r = await env.http.patch(
        URL, json={"prompt_lens_enabled": None}, headers=await env.admin_csrf()
    )
    assert r.status_code == 200
    assert _audit(env.db) == [
        ("admin", "session", 0, 1, "default", "setting"),
        ("admin", "session", 1, 1, "setting", "setting"),
        ("admin", "session", 1, 0, "setting", "default"),
    ]
    last = (await _status(env))["last_change"]
    assert last["actor"] == "admin" and last["via"] == "session"
    assert last["old_enabled"] is True and last["new_enabled"] is False
    assert last["new_source"] == "default" and isinstance(last["ts"], float)


async def test_status_without_changes_has_no_last_change(env: Env) -> None:
    assert (await _status(env))["last_change"] is None
