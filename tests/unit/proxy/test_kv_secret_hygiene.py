"""Secret hygiene of per-key isolation (G7c).

The scope, the scope tag, the tenant tag and the derived GPU salt of an isolated key are
secrets of the key holder: they never appear in a log record, the history row, a debug
capture or the model's effective argv. Capture and history keep the client's body.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time

from app.proxy.request_id import REQUEST_ID_HEADER
from lmw_kv.scopes import gpu_cache_salt, isolation_scope, scope_tag, tenant_tag
from tests.conftest import csrf_header, seed_admin_user, wait_until
from tests.unit.proxy.test_kv_isolation import AUTH_A, CHAT, _go, _kv_on, _nk, _setup
from tests.unit.proxy.test_kv_tier_record import FakeKv


def _secrets(client) -> list[str]:
    nk = _nk(client)
    scope = isolation_scope(nk, "tokA", 0)
    salt = gpu_cache_salt(nk, scope, "client-salt-1")
    vals = [scope.hex(), scope_tag(scope).hex(), tenant_tag(nk, "tokA").hex(), salt]
    return vals + [salt.removeprefix("lmw1:")]


def _assert_clean(text: str, secrets: list[str], where: str) -> None:
    for s in secrets:
        assert s not in text, f"a derived value leaked into {where}"
        assert s.upper() not in text, f"a derived value leaked into {where}"


def test_no_derived_value_leaves_the_proxy(tmp_data_dir, client, caplog):
    _setup(client, tmp_data_dir)
    _kv_on(client, tmp_data_dir)
    fake = FakeKv({"ram_tokens": 0, "nvme_tokens": 0, "load_s": 0.0})
    client.app.state.kv_store.client = lambda: fake
    db = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db) as c:
        c.execute("UPDATE api_tokens SET capture_until = ? WHERE id = 'tokA'", (time.time() + 600,))
        c.commit()
    secrets = _secrets(client)
    body = {**CHAT, "cache_salt": "client-salt-1"}
    with caplog.at_level(logging.DEBUG):
        r, send = _go(client, AUTH_A, body)
        assert r.status_code == 200
        # The engine got the derived values (that is the point) ...
        sent = send.call_args.args[0].content.decode()
        assert secrets[0] in sent and secrets[3] in sent
        rid = r.headers[REQUEST_ID_HEADER]

        def _row():
            with sqlite3.connect(db) as c:
                c.row_factory = sqlite3.Row
                row = c.execute("SELECT * FROM request_history WHERE id = ?", (rid,)).fetchone()
                return dict(row) if row is not None else None

        row = wait_until(_row, what="history row")
        rec = wait_until(lambda: client.app.state.capture_store.get(rid), what="capture")
    # ... and nothing else did.
    for rec_ in caplog.records:
        text = rec_.getMessage() + " " + repr(rec_.args) + " " + (rec_.exc_text or "")
        _assert_clean(text, secrets, f"log record {rec_.name}")
    _assert_clean(json.dumps(row), secrets, "the history row")
    assert row["kv_isolated"] == 1
    cap = json.dumps(rec)
    _assert_clean(cap, secrets, "the capture")
    assert json.loads(rec["request_body"])["cache_salt"] == "client-salt-1"
    if rec.get("engine_body"):
        assert json.loads(rec["engine_body"])["cache_salt"] == "client-salt-1"
    # The admin read route serves the same record.
    seed_admin_user(db)
    tok = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    hdrs = {"Authorization": f"Bearer {tok.json()['access_token']}", **csrf_header(client)}
    got = client.get(f"/api/capture/{rid}", headers=hdrs)
    assert got.status_code == 200
    _assert_clean(got.text, secrets, "the capture route")
    argv = client.get("/api/models/qwen/effective-argv", headers=hdrs)
    assert argv.status_code == 200, argv.text
    _assert_clean(argv.text, secrets, "effective-argv")


# -- fix round 1: an engine error that echoes the request ------------------------

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

import pytest  # noqa: E402


def _echo_400(req, **kw):
    """vLLM's RequestValidationError for a whole-model validator: the request dict in a 400."""
    raw = json.loads(req.content)
    body = json.dumps({"detail": [{"type": "value_error", "input": raw}]}).encode()
    resp = MagicMock()
    resp.status_code = 400
    resp.headers = {"content-type": "application/json"}
    resp.aread = AsyncMock(return_value=body)

    async def _it():
        yield body[:40]
        yield body[40:]

    resp.aiter_bytes = _it
    resp.aclose = AsyncMock()
    return resp


@pytest.mark.parametrize("stream", [False, True])
def test_an_echoed_error_body_never_carries_a_derived_value(tmp_data_dir, client, stream):
    _setup(client, tmp_data_dir)
    _kv_on(client, tmp_data_dir)
    db = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db) as c:
        c.execute("UPDATE api_tokens SET capture_until = ? WHERE id = 'tokA'", (time.time() + 600,))
        c.commit()
    secrets = _secrets(client)
    body = {**CHAT, "cache_salt": "client-salt-1", **({"stream": True} if stream else {})}
    send = AsyncMock(side_effect=_echo_400)
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post("/v1/chat/completions", headers=AUTH_A, json=body)
    assert send.called and r.status_code == 400
    sent = send.call_args.args[0].content.decode()
    assert secrets[3] in sent  # the engine did get them
    _assert_clean(r.text, secrets, "the client's error body")
    assert "<redacted>" in r.text
    rid = r.headers[REQUEST_ID_HEADER]
    rec = wait_until(lambda: client.app.state.capture_store.get(rid), what="capture")
    _assert_clean(json.dumps(rec), secrets, "the capture")
