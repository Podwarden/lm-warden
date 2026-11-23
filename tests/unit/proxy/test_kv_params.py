"""KV-cache tier (step 1): the ``kv_transfer_params`` the engine's connector reads.

For a model with ``kv_store`` on, a client's own ``kv_transfer_params`` never reach the
engine (vLLM would act on ``do_remote_prefill`` / ``do_remote_decode``); LM Warden sets
its own, after the cobs2 ids and the cache-obs chain are computed. With ``kv_store``
off the client's params pass through: an operator's own connector needs them.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from unittest.mock import AsyncMock, patch

from app.proxy.kv import set_kv_params, strip_client_kv_params
from app.proxy.model_cache import invalidate_all
from app.proxy.request_id import REQUEST_ID_HEADER
from tests.conftest import wait_until
from tests.unit.proxy.test_messages import AUTH, _json_upstream, _ready

OK = {
    "id": "c1",
    "object": "chat.completion",
    "model": "qwen",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
CHAT = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
CLIENT_KV = {"do_remote_prefill": True, "remote_host": "evil", "lmw_req": "spoofed"}


# -- pure -----------------------------------------------------------------------


def test_strip_removes_and_reports():
    body: dict[str, Any] = {"messages": [], "kv_transfer_params": {"do_remote_prefill": True}}
    assert strip_client_kv_params(body) is True
    assert body == {"messages": []}
    assert strip_client_kv_params(body) is False
    assert body == {"messages": []}


def test_strip_tolerates_a_non_dict_body():
    assert strip_client_kv_params([1, 2]) is False  # type: ignore[arg-type]


def test_set_kv_params_shape():
    body: dict[str, Any] = {"messages": []}
    set_kv_params(body, req_id="r1", conv="c1", session=None)
    assert body["kv_transfer_params"] == {
        "lmw_req": "r1",
        "lmw_tenant": "",
        "lmw_scope": "",
        "lmw_conv": "c1",
        "lmw_session": None,
    }


def test_set_kv_params_carries_tenant_and_scope_hex():
    body: dict[str, Any] = {"messages": []}
    set_kv_params(body, req_id="r1", conv=None, session=None, tenant="ab" * 16, scope="cd" * 32)
    kv = body["kv_transfer_params"]
    assert kv["lmw_tenant"] == "ab" * 16 and kv["lmw_scope"] == "cd" * 32


# -- forward --------------------------------------------------------------------


def _set(db_path, sql: str) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute(sql)
        db.commit()


def _kv_on(client, tmp_data_dir, tier: str = "ram") -> None:
    # The real supervisor would try to start a daemon once a row wants one.
    client.portal.call(client.app.state.kv_store.stop)
    _set(tmp_data_dir / "vllm-warden.db", f"UPDATE models SET kv_store='{tier}'")
    invalidate_all()  # the proxy's model-row snapshot


def _go(client, body, *, path="/v1/chat/completions", upstream=None):
    send = AsyncMock(return_value=upstream or _json_upstream(OK))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(path, headers=AUTH, json=body)
    return r, send


def _sent(send) -> bytes:
    return send.call_args.args[0].content


def test_off_body_is_byte_identical(tmp_data_dir, client):
    """Golden: with kv_store off, the proxy's only edit is today's priority field."""
    _ready(client, tmp_data_dir)
    r, send = _go(client, CHAT)
    assert r.status_code == 200
    assert _sent(send) == json.dumps({**CHAT, "priority": -5}).encode()


def test_client_kv_params_pass_through_on_chat_with_kv_off(tmp_data_dir, client):
    """F5: with kv_store off the engine's KV connector, if any, is the operator's (NIXL or
    disaggregated prefill through extra_args), and it needs the client's params."""
    _ready(client, tmp_data_dir)
    r, send = _go(client, {**CHAT, "kv_transfer_params": CLIENT_KV})
    assert r.status_code == 200
    sent = json.loads(_sent(send))
    assert sent["kv_transfer_params"] == CLIENT_KV


def test_client_kv_params_are_replaced_on_chat_with_kv_on(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _kv_on(client, tmp_data_dir)
    r, send = _go(client, {**CHAT, "kv_transfer_params": CLIENT_KV})
    assert r.status_code == 200
    kv = json.loads(_sent(send))["kv_transfer_params"]
    assert "do_remote_prefill" not in kv and "remote_host" not in kv
    assert kv["lmw_req"] == r.headers[REQUEST_ID_HEADER]


def test_client_kv_params_pass_through_on_completions_with_kv_off(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    body = {"model": "qwen", "prompt": "hi", "kv_transfer_params": CLIENT_KV}
    r, send = _go(client, body, path="/v1/completions")
    assert r.status_code == 200
    assert json.loads(_sent(send))["kv_transfer_params"] == CLIENT_KV


def test_client_kv_params_are_replaced_on_completions_with_kv_on(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _kv_on(client, tmp_data_dir)
    body = {"model": "qwen", "prompt": "hi", "kv_transfer_params": CLIENT_KV}
    r, send = _go(client, body, path="/v1/completions")
    assert r.status_code == 200
    kv = json.loads(_sent(send))["kv_transfer_params"]
    assert "do_remote_prefill" not in kv and "remote_host" not in kv
    assert kv["lmw_req"] == r.headers[REQUEST_ID_HEADER]


def test_client_kv_params_are_stripped_on_messages(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    body = {
        "model": "qwen",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "kv_transfer_params": CLIENT_KV,
    }
    r, send = _go(client, body, path="/v1/messages")
    assert r.status_code == 200, r.text
    assert "kv_transfer_params" not in json.loads(_sent(send))


def test_client_kv_params_are_stripped_on_responses(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    body = {
        "model": "qwen",
        "input": "hi",
        "max_output_tokens": 16,
        "kv_transfer_params": CLIENT_KV,
    }
    r, send = _go(client, body, path="/v1/responses")
    assert r.status_code == 200, r.text
    assert "kv_transfer_params" not in json.loads(_sent(send))


def test_messages_with_kv_on_carries_lmw_req(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _kv_on(client, tmp_data_dir)
    body = {
        "model": "qwen",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "kv_transfer_params": CLIENT_KV,
    }
    r, send = _go(client, body, path="/v1/messages")
    assert r.status_code == 200, r.text
    kv = json.loads(_sent(send))["kv_transfer_params"]
    assert set(kv) == {"lmw_req", "lmw_tenant", "lmw_scope", "lmw_conv", "lmw_session"}
    assert kv["lmw_req"] == r.headers[REQUEST_ID_HEADER]


def test_kv_on_sets_lmw_req_to_the_request_id(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _kv_on(client, tmp_data_dir, "ram+nvme")
    r, send = _go(client, CHAT)
    assert r.status_code == 200
    sent = json.loads(_sent(send))
    kv = sent.pop("kv_transfer_params")
    assert kv["lmw_req"] == r.headers[REQUEST_ID_HEADER]
    assert kv["lmw_scope"] == ""
    # Nothing else in the body moved.
    assert sent == {**CHAT, "priority": -5}


def _history(db_path, rid: str):
    def _row():
        with sqlite3.connect(db_path) as db:
            return db.execute(
                "SELECT cobs2_head, cobs2_node, cobs2_conv, cobs2_session "
                "FROM request_history WHERE id = ?",
                (rid,),
            ).fetchone()

    return wait_until(_row, what="history row")


def test_kv_on_changes_no_salt_no_cobs2_ids_and_no_cache_obs_chain(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    db_path = tmp_data_dir / "vllm-warden.db"
    chains: list[Any] = []
    idx = client.app.state.cache_index
    orig_match = idx.match

    def spy(model_id, epoch, chain, **kw):
        chains.append(chain)
        return orig_match(model_id, epoch, chain, **kw)

    idx.match = spy
    body = {**CHAT, "cache_salt": "salt-1"}
    r_off, send_off = _go(client, body)
    _kv_on(client, tmp_data_dir)
    r_on, send_on = _go(client, body)
    assert r_off.status_code == r_on.status_code == 200
    off, on = json.loads(_sent(send_off)), json.loads(_sent(send_on))
    assert on["cache_salt"] == off["cache_salt"] == "salt-1"
    kv = on.pop("kv_transfer_params")
    assert on == off
    row_off = _history(db_path, r_off.headers[REQUEST_ID_HEADER])
    row_on = _history(db_path, r_on.headers[REQUEST_ID_HEADER])
    assert row_on[:2] == row_off[:2] and row_on[0] is not None
    # The informational ids are the turn's own cobs2 conv and session.
    assert kv["lmw_conv"] == row_on[2]
    assert kv["lmw_session"] == row_on[3]
    # match runs twice per request (all keys, own key): the same chain each time.
    assert len(chains) == 4 and chains[0] == chains[2] and chains[0] is not None


def test_kv_client_is_none_while_the_engine_runs_with_its_tier_off():
    """F3: an engine whose connector is off records NULL tiers, never GPU-only turns."""
    from types import SimpleNamespace

    from app.proxy.kv import kv_client

    daemon = object()
    state = {"state": "on"}
    app_state = SimpleNamespace(
        kv_store=SimpleNamespace(client=lambda: daemon),
        supervisor=SimpleNamespace(kv_tier=lambda mid: dict(state)),
    )
    model = SimpleNamespace(id="m1", kv_store="ram")
    assert kv_client(app_state, model) is daemon
    state["state"] = "off_until_reload"
    assert kv_client(app_state, model) is None
