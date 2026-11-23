"""Isolated keys on backends that cannot isolate their cache (plan 2, task 10, O4).

llama.cpp and MLX reuse a slot's prefix whatever the ``cache_salt``, so an isolated key
is refused there with a 403, decided first in the pre-flight: before the 503s, the
guard, the live row, admission and any engine call. ``/v1/models`` tells an isolated
key which models it can use; the admin models list says it for every model.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from typing import Any

import pytest

from tests.conftest import jwt_login
from tests.unit.proxy.test_kv_isolation import (
    AUTH_A,
    AUTH_B,
    ROUTES,
    _db,
    _go,
    _refused,
    _sent_json,
    _setup,
)

MSG = "this key is isolated and model qwen cannot isolate its cache"


def _backend(tmp_data_dir, backend: str | None, mid: str = "qwen") -> None:
    with sqlite3.connect(_db(tmp_data_dir)) as db:
        db.execute("UPDATE models SET backend = ? WHERE id = ?", (backend, mid))
        db.commit()
    _drop_model_cache()


@pytest.fixture
def with_mlx(monkeypatch):
    """The linux registry has no MLX; register it as the apple profile would."""
    from app.runtime.backends import registry
    from app.runtime.backends.mlx import MlxBackend

    monkeypatch.setitem(registry._BACKENDS, "mlx", MlxBackend())


def _drop_model_cache() -> None:
    from app.proxy.model_cache import invalidate_all

    invalidate_all()


# -- pure -----------------------------------------------------------------------


class _M:
    def __init__(self, backend: Any) -> None:
        self.backend = backend


def test_can_isolate_per_backend(with_mlx) -> None:
    from app.proxy.kv_scope import cache_isolation, can_isolate

    assert can_isolate(_M("vllm")) is True
    assert can_isolate(_M(None)) is True  # NULL is vLLM on linux (D6)
    assert can_isolate(_M("llamacpp")) is False
    assert can_isolate(_M("mlx")) is False
    assert can_isolate(_M("no-such-engine")) is False  # fail closed
    assert can_isolate(object()) is True  # no backend attribute: the default
    assert (cache_isolation(_M("vllm")), cache_isolation(_M("llamacpp"))) == (
        "isolated",
        "unsupported",
    )


def test_can_isolate_is_false_on_any_registry_error(monkeypatch) -> None:
    from app.proxy.kv_scope import can_isolate
    from app.runtime.backends import registry

    def boom(_name):
        raise RuntimeError("registry broke")

    monkeypatch.setattr(registry, "get", boom)
    assert can_isolate(_M("vllm")) is False


def _marked() -> dict[str, Any]:
    cc = {"type": "ephemeral"}
    return {
        "model": "claude-x",
        "max_tokens": 8,
        "cache_control": dict(cc),
        "system": [{"type": "text", "text": "sys", "cache_control": dict(cc)}],
        "tools": [{"name": "t", "input_schema": {"type": "object"}, "cache_control": dict(cc)}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "a", "cache_control": dict(cc)}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "x",
                        "content": [{"type": "text", "text": "r", "cache_control": dict(cc)}],
                        "cache_control": dict(cc),
                    }
                ],
            },
            {"role": "assistant", "content": "plain"},
        ],
    }


def test_strip_cache_control_removes_every_mark() -> None:
    from app.proxy.kv_scope import strip_cache_control

    body = _marked()
    assert strip_cache_control(body) is True
    assert "cache_control" not in json.dumps(body)
    # Only the marks went: everything else is as it was.
    want = _marked()
    want.pop("cache_control")
    for part in (want["system"], want["tools"], want["messages"][0]["content"]):
        for b in part:
            b.pop("cache_control")
    tr = want["messages"][1]["content"][0]
    tr.pop("cache_control")
    tr["content"][0].pop("cache_control")
    assert body == want
    assert strip_cache_control(body) is False


def test_strip_cache_control_on_odd_shapes_and_batches() -> None:
    from app.proxy.kv_scope import strip_cache_control

    for body in (
        {},
        {"system": "text", "messages": "x", "tools": None},
        {"messages": [1, None, {"content": 5}], "system": [1, "s"]},
    ):
        before = copy.deepcopy(body)
        assert strip_cache_control(body) is False and body == before
    batch = {"requests": [{"custom_id": "1", "params": _marked()}, "junk", {"params": 3}]}
    assert strip_cache_control(batch) is True
    assert "cache_control" not in json.dumps(batch)


def test_cannot_isolate_message() -> None:
    from app.proxy.kv_scope import cannot_isolate_message

    assert cannot_isolate_message("qwen") == MSG


# -- forward --------------------------------------------------------------------


@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize("backend", ["llamacpp", "mlx"])
def test_an_isolated_key_is_refused_on_a_backend_that_cannot_isolate(
    tmp_data_dir, client, with_mlx, route, backend
):
    _setup(client, tmp_data_dir)
    _backend(tmp_data_dir, backend)
    path, body = ROUTES[route]
    r = _refused(client, AUTH_A, body, path)  # no upstream call, never admitted
    assert r.status_code == 403, r.text
    assert MSG in r.text


@pytest.mark.parametrize("route", list(ROUTES))
def test_a_sharing_key_still_works_on_llamacpp(tmp_data_dir, client, route):
    _setup(client, tmp_data_dir)
    _backend(tmp_data_dir, "llamacpp")
    path, body = ROUTES[route]
    r, send = _go(client, AUTH_B, body, path)
    assert r.status_code == 200, r.text
    assert send.called
    assert "cache_salt" not in _sent_json(send)  # llama.cpp: nothing to rewrite


def test_an_unknown_backend_refuses_the_isolated_key(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    _backend(tmp_data_dir, "no-such-engine")
    r = _refused(client, AUTH_A, ROUTES["chat"][1])
    assert r.status_code == 403 and MSG in r.text


def test_vllm_serves_the_isolated_key(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    r, send = _go(client, AUTH_A, ROUTES["chat"][1])
    assert r.status_code == 200
    assert _sent_json(send)["cache_salt"].startswith("lmw1:")


def test_the_403_comes_before_the_503s(tmp_data_dir, client):
    """No network key would be a 503; the backend refusal is decided first."""
    _setup(client, tmp_data_dir)
    _backend(tmp_data_dir, "llamacpp")
    client.app.state.cobs2_key = None
    r = _refused(client, AUTH_A, ROUTES["chat"][1])
    assert r.status_code == 403 and MSG in r.text


def test_the_403_comes_before_the_guard(tmp_data_dir, client, monkeypatch):
    import app.proxy.routes as routes

    seen: list[Any] = []

    async def inspect(*a, **k):
        seen.append(1)
        raise AssertionError("the guard ran")

    _setup(client, tmp_data_dir)
    _backend(tmp_data_dir, "llamacpp")
    monkeypatch.setattr(routes.guard_proxy, "guard_for", lambda *a, **k: object())
    monkeypatch.setattr(routes, "_guard_inspect", inspect)
    r = _refused(client, AUTH_A, ROUTES["chat"][1])
    assert r.status_code == 403 and seen == []


# -- the models lists -----------------------------------------------------------


def _add_llama_model(tmp_data_dir) -> None:
    with sqlite3.connect(_db(tmp_data_dir)) as db:
        db.execute("CREATE TEMP TABLE m AS SELECT * FROM models WHERE id = 'qwen'")
        db.execute("UPDATE m SET id = 'llama', served_model_name = 'llama', backend = 'llamacpp'")
        db.execute("INSERT INTO models SELECT * FROM m")
        db.commit()
    _drop_model_cache()


def test_v1_models_flags_isolation_only_for_an_isolated_key(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    _add_llama_model(tmp_data_dir)
    shared = client.get("/v1/models", headers=AUTH_B).json()["data"]
    assert {e["id"] for e in shared} == {"qwen", "llama"}
    assert all(not any(k.startswith("lmw_") for k in e) for e in shared)
    iso = client.get("/v1/models", headers=AUTH_A).json()["data"]
    assert {e["id"]: e["lmw_cache_isolation"] for e in iso} == {
        "qwen": "isolated",
        "llama": "unsupported",
    }
    # Apart from the flag, the isolated key's list is the sharing key's.
    for e in iso:
        e.pop("lmw_cache_isolation")
    assert sorted(iso, key=lambda e: e["id"]) == sorted(shared, key=lambda e: e["id"])


def test_admin_models_list_carries_cache_isolation(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    _add_llama_model(tmp_data_dir)
    from tests.conftest import seed_admin_user

    seed_admin_user(_db(tmp_data_dir))
    auth = jwt_login(client)
    models = client.get("/api/models", headers=auth).json()["models"]
    assert {m["id"]: m["cache_isolation"] for m in models} == {
        "qwen": "isolated",
        "llama": "unsupported",
    }
    one = client.get("/api/models/llama", headers=auth).json()
    assert one["cache_isolation"] == "unsupported"


# -- the running engine decides, not the row --------------------------------------


def _launched_as(client, monkeypatch, backend: str | None) -> None:
    """The supervisor's launch record says the engine runs ``backend``."""
    import dataclasses as dc

    from app.runtime.variants import variant_of

    sup = client.app.state.supervisor

    def get_variant(model_id):
        base = variant_of(_ROWS[model_id], None)
        return dc.replace(base, descriptor={**base.descriptor, "backend": backend})

    monkeypatch.setattr(sup, "get_variant", get_variant, raising=False)


_ROWS: dict[str, Any] = {}


def _remember_row(tmp_data_dir) -> None:
    import asyncio

    from app.db.database import open_db
    from app.db.repos.models import ModelRepo

    async def go():
        async with open_db(_db(tmp_data_dir)) as db:
            return await ModelRepo(db).get("qwen")

    _ROWS["qwen"] = asyncio.run(go())


def test_a_row_edited_to_vllm_cannot_unlock_a_running_llamacpp_engine(
    tmp_data_dir, client, monkeypatch
):
    _setup(client, tmp_data_dir)  # the row says vLLM
    _remember_row(tmp_data_dir)
    _launched_as(client, monkeypatch, "llamacpp")
    r = _refused(client, AUTH_A, ROUTES["chat"][1])
    assert r.status_code == 403 and MSG in r.text
    iso = client.get("/v1/models", headers=AUTH_A).json()["data"]
    assert iso[0]["lmw_cache_isolation"] == "unsupported"


def test_a_running_vllm_engine_serves_and_salts_whatever_the_row_says(
    tmp_data_dir, client, monkeypatch
):
    _setup(client, tmp_data_dir)
    _remember_row(tmp_data_dir)
    _backend(tmp_data_dir, "llamacpp")  # edited while the vLLM engine runs
    _launched_as(client, monkeypatch, "vllm")
    r, send = _go(client, AUTH_A, ROUTES["chat"][1])
    assert r.status_code == 200, r.text
    assert _sent_json(send)["cache_salt"].startswith("lmw1:")


def test_an_unknown_running_backend_fails_closed(tmp_data_dir, client, monkeypatch):
    _setup(client, tmp_data_dir)
    _remember_row(tmp_data_dir)
    _launched_as(client, monkeypatch, None)  # the launch record names no backend
    r = _refused(client, AUTH_A, ROUTES["chat"][1])
    assert r.status_code == 403 and MSG in r.text


def test_a_missing_network_key_is_logged_once(tmp_data_dir, client, caplog):
    import logging

    import app.proxy.routes as routes

    _setup(client, tmp_data_dir)
    routes._warned_no_network_key = False
    key = client.app.state.cobs2_key
    client.app.state.cobs2_key = None
    with caplog.at_level(logging.WARNING, logger="app.proxy.routes"):
        for _ in range(3):
            _go(client, AUTH_B, ROUTES["chat"][1])
            _refused(client, AUTH_A, ROUTES["chat"][1])
    hits = [r for r in caplog.records if "network key is unavailable" in r.getMessage()]
    assert len(hits) == 1
    # Back, then gone again: warned again.
    client.app.state.cobs2_key = key
    _go(client, AUTH_B, ROUTES["chat"][1])
    client.app.state.cobs2_key = None
    with caplog.at_level(logging.WARNING, logger="app.proxy.routes"):
        _go(client, AUTH_B, ROUTES["chat"][1])
    hits = [r for r in caplog.records if "network key is unavailable" in r.getMessage()]
    assert len(hits) == 2


def test_running_backend_fails_closed_on_errors() -> None:
    from app.proxy.kv_scope import can_isolate_running, running_backend

    class Boom:
        @property
        def descriptor(self):
            raise RuntimeError("x")

    assert running_backend(None, object(), Boom()) is None
    assert can_isolate_running(None, object(), Boom()) is False


def test_key_holder_sees_which_of_their_models_can_isolate(tmp_data_dir, client):
    """The key settings page marks the models an isolated key would be refused on, and the
    isolate dialog names them: ``/api/me/key`` lists the loaded models the key may use with
    the same answer ``/v1/models`` gives an isolated key, for a sharing key too."""
    from tests.unit.proxy.test_kv_isolation import KEY_A
    from tests.unit.stats.test_forest_routes import _forest_login

    _setup(client, tmp_data_dir)
    _add_llama_model(tmp_data_dir)
    k = _forest_login(client, KEY_A)
    d = client.get("/api/me/key", headers=k).json()
    assert {m["id"]: m["cache_isolation"] for m in d["models"]} == {
        "qwen": "isolated",
        "llama": "unsupported",
    }
    # The privacy property: a model the key may not use is never listed.
    with sqlite3.connect(_db(tmp_data_dir)) as db:
        db.execute("UPDATE api_tokens SET allowed_models = 'qwen' WHERE allowed_models IS NULL")
        db.commit()
    d = client.get("/api/me/key", headers=k).json()
    assert [m["id"] for m in d["models"]] == ["qwen"]
