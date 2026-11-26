"""``GET /api/me/engine`` (owner decision 2026-11-26, the leanprompt handoff of 2026-11-26):
the engine facts of the models an inference key may use, with the same credential an app uses
for inference."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app.db.repos.tokens import hash_token
from tests.conftest import bearer

URL = "/api/me/engine"
EPOCH = "run:1"
SPEC = json.dumps(["--speculative-config", json.dumps({"method": "qwen3_5_mtp"})])


def _model(
    c: sqlite3.Connection,
    mid: str,
    *,
    status: str = "loaded",
    args: str = "[]",
    backend: str = "vllm",
    max_len: int | None = 32768,
    dp: int = 1,
) -> None:
    c.execute(
        "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
        "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, trust_remote_code, "
        "extra_args, status, pulled_bytes, backend, data_parallel_size) "
        "VALUES (?, ?, 'org/repo', 'main', '[0]', 1, 'auto', ?, 0.9, 0, ?, ?, 0, ?, ?)",
        (mid, mid, max_len, args, status, backend, dp),
    )


def _key(c: sqlite3.Connection, kid: str, **cols: Any) -> str:
    plain = "vw_" + kid.ljust(33, "k")
    cols = {"allowed_models": None, **cols}
    c.execute(
        f"INSERT INTO api_tokens(id, name, prefix, hash, scope, {', '.join(cols)}) "
        f"VALUES (?, ?, ?, ?, 'inference', {', '.join('?' * len(cols))})",
        (kid, "fake-" + kid, plain[:8], hash_token(plain), *cols.values()),
    )
    return plain


@pytest.fixture
def env(client: Any, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import app.proxy.routes_dp as routes_dp

    monkeypatch.setattr(routes_dp, "engine_epoch", lambda state, m: EPOCH)
    client.get("/healthz")
    db = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db) as c:
        _model(c, "big", args=SPEC, dp=4)
        _model(c, "small", args="[]")
        _model(c, "idle", status="pulled", max_len=None)
        _model(c, "cpp", backend="llamacpp")
        keys = {
            "all": _key(c, "kall"),
            "one": _key(c, "kone", allowed_models="small"),
            "none": _key(c, "knone", allowed_models="no-such-model"),
            "off": _key(c, "koff", engine_facts=0),
            "revoked": _key(c, "krev", revoked_at="2000-01-01 00:00:00"),
            "expired": _key(c, "kexp", expires_at="2000-01-01 00:00:00"),
            "paused": _key(c, "kpau", paused_at="2000-01-01 00:00:00"),
        }
        c.commit()
    sizes = client.app.state.engine_block_sizes
    sizes.set("big", EPOCH, 1584, drops_last_block=True, match_unit=16, prefix_caching=True)
    sizes.set("small", EPOCH, 16, prefix_caching=True)  # a classic small page
    return {"client": client, "keys": keys}


def _get(env: dict[str, Any], who: str) -> Any:
    return env["client"].get(URL, headers=bearer(env["keys"][who]))


def _by_model(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["model"]: m for m in body["models"]}


def test_a_key_sees_only_its_allowed_models(env: dict[str, Any]) -> None:
    assert sorted(_by_model(_get(env, "one").json())) == ["small"]
    assert _get(env, "none").json()["models"] == []
    assert sorted(_by_model(_get(env, "all").json())) == ["big", "cpp", "idle", "small"]


def test_the_fields_of_a_loaded_model_exactly(env: dict[str, Any]) -> None:
    body = _get(env, "all").json()
    assert body["v"] == 1 and isinstance(body["warden_version"], str)
    assert set(body) == {"v", "warden_version", "models"}
    big = _by_model(body)["big"]
    assert big == {
        "model": "big",
        "backend": "vllm",
        "running": True,
        "prefix_caching": True,
        "engine_block_tokens": 1584,
        "engine_match_unit_tokens": 16,
        "engine_drops_last_block": True,
        "speculative": "mtp",
        "max_model_len": 32768,
        "replicas": 4,
        "cache_aware_routing": True,
        "unknown": [],
    }


def test_a_block_of_16_is_reported_as_16(env: dict[str, Any]) -> None:
    small = _by_model(_get(env, "all").json())["small"]
    assert small["engine_block_tokens"] == 16
    assert small["engine_match_unit_tokens"] == 16  # the block, when no unit is set
    assert small["engine_drops_last_block"] is False
    assert small["speculative"] is None and "speculative" not in small["unknown"]
    assert small["replicas"] == 1 and small["cache_aware_routing"] is False


def test_a_model_with_no_traffic_is_listed(env: dict[str, Any]) -> None:
    with sqlite3.connect(env["client"].app.state.settings.db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM request_history").fetchone() == (0,)
    assert "big" in _by_model(_get(env, "all").json())


def test_an_unloaded_model_has_nulls_and_says_which_are_unknown(env: dict[str, Any]) -> None:
    idle = _by_model(_get(env, "all").json())["idle"]
    assert idle["running"] is False
    for field in (
        "prefix_caching",
        "engine_block_tokens",
        "engine_match_unit_tokens",
        "engine_drops_last_block",
        "speculative",
        "max_model_len",
    ):
        assert idle[field] is None, field
    assert idle["unknown"] == [
        "prefix_caching",
        "engine_block_tokens",
        "engine_match_unit_tokens",
        "engine_drops_last_block",
        "speculative",
        "max_model_len",
    ]
    assert idle["replicas"] == 1  # configuration, known without a load


def test_a_backend_that_does_not_say_is_unknown_never_guessed(env: dict[str, Any]) -> None:
    cpp = _by_model(_get(env, "all").json())["cpp"]
    assert cpp["running"] is True and cpp["backend"] == "llamacpp"
    assert cpp["engine_block_tokens"] is None
    assert "engine_block_tokens" in cpp["unknown"] and "speculative" in cpp["unknown"]


def test_a_loaded_engine_that_has_not_reported_yet_is_unknown(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    env["client"].app.state.engine_block_sizes._by_model.clear()
    big = _by_model(_get(env, "all").json())["big"]
    assert big["engine_block_tokens"] is None
    assert {"engine_block_tokens", "engine_match_unit_tokens", "engine_drops_last_block"} <= set(
        big["unknown"]
    )
    assert big["speculative"] == "mtp"  # from the launched args, not a guess


@pytest.mark.parametrize("who", ["revoked", "expired"])
def test_a_dead_key_gets_the_401_inference_gives(env: dict[str, Any], who: str) -> None:
    got = _get(env, who)
    chat = env["client"].post(
        "/v1/chat/completions",
        json={"model": "small", "messages": [{"role": "user", "content": "x"}]},
        headers=bearer(env["keys"][who]),
    )
    assert got.status_code == chat.status_code == 401
    assert got.json() == chat.json()


def test_a_paused_key_gets_the_403_inference_gives(env: dict[str, Any]) -> None:
    got = _get(env, "paused")
    assert got.status_code == 403 and got.json()["detail"] == "token paused"


def test_no_key_and_a_wrong_key_are_401(env: dict[str, Any]) -> None:
    c = env["client"]
    assert c.get(URL).status_code == 401
    assert c.get(URL, headers=bearer("vw_" + "z" * 33)).status_code == 401
    assert c.get(URL, headers=bearer("not-a-key")).status_code == 401


def test_a_key_without_the_permission_is_403_with_a_message(env: dict[str, Any]) -> None:
    got = _get(env, "off")
    assert got.status_code == 403
    assert "engine facts" in got.json()["detail"]


def test_the_answer_has_no_traffic_other_keys_or_hosts(env: dict[str, Any]) -> None:
    with sqlite3.connect(env["client"].app.state.settings.db_path) as c:
        c.execute(
            "INSERT INTO request_history(id, finished_at, model_id, model, token_name, token_id,"
            " client_ip, prompt_tokens, completion_tokens, duration_s, ttft_s, finish_reason,"
            " orphan, started_iso, variant_id, cached_tokens) VALUES ('r1', 1.0, 'big', 'big',"
            " 'fake-kall', 'kall', '192.0.2.9', 1234567, 7654321, 1, 1, 'stop', 0,"
            " '2026-01-01T00:00:00Z', 'v1', 99)"
        )
        c.commit()
    raw = _get(env, "all").text
    for needle in ("1234567", "7654321", "192.0.2.9", "fake-kall", "kall", "localhost", "127.0.0"):
        assert needle not in raw
    for m in _get(env, "all").json()["models"]:
        assert not {"requests", "tokens", "host", "port", "cached_tokens"} & set(m)


def test_it_is_in_the_openapi_schema(env: dict[str, Any]) -> None:
    spec = env["client"].app.openapi()
    op = spec["paths"][URL]["get"]
    assert op["security"] == []  # an inference key, not the admin credential
    assert "EngineFacts" in json.dumps(op)
