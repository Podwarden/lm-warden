"""GET /api/connect/clients (connect-clients plan, Task 1)."""

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.router_rules import RouterRuleRepo
from app.db.repos.settings import SettingsRepo
from tests.conftest import bearer, csrf_header, jwt_login, seed_admin_token

URL = "/api/connect/clients"
_CAPS = ("supports_tools", "supports_vision", "supports_reasoning")


def _model(mid: str, name: str, status: str = "loaded", **over: Any) -> ModelRow:
    row = ModelRow(
        id=mid,
        served_model_name=name,
        hf_repo="a/b",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=65536,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status=status,
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
    )
    return dataclasses.replace(row, **over)


def _run(db_path: Path, fn: Any) -> Any:
    async def go() -> Any:
        async with open_db(db_path) as db:
            return await fn(db)

    return asyncio.run(go())


def _seed(db_path: Path, *rows: ModelRow) -> None:
    async def fn(db: Any) -> None:
        for r in rows:
            await ModelRepo(db).insert(r)
            # insert() does not write the capability flags; the settings
            # PATCH path does, through update_fields.
            caps = {k: getattr(r, k) for k in _CAPS if getattr(r, k) is not None}
            if caps:
                await ModelRepo(db).update_fields(r.id, caps)

    _run(db_path, fn)


def _set(db_path: Path, key: str, value: str) -> None:
    _run(db_path, lambda db: SettingsRepo(db).set(key, value))


def _rule(db_path: Path, pattern: str, target: str, enabled: bool = True) -> None:
    _run(db_path, lambda db: RouterRuleRepo(db).create(pattern, target, enabled))


@pytest.fixture
def api(client, seeded_db):
    return client, jwt_login(client), seeded_db


def _client(body: dict[str, Any], cid: str) -> dict[str, Any]:
    return next(c for c in body["clients"] if c["id"] == cid)


def _file(body: dict[str, Any], cid: str, mid: str, fid: str) -> dict[str, Any]:
    mode = next(m for m in _client(body, cid)["modes"] if m["id"] == mid)
    return next(f for f in mode["files"] if f["id"] == fid)


def test_unauthenticated_is_401(client, seeded_db):
    assert client.get(URL).status_code == 401


def test_an_admin_token_is_accepted(client, seeded_db):
    _, secret = seed_admin_token(seeded_db)
    r = client.get(URL, headers=bearer(secret))
    assert r.status_code == 200, r.text


def test_an_inference_key_is_refused(api):
    c, h, _ = api
    made = c.post("/api/tokens", json={"name": "k"}, headers={**h, **csrf_header(c)})
    assert made.status_code == 201, made.text
    assert c.get(URL, headers=bearer(made.json()["plaintext"])).status_code == 401


def test_the_empty_warden(api):
    c, h, _ = api
    r = c.get(URL, headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["origin"] == "http://testserver"
    assert body["origin_source"] == "request"
    assert body["header_name"] == "X-LMWarden-Key"
    assert body["key_placeholder"] == "vw_YOUR_KEY"
    assert body["router"] == {"enabled": False, "passthrough_unmatched": True, "rules": []}
    assert body["models"] == []
    assert body["selected_model"] is None
    assert [c["id"] for c in body["clients"]][0] == "claude-code"
    # No loaded model: OpenAI snippets carry the placeholder name.
    assert "your-served-model-name" in _file(body, "opencode", "default", "config")["rendered"]


def test_origin_from_public_url(api):
    c, h, db = api
    _set(db, "public_url", "https://warden.example")
    body = c.get(URL, headers=h).json()
    assert body["origin"] == "https://warden.example"
    assert body["origin_source"] == "public_url"
    shell = _file(body, "claude-code", "router", "shell")["rendered"]
    assert "export ANTHROPIC_BASE_URL=https://warden.example\n" in shell
    assert "vw_YOUR_KEY" in shell


def test_origin_honours_forwarded_headers(api):
    c, h, _ = api
    hdrs = {**h, "X-Forwarded-Proto": "https", "X-Forwarded-Host": "proxy.example"}
    body = c.get(URL, headers=hdrs).json()
    assert body["origin"] == "https://proxy.example"
    assert body["origin_source"] == "request"


def test_models_loaded_first_then_by_name_with_capabilities(api):
    c, h, db = api
    _seed(
        db,
        _model("m3", "zeta", supports_tools=1),
        _model("m1", "alpha", status="registered", max_model_len=None),
        _model("m2", "beta", max_model_len=32768, supports_tools=0, supports_reasoning=1),
    )
    body = c.get(URL, headers=h).json()
    assert [m["served_name"] for m in body["models"]] == ["beta", "zeta", "alpha"]
    beta = body["models"][0]
    assert beta == {
        "id": "m2",
        "served_name": "beta",
        "status": "loaded",
        "backend": "vllm",
        "context_window": 32768,
        "supports_tools": False,
        "supports_vision": None,
        "supports_reasoning": True,
    }
    assert body["models"][1]["supports_tools"] is True
    assert body["selected_model"] == "beta"
    rendered = _file(body, "opencode", "default", "config")["rendered"]
    assert '"lmwarden/beta"' in rendered
    assert '"context": 32768' in rendered


def test_selected_model_follows_the_query_when_registered(api):
    c, h, db = api
    _seed(db, _model("m1", "alpha"), _model("m2", "beta", status="registered"))
    assert c.get(URL, params={"model": "beta"}, headers=h).json()["selected_model"] == "beta"
    assert c.get(URL, params={"model": "nope"}, headers=h).json()["selected_model"] == "alpha"
    body = c.get(URL, params={"model": "beta"}, headers=h).json()
    assert "aider --model openai/beta" in _file(body, "aider", "default", "shell")["rendered"]


def test_no_loaded_model_selects_nothing(api):
    c, h, db = api
    _seed(db, _model("m1", "alpha", status="registered"))
    body = c.get(URL, headers=h).json()
    assert body["selected_model"] is None
    assert "your-served-model-name" in _file(body, "aider", "default", "shell")["rendered"]


def test_router_rules_enabled_only_in_position_order(api):
    c, h, db = api
    _seed(db, _model("m1", "alpha"), _model("m2", "beta", status="registered"))
    _set(db, "router_enabled", "true")
    _set(db, "router_passthrough_unmatched", "false")
    _rule(db, "claude-sonnet*", "m2")
    _rule(db, "claude-off*", "m1", enabled=False)
    _rule(db, "gpt-*", "m1")
    body = c.get(URL, headers=h).json()
    assert body["router"] == {
        "enabled": True,
        "passthrough_unmatched": False,
        "rules": [
            {
                "pattern": "claude-sonnet*",
                "target_served_name": "beta",
                "target_status": "registered",
                "fallback": True,
                "example_model": "claude-sonnet-4-5",
            },
            {
                "pattern": "gpt-*",
                "target_served_name": "alpha",
                "target_status": "loaded",
                "fallback": True,
                "example_model": None,
            },
        ],
    }


def test_the_response_carries_templates_and_verify_definitions(api):
    c, h, db = api
    _seed(db, _model("m1", "alpha"))
    body = c.get(URL, headers=h).json()
    shell = _file(body, "claude-code", "router", "shell")
    assert shell["template"].startswith("export ANTHROPIC_BASE_URL={{origin}}")
    assert shell["rendered"].startswith("export ANTHROPIC_BASE_URL=http://testserver")
    router_mode = _client(body, "claude-code")["modes"][0]
    assert router_mode["id"] == "router"
    assert router_mode["verify"]["path"] == "/v1/messages"
    assert router_mode["verify"]["body"]["model"] == "{{verify_model}}"
    codex = _client(body, "codex-cli")
    assert codex["support"] == "local_only"
    (codex_mode,) = codex["modes"]
    assert codex_mode["verify"]["path"] == "/v1/responses"
    assert codex_mode["verify"]["expect"] == "openai_responses"
    config = _file(body, "codex-cli", "default", "config")
    assert 'wire_api = "responses"' in config["rendered"]
    assert 'base_url = "http://testserver/v1"' in config["rendered"]
    assert codex["verified"]["status"] == "run"
    assert set(codex["verified"]) == {"status", "date", "client_version", "note"}
    assert _client(body, "claude-code")["docs"]["url"].startswith("https://")


def test_continue_renders_both_variants(api):
    c, h, db = api
    _seed(db, _model("m1", "alpha", supports_tools=1))
    body = c.get(URL, headers=h).json()
    files = next(m for m in _client(body, "continue")["modes"])["files"]
    assert {f["requires_tools"] for f in files} == {True, False}
    for f in files:
        assert "model: alpha" in f["rendered"]


def test_no_key_material_in_the_response(api):
    c, h, db = api
    _seed(db, _model("m1", "alpha"))
    made = c.post("/api/tokens", json={"name": "k"}, headers={**h, **csrf_header(c)})
    plaintext = made.json()["plaintext"]
    text = c.get(URL, headers=h).text
    assert plaintext not in text
    assert made.json()["prefix"] not in text
