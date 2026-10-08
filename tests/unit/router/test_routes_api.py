"""Control API /api/router/* (#287 task 2)."""

import asyncio
import time
from pathlib import Path

import pytest

from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.router.state import Decision
from tests.conftest import bearer, csrf_header, jwt_login, seed_admin_token

S = "/api/router/settings"
R = "/api/router/rules"


def _model(mid: str = "m1", name: str = "qwen") -> ModelRow:
    return ModelRow(
        id=mid,
        served_model_name=name,
        hf_repo="a/b",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status="loaded",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
    )


def _seed_model(db_path: Path, mid: str = "m1", name: str = "qwen") -> None:
    async def go() -> None:
        async with open_db(db_path) as db:
            await ModelRepo(db).insert(_model(mid, name))

    asyncio.run(go())


def _delete_model(db_path: Path, mid: str) -> None:
    async def go() -> None:
        async with open_db(db_path) as db:
            await ModelRepo(db).delete(mid)

    asyncio.run(go())


@pytest.fixture
def api(client, seeded_db):
    _seed_model(seeded_db)
    return client, {**jwt_login(client), **csrf_header(client)}, seeded_db


def _dec(**over) -> Decision:
    base = dict(
        ts=time.time(),
        path="/v1/messages",
        model_in="claude-haiku-4",
        model_out="qwen",
        rule_id=None,
        route="local",
        reason=None,
        status=200,
        latency_ms=100,
        ttfb_ms=40,
        stream=True,
        token_name="laptop",
    )
    base.update(over)
    return Decision(**base)


def _rule(c, h, pattern="claude-haiku*", **kw):
    r = c.post(R, json={"pattern": pattern, "target_model_id": "m1", **kw}, headers=h)
    assert r.status_code == 201, r.text
    return r.json()


# -- settings ---------------------------------------------------------------


def test_settings_defaults(api):
    c, h, _ = api
    r = c.get(S, headers=h)
    assert r.status_code == 200
    assert r.json() == {
        "enabled": False,
        "upstream_url": "https://api.anthropic.com",
        "passthrough_unmatched": True,
        "local_header_timeout_s": 60,
        "local_nonstream_timeout_s": 120,
        "breaker_threshold": 3,
        "breaker_open_s": 60,
        "max_body_mb": 32,
        "header_name": "X-LMWarden-Key",
    }


def test_settings_patch_persists_and_invalidates(api, monkeypatch):
    c, h, _ = api
    calls = []
    monkeypatch.setattr(c.app.state.router, "invalidate", lambda: calls.append(1))
    r = c.patch(S, json={"enabled": True, "upstream_url": "https://proxy.example"}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["enabled"] is True
    assert calls
    again = c.get(S, headers=h).json()
    assert again["enabled"] is True and again["upstream_url"] == "https://proxy.example"
    assert again["breaker_threshold"] == 3


@pytest.mark.parametrize(
    "body",
    [
        {"header_name": "x"},
        {"upstream_url": "http://proxy.example"},
        {"upstream_url": "https://x/?q=1"},
        {"breaker_threshold": 0},
        {"max_body_mb": 257},
        {"enabled": None},
        {"enabled": "maybe"},
    ],
)
def test_settings_patch_rejects(api, body):
    c, h, _ = api
    assert c.patch(S, json=body, headers=h).status_code == 422


# -- rules ------------------------------------------------------------------


def test_rule_create_shape_and_positions(api, monkeypatch):
    c, h, _ = api
    calls = []
    monkeypatch.setattr(c.app.state.router, "invalidate", lambda: calls.append(1))
    a = _rule(c, h)
    assert a["position"] == 0
    assert a["target_served_name"] == "qwen" and a["target_status"] == "loaded"
    assert a["enabled"] is True and a["fallback"] is True and a["strip_thinking"] is True
    assert a["min_max_tokens"] == 0 and a["created_at"] and a["updated_at"]
    assert _rule(c, h, "claude-opus*")["position"] == 1
    assert len(calls) == 2
    listed = c.get(R, headers=h).json()["rules"]
    assert [x["pattern"] for x in listed] == ["claude-haiku*", "claude-opus*"]


def test_rule_create_validation(api):
    c, h, _ = api
    r = c.post(R, json={"pattern": "x*", "target_model_id": "nope"}, headers=h)
    assert r.status_code == 422 and "target_model_id: unknown model" in r.text
    for pat in ("*", "a[b]", ""):
        r = c.post(R, json={"pattern": pat, "target_model_id": "m1"}, headers=h)
        assert r.status_code == 422 and "pattern" in r.text
    r = c.post(
        R, json={"pattern": "x*", "target_model_id": "m1", "min_max_tokens": 131073}, headers=h
    )
    assert r.status_code == 422


def test_rule_patch(api, monkeypatch):
    c, h, _ = api
    a = _rule(c, h)
    calls = []
    monkeypatch.setattr(c.app.state.router, "invalidate", lambda: calls.append(1))
    assert c.patch(f"{R}/{a['id']}", json={"pattern": "*"}, headers=h).status_code == 422
    r = c.patch(f"{R}/{a['id']}", json={"enabled": False, "min_max_tokens": 512}, headers=h)
    assert r.status_code == 200
    assert r.json()["enabled"] is False and r.json()["min_max_tokens"] == 512
    assert calls
    assert c.patch(f"{R}/{a['id']}", json={"target_model_id": "zzz"}, headers=h).status_code == 422
    assert c.patch(f"{R}/{a['id']}", json={"enabled": None}, headers=h).status_code == 422
    assert c.patch(f"{R}/nope", json={"enabled": True}, headers=h).status_code == 404


def test_rule_order(api, monkeypatch):
    c, h, _ = api
    a, b = _rule(c, h), _rule(c, h, "claude-opus*")
    calls = []
    monkeypatch.setattr(c.app.state.router, "invalidate", lambda: calls.append(1))
    r = c.put(f"{R}/order", json={"ids": [b["id"], a["id"]]}, headers=h)
    assert r.status_code == 200, r.text
    assert [x["id"] for x in r.json()["rules"]] == [b["id"], a["id"]]
    assert [x["position"] for x in r.json()["rules"]] == [0, 1]
    assert calls
    assert c.put(f"{R}/order", json={"ids": [a["id"]]}, headers=h).status_code == 422
    assert c.put(f"{R}/order", json={"ids": [a["id"], a["id"]]}, headers=h).status_code == 422


def test_rule_delete_and_missing_target(api, monkeypatch):
    c, h, db_path = api
    a = _rule(c, h)
    b = _rule(c, h, "claude-opus*")
    calls = []
    monkeypatch.setattr(c.app.state.router, "invalidate", lambda: calls.append(1))
    assert c.delete(f"{R}/{a['id']}", headers=h).status_code == 204
    assert calls
    assert c.delete(f"{R}/{a['id']}", headers=h).status_code == 404
    _delete_model(db_path, "m1")
    rules = c.get(R, headers=h).json()["rules"]
    assert rules[0]["id"] == b["id"]
    assert rules[0]["target_served_name"] is None and rules[0]["target_status"] is None


# -- stats / decisions ------------------------------------------------------


def test_stats_empty_then_counted_then_reset(api):
    c, h, _ = api
    a = _rule(c, h)
    s = c.get("/api/router/stats", headers=h).json()
    assert s["since"] is None and s["enabled"] is False
    assert s["totals"] == {"local": 0, "passthrough": 0, "fallback": 0, "refused": 0, "error": 0}
    assert s["by_reason"] == {}
    assert [r["rule_id"] for r in s["rules"]] == [a["id"]]
    assert s["rules"][0]["target_served_name"] == "qwen"
    assert s["rules"][0]["latency_ms"] == {"p50": None, "p95": None}
    assert [t["model_id"] for t in s["targets"]] == ["m1"]
    assert s["passthrough"]["requests"] == 0

    st = c.app.state.router
    st.record(_dec(rule_id=a["id"]), rule_pattern="claude-haiku*")
    st.record(_dec(route="passthrough", model_out=None))
    s = c.get("/api/router/stats", headers=h).json()
    assert s["totals"]["local"] == 1 and s["totals"]["passthrough"] == 1
    assert s["since"] is not None
    assert s["rules"][0]["local"] == 1 and s["rules"][0]["latency_ms"]["p50"] == 100

    assert c.post("/api/router/stats/reset", headers=h).status_code == 204
    s = c.get("/api/router/stats", headers=h).json()
    assert s["totals"]["local"] == 0 and s["since"] is None
    assert c.get("/api/router/decisions", headers=h).json() == {"decisions": []}


def test_stats_reflect_enabled(api):
    c, h, _ = api
    c.patch(S, json={"enabled": True}, headers=h)
    assert c.get("/api/router/stats", headers=h).json()["enabled"] is True


def test_decisions_limit_and_order(api):
    c, h, _ = api
    st = c.app.state.router
    for i in range(3):
        st.record(_dec(model_in=f"m{i}"))
    assert c.get("/api/router/decisions?limit=0", headers=h).status_code == 422
    assert c.get("/api/router/decisions?limit=201", headers=h).status_code == 422
    d = c.get("/api/router/decisions?limit=2", headers=h).json()["decisions"]
    assert [x["model_in"] for x in d] == ["m2", "m1"]
    assert set(d[0]) == {
        "ts", "path", "model_in", "model_out", "rule_id", "route", "reason",
        "status", "latency_ms", "ttfb_ms", "stream", "token_name",
    }  # fmt: skip
    assert len(c.get("/api/router/decisions", headers=h).json()["decisions"]) == 3


# -- auth -------------------------------------------------------------------

ROUTES = [
    ("GET", S),
    ("PATCH", S),
    ("GET", R),
    ("POST", R),
    ("PATCH", f"{R}/x"),
    ("DELETE", f"{R}/x"),
    ("PUT", f"{R}/order"),
    ("GET", "/api/router/stats"),
    ("POST", "/api/router/stats/reset"),
    ("GET", "/api/router/decisions"),
]


def test_no_credential_is_401(api):
    c, _, _ = api
    c.cookies.clear()
    csrf = csrf_header(c)
    for m, p in ROUTES:
        assert c.request(m, p, json={}, headers=csrf).status_code == 401, (m, p)


def test_admin_token_ok_and_inference_token_refused(api):
    c, h, db_path = api
    inf = c.post("/api/tokens", json={"name": "k"}, headers=h).json()["plaintext"]
    _, admin = seed_admin_token(db_path)
    c.cookies.clear()
    assert c.get(S, headers=bearer(admin)).status_code == 200
    assert c.get("/api/router/stats", headers=bearer(admin)).status_code == 200
    assert c.get(S, headers=bearer(inf)).status_code == 401
    assert c.get(R, headers=bearer(inf)).status_code == 401
