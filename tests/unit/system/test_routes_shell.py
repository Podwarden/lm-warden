"""GET /api/shell/status route — the HTTP seam over the pure shell-status service.

The route is the third of three shell-status-feed parts (planner iter 44): the
bearer/loopback guard (``require_shell_client``) is a ``Depends``, the pure
``build_shell_status`` is called with the supervisor, the in-flight registry
snapshot and the apple-profile memory probes, and the handler only assembles
``ShellStatus`` and the ``Cache-Control: no-store`` header.

``TestClient`` sets the ASGI ``scope['client']`` to ``testclient``, which
``is_loopback_client`` refuses, so a thin wrapper ASGI app pins it to a real
address per test (loopback 127.0.0.1, or a LAN address for the 404 arm) and
passes lifespan through untouched. New names are imported inside the test
functions so this file stays importable on the base tree where the module does
not exist yet.
"""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

TOKEN = "S" * 64


def _live_request(token_name, client_ip, model):
    """A minimal LiveRequest: the service reads only token_name and model."""
    from app.proxy.request_registry import LiveRequest

    return LiveRequest(
        id="req-1",
        token_id=None,
        token_name=token_name,
        client_ip=client_ip,
        model=model,
        model_row_id=model,
        path="/v1/chat/completions",
        prompt_tokens=0,
        max_model_len=None,
        started_monotonic=0.0,
        started_iso="1970-01-01T00:00:00Z",
    )


def _make_client(client_addr, token, supervisor, registry):
    """A TestClient whose ASGI scope carries a fixed loopback/LAN peer."""
    from app.system import routes_shell

    app = FastAPI()
    app.include_router(routes_shell.router)
    app.state.shell_token = token
    app.state.supervisor = supervisor
    app.state.request_registry = registry

    async def asgi_app(scope, receive, send):
        if scope["type"] == "http":
            scope["client"] = client_addr
        return await app(scope, receive, send)

    return TestClient(asgi_app)


def _one_engine_supervisor():
    """A duck-typed supervisor with a single live engine, pid 4242."""
    return SimpleNamespace(
        live_engine_ids=lambda: ["m1"],
        get_variant=lambda mid: SimpleNamespace(
            served_model_name="served-model",
            descriptor={"backend": "mlx"},
        ),
        get_state=lambda mid: SimpleNamespace(value="ready"),
        get_pid=lambda mid: 4242,
    )


# (a) loopback + exact token -> 200, exact top-level keys, no-store,
#     and the bearer never leaks into the body.
def test_shell_status_200_shape_and_no_store():
    from app.system import routes_shell

    registry = SimpleNamespace(snapshot=lambda: [])
    client = _make_client(("127.0.0.1", 50000), TOKEN, _one_engine_supervisor(), registry)
    res = client.get("/api/shell/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert res.status_code == 200
    assert set(res.json()) == {"models", "memory", "in_flight"}
    assert res.headers["cache-control"] == "no-store"
    assert TOKEN not in res.text
    model = res.json()["models"][0]
    assert model["name"] == "served-model"
    assert model["backend"] == "mlx"
    assert model["state"] == "ready"
    assert routes_shell.router.routes[0].include_in_schema is False


# (b) wrong token (last char changed) -> 401; missing header -> 401.
def test_shell_status_wrong_token_and_missing_header_401():
    registry = SimpleNamespace(snapshot=lambda: [])
    client = _make_client(("127.0.0.1", 50000), TOKEN, _one_engine_supervisor(), registry)
    res = client.get("/api/shell/status", headers={"Authorization": f"Bearer {TOKEN[:-1]}x"})
    assert res.status_code == 401
    res = client.get("/api/shell/status")
    assert res.status_code == 401


# (c) non-loopback peer -> 404; loopback peer that is forwarded -> 404.
def test_shell_status_non_loopback_and_forwarded_404():
    registry = SimpleNamespace(snapshot=lambda: [])
    lan = _make_client(("10.0.0.5", 50000), TOKEN, _one_engine_supervisor(), registry)
    res = lan.get("/api/shell/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert res.status_code == 404

    loopback = _make_client(("127.0.0.1", 50000), TOKEN, _one_engine_supervisor(), registry)
    res = loopback.get(
        "/api/shell/status",
        headers={"Authorization": f"Bearer {TOKEN}", "X-Forwarded-For": "1.2.3.4"},
    )
    assert res.status_code == 404


# (d) endpoint disabled (shell_token None) -> 404.
def test_shell_status_disabled_404():
    registry = SimpleNamespace(snapshot=lambda: [])
    client = _make_client(("127.0.0.1", 50000), None, _one_engine_supervisor(), registry)
    res = client.get("/api/shell/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert res.status_code == 404


# (e) linux profile: usable/pressure None and the apple probes never run.
def test_shell_status_linux_profile_never_probes(monkeypatch):
    import app.system.apple.memory as apple_memory
    import app.system.gpu as gpu_mod
    from app import platform_profile

    monkeypatch.setattr(platform_profile, "PROFILE", "linux")
    calls = []
    monkeypatch.setattr(
        gpu_mod,
        "_apple_budget_mib",
        lambda: _boom(calls, "budget"),
        raising=False,
    )
    monkeypatch.setattr(
        apple_memory,
        "read_memory_pressure",
        lambda run: _boom(calls, "pressure"),
        raising=False,
    )
    registry = SimpleNamespace(snapshot=lambda: [])
    client = _make_client(("127.0.0.1", 50000), TOKEN, _one_engine_supervisor(), registry)
    res = client.get("/api/shell/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert res.status_code == 200
    memory = res.json()["memory"]
    assert memory["usable_bytes"] is None
    assert memory["pressure_level"] is None
    assert calls == []


# (f) apple profile: probes run; success values pass through, any probe
#     exception degrades that value to None but never a 500.
def test_shell_status_apple_profile_probes(monkeypatch):
    import app.system.apple.memory as apple_memory
    import app.system.gpu as gpu_mod
    from app import platform_profile
    from app.system.apple.memory import MemoryPressure

    monkeypatch.setattr(platform_profile, "PROFILE", "apple")

    async def budget_ok():
        return 16384

    async def pressure_ok():
        return 7

    monkeypatch.setattr(gpu_mod, "_apple_budget_mib", budget_ok, raising=False)
    monkeypatch.setattr(
        apple_memory,
        "read_memory_pressure",
        lambda run: MemoryPressure(pressure_level=7),
        raising=False,
    )
    registry = SimpleNamespace(snapshot=lambda: [])
    client = _make_client(("127.0.0.1", 50000), TOKEN, _one_engine_supervisor(), registry)
    res = client.get("/api/shell/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert res.status_code == 200
    memory = res.json()["memory"]
    assert memory["usable_bytes"] == 16384 * 2**20
    assert memory["pressure_level"] == 7

    async def budget_boom():
        raise RuntimeError("boom-budget")

    monkeypatch.setattr(gpu_mod, "_apple_budget_mib", budget_boom, raising=False)

    def pressure_boom(run):
        raise RuntimeError("boom-pressure")

    monkeypatch.setattr(apple_memory, "read_memory_pressure", pressure_boom, raising=False)
    res = client.get("/api/shell/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert res.status_code == 200
    memory = res.json()["memory"]
    assert memory["usable_bytes"] is None
    assert memory["pressure_level"] is None


# (g) in-flight group: count and who/model present; client IP and the bearer
#     never appear in the response.
def test_shell_status_in_flight_group_hides_secrets():
    registry = SimpleNamespace(
        snapshot=lambda: [_live_request("alice", "203.0.113.9", "served-model")]
    )
    client = _make_client(("127.0.0.1", 50000), TOKEN, _one_engine_supervisor(), registry)
    res = client.get("/api/shell/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert res.status_code == 200
    in_flight = res.json()["in_flight"]
    assert in_flight["count"] == 1
    assert in_flight["groups"][0] == {"who": "alice", "model": "served-model", "requests": 1}
    assert "203.0.113.9" not in res.text
    assert TOKEN not in res.text


# (h) a missing registry means no in-flight requests, never an error.
def test_shell_status_missing_registry_means_no_requests():
    from app.system import routes_shell

    app = FastAPI()
    app.include_router(routes_shell.router)
    app.state.shell_token = TOKEN
    app.state.supervisor = _one_engine_supervisor()
    # NOTE: no request_registry attribute on app.state

    async def asgi_app(scope, receive, send):
        if scope["type"] == "http":
            scope["client"] = ("127.0.0.1", 50000)
        return await app(scope, receive, send)

    client = TestClient(asgi_app)
    res = client.get("/api/shell/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert res.status_code == 200
    assert res.json()["in_flight"] == {"count": 0, "groups": []}


def _boom(calls, name):
    """Record that a probe ran, then fail: the linux profile must never probe."""
    calls.append(name)
    raise AssertionError(f"apple probe {name} ran under the linux profile")


# (i) no oracle, no leak: the route is absent from OpenAPI; a non-loopback peer
#     gets the same 404 for a wrong token as for the right one, byte-identical to
#     an unmounted path; the 401 body is the fixed string; the token appears in
#     no response body, no response header and no log record.
def test_shell_status_no_oracle_no_leak(caplog):
    import logging

    from app.system import routes_shell

    bare = FastAPI()
    bare.include_router(routes_shell.router)
    assert "/api/shell/status" not in bare.openapi()["paths"]

    caplog.set_level(logging.DEBUG)
    registry = SimpleNamespace(snapshot=lambda: [])
    wrong = f"Bearer {TOKEN[:-1]}x"
    right = f"Bearer {TOKEN}"
    lan = _make_client(("10.0.0.5", 50000), TOKEN, _one_engine_supervisor(), registry)
    lan_right = lan.get("/api/shell/status", headers={"Authorization": right})
    lan_wrong = lan.get("/api/shell/status", headers={"Authorization": wrong})
    unmounted = lan.get("/api/shell/nope", headers={"Authorization": right})
    assert lan_right.status_code == lan_wrong.status_code == unmounted.status_code == 404
    assert lan_right.content == lan_wrong.content == unmounted.content

    loopback = _make_client(("127.0.0.1", 50000), TOKEN, _one_engine_supervisor(), registry)
    bad = loopback.get("/api/shell/status", headers={"Authorization": wrong})
    assert bad.status_code == 401
    assert bad.json() == {"detail": "unauthorized"}
    assert bad.headers["www-authenticate"] == "Bearer"
    ok = loopback.get("/api/shell/status", headers={"Authorization": right})
    assert ok.status_code == 200

    for res in (lan_right, lan_wrong, bad, ok):
        assert TOKEN not in res.text
        assert all(TOKEN not in value for value in res.headers.values())
    assert all(TOKEN not in record.getMessage() for record in caplog.records)
    assert all(TOKEN not in str(record.args) for record in caplog.records)
