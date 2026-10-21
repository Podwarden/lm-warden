"""build_app wires the shell status feed ONLY when VW_SHELL_TOKEN is set.

Part 4 of the OVERSEER shell status feed (planner iter 44): the guard
(``shell_auth.py``), the pure service (``shell_status.py``) and the route
(``routes_shell.py``) already merged; this pins that ``build_app`` reads the
env var through ``parse_shell_token``, stores the bearer on
``app.state.shell_token``, mounts ``routes_shell.router`` only for a valid
token, and pops ``VW_SHELL_TOKEN`` from ``os.environ`` so engine subprocesses
and any later child cannot inherit it.

The route is absent for an unset or empty var, so Linux and CI route tables
stay byte-identical (``test_app_wiring`` / ``test_auth_matrix`` run without
the token). A malformed token fails boot loud with a ValueError that names
the variable, never the value. ``include_in_schema=False`` keeps the route out
of the OpenAPI document either way, so ``api-types.generated.ts`` never
changes.

All ``build_app`` / ``routes_shell`` / ``parse_shell_token`` imports live
INSIDE the test functions: a module-level import of a name that does not exist
yet makes this file an ImportError on the base, which runs nothing (red for
the wrong reason).
"""

import logging
import os

VALID = "a" * 64
INVALID = "b" * 64 + " "


def _has_shell_status_route(app) -> bool:
    return any(getattr(r, "path", None) == "/api/shell/status" for r in app.routes)


# (a) unset / empty VW_SHELL_TOKEN: no route, state None, nothing to pop.
def test_unset_and_empty_token_mount_nothing(monkeypatch):
    from app.main import build_app

    monkeypatch.delenv("VW_SHELL_TOKEN", raising=False)
    app = build_app()
    assert not _has_shell_status_route(app)
    assert getattr(app.state, "shell_token", "MISSING") is None
    assert "VW_SHELL_TOKEN" not in os.environ

    monkeypatch.setenv("VW_SHELL_TOKEN", "")
    app = build_app()
    assert not _has_shell_status_route(app)
    assert app.state.shell_token is None


# (b) a valid token: route mounted, state carries the bearer, the env var is
#     popped so no child process inherits it.
def test_valid_token_mounts_route_and_pops_env(monkeypatch):
    from app.main import build_app

    monkeypatch.setenv("VW_SHELL_TOKEN", VALID)
    app = build_app()
    assert _has_shell_status_route(app)
    assert app.state.shell_token == VALID
    assert "VW_SHELL_TOKEN" not in os.environ


# (b2) the pop is what keeps the token from children: engine envs are closed
#      dicts, but every other child (nvidia-smi probes, ps, sysctl) inherits
#      the live os.environ at spawn time, which is AFTER build_app has run.
def test_child_process_spawned_after_build_never_sees_token(monkeypatch):
    import subprocess
    import sys

    from app.main import build_app

    monkeypatch.setenv("VW_SHELL_TOKEN", VALID)
    build_app()
    probe = "import os; print(os.environ.get('VW_SHELL_TOKEN', 'ABSENT'))"
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=30
    ).stdout.strip()
    assert out == "ABSENT"


# (c) malformed token: boot fails loud, naming the variable but never the
#     value. 31 chars is one under the 32-char minimum; a space is a printable
#     char outside the allowed class.
def test_malformed_token_fails_boot(monkeypatch):
    import pytest

    from app.main import build_app

    for bad, label in (("a" * 31, "31 chars"), (INVALID, "embedded space")):
        monkeypatch.setenv("VW_SHELL_TOKEN", bad)
        with pytest.raises(ValueError) as excinfo:
            build_app()
        msg = str(excinfo.value)
        assert "VW_SHELL_TOKEN" in msg, label
        assert bad not in msg, label


# Boundary of the token rule as it reaches build_app: 31/32 and 256/257 chars.
def test_token_length_boundaries(monkeypatch):
    from app.main import build_app

    monkeypatch.setenv("VW_SHELL_TOKEN", "a" * 31)
    try:
        build_app()
    except ValueError:
        pass
    else:
        raise AssertionError("31 chars accepted")

    monkeypatch.setenv("VW_SHELL_TOKEN", "a" * 32)
    app = build_app()
    assert app.state.shell_token == "a" * 32

    monkeypatch.setenv("VW_SHELL_TOKEN", "a" * 256)
    app = build_app()
    assert app.state.shell_token == "a" * 256

    monkeypatch.setenv("VW_SHELL_TOKEN", "a" * 257)
    try:
        build_app()
    except ValueError:
        pass
    else:
        raise AssertionError("257 chars accepted")


# (e) the route never enters the OpenAPI document, with or without the token.
def test_openapi_never_exposes_shell_status(monkeypatch):
    from app.main import build_app

    monkeypatch.delenv("VW_SHELL_TOKEN", raising=False)
    assert "/api/shell/status" not in build_app().openapi()["paths"]

    monkeypatch.setenv("VW_SHELL_TOKEN", VALID)
    assert "/api/shell/status" not in build_app().openapi()["paths"]


# (d) + (f) end to end over a lifespan app: loopback + bearer -> 200 with the
#     feed shape; no bearer -> 401; a LAN peer -> 404 (no oracle); a build with
#     no token -> 404 from loopback; the token never reaches a log record.
def test_shell_status_end_to_end_and_no_log_leak(tmp_data_dir, migrated_db_template, caplog):
    import shutil
    from types import SimpleNamespace

    import app.main as main_module

    caplog.set_level(logging.DEBUG)
    # 'supervisor' only exists after a lifespan; capture (and restore) whatever
    # a preceding test left on the module-level app.
    original_app = main_module.app
    original_supervisor = getattr(original_app.state, "supervisor", None)
    try:
        shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
        app = _build_app_with_token(VALID)
        main_module.app = app
        app.state.supervisor = SimpleNamespace(
            live_engine_ids=lambda: [],
            get_variant=lambda mid: None,
            get_state=lambda mid: None,
            get_pid=lambda mid: None,
        )
        app.state.request_registry = SimpleNamespace(snapshot=lambda: [])
        _assert_feed_roundtrip()
        assert _no_token_from_loopback()
    finally:
        main_module.app = original_app
        if original_supervisor is not None:
            original_app.state.supervisor = original_supervisor

    for record in caplog.records:
        assert VALID not in record.getMessage()
        assert VALID not in str(record.args)


def _client(app, peer):
    # TestClient pins scope['client'] to 'testclient', which the guard
    # refuses; this shim fixes a real peer per scope and passes lifespan
    # through.
    from fastapi.testclient import TestClient

    async def asgi_app(scope, receive, send):
        if scope["type"] == "http":
            scope["client"] = peer
        return await app(scope, receive, send)

    return TestClient(asgi_app)


def _assert_feed_roundtrip():
    import app.main as main_module

    local = _client(main_module.app, ("127.0.0.1", 50000))
    with local:
        ok = local.get("/api/shell/status", headers={"Authorization": f"Bearer {VALID}"})
        assert ok.status_code == 200
        body = ok.json()
        assert body["models"] == []
        assert body["in_flight"]["count"] == 0
        assert local.get("/api/shell/status").status_code == 401

    lan = _client(main_module.app, ("192.168.1.20", 50000))
    with lan:
        assert (
            lan.get("/api/shell/status", headers={"Authorization": f"Bearer {VALID}"}).status_code
            == 404
        )


def _no_token_from_loopback() -> bool:
    # Rebuild without the token: even a loopback peer must get 404.
    app = _build_app_with_token(None)
    local = _client(app, ("127.0.0.1", 50000))
    with local:
        return local.get("/api/shell/status").status_code == 404


def _build_app_with_token(value: str | None):
    import os

    from app.main import build_app

    # build_app pops the variable, so (re)set it before every build.
    if value is None:
        os.environ.pop("VW_SHELL_TOKEN", None)
    else:
        os.environ["VW_SHELL_TOKEN"] = value
    return build_app()
