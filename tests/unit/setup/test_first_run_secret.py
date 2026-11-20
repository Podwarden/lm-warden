"""lifecycle.bootstrap-admin: a fresh install cannot be claimed by a stranger.

Two ways to set up the first admin (conventions lifecycle.bootstrap-admin@1):

* Wizard, locked by a one-time SETUP CODE. Every POST /api/setup/* needs the
  ``X-Setup-Code`` header until the admin exists. The code is LM_SETUP_CODE
  when the operator (or install.sh, or the Hub) set it, else one the server
  generates, keeps in <data dir>/setup-code (0600) and prints to its log.
* Unattended: LM_BOOTSTRAP_ADMIN_USERNAME + LM_BOOTSTRAP_ADMIN_PASSWORD create
  the admin at boot and close the wizard. Half-set refuses to boot.

The macOS app (it launches the server with a per-launch VW_SHELL_TOKEN, and
binds 127.0.0.1) is exempt for a direct loopback peer: the person at the Mac
is the owner, and the app offers no way to set an env var.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import stat
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.conftest import seed_admin_user

ADMIN = {"username": "owner", "password": "a-long-enough-password"}
CODE_FILE = "setup-code"


def _db(tmp_data_dir: Path) -> Path:
    return tmp_data_dir / "vllm-warden.db"


def _users(tmp_data_dir: Path) -> list[tuple[str, str]]:
    with sqlite3.connect(_db(tmp_data_dir)) as db:
        return db.execute("SELECT username, password_hash FROM users").fetchall()


def _walk(client: TestClient, code: str | None) -> list[int]:
    """welcome -> gpus -> hf_token -> admin; the status of each POST."""
    h = {} if code is None else {"X-Setup-Code": code}
    out = [client.post("/api/setup/welcome", headers=h).status_code]
    out.append(
        client.post("/api/setup/gpus", json={"allowed_gpu_indices": [0]}, headers=h).status_code
    )
    out.append(client.post("/api/setup/hf_token", json={"hf_token": None}, headers=h).status_code)
    out.append(client.post("/api/setup/admin", json=ADMIN, headers=h).status_code)
    return out


@pytest.fixture
def one_gpu(monkeypatch):
    from app.system import gpu as gpu_mod
    from app.system.gpu import GpuInfo

    async def fake_query():
        return [GpuInfo(0, "A4000", 16376, 0, 0)]

    monkeypatch.setattr(gpu_mod, "query_gpus", fake_query)


@pytest.fixture
def boot(tmp_data_dir, migrated_db_template):
    """Build and start an app on a fresh (migrated, empty) database."""
    # Imported here, before any test sets VW_SHELL_TOKEN: the module's own
    # `app = build_app()` reads and pops that variable on first import.
    from app.main import build_app

    started: list[TestClient] = []

    def _boot(**kwargs) -> TestClient:
        if not _db(tmp_data_dir).exists():
            shutil.copyfile(migrated_db_template, _db(tmp_data_dir))
        c = TestClient(build_app(), **kwargs)
        c.__enter__()
        started.append(c)
        return c

    yield _boot
    for c in started:
        c.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# The generated code
# ---------------------------------------------------------------------------


def test_fresh_install_writes_a_private_setup_code_file(tmp_data_dir, boot):
    client = boot()
    path = tmp_data_dir / CODE_FILE
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    code = path.read_text().strip()
    assert len(code.replace("-", "")) >= 16
    state = client.get("/api/setup/state").json()
    assert state == {"step": "welcome", "done": False, "code_required": True, "code_source": "file"}


def test_fresh_install_logs_where_the_code_is(tmp_data_dir, boot, caplog):
    caplog.set_level(logging.WARNING)
    boot()
    code = (tmp_data_dir / CODE_FILE).read_text().strip()
    text = caplog.text
    assert "setup code" in text.lower()
    assert str(tmp_data_dir / CODE_FILE) in text
    assert code in text  # printed once at boot so a container log shows it


def test_anonymous_post_without_the_code_is_refused(tmp_data_dir, boot, one_gpu):
    client = boot()
    r = client.post("/api/setup/welcome")
    assert r.status_code == 403
    assert r.json() == {"detail": "setup code required"}
    r = client.post("/api/setup/admin", json=ADMIN, headers={"X-Setup-Code": "WRONG-CODE-0000"})
    assert r.status_code == 403
    assert r.json() == {"detail": "setup code invalid"}
    assert _users(tmp_data_dir) == []


def test_every_setup_post_needs_the_code(tmp_data_dir, boot, one_gpu):
    client = boot()
    code = (tmp_data_dir / CODE_FILE).read_text().strip()
    assert client.post("/api/setup/welcome", headers={"X-Setup-Code": code}).status_code == 200
    for path, body in (
        ("/api/setup/gpus", {"allowed_gpu_indices": [0]}),
        ("/api/setup/hf_token", {"hf_token": None}),
        ("/api/setup/admin", ADMIN),
    ):
        assert client.post(path, json=body).status_code == 403, path


def test_with_the_code_the_admin_is_created_and_the_code_dies(tmp_data_dir, boot, one_gpu):
    client = boot()
    code = (tmp_data_dir / CODE_FILE).read_text().strip()
    # Lower case and without dashes is the same code: people retype it.
    assert _walk(client, code.lower().replace("-", " ")) == [200, 200, 200, 200]
    assert [u for u, _ in _users(tmp_data_dir)] == ["owner"]
    assert not (tmp_data_dir / CODE_FILE).exists()
    state = client.get("/api/setup/state").json()
    assert state["done"] is True and state["code_required"] is False
    # The code cannot be used again: the wizard is closed for good.
    again = client.post(
        "/api/setup/admin", json={**ADMIN, "username": "x"}, headers={"X-Setup-Code": code}
    )
    assert again.status_code in (400, 409)
    assert [u for u, _ in _users(tmp_data_dir)] == ["owner"]


def test_the_code_survives_a_restart_until_used(tmp_data_dir, boot):
    boot()
    first = (tmp_data_dir / CODE_FILE).read_text()
    boot()
    assert (tmp_data_dir / CODE_FILE).read_text() == first


# ---------------------------------------------------------------------------
# LM_SETUP_CODE
# ---------------------------------------------------------------------------


def test_operator_code_from_env(tmp_data_dir, boot, monkeypatch, one_gpu):
    monkeypatch.setenv("LM_SETUP_CODE", "0f3a9c2e7b1d4a6f8e0c2b4d6f8a0c2e")
    client = boot()
    assert not (tmp_data_dir / CODE_FILE).exists()
    assert client.get("/api/setup/state").json()["code_source"] == "env"
    assert (
        client.post("/api/setup/welcome", headers={"X-Setup-Code": "nope-nope-nope"}).status_code
        == 403
    )
    assert _walk(client, "0f3a9c2e7b1d4a6f8e0c2b4d6f8a0c2e") == [200, 200, 200, 200]


def test_a_short_env_code_refuses_to_boot(tmp_data_dir, monkeypatch):
    from app.setup.bootstrap import BootstrapMisconfigured, setup_code_from_env

    with pytest.raises(BootstrapMisconfigured, match="LM_SETUP_CODE"):
        setup_code_from_env({"LM_SETUP_CODE": "short"})


# ---------------------------------------------------------------------------
# Existing installs
# ---------------------------------------------------------------------------


def test_existing_admin_database_is_unaffected(tmp_data_dir, boot):
    client = boot()
    seed_admin_user(_db(tmp_data_dir))
    (tmp_data_dir / CODE_FILE).unlink(missing_ok=True)
    client = boot()  # restart on a database that has its admin
    assert not (tmp_data_dir / CODE_FILE).exists()
    state = client.get("/api/setup/state").json()
    assert state == {"step": "done", "done": True, "code_required": False, "code_source": None}
    r = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    assert r.status_code == 200


def test_a_leftover_code_file_is_removed_once_an_admin_exists(tmp_data_dir, boot):
    boot()
    assert (tmp_data_dir / CODE_FILE).exists()
    seed_admin_user(_db(tmp_data_dir))
    boot()
    assert not (tmp_data_dir / CODE_FILE).exists()


# ---------------------------------------------------------------------------
# Unattended bootstrap (LM_BOOTSTRAP_ADMIN_*)
# ---------------------------------------------------------------------------


def test_env_bootstrap_creates_the_admin_and_closes_the_wizard(
    tmp_data_dir, boot, monkeypatch, one_gpu
):
    monkeypatch.setenv("LM_BOOTSTRAP_ADMIN_USERNAME", "ops")
    monkeypatch.setenv("LM_BOOTSTRAP_ADMIN_PASSWORD", "from-the-environment")
    client = boot()
    assert [u for u, _ in _users(tmp_data_dir)] == ["ops"]
    assert not (tmp_data_dir / CODE_FILE).exists()
    assert client.get("/api/setup/state").json()["done"] is True
    r = client.post("/api/auth/login", json={"username": "ops", "password": "from-the-environment"})
    assert r.status_code == 200
    # The wizard is closed: nobody can add a second admin through it.
    assert client.post("/api/setup/admin", json=ADMIN).status_code in (400, 409)
    # Every detected GPU is allowed, as the wizard's default would be.
    with sqlite3.connect(_db(tmp_data_dir)) as db:
        draft = db.execute("SELECT draft FROM setup_state").fetchone()[0]
    assert '"allowed_gpu_indices": [0]' in draft


@pytest.mark.parametrize(
    "env,needle",
    [
        ({"LM_BOOTSTRAP_ADMIN_USERNAME": "ops"}, "LM_BOOTSTRAP_ADMIN_PASSWORD"),
        ({"LM_BOOTSTRAP_ADMIN_PASSWORD": "from-the-environment"}, "LM_BOOTSTRAP_ADMIN_USERNAME"),
        (
            {"LM_BOOTSTRAP_ADMIN_USERNAME": "  ", "LM_BOOTSTRAP_ADMIN_PASSWORD": "x" * 20},
            "USERNAME",
        ),
        ({"LM_BOOTSTRAP_ADMIN_USERNAME": "ops", "LM_BOOTSTRAP_ADMIN_PASSWORD": "short"}, "12"),
    ],
)
def test_half_set_bootstrap_refuses_to_boot(env, needle):
    from app.setup.bootstrap import BootstrapMisconfigured, bootstrap_admin_from_env

    with pytest.raises(BootstrapMisconfigured, match=needle):
        bootstrap_admin_from_env(env)


def test_half_set_bootstrap_stops_the_app(tmp_data_dir, boot, monkeypatch):
    monkeypatch.setenv("LM_BOOTSTRAP_ADMIN_USERNAME", "ops")
    with pytest.raises(Exception, match="LM_BOOTSTRAP_ADMIN_PASSWORD"):
        boot()


def test_bootstrap_vars_never_touch_an_existing_admin(tmp_data_dir, boot, monkeypatch, caplog):
    boot()
    seed_admin_user(_db(tmp_data_dir))
    before = _users(tmp_data_dir)
    monkeypatch.setenv("LM_BOOTSTRAP_ADMIN_USERNAME", "ops")
    monkeypatch.setenv("LM_BOOTSTRAP_ADMIN_PASSWORD", "from-the-environment")
    caplog.set_level(logging.WARNING)
    boot()
    assert _users(tmp_data_dir) == before
    assert "LM_BOOTSTRAP_ADMIN_" in caplog.text  # told to remove them


def test_bootstrap_admin_is_temporary_until_its_password_changes(
    tmp_data_dir, boot, monkeypatch, caplog
):
    """lifecycle.retire-bootstrap: the env password also sits in compose files
    and shell history, so the admin it made stays flagged (and the log says
    so at every boot) until a person sets a new password."""
    from tests.conftest import csrf_header

    monkeypatch.setenv("LM_BOOTSTRAP_ADMIN_USERNAME", "ops")
    monkeypatch.setenv("LM_BOOTSTRAP_ADMIN_PASSWORD", "from-the-environment")
    client = boot()
    login = client.post(
        "/api/auth/login", json={"username": "ops", "password": "from-the-environment"}
    )
    assert login.json()["temporary_admin"] is True
    caplog.set_level(logging.WARNING)
    client = boot()
    assert "still has the password from LM_BOOTSTRAP_ADMIN_PASSWORD" in caplog.text

    jwt = client.post(
        "/api/auth/login", json={"username": "ops", "password": "from-the-environment"}
    )
    auth = {"Authorization": f"Bearer {jwt.json()['access_token']}", **csrf_header(client)}
    r = client.patch(
        "/api/settings/runtime", json={"admin_password": "a-new-password-of-mine"}, headers=auth
    )
    assert r.status_code < 300, r.text
    again = client.post(
        "/api/auth/login", json={"username": "ops", "password": "a-new-password-of-mine"}
    )
    assert again.json()["temporary_admin"] is False
    assert (
        client.post(
            "/api/auth/login", json={"username": "ops", "password": "from-the-environment"}
        ).status_code
        == 401
    )
    caplog.clear()
    boot()
    assert "still has the password" not in caplog.text


# ---------------------------------------------------------------------------
# macOS app: a direct loopback peer of a shell-launched server
# ---------------------------------------------------------------------------

SHELL_TOKEN = "s" * 40


async def _asgi_post(app, path: str, headers: dict[str, str], client=("127.0.0.1", 50123)):
    transport = httpx.ASGITransport(app=app, client=client)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8080") as c:
        return await c.post(path, headers=headers)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "shell,peer,forwarded,needs_code",
    [
        (True, "127.0.0.1", False, False),  # the Mac app's own window
        (True, "::1", False, False),
        (True, "192.168.1.20", False, True),  # Share on Local Network: a LAN peer
        (True, "127.0.0.1", True, True),  # Caddy / Tailscale in front of the Mac
        (False, "127.0.0.1", False, True),  # any other native run (a proxy without XFF)
    ],
)
async def test_mac_app_loopback(
    tmp_data_dir, boot, monkeypatch, shell, peer, forwarded, needs_code
):
    if shell:
        monkeypatch.setenv("VW_SHELL_TOKEN", SHELL_TOKEN)
    client = boot()
    headers = {"X-Forwarded-For": "203.0.113.9"} if forwarded else {}
    r = await _asgi_post(client.app, "/api/setup/welcome", headers, client=(peer, 50123))
    assert (r.status_code == 403) is needs_code, r.text


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Review fixes (2026-11-04)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    "headers,needs_code",
    [
        # The Mac app's own window: Host and Origin are the loopback address.
        ({"Origin": "http://127.0.0.1:8080"}, False),
        ({"Host": "localhost:8080", "Origin": "http://localhost:8080"}, False),
        ({"Host": "[::1]:8080"}, False),
        # Any web page the Mac user opens can fire a no-cors POST at loopback.
        ({"Origin": "https://evil.example"}, True),
        ({"Origin": "null"}, True),
        ({"Referer": "https://evil.example/page"}, True),
        # DNS rebinding: evil.example resolved to 127.0.0.1; Host gives it away.
        ({"Host": "evil.example:8080", "Origin": "http://evil.example:8080"}, True),
        ({"Host": "evil.example:8080"}, True),
    ],
)
async def test_mac_exemption_needs_a_loopback_host_and_origin(
    tmp_data_dir, boot, monkeypatch, headers, needs_code
):
    monkeypatch.setenv("VW_SHELL_TOKEN", SHELL_TOKEN)
    client = boot()
    r = await _asgi_post(client.app, "/api/setup/welcome", headers)
    assert (r.status_code == 403) is needs_code, r.text
    state_needs = await _asgi_get_state(client.app, headers)
    assert state_needs is needs_code


async def _asgi_get_state(app, headers: dict[str, str]) -> bool:
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 50123))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8080") as c:
        return (await c.get("/api/setup/state", headers=headers)).json()["code_required"]


def test_env_code_needs_16_characters():
    from app.setup.bootstrap import BootstrapMisconfigured, setup_code_from_env

    with pytest.raises(BootstrapMisconfigured, match="16"):
        setup_code_from_env({"LM_SETUP_CODE": "a" * 15})
    assert setup_code_from_env({"LM_SETUP_CODE": "a" * 16}) == "a" * 16


@pytest.mark.anyio
async def test_wrong_codes_are_throttled(tmp_data_dir, boot):
    """Failed codes reuse the login throttle: held back, then a public address
    is locked out (429) -- even with the right code -- for a while."""
    from app.auth.throttle import LoginThrottle

    client = boot()
    held: list[float] = []

    async def fake_sleep(s: float) -> None:
        held.append(s)

    client.app.state.setup_code_throttle = LoginThrottle(sleep=fake_sleep)
    code = (tmp_data_dir / CODE_FILE).read_text().strip()
    public = ("203.0.113.5", 40000)
    statuses = [
        (
            await _asgi_post(
                client.app, "/api/setup/welcome", {"X-Setup-Code": "WRONG-WRONG-0000"}, public
            )
        ).status_code
        for _ in range(6)
    ]
    assert statuses[:5] == [403] * 5
    assert statuses[5] == 429
    assert any(s > 0 for s in held)
    locked = await _asgi_post(client.app, "/api/setup/welcome", {"X-Setup-Code": code}, public)
    assert locked.status_code == 429
    assert "retry-after" in locked.headers


def test_unwritable_data_dir_falls_back_to_a_logged_code(tmp_data_dir, boot, monkeypatch, caplog):
    import os

    real_open = os.open

    def deny(path, *a, **k):
        if str(path).endswith(CODE_FILE):
            raise PermissionError(13, "Read-only file system")
        return real_open(path, *a, **k)

    monkeypatch.setattr(os, "open", deny)
    caplog.set_level(logging.WARNING)
    client = boot()
    assert not (tmp_data_dir / CODE_FILE).exists()
    code = client.app.state.setup_lock.code
    assert code and code in caplog.text
    assert client.post("/api/setup/welcome", headers={"X-Setup-Code": code}).status_code == 200


def test_unreadable_code_file_does_not_stop_boot(tmp_data_dir, boot, monkeypatch, caplog):
    from pathlib import Path as P

    (tmp_data_dir / CODE_FILE).write_text("ABCD-EFGH-JKMN-PQRS\n")
    real_read = P.read_text

    def deny(self, *a, **k):
        if self.name == CODE_FILE:
            raise PermissionError(13, "Permission denied")
        return real_read(self, *a, **k)

    monkeypatch.setattr(P, "read_text", deny)
    caplog.set_level(logging.WARNING)
    client = boot()
    code = client.app.state.setup_lock.code
    assert code and code in caplog.text


def test_resetting_the_same_env_password_keeps_the_admin_temporary(tmp_data_dir, boot, monkeypatch):
    from tests.conftest import csrf_header

    monkeypatch.setenv("LM_BOOTSTRAP_ADMIN_USERNAME", "ops")
    monkeypatch.setenv("LM_BOOTSTRAP_ADMIN_PASSWORD", "from-the-environment")
    client = boot()
    jwt = client.post(
        "/api/auth/login", json={"username": "ops", "password": "from-the-environment"}
    ).json()["access_token"]
    auth = {"Authorization": f"Bearer {jwt}", **csrf_header(client)}
    r = client.patch(
        "/api/settings/runtime", json={"admin_password": "from-the-environment"}, headers=auth
    )
    assert r.status_code < 300, r.text
    again = client.post(
        "/api/auth/login", json={"username": "ops", "password": "from-the-environment"}
    )
    assert again.json()["temporary_admin"] is True
