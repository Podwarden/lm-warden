"""Home page modes (spec 2026-11-04): what `/` shows, per mode, in core alone
(no extension installed; the website's side is in the website package's own
tests)."""

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import csrf_header, jwt_login, seed_admin_user


def _put(tmp_data_dir: Path, key: str, value: str) -> None:
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db", isolation_level=None) as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO settings(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        db.execute("COMMIT")
        db.execute("PRAGMA wal_checkpoint(FULL)")


@pytest.fixture
def set_setting(client: TestClient, tmp_data_dir: Path):
    return lambda key, value: _put(tmp_data_dir, key, value)


@pytest.fixture
def admin_headers(client: TestClient, tmp_data_dir: Path) -> dict[str, str]:
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    headers = jwt_login(client)
    headers.update(csrf_header(client))
    return headers


def test_fresh_install_defaults_to_builtin(client: TestClient) -> None:
    from app.db.database import open_db  # noqa: F401

    with sqlite3.connect(client.app.state.settings.db_path) as db:  # type: ignore[attr-defined]
        rows = dict(db.execute("SELECT key, value FROM settings"))
    assert rows["home_page_mode"] == "builtin"
    assert rows["home_page_off_action"] == "console"


def test_fresh_install_shows_welcome_page(client: TestClient) -> None:
    for path in ("/", "/_landing"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "LM Warden" in r.text and "fonts.googleapis" not in r.text
        assert "LLM Warden" not in r.text and "Podwarden/lm-warden" in r.text
        assert 'name="robots" content="noindex"' in r.text
        assert r.headers["cache-control"] == "no-cache"
        assert r.headers["content-security-policy"] == (
            "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        )
    assert "https://github.com/Podwarden/lm-warden/blob/main/LICENSE" in r.text


@pytest.mark.parametrize("path", ["/", "/_landing"])
def test_off_console_redirects(client: TestClient, set_setting, path: str) -> None:
    set_setting("home_page_mode", "off")
    r = client.get(path, follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/ui/"


@pytest.mark.parametrize("path", ["/", "/_landing"])
def test_off_404(client: TestClient, set_setting, path: str) -> None:
    set_setting("home_page_mode", "off")
    set_setting("home_page_off_action", "404")
    assert client.get(path, follow_redirects=False).status_code == 404


def test_missing_mode_row_behaves_as_off(client: TestClient, tmp_data_dir: Path) -> None:
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db", isolation_level=None) as db:
        db.execute("DELETE FROM settings WHERE key = 'home_page_mode'")
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/ui/"


def test_custom_without_provider_behaves_as_off(client: TestClient, set_setting) -> None:
    set_setting("home_page_mode", "custom")
    assert client.get("/", follow_redirects=False).status_code == 307


def test_robots_txt_off_disallows_all(client: TestClient, set_setting) -> None:
    set_setting("home_page_mode", "off")
    r = client.get("/robots.txt")
    assert r.status_code == 200 and "Disallow: /" in r.text


def test_robots_txt_builtin_welcome_disallows_all(client: TestClient) -> None:
    r = client.get("/robots.txt")
    assert r.status_code == 200 and r.text == "User-agent: *\nDisallow: /\n"


def test_welcome_page_has_plain_json_404(client: TestClient) -> None:
    r = client.get("/no-such-page", headers={"accept": "text/html"})
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


def test_custom_without_provider_is_refused(client: TestClient, admin_headers) -> None:
    r = client.patch(
        "/api/settings/runtime", headers=admin_headers, json={"home_page_mode": "custom"}
    )
    assert r.status_code == 422, r.text


def test_custom_accepted_when_a_provider_declares_it(client: TestClient, admin_headers) -> None:
    class Custom:
        mode = "custom"
        priority = 50

        async def home(self, request):  # type: ignore[no-untyped-def]
            from fastapi.responses import HTMLResponse

            return HTMLResponse("<p>mine</p>")

        async def not_found(self, request):  # type: ignore[no-untyped-def]
            return None

        async def robots_txt(self, request):  # type: ignore[no-untyped-def]
            return None

    client.app.state.ext.home_providers.append(Custom())  # type: ignore[attr-defined]
    r = client.patch(
        "/api/settings/runtime", headers=admin_headers, json={"home_page_mode": "custom"}
    )
    assert r.status_code == 200, r.text
    assert client.get("/").text == "<p>mine</p>"


def test_settings_round_trip_and_validation(client: TestClient, admin_headers) -> None:
    r = client.patch(
        "/api/settings/runtime",
        headers=admin_headers,
        json={"home_page_mode": "off", "home_page_off_action": "404"},
    )
    assert r.status_code == 200 and r.json()["requires_restart_kinds"] == []
    body = client.get("/api/settings/runtime", headers=admin_headers).json()
    assert body["home_page_mode"] == "off" and body["home_page_off_action"] == "404"
    for bad in ({"home_page_mode": "maybe"}, {"home_page_off_action": "teapot"}):
        assert (
            client.patch("/api/settings/runtime", headers=admin_headers, json=bad).status_code
            == 422
        )


def test_landing_page_enabled_is_no_longer_writable(client: TestClient, admin_headers) -> None:
    r = client.patch(
        "/api/settings/runtime", headers=admin_headers, json={"landing_page_enabled": True}
    )
    assert r.status_code == 400


def test_invalid_mode_wording_omits_custom_in_community(client: TestClient, admin_headers) -> None:
    """Core alone offers off / builtin, so the 422 must not advertise a mode
    this edition cannot serve (spec decision 5)."""
    for bad in ("maybe", "custom"):
        r = client.patch(
            "/api/settings/runtime", headers=admin_headers, json={"home_page_mode": bad}
        )
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        assert "'off', 'builtin'" in detail, detail
        assert "custom" not in detail, detail


def test_invalid_mode_wording_lists_custom_when_a_provider_declares_it(
    client: TestClient, admin_headers
) -> None:
    class Custom:
        mode = "custom"
        priority = 0

        async def home(self, request):  # type: ignore[no-untyped-def]
            return None

        async def not_found(self, request):  # type: ignore[no-untyped-def]
            return None

        async def robots_txt(self, request):  # type: ignore[no-untyped-def]
            return None

    client.app.state.ext.home_providers.append(Custom())  # type: ignore[attr-defined]
    r = client.patch("/api/settings/runtime", headers=admin_headers, json={"home_page_mode": "x"})
    assert r.status_code == 422
    assert "'off', 'builtin', 'custom'" in r.json()["detail"]


@pytest.mark.parametrize("mode", ["off", "builtin"])
def test_admin_token_may_set_off_and_builtin(client: TestClient, tmp_data_dir: Path, mode) -> None:
    """Only a mode beyond Community's off/builtin is session-only (an edition's
    page goes live); the two core modes stay ordinary configuration."""
    from tests.conftest import bearer, seed_admin_token

    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    _tid, secret = seed_admin_token(tmp_data_dir / "vllm-warden.db")
    r = client.patch("/api/settings/runtime", headers=bearer(secret), json={"home_page_mode": mode})
    assert r.status_code == 200, r.text
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        row = db.execute("SELECT value FROM settings WHERE key='home_page_mode'").fetchone()
    assert row[0] == mode


@pytest.mark.parametrize("value", ["custom", " Custom "])
def test_admin_token_custom_in_community_is_422_not_session_only(
    client: TestClient, tmp_data_dir: Path, value: str
) -> None:
    """Community serves no `custom`, so a token asking for it gets the same
    422 a session gets (listing off / builtin), not a 403 that would talk
    about uploaded pages this edition does not have."""
    from tests.conftest import bearer, seed_admin_token

    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    _tid, secret = seed_admin_token(tmp_data_dir / "vllm-warden.db")
    r = client.patch(
        "/api/settings/runtime", headers=bearer(secret), json={"home_page_mode": value}
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and "'off', 'builtin'" in detail, detail
    assert "custom" not in detail and "upload" not in detail.lower(), detail


def test_admin_token_custom_is_session_only_where_the_edition_serves_it(
    client: TestClient, tmp_data_dir: Path
) -> None:
    from tests.conftest import bearer, seed_admin_token

    class Custom:
        mode = "custom"
        priority = 0

        async def home(self, request):  # type: ignore[no-untyped-def]
            return None

        async def not_found(self, request):  # type: ignore[no-untyped-def]
            return None

        async def robots_txt(self, request):  # type: ignore[no-untyped-def]
            return None

    client.app.state.ext.home_providers.append(Custom())  # type: ignore[attr-defined]
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    _tid, secret = seed_admin_token(tmp_data_dir / "vllm-warden.db")
    r = client.patch(
        "/api/settings/runtime", headers=bearer(secret), json={"home_page_mode": "custom"}
    )
    assert r.status_code == 403 and r.json()["detail"]["error_code"] == "session_only"


def test_mode_change_is_logged_with_the_actor(
    client: TestClient, admin_headers, tmp_data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from tests.conftest import bearer, seed_admin_token

    caplog.set_level("INFO", logger="app.settings.routes_api")
    r = client.patch("/api/settings/runtime", headers=admin_headers, json={"home_page_mode": "off"})
    assert r.status_code == 200, r.text
    assert "home page mode builtin -> off by admin (session)" in caplog.text

    tid, secret = seed_admin_token(tmp_data_dir / "vllm-warden.db")
    r = client.patch(
        "/api/settings/runtime", headers=bearer(secret), json={"home_page_mode": "builtin"}
    )
    assert r.status_code == 200, r.text
    assert f"home page mode off -> builtin by admin (admin token {tid})" in caplog.text


def test_unchanged_mode_and_other_keys_are_not_logged_as_a_mode_change(
    client: TestClient, admin_headers, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="app.settings.routes_api")
    client.patch("/api/settings/runtime", headers=admin_headers, json={"home_page_mode": "builtin"})
    client.patch(
        "/api/settings/runtime", headers=admin_headers, json={"home_page_off_action": "404"}
    )
    assert "home page mode" not in caplog.text
