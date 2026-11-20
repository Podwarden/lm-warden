"""security.csrf: the origin check on every state-changing cookie request.

The rule (conventions security.csrf, revision 1):

* A POST/PUT/PATCH/DELETE that carries any cookie this app sets
  (vw_refresh, vw_csrf_id, vw_session) must come from an allowed origin:
  ``Origin`` when the browser sent one, else the origin of ``Referer``. With
  neither, the CSRF token decides (convention: "Origin must match when it is
  present").
* A request carrying none of the app's cookies skips the CSRF layer; the
  route's own auth accepts or refuses it. SDKs, CLIs and curl send no cookie.
* A request authenticated by an API key (vw_) or an admin token (vwa_) carries
  no ambient credential and is never refused for CSRF, cookie or not.

The allowed origins are the same list refresh/logout use: VW_FRONTEND_ORIGIN,
the proxy-derived origin under VW_TRUST_PROXY_ORIGIN, or, with nothing
configured, the origin the request was addressed to (Host).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import bearer, csrf_header, jwt_login, seed_admin_token, seed_admin_user

EVIL = "https://evil.example"
SAME = "http://testserver"  # TestClient's Host; derivation is on (nothing configured)
DEV = "http://localhost:3000"  # Settings.allowed_origins' default
ORIGIN_REFUSED = {"detail": "origin mismatch"}
TOKEN_REFUSED = {"detail": "csrf token invalid"}


@pytest.fixture
def session(client: TestClient, tmp_data_dir):
    """A signed-in console: the cookie jar holds vw_csrf_id (and vw_refresh)."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    return {**jwt_login(client), **csrf_header(client)}


def _mint_key(client: TestClient, session: dict[str, str], origin: str | None) -> object:
    headers = dict(session)
    if origin is not None:
        headers["Origin"] = origin
    return client.post("/api/tokens", json={"name": "k"}, headers=headers)


def test_cross_origin_cookie_post_is_refused(client, session):
    r = _mint_key(client, session, EVIL)
    assert r.status_code == 403
    assert r.json() == ORIGIN_REFUSED


def test_same_origin_cookie_post_passes(client, session):
    assert _mint_key(client, session, SAME).status_code == 201


def test_configured_dev_origin_passes(client, session):
    assert _mint_key(client, session, DEV).status_code == 201


def test_origin_null_is_refused(client, session):
    # A sandboxed iframe or a file:// page sends the literal "null".
    r = _mint_key(client, session, "null")
    assert r.status_code == 403
    assert r.json() == ORIGIN_REFUSED


def test_referer_is_the_fallback_when_origin_is_absent(client, session):
    r = client.post(
        "/api/tokens", json={"name": "k"}, headers={**session, "Referer": f"{EVIL}/page"}
    )
    assert r.status_code == 403
    assert r.json() == ORIGIN_REFUSED
    ok = client.post(
        "/api/tokens", json={"name": "k"}, headers={**session, "Referer": f"{SAME}/ui/tokens"}
    )
    assert ok.status_code == 201


def test_cookies_without_origin_or_referer_fall_to_the_token(client, session):
    # Convention: the Origin must match when it is present. Absent, the
    # HMAC token is what stands between a cookie and a write.
    assert _mint_key(client, session, None).status_code == 201
    no_token = {k: v for k, v in session.items() if k != "X-CSRF-Token"}
    r = client.post("/api/tokens", json={"name": "k"}, headers=no_token)
    assert r.status_code == 403
    assert r.json() == TOKEN_REFUSED


def test_a_valid_token_does_not_excuse_a_foreign_origin(client, session):
    # Belt and braces: the token is valid, the origin is not.
    r = _mint_key(client, session, EVIL)
    assert r.json() == ORIGIN_REFUSED


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/auth/login", {"username": "admin", "password": "hunter2"}),
        ("/api/auth/sse-ticket", {"path": "/x"}),
        ("/api/setup/welcome", {}),
    ],
)
def test_paths_that_skip_the_token_still_get_the_origin_check(client, session, path, body):
    """The token-exempt routes are no longer origin-exempt: a cross-site form
    or fetch that rides the browser's cookies is refused there too."""
    headers = {k: v for k, v in session.items() if k != "X-CSRF-Token"}
    r = client.post(path, json=body, headers={**headers, "Origin": EVIL})
    assert r.status_code == 403
    assert r.json() == ORIGIN_REFUSED


def test_v1_never_honours_a_cookie(client, session):
    """/v1 skips the CSRF pair entirely (app/main.py PathBypass): it is
    authenticated by an API key header only, so a cookie opens nothing there."""
    headers = {k: v for k, v in session.items() if k not in ("X-CSRF-Token", "Authorization")}
    r = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={**headers, "Origin": EVIL},
    )
    assert r.status_code == 401


def test_no_cookie_no_token_is_left_to_the_route_auth(client, tmp_data_dir):
    """A CLI with a session JWT and no cookie jar: nothing ambient, no CSRF."""
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    jwt = jwt_login(client)
    client.cookies.clear()
    r = client.post("/api/tokens", json={"name": "cli"}, headers=jwt)
    assert r.status_code == 201, r.text
    # ... and without the JWT, it is the route's 401, not a CSRF 403.
    client.cookies.clear()
    assert client.post("/api/tokens", json={"name": "cli"}).status_code == 401


def test_admin_token_is_never_refused_for_csrf(client, session, tmp_data_dir):
    _, admin = seed_admin_token(tmp_data_dir / "vllm-warden.db")
    # Cookies in the jar, a foreign Origin, no CSRF token: the bearer is the
    # credential, and a cross-site page cannot set it.
    r = client.post("/api/tokens", json={"name": "a"}, headers={**bearer(admin), "Origin": EVIL})
    assert r.status_code == 201, r.text
    client.cookies.clear()
    r = client.post("/api/tokens", json={"name": "b"}, headers=bearer(admin))
    assert r.status_code == 201, r.text


def test_api_key_on_v1_without_origin_or_cookie_is_not_refused_for_csrf(client, session):
    key = _mint_key(client, session, SAME).json()["plaintext"]
    client.cookies.clear()
    r = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
        headers=bearer(key),
    )
    assert r.status_code != 403, r.text


def test_api_key_from_a_browser_on_another_origin_is_not_refused_for_csrf(client, session):
    # A web client on another origin calling /v1 with a key: it carries this
    # host's cookies only if the user also uses the console here.
    key = _mint_key(client, session, SAME).json()["plaintext"]
    r = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
        headers={**bearer(key), "Origin": EVIL},
    )
    assert r.json() != ORIGIN_REFUSED


def test_macos_app_loopback_origin_passes(client, session):
    """The Mac app serves its static UI from http://127.0.0.1:<port> and sets
    no VW_FRONTEND_ORIGIN, so the request's own Host is the allowed origin."""
    headers = {**session, "Host": "127.0.0.1:8080", "Origin": "http://127.0.0.1:8080"}
    r = client.post("/api/tokens", json={"name": "mac"}, headers=headers)
    assert r.status_code == 201, r.text
    other = {**headers, "Origin": "http://127.0.0.1:9999"}
    assert client.post("/api/tokens", json={"name": "mac"}, headers=other).status_code == 403


def test_configured_origin_list_is_honoured_exactly(
    tmp_data_dir, migrated_db_template, monkeypatch
):
    import shutil

    from app.main import build_app

    monkeypatch.setenv("VW_FRONTEND_ORIGIN", "https://lm.example.com,https://alt.example.com")
    shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    # https: an all-https allow-list makes the cookies Secure (app/auth/cookies.py),
    # and a Secure cookie never travels over http.
    with TestClient(build_app(), base_url="https://testserver") as c:
        c.get("/healthz")
        seed_admin_user(tmp_data_dir / "vllm-warden.db")
        s = {**jwt_login(c), **csrf_header(c)}
        assert _mint_key(c, s, "https://alt.example.com").status_code == 201
        # Derivation is off once an allow-list is configured.
        assert _mint_key(c, s, "https://testserver").json() == ORIGIN_REFUSED


def test_no_cors_middleware_while_bearers_skip_the_csrf_layer(client):
    """csrf_check lets any vw_/vwa_ bearer through, whatever its Origin. That is
    safe only because no CORS middleware lets another site read the answers or
    send credentialed preflighted requests. Adding CORS means revisiting
    app/auth/csrf.py::_key_bearer first -- then update this test."""
    from starlette.middleware.cors import CORSMiddleware

    assert all(m.cls is not CORSMiddleware for m in client.app.user_middleware)
    r = client.options(
        "/api/tokens",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in r.headers
