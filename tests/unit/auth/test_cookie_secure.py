"""The ``Secure`` flag on the session cookies is derived, not hardcoded.

Regression cover for the plain-HTTP session bug: ``vw_refresh`` was set with
``secure=True`` unconditionally, so a browser on the documented quick-start
URL (``http://YOUR-HOST:8080/ui/``) discarded it and every page load answered
``401 missing refresh cookie``. The flag must now follow the scheme the
browser actually used -- and both cookies must agree, since they describe the
same connection.

These assertions are deliberately about the wire format (the ``set-cookie``
header), not about the helper's return value: a silent regression here is a
security downgrade on TLS or a broken session on HTTP, and both are only
visible in the header the browser reads.
"""

from __future__ import annotations

import sqlite3

import bcrypt
import pytest


def _seed_admin(db_path, username="admin", pw="hunter2"):
    h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", (username, h))
        db.execute("UPDATE setup_state SET step = 'done' WHERE id = 1")
        db.commit()


def _refresh_cookie_header(response) -> str:
    for raw in response.headers.get_list("set-cookie"):
        if raw.startswith("vw_refresh="):
            return raw
    raise AssertionError(f"no vw_refresh cookie in {response.headers.get_list('set-cookie')!r}")


def _login(client, tmp_data_dir, *, url="/api/auth/login", headers=None):
    client.get("/healthz")  # boot, run migrations
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        url,
        json={"username": "admin", "password": "hunter2"},
        headers=headers or {},
    )
    assert r.status_code == 200, r.text
    return r


def _is_secure(cookie_header: str) -> bool:
    return "secure" in {a.strip().lower() for a in cookie_header.split(";")}


# ---------------------------------------------------------------------------
# The three schemes
# ---------------------------------------------------------------------------


def test_plain_http_login_cookie_is_not_secure(tmp_data_dir, client):
    """The bug. Over http:// the refresh cookie must be storable."""
    r = _login(client, tmp_data_dir)
    assert not _is_secure(_refresh_cookie_header(r))


def test_https_login_cookie_is_secure(tmp_data_dir, client):
    """Direct TLS keeps the flag with no configuration at all."""
    r = _login(client, tmp_data_dir, url="https://testserver/api/auth/login")
    assert _is_secure(_refresh_cookie_header(r))


def test_trusted_forwarded_proto_https_makes_cookie_secure(tmp_data_dir, client):
    """A TLS-terminating proxy the operator opted into trusting."""
    client.app.state.settings = _with(client, trust_proxy_origin=True)
    r = _login(client, tmp_data_dir, headers={"X-Forwarded-Proto": "https"})
    assert _is_secure(_refresh_cookie_header(r))


def test_untrusted_forwarded_proto_is_ignored(tmp_data_dir, client):
    """Without VW_TRUST_PROXY_ORIGIN the header is just client input.

    Honouring it unconditionally would let any caller re-arm the flag and
    break its own session -- and, worse, would make the flag a thing the
    client decides rather than a fact about the connection.
    """
    r = _login(client, tmp_data_dir, headers={"X-Forwarded-Proto": "https"})
    assert not _is_secure(_refresh_cookie_header(r))


def test_trusted_forwarded_proto_http_keeps_cookie_storable(tmp_data_dir, client):
    """Trusted proxy reporting plain http outranks our own socket."""
    client.app.state.settings = _with(client, trust_proxy_origin=True)
    r = _login(client, tmp_data_dir, headers={"X-Forwarded-Proto": "http"})
    assert not _is_secure(_refresh_cookie_header(r))


def test_forwarded_proto_chain_uses_client_most_hop(tmp_data_dir, client):
    """`X-Forwarded-Proto: https, http` -- the browser's hop is the first."""
    client.app.state.settings = _with(client, trust_proxy_origin=True)
    r = _login(client, tmp_data_dir, headers={"X-Forwarded-Proto": "https, http"})
    assert _is_secure(_refresh_cookie_header(r))


def test_https_frontend_origin_implies_secure(tmp_data_dir, client):
    """A pinned https:// public URL is a statement about the browser's scheme.

    This is the "Caddy in front, VW_TRUST_PROXY_ORIGIN never set" deployment:
    nothing forwards a trusted header, our socket is plaintext, and the only
    evidence that the browser is on TLS is the origin the operator configured.
    """
    client.app.state.settings = _with(client, allowed_origins=("https://llm.example.com",))
    r = _login(client, tmp_data_dir)
    assert _is_secure(_refresh_cookie_header(r))


def test_mixed_scheme_frontend_origin_stays_storable(tmp_data_dir, client):
    """Some browsers really are on http, so the flag would lock them out."""
    client.app.state.settings = _with(
        client,
        allowed_origins=("https://llm.example.com", "http://llm.internal:8080"),
    )
    r = _login(client, tmp_data_dir)
    assert not _is_secure(_refresh_cookie_header(r))


# ---------------------------------------------------------------------------
# The two cookies must agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected_secure",
    [("/api/csrf", False), ("https://testserver/api/csrf", True)],
)
def test_csrf_cookie_follows_the_same_scheme(tmp_data_dir, client, url, expected_secure):
    r = client.get(url)
    assert r.status_code == 200
    minted = [raw for raw in r.headers.get_list("set-cookie") if raw.startswith("vw_csrf_id=")]
    assert minted, "first request must mint vw_csrf_id"
    assert _is_secure(minted[0]) is expected_secure


def test_both_cookies_agree_over_plain_http(tmp_data_dir, client):
    """The asymmetry WAS the bug -- pin that they can never diverge again."""
    csrf = client.get("/api/csrf")
    csrf_cookie = next(
        raw for raw in csrf.headers.get_list("set-cookie") if raw.startswith("vw_csrf_id=")
    )
    login = _login(client, tmp_data_dir)
    assert _is_secure(csrf_cookie) == _is_secure(_refresh_cookie_header(login))


# ---------------------------------------------------------------------------
# Logout must be able to remove what login set
# ---------------------------------------------------------------------------


def test_logout_deletion_matches_the_cookie_it_set(tmp_data_dir, client):
    """Over HTTPS the deletion must also carry Secure, or browsers reject it."""
    from tests.conftest import csrf_header

    login = _login(client, tmp_data_dir, url="https://testserver/api/auth/login")
    jwt = login.json()["access_token"]
    headers = {
        "Authorization": f"Bearer {jwt}",
        "Origin": "https://testserver",
        **csrf_header(client),
    }
    r = client.post("https://testserver/api/auth/logout", headers=headers)
    assert r.status_code == 204, r.text
    assert _is_secure(_refresh_cookie_header(r))


# ---------------------------------------------------------------------------


def _with(client, **overrides):
    """A copy of the live Settings with fields replaced."""
    import dataclasses

    return dataclasses.replace(client.app.state.settings, **overrides)
