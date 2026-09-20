"""An admin token on the control API (spec 2026-09-19, decisions 2-10):

  * accepted by require_jwt, acting as its owner; principal recorded
  * the same refusals as an inference key: expired/revoked 401, paused 403,
    unknown 401; a rotated predecessor works through its grace window only
  * CSRF skipped for `Bearer vwa_...`, and only for that
  * never a /v1 credential; stress runs accept it; the ticket mint refuses it
  * a session-only route identifies the token (so its refusal can be
    audited) and then refuses it with 403 `session_only`
"""

import sqlite3
from datetime import timedelta

import pytest
from fastapi import Depends, HTTPException, Request

from app.auth.admin_audit import STATE_KEY, AdminAuditContext
from app.auth.deps import require_jwt, require_session
from app.db.repos.tokens import sqlite_utc_in, sqlite_utc_now
from tests.conftest import bearer, csrf_header, jwt_login, seed_admin_token


def test_an_admin_token_reads_the_control_api(client, seeded_db):
    _, secret = seed_admin_token(seeded_db)
    r = client.get("/api/tokens", headers=bearer(secret))
    assert r.status_code == 200, r.text


def test_it_acts_as_its_owner_and_records_the_principal(client, seeded_db):
    tid, secret = seed_admin_token(seeded_db, created_by="ops")

    async def whoami(request: Request, user: str = Depends(require_jwt)) -> dict[str, str]:
        return {"user": user, "principal": request.state.principal}

    client.app.add_api_route("/api/_whoami_test", whoami)
    r = client.get("/api/_whoami_test", headers=bearer(secret))
    assert r.json() == {"user": "ops", "principal": f"admin_token:{tid}"}
    r = client.get("/api/_whoami_test", headers=jwt_login(client))
    assert r.json() == {"user": "admin", "principal": "session"}


def test_use_stamps_last_used_at(client, seeded_db):
    tid, secret = seed_admin_token(seeded_db)
    client.get("/api/tokens", headers=bearer(secret))
    with sqlite3.connect(seeded_db) as db:
        (last_used,) = db.execute(
            "SELECT last_used_at FROM api_tokens WHERE id = ?", (tid,)
        ).fetchone()
    assert last_used is not None


@pytest.mark.parametrize(
    ("state", "status", "detail"),
    [
        ({"expires_at": "past"}, 401, "token expired"),
        ({"revoked_at": "past"}, 401, "token revoked"),
        ({"paused_at": "now"}, 403, "token paused"),
        # A refreshed predecessor: revoked_at in the future = grace window.
        ({"rotated_at": "now", "revoked_at": "future"}, 200, None),
        ({"rotated_at": "now", "revoked_at": "past"}, 401, "token revoked"),
    ],
)
def test_token_life(client, seeded_db, state, status, detail):
    when = {
        "past": sqlite_utc_in(timedelta(minutes=-1)),
        "now": sqlite_utc_now(),
        "future": sqlite_utc_in(timedelta(hours=1)),
    }
    _, secret = seed_admin_token(seeded_db, **{k: when[v] for k, v in state.items()})
    r = client.get("/api/tokens", headers=bearer(secret))
    assert r.status_code == status, r.text
    if detail is not None:
        assert r.json()["detail"] == detail


def test_an_unknown_or_non_admin_vwa_secret_is_unknown(client, seeded_db):
    r = client.get("/api/tokens", headers=bearer("vwa_" + "z" * 56))
    assert (r.status_code, r.json()["detail"]) == (401, "unknown token")
    _, secret = seed_admin_token(seeded_db, scope="inference")
    r = client.get("/api/tokens", headers=bearer(secret))
    assert (r.status_code, r.json()["detail"]) == (401, "unknown token")


def test_the_bearer_scheme_is_case_insensitive(client, seeded_db):
    _, secret = seed_admin_token(seeded_db)
    r = client.get("/api/tokens", headers={"Authorization": f"bearer {secret}"})
    assert r.status_code == 200, r.text


def test_an_admin_token_skips_csrf(client, seeded_db):
    _, secret = seed_admin_token(seeded_db)
    r = client.post("/api/tokens", json={"name": "from-a-script"}, headers=bearer(secret))
    assert r.status_code == 201, r.text


def test_a_session_without_csrf_is_still_refused(client, seeded_db):
    r = client.post("/api/tokens", json={"name": "x"}, headers=jwt_login(client))
    assert (r.status_code, r.json()["detail"]) == (403, "csrf token invalid")


def test_a_bogus_admin_token_passes_csrf_but_not_auth(client, seeded_db):
    r = client.post("/api/tokens", json={"name": "x"}, headers=bearer("vwa_" + "z" * 56))
    assert r.status_code == 401


def test_v1_refuses_an_admin_token(client, seeded_db):
    _, secret = seed_admin_token(seeded_db)
    r = client.get("/v1/models", headers=bearer(secret))
    assert (r.status_code, r.json()["detail"]) == (401, "invalid token format")


def test_v1_refuses_a_non_inference_row_even_with_a_vw_secret(client, seeded_db):
    _, secret = seed_admin_token(seeded_db, plaintext="vw_" + "q" * 56)  # scope 'admin'
    r = client.get("/v1/models", headers=bearer(secret))
    assert (r.status_code, r.json()["detail"]) == (401, "unknown token")


def test_stress_routes_accept_an_admin_token(client, seeded_db):
    _, secret = seed_admin_token(seeded_db)
    r = client.get("/api/models/nope/capabilities", headers=bearer(secret))
    assert r.status_code == 404, r.text  # past the operator gate; no such model


def test_the_stress_refusal_names_inference_tokens(client, seeded_db):
    session = {**jwt_login(client), **csrf_header(client)}
    inference = client.post("/api/tokens", json={"name": "k"}, headers=session).json()["plaintext"]
    r = client.get("/api/models/nope/capabilities", headers=bearer(inference))
    assert r.status_code == 403
    assert r.json()["detail"]["error_code"] == "operator_only"
    assert "Inference tokens" in r.json()["detail"]["message"]


def test_the_sse_ticket_mint_is_session_only(client, seeded_db):
    _, secret = seed_admin_token(seeded_db)
    r = client.post("/api/auth/sse-ticket", json={"path": "/api/stats/live"}, headers=bearer(secret))
    assert r.status_code == 403
    assert r.json()["detail"]["error_code"] == "session_only"


def test_a_session_only_route_refuses_the_kind_not_the_state(client, seeded_db):
    """Only identified, not validated: an expired admin token is still an
    admin token, and still 403 `session_only`. An unknown vwa_ secret is 401."""
    _, secret = seed_admin_token(seeded_db, expires_at=sqlite_utc_in(timedelta(minutes=-1)))
    body = {"path": "/api/stats/live"}
    r = client.post("/api/auth/sse-ticket", json=body, headers=bearer(secret))
    assert (r.status_code, r.json()["detail"]["error_code"]) == (403, "session_only")
    r = client.post("/api/auth/sse-ticket", json=body, headers=bearer("vwa_" + "z" * 56))
    assert (r.status_code, r.json()["detail"]) == (401, "unknown token")


def _request(app, headers: dict[str, str]) -> Request:
    return Request({
        "type": "http",
        "app": app,
        "method": "POST",
        "path": "/api/auth/sse-ticket",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("192.0.2.9", 50000),
        "state": {},
    })


async def test_require_session_sets_the_audit_context_before_refusing(live_app, tmp_data_dir):
    """A leaked admin token probing a session-only route leaves a trail: the
    token is identified and the audit context set BEFORE the 403."""
    tid, secret = seed_admin_token(tmp_data_dir / "vllm-warden.db", created_by="ops")
    request = _request(live_app, {**bearer(secret), "X-Forwarded-For": "198.51.100.7"})
    with pytest.raises(HTTPException) as exc:
        await require_session(request)
    assert exc.value.status_code == 403
    assert getattr(request.state, STATE_KEY) == AdminAuditContext(
        token_id=tid, username="ops", client_ip="198.51.100.7", peer_ip="192.0.2.9",
    )


async def test_an_unknown_admin_secret_sets_no_audit_context(live_app):
    request = _request(live_app, bearer("vwa_" + "z" * 56))
    with pytest.raises(HTTPException) as exc:
        await require_session(request)
    assert exc.value.status_code == 401
    assert getattr(request.state, STATE_KEY, None) is None


async def test_a_refused_known_token_carries_the_audit_context(live_app, tmp_data_dir):
    tid, secret = seed_admin_token(
        tmp_data_dir / "vllm-warden.db", revoked_at=sqlite_utc_in(timedelta(minutes=-1))
    )
    request = _request(live_app, bearer(secret))
    with pytest.raises(HTTPException) as exc:
        await require_jwt(request)
    assert (exc.value.status_code, exc.value.detail) == (401, "token revoked")
    assert getattr(request.state, STATE_KEY).token_id == tid
