"""Per-key permissions (owner decision 2026-11-26, migration 0066): ``prompt_lens`` and
``engine_facts``, set by the admin on create and edit, carried by a rotation, shown to the
holder read-only, and enforced on the server wherever Prompt Lens is offered."""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path
from typing import Any

from tests.conftest import bearer, csrf_header, jwt_login
from tests.unit.proxy.test_prompt_lens_forward import _enable_prompt_lens
from tests.unit.stats.test_forest_routes import _forest_login

SESSIONS = "/api/me/prompt-lens/sessions"


def _admin(client: Any, seeded_db: Path) -> dict[str, str]:
    return {**jwt_login(client), **csrf_header(client)}


def _create(client: Any, h: dict[str, str], **body: Any) -> dict[str, Any]:
    r = client.post("/api/tokens", json={"name": "perm-key", **body}, headers=h)
    assert r.status_code == 201, r.text
    return r.json()


def test_a_new_key_starts_without_prompt_lens_and_with_engine_facts(client, seeded_db) -> None:
    h = _admin(client, seeded_db)
    made = _create(client, h)
    assert (made["prompt_lens"], made["engine_facts"]) == (False, True)
    got = client.get(f"/api/tokens/{made['id']}", headers=h).json()
    assert (got["prompt_lens"], got["engine_facts"]) == (False, True)
    ticked = _create(client, h, prompt_lens=True, engine_facts=False)
    assert (ticked["prompt_lens"], ticked["engine_facts"]) == (True, False)


def test_an_existing_key_keeps_prompt_lens_after_the_migration(client, seeded_db) -> None:
    with sqlite3.connect(seeded_db) as c:  # a row written before 0066 knew the columns
        c.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES "
            "('old1', 'old', 'vw_aaaa', 'h', 'inference')"
        )
        c.commit()
    h = _admin(client, seeded_db)
    got = client.get("/api/tokens/old1", headers=h).json()
    assert (got["prompt_lens"], got["engine_facts"]) == (True, True)


def test_the_admin_edits_both_and_each_change_is_audited(client, seeded_db, flush_audit) -> None:
    h = _admin(client, seeded_db)
    tid = _create(client, h)["id"]
    r = client.patch(
        f"/api/tokens/{tid}", json={"prompt_lens": True, "engine_facts": False}, headers=h
    )
    assert r.status_code == 200, r.text
    assert (r.json()["prompt_lens"], r.json()["engine_facts"]) == (True, False)
    # The same values again change nothing and write nothing.
    client.patch(f"/api/tokens/{tid}", json={"prompt_lens": True}, headers=h)
    flush_audit()
    with sqlite3.connect(seeded_db) as c:
        paths = [
            p
            for (p,) in c.execute(
                "SELECT path FROM admin_audit WHERE token_id = ? AND path LIKE '%key.%' ORDER BY ts",
                (tid,),
            )
        ]
    assert [p.split(" ")[-1] for p in paths] == ["key.prompt_lens:on", "key.engine_facts:off"]


def test_null_is_refused(client, seeded_db) -> None:
    h = _admin(client, seeded_db)
    tid = _create(client, h)["id"]
    for flag in ("prompt_lens", "engine_facts"):
        assert client.patch(f"/api/tokens/{tid}", json={flag: None}, headers=h).status_code == 422


def test_a_rotated_key_inherits_both(client, seeded_db) -> None:
    h = _admin(client, seeded_db)
    tid = _create(client, h, prompt_lens=True, engine_facts=False)["id"]
    r = client.post(f"/api/tokens/{tid}/rotate", json={"grace_hours": 1}, headers=h)
    assert r.status_code == 201, r.text
    new = client.get(f"/api/tokens/{r.json()['id']}", headers=h).json()
    assert (new["prompt_lens"], new["engine_facts"]) == (True, False)


def test_the_holder_sees_the_state_read_only(client, seeded_db) -> None:
    h = _admin(client, seeded_db)
    made = _create(client, h, prompt_lens=True, engine_facts=False)
    fh = _forest_login(client, made["plaintext"])
    me = client.get("/api/me/key", headers=fh).json()
    assert (me["prompt_lens"], me["engine_facts"]) == (True, False)
    # A holder cannot change it: the body field is not part of the holder's patch.
    r = client.patch("/api/me/key", json={"prompt_lens": False}, headers=fh)
    assert r.status_code == 422


def test_prompt_lens_is_refused_without_the_key_permission(client, seeded_db, tmp_data_dir) -> None:
    h = _admin(client, seeded_db)
    off = _create(client, h)
    on = _create(client, h, prompt_lens=True)
    _enable_prompt_lens(client)
    fh_off, fh_on = _forest_login(client, off["plaintext"]), _forest_login(client, on["plaintext"])
    assert client.get("/api/me/key", headers=fh_off).json()["prompt_lens_view_available"] is False
    assert client.get("/api/me/key", headers=fh_on).json()["prompt_lens_view_available"] is True
    refused = client.get(SESSIONS, headers=fh_off)
    assert refused.status_code == 404
    assert client.get(SESSIONS, headers=fh_on).status_code == 200
    for path in ("/api/me/prompt-lens/kept", "/api/me/prompt-lens/trends"):
        assert client.get(path, headers=fh_off).status_code == 404
    assert client.post("/api/me/prompt-lens/ticket", headers=fh_off).status_code == 404
    # The admin switches it on: the next call works, no sign-in again.
    client.patch(f"/api/tokens/{off['id']}", json={"prompt_lens": True}, headers=h)
    assert client.get(SESSIONS, headers=fh_off).status_code == 200


def test_the_platform_switch_still_decides(client, seeded_db) -> None:
    h = _admin(client, seeded_db)
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=False
    )
    made = _create(client, h, prompt_lens=True)
    fh = _forest_login(client, made["plaintext"])
    assert client.get("/api/me/key", headers=fh).json()["prompt_lens_view_available"] is False
    assert client.get(SESSIONS, headers=fh).status_code == 404


def test_an_inference_bearer_still_works_for_the_key_flags(client, seeded_db) -> None:
    h = _admin(client, seeded_db)
    made = _create(client, h)
    assert client.get("/api/me/key", headers=bearer(made["plaintext"])).status_code == 401
