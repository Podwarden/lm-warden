"""Owner decision 2026-11-22 (replaces D8's two switches): a key holder may
open captures of their OWN requests whenever Prompt Lens is on for the platform.
There is no per-key switch any more; the 0059 column is inert."""

from __future__ import annotations

import dataclasses

from tests.conftest import csrf_header
from tests.unit.stats.test_forest_routes import _forest_login, _ready_key


def test_available_follows_the_platform_only(tmp_data_dir, client):
    _db, plaintext, _admin = _ready_key(client, tmp_data_dir)
    k = _forest_login(client, plaintext)
    me = client.get("/api/me/key", headers=k).json()
    assert me["prompt_lens_view_available"] is False
    assert "godmode_self_view" not in me
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=True
    )
    assert client.get("/api/me/key", headers=k).json()["prompt_lens_view_available"] is True


def test_the_per_key_switch_is_gone(tmp_data_dir, client):
    db_path, _plaintext, admin = _ready_key(client, tmp_data_dir)
    h = {**admin, **csrf_header(client)}
    assert "godmode_self_view" not in client.get("/api/tokens/tok1", headers=h).json()
    # An old client sending the field changes nothing (the column stays inert).
    client.patch("/api/tokens/tok1", json={"godmode_self_view": True}, headers=h)
    import sqlite3

    with sqlite3.connect(db_path) as db:
        assert (
            db.execute("SELECT godmode_self_view FROM api_tokens WHERE id='tok1'").fetchone()[0]
            == 0
        )
