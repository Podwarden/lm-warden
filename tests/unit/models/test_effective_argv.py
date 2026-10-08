"""Endpoint tests for ``GET /api/models/{id}/effective-argv``.

This endpoint carries sub-project B's ONE deliberate observable change: the
response now begins with ``["vllm", "serve"]``, where before B it returned only
the post-binary tail. B's plan listed this file as "existing"; it did not
exist, so B's single behaviour change had no backend coverage at all. It does
now -- both the new shape and the parts that must NOT have moved.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import bcrypt

from tests.conftest import csrf_header


def _seed_done(db_path: Path) -> None:
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step = 'done', draft = ? WHERE id = 1",
            (json.dumps({"allowed_gpu_indices": [0, 1, 2, 3]}),),
        )
        db.commit()


def _auth(client) -> dict[str, str]:
    r = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _create_model(client, auth, **over) -> str:
    body = {
        "served_model_name": "demo",
        "hf_repo": "org/model",
        "gpu_indices": [0],
    }
    body.update(over)
    r = client.post("/api/models", json=body, headers={**auth, **csrf_header(client)})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_effective_argv_includes_argv_zero_and_its_subcommand(client, tmp_data_dir):
    """B's one deliberate observable change, stated as an assertion.

    Before B this endpoint returned the post-binary tail only -- the response
    model's own description said so -- because it called build_vllm_args
    directly. It now asks the row's backend for a LaunchPlan, and a LaunchPlan
    carries argv[0]. Anything downstream that prepended 'vllm serve' itself
    would now double it.
    """
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _auth(client)
    model_id = _create_model(client, auth)

    r = client.get(f"/api/models/{model_id}/effective-argv", headers=auth)
    assert r.status_code == 200, r.text
    argv = r.json()["argv"]

    assert argv[:2] == ["vllm", "serve"]
    # ...and the tail is unchanged: the flags still start at --model.
    assert argv[2] == "--model"
    assert argv[3] == "org/model"


def test_effective_argv_still_previews_the_placeholder_port(client, tmp_data_dir):
    """Not changed by B, and the byte-identity check at release time depends on
    it: both sides of that diff use port 10000, so the expected delta is
    exactly the two leading tokens and nothing else."""
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _auth(client)
    model_id = _create_model(client, auth)

    argv = client.get(f"/api/models/{model_id}/effective-argv", headers=auth).json()["argv"]
    assert argv[argv.index("--port") + 1] == "10000"


def test_effective_argv_binds_loopback_under_the_local_driver(client, tmp_data_dir):
    """#211's fail-narrow default survives the move onto the backend.

    The preview promises to show what the supervisor would run, and the bind
    host is driver-dependent, so the preview has to resolve it the same way --
    now via Backend.bind_host(driver_name) rather than the builder's own
    driver= parameter."""
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _auth(client)
    model_id = _create_model(client, auth)

    argv = client.get(f"/api/models/{model_id}/effective-argv", headers=auth).json()["argv"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


def test_effective_argv_requires_auth(client, tmp_data_dir):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _auth(client)
    model_id = _create_model(client, auth)
    assert client.get(f"/api/models/{model_id}/effective-argv").status_code == 401


def test_effective_argv_404s_on_an_unknown_model(client, tmp_data_dir):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _auth(client)
    assert client.get("/api/models/nope/effective-argv", headers=auth).status_code == 404
