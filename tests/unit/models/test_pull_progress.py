"""Regression tests for the SSE pull-progress stream.

Pins the v2026.05.10.2 fix: pull progress was never persisted (pulled_bytes
and pulled_total stayed at insert-time defaults of 0/None), so the FE
progress bar never updated. This file covers:

  1. The SSE endpoint /api/models/{id}/pull/progress streams the current
     row state and terminates when status leaves pulling/registered.
  2. run_pull writes pulled_total upfront and pulled_bytes at completion,
     so the SSE consumer sees real numbers across the lifecycle.
"""

import json
import sqlite3

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from tests.conftest import csrf_header, jwt_login, seed_admin_user


# #55 fix — route through the shared barrier+retry helpers.
def _seed_login(client, tmp_data_dir):
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0, 1])


def _jwt_login(client, username="admin", password="hunter2"):
    return jwt_login(client, username=username, password=password)


async def _seed_model(db_path, *, status="pulling", pulled_bytes=0, pulled_total=None):
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await ModelRepo(db).insert(
            ModelRow(
                id="m1",
                served_model_name="m1",
                hf_repo="o/r",
                hf_revision="main",
                gpu_indices=[0],
                tensor_parallel_size=1,
                dtype=None,
                max_model_len=None,
                gpu_memory_utilization=0.9,
                trust_remote_code=False,
                extra_args=[],
                extra_env={},
                status=status,
                pulled_bytes=pulled_bytes,
                pulled_total=pulled_total,
                last_error=None,
            )
        )


def _create_model_via_api(client, served="x", auth=None):
    h = csrf_header(client)
    headers = {**auth, **h} if auth else h
    r = client.post(
        "/api/models",
        json={
            "served_model_name": served,
            "hf_repo": "o/r",
            "gpu_indices": [0],
        },
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _set_row(db_path, model_id, **fields):
    """Patch arbitrary columns on a model row via raw sqlite."""
    cols = ", ".join(f"{k} = ?" for k in fields)
    with sqlite3.connect(db_path) as db:
        db.execute(f"UPDATE models SET {cols} WHERE id = ?", (*fields.values(), model_id))
        db.commit()


def test_progress_endpoint_streams_row_state(tmp_data_dir, client):
    """SSE stream emits current row JSON for a 'pulled' row and terminates
    (status is not pulling/registered, so the generator returns after one
    yield). Pins the response shape consumed by _card.html."""
    client.get("/healthz")
    _seed_login(client, tmp_data_dir)
    auth = _jwt_login(client)
    mid = _create_model_via_api(client, auth=auth)
    _set_row(
        tmp_data_dir / "vllm-warden.db", mid, status="pulled", pulled_bytes=1000, pulled_total=2000
    )

    with client.stream("GET", f"/api/models/{mid}/pull/progress", headers=auth) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        # Anti-buffering headers (#50). Without these, nginx/Caddy
        # buffer the SSE response, so the first-setup operator sees
        # the progress bar stay at 0% and then jump to 100% rather
        # than tick smoothly — the worst UX moment for this endpoint
        # (it runs during initial model pull, the user's first real
        # interaction with the app).
        assert r.headers["x-accel-buffering"] == "no"
        assert r.headers["cache-control"] == "no-cache"
        body = "".join(r.iter_text())

    payloads = [line[len("data: ") :] for line in body.splitlines() if line.startswith("data: ")]
    assert payloads, f"no data: lines in {body!r}"
    first = json.loads(payloads[0])
    assert first["status"] == "pulled"
    assert first["bytes"] == 1000
    assert first["total"] == 2000
    assert first["last_error"] is None


def test_progress_endpoint_emits_missing_when_row_absent(tmp_data_dir, client):
    """Missing row → SSE emits status=missing then closes. Does NOT 404,
    because EventSource on the FE doesn't surface HTTP errors usefully."""
    client.get("/healthz")
    _seed_login(client, tmp_data_dir)
    auth = _jwt_login(client)

    with client.stream("GET", "/api/models/nope/pull/progress", headers=auth) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert '"status": "missing"' in body, body


def test_progress_endpoint_requires_auth(tmp_data_dir, client):
    """Auth dep returns 401 without a session. Seed setup as done first
    so the setup-gate redirect doesn't mask the auth check."""
    client.get("/healthz")
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "UPDATE setup_state SET step = 'done', draft = ? WHERE id = 1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.commit()
    r = client.get("/api/models/m1/pull/progress")
    assert r.status_code == 401


async def test_run_pull_persists_total_and_final_bytes(tmp_data_dir, monkeypatch):
    """run_pull writes pulled_total upfront (drives the bar denominator)
    and a final pulled_bytes at completion (fills the bar). Without this,
    the FE progress bar can't render anything useful."""
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed_model(settings.db_path, status="registered")

    async def fake_estimate(*a, **kw):
        return 4096

    async def fake_disk_check(*a, **kw):
        return None

    async def fake_download(*a, **kw):
        return "/tmp/fake"

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)
    # Simulate an in-flight download size at completion.
    monkeypatch.setattr(pt, "_snapshot_dir_size", lambda cache, repo: 4096)

    await pt.run_pull("m1", settings)

    with sqlite3.connect(settings.db_path) as db:
        status, pulled_bytes, pulled_total = db.execute(
            "SELECT status, pulled_bytes, pulled_total FROM models WHERE id = 'm1'"
        ).fetchone()
    assert status == "pulled"
    assert pulled_total == 4096
    assert pulled_bytes == 4096


async def test_snapshot_dir_size_walks_blobs(tmp_path):
    """_snapshot_dir_size sums file sizes under cache/models--o--r/blobs."""
    from app.models.pull_task import _snapshot_dir_size

    blobs = tmp_path / "models--o--r" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "a").write_bytes(b"x" * 100)
    (blobs / "b").write_bytes(b"y" * 250)
    assert _snapshot_dir_size(tmp_path, "o/r") == 350


async def test_snapshot_dir_size_returns_zero_when_missing(tmp_path):
    from app.models.pull_task import _snapshot_dir_size

    assert _snapshot_dir_size(tmp_path, "o/r") == 0
