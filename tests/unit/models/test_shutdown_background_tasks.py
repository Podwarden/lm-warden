"""CI audit A1 (#270) — the load/pull/unload runners must not outlive the app.

The model routes used to start their runners with a bare
``asyncio.create_task`` and drop the reference. CPython keeps only a weak
reference to a task (bpo-88831), so nothing could cancel them at shutdown:
a shutdown mid-unload abandoned the runner, which in tests kept polling (or
holding an ``open_db`` connection) after the loop closed — the "Event loop
is closed" flakes — and in production skipped the terminal row write and
the port release. The runners now register themselves in a set on
``app.state`` and the lifespan's finally cancels + gathers that set before
the DB-using loops stop.

These tests drive the app's lifespan on their own event loop (like the
``live_app`` fixture) instead of through TestClient: TestClient's portal
cancels EVERY remaining task when it stops, which would mask a runner the
lifespan forgot to cancel. Observing the tasks after the lifespan's
``finally`` — while the loop is still open — is the production moment: a
runner still pending there is one the app abandoned.
"""

import asyncio
import shutil
import sqlite3
from unittest.mock import patch

import httpx
from httpx import ASGITransport

from app.main import build_app
from tests.conftest import seed_admin_user

# Coroutine names of the fire-and-forget runners the model routes spawn
# (app/models/routes_api.py): the unload closure, the load runner and the
# pull runner.
_RUNNER_NAMES = {"runner", "start_engine", "run_pull"}


def _seed_loaded_model(db_path, model_id: str) -> None:
    """Seed a setup-done admin and a model row already in 'loaded' status.

    No supervisor state is registered for it (no engine was ever spawned in
    this process), so ``ensure_unloadable`` and ``Supervisor.unload`` both
    find nothing to refuse — exactly the shape a real teardown has once the
    engine process is gone.
    """
    seed_admin_user(db_path, allowed_gpu_indices=[0, 1])
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes, pulled_total, last_error) "
            "VALUES (?, ?, 'o/r', 'main', '[]', 1, NULL, NULL, 0.9, 0, '[]', "
            "'loaded', 0, NULL, NULL)",
            (model_id, model_id),
        )
        db.commit()


def _record_create_tasks(monkeypatch) -> list[asyncio.Task]:
    """Wrap ``asyncio.create_task`` so every task the process creates is
    recorded; the wrapper returns the real task, so behaviour is unchanged."""
    created: list[asyncio.Task] = []
    real_create_task = asyncio.create_task

    def recording_create_task(coro, **kwargs):
        task = real_create_task(coro, **kwargs)
        created.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", recording_create_task)
    return created


async def _login_and_unload(app, model_id: str) -> httpx.Response:
    """Real login through the real routes, then POST the unload."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        r = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "hunter2"},
        )
        assert r.status_code == 200, r.text
        headers = {
            "Authorization": f"Bearer {r.json()['access_token']}",
        }
        r = await client.get("/api/csrf")
        assert r.status_code == 200, r.text
        headers["X-CSRF-Token"] = r.json()["csrf"]
        return await client.post(f"/api/models/{model_id}/unload", headers=headers)


async def test_unload_runner_is_cancelled_at_lifespan_exit(
    tmp_data_dir, migrated_db_template, monkeypatch
):
    """Red before R17: the unload runner was a bare create_task, so a
    shutdown with the teardown still in flight left it PENDING when the
    lifespan's finally ended — outliving the app. After the fix the
    finally cancels it and gathers it, so it is done (cancelled) at
    lifespan exit."""
    db_path = tmp_data_dir / "vllm-warden.db"
    shutil.copyfile(migrated_db_template, db_path)
    created = _record_create_tasks(monkeypatch)

    entered: list[str] = []

    async def hang_unload(self, model_id, *, force=False):
        entered.append(model_id)
        await asyncio.Event().wait()  # never set: the teardown never completes

    app = build_app()
    with patch("app.runtime.supervisor.Supervisor.unload", new=hang_unload):
        async with app.router.lifespan_context(app):
            _seed_loaded_model(db_path, "hang-model")
            r = await _login_and_unload(app, "hang-model")
            assert r.status_code == 202, r.text

            runners = [t for t in created if t.get_coro().__name__ == "runner"]
            assert len(runners) == 1, "the unload runner task was never started"
            # Prove the hang actually took: the runner entered the patched
            # teardown and is parked in it, not done through some shortcut.
            for _ in range(100):
                if entered:
                    break
                await asyncio.sleep(0.02)
            assert entered == ["hang-model"], (
                "the patched teardown was never entered; the runner took a "
                "shortcut instead of hanging"
            )
            assert not runners[
                0
            ].done(), "the runner finished before shutdown; the hang path was not driven"
        # --- the lifespan's finally has just run; the loop is still open ---

    loop_pending = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and t.get_coro().__name__ in _RUNNER_NAMES
    ]
    assert not loop_pending, (
        f"runner task(s) still pending on the loop at lifespan exit: "
        f"{loop_pending!r} — a shutdown mid-unload abandoned the runner "
        "(CI audit A1): it would keep polling (or holding an open_db "
        "connection) after the loop closes"
    )
    assert runners[
        0
    ].cancelled(), f"the unload runner was not cancelled at shutdown: {runners[0]!r}"


async def test_normal_unload_still_completes_and_drains_the_tracker(
    tmp_data_dir, migrated_db_template, monkeypatch
):
    """Control: a teardown that actually finishes still drives the row to
    its terminal state, and the finished runner discards itself from the
    app's tracker instead of accumulating in it."""
    db_path = tmp_data_dir / "vllm-warden.db"
    shutil.copyfile(migrated_db_template, db_path)
    created = _record_create_tasks(monkeypatch)

    async def quick_unload(self, model_id, *, force=False):
        await asyncio.sleep(0.05)

    app = build_app()
    with patch("app.runtime.supervisor.Supervisor.unload", new=quick_unload):
        async with app.router.lifespan_context(app):
            _seed_loaded_model(db_path, "ok-model")
            r = await _login_and_unload(app, "ok-model")
            assert r.status_code == 202, r.text

            row = None
            for _ in range(100):
                with sqlite3.connect(db_path) as db:
                    row = db.execute("SELECT status FROM models WHERE id='ok-model'").fetchone()
                if row and row[0] == "pulled":
                    break
                await asyncio.sleep(0.05)
            assert (
                row is not None and row[0] == "pulled"
            ), f"expected terminal 'pulled', got {row[0] if row else None}"

    runners = [t for t in created if t.get_coro().__name__ == "runner"]
    assert len(runners) == 1
    assert runners[0].done() and not runners[0].cancelled()
    assert (
        app.state.model_bg_tasks == set()
    ), f"a finished runner is stuck in the tracker: {app.state.model_bg_tasks!r}"
