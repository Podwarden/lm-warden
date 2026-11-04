"""A pull must survive the things that killed one on lmwarden.com (2026-11-04).

A 27B pull (18 shards) was 1.2 GB in when the api process died. Three
things then went wrong, one per test group below:

1. The download ran inside the api process, so whatever killed it took the
   whole API down (and every chat with it). The download now runs in a child
   process: a crash there fails the pull, not the warden.
2. huggingface_hub >= 1.x writes each attempt to ``<etag>.<uuid8>.incomplete``
   and never resumes such a file, so every interrupted attempt leaves dead
   partial files that the progress counter then counts. A new pull clears
   them first.
3. (boot reconcile, tests/unit/runtime/test_boot_reconcile.py) the restart
   zeroed the row's progress and left it 'registered' with nothing running —
   the UI showed a busy bar at "0 B" forever.
"""

from __future__ import annotations

import sqlite3
import sys

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow


async def _seed(db_path, model_id="m1", repo="o/r", status="registered"):
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await ModelRepo(db).insert(
            ModelRow(
                id=model_id,
                served_model_name=model_id,
                hf_repo=repo,
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
                pulled_bytes=0,
                pulled_total=None,
                last_error=None,
            )
        )


# --------------------------------------------------------------------------
# 1. the download runs in a child process
# --------------------------------------------------------------------------


def _emit(payload: str) -> str:
    """Python source for one result line; ``payload`` is an expression."""
    from app.models.pull_task import WORKER_RESULT_MARKER

    return "print(" + repr(WORKER_RESULT_MARKER) + " + json.dumps(" + payload + "))"


def _fake_worker(monkeypatch, code: str) -> None:
    from app.models import pull_task as pt

    monkeypatch.setattr(pt, "_worker_argv", lambda: [sys.executable, "-c", code])


async def test_a_crashing_downloader_fails_the_pull_not_the_api(tmp_data_dir, monkeypatch):
    """os.abort() in the child stands in for a native crash (hf_xet is Rust)
    or an OOM kill. The api process must live on and record why."""
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed(settings.db_path)

    async def fake_estimate(*a, **kw):
        return 1024

    async def no_shortage(*a, **kw):
        return None

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", no_shortage)
    _fake_worker(monkeypatch, "import os, sys; sys.stdin.read(); os.abort()")

    await pt.run_pull("m1", settings)

    with sqlite3.connect(settings.db_path) as db:
        status, err = db.execute("SELECT status, last_error FROM models WHERE id='m1'").fetchone()
    assert status == "failed"
    assert "downloader process" in err
    assert "signal" in err


async def test_worker_success_returns_the_snapshot_path(tmp_data_dir, monkeypatch):
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    _fake_worker(
        monkeypatch,
        "import json, sys; req = json.loads(sys.stdin.read()); "
        "print('noise on stdout'); " + _emit("{'ok': True, 'path': '/snap/' + req['repo_id']}"),
    )

    path = await pt._snapshot_download("m1", settings, "o/r", "main", "hf_secret")

    assert path == "/snap/o/r"


async def test_worker_receives_the_token_on_stdin_not_argv(tmp_data_dir, monkeypatch):
    """argv is world-readable in /proc; the HF token must not be in it."""
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    _fake_worker(
        monkeypatch,
        "import json, sys; req = json.loads(sys.stdin.read()); "
        "assert 'hf_secret' not in ' '.join(sys.argv); "
        + _emit("{'ok': True, 'path': req['token']}"),
    )

    assert await pt._snapshot_download("m1", settings, "o/r", "main", "hf_secret") == "hf_secret"


async def test_worker_auth_refusal_maps_to_the_typed_envelope(tmp_data_dir, monkeypatch):
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    _fake_worker(
        monkeypatch,
        "import json, sys; sys.stdin.read(); "
        + _emit("{'ok': False, 'auth': True, 'error': 'GatedRepoError: no'}"),
    )

    with pytest.raises(pt.PullAuthRequired):
        await pt._snapshot_download("m1", settings, "o/r", "main", None)


async def test_worker_ordinary_error_is_reported_verbatim(tmp_data_dir, monkeypatch):
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    _fake_worker(
        monkeypatch,
        "import json, sys; sys.stdin.read(); "
        + _emit("{'ok': False, 'auth': False, 'error': 'OSError: disk full'}"),
    )

    with pytest.raises(pt.PullWorkerError, match="OSError: disk full"):
        await pt._snapshot_download("m1", settings, "o/r", "main", None)


def test_real_worker_module_reports_a_failure_as_a_result_line(tmp_path):
    """The real entry point, end to end, offline: an invalid repo id fails
    inside huggingface_hub before any network call, and the worker must turn
    that into a result line rather than a traceback with no verdict."""
    import json
    import subprocess

    from app.models import pull_task as pt

    req = {
        "repo_id": "not a valid repo id!!",
        "revision": "main",
        "cache_dir": str(tmp_path),
        "token": None,
        "allow_patterns": None,
    }
    proc = subprocess.run(
        pt._worker_argv(),
        input=json.dumps(req).encode(),
        capture_output=True,
        timeout=60,
        cwd=str(pt._APP_ROOT),
        env={"HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"},
    )
    lines = [
        ln for ln in proc.stdout.decode().splitlines() if ln.startswith(pt.WORKER_RESULT_MARKER)
    ]
    assert lines, proc.stderr.decode()[-2000:]
    result = json.loads(lines[-1][len(pt.WORKER_RESULT_MARKER) :])
    assert result["ok"] is False
    assert result["error"]


def test_worker_and_parent_agree_on_the_result_marker():
    from app.models import pull_task, pull_worker

    assert pull_worker.WORKER_RESULT_MARKER == pull_task.WORKER_RESULT_MARKER


# --------------------------------------------------------------------------
# 2. dead partial files from earlier attempts are cleared
# --------------------------------------------------------------------------

_SHA = "1d87b85e29b46181c619ef9f9175cdbf1c42e7261a9c49c9b4f126c5ca6b51e3"


def _blobs(cache_dir, repo="o/r"):
    d = cache_dir / ("models--" + repo.replace("/", "--")) / "blobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_purge_removes_only_uuid_suffixed_incompletes(tmp_path):
    from app.models.pull_task import purge_orphan_incompletes

    blobs = _blobs(tmp_path)
    (blobs / f"{_SHA}.3c1d9d34.incomplete").write_bytes(b"x" * 10)  # hf >= 1.x attempt
    (blobs / f"{_SHA[:40]}.0b3da969.incomplete").write_bytes(b"")  # git-sha etag
    (blobs / f"{_SHA}.incomplete").write_bytes(b"y" * 5)  # resumable (older hf)
    (blobs / _SHA).write_bytes(b"z" * 7)  # finished blob

    freed = purge_orphan_incompletes(tmp_path, "o/r")

    assert freed == 10
    assert sorted(p.name for p in blobs.iterdir()) == sorted([f"{_SHA}.incomplete", _SHA])


def test_purge_with_no_cache_dir_is_a_no_op(tmp_path):
    from app.models.pull_task import purge_orphan_incompletes

    assert purge_orphan_incompletes(tmp_path, "never/pulled") == 0


async def test_run_pull_clears_dead_partials_before_downloading(tmp_data_dir, monkeypatch):
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed(settings.db_path)
    blobs = _blobs(settings.hf_cache_dir)
    orphan = blobs / f"{_SHA}.3c1d9d34.incomplete"
    orphan.write_bytes(b"x" * 100)
    seen: dict[str, bool] = {}

    async def fake_estimate(*a, **kw):
        return 1024

    async def no_shortage(*a, **kw):
        return None

    async def fake_download(*a, **kw):
        seen["orphan_at_download"] = orphan.exists()
        return "/snap"

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", no_shortage)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)

    assert seen == {"orphan_at_download": False}


async def test_run_pull_leaves_partials_alone_while_another_pull_of_the_repo_runs(
    tmp_data_dir, monkeypatch
):
    """Two rows can share a repo. The other pull's live attempt file looks
    exactly like an orphan, so deleting it would break that pull."""
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed(settings.db_path, model_id="m1")
    live = _blobs(settings.hf_cache_dir) / f"{_SHA}.3c1d9d34.incomplete"
    live.write_bytes(b"x")
    monkeypatch.setattr(pt, "_ACTIVE_REPOS", {"o/r": 1})

    async def fake_estimate(*a, **kw):
        return 1024

    async def no_shortage(*a, **kw):
        return None

    async def fake_download(*a, **kw):
        return "/snap"

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", no_shortage)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)

    assert live.exists()
    assert pt._ACTIVE_REPOS == {"o/r": 1}  # our own count was released


async def test_a_second_run_of_the_same_pull_does_nothing(tmp_data_dir, monkeypatch):
    """Double-click, or an operator POST racing the boot-time resume: two
    run_pull tasks for one model. Only one may download; the second must not
    purge the first one's live partial file either."""
    import asyncio

    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed(settings.db_path)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_estimate(*a, **kw):
        return 1024

    async def no_shortage(*a, **kw):
        return None

    async def slow_download(*a, **kw):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "/snap"

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", no_shortage)
    monkeypatch.setattr(pt, "_snapshot_download", slow_download)

    first = asyncio.create_task(pt.run_pull("m1", settings))
    await started.wait()
    assert pt.pull_is_running("m1")
    await pt.run_pull("m1", settings)  # returns at once
    release.set()
    await first

    assert calls == 1
    assert not pt.pull_is_running("m1")


def test_worker_inherits_isolated_and_no_bytecode_flags(monkeypatch):
    """The macOS bundle runs `python -I -B`; the child must too, or user
    site-packages can shadow huggingface_hub and __pycache__ lands inside the
    signed .app."""
    import types

    from app.models import pull_task as pt

    monkeypatch.setattr(pt.sys, "flags", types.SimpleNamespace(isolated=1, dont_write_bytecode=1))
    argv = pt._worker_argv()
    assert argv[1:3] == ["-I", "-B"]
    assert argv[-2:] == ["-m", "app.models.pull_worker"]
