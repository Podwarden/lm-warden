"""Integration tests for the HF cache management API.

Drives the real FastAPI app through ``TestClient``, seeds rows directly
in SQLite (mirroring the pattern in tests/unit/models/test_routes_crud.py),
and creates physical cache directories under ``VW_HF_CACHE_DIR`` so the
scanner has real on-disk state to walk.

Covers every case from the spec § Testing / Backend integration block.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tests.conftest import jwt_login, seed_admin_user

# ---------------------------------------------------------------------------
# Test helpers — mirror tests/unit/models/test_routes_crud.py
# ---------------------------------------------------------------------------


# #55 fix — local shims route through the shared barrier+retry helpers
# so this module picks up the WAL-race mitigation without touching the
# domain-specific ``_insert_model`` / ``_make_cache_dir`` helpers below.
def _seed_done(db_path: Path, allowed: list[int] | None = None) -> None:
    seed_admin_user(db_path, allowed_gpu_indices=allowed)


def _jwt_login(client, username: str = "admin", password: str = "hunter2") -> dict[str, str]:
    return jwt_login(client, username=username, password=password)


def _insert_model(
    db_path: Path,
    *,
    model_id: str,
    served_name: str,
    hf_repo: str,
    status: str,
    updated_at: str | None = None,
) -> None:
    """Insert a row directly — avoids hitting the create endpoint with its
    GPU-allocation policy (irrelevant to cache tests)."""
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models("
            "  id, served_model_name, hf_repo, hf_revision, gpu_indices,"
            "  tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization,"
            "  trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error,"
            "  extra_env"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                model_id,
                served_name,
                hf_repo,
                "main",
                json.dumps([0]),
                1,
                None,
                None,
                0.9,
                0,
                json.dumps([]),
                status,
                0,
                None,
                None,
                json.dumps({}),
            ),
        )
        if updated_at is not None:
            db.execute(
                "UPDATE models SET updated_at = ? WHERE id = ?",
                (updated_at, model_id),
            )
        db.commit()


def _make_cache_dir(cache_root: Path, repo: str, *, size: int = 1024) -> Path:
    """Drop a fake ``models--<org>--<name>/blobs/<sha>`` tree of ``size`` bytes."""
    safe = "models--" + repo.replace("/", "--", 1)
    target = cache_root / safe / "blobs"
    target.mkdir(parents=True, exist_ok=True)
    (target / "deadbeef").write_bytes(b"x" * size)
    return cache_root / safe


# ---------------------------------------------------------------------------
# GET /api/cache/models
# ---------------------------------------------------------------------------


def test_list_returns_scanner_joined_with_db_rows(tmp_data_dir, client):
    """Cached repo with a matching DB row returns non-empty
    ``matched_models``; orphan cache returns an empty list."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    cache_root = tmp_data_dir / "hf-cache"
    cache_root.mkdir(exist_ok=True)
    _make_cache_dir(cache_root, "Qwen/Qwen3-9B", size=2048)
    _make_cache_dir(cache_root, "Orphan/Repo", size=512)
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m1",
        served_name="qwen",
        hf_repo="Qwen/Qwen3-9B",
        status="pulled",
    )
    auth = _jwt_login(client)
    r = client.get("/api/cache/models", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    by_repo = {row["repo"]: row for row in body}
    assert set(by_repo) == {"Qwen/Qwen3-9B", "Orphan/Repo"}
    assert by_repo["Qwen/Qwen3-9B"]["matched_models"] == [
        {"id": "m1", "served_model_name": "qwen", "status": "pulled"},
    ]
    assert by_repo["Orphan/Repo"]["matched_models"] == []
    # Sanity-check size joining — we wrote 2048 bytes plus dir overhead
    # so size_bytes must be at least 2048.
    assert by_repo["Qwen/Qwen3-9B"]["size_bytes"] >= 2048


def test_list_requires_jwt(tmp_data_dir, client):
    """Cache endpoints inherit the standard ``require_jwt`` gate —
    anonymous requests return 401, not 200 with the cache list."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/cache/models")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# DELETE /api/cache/models/{repo:path}
# ---------------------------------------------------------------------------


def test_delete_active_row_refused(tmp_data_dir, client):
    """A row in ``loaded`` (or any ACTIVE_STATUSES member) must block
    the cache delete with 409 — and ``?force=true`` must NOT override
    this; only the ``pulled``/``idle`` warning is forcible."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    cache_root = tmp_data_dir / "hf-cache"
    cache_root.mkdir(exist_ok=True)
    _make_cache_dir(cache_root, "live/model")
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m1",
        served_name="live",
        hf_repo="live/model",
        status="loaded",
    )
    auth = _jwt_login(client)
    from tests.conftest import csrf_header

    r = client.delete(
        "/api/cache/models/live/model",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 409, r.text
    assert "active" in r.text.lower()
    # ``?force=true`` is not an escape hatch for active rows.
    r2 = client.delete(
        "/api/cache/models/live/model?force=true",
        headers={**auth, **csrf_header(client)},
    )
    assert r2.status_code == 409, r2.text
    # Dir still on disk.
    assert (cache_root / "models--live--model").exists()


def test_delete_idle_row_requires_force(tmp_data_dir, client):
    """``status=pulled`` is a benign-but-alive row — DELETE returns 409
    with a force-required message; the cache dir stays put."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    cache_root = tmp_data_dir / "hf-cache"
    cache_root.mkdir(exist_ok=True)
    _make_cache_dir(cache_root, "ok/model")
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m1",
        served_name="ok",
        hf_repo="ok/model",
        status="pulled",
    )
    auth = _jwt_login(client)
    from tests.conftest import csrf_header

    r = client.delete(
        "/api/cache/models/ok/model",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 409, r.text
    assert "force" in r.text.lower()
    assert (cache_root / "models--ok--model").exists()


def test_delete_with_force_succeeds_for_idle(tmp_data_dir, client):
    """``?force=true`` against a ``pulled`` row → 204, cache gone,
    DB row preserved (the row may still be valid metadata for a
    re-pull — see spec § Non-goals)."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    cache_root = tmp_data_dir / "hf-cache"
    cache_root.mkdir(exist_ok=True)
    _make_cache_dir(cache_root, "ok/model")
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m1",
        served_name="ok",
        hf_repo="ok/model",
        status="pulled",
    )
    auth = _jwt_login(client)
    from tests.conftest import csrf_header

    r = client.delete(
        "/api/cache/models/ok/model?force=true",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 204, r.text
    assert not (cache_root / "models--ok--model").exists()
    # Row preserved.
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        n = db.execute("SELECT COUNT(*) FROM models WHERE id = 'm1'").fetchone()[0]
    assert n == 1


def test_delete_orphan_succeeds_without_force(tmp_data_dir, client):
    """No matching row → straight to 204; ``?force`` not required."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    cache_root = tmp_data_dir / "hf-cache"
    cache_root.mkdir(exist_ok=True)
    _make_cache_dir(cache_root, "orphan/model")
    auth = _jwt_login(client)
    from tests.conftest import csrf_header

    r = client.delete(
        "/api/cache/models/orphan/model",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 204, r.text
    assert not (cache_root / "models--orphan--model").exists()


def test_delete_rejects_traversal_repo_path(tmp_data_dir, client):
    """Defence-in-depth: the DELETE handler validates ``repo`` against
    ``^[\\w.-]+/[\\w.-]+$`` BEFORE touching the DB or filesystem, and
    rejects with 400 anything that isn't a clean single-slash HF repo
    id.

    The downstream ``.exists()`` check would 404 these inputs even
    without the guard (``_safe_dirname`` only replaces the first ``/``,
    so multi-segment inputs construct two-segment on-disk paths that
    don't exist). But routing inputs that look like traversal attempts
    through a 400 at the boundary is the contract — see CR feedback on
    !MR for vllm-warden#114.

    Cases pinned:
      - ``a/b/c``           — multi-slash; ``_safe_dirname`` would smear
                              two segments onto disk if not for the guard
      - ``..%2Fetc%2Fpasswd`` — encoded traversal; httpx/Starlette decode
                                the percent-escapes server-side, then the
                                regex must catch ``..`` (not in alnum + ``_-.``)
      - ``has space/repo``  — whitespace; alphabet violation
      - alphabet violations — ``+`` outside the allowed set

    The httpx ``TestClient`` (httpx.URL) normalises ``..`` segments in the
    raw path before sending, so we can't drive a ``GET /a/../b`` directly
    — that's why this test exercises percent-encoded traversal plus the
    multi-segment and alphabet-violation cases that DO survive client-side
    normalisation. The plain ``..`` path is verified by the regex unit test
    on ``_REPO_RE`` itself (see scanner/route module docstrings).
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    (tmp_data_dir / "hf-cache").mkdir(exist_ok=True)
    auth = _jwt_login(client)
    from tests.conftest import csrf_header

    # Three-segment input — single-slash invariant.
    r = client.delete(
        "/api/cache/models/a/b/c",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 400, r.text
    assert "invalid repo" in r.text.lower()
    # Percent-encoded ``..`` traversal — the encoded form survives httpx
    # path normalisation; the server decodes, the regex catches ``..``
    # (the dot is allowed but the trailing ``/`` after two literal dots
    # would only validate if alphabet-clean — ``..`` org name with empty
    # name segment fails the alnum requirement after the slash).
    r = client.delete(
        "/api/cache/models/..%2Fetc%2Fpasswd",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 400, r.text
    # Alphabet violation — space is outside the allowed ``[\w.-]`` set.
    r = client.delete(
        "/api/cache/models/has space/repo",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 400, r.text
    # Alphabet violation — ``+`` not in the allowed set.
    r = client.delete(
        "/api/cache/models/org+plus/repo",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 400, r.text


def test_repo_validator_rejects_dotdot_traversal() -> None:
    """Pin the input-validator behaviour directly — the integration test
    above can't drive a literal ``..`` segment through httpx (the client
    normalises it out before the request leaves the harness), so this
    unit test pins ``_is_valid_repo``'s contract: any ``..`` token
    anywhere in the string rejects, multi-slash rejects, alphabet
    violations reject, every legitimate HF repo id passes.
    """
    from app.cache.routes_api import _is_valid_repo

    # Reject: traversal vocabulary (the ``..`` substring guard catches
    # these even though ``.`` is in the alphabet — see _is_valid_repo
    # docstring for the rationale).
    assert not _is_valid_repo("../../etc/passwd")
    assert not _is_valid_repo("../etc")
    assert not _is_valid_repo("a/../b")
    assert not _is_valid_repo("foo..bar/baz")
    # Reject: multi-slash.
    assert not _is_valid_repo("a/b/c")
    # Reject: empty segments.
    assert not _is_valid_repo("")
    assert not _is_valid_repo("/")
    assert not _is_valid_repo("/name")
    assert not _is_valid_repo("org/")
    # Reject: alphabet violations.
    assert not _is_valid_repo("has space/repo")
    assert not _is_valid_repo("org+plus/repo")
    # Accept: legitimate HF shapes (alnum, ``_-.``, single ``.`` ok).
    assert _is_valid_repo("Qwen/Qwen3.6-27B-GGUF")
    assert _is_valid_repo("meta-llama/Llama-3.2-1B-Instruct")
    assert _is_valid_repo("unsloth/Llama-3.2-1B-Instruct-GGUF")
    assert _is_valid_repo("a/b")
    assert _is_valid_repo("org.with.dots/name.with.dots")


def test_delete_no_cache_dir_returns_404(tmp_data_dir, client):
    """Endpoint is idempotent for "already gone" — second delete or
    typo-repo returns 404, not 500."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    (tmp_data_dir / "hf-cache").mkdir(exist_ok=True)
    auth = _jwt_login(client)
    from tests.conftest import csrf_header

    r = client.delete(
        "/api/cache/models/nonexistent/repo",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/cache/models/gc
# ---------------------------------------------------------------------------


def _hours_ago_iso(hours: float) -> str:
    """SQLite ``datetime('now')`` format: ``"YYYY-MM-DD HH:MM:SS"`` (UTC)."""
    return (datetime.now(UTC) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")


def test_gc_dry_run_returns_candidates_without_deleting(tmp_data_dir, client):
    """``dry_run=true`` returns the candidate list AND leaves all dirs
    intact. Both orphan and stale-failed cases appear in the preview."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    cache_root = tmp_data_dir / "hf-cache"
    cache_root.mkdir(exist_ok=True)
    _make_cache_dir(cache_root, "orphan/dead")
    _make_cache_dir(cache_root, "stale/failed")
    _make_cache_dir(cache_root, "keep/me")
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m_stale",
        served_name="stale",
        hf_repo="stale/failed",
        status="failed",
        updated_at=_hours_ago_iso(48),
    )
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m_keep",
        served_name="keep",
        hf_repo="keep/me",
        status="pulled",
    )
    auth = _jwt_login(client)
    from tests.conftest import csrf_header

    r = client.post(
        "/api/cache/models/gc?dry_run=true",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dry_run"] is True
    assert body["deleted_paths"] == []
    candidate_repos = sorted(c["repo"] for c in body["candidates"])
    assert candidate_repos == ["orphan/dead", "stale/failed"]
    # ``keep/me`` (status=pulled) must NOT appear.
    # Dirs intact:
    assert (cache_root / "models--orphan--dead").exists()
    assert (cache_root / "models--stale--failed").exists()
    assert (cache_root / "models--keep--me").exists()


def test_gc_real_run_deletes_only_candidates(tmp_data_dir, client):
    """``dry_run=false`` actually removes orphan + stale-failed,
    leaves ``pulled``/``idle``/``loaded``/``loading`` alone."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    cache_root = tmp_data_dir / "hf-cache"
    cache_root.mkdir(exist_ok=True)
    _make_cache_dir(cache_root, "orphan/dead")
    _make_cache_dir(cache_root, "stale/failed")
    _make_cache_dir(cache_root, "pulled/keep")
    _make_cache_dir(cache_root, "loaded/keep")
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m_stale",
        served_name="stale",
        hf_repo="stale/failed",
        status="failed",
        updated_at=_hours_ago_iso(48),
    )
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m_pulled",
        served_name="pulled",
        hf_repo="pulled/keep",
        status="pulled",
    )
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m_loaded",
        served_name="loaded",
        hf_repo="loaded/keep",
        status="loaded",
    )
    auth = _jwt_login(client)
    from tests.conftest import csrf_header

    r = client.post(
        "/api/cache/models/gc?dry_run=false",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dry_run"] is False
    assert len(body["deleted_paths"]) == 2
    assert not (cache_root / "models--orphan--dead").exists()
    assert not (cache_root / "models--stale--failed").exists()
    assert (cache_root / "models--pulled--keep").exists()
    assert (cache_root / "models--loaded--keep").exists()


def test_gc_failed_threshold_zero_includes_fresh_failures(tmp_data_dir, client):
    """``failed_older_than_hours=0`` includes a row whose ``updated_at``
    is "right now" — operator override for "I know this failed, sweep
    it" without waiting 24h."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    cache_root = tmp_data_dir / "hf-cache"
    cache_root.mkdir(exist_ok=True)
    _make_cache_dir(cache_root, "fresh/failed")
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m_fresh",
        served_name="fresh",
        hf_repo="fresh/failed",
        status="failed",
        updated_at=_hours_ago_iso(0.01),
    )
    auth = _jwt_login(client)
    from tests.conftest import csrf_header

    # Default 24h excludes the fresh failure.
    r = client.post(
        "/api/cache/models/gc?dry_run=true",
        headers={**auth, **csrf_header(client)},
    )
    assert r.json()["candidates"] == []
    # Threshold=0 includes it.
    r2 = client.post(
        "/api/cache/models/gc?dry_run=true&failed_older_than_hours=0",
        headers={**auth, **csrf_header(client)},
    )
    repos = [c["repo"] for c in r2.json()["candidates"]]
    assert repos == ["fresh/failed"]


def test_gc_excludes_failed_when_mixed_with_active(tmp_data_dir, client):
    """If a repo backs BOTH a stale-failed row AND any non-failed row,
    GC must skip it — ``failed_stale`` reason requires ALL matching
    rows to be failed."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    cache_root = tmp_data_dir / "hf-cache"
    cache_root.mkdir(exist_ok=True)
    _make_cache_dir(cache_root, "mixed/repo")
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m_failed",
        served_name="mf",
        hf_repo="mixed/repo",
        status="failed",
        updated_at=_hours_ago_iso(72),
    )
    _insert_model(
        tmp_data_dir / "vllm-warden.db",
        model_id="m_pulled",
        served_name="mp",
        hf_repo="mixed/repo",
        status="pulled",
    )
    auth = _jwt_login(client)
    from tests.conftest import csrf_header

    r = client.post(
        "/api/cache/models/gc?dry_run=true",
        headers={**auth, **csrf_header(client)},
    )
    assert r.json()["candidates"] == []
    assert (cache_root / "models--mixed--repo").exists()
