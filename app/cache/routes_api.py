"""HF cache management API.

Three endpoints, all gated by ``require_jwt``:

- ``GET /api/cache/models``                 — list every cached repo with
  size, mtime, and the model rows (if any) that own it.
- ``DELETE /api/cache/models/{repo:path}``  — drop a single repo's cache,
  refusing if any matching model row is active and warning if any are
  ``pulled``/``idle`` (overridable via ``?force=true``).
- ``POST /api/cache/models/gc``             — sweep orphans + stale
  ``failed`` rows in one shot. Dry-run preview by default.

See ``docs/superpowers/specs/2026-05-20-hf-cache-management-design.md``
and vllm-warden#114 for the full design rationale.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from app.auth.deps import require_jwt
from app.cache.scanner import _PREFIX, CachedRepo, scan_hf_cache
from app.db.constants import ACTIVE_STATUSES
from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow

# HF repo id shape: exactly one ``/`` separating org and name, both
# segments restricted to alnum + ``_-.``. The single-slash requirement
# stops ``a/b/c`` from slipping past _safe_dirname's first-only replace
# and creating a two-segment on-disk path. Leading/trailing slashes,
# whitespace, and anything outside the alphabet all reject.
#
# We additionally forbid any ``..`` substring (checked separately at the
# handler since the alphabet itself permits ``.``): a literal ``..``
# token isn't a filesystem traversal once ``_safe_dirname`` encodes ``/``
# to ``--`` (the on-disk dir would be ``models--..--etc``, a sibling not
# a parent), but real HF repo ids never contain ``..`` so refusing them
# loses nothing and removes the cognitive load of having to argue about
# the encoder's invariants every code review.
_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def _is_valid_repo(repo: str) -> bool:
    """Validate ``repo`` matches the HF ``<org>/<name>`` shape AND has no
    ``..`` substring.

    Two checks because we can't express "alphabet includes ``.`` but
    not the ``..`` bigram" in a clean regex without negative lookaheads
    that obscure the intent. Split is auditable.
    """
    if ".." in repo:
        return False
    return _REPO_RE.fullmatch(repo) is not None


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cache")


# ---------------------------------------------------------------------------
# Response shapes
# ---------------------------------------------------------------------------


class MatchedModelRef(BaseModel):
    """Slim projection of a ``models`` row for inline display.

    We surface only the fields the operator needs to decide whether the
    cache entry is safe to delete: row id, served name (the operator's
    label), and status. Everything else is one click away on the model
    detail page.
    """

    id: str
    served_model_name: str
    status: str


class CachedRepoView(BaseModel):
    repo: str
    path: str
    size_bytes: int
    last_modified: float
    # ``matched_models`` is a list (not Optional) because the same
    # ``hf_repo`` can back multiple rows (different GPU mappings,
    # different served_model_names). The safety check in DELETE iterates
    # this — never collapse to a single Optional.
    matched_models: list[MatchedModelRef]


class GcCandidate(BaseModel):
    repo: str
    reason: Literal["orphan", "failed_stale"]
    size_bytes: int
    # ``[]`` for ``orphan`` (no matching rows at all), >=1 for
    # ``failed_stale`` (all rows are failed and old enough).
    matched_rows: list[MatchedModelRef]


class GcResult(BaseModel):
    dry_run: bool
    total_bytes_freed: int
    candidates: list[GcCandidate]
    # Empty when ``dry_run=true``; populated with absolute paths that
    # were actually removed when ``dry_run=false``.
    deleted_paths: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref(row: ModelRow) -> MatchedModelRef:
    return MatchedModelRef(id=row.id, served_model_name=row.served_model_name, status=row.status)


def _index_rows_by_repo(rows: list[ModelRow]) -> dict[str, list[ModelRow]]:
    """Group ``models`` rows by ``hf_repo`` for O(1) join with scanner output.

    Multi-row case (same ``hf_repo`` backing several served_model_names
    with different GPU mappings) is supported by the list value — see
    ``ModelRepo.list_by_repo`` docstring.
    """
    out: dict[str, list[ModelRow]] = {}
    for r in rows:
        out.setdefault(r.hf_repo, []).append(r)
    return out


def _to_view(cached: CachedRepo, rows: list[ModelRow]) -> CachedRepoView:
    return CachedRepoView(
        repo=cached.repo,
        path=str(cached.path),
        size_bytes=cached.size_bytes,
        last_modified=cached.last_modified,
        matched_models=[_ref(r) for r in rows],
    )


def _safe_dirname(repo: str) -> str:
    """Encode ``<org>/<name>`` back to the on-disk ``models--<org>--<name>``.

    Inverse of ``scanner._decode_repo_name``: replace the first "/" with
    "--". Repos without a "/" (legacy or anonymous) get the prefix
    treatment too — still unambiguous because the cache layout is
    "``models--<everything>``".

    Callers MUST first validate ``repo`` against ``_REPO_RE`` (the DELETE
    handler does this on entry); without that, ``a/b/c`` would encode to
    ``models--a--b/c`` and resolve to a TWO-segment on-disk path. Today
    the downstream ``.exists()`` would 404 such a request, but the input
    validation makes the safety property explicit at the boundary.
    """
    return _PREFIX + repo.replace("/", "--", 1)


def _delete_repo_dirs(cache_root: Path, safe: str) -> list[str]:
    """Remove ``cache_root/{safe}`` and ``cache_root/hub/{safe}`` if they exist.

    Returns the list of absolute paths actually removed (empty if no
    matching dir existed, in which case the route returns 404).
    Pure-sync — wrap callers in ``asyncio.to_thread`` so a multi-GiB
    rmtree doesn't block the event loop.
    """
    removed: list[str] = []
    for parent in (cache_root, cache_root / "hub"):
        target = parent / safe
        if not target.exists():
            continue
        try:
            shutil.rmtree(target)
            removed.append(str(target))
        except OSError as exc:
            # Surface the failure as a 500 — partial deletion is
            # observable to the operator and worth a noisy retry.
            logger.exception("delete: rmtree %s failed", target)
            raise HTTPException(500, f"failed to remove {target}: {exc}") from exc
    return removed


def _hours_since(iso_ts: str | None, now: datetime) -> float | None:
    """Convert a SQLite ``datetime('now')`` text timestamp to "hours ago".

    Returns ``None`` for unparseable input — the GC caller treats that
    as "ineligible" so a row with malformed metadata is never
    auto-deleted.
    """
    if not iso_ts:
        return None
    try:
        # SQLite ``datetime('now')`` returns "YYYY-MM-DD HH:MM:SS" in UTC,
        # no timezone suffix. Parse as naive-UTC then attach UTC explicitly.
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
    except ValueError:
        return None
    return (now - dt).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/models", response_model=list[CachedRepoView])
async def list_cached_models(
    request: Request,
    _user: str = Depends(require_jwt),
) -> list[CachedRepoView]:
    settings = request.app.state.settings
    repos = await asyncio.to_thread(scan_hf_cache, settings.hf_cache_dir)
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()
    by_repo = _index_rows_by_repo(rows)
    return [_to_view(r, by_repo.get(r.repo, [])) for r in repos]


@router.delete("/models/{repo:path}", status_code=204)
async def delete_cached_model(
    repo: str,
    request: Request,
    force: bool = False,
    _user: str = Depends(require_jwt),
) -> Response:
    """Drop one repo's HF cache directory.

    Safety ladder (descending in severity):

    1. Any matching row in ``ACTIVE_STATUSES`` (``loaded``/``loading``/
       ``unloading``/``pulling``) → 409, NOT overridable. Deleting the
       cache out from under a running vLLM would crash the worker.
    2. Any matching row in ``("pulled", "idle")`` → 409 unless
       ``?force=true``. These rows are user-owned but not active; force
       is the operator opt-in.
    3. Otherwise (orphan, or only ``failed`` rows) → proceed.

    Returns 404 if no cache dir is found (idempotent for "already gone").

    ``repo`` is validated against ``_REPO_RE`` BEFORE any DB lookup or
    filesystem touch — reject ``../etc/passwd``, ``a/b/c``, empty, and
    double-slash with 400. The on-disk encoder ``_safe_dirname`` only
    replaces the first ``/``, so a multi-segment input WOULD construct
    a two-segment path; today the downstream ``.exists()`` returns False
    and 404s, but the explicit guard fails closed at the boundary.
    """
    if not _is_valid_repo(repo):
        raise HTTPException(
            400,
            f"invalid repo {repo!r}: expected ``<org>/<name>`` with one "
            f"slash, alnum + ``_-.`` only, no ``..`` substring",
        )
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_by_repo(repo)

    in_use = [r for r in rows if r.status in ACTIVE_STATUSES]
    if in_use:
        raise HTTPException(
            409,
            f"refusing: repo {repo!r} backs {len(in_use)} active model(s) "
            f"({', '.join(r.id for r in in_use)})",
        )
    benign_but_alive = [r for r in rows if r.status in ("pulled", "idle")]
    if benign_but_alive and not force:
        raise HTTPException(
            409,
            f"repo {repo!r} backs {len(benign_but_alive)} pulled-but-unloaded "
            f"row(s); pass ?force=true to delete cache anyway "
            f"(the row(s) will move to 'failed' on next load attempt)",
        )

    safe = _safe_dirname(repo)
    deleted_paths = await asyncio.to_thread(_delete_repo_dirs, settings.hf_cache_dir, safe)
    if not deleted_paths:
        raise HTTPException(404, f"no cache dir found for repo {repo!r}")
    return Response(status_code=204)


@router.post("/models/gc", response_model=GcResult)
async def gc_cached_models(
    request: Request,
    dry_run: bool = True,
    failed_older_than_hours: int = 24,
    _user: str = Depends(require_jwt),
) -> GcResult:
    """Sweep clearly-disposable cache: orphans + stale failed rows.

    A repo is a candidate iff EITHER:
      - it has zero matching ``models`` rows (orphan), OR
      - every matching row is ``status=failed`` AND ``updated_at`` is
        more than ``failed_older_than_hours`` hours ago.

    Any other state — even a single ``pulled``/``idle``/active row —
    excludes the repo. The default 24h window is long enough that the
    operator has seen the failure and chosen not to retry; pass
    ``failed_older_than_hours=0`` to include freshly-failed rows in
    the sweep.

    ``dry_run=true`` (default) returns the candidate list with no
    side effects. Pass ``dry_run=false`` after operator confirms the
    preview.
    """
    settings = request.app.state.settings
    repos = await asyncio.to_thread(scan_hf_cache, settings.hf_cache_dir)
    async with open_db(settings.db_path) as db:
        all_rows = await ModelRepo(db).list_all()

    # ``updated_at`` rides on every row now (see ModelRow / _MODEL_COLS in
    # app/db/repos/models.py), so the per-row roundtrip the original
    # implementation made via ``ModelRepo(db).updated_at(row.id)`` is gone —
    # one SELECT covers everything.
    by_repo = _index_rows_by_repo(all_rows)
    now = datetime.now(UTC)
    candidates: list[GcCandidate] = []
    for repo in repos:
        rows = by_repo.get(repo.repo, [])
        if not rows:
            candidates.append(
                GcCandidate(
                    repo=repo.repo,
                    reason="orphan",
                    size_bytes=repo.size_bytes,
                    matched_rows=[],
                )
            )
            continue
        if all(r.status == "failed" for r in rows):
            ages = [_hours_since(r.updated_at, now) for r in rows]
            # ALL rows must be older than threshold; a single fresh
            # failure means the operator may still be triaging it.
            if ages and all(age is not None and age >= failed_older_than_hours for age in ages):
                candidates.append(
                    GcCandidate(
                        repo=repo.repo,
                        reason="failed_stale",
                        size_bytes=repo.size_bytes,
                        matched_rows=[_ref(r) for r in rows],
                    )
                )

    total = sum(c.size_bytes for c in candidates)
    if dry_run:
        return GcResult(
            dry_run=True,
            total_bytes_freed=total,
            candidates=candidates,
            deleted_paths=[],
        )

    deleted_paths: list[str] = []
    for cand in candidates:
        safe = _safe_dirname(cand.repo)
        removed = await asyncio.to_thread(_delete_repo_dirs, settings.hf_cache_dir, safe)
        deleted_paths.extend(removed)
    return GcResult(
        dry_run=False,
        total_bytes_freed=total,
        candidates=candidates,
        deleted_paths=deleted_paths,
    )
