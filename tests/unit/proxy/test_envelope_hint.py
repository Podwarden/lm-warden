"""Tests for the post-overhaul envelope-hint stub.

Slice S1 of the 2026-05 overhaul replaced the bench-coupled enrichment
with a slim ``models.last_error`` attacher. These tests pin the new
contract:

  * non-5xx → no enrichment
  * non-envelope JSON / malformed body → no enrichment
  * 5xx + envelope shape + no last_error → no enrichment
  * 5xx + envelope shape + last_error → attach ``hint`` field
  * pre-existing ``hint`` → attach under ``last_error_hint`` instead
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.proxy.envelope_hint import _is_error_envelope, enrich_5xx_from_db


def _envelope_body() -> bytes:
    return json.dumps(
        {"error": {"message": "out of memory", "type": "internal_server_error"}}
    ).encode("utf-8")


async def _setup_db(tmp_path: Path, *, model_id: str = "m1", last_error: str | None = None) -> Path:
    db_path = tmp_path / "vw.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(
            ModelRow(
                id=model_id,
                served_model_name=model_id,
                hf_repo="org/repo",
                hf_revision="main",
                gpu_indices=[0],
                tensor_parallel_size=1,
                dtype=None,
                max_model_len=None,
                gpu_memory_utilization=0.9,
                trust_remote_code=False,
                extra_args=[],
                status="failed",
                pulled_bytes=0,
                pulled_total=None,
                last_error=last_error,
                extra_env={},
            )
        )
    return db_path


# --------------------------------------------------------------------------- #
# Trigger gates                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_enrichment_for_non_5xx_status(tmp_path: Path) -> None:
    db_path = await _setup_db(tmp_path, last_error="boom")
    out = await enrich_5xx_from_db(
        db_path=db_path,
        model_id="m1",
        status_code=200,
        body_bytes=_envelope_body(),
        asked=None,
    )
    assert out is None


@pytest.mark.asyncio
async def test_no_enrichment_when_body_is_not_json(tmp_path: Path) -> None:
    db_path = await _setup_db(tmp_path, last_error="boom")
    out = await enrich_5xx_from_db(
        db_path=db_path,
        model_id="m1",
        status_code=500,
        body_bytes=b"<html>oops</html>",
        asked=None,
    )
    assert out is None


@pytest.mark.asyncio
async def test_no_enrichment_when_body_lacks_error_envelope(tmp_path: Path) -> None:
    db_path = await _setup_db(tmp_path, last_error="boom")
    out = await enrich_5xx_from_db(
        db_path=db_path,
        model_id="m1",
        status_code=500,
        body_bytes=json.dumps({"detail": "no good"}).encode(),
        asked=None,
    )
    assert out is None


@pytest.mark.asyncio
async def test_no_enrichment_when_last_error_missing(tmp_path: Path) -> None:
    db_path = await _setup_db(tmp_path, last_error=None)
    out = await enrich_5xx_from_db(
        db_path=db_path,
        model_id="m1",
        status_code=500,
        body_bytes=_envelope_body(),
        asked=None,
    )
    assert out is None


@pytest.mark.asyncio
async def test_no_enrichment_when_model_missing(tmp_path: Path) -> None:
    db_path = await _setup_db(tmp_path, last_error="boom")
    out = await enrich_5xx_from_db(
        db_path=db_path,
        model_id="does-not-exist",
        status_code=500,
        body_bytes=_envelope_body(),
        asked=None,
    )
    assert out is None


# --------------------------------------------------------------------------- #
# Happy path                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_enrichment_attaches_last_error_hint(tmp_path: Path) -> None:
    db_path = await _setup_db(tmp_path, last_error="cuda OOM at warmup")
    out = await enrich_5xx_from_db(
        db_path=db_path,
        model_id="m1",
        status_code=502,
        body_bytes=_envelope_body(),
        asked=None,
    )
    assert out is not None
    parsed = json.loads(out)
    err = parsed["error"]
    # Original fields preserved verbatim.
    assert err["message"] == "out of memory"
    assert err["type"] == "internal_server_error"
    # Hint attached.
    assert err["hint"] == {"model_id": "m1", "last_error": "cuda OOM at warmup"}


@pytest.mark.asyncio
async def test_enrichment_uses_last_error_hint_when_hint_field_present(tmp_path: Path) -> None:
    """Pre-existing ``hint`` is preserved; ours goes under ``last_error_hint``."""
    db_path = await _setup_db(tmp_path, last_error="cuda OOM at warmup")
    body = json.dumps({"error": {"message": "boom", "hint": "from upstream"}}).encode()
    out = await enrich_5xx_from_db(
        db_path=db_path,
        model_id="m1",
        status_code=500,
        body_bytes=body,
        asked=None,
    )
    assert out is not None
    err = json.loads(out)["error"]
    assert err["hint"] == "from upstream"
    assert err["last_error_hint"] == {"model_id": "m1", "last_error": "cuda OOM at warmup"}


# --------------------------------------------------------------------------- #
# Helper coverage                                                             #
# --------------------------------------------------------------------------- #


def test_is_error_envelope_accepts_dict_error_field() -> None:
    assert _is_error_envelope({"error": {"message": "x"}})
    assert not _is_error_envelope({"error": "string"})
    assert not _is_error_envelope({"detail": "x"})
    assert not _is_error_envelope("plain string")
    assert not _is_error_envelope(None)
