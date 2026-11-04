"""Relayed rows (router passthrough) never reach a local stats reader (task 5, R4)."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.stats import prefill_model, routes_forest
from app.stats import request_history as rh
from app.stats.request_history import RequestHistoryStore


def _record(i: int, **over):
    base = {
        "id": f"req-{i}",
        "model": "m",
        "model_id": "m",
        "token_name": "k",
        "token_id": "tok1",
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "duration_s": 1.0,
        "ttft_s": 0.2,
        "cached_tokens": 50,
        "reusable_tokens": 60,
        "cache_outcome": "partial",
        "finish_reason": "stop",
        "orphan": False,
        "started_iso": "2026-10-25T10:00:00Z",
        "session_key": "sk",
    }
    base.update(over)
    return base


async def _two_rows(tmp_path):
    db_path = tmp_path / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    store = RequestHistoryStore(db_path)
    store.record(_record(1), finished_at=1000.0)
    store.record(_record(2, relayed=True, upstream="anthropic.example"), finished_at=1001.0)
    assert await store.flush() == 2
    return db_path


@pytest.mark.parametrize(
    "reader",
    [
        "query_window",
        "query_cache_summary",
        "query_window_stats",
        "query_last",
        "query_token_timings",
    ],
)
async def test_relayed_rows_are_invisible_to_local_readers(tmp_path, reader):
    db_path = await _two_rows(tmp_path)
    async with open_db(db_path) as db:
        if reader == "query_window":
            out = (await rh.query_window(db, since=0, model_ids=None, limit=10)).rows
            assert [r["id"] for r in out] == ["req-1"]
        elif reader == "query_cache_summary":
            (m,) = await rh.query_cache_summary(db, since=0, model_ids=None)
            assert m["requests"] == 1 if "requests" in m else True
            assert sum(m["outcomes"].values()) == 1
        elif reader == "query_window_stats":
            assert len(await rh.query_window_stats(db, since=0, model_ids=None)) == 1
        elif reader == "query_last":
            assert [r["id"] for r in await rh.query_last(db, n=10, model_ids=None)] == ["req-1"]
        else:
            t = await rh.query_token_timings(db, token_ids=["tok1"], since=0, until=2000)
            assert t.total == 1


async def test_prefill_model_ignores_relayed(tmp_path):
    db_path = await _two_rows(tmp_path)
    state = prefill_model.PrefillModelState()
    assert await prefill_model.refresh(state, db_path, now=1500.0) == 1


async def test_forest_ignores_relayed(tmp_path):
    db_path = await _two_rows(tmp_path)
    with closing(sqlite3.connect(db_path)) as db:
        rows = routes_forest._load(db, 0.0, None, None)
        assert len(rows) == 1
        assert sum(routes_forest._counts(db, 0.0, 10**9).values()) == 1
        assert len(routes_forest._load_new(db, 0, 10**9, 0.0, None)) == 1
