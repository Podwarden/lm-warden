"""Bounded, indexed session queries (sessions pages, task 6)."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.sessions import store
from app.sessions.store import (
    descendants,
    list_rows,
    lookup_ref,
    oldest_finished_at,
    request_row,
    root_of,
    rows_for,
    tool_names,
    toolsets,
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "vllm-warden.db"

    async def mig():
        async with open_db(path) as d:
            await apply_migrations(d)

    asyncio.run(mig())
    conn = sqlite3.connect(path)
    yield conn
    conn.close()


def _insert(db, rid, *, tok="tok1", sess="S", parent=None, fin=1000.0, dur=2.0, **cols):
    base = {
        "id": rid,
        "finished_at": fin,
        "model_id": "m",
        "model": "m",
        "token_id": tok,
        "token_name": "k-" + (tok or "-"),
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "duration_s": dur,
        "started_iso": "2026-10-25T10:00:00Z",
        "orphan": 0,
        "cobs2_session": sess,
        "session_key": sess,
        "parent_session_key": parent,
    }
    base.update(cols)
    keys = ", ".join(base)
    db.execute(
        f"INSERT INTO request_history({keys}) VALUES ({', '.join('?' for _ in base)})",
        tuple(base.values()),
    )
    db.commit()


def test_lookup_ref(db):
    _insert(db, "r1", cobs2_sess_ref="a" * 32)
    assert lookup_ref(db, "a" * 32) == ("tok1", "S")
    assert lookup_ref(db, "b" * 32) is None


def test_root_of_walks_up_and_is_cycle_safe(db):
    _insert(db, "a1", sess="a")
    _insert(db, "b1", sess="b", parent="a")
    _insert(db, "c1", sess="c", parent="b")
    assert root_of(db, "tok1", "c") == "a"
    _insert(db, "x1", sess="x", parent="y")
    _insert(db, "y1", sess="y", parent="x")
    assert root_of(db, "tok1", "x") in {"x", "y"}
    for i in range(12):
        _insert(db, f"d{i}", sess=f"d{i}", parent=f"d{i + 1}" if i < 11 else None)
    assert root_of(db, "tok1", "d0") == "d8"


def test_root_stops_at_topmost_ancestor_with_rows(db):
    _insert(db, "b1", sess="b", parent="ghost")
    _insert(db, "c1", sess="c", parent="b")
    assert root_of(db, "tok1", "c") == "b"


def test_root_is_scoped_by_token(db):
    _insert(db, "a1", sess="a", tok="tok2")
    _insert(db, "b1", sess="b", parent="a")
    assert root_of(db, "tok1", "b") == "b"


def test_descendants_finds_children_and_grandchildren(db):
    _insert(db, "a1", sess="a")
    _insert(db, "b1", sess="b", parent="a")
    _insert(db, "c1", sess="c", parent="b")
    _insert(db, "z1", sess="z", parent="a", tok="tok2")
    assert descendants(db, "tok1", "a") == ["a", "b", "c"]


def test_rows_for_is_scoped_by_token_and_bounded(db):
    for i in (1, 2, 3):
        _insert(db, f"r{i}", fin=1000.0 + i)
    _insert(db, "o1", tok="tok2", fin=1010.0)
    rows, trunc = rows_for(db, "tok1", ["S"], limit=2)
    assert [r.id for r in rows] == ["r2", "r3"] and trunc
    assert all(r.token_id == "tok1" for r in rows)
    assert rows[0].started_at == pytest.approx(1002.0 - 2.0)
    rows, trunc = rows_for(db, "tok1", ["S"], limit=10)
    assert len(rows) == 3 and not trunc


def test_rows_decode_shape_and_tools_out_defensively(db):
    _insert(
        db,
        "ok",
        cobs2_shape=json.dumps({"chars": 5}),
        tools_out=json.dumps([["agent", "ab12cd34"]]),
        relayed=1,
    )
    _insert(db, "bad", cobs2_shape="{nope", tools_out="[1,2", fin=1001.0)
    rows, _ = rows_for(db, "tok1", ["S"], limit=10)
    by = {r.id: r for r in rows}
    assert by["ok"].shape == {"chars": 5} and by["ok"].tools_out == (("agent", "ab12cd34"),)
    assert by["ok"].relayed is True
    assert by["bad"].shape is None and by["bad"].tools_out == () and by["bad"].relayed is False
    assert request_row(db, "ok").id == "ok"
    assert request_row(db, "nope") is None


def test_queries_use_the_new_indexes(db):
    plan = " ".join(
        str(r) for r in db.execute("EXPLAIN QUERY PLAN " + store._ROWS_SQL_FOR(1), ("tok1", "S", 5))
    )
    assert "idx_request_history_cobs2_session" in plan


def test_list_rows_skips_sessionless_and_respects_since(db):
    _insert(db, "old", fin=500.0)
    _insert(db, "new", fin=1500.0, tools_out=json.dumps([["read", "x"]]))
    _insert(db, "none", sess=None, fin=1600.0)
    _insert(db, "other", tok="tok2", fin=1700.0)
    rows, trunc = list_rows(db, since=1000.0, token_id="tok1", limit=10)
    assert [r.id for r in rows] == ["new"] and not trunc and rows[0].has_tools
    rows, _ = list_rows(db, since=1000.0, token_id=None, limit=10)
    assert [r.id for r in rows] == ["other", "new"]
    rows, trunc = list_rows(db, since=0.0, token_id=None, limit=1)
    assert len(rows) == 1 and trunc
    assert oldest_finished_at(db) == 500.0


def test_toolsets_and_tool_names_are_scoped_to_the_token(db):
    db.execute(
        "INSERT INTO cobs2_toolsets VALUES (?,?,?)",
        ("d" * 32, json.dumps([["h1", "agent", 10]]), 1.0),
    )
    db.executemany(
        "INSERT INTO session_tools VALUES (?,?,?,?,?,?,?)",
        [
            ("tok1", "h1", "Task", "agent", 10, 1.0, 1.0),
            ("tok2", "h1", "Other", "agent", 10, 1.0, 1.0),
        ],
    )
    db.commit()
    assert toolsets(db, ["d" * 32, "e" * 32]) == {"d" * 32: [("h1", "agent", 10)]}
    assert tool_names(db, "tok1", ["h1", "h2"]) == {"h1": "Task"}


def _plan(db, sql, args):
    return " ".join(str(r) for r in db.execute("EXPLAIN QUERY PLAN " + sql, args))


def test_descendants_query_uses_the_parent_index_without_analyze(db):
    """Final review I4: production never runs ANALYZE; the child-session lookup must
    still use idx_request_history_parent_session, not scan the key's session rows."""
    assert "sqlite_stat1" not in {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
    plan = _plan(db, store._DESC_SQL, ("tok1", "S"))
    assert "idx_request_history_parent_session" in plan


def test_rows_for_many_sessions_merges_newest_first_and_bounds(db):
    for i in range(3):
        _insert(db, f"a{i}", sess="A", fin=1000.0 + 2 * i)
        _insert(db, f"b{i}", sess="B", fin=1001.0 + 2 * i)
    rows, trunc = rows_for(db, "tok1", ["A", "B"], limit=4)
    assert [r.id for r in rows] == ["a1", "b1", "a2", "b2"]
    assert trunc


# -- sessions redesign, task 2 ------------------------------------------------

from app.sessions.store import ModelEngine, llamacpp_slots, model_engines  # noqa: E402


def test_rows_carry_source_own_lens_and_ttft_estimate(db):
    _insert(
        db,
        "r1",
        session_source="claude_code_header",
        reusable_tokens_own=900,
        cached_ttft_est_tokens=700,
    )
    (r,), _ = rows_for(db, "tok1", ["S"], limit=10)
    assert (r.session_source, r.reusable_tokens_own, r.cached_ttft_est_tokens) == (
        "claude_code_header",
        900,
        700,
    )


def test_list_rows_read_dw_and_chars_in_sql(db):
    _insert(
        db,
        "r1",
        cobs2_shape='{"chars":4000,"dw":{"dup":10,"boil":20,"reason":30}}',
        tools_out='[["read","ab12cd34"]]',
    )
    _insert(db, "r2", cobs2_shape='{"chars":4000}', fin=1001.0)
    _insert(db, "r3", cobs2_shape="not json", fin=1002.0)
    rows = {r.id: r for r in list_rows(db, since=0, token_id=None)[0]}
    assert (
        rows["r1"].dw == (10, 20, 30)
        and rows["r1"].chars == 4000
        and rows["r1"].called == ("ab12cd34",)
    )
    assert rows["r2"].dw is None and rows["r3"].chars is None and rows["r2"].called == ()


def _model(db, mid, backend, dp=1, aff=1, mns=None, extra="[]"):
    db.execute(
        "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, tensor_parallel_size, dtype,"
        " max_model_len, gpu_memory_utilization, trust_remote_code, extra_args, status, backend,"
        " data_parallel_size, dp_affinity_enabled, max_num_seqs)"
        " VALUES (?, ?, 'o/r', 'main', '[0]', 1, NULL, NULL, 0.9, 0, ?, 'loaded', ?, ?, ?, ?)",
        (mid, mid, extra, backend, dp, aff, mns),
    )
    db.commit()


def test_model_engines_reads_dp_affinity_backend_and_slots(db):
    _model(db, "qwen", "vllm", dp=3, aff=1)
    _model(db, "ll", "llamacpp", mns=8)
    _model(db, "ll2", "llamacpp", extra='["-np", "6"]')
    _model(db, "mx", "mlx")
    got = model_engines(db, ["qwen", "ll", "ll2", "mx", "gone"])
    assert got["qwen"] == ModelEngine("vllm", 3, True, 4, True)
    assert got["ll"].slots == 8 and got["ll2"].slots == 6
    assert got["mx"].cache_known is False
    assert "gone" not in got


@pytest.mark.parametrize(
    "backend,mns,extra,s",
    [
        ("llamacpp", 8, [], 8),
        ("llamacpp", None, ["--parallel=2"], 2),
        ("llamacpp", None, ["--parallel", "3"], 3),
        ("llamacpp", None, [], 4),
        ("vllm", 256, [], 4),
    ],
)
def test_llamacpp_slots(backend, mns, extra, s):
    assert llamacpp_slots(backend, mns, extra) == s
