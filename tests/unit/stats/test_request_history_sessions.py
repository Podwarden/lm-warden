"""request_history persists the sessions-pages fields (migration 0053): session,
conversation, shape, parent request, toolsets and tool names, relay flag."""

from __future__ import annotations

import asyncio
import json
import sqlite3

from app.cobs2.shape import Toolset
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.proxy.request_registry import LiveRequest, finished_record
from app.stats import request_history as rh
from app.stats.request_history import TOOL_NAME_REFRESH_S, RequestHistoryStore, prune

TS = Toolset(
    "d" * 32, (("ab12cd34", "read", 10), ("ef56ab78", "agent", 20)), ("Read", "Task"), False
)


def _record(i: int, **over):
    base = {
        "id": f"req-{i}",
        "model": "model-a",
        "model_id": "id-model-a",
        "token_name": "key-a",
        "token_id": "tok1",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "duration_s": 1.0,
        "finish_reason": "stop",
        "orphan": False,
        "started_iso": "2026-10-25T10:00:00Z",
    }
    base.update(over)
    return base


def _ts_fields(ts: Toolset = TS) -> dict:
    return {
        "cobs2_toolset": ts.digest,
        "cobs2_toolset_tools": [list(e) for e in ts.entries],
        "cobs2_toolset_names": list(ts.names),
    }


async def _migrated(tmp_path):
    db_path = tmp_path / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    return db_path


def _q(db_path, sql, args=()):
    with sqlite3.connect(db_path) as db:
        return db.execute(sql, args).fetchall()


def test_read_columns_are_unchanged() -> None:
    assert rh._READ_COLUMNS[-1] == "thinking_forced_off"
    assert "cobs2_session" not in rh._READ_COLUMNS
    assert rh._COLUMNS[-len(rh._LENS_COLUMNS) :] == rh._LENS_COLUMNS
    tail = rh._COLUMNS[: -len(rh._LENS_COLUMNS) - len(rh._KV_ISO_COLUMNS)]
    assert tail[-len(rh._DIET_COLUMNS) :] == rh._DIET_COLUMNS
    tail = tail[: -len(rh._DIET_COLUMNS)]
    assert tail[-len(rh._SESSIONS_COLUMNS) :] == rh._SESSIONS_COLUMNS
    assert rh._COLUMNS[: -len(rh._LENS_COLUMNS)][-len(rh._KV_ISO_COLUMNS) :] == rh._KV_ISO_COLUMNS


async def test_new_fields_round_trip_and_toolset_upserts(tmp_path) -> None:
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    shape = '{"chars":10,"brk":null}'
    store.record(
        _record(
            1,
            cobs2_session="sess",
            cobs2_conv="01" + "a" * 32,
            cobs2_shape=shape,
            cobs2_parent_req="req-0",
            cobs2_sess_ref="f" * 32,
            relayed=True,
            upstream="anthropic",
            **_ts_fields(),
        ),
        finished_at=1000.0,
    )
    store.record(_record(2, **_ts_fields()), finished_at=1005.0)
    assert await store.flush() == 2
    row = _q(
        db_path,
        "SELECT cobs2_session, cobs2_conv, cobs2_shape, cobs2_parent_req, cobs2_sess_ref,"
        " cobs2_toolset, relayed, upstream, served_gpu_tokens, served_ram_tokens,"
        " served_nvme_tokens, served_remote_tokens, load_s, node_id"
        " FROM request_history WHERE id='req-1'",
    )[0]
    assert row == (
        "sess", "01" + "a" * 32, shape, "req-0", "f" * 32, "d" * 32, 1, "anthropic",
        None, None, None, None, None, None,
    )  # fmt: skip
    assert _q(db_path, "SELECT relayed FROM request_history WHERE id='req-2'") == [(None,)]
    ts = _q(db_path, "SELECT digest, tools, last_seen FROM cobs2_toolsets")
    assert len(ts) == 1 and ts[0][0] == "d" * 32 and ts[0][2] == 1005.0
    assert json.loads(ts[0][1]) == [["ab12cd34", "read", 10], ["ef56ab78", "agent", 20]]
    assert "Read" not in ts[0][1]


async def test_bad_values_cost_one_field_not_the_row(tmp_path) -> None:
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1, cobs2_conv=123, cobs2_shape=[1], load_s="x"), finished_at=1000.0)
    assert await store.flush() == 1
    assert _q(db_path, "SELECT cobs2_conv, cobs2_shape, load_s FROM request_history") == [
        (None, None, None)
    ]


async def test_wait_for_returns_true_after_commit_and_false_on_timeout(tmp_path) -> None:
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    assert await store.wait_for("req-1", 0.05) is False
    store.record(_record(1), finished_at=1000.0)

    async def later() -> None:
        await asyncio.sleep(0.05)
        await store.flush()

    task = asyncio.create_task(later())
    assert await store.wait_for("req-1", 2.0) is True
    await task
    assert await store.wait_for("req-1", 0.0) is True  # already committed


async def test_prune_drops_stale_toolsets_and_tool_names(tmp_path) -> None:
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1, **_ts_fields()), finished_at=1000.0)
    assert await store.flush() == 1
    cutoff = 1000.0 + TOOL_NAME_REFRESH_S + 1
    async with open_db(db_path) as db:
        await prune(db, cutoff=cutoff, max_rows=0)
        await db.commit()
    assert _q(db_path, "SELECT COUNT(*) FROM cobs2_toolsets") == [(0,)]
    assert _q(db_path, "SELECT COUNT(*) FROM session_tools") == [(0,)]


async def test_prune_keeps_tools_within_the_refresh_slack(tmp_path) -> None:
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1, **_ts_fields()), finished_at=1000.0)
    assert await store.flush() == 1
    async with open_db(db_path) as db:
        await prune(db, cutoff=1000.0 + 10, max_rows=0)
        await db.commit()
    assert _q(db_path, "SELECT COUNT(*) FROM session_tools") == [(2,)]


async def test_session_tools_upserts_only_new_changed_or_stale(tmp_path, monkeypatch) -> None:
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    writes: list[int] = []
    real = store._tool_upserts_due

    def spy(*a, **kw):
        out = real(*a, **kw)
        writes.append(len(out))
        return out

    monkeypatch.setattr(store, "_tool_upserts_due", spy)
    for i in range(3):
        store.record(_record(i, **_ts_fields()), finished_at=1000.0 + i)
    assert writes == [2, 0, 0]
    bigger = Toolset("e" * 32, (("ab12cd34", "read", 99),), ("Read",), False)
    store.record(_record(10, **_ts_fields(bigger)), finished_at=1010.0)
    assert writes[-1] == 1
    store.record(_record(11, **_ts_fields()), finished_at=1010.0 + TOOL_NAME_REFRESH_S + 1)
    assert writes[-1] == 2  # stale refresh (and the size change back)
    store.record(_record(12, token_id="tok2", **_ts_fields()), finished_at=1020.0)
    assert writes[-1] == 2
    assert await store.flush() == 6
    rows = _q(
        db_path, "SELECT token_id, tool_hmac, name, family, size FROM session_tools ORDER BY 1, 2"
    )
    assert rows == [
        ("tok1", "ab12cd34", "Read", "read", 10),
        ("tok1", "ef56ab78", "Task", "agent", 20),
        ("tok2", "ab12cd34", "Read", "read", 10),
        ("tok2", "ef56ab78", "Task", "agent", 20),
    ]
    last = _q(
        db_path,
        "SELECT last_seen FROM session_tools WHERE token_id='tok1' AND tool_hmac='ab12cd34'",
    )
    assert last == [(1010.0 + TOOL_NAME_REFRESH_S + 1,)]


async def test_session_tools_never_written_for_rows_without_a_token(tmp_path) -> None:
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1, token_id=None, **_ts_fields()), finished_at=1000.0)
    assert await store.flush() == 1
    assert _q(db_path, "SELECT COUNT(*) FROM session_tools") == [(0,)]
    assert _q(db_path, "SELECT COUNT(*) FROM cobs2_toolsets") == [(1,)]


def test_finished_record_carries_session_fields() -> None:
    req = LiveRequest(
        id="r1",
        token_id="tok1",
        token_name="k",
        client_ip="1.2.3.4",
        model="model-a",
        model_row_id="m",
        path="/v1/chat/completions",
        prompt_tokens=1,
        max_model_len=None,
        started_monotonic=0.0,
        started_iso="2026-10-25T10:00:00Z",
        cobs2_session="sess",
        cobs2_conv="01" + "a" * 32,
        cobs2_shape='{"chars":1}',
        cobs2_parent_req="r0",
        cobs2_sess_ref="f" * 32,
        cobs2_toolset=Toolset("d" * 32, (("ab12cd34", "read", 10),), ("Read",), False),
        relayed=True,
        upstream="anthropic",
    )
    rec = finished_record(req, now=1.0)
    assert rec["cobs2_toolset"] == "d" * 32
    assert rec["cobs2_toolset_tools"] == [["ab12cd34", "read", 10]]
    assert rec["cobs2_toolset_names"] == ["Read"]
    assert rec["relayed"] is True and rec["upstream"] == "anthropic"
    assert (rec["cobs2_session"], rec["cobs2_parent_req"], rec["cobs2_sess_ref"]) == (
        "sess",
        "r0",
        "f" * 32,
    )
