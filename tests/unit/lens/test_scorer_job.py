"""The quiet-session scorer (analysis Task 2.3): written once when quiet,
rewritten in place when it resumes, never while active, canary-clean (A-10),
pruned with request_history."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import sqlite3
from dataclasses import replace
from typing import Any

from app.lens.scorer_job import ScorerJob
from app.lens.store import LensStore
from app.stats import request_history
from tests.unit.lens import seed
from tests.unit.lens.test_canary import CANARY, run_session_with_canary
from tests.unit.lens.test_routes_ui import A, Env, env, put_rows, session_rows  # noqa: F401
from tests.unit.proxy.test_lens_hot_path import _rows
from tests.unit.proxy.test_prompt_lens_forward import _ready_gm

LAST = seed.T0 + 30 * 23 + 2  # the last of 24 seeded turns finishes here
QUIET = 600.0
DONE = LAST + 10 + QUIET  # B's session starts 7 s later: both are quiet


def _tables(db_path) -> tuple[list[Any], list[Any]]:
    with sqlite3.connect(db_path) as c:
        return (
            c.execute("SELECT * FROM lens_session_scores").fetchall(),
            c.execute("SELECT * FROM lens_findings").fetchall(),
        )


def _job(env: Env) -> ScorerJob:  # noqa: F811
    st = env.client.app.state
    st.settings = dataclasses.replace(st.settings, lens_quiet_s=QUIET)
    return ScorerJob(st)


def test_a_session_quiet_for_600s_is_written_once(env: Env) -> None:  # noqa: F811
    job = _job(env)
    assert job.run_once_sync(DONE) == 2  # keys A and B
    scores, findings = _tables(env.db)
    assert len(scores) == 2 and findings
    assert job.run_once_sync(DONE + 500) == 0  # nothing moved
    assert len(_tables(env.db)[0]) == 2
    got = LensStore(sqlite3.connect(env.db)).sessions(A, since=0)
    assert len(got) == 1 and got[0].turns == 24 and got[0].harness == "other"
    assert got[0].sess_ref.startswith("s_")


def test_an_active_session_is_not_written(env: Env) -> None:  # noqa: F811
    assert _job(env).run_once_sync(LAST + QUIET - 5) == 0
    assert _tables(env.db) == ([], [])


def test_a_resumed_session_is_updated_in_place(env: Env) -> None:  # Review Focus 5  # noqa: F811
    job = _job(env)
    assert job.run_once_sync(DONE) == 2
    before = LensStore(sqlite3.connect(env.db)).sessions(A, since=0)[0]
    # Three more turns of A's session, an hour later.
    more = session_rows(A, "reqa", turns=27)[24:]
    late = [
        replace(r, started_at=r.started_at + 3600, finished_at=r.finished_at + 3600) for r in more
    ]
    put_rows(env.db, late, secret=env.secret)
    assert job.run_once_sync(DONE + 3700) == 1
    after = LensStore(sqlite3.connect(env.db)).sessions(A, since=0)
    assert len(after) == 1  # one row, never two
    assert after[0].sess_ref == before.sess_ref
    assert after[0].turns == 27 and after[0].last_at > before.last_at


def test_the_scores_and_findings_come_from_the_keys_own_rows(env: Env) -> None:  # noqa: F811
    _job(env).run_once_sync(DONE)
    s = LensStore(sqlite3.connect(env.db))
    row = s.sessions(A, since=0)[0]
    assert row.e is not None and row.e < 100  # the E-01 line changes every turn
    rules = {f.rule for f in s.findings(A, row.sess_ref)}
    assert {"E-01", "R-04"} <= rules
    assert row.rules["E"] == ["E-01"] or "E-01" in row.rules["E"]


def test_a_failing_session_does_not_stop_the_pass(env: Env, monkeypatch, caplog) -> None:  # noqa: F811
    from app.lens import service

    real = service.score_rows

    def boom(db, state, rows, **kw):
        if rows and rows[0].token_id == A:
            raise RuntimeError("CANARY-in-an-exception-message")
        return real(db, state, rows, **kw)

    monkeypatch.setattr(service, "score_rows", boom)
    caplog.set_level(logging.DEBUG)
    assert _job(env).run_once_sync(DONE) == 1
    assert "CANARY-in-an-exception-message" not in caplog.text  # the type only


def test_retention_prune_removes_scores_with_their_requests(env: Env) -> None:  # noqa: F811
    _job(env).run_once_sync(DONE)
    assert len(_tables(env.db)[0]) == 2

    async def prune() -> None:
        import aiosqlite

        async with aiosqlite.connect(env.db) as db:
            await request_history.prune(db, cutoff=LAST + 10, max_rows=0)
            await db.commit()

    asyncio.run(prune())
    assert _tables(env.db) == ([], [])


def test_canary_never_in_scores_findings_or_logs(
    tmp_data_dir: Any, client: Any, caplog: Any
) -> None:  # A-10
    import time

    caplog.set_level(logging.DEBUG)
    plaintext = _ready_gm(client, tmp_data_dir)
    run_session_with_canary(client, plaintext)
    _rows(tmp_data_dir, 2)  # the history writer is asynchronous
    st = client.app.state
    st.settings = dataclasses.replace(st.settings, lens_quiet_s=1.0)
    assert ScorerJob(st).run_once_sync(time.time() + 3600) >= 1
    scores, findings = _tables(tmp_data_dir / "vllm-warden.db")
    assert scores and CANARY not in json.dumps([scores, findings], default=str)
    assert CANARY not in caplog.text
