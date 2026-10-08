"""Persistence for stress runs (design §7.2).

The load-bearing behaviour is `current_for`, which is the ONLY path by which a
measured number reaches a client. Every condition it enforces exists because
publishing without it would state something untrue, so each is tested
separately rather than as one happy path.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from app.db.repos.stress_runs import StressRunRepo, StressRunRow

_SQL_DIR = Path(__file__).resolve().parents[3] / "app" / "db" / "sql"
_DDL = _SQL_DIR / "0029_stress_runs.sql"
#: EVERY stress migration, in order. A fixture pinned to 0029 alone broke 34
#: tests the moment 0030 added a column, because the repo's SELECT list moved
#: with the schema while the test schema did not. Globbing means the next
#: migration is picked up without touching a fixture.
_STRESS_DDL = sorted(_SQL_DIR.glob("00*_stress*.sql"))


def _apply_stress_schema(execute) -> None:
    for f in _STRESS_DDL:
        execute(f.read_text())


@pytest.fixture
async def db():
    """An in-memory DB with the migration applied, CLOSED afterwards.

    aiosqlite runs a thread per connection, so a test that opens one and walks
    away leaks a thread for the life of the process. Every other suite in this
    repo uses `async with aiosqlite.connect(...)` for exactly that reason.
    """
    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE models (id TEXT PRIMARY KEY)")
        await conn.execute("INSERT INTO models (id) VALUES ('m1')")
        for _f in _STRESS_DDL:
            await conn.executescript(_f.read_text())
        await conn.commit()
        yield conn


def _row(**over) -> StressRunRow:
    base = dict(
        id="r1",
        model_id="m1",
        fingerprint="sha256:aaa",
        mode="conservative",
        seed=7,
        probe_suite_hash="suite-v1",
        algorithm_version=1,
    )
    base.update(over)
    return StressRunRow(**base)


# ---- schema --------------------------------------------------------------


def test_the_migration_applies_cleanly():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE models (id TEXT PRIMARY KEY)")
    _apply_stress_schema(con.executescript)
    cols = {r[1] for r in con.execute("PRAGMA table_info(stress_runs)")}
    assert {
        "fingerprint",
        "limits",
        "non_monotone",
        "traffic_observed",
        "loads_since_measurement",
        "recommended_config",
        "progress",
    } <= cols


def test_the_migration_has_no_transaction_hostile_statement():
    """The runner refuses PRAGMA/VACUUM/BEGIN inside a migration.

    Each file runs in one transaction so a half-applied migration is
    impossible; a statement that ends the transaction would break that.
    """
    hostile = {"pragma", "vacuum", "begin", "commit", "end", "rollback", "attach", "detach"}
    for stmt in "\n".join(f.read_text() for f in _STRESS_DDL).split(";"):
        body = "\n".join(ln for ln in stmt.splitlines() if not ln.strip().startswith("--")).strip()
        if body:
            assert body.split()[0].lower() not in hostile


def test_deleting_a_model_cascades_to_its_runs():
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("CREATE TABLE models (id TEXT PRIMARY KEY)")
    con.executescript(_DDL.read_text())
    con.execute("INSERT INTO models (id) VALUES ('m1')")
    con.execute(
        "INSERT INTO stress_runs (id, model_id, fingerprint, mode, seed, "
        "probe_suite_hash, algorithm_version) VALUES ('r1','m1','fp','quick',1,'s',1)"
    )
    con.execute("DELETE FROM models WHERE id = 'm1'")
    assert con.execute("SELECT count(*) FROM stress_runs").fetchone()[0] == 0


# ---- round trip ----------------------------------------------------------


async def test_insert_and_get_round_trip(db):
    repo = StressRunRepo(db)
    await repo.insert(_row())
    got = await repo.get("r1")
    assert got is not None
    assert got.status == "running"
    assert got.limits is None


async def test_finish_stores_the_json_payloads(db):
    repo = StressRunRepo(db)
    await repo.insert(_row())
    await repo.finish(
        "r1",
        status="completed",
        limits={"quality_context": {"value": 122880}},
        observations=[{"cls": "pass"}],
    )
    got = await repo.get("r1")
    assert got.limits["quality_context"]["value"] == 122880
    assert got.observations == [{"cls": "pass"}]
    assert got.finished_at is not None


# ---- current_for: the only path to a client ------------------------------


async def test_current_for_returns_a_completed_run(db):
    repo = StressRunRepo(db)
    await repo.insert(_row())
    await repo.finish("r1", status="completed", limits={"a": 1})
    assert (await repo.current_for("m1", "sha256:aaa")) is not None


async def test_a_different_fingerprint_gets_nothing(db):
    """The measurement is not about this model as it currently stands."""
    repo = StressRunRepo(db)
    await repo.insert(_row())
    await repo.finish("r1", status="completed", limits={"a": 1})
    assert (await repo.current_for("m1", "sha256:different")) is None


async def test_an_interrupted_run_is_never_published(db):
    """It holds a bracket that was never confirmed.

    Publishing its lower bound would understate the limit while carrying the
    full confidence of a completed measurement.
    """
    repo = StressRunRepo(db)
    await repo.insert(_row())
    await repo.finish("r1", status="interrupted")
    assert (await repo.current_for("m1", "sha256:aaa")) is None


async def test_a_non_monotone_run_is_never_published(db):
    repo = StressRunRepo(db)
    await repo.insert(_row())
    await repo.finish("r1", status="completed", limits={"a": 1}, non_monotone=True)
    assert (await repo.current_for("m1", "sha256:aaa")) is None


async def test_a_traffic_tainted_run_is_never_published(db):
    """KV is a shared pool; foreign traffic moved the measurement."""
    repo = StressRunRepo(db)
    await repo.insert(_row())
    await repo.finish("r1", status="completed", limits={"a": 1}, traffic_observed=True)
    assert (await repo.current_for("m1", "sha256:aaa")) is None


async def test_the_newest_matching_run_wins(db):
    repo = StressRunRepo(db)
    await repo.insert(_row(id="old"))
    await repo.finish("old", status="completed", limits={"v": 1})
    await db.execute("UPDATE stress_runs SET started_at = '2020-01-01' WHERE id='old'")
    await repo.insert(_row(id="new"))
    await repo.finish("new", status="completed", limits={"v": 2})
    got = await repo.current_for("m1", "sha256:aaa")
    assert got.limits["v"] == 2


# ---- boot reconciliation -------------------------------------------------


async def test_a_running_row_is_marked_interrupted_at_boot(db):
    """The run is in-process, so a warden restart kills it mid-search."""
    repo = StressRunRepo(db)
    await repo.insert(_row())
    assert await repo.mark_interrupted_on_startup() == 1
    got = await repo.get("r1")
    assert got.status == "interrupted"
    assert "restarted" in (got.last_error or "")


async def test_boot_reconciliation_leaves_finished_runs_alone(db):
    repo = StressRunRepo(db)
    await repo.insert(_row())
    await repo.finish("r1", status="completed", limits={"a": 1})
    assert await repo.mark_interrupted_on_startup() == 0


async def test_reloads_are_counted_against_a_completed_measurement(db):
    """An identical-config reload leaves the fingerprint unchanged.

    Whether a measurement should survive that is genuinely unknown, so the
    count is recorded to make it answerable from data later.
    """
    repo = StressRunRepo(db)
    await repo.insert(_row())
    await repo.finish("r1", status="completed", limits={"a": 1})
    await repo.bump_loads_since_measurement("m1")
    await repo.bump_loads_since_measurement("m1")
    assert (await repo.get("r1")).loads_since_measurement == 2


async def test_a_valueless_newest_run_does_not_hide_an_older_measurement(db):
    """Found in review, 2026-09-04: a regression in the cooldown fix.

    `current_for` fetched the newest row and THEN asked whether it measured
    anything, so a newer valueless run hid a valid older one. The caller
    (`published_record`) then found the old run through its history fallback
    and served it as `provenance: stale` with "the model, engine, hardware or
    co-resident set has changed" -- when nothing had. Withdrawing a live
    measurement and giving a false reason is worse than the bug the condition
    was added to fix.
    """
    repo = StressRunRepo(db)
    await repo.insert(_row(id="old"))
    await repo.finish("old", status="completed", limits={"quality_context": 32768})
    await repo.insert(_row(id="new"))
    await repo.finish("new", status="completed", limits={"quality_context": {"raw_confirmed": 0}})
    # started_at defaults to datetime('now') at SECOND resolution, so both rows
    # would otherwise tie and the ordering this test depends on would be
    # undefined -- the test would pass or fail by luck.
    await db.execute(
        "UPDATE stress_runs SET started_at = ? WHERE id = ?", ("2026-09-04 01:00:00", "old")
    )
    await db.execute(
        "UPDATE stress_runs SET started_at = ? WHERE id = ?", ("2026-09-04 02:00:00", "new")
    )
    await db.commit()

    got = await repo.current_for("m1", "sha256:aaa")
    assert got is not None, "the older real measurement was hidden"
    assert got.id == "old"


# ---------------------------------------------------------------------------
# starting a run wipes what the last one collected (operator request)
#
# "restart the test should wipe collected data and start over" (2026-09-04).
# Observed on bonus with a thorough run already in flight: three runs coexisted
# for one fingerprint and current_for went on serving the OLD measurement --
# 1,616 confirmed, 1,454 published -- which was by then known to be an artefact
# of the harness's own token budget. Pressing the button changed nothing an
# operator or a client could see.
#
# This deliberately reverses the rule added earlier the same day that lets
# current_for scan PAST a valueless newest run to an older valid one. That rule
# is right for a run that merely FAILED; it is wrong for one the operator
# deliberately started, because they asked to measure again.
# ---------------------------------------------------------------------------


async def test_starting_a_run_clears_the_previous_ones(db):
    """Finished, hence clearable. `_row()` starts a run as 'running'."""
    repo = StressRunRepo(db)
    await repo.insert(_row(id="old"))
    await repo.finish("old", status="completed", limits={"quality_context": 1616})

    assert await repo.clear_for_model("m1") == 1
    assert await repo.get("old") is None
    assert await repo.current_for("m1", "sha256:aaa") is None


async def test_another_model_is_untouched(db):
    """Scoped to the model under test: wiping a neighbour's measurement would
    destroy data nobody asked to replace.

    Both runs are FINISHED first. `_row()` defaults to 'running', and
    clear_for_model deliberately never deletes a running row — an earlier
    version of this test inserted the defaults and then asserted a deletion
    the guard correctly refused, so it was failing the code rather than the
    other way round.
    """
    await db.execute("INSERT INTO models (id) VALUES ('m2')")
    await db.commit()
    repo = StressRunRepo(db)
    await repo.insert(_row(id="mine", model_id="m1"))
    await repo.insert(_row(id="theirs", model_id="m2"))
    await repo.finish("mine", status="completed", limits={"quality_context": 1})
    await repo.finish("theirs", status="completed", limits={"quality_context": 1})

    await repo.clear_for_model("m1")

    assert await repo.get("mine") is None
    assert await repo.get("theirs") is not None


async def test_a_run_in_flight_is_never_cleared(db):
    """The route already refuses a second run while one is active, but the repo
    must not make that check load-bearing: deleting a running row would orphan
    a live asyncio task whose `finish` then writes nothing, leaving the UI
    polling a run that no longer exists.
    """
    repo = StressRunRepo(db)
    await repo.insert(_row(id="live", status="running"))

    assert await repo.clear_for_model("m1") == 0
    assert await repo.get("live") is not None


async def test_clearing_an_untested_model_is_not_an_error(db):
    assert await StressRunRepo(db).clear_for_model("never-tested") == 0
