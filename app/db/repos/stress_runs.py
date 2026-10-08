"""Persistence for stress runs (design §7.2).

Results are keyed on ``(model_id, fingerprint)``, not on ``model_id`` alone.
The same model measured at two context sizes produces two rows and both are
true — neither supersedes the other, because they describe different
configurations. That is what makes the reload sweep expressible at all.

Reading is deliberately narrow: ``current_for`` is the only way a client-facing
surface gets a number, and it refuses to return anything that is not a
completed, publishable run under the exact fingerprint asked for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import aiosqlite

from app.stress.measured import has_measurement


@dataclass
class StressRunRow:
    id: str
    model_id: str
    fingerprint: str
    mode: str
    status: str = "running"
    truncated_by: str | None = None
    traffic_observed: bool = False
    non_monotone: bool = False
    loads_since_measurement: int = 0
    limits: dict | None = None
    observations: list | None = None
    recommended_config: dict | None = None
    seed: int = 0
    probe_suite_hash: str = ""
    algorithm_version: int = 1
    started_at: str | None = None
    #: Mid-run phase and probe count; None once the run is over.
    progress: dict | None = None
    finished_at: str | None = None
    last_error: str | None = None
    _unused: dict = field(default_factory=dict, repr=False)


_COLS = (
    "id, model_id, fingerprint, mode, status, truncated_by, traffic_observed, "
    "non_monotone, loads_since_measurement, limits, observations, "
    "recommended_config, seed, probe_suite_hash, algorithm_version, "
    "started_at, finished_at, last_error, progress"
)


def _decode(row: tuple) -> StressRunRow:
    return StressRunRow(
        id=row[0],
        model_id=row[1],
        fingerprint=row[2],
        mode=row[3],
        status=row[4],
        truncated_by=row[5],
        traffic_observed=bool(row[6]),
        non_monotone=bool(row[7]),
        loads_since_measurement=row[8],
        limits=json.loads(row[9]) if row[9] else None,
        observations=json.loads(row[10]) if row[10] else None,
        recommended_config=json.loads(row[11]) if row[11] else None,
        seed=row[12],
        probe_suite_hash=row[13],
        algorithm_version=row[14],
        started_at=row[15],
        finished_at=row[16],
        last_error=row[17],
        progress=json.loads(row[18]) if row[18] else None,
    )


class StressRunRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def insert(self, row: StressRunRow) -> None:
        await self.db.execute(
            "INSERT INTO stress_runs (id, model_id, fingerprint, mode, status, seed, "
            "probe_suite_hash, algorithm_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.id,
                row.model_id,
                row.fingerprint,
                row.mode,
                row.status,
                row.seed,
                row.probe_suite_hash,
                row.algorithm_version,
            ),
        )
        await self.db.commit()

    async def get(self, run_id: str) -> StressRunRow | None:
        cur = await self.db.execute(f"SELECT {_COLS} FROM stress_runs WHERE id = ?", (run_id,))
        got = await cur.fetchone()
        return _decode(got) if got else None

    async def finish(
        self,
        run_id: str,
        *,
        status: str,
        limits: dict | None = None,
        observations: list | None = None,
        recommended_config: dict | None = None,
        truncated_by: str | None = None,
        non_monotone: bool = False,
        traffic_observed: bool = False,
        last_error: str | None = None,
    ) -> None:
        await self.db.execute(
            "UPDATE stress_runs SET status = ?, limits = ?, observations = ?, "
            "recommended_config = ?, truncated_by = ?, non_monotone = ?, "
            "traffic_observed = ?, last_error = ?, finished_at = datetime('now') "
            "WHERE id = ?",
            (
                status,
                json.dumps(limits) if limits is not None else None,
                json.dumps(observations) if observations is not None else None,
                json.dumps(recommended_config) if recommended_config is not None else None,
                truncated_by,
                int(non_monotone),
                int(traffic_observed),
                last_error,
                run_id,
            ),
        )
        await self.db.commit()

    async def current_for(self, model_id: str, fingerprint: str) -> StressRunRow | None:
        """The newest publishable measurement for exactly this configuration.

        Four conditions, each of which exists because publishing without it
        would tell a client something untrue:

        * ``status = 'completed'`` — an interrupted run has a bracket but no
          confirmation, so its lower bound would understate the limit while
          carrying full confidence.
        * matching ``fingerprint`` — a measurement from different hardware,
          engine build or co-resident set is not about this model as it stands.
        * ``non_monotone = 0`` — the search was unsound; a visible gap beats a
          number that may be several times too low.
        * ``traffic_observed = 0`` — KV is a shared pool, so foreign traffic
          moved the measurement by an unknown amount.

        A fifth condition cannot be expressed in this SQL and is applied after
        the fetch: the run must actually have MEASURED something. Completing
        and measuring are different states, and treating them as one cost an
        operator their retest (2026-09-04) — a run that confirmed nothing held
        the six-hour cooldown, so pressing the button returned `200 reused`
        naming the dead run, with no error and no way out. A run that learned
        nothing is the strongest reason to allow another, not to refuse one.
        """
        cur = await self.db.execute(
            f"SELECT {_COLS} FROM stress_runs "
            "WHERE model_id = ? AND fingerprint = ? AND status = 'completed' "
            "AND non_monotone = 0 AND traffic_observed = 0 "
            "ORDER BY started_at DESC",
            (model_id, fingerprint),
        )
        # Newest FIRST but not newest ONLY. A single LIMIT 1 plus a post-fetch
        # test hides a valid older measurement behind a newer valueless run --
        # and `published_record` then falls back to history and serves that
        # same older run as `provenance: stale` with "the hardware changed",
        # which is false. Withdrawing a live measurement and lying about why is
        # worse than the cooldown bug this condition was added to fix.
        for got in await cur.fetchall():
            row = _decode(got)
            if has_measurement(row.limits, row.recommended_config):
                return row
        return None

    async def update_progress(self, run_id: str, progress: dict) -> None:
        """Overwrite this run's progress. Cheap, frequent, and lossy by design.

        Called from the heartbeat that already ticks for the lease, so it adds
        no timer of its own. Nothing reads the previous value, so an
        overwrite is correct and a history would be dead weight -- a run emits
        hundreds of these and only the last one is ever wanted.
        """
        await self.db.execute(
            "UPDATE stress_runs SET progress = ? WHERE id = ?",
            (json.dumps(progress), run_id),
        )
        await self.db.commit()

    async def clear_for_model(self, model_id: str) -> int:
        """Delete this model's finished runs. Returns how many went.

        Called only for an EXPLICIT reset -- the "wipe and run again" action
        offered beside a finished run's results. Never on an ordinary start.

        That distinction is the whole design. A run that
        merely FAILED must not withdraw a good older measurement, which is why
        `current_for` scans past valueless runs. A wipe the operator asked for
        is the opposite case: they have looked at the results and decided to
        replace them. Only they can tell those apart, so only they trigger this.

        Running rows are never deleted. The route already refuses to start a
        second run while one is active, but a delete here would orphan a live
        asyncio task whose ``finish`` would then write nothing, leaving the UI
        polling a run that no longer exists -- and correctness should not
        depend on a check in a different module.
        """
        cur = await self.db.execute(
            "DELETE FROM stress_runs WHERE model_id = ? AND status != 'running'",
            (model_id,),
        )
        await self.db.commit()
        return cur.rowcount or 0

    async def list_for_model(self, model_id: str, limit: int = 20) -> list[StressRunRow]:
        cur = await self.db.execute(
            f"SELECT {_COLS} FROM stress_runs WHERE model_id = ? "
            "ORDER BY started_at DESC LIMIT ?",
            (model_id, limit),
        )
        return [_decode(r) for r in await cur.fetchall()]

    async def mark_interrupted_on_startup(self) -> int:
        """A run is in-process, so a warden restart kills it mid-search.

        Such a row must never be published: it holds a bracket that was never
        confirmed. Called from boot reconciliation alongside the model sweep.
        """
        cur = await self.db.execute(
            "UPDATE stress_runs SET status = 'interrupted', "
            "finished_at = datetime('now'), "
            "last_error = 'warden restarted while this run was in progress' "
            "WHERE status = 'running'"
        )
        await self.db.commit()
        return cur.rowcount or 0

    async def bump_loads_since_measurement(self, model_id: str) -> None:
        """Called on every load of this model.

        An identical-config reload leaves the fingerprint unchanged, so a
        measurement survives it — and whether that is correct is unknown. This
        counter is what lets the question be answered from data later rather
        than assumed now.
        """
        await self.db.execute(
            "UPDATE stress_runs SET loads_since_measurement = loads_since_measurement + 1 "
            "WHERE model_id = ? AND status = 'completed'",
            (model_id,),
        )
        await self.db.commit()
