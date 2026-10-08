import aiosqlite


class SamplesRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def add_model_sample(
        self,
        model_id: str,
        minute: int,
        delta_requests: int,
        delta_prompt: int,
        delta_completion: int,
        delta_cached: int | None = None,
    ) -> None:
        # delta_cached is the request's MEASURED cached prompt tokens; None =
        # the engine did not report it, which adds to neither column.
        await self.db.execute(
            "INSERT INTO model_samples(model_id, minute, requests, prompt_tokens, "
            "completion_tokens, cached_tokens, cached_measured_requests) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(model_id, minute) DO UPDATE SET "
            "requests = requests + excluded.requests, "
            "prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
            "completion_tokens = completion_tokens + excluded.completion_tokens, "
            "cached_tokens = cached_tokens + excluded.cached_tokens, "
            "cached_measured_requests = cached_measured_requests "
            "+ excluded.cached_measured_requests",
            (
                model_id,
                minute,
                delta_requests,
                delta_prompt,
                delta_completion,
                int(delta_cached) if delta_cached is not None else 0,
                1 if delta_cached is not None else 0,
            ),
        )
        await self.db.commit()

    async def add_gpu_sample(
        self,
        gpu_index: int,
        minute: int,
        utilization_pct: int,
        memory_used_mib: int,
        memory_total_mib: int,
        name: str | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO gpu_samples(gpu_index, minute, utilization_pct, "
            "memory_used_mib, memory_total_mib, name) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(gpu_index, minute) DO UPDATE SET "
            "utilization_pct = excluded.utilization_pct, "
            "memory_used_mib = excluded.memory_used_mib, "
            "memory_total_mib = excluded.memory_total_mib, "
            "name = COALESCE(excluded.name, name)",
            (gpu_index, minute, utilization_pct, memory_used_mib, memory_total_mib, name),
        )
        await self.db.commit()

    async def model_samples_since(self, model_id: str, since_minute: int) -> list[dict]:
        cur = await self.db.execute(
            "SELECT minute, requests, prompt_tokens, completion_tokens "
            "FROM model_samples WHERE model_id = ? AND minute >= ? ORDER BY minute",
            (model_id, since_minute),
        )
        return [
            {"minute": r[0], "requests": r[1], "prompt_tokens": r[2], "completion_tokens": r[3]}
            for r in await cur.fetchall()
        ]

    async def add_power_sample(
        self,
        gpu_idx: int,
        minute: int,
        watts: float,
    ) -> None:
        """Accumulate one 5s power-draw sample into the per-(gpu, minute) bucket.

        S7 (#124) — power_samples uses a write-path aggregator (watts_sum +
        samples counter) rather than last-write-wins so the read side can
        report a true minute-average rather than "whichever 5s tick landed
        last in this minute". Schema: app/db/sql/0019_power_samples.sql.
        """
        await self.db.execute(
            "INSERT INTO power_samples(gpu_idx, minute, watts_sum, samples) "
            "VALUES (?, ?, ?, 1) "
            "ON CONFLICT(gpu_idx, minute) DO UPDATE SET "
            "watts_sum = watts_sum + excluded.watts_sum, "
            "samples = samples + 1",
            (gpu_idx, minute, float(watts)),
        )
        await self.db.commit()

    async def power_samples_since(self, since_minute: int) -> list[dict]:
        """Return per-(gpu, minute) average watts for minutes >= since_minute.

        ``avg_watts`` is computed inline as ``watts_sum / samples`` so the
        caller never has to remember the accumulator shape. Rows are
        ordered (minute ASC, gpu_idx ASC) for stable chart layouts.
        """
        cur = await self.db.execute(
            "SELECT gpu_idx, minute, watts_sum, samples, "
            "       watts_sum / NULLIF(samples, 0) AS avg_watts "
            "FROM power_samples WHERE minute >= ? "
            "ORDER BY minute, gpu_idx",
            (since_minute,),
        )
        return [
            {
                "gpu_idx": r[0],
                "minute": r[1],
                "watts_sum": r[2],
                "samples": r[3],
                "avg_watts": r[4],
            }
            for r in await cur.fetchall()
        ]

    async def gpu_samples_since(self, since_minute: int) -> list[dict]:
        cur = await self.db.execute(
            "SELECT gpu_index, minute, utilization_pct, memory_used_mib, memory_total_mib, name "
            "FROM gpu_samples WHERE minute >= ? ORDER BY minute, gpu_index",
            (since_minute,),
        )
        return [
            {
                "gpu_index": r[0],
                "minute": r[1],
                "utilization_pct": r[2],
                "memory_used_mib": r[3],
                "memory_total_mib": r[4],
                "name": r[5],
            }
            for r in await cur.fetchall()
        ]

    async def prune_older_than(self, before_minute: int) -> None:
        await self.db.execute("DELETE FROM model_samples WHERE minute < ?", (before_minute,))
        await self.db.execute("DELETE FROM gpu_samples WHERE minute < ?", (before_minute,))
        # S7 (#124) — power_samples follows the same retention as gpu_samples
        # (one row per (gpu, minute), so volume is identical) and shares the
        # pruner schedule.
        await self.db.execute("DELETE FROM power_samples WHERE minute < ?", (before_minute,))
        await self.db.commit()
