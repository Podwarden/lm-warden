-- vllm-warden 2026-05 overhaul · S7 (stats-v2) · #124
--
-- Per-GPU power draw samples for the stats-v2 page (`/api/stats/v2/overview`).
-- Same minute-bucketing as the existing ``gpu_samples`` and ``model_samples``
-- tables so the three timelines line up for cross-cutting stats joins.
-- Rollup strategy: READ-PATH rollup. The collector writes one row per
-- (gpu_idx, minute) and UPSERTs in-bucket, picking up the AVG-style
-- accumulator (running mean across the 5s-cadence sample passes inside a
-- given minute). This matches the existing ``gpu_samples`` precedent (see
-- ``SamplesRepo.add_gpu_sample`` — same UPSERT shape, last-write-wins for
-- util/mem; for ``watts`` we want a representative average across the
-- minute so we store ``watts_sum`` + ``samples`` and the read-side divides).
--
-- Forward-only: this project's migration runner (app/db/migrations.py) does
-- not support `down.sql`. Manual rollback recipe is documented inline so an
-- operator can revert by hand if a release is yanked.
--
-- Rollback (manual, exercised in the migration up/down unit test fixture):
--
--   BEGIN;
--     DROP INDEX IF EXISTS idx_power_samples_gpu_idx_ts;
--     DROP TABLE IF EXISTS power_samples;
--     DELETE FROM schema_migrations WHERE filename = '0019_power_samples.sql';
--   COMMIT;

-- ---- power_samples (per-GPU minute-bucket power-draw rollup) -------------
-- Columns:
--   gpu_idx     — GPU index matching gpu_samples.gpu_index (loose FK in spirit)
--   minute      — floor(epoch_seconds / 60); same bucket integer as
--                 gpu_samples.minute and model_samples.minute for joins
--   watts_sum   — running sum of all watt samples landed in this bucket
--   samples     — number of samples that contributed to watts_sum; avg-watts
--                 in this bucket = watts_sum / NULLIF(samples, 0)
-- Composite PK on (gpu_idx, minute) keeps one row per (GPU, minute) bucket
-- and makes the UPSERT in SamplesRepo.add_power_sample trivially correct.
CREATE TABLE power_samples (
    gpu_idx INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    watts_sum REAL NOT NULL DEFAULT 0.0,
    samples INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (gpu_idx, minute)
);

-- Read path is "last N minutes per gpu, newest first" (the overview chart).
-- A (gpu_idx, ts DESC) index would be ideal but our minute integer column
-- already gives DESC ordering for free; this index makes the per-GPU range
-- scan use a bounded index range rather than a full table scan once power
-- rows accumulate (one row per GPU per minute → 1440 rows/day/GPU).
CREATE INDEX idx_power_samples_gpu_idx_ts ON power_samples(gpu_idx, minute DESC);
