-- Migration 0017: drop benchmark-v2 persistence (run / load-config attempt / cell).
--
-- 2026-05 overhaul, Slice S1 — bench-removal. The benchmark-v2 subsystem
-- (introduced in migration 0012) is being excised entirely; the slice
-- replaces "we know best" auto-config with a "Suggest values" preset
-- button that is planned to land in S3/S4 against an external Hub
-- catalog rather than a per-deployment grid search.
--
-- This migration is idempotent: ``DROP TABLE IF EXISTS`` makes re-running
-- the migration safe (the upstream runner also skips already-applied
-- entries via schema_migrations, but defensive in case of manual replay).
--
-- Order matters: children first. ``bench_cell`` references
-- ``bench_load_config_attempt`` and ``bench_load_config_attempt``
-- references ``bench_run``, both with ON DELETE CASCADE. Dropping
-- parents first while children still exist would fail in strict mode.
-- SQLite's default mode tolerates it, but explicit child-first ordering
-- documents intent and survives a future PRAGMA foreign_keys=ON default.
--
-- Indexes and the partial unique index from 0012 (bench_run_one_active_per_model_idx,
-- bench_run_model_idx, bench_cell_attempt_idx) are dropped automatically
-- when their parent tables are dropped.
--
-- Row counts: no programmatic logging here — SQLite has no ``RAISE NOTICE``
-- equivalent. Operators wanting a pre-drop count can run
-- ``SELECT COUNT(*) FROM bench_run`` against the live DB before deploying.

DROP TABLE IF EXISTS bench_cell;
DROP TABLE IF EXISTS bench_load_config_attempt;
DROP TABLE IF EXISTS bench_run;
