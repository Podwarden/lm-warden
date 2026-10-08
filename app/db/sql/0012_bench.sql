-- Migration 0012: benchmark-v2 persistence (run / load-config attempt / cell).
--
-- Spec: docs/superpowers/specs/2026-05-13-vllm-warden-benchmark-v2-design.md
-- §Persistence — SQLite schema.
--
-- Three tables form a run → load-config-attempt → cell hierarchy. Cascade
-- delete propagates from run to its attempts to its cells so DELETE
-- /api/bench/runs/{id} cleans up without orphans.
--
-- The partial unique index `bench_run_one_active_per_model_idx` enforces
-- the active-run lock from the spec (P0-4): at most one queued|running|
-- paused run per model. POST /api/bench/runs catches the resulting
-- IntegrityError and returns 409 with the existing run_id.
--
-- `pause_requested` is the cooperative pause flag (P0-5): the cli polls
-- this between cells, never mid-cell.

CREATE TABLE bench_run (
  run_id           TEXT PRIMARY KEY,           -- ulid
  model_id         TEXT NOT NULL,
  gpu_set          TEXT NOT NULL,              -- e.g. "0,1"
  matrix_hash      TEXT NOT NULL,              -- sha256 of resolved matrix + corpus manifest
  status           TEXT NOT NULL,              -- queued|running|paused|done|cancelled|failed
  pause_requested  INTEGER NOT NULL DEFAULT 0, -- cooperative pause flag (spec P0-5)
  created_at       INTEGER NOT NULL,
  started_at       INTEGER,
  ended_at         INTEGER,
  pid              INTEGER,
  summary_json     TEXT
);

-- Active-run lock (spec P0-4): at most one queued|running|paused per model.
CREATE UNIQUE INDEX bench_run_one_active_per_model_idx
  ON bench_run(model_id)
  WHERE status IN ('queued', 'running', 'paused');

-- Fast lookup by (model_id, status) for /api/bench/runs?model_id= and the
-- /benchmarks dashboard cross-model summary.
CREATE INDEX bench_run_model_idx
  ON bench_run(model_id, status);

CREATE TABLE bench_load_config_attempt (
  attempt_id       TEXT PRIMARY KEY,           -- ulid
  run_id           TEXT NOT NULL REFERENCES bench_run(run_id) ON DELETE CASCADE,
  load_config_json TEXT NOT NULL,              -- {quantization, tp, gpu_mem, max_model_len, max_num_seqs?}
  load_ok          INTEGER,                    -- 0/1/NULL
  load_ms          INTEGER,
  load_error       TEXT,
  envelope_json    TEXT,                       -- {suggested_concurrent_requests, max_new, limited_by}
  created_at       INTEGER NOT NULL,
  ended_at         INTEGER
);

CREATE TABLE bench_cell (
  cell_id          TEXT PRIMARY KEY,           -- ulid
  attempt_id       TEXT NOT NULL REFERENCES bench_load_config_attempt(attempt_id) ON DELETE CASCADE,
  concurrency      INTEGER NOT NULL,
  prompt_size      TEXT NOT NULL,              -- bucket: "1k","4k","16k","31k","128k","1m_plus"
  max_new          INTEGER NOT NULL,
  status           TEXT NOT NULL,              -- pending|running|ok|fail|skipped
  duration_ms      INTEGER,
  agg_tps          REAL,
  p50_ttft_ms      INTEGER,
  p95_ttft_ms      INTEGER,
  p50_latency_ms   INTEGER,
  p95_latency_ms   INTEGER,
  pass_rate        REAL,
  ok_count         INTEGER,
  fail_count       INTEGER,
  error            TEXT,
  created_at       INTEGER NOT NULL,
  ended_at         INTEGER,
  UNIQUE(attempt_id, concurrency, prompt_size, max_new)
);

CREATE INDEX bench_cell_attempt_idx
  ON bench_cell(attempt_id);
