-- Per-request history: one row per completed /v1 request, as the proxy saw it.
--
-- Until now the only per-request record was the in-memory FinishedRing
-- (200 rows, 15 minutes, dies with the process). Two operator complaints
-- turned out to be the same missing thing: the "recently finished" table
-- answered no question because nothing aggregated, and the latency
-- distributions covered "since you opened the tab, at most 5 minutes" because
-- the engine publishes only cumulative histograms and the page had nowhere
-- else to look. Both are served from this table.
--
-- The latency columns are the PROXY's measurements (TTFT at the first streamed
-- frame, duration at the end of the stream) and exist identically for every
-- backend. llama.cpp publishes no latency histogram at all; this table is what
-- gives GGUF models a latency panel.
--
-- `model_id` is the models table's ROW id (what every `?models=` filter uses)
-- and `model` is the served name (what a human reads). They differ, and
-- conflating them was the bug fixed in !384. NOT a foreign key on purpose:
-- history should outlive a deleted model row -- the served name is carried so
-- the row stays readable -- and `model_samples`' ON DELETE CASCADE is why the
-- token chart forgets a model the moment it is removed.
--
-- `finished_at` is wall-clock epoch seconds, the key for windows and retention.
-- `started_iso` is kept as written by the registry for display.
--
-- Volume: thousands of requests a day at ~200 bytes a row with both indexes is
-- on the order of a megabyte a day. Retention is enforced by the stats pruner
-- (app/runtime/stats_pruner.py) by age AND by row count, both configurable --
-- see Settings.request_history_retention_days / request_history_max_rows.

CREATE TABLE request_history (
  id TEXT PRIMARY KEY,
  finished_at REAL NOT NULL,
  model_id TEXT NOT NULL,
  model TEXT NOT NULL,
  token_name TEXT,
  client_ip TEXT,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  duration_s REAL NOT NULL,
  -- NULL when no token ever arrived (immediate error, abort before the first
  -- frame, or a non-streaming request, where there is no first frame to time).
  ttft_s REAL,
  finish_reason TEXT,
  orphan INTEGER NOT NULL DEFAULT 0,
  started_iso TEXT NOT NULL
);

-- Every read is "newest first within a window", optionally for a model set;
-- the pruner deletes by finished_at and by rank over it.
CREATE INDEX idx_request_history_finished_at ON request_history(finished_at);
CREATE INDEX idx_request_history_model_finished
  ON request_history(model_id, finished_at);
