-- Model load history: per-step durations from successful loads, so the next
-- load of the same model + config can show "usually ~6m" bars and a ready-by
-- time (#model-load-progress, Task 7).
--
-- Rows are keyed on (model_id, fingerprint) so the same model at two context
-- sizes or two engine versions keeps two independent histories — the
-- fingerprint is the identity of what was loaded (see load_fingerprint in
-- app/db/repos/load_history.py). The repo keeps only the newest 20 per model
-- so the table stays bounded; `usual` takes the median of the last 5.
--
-- FK to models(id) ON DELETE CASCADE: deleting a model drops its history,
-- matching the convention in 0004/0006/0022/0029.

CREATE TABLE model_load_history (
  id INTEGER PRIMARY KEY,
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  fingerprint TEXT NOT NULL,
  finished_at TEXT NOT NULL DEFAULT (datetime('now')),
  total_s REAL NOT NULL,
  steps_json TEXT NOT NULL
);

CREATE INDEX idx_model_load_history_model_fp_time
  ON model_load_history(model_id, fingerprint, finished_at DESC);
