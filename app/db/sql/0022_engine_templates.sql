-- #162 engine templates: user-defined templates, per-model engine axis,
-- and trial-and-error stack attempts.

ALTER TABLE models ADD COLUMN engine_channel TEXT;
ALTER TABLE models ADD COLUMN engine_vllm_version TEXT;
ALTER TABLE models ADD COLUMN engine_image TEXT;

CREATE TABLE engine_templates (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  payload TEXT NOT NULL,                      -- full ModelTemplate as JSON
  source TEXT NOT NULL DEFAULT 'user'
    CHECK (source IN ('user')),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE stack_attempts (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  channel TEXT NOT NULL,
  vllm_version TEXT NOT NULL,
  image TEXT,
  result TEXT NOT NULL DEFAULT 'pending'
    CHECK (result IN ('pending','ok','failed')),
  error TEXT,
  category TEXT,                              -- stack_classifier category on failure
  suggested_next TEXT,                        -- JSON suggestion payload
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_stack_attempts_model ON stack_attempts(model_id, created_at);
