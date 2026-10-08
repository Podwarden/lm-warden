CREATE TABLE model_runtime (
  model_id TEXT PRIMARY KEY REFERENCES models(id) ON DELETE CASCADE,
  pid INTEGER,
  port INTEGER,
  started_at TEXT,
  health_ok INTEGER NOT NULL DEFAULT 0,
  last_health_at TEXT
);
