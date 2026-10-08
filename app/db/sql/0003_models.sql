CREATE TABLE models (
  id TEXT PRIMARY KEY,
  served_model_name TEXT NOT NULL UNIQUE,
  hf_repo TEXT NOT NULL,
  hf_revision TEXT NOT NULL DEFAULT 'main',
  gpu_indices TEXT NOT NULL,
  tensor_parallel_size INTEGER NOT NULL DEFAULT 1,
  dtype TEXT,
  max_model_len INTEGER,
  gpu_memory_utilization REAL NOT NULL DEFAULT 0.9,
  trust_remote_code INTEGER NOT NULL DEFAULT 0,
  extra_args TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'registered'
    CHECK (status IN ('registered','pulling','pulled','loading','loaded','unloading','failed')),
  pulled_bytes INTEGER NOT NULL DEFAULT 0,
  pulled_total INTEGER,
  last_error TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
