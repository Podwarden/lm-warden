CREATE TABLE counters (
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  token_id TEXT REFERENCES api_tokens(id) ON DELETE SET NULL,
  requests INTEGER NOT NULL DEFAULT 0,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (model_id, token_id)
);

CREATE TABLE model_samples (
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  minute INTEGER NOT NULL,
  requests INTEGER NOT NULL DEFAULT 0,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (model_id, minute)
);

CREATE TABLE gpu_samples (
  gpu_index INTEGER NOT NULL,
  minute INTEGER NOT NULL,
  utilization_pct INTEGER,
  memory_used_mib INTEGER,
  memory_total_mib INTEGER,
  PRIMARY KEY (gpu_index, minute)
);
