-- Migration 0010: create settings key/value store and seed defaults.
--
-- Why a new table: configuration was previously scattered between the
-- setup_state.draft JSON blob (allowed_gpu_indices) and ad-hoc files
-- (hf-token). Phase-3 of the UI redesign needs a single durable surface
-- for runtime tunables that the new /api/settings/runtime endpoint reads
-- and writes. INSERT OR IGNORE makes the seed idempotent so re-running
-- the migration (e.g. on a partially-built test DB) is safe.

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO settings (key, value) VALUES
  ('session_access_ttl_minutes', '15'),
  ('session_refresh_ttl_days', '7'),
  ('sse_ticket_ttl_seconds', '60'),
  ('default_token_expiration_days', '365'),
  ('rotation_grace_hours', '24'),
  ('log_retention_lines', '5000'),
  ('vllm_version', '0.9.2'),
  ('default_gpu_indices', '[0]');
