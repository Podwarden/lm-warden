-- vllm-warden 2026-05 overhaul · S5 (tokens-v2) · closes #104
--
-- Adds per-token rate-limit (sliding 10s window) and priority (0..9, STRICT)
-- to api_tokens, plus a minute-bucket per-token usage rollup feeding the new
-- GET /api/tokens/{id}/usage endpoint.
--
-- Forward-only: this project's migration runner (app/db/migrations.py) does
-- not support `down.sql`. Manual rollback recipe is documented inline below
-- so an operator can revert by hand if a release is yanked.
--
-- Rollback (manual, untested in CI — exercised in the migration up/down unit
-- test fixture instead). Triggers reference the columns they guard, so they
-- MUST be dropped BEFORE the columns or ALTER TABLE DROP COLUMN errors with
-- "error in trigger ... after drop column: no such column: NEW.priority":
--
--   BEGIN;
--     DROP TRIGGER IF EXISTS api_tokens_priority_range_insert;
--     DROP TRIGGER IF EXISTS api_tokens_priority_range_update;
--     DROP TRIGGER IF EXISTS api_tokens_rate_limit_tps_range_insert;
--     DROP TRIGGER IF EXISTS api_tokens_rate_limit_tps_range_update;
--     DROP TABLE IF EXISTS token_usage_minute;
--     -- 0035's sort index covers priority; an index on a column blocks its
--     -- DROP COLUMN just as a trigger does.
--     DROP INDEX IF EXISTS idx_api_tokens_priority_id;
--     -- SQLite has no DROP COLUMN before 3.35 — older runners must recreate
--     -- api_tokens. The runtime image (Ubuntu 22.04 base) ships SQLite
--     -- 3.37.2, which is new enough for the direct DROP COLUMN form below.
--     ALTER TABLE api_tokens DROP COLUMN priority;
--     ALTER TABLE api_tokens DROP COLUMN rate_limit_tps;
--     DELETE FROM schema_migrations WHERE filename = '0018_tokens_rate_priority.sql';
--   COMMIT;

-- ---- rate_limit_tps (NULL = unlimited; sliding 10s window in the proxy) ----
-- Existing rows backfill to NULL via column add (no DEFAULT clause needed
-- because SQLite already fills new columns with NULL on schema migration).
ALTER TABLE api_tokens ADD COLUMN rate_limit_tps INTEGER;

-- ---- priority (0..9, STRICT scheduler; 9 always served first) ----
-- Existing rows backfill to 5 via DEFAULT. CHECK enforces the 0..9 range
-- but SQLite cannot add a CHECK on ALTER TABLE in a single statement, so
-- the CHECK is applied via a CREATE TRIGGER below (run on INSERT/UPDATE).
ALTER TABLE api_tokens ADD COLUMN priority INTEGER NOT NULL DEFAULT 5;

CREATE TRIGGER api_tokens_priority_range_insert
BEFORE INSERT ON api_tokens
FOR EACH ROW
WHEN NEW.priority IS NOT NULL AND (NEW.priority < 0 OR NEW.priority > 9)
BEGIN
    SELECT RAISE(ABORT, 'priority must be between 0 and 9');
END;

CREATE TRIGGER api_tokens_priority_range_update
BEFORE UPDATE OF priority ON api_tokens
FOR EACH ROW
WHEN NEW.priority IS NOT NULL AND (NEW.priority < 0 OR NEW.priority > 9)
BEGIN
    SELECT RAISE(ABORT, 'priority must be between 0 and 9');
END;

CREATE TRIGGER api_tokens_rate_limit_tps_range_insert
BEFORE INSERT ON api_tokens
FOR EACH ROW
WHEN NEW.rate_limit_tps IS NOT NULL AND NEW.rate_limit_tps <= 0
BEGIN
    SELECT RAISE(ABORT, 'rate_limit_tps must be > 0 when set (NULL = unlimited)');
END;

CREATE TRIGGER api_tokens_rate_limit_tps_range_update
BEFORE UPDATE OF rate_limit_tps ON api_tokens
FOR EACH ROW
WHEN NEW.rate_limit_tps IS NOT NULL AND NEW.rate_limit_tps <= 0
BEGIN
    SELECT RAISE(ABORT, 'rate_limit_tps must be > 0 when set (NULL = unlimited)');
END;

-- ---- token_usage_minute (per-token minute-bucket rollup) ------------------
-- Same one-minute bucketing as the existing per-model `model_samples` table
-- so the two timelines line up for cross-cutting stats joins later (S7).
-- token_id is a FK in spirit but no constraint — schema_migrations runs
-- before the foreign_keys pragma flip in main.py, and the existing tables
-- already follow that loose convention.
CREATE TABLE token_usage_minute (
    token_id TEXT NOT NULL,
    minute INTEGER NOT NULL,  -- floor(epoch_seconds / 60)
    requests INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (token_id, minute)
);

-- Index supports the "last 24h" range query in routes_api.py — bounded
-- minute scan rather than full table scan once usage rows accumulate.
CREATE INDEX idx_token_usage_minute_minute ON token_usage_minute(minute);
