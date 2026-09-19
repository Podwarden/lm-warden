-- Per-(key, model VARIANT) minute-bucket rollup for the token page's "Usage by
-- model" card and its per-model tokens chart.
--
-- None of the existing stores can answer "which models did this key use, and
-- how much, over <period>":
--   * token_usage_minute is (token_id, minute) -- no model dimension;
--   * request_history has the model but keeps 30 days / 200k rows, and has
--     carried token_id only since 0033;
--   * counters is (model, token) but all-time, with no time axis.
-- So this is a new rollup, written by the proxy next to token_usage_minute
-- (app/proxy/routes.py, _record_counters) with the same minute integer, only
-- when the request carried a key.
--
-- Attribution is to the exact VARIANT, not just models.id: a models row is
-- edited in place (the try-stack rewrites its engine image/version, the
-- settings form its quantization, dtype, context, args, revision), and one id
-- would merge "vLLM 0.9 FP8" with "vLLM 0.11 AWQ". See app/runtime/variants.py
-- for the identity fields and why the variant is taken from the RUNNING
-- engine, not the current row.
--
-- No backfill: there is nothing to backfill it from (see above). The series
-- endpoint reports the earliest usage row store-wide as `by_model_since` so the
-- UI can say where the breakdown starts.

-- ---- model_variants ----------------------------------------------------------
-- One row per variant ever served. id = first 16 hex of sha256 over the
-- canonical JSON of {model_id, descriptor}. descriptor is that canonical JSON
-- of the identity fields only -- never extra_env, which can hold secrets.
-- served_model_name is the name when first seen; display names are resolved
-- through `models` at read time and fall back to this, then to the id.
CREATE TABLE IF NOT EXISTS model_variants (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    served_model_name TEXT NOT NULL,
    descriptor TEXT NOT NULL,
    first_seen REAL NOT NULL  -- epoch seconds
);

-- ---- token_model_usage_minute ------------------------------------------------
-- model_id is a plain column (denormalised from the variant) so the card and
-- the chart group by model without a join.
CREATE TABLE IF NOT EXISTS token_model_usage_minute (
    token_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    minute INTEGER NOT NULL,  -- floor(epoch_seconds / 60)
    requests INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (token_id, variant_id, minute)
);

-- The PK puts variant_id between the key and the minute, so a "this key set
-- over [from, to)" scan would read every minute of the key; this serves it.
CREATE INDEX IF NOT EXISTS idx_token_model_usage_minute_token_minute
    ON token_model_usage_minute(token_id, minute);

-- by_model_since is MIN(minute) over the whole table on every page poll; this
-- lets SQLite read its first entry instead of scanning (same reasoning as 0034).
CREATE INDEX IF NOT EXISTS idx_token_model_usage_minute_minute
    ON token_model_usage_minute(minute);

-- ---- request_history.variant_id ----------------------------------------------
-- Which variant served each request. Nullable: every older row stays NULL.
ALTER TABLE request_history ADD COLUMN variant_id TEXT;
