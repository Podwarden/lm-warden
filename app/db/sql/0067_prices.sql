-- 0067_prices.sql
--
-- Prices tab (#347). ``price_catalogue`` is the table of public list prices the
-- install keeps: refreshed from OpenRouter / LiteLLM (app/lens/price_sources.py)
-- and seeded from the snapshot shipped with the release (app/lens/price_snapshot.json).
-- It never holds an admin's price: an override is a ``rate_cards`` row
-- (org_id NULL), which wins over the catalogue (app/lens/prices.py).
--
-- host       the provider's API host, as ``request_history.upstream`` records it
-- model_key  the model name folded for matching (price_sources.model_key)
-- *_micros   per million tokens, micro-units of ``currency`` (100000 = 0.10)
-- source     "openrouter", "litellm" or "snapshot"
-- fetched_at ISO UTC time the price was read from that source
--
-- price_audit: one row per deliberate change of a price override, of a model's
-- reference card or of the refresh interval (who, when, before and after).
-- Rollback: DROP TABLE price_catalogue; DROP TABLE price_audit;
CREATE TABLE price_catalogue (
  host TEXT NOT NULL,
  model_key TEXT NOT NULL,
  model TEXT NOT NULL,
  currency TEXT NOT NULL DEFAULT 'USD',
  input_per_m_micros INTEGER NOT NULL,
  output_per_m_micros INTEGER,
  cache_read_per_m_micros INTEGER,
  cache_write_per_m_micros INTEGER,
  source TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  PRIMARY KEY (host, model_key)
);

CREATE TABLE price_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  provider TEXT,
  model TEXT,
  old_value TEXT,
  new_value TEXT
);
