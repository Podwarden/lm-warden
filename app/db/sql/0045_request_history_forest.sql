-- Session forest (docs/superpowers/specs/2026-10-05-session-forest-design.md §3).
--
-- session_key / parent_session_key: HMAC-SHA256(cookie_secret, raw id)[:16] hex.
--   The raw session id (it can embed an account uuid) never reaches the DB.
-- turn_index: 0-based position of the request in its session (proxy LRU).
-- batch_id: the X-Batch-Id request header, <= 64 printable chars; groups a batch
--   into one tree.
-- tools_out: JSON [[family, name_hmac8], ...] the model emitted in this response.
-- tools_in:  JSON [[family, tokens, failed], ...] tool results this request carried.
-- All NULL on rows from before this migration and whenever a signal is absent.
ALTER TABLE request_history ADD COLUMN session_key TEXT;
ALTER TABLE request_history ADD COLUMN parent_session_key TEXT;
ALTER TABLE request_history ADD COLUMN turn_index INTEGER;
ALTER TABLE request_history ADD COLUMN batch_id TEXT;
ALTER TABLE request_history ADD COLUMN tools_out TEXT;
ALTER TABLE request_history ADD COLUMN tools_in TEXT;
CREATE INDEX idx_request_history_session_key ON request_history(session_key, finished_at);
CREATE INDEX idx_request_history_token_batch ON request_history(token_id, batch_id);
