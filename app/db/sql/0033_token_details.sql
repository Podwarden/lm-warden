-- Token details page (docs/superpowers/specs/2026-09-18-token-details-design.md §2).
--
-- api_tokens.paused_at: when the operator paused this key, NULL = not paused.
-- require_bearer (app/proxy/auth.py) answers a paused key with 403 "token
-- paused" once the expired and revoked checks have passed, so a dead key still
-- gets its 401. Requests already in flight are not touched.
--
-- request_history.token_id: the api_tokens row id the request authenticated
-- with. Until now a row named its key by `token_name` only, and names are
-- reused across rotations: the key called "ip-harness" last week is today's
-- "ip-harness (old 1)". There is NO BACKFILL for exactly that reason. Matching
-- old rows by name would file one key's traffic under another, so rows written
-- before this migration read NULL and the token page says per-token timings
-- start at the deploy that shipped this file.
--
-- Not a foreign key, for the same reason model_id is not one: history should
-- outlive a deleted token row.

ALTER TABLE api_tokens ADD COLUMN paused_at TEXT;
ALTER TABLE request_history ADD COLUMN token_id TEXT;
-- The series endpoint reads "these keys, this window" (app/tokens/series.py).
CREATE INDEX idx_request_history_token ON request_history(token_id, finished_at);
