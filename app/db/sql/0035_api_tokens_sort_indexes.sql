-- GET /api/tokens is paged and sortable, and must stay cheap with millions of
-- keys (app/db/repos/tokens.py, token_page_sql).
--
-- One index per index-servable sort order, each ending in `id`: the list
-- always breaks ties on id in the sort's own direction, so the whole ORDER BY
-- -- key then id -- comes straight off the index, walked forwards or backwards,
-- and a page reads only offset + limit entries instead of sorting the table.
-- NULLs (never used, never expires) sort last in both directions: SQLite's
-- NULLS LAST over an ascending walk is two passes over the same index, so the
-- nullable keys need nothing extra. `name` is sorted case-insensitively, so
-- its index carries the same collation. The usage_24h and status sorts are
-- computed per row and cannot use an index.
--
-- idx_api_tokens_hidden is partial over exactly the keys the list hides
-- (revoked, never rotated), leads with expires_at and carries both columns of
-- its WHERE so it covers every query against it. The unsearched `total` is
-- COUNT(*) of the table -- a walk of its smallest index -- minus the hidden
-- keys, and `near_expiry` is a covering range count on the expires index
-- minus the hidden keys in the same range: no table row is read to test the
-- visibility rule (token_counts_sql names this index with INDEXED BY).
--
-- idx_api_tokens_rotated_from serves each page row's successor lookup, the
-- status sort's "orphan" test, TokenRepo.lineage's forward walk, and the
-- ON DELETE SET NULL action a DELETE of any key runs against rotated_from --
-- which otherwise scans the whole table.
--
-- Two older single-column indexes are left-prefixes of the new composites and
-- are dropped: 0009's idx_tokens_expires_at (expires_at -> expires_at, id) and
-- 0007's idx_api_tokens_prefix (prefix -> prefix, id). Every query that used
-- them -- the near-expiry range count, a prefix lookup -- is served as well by
-- the wider index, and each drop is one index fewer to maintain on every
-- insert. IF EXISTS, like the creates, so a hand-repaired DB replays cleanly.

CREATE INDEX IF NOT EXISTS idx_api_tokens_created_id ON api_tokens(created_at, id);
CREATE INDEX IF NOT EXISTS idx_api_tokens_last_used_id ON api_tokens(last_used_at, id);
CREATE INDEX IF NOT EXISTS idx_api_tokens_expires_id ON api_tokens(expires_at, id);
CREATE INDEX IF NOT EXISTS idx_api_tokens_name_id ON api_tokens(name COLLATE NOCASE, id);
CREATE INDEX IF NOT EXISTS idx_api_tokens_priority_id ON api_tokens(priority, id);
CREATE INDEX IF NOT EXISTS idx_api_tokens_prefix_id ON api_tokens(prefix, id);
CREATE INDEX IF NOT EXISTS idx_api_tokens_hidden
    ON api_tokens(expires_at, revoked_at, rotated_at)
    WHERE revoked_at IS NOT NULL AND rotated_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_api_tokens_rotated_from ON api_tokens(rotated_from);
DROP INDEX IF EXISTS idx_tokens_expires_at;
DROP INDEX IF EXISTS idx_api_tokens_prefix;
