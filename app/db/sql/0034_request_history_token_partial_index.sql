-- #251: `latency_since` on every token-page poll.
--
-- GET /api/tokens/{id}/series asks when per-token history begins:
--   SELECT MIN(finished_at) FROM request_history WHERE token_id IS NOT NULL
-- (app/stats/request_history.py, earliest_token_finished_at). Without this
-- index the planner walks idx_request_history_finished_at from the oldest end
-- until it meets a row with a token_id -- every row written before 0033, on
-- every 10 s poll. A partial index over exactly the rows the WHERE keeps lets
-- SQLite's min() optimisation read its first entry and stop.
--
-- SQLite uses a partial index only when it can prove the query's WHERE implies
-- the index's, so the query keeps the same `token_id IS NOT NULL` term; the
-- EXPLAIN QUERY PLAN test in
-- tests/unit/db/test_migration_0034_token_partial_index.py pins that.

CREATE INDEX IF NOT EXISTS idx_request_history_token_finished
    ON request_history(finished_at) WHERE token_id IS NOT NULL;
