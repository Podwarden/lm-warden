-- Inputs for the learned prefill model (app/stats/prefill_model.py).
--
-- dp_rank: the data-parallel replica the request was routed to (NULL when
-- unrouted or single-replica).
-- inflight_same_rank_at_start: other requests already admitted on the same
-- model and replica when this one was admitted. 0 means it prefilled alone, the
-- only case where (prompt - cached) / ttft measures the engine rather than its
-- queue. NULL on rows from before this migration.
-- gap_s: seconds since the same conversation's previous request finished (the
-- prefix-memory entry's age), NULL when there was no usable previous turn. Lets
-- the model learn how fast the engine evicts an idle session's cache.
ALTER TABLE request_history ADD COLUMN dp_rank INTEGER;
ALTER TABLE request_history ADD COLUMN inflight_same_rank_at_start INTEGER;
ALTER TABLE request_history ADD COLUMN gap_s REAL;
