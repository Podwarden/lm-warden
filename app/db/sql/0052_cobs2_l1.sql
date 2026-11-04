-- cobs2 L1 (spec §6): content-derived conversation identity, keyed with the network key. Hex ids only.
ALTER TABLE request_history ADD COLUMN cobs2_head TEXT;
ALTER TABLE request_history ADD COLUMN cobs2_node TEXT;          -- the request's last body node (exact)
ALTER TABLE request_history ADD COLUMN cobs2_parent_node TEXT;   -- what it continued / forked / compacted from
ALTER TABLE request_history ADD COLUMN cobs2_edge TEXT;          -- new|continue|fork|compacted_from|subagent
ALTER TABLE request_history ADD COLUMN cobs2_reply_msg TEXT;     -- the generated reply's msg id (exact)
ALTER TABLE request_history ADD COLUMN cobs2_compaction_request INTEGER;  -- 1 when this request asked for a summary
-- compacted|harness_rewrite: the cobs2 cause of a relabelled miss, or of a hit/partial whose prefix broke
-- (app/cache_obs/classify.py::divergence_cause); NULL when cobs2 explains nothing. _own: the own lens.
ALTER TABLE request_history ADD COLUMN cobs2_divergence_cause TEXT;
ALTER TABLE request_history ADD COLUMN cobs2_divergence_cause_own TEXT;
