-- The own-lens divergence point (#291). diverged_at is matched against every
-- key's earlier prompts, so its "later messages were already seen" evidence
-- can come from another key's traffic; the own lens must not read it.
-- diverged_at_own is the same index, matched against this key's prompts only
-- (spec 2026-10-07 §3.3 already classified cache_outcome_own with it; now it
-- is kept). NULL = no divergence evidence, or a row written before 0048.
ALTER TABLE request_history ADD COLUMN diverged_at_own INTEGER;
