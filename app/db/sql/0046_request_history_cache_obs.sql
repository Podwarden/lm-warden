-- Prompt-cache observation per finished request (app/cache_obs/, spec
-- docs/superpowers/specs/2026-10-07-unified-cache-visibility-design.md §3.4).
--
-- reusable_tokens: prompt tokens that matched a recently processed prefix ON
-- THE SERVED REPLICA (the potential R; approximate, bytes scaled to tokens).
-- reusable_tokens_fleet: best match on any replica. reusable_tokens_own: match
-- against this token's own earlier prompts only (own lens).
-- cache_outcome / cache_outcome_own: hit | partial | lost | misrouted |
-- diverged | cold (own never misrouted). diverged_at: message index where the
-- client rewrote its prefix (-1 = tools).
-- cached_source: engine | estimated. cached_ttft_est_tokens: the TTFT-based
-- estimate for engines that do not report cached tokens; cached_tokens stays
-- MEASURED only. NULL everywhere = unknown, never 0.
ALTER TABLE request_history ADD COLUMN reusable_tokens INTEGER;
ALTER TABLE request_history ADD COLUMN reusable_tokens_fleet INTEGER;
ALTER TABLE request_history ADD COLUMN reusable_tokens_own INTEGER;
ALTER TABLE request_history ADD COLUMN cache_outcome TEXT;
ALTER TABLE request_history ADD COLUMN cache_outcome_own TEXT;
ALTER TABLE request_history ADD COLUMN diverged_at INTEGER;
ALTER TABLE request_history ADD COLUMN cached_source TEXT;
ALTER TABLE request_history ADD COLUMN cached_ttft_est_tokens INTEGER;
