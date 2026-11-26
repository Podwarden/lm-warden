-- 0063_lens_fields.sql
--
-- Prompt Lens analysis, stage 1 (spec 2026-11-23, sections 4-8; v1 5.3). What
-- the session analysis reads after the ring and the lineage have forgotten a
-- request: codes, counts, sizes and positions only, never prompt, reply,
-- reasoning or tool text (A-3, A-10). Every column is nullable with no
-- default: NULL is "not computed" (kvsig off, a lens fault, a row from before
-- this migration), so old rows read as "not attributed", never as a cause.
-- Plain ADD COLUMNs: SQLite does not rewrite request_history for them.
--
-- Layer 1 (computed in the kvsig pass from the key's own lineage, so own-lens
-- by construction; misrouted stays in the existing fleet fields):
--   lens_break          'sys', 'tools', 'node:<i>' or NULL (the first broken part)
--   lens_cause          a code of app/lens/causes.py CAUSES, or NULL
--   lens_span_class     the class of the changed system prompt line (app/lens/probes.py
--                       SPAN_CLASSES); the line's text is read and dropped
--   lens_span_line      that line, 1-based
--   lens_lost_tokens    counterfactual reuse minus actual reuse, >= 0
--   lens_cf_tokens      the counterfactual reuse (the prefix shared with the parent
--                       had the broken part not changed)
--   lens_other_breaks   further broken parts after the first
-- Relays (app/router/relay_record.py):
--   cache_write_tokens  the upstream's cache_creation_input_tokens
--   cache_marks         cache_control marks counted in the relayed body
-- Layer 3 (filled by analysis Task 1.4; created here, one migration per stage):
--   lens_repeat_results tokens of tool results repeated within the prompt
--   max_tokens_req      the request's max_tokens (or max_completion_tokens)
--   reasoning_tokens    the reply's reasoning tokens
--   reply_chars         the reply's size in chars
--   calls_n             tool calls in the reply
--   calls_bad           JSON, at most 16 [k, hash8, code] of the non-ok calls
--                       (k: the call's index in the reply; null for leaked markup)
--   tools_lint          JSON, one int per offered tool
--
-- Number: 0062 (model_aliases) is the last on develop; 0063 is free (checked
-- 2026-11-25).
--
-- Rollback: inert when unread; SQLite >= 3.35 can
-- ALTER TABLE request_history DROP COLUMN lens_break;
-- ALTER TABLE request_history DROP COLUMN lens_cause;
-- ALTER TABLE request_history DROP COLUMN lens_span_class;
-- ALTER TABLE request_history DROP COLUMN lens_span_line;
-- ALTER TABLE request_history DROP COLUMN lens_lost_tokens;
-- ALTER TABLE request_history DROP COLUMN lens_cf_tokens;
-- ALTER TABLE request_history DROP COLUMN lens_other_breaks;
-- ALTER TABLE request_history DROP COLUMN cache_write_tokens;
-- ALTER TABLE request_history DROP COLUMN cache_marks;
-- ALTER TABLE request_history DROP COLUMN lens_repeat_results;
-- ALTER TABLE request_history DROP COLUMN max_tokens_req;
-- ALTER TABLE request_history DROP COLUMN reasoning_tokens;
-- ALTER TABLE request_history DROP COLUMN reply_chars;
-- ALTER TABLE request_history DROP COLUMN calls_n;
-- ALTER TABLE request_history DROP COLUMN calls_bad;
-- ALTER TABLE request_history DROP COLUMN tools_lint;
ALTER TABLE request_history ADD COLUMN lens_break TEXT;
ALTER TABLE request_history ADD COLUMN lens_cause TEXT;
ALTER TABLE request_history ADD COLUMN lens_span_class TEXT;
ALTER TABLE request_history ADD COLUMN lens_span_line INTEGER;
ALTER TABLE request_history ADD COLUMN lens_lost_tokens INTEGER;
ALTER TABLE request_history ADD COLUMN lens_cf_tokens INTEGER;
ALTER TABLE request_history ADD COLUMN lens_other_breaks INTEGER;
ALTER TABLE request_history ADD COLUMN cache_write_tokens INTEGER;
ALTER TABLE request_history ADD COLUMN cache_marks INTEGER;
ALTER TABLE request_history ADD COLUMN lens_repeat_results INTEGER;
ALTER TABLE request_history ADD COLUMN max_tokens_req INTEGER;
ALTER TABLE request_history ADD COLUMN reasoning_tokens INTEGER;
ALTER TABLE request_history ADD COLUMN reply_chars INTEGER;
ALTER TABLE request_history ADD COLUMN calls_n INTEGER;
ALTER TABLE request_history ADD COLUMN calls_bad TEXT;
ALTER TABLE request_history ADD COLUMN tools_lint TEXT;
