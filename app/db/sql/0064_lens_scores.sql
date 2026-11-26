-- 0064_lens_scores.sql
--
-- Prompt Lens analysis, stage 2 (spec 2026-11-23, sections 10 and 12): the score
-- and the findings of a session, written once it has been quiet for a while
-- (lens_quiet_s, default 10 minutes) and rewritten in place when it resumes and
-- goes quiet again, so a session is one row, never two.
--
-- Text-free (A-10): numbers, rule ids, cause codes, keyed tool hashes and turn
-- numbers only. The score's per-rule terms are stored (``terms``), so a rule
-- marked "won't fix" (stage 3) is left out on read, without a rescan.
--
--   lens_session_scores.key_id         the key the session belongs to (api_tokens.id)
--   lens_session_scores.sess_ref       the sessions page's session id ("s_<hex>")
--   lens_session_scores.started/last_at   first request start, last request end
--   lens_session_scores.model/route/harness   labels ("route": "local" or "relay")
--   lens_session_scores.version        prompt version (stage 3), NULL until then
--   lens_session_scores.e/c/r/overall  0-100, NULL when the layer was not scored
--   lens_session_scores.ranges         JSON {"E": [lo, hi], ...}, 10th-90th percentile
--   lens_session_scores.confidence     JSON {"E": "high", ...}
--   lens_session_scores.coverage       JSON {"E": 0.97, ...}
--   lens_session_scores.terms          JSON per layer: terms, inputs, reason, turns
--   lens_session_scores.rules          JSON: fired rule ids by layer
--   lens_session_scores.score_version  app/lens/scoring.py SCORE_VERSION
--   lens_findings                      one row per fired rule (and tool), as Finding.to_json()
--
-- Rows are pruned with request_history (app/stats/request_history.py prune): a
-- session row goes once its last request is older than the retention cutoff.
--
-- Number: 0063 is the last on develop (checked 2026-11-26).
--
-- Rollback: nothing else reads these tables.
-- DROP TABLE lens_findings;
-- DROP TABLE lens_session_scores;
CREATE TABLE lens_session_scores (
  key_id TEXT NOT NULL,
  sess_ref TEXT NOT NULL,
  started REAL NOT NULL,
  last_at REAL NOT NULL,
  model TEXT NOT NULL,
  route TEXT NOT NULL,
  harness TEXT NOT NULL,
  version TEXT,
  turns INTEGER NOT NULL,
  e INTEGER,
  c INTEGER,
  r INTEGER,
  overall INTEGER,
  ranges TEXT NOT NULL,
  confidence TEXT NOT NULL,
  coverage TEXT NOT NULL,
  terms TEXT NOT NULL,
  rules TEXT NOT NULL,
  score_version INTEGER NOT NULL,
  PRIMARY KEY (key_id, sess_ref)
);
CREATE INDEX idx_lens_scores_key_last ON lens_session_scores(key_id, last_at DESC);
CREATE INDEX idx_lens_scores_last ON lens_session_scores(last_at);

CREATE TABLE lens_findings (
  key_id TEXT NOT NULL,
  sess_ref TEXT NOT NULL,
  rule TEXT NOT NULL,
  turns TEXT NOT NULL,
  counts TEXT NOT NULL,
  tool TEXT NOT NULL DEFAULT '',
  cause TEXT,
  saving TEXT NOT NULL,
  confidence TEXT NOT NULL,
  PRIMARY KEY (key_id, sess_ref, rule, tool),
  FOREIGN KEY (key_id, sess_ref) REFERENCES lens_session_scores(key_id, sess_ref) ON DELETE CASCADE
);
