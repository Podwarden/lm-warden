-- 0065_lens_versions_wontfix.sql
--
-- Prompt Lens analysis, stage 3 (spec 2026-11-23, sections 11.4 and 13): the
-- prompt version a request was sent with, and the rules a key marked "won't fix".
--
--   request_history.lens_version   the prompt version: "h:<name>" when the client
--                                  sent X-Prompt-Version (cleaned, 64 printable
--                                  characters at most), else 8 hex detected from
--                                  the stable system prompt lines and the toolset.
--                                  NULL when kvsig is off or nothing was computed.
--                                  A keyed hash or a client's own label, never text.
--   lens_wontfix.key_id            the key (api_tokens.id) that marked the rule; reads
--                                  follow the key's rotation chain
--   lens_wontfix.rule              a catalogue rule id ("C-01"); R-09 is never marked
--   lens_wontfix.version           '' for every prompt version, else the one version
--                                  the mark is limited to (NOT NULL: SQLite treats
--                                  NULLs in a primary key as distinct)
--   lens_wontfix.by                "key" (the key holder) or "admin"
--   lens_wontfix.at                epoch seconds
--
-- lens_session_scores.version (migration 0064) is filled by the scorer from the
-- session's newest request version.
--
-- Number: 0064 is the last on develop (checked 2026-11-26).
--
-- Rollback: nothing else reads these.
-- DROP TABLE lens_wontfix;
-- ALTER TABLE request_history DROP COLUMN lens_version;
ALTER TABLE request_history ADD COLUMN lens_version TEXT;

CREATE TABLE lens_wontfix (
  key_id TEXT NOT NULL,
  rule TEXT NOT NULL,
  version TEXT NOT NULL DEFAULT '',
  by TEXT NOT NULL,
  at REAL NOT NULL,
  PRIMARY KEY (key_id, rule, version)
);
