-- 0061_prompt_lens_keep_captures.sql
--
-- Prompt Lens "keep captures" (owner decision 2026-11-22, rulings G-R4 and
-- G-R12). An admin-only per-key switch. While Prompt Lens is on, a key with it on
-- has its last finished requests kept on disk -- the structured capture (the
-- request_start, kvsig and request_end events as the admin sees them) and the
-- whole sections (thinking, reply, tool calls, the wide prompt capture), each
-- within the in-memory store's own budgets -- so they outlive the ring and a
-- restart. At most 50 rows and 32,000,000 chars per key, oldest dropped first.
-- Key holders only ever read them through the key-holder view.
--
-- api_tokens.keep_captures: 0 = off; any other value = on, and it is the
-- switch's generation (a random number set when it is turned on). A capture is
-- written only while its key still has the generation it started under, so a
-- switch turned off and on again never brings an older capture back.
--
-- Rows go with the key (ON DELETE CASCADE; TokenRepo.delete also deletes those
-- of the keys it was rotated from) and are deleted, for the whole rotation
-- lineage, when the switch is turned off. Content is never logged.
--
-- Each section is its own column, so a section read (and a ?from= refetch)
-- reads that section only, and a text read only from its offset (substr).
--
-- Number: 0059 (godmode_self_view) is in production and feat/kv-tiers-plan2
-- takes 0060, so this is 0061.
--
-- Rollback: inert when every key has 0; SQLite >= 3.35 can
-- ALTER TABLE api_tokens DROP COLUMN keep_captures; DROP TABLE prompt_lens_kept;
-- DROP TABLE prompt_lens_switch_audit;
ALTER TABLE api_tokens ADD COLUMN keep_captures INTEGER NOT NULL DEFAULT 0;

CREATE TABLE prompt_lens_kept (
  req_id TEXT PRIMARY KEY,
  token_id TEXT NOT NULL REFERENCES api_tokens(id) ON DELETE CASCADE,
  request_id TEXT,
  ended_ts REAL NOT NULL,
  -- The request_end's hub seq (the admin's section view carries it).
  end_seq INTEGER NOT NULL DEFAULT 0,
  -- Every stored text together, in chars: the per-key 32M cap sums it.
  chars INTEGER NOT NULL,
  -- What a list shows, as JSON: model, times, finish reason, token counts,
  -- and each section's size.
  summary TEXT NOT NULL,
  -- [request_start, kvsig, request_end] as JSON, admin form (no prompt_parts:
  -- the wide prompt below replaces it).
  events TEXT NOT NULL,
  thinking TEXT NOT NULL DEFAULT '',
  thinking_dropped INTEGER NOT NULL DEFAULT 0,
  reply TEXT NOT NULL DEFAULT '',
  reply_dropped INTEGER NOT NULL DEFAULT 0,
  -- {"calls": [...], "dropped_chars": n, "dropped_calls": n} as JSON.
  tool_calls TEXT NOT NULL DEFAULT '{}',
  -- The wide structured prompt capture (PromptParts) as JSON, or 'null'.
  prompt TEXT NOT NULL DEFAULT 'null'
);

CREATE INDEX idx_prompt_lens_kept_token ON prompt_lens_kept(token_id, ended_ts);
CREATE INDEX idx_prompt_lens_kept_request ON prompt_lens_kept(request_id);

-- Every save of the Prompt Lens switch (Settings, runtime key
-- prompt_lens_enabled), including "Use server default": who, when, through
-- what, and the value and source in effect before and after. The admin-token
-- trail (admin_audit, 0037) records requests, not values, and never a
-- session's, so this is its own small table. Not pruned: one row per
-- deliberate save.
-- Rollback: DROP TABLE prompt_lens_switch_audit;
CREATE TABLE prompt_lens_switch_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  actor TEXT NOT NULL,
  -- "session" or "admin token <id>".
  via TEXT NOT NULL,
  old_enabled INTEGER NOT NULL,
  new_enabled INTEGER NOT NULL,
  -- "setting", "env" or "default" (app/proxy/prompt_lens_switch.py).
  old_source TEXT NOT NULL,
  new_source TEXT NOT NULL,
  env_var TEXT
);
