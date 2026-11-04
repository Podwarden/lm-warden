-- Sensitive data guard (2026-10-29). No secret value is stored anywhere.
-- A key's guard: NULL = inherit the platform setting (settings.key = 'guard'); {"off": true}; {"override": {...}}.
ALTER TABLE api_tokens ADD COLUMN guard TEXT;
-- What the guard did on this request: {"m": mode, "b": blocked, "f": [[rule, count, new]], "r": restored}.
ALTER TABLE request_history ADD COLUMN guard TEXT;
-- The action log: one row per value acted on. ph = the placeholder's keyed hash (10 hex), never the value.
CREATE TABLE IF NOT EXISTS guard_events (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    token_id TEXT,
    request_id TEXT,
    sess_ref TEXT,
    action TEXT NOT NULL,
    rule_id TEXT,
    role TEXT,
    msg_index INTEGER,
    kind TEXT,
    length INTEGER,
    ph TEXT,
    relayed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_guard_events_token_ts ON guard_events(token_id, ts);
CREATE INDEX IF NOT EXISTS idx_guard_events_ts ON guard_events(ts);
