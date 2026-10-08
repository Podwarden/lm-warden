-- Claude Code model router (#287).
--
-- router_rules: operator rules mapping a requested model name (glob) to a
-- local model row. Keyed by models.id, deliberately WITHOUT a foreign key: a
-- deleted target must stay visible in the UI (and fall back / refuse at
-- request time) instead of silently cascading the rule away.
CREATE TABLE router_rules (
    id              TEXT PRIMARY KEY,
    position        INTEGER NOT NULL,
    pattern         TEXT NOT NULL,
    target_model_id TEXT NOT NULL,
    enabled         INTEGER NOT NULL DEFAULT 1,
    fallback        INTEGER NOT NULL DEFAULT 1,
    strip_thinking  INTEGER NOT NULL DEFAULT 1,
    min_max_tokens  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_router_rules_position ON router_rules(position);

-- anthropic_relay: 1 = this inference key may cause the warden to relay its
-- request (with the caller's own Anthropic credential) to Anthropic.
ALTER TABLE api_tokens ADD COLUMN anthropic_relay INTEGER NOT NULL DEFAULT 0;
