-- chat2: per-user persisted chats with linear messages, image attachments
-- stored on disk and refcounted by sha256, per-user defaults, and the
-- billing foundation (append-only usage ledger + rate cards). This is the
-- first per-user data in the app: users.id is INTEGER, and every chat2
-- query is scoped by that integer (the JWT subject is the username and is
-- resolved once per request by app.chat2.identity.current_user_id).
-- usage_ledger deliberately has no FK to chats/messages: deleting a chat
-- must never erase billable usage. Context-full state is derived at read
-- time (last usage + current settings + current model window) and is not
-- stored. See docs/superpowers/specs/2026-08-23-chat2-design.md §3.
ALTER TABLE models ADD COLUMN supports_tools INTEGER;
ALTER TABLE models ADD COLUMN supports_vision INTEGER;
ALTER TABLE models ADD COLUMN supports_reasoning INTEGER;

CREATE TABLE chats (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  org_id TEXT,
  title TEXT NOT NULL,
  title_source TEXT NOT NULL DEFAULT 'auto',
  model TEXT,
  settings_json TEXT NOT NULL,
  forked_from_chat_id TEXT,
  forked_at_seq INTEGER,
  cost_micros_total INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_message_at TEXT
);
CREATE INDEX idx_chats_user ON chats(user_id, last_message_at DESC);

CREATE TABLE messages (
  id TEXT PRIMARY KEY,
  chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  role TEXT NOT NULL,
  parts_json TEXT NOT NULL,
  model TEXT,
  settings_snapshot_json TEXT NOT NULL,
  usage_json TEXT,
  finish_reason TEXT,
  error_json TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (chat_id, seq)
);

CREATE TABLE attachments (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  org_id TEXT,
  chat_id TEXT REFERENCES chats(id) ON DELETE CASCADE,
  message_id TEXT,
  kind TEXT NOT NULL,
  mime TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  width INTEGER,
  height INTEGER,
  created_at TEXT NOT NULL,
  evicted_at TEXT
);
CREATE INDEX idx_attachments_chat ON attachments(chat_id);
CREATE INDEX idx_attachments_sha ON attachments(user_id, sha256);

CREATE TABLE user_chat_defaults (
  user_id INTEGER PRIMARY KEY REFERENCES users(id),
  model TEXT,
  settings_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE rate_cards (
  id TEXT PRIMARY KEY,
  org_id TEXT,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  input_per_m_micros INTEGER NOT NULL,
  output_per_m_micros INTEGER NOT NULL,
  reasoning_per_m_micros INTEGER,
  cache_read_per_m_micros INTEGER,
  cache_write_per_m_micros INTEGER,
  markup_pct INTEGER NOT NULL DEFAULT 0,
  currency TEXT NOT NULL,
  effective_from TEXT NOT NULL,
  UNIQUE (org_id, provider, model, effective_from)
);
CREATE INDEX idx_rate_cards_lookup ON rate_cards(provider, model, org_id, effective_from DESC);

CREATE TABLE usage_ledger (
  id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL UNIQUE,
  provider_request_id TEXT,
  user_id INTEGER NOT NULL,
  org_id TEXT,
  chat_id TEXT,
  message_id TEXT,
  purpose TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL,
  completion_tokens INTEGER NOT NULL,
  reasoning_tokens INTEGER,
  cache_read_tokens INTEGER,
  cache_write_tokens INTEGER,
  estimated INTEGER NOT NULL DEFAULT 0,
  outcome TEXT NOT NULL,
  rate_card_id TEXT,
  cost_micros INTEGER,
  currency TEXT,
  cost_status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_ledger_user ON usage_ledger(user_id, created_at);
CREATE INDEX idx_ledger_org ON usage_ledger(org_id, created_at);
CREATE INDEX idx_ledger_chat ON usage_ledger(chat_id);
