CREATE TABLE setup_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  step TEXT NOT NULL DEFAULT 'welcome',
  draft TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT INTO setup_state(id, step, draft) VALUES (1, 'welcome', '{}');
