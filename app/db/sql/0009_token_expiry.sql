ALTER TABLE api_tokens ADD COLUMN expires_at TEXT NULL;
ALTER TABLE api_tokens ADD COLUMN rotated_at TEXT NULL;
ALTER TABLE api_tokens ADD COLUMN rotated_from TEXT NULL REFERENCES api_tokens(id) ON DELETE SET NULL;

UPDATE api_tokens
   SET expires_at = datetime(created_at, '+365 days')
 WHERE expires_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_tokens_expires_at ON api_tokens(expires_at);
