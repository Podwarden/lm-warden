-- Sign-in sessions (verified finding #2): every admin login and every key sign-in gets a row, and its tokens
-- carry the id. Logout ends one; a password or username change ends the user's; rotating or revoking a key ends
-- its holders'. kind: 'admin' (subject = username) | 'key' (subject = api_tokens.id).
CREATE TABLE IF NOT EXISTS auth_sessions (
    sid TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    created_at REAL NOT NULL,
    revoked_at REAL
);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_subject ON auth_sessions(kind, subject);
