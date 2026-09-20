-- Admin tokens and their audit trail
-- (docs/superpowers/specs/2026-09-19-admin-tokens-design.md).
--
-- An admin token is an api_tokens row with scope = 'admin' and a `vwa_`
-- secret. It calls the control API (/api/*) as the user that issued it, so
-- the row records that user in created_by. Inference rows (every row before
-- this migration) keep created_by NULL: there is nothing to backfill and they
-- never act as a user.
--
-- idx_api_tokens_scope lets the admin list and the inference list's count
-- arithmetic seek the (few) admin rows. The inference page CTE must NOT use it
-- -- it keeps walking the 0035 sort indexes (app/db/repos/tokens.py filters
-- with an unindexable `+t.scope`).
--
-- admin_audit: one row per request an admin token made, written shortly after
-- the response -- queued there, committed in batches by a background writer
-- (app/auth/admin_audit.py). `path` is the route TEMPLATE, never the
-- concrete URL. ts is epoch seconds. No foreign key to api_tokens: the trail
-- must outlive whatever happens to the row. (token_id, ts DESC) serves one
-- token's newest-first page; (ts) serves the pruner's age cut and row cap
-- (90 days / 200k rows, app/runtime/stats_pruner.py). client_ip is the
-- proxy-reported address (X-Forwarded-For aware, like request_history, so it
-- can be forged); peer_ip is the raw socket peer next to it, forgeable by
-- neither client nor proxy misconfiguration.

ALTER TABLE api_tokens ADD COLUMN created_by TEXT;

CREATE INDEX IF NOT EXISTS idx_api_tokens_scope ON api_tokens(scope);

CREATE TABLE IF NOT EXISTS admin_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    token_id TEXT NOT NULL,
    username TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    client_ip TEXT,
    peer_ip TEXT
);

CREATE INDEX IF NOT EXISTS idx_admin_audit_token_ts ON admin_audit(token_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_admin_audit_ts ON admin_audit(ts);
