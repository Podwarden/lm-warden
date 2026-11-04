-- Sessions pages (docs/superpowers/specs/2026-10-25-sessions-pages-design.md). Ids, sizes and counts only.
-- The lineage session actually used (harness key or cobs2 fallback), the conversation (loose head + first
-- loose node), the request's shape (sizes/counts JSON, <= 200 B), the request that produced its parent node,
-- a keyed lookup ref for (token, session), and the digest of the offered toolset (see cobs2_toolsets).
ALTER TABLE request_history ADD COLUMN cobs2_session TEXT;
ALTER TABLE request_history ADD COLUMN cobs2_conv TEXT;
ALTER TABLE request_history ADD COLUMN cobs2_shape TEXT;
ALTER TABLE request_history ADD COLUMN cobs2_parent_req TEXT;
ALTER TABLE request_history ADD COLUMN cobs2_sess_ref TEXT;
ALTER TABLE request_history ADD COLUMN cobs2_toolset TEXT;
-- 1 for a request relayed upstream (router mode); NULL for a local one.
ALTER TABLE request_history ADD COLUMN relayed INTEGER;
ALTER TABLE request_history ADD COLUMN upstream TEXT;
-- Reserved for the cache tiers (plan 4, multi-node): NULL until filled.
ALTER TABLE request_history ADD COLUMN served_gpu_tokens INTEGER;
ALTER TABLE request_history ADD COLUMN served_ram_tokens INTEGER;
ALTER TABLE request_history ADD COLUMN served_nvme_tokens INTEGER;
ALTER TABLE request_history ADD COLUMN served_remote_tokens INTEGER;
ALTER TABLE request_history ADD COLUMN load_s REAL;
ALTER TABLE request_history ADD COLUMN node_id TEXT;

CREATE INDEX idx_request_history_cobs2_session ON request_history(token_id, cobs2_session, finished_at);
CREATE INDEX idx_request_history_sess_ref ON request_history(cobs2_sess_ref, finished_at);
CREATE INDEX idx_request_history_parent_session ON request_history(token_id, parent_session_key)
    WHERE parent_session_key IS NOT NULL;

-- An offered toolset: JSON list of [hash8, family, chars]. No names, descriptions or schemas.
CREATE TABLE cobs2_toolsets (
    digest TEXT PRIMARY KEY,
    tools TEXT NOT NULL,
    last_seen REAL NOT NULL
);

-- Plain tool NAMES per key (owner decision 2026-10-25): names only, never descriptions or schemas.
-- Shown only to that key's holder and the admin; pruned with the history retention.
CREATE TABLE session_tools (
    token_id TEXT NOT NULL,
    tool_hmac TEXT NOT NULL,
    name TEXT NOT NULL,
    family TEXT NOT NULL,
    size INTEGER NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    PRIMARY KEY (token_id, tool_hmac)
);
CREATE INDEX idx_session_tools_last_seen ON session_tools(last_seen);
