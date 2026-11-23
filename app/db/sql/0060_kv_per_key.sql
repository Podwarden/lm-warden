-- KV-cache tiers plan 2: per-key sharing and eviction.
--
-- Every statement is an ADD COLUMN with a constant default, a CREATE, or an UPDATE of a
-- small table (api_tokens). SQLite stores a constant default in the schema, so the ADD
-- COLUMNs do not rewrite request_history, the one large table.
--
-- api_tokens: how a key shares the KV cache.
--   kv_isolation     'sharing' (default for every key, new and existing) or 'isolated'.
--   kv_tenant_id     the key family's stable id; rotation copies it, so a successor keeps
--                    its cache. Never empty: the tenant tag derivation fails closed on ''.
--   kv_epoch         bumped by a reset; part of an isolated key's scope.
--   kv_max_pct_ram / kv_max_pct_nvme   per-key caps on each tier, NULL = no cap.
ALTER TABLE api_tokens ADD COLUMN kv_isolation TEXT NOT NULL DEFAULT 'sharing'
    CHECK (kv_isolation IN ('sharing', 'isolated'));
ALTER TABLE api_tokens ADD COLUMN kv_tenant_id TEXT NOT NULL DEFAULT '';
ALTER TABLE api_tokens ADD COLUMN kv_epoch INTEGER NOT NULL DEFAULT 0;
ALTER TABLE api_tokens ADD COLUMN kv_max_pct_ram INTEGER;
ALTER TABLE api_tokens ADD COLUMN kv_max_pct_nvme INTEGER;

-- Backfill, step 1: a rotation chain is one family, keyed by its root's id. The root is the
-- row with no predecessor (rotated_from NULL, which is also what ON DELETE SET NULL leaves
-- an orphaned successor) or one whose predecessor row is gone. So a successor in its
-- predecessor's grace window shares its predecessor's tenant from the first request.
WITH RECURSIVE chain(id, root) AS (
    SELECT id, id FROM api_tokens
    WHERE rotated_from IS NULL OR rotated_from NOT IN (SELECT id FROM api_tokens)
    UNION ALL
    SELECT t.id, c.root FROM api_tokens t JOIN chain c ON t.rotated_from = c.id
)
UPDATE api_tokens SET kv_tenant_id = (SELECT root FROM chain WHERE chain.id = api_tokens.id)
WHERE kv_tenant_id = '' AND id IN (SELECT id FROM chain);

-- Backfill, step 2: any row step 1 did not reach is its own family.
UPDATE api_tokens SET kv_tenant_id = id WHERE kv_tenant_id = '';

-- A row inserted without a tenant id (any insert path, not only TokenRepo.create) takes
-- its own id, so no row ever holds ''. Rotation inserts the predecessor's id and is untouched.
CREATE TRIGGER api_tokens_kv_tenant_default
AFTER INSERT ON api_tokens
FOR EACH ROW
WHEN NEW.kv_tenant_id = ''
BEGIN
    UPDATE api_tokens SET kv_tenant_id = NEW.id WHERE id = NEW.id;
END;

-- No later code path may blank a tenant id: the tenant tag derivation fails closed on ''.
CREATE TRIGGER api_tokens_kv_tenant_not_blank
BEFORE UPDATE OF kv_tenant_id ON api_tokens
FOR EACH ROW
WHEN NEW.kv_tenant_id = ''
BEGIN
    SELECT RAISE(ABORT, 'kv_tenant_id must not be empty');
END;

-- models: the model's part of each tier under the fair eviction policy.
ALTER TABLE models ADD COLUMN kv_store_weight INTEGER NOT NULL DEFAULT 1;

-- request_history: whether the request ran isolated, and for an isolated request the prompt
-- tokens it recomputed that a shared prefix would have served. NULL on rows written before
-- this migration.
ALTER TABLE request_history ADD COLUMN kv_isolated INTEGER;
ALTER TABLE request_history ADD COLUMN kv_iso_cost_tokens INTEGER;

-- Scope tags retired by a reset, a revoke, a delete or an admin switch back to sharing.
-- The api re-sends them to the store on every connect; the store refuses their blocks.
-- scope_tag: lower-case hex of the 16-byte tag (32 chars)
CREATE TABLE kv_retired_scopes (
    scope_tag TEXT PRIMARY KEY, kv_tenant_id TEXT NOT NULL, retired_at TEXT NOT NULL);
