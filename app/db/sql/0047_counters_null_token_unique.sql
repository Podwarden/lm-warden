-- One counters row per (model, NULL token) (#298).
--
-- counters' PRIMARY KEY (model_id, token_id) does not cover the NULL token:
-- SQLite treats NULLs as distinct in a composite key, so any number of
-- (model, NULL) rows can coexist. pw-prod had two for one model (4 and 32
-- requests). Two paths could create the second one:
--   * deleting a key. token_id is REFERENCES api_tokens ON DELETE SET NULL and
--     foreign_keys is ON, so DELETE FROM api_tokens turned every (model, key)
--     row into a (model, NULL) row next to the one already there. This is the
--     likely source: the writers update every matching row, so two rows that
--     exist together grow together, and two far-apart counts mean one row
--     arrived later with its own history -- a deleted key's.
--   * the pre-#279 CountersRepo.increment, a SELECT then INSERT outside any
--     transaction on a per-request connection: two first requests for a model
--     could both see no row and both INSERT. That yields near-equal rows.
--
-- This migration merges the duplicates, adds a partial unique index so a
-- second (model, NULL) row is refused, and replaces the FK's SET NULL with a
-- trigger that folds a deleted key's rows into the model's NULL row (the SET
-- NULL would otherwise violate the index and fail the key's DELETE). A deleted
-- key's usage keeps counting toward the model as unattributed, as before.

-- ---- merge -------------------------------------------------------------------
-- Sum every (model, NULL) group into its lowest rowid, then drop the rest.
UPDATE counters SET
  requests = (SELECT SUM(c.requests) FROM counters c
              WHERE c.model_id = counters.model_id AND c.token_id IS NULL),
  prompt_tokens = (SELECT SUM(c.prompt_tokens) FROM counters c
                   WHERE c.model_id = counters.model_id AND c.token_id IS NULL),
  completion_tokens = (SELECT SUM(c.completion_tokens) FROM counters c
                       WHERE c.model_id = counters.model_id AND c.token_id IS NULL)
WHERE token_id IS NULL
  AND rowid = (SELECT MIN(c.rowid) FROM counters c
               WHERE c.model_id = counters.model_id AND c.token_id IS NULL);

DELETE FROM counters
WHERE token_id IS NULL
  AND rowid <> (SELECT MIN(c.rowid) FROM counters c
                WHERE c.model_id = counters.model_id AND c.token_id IS NULL);

-- ---- index -------------------------------------------------------------------
CREATE UNIQUE INDEX idx_counters_null_token_model ON counters(model_id) WHERE token_id IS NULL;

-- ---- key deletion --------------------------------------------------------------
-- BEFORE DELETE, so the key's rows are gone by the time the FK action runs and
-- its SET NULL has nothing left to touch. Plain UPDATE/DELETE rather than an
-- UPSERT, which needs a newer SQLite inside a trigger than the image may carry.
-- Order matters: fold into existing NULL rows first, then relabel the rows of
-- models that have none (no collision possible), then drop what was folded.
CREATE TRIGGER counters_fold_deleted_token BEFORE DELETE ON api_tokens
BEGIN
  UPDATE counters SET
    requests = requests + (SELECT t.requests FROM counters t
                           WHERE t.model_id = counters.model_id AND t.token_id = OLD.id),
    prompt_tokens = prompt_tokens + (SELECT t.prompt_tokens FROM counters t
                                     WHERE t.model_id = counters.model_id AND t.token_id = OLD.id),
    completion_tokens = completion_tokens + (SELECT t.completion_tokens FROM counters t
                                             WHERE t.model_id = counters.model_id AND t.token_id = OLD.id)
  WHERE token_id IS NULL
    AND model_id IN (SELECT model_id FROM counters WHERE token_id = OLD.id);
  UPDATE counters SET token_id = NULL
  WHERE token_id = OLD.id
    AND NOT EXISTS (SELECT 1 FROM counters n
                    WHERE n.model_id = counters.model_id AND n.token_id IS NULL);
  DELETE FROM counters WHERE token_id = OLD.id;
END;
