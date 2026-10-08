-- Stress runs: measured operating limits per model, per configuration.
--
-- A separate table rather than columns on `models`, for two concrete reasons
-- specific to this codebase:
--
--   1. `_PATCHABLE_MODEL_FIELDS` (app/settings/routes_api.py) is DERIVED from
--      ModelRow's fields minus a blocklist, so any new `models` column becomes
--      operator-patchable the moment it exists. Machine-written measurements
--      must never be hand-editable — an operator who edits one has produced a
--      number with `provenance: measured` that was not measured.
--   2. tests/unit/models/test_model_serialisation.py fails the build unless
--      every new `models` column is classified into one of three field sets.
--
-- `fingerprint` is the identity of what was measured (design §7.3). A run is
-- only valid for the exact model + load config + engine build + hardware +
-- co-resident set it ran against, so results are keyed on it rather than on
-- model_id alone: the same model at two context sizes yields two rows, both
-- true, neither superseding the other.

CREATE TABLE stress_runs (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  fingerprint TEXT NOT NULL,

  mode TEXT NOT NULL
    CHECK (mode IN ('conservative','quick','thorough')),

  -- `interrupted` is written at boot to any row still 'running', because the
  -- run is in-process and dies with the warden. Such a row is never published:
  -- a partial search has a bracket but no confirmation, and publishing its
  -- lower bound would understate the limit with full confidence.
  status TEXT NOT NULL DEFAULT 'running'
    CHECK (status IN ('running','completed','aborted','interrupted','failed_unrecovered')),

  -- Why the run stopped short, when it did. Distinct from `status` because a
  -- completed run can still be truncated (crash budget, a wedged engine).
  truncated_by TEXT,

  -- Set when the measurement conditions were violated but the run continued
  -- under `force`. A tainted run is recorded and never published.
  traffic_observed INTEGER NOT NULL DEFAULT 0,

  -- Phase D verdict. A non-monotone axis publishes nothing: bisection assumes
  -- an upward-closed failure region, and without that assumption the confirmed
  -- value can be several times too low while carrying full confidence.
  non_monotone INTEGER NOT NULL DEFAULT 0,

  -- Counts reloads of this model since the run finished. The fingerprint is
  -- unchanged by an identical-config reload, so a measurement survives one --
  -- and whether that is correct is genuinely unknown (design §10.1). This
  -- column is what makes the question answerable later instead of assumed now.
  loads_since_measurement INTEGER NOT NULL DEFAULT 0,

  -- The measured limits, as a JSON object of limit-name -> limit-object.
  -- Each carries value/provenance/confidence/limited_by/raw_confirmed/
  -- first_observed_failure and the oracle parameters, because a probabilistic
  -- edge cannot be published as a scalar.
  limits TEXT,

  -- Every probe outcome, for forensics and for re-deriving a limit without
  -- re-running the GPU work.
  observations TEXT,

  -- A different configuration that measured better (design §6.2). Advisory
  -- only, and carries its own fingerprint so it is unmistakable that it
  -- describes a config the model is NOT currently running.
  recommended_config TEXT,

  seed INTEGER NOT NULL,
  probe_suite_hash TEXT NOT NULL,
  algorithm_version INTEGER NOT NULL,

  started_at TEXT NOT NULL DEFAULT (datetime('now')),
  finished_at TEXT,
  last_error TEXT
);

-- Serving a client the current measurement is a lookup by (model, fingerprint)
-- ordered by recency; the cooldown check is the same query.
CREATE INDEX idx_stress_runs_model_fp ON stress_runs(model_id, fingerprint, started_at);
CREATE INDEX idx_stress_runs_model ON stress_runs(model_id, started_at);
