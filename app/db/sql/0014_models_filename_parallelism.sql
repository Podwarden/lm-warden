-- Migration 0014: per-file download + parallelism strategy + KV batch sizing
-- on the models row (#85).
--
-- ``filename`` narrows the pull to one weights file (plus config/tokenizer)
-- when the user picked a specific quant variant in the Add Model wizard.
-- NULL preserves the legacy "pull the whole repo" path.
--
-- ``parallelism_strategy`` records the wizard's tp/pp/auto choice. Runtime
-- wiring (which vLLM flag we set) is downstream in #82.5; persisting it now
-- so the wizard can round-trip the user's choice.
--
-- ``max_batch_size`` feeds the KV-reserve math in app/models/fit.py.
--
-- SQLite does not allow adding a NOT NULL column with no default to an
-- existing table without a rewrite. ``filename`` is nullable (legacy rows
-- and "whole repo" pulls both want NULL); the other two have defaults that
-- match the Pydantic ModelCreate defaults so existing rows behave like the
-- pre-#85 wizard did.

ALTER TABLE models ADD COLUMN filename TEXT;
ALTER TABLE models ADD COLUMN parallelism_strategy TEXT NOT NULL DEFAULT 'auto';
ALTER TABLE models ADD COLUMN max_batch_size INTEGER NOT NULL DEFAULT 1;
