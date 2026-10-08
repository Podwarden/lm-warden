-- Migration 0011: persist user-set quantization and max_num_seqs on the
-- models row.
--
-- The model-load wizard lets the user pick a quantization mode and a
-- max_num_seqs cap; until now these were applied only at subprocess spawn
-- and never persisted, so re-opening the wizard lost the user's choice.
-- The benchmark-v2 sweep also drives both values via in-memory `overrides`
-- on cmd_builder (see app/runtime/cmd_builder.py) but MUST NOT touch the
-- models row when it does so — these columns exist only so the wizard can
-- save the user's last choice.
--
-- SQLite does not allow adding a NOT NULL column with no default to an
-- existing table without a rewrite; both columns are nullable.

ALTER TABLE models ADD COLUMN quantization TEXT;
ALTER TABLE models ADD COLUMN max_num_seqs INTEGER;
