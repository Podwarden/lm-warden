-- Migration 0015: optional --hf-config-path / --tokenizer plumbing for GGUF
-- repos that omit `config.json` (#106).
--
-- vLLM 0.20.0 rejects ``repo_id:quant_type`` model args when the quantized
-- GGUF repo lacks `config.json` (common for unsloth republishes). The recipe
-- vLLM prints points at `--hf-config-path <original_repo>`. We also expose
-- `--tokenizer <repo>` because the same upstream-vs-quant split typically
-- holds for the tokenizer.
--
-- Both columns are nullable — non-GGUF rows, and GGUF rows that ship
-- `config.json` natively, never need to set them. The launcher omits the
-- flags when the column is NULL (mirrors the v17.17 #100 cmd_builder
-- pattern). No default needed: SQLite ALTER TABLE ADD COLUMN of a nullable
-- TEXT is a metadata-only operation, existing rows decode to None.

ALTER TABLE models ADD COLUMN hf_config_repo TEXT;
ALTER TABLE models ADD COLUMN tokenizer_repo TEXT;
