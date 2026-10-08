-- Migration 0023: remove the ghost `hf_cache_dir` setting.
--
-- `hf_cache_dir` was seeded by 0010 and exposed as an editable runtime key in
-- the settings API + UI, but its DB value was NEVER read. The HF model-cache
-- path is derived entirely from the environment: `VW_HF_CACHE_DIR` →
-- `settings.hf_cache_dir` (config.py), which both the pull task
-- (`snapshot_download(cache_dir=...)`) and the engine subprocess
-- (`HF_HUB_CACHE`, see app/runtime/env_builder.py) consume. Editing the KV row
-- in the UI therefore did nothing — a misleading, inert control.
--
-- 0010's seed line is removed in the same change so fresh DBs never get the
-- row; this DELETE converges existing DBs (the migration runner is
-- filename-tracked, so already-applied 0010 is never re-run on an existing DB).
-- DELETE on a non-existent key is a harmless no-op, so this is idempotent.

DELETE FROM settings WHERE key = 'hf_cache_dir';
