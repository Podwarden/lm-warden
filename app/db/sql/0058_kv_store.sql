-- KV-cache tiers (lmw_kv): 'off' = engine default, 'ram' = host RAM tier, 'ram+nvme' = RAM spilling to NVMe.
-- Changes the engine argv, so it is reload-only.
ALTER TABLE models ADD COLUMN kv_store TEXT NOT NULL DEFAULT 'off';
