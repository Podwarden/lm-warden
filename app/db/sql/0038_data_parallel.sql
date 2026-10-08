-- First-class data-parallel layout and replica-affinity routing (#286).
--
-- data_parallel_size: number of identical vLLM replicas. The invariant is
-- tensor_parallel_size * data_parallel_size == len(gpu_indices). Default 1
-- keeps every existing row single-replica.
--
-- dp_affinity_enabled: 0/1; when 1 and data_parallel_size > 1 the proxy pins a
-- conversation to one replica via the X-data-parallel-rank header.
--
-- dp_spill_threshold: per-replica in-flight count above which the proxy spills
-- a request to the least-loaded replica. NULL = auto (max_num_seqs / 4).
ALTER TABLE models ADD COLUMN data_parallel_size INTEGER NOT NULL DEFAULT 1;
ALTER TABLE models ADD COLUMN dp_affinity_enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE models ADD COLUMN dp_spill_threshold INTEGER;
