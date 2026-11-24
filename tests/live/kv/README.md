# tests/live/kv: KV-cache tier tests on a GPU

These tests drive a running LM Warden whose model has `kv_store` set to `ram+nvme`.
They prove on real hardware what the unit tests cannot: the restored KV gives the
same tokens as a vanilla run, a restore is much faster than recomputing, and the
store failing never breaks a request.

Three files. `test_kv_live.py` is step 1 (tiers, parity, fail-open).
`test_kv_isolation_live.py` is Plan 2 (per-key isolation, two rows on one namespace,
the fair policy under a flood). `test_kv_disk_only_live.py` is the disk only mode
(`kv_store` `nvme`: blocks live on disk, RAM holds them only while they move).

Step 1's and Plan 2's files run on the tier in `VW_KV_LIVE_TIER` (`ram+nvme` by
default). With `nvme` a check that expects RAM tokens accepts store tokens from RAM or
NVMe, and two checks that need a RAM tier skip with that reason: G12 (it needs a `ram`
row beside the tier under test) and the fair flood smoke (it measures the RAM tier's
share).

They are opt-in. The `live` marker keeps them out of a plain `pytest` run, and a
missing env var skips the whole directory with the list of what is missing.

## What the run does to the instance

The run takes about 20 to 30 minutes. It unloads and loads the model several times,
changes its `kv_store` and its revision, and puts both back at the end. It mints a
short-lived inference key and revokes it at the end. Do not run it on an instance
that serves real traffic.

In order, the tests share one run:

1. **Vanilla pass.** `kv_store` off, fresh engine, one greedy 64-token completion
   per prompt size (2,048, 20,000 and 50,000 seeded random token ids). This gives
   the reference tokens and the cold TTFT.
2. **Save pass.** `kv_store` `ram+nvme`, fresh engine, the same prompts. The
   connector writes every whole block to RAM and NVMe.
3. **Restore pass.** The engine is restarted, so vLLM's own GPU prefix cache is
   empty, and the same prompts are sent again.

Then:

| Test | Checks |
|---|---|
| `test_kv_transfer_config_asks_for_recompute` | the engine argv has the connector and `kv_load_failure_policy: recompute` |
| `test_parity_after_restore` | restore tokens equal vanilla tokens, or differ first at a bf16 near-tie (top-2 logprob margin under `VW_KV_LIVE_TIE_MARGIN`), the rule the spike used |
| `test_restore_after_engine_restart` | at 20k+ tokens, history shows RAM or NVMe tokens and TTFT is at least 5x lower than vanilla cold |
| `test_cached_tokens_include_connector_tokens` | vLLM's `cached_tokens` includes the connector's tokens, and `gpu + ram + nvme == cached` |
| `test_fail_open_when_daemon_killed_mid_load` | the store daemon is killed while the 50k prompt loads; the request completes with vanilla tokens, the engine keeps serving and logs no error |
| `test_nvme_tier_serves_after_daemon_restart` | a fresh daemon has an empty RAM pool; the prompt loads from NVMe with vanilla tokens |
| `test_failed_load_recomputes` | the NVMe records are damaged behind a fresh daemon's back; the lookup still matches, the load fails, vLLM recomputes and the output is vanilla's |
| `test_old_namespace_never_loads` | a fresh prompt restores under the original revision (control); after the revision changes, nothing stored under the old one loads |

Every pass salts its prompts with a fresh run id, so blocks left on NVMe by an
earlier run never count.

### Plan 2: `test_kv_isolation_live.py`

It runs before `test_kv_live.py` (file order) and takes about 40 to 60 minutes. It mints
keys A (isolated), B (sharing), C (isolated) and D (sharing, switched during G11), turns on
debug capture for them (6 hours), and deletes them at the end. It sets `kv_evict_policy`
to `fair` for one test and back to what it was.

| Test | Checks |
|---|---|
| `test_g7_isolation_each_tier` | A sends a prompt with a client salt (8k in the GPU case, so all six prompts fit the GPU prefix cache; 20k in the RAM and NVMe cases). B (no salt, A's client salt, a reserved `lmw1:` value: A's own derived salt when the run can compute it) and C (no salt, A's client salt) send it too: no cached token and no tier token for any of them. A's repeat hits. Three times: A's blocks on the GPU; after an engine restart (RAM path); after a daemon restart with RAM empty and an engine restart (NVMe path) |
| `test_g8_gpu_isolation_without_daemon` | the GPU case of G7 with `kv_store` off and no daemon process (checked before and after) |
| `test_g11_parity_mixed_keys` | greedy parity with vanilla (kv_store off) for A's and B's save and restore, and for key D across sharing, a switch to isolated (that turn must reuse nothing), a restore from its isolated blocks and a switch back to sharing |
| `test_g12_two_rows_one_namespace` | two rows on a small model's weights, one `ram+nvme`, one `ram`. Control: the `ram+nvme` row alone writes segment files. After the `ram` row loads too, prompts to both rows create or grow no segment file under that namespace. Second control: after the `ram` row unloads, the `ram+nvme` row writes again |
| `test_fair_flood_smoke` | `kv_evict_policy` `fair`. A flooder sends 10 times its share of the RAM tier (the share is the tier over the 2 keys of the test, Task 15 N1), then the engine restarts and the victim's session loads from RAM or NVMe |
| `test_g7c_no_secret_in_logs` | last, so it covers the whole run: the tenant tag, scope, scope tag and derived salt of every key (each epoch, each client salt sent), and B's rewritten reserved salt, appear in no api log line, engine log line, history row, capture (API and files) or `effective-argv`. A value a client sent itself is exempt only in that request's own capture and history row. Values are never printed |

### Disk only: `test_kv_disk_only_live.py`

It runs first (file order) and takes about 10 minutes. It sets the model's `kv_store` to
`off` for a vanilla pass and to `nvme` for the rest, and puts it back at the end. The
instance needs `kv_store_ram_gib` 1, a disk folder at `/kv` with the marker file
(`kv_store_nvme_marker` true) and `--prefix-match-unit 16` in the model's `extra_args`.
It reads the daemon's socket, its NVMe manifest and the page cache, so it runs inside the
api container.

The shared run: a vanilla pass of the long prompt (`VW_KV_LIVE_LONG_SIZE`), a save pass
with `nvme`, a wait for the writes to land, an engine restart and a resume.

| Test | Checks |
|---|---|
| `test_disk_only_argv_and_pool` | the engine argv has `"tiers": "nvme"`; `GET /api/system/kv-store` shows a 1 GiB pool and at least one staging namespace |
| `test_long_prompt_resumes_from_disk` | the resume loads exactly `floor((P - 1) / B) * B` tokens from NVMe (MTP with `disable_eagle_block_drop`, or no MTP) and none from RAM or the GPU; TTFT at least `VW_KV_LIVE_MIN_SPEEDUP` times lower than vanilla cold; output identical to the save pass's cold output, and to vanilla's or differing first at a near-tie |
| `test_ram_returns_to_empty` | `ram_used` and `staging.in_flight` are 0 once the save pass's writes land, and again after the resume |
| `test_segments_stay_out_of_the_page_cache` | at most 64 MiB of the segment files are in the page cache after the save pass and after the resume |
| `test_weights_stay_in_the_page_cache` | the resident share of the weight files drops by at most 2 percentage points between the model load and the end of the resume |
| `test_state_snapshots_saved_with_match_unit_16` | each GDN state namespace holds exactly the snapshots at vLLM's chunk ends floored to `B` (each chunk starts where the last one ended) plus `floor((P - 1) / B) * B`: read from the manifest against the prompt's attention chain, as a block count from `stats.by_namespace`, and by a `lookup` of each key |
| `test_partial_head_block_is_private` | X (20,000 tokens) is saved and flushed from the GPU by unique prompts; W, X's first 10,000 tokens (16-aligned, not a multiple of `B`), is sent; X again gets `gpu = floor(c_W / B) * B` and the store serves the rest up to `floor((P - 1) / B) * B`, with X's cold output; W again hits `c_W` on the GPU with its first output |
| `test_write_volume_and_counters` | `stats.reuse.nvme_bytes_written` during the run within 10% of the records the manifest gained; `/api/kv/usage` per model adds up to the same; the resume grew the key's own `reuse.reloads` by one per position loaded (one request, not one per group) |

The derived values are computed by the test (re-implemented from `lmw_kv/PROTOCOL.md`
and cross-checked against `lmw_kv.scopes` when it is importable), from the network key
file and the key rows in the database. So G7's "A's derived salt" and G7c need the run
inside the api container.

## Env contract

Put these in a private env file outside the repo.

Required:

| Var | Meaning |
|---|---|
| `VW_KV_LIVE_BASE_URL` | the instance, no trailing slash |
| `VW_KV_LIVE_ADMIN_TOKEN` | an admin token (`vwa_...`), never printed |
| `VW_KV_LIVE_MODEL` | the model's served name |
| `VW_KV_LIVE_ACK_RELOADS=1` | you accept that the run reloads the model |

Optional:

| Var | Default | Meaning |
|---|---|---|
| `VW_KV_LIVE_KILL_DAEMON_CMD` | none | a shell command that kills the store daemon, for example `pkill -KILL -f lmw_kv.daemon` when the tests run inside the api container. Without it the two daemon tests skip. |
| `VW_KV_LIVE_KILL_DELAY_S` | 0.1 | seconds between sending the 50k prompt and running the kill command |
| `VW_KV_LIVE_DAEMON_WAIT_S` | 15 | seconds to let the api restart the daemon |
| `VW_KV_LIVE_BREAK_NVME_CMD` | none | a shell command that damages the NVMe records, for example `find /kv -name '*.seg' -exec truncate -s 54 {} +` (keeps each segment's 54-byte header). Without it the failed-load test skips. It runs late in the file: the old-namespace test does not need NVMe. |
| `VW_KV_LIVE_LOG_CMD` | none | a shell command that prints the engine log; its `LM Warden KV` lines go into the report |
| `VW_KV_LIVE_OTHER_REVISION` | none | another commit of the same model, for the old-namespace test |
| `VW_KV_LIVE_SIZES` | `2048,20000,50000` | prompt sizes in tokens |
| `VW_KV_LIVE_MIN_SPEEDUP` | 5 | TTFT ratio the restore must reach at 20k+ |
| `VW_KV_LIVE_TIE_MARGIN` | 0.01 | top-2 logprob margin that counts as a near-tie |
| `VW_KV_LIVE_LOAD_TIMEOUT_S` | 1200 | seconds a load or pull may take |
| `VW_KV_LIVE_VERIFY_TLS` | 1 | 0 to skip TLS checks |
| `VW_KV_LIVE_REPORT_DIR` | `~/.local/state/vw-kv-live` | where the JSON report of each run goes |
| `VW_KV_LIVE_DAEMON_ALIVE_CMD` | none | exits 0 while a store daemon runs, for example `pgrep -f '[l]mw_kv[.]daemon'`. Without it G8 skips |
| `VW_KV_LIVE_API_LOG_CMD` | none | prints the api log (G7c), for example `cat /data/logs/api-stdout.log` when the api's output is tee'd there |
| `VW_KV_LIVE_SEGMENTS_CMD` | none | prints `<path> <size>` per segment file, for example `find /kv -name '*.seg' -printf '%p %s\n'`. Without it G12 skips |
| `VW_KV_LIVE_NETWORK_KEY_FILE` | none | the cobs2 network key file (`/data/cobs2_network_key`), read only, for G7 and G7c |
| `VW_KV_LIVE_DB_PATH` | none | the database (`/data/vllm-warden.db`), opened read-only for the keys' tenant ids and epochs. Without these two G7c skips and G7 sends a random `lmw1:` value |
| `VW_KV_LIVE_LOG_DIR` | none | the engine log directory (`/data/logs`): G7c reads every log file in it whole. Without it G7c reads `VW_KV_LIVE_LOG_CMD`'s lines written during the run |
| `VW_KV_LIVE_CAPTURE_DIR` | none | the capture directory (`/data/capture`), grepped by G7c besides the capture API |
| `VW_KV_LIVE_G12_HF_REPO`, `VW_KV_LIVE_G12_REVISION` | none | a small model (full commit id) for G12's two rows. Without them G12 skips |
| `VW_KV_LIVE_G12_GPUS` | none | two GPU indices, one per G12 row: LM Warden runs one loaded model per GPU. Without them G12 skips |
| `VW_KV_LIVE_G12_DTYPE` | `bfloat16` | both rows' dtype (the namespace includes it); `float16` when one GPU has no bf16 |
| `VW_KV_LIVE_G12_GPU_UTIL` | 0.4 | each G12 row's `gpu_memory_utilization` |
| `VW_KV_LIVE_GPU_CASE_SIZE` | 8192 | prompt size of G7's and G8's GPU case: six prompts must fit the GPU prefix cache together, or A's blocks are evicted before the others try them |
| `VW_KV_LIVE_NVME_FLUSH_S` | 15 | seconds to let the write-through to NVMe finish before a check that needs it |
| `VW_KV_LIVE_FLOOD_FACTOR` | 10 | the flood in multiples of the flooder's RAM share |
| `VW_KV_LIVE_FLOOD_MAX_TOKENS` | 0 | a cap on the flood's tokens (0: none) |
| `VW_KV_LIVE_TIER` | `ram+nvme` | the tier step 1's and Plan 2's files run on: `ram+nvme` or `nvme` |
| `VW_KV_LIVE_SOCKET` | none | the daemon's socket (`/dev/shm/lmw-kv.sock` in the api container). Without it, or the two dirs below, the disk only file skips |
| `VW_KV_LIVE_SEGMENT_DIR` | none | the NVMe directory (`/kv`): segment files and manifests |
| `VW_KV_LIVE_WEIGHTS_DIR` | none | a directory holding the model's `*.safetensors` (symlinks are followed) |
| `VW_KV_LIVE_LONG_SIZE` | 50000 | the disk only file's long prompt, in tokens |
| `VW_KV_LIVE_MATCH_UNIT` | 16 | the model's `--prefix-match-unit` |
| `VW_KV_LIVE_BLOCK_SIZE` | none | the engine block size; without it the test reads it from the engine log (`VW_KV_LIVE_LOG_CMD`) |
| `VW_KV_LIVE_GPU_FLUSH_TOKENS` | 160000 | tokens of unique 20k prompts that evict the GPU prefix cache (more than it holds) |
| `VW_KV_LIVE_WRITE_TOLERANCE` | 0.10 | how far `nvme_bytes_written` and the reports may be from the records written |

## Setting up the instance

1. The api service needs a `/dev/shm` of at least twice the RAM tier plus 2 GiB
   plus the engine, and a fast local disk mounted at `/kv`
   (see `docs/operating.md`, "KV-cache tiers"). The daemon keeps 2 GiB of
   `/dev/shm` free, and the tests kill it: the new daemon sizes its pool while the
   running engine still maps the old one. With less, the new pool is shrunk, and a
   disk only pool (1 GiB, the minimum) is not made at all, so the store stays off
   until the next daemon start. 8 GiB works for the disk only run; 4 GiB did not.
2. Runtime settings: `kv_store_nvme_dir` `/kv`, `kv_store_ram_gib` to fit
   `/dev/shm`. Restart the api after changing them.
3. Register the model with a pinned revision (a full commit id) and a
   `max_model_len` of at least 50,112. The model's `kv_store` can start at any
   value; the run sets it.
4. Mint an admin token for the run and revoke it afterwards.
5. Plan 2 only. G12 needs two GPUs the instance may use (`allowed_gpu_indices`),
   because LM Warden runs one loaded model per GPU. G7c needs the api's output in
   a file: run the api with its output tee'd, for example
   `bash -c 'exec > >(tee -a /data/logs/api-stdout.log) 2>&1; exec uvicorn ...'`.

## Running it

```
set -a; . /path/outside/repo/vw-kv-live.env; set +a
pytest -m live tests/live/kv -v -s
```

To run inside the api container (so the kill command and the log command are
local), copy `tests/__init__.py`, `tests/live/__init__.py` (both may be empty)
and `tests/live/kv/` into a directory there, install `pytest` and run
`python -m pytest -o addopts= -p no:cacheprovider tests/live/kv -v -s` from that
directory with `VW_KV_LIVE_BASE_URL=http://127.0.0.1:8080`. Without
`tests/live/conftest.py` nothing adds the `live` marker there, so leave out `-m live`.
Set `PYTHONUNBUFFERED=1` when the output goes to a file, or the progress lines
arrive only at the end.

## Results

### 2026-11-23, disk only read path, RTX A4000 (16 GB, sm_86), TP1

The disk only resume was 1.9x faster than cold (below, same day). It is now 9x at 50k and
5.9x at 20k. Same host, model, args and settings as the run below (`shm_size` 8 GiB), image
built from this branch at `17b89458` (llama.cpp for sm_86 only). The tests ran inside the
api container with `VW_KV_LIVE_TIER=nvme`.

**Where the time went (before).** A 50k resume loads 115 records (2.19 GB) in 5.7 s. The
daemon promoted them one at a time: 4.9 s, as pread 0.7 s, the CRC with a 19.5 MB slice
copy 1.4 s, a second slice copy 0.5 s, the AES-GCM decrypt into a new buffer 1.2 s, the
page cache drop 0.5 s, the copy into the RAM slot 0.2 s and the rest 0.3 s. The GPU copy
(0.85 s) ran only after each window's reads. On this host direct reads of the same
records run at 1.4 GB/s on one thread and 4.3 GB/s on four. A copy from pinned memory to
the GPU runs at 2.85 GB/s (the card's PCIe link): 0.77 s at 50k and 0.31 s at 20k, the
floor of any resume here.

**What changed.**

1. A record is read into a reused buffer and checked on views, with no slice copy, and
   decrypted with `AESGCM.decrypt_into` straight into its RAM slot. Same checks in the same
   order: frame, CRC, key, group, then the GCM tag. `decrypt` (a new 19.5 MB buffer each
   time) does not speed up across threads (1.4 to 1.8 GB/s); `decrypt_into` does (5.8 GB/s
   on one thread, 11.3 on eight).
2. A load's promotions run in 8 worker threads and finish in key order. The store's own
   steps and their order are unchanged: a block that needs a policy decision (no free slot,
   or a `fair` allocation outside disk only mode) waits for the earlier ones first.
3. The connector reads the next window while it copies the current one to the GPU (two
   half windows, so the pins stay within one window).

| Prompt | Cold | Before | After (warm engine) | After (first request after restart) |
|---|---|---|---|---|
| 20,000 | 3.88 to 4.19 s | 2.03 s (1.9x) | 0.65 to 0.68 s, load 0.41 s (5.7x to 6.4x) | not measured |
| 50,000 | 12.33 to 12.46 s | 6.38 s (1.9x) | 1.10 s, load 0.79 s (11.3x) | 1.39 to 1.61 s, load 0.79 to 0.94 s (7.7x to 9.0x) |

Each change on its own, from a driver on the same prompts: change 1 took the 50k resume from
6.3 s to 2.1 s and the warm 20k one to 0.90 s; change 2 to 1.56 s and 0.73 s; change 3 to
1.44 to 1.61 s and 0.65 to 0.68 s. A first request after an engine restart carries about
0.55 s of engine warm-up besides the load; a warm one about 0.25 s.

**Live suite on the final image** (`pytest tests/live/kv -v`, 28.7 minutes): 19 passed, 1
failed, 2 skipped. The failure was the weights page cache check, which found no weight
files: the run's `VW_KV_LIVE_WEIGHTS_DIR` pointed at a folder that does not exist. The
disk only file then ran again on its own with the right folder (below). The skips are G12
and the fair flood (no RAM tier on `nvme`).

| Test | Result |
|---|---|
| Resume from disk (50k) | 49,632 `nvme` tokens exactly, 0 RAM, 0 GPU. TTFT 12.46 s cold, 13.26 s save pass, 1.39 s resume (load 0.79 s): **9.0x**. Identical to the save pass and to vanilla |
| RAM back to empty | `ram_used` 0 and `in_flight` 0 after the save pass (writes landed at once) and after the resume |
| Segments in the page cache | 1.4 MiB resident of 9.0 GB in 167 segment files, after the save pass and after the resume |
| Weights in the page cache | re-run (below): 92.9% resident after the load and after the resume (0.0 points drop) |
| State snapshots | each of the 3 GDN namespaces: 7,920 ... 47,520 and 49,632, 7 blocks, each found by `lookup` |
| Partial head block | W first: `nvme` 7,920, TTFT 0.81 s. X after W: `gpu` 9,504 + `nvme` 10,032, TTFT 0.53 s, identical to X cold. W again: `gpu` 9,984, identical |
| Write volume and counters | 532 records, 10,127,916,540 B, equal to `nvme_bytes_written`; reloads +94 for 94 positions |
| G7, G8, G11, G7c | pass. G7 NVMe path: A's repeat 19,536 from NVMe, TTFT 0.75 s (2.95 s before). G11 restores 0.67 to 1.12 s |
| Restore (step 1) | 2,048: 0.60 s; **20,000: 0.66 s vs 3.88 s cold (5.9x)**; 50,000: 1.10 s vs 12.41 s (11.3x). Parity identical at every size |
| Daemon killed during the 50k load | output identical, engine serving; the kill landed after `begin_load` |
| NVMe after a daemon restart | 19,536 `nvme`, TTFT 1.09 s, load 0.52 s |
| Failed load, old namespace | pass (one `load failed` line, vanilla output; the other revision 0) |

**The disk only file again** (on a fresh api, with the weights folder right): **8 passed** in
5.7 minutes. Resume 1.58 s against 12.38 s cold (load 0.98 s): 7.8x, identical output.
Weights 92.9% resident after the load and after the resume (0.0 points). Segments 3.6 MiB
resident of 6.1 GB in 756 files. `ram_used` and `in_flight` 0 after the save pass and the
resume. Snapshots, partial head block, write volume (532 records, 10.13 GB, equal to
`nvme_bytes_written`) and reloads (+94) as above.

### 2026-11-23, disk only (`kv_store` `nvme`), RTX A4000 (16 GB, sm_86), TP1

- **Hardware.** One host: AMD Ryzen 9 5950X, 62 GB RAM, NVMe disk (ext4).
- **Setup.** A temporary LM Warden on the test host (Kubernetes, local-subprocess driver),
  image built from this branch at `253a87a7` (llama.cpp built for sm_86 only), vLLM
  0.32.0. Model `Qwen/Qwen3.5-4B` @ `851bf6e8`, bf16, `max_model_len` 65536,
  `gpu_memory_utilization` 0.9, `--prefix-match-unit 16`, MTP with
  `disable_eagle_block_drop`, `--max-num-batched-tokens 8192 --max-num-seqs 8
  --max-cudagraph-capture-size 16`. Block size 528, GPU prefix cache 113,465 tokens.
  `kv_store_ram_gib` 1, `kv_store_nvme_dir` `/kv` with the marker, `kv_store_nvme_cap_gb`
  60, api `shm_size` 8 GiB. The tests ran inside the api container with
  `VW_KV_LIVE_TIER=nvme`.
- **Result.** `pytest tests/live/kv -v`: **18 passed, 2 failed, 2 skipped** in 27.3
  minutes. Both failures are the 5x speed-up checks: a disk only resume is about 1.9x
  faster than cold, not 5x (below). The skips are G12 and the fair flood (no RAM tier).

Disk only (`test_kv_disk_only_live.py`, P = 50,000):

| Check | Result |
|---|---|
| Argv and pool | `"tiers": "nvme"`; pool 1 GiB granted; 4 staging namespaces |
| Resume from disk | 49,632 `nvme` tokens (exactly `floor((P - 1) / B) * B`), 0 RAM, 0 GPU. TTFT 12.38 s cold, 13.19 s save pass, 6.38 s resume: **1.9x (fails the 5x check)**. Load 5.79 s for 115 records (2.2 GB) |
| Parity | resume identical to the save pass and to vanilla |
| RAM back to empty | `ram_used` 0 and `staging.in_flight` 0 after the save pass and after the resume; the writes had landed when the stream ended (0.0 s wait) (1) |
| Segments in the page cache | 0.9 MiB resident of 4.4 GB in 78 segment files, after the save pass and after the resume (1) |
| Weights in the page cache | 97.9% resident after the load and after the resume (0.0 points drop) (1) |
| State snapshots | each of the 3 GDN namespaces holds exactly 7,920, 15,840, 23,760, 31,680, 39,600, 47,520 and 49,632 (7 blocks in `stats.by_namespace`, each found by `lookup`) |
| Partial head block | W (10,000 tokens): its tail checkpoint `c_W` = 9,984 (18 blocks + 480). X (20,000) after W: `gpu` 9,504 + `nvme` 10,032, output identical to X's cold output. W again: `gpu` 9,984, output identical |
| Write volume | 532 records written during the file (10.13 GB): 427 of 19,464,285 B (attention) and 105 of 17,301,589 B (GDN state). `nvme_bytes_written` 10,127,916,540, equal to the manifest's record bytes; `/api/kv/usage` per model the same |
| Reloads | the resume grew the key's own `reuse.reloads` by 94: one per position loaded (94), not one per namespace or window |
| Empty load windows | no counter exists, and a window that waits and then succeeds logs nothing. 0 `load failed` lines outside the failed-load test. With a 1 GiB pool and one rank a window is 27 blocks and nothing else held slots during a resume, so the estimate is 0 empty windows and 0 s added |

(1) These figures are from the first pass of this file on the same image and settings
(with a 4 GiB `/dev/shm`), which passed the same checks as the final run; the final
run's report file went with the deployment before it was copied. Every other figure is
from the final run.

Step 1 and Plan 2 on `nvme`, same run:

| Test | Result |
|---|---|
| G7 | B and C: 0 everywhere in all three cases. A's repeat: 8,176 from the GPU; 19,536 from NVMe after an engine restart (TTFT 2.80 s, load 2.57 s); 19,536 from NVMe after a daemon and engine restart (load 2.37 s) |
| G8 | B and C: 0 cached. A's repeat: 8,176 cached |
| G11 | all 8 comparisons with vanilla identical; restores 19,536 from NVMe; D's isolated turn reused nothing |
| G12, fair flood | skipped (no RAM tier on `nvme`) |
| G7c | 38 requests; 8 tenant tags, 10 scopes, 10 scope tags and 152 derived salts found in none of the api log, the engine logs, 38 history rows, 38 captures, 62 capture files and `effective-argv` |
| Restore (step 1) | 2,048: 1,584 `nvme`, TTFT 0.78 s; 20,000: 19,536, 2.03 s vs 3.88 s cold (**1.9x, fails the 5x check**); 50,000: 49,632, 4.97 s, load 4.63 s. Parity identical at every size; `cached_tokens` = gpu + ram + nvme |
| Daemon killed during the 50k load | output identical, engine serving. The kill landed before the lookup: the worker turned its store off ("the daemon restarted") and the request recomputed (0 cached) |
| NVMe after a daemon restart | 19,536 `nvme`, TTFT 2.95 s, load 2.39 s, identical |
| Failed load | one `load failed` line, vanilla output, 0 cached |
| Old namespace | control 19,536 from NVMe; after the revision changed, 0 |

- **Disk read speed.** The disk is not the limit: direct reads of the same segment
  files ran at 5.2 GB/s, while a resume reads about 0.4 GB/s (115 records, 2.2 GB, in
  5.8 s). Per 19.5 MB record, the CRC takes about 3.6 ms and the AES-GCM decrypt about
  7.6 ms (about 1.3 s of the 5.8 s); the rest was not broken down.
- **`/dev/shm`.** A first attempt with a 4 GiB `/dev/shm` failed G7's NVMe case: after
  the daemon was killed, the new one could fit only 0.99 GiB beside the old pool (still
  mapped by the engine) and the 2 GiB reserve, under the 1 GiB minimum, so it ran
  with no RAM tier and the disk only store stayed off. 8 GiB passed.

### 2026-11-22, RTX A4000 (16 GB, sm_86) and Quadro RTX 5000 (16 GB, sm_75), TP1

- **Hardware.** One host: AMD Ryzen 9 5950X, 62 GB RAM, NVMe disk.
- **Setup.** A temporary LM Warden on the test host (Kubernetes, local-subprocess driver),
  image built from this branch at `5623e8c6` (the run's tests and `tools/` changes are
  not in the image), vLLM 0.32.0. Model `Qwen/Qwen3.5-4B` @ `851bf6e8`, bf16, on the
  A4000, `max_model_len` 65536, `gpu_memory_utilization` 0.9, `--max-num-batched-tokens
  8192 --max-num-seqs 4` (GPU prefix cache 113,562 tokens, block size 528). api `shm_size`
  12 GiB, `kv_store_ram_gib` 8, `kv_store_nvme_dir` `/kv` on a `local-path` volume on the
  host's NVMe disk. G12 used `Qwen/Qwen3.5-0.8B` @ `2fc06364`, float16 (the RTX 5000 has
  no bf16), one row on each GPU. The tests ran inside the api container.
- **Result.** `pytest tests/live/kv -v`: **14 passed** in 36.6 minutes (both files, one
  run). In that run G7c read only part of the engine log (a "lines since the start"
  marker matched a repeated line), so the Plan 2 file ran again on its own with G7c
  reading every engine log file whole: **6 passed** in 24.7 minutes, the same numbers as
  below.

Plan 2 (`test_kv_isolation_live.py`):

| Gate | Result |
|---|---|
| G7, GPU case (8k) | B (no salt, A's client salt, A's own derived salt) and C (no salt, A's client salt): 0 cached tokens and 0 in every tier, while A's blocks sat in the GPU cache. A's repeat: 7,920 tokens from the GPU |
| G7, RAM path (20k, engine restarted) | B and C: 0 everywhere. A's repeat: 19,536 from RAM, TTFT 0.45 s (4.0 s cold) |
| G7, NVMe path (20k, daemon and engine restarted) | B and C: 0 everywhere. A's repeat: 19,536 from NVMe, TTFT 1.8 s |
| G8 (kv_store off, no daemon process before or after) | B and C: 0 cached tokens. A's repeat: 7,920 cached |
| G11 | all 8 comparisons with vanilla IDENTICAL: A and B save and restore (19,536 from RAM), D sharing, D switched to isolated (reused nothing), D's isolated restore (19,536 from RAM), D switched back to sharing (19,536 reused) |
| G12 | control: the `ram+nvme` row alone grew 4 segment files. With the `ram` row loaded too, prompts to both rows grew 0 files under that namespace. After the `ram` row unloaded, the `ram+nvme` row grew 5 again |
| Fair flood smoke | `fair`; the flooder sent 53 prompts of 20k (1.06 M tokens, 4.9 times the 8 GiB RAM tier at 39,793 bytes per token, in 356 s). After an engine restart the victim's session loaded 19,536 tokens from RAM (97.7% of the prompt), TTFT 0.72 s |
| G7c (second run) | 46 requests; 8 tenant tags, 10 scopes, 10 scope tags and 152 derived salts searched in the api log (0.27 M characters), every engine log file whole (2.2 M), 46 history rows, 44 captures (API) and 133 capture files, and `effective-argv`: none found. Four requests carried A's derived salt as B's own client salt; that value is exempt only in those requests' own capture and history row |

Step 1 (`test_kv_live.py`), same run:

| Prompt | Vanilla cold TTFT | Save pass TTFT | Restore TTFT (RAM) | Speed-up | Load time | Parity |
|---|---|---|---|---|---|---|
| 2,048 | 0.640 s | 0.694 s | 0.447 s | 1.4x | 0.040 s | identical |
| 20,000 | 3.723 s | 3.991 s | 0.445 s | 8.4x | 0.245 s | identical |
| 50,000 | 11.875 s | 12.501 s | 0.860 s | 13.8x | 0.592 s | identical |

- NVMe after a daemon restart: 19,536 `nvme` tokens, TTFT 1.91 s, load 1.46 s.
- Daemon killed during the 50k load: output identical, engine serving; the kill landed
  after `begin_load` (the copy finished from the mapped RAM file) and the turn's tier
  columns read null while `cached_tokens` was 49,632 (the known lost-hit-record case).
- Failed load (NVMe records damaged): one `load failed` line, vanilla output, 0 cached.
- Old namespace: the control restored 19,536 from RAM under the original revision; after
  the revision changed, 0 cached and 0 from the store.
- The stricter step 1 checks added after 2026-11-21 (the control, the kill exit code,
  exactly one failed load) ran on a GPU for the first time here and passed.

### 2026-11-21, RTX A4000 (16 GB, sm_86), one GPU, TP1

- **Setup.** A temporary LM Warden on the test host (Kubernetes, local-subprocess driver),
  image built from this branch at `468de818`, vLLM 0.32.0, V2 model runner. Model
  `Qwen/Qwen3.5-4B` @ `851bf6e8`, bf16, `max_model_len` 65536,
  `gpu_memory_utilization` 0.9, `--max-num-batched-tokens 8192 --max-num-seqs 4`.
  vLLM picked `mamba_cache_mode=align` and block size 528 for all four KV-cache groups
  on its own. api `shm_size` 12 GiB, `kv_store_ram_gib` 8, `kv_store_nvme_dir` `/kv` on
  a `local-path` volume on the host's NVMe disk.
- **Result.** 8 passed (`pytest tests/live/kv -v`, 11.4 minutes). The tests ran inside
  the api container, so the kill and log commands were local.

| Prompt | Vanilla cold TTFT | Save pass TTFT | Restore TTFT (RAM) | Speed-up | Load time | Parity |
|---|---|---|---|---|---|---|
| 2,048 | 0.641 s | 0.677 s | 0.471 s | 1.4x | 0.039 s | identical |
| 20,000 | 3.653 s | 3.951 s | 0.414 s | 8.8x | 0.244 s | differs at token 23 on an exact bf16 tie (top-2 margin 0.0000) |
| 50,000 | 11.526 s | 12.355 s | 0.824 s | 14.0x | 0.589 s | identical |

- **History.** Each restored turn shows `ram` = vLLM's `cached_tokens` (1,584, 19,536
  and 49,632: whole 528-token blocks, the last prompt token is never loaded) and
  `gpu` = 0. So vLLM's `cached_tokens` includes the tokens the connector loaded.
- **Save cost.** The save pass is 7 to 8 percent slower than vanilla at 20k and 50k
  (D2H copies and one `begin_save` round trip per block and group).
- **NVMe.** After a daemon restart (empty RAM pool) the 20k prompt loaded from NVMe:
  19,536 `nvme` tokens, TTFT 1.96 to 2.89 s, load 1.5 to 2.3 s, output identical. On
  this disk the NVMe tier is only 1.3 to 1.9 times faster than recomputing at 20k.
- **Daemon killed during a 50k load.** The output was identical and the engine kept
  serving. In all four runs the kill landed after `begin_load`: the copy finished from
  the RAM file the worker still had mapped, and only the hit record was lost.
- **Failed load.** NVMe records damaged behind a fresh daemon's back: the lookup matched,
  the load failed, the engine logged `LM Warden KV load failed for ...; vLLM recomputes
  it`, and the 50k request completed with vanilla output and 0 cached tokens.
- **Old namespace.** After switching the revision to `bbaaae1f` (same weights) the 20k
  prompt got 0 cached tokens and 0 from the store. That run had no positive control yet;
  the control (a restore under the original revision first) was added afterwards and
  has not run on a GPU.
- **Since this run** the tests got stricter (the control above, the kill command must
  succeed, exactly one failed load during the failed-load test). They run again at the
  next GPU gate.
