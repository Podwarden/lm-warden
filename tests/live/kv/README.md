# tests/live/kv: KV-cache tier tests on a GPU

These tests drive a running LM Warden whose model has `kv_store` set to `ram+nvme`.
They prove on real hardware what the unit tests cannot: the restored KV gives the
same tokens as a vanilla run, a restore is much faster than recomputing, and the
store failing never breaks a request.

Two files. `test_kv_live.py` is step 1 (tiers, parity, fail-open).
`test_kv_isolation_live.py` is Plan 2 (per-key isolation, two rows on one namespace,
the fair policy under a flood).

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

## Setting up the instance

1. The api service needs a `/dev/shm` big enough for the RAM tier plus 1 GiB
   plus the engine, and a fast local disk mounted at `/kv`
   (see `docs/operating.md`, "KV-cache tiers").
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
