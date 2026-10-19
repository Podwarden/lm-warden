# Engine load log fixtures

Byte-faithful (or synthetic, as noted) engine startup logs used by the
model-load-progress detection tests. Later tasks parse these to detect load
steps, so the real logs preserve raw bytes including `\r`, ANSI-free tqdm
bars, pids, `(Worker_TPn pid=…)`/`(EngineCore_DPn pid=…)` prefixes, and
timestamps.

## Files

### `vllm-0.26.0-dp3-tp2-fp8.log`

- **Source host:** lmw48 (7× RTX 5090) — no IP recorded here.
- **Date:** 2026-10-05
- **Model:** `Qwen/Qwen3.8-27B-FP8` (public HF repo — kept as-is)
- **Layout:** 3 replicas × TP2, `api_server_count` 3, DP3, FP8 KV cache
- **vLLM version:** 0.26.0
- **Line range taken:** lines 1–682 of the docker volume log file
  `/var/lib/docker/volumes/vllm-warden_vw-data/_data/logs/ddfe51df203e2cbd.log.1`
  — from the docker log header (line 1, a truncation notice) through the
  banner (lines 2–3), the `non-default args` line (line 4), and the full cold
  start ending at the third `Application startup complete.` (line 682,
  `ApiServer_2`).
- **Transfer method:** `ssh ubuntu@<host> 'sudo sed -n "1,682p" <file>'`
  (read-only, no changes on the source host).
- **Scrubs applied:**
  - `172.18.0.2` (Docker bridge IP, internal to the container) → `172.17.0.2`
    (placeholder, also a Docker bridge range but not the actual container IP).
  - `127.0.0.1` (universal loopback) kept as-is — it is the `host` config value
    and reveals nothing.
  - No hostnames, HF usernames/orgs (other than the public `Qwen/Qwen3.8-27B-FP8`
    repo), API keys, or tokens were present. The `HF_TOKEN` string that appears
    is the standard HuggingFace environment-variable name in a generic
    "unauthenticated requests" warning — no token value is present.
- **Key markers (for verification):**
  - `Loading safetensors checkpoint shards` — 207 occurrences (shard-loading
    progress bars across 3 DP replicas × 2 TP workers)
  - `init engine (profile, create kv cache, warmup model) took` — 3 occurrences
    (one per DP replica)
  - `Capturing CUDA graphs` — 9 occurrences (across 3 replicas)

### `llamacpp-b10731.synthetic.log`

- **Synthetic** — no real load log exists for llama.cpp b10731. This file is a
  short (~62 lines) plausible `llama-server` startup log built from the real
  log format strings in the llama.cpp source (server.cpp, server-context.cpp)
  fetched at the scratchpad `lsrc/` directory.
- **Build:** `b10731-0eadefebd` (matching the upstream tag used by the
  existing `tests/fixtures/llamacpp/` fixtures).
- **Model:** `qwen2.5-7b-instruct-q4_k_m.gguf` (plausible local model path,
  not a real capture).
- **Source strings used (from `lsrc/`):**
  - `SRV_INF("loading model '%s'\n", …)` — `server-context.cpp:1042`
  - `SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n", …)` —
    `server-context.cpp:1250`
  - `SRV_INF("%s", "model loaded\n")` — `server.cpp:489`
  - `SRV_INF("listening on %s\n", …)` — `server.cpp:515`
  - `SRV_WRN("%s", "CORS is set to allow all origins …")` — `server.cpp:324`
  - `common_params_print_info: verbosity = 3` — from existing llama.cpp
    fixture `log_startup_ok.txt`
  - `llama threadpool init, n_threads = 8` — from existing fixture
  - `llama_model_loader` / `load_tensors` / `llama_kv_cache_init` /
    `llama_new_context_with_model` — canonical llama.cpp core log format
    (the core library source `src/llama-model-loader.cpp` etc. is not in the
    server source snapshot, but these format strings are stable across
    b10731).
- **Coverage:** start, model loader metadata, tensor dump, load_tensors
  progress (0%–100%) and end, context/KV init, "model loaded", server
  listening.
- **No scrubs needed** — all values are synthetic; the bind address
  `0.0.0.0:8080` is the llama-server default.

## Publish safety

Both files have been checked against the `publish/denylist-block.txt` and
`publish/denylist-warn.txt` denylists — no hits. Additional grep checks for
server and LAN address prefixes, internal domains, and `vw_` / `hf_` key
prefixes confirmed no infrastructure, credentials, or internal hostnames are
present.

## GLM-5.2 NVFP4 on 8× B300 (TP8 + expert parallel), before and after the engine bump

Two real cold starts of the same model with the same settings, captured on 2026-10-18 from the 8× NVIDIA
B300 SXM6 host (NVLink through one NVSwitch). Each file runs from the vLLM banner to the
`Application startup complete.` line; nothing after it (no client requests) is included.

| File | vLLM | Engine start → HTTP up | Warden `loaded` |
|---|---|---|---|
| `vllm-0.26.0-tp8-glm52-nvfp4-b300.log` | v0.26.0 | 13:09 | 15:50 |
| `vllm-0.32.0-tp8-glm52-nvfp4-b300.log` | v0.32.0 | ~11:00 | 11:24 |

Model `nvidia/GLM-5.2-NVFP4`, `tensor_parallel_size` 8, `enable_expert_parallel`, `kv_cache_dtype`
`fp8_e4m3`, `max_model_len` 262144. The v0.26.0 run has a ~5.5 min stretch with no log output between
KV-cache sizing and CUDA-graph capture, and the warden wrote its "still loading after 600s" note into
`last_error` at 600 s while the load was healthy. On v0.32.0 the weight lines come from
`model_runner.py` (the new default runner) and every TP rank logs its own `Model loading took` line.

Scrubs: the Docker bridge address `172.18.0.2` is replaced with `172.17.0.2`; the files were checked
for hostnames, public IPs, tokens and usernames.

## Qwen3.8-27B-FP8 on lmw48 (7× RTX 5090), 3 replicas × TP2, on vLLM 0.32.0

`vllm-0.32.0-dp3-tp2-fp8.log` is the same model and layout as `vllm-0.26.0-dp3-tp2-fp8.log`, captured on
2026-10-18 after the engine bump. Engine init took 171 s per replica (254 s on 0.26.0). On 0.32.0 the
top-level `non-default args` line of a data-parallel launch is prefixed `(MainProcess pid=…)`. Same scrubs
as above (Docker bridge address only); no client requests are included.
