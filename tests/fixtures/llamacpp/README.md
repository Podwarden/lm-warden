# Captured `llama-server` output — sub-project C, Task 2

Everything sub-project C parses — the argv it builds, the health handling, the
`llamacpp:` metric dialect, the startup-log grammar — is written against the
files in this directory, not against anyone's recollection of what
`llama-server` prints. If a fixture goes stale, the parser written against it is
wrong and nothing else will say so.

## Provenance

| | |
|---|---|
| Upstream build tag | **`b11719`** |
| Upstream commit | `5aeac37a3` (`5aeac37a` — "ggml-cuda: fattn do not check oob if k_VKQ_max = (k_VKQ_max - nbatch_fa) (#30572)") |
| Image | `llama-cpp-cpu-b11719` (locally built from `ubuntu:24.04` + llama.cpp git tag `b11719`; the upstream GHCR `server-bNNNNN` prebuilt tags are garbage-collected after a few builds, so b11719 was not available as a prebuilt image) |
| Image digest | `sha256:bd977beaad8d932f59414c9b05944f4cdaa7e2317fc22557fbadf9d0db6aff22` (arm64 manifest; single-arch image) |
| Platform captured | `linux/arm64` (CPU-only, no GPU) |
| Reported version string | `version: 0.7.0-dev (build 11719, commit 5aeac37a3)` |
| GGUF used | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` → `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (668,788,096 B) |
| Host | Apple Silicon, Docker/colima `linux/aarch64`, **CPU-only, no GPU** |
| Capture date | 2026-10-18 |

Task 13 pins the same `b11719` tag in the `Dockerfile`, so the fixtures and the
shipped binary cannot drift.

### History: previous capture at b10731

The fixtures were originally captured at `b10731` (commit `0eadefebd`,
2026-09-01) from the upstream prebuilt image
`ghcr.io/ggml-org/llama.cpp:server-b10731` (image index digest
`sha256:bdce328e7152af01fc8724ef63ebefa1f39639fb6250731ca4fa6af04e1690b6`,
arm64 manifest digest `sha256:0ab379e88348888e68a32d255b932eb9170661f0418afcec144a1ba4040621db`).
That tag has since been garbage-collected from GHCR. The b11719 re-capture
builds the same source locally because no prebuilt `server-b11719` image exists
on GHCR either.

### Why a CPU-only, arm64 capture

The plan's capture recipe asks for `server-cuda13-b11719` on a box with an
NVIDIA GPU, and notes that "a CPU-only capture is acceptable for everything
except `log_oom.txt`". The only NVIDIA host in scope is `bonus`, which this
plan's Global Constraints make **read-only until Task 16**, and the one other
GPU box in the inventory has an unloaded driver. So the whole corpus was
captured from the *same upstream build tag*, on the CPU image, on arm64.

What this changes, and what it does not:

- **Does not change**: the HTTP surface (`/health`, `/v1/models`, `/props`,
  `/metrics`), the `llamacpp:` metric names and types, `--help`, `--version`
  format, and the `llama_model_load` / `srv` / `cmn` log grammar. All of these
  are architecture-independent and are what Tasks 7, 9 and 10 parse.
- **Does change**: no `ggml_cuda_init` / device-enumeration lines appear in the
  logs, and `log_oom.txt` is a **host-RAM** allocation failure rather than a
  CUDA device OOM (see below).

## The files

| File | What it is |
|---|---|
| `version.txt` | `--version`, stdout+stderr |
| `help.txt` | `--help`, 744 lines |
| `health_ready.json` | `GET /health` once the model is resident → `200` |
| `health_loading.json` | `GET /health` during load → `503`, the `middleware_server_state` pre-router envelope |
| `models.json` | `GET /v1/models` |
| `props.json` | `GET /props` |
| `metrics.txt` | `GET /metrics` after one real completion |
| `metrics_second_scrape.txt` | the *immediately following* `GET /metrics` — see property 1 |
| `log_startup_ok.txt` | a successful load, serve, and one completion |
| `log_unknown_arch.txt` | a GGUF whose `general.architecture` is `notarealarch` |
| `log_ctx_capped.txt` | `--ctx-size 8192` against a 2048-train model: warns, caps, and serves |
| `log_wrong_shard.txt` | a GGUF declaring `split.no=1, split.count=2` loaded without its siblings |
| `log_oom.txt` | `--ctx-size 999999`: the KV-cache buffer allocation fails |

## Capture commands

```bash
IMG=llama-cpp-cpu-b11719   # locally built: ubuntu:24.04 + llama.cpp git tag b11719, CPU-only
MODELS=$PWD/models          # holds the tinyllama GGUF
MODEL=/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

docker run --rm $IMG --version > version.txt 2>&1
docker run --rm $IMG --help    > help.txt    2>&1

docker run -d --name lcap -p 18080:18080 -v "$MODELS:/models" $IMG \
  --model $MODEL --host 0.0.0.0 --port 18080 --alias probe --metrics --no-webui
# poll /health in a tight loop and keep the first 503 body
curl -sS http://127.0.0.1:18080/health      > health_loading.json    # during load
curl -sS http://127.0.0.1:18080/health      > health_ready.json      # once 200
curl -sS http://127.0.0.1:18080/v1/models   > models.json
curl -sS http://127.0.0.1:18080/props       > props.json
curl -sS http://127.0.0.1:18080/v1/completions -H 'content-type: application/json' \
  -d '{"model":"probe","prompt":"hello","max_tokens":16,"stream":false}' > /dev/null
curl -sS http://127.0.0.1:18080/metrics     > metrics.txt
curl -sS http://127.0.0.1:18080/metrics     > metrics_second_scrape.txt
docker logs lcap                            > log_startup_ok.txt 2>&1
docker rm -f lcap

R() { docker run --rm -v "$MODELS:/models" $IMG "$@" ; }
R --model /models/broken.gguf                 --port 18080 > log_unknown_arch.txt 2>&1
R --model $MODEL --ctx-size 8192              --port 18080 > log_ctx_capped.txt   2>&1
R --model /models/shard-00002-of-00002.gguf   --port 18080 > log_wrong_shard.txt  2>&1
R --model $MODEL --ctx-size 999999            --port 18080 > log_oom.txt          2>&1
```

`broken.gguf` and `shard-00002-of-00002.gguf` are 320-byte hand-written GGUF v3
headers — the first declaring `general.architecture = "notarealarch"`, the
second declaring `split.no = 1`, `split.count = 2` (both `uint16`; llama.cpp
rejects a `uint32` `split.count` with a *different*, type-mismatch error, which
is itself worth knowing).

## The four properties verified by hand at capture time

### 1. Are the `llamacpp:` counters monotonic across two scrapes while the two `*_seconds` throughput gauges reset?

**Yes — exactly as upstream's source says.** `diff metrics.txt
metrics_second_scrape.txt` is two lines and only two lines:

```
33c33
< llamacpp:prompt_tokens_seconds 43.3623
---
> llamacpp:prompt_tokens_seconds 0
36c36
< llamacpp:predicted_tokens_seconds 44.7497
---
> llamacpp:predicted_tokens_seconds 0
```

Every counter (`prompt_tokens_total`, `prompt_tokens_cached_total`,
`prompt_seconds_total`, `tokens_predicted_total`,
`tokens_predicted_seconds_total`, `n_decode_total`, `n_tokens_max`, the three
`spec_decode_*` counters) is byte-identical between the two scrapes. **Task 10's
rate math is therefore sound**, and the two gauges must never be read: they are
averaged over the window since the last scrape and the bucket is reset by the
act of scraping (`tools/server/server-context.cpp:4657-4659`), so two
concurrent scrapers would halve each other's numbers.

### 2. Does the log contain ANSI escape sequences when stdout is not a tty?

**No — zero.** `log_startup_ok.txt` contains no `\x1b[` at all.
`--log-colors` defaults to `auto`, and `help.txt:210-212` documents that `auto`
"enables colors when output is to a terminal". The supervisor writes the engine
log to a file descriptor, so colour is off by construction. **No colour flag is
needed in Task 7's argv**, and `test_logs_carry_no_ansi_escapes` is the tripwire
if that ever changes.

### 3. Does the last line before a fast crash survive?

**Almost always, but not guaranteed.** Six captures of the same sub-20 ms
failure produced the terminal line
`E srv  llama_server: exiting due to model loading error` five times and lost it
once. llama.cpp uses its own logging thread with an explicit flush on exit
rather than relying on C++ stdio buffering, so this is a rare shutdown race
rather than the block-buffering class of bug that the 2026-05-15
`PYTHONUNBUFFERED` incident recorded at `env_builder.py:256-265` fixed for vLLM.

**Consequence for Task 9: the diagnosis grammar must key on the *diagnostic*
lines, never on the terminal line.** Each real cause appears at least twice in
its log — llama.cpp runs the load once inside `common_fit_params` and once for
real — so `llama_model_load: error loading model: …` is the reliable anchor.
The remedy if this ever regresses is a documented llama.cpp logging flag
(`--log-file`), **never** a wrapper program (spec §2).

### 4. The exact `--version` output format

```
version: 0.7.0-dev (build 11719, commit 5aeac37a3)
built with GNU 13.3.0 for Linux aarch64
```

Two lines. The build number is the `bNNNNN` tag without the `b`, and the commit
is the abbreviated upstream SHA. Task 8's `/api/system/backends` `version` field
and Task 13's baked build id are both derived from the `b11719` tag rather than
by shelling out to the binary, so they agree by construction.

## Changes in log grammar from b10731 to b11719

The re-capture at b11719 introduced the following surface changes in the logs.
None of them change the diagnostic anchors the parser keys on, but they are
noted here so a future diff is interpretable:

1. **Server init line**: `I srv  llama_server: initializing ...` is now the
   first line of every log (was absent at b10731).
2. **CORS warning**: `W srv  llama_server: security: no API key is set and CORS
   allows all origins (see ...)` replaces the three-line `CORS is set to allow
   all origins ('*') and no API key is set` block.
3. **UI disabled line**: `I srv  init_listene: The UI is disabled` (was
   `I srv          init: The UI is disabled` — the function name in the log
   prefix changed).
4. **Slot LRU line**: `I slot get_availabl: id  0 | task -1 | selected slot by
   LRU, n_tokens = 0, t_last = -1` now includes `n_tokens = 0` (was absent).
5. **Graph reallocs line**: `I slot print_timing: id  0 | task 0 |   graph
   reallocs =          0    /     0        (same size) / (changed size)` is a
   new line after `graphs reused`.
6. **OOM log**: `E ggml_backend_buft_alloc_buffer_n_default: failed to
   allocate CPU buffer of size ...` replaces the b10731 line
   `E alloc_tensor_range: failed to allocate CPU buffer of size ...`.
   The backend-neutral `failed to allocate buffer for kv cache` /
   `failed to allocate buffer of size` anchors are unchanged, so the
   diagnostics regex still matches.
7. **models.json**: `vocab_type` changed from `true` (boolean) to `1`
   (integer), and a new `architecture` field with
   `input_modalities`/`output_modalities` was added to the model object.
8. **props.json**: `build_info` is now `b11719-5aeac37a3` (was
   `b10731-0eadefebd`); `media_marker` changed (random per-run token).

## Known gap: `log_oom.txt` is a host-RAM failure, not a CUDA device OOM

`--ctx-size 999999` on a 2048-train model makes llama.cpp try to allocate a
21 GB KV cache and fail:

```
W common_fit_params: failed to fit params to free device memory: was unable to fit model into system memory by reducing context, abort
E ggml_aligned_malloc: insufficient memory (attempted to allocate 21488.50 MB)
E ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 22532325376
E ggml_backend_buft_alloc_buffer_n_default: failed to allocate CPU buffer of size 22532325376
E llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
E srv    load_model: failed to create_context with model '…'
```

On a CUDA host the same code path emits `ggml_backend_cuda_buffer_type_alloc_buffer`
and `ggml_cuda_host_malloc` instead of the CPU spellings, wrapping the same
`failed to allocate buffer for kv cache` /
`failed to create_context` lines. Task 9's grammar therefore matches on the
**backend-neutral** anchors that this fixture does contain, and additionally on
the CUDA spellings read from the pinned `b11719` source tree
(`ggml/src/ggml-cuda/ggml-cuda.cu`). A real CUDA OOM capture from `bonus` should
replace this file the first time one occurs after Task 16.
