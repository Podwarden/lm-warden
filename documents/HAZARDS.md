# Hazards worth reading before you load anything

Each of the following is a thing that costs an afternoon to diagnose and one
paragraph to prevent. They are true of LM Warden however it was deployed.

## Shared memory: tensor-parallel models need 2 GiB of `/dev/shm`

<!-- The `shared:` marker pairs in this file fence generated regions. Their text
     is shared with the PodWarden Hub catalogue listing for this app, and
     `scripts/sync-shared-docs.py` rewrites them -- a hand edit inside a pair is
     reverted on the next sync and fails CI in the meantime. Everything outside
     those markers is hand-written: edit it freely. -->

<!-- shared:shm-sigbus -->
vLLM's tensor-parallel workers communicate over POSIX shared memory in
`/dev/shm`. A container or Kubernetes pod that asks for nothing gets a **64 MiB**
`/dev/shm`, and any model with `tensor_parallel_size > 1` then dies with an
uncatchable `SIGBUS` inside `shm_broadcast.enqueue` — **minutes into serving,
not at startup**, because tmpfs allocates lazily. The process is gone with no
Python traceback, and the model looks like it worked.

**2 GiB is the floor.** A TP=4 engine has been observed holding ~1.1 GB of ring
buffer segments, on top of one `shm_broadcast` ring per message queue at 160 MiB
each. Two GiB clears that with headroom and sits far enough above the 64 MiB
default that an undersized `/dev/shm` can only mean nobody configured one.

- Raw `docker compose`: `shm_size: 2g` on the service that runs the engine.
- Kubernetes: a `medium: Memory` `emptyDir` mounted at `/dev/shm`.

Shared memory counts against the pod's and the node's memory, so size it against
what the node really has free; exceeding the limit evicts the pod. On startup the
control plane logs an `engine shm:` warning when `/dev/shm` is below the floor —
if a multi-GPU model dies without a traceback, that log line is the first place
to look.

Capping vLLM's message-queue chunk size is *not* the fix. It makes the ring fit
inside 64 MiB, and it costs throughput: payloads above the chunk limit fall back
to a slower copy path that multimodal requests hit constantly.
<!-- /shared:shm-sigbus -->

## One loaded model per GPU

<!-- shared:one-model-per-gpu -->
A GPU is claimed exclusively by the model loaded on it. A second load onto a busy
card is refused:

```
GPU 0 is already serving 'qwen2.5-1.5b' — unload it first, or load this model
on a free GPU. LM Warden runs one loaded model per GPU.
```

**This is an ownership claim, not a VRAM check.** It is decided in the control
plane before any engine process starts, so it is not a capacity problem and it
has no tuning workaround: lowering `gpu_memory_utilization` to make room does
not let a second model share the card, because nothing ever measures the room.
That misdiagnosis is an easy one to make — two 1 GB models on a 16 GB card
plainly *fit* — and the refusal happens anyway.

Over the API the refusal is **asynchronous**. `POST /api/models/{id}/load`
answers `202 {"status":"loading"}` like any other load, and the claim is
rejected a moment later — so the row lands in `failed` with the message above in
`last_error`, and a script that only checks the POST's status code will believe
it succeeded. Poll `GET /api/models/{id}` for the outcome.

Switching models on a card means unload, then load. The same error can also mean
a stale claim left by a model that is already gone; unloading it clears the
claim.
<!-- /shared:one-model-per-gpu -->

## `gpu_memory_utilization` reserves a fraction of the whole card

<!-- shared:gpu-memory-utilization -->
`gpu_memory_utilization` is a fraction of the **card**, not of the model. vLLM
reserves that share up front and fills whatever is not holding weights with KV
cache. At the default `0.9`, a 1.5B model whose weights are ~3 GB occupies
**~15 GB of a 16 GB card** — about 10.5 GB of it KV cache. Nothing is wrong,
and the engine log says so plainly:

```
Available KV cache memory: 10.55 GiB
GPU KV cache size: 395,264 tokens
```

Three consequences:

- **VRAM is a step function, not a curve.** The whole reservation happens at
  load. It does not grow with traffic, and a VRAM graph over time is flat
  between loads however busy the server is.
- **Sizing a card by model weights is wrong.** Size it by weights *plus* the
  context you intend to serve, or lower `gpu_memory_utilization` and accept
  fewer concurrent requests.
- **A card that looks 94 % full is not a leak.** It is the reservation, and the
  free-VRAM figure `nvidia-smi` reports for a serving card says almost nothing
  about how much more work that card could take.

llama.cpp has no equivalent flag — `llama-server` offers a layer count
(`--n-gpu-layers`), not a VRAM fraction — so on that engine the whole card is
treated as available and the lever is `--ctx-size` instead.
<!-- /shared:gpu-memory-utilization -->

## What fits on what

<!-- shared:what-fits-on-what -->
Measured on 16 GiB Ampere/Turing cards (RTX A4000 and Quadro RTX 5000, sm_86 and
sm_75), driving each model through a chat playground and judging the answer —
not by a health probe.

| Goal | What works | What does not |
|---|---|---|
| `openai/gpt-oss-20b` | **two** 16 GiB cards, TP=2, `max_model_len` 32000, `gpu_memory_utilization` 0.7–0.9 | one 16 GiB card — the 20B MoE does not fit |
| Llama 3.1 8B / Mistral 7B class on **one** card | a pre-quantized **AWQ-INT4** checkpoint (e.g. `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`) with `--enforce-eager`, `gpu_memory_utilization` 0.9 | the bf16 repo — weights are ~15 GiB and it OOMs during load |
| Qwen2.5 14B on one card | `Qwen/Qwen2.5-14B-Instruct-AWQ`, `--enforce-eager`, 0.92 | anything FP8 (see below) |
| Two models at once | one model per card, each fitting its own card | splitting one card between two models |

Three rules those rows encode:

1. **A bf16 8B model does not fit one 16 GiB card.** Weights alone are ~15 GiB.
   Either split it across two cards (TP=2) or use a quantized checkpoint.
2. **FP8 weights on sm_86 and older produce gibberish, not an error.** FP8 is
   emulated below compute capability 8.9; the model loads, streams tokens, and
   the output is garbage. Native FP8 starts at Ada (sm_89).
3. **`--enforce-eager` avoids the compile-time OOM and shortens the load.** It
   skips inductor autotuning, which is both where the tight single-GPU load
   dies and a large part of load latency.

These combos were validated against vLLM 0.20.0. The rules above have held
across base-image bumps; the exact numbers are worth re-checking on a newer
engine before you depend on them.
<!-- /shared:what-fits-on-what -->

## First loads are slow, and that is normal

<!-- shared:first-loads-are-slow -->
A model is not reported `loaded` when the process starts. The weights go to the
GPU, CUDA graphs are captured, and a warmup request is served end to end first —
so the first load of a model is measured in tens of seconds for a small one and
minutes for a large one. What pushes it into the minutes: kernel autotuning for
NVFP4/FP8 checkpoints on first start, vision-tower profiling for multimodal
models, tensor-parallel NCCL and shared-memory startup, and hybrid-attention page
size reconciliation.

The warmup probe budget (`VW_WARMUP_PROBE_TIMEOUT_S`, default 600 s) is
deliberately generous, because the cost is asymmetric. An engine that *dies* is
reported immediately by the on-exit callback whatever this value says, and a
probe that cannot connect fails at once — so a larger budget only extends how
long a **live** engine is allowed to finish starting. Set it too low and a
healthy engine is marked failed while its process keeps running and holds the
GPUs.
<!-- /shared:first-loads-are-slow -->

## Measuring instead of guessing

An afternoon spent bisecting `max_model_len` until the engine stops crashing
produces a number that works and no idea why.
`POST /api/models/{id}/stress` runs that search deliberately: it probes the
loaded model for the context length it can actually *serve* — needle
retrieval at several offsets, not "did it start" — and reports what it
established.

Three modes, and they differ in what they establish rather than only in how long
they take: `conservative` (crash budget 0, 5/5 confirmations), `quick` (the
same, crash budget 2), and `thorough` (crash budget 4, 7/7 confirmations, three
needle offsets, and the configuration sweep). **Only `thorough` can answer "what
should I set `max_model_len` to?"** — the sweep is the sole source of a
recommendation, and a cached cheaper run is never passed off as one.

When a run produces a recommendation that differs from what is running, the
results screen offers **Apply and reload** — unload, persist, load, as one
server-side operation — and it says plainly that applying invalidates the
measurement that produced it.
