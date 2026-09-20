# Architecture

One published port. Caddy fans out to internal-only `api` and `ui` containers:

| Path | Backend | Notes |
|---|---|---|
| `/` | FastAPI `/_landing` | Public landing page (can be disabled) |
| `/ui/*` | Next.js | Browser UI |
| `/api/*` | FastAPI | JWT-gated control plane |
| `/v1/*` | FastAPI | OpenAI-compatible proxy (token-gated) |
| `/healthz` | FastAPI | Liveness probe |

Routing lives in `deploy/caddy/Caddyfile`. The `api` container shares the host
PID namespace so GPU process attribution can map host PIDs back to
supervisor-tracked engine workers.

Inside the api container, two axes compose rather than multiply:

- a **driver** answers *where* an engine process runs — an in-container
  subprocess (the default) or a sibling container over the Docker socket;
- a **backend** answers *which program runs*, with what argv, env, health probe,
  log grammar and capabilities.

Adding either is O(1), not O(n×m). Both engines speak OpenAI over HTTP and both
serve `/health`, which is why the proxy, the watchdog and the warmup probe
needed no engine-specific code at all. `GET /api/system/backends` reports what
this build has and — separately — what the *active driver* will actually let it
do, so a control that cannot work is disabled with a reason instead of a load
failing forty seconds in.

### What the driver does and does not separate

The two drivers are not equally exposed, and the difference is worth stating
exactly rather than as "docker is more isolated".

Under the **local-subprocess** driver the engine is a child process inside the
api container. It shares that container's filesystem, so it can read and write
everything the warden can — the SQLite database, the JWT secret, the HF token.
That is inherent to the design and is documented, not a defect: a launch is argv
plus env, and an engine started with attacker-chosen argv in the warden's own
container is equivalent to code execution in the warden.

Under the **docker** driver the engine is a sibling container. It is handed the
model-cache volume and nothing else: **the warden's data volume is not mounted
into it** (#265 — it used to be, read-write, which made this driver no better
than the one above). What it still shares with the control plane is the
control-plane network — it must, because that is how the proxy and the health
probe reach it — the host IPC namespace (`ipc_mode=host`, for tensor-parallel
shared memory) and the GPUs it was pinned to. So the honest claim is: under the
docker driver the engine cannot read the database or the signing secret; it is
not sandboxed from the warden's *API*, which it can reach over that network like
any other client on it.

Neither driver gives the engine the Docker socket. The warden holds it.

---

## Two engines, and the model that made us add the second

There is one product invariant, and everything below follows from it:

> **We ship mainline runtimes. No monkeypatching, ever.**

No patched engine image, no vendored fork, no `sh -c "patch && exec …"`
entrypoint, no `sitecustomize.py`, no `LD_PRELOAD`. A launch is argv plus
environment and nothing else. That is enforced by the type the launch path
speaks — it has no field to put a patch in — and by a test that fails if the
vocabulary reappears.

The invariant is easy to hold right up until a model you want will not load.

**The worked example.** `ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ` quantises the
embedding table. Mainline vLLM's weight loader has no entry for it, so the load
dies with:

```
ValueError: There is no module or parameter named 'embed_tokens.weight_packed'
in Qwen3_5Model
```

The model card's own remedy is a newer vLLM **plus a Python monkeypatch applied
to the interpreter before the server starts**. Under the invariant that is not
"hard", it is forbidden. So on vLLM this model is unsupported, and stays
unsupported until upstream adds the format.

The same authors publish a GGUF conversion, and their model card says it runs
unmodified in llama.cpp. It does. Measured, on one 16 GiB RTX A4000:

| | |
|---|---|
| Repo | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` |
| Weights | `Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf` — 10,094,357,632 B |
| Vision projector | `mmproj-Qwen3.8-27B-BF16.gguf` — 931,146,528 B |
| Context | `--ctx-size 8192`, requested and held |
| Steady-state VRAM | 12,039 MiB of 16,376 MiB |
| CPU offload | none — llama.cpp's own sizing kept every layer on the card |

Coherent multi-turn chat across a deliberate distractor, correct arithmetic,
exact format compliance, and a correct description of an image generated for the
test — through the product's own `/v1/chat/completions`, not against the engine
port.

The claim is narrow: *a model the product genuinely could not serve is served,
unmodified, by a different mainline backend.* One row, one field. It is not
that vLLM is bad. It is that being locked to a single engine does not make the
other models slower — it makes them unavailable.

Three consequences:

- **GGUF quantisation fits larger models onto smaller cards.** A 27B model on
  one 16 GiB card is a 3-bit quant's whole point.
- **Older GPUs stay useful as vLLM's support moves on.** Turing is not on
  vLLM's happy path any more; llama.cpp still serves it.
- **Heterogeneous boxes are ordinary, not exotic.** An Ampere card next to a
  Turing card is what a machine that grew looks like, and the stats page reads
  them as two different cards rather than averaging them into a fiction.

### What the second engine costs

**llama.cpp publishes no latency histograms at all.** vLLM exposes four; the
llama.cpp `/metrics` endpoint has none. The choice was to leave the latency
panels blank for GGUF models, or to measure latency somewhere both engines have
in common — so the panels now come from the **proxy's own** measurements: TTFT
at the first streamed frame, duration at the end of the stream, persisted per
request. GGUF models got a latency panel for the first time because of that gap.
What is given up is the engine's per-token resolution; the stored inter-token
latency is a mean per request and is labelled that way everywhere it appears.

The same rule runs through the rest of the metrics: a number the engine does not
report renders as **"not reported"**, never as `0`. A zero says "plenty of KV
headroom" or "the engine is idle"; the truth is "this engine is silent about
that", and an operator who acts on the first has been actively misled. On
llama.cpp that means KV-cache usage, sleep state, preemptions and MFU are blank
rather than reassuring.

Two more limits worth knowing before you pick it:

- **llama.cpp's version is fixed by the warden image** (`b10731`). Bumping it is
  a warden release, not a per-model choice. vLLM's version can be pinned per
  model, but only under the Docker engine driver — the UI disables the control
  and says which of the two reasons applies rather than offering a switch that
  does nothing.
- **A GGUF-only repository ships no tokenizer**, so token accounting falls back
  to a character estimate (which also drives per-key usage figures). Set
  `tokenizer_repo` to the upstream safetensors sibling and counts are exact,
  local and free. The degradation is reported rather than hidden.

Choosing between them is mostly automatic: pick a `.gguf` file in **Add model**
and the wizard pre-selects llama.cpp. It is a pre-selection, not a rule — a GGUF
that vLLM can also serve stays available to vLLM.
