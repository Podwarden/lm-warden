# LLM Warden

**Your own OpenAI-compatible API, on your own NVIDIA GPUs. Two mainline engines
— vLLM and llama.cpp — behind one control plane, one port, and a browser UI.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Engines](https://img.shields.io/badge/engines-vLLM%200.26.0%20%C2%B7%20llama.cpp%20b10731-4b8bbe.svg)](#two-engines-and-the-model-that-made-us-add-the-second)
[![Deploy](https://img.shields.io/badge/deploy-Docker%20Compose-2496ed.svg)](documents/INSTALL.md)

![LLM Warden stats — host VRAM, GPU utilisation and power, and a week of GPU utilisation and power draw](assets/screenshots/01-stats-overview.jpg)

You have NVIDIA GPUs — in a rack, in a workstation, or two cards bought
eighteen months apart in a box under a desk. You want what is on them to be
reachable the way a hosted API is reachable: a base URL, a key, and a client
library that already exists. And you want the things a hosted API gives you as
a matter of course and a bare engine does not give you at all — a separate key
per consumer, usage attributed to it, and a straight answer about what the
hardware is doing — without one container per model, a
`--tensor-parallel-size` you arrived at by bisection, and `nvidia-smi` open in
a second terminal to find out why a request is slow.

LLM Warden is the control plane around the engines. Pull a model from
HuggingFace, load it, mint a key, open that key's page to see what it spent
and how long its requests waited, watch what the card is actually doing. One
published port, so your own TLS terminator, ingress, SSO or network policy
sits in front of it unchanged. Nothing leaves the host:
no account, no licence check, no analytics, and an
[offline install](documents/INSTALL.md#a7-offline--air-gapped-install) for machines with
no route out at all.

The engines themselves are upstream and unmodified — that is a hard rule, and
[the second engine exists because of it](#two-engines-and-the-model-that-made-us-add-the-second).

---

## What it is, and what it is not

It is a wrapper, and that is the whole claim. It does not make models faster,
it does not fork an engine, and it will not do anything to your tokens that
`vllm serve` would not. What it adds is the part nobody ships: model
lifecycle, per-key auth with usage accounting, and an honest view of the GPU.

**The shape of it.** Three containers behind one Caddy front door on a single
published port; JWT sessions with CSRF on the control plane and bearer tokens
on `/v1/*`; a SQLite store with numbered migrations; an engine watchdog that
probes `/health` rather than trusting the process tree; `mypy --strict`
against a committed baseline, and a pytest suite that runs in a container.
`make lint`, `make typecheck`, `make test` — all three on a clean checkout, no
host Python.

**Nothing reports to us.** No account, no licence check, no analytics, no
callback — [`install.sh`](install.sh) and
[`docker-compose.yml`](docker-compose.yml) are the entire deployment and you
can read both. No part of a request leaves the host. The stack makes outbound
calls only when you ask it to: huggingface.co to pull weights
(`HF_HUB_OFFLINE=1` stops even that), Docker Hub to list published vLLM tags
when you open the engine-version picker, and the release registry for the
images — which `docker load` replaces entirely, see
[Offline / air-gapped install](documents/INSTALL.md#a7-offline--air-gapped-install). On a
host with no route out, none of the three is needed.

**Leaving is a `base_url` change.** Apache-2.0, OpenAI-compatible on the way
in. The weights sit in an ordinary HuggingFace cache volume that
`make export-hf-cache` hands you as a tarball. The exact argv each engine was
launched with is a GET away (`/api/models/{id}/effective-argv`), so
reproducing a working configuration outside this product is copy and paste.

**What is automatic, and what is not.** A dead engine under a live wrapper is
detected, its evidence is captured and the model is reloaded without you —
that one exists because the engine core died four times in one day while the
`vllm serve` wrapper stayed alive and the proxy kept forwarding to a corpse.
Other things are deliberately not automatic: the wall-clock request reaper is
**off by default** (`request_max_wall_s`), and a GPU is claimed by one model
until you unload it.

The whole path — install, wizard, key, register, pull, load, a real completion
— was walked end to end from the published release, on a host it had not been
developed on, following only what was written down. Four things the
documentation did not say then have sections of their own now:
[a first run with no browser](documents/API.md#first-run-without-a-browser), that
[pull and load are separate asynchronous steps](documents/API.md#adding-a-model-from-the-api)
and pull progress is an SSE stream, that
[a GPU serves one loaded model at a time](documents/HAZARDS.md#one-loaded-model-per-gpu),
and that
[`gpu_memory_utilization` is a fraction of the whole card](documents/HAZARDS.md#gpu_memory_utilization-reserves-a-fraction-of-the-whole-card).
[documents/INSTALL.md](documents/INSTALL.md) is that kind of walk, recorded in full: every command
run, every block of output as the terminal printed it.

## Will it run on my hardware?

NVIDIA only — no ROCm, no Metal, no CPU serving
path. vLLM's own support matrix applies unchanged, because it is upstream's
image. **The floor is Turing (sm_75)**: the base image ships CUDA 13, which
dropped Maxwell, Pascal and Volta, so a GTX 1080 Ti, a Titan X or a V100 will
not work here and no rebuild changes that. Above that floor the llama.cpp
binary shipped in the api image carries native SASS for **every architecture
CUDA 13 can target** — Turing through Blackwell, including Ada, Hopper and the
RTX 50-series — plus PTX, so a card newer than this release JIT-compiles on
first load instead of finding no backend at all. Narrowing the list to your own
cards is a build argument and makes the build much shorter:
[the build works from a plain clone](.github/CONTRIBUTING.md#build-from-source), which
is also where the current list is written down. And FP8 weights on Ampere are numerically
broken upstream, not slow: the model loads, streams tokens, and emits garbage.
The product warns about that rather than letting you discover it.

Above the floor, the question is what fits. Measured on 16 GiB cards,
[What fits on what](documents/HAZARDS.md#what-fits-on-what) says which model classes run
on one card, which need two, and which quantisations to avoid; the rest of
[documents/HAZARDS.md](documents/HAZARDS.md) is what to know before the first load. If none of
this applies to your machine, the next section says so plainly.

## Don't use this if…

None of these are solvable by configuration.

- **You have no NVIDIA GPU.** There is no ROCm, Apple Silicon or CPU-serving
  path. `--gpus none` starts the control plane for evaluation and CI; nothing
  can be loaded on it.
- **You need to serve one model across several machines.** This is a
  single-host Compose stack. Multiple GPUs in one box, yes; multi-node, no.
- **You want a hosted API.** Nobody operates this for you. You supply the
  hardware, the driver, the disk and the electricity.
- **You need a backend that is not vLLM or llama.cpp** — TensorRT-LLM, SGLang,
  MLX, Ollama's runtime. The seam to add one is real and small, but nothing
  else ships today.
- **You want several models resident on one card.** One loaded model per GPU is
  an ownership rule enforced before an engine starts. No amount of tuning gets
  past it.
- **You want tensor parallelism out of llama.cpp.** It splits layers, not
  tensors. For a model too large for one card, vLLM is still the right answer.
- **Your host is not Linux x86_64.** The api image is built on upstream's CUDA
  base and needs both.

---

## Quick start

A Linux host with Docker, Docker Compose v2.24+ (v5.x is fine), an NVIDIA GPU,
the NVIDIA Container Toolkit, and 40 GB free where Docker keeps its images:

```bash
git clone https://github.com/Podwarden/vllm-warden.git
cd vllm-warden
./install.sh
make smoke      # 200s across / /_landing /ui/ /api/csrf /healthz
```

The installer checks the host, lets you pick GPUs, generates the secrets, pulls
the release images and offers to start the stack — every route, every flag and
the offline install are in [documents/INSTALL.md](documents/INSTALL.md). Open
`http://YOUR-HOST:8080/ui/`; a first-run wizard covers GPU selection, a
HuggingFace token and your admin account. Then **Models → Add model**.

Your OpenAI-compatible endpoint is live at:

```
http://YOUR-HOST:8080/v1/chat/completions
```

Point any OpenAI client at it — LangChain, OpenWebUI, the `openai` SDK, your
agents. Only the `base_url` and the key change.

Scripting the whole thing instead of clicking?
[First run without a browser](documents/API.md#first-run-without-a-browser) is the
six-call version, and
[Adding a model from the API](documents/API.md#adding-a-model-from-the-api) is the rest.

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
dies with `ValueError: There is no module or parameter named
'embed_tokens.weight_packed'`. The model card's own remedy is a newer vLLM
**plus a Python monkeypatch applied to the interpreter before the server
starts**. Under the invariant that is not "hard", it is forbidden. So on vLLM
this model is unsupported, and stays unsupported until upstream adds the format.

The same authors publish a GGUF conversion, and their model card says it runs
unmodified in llama.cpp. It does. Measured on one 16 GiB RTX A4000, the
IQ3_XXS quant with its vision projector holds `--ctx-size 8192` at 12,039 MiB
of 16,376 MiB with no CPU offload, and passes a multi-turn chat, arithmetic,
format-compliance and image-description check through the product's own
`/v1/chat/completions`, not against the engine port. The full measurement is
in [documents/ARCHITECTURE.md](documents/ARCHITECTURE.md#two-engines-and-the-model-that-made-us-add-the-second).

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

---

## What you get

A bare engine is a single-model process. This is what sits around it — and
once the hardware is bought, a request costs electricity rather than a
per-token line on someone's invoice.

| A bare engine | With LLM Warden |
|---|---|
| One model per container, restart to switch | Register, pull, load and unload from the browser |
| One engine, take it or leave it | vLLM **and** llama.cpp, chosen per model |
| A single shared API key, or none | A key per consumer, each with its own page, priority lane, pause switch and rotation grace window |
| No record of who used what | Requests, prompt tokens and completion tokens rolled up per key and per client IP, plus each key's queue wait and latency |
| No way to see what a client actually sent | God mode: an opt-in, in-memory live view of one key's prompts and completions, docked on that key's page and streaming only while open; off by default |
| A port per engine to expose | One published port — your own TLS terminator, ingress, SSO or network policy goes in front of it |
| `nvidia-smi` in a second terminal | Per-card VRAM, utilisation, power, temperature against the driver's own throttle point, PCIe width, ECC, NVLink |
| No idea why a request is slow | Live request table, TTFT and duration distributions from the proxy, KV-cache pressure and preemptions where the engine reports them |
| Hand-edited `--tensor-parallel-size` | Guided setup, a fit preview before you pull, and a stress test that measures the ceiling |
| A dead engine nobody notices | `/health` watchdog, evidence captured, model reloaded |

- **Browser UI** — models, live engine logs, chat playground, stats, and a page
  per API key.
- **OpenAI-compatible gateway** at `/v1/*` — drop-in for any existing client.
  `GET /v1/models` reports `max_model_len`, so clients stop guessing the context
  window.
- **Model lifecycle** — pull from HuggingFace, hot-swap without restarting the
  container, per-model settings, GGUF on either engine.
- **Per-key auth and accounting** — one token per consumer, each with its own
  priority lane and rotation grace window. Requests, prompt
  tokens and completion tokens are attributed to the key that spent them and
  to the IP that called.
- **A page per key** — rename it, set its priority, pause it (its clients get
  `403 token paused` until you resume it; requests already running finish),
  test, rotate or delete it, and see the keys it replaced. Its charts show
  tokens and requests per minute, queue wait and latency (first token or full
  response, median and 95th percentile) over 1 h, 6 h, 24 h, 7 d or any period
  you drag or type, with the keys it was rotated from included by default.
  Binning happens on the server, so a week is still a few hundred points.
  Per-request timings are recorded per key from v2026.09.18.1 on; an earlier
  period is marked "Not recorded" rather than drawn as an idle one.
  `GET /api/tokens/{id}/series` returns the same series the page draws.
- **Request-level visibility** — a live table of what is in flight: key,
  client IP, model, context used against `max_model_len`, prefill or decode,
  and whether the caller has already disconnected. Finished requests are
  persisted with TTFT, duration and how each one ended. All of that is
  metadata: the `request_history` table has no column that holds prompt or
  completion text.
- **Reading what a client actually sent** — god mode is an opt-in live view of
  prompts, completions and inline images, off by default and held only in a
  bounded in-memory ring. It lives in a dock at the bottom of each key's page
  and shows that key's traffic alone, its earlier keys included; it streams
  only while the dock is open, and a closed dock holds no connection. It, and
  the one other feature that can capture content, are described together with
  their bounds under
  [Where request content can end up](documents/OPERATING.md#where-request-content-can-end-up).
- **HuggingFace cache manager** — see what is on disk, garbage-collect orphans,
  export and import the whole cache as a tarball.
- **GPU observability** — per-card telemetry with engine process attribution,
  plus an interconnect graph read from `nvidia-smi topo -m`, so you can see
  whether two cards have a path to each other before splitting a model across
  them.
- **Single-port topology** — one Caddy front door on `:8080` serves UI, control
  API and the OpenAI shim, so one reverse-proxy rule covers everything. It does
  not want to own your edge: terminate TLS where you already terminate it.
- **Runs with no route out** — three image tarballs plus an optional
  model-cache tarball are the entire transport, and `HF_HUB_OFFLINE=1` stops
  the stack contacting huggingface.co at all. See
  [Offline / air-gapped install](documents/INSTALL.md#a7-offline--air-gapped-install).

<table>
<tr>
<td width="50%"><img src="assets/screenshots/05-quant-fit.jpg" alt="Add model: every quant of a 35B GGUF repo, each marked fits or won't fit"><br>
<b>Pick the quant your hardware can actually run.</b> Point it at a repo and it lists
every file with a verdict — here a 35B where BF16, Q8_0 and MXFP4 are all too big
and <code>UD-IQ2_M</code> fits.</td>
<td width="50%"><img src="assets/screenshots/06-quant-fit-combined.jpg" alt="The same 20 GiB file marked fits once four GPUs are selected"><br>
<b>Tick more cards and the budget adds up.</b> The 20.22 GiB file that will not fit
on one 16 GiB card fits across four — the verdict recomputes as you select GPUs.</td>
</tr>
<tr>
<td><img src="assets/screenshots/03-gpu-cards-mixed.jpg" alt="Two mismatched GPUs side by side with differs markers and a reduced-link-width warning"><br>
<b>Mismatched cards, read honestly.</b> An Ampere A4000 beside a Turing Quadro:
architecture, VRAM, power cap and ECC all differ, and the reduced PCIe link width
is called out rather than buried.</td>
<td><img src="assets/screenshots/04-gpu-cards-four.jpg" alt="Four identical A4000s with per-card clocks, temperature, power and fan"><br>
<b>Per card, not per host.</b> Clocks, temperature, power, fan, driver and CUDA for
each card, with the throttle threshold marked on the temperature bar.</td>
</tr>
<tr>
<td><img src="assets/screenshots/07-latency-and-cache.jpg" alt="Latency distributions, KV cache occupancy and preemption rate"><br>
<b>Latency measured at the proxy.</b> TTFT, inter-token and duration
distributions the same way for every backend — plus KV pool occupancy, prefix
cache hit rate and the preemption rate.</td>
<td><img src="assets/screenshots/08-interconnect-and-keys.jpg" alt="Interconnect graph and per-key token usage table"><br>
<b>How the cards reach each other.</b> The PCIe topology, with negotiated lane
widths on each edge — worth knowing before splitting a model. Below it, token
usage per API key.</td>
</tr>
<tr>
<td><img src="assets/screenshots/02-models-list.jpg" alt="Two models loaded on separate GPUs"><br>
<b>One loaded model per GPU.</b> Each model owns its card; the list shows which
index each one holds.</td>
<td><img src="assets/screenshots/09-settings-presets.jpg" alt="Model settings with hardware presets and a suggest-values control"><br>
<b>Starting points, not blank fields.</b> Presets for common card/model shapes,
and a suggestion pass driven by the model config and the VRAM actually detected.</td>
</tr>
<tr>
<td><img src="assets/screenshots/10-token-details.jpg" alt="A key's page over 7 days: priority P0 to P9, details, rotation history listing two earlier keys, a history strip with rotation markers and a selected window, and tokens-per-minute and requests-per-minute charts"><br>
<b>Every key has its own page.</b> Rename, pause, test, rotate or delete it and set
its priority. The history strip spans the key's whole life, earlier keys
included, with each rotation marked — drag the window to chart any period, or
type exact times.</td>
<td><img src="assets/screenshots/11-token-timings.jpg" alt="The same page over the last hour: tokens and requests per minute, queue wait, and time-to-first-token latency with the median solid and the 95th percentile dashed"><br>
<b>What one key waited for.</b> Queue wait and latency — first token or full
response — per key, median and 95th percentile, next to the tokens and requests
it sent. A bin with no requests is left empty rather than drawn as zero.</td>
</tr>
<tr>
<td colspan="2"><img src="assets/screenshots/12-token-godmode-dock.jpg" alt="The god-mode dock open at the bottom of a key's page, live, replaying that key's recent requests and streaming a response as it is generated"><br>
<b>God mode, one key at a time.</b> Open the dock at the bottom of a key's page to
watch that key's prompts and responses as they happen, earlier keys included.
Close it and the stream stops — nothing is watched in the background. Off by
default (<code>VW_GODMODE_ENABLED</code>), and held only in memory.</td>
</tr>
</table>

---

## Where request content can end up

By default no prompt or completion text is stored anywhere. `request_history`
— the table behind the requests chart and the per-key rollups — has no column
that holds it. Two diagnostic features can capture content: god mode
(`VW_GODMODE_ENABLED`), an in-memory ring watched one key at a time from a
dock on that key's page, streaming only while the dock is open; and the
content log (`VW_CONTENT_LOG_ENABLED`), which writes to disk and only for
token ids on an explicit allowlist. Both are off by default, and with both off
the proxy's forward path is the code it would be in a build that never had
them. Their bounds, and what you must arrange yourself before switching the
second one on, are under
[Where request content can end up](documents/OPERATING.md#where-request-content-can-end-up).

---

## Where to go from here

Each of these is one hop from here and says what it holds.

- [documents/INSTALL.md](documents/INSTALL.md) — the step-by-step install manual, recorded from two
  real installs: the published images (Path A), a build from source (Path B),
  the unattended flags, the no-clone one-liner, the offline / air-gapped
  install, what `make uninstall` does and does not free, and a symptom-to-cause
  table.
- [documents/API.md](documents/API.md) — driving it without a browser: the six-call first run,
  minting and managing a key (rename, pause, rotation history, its usage and
  timing series, its god-mode stream), and register / pull / load from the API,
  including the four fields only a llama.cpp row has.
- [documents/HAZARDS.md](documents/HAZARDS.md) — five things that cost an afternoon to diagnose and
  a paragraph to prevent: `/dev/shm` and tensor parallelism, one loaded model
  per GPU, what `gpu_memory_utilization` really reserves, what fits on a 16 GiB
  card, why first loads are slow — and how to measure `max_model_len` instead
  of bisecting it.
- [documents/OPERATING.md](documents/OPERATING.md) — the day-to-day `make` targets, upgrading, the
  URLs once it is running, HTTP against HTTPS, managing API keys from their
  pages, and where request content can end up.
- [documents/ARCHITECTURE.md](documents/ARCHITECTURE.md) — one port and three containers, drivers
  against backends, and the full account of the two engines and what the second
  one costs.
- [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) — building both images from source, how
  long that takes and how to make it shorter, the dev targets, and how to add a
  third backend.

## License

[Apache License 2.0](LICENSE).

## Trademarks

vLLM is a project of the [vLLM team](https://github.com/vllm-project/vllm).
llama.cpp is a project of
[ggml.ai and its contributors](https://github.com/ggml-org/llama.cpp). PodWarden
is a trademark of its operators. LLM Warden is not affiliated with or endorsed
by any of them.
