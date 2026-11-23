# Architecture

One published port. Caddy fans out to internal-only `api` and `ui` containers:

| Path | Backend | Notes |
|---|---|---|
| `/` | FastAPI `/_landing` | The home page setting: the built-in welcome page, or (Off) a redirect to `/ui/` or a 404 |
| `/ui/*` | Next.js | Browser UI |
| `/api/*` | FastAPI | JWT-gated control plane |
| `/v1/*` | FastAPI | OpenAI-compatible proxy (token-gated) |
| `/healthz` | FastAPI | Liveness probe |
| `/readyz` | FastAPI | Readiness: `200` only when a model can serve |

Which replica answers a request, and the Claude Code router: [ROUTING.md](ROUTING.md).

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

## What is automatic, and what is not

A dead engine under a live wrapper is detected, its evidence is captured and
the model is reloaded without you. That watchdog probes the engine's `/health`
rather than trusting the process tree, and it exists because the engine core
died four times in one day while the `vllm serve` wrapper stayed alive and the
proxy kept forwarding to a corpse.

Other things are deliberately not automatic:

- The wall-clock request reaper is **off by default** (`request_max_wall_s`,
  set through `VW_REQUEST_MAX_WALL_S`; `0` disables it). A long request is
  yours to bound.
- A GPU is claimed by one model until you unload it. Nothing evicts a loaded
  model to make room for another; see
  [One loaded model per GPU](HAZARDS.md#one-loaded-model-per-gpu).

## Leaving is a `base_url` change

Apache-2.0, OpenAI-compatible on the way in. The weights sit in an ordinary
Hugging Face cache volume that `make export-hf-cache` hands you as a tarball
([Day-to-day](OPERATING.md#day-to-day)). The exact argv each engine was
launched with is a GET away (`GET /api/models/{id}/effective-argv`), so
reproducing a working configuration outside this product is copy and paste.

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

- **llama.cpp's version is fixed by the warden image** (`b12644`). Bumping it is
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

## Conversation identity (cobs2 L1)

The proxy gives every chat request a set of content ids: short keyed hashes of
what the conversation says, not of who sent it. They let the warden tell one
turn from the next, notice a fork, a compaction or a sub-agent, explain cache
misses, and place a conversation on a replica by its content. The code is in
`app/cobs2/` (canonical form, ids, key, lineage, reply capture) and in
`_forward` in `app/proxy/routes.py`.

### What is hashed

Every API shape the proxy accepts reaches `_forward` as an OpenAI chat body
(`/v1/messages` and `/v1/responses` are translated first), and that body is
turned into a canonical form before anything is hashed:

- **The head** is every `system` and `developer` message, wherever it sits,
  plus the `tools` list. It gets one id.
- **The body** is every other message, in order. Each message gets its own id,
  which depends only on that message: role, text, media (by SHA-256 of the
  bytes, or of the URL), reasoning, tool calls (name and arguments) and tool
  results. A reply hashed on its own therefore has the same id as that reply
  when the client sends it back in the next request.
- **Nodes** chain the body: node 0 is a hash of the head id and message 0, and
  node *i* is a hash of node *i-1* and message *i*. A node fixes everything
  before it, so "the deepest node we already know" is "how much of this
  conversation we have seen".

**Tool-call ids never enter a hash.** Harnesses make them up per request, so a
call is identified by its position in its own assistant message, and a tool
result by the position of its call in the nearest earlier assistant message
that has that id (-1 when none does).

Each id exists in two strengths:

- **Exact** follows the text as sent. Tool-call arguments and tool schemas keep
  the client's key order, because chat templates render them in that order and
  a reorder really does change the prompt.
- **Loose** ignores noise that harnesses rewrite between turns without changing
  the meaning. It drops reasoning, `<system-reminder>` and
  `<environment_context>` blocks, lines that hold only an ISO date or
  date-time, whitespace differences and known harness preambles (Aider's
  example conversation), and it sorts JSON keys in tool arguments and tools.
  The list of what is dropped is versioned (`LOOSE_TABLE_VERSION` in
  `app/cobs2/canonical.py`, now 1): change the list and bump it, because loose
  ids from two versions do not compare.

A request's `cache_salt`, when it sends one, is mixed into every id, so two
salts never share ids.

### The key

Every id is an HMAC-SHA-256 under a per-install network key, cut to 16 bytes
and prefixed with one epoch byte: 34 hex characters. Each kind of id (head,
message, node, and their loose forms) has its own label inside the HMAC, so
ids of different kinds never collide.

The key is 32 random bytes in `<data_dir>/cobs2_network_key` (next to
`vllm-warden.db`), mode 0600, written on first start as one line,
`v1:<64 hex characters>`. Two warden processes starting at once agree on one
key. An empty or corrupt file stops startup with an error instead of being
replaced: a new key would change every id. Back it up with the database. The
key is never logged and no API returns it.

### What is stored

No text. `request_history` gains these columns (migration
`0052_cobs2_l1.sql`), all hex ids or short fixed values:

| Column | Holds |
|---|---|
| `cobs2_head` | the head id (exact) |
| `cobs2_node` | the request's last body node (exact) |
| `cobs2_parent_node` | the node this request continued, forked or was compacted from |
| `cobs2_edge` | `new`, `continue`, `fork`, `compacted_from` or `subagent` |
| `cobs2_reply_msg` | the message id of the reply the warden generated (exact) |
| `cobs2_compaction_request` | 1 when this request asked the model for a summary |
| `cobs2_divergence_cause` | `compacted` or `harness_rewrite`, see below |
| `cobs2_divergence_cause_own` | the same cause, judged on the key's own traffic |

Migration `0053_sessions.sql` adds the sessions columns: `cobs2_session` (the
lineage session used), `cobs2_conv` (the conversation id: loose head + first
loose body node), `cobs2_shape` (sizes and counts, at most 200 bytes),
`cobs2_parent_req` (the request that produced the parent node),
`cobs2_sess_ref` (a keyed lookup of key + session), `cobs2_toolset` (a digest
of the offered tools), `relayed` and `upstream`, and the reserved cache-tier
columns (`served_*_tokens`, `load_s`, `node_id`). The cache summary reads the
divergence causes; the sessions API (below, and documents/API.md, "Sessions")
reads the rest.

The reply id is taken only from a complete reply: the engine sent a finish
reason, `[DONE]` or a whole JSON body. A reply that was cut short (the client
went away, the reaper ended it, the upstream failed, the engine sent an error
frame) gets no id, because it is not the message the client will send back.
A reply with more than about 4 MiB of text gets no id either.

### Edges and lineage

Each request is linked to what came before it in **its own session**, by one
edge:

- **new**: nothing in this session matches.
- **continue**: the request extends a node this session has seen (an identical
  retry is also a continue). A request whose prefix is unknown but which
  carries a reply this session produced also continues the turn that reply
  answered, flagged as a history rewrite, and stays in that conversation's
  lane: some harnesses rewrite the first message after the first turn.
- **fork**: the deepest known node already has a different next message, so
  the client went back and took another branch. Fork follows git: one session
  can carry several conversations side by side without them reading as forks.
- **compacted_from**: the body collapsed (no known node) and this looks like
  the conversation after a compaction, with the same head.
- **subagent**: a head this model and salt have not seen, while the client
  names a parent session, or another request of the same key with a different
  head is in flight.

The session is the client's session id (the same signal as the dashboard's
Session column), else a digest of the API token and the loose head (so a
changed reminder or date line in the system prompt keeps the session). Nothing continues across sessions: a user who reopens the same
conversation in a new harness session starts at `new`.

Lineage lives in memory only (`Lineage` in `app/cobs2/lineage.py`): ids and
digests, at most 50,000 entries per table, each forgotten 6 hours after it was
last seen. A loose index runs beside the exact one, so a harness that rewrote
an earlier message still continues at the right depth.

A compaction is recognised two ways:

- **By fingerprint.** The previous request ended with a user message that
  contained a known summarize prompt (Claude Code, OpenCode, Cline). The next collapsed
  request with the same head is `compacted_from` it. When a harness sends the
  summarize call under its own system prompt, the match is allowed across
  heads, but only in the shape "this head's last turn, then the summarize
  call", within 10 minutes, and only once.
- **By reply overlap.** For each recent conversation in the session (up to
  four, one per head) the warden keeps a sketch of its last reply: 64 keyed
  hash values over its first 2,000 five-word shingles, with a leading
  `<analysis>` block removed. A collapsed request whose first message overlaps
  that sketch by half or more (estimated Jaccard) is `compacted_from` that
  reply's node. The sketch is kept in memory and is never stored.

### Miss causes

The cache classifier (`app/cache_obs/classify.py`) uses the edge to name two
causes:

- **compacted**: the edge is `compacted_from`.
- **harness_rewrite**: the loose chain matched deeper than the exact one, and
  the exact node at that depth really differs from the one recorded (so a node
  the memory evicted does not count as a rewrite).

They replace an outcome only when it was `diverged` or `cold`, after every
other rule. A `lost` request keeps its label: it still had 256 or more
reusable tokens and the engine evicted them, which must stay visible. A `hit`,
`partial` or `misrouted` request keeps its label too. On a hit or partial the
cause goes in `cobs2_divergence_cause` instead: a `harness_rewrite` when the
prefix broke mid-prompt, a `compacted` always. A relabelled row carries its
cause there as well; `lost` and `misrouted` rows carry none. The cache summary counts
those causes per model in `divergence_causes`, over broken prefixes only. The
dashboard's Prompt cache panel shows these causes per model.

### The replica key

On a data-parallel model, a request with a harness session id is placed by
that session narrowed to one conversation. The narrowing is the **loose** head
id plus the **loose** first node (`conversation_key` in
`app/proxy/dp_affinity.py`), so a changed reminder or date in the system
prompt or the first message keeps the conversation on its replica and its KV.
The key does not depend on the body's size: when the full ids are skipped (the
size cap, or an error), the same two ids are computed from the head, the tools
and the first message alone. Without a session id the key is the existing hash
of the first user message, unchanged.

### Fail-open and the hot-path budget

Identity must never fail or slow a request:

- Every step is wrapped. Any error leaves the request's cobs2 fields empty, and
  the replica key falls back to the previous discriminator.
- Ids are computed after admission and before replica routing. A body over
  256 KiB is hashed in a worker thread, so a big prompt does not hold the event
  loop. Lineage runs on the loop.
- A request whose canonical form is over 2 MiB (UTF-8) gets no ids and no
  lineage; its reply still gets an id. The texts
  scanned for a compaction are bounded: the first and last 32k characters of
  the last user message, the first 64k characters of the first message, the
  first 256k characters of the reply.
- An id memo means each request mostly hashes only its new tail. For a
  200 KB coding-agent body the target is under 2 ms at p99 on a quiet machine
  (about 1.2 ms median measured on an M4); the first request of a process,
  with nothing memoised, is about 3 ms.

### Known limits

- **A restart forgets lineage.** Ids stay the same (the key is on disk), but
  the first request after a restart is `new`.
- **A per-request session signal gives only `new`.** When the only session
  signal is `x-client-request-id` and the client sends a fresh one per
  request, every request is its own session.
- **With auth off, the fallback session is shared.** Without a session id, the
  session is a digest of the token and the head, and every anonymous client
  uses the same token, so clients with the same system prompt share a session.
- **A request that fails upstream still enters lineage.** Lineage is recorded
  before the request is forwarded.
- **A request cancelled while it is hashed in the worker thread is not in
  request history.** Its slot and live row are released; it was never served.
- **`/v1/responses` keeps the old replica key.** That route supplies its own
  key (Codex's thread id), so the content ids do not narrow it yet.
- **A compaction may move a conversation's replica.** The first message
  changes, so the key does. Only the head's cache is lost, and the old KV is
  mostly useless after a compaction anyway.

## Sessions

The sessions pages and API (documents/API.md, "Sessions") are built from
`request_history` rows only, never from the in-memory lineage, so they survive
a restart.

- **Session.** A key plus the harness's session id. Traffic with no session id
  falls back to the cobs2 session (a digest of key and loose head), split into
  separate sessions at 30-minute idle gaps. A subagent session that names its
  parent (OpenCode, Codex) is folded into the root session, at most 8 levels
  up; Claude Code sends one id for the main agent and its subagents.
- **Conversation.** A lane inside a session: one cobs2 conversation id. The
  earliest lane with no parent is `main`. A lane whose first turn is
  `compacted_from` continues its parent as "<parent> after compaction".
- **Side call.** A one-turn conversation with no tool use that is not a
  compaction continuation (a harness's title or topic call). Side calls are
  collapsed into one row and are never a subagent's parent.
- **Inferred parent.** A lane that starts `new` or `subagent`, has no known
  parent, and has two or more turns or any tool use, gets as parent the latest
  earlier turn of another lane that called an `agent`-family tool and finished
  no more than 10 minutes before the lane started (or was still running). The
  link is drawn and returned as `inferred`.
- **Lineage unknown.** A turn whose parent node is in no row of the session
  (lineage forgot it: a restart or the 6 h window), a later turn of a lane that
  starts over as `new`, and a turn recorded without cobs2 ids (over the size
  cap) are marked, never given an invented parent.
- **Relayed rows.** In router mode, `/v1/messages` requests relayed to
  Anthropic are recorded too (`relayed = 1`): ids and lineage from the
  translated body, tools and the reply id from the captured response,
  usage and timings from the upstream. Recording runs in background tasks
  the relay never waits for. Their cache numbers are the upstream's prompt
  cache, so every local stats reader (cache panel, latency, throughput,
  prefill model, forest) ignores relayed rows; they appear only on the
  sessions pages, marked as relayed.
- **Tools.** `cobs2_toolsets` holds each offered toolset as
  `[hash, family, chars]` entries (no names). `session_tools` holds plain tool
  NAMES per key (never descriptions or schemas), shown only to that key's holder
  and the admin. Both are pruned with request history, with a one-day slack for
  the name refresh.
- **Harness.** Taken from how the harness sent its session id:
  `claude_code_header` and `anthropic_metadata` are Claude Code;
  `x_session_id` and `x_session_affinity` are OpenCode (Hermes and Aider
  too); `session_id_header` and `prompt_cache_key` are Codex (pi too);
  `openai_user` is the OpenAI SDK; anything else is `other`. A session takes
  its most common row harness.
- **Dead weight.** Measured when the request is canonicalised, in characters,
  into `cobs2_shape.dw = {dup, boil, reason}`. `dup` is the whole of each
  message whose exact id already appeared earlier in the same prompt; `boil`
  (`<system-reminder>` / `<environment_context>` blocks and date-only lines)
  and `reason` (reasoning parts) count only in the head and in first
  occurrences, so nothing is counted twice. Per-message results are memoised by
  message id (ids and three ints, never text; 65,536 entries, LRU), so a turn
  scans only its new messages. The shape is capped at 256 B; over the cap `dw`
  is dropped first. A failure drops `dw` only, never the shape, the row or the
  request. On `/v1/messages` thinking blocks are dropped before the engine, so
  those turns record `reason = 0`, local and relayed alike. Unused tools come
  from the toolset: tools offered but never called (per conversation for a
  turn or a conversation, per session for the session). Characters become
  tokens at the turn's own prompt-tokens-per-character ratio, and a turn's
  total is capped at its prompt.
- **Prompt anatomy.** Per turn: dead weight `d` first. With cached `c` and the
  previous prompt of the lane `q`, `new = min(max(0, p − q), p − c)` and
  `recomputed = p − c − new`. Cached, recomputed and new are then scaled by
  `1 − d/p`, so the parts sum to the prompt. An unknown cache makes the rest
  `unknown`.
- **Saved vs plain vLLM.** Plain `vllm serve` with data parallelism routes by
  load, not by prefix, so a turn finds its prefix on its replica 1 time in N.
  The figure is that expected value per local turn,
  `cached × (N − 1) / N ÷ cold rate`, with N the model's current
  `data_parallel_size`. It is 0 when N = 1 or `dp_affinity_enabled = 0` (LM
  Warden then routes by load too), and null for relayed-only sessions.
- **Engine replay.** This session's local turns alone, in start order, no
  other traffic, prefix = `min(lane's previous prompt, this prompt)`:
  LM Warden is what was measured; plain vLLM pays the same plus the expected
  replica misses above; llama.cpp keeps an LRU of S slots, one lane per slot
  (S = `max_num_seqs`, else `--parallel` / `-np`, else 4); mlx-lm keeps only
  the last prompt; oMLX keeps every prefix but its disk load time is not
  estimated. Time is recomputed tokens at each turn's cold rate; turns without
  one are counted in `skipped_time`.
- **Pauses and evictions.** Per conversation lane (parallel agents interleave),
  relayed turns excluded. A resume starts ≥ 60 s after the lane's previous turn
  ended. The cache lens is the key's own (`reusable_tokens_own`,
  `cache_outcome_own`) when recorded, else the fleet's; cached tokens fall back
  to the TTFT estimate on engines that report none. A resume is evicted when
  reusable ≥ 256 tokens and the outcome is `lost` or at least half the reusable
  prompt was prefilled again; with no cache data it is unassessed. The disk-tier
  hint keeps prefixes up to the top of the longest bucket with an eviction (for
  `4h+`, the longest evicted pause) and sums the lost time of evictions within
  it. The list's pause summary builds lanes from key, session and conversation
  before the 30-minute split, so a long pause in session-less traffic still
  counts.
