# Driving LM Warden from the API

The browser UI is a client of the control API and nothing more. Everything it
does — the first-run wizard, minting a key, registering, pulling and loading a
model — is a handful of `curl` calls, and this document is those calls, for a
script, a CI job or an agent that will never open `/ui/`.

[INSTALL.md](INSTALL.md) walks the same first run and the same model add as a
recorded transcript, with the failures it hit on the way and what they meant.

## First run without a browser

<!-- The `shared:` marker pairs in this file fence generated regions. Their text
     is shared with the PodWarden Hub catalogue listing for this app, and
     `scripts/sync-shared-docs.py` rewrites them -- a hand edit inside a pair is
     reverted on the next sync and fails CI in the meantime. Everything outside
     those markers is hand-written: edit it freely. -->

<!-- shared:headless-first-run -->
The wizard is a browser flow, but it is only a client of `/api/setup/*` — so a
first run can be scripted end to end. Everything below runs against a freshly
started stack on `http://127.0.0.1:8080` and needs nothing but `curl` and `jq`.

`/api/setup/*` is exempt from the CSRF check, so these six calls need no token
and no cookie jar. They are a strict state machine: posting out of order returns
`400 not at <step> step (current: <where you are>)`, and `GET /api/setup/state`
always tells you where that is — so an interrupted run resumes rather than
restarts.

```bash
W=http://127.0.0.1:8080

# 1. Where are we? A fresh install answers {"step":"welcome","done":false}.
curl -s $W/api/setup/state

# 2. Acknowledge the welcome step.                    -> {"step":"gpus"}
curl -s -X POST $W/api/setup/welcome

# 3. List the GPUs the container can see, with their indices.
curl -s $W/api/setup/gpus
# [{"index":0,"name":"NVIDIA RTX A4000","memory_total_mib":16376,
#   "memory_used_mib":0,"utilization_pct":0}, ...]

# 4. Choose which of those indices the engine may use.  -> {"step":"hf_token"}
curl -s -X POST $W/api/setup/gpus -H 'Content-Type: application/json' \
     -d '{"allowed_gpu_indices":[0,1]}'

# 5. HuggingFace token. JSON null is valid and is the right answer unless you
#    need gated models (Llama, Mistral, gpt-oss).      -> {"step":"admin"}
curl -s -X POST $W/api/setup/hf_token -H 'Content-Type: application/json' \
     -d '{"hf_token":null}'

# 6. Create the admin account.                          -> {"step":"done"}
#    Password rules: at least 12 characters and at most 72 BYTES. The upper
#    bound is bcrypt's, and it is rejected rather than silently truncated —
#    worth knowing before you generate a long passphrase.
curl -s -X POST $W/api/setup/admin -H 'Content-Type: application/json' \
     -d '{"username":"admin","password":"CHANGE-ME-to-12-plus-chars"}'
```

`GET /api/setup/gpus` returns `404` once setup is done — deliberate, so a
completed install does not report its hardware to anonymous callers.

**Then mint an API key.** Unlike `/api/setup`, the rest of the control API does
enforce CSRF, so this is a two-header call: the CSRF token from `/api/csrf`
plus the JWT from the login.

```bash
# The response key is `csrf`, NOT `csrf_token`. Reading the wrong field sends
# an empty header and the server answers a flat
# `403 {"detail":"csrf token invalid"}` that says nothing about which field
# was wrong — so this one line is worth copying exactly.
CSRF=$(curl -s -c jar $W/api/csrf | jq -r .csrf)

JWT=$(curl -s -b jar -c jar -X POST $W/api/auth/login \
        -H 'Content-Type: application/json' \
        -d '{"username":"admin","password":"CHANGE-ME-to-12-plus-chars"}' | jq -r .access_token)

curl -s -b jar -X POST $W/api/tokens \
     -H 'Content-Type: application/json' \
     -H "Authorization: Bearer $JWT" -H "X-CSRF-Token: $CSRF" \
     -d '{"name":"my-first-key"}'
# {"id":"53e7011c…","name":"my-first-key","plaintext":"vw_su3yl…",
#  "prefix":"vw_su3yl","expires_at":"2027-09-06 17:01:32", …}
```

**`plaintext` is shown exactly once.** Only a SHA-256 hash is stored, so the
token list can never show it again — save it now, or mint another key. It is
the bearer token for `/v1/*`:

```bash
curl -s $W/v1/models -H "Authorization: Bearer vw_su3yl…"
```

One more thing a script needs to know: `/v1/*` is bearer-gated and CSRF-exempt
— it is an API for programs, not for browsers.
<!-- /shared:headless-first-run -->

---

## Adding a model from the API

**Models → Add model** in the UI drives exactly these calls. Every one is
JWT-gated and CSRF-checked; `$JWT` and `$CSRF` are the two values produced by
the login sequence in *First run without a browser*, and `AUTH` below is just
those two headers.

Register, pull, load are **three separate, asynchronous steps**. Each returns
`202` immediately and reports progress somewhere else — the row's `status`
walks `registered → pulling → pulled → loading → loaded`, and you poll
`GET /api/models/{id}` for it.

```bash
# -b jar is REQUIRED, not optional: the CSRF token is bound to the cookie the
# jar holds, and the header alone gets you 403 {"detail":"csrf token invalid"}.
AUTH=(-b jar -H "Authorization: Bearer $JWT" -H "X-CSRF-Token: $CSRF"
      -H 'Content-Type: application/json')

# 1. Register. `gpu_indices` is required and must be a subset of what you
#    allowed in the wizard. `backend` defaults to "vllm"; the other value is
#    "llamacpp", which wants a GGUF `filename`.
curl -s -X POST $W/api/models "${AUTH[@]}" -d '{
      "served_model_name": "qwen2.5-1.5b",
      "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct",
      "gpu_indices": [0],
      "max_model_len": 8192
    }'
# 201 {"id":"9895a1829559533c","served_model_name":"qwen2.5-1.5b",
#      "status":"registered"}

M=9895a1829559533c

# 2. Pull the weights from HuggingFace.        202 {"status":"pulling","force":false}
curl -s -X POST $W/api/models/$M/pull "${AUTH[@]}"

# Progress is a SERVER-SENT EVENT stream — `data: {…}` lines, one per second,
# not a JSON document. Read it line by line; piping it to `jq` will not work.
curl -sN $W/api/models/$M/pull/progress -H "Authorization: Bearer $JWT"
# data: {"status": "pulling", "bytes": 2126013055, "total": 3098973447, ...}
# data: {"status": "pulled",  "bytes": 3098973447, "total": 3098973447, ...}

# 3. Load it onto the GPU.                     202 {"status":"loading","port":10000}
curl -s -X POST $W/api/models/$M/load "${AUTH[@]}"

# 4. Watch it become `loaded` (or `failed`, with `last_error` saying why).
curl -s $W/api/models/$M -H "Authorization: Bearer $JWT" | jq '.status, .last_error'

# 5. Free the GPU again.                       202
curl -s -X POST $W/api/models/$M/unload "${AUTH[@]}"
```

Loading is not instant: the weights go to the GPU, CUDA graphs are captured and
a warmup request is served before the model is reported `loaded`. Tens of
seconds is normal for a small model, minutes for a large one.

`POST /api/models/fit-preview` answers "will this fit?" before you pull
anything, returning a `green`/`yellow`/`orange`/`red` verdict with the
arithmetic behind it.

**On a llama.cpp row**, four fields have no vLLM equivalent and are worth
knowing:

- `filename` — point it at **shard one** of a split set
  (`…-00001-of-000NN.gguf`) and llama.cpp finds the rest itself.
- `mmproj_filename` — the vision projector, a separate GGUF beside the weights.
  A vision model started without it loads, serves, and silently ignores every
  image, so a set-but-missing projector is a hard error before any process
  starts.
- `tokenizer_repo` — the safetensors sibling, for exact token accounting.
- `n_gpu_layers` — leave it NULL and llama.cpp sizes itself to the card. An
  explicit integer is a deliberate partial CPU offload: it works, it is
  llama.cpp's real differentiator, and it is a performance cliff. It is never
  chosen for you.

A load refused because the card is already serving another model is refused
**asynchronously** — the `POST` still answers `202`. Why, and what to do about
it, is under [One loaded model per GPU](HAZARDS.md#one-loaded-model-per-gpu).
`POST /api/models/{id}/stress`, which measures the context length a loaded
model can actually serve rather than the one it starts with, is described under
[Measuring instead of guessing](HAZARDS.md#measuring-instead-of-guessing).

### Data-parallel models

To serve several replicas of a model that fits on fewer GPUs, set
`data_parallel_size` and list all the GPUs; `tensor_parallel_size` is derived
so that `tensor_parallel_size x data_parallel_size == len(gpu_indices)`
(anything else is a 422). vLLM only.

```bash
# A 7-GPU host: seven one-GPU replicas behind one port.
curl -s -X POST $W/api/models "${AUTH[@]}" -d '{
      "served_model_name": "qwen2.5-1.5b",
      "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct",
      "gpu_indices": [0,1,2,3,4,5,6],
      "data_parallel_size": 7,
      "dp_affinity_enabled": true,
      "dp_spill_threshold": 64
    }'
```

`dp_affinity_enabled` (default `true`, a JSON boolean) and `dp_spill_threshold`
(default unset, which means `max(1, max_num_seqs / 4)`, so 64 with vLLM's
default) are optional and can be changed on a loaded model through
`PATCH /api/models/{id}/settings` without a reload. Do not put
`--tensor-parallel-size`, `--data-parallel-size` or `--pipeline-parallel-size`
in `extra_args`; that is a 422. The proxy pins each conversation to one replica
(by `metadata.user_id`, then `user`, then `X-Session-Id`, then a hash of the
first user message) and spills to the least-loaded replica above the threshold;
a client that sends its own `X-data-parallel-rank` header is left alone.
`GET /api/models/{id}/dp-routing` (JWT) returns the per-replica routing counters
joined with the engine's per-replica load, KV-cache and prefix-cache hit rate.
Details and tuning are in [ROUTING.md](ROUTING.md#replica-routing).

`extra_args` also refuses unambiguous argparse abbreviations of those flags
(`--tensor-parallel 2`, `--data-parallel-s 7`). Model templates
(`GET`/`POST /api/models/templates`) carry `data_parallel_size` (default 1);
`POST /api/models` with a `template_id` uses the template's value unless the body
sets its own. `DELETE /api/models/{id}/layout-notice` (JWT, 204) dismisses the
info-level `layout_notice` a boot pass left on a model; a `warning` notice answers
409 (edit `extra_args` instead), an unknown model 404.

### Changing a model after it is registered

`PATCH /api/models/{id}/settings` takes any subset of the model's settings; an
omitted key is left alone. The body is merged into the stored row and **the
merged row is validated by the same rules as `POST /api/models`** — the same
slug patterns, the same length and numeric bounds, the same `extra_env`
allowlist, and the same cross-field rules. There is no value this endpoint
accepts that register would refuse.

```bash
curl -s -X PATCH $W/api/models/$M/settings "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"max_model_len": 32768, "extra_args": ["--enforce-eager"]}'
# 200 {"ok": true}
```

Two rules catch people out, and both come from register:

- **`tensor_parallel_size` must equal `len(gpu_indices)`.** Changing the GPU
  selection therefore means sending the new parallel size in the same request.
  A row where the two disagree cannot load, which is why the pair is refused
  rather than half-applied.
- **`extra_args` is a list of strings.** A bare string is refused; it used to
  be accepted and then appended to the engine's argv one character at a time.

What comes back:

| Status | When |
|---|---|
| 400 | a key that is not a setting (`status`, `prior_status`, or an unknown name), a malformed `gpu_indices` / `backend` / capability flag, or a GPU index outside the wizard's allow-list |
| 403 | an admin token turning `trust_remote_code` on (see below) |
| 404 | no such model |
| 409 | the model is `loaded` (unload first), or the new `served_model_name` already belongs to another model |
| 422 | a value that breaks one of the rules above; the body names the field |

A refused PATCH writes nothing at all — not one of its keys.

Only the capability flags (`supports_tools`, `supports_vision`,
`supports_reasoning`) may be changed while the model is loaded: they are
metadata no engine reads, and the loaded model is exactly the one you are
looking at when you notice one is wrong. Mixing any other setting into the same
body puts the whole request back under the 409.

The same rules apply wherever else the API writes a model row — registering
from a template, pinning an engine with `POST /api/models/{id}/try-stack`, and
applying a measured context with `POST /api/models/{id}/stress/apply` all go
through one writer, so none of them can store something register would refuse.

Because the whole merged row is checked, any of those can refuse over a column
the request does not mention — most often an `extra_env` key written before
that allowlist was enforced, which loads and serves fine because the engine
drops unknown keys at launch. On `POST /api/models` and the settings PATCH you
can send the corrected value in the same request and it is accepted; try-stack
and stress/apply take no such body, so fix the column with a settings PATCH
first. `POST /api/models/{id}/stress/apply` checks the row **before** it
unloads anything, so a `422` there normally means the model is still serving on
its current context and nothing was written. The check cannot close the window
entirely: if the row changes underneath it, the refusal lands after the unload,
the engine is restarted before the `422` is returned, and a restart that itself
fails leaves the row `failed` for the watchdog.

---

## Managing a key from the API

Each key's page in the UI (`/ui/tokens/{id}`) drives these calls. They are
JWT-gated like the rest of the control API, and the ones that change something
are CSRF-checked, so `$W`, `$JWT` and `AUTH` are the same as in the two sections
above. `T` is a key's `id` — the `id` in the `POST /api/tokens` response, or
any item of `GET /api/tokens`.

```bash
T=53e7011c…

# 1. One key: every field a GET /api/tokens item carries, plus `lineage` —
#    the rotation chain it belongs to, oldest first. 404 for an unknown id.
curl -s $W/api/tokens/$T -H "Authorization: Bearer $JWT"
# {"id":"53e7011c…","name":"my-first-key","prefix":"vw_su3yl",
#  "priority":5,"is_paused":false,"paused_at":null,
#  "is_expired":false,"is_revoked":false,"successor_id":null,
#  "usage_24h":{"requests":12,"prompt_tokens":48210,"completion_tokens":3120,
#               "total_tokens":51330}, …,
#  "lineage":[
#    {"id":"0b91…","name":"my-first-key (old 1)","created_at":"…",
#     "rotated_at":"…","is_revoked":false,"in_grace":true,"is_self":false},
#    {"id":"53e7011c…","name":"my-first-key","created_at":"…",
#     "rotated_at":null,"is_revoked":false,"in_grace":false,"is_self":true}]}

# 2. Rename, reprioritise, pause or resume. Every field is optional; omit one
#    to leave it alone. The response is the full key, the same shape as (1).
curl -s -X PATCH $W/api/tokens/$T "${AUTH[@]}" -d '{"name":"ci-runner"}'
curl -s -X PATCH $W/api/tokens/$T "${AUTH[@]}" -d '{"priority":7}'
curl -s -X PATCH $W/api/tokens/$T "${AUTH[@]}" -d '{"paused":true}'
# 200 {… "is_paused":true,"paused_at":"2026-09-18 21:04:11", …}
curl -s -X PATCH $W/api/tokens/$T "${AUTH[@]}" -d '{"paused":false}'

# 3. Rotate: a new key that keeps the name, and the old one working for
#    `grace_hours` more (default 24; 0 cuts it off on its next request).
curl -s -X POST $W/api/tokens/$T/rotate "${AUTH[@]}" -d '{"grace_hours":24}'
# 201 {"id":"7c2e…","name":"ci-runner","plaintext":"vw_…","prefix":"vw_…",
#      "rotated_from":"53e7011c…","renamed_to":"ci-runner (old 1)",
#      "grace_hours":24}
```

What the PATCH fields do:

- `name` — trimmed, then 1–64 characters. Duplicate names are allowed.
- `priority` — 0 to 9, served strictly highest first. It is the only per-key
  fairness control: per-key rate limits were removed in v2026.09.18.2, and a
  create or PATCH body that still carries `rate_limit_tps`, even as `null`, is
  refused with **422** rather than quietly ignored.
- `paused` — `true` stamps `paused_at` (pausing again keeps the first time);
  `false` clears it. A paused key's next `/v1` request gets
  **`403 {"detail":"token paused"}`** — not the 401 an expired, revoked or
  unknown key gets, so a client can tell the two apart. Requests already
  running finish; nothing queued is aborted. Pausing a key that is expired, or
  revoked with its grace window over, is a **409**. An old key still inside its
  grace window can be paused, which cuts it off early. Rotating a paused key
  gives a successor that starts unpaused; the old key stays paused.

`POST /api/tokens/{id}/test` also reports `"paused"` beside `"revoked"` and
`"expired"`. Rotating a key that was already rotated is a 409 — rotate its
successor instead.

### Listing keys

`GET /api/tokens` returns **one page** of keys, sorted and optionally searched
on the server — the list is built for installs with millions of keys, so a
script that wants every key must page through it.

```bash
# The defaults, spelled out: newest first, 50 per page, no search.
curl -s "$W/api/tokens?sort=created&dir=desc&limit=50&offset=0" \
     -H "Authorization: Bearer $JWT"
# {"items":[{"id":"53e7011c…","name":"ci-runner", …}, …],
#  "total":12345,"limit":50,"offset":0,"near_expiry":3}

# Every key whose name contains "bot", any case, busiest first.
curl -s "$W/api/tokens?q=bot&sort=usage_24h&dir=desc" -H "Authorization: Bearer $JWT"

# All of them, 500 at a time.
for ((off = 0; ; off += 500)); do
  page=$(curl -s "$W/api/tokens?limit=500&offset=$off" -H "Authorization: Bearer $JWT")
  jq -c '.items[]' <<<"$page"
  (( off + 500 < $(jq .total <<<"$page") )) || break
done
```

| Parameter | Default | Values |
|---|---|---|
| `sort` | `created` | `name`, `prefix`, `created`, `expires`, `last_used`, `priority`, `usage_24h`, `status` |
| `dir` | `desc` | `asc`, `desc` |
| `limit` | `50` | 1–500 |
| `offset` | `0` | 0 or more |
| `q` | empty | up to 64 characters |
| `near_expiry` | `0` | `0`, `1` |

- **`sort`** — `name` ignores case. `usage_24h` is prompt + completion tokens
  over the last 24 hours. `status` follows the badge the UI shows: Paused,
  then Revoked (a rotated key whose grace is over, or whose successor was
  deleted), Expired, Grace, Expiring soon, Active. Keys that were never used
  or never expire sort **last in both directions**, and ties are broken by
  `id` in the same direction, so paging never repeats or skips a key.
- **`q`** — trimmed; matches keys whose name contains it, ignoring case (ASCII
  letters). `%`, `_` and `\` are matched literally. Empty means no filter.
- **`near_expiry=1`** — only keys that expire within 30 days and have not
  expired yet: exactly the keys the `near_expiry` count counts. Combines with
  `q`, `sort` and paging. The UI's banner applies it.
- **Response** — `items` is the page, each item the `GET /api/tokens/{id}`
  shape without `lineage`. `total` counts every key the list shows (after
  `q`), across all pages. `near_expiry` counts those of them that expire
  within 30 days and have not expired yet — what the UI's "expiring soon"
  banner says. An `offset` past the end is not an error: `items` is empty and
  `total` still tells you where the end is.
- **422** — `limit` outside 1–500, a negative `offset`, an unknown `sort` or
  `dir` (both are case-sensitive), a `q` longer than 64 characters, or a
  `near_expiry` other than `0` / `1`.
- **`last_used_at` is accurate to a minute.** A request stamps it only when
  the stored value is empty or more than 60 seconds old, so a busy key does
  not rewrite its row (and the index the `last_used` sort reads) on every
  request. Read it as "used at most a minute after this"; the same goes for
  `GET /api/tokens/{id}`.
- A key revoked **without** a rotation is left out of the list, as it always
  was; `GET /api/tokens/{id}` still answers for it. A rotated key stays listed
  after its grace window closes.

**What a page costs.** Every sort but `usage_24h` and `status` reads its page
straight off an index (migration 0035), and `total` / `near_expiry` are
index-only counts, so the first page costs about the same at a hundred keys or
a million (under 10 ms). A deep `offset` walks that many index entries —
around half a second at offset 500,000. `status` and `usage_24h` are computed
per key and scan the list: roughly 0.2–0.8 s and 1 s respectively per million
keys. A `q` search reads every name — a leading-wildcard `LIKE` cannot seek an
index — so its `total` costs a scan of the keys (100–400 ms per million),
while the page itself stays fast whenever matches are common. If search ever
becomes the bottleneck, an FTS5 `trigram` table over `name` would serve the
same substring match from an index.

### Usage and timings for any window

`GET /api/tokens/{id}/series` is what the page's charts and history strip draw:
usage and per-request timings for one key, binned on the server.

```bash
NOW=$(date +%s)
curl -s "$W/api/tokens/$T/series?from=$((NOW - 86400))&to=$NOW" \
     -H "Authorization: Bearer $JWT"
# {"token_ids":["53e7011c…","0b91…"],
#  "from_minute":29828160,"to_minute":29829600,"bin_minutes":5,
#  "latency_since":1789689600.0,
#  "timing_sample":{"total":1480,"used":1480,"stride":1},
#  "totals":{"requests":1480,"prompt_tokens":61203344,"completion_tokens":402118},
#  "bins":[{"minute":29828160,
#           "requests_per_min":1.2,"prompt_per_min":48210.4,
#           "completion_per_min":331.0,"peak_prompt":97310,
#           "peak_completion":702,"requests":6,"n":6,
#           "queue_p50":0.0,"queue_p95":0.41,
#           "ttft_p50":1.9,"ttft_p95":6.3,
#           "duration_p50":11.2,"duration_p95":38.5}, …]}
```

| Parameter | Default | Meaning |
|---|---|---|
| `from`, `to` | required | The window, in epoch seconds. `to` is exclusive. |
| `chain` | `1` | `1` includes the keys this one was rotated from; `0` is this key alone. |
| `max_bins` | `360` | At most this many bins (1–360). |
| `timings` | `1` | `0` skips the per-request timings: every timing field is `null`, `timing_sample` is `{0, 0, 1}` and `latency_since` is `null`. Usage only, and much cheaper over a long window. |

How to read the response:

- **Bin width** (`bin_minutes`) is the smallest step of 1, 2, 5, 10, 15, 30,
  60, 120, 360, 720, 1440 or 10080 minutes that fits the window in
  `max_bins` bins, so 1h and 6h come back in 1-minute bins, 24h in 5 and 7d
  in 30. Bins are aligned to UTC, so they do not move between polls. `minute`
  is a bin's start, in minutes since the epoch.
- **Rates** are the bin's sum divided by its width: a minute with no traffic
  counts as zero, so a quiet key does not look busy. `peak_prompt` and
  `peak_completion` are the busiest single minute inside the bin, across the
  whole chain. `totals` are exact.
- **Timings** — `queue_*` (waiting for a slot), `ttft_*` (first token) and
  `duration_*` (full response), in seconds, median and 95th percentile. A bin
  with no requests has `null`, not `0`. `n` is how many requests the
  percentiles came from.
- **`latency_since`** is the oldest per-key timing on record (epoch seconds),
  or `null` if there is none. Timings are recorded per key from v2026.09.18.1
  on and pruned with the rest of request history, so a window reaching further
  back has timing gaps that mean "not recorded", not "idle".
- **Sampling.** Past 50,000 matching requests the percentiles come from every
  `stride`-th request across the whole window; `timing_sample` says how many
  there were (`total`) and how many were used (`used`). Request and token
  counts are never sampled.
- Bins with no data are left out; fill the gaps yourself.

A `to` later than the server's clock is clamped to it rather than refused, so
a client whose clock runs fast still gets a chart. Otherwise the window must
have `from` before `to` and be no longer than 366 days, or the answer is
**422**. An unknown id is a 404.

The older `GET /api/tokens/{id}/usage?range=1h|24h|7d` is still there and
unchanged: raw per-minute rows for one key, with no binning and no timings.

### God mode for one key

God mode is off unless `VW_GODMODE_ENABLED` is set (see
[Where request content can end up](OPERATING.md#where-request-content-can-end-up)).
The page asks first, and shows its dock only when the answer is yes:

```bash
curl -s $W/api/admin/godmode/status -H "Authorization: Bearer $JWT"
# {"enabled":true}
```

The stream itself is a server-sent-event stream behind a single-use ticket,
because a browser's `EventSource` cannot send an `Authorization` header. Mint
the ticket for the **bare** path — no query string — then pass it, and
`token_ids`, as query parameters:

```bash
TICKET=$(curl -s -X POST $W/api/auth/sse-ticket "${AUTH[@]}" \
           -d '{"path":"/api/admin/godmode/stream"}' | jq -r .ticket)

curl -sN "$W/api/admin/godmode/stream?token_ids=$T,0b91…&ticket=$TICKET"
# data: {"type":"request_start","req_id":"…","token_id":"53e7011c…", …}
# data: {"type":"delta","req_id":"…","token_id":"53e7011c…", …}
# data: {"type":"request_end","req_id":"…","token_id":"53e7011c…", …}
```

`token_ids` is a comma-separated list of at most 20 key ids. Both the replay
of recent requests and the live events are narrowed to those keys, and every
event carries `token_id`, so a request is never split from its deltas. An
empty list, or more than 20 ids, is a **422**. Leave the parameter out and the
stream is every key's traffic, exactly as before it existed — scripts that
already read it are unaffected. With god mode off, the ticket mint and the
stream both answer **409** `god mode is disabled (VW_GODMODE_ENABLED)`.

The stream is open only as long as you hold it. The page's dock works the same
way: it connects when you open it and disconnects when you close it.

---

## Claude Code and the Anthropic Messages API

`POST /v1/messages` speaks the Anthropic Messages API, so Claude Code and the
Anthropic SDKs can use a warden as their base URL. It is not a second proxy:
the request is translated into an OpenAI chat request and goes through the
same path as `/v1/chat/completions`. Admission, key priority, usage
accounting, live stats, god mode and the runaway detector all work the same
way, and the engine sees an ordinary chat completion.

```bash
curl -s $W/v1/messages -H "x-api-key: vw_su3yl…" \
     -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' \
     -d '{"model":"qwen2.5-1.5b","max_tokens":256,
          "system":"Be terse.",
          "messages":[{"role":"user","content":"Name one prime."}]}'
# {"id":"msg_…","type":"message","role":"assistant","model":"qwen2.5-1.5b",
#  "content":[{"type":"text","text":"7"}],
#  "stop_reason":"end_turn","stop_sequence":null,
#  "usage":{"input_tokens":21,"output_tokens":2,
#           "cache_creation_input_tokens":0,"cache_read_input_tokens":0}}
```

The key is an ordinary inference key (`vw_…`), with the same expiry, pause,
revocation and per-model allow list. These two routes, and only these, also
accept it as `x-api-key`, which is where the Anthropic SDKs send it;
`Authorization: Bearer` works too. `model` is the **served model name**, the
same one `/v1/models` lists. Unless the router below is on, there is no
mapping from `claude-…` names, so a request for a model the warden does not
serve gets a `404`. Every `/v1` route also accepts the key in an
`X-LMWarden-Key` header, which the router mode uses.

**Claude Code** can use a warden in two modes.

*Local only.* Everything goes to the warden: name the served model in every
model slot, including the small one Claude Code uses for background work, and
send the key as the auth token. No Anthropic account is involved.

```bash
export ANTHROPIC_BASE_URL=$W                  # no /v1: Claude Code adds the path
export ANTHROPIC_AUTH_TOKEN=vw_su3yl…         # sent as Authorization: Bearer
export ANTHROPIC_MODEL=qwen3.8-27b-fp8
export ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-27b-fp8
export ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-27b-fp8
export ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-27b-fp8
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192     # see max_tokens below
claude
```

*Router: Claude Code with your subscription.* Keep your normal Claude login and
let the warden serve only the models you choose. Claude Code needs to be
version 2.1.227 or newer. The warden key goes in its own header, so the login
(`Authorization` / `x-api-key`) still reaches Anthropic:

```bash
export ANTHROPIC_BASE_URL=$W
export ANTHROPIC_CUSTOM_HEADERS="X-LMWarden-Key: vw_su3yl…"
claude
```

Do not also set `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` to the warden
key; that would replace the login. The key must have `anthropic_relay`
(**May relay to Anthropic**: `"anthropic_relay": true` on `POST /api/tokens` or
`PATCH /api/tokens/{id}`, off by default), or any request that would reach
Anthropic is refused with a `403`; requests that a rule serves locally need no
flag. The router is off until you enable it, and rules decide which requested
models run locally. With `$ACCESS` an admin session or admin token and `$CSRF`
as in the sections above:

```bash
# Turn it on (GET returns the same object; every field is optional on PATCH).
curl -s -X PATCH $W/api/router/settings -H "Authorization: Bearer $ACCESS" \
     -H "X-CSRF-Token: $CSRF" -b jar -H 'Content-Type: application/json' \
     -d '{"enabled":true}'
# {"enabled":true,"upstream_url":"https://api.anthropic.com",
#  "passthrough_unmatched":true,"local_header_timeout_s":60,
#  "local_nonstream_timeout_s":120,"breaker_threshold":3,"breaker_open_s":60,
#  "max_body_mb":32,"header_name":"X-LMWarden-Key"}

# Claude Haiku -> a local model (target_model_id is the model row's id).
curl -s -X POST $W/api/router/rules -H "Authorization: Bearer $ACCESS" \
     -H "X-CSRF-Token: $CSRF" -b jar -H 'Content-Type: application/json' \
     -d '{"pattern":"claude-haiku*","target_model_id":"9895a1829559533c",
          "fallback":true,"strip_thinking":true}'
# 201 {"id":"a1b2…","position":0,"pattern":"claude-haiku*",
#      "target_served_name":"qwen2.5-1.5b","target_status":"loaded", …}

# Order matters (first match wins); the body must list every rule id once.
curl -s -X PUT $W/api/router/rules/order -H "Authorization: Bearer $ACCESS" \
     -H "X-CSRF-Token: $CSRF" -b jar -H 'Content-Type: application/json' \
     -d '{"ids":["a1b2…","c3d4…"]}'

# Where did requests go? Counters are per process; the process keeps the last
# 200 decisions (limit 1 to 200, default 100; the Router activity page shows 100).
curl -s $W/api/router/stats -H "Authorization: Bearer $ACCESS"
curl -s "$W/api/router/decisions?limit=20" -H "Authorization: Bearer $ACCESS"
```

Rules are also `GET /api/router/rules`, `PATCH`/`DELETE
/api/router/rules/{id}`, and `POST /api/router/stats/reset` clears the
counters. Everything else (Opus, Sonnet, and any other `/v1/*` path) is passed
through to Anthropic byte for byte, streaming included, with `Authorization` /
`x-api-key` forwarded only there; the warden key is never sent to Anthropic. A
rule can fall back to Anthropic when the local model fails before its first
byte, or refuse with a `529` so Claude Code retries; a model the warden serves
by name is always answered locally. Passthrough tokens are counted in the
router's own stats, not in a key's usage. The full behaviour, with every
refusal reason and the failure table, is in [ROUTING.md](ROUTING.md#the-router).

Router behaviours worth knowing when you call the warden directly: a request
body above `max_body_mb` is a `413` whether or not it declares a length; while a
rule's breaker is half-open only one request probes the local model and the
rest fall back or get `529`; an unexpected error in the local leg is treated as
a local failure (`local_error`), not a `500`; a relayed path with a `.`/`..`
segment in any encoding is a `400`; more than 256 relayed requests in flight
get `529` after waiting up to 10 s; a key with a model allow list is not relayed
on paths with no model (the catch-all) and a fallback is held to its list; and
unknown `/v1` paths keep answering `404`/`405`/`307` as they did before the
router when it is off, authenticated or not.

What is translated, and how:

- **Request.** `system` (a string or text blocks) becomes a leading system
  message. `tool_use` blocks become the assistant's `tool_calls`.
  `tool_result` blocks become `role: tool` messages, placed before the rest of
  the user turn. `tools[].input_schema` becomes `function.parameters`, and
  `tool_choice` `auto`/`any`/`tool`/`none` becomes
  `auto`/`required`/named function/`none`. `disable_parallel_tool_use`
  becomes `parallel_tool_calls: false`. `stop_sequences` becomes `stop`, and
  `max_tokens`, `temperature`, `top_p` and `top_k` pass through. Images
  (base64 or URL) become `image_url` parts. An image inside a `tool_result`
  is moved into the user message that follows, because an OpenAI tool
  message can only carry text. Server tools such as `web_search_…` are
  dropped, since the engine cannot run them. `cache_control`, `metadata` and
  other fields the engine has no use for are ignored.
- **Response.** Text becomes a `text` block and each tool call becomes a
  `tool_use` block with its arguments parsed into `input`. `finish_reason`
  maps `stop` to `end_turn` (or to `stop_sequence` when the engine names the
  matched string), `length` to `max_tokens`, `tool_calls` to `tool_use`, and
  a runaway-detector cut to `max_tokens`. A turn that made tool calls is
  always `tool_use`. `usage.prompt_tokens`/`completion_tokens` become
  `input_tokens`/`output_tokens`, and a prefix-cache hit the engine reports
  is shown as `cache_read_input_tokens`.
- **Streaming** (`"stream": true`) sends Anthropic's events: `message_start`,
  then `content_block_start`/`content_block_delta`/`content_block_stop` per
  block, then `message_delta` (stop reason and final usage) and
  `message_stop`. Text streams as it is generated. Each tool call is sent
  whole, as one `input_json_delta`, once the engine has finished it, so its
  arguments are valid JSON before the client parses them. An engine error in
  the middle of a stream arrives as an `error` event.
- **Thinking.** The engine's reasoning channel (`--reasoning-parser`) is
  returned as `thinking` blocks only when the request enables `thinking`,
  which is also how Anthropic's API behaves. Their `signature` is empty.
  Thinking blocks a client sends back are dropped.
- **System messages inside `messages`.** Claude Code sends `role: "system"`
  entries (content a string or text blocks) in the middle of `messages` for
  model names it does not recognise (beta `mid-conversation-system-2026-04-07`).
  They are accepted. Chat templates such as Qwen's allow a system message only
  first, so a run of them before the first user or assistant message is merged
  into the top-level `system`, and a later one is folded, wrapped in
  `[system message] … [end system message]`, into the start of the next user
  turn (after any `tool_result` content, so a tool call and its result are never
  separated). With no user turn after it, it is appended to the end of the
  conversation. Empty ones are dropped, `count_tokens` counts the same text, and
  the router's pass-through to Anthropic is untouched.
- **Errors** use Anthropic's envelope,
  `{"type":"error","error":{"type":…,"message":…}}`, with the same status the
  OpenAI routes would return. `400` is `invalid_request_error` (also for a
  missing `max_tokens` or a content block that cannot be translated), `401`
  is `authentication_error`, `403` is `permission_error` (paused key, or a
  model outside the key's allow list), `404` is `not_found_error` (model not
  served or not loaded), `429` is `rate_limit_error`, and `5xx` is
  `api_error`. An engine's own error message is passed on unchanged.

`POST /v1/messages/count_tokens` takes the same body without `max_tokens` and
answers `{"input_tokens": n}`. The count comes from the served model's
tokenizer and does not include the chat template's framing tokens. With the
router on, a matching rule counts locally even while its breaker is open; if
the local count fails the request goes to Anthropic, and where the key may not
relay, the local error (for example `529`) is returned instead of a `403`.

**`max_tokens` is lowered to fit the model's context.** Claude Code asks for a
large output budget by default (32,000 or more), and vLLM refuses a request
whose prompt plus `max_tokens` exceeds the model's context window. So before a
request reaches the engine, the warden counts the prompt with the model's
tokenizer and, if prompt plus `max_tokens` would not fit `max_model_len`, lowers
`max_tokens` to what is left, keeping a margin of 256 tokens plus 5 % of the
prompt (25 % when no tokenizer is available and the count is an estimate). This
applies to every `/v1/messages` request answered locally: with the router off,
for a served model name, and on a router rule's local leg alike. It never raises
`max_tokens`, and leaves it alone when the window is unknown or the prompt alone
does not fit (the engine's `400` then follows the usual error or fallback path).
`/v1/chat/completions` is not clamped: OpenAI clients set their own
`max_tokens`. `CLAUDE_CODE_MAX_OUTPUT_TOKENS` still sets the budget Claude Code
asks for.

### Connect a client

`GET /api/connect/clients` returns, in one read, everything needed to point a
harness at this warden: per-tool setup files (Claude Code, the Anthropic and
OpenAI SDKs, OpenCode, Aider, Grok CLI, Continue, Cline, Cursor and Codex CLI), the origin to use, the router's header
name and enabled rules, and the registered models with their effective context
window and capability flags. It is what the **Connect** page (`/connect`)
renders. A session or an admin token; `?model=<served name>` renders the
snippets for that model instead of the first loaded one.

```bash
curl -s "$W/api/connect/clients?model=qwen2.5-1.5b" -H "Authorization: Bearer $ACCESS"
# {"origin":"https://warden.example","origin_source":"public_url",
#  "header_name":"X-LMWarden-Key","key_placeholder":"vw_YOUR_KEY",
#  "router":{"enabled":true,"passthrough_unmatched":true,
#            "rules":[{"pattern":"claude-haiku*","target_served_name":"qwen2.5-1.5b",
#                      "target_status":"loaded","fallback":true,
#                      "example_model":"claude-haiku-4-5"}]},
#  "models":[{"id":"9895a1…","served_name":"qwen2.5-1.5b","status":"loaded",
#             "backend":"vllm","context_window":32768,"supports_tools":true,
#             "supports_vision":null,"supports_reasoning":false}],
#  "selected_model":"qwen2.5-1.5b",
#  "clients":[{"id":"claude-code","name":"Claude Code","group":"anthropic",
#    "protocol":"anthropic","support":"full","requirements":[…],
#    "docs":{"url":"https://…","accessed":"2026-10-04"},"verified":null,
#    "modes":[{"id":"router","needs_relay_key":true,
#      "files":[{"id":"shell","language":"bash",
#        "template":"export ANTHROPIC_BASE_URL={{origin}}\n…",
#        "rendered":"export ANTHROPIC_BASE_URL=https://warden.example\n…"}, …],
#      "verify":{"method":"POST","path":"/v1/messages",
#                "headers":{"{{header}}":"{{key}}", …},
#                "body":{"model":"{{verify_model}}","max_tokens":16, …},
#                "expect":"anthropic_message", …}}, …]}, …]}
```

`origin` is the `public_url` setting when it is set (`origin_source:
"public_url"`), otherwise the address this request came in on, honouring
`X-Forwarded-Proto`/`X-Forwarded-Host` (`"request"`). `support` is `full`
(Anthropic protocol: router mode and local only), `local_only` (OpenAI
protocol: local models, no pass-through to OpenAI or xAI), `documented`
(editors configured from their docs; the request they send was tested, the
editor was not run) or `unsupported` (no modes). `verified` records the live
run per client, or is `null`.

Templates use five placeholders, replaced as plain text: `{{origin}}` (no `/v1`
for Anthropic clients, which add the path; `{{origin}}/v1` is written out for
OpenAI clients), `{{key}}`, `{{model}}` (a served name), `{{header}}` and
`{{context}}` (the model's context window). A `verify` body may also use
`{{verify_model}}`: in router mode the first rule's `example_model`, otherwise
the model. `rendered` is the template filled in for this warden, with
`{{key}}` always `vw_YOUR_KEY` and `{{model}}` `your-served-model-name` when no
model is loaded; an unknown context window renders as `32768` with a comment
saying it is a guess. The response never contains a key: the warden stores
only hashes. A requirement's `modes` names the modes it applies to (`null`:
all), and a file's `requires_tools` (`true`/`false`) marks a variant meant only
for a model whose `supports_tools` matches.

---

## Codex CLI and the OpenAI Responses API

`POST /v1/responses` speaks the OpenAI Responses API, which is the only wire
format OpenAI Codex CLI uses. Like `/v1/messages` it is not a second proxy: the
request is translated into an OpenAI chat request for the served model and goes
through the same path as `/v1/chat/completions`, so admission, key priority,
usage accounting, live stats, the session column, replica affinity, the
prefix-cache estimate, god mode and the runaway detector all apply. The
warden's own `/v1/responses` is used rather than the engine's because vLLM's
streams tool calls of non-harmony models (Qwen) as text, and llama.cpp has none.

```bash
export LMWARDEN_KEY=vw_su3yl…
curl -s $W/v1/responses -H "Authorization: Bearer $LMWARDEN_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"model":"qwen2.5-1.5b","input":"Reply with OK.","max_output_tokens":16}'
```

What is served:

- **Models.** Served model names only, with the router on or off. Router rules
  (Claude-name globs) and the Anthropic relay do not apply to this route, and a
  name that is not a loaded served model is a `404`.
- **Stateless.** Codex sends `store: false` and the whole conversation in
  `input` every turn. `previous_response_id`, `conversation` and `background:
  true` are refused with a `400` that names the field. `GET`/`DELETE
  /v1/responses/{id}`, `/cancel`, `/compact` and `/v1/conversations/*` are not
  served.
- **Input.** A string, or items: `message` (`user`, `assistant`, `system`,
  `developer`; `input_text`, `output_text`, `refusal`, `input_image` as a URL or
  data URI), `function_call`, `function_call_output` (a string, or text and
  image items), `custom_tool_call`, `custom_tool_call_output`, and `reasoning` and
  `web_search_call` (accepted and dropped). `instructions` and a leading run of `developer` and
  `system` items form the system prompt; a later one is folded into the next
  user turn. Anything else (`input_file`, `input_audio`, `item_reference`,
  `local_shell_call`, `compaction`, `mcp_*`, ...) is a `400` naming the item type and advising a new Codex thread.
- **Tools.** `function` tools, and `custom` (freeform, for example `apply_patch`)
  tools, which the model calls through one string argument `input` and which come
  back as `custom_tool_call` items. A `namespace` tool (Codex sends one,
  `multi_agent_v1`, for its sub-agent tools) is flattened into its member
  functions, and a call to one comes back with its `namespace`. `web_search*`
  tools are dropped; other hosted tool types (`local_shell`, `tool_search`,
  `mcp`, `file_search`, `code_interpreter`, `image_generation`,
  `computer_use_preview`) are a `400`.
  `tool_choice` is `auto`, `none`, `required`, `{type: function|custom, name}`.
  Tool calls are sent whole after the model finishes, not as argument deltas; if
  the model's arguments are not valid JSON they are passed through as written.
- **Other fields.** `max_output_tokens` (lowered to fit the model's context, as for
  `/v1/messages`), `temperature`, `top_p`, `parallel_tool_calls`, `text.format`
  (`json_schema`, `json_object`), `stream`. Accepted and ignored: `store`,
  `include`, `prompt_cache_key` (used for affinity only), `user`, `metadata`,
  `client_metadata`, `safety_identifier`, `service_tier`, `truncation`,
  `stream_options`, `text.verbosity`. Only fields the engine understands are
  forwarded to it.
- **Reasoning.** Thinking is on by default: Codex sends `reasoning: {"summary":
  "auto"}`, which the warden treats as on (the chat template's own default
  applies), and the model's reasoning is returned as a `reasoning` item on the
  summary channel (`response.reasoning_summary_text.delta`), which is what Codex
  displays. It is off when `reasoning` is absent or `null`, or its `effort` is
  `none`; the warden then sends `chat_template_kwargs.enable_thinking = false`.
  To turn it off for Codex, set `model_reasoning_effort = "none"` in
  `config.toml` (verified against Codex 0.160.0: the forwarded request then has
  thinking off and no reasoning items come back).
- **Streaming.** `event:`/`data:` frames with a rising `sequence_number`, no
  `[DONE]`. A turn always ends with `response.completed`; one cut short by the
  token limit has `status: "incomplete"` and `incomplete_details.reason:
  "max_output_tokens"` (Codex retries a `response.incomplete`, so it is never
  sent). An engine error inside the stream is `response.failed`; an engine error
  before the stream starts is a plain HTTP error.
- **Usage.** `input_tokens` includes the cached prefix, reported in
  `input_tokens_details.cached_tokens`; `output_tokens_details.reasoning_tokens`
  is the engine's count or `0`.
- **Errors.** OpenAI's envelope, `{"error": {"message", "type", "param", "code"}}`,
  for every refusal including a missing or bad key: `400`, `404`, `413` and `422`
  are `invalid_request_error`, `401` `authentication_error`, `403`
  `permission_error`, `429` `rate_limit_error`, `5xx` `server_error`. The key
  is `Authorization: Bearer vw_…` (or `X-LMWarden-Key`); `x-api-key` is not read.
  A `GET` to `/v1/responses` is a `405` with `Allow: POST`.
- **Not served.** `GET`/`DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`
  and `POST /v1/responses/compact` (Codex's remote compaction) do not exist here,
  and neither do `/v1/conversations/*`. With the router off they are a plain
  `404 {"detail": "Not Found"}`. With the router on they fall to the `/v1`
  catch-all, which relays unknown paths to the Anthropic upstream for a key that
  may relay (and passthrough is on), so the answer is Anthropic's error, not the
  warden's; a model-restricted key or one that may not relay gets a `403`, and
  passthrough off a `404`. Codex falls back to compacting locally.

Codex's `session-id` header and `prompt_cache_key` identify the thread, so its
turns stay on one replica and the Session column shows the thread. A request
with neither falls back to a hash of the first real user message, skipping
Codex's `<environment_context>` item.

Codex setup (Connect writes this for the picked model; the engine needs tool
calling enabled, e.g. vLLM `--enable-auto-tool-choice --tool-call-parser ...`):

```toml
# ~/.codex/config.toml
model = "qwen2.5-1.5b"
model_provider = "lmwarden"
# Codex assumes a 272k window for models it does not know; set the real one.
model_context_window = 32768

[model_providers.lmwarden]
name = "LM Warden"
base_url = "https://warden.example/v1"
env_key = "LMWARDEN_KEY"
wire_api = "responses"
```

```bash
export LMWARDEN_KEY=vw_su3yl…
codex exec "Reply with OK."
```

Set `web_search = "disabled"` in the same file: Codex otherwise offers a hosted
search tool the warden drops. A model that is not loaded answers `404` and Codex
retries for several seconds before it reports the error.

---

## Session forest

`GET /api/stats/forest?range=48h[&since=<cursor>]` — trees of agent sessions for the forest view.
Ranges: `1h`, `6h`, `24h`, `48h`, `7d` (anything else, `30d` included, is a 400: the forest never holds more
than 7 days). The forest view offers `1h`, `6h`, `24h` and `7d`. The range picks the trees with a turn finished in `[now − range, now]`; each tree is sent whole (every
turn of the last 7 days), never cut at the window edge, so `t0` is about `now − 7 d` for every range up to `7d`. Admins see every key; a key holder sees only their key.

`POST /api/forest/login` with `Authorization: Bearer <your API key>` returns a 12-hour token for the forest
routes only.

Response fields: `range`, `t0`, `now`, `truncated_before` (only when the server holds fewer rows than the window has:
past its row budget, a key's oldest rows before this absolute time are left out, so its oldest trees may miss that
wood), `models` (per variant: `prefill_tps`, `decode_tps` — 7-day p75 —
`n`, the number of requests sampled, and `max_model_len`), `trees`, `flowers`, `ids`, `cursor`, `full`.
The body is gzip-compressed when the request sends `Accept-Encoding: gzip`. The server rebuilds at most
every 3 s per (range, viewer); requests in between get the same data, but `now` is always the time of the
request itself (absolute unix seconds), so the data can be up to 3 s older than `now`. The forest view runs
its clock on the browser and uses `now` only to estimate the clock skew.

Client contract:

- **Times are relative.** Every time in `trees` and `flowers` is seconds after `t0` (absolute =
  relative + `t0`). `t0` can change from one response to the next, so rebase what you hold by the
  difference. The window `[now − range, now]` picks trees; it never trims them.
- **Polling.** Poll with `since=<cursor>` (the `cursor` of your last response, every ~2 s). A delta
  carries every *whole* tree with a turn finished after `since − 5 s` — a 5 s overlap, so rows written
  late are not missed; replace held trees by `id`.
- **`ids`** always lists every current tree, whether or not it is in the response. Drop any tree you
  hold whose id is not in `ids`.
- **Flowers** (single-turn requests without tools) have no ids; each is `[start, prompt_tokens,
  completion_tokens]`. A delta re-sends every flower whose absolute start (`t0` + its start) is strictly
  greater than `since − 600 s`: replace all the flowers you hold whose absolute start is greater than that
  by the ones sent.
- **`full`.** A tree's id can change without the tree growing (a spell's id follows its first session
  still in the window, so it can change as the window slides past it). **If `full` is true or `ids`
  contains an id you don't hold, refetch without `since`.** A `full: true` response — every response
  without `since`, and a delta the server cannot vouch for — already carries every tree and flower, so
  you may use it as that refetch.
- **Order.** Apply a delta first, then check `ids` for unknown ids.

Each tree is `{id, key, start, end, last, last_at, traits, sessions}`: `key` is the API key's name;
`start` and `end` are its first and last request start and `last` its last finish (relative to `t0`);
`last_at` is that last finish, absolute; `traits` are the shape inputs; `sessions` its root sessions. Each
session is `{id, variant, turns, children, at?}`: `children` are its subagent sessions, and a child also has
`at`, the index of the parent turn it branched from. A turn has 8 elements: `[start, prompt_tokens,
completion_tokens, cached_tokens, tools_in, tools_out, finish_reason, n]`, where `start` is relative to `t0`
and `n` is the number of requests the turn stands for. The level of detail is fixed by index: the first
2000 turns of a session are sent one by one; later turns go in blocks of 8 (a block's `n` is 8). Only
closed blocks are fixed: past 2000 turns, a trailing incomplete block is sent turn by turn until it fills,
and when it closes those (up to 8) newest single turns are replaced by the one merged turn (the time of its
last turn), so a child's `at` in that block moves too. Everything before the newest block never changes
shape as the tree grows. A child's `at` uses the same mapping.

`GET /api/stats/requests` (requests in flight) also accepts a forest token. It is then scoped to that key:
only its rows, with `count`, `by_token` and `by_ip` computed from those rows alone. Each row's
`forest_session` is the id the forest gives that request's session (the keyed hash its history row stores
as the session key; `null` when the request has no session id); `session_id` stays the raw client id.

Optional request header on inference calls: `X-Batch-Id: <id>` (≤ 64 printable characters). Requests with
the same batch id grow one tree. Claude Code: `ANTHROPIC_CUSTOM_HEADERS="X-Batch-Id: my-batch-42"`.

Privacy: session ids and tool names are stored only as keyed hashes.

---

## Admin API

Everything above signs in with the admin password and carries a CSRF token. A
program that should not hold that password uses an **admin token**: a `vwa_…`
secret that calls the control API (`/api/*`) with the same power as signing
in — settings, model register/pull/load/unload, inference-token management,
statistics, god mode, stress runs and the cache. Issue one in
**Settings → Admin tokens**, or from a signed-in session:

```bash
curl -s -b jar -X POST $W/api/admin-tokens \
     -H 'Content-Type: application/json' \
     -H "Authorization: Bearer $JWT" -H "X-CSRF-Token: $CSRF" \
     -d '{"name":"ci-deploy","expires_in_days":90}'
# 201 {"id":"4f0c…","name":"ci-deploy","prefix":"vwa_k2x7","created_by":"admin",
#      "expires_at":"2026-12-18 09:12:40","status":"active", …,
#      "plaintext":"vwa_k2x7…"}
```

`expires_in_days` is required: `30`, `90` or `365`, or JSON `null` for a token
that never expires — a value you have to send on purpose. **`plaintext` is
shown exactly once**; only its SHA-256 hash is stored. The `vwa_` prefix
(inference keys are `vw_`) tells a leaked secret's kind at a glance, and secret
scanners can key on it.

With the token there is no login, no cookie jar and no CSRF header:

```bash
export VW_ADMIN_TOKEN=vwa_k2x7…
A=(-H "Authorization: Bearer $VW_ADMIN_TOKEN")

# The whole API, described. Not public: it needs an admin token or a session.
curl -s "${A[@]}" $W/api/openapi.json | jq '.paths | keys | length'

# Anything the UI does.
curl -s "${A[@]}" $W/api/models
curl -s "${A[@]}" -X POST $W/api/models/9895a1829559533c/load     # 202

# Streams take the header too -- no SSE ticket.
curl -sN "${A[@]}" $W/api/stats/live
```

The token acts as the user that issued it (`created_by`). Neither changing
the admin password nor renaming the operator revokes a token issued earlier
— it keeps calling the API as the `created_by` username recorded at issue
time. After a username change, revoke and reissue every admin token: it
still gets `200` from routes that only check it is a known admin token
(e.g. `/api/version`), but `401 unknown subject` from routes that resolve
identity by name (e.g. the chat2 surface), which is confusing enough to
avoid on purpose. FastAPI's own `/docs`, `/redoc` and `/openapi.json` are
switched off; `GET /api/openapi.json` is the spec, with a `bearerAuth`
scheme on every route except the public ones — `/healthz`, `/api/setup/*`,
`/api/auth/login`, `/api/auth/refresh`, `/api/csrf`, and the signed chat2
attachment download `GET /api/chat2/attachments/{attachment_id}` (it
carries its own `?t=` signature) — and `/v1/*`, which marks them
`security: []`.

**Treat an admin token like the admin password.** The session-only routes
below stop a leaked token from managing admin tokens *through the API* —
they are an API-level guardrail, not containment. `trust_remote_code` is
fenced the same way (#256): it is a stored declaration that the target
repository's Python may be executed with the warden's own privileges, and an
admin token must not be able to write it.

Where that execution actually happened is worth stating plainly, because this
page said the wrong thing until #264. **The `trust_remote_code` column has
never been passed to an engine.** Neither the vLLM nor the llama.cpp launcher
builds `--trust-remote-code` from it, under any driver. Its one consumer was
the warden's *own* process: the proxy's tokenizer cache handed the flag to
`AutoTokenizer.from_pretrained` on `/v1` data-plane requests, for prompt and
completion token accounting. So a row with the flag made the warden import that
repository's Python into the process holding the SQLite database and
`VW_JWT_SECRET` — on inference traffic, not at load, and identically under
every driver. **As of #264 the flag no longer reaches the tokenizer either**:
the cache always loads with `trust_remote_code=False`, and a repo whose
tokenizer can only be built by running its own code is token-*estimated*
instead (the same visible degradation a GGUF-only repo already gets — see
`tokenizer_repo`). The column now reaches no code path at all; the four
refusals below stay as a fence on the stored grant.

Four paths are refused for an admin token, which is every way the API has to
set the flag or load a model carrying it:

* `POST /api/models` — refused when the **effective** value is true, so a
  `template_id` whose template carries the flag is refused too, builtin
  templates included;
* `POST /api/models/templates` — refused when the template being saved carries
  the flag, so a token cannot leave the instruction lying around;
* `PATCH /api/models/{id}/settings` — refused when it would turn the flag *on*
  (turning it off is fine), because the flag is persisted the same either way;
* `POST /api/models/{id}/load` — refused when the row already has it, however
  that row got there. Loading does not itself execute anything from the
  repository — the column never reaches the engine — so this is a check on the
  stored grant, kept because a row can predate the register-time refusal.

`prior_status`, the column the watchdog's restart sweep keys on, is not
patchable at all any more — by a token or by a session. It was the other half
of the same hole: an admin token that could write it could have the watchdog
start the engine, with whatever argv the row carries, without ever calling
`/load`.

Those four paths are closed, but they were never containment, and the
`trust_remote_code` column was never the way to reach code execution in the
engine. `extra_args` is, and always has been: it is appended to the engine's
argv verbatim and last, so an admin token can put `--trust-remote-code` there —
and `--model <repo>` with it — through `POST /api/models` or `PATCH
/api/models/{model_id}/settings`, then load the model normally: the column
stays false, so the refusals above never fire. `extra_args` is also the *only*
way to make an engine trust remote code, which is why the engine-log diagnosis
for "this model requires trust_remote_code" now says to put
`--trust-remote-code` there rather than to turn the column on.
`engine_image` is the same shape wherever the driver runs a container
image — and note what that reaches. Under the default local-subprocess driver
the engine runs inside the api container, so argv-level code execution there is
code execution in the warden, with the database and the JWT secret on the same
filesystem. Under the docker driver the engine is a sibling container that is
handed the model-cache volume and nothing else: as of #265 the warden's data
volume is no longer mounted into it, so a chosen image or argv cannot read
`vllm-warden.db` or `jwt_secret`. It can still address the warden's API over the
shared control-plane network, as any client on that network can. Both are left
open deliberately: `POST /api/models` already accepts
them, so fencing only the PATCH would move the power one route over rather
than remove it — if engine start-up inputs are to become session-only, that
decision belongs at the register route. `extra_env`, by contrast, *is* fenced:
keys are allowlisted by prefix, and `LD_PRELOAD`, `PYTHONPATH`, `HF_TOKEN`,
`PATH` and friends are refused at write time.

**Treat an admin token as equivalent to code execution in the warden
container.** If a token leaks, revoke it immediately, then rotate
`VW_JWT_SECRET`/`jwt_secret` — treat it exactly like you would a leaked
admin password.

**What an admin token cannot do through the API.** These are
**session-only**, and answer an admin token with `403
{"detail":{"error_code":"session_only", …}}`:

| Route | Why |
|---|---|
| `GET/POST /api/admin-tokens`, `POST /api/admin-tokens/{id}/rotate`, `DELETE /api/admin-tokens/{id}`, `GET /api/admin-tokens/{id}/audit` | These are the only API routes that manage admin tokens, so requiring a session here is what makes "revoke it in the UI" an effective response to a leak: a leaked token cannot use *this* API to mint itself a sibling, extend its own life, or revoke or hide the trail of another admin token. It does not mean the token is contained — see the warning above. |
| `POST /api/auth/logout`, `POST /api/auth/sse-ticket` | They belong to a browser session. A program sends the header to a stream directly. |
| `PATCH /api/settings/runtime` when the body carries `admin_username`, `admin_password`, `session_access_ttl_minutes`, `session_refresh_ttl_days`, `sse_ticket_ttl_seconds` or `public_url` | A leaked token must not be able to change the operator's credentials or the session's lifetimes and lock the operator out of the UI, which is where it gets revoked. `public_url` is session-only too, as defence in depth: it is the host that client-facing URLs are built from, and a leaked token must not be able to steer it to a host it controls. (The curl examples shown next to a new admin secret use the page's own address and ignore it.) The refusal is on the whole request — nothing in the body is applied, not even its other keys. Every other runtime key stays open to an admin token. |
| `POST /api/models` when the model's **effective** `trust_remote_code` is true — the body's value, or the template's when the body does not say | `trust_remote_code` declares that the target repository's Python may be executed with the warden's privileges. Setting it needs a signed-in session, so a leaked token cannot persist that grant. The check is on the merged value, not the body key, because the builtin `gpt-oss-20b` template carries the flag and a `template_id` would otherwise be a one-request way around it. Refused before any write — nothing is persisted. An explicit `trust_remote_code: false` still wins over a template that sets it, and registering without the flag is unaffected. |
| `POST /api/models/templates` when the template being saved sets `trust_remote_code: true` | A template is a register-time prefill, so saving one is storing an instruction to set the flag on every model made from it. Refused before any write. Templates without it, and every other template field, stay open to an admin token. |
| `PATCH /api/models/{id}/settings` when the body sets `trust_remote_code` to a true value | Patching the flag on persists the same grant by a different verb. Checked first, before validation and before any write, so a refused PATCH changes nothing at all. Setting it to `false` is open to an admin token, as is every other setting on this route. |
| `POST /api/models/{id}/load` when the row's `trust_remote_code` is true | A row can predate the register-time refusal, so the flag is checked again here regardless of how the row was created. Loading does **not** itself run anything from the repository — the column is never passed to the engine (#264) — so this refuses on the stored grant, not on an execution about to happen. Every other model — and unload, list and everything else on a `trust_remote_code` model — stays open to an admin token. |

The setup wizard (`/api/setup/*`) and `/api/auth/login` take no credential at
all. An admin token is never a `/v1` credential (`401 invalid token format`).
An inference key never opens `/api/*` either — `401` — except the three
stress routes (`POST /api/models/{id}/stress`, `GET
/api/models/{id}/capabilities`, `POST /api/models/{id}/stress/apply`), which
name the reason: `403 {"detail":{"error_code":"operator_only", …}}`, because
starting a stress run deliberately crashes an engine and an inference key is a
data-plane credential.

**Refusals** are the same ladder an inference key gets: `401
{"detail":"token expired"}`, `401 {"detail":"token revoked"}`, `401
{"detail":"unknown token"}`. There is no API to pause an admin token — pausing
only ever touches an inference key — so an admin token never answers `403
token paused`.

**Refresh and revoke** (session-only):

```bash
# A new secret with the same name, owner and term, starting now. The old one
# keeps working for grace_hours: 0, 1 (the default) or 24.
curl -s -X POST $W/api/admin-tokens/4f0c…/rotate "${AUTH[@]}" -d '{"grace_hours":1}'
# 201 {"id":"a81d…","name":"ci-deploy","rotated_from":"4f0c…", …, "plaintext":"vwa_…"}

# Revoke now. The row stays, dimmed in the UI, so its trail stays readable;
# its open streams are closed.
curl -s -X DELETE $W/api/admin-tokens/4f0c… "${AUTH[@]}"     # 204
```

A refreshed, revoked or expired token cannot be refreshed again (`409`): issue
a new one.

Revoking a refreshed token does **not** end its predecessor's grace window —
they are separate rows. If you refreshed `ci` and are now revoking it in
response to a leak, also revoke `ci (old 1)` (or whatever the predecessor was
renamed to); it is listed with status "grace" and stays usable, with the
original secret, until its grace period ends.

**Streams.** An admin token opens the same server-sent-event streams a session
does by sending the header instead of a ticket — god mode, model logs, live
stats, header metrics and model pull progress — plus the chat2 turn streams and
the chat playground's completion proxy (`POST /api/chat/completions`), which
always took a bearer header (never a ticket) and now accept an admin token too.
While a stream is open, it is re-checked every 60 seconds: it ends when the
token expires, is revoked, falls out of a refresh's grace window, or is
deleted. Revoking a token, or refreshing it with `grace_hours: 0`, ends its
open streams at once instead of waiting for the next check. That covers every
stream the warden opens under a credential, an in-flight playground generation
included — revoking a token stops the text mid-sentence rather than letting it
run to the end.

**Audit.** Every request an admin token makes to a route is recorded shortly
after the response, never on the response path: the row is queued when the
response is known and a background writer commits a batch of them — at most
200 rows or one second behind, whichever comes first — so a polling script does
not pay a database commit per call. Each row keeps the time the request
*arrived*, so the trail reads in order regardless of how the batches fell. The
queue is bounded (10 000 rows); if a stalled data volume ever overflowed it the
rows are dropped and counted, and so is a batch the database refuses, so both
kinds of gap show up in one number. Drops are logged from the enqueue side (at
most once a minute) as well as at each flush, so a gap in a trail is visible in
the log rather than silent even if the background writer itself is what
stopped. A clean shutdown writes what is still queued, including a batch the
writer had already picked up; a `kill -9` does not. What each row holds: time, method, the
**route template**
(`/api/models/{model_id}/load`, not the concrete URL), status, duration,
`client_ip` (proxy-reported, from `X-Forwarded-For` — the holder of the token
can forge it) and `peer_ip` (the TCP socket peer, which it cannot). A refused
attempt is audited too, as soon as an auth dependency has identified the
secret as a known admin token — an expired or revoked token still being
tried, or one probing a session-only route, lands in its own trail, and so
does a well-formed body that fails validation (`422`). A secret matching no
admin token; anything refused before routing (an unmatched `404`/`405`) or by
an outer layer (chat2's request-body-size `413`); and a malformed or
undecodable JSON body — FastAPI rejects it before any dependency, including
auth, runs — leave no row: no admin token was ever identified for that
request. A stream's row is written once, when the stream closes, covering
its whole open duration. Rows are kept 90 days, at most 20 000 per token, and
at most 200 000 across all tokens (each pass drops the oldest rows first). The
prune runs the 90-day window first, then the per-token cap, then the global
cap. The per-token cap is what keeps one token generating a flood of requests
from evicting anybody else's history: by the time the global cap is applied,
the flood has already been trimmed to that token's own 20 000. Read them
newest first, 50 at a time,
passing `next_before` back as `before`:

```bash
curl -s $W/api/admin-tokens/4f0c…/audit?limit=50 -H "Authorization: Bearer $JWT"
# {"items":[{"id":812,"ts":1758300000.25,"method":"POST",
#            "path":"/api/models/{model_id}/load","status":202,
#            "duration_ms":12,"client_ip":"198.51.100.7","peer_ip":"127.0.0.1",
#            "username":"admin"}, …],
#  "next_before":1758290000.0}
```
