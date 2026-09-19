# Driving LLM Warden from the API

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
#    Password rules: at least 6 characters and at most 72 BYTES. The upper
#    bound is bcrypt's, and it is rejected rather than silently truncated —
#    worth knowing before you generate a long passphrase.
curl -s -X POST $W/api/setup/admin -H 'Content-Type: application/json' \
     -d '{"username":"admin","password":"CHANGE-ME"}'
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
        -d '{"username":"admin","password":"CHANGE-ME"}' | jq -r .access_token)

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
