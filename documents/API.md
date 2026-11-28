# Driving LM Warden from the API

The browser UI is a client of the control API and nothing more. Everything it
does — the first-run wizard, minting a key, registering, pulling and loading a
model — is a handful of `curl` calls, and this document is those calls, for a
script, a CI job or an agent that will never open `/ui/`.

[INSTALL.md](INSTALL.md) walks the same first run and the same model add as a
recorded transcript, with the failures it hit on the way and what they meant.

The calls were recorded against the Linux install. The Mac app (beta) serves
the same OpenAI-compatible API, on `http://127.0.0.1:8080` by default; it has
no vLLM, so the vLLM-only fields below (tensor and data parallelism among
them) do not apply there.

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

A fresh install cannot be claimed by whoever reaches it first: every
`POST /api/setup/*` needs the install's one-time **setup code** in an
`X-Setup-Code` header. It is `LM_SETUP_CODE` in `.env` when that is set
(`install.sh` generates one and prints it at the end), otherwise the server
generates one, keeps it in `/data/setup-code` and prints it in its log (on
PodWarden Hub, it is `LM_SETUP_CODE` in the deployment's settings). It works
until the admin exists, then never again. To skip the wizard entirely,
set `LM_BOOTSTRAP_ADMIN_USERNAME` and `LM_BOOTSTRAP_ADMIN_PASSWORD` instead:
the admin is created at first boot.

These calls send no cookie, so they need no CSRF token and no cookie jar.
They are a strict state machine: posting out of order returns
`400 not at <step> step (current: <where you are>)`, and `GET /api/setup/state`
always tells you where that is — so an interrupted run resumes rather than
restarts.

```bash
W=http://127.0.0.1:8080
# The setup code: from .env, or from the container.
CODE=$(grep '^LM_SETUP_CODE=' .env | cut -d= -f2)
[ -n "$CODE" ] || CODE=$(docker compose exec -T api cat /data/setup-code)
S="X-Setup-Code: $CODE"

# 1. Where are we? A fresh install answers
#    {"step":"welcome","done":false,"code_required":true,"code_source":"env"}.
curl -s $W/api/setup/state

# 2. Acknowledge the welcome step.                    -> {"step":"gpus"}
curl -s -X POST $W/api/setup/welcome -H "$S"

# 3. List the GPUs the container can see, with their indices.
curl -s $W/api/setup/gpus
# [{"index":0,"name":"NVIDIA RTX A4000","memory_total_mib":16376,
#   "memory_used_mib":0,"utilization_pct":0}, ...]

# 4. Choose which of those indices the engine may use.  -> {"step":"hf_token"}
curl -s -X POST $W/api/setup/gpus -H "$S" -H 'Content-Type: application/json' \
     -d '{"allowed_gpu_indices":[0,1]}'

# 5. HuggingFace token. JSON null is valid and is the right answer unless you
#    need gated models (Llama, Mistral, gpt-oss).      -> {"step":"admin"}
curl -s -X POST $W/api/setup/hf_token -H "$S" -H 'Content-Type: application/json' \
     -d '{"hf_token":null}'

# 6. Create the admin account.                          -> {"step":"done"}
#    Password rules: at least 12 characters and at most 72 BYTES. The upper
#    bound is bcrypt's, and it is rejected rather than silently truncated —
#    worth knowing before you generate a long passphrase.
curl -s -X POST $W/api/setup/admin -H "$S" -H 'Content-Type: application/json' \
     -d '{"username":"admin","password":"CHANGE-ME-to-12-plus-chars"}'
```

`GET /api/setup/gpus` returns `404` once setup is done — deliberate, so a
completed install does not report its hardware to anonymous callers.

**Then mint an API key.** With a cookie jar, the rest of the control API
enforces CSRF, so this is a two-header call: the CSRF token from `/api/csrf`
plus the JWT from the login. (Without a cookie jar the JWT alone is enough: a
request carrying no LM Warden cookie skips the CSRF check.)

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

`engine_vllm_version` pins the engine binary a model launches. A bare
version id with no `engine_channel` and no `engine_image` selects an engine
bundled in the image under `/opt/lmw-engines/<version id>/` — the warden
launches that install's `vllm` instead of the one on PATH, and refuses the
load (before claiming any GPU) when the id names no bundled engine. A pin
that sets `engine_channel` selects the docker-driver image path instead and
is never treated as bundled.

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

### Watching a load

`POST /api/models/{id}/load` answers `202` and the row's `status` walks to
`loading`, but the row alone does not say *where* the load is. Two endpoints do.
Both are JWT-gated (an admin token works too) and return the same shape:

- `GET /api/models/{model_id}/load-progress` — one model's live load. `404`
  when no load is in flight for `model_id` (it finished, never started, or was
  superseded — poll `GET /api/models/{model_id}` for the row's `status`).
- `GET /api/models/load-progress` — every active load at once, for the
  dashboard cards. An empty array when nothing is loading. Registered before
  `/{model_id}` so the path `load-progress` is not captured as a model id.

```bash
# One model's load (404 when nothing is loading for it).
curl -s $W/api/models/$M/load-progress "${AUTH[@]}"

# Every active load (empty array when none).
curl -s $W/api/models/load-progress "${AUTH[@]}"
```

The response is a snapshot built every poll (the UI polls roughly every 2 s)
and is total — it never raises mid-load. Its fields:

| Field | Meaning |
|-------|---------|
| `model_id` | The model being loaded. |
| `backend` | `vllm` or `llamacpp`. |
| `layout` | A human-readable parallelism label (e.g. `3 replicas × TP2`, `TP2`), or `null` for a single-engine TP1 load, or when the model row was deleted mid-load. |
| `elapsed_s` | Seconds since the load started (monotonic). |
| `state` | `progressing` (no stall), `stalled` (amber — no progress for `VW_LOAD_STALL_WARN_S`), or `give_up` (the warden is stopping the engine — no progress for `VW_LOAD_STALL_TIMEOUT_S`, or the absolute cap `VW_LOAD_MAX_S` was reached). |
| `stalled_for_s` | How long the load has been stalled (0 while progressing). |
| `give_up_at` | The wall-clock datetime the load will be given up at if nothing moves, or `null`. |
| `reason` | Plain-English reason for a `stalled`/`give_up` verdict (e.g. an NCCL init guess), or `null`. |
| `hint` | A short suggested action for a `stalled`/`give_up` verdict, or `null`. |
| `usual_total_s` | The median total load time from the last five successful loads with the same engine settings, or `null` when no history exists yet. |
| `ready_eta` | A wall-clock datetime estimate of when the load will finish, based on `usual_total_s`, or `null` without history. |
| `current_step` | The key of the running step (one of `START`, `WEIGHTS`, `COMPILE`, `KV`, `TUNE`, `GRAPHS`, `WARMUP`). |
| `steps` | The step checklist — see `StepOut` below. |
| `vitals` | Vital signs — see `VitalsOut` below. |

Each entry in `steps` (`StepOut`):

| Field | Meaning |
|-------|---------|
| `key` | Step key: `START`, `WEIGHTS`, `COMPILE`, `KV`, `TUNE`, `GRAPHS`, `WARMUP` (vLLM); llama.cpp uses `START`, `WEIGHTS`, `WARMUP` only. |
| `label` | Human-readable step label (e.g. "Load weights", "Tune kernels"). |
| `status` | `done` (the step ended), `running` (the first not-done step that has started, or the first not-done step while progressing), `stalled` (that same step when the verdict is not progressing), `pending`, or `skipped` (a step the engine never reports, e.g. `TUNE` on a model with no AutoTuner, once a later step has started). |
| `duration_s` | The step's duration when done, or `null`. |
| `elapsed_s` | The step's elapsed time while running, or `null`. |
| `count` / `total` | The step's counter progress (e.g. weights shards `30/66`, CUDA graphs `8/10`, AutoTuner ops `5/22`), or `null`. |
| `usual_s` | The historical per-step duration from load history, or `null` without history. |
| `detail` | Step-specific extra text (e.g. `"trtllm_fp4_block_scale_moe · op 2"` for `TUNE`), or `null`. |

`vitals` (`VitalsOut`) — a time series of up to 120 points:

| Field | Meaning |
|-------|---------|
| `series` | One point per vitals sample (downsampled to ≤ 120 evenly spaced points plus the newest). Each point has `t_s` (seconds since load start), `cpu_cores` (cores busy, `ΔCPU-s / Δt`; `null` for the first point or when a sample is missing), and `gpu_mem_mib` (GPU memory used, in MiB; `null` when the probe failed). |
| `gpu_mem_total_mib` | The summed capacity of the model's GPUs, in MiB, or `null` when not known (e.g. the docker driver). |
| `last_log_age_s` | Seconds since the engine last wrote a log line. |
| `last_log_line` | The last log line, truncated to ≤ 200 characters. |
| `log_quiet_normal` | `true` when the log has been quiet for > 60 s while CPU or GPU memory still moved — a "slow but alive" signal. |

Progress means: a new log line, a step or counter change, GPU memory growth,
or disk reads. A stuck load (no progress past the stall timeout, or past the
absolute cap) is stopped — the engine is killed and the GPUs released — and the
model is marked `failed` with the stall reason. There is no automatic retry.

---

## Model aliases

A **model alias** is a named, first-class object clients call. It holds an
**ordered list of target models** (loaded or not) plus optional proxy-side
settings; it is not a rule, not a template, and not a copy of a model. The
proxy resolves a request's `model` against alias names **first**, then served
names, so an alias may share the name of a model and then wins (`lmw-default`
→ `[qwen]` answers as `lmw-default` while `qwen` is served). It is how clients
survive a model swap.

Rules (owner decisions 2026-11-23/24):

| Rule | Meaning |
|------|---------|
| D4a name priority | A request's `model` is matched against alias names **first**, then model served names. An alias may have the same name as a model and then wins. Alias names are unique among aliases only. |
| D4b fallback | The first target in the list that is **loaded** serves the request. If none is loaded, the answer is `404` `no model behind alias '<name>' is loaded` -- the same status as today's not-loaded answer. |
| D4c settings | `guard` (sensitive data): `inherit` \| `sanitize` \| `block` -- the **stricter** of the key's guard and the alias's applies (block > sanitize > none); an alias never weakens a key. `thinking_policy`: `inherit` \| `client` \| `on` \| `off` -- precedence key > alias > model. `params`: a map of request parameter → `{value, mode}`, mode `default` (used only when the client sent none), `force` (always replaces) or `cap` (numeric only: `min(client, value)`, or `value` when the client sent none). Allowed parameters: `max_tokens` int ≥1 (all three modes), `temperature` float 0–2, `top_p` float 0–1, `top_k` int ≥−1, `min_p` float 0–1, `presence_penalty` / `frequency_penalty` float −2–2, `repetition_penalty` float 0–2, `seed` int, `reasoning_effort` one of `low|medium|high` (default/force only), and `chat_template_kwargs.<key>` entries (any JSON scalar, default/force only). Anything else is refused at write time (`422`). |
| D4d allow-list | A request through an alias is allowed when the key's model allow-list contains the alias name **or** the served name of the model that serves it. |
| D4e `/v1/models` | `/v1/models` lists each alias that has a loaded target (allowed for the key) under the alias name, and hides a model whose served name is shadowed by an alias. |

Endpoints (JWT; the same credential as every other control route):

| Route | Meaning |
|-------|---------|
| `GET /api/aliases` | `{"aliases": [...]}` — each alias with its targets (`id`, `served_model_name`, `status`), settings and `enabled`. |
| `POST /api/aliases` | `201`; `name` (same pattern and length as `served_model_name`) is unique among aliases (a duplicate is `409`); targets must exist (`422`); 0–8 distinct targets. A name may equal a model's served name. |
| `GET /api/aliases/{id}` | `200` or `404`. |
| `PATCH /api/aliases/{id}` | `200`; omitted keys are left alone; a duplicate name is `409`; `422` for a value that breaks the rules above. |
| `DELETE /api/aliases/{id}` | `204`; `404` if unknown. |

Deleting a model **removes it from every alias's `targets`** (the alias itself
survives, with an empty or shortened list). Re-adding it means re-adding it
to the targets.

---

## Health and readiness

Two public routes (no credential) for two different questions, and an admin
route with the detail:

| Route | Question | Status |
|-------|----------|--------|
| `GET /healthz` | Is the warden process up? (liveness) | Always `200` while the api answers |
| `GET /readyz` | Can it serve a completion right now? (readiness) | `200` or `503` |
| `GET /api/admin/readiness` | Same as `/readyz`, per model (admin credential) | `200`, `503`, `404` |

`/healthz` never fails over models. A model load takes minutes, and a
liveness probe that failed meanwhile would get the api restarted mid-load, so
the load would never finish. It reports model state in its body instead:

```
{"ok": true,
 "gpu": {"state": "ok", "error": null, "since": <epoch s>, "checked_at": <epoch s>},
 "models": {"serving": false,
            "counts": {"loading": 1, "ready": 0, "failed": 0,
                       "unloading": 0, "unloaded": 2}}}
```

`serving` is `true` when at least one model passes the router's own check
(status `loaded` and the engine is running), the check a `/v1` request meets
before it is forwarded. `/healthz` makes no engine call to get it. If the
models table cannot be read, `models` is `{"serving": null, "counts": null}`
and the status is still `200`. The states:

| State | Meaning |
|-------|---------|
| `loading` | Loading, or failed and about to be restored by the watchdog after a restart (only while the watchdog will do it: probing enabled, or the boot restore has not run yet) |
| `ready` | The router forwards requests to it |
| `failed` | The load failed or the engine is gone, and nothing will restore it by itself |
| `unloading` | Being unloaded |
| `unloaded` | Not meant to be serving (registered, pulling, pulled) |

`/readyz` answers `200` only when at least one `ready` model's engine also
answers its own health path now (2 s timeout). Otherwise `503`. The body is
only a verdict and a reason code:

```
200 {"ready": true,  "reason": null}
503 {"ready": false, "reason": "loading"}
```

| `reason` | When |
|----------|------|
| `loading` | A model is loading or being restored |
| `engine_unhealthy` | A model is `ready` but its engine did not answer |
| `failed` | Every model meant to serve has failed |
| `unloading` | The only model meant to serve is being unloaded |
| `no_models_configured` | No local model is loaded or loading, including on purpose |
| `unavailable` | The models table could not be read |

Both public routes are deliberately anonymous: they never name a model, say
when it changed, or carry an error message. An admin (session or admin token)
gets that on `GET /api/admin/readiness`, with the same status codes as
`/readyz` and the same reason codes:

```
curl -s -H "Authorization: Bearer $VW_ADMIN_TOKEN" http://HOST:8080/api/admin/readiness
503 {"ready": false, "reason": "loading",
     "models": {"serving": false, "counts": {...},
                "items": [{"id": "9895a1829559533c", "name": "qwen2.5-1.5b",
                           "state": "loading", "since": "2026-11-22T12:24:10Z"}]}}
```

Each item may also carry `engine` (`ok` or `unhealthy`, for a `ready` model)
and `error` (the model's `last_error`, for a `failed` one). Models that are not
meant to serve are counted, not listed. `?model=<id or served name>` narrows
the verdict to one model (`404` for an unknown one).

Things to know:

- **Ready means at least one model.** On a multi-model warden `/readyz` is
  `200` as soon as any model can serve. To wait for a particular model, ask
  `/api/admin/readiness?model=<name>`.
- **Relayed models do not count.** A warden that serves only through router
  rules relaying to an upstream (for example Claude names sent on to
  Anthropic) and has no local model answers `503 no_models_configured`, even
  though `/v1/messages` works. `/readyz` checks local engines only.
- **The answer is shared for about 1 s.** Every caller gets the same result
  for a second, and concurrent callers wait on one check; a model status
  change starts a new one at once.
- **Front door.** The repo's Caddyfile routes `/readyz` to the api. A
  PodWarden Hub catalogue install carries its own Caddyfile, which (as of
  2026-11-22) answers unknown paths with Caddy's own `404`, so `/readyz`
  returns `404` there until that Caddyfile gains a `handle /readyz` route.
  `/api/admin/readiness` works there already (it is under `/api/`).

Wait on `/readyz` (or the admin route), not `/healthz`, before sending
traffic after a deploy or a restart. Keep `/healthz` as the container's
liveness probe. Do not make `/readyz` the api container's readiness probe
either: a warden with no model loaded is correctly not ready, and the same
container serves the UI and the API you would use to load one.

---

## Engine capability

`GET /api/system/engine` (JWT) reports the active engine driver's
capability: `driver` (`"docker"` or `"subprocess"`), `supports_version_select`
(whether this driver can honor an engine-version pin), `vllm_version` (the
vLLM package version baked into the image, or null) and `bundled_vllm` — the
sorted list of version ids of the engines bundled in the image under
`/opt/lmw-engines` (empty when none is installed). A model row's
`engine_vllm_version` names one of those ids; see the model section above.

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

### Prompt Lens

**Prompt Lens** is a tool for seeing what a
client really sent and what came back, prompt part by prompt part. It is off
unless turned on: by the switch in Settings, General, Privacy (the runtime
setting `prompt_lens_enabled`, applied at once), or, while that switch has never
been saved, by `VW_PROMPT_LENS_ENABLED` (see
[Where request content can end up](OPERATING.md#where-request-content-can-end-up)).
The console asks first, and shows the Prompt Lens menu entry and page only when the answer is yes; the answer
also says where the value in effect comes from (`source`: `setting`, `env` or
`default`, and which `env_var`):

```bash
curl -s $W/api/admin/prompt-lens/status -H "Authorization: Bearer $JWT"
# {"enabled":true,"source":"env","setting":null,"env_var":"VW_PROMPT_LENS_ENABLED","env_value":true}
```

Only a signed-in session can change the switch (`PATCH /api/settings/runtime`
`{"prompt_lens_enabled": false}`; `null` clears it, "Use server default", so
the env variable, else the default (on), decides again; an admin token gets 403 `session_only`). Every
save is a row in `prompt_lens_switch_audit` (who, when, through what, value and
source before and after); the status carries the newest as `last_change`.
Turning it off stops capturing at once (a request already running captures
nothing more, even after it is turned back on), ends every open Prompt Lens
stream (admin and key holder) and clears the ring and the section store. A
key's kept captures stay on disk, and while it is off the admin routes
`/kept`, `/kept/{req_id}` and `/requests/{req_id}/section/{kind}` still serve
them (read-only; nothing from memory); the key holder's routes answer 404.

A key holder also needs the admin's permission for their key (below, "Per-key
permissions"): Prompt Lens for a holder is on when the switch above is on **and**
the key's `prompt_lens` permission is on.

Prompt Lens was called god mode before 2026-11-22; the old routes, field and
`VW_GODMODE_*` variables were removed in the release after v2026.11.24.1.

**The admin stream.** A server-sent-event stream behind a single-use ticket,
because a browser's `EventSource` cannot send an `Authorization` header. Mint
the ticket for the **bare** path — no query string — then pass it, and
`token_ids`, as query parameters:

```bash
TICKET=$(curl -s -X POST $W/api/auth/sse-ticket "${AUTH[@]}" \
           -d '{"path":"/api/admin/prompt-lens/stream"}' | jq -r .ticket)

curl -sN "$W/api/admin/prompt-lens/stream?token_ids=$T,0b91…&ticket=$TICKET"
# data: {"type":"ring","oldest_ts":…,"events":1840,"chars":3912007,"chars_max":4000000}
# data: {"type":"request_start","req_id":"…","token_id":"53e7011c…", …}
# data: {"type":"delta","req_id":"…","token_id":"53e7011c…", …}
# data: {"type":"request_end","req_id":"…","token_id":"53e7011c…", …}
```

`token_ids` is a comma-separated list of at most 20 key ids. Both the replay
of recent requests and the live events are narrowed to those keys, and every
event carries `token_id`, so a request is never split from its deltas. An
empty list, or more than 20 ids, is a **422**. Leave the parameter out and the
stream is every key's traffic. With Prompt Lens off, the ticket mint and the
stream both answer **409** `Prompt Lens is disabled (Settings, or VW_PROMPT_LENS_ENABLED)`.

**Scripts: the admin stream now starts with a `ring` frame.** Right after the
`: connected` comment comes `{"type":"ring",…}`: how many events and
characters the replay ring holds (`events`, `chars`, `chars_max`) and the
server time of the oldest one (`oldest_ts`, or null when the ring is empty).
It repeats every 30 seconds.
A script that reads only the request events should skip frames whose `type`
it does not know. The key holder's stream never has it (those counts are
every key's).

The stream is open only as long as you hold it. The Prompt Lens page works the
same way: it connects when you open it and disconnects when you leave it.

**Events.** Every data frame is one JSON object with `type`:

| `type` | Fields |
|---|---|
| `request_start` | `req_id`, `ts`, `seq`, `token_id`, `token_label`, `request_id` (the `X-LMW-Request-Id`), `model`, `served_name`, `client_ip`, `stream`, `prompt` and `prompt_elided` (the flat prompt, kept one more release for scripts), `prompt_parts` (the prompt by message: the newest messages whole, older ones counted in `older`/`omitted`), `session_id`, `turn`, `media` (inline images, by id) |
| `kvsig` | `req_id`, `token_id`, `ts`, `seq`, `session_id`, `turn`, `queued_s` (sent at admission), `available` (`reason` when false), `profile`, `head8`, `conv8`, `edge`, `parent_request_id`, `exact_depth`, `loose_depth`, `n_nodes`, `reusable_own`, `reusable_any`, `parts` (each `p` = `sys`, `tools` or `nodes`, `from`, `to`, `chars`, `roles`, `exact8`, `loose8`, `state` = `exact`, `loose`, `new` or `other`) |
| `delta` | `req_id`, `token_id`, `ts`, `seq`, `channel` (`reasoning`, `content` or `tool_call`), `text`; a `tool_call` fragment also has `index`, and `call_id`, `name` and `elided` when it carries them |
| `progress` | `req_id`, `token_id`, `ts`, `seq`, `phase`, `frames` (`reasoning`, `content`, `tool_call`), `completion_tokens`. Live only: never replayed |
| `request_end` | `req_id`, `token_id`, `ts`, `seq`, `finish_reason`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `cached_tokens`, `queued_s`, `ttft_s`, `duration_s`, `decode_tps`, `own_reused_tokens`, `cache` (the cache observation), and `tool_calls_dropped` / `tool_call_chars_dropped` only when tool-call text was cut |
| `ring` | admin stream only, above |
| `lagged` | the client fell behind its queue and the oldest events it had not read were dropped; sent once per gap |

Every request that has a `request_start` gets exactly one `request_end`. A
request the sensitive data guard blocks before it starts has neither: it is in
the guard log, not in Prompt Lens. Requests relayed to Anthropic by the router
do not appear in Prompt Lens yet.

**What a key holder never receives.** The key holder's stream, sections and
kept captures remove, from every event: `client_ip`, `token_label`, the hub's
`seq` (their stream numbers its own frames instead), `reusable_any`,
`cached_tokens`, `cache` and the names inside it (`reusable_tokens`,
`reusable_tokens_fleet`, `cache_outcome`, `divergence_cause`,
`other_key_label`). A prompt part another key made cached reads `new` for
them (the admin sees `other`). They never get a `ring` frame.

**Whole sections.** The stream carries a request's text as it streams; the
Prompt Lens page keeps a bounded tail of it. The whole section comes from:

```bash
curl -s "$W/api/admin/prompt-lens/requests/$REQ_ID/section/thinking?from=0" -H "Authorization: Bearer $JWT"
# {"kind":"thinking","seq":91822,"live":true,"dropped_chars":0,"offset":0,"text":"…","dropped_calls":0,"kept":false}
```

`kind` is `thinking`, `reply`, `tool_calls` (`calls`, by index), `tool_results`
or `prompt` (`messages`, system and tools first, and `older`, the collapsed
summary). `offset` is where `text` starts, so `offset` plus its length is the
section's length: fetch again with `?from=` that number to get only what came
since. `dropped_chars` counts the oldest text the store dropped. Bounds:
[Where request content can end up](OPERATING.md#where-request-content-can-end-up).
A key holder reads their own with `GET /api/me/prompt-lens/requests/{req_id}/section/{kind}`;
`seq` is then `null`.

**Findings on a request.** Once a request ended, its findings come from the
stored request history of its session (layers E reuse, C construction and R
replies; every rule, its threshold and its fix are in `app/lens/catalogue.py`):

```bash
curl -s "$W/api/admin/prompt-lens/requests/$REQ_ID/findings" -H "Authorization: Bearer $JWT"
# {"turn":9,"names":{"5c1e0a9f":"search_docs"},"findings":[
#   {"rule":"E-01","turns":[2,3,9],"counts":{"line":2,"class":1,"turns":23,"lost":239200,"sessions":1},
#    "tool":null,"cause":"sys_changed","saving":{"tokens":239200,"prefill_s":null,"decode_s":null,"money":null},
#    "confidence":"high","muted":false,"part":"sys","guard":null},
#   {"rule":"R-04","turns":[7,9,12,14],"counts":{"bad":4,"calls":6,"code":3},"tool":"5c1e0a9f",
#    "cause":"schema_undocumented","saving":{…},"confidence":"high","muted":false,"part":"call:0","guard":null}]}
```

`req_id` is the Prompt Lens id of the request (in memory or a kept capture) or
its stored request id. Each finding lists the session turns it is about (this
request's `turn` among them), its counts, a keyed tool-name hash (`tool`), the
likely cause, the saving and a confidence. `counts.info` = 1 marks a notice,
never a fix; E-01's `counts.class` indexes the span classes (`date`, `time`,
`path`, `uuid_or_hex_id`, `version`, `number`, `url`, `provider_or_model_id`,
`user_name_or_email`, `other`) and R-04's `counts.code` the call verdicts
(`ok`, `json`, `unknown_tool`, `missing_arg`, `type`, `leaked_markup`). `part`
says where the finding sits on the request: `sys`, `tools`, `node:<i>`,
`call:<k>` (the reply's tool call by index), or `null`. `guard` (R-09 only)
holds the guard rule ids and where they matched. `names` maps the tool hashes
above to the names the caller already sees on the Sessions pages. Nothing in
the answer is prompt, reply or tool text. The admin route answers for any key,
also while Prompt Lens is off; **404** `no such request` for an unknown id.

A key holder reads their own with `GET /api/me/prompt-lens/requests/{req_id}/findings`:
the analysis uses their own key's figures only (no other key's reuse, no
fleet-wide cause), and names only their own tools. Prompt Lens off, an unknown
id, an evicted one and another key's all answer the same **404** `no request
of yours with this id`; an admin session or admin token gets **403**. A key may
make 60 findings reads a minute; past that the answer is **429** with
`Retry-After`.

**Scores and the scorecard.** A session that has been quiet for 10 minutes
is scored on its stored requests (so it works with Prompt Lens' capture off,
and after the ring has forgotten it): E (cache reuse), C (construction), R
(replies) and the overall, each 0-100 with 100 meaning "nothing measurable to
fix in this layer". The formulas, minimums and weights are in the analysis
spec (10) and are published in every answer (`weights`, `score_version`).

```bash
curl -s "$W/api/admin/prompt-lens/sessions?since=0&key=$TOKEN_ID" -H "Authorization: Bearer $JWT"
# {"sessions":[{"session_id":"s_9f2c…","model":"qwen3","route":"local","harness":"other","turns":24,
#   "overall":62,"e":48,"c":71,"r":74,"confidence":"high","reasons":{},"score_version":1,
#   "key_name":"alpha-key","key_id":"…"}],"next_cursor":null}
curl -s "$W/api/admin/prompt-lens/sessions/s_9f2c…/scorecard" -H "Authorization: Bearer $JWT"
# {"session":{…},"scores":{"overall":{"layer":"overall","scored":true,"value":62,"lo":57,"hi":66,
#   "confidence":"high","turns":24,"inputs":{"from":["E","C","R"],…}},
#   "R":{"layer":"R","scored":false,"turns":3,"reason":"3 of 5 assistant turns"}, …},
#  "top":[…3 findings…],"notice":null,"findings":[…],"turns":[{"n":1,"reused":0,"recomputed":0,"new":6000,"markers":[]},…],
#  "cost":{…},"weights":{"E":0.45,"C":0.15,"R":0.4},"score_version":1,"review":null}
```

The list takes `key` (admin only), `model` and `harness` (the client: `claude_code`,
`codex`, `opencode`, `openai_sdk`, `other`); each may repeat (`key=a&key=b`) and
matches any of its values, and the three combine. A key holder's list takes `model`
and `harness` only.

A layer below its minimum data has **no number**: its entry is
`{"layer", "scored": false, "reason", "turns"}` and the overall needs two
scored layers. A score has a 10th-90th percentile range (`lo`, `hi`, a
bootstrap seeded by the session id, so a reload shows the same range) and a
confidence (`high`, `medium`, `low`). `top` is the three fixes worth most
(money with a price card, else GPU-seconds, else tokens; weighted by
confidence); `notice` is a secret in the prompt (R-09), pinned above them. The
scorecard holds codes, counts and positions only: never prompt, reply or tool
text. Money appears only with a price: a relayed route uses its provider's
price, an own model the reference price an admin picked per model (see
[Prices](#prices) below; the price's date is shown with every money figure).
A model with no known price shows tokens, as before.

A key holder reads their own with `GET /api/me/prompt-lens/sessions` and
`GET /api/me/prompt-lens/sessions/{ref}/scorecard`: the key is the sign-in's
(a `key` parameter is ignored), the figures are their own key's only (no rank,
percentile or other key's number), and an unknown ref, an evicted session and
another key's all answer the same **404**; an admin session or admin token gets
**403**. Each list item carries `key_name`, the API key's name (null when the
key has been deleted); a key holder gets its own key's name only, and `key_id`
stays admin-only. `since` (epoch seconds, default 30 days), `model`, `limit` (at most
200) and `cursor` page the list. A key may open 30 scorecards a minute; past
that the answer is **429** with `Retry-After`.

**Prompt versions, trends and the comparison.** Every request is stamped with
a prompt version: detected from the system prompt's lines (lines holding a
date, a time, a path, an id or a version number are left out, so a changing
working directory never starts a version) plus the tool list, or named by the
client with an `X-Prompt-Version` header (at most 64 printable characters; it
wins over detection and is shown as the client wrote it). The stored value is a
keyed hash or that name, never prompt text. With kvsig off nothing is computed.

```bash
curl -s "$KEY_HOLDER/api/me/prompt-lens/trends?by=week&model=qwen3" -H "Authorization: Bearer $KEY_HOLDER_TOKEN"
# {"by":"week","model":"qwen3","models":["qwen3"],
#  "points":[{"bucket":"2026-11-23","n":9,"medians":{"overall":64,"E":58,"C":71,"R":80},"band":[55,70],"start":…}],
#  "versions":[{"id":"3fa9c2d1","label":"v1","since":…,"n":9,"medians":{…},"rules":{"E-01":7},"resolved":[],"new":[]}], …}
curl -s "$W/api/me/prompt-lens/compare?a=3fa9c2d1&b=b17e0c44" -H "Authorization: Bearer $KEY_HOLDER_TOKEN"
# {"a_label":"v1","b_label":"v2","n_a":23,"n_b":31,"diffs":{"overall":12,"E":18,"C":3,"R":4},
#  "verdict":"better_clear","like_for_like":true,"note":null,"resolved":["E-01"],"new":[],"p":0.004}
```

`GET /api/me/prompt-lens/trends` (`by=day|week|version`, `model`, `since`, `tz`
in minutes east of UTC), `/versions` and `/compare` read the signed-in key's own
scored sessions, across its rotation chain, with its won't-fix marks applied
(below). A bucket with fewer than 3 scored sessions has `null` medians and no
`band`, never a 0. `compare` sets version `b` against the earlier `a` on the
same model: the median difference per layer (needs 5 scored sessions a side,
else `verdict` is `not_enough`), and a verdict from a rank test on the per-session
overall: `better_clear` / `worse_clear` at p < 0.05, `better_unclear` /
`worse_unclear`, `no_change`. `like_for_like` is false, with a `note`, when the
sessions' harness or route differs or their typical length differs by half.
`resolved` lists the rules that fired on `a` and not on `b`. A key holder cannot
name a key: a `key` parameter is **422**. The admin's routes
(`/api/admin/prompt-lens/trends|versions|compare|wontfix`) take a required
`key=`, and `GET /api/admin/prompt-lens/fleet?model=&since=` gives the
distribution (10th, 25th, 50th, 75th and 90th percentile) of the stored scores
over every key; it is the admin's alone (a key holder's sign-in or an inference
key is **401**).

**Won't fix.** A key holder (or the admin, for any key) can mark a rule "won't
fix" for the key, for every prompt version or one. A marked rule stops counting
in the scores (the scorecard's `excluded_rules` names it and `inputs.with_marks`
holds the value with it counted), its findings stay listed with `muted: true`
and `muted_by` (`key` or `admin`) and never enter `top`, and unmarking restores
the scorecard at once. A mark limited to one version stops applying when the
prompt version changes. Marks follow the key's rotation chain.
`GET /api/me/prompt-lens/wontfix` lists them; `PUT /api/me/prompt-lens/wontfix/{rule}`
with `{"version": "<id>" | null}` marks; `DELETE …/wontfix/{rule}?version=<id>`
removes the mark for that version (no `version`: the one for every version). The
answer is `{"marks":[{"rule","version","by","at"}],"changed":bool}`. An unknown
rule and **R-09** (a secret in the prompt: a notice, not a score term) answer
**422**. Every mark and unmark writes one `admin_audit` row (the key, who, the
rule and the version are in `path`, after the route template). The stored
scores and the fleet distribution are computed without
the marks, so a mark cannot move a key within the fleet.

**Keep captures.** An admin can set a key to keep its last finished requests
on disk: `PATCH /api/tokens/{id}` with `{"keep_captures": true}` (off by
default; the key's page has the switch). The key's last 50 finished requests,
at most 32,000,000 characters in all, are written to the database, oldest
dropped first, prompts included. They outlive the in-memory store and a
restart. Turning it off deletes the rows of the key and of the keys it was
rotated from; deleting the key deletes them too. While it is on, the key
holder's capture note says the last 50 are kept on disk for this key. Lists
and reads: `GET /api/admin/prompt-lens/kept?token_ids=…`,
`GET /api/admin/prompt-lens/kept/{req_id}`, and for the key holder
`GET /api/me/prompt-lens/kept` and `GET /api/me/prompt-lens/kept/{req_id}`. A key may
make 60 kept reads a minute; past that the answer is **429** with
`Retry-After`.

**A key holder's live stream.** The key holder's sign-in mints a single-use,
60-second ticket and opens the stream of its own key's requests:

```bash
TICKET=$(curl -s -X POST $W/api/me/prompt-lens/ticket -H "Authorization: Bearer $KEY_HOLDER_TOKEN" | jq -r .ticket)
curl -sN "$W/api/me/prompt-lens/stream?ticket=$TICKET"
```

The stream takes no key parameter and answers **404** while Prompt Lens is off.
It replays the key's own recent events, then streams new ones. Every frame
carries `seq`, numbered from 1 on each connection, so a client that reconnects
rebuilds its view from the replay and tells requests apart by `req_id`. The
stream ends when the key is revoked, rotated or paused, when that sign-in ends
or expires, and when Prompt Lens is switched off. A key may hold 4 open streams
and mint 30 tickets a minute; past either, the mint (or the stream) answers
**429** with `Retry-After`. Tickets never appear in the access log.

**The replay window.** The admin stream replays the shared ring (2000 events
and ~4,000,000 characters by default). A key holder replays a window of their
own key's events only, so what they get back depends on their own traffic,
never on how busy other keys are: at most 500 events and 400,000 characters,
64 request starts, dropped after 30 minutes without an event of that key. The
server keeps at most 128 such windows; past that, the least recently active
key's window goes first. That is the one place where other keys' activity
shows: only when more than 128 keys were active within a key's idle time.
Text streams one delta per token, so a long reasoning stream fills a window
fast, and a running request's `request_start` can leave the replay before you
connect. The replay therefore opens with the `request_start` (and `kvsig`) of
every request still running whose own start already left it, marked
`"restored": true` and numbered like any replayed frame. For a key holder these
come from the key's own window and are in the key holder's view (no client IP
or key label, no other-key figure, the parts as if no other key had cached
anything); they change no window size or eviction. A client merges a restored
start into what it already holds and never resets text already received.

**Deprecated: `GET /api/me/requests/{id}/capture`.** The key holder's old
capture route still answers, the same as before, and is marked deprecated in
the API schema. The UI no longer calls it (it reads the stream and the
section routes); it goes in the next release.

```bash
curl -s $W/api/me/requests/$REQUEST_ID/capture -H "Authorization: Bearer $KEY_HOLDER_TOKEN"
# {"id":"…","model":"qwen","live":false,"prompt":"…","reply":"…","reasoning":"",
#  "finish_reason":"stop","timing":{"queued_s":0.2,"ttft_s":1.3,"duration_s":6.1,…},
#  "note":"Captures are kept briefly in server memory, …"}
```

**What it cannot hide (accepted).**

- `phase`, `queued_s` and `ttft_s` are the server's real timings, so they rise
  when the server is busy with other keys' work. They show how loaded the
  server is, not who loaded it.
- The API's own `usage.cached_tokens` in a response is the engine's count. The
  engine's prefix cache is shared by every key that is not isolated, so a
  prompt another key sent recently can make it higher. That is a side channel
  of the shared cache itself, not of Prompt Lens; a key that needs strict
  separation gets its own isolated KV cache (KV-cache tiers, isolated keys).
- With more than 128 key windows active, the least recently active key's
  replay window goes (above).
- The whole-section store evicts per key first: each key has its own budget
  (a quarter of `VW_PROMPT_LENS_SECTION_CHARS`, 4,000,000 characters by default,
  and 64 requests), and a key under it keeps its sections whatever the others
  send. Past the store's total bounds the heaviest key's oldest request goes,
  so the key with the most text kept can lose its oldest section to the
  others' combined traffic. That tells it only that total traffic was high.

### Per-key permissions

An admin sets two permissions on a key, when creating it (`POST /api/tokens`)
or later (`PATCH /api/tokens/{id}`, the key's page, "Permissions"). Both come
back on every key (`GET /api/tokens`, `GET /api/tokens/{id}`):

| Field | Meaning | New key | Existing keys |
|---|---|---|---|
| `prompt_lens` | the key's holder may use Prompt Lens on their own traffic: the key area's pages, the Prompt Lens API and the Prompt Lens MCP tools. The platform switch (Settings, General, Privacy) must be on too | `false` unless sent as `true` | `true` |
| `engine_facts` | the key may read the engine facts of the models it may use (`GET /api/me/engine`, the `engine_facts` MCP tool) | `true` | `true` |

```bash
curl -s -X POST $W/api/tokens "${AUTH[@]}" -d '{"name":"ci-runner","prompt_lens":true,"engine_facts":true}'
curl -s -X PATCH $W/api/tokens/$T "${AUTH[@]}" -d '{"prompt_lens":false}'
```

A rotation gives the successor both flags. The server decides on every call:
without `prompt_lens` every Prompt Lens route of the key holder (the pages'
routes under `/api/me/prompt-lens/`, the stream ticket, kept captures,
`/api/lens/me/*`) answers the same **404** as when Prompt Lens is off, and the
MCP endpoint lists no Prompt Lens tool. A key holder sees both flags,
read-only, in `GET /api/me/key` and on the key area's Settings page; they
cannot change them (`PATCH /api/me/key` refuses the fields). Each change an
admin makes is one `admin_audit` row whose `path` ends in `key.prompt_lens:on`
or `key.engine_facts:off`.

### Engine facts

`GET /api/me/engine` returns static facts about the engine behind each model
the key may use, with the credential an app already uses for inference: an
inference key (`Authorization: Bearer vw_…`, or `X-LMWarden-Key`), accepted
exactly as `/v1/chat/completions` accepts it. A missing, malformed, unknown,
expired or revoked key is **401**, a paused key **403** `token paused`, and a
key whose `engine_facts` permission is off **403**. An admin credential is not
accepted here. It is meant for tools such as
`leanprompt profile detect --lmwarden`. It carries no traffic figure, no other
key and no host name.

```bash
curl -s $W/api/me/engine -H "Authorization: Bearer $LMWARDEN_API_KEY"
# {"v":1,"warden_version":"2026.11.26.1","models":[
#   {"model":"qwen3-4b","backend":"vllm","running":true,"prefix_caching":true,
#    "engine_block_tokens":1584,"engine_match_unit_tokens":16,"engine_drops_last_block":true,
#    "speculative":"mtp","max_model_len":32768,"replicas":4,"cache_aware_routing":true,
#    "unknown":[]}]}
```

Every model the key may use and that is configured is listed, whether or not it
has served a request and whether or not it is loaded. A key with no allowed
models gets `{"models": []}`. Fields:

- `model` is the served model name, the one a client sends in `model`; `backend`
  is `vllm` or `llamacpp`; `running` is true while the model is loaded.
- `prefix_caching` is whether the running engine has prefix caching on.
- `engine_block_tokens` is the real KV block size, a classic 16 included (the
  dashboard's small-page filter is not applied). Reuse is in whole blocks.
  `engine_match_unit_tokens` is `--prefix-match-unit` when set, else the block
  size. `engine_drops_last_block` is true when speculative decoding that reads
  the target model's hidden states (EAGLE, MTP and similar) makes the engine
  recompute one more unit on every hit.
- `speculative` is the method (`mtp`, `eagle`, …), or `null` when it is off.
- `max_model_len` is the context length; `replicas` the engine copies serving
  the model; `cache_aware_routing` is true when the warden keeps a conversation
  on the replica that holds its cache (more than one replica).
- **Unknown is `null`, never a guess.** A value the warden cannot determine now
  (the model is not loaded, the engine has not reported its block size yet, a
  backend that does not say) is `null` and its field is named in `unknown`.
  For a model that is not loaded, every engine-run field and an unset
  `max_model_len` are unknown; `replicas` and `cache_aware_routing` come from
  the configuration.

### Prices

Settings, Prices tab. Admin session or admin token, like every `/api/admin` route.
Prices are **per million tokens, in micro-units of the currency** (`3000000` =
3.00 USD per million). The price of a model is, in order: the admin's
**override**, else the public list price in the install's price table. The table
is seeded a few seconds after startup from the snapshot shipped with the release
(`app/lens/price_snapshot.json`, with its source and date), so an offline
install has prices from its first minute (Prompt Lens reads the snapshot itself until the table is seeded), then refreshed in the background from
public lists: OpenRouter's `GET https://openrouter.ai/api/v1/models` (primary)
and, only when that fails, LiteLLM's `model_prices_and_context_window.json` on
`raw.githubusercontent.com` (MIT). Only those two HTTPS hosts are contacted,
without credentials or redirects, with timeouts and a 16 MiB cap, and nothing
about the install is sent. A failed refresh keeps the last good table. The
refresh is off for `HF_HUB_OFFLINE=1` or `VW_PRICE_REFRESH=0`, and runs every
`refresh_interval_hours` (default 24, 0 = off). It never delays startup.

| Route | What it does |
|---|---|
| `GET /api/admin/prices` | `providers` (the router's upstream, every upstream seen in the last 90 days, every provider priced by hand; each with its `models` and `catalogue_models`), `local` (own models with their reference and `priced_as`), `refresh` (interval, `last_success`, `last_error`, `last_source`, the snapshot's `source` and `fetched_at`) |
| `GET /api/admin/prices/models?provider=&q=&limit=` | catalogue models of one provider whose name contains `q` |
| `PUT /api/admin/prices/override` | `{provider, model, currency, input_per_m_micros, output_per_m_micros?, cache_read_per_m_micros?, cache_write_per_m_micros?}`: the admin's price; returns the entry |
| `DELETE /api/admin/prices/override?provider=&model=` | removes it so the list price applies again; **404** when there was none |
| `POST /api/admin/prices/refresh` | fetch now; a failure answers **200** with `last_error` |
| `PUT /api/admin/prices/settings` | `{refresh_interval_hours}` (0 to 720) |
| `GET /api/admin/prices/audit?limit=` | newest changes: who, when, what, before and after |

An entry is `{model, price, source, fetched_at}` with `source` one of `override`
(`fetched_at` is its date), `openrouter`, `litellm` or `snapshot`; an override
also carries `override.by`, `list_price` and `list_source`. `price` is null for a
model with no known price. `provider` is the provider's API host as the request
history records it (`api.anthropic.com`); model names match by folding case,
dots and a trailing date, so `claude-sonnet-4.5` and
`claude-sonnet-4-5-20250929` are one model. Every override set or cleared, every
reference card change and every interval change is a `price_audit` row.

**Kept from v2026.11.26.1, now thin wrappers:** `GET /api/admin/lens/rate-cards`
(`cards` are the overrides; `defaults` is always `[]`, the shipped list prices
are the price table now; `reference` as before), `PUT /api/admin/lens/rate-cards`
(each card is an audited override dated `effective_from`) and
`PUT /api/admin/lens/reference-card/{model_id}` (audited).

### The Prompt Lens API

The same reads as the Prompt Lens page, for scripts and assistants, with the
credential a client already has. `/api/lens/me/*` takes an **inference key**
(`vw_…`, as `/v1` does) or the key holder's sign-in token, and answers for that
key only: the key is the credential's, a `key=` parameter is **422**, an admin
credential is **403**, and a key without the `prompt_lens` permission (or with
the platform switch off) gets the one **404** of the page's routes. An unknown,
evicted and another key's id all answer that same 404. Calls are limited per
credential, shared with the MCP endpoint: 60 a minute for a key holder (and 2
concurrent diffs), 300 for an admin (`VW_LENS_RATE_KEY_PER_MIN`,
`VW_LENS_RATE_ADMIN_PER_MIN`); past it **429** with `Retry-After`. The log keeps
the route, a hash of the credential, the status and the time, never a query or
a body. Answers are codes, counts, sizes, positions and keyed hashes: no
prompt, reply or tool text, and no rank, percentile or other key's number.

| Route | Answer |
|---|---|
| `GET /api/lens/me/requests?since=&before=&model=&limit=` | the key's finished requests, newest first (`next_before` pages on) |
| `GET /api/lens/me/requests/{id}` | cache figures, the first part that broke reuse (`break`, `cause`, `span_line`, `lost_tokens`) and its findings |
| `GET /api/lens/me/requests/{id}/diff?against=` | what changed against the parent turn (or `against`): first broken part, cause, position, token deltas |
| `GET /api/lens/me/requests/{id}/findings` | the findings that touch the request, with their parts |
| `GET /api/lens/me/sessions?since=&model=&limit=&cursor=` | scored sessions with scores |
| `GET /api/lens/me/sessions/{id}/scorecard` | scores, ranges, confidence, top 3, findings, turn strip |
| `GET /api/lens/me/sessions/{id}/cost` | the session's cost and per-turn bars |
| `GET /api/lens/me/sessions/{id}/findings?layer=&rule=` | the session's findings |
| `GET /api/lens/me/sessions/{id}/report?format=json\|html&tool_names=0\|1` | the downloadable report (below) |
| `GET /api/lens/me/trends`, `/versions`, `/compare` | as `/api/me/prompt-lens/…` |
| `GET /api/lens/me/score?since=&model=` | leanprompt's measured score |
| `GET\|PUT\|DELETE /api/lens/me/wontfix[/{rule}]` | won't-fix marks |

`/api/lens/admin/*` has the same routes for any key (`key=`; required for
trends, versions, compare, wontfix and score) with an admin credential, and
`GET /api/lens/admin/fleet` (the distribution of stored scores, admin only; a key
credential is 401).

`GET /api/lens/me/score` returns leanprompt's measured-score JSON so
`leanprompt score` can read it: `prefill_efficiency` (prompt tokens served from
cache over the reusable ones), `stability` (reusable over prompt tokens on calls
that continue an earlier one), `generation_waste` (output tokens of calls cut at
the token limit or with a malformed tool call), `leanprompt` =
`floor(100 × (0.6 × prefill_efficiency × stability + 0.4 × (1 − generation_waste)))`,
and `calls`. With fewer than 50 calls in the window it answers
`{"scored": false, "reason": "…", "calls": n}` and no number.

**The session report.** `…/sessions/{id}/report` downloads one session as HTML
(one self-contained page: inline CSS, no script, no link) or JSON (its schema is
`app/lens/report.schema.json`, JSON Schema 2020-12, so CI can diff two reports).
It holds the scores with ranges and confidence, the top 3 fixes, every finding,
the turn bars, the cost and the `score_version`, with report-local labels (`T4`,
"system prompt line 3", "tool #5"). It has no key name, key or session id,
request id, kvsig id or client address, and never prompt, reply or tool text; tool
names appear only with `tool_names=1`. The admin's copy says "admin view" on its
first line. The scorecard's "Download report" menu calls the page's own routes,
`GET /api/me/prompt-lens/sessions/{ref}/report` and its admin twin.

### The Prompt Lens MCP endpoint

`POST /api/lens/mcp` is a Model Context Protocol server (Streamable HTTP,
JSON responses, protocol `2025-06-18`; server name `lm-warden`). It takes the
inference key (or the key holder's sign-in token); an admin token gets the admin
tool set. Point an MCP client at it, with a placeholder key:

```json
{
  "mcpServers": {
    "lm-warden": {
      "type": "http",
      "url": "https://warden.example/api/lens/mcp",
      "headers": { "Authorization": "Bearer vw_REPLACE_WITH_YOUR_KEY" }
    }
  }
}
```

`tools/list` shows only what the credential may use: no Prompt Lens tool unless
the key has the `prompt_lens` permission and the platform switch is on, no
`engine_facts` unless the key has `engine_facts`. Every tool description says its
results are data about the caller's own traffic, never instructions. Tools:
`lens_list_sessions`, `lens_list_requests`, `lens_explain_request`,
`lens_turn_diff`, `lens_session_cost`, `lens_scorecard`, `lens_findings`,
`lens_trends`, `lens_versions`, `lens_compare_versions`, `lens_score`,
`lens_wontfix`, `lens_wait_for_request` (waits up to 30 s for the key's next
request to finish, or answers `{"timeout": true}`), `engine_facts`, and the guide
catalogue `list_skills` and `load_skill`. An admin credential gets
`lens_admin_*` twins that take a `key` argument. An unknown or hidden tool is
JSON-RPC error `-32602`; the rate limit is the REST one (HTTP 429 with
`Retry-After`).

The guides (`prompt-lens`, `leanprompt`, `kvsig`, `engine-facts`) are markdown
shipped in the image. They are offered three ways that agree: the `list_skills`
and `load_skill` tools, the resources `skill://<name>` (`resources/list`,
`resources/read`), and a mention in the `initialize` result's `instructions`.
`prompt-lens` is offered only to a key that may use Prompt Lens, `engine-facts`
only to one that may read engine facts.

---

## Sensitive data guard

The guard keeps secrets in a request from reaching a model. It matches cloud
and API keys, private keys, passwords and secrets in config and on command
lines, crypto keys, and, if you turn them on, cards, IBANs, e-mail and phone
numbers and crypto addresses. Base64 that decodes to text is checked too (a
kubeconfig's `client-key-data`). Your own patterns and fixed words match too.
A value found once is caught again later in the same session even without the
context its rule needed (in prose, or after the agent compacts the
conversation); that match uses the value's keyed fingerprint, never the value.

In **sanitize** mode each match is replaced with a placeholder such as
`⟦LMW-SECRET-3a9f1c2e7b⟧` before the model sees it. Placeholders the model
writes in its answer are put back to the real value before the client gets
it. A line from LM Warden at the end of the answer lists what was replaced,
masked. **block** mode answers the first request in a session that carries a
new value itself, instead of the model, and says what it found. The same
conversation sent again goes through sanitized. A value in the system prompt
or tool definitions is never blocked, only replaced.

The guard covers local requests and, with the router on, requests relayed to
Anthropic. It is off until you turn it on. The UI is **Settings → Sensitive
data** for the platform default and the **Sensitive data** card on a key's
page. These calls drive both. `$W`, `AUTH` and `T` are as in the sections
above; an admin token works the same way (see [Admin API](#admin-api)).

**Nothing stores a value.** Request history, the debug capture, Prompt Lens and
the content log see placeholders only. The action log keeps the rule, where
the value was, its length and a keyed fingerprint (`ph`). The same
fingerprint means the same value for that key, nothing more.

### The settings object

The platform default and a key's own settings have the same shape. Every
field is optional; the defaults are shown.

```json
{
  "mode": "sanitize",
  "packs": ["cloud-api-keys", "private-keys", "generic-credentials", "crypto-keys"],
  "rules": [],
  "overrides": {},
  "restore": "all",
  "trailer": "new_only",
  "fail": "closed"
}
```

| Field | Values |
|---|---|
| `mode` | `sanitize`, or `block` (block once per session, then sanitize) |
| `packs` | built-in rule packs, from `GET /api/guard/packs`: `cloud-api-keys`, `private-keys`, `generic-credentials`, `crypto-keys`, `payment-cards`, `iban`, `pii-contact`, `crypto-addresses` |
| `rules` | your own rules, at most 100 (below) |
| `overrides` | `{"<pack or rule id>": "sanitize" \| "block"}`, the action for one pack or rule |
| `restore` | `all` (text and tool-call arguments), `text_only`, `off` |
| `trailer` | the line at the end of the answer: `new_only` (when something new was replaced), `always`, `off` |
| `fail` | `closed` refuses a request the guard cannot check; `open` sends it unchecked |

A custom rule is a pattern or a list of fixed words:

```json
{"id": "corp-hostnames", "type": "regex", "regex": "corp-[a-z0-9]{8}\\.internal",
 "keywords": ["corp-"]}
{"id": "codenames", "type": "fixed", "values": ["Project Halcyon", "HALCYON-7"]}
```

- **`id`:** 1–48 lowercase letters, digits or dashes.
- **Pattern:** at most 512 characters, with no back-references or look-arounds. A repeated group may not hold another repeat or alternatives (`(a+)+`, `(a|aa)+`, `(a{1,30}){1,30}`): those can take exponential time, so the rule is refused when you save it. Write flat patterns, one repeat per character class.
- **`keywords`:** optional and lowercase, anywhere in the match. The pattern only runs near them, which keeps long requests fast.
- **Fixed words:** 1–1000 values, matched case-insensitively.
- **`overrides`** must name a pack, a built-in rule or one of your rules; a typo is refused.

### Platform default

```bash
# What is set now. null = the guard is off unless a key turns it on for itself.
curl -s $W/api/guard/settings "${AUTH[@]}"
# {"guard": null}

# Turn it on for every key. The answer is the stored, normalised settings.
curl -s -X PUT $W/api/guard/settings "${AUTH[@]}" \
     -d '{"guard": {"mode": "sanitize"}}'
# {"guard": {"mode":"sanitize","packs":["cloud-api-keys","private-keys",
#            "generic-credentials","crypto-keys"],"rules":[],"overrides":{},
#            "restore":"all","trailer":"new_only","fail":"closed"}}

# Turn it off again.
curl -s -X PUT $W/api/guard/settings "${AUTH[@]}" -d '{"guard": null}'
```

A setting the server cannot use is a **422** whose `detail` says why, for
example `unknown pack 'cloud'` or `nested repetition can take exponential time`.
Nothing is saved in that case.

### One key

The same `guard` value can be given when the key is created
(`POST /api/tokens` with `"guard": {...}` next to `name`); a setting the
server refuses is a **422** and no key is created.

Each key does one of three things: uses the platform default, switches the
guard off for itself, or has its own settings. The key's own settings start
from the platform default and replace only the fields they name.

```bash
# Use the platform default (what every key does until told otherwise).
curl -s -X PATCH $W/api/tokens/$T "${AUTH[@]}" -d '{"guard": null}'

# Off for this key, whatever the platform says.
curl -s -X PATCH $W/api/tokens/$T "${AUTH[@]}" -d '{"guard": {"off": true}}'

# Own settings: block on this key, everything else as the platform has it.
curl -s -X PATCH $W/api/tokens/$T "${AUTH[@]}" \
     -d '{"guard": {"override": {"mode": "block"}}}'

# What the key stores, and what is in effect for it.
curl -s $W/api/tokens/$T "${AUTH[@]}" | jq '{guard, guard_effective}'
# {"guard": {"override": {"mode": "block"}},
#  "guard_effective": {"mode": "block", "packs": [...], "rules": [], ...,
#                      "source": {"mode": "key", "packs": "platform",
#                                 "trailer": "default", ...}}}
```

`guard_effective` is every setting in force for the key, defaults included,
or `null` when the guard is off for it. `source` says for each field whether
it came from the key, the platform, or the default.

### Try a setting

`POST /api/guard/test` runs a setting over some text and shows what a model
would see. Nothing is stored. Leave out `guard` to test the platform default.

```bash
curl -s -X POST $W/api/guard/test "${AUTH[@]}" -d '{
  "text": "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
  "guard": {"mode": "sanitize", "packs": ["cloud-api-keys"]}}'
# {"ms": 0.4,
#  "preview": "export AWS_ACCESS_KEY_ID=⟦LMW-SECRET-7c41e09b2a⟧",
#  "findings": [{"rule_id": "aws-access-key", "label": "AWS access key",
#                "start": 25, "end": 45, "length": 20,
#                "masked": "AKIA… (20 chars)", "action": "sanitize"}]}
```

`GET /api/guard/packs` lists every pack with its rules (`id`, `label`, and the
public prefix shown when masking, such as `AKIA`) and which packs are on by
default.

### The action log

```bash
# Last 7 days for one key; every parameter is optional.
curl -s "$W/api/guard/events?token=$T&action=blocked&limit=200" "${AUTH[@]}"
# {"events": [{"ts": 1793300000.1, "token_id": "53e7011c…", "request_id": "…",
#              "sess_ref": null, "action": "blocked", "rule_id": "aws-secret-key",
#              "role": "tool", "msg_index": 14, "kind": "tool result",
#              "length": 40, "ph": "0d93aa51fe", "relayed": false}],
#  "totals": {"sanitized": 312, "blocked": 3, "restored": 0, "error": 0,
#             "distinct_values": 17}}
```

- **Parameters:**
  - `token`: a key id;
  - `action`: `sanitized`, `blocked`, `restored` or `error`, anything else is a **400**;
  - `rule`: a rule id;
  - `since`: epoch seconds, default 7 days ago;
  - `limit`: 1–2000, default 200.
- **`msg_index`:** counts from 1; `0` is the system prompt or the tool definitions.
- **Key holders:** someone signed in with their own key reads that key's log at `GET /api/me/guard/events`, with `since` and `limit`. They cannot change the key's guard.

### What a client sees

- **Sanitized:** the model's answer with values put back, then the trailer when
  one is due. The trailer names each value by rule and a mask: a key's public
  prefix (`AKIA…`), or only its length for passwords and other general
  findings. A relayed answer also carries the header
  `X-LMWarden-Guard: sanitized;n=2`.
- **Blocked:** a normal `200` answer in the request's own format: chat,
  completions, or an Anthropic message, streamed if the request asked for a
  stream. It comes from the model `lm-warden-guard`, with
  `X-LMWarden-Guard: blocked;n=1`, `X-LMW-Request-Id`, and a text naming each
  value, masked, and where it was. A blocked request still gets a
  request-history row, with phase `blocked` and no completion. It never takes
  an engine slot.
- **Not checked** (fail closed): **422** with `X-LMWarden-Guard: failed;reason=…`.
  The reason is `too_large` (over 8 MiB of text), `too_many_findings` (over 2,000),
  `unscannable` (a legacy `prompt` given as token ids), or `scanner_error`.
- **A value inside a signed Anthropic `thinking` block** blocks the request in
  any mode: it cannot be replaced without breaking the signature.
- **With the router on**, unparsed passthrough paths (the catch-all, for example
  message batches) answer a guarded key **403** `guard_no_passthrough`.
  `count_tokens` is sanitized and never blocked.

Block once per session is remembered in memory. After a restart, the first
request that still carries a value blocks once more.

---

## Claude Code and the Anthropic Messages API

`POST /v1/messages` speaks the Anthropic Messages API, so Claude Code and the
Anthropic SDKs can use a warden as their base URL. It is not a second proxy:
the request is translated into an OpenAI chat request and goes through the
same path as `/v1/chat/completions`. Admission, key priority, usage
accounting, live stats, Prompt Lens and the runaway detector all work the same
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

Thinking can be switched off for every chat request a token or a model serves
(on models whose chat template has a thinking switch):
`"thinking_policy": "off"` on `POST /api/tokens` or `PATCH /api/tokens/{id}`,
or on `PATCH /api/models/{id}/settings`. The default, `"client"`, leaves the
request in charge; any other value is a `422`. Thinking is off when either the
token or the model says `off`. On chat requests the proxy sets
`chat_template_kwargs.enable_thinking` to `false`, overriding the client. That
covers `/v1/messages` and `/v1/responses` (translated to chat first) and the
local leg of a router rule; `/v1/completions` is untouched. The change takes
effect on the next request, and a model setting can be changed while the model
is loaded, with no reload. A rotated token's successor inherits the policy.

Only models whose chat template has an `enable_thinking` switch are affected
(#248). `GET /api/models` shows `thinking_policy_supported`: `true`, `false`, or
`null` when the warden cannot tell, which is usual for GGUF models, whose
template lives inside the model file. It reads the model's Hugging Face chat template, so an
operator `--chat-template` in `extra_args` (or a list-form named template) is not reflected.
`GET /api/models/{id}` and `GET /api/models/{id}/settings` carry the same field. Every chat
request served under an effective `off` policy is marked `thinking_forced_off` on the live request
row and in request history, including requests that already had thinking off. Requests relayed to
Anthropic (the router relay on a relay-enabled key) are not subject to `thinking_policy`. The admin
stress runner talks to engines directly and is not subject to the policy.

```bash
curl -s -X PATCH $W/api/tokens/53e7011c… -H "Authorization: Bearer $ACCESS" \
     -H "X-CSRF-Token: $CSRF" -b jar -H 'Content-Type: application/json' \
     -d '{"thinking_policy":"off"}'
```

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

### Supported clients

The catalogue behind that endpoint has ten entries. Each says which protocol
the client speaks, which modes it has, and what it needs from the model. Every
client sends the warden key; the origin is the warden's root URL for the
Anthropic-protocol clients (they add `/v1/messages` themselves) and its `/v1`
for the OpenAI ones.

| Client | Modes | Endpoint | Requirements (from the catalogue) |
|---|---|---|---|
| Claude Code | Router: keep your Claude login; Local only (no Anthropic account) | `POST /v1/messages` | Router: Claude Code 2.1.227 or newer, routing on with an enabled rule, a key allowed to relay. At least 49,152 tokens of context (the first turn is about 38,000), 131,072 or more for real sessions. |
| Anthropic SDK (Python, TypeScript, curl) | Router: keep your Anthropic key; Local only | `POST /v1/messages` | Router: routing on with an enabled rule, a key allowed to relay. |
| OpenCode | Local models | `POST /v1/chat/completions` | Tool calling; 32,768 tokens of context or more recommended. |
| Aider | Local models | `POST /v1/chat/completions` | 32,768 tokens or more recommended; no tool calling needed (it edits through text formats). |
| Grok CLI | Grok Build (official); Community grok-cli | `POST /v1/chat/completions` | Tool calling; 32,768 tokens or more recommended. |
| OpenAI SDK (Python, TypeScript, curl) | Local models | `POST /v1/chat/completions` | None. |
| Continue | Local models | `POST /v1/chat/completions` | 32,768 tokens or more recommended; two config variants, chosen by whether the model supports tools. |
| Cline | Local models | `POST /v1/chat/completions` (streaming) | Tool calling; 32,768 tokens or more recommended. |
| Cursor | Chat with a local model | `POST /v1/chat/completions` | The warden's URL must be reachable from the internet. |
| Codex CLI | Local models | `POST /v1/responses` | Tool calling; 32,768 tokens or more recommended. |

The catalogue's own notes, per client:

- **Claude Code.** Router mode keeps your Claude login: the warden key travels
  in `ANTHROPIC_CUSTOM_HEADERS`, the Claude models your rules name run here,
  and every other model and `/v1` path goes to Anthropic. Never put the warden
  key in `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` in router mode: that
  would replace your login. Local only runs everything, background tasks
  included, on one served model, and nothing goes to Anthropic. Claude Code may
  warn that a local model name is not in its catalogue; that is informational,
  so set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the real window.
- **Anthropic SDK.** Router mode: `base_url` is the warden, your Anthropic key
  stays in `api_key`, the warden key goes in default headers. Local only: the
  warden key is the `api_key` and the model is a served name.
- **OpenCode.** Local models only, over Chat Completions through
  `@ai-sdk/openai-compatible` (`@ai-sdk/openai` would use the Responses API,
  which the warden translates for Codex CLI only). The model id must be a
  served name from `/v1/models`. For a committed file, use `{env:LMWARDEN_KEY}`
  instead of the key. Router mode through OpenCode's Anthropic provider is not
  covered yet.
- **Aider.** Local models only. The metadata file only stops Aider warning
  about an unknown model and sets its context size.
- **Grok CLI.** Local models only (`api_backend = chat_completions`); there is
  no pass-through to xAI and a bring-your-own-key model needs no xAI login.
  `session_summary` keeps Grok's session-title requests on your model; without
  it Grok asks the warden for `grok-4.6` and gets a 404. Grok adds its own
  sections to `config.toml` when it runs. The community `grok-cli` (npm
  `grok-dev`) is a different tool: it needs Bun, chat works, tool calls do not,
  because its xAI SDK rejects tool-call chunks that arrive in parts.
- **OpenAI SDK.** Local models only. There is no pass-through to OpenAI: a
  request for `gpt-*` gets 404. Chat Completions and Completions are served;
  `/v1/responses` is a stateless translation to Chat Completions.
- **Continue.** Local models only. The config follows Continue's docs and the
  request it sends was tested from the Connect page; the editor itself was not
  run. Router mode would need custom headers Continue does not document.
- **Cline.** Local models only. Cline has no config file: enter the values
  under API Provider > OpenAI Compatible. Its Anthropic provider has a custom
  base URL but no custom header, so router mode is not available.
- **Cursor.** Requests are issued from Cursor's servers, so a LAN or localhost
  warden will not work. Custom keys only work with chat models: Tab completion
  keeps using Cursor's own models, and Agent with Cursor's models rejects custom
  keys. Cursor's current docs no longer describe the Override OpenAI Base URL
  setting; it is known from Cursor staff forum replies.
- **Codex CLI.** See [Codex CLI and the OpenAI Responses
  API](#codex-cli-and-the-openai-responses-api) below.

Public pages for each client, with the same files and placeholders, are at
`/coding-agents/<id>` on the website (`claude-code`, `anthropic-sdk`,
`opencode`, `aider`, `grok-cli`, `openai-sdk`, `continue`, `cline`, `cursor`,
`codex-cli`).

---

## Codex CLI and the OpenAI Responses API

`POST /v1/responses` speaks the OpenAI Responses API, which is the only wire
format OpenAI Codex CLI uses. Like `/v1/messages` it is not a second proxy: the
request is translated into an OpenAI chat request for the served model and goes
through the same path as `/v1/chat/completions`, so admission, key priority,
usage accounting, live stats, the session column, replica affinity, the
prefix-cache estimate, Prompt Lens and the runaway detector all apply. The
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

`GET /api/stats/forest?range=48h[&since=<cursor>]` — trees of agent sessions (the data only: the UI has no view
of it at present).
Ranges: `1h`, `6h`, `24h`, `48h`, `7d` (anything else, `30d` included, is a 400: the forest never holds more
than 7 days). The range picks the trees with a turn finished in `[now − range, now]`; each tree is sent whole (every
turn of the last 7 days), never cut at the window edge, so `t0` is about `now − 7 d` for every range up to `7d`. Admins see every key; a key holder sees only their key.

`POST /api/forest/login` with `Authorization: Bearer <your API key>` returns a 12-hour token for the forest
routes only.

Response fields: `range`, `t0`, `now`, `truncated_before` (only when the server holds fewer rows than the window has:
past its row budget, a key's oldest rows before this absolute time are left out, so its oldest trees may miss that
wood), `models` (per variant: `prefill_tps`, `decode_tps` — 7-day p75 —
`n`, the number of requests sampled, and `max_model_len`), `trees`, `flowers`, `ids`, `cursor`, `full`.
The body is gzip-compressed when the request sends `Accept-Encoding: gzip`. The server rebuilds at most
every 3 s per (range, viewer); requests in between get the same data, but `now` is always the time of the
request itself (absolute unix seconds), so the data can be up to 3 s older than `now`. A client that animates
the data should run its own clock and use `now` only to estimate the clock skew.

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

## Sessions

Sessions group requests the way a harness works: one harness session (Claude Code's session id, OpenCode's,
Codex's), its conversations (the main agent, each subagent, the continuation after a compaction), and each
conversation's turns. Ids, sizes, counts and timings only: no prompt or reply text is stored or returned.

Every route takes the admin session, or a key holder's read token from `POST /api/forest/login` (send your API
key once as `Authorization: Bearer <key>`; the answer is a 12-hour token). A key holder sees only their own key's
sessions and gets 404 for anyone else's. A raw `vw_` key is not accepted on these routes. Any read failure answers
`503 {"detail": "sessions unavailable"}`; the rest of the dashboard keeps working.

**Refs.** A session is addressed as `s_<32 hex>` (a keyed lookup of key + session; for traffic that sends no
harness session id, plus 12 hex of the segment's first request, since such traffic is split at 30-minute idle
gaps), or as `r_<request id>`, which resolves to the session that request belongs to. Raw harness session ids never
appear in a URL or a response.

**`X-LMW-Request-Id`.** Every proxied response (`/v1/chat/completions`, `/v1/completions`, `/v1/messages`,
`/v1/responses`, and `/v1/messages` relayed in router mode) carries the id of its history row in this header.

### `GET /api/sessions`

`?range=24h&group=key&view=all&where=all&sort=newest&q=&model=&since=&until=&offset=&limit=50&per_group=3&tz_offset_min=0`
— the sessions overview: sessions that started in the window, filtered, counted per ready view, and grouped.

- `range`: `1h`, `6h`, `24h`, `7d` or `30d`. `limit`: 1–100. Any other value of an enum below is a 400.
- `model`: a served model name. `token`: a key id (admin only; a key holder is always scoped to their own key).
  `q`: a `session_id` prefix, or part of a key's name.
- `where`: `all`, `local`, `relayed` or `hybrid` (some turns local, some relayed).
- `since`, `until`: epoch seconds, a narrower span inside the range (the chart's selection); clamped to it.
- `view`: `all`, `running` (a request in the last 120 s), `biggest` (top 50 by prompt tokens), `dead_weight` (over
  20% of prompt tokens, measured sessions only), `cache_trouble` (under 60% from cache, where the cache is known),
  `long_running` (over 1 h) or `evicted` (at least one resume found its cache evicted).
- `group`: `key`, `harness`, `day` (in the browser's time zone: pass `tz_offset_min`, −840 to 840) or `none`.
  Each group carries its first `per_group` (1–100, default 3) sessions; `group_key` + `offset` pages through one
  group, `limit` at a time. With `group=none`, `offset` pages the flat list.
- `sort`: `newest`, `prompt_tokens`, `dead_weight`, `duration` or `evicted_lost`; unknown values sort last.
- Answer: `{total, views, histogram, pause_summary, groups, sessions, next_cursor, truncated_before,
  oldest_data_at}`.
  - `total`: sessions after the filters and the view. `views`: the count of each view after the filters, so a
    view's chip shows what it would list.
  - `histogram: {bin_s: 3600, start, bins: [{t, sessions, over_dw}]}`: sessions started per hour over the whole
    range (filters applied, `since`/`until` not), and how many of them carry over 10% dead weight.
  - `pause_summary`: the resumes after a pause across the listed sessions, as in a session's `pauses` below.
  - `groups[]` (null with `group=none`): `key`, `label`, `harnesses`, `where: {local, relayed, hybrid}` (counts),
    `sessions`, `duration_s` (summed), `prompt_tokens`, `cached_share`, `anatomy`, `dead_weight_share`,
    `saved_vs_vanilla_s`, `evicted_resumes`, `evicted_lost_s`, `items` (sessions) and `more` (how many are left).
  - `sessions`: the flat list with `group=none`; empty otherwise.
- Without `group`, `sort` and `offset`, the answer is the older paged list (newest activity first; pass
  `next_cursor` back as `cursor`), with the keys above added. `truncated_before` is set when the window held more
  rows than one scan reads (200,000): sessions older than it may be missing. The scan is cached for 5 s per key and
  range, so changing a filter or a view does not rescan.
- Each session: `session_id`, `token_id`, `token_name`, `models`, `started_at`, `last_at`, `duration_s`,
  `requests`, `conversations` (side calls not counted), `prefill_tokens`, `cached_tokens`, `cached_share` (local
  requests only), `relayed_requests`, `generated_tokens`, `compactions`, `active`, plus:
  - `harness`: `claude_code`, `opencode`, `codex`, `openai_sdk` or `other`, from how the harness sent its session id;
  - `where`, `prompt_tokens` (every request, relayed included), `anatomy` (below), `cache_known`, and
    `cache_share`: cached over prompt tokens of the requests whose cache is known (relayed: the upstream's), null
    when none is; this, not the anatomy, is the "% from cache" and what `cache_trouble` tests. `cached_share`
    (local only) is null too when no local request reported cached tokens;
  - `dead_weight_share`, `dead_weight_tokens`, `dead_weight_measured` (≈; null when no turn was measured);
  - `saved_vs_vanilla_s`, `recomputed_s`, `cache_served_s` (≈ seconds, below), `replicas`;
  - `evicted_resumes`, `evicted_lost_s` (≈). For traffic with no harness session id, a resume across a gap of
    more than 30 minutes belongs to no single listed session (each side is its own session), so it counts only
    in `pause_summary`.

### `GET /api/sessions/{session_id}`

One session, by `s_…` or `r_…`; the answer's `session_id` is always the canonical `s_…`. 404 `session not found`,
or 404 `not recorded yet` for an `r_` ref whose request is still running and whose session has no recorded turn
yet. At most the newest 5,000 turns are returned (`truncated: true` beyond that).

- `totals`: `started_at`, `last_at`, `wall_s`, `requests`, `conversations`, `side_calls`, `compactions`,
  `prefill_tokens`, `cached_tokens`, `cached_share`, `generated_tokens` (local requests), `prompt_tokens` (every
  request), `cache_share` (as in the list), `replicas` (the data-parallel size most of the local turns ran on; null for a model no longer
  configured), `where`, and `relayed: {requests, prefill_tokens, cached_tokens, upstream}` (the upstream's own
  prompt cache) or null. Three ≈ times, each null when the model has no measured cold rate, its cache is unknown,
  or the turns were relayed:
  - `cache_served_s`: the prefill time the cached tokens would have taken at the model's measured cold rate.
  - `saved_vs_vanilla_s`: prefill time LM Warden's placement spared against plain vLLM, which routes data-parallel
    turns by load. Plain vLLM lands a turn on the replica holding its prefix 1 time in N, so the figure is the
    expected value `cached × (N − 1) / N ÷ cold rate` per local turn. It is 0 on a model with one replica or with
    affinity turned off.
  - `recomputed_s`: prefill time spent on the part of each prompt the cache did not have.
- `harness`, as in the list.
- `anatomy: {cached, dead, recomputed, new, unknown}` (≈ tokens; summed over turns, and the parts add up to the
  prompt). Dead weight comes first. The rest is split, in proportion, into what came from cache, what the previous
  prompt already held but had to be prefilled again (`recomputed`), and what is new this turn. A turn whose cache is
  unknown (an engine that reports no cached tokens) counts as `unknown`, never as a miss. A relayed turn's cached
  part is the upstream's.
- `dead_weight` (≈): `total_tokens`, `share`, `per_turn` and `prompt_tokens` (the base of `share`), all over the
  measured turns; `turns`, `turns_measured` (a turn is measured when it recorded duplicates, boilerplate and
  reasoning; turns recorded before dead weight was tracked are left out, never counted as 0), and `kinds`
  (each over the turns that know it): `unused_tools` (tools offered
  but never called in this session), `dup` (a message sent again within one prompt), `boil` (harness boilerplate:
  `<system-reminder>` and `<environment_context>` blocks and date-only lines), `reason` (reasoning sent back), each
  `{tokens, share, per_turn}` or nulls when unknown.
- `engine_replay` (≈, or null with no local turn): `{scope: "session", turns, skipped_time, slots, replicas,
  disk_load_excluded: true, engines: [{engine, recompute_tokens, recompute_s, measured}]}` for `lm_warden` (what
  happened: `measured: true`), `vllm`, `llamacpp`, `mlx_lm` and `omlx`. It replays this session's local turns alone
  through each engine's cache rules. oMLX's disk load time is not included. `skipped_time` counts turns without a
  cold rate, whose time is left out.
- `pauses`: `{resumes, evicted, unassessed, lost_tokens, lost_s, buckets, disk_tier_hint}`.
  - A resume is a local turn starting at least 60 s after its conversation's previous turn ended.
  - It is evicted when the prompt it could reuse was at least 256 tokens and either the cache outcome was `lost`, or
    at least half of the reusable prompt had to be prefilled again.
  - A resume with no cache data is `unassessed`.
  - `buckets`: `1-5m`, `5-15m`, `15-60m`, `1-4h`, `4h+`, each `{label, min_s, max_s, resumes, evicted,
    lost_tokens, lost_s}`.
  - `disk_tier_hint: {keep_s, recovered_s}` (≈, or null): a disk tier keeping prefixes for `keep_s` would have
    recovered `recovered_s` of prefill, disk load time not included.
- `conversations[]`: `id`, `label` (`main`, `subagent #k`, `conversation N`, `main after compaction`, …), `kind`
  (`main`, `subagent`, `compaction`, `conversation`, `unlinked`), `parent: {conversation_id, request_id, link}`
  where `link` is `known` (the harness sent a parent session id), `inferred` (LM Warden's guess, below) or
  `compaction`; plus `first_at`, `last_at`, `turns`, token sums, `relayed`, `lineage_unknown` and `tools`.
- `side_calls[]`: a harness's one-shot calls (title, topic, …): `request_id`, `started_at`, `model`,
  `prompt_tokens`.
- `turns[]`: `request_id`, `conversation_id`, `turn`, `started_at`, `finished_at`, `edge`, `node`, `parent_node`,
  `parent_request_id`, `prompt_tokens`, `cached_tokens`, `generated_tokens`, `reusable_tokens`, `cache_outcome`,
  `cache_outcome_own`, `divergence_cause`, `divergence_cause_own`, `dp_rank`, `queued_s`, `ttft_s`, `prefill_s`,
  `gen_s`, `finish_reason`, `shape` (sizes in characters: `chars`, `sys`, `tools`, `tools_chars`, `msgs` by role,
  `last_user`, `brk`, the first body message the session had not seen, on a fork or a rewrite, and `dw`, dead
  weight in characters, when measured), `tiers`
  (`gpu`, `ram`, `nvme`, `remote`, `load_s`: reserved for the cache tiers, null for now), `node_id`, `relayed`,
  `upstream`, `model`, `lineage_unknown`, `side_call`, plus `dead_weight` (≈ tokens per kind, `total`, `share`;
  null when not measured), `anatomy`, `cache_known`, `pause_s` (since the conversation's previous turn ended),
  `evicted_on_resume`, `lost_tokens` and `lost_s` (≈).
- Each conversation also carries `dead_weight`, `anatomy` and `saved_vs_vanilla_s`. Its unused tools are counted
  against what that conversation called; the session's against what the session called.
- `tools` (session-wide; also per conversation): `offered`, `called`, `unused`, `unused_tokens_per_turn`,
  `unused_tokens_total`, `unused_share` (estimates: `estimated: true`), `truncated` (a harness offered more than
  128 tools), and `rows[]`: `id` (8-hex keyed hash of the tool name), `name`, `family`, `chars`, `tokens`,
  `called`, `offered_turns`. `name` is the plain tool name, kept per key: it is shown only to that key's holder
  and the admin, and a tool's description or schema is never stored.

### `GET /api/requests/{request_id}?wait_ms=0`

One turn by its `X-LMW-Request-Id`: the turn fields above plus `session_id`. History rows commit in batches about
a second after a request ends; `wait_ms` (0–5000) waits for the commit before answering 404.

### Test-agent recipe

1. `POST /api/forest/login` with `Authorization: Bearer <key>` → `token`.
2. Send requests as usual, with the harness's session header (e.g. `X-Claude-Code-Session-Id`).
3. Read `X-LMW-Request-Id` from each response.
4. `GET /api/requests/<id>?wait_ms=5000` with `Authorization: Bearer <token>` → `session_id`.
5. `GET /api/sessions/<session_id>` → the tree: edges, inferred links, cache per turn, replica, tools.

## Stats

The Stats page reads these. All take the admin session or an admin token.

`GET /api/stats/v2/speed?range=1h&model=<model id>` gives per-request latency
and speed for one model over the window: `points` (one per request: `ttft`
including the warden queue, `decode`, `prefill` from computed tokens with
`prefill_alone`, `duration`, the key and client IP) and `summary` per figure.
A summary follows the low-volume rules: `{"mode":"none"}` with no requests,
`"list"` with the values for 1 to 4, `"max"` with `p50` and `max` for 5 to 19,
`"p95"` with `p50`, `p95` (and `p99` from 100 requests) above that. Add
`group=key`, `group=ip` or `group=model` (with or without `model`) for the
window columns of the Requests table's Group by; key groups split by model.

`GET /api/stats/requests` (in flight) rows carry `ttft_s`, `ttft_total_s`
(warden queue included), `out_tokens` (every channel, thinking and tool calls
included) and `decode_tps_5s`, and the response adds `groups.key`, `groups.ip`
and `groups.model` with each group's in-flight count and summed live speed.

The live stream's per-model block (`GET /api/stats/live`) adds `now`
(`generation_tps_10s`, `prefill_computed_tps_10s`, `warden_queue`) and, on the
first frame of a connection, `history_5m`.

### Stats for a key holder

Someone signed in with their own key (`POST /api/forest/login`) reads their own
figures. Every route binds the key from the sign-in and takes no key id:

| Route | Returns |
|---|---|
| `GET /api/me/stats/speed?range&model` | The same tiles and points as `/v2/speed`, from their requests only, with `prompt` speed (prompt / TTFT) in place of computed prefill; `models` lists the models the key used |
| `GET /api/me/stats/tokens?range&model` | Prompt and completion tokens per minute, booked to the minutes they were produced |
| `GET /api/me/stats/cache?range&model` | Tokens reused from the key's own earlier prompts, own-lens outcomes, prefill saved at the configured rate |
| `GET /api/me/stats/requests` | Their requests in flight, with `expected_start_s` estimated from their own requests, and a `state` per model |

`speed`, `tokens` and `cache` take an optional `until` (ISO 8601; a time in the
future reads as now, an unreadable one is a **422**): the window then ends at
that time instead of now. `speed` also returns `last_request_at`, when the
key's newest request finished, whatever the window. The page uses both to
show an idle key's last activity instead of an empty window.

None of them carries anything fleet-wide: no other key's figures or names, no
client IP, no install, engine or host figure, no queue count, and no measured
cached token count (on a shared prefix cache it could reveal that someone else
sent the same text).

---

## Admin API

Everything above signs in with the admin password and carries a CSRF token. A
program that should not hold that password uses an **admin token**: a `vwa_…`
secret that calls the control API (`/api/*`) with the same power as signing
in — settings, model register/pull/load/unload, inference-token management,
statistics, Prompt Lens, stress runs and the cache. Issue one in
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
scheme on every route except the public ones — `/healthz`, `/readyz`, `/api/setup/*`,
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
does by sending the header instead of a ticket — Prompt Lens, model logs, live
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
