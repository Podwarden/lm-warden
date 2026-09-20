# Changelog

All notable changes to LLM Warden are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once the first
release ships.

## [Unreleased]

## [v2026.09.20.1] — 2026-09-20

### Fixed

- **A cancelled database open no longer leaves a thread behind that stops the
  process from exiting.** Every SQLite connection is driven by a worker thread
  of aiosqlite's own, created non-daemon and parked on a queue until the
  connection is closed; CPython joins such threads during interpreter shutdown,
  so one that is never closed hangs the process *after* all of its work is
  finished, with nothing printed. A cancellation landing inside the connect —
  which is what the watchdog's background passes get when shutdown tears them
  down — did exactly that: the connection object was never handed back, so
  nothing could close it, and aiosqlite does not stop the thread on that path
  either. `open_db` now runs the connect as a shielded task, waits for it to
  finish before closing what it produced, and re-raises the cancellation
  afterwards, so a connection is no longer abandoned on any cancellation a
  caller can normally deliver. It is not an absolute "never": a caller that
  cancels the same open repeatedly, or a connect that has not returned 10 s
  after being cancelled, is abandoned on purpose — the alternative is an
  uncancellable task, which hangs shutdown exactly the way this bug did. That
  abandons one *daemon* thread, which cannot block interpreter exit, and it is
  logged at `error`. The close is deliberately left unshielded: aiosqlite
  queues it before awaiting and stops the worker from a `finally`, so a
  cancelled close still closes the connection, and shielding it only added an
  uncancellable wait — up to `busy_timeout` — to every client disconnect and
  every shutdown. The wait is bounded all the same: if the connection's worker
  thread is gone, nothing will ever acknowledge the close, and a `finally` that
  parks forever is a process that will not exit — the same failure this entry
  is about. Giving up on the acknowledgement after 10 s costs nothing, because
  the close is already queued on that thread. A close that fails no longer
  masks the error it was unwinding from. The test suite was hanging after its own summary line roughly
  one run in three because of this (#260), and the CI integration job wedged the
  same way (#43); the workaround that daemonised aiosqlite's threads for
  `tests/integration` only is gone with it. A hang like it is now diagnosable
  rather than silent: `faulthandler_timeout` makes a single hung test dump every
  thread's stack (it does not end the run — pytest arms that timer without
  `exit=True`), and the end of a test session names any lingering non-daemon
  thread and arms a faulthandler dump that prints every thread's stack and then
  exits non-zero instead of burning a CI slot until the job timeout. That timer
  is armed only in the process CI waits on: an xdist worker finishes its session
  and then idles until the controller releases it, which on a loaded machine is
  minutes, and arming it there killed healthy workers.
- `PRAGMA busy_timeout` is now set before `PRAGMA journal_mode = WAL` rather
  than after it. Switching to WAL takes an exclusive lock, so the statement most
  likely to contend with a background writer was the one running with sqlite3's
  5 s default instead of the 30 s the line below it intended.
- **`PATCH /api/models/{id}/settings` now validates `extra_env` the same way
  `POST /api/models` does, and answers `422` instead of `200` (#261).** The
  PATCH took a raw `dict[str, Any]` body and never ran `ModelCreate`'s
  allowlist check, so a signed-in session — or an admin token — could PATCH a
  hard-locked key such as `LD_PRELOAD` or `HF_TOKEN` and get it persisted.
  This was not code execution: `filter_extra_env` fails closed at launch when
  it sees a hard-locked key, so the practical effect was a persistent
  per-model denial of service (the row could no longer load until an
  operator edited it back out by hand), reachable by session or token alike.
  The effective backend used is the one being set in the same PATCH when
  `backend` is in the body, else the model's current backend.
- **The engine-template prefill was a third way to persist the same
  `extra_env` (#261).** `POST /api/models/templates` validated nothing, and
  registering with a `template_id` copies the template's `extra_env` verbatim
  when the body does not set one — so the payload `POST /api/models` refuses
  was still storable in two authenticated calls, with the same
  fails-closed-at-load denial of service. `trust_remote_code` needed this exact
  treatment in #256, and `extra_env` now gets it too: every one of the four
  doors — the register body key, the **merged** value at register, a template
  create, and the settings PATCH — goes through one allowlist check
  (`app.models.schemas`), each before any write. The merged check is the one
  that also covers a template stored before this fix.

  A template is held to the **union** of every backend's allowlist, not to one
  backend's: which backend it serves is decided by the register body, so there
  is no per-backend answer to give when it is saved. Hard-locked keys
  (`LD_PRELOAD`, `HF_TOKEN`, `PATH`, …) are global and refused there as
  everywhere else — that is the half this door exists to stop — and the
  per-backend rule is still enforced on the merged value at register. So a
  llama.cpp-only `GGML_*` template saves as it always did, and is refused only
  if it is then registered against vLLM.

  One behaviour change to know about: registering a model whose merged
  `extra_env` is not on its **own** backend's allowlist is now a `422` instead
  of a row whose env is silently dropped at launch — which is reachable with
  the builtin `gpt-oss-20b` template and `backend: llamacpp`. An explicit
  `"extra_env": {}` in the body now overrides a template's, so there is a way
  to say "none of it" (previously the merge was truthiness-based and an empty
  dict fell through to the template's).

  There is still no sweep for rows poisoned by the old PATCH, so an existing
  install may hold one; it surfaces as a model that refuses to load until its
  `extra_env` is edited. *(#266 below sweeps the two unusable shapes; a
  well-formed key that is merely off the allowlist is left alone, and what that
  means for you is spelled out there.)*
- **Boot now cleans up the two unusable shapes a pre-#261 settings PATCH could
  write, instead of leaving such a row to refuse to load with an error that
  points at the environment rather than at the row's history (#266).** Both are
  unusable no matter what an operator does next, so the sweep fixes the row
  rather than only reporting it — logging exactly what it changed, model by
  model, so the log alone tells you which model was affected and what happened
  to it:

  - `extra_env` carrying a hard-locked key (`LD_PRELOAD`, `PYTHONPATH`,
    `HF_TOKEN`, `PATH`, …) — the key is removed, every other key on the row
    is kept, and the log names both the model and the key(s) removed.
  - Either `extra_env` or `extra_args` in a shape registering a model would
    never have accepted at all (the old PATCH wrote whatever JSON a client
    sent, with no shape check) — the column is reset to its empty default
    (`{}` / `[]`) and the log says what was there before.

  If you see one of these lines in the startup log after upgrading, no
  action is needed — the row is already fixed. If the model still won't
  load, the cause is unrelated to this.

  **What the sweep deliberately does not touch, and what to do about it.**
  A third shape is possible and is left exactly as it is: an `extra_env` key
  that is a well-formed string→string entry but simply not on the backend's
  allowlist (`FOO_BAR`, or a `GGML_*` key on a vLLM row). Such a row loads and
  serves normally — `filter_extra_env` drops unknown keys at launch — so
  removing the key would be this release silently editing a working model's
  configuration, which is not a thing a boot sweep should do. What changed for
  it is on the *write* side (#262): because every write now validates the whole
  merged row, **any settings PATCH on such a row answers `422` naming the key**,
  including one that changes something else entirely. The same is true of
  pinning an engine (`try-stack`) and of applying a stress recommendation.

  To find such a row: `GET /api/models/{id}/settings` and read `extra_env`. To
  fix it: PATCH the corrected `extra_env` — in the same request as whatever you
  were trying to change, which is accepted — or send `"extra_env": {}` to clear
  it. Reading the model, loading it and serving from it are unaffected
  throughout; only writes are refused.

  This also closes a worse, silent
  failure the same poisoning could cause: a malformed `extra_env` or
  `extra_args` broke `GET /api/models` outright (a 500, hiding every model
  in the list, not just the affected one), which the old fail-closed-at-load
  behaviour for `extra_env` never did on its own.

### Security

- **The docker engine driver no longer mounts the warden's database into engine
  containers (#265).** Every sibling engine container was started with the
  `vllm-warden-data` volume bind-mounted at `/data`, mode `rw`. That volume holds
  `vllm-warden.db` — api tokens, the admin-token audit trail, models, sessions —
  and the auto-minted `jwt_secret`. So the engine, which runs a model's code
  under an image an operator or an admin token names (`engine_image`) with argv
  an admin token can extend (`extra_args`), could read the control plane's own
  database and secret, mint a session, or edit the audit trail. It also meant the
  docker driver was no more separated from warden state than the
  local-subprocess driver it is supposed to improve on. The engine container is
  now given the model-cache volume and nothing else.

  **Operators running `VW_ENGINE_DRIVER=docker` need do nothing, with one
  exception.** No new volume is introduced and nothing is renamed, so an existing
  install picks this up on the next warden restart; already-running engine
  containers keep the old mount until they are next reloaded. The exception is a
  deployment that moved `VW_HF_CACHE_DIR` to a path *under* `/data` — not the
  default, and not what any shipped compose file or Hub template does, all of
  which mount the cache volume at `/root/.cache/huggingface`. Such a deployment
  was reaching its model cache **through** the `/data` mount, and without it the
  engine would re-download every model into the container's writable layer (the
  2026-06-15 ENOSPC failure in a new costume). Move the cache back to the default
  path before restarting; the warden now logs a warning at startup whenever
  `VW_HF_CACHE_DIR` is not the path the driver mounts the cache volume at, since
  the mount point in the engine container is fixed.

  The engine needed nothing on that volume. Its env names no path under `/data`
  (`HF_HUB_CACHE` points at the model cache — the 2026-06-15 fix), its argv names
  files only inside the model cache, and the per-model log file under
  `<data_dir>/logs` is written by the *warden*, which streams the container's
  stdout/stderr from outside and writes it — exactly as the watchdog assembles
  crash bundles from outside. Read-only was considered and rejected: the threat
  is reading the token table and the signing secret, which a read-only mount
  serves perfectly well. Unit tests now pin the whole container spec's volume
  map, so a future mount has to be argued for rather than slipped in.
  `documents/ARCHITECTURE.md`, `docs/backends.md` and `documents/API.md` now
  state what each driver does and does not separate — including what the docker
  driver still shares with the control plane (the network the proxy reaches the
  engine on, the host IPC namespace, the pinned GPUs) rather than implying a
  sandbox.
- **The proxy's tokenizer no longer executes model-repository code.** A model
  row's `trust_remote_code` column was passed to
  `AutoTokenizer.from_pretrained` on every `/v1` request, for prompt and
  completion token accounting — which means it ran that Hugging Face
  repository's Python **inside the warden's own process**: the process holding
  the SQLite database and `VW_JWT_SECRET`, on data-plane traffic, identically
  under every engine driver. The column was never emitted to an engine (no
  `--trust-remote-code` is built for vLLM or llama.cpp), so the tokenizer was
  its only consumer and this was the only in-process code-execution path the
  product had. The cache now loads with `trust_remote_code=False` always, and
  the flag is no longer accepted as an argument at all (#264).

  The cost is accounting precision, in one narrow case: a repo whose tokenizer
  can only be built by running its own code now falls back to the existing
  character estimate, which is logged once and reported through the
  `estimating()` surface, exactly as a GGUF-only repo already does. No shipped
  template is affected — the only builtin carrying the flag,
  `openai/gpt-oss-20b`, ships a plain fast tokenizer that loads without it.
  Nothing enforces limits from these counts (per-token rate limits were removed;
  chat budgets and stress runs use the engine's own `usage`), and the warden's
  counts were already approximate by design: they tokenize the joined message
  text without the chat template.

### Fixed

- **The "this model requires trust_remote_code" engine diagnosis now gives
  advice that works.** It told the operator to enable `trust_remote_code` for
  the model — but that column is never passed to the engine, so following it
  led back to the identical failure. It now says to add `--trust-remote-code`
  to the model's `extra_args`, which is the argv the engine actually receives
  (#264). Making the column emit the flag is a separate decision and is not
  part of this change.

### Changed

- **`documents/API.md` said `trust_remote_code` ran "inside the warden's engine
  process … under the default local-subprocess driver".** It did not: it ran in
  the warden's own process, via the proxy tokenizer, on `/v1` traffic, under
  every driver — and as of the change above it runs nowhere. The Admin API
  section now says so, including the `POST /api/models/{id}/load` refusal row,
  which described load-time engine execution that never happened. The four
  session-only refusals around the column are **unchanged in behaviour**: they
  fence a stored grant an admin token must not be able to write. The refusal
  messages and the in-code comments that inherited the same wrong mechanism
  were corrected with them (#264).

- **Every API path that writes a model now applies register's rules, because
  they all go through one writer.** The model row is what the runtime trusts —
  it becomes the engine's argv and env, its container image, and the thing the
  watchdog restarts with nobody at the keyboard — but until now every rule
  about what may be *on* a row was attached to a *request*: the full set on
  `POST /api/models`, a thinner copy on `POST /api/models/templates`, a
  hand-written subset on the settings PATCH, and none at all on
  `POST /api/models/{id}/try-stack` and `POST /api/models/{id}/stress/apply`.
  So the same hole kept reappearing one door at a time. There is now a
  full-row validator (`ModelSpec`) and a single writer that consults the
  per-column policy table, checks the principal, enforces the unload-first
  guard, merges, validates the *merged row*, and runs the checks a single row
  cannot state — and refuses before writing anything, ever partially. The
  state machine keeps its own named methods for `status`, `prior_status`,
  `pulled_*` and `last_error`; the watchdog, boot reconciliation and the pull
  task cannot reach the operator writer at all. Two tripwires keep it honest:
  the policy table already fails at startup if a new column is unclassified,
  and the test suite now fails if an operator-writable column has no rule
  (#262, #265).

- **`POST /api/models/{id}/try-stack` and `POST /api/models/{id}/stress/apply`
  no longer write the model row by hand.** Try-stack wrote the engine channel,
  version and image with a raw `UPDATE`, including onto a model that was
  currently serving — it now answers `409` and asks for an unload, which is
  what pinning an image already required to take effect. Stress-apply wrote
  `max_model_len` the same way and then restarted the engine off the result;
  it also now records the unload it performs, so the row no longer claims
  `loaded` with no process behind it for the length of the restart (#265).

  **`POST /api/models/{id}/stress/apply` can now refuse where it used to
  succeed.** Because the write validates the whole merged row, "Apply and
  reload" answers `422` when any column on the row breaks a rule — including a
  column the request does not touch, such as an `extra_env` key left by a
  pre-#261 PATCH (see #266 above). The row is checked **before** the engine is
  touched, so in the ordinary case the model keeps serving on its current
  context, nothing is written, and the message names the column to fix. If the
  row changes in the moment between that check and the write, the refusal
  arrives after the unload: the engine is restarted before the `422` is
  returned, and if that restart itself fails the row is left `failed` for the
  watchdog to pick up. Fix the named column with a settings PATCH and apply
  again.

- **`POST /api/models` accepts the `supports_tools` / `supports_vision` /
  `supports_reasoning` capability flags.** They are model columns an operator
  may set, and until now the only way to set one was a second request to the
  settings PATCH. Omitted is still the default and still means "auto-detect".

### Fixed

- **The per-model settings PATCH silently accepted values the register route
  refuses.** `PATCH /api/models/{id}/settings` skipped essentially all of
  register's validation: slug patterns and length bounds on
  `served_model_name`, `hf_repo`, `filename` and `mmproj_filename`; the
  numeric bounds on `max_model_len`, `gpu_memory_utilization`,
  `max_batch_size`, `n_gpu_layers` and `tensor_parallel_size`; the
  `tp`/`pp`/`auto` choice; `gpu_indices` uniqueness; and the rule tying
  `tensor_parallel_size` to `gpu_indices`. It also wrote `extra_args`,
  `extra_env` and `gpu_indices` as whatever JSON type arrived, so a *string*
  `extra_args` round-tripped as a string and was then appended to the engine's
  argv one character at a time. All of these are now `422`, and a refused
  PATCH writes none of its keys. Two consequences worth knowing before you
  upgrade: changing `gpu_indices` now requires sending the matching
  `tensor_parallel_size` in the same request (the settings page does this for
  you), and a `422` carries the standard list of field errors rather than a
  bare string (#262).

- **A `served_model_name` collision on the settings PATCH was a 500.** The
  column is `UNIQUE` in SQL and the PATCH never looked, so SQLite raised an
  `IntegrityError` that reached the client as an unexplained server error.
  It is now the same `409` register has always answered with, and the settings
  page shows it as a save error on the field rather than as the unload-first
  banner (#262).

- **A model row written before a rule existed can still be read.** Validation
  applies to writes only: `GET /api/models`, the model page and the settings
  GET decode such a row unchanged. A *change* to one is refused until the
  offending column is fixed, which the same request can do — patching a valid
  `extra_env` alongside the edit you wanted is accepted.

## [v2026.09.19.4] — 2026-09-19

### Changed

- **Admin tokens can no longer set the `trust_remote_code` column, on any
  route.** A final security review of the v2026.09.19.3 admin
  tokens found that a leaked one could register a model with
  `trust_remote_code: true` and load it, running that Hugging Face
  repository's Python inside the warden container — from there it could
  read `VW_JWT_SECRET`, mint a session, or edit `api_tokens` /
  `admin_audit` directly, making the session-only fence around token
  management an API-level guardrail rather than real containment (#256).
  Four paths now answer `403 session_only` for an admin token, which is every
  way the API has to put such a model in front of an engine, and each is
  refused before any write:
  - `POST /api/models`, when the model's **effective** `trust_remote_code` is
    true — the body's value, or the template's when the body does not say. The
    merged value is what matters, because the builtin `gpt-oss-20b` template
    carries the flag, so a `template_id` alone would otherwise have been
    enough. An explicit `trust_remote_code: false` still wins over a template
    that sets it.
  - `POST /api/models/templates`, when the template being saved carries the
    flag.
  - `PATCH /api/models/{id}/settings`, when it would turn the flag **on**;
    turning it off, and every other setting on that route, stays open to an
    admin token.
  - `POST /api/models/{id}/load`, when the row already has it — checked at
    load time, since a row can predate this change — while unload, list and
    every other model stay open to an admin token.

  `prior_status` is no longer patchable by anyone. It is the column the
  watchdog's restart sweep keys on, and it was the other half of the same
  hole: an admin token that could set it, and the flag, on a `failed` row
  could have the watchdog start the engine for it without ever calling
  `/load`. It belongs to the crash path and the restart sweep, like `status`.

  Every refusal is session-only: a signed-in session is unaffected by all of
  it, and each refused attempt lands in the token's audit trail.

  This closes the column, not the capability: `extra_args` still reaches the
  engine's argv verbatim and last, so an admin token can pass
  `--trust-remote-code` that way, and `engine_image` names the image a
  container driver runs. An admin token remains equivalent to code execution
  in the warden container — see `documents/API.md`, "Admin API".
- Admin-token audit pruning now caps each token at 20 000 rows as well as the
  existing 200 000 rows overall. The prune runs the 90-day window first, then
  the per-token cap, then the global cap. Previously the cap was global only,
  so a single token making requests once a second could fill the table in
  about 2.3 days and evict every other token's history along with its own; the
  per-token pass trims that flood to its own 20 000 rows before the global cap
  is applied, so a noisy or leaked token only evicts its own trail. (#257)
- **The admin-token audit trail is written in batches, and a clean shutdown no
  longer loses rows.** Each request used to open its own database connection
  and commit its audit row once the response had finished, so a script polling
  the API paid a commit per call, and anything still in flight when the warden
  stopped was lost. The middleware now hands the finished row to an in-process
  queue and one background writer commits a batch — at most 200 rows, or one
  second behind, whichever comes first — with a final flush at shutdown. What
  is recorded, and when a request counts as audited, is unchanged; a row is
  durable up to a second after the response rather than immediately, and each
  row still carries the time its request arrived, so the trail reads in order.
  The queue is bounded at 10 000 rows: if a stalled data volume ever filled it,
  the extra rows are dropped and counted — as is a batch the database refuses,
  so both kinds of gap show up in one number. Drops are logged from the enqueue
  side too, at most once a minute, not only at each flush, so a gap in a trail
  shows up in the log instead of passing silently even if the background writer
  is what stopped. The writer survives an unexpected error and keeps flushing
  rather than ending its task; a clean shutdown writes what is still queued,
  including a batch it had already picked up.
- **Revoking a token now also ends an in-flight chat-playground generation.**
  Six streams already ended when the credential that opened them did — god
  mode, model logs, live stats, header metrics, model pull progress and the
  chat2 turn streams — but the playground's own completion stream did not, so a
  revoked admin token (or a signed-out session) kept generating to the end. It
  is now registered and re-checked like the other six: a revoke, a refresh with
  `grace_hours: 0`, an expiry or a logout stops it where it is. The streaming
  payload the playground reads is unchanged. The chat2 turn route's replay of an
  already-recorded turn is registered too, so "every stream the warden opens
  under a credential" needs no footnote — there is nothing to cancel in a single
  pre-computed frame, but the claim is easier to trust than to qualify.

## [v2026.09.19.3] — 2026-09-19

### Added

- **Admin tokens: the whole control API for programs.** Settings → Admin
  tokens issues `vwa_…` secrets that call `/api/*` with the same power as
  signing in — settings, models, inference tokens, statistics, god mode,
  stress runs and the cache — with no password, cookie jar or CSRF header.
  Expiry is 30, 90 (default) or 365 days, or Never, which has to be chosen
  explicitly and warns. The secret is shown once. Refresh issues a new secret
  and keeps the old one working for 0, 1 or 24 hours (revoking the successor
  does not end the predecessor's grace window — revoke both); revoke is
  immediate and closes the token's open streams (god mode, model logs, live
  stats, header metrics, pull progress and chat2 turns), which are otherwise
  re-checked every 60 seconds. Every request an admin token makes to a route
  is recorded — route template, status, duration, the proxy-reported
  `client_ip` and the socket `peer_ip` — for 90 days (at most 200 000 rows
  across all tokens), including refused attempts by a known token; requests
  refused before routing, and oversize or malformed bodies, are not. The
  Activity view pages through it. Issuing, refreshing, revoking and reading
  the audit need a signed-in session, and so does changing the operator's own
  credentials, a session's lifetimes, or `public_url` (the host client-facing
  URLs are built from) through `PATCH /api/settings/runtime`:
  a leaked token cannot use this API to copy, extend or hide itself, or lock
  the operator out of the UI where it gets revoked, or steer the host
  client-facing URLs point at. The curl examples next to a new admin secret
  use the page's own address. That is an
  API-level guardrail, not containment — an admin token is root-equivalent
  (it can load a model with `trust_remote_code`), so treat a leaked one like
  a leaked admin password: revoke it, then rotate `VW_JWT_SECRET`. An admin
  token is never accepted on `/v1`, and the inference-token API never shows
  admin tokens. See `documents/API.md`, "Admin API".
- **`GET /api/openapi.json`** serves the API description to a session or an
  admin token, with a `bearerAuth` scheme and `security: []` on the public
  routes and `/v1`. `GET /api/models` and `GET /api/models/{id}` are now typed
  in it.

### Changed

- **FastAPI's `/docs`, `/redoc` and `/openapi.json` are gone.** The API
  description is no longer public; use `GET /api/openapi.json` with
  credentials.
- The SSE streams gated on a single-use ticket (`/api/models/{id}/logs/stream`,
  `/api/admin/godmode/stream`, `/api/stats/live`,
  `/api/header/metrics/stream`) accept `Authorization: Bearer vwa_…` in place
  of `?ticket=`. `ticket` is therefore optional in the spec, and a request
  with neither is `401 missing ticket` instead of `422`. Model pull progress
  and the chat2 turn streams already took a bearer header for a session and
  now also accept an admin token.
- Stress runs accept an admin token; the `operator_only` refusal now names
  inference tokens (`vw_…`) only.

## [v2026.09.19.2] — 2026-09-19

### Added

- **A live test that proves priority queueing on a running install.**
  `tests/live/test_priority_queueing.py` (marker `live`, deselected by
  default) creates ten keys across priorities 0–9, fills every engine slot,
  then sends a burst and checks that requests are admitted strictly by
  priority and first-in-first-out within a priority, over several rounds.
  While it runs it pauses every other active key so real traffic can't
  interfere, and it restores them afterwards, including after Ctrl-C,
  SIGTERM or a crash (`python -m tests.live.restore --state …`). All settings
  come from `VW_LIVE_*` environment variables; nothing about the target is in
  the repository. See `tests/live/README.md`.

### Fixed

- **Lost GPU telemetry is reported instead of shown as an idle 0.** When
  `nvidia-smi` stopped answering inside the api container, the header showed
  VRAM 0 % next to a model still marked loaded. `/stats` kept showing the last
  sample from before the failure. The log repeated `nvidia-smi exit 255:`
  about 1,300 times an hour with nothing after the colon: the reason,
  `Failed to initialize NVML: Unknown Error`, is printed on stdout, and only
  stderr was read. That state means the running model keeps serving but the
  next load fails. Now:
  - the error names nvidia-smi's own message, from stdout or stderr;
  - a hung nvidia-smi is killed rather than left behind;
  - the api log records the loss once, and the recovery once;
  - the header gauges read `--`, and their tooltip opens with
    `GPU telemetry unavailable: <reason>` while still listing the models;
  - `/stats` shows a banner and `—` tiles instead of stale numbers;
  - `/healthz` stays `200` but reports `gpu: {state, error, since,
    checked_at}` from memory, without running nvidia-smi.
  Optional probes (NVLink, thermal limits, topology, throttle reasons) fail by
  design on some drivers, so they no longer log warnings or change the state.
  The header stream's VRAM and utilization fields, and the overview's current
  GPU numbers, are `null` rather than `0` when nothing was measured. (#255)

- **The installer requests the GPUs through CDI, so a systemd reload can no
  longer take them away.** The legacy `driver: nvidia` request lets the NVIDIA
  hook grant the GPUs outside the cgroup manager. On hosts using Docker's
  systemd cgroup driver, the next `systemctl daemon-reload` (unattended upgrades
  run one) revoked them from the running container. `nvidia-smi` then failed
  with `Failed to initialize NVML: Unknown Error`, the VRAM gauges read 0, and
  new model loads would fail until the container restarted. `install.sh` now
  writes `driver: cdi` with `nvidia.com/gpu=all` or one device per index
  whenever Docker has CDI enabled and the toolkit's spec names the GPUs. In that
  case it no longer needs the `nvidia` runtime, and it sets
  `NVIDIA_VISIBLE_DEVICES=void`. Otherwise it keeps the hook and, on a
  systemd-cgroup host, warns. On such hosts it also creates the NVIDIA
  `/dev/char` symlinks systemd needs and installs NVIDIA's udev rule that
  recreates them at boot. None of this restarts anything. `GPU_REQUEST` and
  `GPU_HOST_PREP` override the choices; see `documents/INSTALL.md`. (#254)

- **Pages keep refreshing after you switch back to their tab.** `/stats`
  (including its system panel), `/cache`, `/models`, a model's page and the
  settings model tab stopped polling for good the first time their tab was
  hidden, until a reload. SWR ends its polling loop on an interval of 0, which
  they returned while hidden; they now use a fixed interval, and SWR itself
  skips the refresh while the tab is hidden. (#251)

## [v2026.09.19.1] — 2026-09-19

### Added

- **Breadcrumb trail and smart back button on every page.** A one-row strip
  under the nav shows where you are (`Home › Models › Qwen3-8B › Settings`,
  with detail pages named after the model or token) and a `← <previous page>`
  button that returns to the page you came from at the scroll position you
  left it, or `← Home` on a fresh load (Home is the Models page; the button
  is hidden where it would lead back to the page you are on); it replaces the
  token and model pages' own "API tokens /" and "← Back to" links.
- **Migration 0035: indexes for sorting the token list.** Composite
  `(column, id)` indexes on `api_tokens` for `created_at`, `last_used_at`,
  `expires_at`, `name` (case-insensitive), `priority` and `prefix`, so a page
  of the list is read off an index instead of sorting every key; one on
  `rotated_from` for the successor lookup; and a partial index over the keys
  the list hides, so its `total` and `near_expiry` counts never scan the
  table. The `rotated_from` index also stops deleting a key from scanning the
  whole table for its successor. 0009's `idx_tokens_expires_at` and 0007's
  `idx_api_tokens_prefix` are dropped: the new composites start with the same
  column and serve everything they did. Created with `IF NOT EXISTS`;
  nothing to backfill. 0018's manual rollback recipe now drops
  `idx_api_tokens_priority_id` before the `priority` column.
- **Search the token list by name.** `GET /api/tokens?q=` keeps keys whose
  name contains the text, ignoring case, with `%`, `_` and `\` matched
  literally (up to 64 characters). The tokens page has a search box for it
  (debounced, kept in the URL as `?q=`), and says "No tokens match …" with a
  **Clear search** button when nothing does.
- **Filter the token list to keys expiring soon.** `GET /api/tokens?near_expiry=1`
  keeps only the keys the `near_expiry` count counts (expiring within 30 days,
  not yet expired). The expiring-soon banner's **Show them** applies it, shown
  as a removable chip (`?expiring=1` in the page URL).
- **Usage by model on the token page.** A new card between the summary and
  the charts lists, for the selected period and key set (the "Include earlier
  keys" toggle applies), each model's requests, prefill, generation and total
  tokens and its share of the key's traffic, busiest first. Usage is
  attributed to the exact model VARIANT the engine was running -- backend,
  engine version or image, repo and revision, files, quantization, dtype,
  context, max sequences, trust-remote-code and extra args, never `extra_env` --
  plus what the launch actually ran: the engine image (the pin, or the
  driver's default), or the in-container engine's version, and the commit the
  revision resolved to in the HF cache. So a model whose row was changed in
  place (a try-stack engine swap, a settings edit), a warden upgrade that
  moved the default engine, or a `main` that moved upstream shows each
  configuration it was served as, expandable under the model with a compact
  settings line. The variant is fixed when the engine starts; editing a model
  without restarting it does not move attribution. Values of secret-looking
  extra-args flags (`key`, `token`, `secret`, `pass`, `auth`, `cred`) are
  stored as a digest, never in clear.
  The card says from when per-model data exists.
- **Per-model tokens chart.** The tokens-per-minute chart draws one line per
  model (for the active Prefill/Generation switch) when the period saw more
  than one, with a legend and per-model tooltip rows; the part of the period
  before per-model recording began keeps the total as a dashed "All models"
  line, and the footnote says so.
- **Migration 0036: per-key, per-variant minute rollup.** New
  `model_variants` (id = first 16 hex of sha256 over the canonical JSON of the
  model id and its identity fields) and `token_model_usage_minute` keyed on
  `(token_id, variant_id, minute)` with `model_id` alongside, written by the
  proxy next to the per-key rollup; `request_history.variant_id` (nullable)
  records the variant of each request. No backfill: per-model data starts at
  the upgrade. Recording is bookkeeping: a failure to write it is logged and
  never fails the proxied response.
- **`GET /api/tokens/{id}/series` returns `by_model`, `model_bins` and
  `by_model_since`.** `by_model` lists models (with their `variants`) and
  their totals and share; `model_bins` carries sparse per-bin, per-model
  prefill and generation rates on the same bins; `by_model_since` is the
  first per-model minute store-wide. `by_model=0` skips all three; the
  history strip passes it.

### Changed

- **The token list is paged and sortable, on the server.**
  `GET /api/tokens` takes `sort` (`name`, `prefix`, `created`, `expires`,
  `last_used`, `priority`, `usage_24h`, `status`), `dir` (`asc`/`desc`),
  `limit` (1–500) and `offset`, and its response gains `total`, `limit`,
  `offset` and `near_expiry` beside `items`. **The default page is 50 keys,
  newest first — a client that read the whole list from one call must now
  page through it** (see *Listing keys* in documents/API.md). Unknown values
  and out-of-range numbers are a 422. A page is two queries however many keys
  exist: the per-key "last 24h" query and the successor scan that made the
  old list slow with many keys are gone. On `/ui/tokens` the column headers
  sort (click again to reverse), a bar under the table pages it with 25, 50
  or 100 rows, sort and page are kept in the URL, and the expiring-soon
  banner counts across the whole list instead of listing the names.
- **`last_used_at` is stamped at most once a minute per key.** It is now
  indexed for the list's sort, and rewriting it on every proxied request cost
  a page write and a WAL append (~4 KB) per request for a busy key. A request
  updates it only when it is empty or more than 60 seconds old, so it reads
  accurate to a minute.
- **Prefill and generation.** The token page's summary reads "prefill tokens"
  and "generation tokens", and the tokens chart's switch "Prefill" /
  "Generation" (were prompt and completion).
- **Migration 0034: a partial index for the token page's `latency_since`.**
  `idx_request_history_token_finished` covers `request_history(finished_at)`
  for rows with a token id, so the "per-token timings start at" lookup on
  every `/series` poll reads one index entry instead of walking every older
  row without a token id. Created with `IF NOT EXISTS`; nothing to backfill
  (#251).
- **Documentation and README describe the token details page.** The
  public docs and the README now cover the per-key page, its usage charts
  and the god-mode dock that replaced the standalone page, with new
  screenshots; the stats screenshot no longer shows the removed God Mode
  button (#251).

### Fixed

- **Rotation history no longer says "in grace" for a key that is refused.**
  A predecessor that is paused, or that expired inside its grace window, now
  reads "cut off" on the token page's lineage card: `in_grace` is false for
  it, matching the 403/401 it actually gets (#251).
- **A custom range that lies entirely in the future gets a clear error.**
  `GET /api/tokens/{id}/series` clamps a future `to` to the server's clock;
  when that leaves nothing, the 422 now says "'from' is after the server's
  current time" instead of "'from' must be before 'to'", which the client had
  not got wrong (#251).
- **God mode's "Live" badge appears as soon as the stream opens.** The stream
  now starts with an SSE comment (`: connected`), so the browser's open event
  fires at once instead of waiting up to 15 s for the first event or
  keepalive. SSE parsers ignore comments; the `data:` frames are unchanged
  (#251).
- **A god-mode dock filtered to one key keeps alive on time.** Other keys'
  traffic could stretch the gap between keepalives to about 30 s; it is now
  at most the 15 s interval (#251).
- **The token page's history strip: the selection brush no longer hangs past
  the strip's right edge** on a young key, where the strip ends up to a minute
  before the selected window does.
- **The token page stops polling a token deleted elsewhere.** Once the token
  returns 404 the page shows "Token not found" and makes no further token,
  chart or strip requests (it used to keep getting 404s). The token itself is
  also re-polled every 10 s on the 1h and 6h ranges again; the page's own
  re-renders kept restarting that timer, so it never fired.
- **One header-metrics stream ticket per page load, not two.** On a fresh
  load the nav bar's live metrics minted its first SSE ticket before the
  session was restored, got a 401, and minted again.

## [v2026.09.18.2] — 2026-09-18

### Changed

- Token names on the stats page now link to the token's details page.

### Removed

- **Per-token rate limits (`rate_limit_tps`).** The limit counted only the
  tokenizer's estimate of *prompt* tokens over a 10 s sliding window, so
  completions were never charged, and any prompt larger than ten times the
  key's rate was rejected with 429 on every attempt, forever. Priority remains
  the per-key fairness control. `rate_limit_tps` is gone from every token API
  response (list, `GET /api/tokens/{id}`, create, PATCH, `/test`); a create or
  PATCH body that still sends it is refused with 422 rather than silently
  ignored. The tokens table loses its Rate column, the create dialog and the
  token page's Limits card lose the rate field, and the
  `VW_RATE_LIMIT_WINDOW_S` setting is gone. The `api_tokens.rate_limit_tps`
  column is left in place, unused — SQLite cannot drop it cheaply — and
  rotation no longer copies it to the successor.

### Fixed

- **The token page's custom range: Apply never looks like it did nothing.**
  Apply now always refetches the charts and the history strip, even when the
  times are unchanged (a custom range does not poll, so it used to neither
  refetch nor respond). Times outside the key's lifetime are clamped as
  before, but the fields now show the clamped times, with a short
  "Adjusted to this key's lifetime." note until the next edit. The history
  strip of a key younger than two days labels its axis with times instead of
  the same date five times over.

## [v2026.09.18.1] — 2026-09-18

### Added

- **A token details page, `/ui/tokens/{id}`.** Everything that was spread
  across the tokens table and a shared god-mode page now lives on the one
  key: rename, rate/priority, pause/resume, test, rotate and delete, plus
  usage charts over 1h / 6h / 24h / 7d or a custom window, a history strip
  for picking an older period, and a per-key god-mode dock that streams
  prompt/output events for that key alone, and only while the dock is open.
  New `GET /api/tokens/{id}` serves the page's detail — every field the list
  carries, plus `lineage`, the key's rotation chain oldest-first — and
  `PATCH /api/tokens/{id}` now accepts `name` and `paused` alongside the
  existing fields. `GET /api/tokens/{id}/series` bins usage and timings
  server-side for any window (≤360 bins, with timing percentiles), sampling
  at a uniform stride once a window holds more than 50,000 request-history
  rows so a busy key over 7d stays cheap. `GET /api/admin/godmode/status`
  reports whether god mode is switched on (`{enabled}`), and the existing god-mode
  stream takes `?token_ids=` to narrow it to one key's chain instead of the
  whole fleet. God-mode events now carry `token_id` throughout.

### Changed

- **A paused key is refused with 403, not 401.** `require_bearer` answers a
  paused token with `403 {"detail": "token paused"}` once the expired and
  revoked checks have passed — those still 401, since a paused key is a
  live key an operator switched off, not a dead one. `GET /api/tokens` items
  carry `paused_at` and `is_paused` alongside the existing fields.
  Migration 0033 adds `api_tokens.paused_at` and `request_history.token_id`
  (plus `idx_request_history_token`); there is no backfill, so per-key
  latency and queue-wait history starts at the deploy that ships this file,
  not before.

  Rollback note: rolling back to an image from before this feature, after
  0033 has run, is schema-safe — the added columns are simply ignored. It is
  not behavior-safe: any key paused while the new image was live goes back
  to being treated as live the moment the old image starts serving requests
  again, since nothing in the old code reads `paused_at`.

### Removed

- **The standalone `/ui/godmode` page, and the God Mode link on
  `/ui/stats`.** God mode now lives in the token details page's dock, scoped
  to the key you opened it from.

## [v2026.09.11.2] — 2026-09-11

### Fixed

- **Two intermittent unit-test failures fixed.**
  `test_unload_route_evicts_tokenizer_cache` used
  `asyncio.get_event_loop().run_until_complete(...)`, which raises
  `RuntimeError: There is no current event loop in thread 'MainThread'` when
  nothing earlier in that worker happened to set one — intermittent under
  `pytest -n auto` and deterministic when `tests/unit/stats` ran alone. It now
  uses `asyncio.run`, which owns its loop. Separately, the request-history
  store's background-writer drain was polled with a 5-second deadline, too
  tight on a loaded runner; it is 30s, which bounds the failure case without
  slowing the passing one.

- **The token-throughput panel no longer presents two misleading numbers as
  engine speed.** Shipped in v2026.09.11.1 and caught on the first real
  deployment. The per-request basis is wrong in *both* directions on a
  long-context, concurrent workload:

  - `prompt_tokens / ttft` counts prefix-cached tokens the engine never
    computed. Measured on a live 27B: median prompt 46,363 tokens, median TTFT
    1.9s — about 26,000 tok/s. Prefilling 46k tokens through a 27B is roughly
    2.5e15 FLOPs, which four A4000s at a generous 100 TFLOPS would need ~25
    *seconds* for. The tokens came from the prefix cache. It measures cache
    hits, not compute, so it is no longer labelled a prefill rate — it is
    "prompt intake", and says what it counts.
  - `(completion_tokens - 1) / (duration - ttft)` is one request's share of a
    batched engine, not the engine's aggregate — 1.5 tok/s per request on a box
    doing far more. It now says so.

  The panel therefore **defaults to the wall-clock basis**, which reads the
  minute rollup and is distorted by neither. The per-request basis remains
  available for comparing requests against each other, which is what it is
  actually good for.

  The panel also moved under Context and cache at the same two-thirds width,
  sized to sit level with the Preemptions tile, and is rendered even when the
  engine is idle — it reads persisted history, which outlives the engine.


## [v2026.09.11.1] — 2026-09-11

### Added

- **Admission-queue wait is now recorded per request** (`request_history.queued_s`,
  migration 0032). The proxy admits every `/v1` request through a per-engine,
  strict-priority gate, and time spent waiting there was in no stored column:
  `started_monotonic` — the clock both `ttft_s` and `duration_s` are measured
  from — is read once the slot is held, so a request that queued for ten seconds
  looked identical to one that did not. `PriorityScheduler`'s own docstring warns
  that priority-9 traffic can starve lower priorities indefinitely, and nothing
  in the product could show it happening. The column measures the acquire alone,
  so it excludes tokenization and the rate-limit check, and it is backend-
  independent: `acquire` sits in the single `_forward` path ahead of any backend
  branching. Nullable rather than `DEFAULT 0` — rows written before the migration
  were not measured, which is not the claim "waited zero seconds". It surfaces on
  `GET /api/stats/v2/requests`.

### Changed

- **The stats page reports prefill and generation speed separately, instead of
  one blended tokens-per-second tile.** `current.tps` is
  `(prompt_tokens + completion_tokens) / 60` over the last full minute, which
  averages two quantities that move independently: prefill is compute-bound and
  runs in the hundreds or thousands of tok/s, generation is
  memory-bandwidth-bound and runs in the tens. The result described neither, and
  it moved whenever the prompt-to-completion ratio moved even though the
  hardware had not. A new `Token throughput` panel replaces the tile with two
  series, each as average / max / mode over the selected window, leaving the
  current row as three genuinely host-scoped tiles.

  `GET /api/stats/v2/throughput` serves it, with two bases the response names.
  `basis=request` derives per-request engine speed from the store the latency
  panels already read — `prompt_tokens / ttft` and
  `(completion_tokens - 1) / (duration - ttft)` — so idle time is invisible to
  it. `basis=wallclock` reads the `model_samples` minute rollup and counts idle
  minutes as zero. Two caveats the panel states rather than hides: TTFT is timed
  from arrival and so includes queue wait, which `request_history` has no column
  to subtract; and a prefix-cache hit counts prompt tokens it never computed, so
  prefill is bimodal by construction. That is why `mode` is reported at all —
  over bins about 5% wide, and null rather than invented when the sample is too
  small or nothing repeats, since an average across the two humps lands in the
  trough between them.

## [v2026.09.10.1] — 2026-09-10

### Fixed

- **An idle GPU is no longer reported as throttled.** The stats panel showed
  `Power cap — clock held at 6%` on healthy, idle cards. NVML raises
  `sw_power_cap` whenever the driver holds clocks below maximum — including
  when the only reason is that there is no work to do — and says so with a
  dedicated `GpuIdle` bit that `app/system/gpu.py` already parsed into
  `throttle.idle`. The verdict never read it. The percentage compounded the
  error: it is only ever `sm_clock / sm_clock_max`, so an idle 180 MHz of 3090
  rendered as "held at 6%" — a number that reads as severity while measuring
  nothing but the idle clock. A soft power cap is now suppressed when the
  driver reports the card idle, hardware faults (`hw_thermal`,
  `hw_power_brake`, `hw_slowdown`) still surface either way, and the ratio is
  worded as `clock at N% of max`.

- **A model's GPU indices can no longer be edited to a set the host disallows.**
  `PATCH /api/models/{id}/settings` was the one write path that skipped the
  setup allowlist. Creation rejects an out-of-range index with a 400 and the
  load runner with a 422, but `_derive_patchable_model_fields` makes new
  `ModelRow` columns patchable by default, so `gpu_indices` became writable
  with no value check and the handler wrote the list verbatim into SQLite. The
  row then loaded nowhere, and the complaint —
  `gpu_indices [0, 1, 2, 3] not subset of allowed [0, 1]` — arrived at load
  time, far from the edit that caused it, on a page that shows no allowlist.
  The check now mirrors creation's, with the same status and wording, and
  additionally rejects a non-list, a non-integer element and an empty list.

## [v2026.09.07.3] — 2026-09-07

### Changed

- **The public documentation moved off the repository root into `documents/`,
  and `CONTRIBUTING.md` into `.github/`.** A visitor arriving at the GitHub
  mirror met eight Markdown files before any code. Now the root carries
  `README.md` and the changelog only: `API.md`, `ARCHITECTURE.md`,
  `HAZARDS.md`, `INSTALL.md` and `OPERATING.md` are `documents/`, and
  `CONTRIBUTING.md` is `.github/CONTRIBUTING.md` — one of the three locations
  GitHub reads community-health files from, so the "Contributing guidelines"
  prompt on issues and pull requests keeps working. The new folder is
  `documents/`, not `docs/`, deliberately: `docs/` is the *internal* directory
  and `publish/exclude.txt` denies it wholesale, so a new internal note is
  private the moment it is written. Publishing out of `documents/` leaves that
  deny-by-default posture untouched rather than inverting it into an allow
  list. `exclude.txt` gains no new rule, only a comment naming the invariant —
  `docs/` is internal, `documents/` is public — and `docs/shared/` stays where
  it is, a build input for `scripts/sync-shared-docs.py` rather than reading
  material.
- `MANAGED_DOCUMENTS` in `scripts/sync-shared-docs.py` now names
  `documents/INSTALL.md`, `documents/API.md` and `documents/HAZARDS.md`
  (`README.md` and `docs/catalog-shared-regions.md` are unchanged), so the
  seven machine-managed `shared:` regions stay managed instead of silently
  falling out of the check. Cross-links, the four `assets/screenshots/`
  image paths in `INSTALL.md`, and the references from `install.sh`,
  `.env.example`, `Dockerfile`, `Makefile`, `.gitlab-ci.yml` and the
  headless-first-run test docstring were retargeted with them.

### Fixed

- **The "vLLM Warden" wordmark guard only ever scanned `README.md`, so the
  documents the README was split into had no guard at all.** `SCANNED_FILES`
  in `tests/unit/test_product_name.py` was `("README.md",)` and stayed that
  way through v2026.09.07.2, which means five public documents and
  `CONTRIBUTING.md` — the majority of the prose a stranger reads — could have
  carried the old product name past CI. They happened to be clean, which is
  luck rather than a guard; the catalogue row is what the same gap looks like
  when the luck runs out, its name, About text and `stack_label` all still
  branded "vLLM Warden" with no check that enumerated them. The guard now
  *discovers* its surfaces instead of listing them: every `.md` at the
  repository root, and every `.md` under `documents/` and `.github/` at any
  depth, alongside the existing `app/`, `frontend/src/`, `deploy/hub/` and
  `publish/` trees. A document added next month is covered without anyone
  editing the test. The root glob is deny-by-default with exactly one named
  exemption, `changelog.md`, pinned by its own assertion so a second one
  cannot be added quietly — the file is a historical record and quotes the old
  name three times by design. `docs/` stays unscanned for the same reason.

## [v2026.09.07.2] — 2026-09-07

### Changed

- **The README is split by section: 1,176 lines became 399, and the rest moved
  to six root documents it links to.** Nobody finishes a 1,200-line README.
  What stays answers four questions in order — what this is, whether it runs
  on your hardware, how to try it, and where the depth is: the opening, *What
  it is, and what it is not*, a new *Will it run on my hardware?* section
  (the NVIDIA-only and CUDA 13 floor paragraph, verbatim), *Don't use this
  if…*, the quick start, the two-engine story in short form with every
  limitation it carried (no llama.cpp latency histograms, "not reported" never
  rendered as `0`, the fixed llama.cpp pin, the GGUF tokenizer fallback), the
  *What you get* table and the nine screenshots, a short pointer on where
  request content can end up, and a map of the other documents that says what
  each one holds. Everything else moved, verbatim, to files at the repository
  root — the only place the publish ships, since `publish/exclude.txt` strips
  `/docs/`: the install variants (unattended flags, the no-clone one-liner,
  the offline / air-gapped route) and the mandatory cookie secret merged into
  `INSTALL.md` as A5–A7 and a *Before you start* subsection rather than
  spawning a third install document; the headless first run and *Adding a
  model from the API* are `API.md`; the five load-time hazards and *Measuring
  instead of guessing* are `HAZARDS.md`; *Day-to-day*, upgrading, HTTP against
  HTTPS and the full *Where request content can end up* are `OPERATING.md`;
  the topology and the full two-engine account are `ARCHITECTURE.md`; building
  from source, the image internals, the dev targets and adding a backend are
  `CONTRIBUTING.md`. The seven machine-managed `shared:` regions moved with
  their sections, so `scripts/sync-shared-docs.py` now lists `INSTALL.md`,
  `API.md` and `HAZARDS.md` alongside `README.md` and the catalogue hand-off in
  `MANAGED_DOCUMENTS`; the fragments, the check and the CI job are unchanged.
  `install.sh` and `.env.example` now point at `INSTALL.md` for the offline
  route instead of a README section that no longer exists.

### Fixed

- **The public content-log description understated its own protections.**
  *Where request content can end up* — README until this change, now
  `OPERATING.md` — still said the file "has no rotation and no retention" and
  "is created with the process's default permissions", and that the
  `VW_CONTENT_LOG_*` variables were absent from `.env.example`. All three were
  overtaken by v2026.09.07.1, the same release the text shipped in: the file
  is created `0600` in a `0700` directory at creation rather than at the umask
  (an existing file keeps its mode, so a pre-v2026.09.07.1 log still wants one
  `chmod 600`), `VW_CONTENT_LOG_MAX_BYTES` caps the file at 512 MiB and the
  writer stops with one warning rather than rotating or deleting (`<= 0`
  switches it off, as for `VW_ENGINE_LOG_MAX_BYTES`), and all five variables
  are in `.env.example`. What was still true stays: nothing prunes the file,
  it shares the volume with the SQLite database, and `VW_CONTENT_LOG_MAX_CHARS`
  bounds a record, not the file.

## [v2026.09.07.1] — 2026-09-07

### Security

- **The content log was created at the process umask — world-readable prompts.**
  `content_log.py` created `/data/logs/` and `content.jsonl` with no mode at
  all, so on a stock image (umask 0o022) a file holding users' prompts and
  completions verbatim landed as 0o644 in a 0o755 directory. The directory is
  now created 0700 and the file 0600, both applied by `mkdir(2)`/`open(2)` at
  creation, so there is no window in which the file exists world-readable and
  the modes do not depend on the umask. An **existing** file or directory keeps
  the mode it has: this runs on every logged request and must not undo an
  operator's deliberate widening once per request. The flip side is that a
  content log written by an earlier build keeps its old mode — if you have one,
  `chmod 600` it once.

### Added

- **The five content-logging variables are in the configuration contract at
  last.** `VW_CONTENT_LOG_ENABLED`, `VW_CONTENT_LOG_TOKENS`,
  `VW_CONTENT_LOG_PATH`, `VW_CONTENT_LOG_MAX_CHARS` and `VW_RUNAWAY_MODE` have
  been read by `app/config.py` since the content logger shipped, but appeared
  in neither `.env.example` nor the PodWarden Hub catalogue's `env_schema` —
  while `VW_GODMODE_ENABLED`, which only ever holds content in RAM, was in
  both. An operator auditing what this stack can retain, by reading the file
  the README calls the configuration contract, concluded it retains nothing.
  The entries spell out what the defaults do not: content logging is off and
  needs *both* the flag and the request's token id allowlisted (there is no
  log-everything mode); the destination is the persistent data volume, shared
  with the SQLite database; nothing rotates or prunes it — `stats_pruner.py`
  covers `request_history` and the sample tables only — so filling that volume
  is an outage, not a disk-space nuisance; `VW_CONTENT_LOG_MAX_CHARS` caps a
  record, not the file; and `VW_RUNAWAY_MODE=log` writes *through* the
  content-log sink, so it captures nothing on its own. No default changed.

- **`VW_CONTENT_LOG_MAX_BYTES` — a ceiling on the content log itself, default
  512 MiB.** `VW_CONTENT_LOG_MAX_CHARS` has only ever bounded one *record*; the
  file had no cap at all, on the volume that also holds the SQLite database.
  512 MiB because the Hub template pins `vllm-warden-data` at 10 GiB shared
  with that database, so it leaves the database an order of magnitude of
  headroom while still holding roughly 6,000 records at the ~80 KB a
  default-capped record costs. **On reaching the ceiling the logger stops
  writing and logs one warning naming the file and the limit. It does not
  rotate and it does not delete.** The hazard here is an outage rather than a
  tidiness problem, so the right response is a human looking at it — and
  keeping one rotated generation, as `VW_ENGINE_LOG_MAX_BYTES` does for engine
  logs, would both make the real budget 2x the number in the variable and
  throw away the incidents the logger was armed to capture. Archive or remove
  the file, or turn `VW_CONTENT_LOG_ENABLED` off, to resume. `0` or negative
  switches the ceiling off, the same convention `VW_ENGINE_LOG_MAX_BYTES`
  uses. In `.env.example`, in the Hub `env_schema` **and** in both compose
  `environment:` blocks that reach a container.

### Fixed

- **The content log wrote synchronously on the event loop.** `write_entry` did
  open/write/close with no `await`, called straight from the async forward
  path (five call sites in `app/proxy/routes.py`). The image runs a **single**
  uvicorn worker, so nothing absorbed it: every *other* in-flight request —
  SSE chunks being pumped to other clients included — stalled for the duration
  of an ~80 KB synchronous disk write, once per logged request. The module's
  own contract claimed it "never breaks or delays the response", which was
  true only of the request being logged, whose scheduler slot is released
  first. The write now happens on a worker thread via
  `anyio.to_thread.run_sync`, and `write_entry` is a coroutine the forward path
  awaits. Ordering survives the move: a thread pool is exactly the interleaving
  the single worker used to rule out, so each record's place in the file is
  fixed on the event loop before the offload and the worker drains that queue
  under a lock — order in the file is submission order, and concurrent callers
  coalesce into one append instead of one apiece. Only reachable with content
  logging enabled; disabled, the forward path is unchanged.

- **`VW_CONTENT_LOG_PATH` silently escaped the data volume.** It defaulted to
  the literal `/data/logs/content.jsonl` while the volume it belongs on comes
  from `VW_DATA_DIR` — a knob the Hub template exposes and interpolates. Move
  the data directory and the content log kept writing to `/data/logs/`, which
  is then no longer the mount: the records landed on the container's writable
  layer, filled the **node's** disk instead of the volume, and vanished on
  restart, silently on all three counts. It now defaults to
  `<VW_DATA_DIR>/logs/content.jsonl`; an explicit `VW_CONTENT_LOG_PATH` still
  wins, and blank counts as unset (the compose files render it as
  `"${VW_CONTENT_LOG_PATH:-}"`, and an empty value would otherwise become
  `Path(".")` — the process working directory). Both compose sources and the
  Hub `env_schema` dropped their hardcoded `/data/logs/content.jsonl`, which
  had been overriding the derivation on every install. For anyone on the
  default `VW_DATA_DIR=/data` the path is byte-identical to before.

- **`VW_CONTENT_LOG_MAX_CHARS=-1` disabled the per-record cap.** `_cap` read
  `if max_chars >= 0 and len(text) > max_chars`, so a negative value wrote
  every record in full — flatly contradicting the module's contract, and `-1`
  is exactly the value an operator reaches for, since `VW_REQUEST_MAX_WALL_S=0`
  does mean "no cap" a few lines away in the same `.env.example`. An
  undocumented escape hatch on the only bound the records had. A character
  count has no negative reading, so a negative one is now clamped to 0 —
  capture nothing, keep the record's metadata — with a warning at start-up
  naming the variable. That is the fail-safe direction: an operator who meant
  "unlimited" sees empty prompts immediately and fixes the value, where the old
  behaviour handed them unbounded records and an eventual full volume with no
  warning at all. `.env.example`, the config comment and the docstring now all
  say the same thing: there is no value that means unlimited.

- **The write-failure handler amplified the failure it reports.** Every
  exception was swallowed with `log.warning(..., exc_info=True)` — a full
  traceback per logged request. The failure this file causes is a full volume,
  so once it arrived the handler added write pressure to a disk already out of
  space, once per request, for as long as the logger stayed armed. The first
  failure is now logged in full, naming the path; identical repeats (keyed on
  the exception's type and message, so a *different* fault still gets its own
  report) are counted and suppressed. A successful write clears the
  suppression and reports how many were swallowed, so a transient fault cannot
  silence the warning permanently.

- **Those same five knobs reached no container.** `install.sh` writes the api
  service's `environment:` block explicitly, and Compose passes nothing to a
  service that does not name a variable — so setting any of them in `.env` did
  nothing whatever. They are now forwarded in the generated
  `docker-compose.override.yml` with their existing defaults, and added to the
  Hub template's compose source for the same reason (an `env_schema` entry
  alone advertises a knob that never arrives). Behaviour for anyone not setting
  them is unchanged.

- **The shipped llama.cpp only had kernels for the two cards on one developer's
  desk.** `Dockerfile` compiled it with
  `-DCMAKE_CUDA_ARCHITECTURES="75-real;86-real"` — Turing and Ampere as native
  SASS, and `-real` emits SASS and nothing else. `cuobjdump --list-elf` on the
  published `libggml-cuda.so` returned exactly `sm_75, sm_86`, and
  `--list-ptx` returned nothing at all, so there was no JIT fallback either.
  On an Ada (RTX 40-series, L4, L40S), Hopper (H100) or Blackwell (RTX 50-series,
  B200) card the product's second engine was not slow, it was **absent** — on
  hardware it should be good at. The narrowing was deliberate and documented,
  but it optimised for one build host rather than for everyone who installs the
  published image.

  The list is now `ARG LLAMACPP_CUDA_ARCHS`, defaulting to every architecture
  the base image's toolchain can target. Not assumed — read out of the pinned
  base with `nvcc --list-gpu-arch` (CUDA 13.0, V13.0.88):
  `75 80 86 87 88 89 90 100 103 110 120 121`, all as `-real`, **plus
  `90-virtual`**. That PTX entry is the point of the change as much as the SASS
  is: it is the difference between a card nobody has heard of yet being slow for
  a few seconds on first load and it not working at all. It is `compute_90`
  rather than the numerically highest because llama.cpp's own CMake rewrites any
  plain `12X` to `12Xa` (verified by running it: *"Replacing 121-virtual … with
  121a-virtual"*), and an `-a` PTX blob JITs for that one architecture and
  nothing else — `90` is the highest architecture-generic PTX CUDA 13 offers and
  is upstream llama.cpp's own forward-compatibility fallback.

  Verified in the built binary rather than inferred from a clean compile:
  `cuobjdump --list-elf` now reports `sm_75 80 86 87 88 89 90 100 103 110 120a
  121a` — the list exactly, with `120a`/`121a` being upstream's deliberate FP4
  substitution — and `--list-ptx` reports 142 blobs at `.target sm_90`. The
  RPATH arrangement, the `GGML_STATIC` decision, the full clone and the `ldd`
  assertion are untouched and still pass; `llama-server --version` from the
  built image reports `build 10731`, and the rebuilt `libggml-cuda.so` was
  loaded on real hardware (an sm_86 RTX A4000 and an sm_75 Quadro RTX 5000),
  where it registers the CUDA backend and enumerates both cards.

  **This costs build time, and it is a build argument precisely so you do not
  have to pay it.** Measured on 16 cores / 32 threads, the llama.cpp compile
  step alone: 158 s for `86-real`, 240 s for the old `75-real;86-real`, 1069 s
  for the new default — about 77 s of fixed work plus 81 s per architecture, so
  a cold `docker compose build` on that host goes from 5 min to ~19 min, and on
  4–8 cores the step runs for 1–3 hours. `/opt/llamacpp` grows 85 MB → 285 MB
  and the api image 28.9 GB → 29.3 GB (+1.4%). Anyone building from source for
  a card they own should pass
  `--build-arg LLAMACPP_CUDA_ARCHS="86-real"` and finish *faster than before
  this change*. The wide list stays the default because the default is what CI
  publishes and what an unknown card has to run on; the README and `INSTALL.md`
  both carry the table and the 4-core figure up front.

### Changed

- **The llama.cpp compile no longer takes `-j$(nproc)` on trust — it is bounded
  by memory as well as by cores.** Widening the architecture list is what made
  the old `-j"$(nproc)"` unsafe: each `.cu` became 13 device compilations
  instead of 2, so every parallel lane now holds a `cicc`/`cudafe++`/`ptxas`
  child roughly 6.5x longer and all of them are heavy at once, where before
  they were briefly heavy and then idle. Core count stopped describing the
  peak. On the first CI run that ever attempted the full list this was not
  subtle: on a 12-core, 3.7 GiB shared runner the 12 nvcc lanes' children held
  1921 MB resident plus 797 MB swapped, and the kernel's OOM killer took
  **buildkitd** — not any compiler — 14% into the compile, so the build died as
  `rpc error: code = Unavailable ... EOF` with nothing in it naming memory. Any
  other pipeline building on that host during the window would have failed the
  same way.
  The build now allows one job per 512 MiB of build memory, capped by `nproc`
  and following a cgroup limit when the builder has one. 512 MiB is the largest
  single lane measured in that OOM report, rounded down — not a small number
  picked to feel safe. Because it is a per-lane budget rather than a ceiling it
  binds only below 512 MiB per core, so the 16C/32T host the architecture
  figures were measured on still gets all 32 lanes and is not slowed at all;
  the 3.7 GiB runner drops to `-j7` and completes instead of being killed. The
  chosen value is echoed into the build log, and
  `--build-arg LLAMACPP_BUILD_JOBS=N` overrides the whole calculation.

- **CI's merge-request image build compiles two CUDA architectures, not
  thirteen.** No shipped behaviour changes here and no default moves — the
  Dockerfile still defaults to the full list, which is what `docker build`
  from a plain clone gets and what the published image is built from — but what
  a green merge-request pipeline *means* does change, so it is written down.
  `build-image` is a smoke test: it runs `--output type=cacheonly`, pushes
  nothing, and nothing downstream reads it. With the widened list it took
  3439 s, of which 3386.6 s was the llama.cpp compile alone (pipeline 21410,
  job 244827), on one of only two runners with the disk for a ~9 GB CUDA base.
  Worse, it was a trap: that run had 12 cores and 6966 MiB and so compiled at
  `-j12`, while the other eligible runner has 3839 MiB and lands on `-j7`,
  which extrapolates the same compile to ~5900 s against a 3600 s job timeout.
  Both runs of the full list happened to be scheduled onto the roomier host;
  the first one that was not would have failed as an unexplained timeout.
  Merge-request builds now pass `LLAMACPP_CUDA_ARCHS=86-real;90-virtual` —
  one of each kind, so both the SASS and the PTX code paths are still
  exercised. Measured on the same runner at the same `-j12`, the compile step
  goes **3386.6 s → 1368.6 s**. Those two points fit about 1000 s of fixed work
  plus ~184 s per architecture, which is worth writing down because it is not
  the shape the 16C/32T figures above imply: the architecture list was ~2020 s
  of the old 3387 s and that is what this recovers, but the ~1370 s left is
  llama.cpp's own C++ compile and link and no shorter list touches it. Dropping
  to a single architecture would save ~184 s and cost a codegen path, so this
  is the floor for the arg. The slow runner's `-j7` extrapolation comes down
  from ~5900 s to ~2400 s, which is the point: inside the timeout instead of
  past it. A narrow build can never hit the registry build cache, which `main`
  writes with the full list — but it does hit the runner's own builder cache,
  so that ~1370 s is paid once per build host per 24 h and the next push to the
  same merge request came back green in **11.66 s**.
  **A green merge-request build no longer proves the twelve-architecture
  compile works.** That is proven, unnarrowed, on every push to `develop`,
  again on `main` (which is also the sole writer of the BuildKit cache and so
  must build the full list or leave the next release to compile it cold), and
  finally on the tag by `publish:images`, which builds the artefact that
  ships. To force the full compile on a branch before merging,
  `glab ci run -b <branch> --variables VW_FORCE_IMAGE_BUILD:1` still does it:
  that is not a merge-request event, so it gets the Dockerfile default.

- **The public README speaks to whoever owns the GPUs, and stops narrating its
  own argument.** Two problems, both in the front half. First, the
  objection-handling scaffolding was visible: a section literally titled
  *"What this probably looks like from here"* opened with *"You have seen a lot
  of these"* and then answered seven quoted objections in bold. Every fact in
  it survives — it is a wrapper and that is the whole claim; three containers,
  JWT/CSRF, numbered migrations, the `/health` watchdog, `mypy --strict`; the
  exact shape of the GPU support; nothing phones home; leaving is a `base_url`
  change; what is automatic and what deliberately is not — but they now arrive
  as plain statements under **What it is, and what it is not**, a title that
  says what the section contains rather than what it is doing to the reader.
  The quoted-objection format, the "you have seen"/"possibly true, and here is
  the specific shape of it"/"partly"/"genuinely"/"without the marketing"
  framing devices, and the two sentences that commented on how READMEs are
  written are gone. `## Don't use this if…` stays: a real disqualification list
  is information, not a move.
- Second, the opening addressed only a hobbyist ("a machine under a desk").
  It now names the frame both audiences share — NVIDIA GPUs in a rack, a
  workstation or under a desk; an OpenAI-compatible API on them, with keys,
  accounting and visibility — and surfaces capability that already shipped but
  was under-sold or buried: per-key usage accounting (requests, prompt and
  completion tokens, per key and per client IP, `GET /api/tokens/{id}/usage`),
  per-token rate limiting and its `VW_RATE_LIMIT_WINDOW_S` window, the single
  published port that expects your own TLS terminator/ingress/SSO in front of
  it, request-level visibility over in-flight and finished requests, the
  air-gapped install as a headline rather than an Install subsection, and the
  fact that a request costs electricity rather than a per-token invoice line.
- **New section: "Where request content can end up".** The README makes a
  data-residency claim in its opening ("Nothing leaves the host"), and both
  content-capture features are individually honest about their bounds — but a
  reader assembling those into "this product never puts my prompts on disk"
  would be wrong, and would find that out from a `content.jsonl` on a volume.
  The question is now answered in one place, in order: the default (neither is
  on, and with them off the forward path is the code it would be without
  them); God Mode, in-memory and bounded and gone on restart; the content log,
  which writes prompt and completion text to `/data/logs/content.jsonl` on the
  same persistent volume as the SQLite database, gated by
  `VW_CONTENT_LOG_ENABLED` **and** an explicit `VW_CONTENT_LOG_TOKENS`
  allowlist that is empty by default, with `VW_CONTENT_LOG_MAX_CHARS` (40000)
  capping each record; and what neither does — `request_history` has no
  payload column, and the metadata path stays metadata. Three properties of
  that file are stated because none is obvious from the switch: it has no
  rotation and no retention and nothing ever prunes it, it shares a volume
  with the database so filling that volume is an outage, and it is created
  with default permissions. `VW_RUNAWAY_MODE=log` is documented as having no
  sink of its own — it enriches a record the content log was already writing.
- **God Mode is documented in the README for the first time, with its
  constraints rather than as a bare feature.** It is the only way to read what
  a client actually sent, and the qualifications are the load-bearing part for
  anyone evaluating this for a team: off by default (`VW_GODMODE_ENABLED`),
  and with it off the proxy forward path is the code it was before the feature
  existed; a bounded in-memory ring (2000 events / ~4M characters by default)
  that is evicted oldest-first and lost on restart; nothing written to the
  database or to disk; long prompts captured head-plus-tail, display-only,
  never altering what is forwarded; the bearer token never reaching the
  viewer. Stated equally plainly: while it is enabled, prompt and completion
  text is held in the api container's memory and any holder of the admin
  session can read it, because the project has no roles. Documented as a
  switch to turn on, use and turn off — not a posture to leave a shared
  deployment in. The neighbouring `request_history` bullet now says explicitly
  that the persisted table is metadata and has no prompt or completion column,
  so the two are not confused.
  No limitation was softened: NVIDIA-only, the FP8-on-Ampere warning, the
  llama.cpp architecture pin, the tensor-parallelism limit and the cost of the
  second engine (no latency histograms) all stay as plainly stated as before.
- **The llama.cpp CUDA architecture list is stated in exactly one place.** It
  was written out in three (the objections section, the build-timing note and
  the build-from-source note); the canonical statement is now the one under
  [Build from source](README.md#build-from-source), where the `Dockerfile` line
  you would change actually lives, and the other mentions cross-reference it.
  Widening the build is one line to update rather than three.

## [v2026.09.06.6] — 2026-09-06

### Fixed

- **`make smoke` swallowed the one thing it existed to report.** Under `set -e`
  the bare `code=$(curl ...)` assignment aborted the recipe the moment curl
  exited non-zero, so neither the per-path `printf` nor the `FAIL:` branch ever
  ran and make printed a naked `*** [smoke] Error 7`. The carefully worded
  failure message was unreachable for exactly the failure people actually hit —
  running smoke against a stack that is not up, or published on a different
  port. The curl status is now trapped, and the message names the full URL, what
  went wrong in English (connection refused, DNS, timeout, empty reply, broken
  mid-response), the curl exit code, and how to point `SMOKE_URL` somewhere
  else. A `--max-time 15` also stops a hung app from hanging the smoke run
  indefinitely. Found by a from-source install trial following the public
  README on a clean host.

### Added

- **`INSTALL.md` — a step-by-step installation manual written from a recorded
  install trial.** Two people-facing routes as separate top-level paths: run the
  published images, or build both images from source (including
  `@podwarden/chat-ui`, compiled from its public GitHub mirror at the pinned
  SHA). Every command and every block of expected output is taken from the trial
  transcripts rather than from what the software is meant to print, and each
  failure a first-time reader actually hit is documented inline at the step where
  it happens, with the verbatim error and the remedy: the Container Toolkit
  installed-but-not-registered preflight refusal, the Docker-daemon restart that
  `GPU_TOOLKIT_INSTALL=yes` performs, the `make restart && make smoke` race and
  its bare `Error 56`, `jq` being needed but absent from the requirements, the
  `403 {"detail":"csrf token invalid"}` that means either the wrong JSON field or
  a missing `-b jar`, the bare `vllm subprocess exited unexpectedly (rc=1)` that
  is really another process holding the card, the GPU-already-claimed refusal as
  an exclusive ownership claim that no amount of `gpu_memory_utilization` can
  satisfy, the benign first-load log-rotation traceback, the seven red pip
  `ERROR:` lines a clean build prints, and `make uninstall` freeing 4 GB of 62
  while leaving ~58 GB of images behind and refusing to run without a tty.
  Closes with a symptom-to-cause table.

  It lives at the repository root rather than under `docs/`, which
  `publish/exclude.txt` strips from the public snapshot in full — a manual for
  strangers has to reach the strangers. Its three new screenshots are under
  `assets/screenshots/install/`, which ships for the same reason.

- **The hazards the README and the Hub catalogue listing both document are now
  single-sourced.** Seven of them — the 64 MiB `/dev/shm` default and the
  tensor-parallel `SIGBUS` that follows minutes into serving, one loaded model
  per GPU as an exclusive claim rather than a VRAM check,
  `gpu_memory_utilization` reserving a fraction of the whole card, what fits on
  a 16 GiB card, why a first load takes minutes, the mandatory
  `VW_COOKIE_SECRET`, and the headless first-run sequence — were two
  hand-maintained copies, one in `README.md` and one in the catalogue row's
  long `content_md`. Copies drift, and the copy that is wrong is always the one
  someone is reading when it matters; `deploy/hub/` is this repo's standing
  proof, an unpublished mirror of that row that fell to 5 `env_schema` entries
  against the row's 16.

  Each hazard is now one self-contained fragment under `docs/shared/`, and both
  surfaces carry marker pairs — an opening `<!-- shared:shm-sigbus -->` and a
  closing `<!-- /shared:shm-sigbus -->` — that `scripts/sync-shared-docs.py`
  fills. `--check` re-assembles and diffs instead of writing, and runs in CI as
  `lint:shared-docs`; `make sync-shared-docs` / `make check-shared-docs` are
  the local wrappers. Same contract as
  `frontend/src/lib/api-types.generated.ts`: the marked regions are generated,
  they are never hand-edited, and CI computes the diff rather than trusting
  anyone to remember.

  **Marker blocks, deliberately not a generated README.** Only the seven fenced
  regions are machine-managed; every other line of `README.md` stays
  hand-written, so a contributor who opens it to fix a typo just fixes it. A
  wholly generated README would make the cheapest possible contribution
  require knowing an assembler exists — which is how a documentation pipeline
  quietly stops being used. The script's docstring says so, at length, so the
  design is not "simplified" away later.

  Fragments hold the body of a hazard, never its heading: the README's headings
  are linked from within the README (`#one-loaded-model-per-gpu`,
  `#first-run-without-a-browser`) and the catalogue words some of its own for a
  reader looking at a deployment form. Surface-specific material stays out —
  the `x-podwarden` block, `storage_class`/Longhorn, "the GPU lands on the
  primary service because Caddy owns the published port" and the topology table
  are catalogue-only; building from source, the pinned chat-ui SHA, the `make`
  targets and the contributing section are README-only. A hazard that is mostly
  shared with a surface-specific tail is split, with the tail after the closing
  marker — `/dev/shm` is the worked example.

  `publish/exclude.txt` strips `/docs/`, so the fragments never reach GitHub;
  they are a build input and `README.md` links to none of them. The assembler
  therefore reports and exits 0 in a public clone, where its inputs are absent
  by design. The catalogue half is staged in `docs/catalog-shared-regions.md`,
  which carries the assembled regions plus the mapping and splice procedure for
  the Hub row — the row itself is a hand-curated column this repo does not own
  and no script here writes it.

### Changed

- **The public README is restructured around the second engine, the reader's
  objections, and the gaps a trial install found.** The old one was accurate
  and opened by describing a control plane over vLLM; it barely mentioned that
  this product now runs two mainline backends, and it buried the four things a
  stranger following it literally had to reverse-engineer from source.

  What is new. A **"Two engines, and the model that made us add the second"**
  section that argues from the worked example rather than a feature list:
  `ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ` quantises the embedding table, mainline
  vLLM's loader has no entry for `embed_tokens.weight_packed`, the model card's
  remedy is a monkeypatch the product invariant forbids, and the same authors'
  GGUF conversion runs unmodified on llama.cpp — with the measured single-card
  numbers from `docs/tested-stacks.md`. It states the cost in the same breath:
  llama.cpp publishes no latency histograms, which is why the latency panels
  are measured by the proxy and why "not reported" is never rendered as `0`.
  An **accusation audit** naming the objections a reader arrives with (another
  wrapper, a Gradio script with a Dockerfile, it won't support my GPU, it will
  phone home, lock-in, the demo works and then nothing does) and conceding what
  is fair — including that the shipped llama.cpp is compiled `75-real;86-real`
  with no PTX fallback, so any other card needs a rebuild. A **"Don't use this
  if…"** section of real disqualifiers. And a **Hazards** section carrying the
  `/dev/shm` SIGBUS floor, the exclusive one-model-per-GPU claim (an ownership
  rule, not a VRAM check — lowering `gpu_memory_utilization` cannot get past
  it), the whole-card reservation with its 10.55 GiB measurement, what fits on
  16 GiB Ampere/Turing cards, why first loads take minutes, and the mandatory
  `VW_COOKIE_SECRET`.

  Those hazard sections and *First run without a browser* are written as
  **self-contained blocks** — no sentence in one depends on its neighbours or
  on README-only framing — so the same text can serve the catalogue listing
  without a rewrite. `docs/catalog-copy.md` (internal; `docs/` is stripped from
  the public snapshot) carries the derived catalogue copy.

  Two corrections. The architecture table said `/healthz` was served by
  Next.js; the Caddyfile proxies it to `api`, and it now says so. The old
  "Why you might want it" table credited an automatic request reaper; the
  wall-clock reaper is `request_max_wall_s` and defaults to `0.0`, so the
  README now says it is off by default rather than implying otherwise.

  Screenshots are **unreferenced for now**. Every image under
  `assets/screenshots/` predates the LLM Warden rename — the nav wordmark in
  each still reads "vLLM Warden" — and `03-stats-dashboard.png` shows a page
  the stats rebuild removed. The files are left in place and the README carries
  commented slots naming what each fresh capture should show, rather than
  shipping a shop window that is eight releases out of date.
## [v2026.09.06.5] — 2026-09-06

### Fixed

- **The published tree could not build its own UI image.** `frontend/Dockerfile`
  copied `.npmrc` by name and demanded an npm credential as a `required=true`
  BuildKit secret. The publish deliberately strips `.npmrc` — it routes the
  `@podwarden` scope at an authenticated registry a public clone cannot reach —
  so every `docker compose build ui` from GitHub died at
  `"/.npmrc": not found`, and the empty-file workaround then died at
  `secret npm: not found`. The exclusion was right and the consumer was wrong.
  `.npmrc*` is now a glob and the secret is optional, guarded by an explicit
  check so a checkout that DOES carry an `.npmrc` still fails fast and by name
  when the secret is missing — the fail-fast `required=true` existed for, minus
  the demand on builds with nothing to authenticate to. This also means
  `docker compose build ui` works for the first time anywhere: the compose
  `ui.build` block never declared a `secrets:` key, so it could not have passed
  one internally either.

### Added

- **The UI image compiles `@podwarden/chat-ui` from source.** A new `chatui`
  build stage clones `github.com/Podwarden/chat-ui` at a pinned 40-character
  commit, runs that repo's own `npm ci && npm run build && npm pack`, and
  overlays the result on what `npm ci` installed. `package-lock.json` still
  pins chat-ui's ~20 runtime dependencies with integrity hashes — that closure
  is what a lockfile is for — while the component's own code is now something
  the operator compiled rather than a tarball they downloaded. Same shape as
  the llama.cpp stage in the root `Dockerfile`, and for the same reason.

  A plain npm git dependency was tried first and does not work: npm builds a
  git dependency only when it declares `prepare`, chat-ui declares `prepack`,
  and `files: ["dist", ...]` then filters `src/` out of the pack — npm 10.8.2
  installs LICENSE/NOTICE/README/config/package.json and no code at all, which
  fails at bundle time rather than install time. Making it work would mean
  changing chat-ui's `package.json`, i.e. blocking this repo on another repo's
  release cycle.

  The pin is a SHA, not a tag, because the public chat-ui mirror carries no
  tags — it receives one squashed "Release vX" commit per release.
  `CHATUI_REPO` / `CHATUI_REF` / `CHATUI_VERSION` are overridable build args, so
  a fork or an unreleased chat-ui needs no edit to this repo. The build asserts
  the pinned version against both the lockfile and the freshly built tarball,
  and `tests/unit/system/test_chatui_pin.py` fails if `CHATUI_VERSION` drifts
  from `frontend/package.json`, if `CHATUI_REF` stops being a full SHA, or if
  `CHATUI_REPO` stops being the public mirror.

  Verified reproducible: the tarball built from source has sha1
  `6296e1db...` / `sha512-AiBzIbTJCMtFw...DqVsyEoruMEJg==`, byte-identical to
  the copy on `registry.npmjs.org` that `package-lock.json` already pins.

### Changed

- **README "Build from source" now describes a path that exists.** It said the
  UI image needs a private-registry read token and told readers to keep running
  the published image instead; neither is true any more. It now documents both
  images end to end — the registry-free `./install.sh --no-pull --no-start`
  bootstrap, `docker compose build`, the chat-ui pin and how to point it at a
  fork, what needs x86_64 and disk, and what needs a GPU (running, not
  building).
- **Sessions survive a plain-HTTP install.** The refresh cookie was always set
  `Secure`, and a browser silently discards a `Secure` cookie delivered over
  `http://` — the URL the quick start tells you to open. The access token
  lives only in memory, so `POST /api/auth/refresh` answered `401 missing
  refresh cookie` and every page load, reload or pasted link signed the
  operator out; on a 15-minute token even sitting still on the live dashboard
  did it. In-app navigation kept working, which is what made it easy to miss.
  Both session cookies now derive the flag from the scheme the browser
  actually used: a trusted `X-Forwarded-Proto` when `VW_TRUST_PROXY_ORIGIN=1`,
  else the connection's own scheme, else an all-`https://` `VW_FRONTEND_ORIGIN`
  — so a TLS deployment keeps `Secure` with or without a terminating proxy,
  and a LAN install stays signed in. The CSRF cookie, which was hardcoded the
  other way, now agrees with it; that disagreement was the bug's sharpest
  edge, leaving a session that looked half-alive rather than absent.
- **A GPU that is already busy says which model has it.** Loading a second
  model onto an occupied card reported `GPUs [0] already claimed`, naming
  neither the occupant nor a remedy, and reading like a capacity problem —
  so the natural response was to lower `gpu_memory_utilization`, which cannot
  help, because the claim is refused before the engine starts. It now says
  `GPU 0 is already serving 'qwen2.5-1.5b' — unload it first, or load this
  model on a free GPU. LLM Warden runs one loaded model per GPU.`
- **`fit-preview` and `POST /api/models` no longer disagree about their own
  fields.** The pre-check demanded `repo_id` where the create call takes
  `hf_repo`, and required a `filename` the create call leaves optional — so
  writing one body and reusing it for the other produced `Field required` for
  a field that appears in neither the UI nor the other endpoint. `fit-preview`
  now accepts `hf_repo` (`repo_id` still works) and resolves the weights file
  itself when `filename` is omitted. A GGUF repo, which publishes several
  independent quantisations that fit very differently, still requires an
  explicit `filename` — and now says so, listing the candidates.
- **The most likely first-run failure is now diagnosed.** A card already held
  by another process makes vLLM fail its pre-flight free-memory check before
  it profiles anything; that error matched no rule, so the model showed only
  `vllm subprocess exited unexpectedly (rc=1)`. It now reads `Not enough free
  VRAM on cuda:0 to start: 1.65 GiB free of 15.6 GiB, but this model asks for
  2.03 GiB (gpu_memory_utilization 0.13). Something else is holding the card
  …`.
- **No more `FileNotFoundError` traceback on a model's first load.** Log
  rotation ran before the log file existed and logged the miss as a warning
  with a stack trace. Nothing was ever lost; it simply looked like a fault at
  the exact moment a new operator is watching.
- **The engine-version control no longer describes itself as pointless.** On
  the in-container engine driver it explained that a version pin "would be
  silently discarded" — the pre-#177 behaviour. The supervisor has refused
  such a pin outright since; the sentence now says so, and names the setting
  that would enable the control.
- **`/ui/` no longer flashes an empty Models page before sending you to the
  sign-in form.** The shell rendered before the session was known, so a
  signed-out visitor's first impression of the product was a page that
  appeared broken and then threw them out. The app now waits for the one
  session check it was already going to make.

### Added

- **The README documents a first run without a browser.** The setup wizard
  was only ever described as a browser flow, so an unattended install had no
  documented path at all: the six `/api/setup/*` calls, their order, the
  400 that tells you which step you are actually on, the password rules
  (6 characters minimum, 72 bytes maximum), and the CSRF bootstrap whose
  response key is `csrf` and not `csrf_token` — a wrong guess there returns a
  flat `403 csrf token invalid` that points at nothing.
- **The README documents the model lifecycle API** — register, pull, load,
  unload — including that the three are separate asynchronous steps and that
  pull progress is a server-sent event stream rather than a JSON document.
- **The README explains two things about GPUs that surprise everyone**: a GPU
  hosts one loaded model at a time, and `gpu_memory_utilization` is a fraction
  of the whole card rather than of the model — which is why a 1.5B model sits
  in ~15 GB of a 16 GB card at the default 0.9, and why sizing a card by
  weights alone is wrong.
- `make smoke` is now offered as the post-install check for every install,
  not only for a build from source.

### Changed

- The installer warns, before it acts, that installing the NVIDIA Container
  Toolkit restarts the Docker daemon — which restarts every container on the
  host, not only this stack's. The README's flag table and requirements say
  the same.

## [v2026.09.06.4] — 2026-09-06

### Fixed

- **Stats: the interconnect graph dropped GPU 0 and overlapped its own hub.**
  `nvidia-smi topo -m` UNDERLINES its header row, so the first column arrives
  as `\x1b[4mGPU0` rather than `GPU0`. Matching `GPU<n>` skipped it, and a
  four-card host rendered as THREE cards — with a legend that counted three
  and advised on splitting a model across them. Escapes are stripped before
  parsing, and the regression test carries the real header verbatim, escapes
  included, so a naive matcher fails it. Separately the ring radius was a
  bare 120: a node beside the hub needs half of each box plus a gap, 124px,
  and the worst case is a three-card ring whose lower nodes sit at 30°, where
  120 buys 104px — an 8px collision. The radius is now derived from the box
  sizes and the angle, so it cannot drift out of agreement with them, and the
  SVG width follows from the ring rather than a second unrelated constant.
  Both were found by opening the deployed page; the branch had passed lint,
  types, 2291 Python tests and 712 frontend tests.


## [v2026.09.06.3] — 2026-09-06

### Added

- **Stats: the GPU cards are dials, and the cards' interconnect is drawn.**
  The card's head is a large utilisation number beside two bar gauges —
  load, and temperature with a tick at the driver's throttle point. The
  earlier concentric arcs were rejected twice: hard to read, ~120 px for
  two numbers, and a meaningless stub at 0 %. Core and memory clock now
  read against their own ceilings (`1350 / 2100 MHz`), because the ceiling
  is what makes a low reading readable — 210 MHz idle is normal, 1200 MHz
  at full load is a card being held back. The specs pair onto three lines
  (Driver | CUDA, Link, NVLink | ECC) divided by a hairline, wrapping
  independently on a narrow card, and a spec that disagrees with the other
  cards on the host is marked "◂ differs". Under the name: the generation,
  `Ampere · compute 8.6`, derived from the CUDA compute capability alone —
  nvidia-smi has no architecture field, the map is fixed, and an
  unrecognised capability shows its raw number and says the generation is
  not recognised rather than inventing a label the day a new one ships.
  The throttle verdict now comes from the driver's
  `clocks_throttle_reasons`, not a temperature ratio: on one live host all
  four cards report `sw_thermal_slowdown = Active` at 91–95 °C holding
  57–74 % of max clock, and the card says "Thermal slowdown — clock held at
  64 %". Thermal, power cap and hardware slowdown are named separately (they
  are different problems); a hardware slowdown is a fault, a software one
  a warning. That family is queried on its own because an unknown field
  name rejects the whole nvidia-smi query — a driver that dropped the old
  spelling gets the `clocks_event_reasons` spelling tried next, and one
  that knows neither leaves the verdict "not reported", never "not
  throttled". Below the cards, a new "How the cards reach each other"
  graph from `nvidia-smi topo -m`. No NVLink (both real hosts): a hub star
  to a `CPU · PCIe host bridge` node with each spoke labelled by the card's
  negotiated lane width, dashed and fault-coloured when it gets fewer lanes
  than it can use — and no GPU-to-GPU line, because `PHB` means the pair
  has no direct path. Bridged pairs: thick `NV2` edges card-to-card plus
  the PCIe spokes. Every pair NVLink: one NVSwitch node they all join, not
  a 28-line mesh. Up to four cards sit on an ellipse; past that, two rows,
  scrolling — 16-GPU hosts exist. `GET /api/system/gpus` gains
  `architecture`, `sm_clock_max_mhz`, `mem_clock_mhz`, `mem_clock_max_mhz`,
  a `throttle` block per card, and a top-level `topology` matrix.

## [v2026.09.06.2] — 2026-09-06

### Added

- **Stats: the GPU cards carry real per-card telemetry.** Each card on the
  System configuration panel now shows VRAM used / total with a meter, GPU
  utilisation, temperature against the driver's own throttle point, fan
  speed, power draw against *that card's* power limit, ECC mode, NVLink
  standing, SM clock and P-state — next to the driver and CUDA versions it
  already had. Fleets are heterogeneous: an RTX A4000 (16376 MiB, 140 W, ECC
  off) sits beside a Quadro RTX 5000 (15360 MiB, 230 W, ECC on) on a real
  host, so nothing is shared across cards. NVLink is a separate
  `nvidia-smi nvlink -s` call with three distinct answers — "not supported by
  this card", "supported, all links inactive", "N of M links active" — and
  the panel words each one; a card without NVLink silicon never reads as
  "inactive". A metric the hardware does not report renders as the words
  "not reported", never 0 and never blank: a fan-less card in a VM, a card
  with no power sensor and a card reading 0 W now look different. The live
  feed (`GET /api/system/gpus`, new `telemetry` block per card) degrades
  independently of the static inventory, so a failed probe leaves the
  static rows in place and says the live feed is unavailable.
- **Stats: PCIe generation and link width per card, read honestly.** Two
  traps from a real host. Both cards idle at PCIe Gen 1 against a max of
  Gen 3 — that is power saving, not a fault, and the card shows "Gen 1 now
  · up to Gen 3" without flagging it. GPU 0 negotiated x4 lanes of a
  possible x16 — a x4 slot or a riser, permanent, and on a host with no
  usable NVLink the only path between cards for tensor-parallel serving;
  that IS flagged, as a "reduced link width" chip in words and shape.
  The generation ceiling shown is the *negotiated* max (what the link will
  actually do); when the newer `gpumax`/`hostmax` fields say both
  endpoints can do more ("card Gen 4, host Gen 4 — link settled at Gen 3")
  that appears as a labelled note rather than a flattering substitute. A
  driver that lacks those fields loses only that note — they are queried
  separately so an unknown field name cannot take the rest of the
  telemetry down with it.
- **Stats: a hot idle card is said, not just coloured.** Temperature is
  shown against the driver's slowdown point with a meter, and gains a
  worded chip when it matters: "near throttle point" within 15% of it, or
  "hot for an idle card" within 25% at ≤5% utilisation — the case of four
  cards at 78–84 °C in P2 doing nothing, which a meter alone would bury.

### Removed

- **Stats: the VRAM-over-time chart.** It only moved when a model was
  loaded or unloaded — the deployed 7-day chart was one flat block at
  59.4 GiB with median equal to peak. That is structural, not a quirk of
  one box: vLLM pre-allocates its KV cache from `gpu_memory_utilization`
  at load, and llama.cpp allocates weights plus a KV cache sized to
  `n_ctx` at load, so on both engines VRAM is a step function while
  serving. Current VRAM lives on the per-GPU cards instead. The `vram`
  series stays in `GET /api/stats/v2/overview` and the sampler is
  untouched; only the chart is gone.

- **Stats: per-request history is persisted, and the latency distributions
  and the requests chart honour the window.** Two operator complaints had one
  cause: "recently finished makes little sense" (ten identical-looking rows,
  kept 15 minutes, nothing aggregated) and "why are the TTFT/ITL charts only
  5 minutes" (the engine publishes only cumulative histograms, so the page
  kept bucket snapshots in React state — the "last 5 minutes" it drew was
  really "since you opened the tab, capped at 5 minutes"). Nothing persisted a
  request. Now `_deregister` enqueues every completed request and a background
  writer lands it in a new `request_history` table (migration 0031); the
  proxy never touches the database on the slot-release path, and bookkeeping
  still cannot fail a request. Retention is 30 days by age AND 200k rows by
  count (`VW_REQUEST_HISTORY_RETENTION_DAYS`, `VW_REQUEST_HISTORY_MAX_ROWS`),
  pruned hourly with the other stats tables: 30 days is four times the widest
  window so 7d is always served in full and a last-N basis keeps depth when
  traffic is thin; the row cap bounds a load test to ~40 MB whatever its rate.
  `GET /api/stats/v2/requests` serves the rows over the window (every k-th row
  when there are more than 2000, and it says which k — a newest-N cut would
  leave the left of a time axis empty) and `GET /api/stats/v2/latency` serves
  TTFT, per-request mean ITL and duration distributions with exact quantiles,
  on the window or on the last 500 requests. Every response carries a
  `coverage` block, so a window the store cannot serve in full — history
  younger than the window, or a retention shorter than it — is stated
  precisely rather than drawn narrower than the button promises.

  The latency panels now come from the PROXY's own measurements (TTFT at the
  first streamed frame, duration at the end of the stream), which exist
  identically for every backend: **llama.cpp models get a latency panel for
  the first time** — that engine publishes no histogram at all. The engine
  histograms are no longer shown on the page: two "TTFT" panels from two
  sources over two windows would silently disagree. What is given up is the
  engine's per-token ITL resolution; the store's ITL is the mean gap per
  request and is labelled "mean per request" wherever it appears.

  "Recently finished" is replaced by a requests chart: time on x, duration on
  a log y, one mark per request, colour by client (or by model when one
  client dominates, or nothing when there is one of each — the legend says
  which), size by generated tokens, finish reason by SHAPE (filled / hollow /
  cross — never by colour, which is already carrying identity and is the same
  amber for positive and warn in retro-dark). Prompt tokens and TTFT ride in
  the tooltip. Clicking a legend chip isolates that client. The table
  survives as a collapsed detail view under the chart, newest first, with the
  window's depth rather than a ring's. `GET /api/stats/live/finished` remains
  for a UI image older than this API and reads the same store.

### Fixed

- **Stats: the GPU card's CUDA version was always "—".** `nvidia-smi
  --version` on R610 drivers prints `CUDA version : Deprecated, see "CUDA
  UMD version" instead` and puts the number under `CUDA UMD version : 13.3`;
  the parser wanted a capital-V `CUDA Version` followed by digits, matched
  neither, and left the slot null. It now reads either key
  (case-insensitively, digits required), and falls back to the plain
  `nvidia-smi` banner for drivers that predate `--version`.
- **Stats: the 24h reference labels were clipped, and overlapped each other.**
  They were pinned to fixed corners, which fails in two ordinary cases. A line
  at the TOP of the scale had its label drawn above it, outside the plot, where
  the panel clipped it — GPU utilisation pinned at 100% is not an edge case,
  and VRAM sitting at the card total is another. And median and peak both hugged
  the right edge, so when the two values are close they collided: a 511 W median
  under a 555 W peak was an unreadable smear. Placement is now computed from
  where each line sits in its own domain — upper part gets its label below,
  otherwise above, so neither edge can clip it — and peak keeps the right edge
  while median takes the left, so the two can never overlap at any values.

## [v2026.09.06.1] — 2026-09-06

### Added

- **Stats: a finished request outlives its own completion.** The live registry
  holds only in-flight requests and dropped each one in the streaming
  `finally`, so its duration, time-to-first-token and finish reason — the three
  things that only become knowable at that instant — were discarded exactly
  when they became worth having. That is why the dashboard went blank the
  moment something interesting ended. A bounded ring now keeps them, served by
  `GET /api/stats/live/finished`, scoped by the same `?models=` rules as every
  other stats endpoint. TTFT is measured by the proxy at the first streamed
  frame rather than taken from the engine: llama.cpp publishes no latency
  histogram at all, and the proxy is the one vantage point that sees every
  backend identically. Bounded by count *and* age, because either alone fails —
  a burst would blow the memory bound, and a quiet week would show last Tuesday
  as though it were current. In memory and process-local: it survives a reload
  and a navigation away, not a warden restart.
- **Stats: latency is emitted as a distribution, not only five quantiles.** The
  engine's full histogram buckets were fetched and then reduced to p50/p90/p99,
  throwing away the shape. TTFT is bimodal by construction — a prefix-cache hit
  lands near 0.3 s and a cold prefill near 6 s — so a single percentile falls in
  the valley between the two humps and describes a request that never happened.
  The buckets now ride alongside the quantiles, with `+Inf` encoded as `null`
  because JSON has no infinity and dropping that bucket would discard the whole
  tail. `bucket_deltas` turns two cumulative reads into a real windowed
  distribution, and returns nothing rather than a wrong number when there is no
  earlier read, when an engine restart has zeroed the counters, or when an
  engine upgrade has moved the bucket boundaries.

### Fixed

- **Stats: a finished-request record's `model_id` held the SERVED model name,
  so scoping the panel matched nothing.** Every other stats endpoint filters
  `?models=` on the models table's row id — `/api/stats/v2/overview` validates
  exactly that — but the proxy wrote the served name under `model_id`, so the
  same selection that scoped the overview correctly matched no finished rows
  at all whenever a model's id and served name differ (they usually do). The
  failure was silent: an empty table reads as "no requests", not as a bug. The
  record now carries `model` (served name, for display) and `model_id` (row id,
  for filtering) as distinct values, and the page scopes the endpoint instead
  of narrowing client-side. That also fixes a second fault in the workaround:
  the endpoint applies its row limit BEFORE the page could narrow, so a busy
  unselected model could push every selected row out of the response.

  Building the record was inline in the proxy's `_deregister` and the ring's
  tests inject records of their own making, so nothing ever checked what the
  proxy actually wrote — which is how this shipped. It is now
  `request_registry.finished_record()`, tested directly.

### Changed

- **Stats: one page.** Everything from `/ui/stats/live` moves into `/ui/stats`,
  next to the minute-level history that page already kept — which was the
  existing answer to "the charts reset"; Live's ~80 seconds of client-side
  React state was the anomaly. The merged page keeps one control bar (models +
  window), and the scoping rules are explicit rather than implied: the model
  selection scopes tokens, latency, KV, requests and preemptions, but NOT
  VRAM, GPU utilisation or power — two engines share a card, so watts cannot
  be attributed to one model, and those panels carry a `host` badge instead of
  silently re-scoping. KV and context are never combined across models (each
  engine has its own pool and its own `max_model_len`); the window governs
  history panels only, with live panels saying "not the window"; longer
  windows bucket up (1 min → 5 min → 30 min) and the heading says which; the
  history charts carry a fixed 24h reference (dashed median of busy minutes,
  dotted peak, the peak included in the y-scale); latency renders as the
  measured windowed distribution with percentile markers on it, and says "not
  reported" for an engine that publishes no histograms; every cumulative
  figure is greyed and labelled "since engine start". The model control adapts:
  inline chips up to four models, a summary button with a filterable checklist
  above that.

### Removed

- **`/ui/stats/live` — 404, not a redirect.** There is one stats page, and a
  silent redirect would leave people believing the old route still exists. The
  nav link went with it; the app's 404 page names Stats for the bookmark or
  the tab left open.
- **The MFU panel, and the `mfu` block from the live frame.** `mfu_estimate`
  was a literal `None` left for "a later iteration" — nothing ever resolved
  peak device FLOPs — and the `0 FLOP/s` line beneath it was a *cumulative*
  counter rendered through a per-second formatter, so its unit was wrong even
  when its value was not zero. Measured GPU utilisation and power answer the
  same question with numbers the box actually reports.

### Fixed

- **A scrape error no longer risks the whole stats page.** `_null_frame`
  emits `engine`/`throughput`/`cache`/`latency` as JSON null while keeping
  `model_id`, so the block reached a render path whose TypeScript types
  falsely declared those non-nullable and dereferenced them unguarded — and
  there was no error boundary under `/stats` to contain the crash. The types
  now say null, every render site guards, and the route has an error boundary.
- **The scrape-error banner no longer claims "showing the last good values".**
  Nothing holds a last good frame — the stream replaces state on every
  message — so the banner now says what actually happens: that model's live
  panels pause until the next successful scrape.

## [v2026.09.05.2] - 2026-09-05

### Changed

- **`@podwarden/chat-ui` bumped 0.1.8 -> 0.1.23, so the public snapshot is
  installable.** The pin was one day older than the package's public npm
  mirror, which only goes back to 0.1.19 — so `github.com/Podwarden/vllm-warden`
  would have shipped a dependency nobody outside could resolve, and
  `publish:github` failed closed on the internal registry hostname before it
  could. The two registries are not divergent lineages: the GitLab registry
  carries a continuous 0.1.0 -> 0.1.23 and public npm mirrors the recent tail,
  now within about a minute of each other. Confirmed identical rather than
  assumed — the 0.1.23 tarball is byte-for-byte the same on both
  (sha1 `6296e1db…`), and the dependency set is unchanged across all fifteen
  versions, so nothing but the package itself moves.

## [v2026.09.05.1] - 2026-09-05

### Added

- **`GET /v1/models` now reports `max_model_len`, so clients stop guessing the
  context window.** OpenAI has never specified a context field on this
  endpoint, so every OpenAI-compatible client either hardcodes a number or
  carries a catalog keyed by model name — neither of which can know what a
  private warden was launched with. vLLM answers this upstream by putting
  `max_model_len` on the model card (vllm-project/vllm#4643); warden was
  building its entries from the database and emitting only `id`/`object`/
  `owned_by`, so that answer never reached the client. The field is additive
  and clients that do not know it ignore it. It sits at the top level rather
  than inside `warden`, because it is a CONFIGURED fact known the moment the
  engine loads, whereas the `warden` block is gated on a stress run that most
  wardens never have — gating a known ceiling behind an optional measurement
  is what left clients guessing in the first place. The value is the engine's
  own resolution order, not the row's: a live override first (a
  reload-with-config or a stress sweep leaves the row untouched), then the
  row, then the model's `max_position_embeddings` — because for a NULL row
  that config value *is* the window both engines derive, not a guess about
  it. When no source states one the key is omitted rather than sent as null,
  so a client can still tell "unknown" apart from a real ceiling.

### Fixed

- **Installer: a free-disk preflight, and the documented image size was the
  compressed one.** `install.sh` checked Docker, Compose and the NVIDIA
  runtime and never free space, so a 30 GB VM passed every check and then
  died partway through `docker compose pull` with "no space left on device"
  (#249). The api image is 9.2 GB compressed on the wire — the figure
  `docs/backends.md` quoted — but 28.9 GB as Docker stores it: 19.7 GB
  unpacked, plus the compressed layers that the containerd image store keeps
  alongside. The installer now measures free space on the Docker data root
  (not `/`; they are often different filesystems), says the number, warns
  under the 40 GB the stack wants and refuses a first pull under 20 GB, where
  the image cannot even be unpacked — but only a first pull: a release already
  in the store shares its layers, and `--no-pull` has nothing to refuse. A
  check that cannot run (no data root from `docker info`, a `df` it could not
  parse) says so and carries on, the way the toolkit preflight does. The
  README states the disk requirement next to Docker, Compose and the GPU, with
  what models add on top. Also: the override log line printed `${VERSION}`
  and `${WARDEN_PORT}` literally and labelled the GPU index list "GPUs: 0",
  which on a one-GPU host reads as no GPU at all; it now prints the values and
  a count first ("GPUs: 1 (index 0)"). The model-pull runbook now says in as
  many words that `/pull/progress` is an SSE stream, not JSON to poll.


## [v2026.09.04.2] - 2026-09-04

### Added

- **Stress test: apply the measured setting and reload.** A run could tell you
  `max_model_len` should be 32,768 and then leave you to set it by hand. The
  results screen now offers **Apply and reload** beside the recommendation,
  and only when the recommendation actually differs from what is running —
  reloading to arrive at the setting already in force costs the model its
  availability and changes nothing. It is one server-side operation rather than
  three calls from the browser, because the settings patch refuses to touch a
  loaded model: applying means unload, persist, load, and a client that closes
  its tab midway would leave the model unloaded with nobody to finish. Refused,
  with a reason, when the run never finished, produced no recommendation, was
  measured under a different hardware fingerprint, or when a stress run
  currently holds the model. The response says plainly that the measurement
  described the configuration just replaced — applying invalidates its own
  evidence, which is why it is offered rather than done automatically.

### Fixed

- **The prebuilt CI dependency image is tagged per architecture, so an
  Apple-silicon workstation stops emulating.** The tag was a digest of the
  requirements pair and nothing else. `make deps-image` pulls before it builds,
  the CI builder runs on amd64 runners, so the registry held an amd64-only tag
  — and `docker pull` on an arm64 Mac fetches that quite happily and runs it
  under QEMU. The pull therefore *succeeded*, the local native build never ran,
  and every `make test-unit` / `lint` / `typecheck` was emulated. The prebuilt
  image, whose entire purpose was to make local runs fast, was making them
  slower than having no image at all. Measured on an M-series Mac, same suite,
  same command: **`make test-unit` 2743 s → 44 s**, `make lint` minutes → 1.5 s.
  The tag now carries `-amd64` / `-arm64`, so the arm64 pull misses and falls
  through to a local build — native, about as long as the pip install it
  replaces (69 s), and `make deps-image` now pushes what it built, so the next
  workstation pulls in ~10 s instead of rebuilding. The tag is content-addressed
  (requirements digest plus architecture), which is what makes pushing from a
  workstation safe: it is a shared cache, idempotent, and can only ever add the
  image someone else would have built identically. CI is unaffected beyond
  rebuilding its image once under the new tag, and still does not build arm64 —
  on an amd64 runner that would mean emulating the thing this change avoids.

- **`make typecheck` is green on a clean checkout and CI now runs it.** (#243)
  `mypy --strict app/` had 322 errors in 57 files on `develop` and nothing in
  `.gitlab-ci.yml` invoked it, so the Makefile advertised a check that was red
  for everyone and gated nowhere; on 2026-09-03 five agents each spent a full
  Docker mypy run proving the failure was not theirs. The errors are now
  recorded in `mypy-baseline.txt` (mypy's own output minus line numbers, plus
  the offending source line, sorted, human-readable) and
  `scripts/mypy-baseline.py` fails when the tree drifts from it in either
  direction: an error not in the baseline is a regression; a baseline entry
  mypy no longer reports is a stale baseline that `make typecheck-baseline`
  shrinks. mypy's configuration is unchanged (`strict = true`), no module is
  `ignore_errors`'d, and the mechanical `no-untyped-def`/`type-arg` debt stays
  visible as a to-do list rather than being silenced. A new `typecheck:mypy`
  CI job runs the same script in the same prebuilt dependency container as
  `lint`. `.mypy_cache/` is gitignored.

## [v2026.09.04.1] - 2026-09-04

### Fixed

- **Stress test: a probe is only retired after failing the baseline twice.**
  Excluding a probe used to rest on a single sample, and only two probes judge
  the context ladder — so two unlucky readings abort the whole run while four
  other probes may be answering perfectly. That contradicted the argument the
  rest of the search is built on: every ladder verdict is replicated because one
  sample near a threshold is a coin flip, while the baseline, which shapes the
  entire run, decided on one. A dropped stream alone trips `abrupt` through a
  missing `finish_reason`, and that was enough to cost the run an oracle
  permanently. The re-check costs one extra probe per failing probe, at the
  baseline, which is the cheapest place in the run — and when the second
  reading is clean it becomes the reference, since a reference taken from a
  failed probe would libel every later verdict measured against it.

- **Stress test: the probe budget follows the engine's real context, not a
  floor.** Three runs on the same model aborted with `aborted_no_baseline`
  because `models.max_model_len` was NULL — nobody has to set it — so every
  budget fell back to the unknown-context floor of 1,024 tokens and the needle
  probe starved at exactly that. The engine's own `/props` was reporting
  `n_ctx: 131072` throughout: the harness had 131k available and used 1k. It
  now asks the engine what context it will accept, for both engine shapes
  (llama.cpp `/props`, vLLM `/v1/models`), and the reference prompt is
  re-clamped once that answer arrives. This is deliberately the opposite policy
  to the model-ceiling resolver, which refuses the same number: that asks *how
  far can this model go*, where a running engine only echoes our own
  `--ctx-size` back; this asks *what will the engine accept right now*, where
  that configured value is exactly the authority.

## [v2026.09.03.9] - 2026-09-03

### Changed

- **Stress test: nothing about the probe budget is tuned to a particular
  model.** The escalation ladder stopped at a hardcoded 1,024 tokens, chosen
  because it suited the model in front of us. On `gpt-oss-20b-gguf` the needle
  probe reached that ceiling and the run aborted with `aborted_no_baseline` —
  and raising the constant would have fixed that one model while mis-sizing
  every other, since a 2k-context model cannot spend 1,024 tokens reasoning
  about a 1.5k prompt and a 262k model can spend far more. The ceiling is now
  the arithmetic that holds for every model: the answer must fit in the context
  alongside the prompt, using the engine's own token count. The same applies to
  the reference condition — a 1,024-token baseline is a harness default, not a
  fact about any model, and it is now clamped for models that cannot hold it
  rather than reporting our default as their limit — which also means the
  length ladder now always has somewhere to climb, since the reference prompt
  can no longer fill the whole context. The remaining constants are bounds on
  run cost, and are documented as such.

- **CI pins the pytest-xdist worker count instead of using `-n auto`.**
  Investigating why jobs were queueing 400-440 s to do 3-85 s of work
  (vllm-warden#246) turned up two things, neither of them the suspected
  "someone reset `concurrent` to 2". Both `bigdisk` runners are configured as
  documented, `concurrent = 4` on 12 cores. But `concurrent` is a per-**host**
  cap shared by every runner registered on it, and runner01 has four
  registrations — one of which serves a different project entirely — so our
  jobs compete for those slots with someone else's builds. And `-n auto` is
  one worker per CPU, so a single `unit-tests` job claimed all 12; four
  concurrent jobs would have put 48 pytest workers on 12 cores. Raising
  `concurrent` without fixing that would have made contention worse rather
  than better. CI now runs `-n ${VW_XDIST_WORKERS:-4}`, leaving room for the
  rest of the DAG, which since the stage-to-DAG change all becomes runnable at
  once. `make test-unit` keeps `-n auto` locally, where the suite is the only
  thing on the machine.

### Added

- **Chat: `reasoning_effort` is a per-chat setting.** (#241) Current Qwen chat
  templates take `reasoning_effort` next to `enable_thinking` and default to
  the most expensive level (`xhigh`), and nothing could set it. `SettingsPatch`
  now accepts `reasoning_effort` (a bounded string; `""` returns to the engine
  default) and the turn route forwards it as
  `chat_template_kwargs.reasoning_effort` while thinking is on. The level is
  passed **verbatim in the model's own vocabulary**, never mapped: the catalog
  reads each loaded model's chat template and publishes the values its
  membership test accepts as `ModelInfo.reasoning_efforts` (Qwen3.8:
  `xhigh`/`medium`/`low`), and the turn route only ever sends a value that
  list contains — a validating template `raise_exception`s on an unknown
  level rather than ignoring it, so a level chosen for one model can never
  reach a model with a different vocabulary, and a model whose template we
  cannot read gets no key at all. The settings panel control lands in
  `@podwarden/chat-ui` and renders whatever list the backend publishes.

### Fixed

- **Chat: every chat envelope says whether its model is servable.** (#240)
  `GET /chats`, `GET /chats/{id}`, `POST /chats`, `PATCH /chats/{id}` and
  `POST /chats/{id}/fork` now carry `model_loaded: bool` — the same
  `get_model_info` check the turn route 404s on. A chat pins its model by name
  and nothing constrains that name to the loaded set, while the catalog lists
  loaded models only; the settings panel's `<select>` therefore had a value
  matching no option and the browser showed the first loaded model instead,
  indistinguishable from a real selection, until the turn failed with
  `Model "X" is not loaded.` on a model the operator could not see anywhere.
  Three chats on a client install were found in this state on 2026-09-03 and
  had to be repaired with `PATCH /api/chat2/chats/{id}` from outside the UI.
  The panel fix — the chat's own model always listed, marked unavailable when
  unloaded, with a one-click switch to a loaded model — is in
  `@podwarden/chat-ui` 0.1.8, which this change pins (superseding 0.1.7,
  whose control hard-coded Qwen3.8's level names for every reasoning model).

## [v2026.09.03.8] - 2026-09-03

### Added

- **Stress test: the running modal now says what it is doing, and when it will
  be done.** A run in flight showed "0 probe(s)", "No probes recorded yet", and
  a bar rendered FULL — which reads as *finished*, the worst of the three
  possible wrong answers. Observations are only written when a run ends, and a
  thorough run takes hours. Each phase now reports its name, what it is
  probing, how far through it is, and an ETA, persisted by the heartbeat that
  already ticks for the lease. The ETA is that phase's own measured pace
  applied to its own remainder, never extrapolated across phases — a length
  rung is one HTTP probe while a sweep candidate is an engine reload of up to
  ten minutes — and a phase that cannot bound its remainder reports no ETA
  rather than a guess. The indeterminate bar is now a narrow pulsing segment
  instead of a full one.

- **Stress test: "Wipe and run again", beside a finished run's results.** The
  published record deliberately falls back past a run that measured nothing, so
  a failed run cannot withdraw a good older measurement. An operator who has
  read the results and decided to replace them is the opposite case, and only
  they can tell the two apart — so it is offered, never taken automatically. It
  returns to the confirm screen rather than starting immediately: having read a
  result does not imply acknowledging that the next run can take the engine
  down.

## [v2026.09.03.7] - 2026-09-03

### Added

- **An installer that ships in the repository.** Until now the only documented
  install path was a script rendered on demand by the PodWarden Hub from a
  catalog row, so an Apache-2.0 product could not be installed without the
  vendor's server, air-gapped installs were impossible, and the repository and
  the real install drifted silently -- on 2026-09-03 the shipped compose
  reserved every GPU on `api` while the Hub-rendered override gave GPUs to
  `caddy`, and because Compose appends sequences the merged stack asked for 3
  GPUs on a 2-GPU host. `git clone && ./install.sh` now produces a running
  stack with no access to podwarden.com. `install.sh` (POSIX sh, works piped
  from curl, every prompt on /dev/tty; `--dir`, `--gpus`, `--origin`, `--port`,
  `--version`, `--no-generate-secrets`, `--no-pull`, `--yes`, `--check`,
  `GPU_TOOLKIT_INSTALL=yes|no`) preflights Docker, Compose 2.24+ and -- the
  check that broke the client install -- whether `docker info` lists the
  `nvidia` runtime, not merely whether `nvidia-smi` works; it offers to install
  the NVIDIA Container Toolkit, lets you pick GPUs, generates the secrets, and
  writes `.env` plus a `docker-compose.override.yml` whose device list carries
  the `!override` tag so it replaces the base reservation instead of appending
  to it. `.env.example` is committed and is the single source of truth for the
  env contract; `make/operator.mk` (included by the root Makefile, shipped
  verbatim as an install's Makefile) adds `start` / `stop` / `restart` /
  `logs` / `status` / `pull` / `config` / `preflight` / `uninstall` and the
  air-gapped transport targets `save-images` / `load-images` /
  `export-hf-cache` / `import-hf-cache`; the README documents the offline path
  (pin `VERSION`, `docker save`/`load` all three images, pre-seed the HF cache
  volume, `HF_HUB_OFFLINE=1`). CI now runs `scripts/check-env-contract.sh`
  (every `${VAR}` in compose or the installer is described in `.env.example`,
  every knob is consumed, defaults agree with `app/config.py`), a stubbed
  self-test of the installer under dash and busybox, an allow-failure drift
  check against the env contract the Hub currently renders, and
  `installer:stack-up`, which runs the real installer against the repo's own
  compose on a bigdisk runner and asserts every service reaches healthy
  (#242).

### Fixed

- **Stress test: a longer run is no longer refused because a shorter one just
  finished.** Asking for a longer mode straight after a `conservative` run
  returned `200 reused` naming that run. The cooldown asked "is there a recent
  measurement for this fingerprint?" and never "does it answer THIS request?".
  The modes are not interchangeable: `thorough` confirms 7/7 rather than 5/5,
  probes three needle offsets rather than one, and is the only mode that runs
  the configuration sweep — so it is the only mode that can produce a
  `recommended_config` at all. A cached run is now reused only when its mode is
  at least as demanding as the one requested.

- **Stress test: the published context limit was our own token budget again.**
  The first successful run on `gpt-oss-20b-gguf` confirmed 1,616 tokens and
  published 1,454 as `measured`, `confirmed` — for a model that had filled
  128k earlier the same day. Every failure from 1,664 upward carried
  `abrupt + reasoning_only + wrong`, the starvation signature: the needle
  probe's budget settled at 96 tokens against the 1k baseline, and a reasoning
  model's chain grows with the input, so longer rungs starved. Escalation ran
  once, at the baseline, and the ladder ignored `starved` thereafter. The
  ladder and the limit suite now re-calibrate on that signature instead of
  recording it as a gate failure. Budgets only ever grow and are carried
  forward, so a later rung is never judged under a smaller budget than an
  earlier one.

## [v2026.09.03.6] - 2026-09-03

### Changed
- **CI: one pipeline per push, a DAG instead of stages, and dependencies baked
  into a prebuilt image.** Measured over 23 pipelines / 181 jobs, most of the
  wall clock was not build or test work. Every push ran **two** full pipelines
  — a branch pipeline and an MR pipeline on the same SHA (verified pairs
  20317/20316, 20307/20306, 20301/20300, 20295/20294) — so half of all runner
  load was duplicate; a `workflow:` block now suppresses the redundant push
  pipeline when the branch already has an open MR. `npm ci` stalled 5-7 minutes
  in 12 of 21 frontend jobs, while the one job using a glibc image with
  `--no-audit --no-fund` stalled 0 of 21 (same install: `added 690 packages in
  7m` on alpine vs `in 10s` on slim), so the frontend jobs moved to
  `node:20-slim`. Gate-rejected MRs still ran their whole job set — 1245-1685
  runner-s each after the gate failed in 4-15 s — so every job now `needs:` the
  gate. `cleanup-stale-caches` did 1-5 s of work but queued 627 s median /
  1737 s max in stage 1, gating everything behind it; it is deleted, because
  `get_sources` already removes those directories at checkout. Finally,
  `pip install` ran 22-55 s in each of four jobs: `scripts/ci-deps-image.sh`
  resolves an image tagged with a hash of both requirements files, built only
  when they change, and falls back to `pip install` when the tag is absent so a
  cold cache degrades rather than fails. The same resolver backs the Makefile,
  so local `make` targets stop reinstalling from PyPI on every invocation.

### Fixed
- **Stress test: three defects found by a code review of the day's fixes.**
  `current_for` fetched only the newest run before asking whether it had
  measured anything, so a newer valueless run hid a valid older measurement —
  which was then republished as `provenance: stale` with "the hardware
  changed", when nothing had. `has_measurement` counted an axis the runner
  explicitly withholds (`invalid_reason: prefix_cache_not_defeated` sits beside
  a positive `raw_confirmed`). And the stress lease guarded
  `restore_after_warden_restart` and `restart_crashed_models` but not
  `check_once` — the path a stress run actually provokes, since a wedged engine
  leaves the row reading `loaded` and `wants_restart` is never consulted.

- **Stress test: a run that measured nothing no longer blocks the retest.**
  `current_for`'s docstring calls its result "the newest publishable
  measurement" and lists four conditions — none of which asked whether the run
  measured anything. A run that completed having confirmed no value satisfied
  all four and held the six-hour cooldown, so pressing **Stress test** returned
  `200 reused` naming the dead run: no new run, no error, and no way out,
  because the UI reveals its `force` control only on a `409`. Completing and
  measuring are now distinct: a run must carry a recommendation or a positive
  `raw_confirmed` on some axis to hold the slot. A real measurement is still
  reused, so the cooldown continues to protect hours of GPU time.

- **Stress test: a probe that fails at the baseline no longer vetoes the whole
  run.** With the budget fix in, `gpt-oss-20b-gguf` calibrated a clean baseline
  and still confirmed nothing: `needle` passed at 1,024, 960, 896 and 8,192
  while `summary` tripped `repeating` at **every** length, the baseline
  included, so every rung failed. Asked the same question directly the model
  answered in two clean sentences — but the argument does not rest on that. A
  gate that reads the same at the reference length as at the axis maximum
  cannot locate a threshold on that axis; it only hides one. Probes that trip a
  gate at the baseline are now excluded from the ladder and reported as
  `excluded_probes`, so "summary was dropped because it tripped repeating at
  the baseline" reaches the operator instead of a silent zero. When every
  ladder probe is excluded the run aborts rather than publishing the axis
  maximum on no evidence. `gates.py` already refused to judge against a
  degenerate baseline for the SHORT gate; this applies the same rule to the
  rest.

- **Stress test: the runner's bookkeeping no longer poses as a measurement.**
  `neighbours` and `baseline` were rendered as MEASURED LIMITS cards, each with
  a "measured" badge and a value of "—". They are the runner's own notes. A
  badge that appears on scratch notes is worth nothing on a real number.

- **The unit suite runs in 31 s instead of 388 s, and three real bugs it was
  hiding are fixed.** Same assertions, same counts (2033 passed, 2 skipped
  before and after), zero production changes. bcrypt at the production cost
  factor was ~65% of the runtime on its own — ~640 `hashpw`/`checkpw` calls at
  ~370 ms each, because every `seed_admin_user` hashes and every `jwt_login`
  verifies — so an autouse fixture lowers the *test* cost factor to bcrypt's
  minimum; the hash format, the verify path and production defaults are
  untouched. Schema migrations now run once into a session template that each
  test copies, rather than 28 fsync'd transactions per test. What the speed
  exposed matters more than the speed: unit tests were reaching
  **huggingface.co** for `AutoTokenizer.from_pretrained`, now forced offline
  into the character-estimate fallback the cache already implements; the
  watchdog's first tick was racing the proxy tests' patched
  `httpx.AsyncClient.send` and becoming `calls[0]` — invisible while bcrypt was
  slow, then failing 4 runs in 12 once it was fast; and two tests calling
  `importlib.reload()` in `finally` while `monkeypatch.setenv` was still live
  were baking `VW_ENGINE_NETWORK=""` into the module for every later test,
  which passed serially only because the victim happens to run before the
  culprit.

## [v2026.09.03.5] - 2026-09-03

### Fixed

- **Stress test: a reasoning model no longer fails every probe.** The first
  production run failed all six probes at the baseline on
  `gpt-oss-20b-gguf` — `["abrupt", "reasoning_only", "wrong"]` on every one —
  and correctly published nothing. The model was healthy throughout. The suite
  budgets `max_tokens` at ~4x each probe's ANSWER (16 for arithmetic), which is
  sound for a model that answers directly and impossible for one that reasons
  first: measured against the live engine, the same prompt returned
  `content=''` with a truncated chain at `max_tokens=16` and `'42'` at 64, and
  gpt-oss needs 56 completion tokens to add two numbers. Every gate then fired
  truthfully about an artefact of our own configuration. The baseline now
  detects the one signature that can only mean a starved budget — no content,
  reasoning present, `finish_reason: "length"` — and escalates 4x up to 1024
  tokens until the model can answer, once, carrying the settled budget through
  the sweep and into the concurrency probe. This is measured rather than read
  from `models.supports_reasoning`, which is tri-state and was NULL for the
  very model that exposed the bug while being 1 for another. A model that
  answers within its budget is unaffected, and `REASONING_ONLY` still fires on
  a model that genuinely reasons without answering.

- **A run that cannot establish a baseline now says so.** Previously it
  completed with `invalid_reason: no_confirmed_value`, which reads as a verdict
  on the model rather than on the harness — after spending the whole sweep to
  reach it. It now aborts immediately with `aborted_no_baseline`.

- **Stress test: a completed run now shows what it measured.** Every finished
  run rendered *"No recommendation. Nothing met the bar for publication."* —
  successful ones included. `GET /capabilities` returns `limits` and
  `recommended_config` at the top level, while `runs[]` carries only
  `_run_summary`, which omits both; the result screen reads them off the run
  object, so they were always `undefined`. The modal now attaches the published
  record to the run it came from. The attribution is guarded on a new
  `record_run_id`, because the record is keyed on the FINGERPRINT rather than
  on the latest run: after a run that published nothing, an earlier run's
  numbers are still standing, and presenting those as the finished run's own
  findings would be exactly the confidently-wrong output this feature exists to
  prevent.

## [v2026.09.03.4] - 2026-09-03

### Added

- **Stress test: measure what a model can actually do here.** A model's real
  limits depend on the card it landed on, not on its model card, and until now
  the only way to find them was to raise a setting and wait for something to
  break. A **Stress test** button on the model page runs a sweep and answers
  the operator's actual question — *what should `max_model_len` be for this
  model on this hardware?* Failure is defined by output, not by exit code: a
  run fails when the model goes empty, abrupt, repeating, short against its own
  baseline, reasoning-only, or wrong on a graded probe, because a model
  degrades long before an engine dies — and an engine still dies when a run
  induces OOM, which is measured too. Every published number is measured; a
  bound that was never confirmed is not published at all, and a measurement
  taken on different hardware is flagged stale rather than reused. Results
  reach API clients as an additive `warden` block on `GET /v1/models`, which
  carries the measured limits for the configuration the engine is actually
  running and never a recommendation for one it is not. The watchdog stands
  down for a model under a live stress lease, so a run that crashes an engine
  on purpose no longer races the recovery loop for the restart budget.

### Fixed

- **`VW_WARMUP_PROBE_TIMEOUT_S` default raised again, `300.0` → `600.0`.**
  The `60 → 300` bump in v2026.09.03.3 removed the worst of the cliff but not
  the failure mode. Deploying that release to the same two-GPU client host
  recreated the `api` container; the watchdog auto-restored
  `nvidia/Qwen3.6-35B-A3B-NVFP4` (TP=2, NVFP4, multimodal) and it took roughly
  **8 minutes** to reach `loaded` — the third such measurement on that box,
  after 4m17s and 8.5min for two different models. The install only survived
  because `.env` carried a hand-set `600.0`. The cost of this value is
  asymmetric: a subprocess that dies is reported at once by the `on_exit`
  callback in `start_engine` no matter what the timeout is, and a probe that
  cannot connect returns immediately — so a larger budget only extends how
  long a *live* engine is allowed to finish starting, while too small a budget
  marks a healthy engine `failed` while it keeps running and holding the GPUs.
  A fixed budget is still a guess about hardware we do not control; starting
  the probe clock only once `/health` passes is tracked as #235 part 2.

## [v2026.09.03.3] - 2026-09-03

### Fixed
- **#235 — `VW_WARMUP_PROBE_TIMEOUT_S` default raised `60.0` → `300.0`.**
  Measured live on a client install (two RTX 5060 Ti, vLLM 0.26.0, TP=2):
  `unsloth/Qwen3.8-27B-NVFP4` reported `init engine (profile, create kv
  cache, warmup model) took 123.13 s` and did not accept connections until
  roughly 4m17s after spawn; `nvidia/Qwen3.6-35B-A3B-NVFP4` took about 8.5
  minutes to reach `loaded`. Both are structural, not pathological: NVFP4/FP8
  kernel autotuning (FlashInfer autotunes `fp4_gemm`/`fp8_gemm` on first
  start), multimodal vision-tower profiling, tensor-parallel NCCL/shm
  startup, and hybrid-attention page-size reconciliation all blow through a
  60s budget on their own. A model that loaded and served perfectly was
  marked `failed` while its engine kept running and holding both GPUs. This
  change only raises the default (`app/config.py`, `docs/operating.md` env
  table); the probe still counts startup time against its budget rather
  than starting the clock once `/health` passes — tracked separately as
  #235 part 2.

- **`GET /api/models` and `GET /api/models/{id}` no longer drop engine
  settings and capability flags.** (#237) On a two-GPU install, `PATCH
  /api/models/{id}/settings {"supports_reasoning": true}` returned `{"ok":
  true}`, the DB really updated, and neither read endpoint showed it —
  confirmable only by opening SQLite directly. The detail endpoint omitted
  `supports_tools` / `supports_vision` / `supports_reasoning` entirely (not
  `null` — absent), and the list endpoint additionally dropped
  `max_model_len`, `gpu_memory_utilization` and `extra_args`, so a list view
  had no way to show engine sizing at all. Both routes now serialise through
  one function in `app/models/serialisation.py` (`model_detail` and
  `model_summary` are thin wrappers around it), so the two responses are
  field-for-field identical and a future migration cannot add a column that
  reaches one route but not the other. The three capability columns are
  tri-state in the DB (`1` = operator said yes, `0` = operator said no,
  `NULL` = nobody has said, and `app/chat2/catalog.py` auto-detects vision
  only while the column is `NULL`), so they serialise as `true` / `false` /
  `null` rather than being coerced to a boolean, which would have silently
  turned "nobody has said" into "operator said no".

- **Load-failure diagnosis no longer reports the PREVIOUS run's fault.** The
  engine log (`{logs_dir}/{model_id}.log`) is appended across every load
  attempt, and the diagnoser was handed its last 200 lines — a window that
  straddles run boundaries. An earlier attempt's recognised traceback was
  matched and reported as the current attempt's cause, *replacing* the accurate
  generic message with a confident, specific, wrong one. Seen twice live on a
  two-GPU client install: a run that got all the way through profiling
  (`Available KV cache memory: 2.31 GiB`, `init engine … took 123.13 s`) was
  reported as "GPU ran out of memory loading the model", and a run at
  `max_model_len=32768` that reached a 269,633-token KV cache was reported with
  the previous run's "wants 262144 tokens" verbatim — advice that would have the
  operator lower `max_model_len` a second time for nothing. Every engine driver
  now stamps a run-boundary sentinel
  (`===== vllm-warden run <uuid> started <iso-ts> =====`) into the log when it
  opens it for a spawn, and the diagnoser reads only from the last sentinel on
  (still capped at 200 lines). The full history stays in the file — comparing
  attempt N-1 with attempt N is exactly why we delimit rather than truncate —
  and logs written before sentinels existed fall back to the old whole-tail
  read, so no install loses diagnosis on upgrade. (#234)

- **A model can no longer be stranded in `loading` forever.** Restarting the
  `api` container mid-load left the row at `status='loading'` with no engine
  anywhere, and every route out was closed: `load` refuses from `loading`,
  and so did `unload` — `?force=true` included. The only escape was editing
  SQLite by hand, which an operator without a shell on the container does not
  have (observed on a client install, 2026-09-03). Two fixes, either of which
  removes the dead end:
  - Boot reconciliation (`app/runtime/boot_reconcile.py`) now demotes every
    row left in a transient status (`loading`, `unloading`, `pulling`) whose
    engine the supervisor does not hold: to `pulled` when the weights are
    complete in the HF cache, `registered` when they are not, with
    `last_error` saying the status was recovered after a restart. It runs
    before `mark_runtime_dead_on_startup`, and deliberately leaves both
    `loaded` rows and a watchdog restore in flight to that call, which
    records the `prior_status` the automatic restore keys on.
  - `POST /api/models/{id}/unload?force=true` now succeeds from `loading` and
    `unloading`: it tears down whatever the supervisor still holds (nothing,
    after a restart) and writes a terminal status unconditionally. Plain
    `unload` keeps its guard, and force still refuses from `pulling` — no
    engine is involved in a download, and this route's terminal status would
    be a lie for a half-finished pull.

- **#238 — the HF cache reported roughly twice the disk it was using.** On a
  client install `GET /api/cache/models` claimed 46.9 GB for a repo that `du`
  measured at 22 GiB, and listed every repo twice; the reported total was
  52.2 GB against a real 28.6 GB. Two mechanisms, both reproduced in a fixture
  tree before fixing. The scan deduplicated by resolved directory path, so a
  repo split across `<cache>/models--org--name` (where the weights live) and
  `<cache>/hub/models--org--name` (HF's own layout) became two rows. And the
  size walk used `os.walk(followlinks=False)` — correct — but then `os.stat`,
  which *does* follow symlinks, on every name it yielded: HF's
  `snapshots/<rev>/file` is a symlink into `blobs/<sha>`, so each blob was
  counted once as itself and again as its symlink. Directories are now grouped
  by decoded repo id and every file counted once, keyed on `(st_dev, st_ino)`.
  This is the number an operator sizes a disk against and the one the reclaim
  view shows before a delete.

- **#239 — the chat's "Enable thinking" toggle was invisible on every new
  model.** `@podwarden/chat-ui` renders the control only when the catalog says
  `supports_reasoning`, and that column defaults to `NULL`, which
  `app/chat2/catalog.py` collapsed to `False`. Unlike `supports_vision` it had
  no auto-detection, so the toggle stayed hidden until an operator discovered —
  unprompted — that they had to `PATCH` the flag by hand. Everything behind it
  already worked. While the column is `NULL` the flag is now inferred from the
  model's chat template, which is decisive for this family: a template that
  branches on `enable_thinking` belongs to a model that reasons. Detection
  reads the standalone `chat_template.jinja` first — newer repos ship it as its
  own file and leave `tokenizer_config.json` without the key — and falls back
  to the tokenizer config. An explicit operator `1`/`0` still wins, and a
  missing or unreadable cache degrades to `False` rather than raising on the
  chat's hot path.

## [v2026.09.03.2] - 2026-09-03

### Added
- **Chat: LaTeX-style math renders.** Models write display math as `\[ ... \]`
  and inline math as `\( ... \)`; those reached the screen as plain bracketed
  text. Bump `@podwarden/chat-ui` to 0.1.6: balanced LaTeX delimiters are
  rewritten to the `$`-forms before parsing and render through the existing
  sanitized KaTeX pipeline. Code fences and inline code are never touched, and
  a half-streamed `\[` stays literal until its `\]` arrives.

### Fixed

- **The fit preview knows which engine the verdict is about.** An operator
  picking `gpt-oss-20b-Q8_0.gguf` (11.28 GiB) for **llama.cpp** on a 16 GiB
  card was told "won't fit". Three errors compounded to produce it, and each
  is fixed:
  - `POST /api/models/fit-preview` had no `backend` field at all, so it
    always applied vLLM's `gpu_memory_utilization` — 0.9, hiding 10% of the
    card. llama-server has no such flag (`--help` in the shipped b10731 image
    offers only `-ngl`, a layer count), so the operator's number corresponds
    to nothing. Backends now declare `vram_cap_fraction`: `None` for vLLM,
    meaning the operator's flag is a real cap and is honoured; `1.0` for
    llama.cpp, meaning no engine-imposed fraction. The headroom judgement
    stays where it was — the verdict ladder already calls 0.80–1.0 "tight" —
    rather than being applied twice via an invented safety margin. Declared
    on `BackendCapabilities` rather than branched on the backend name, for
    the reason `needs_local_model_path` already documents.
  - `head_dim` was always derived as `hidden_size // num_attention_heads`.
    gpt-oss-20b declares `head_dim: 64`; the derivation gives `2880 // 64 =
    45`, a 30% undercount of the entire KV term — the direction
    `dtype_bytes_from_torch_dtype` already names as dangerous, because it
    turns a red row green.
  - Every layer was charged the full context. gpt-oss-20b runs interleaved
    sliding-window attention: 12 of its 24 layers hold a 128-token window,
    not 131072 tokens. Counted only from an explicit `config.layer_types`,
    so a model that implies sliding attention some other way keeps today's
    over-counting estimate rather than inviting a per-family guess that could
    silently under-count.
  Net for the reported case: KV 4.22 GiB → 3.00 GiB, and the verdict goes
  from **"won't fit"** to **"tight"** (ratio 0.87) on llama.cpp. It is
  genuinely tight — 11.28 GiB of weights on a 16 GiB card at full 131k
  context — so "tight" is the honest answer, not "fits".
  `recommend_max_model_len` moves in lockstep; it is the inverse of the
  reserve, and a recommendation computed from a different KV shape would
  paste in a context that produces a different verdict. The discovery config
  projection widens by `head_dim`, `sliding_window` and `layer_types` —
  the route reads `discovery["config"]`, so a key absent there is a key the
  math cannot see. `GET /api/system/backends` now returns
  `vram_cap_fraction` so the dialog's client-side budget recompute uses the
  same cap the server did, and switching engine drops the cached verdicts
  instead of showing the previous engine's answer.

## [v2026.09.03.1] - 2026-09-03

### Fixed

- **The GGUF architecture warning no longer fires on llama.cpp, and its link
  no longer 404s.** `gguf_arch_unsupported` is measured against
  `KNOWN_GGUF_ARCHES`, which is *vLLM's* GGUF-loader allowlist. It was
  rendered with no backend gate, so picking a GGUF for llama.cpp produced
  "Architecture … is not in the vLLM-known GGUF allowlist — load is likely to
  fail" about the one file format llama.cpp is built around. Reported against
  a live install serving `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` on llama.cpp
  while sitting outside that allowlist. The gate goes on
  `warningsByFilename`, the single path a warning takes to either render
  site, so neither `ShardFamilyRows` nor `FileRow` needs a new prop.
  The banner also linked to `/docs/operating.md#supported-gguf-architectures`
  — a repo-relative path used as a browser href. Nothing has ever served
  `/docs`: the Caddyfile has no `handle /docs*` and the Dockerfile never
  copies `docs/` into the image, so it 404'd in every deployment. It was the
  only `/docs/` href in the frontend, so rather than add a docs subsystem for
  one link the banner now carries the remedy directly — switch the engine to
  llama.cpp. `docs/operating.md` says the same, having been written when vLLM
  was the only backend.
  The two #101 tests covering this banner were asserting the defect: both
  pick a `.gguf`, `backendForFilename` maps that to llama.cpp, and they then
  required the vLLM banner to be present. They now select vLLM first.

## [v2026.09.02.3] - 2026-09-02

### Changed

- **The product is now called LLM Warden.** Two engines ship, so a name that
  claims one of them was wrong: the nav wordmark, the landing page title and
  copy, the login and setup-wizard headings, the browser tab titles, the
  FastAPI/OpenAPI title, the engine-pin error message, the Hub catalogue's
  display name and stack label, and the READMEs all read "LLM Warden". Prose
  that framed the product as *being* vLLM was corrected with it — the README's
  opening paragraph and the landing hero now say the warden wraps an engine
  rather than that it wraps vLLM, and `docs/backends.md` no longer claims vLLM
  is the only backend.
  This is a **display-only** rename, deliberately: nothing an identifier
  depends on moved. The `vllm-warden` catalogue slug and `app_family`, the
  `vw_`/`VW_` token, cookie and env prefixes, the `pw-vllm-warden` namespace,
  `/data/vllm-warden.db`, image and container names, the `vllm_warden` module
  path and the repository itself are all unchanged, so no deployment, token,
  bookmark or install script breaks. Bare "vLLM" is also untouched wherever it
  means the *inference engine* — "vLLM version", `vllm/vllm-openai`, live vLLM
  logs — because that text is correct and renaming it would make the product
  lie about what it is running.
  `tests/unit/test_product_name.py` is the guard: it scans every user-visible
  surface for the two-word name with a whitespace separator, and asserts that
  the bare engine name and the hyphenated identity slug still pass.
- **Choose which models the stats cover.** `/stats`, `/stats/live` and
  `/godmode` share one model selection — all models, one, or any combination —
  and at least one always stays selected: the last checkbox is *disabled*
  rather than silently re-ticking, because a click that appears to do nothing
  leaves the operator unable to tell whether it registered. `GET
  /api/stats/v2/overview` gained `?models=`; absent still means the whole
  deployment, an *empty* selection is a 400 (it could mean "all" or "none", and
  guessing turns a client bug into a wrong number), and an unknown id is a 400
  rather than a silently narrower answer. What "narrows" means differs by
  series and the page says so: tokens are per model, while VRAM, GPU
  utilisation and power have no model dimension anywhere and resolve to the
  cards the selection occupies.
- **llama.cpp's real knobs are controls, not free text.** Flash attention and
  the two KV cache types (`--flash-attn`, `--cache-type-k`, `--cache-type-v`)
  had no first-class field, so the only way to set them was to hand-type a flag
  into Extra args. They are now selects on the model settings page, backed by
  `extra_args` rather than by new columns — which is where a hand-typed flag
  already goes, so no migration, no schema growth per flag, and precedence
  unchanged. A managed flag has exactly one home: it is read out of Extra args
  into its control, so setting one can never duplicate or contradict the same
  flag typed by hand. Leaving a control on "default" omits the flag, so the
  launch command of an untouched row is byte-identical to before. Values were
  read off `llama-server --help` in the shipped image, not from upstream docs:
  `--flash-attn` takes `on|off|auto` in this build rather than being a bare
  switch. `--split-mode`, `--main-gpu` and `--tensor-split` deliberately stay
  derived (decision D4) and the section says so instead of offering a control
  that would compete with the GPU selection.
- **The header named one model while two were serving.** `_active_model` ended
  in `LIMIT 1` under a comment asserting the supervisor enforces single-model
  loading, so the second engine never reached the frontend at all. The header
  metrics frame gained `active_models` and the chip renders one entry per
  model, each with its own dot; two inline and the rest folded into a `+N`
  counter so its width stays bounded at any fleet size, with every model named
  in the tooltip and the accessible label. The cluster's accent colour now
  summarises the *worst* status, so three healthy engines cannot paint over a
  fourth that crashed. The two aggregates were checked and left as they were —
  pooled VRAM and busiest-card GPU utilisation are both whole-box questions —
  but the GPU readout now says "on the busiest card", which with one card it
  never had to.
- **The live stats view showed one engine on a two-engine box**, and which one
  was whatever SQLite returned first. `/api/stats/live` now carries a block per
  loaded model, with rate state tracked per model — two engines' cumulative
  counters share no origin, and one snapshot would have produced a rate for the
  second model computed against the first model's totals. The per-model block is
  byte-identical to before, so the eleven panels reading it field by field are
  untouched. Where several models' figures ARE combined, an unreported metric is
  skipped rather than counted as zero, and the tile says "1 of 2 models report
  this"; a metric no selected engine reports renders an em dash, never `0`.
  Latency percentiles and KV fractions are deliberately not combined at all.
  llama.cpp model.** There is no llama.cpp image resolver, so honouring such a
  pin would have resolved a *vLLM* image and launched vLLM under the operator's
  model name. The card is now absent for a backend with no image catalogue, and
  visible-but-disabled where only the *driver* is the obstacle — hide what an
  engine has no concept of, disable and explain what this deployment merely
  cannot do. `GET /api/system/backends` gained `version_pin_reason_code`
  (`engine` / `driver` / `backend`) so a client can branch without
  string-matching a human sentence, and the panel now renders the server's
  sentence verbatim instead of hardcoding one about the in-container driver.
- `version_pin_reason` blamed the *driver* for llama.cpp on the deployment we
  run. Both obstacles apply there, and the driver was checked first — a true
  sentence about vLLM and a false promise about llama.cpp, whose pin no driver
  can make available. An operator acting on it would have migrated a deployment
  to unlock a control that would still be dead.
- **"Why is Extra args empty but Effective argv has so many arguments?"** The
  distinction was correct and entirely invisible. Extra args is now labelled as
  the operator's own additions and the Effective argv panel says it is the whole
  generated command.

## [v2026.09.02.2] - 2026-09-02

### Added

- The Add Model wizard names the model already serving from a GPU you pick, and
  stops offering GPUs outside the deployment's `allowed_gpu_indices`. An
  occupied GPU stays selectable — co-locating two small models on one card is a
  legitimate thing to do on purpose; the warning is there so it is not done by
  accident. `GET /api/system/gpus` gained `allowed_indices` to make the second
  half possible: the allowlist was enforced with a 400 on create but published
  nowhere, so the wizard learned about it only by being refused.
- The Add Model wizard offers an engine version, disabled where the deployment
  cannot honour a pin and carrying the server's explanation of why. `GET
  /api/system/backends` gained `version_pin_reason`, non-null exactly when
  `version_pin_available` is false — the frontend sees one boolean and cannot
  tell whether the obstacle is the driver (fixable: run the docker engine
  driver) or the backend (llama.cpp has no image catalogue to pin against).

### Fixed

- **Every model reported `Engine: vllm`**, including one demonstrably served by
  `llama-server`. The data was never wrong: both model read paths built their
  responses as dict literals, so the columns migrations 0027 and 0028 added
  (`backend`, `mmproj_filename`, `n_gpu_layers`) had to be remembered at a
  second site and were not. `GET /api/models` and `GET /api/models/{id}` now
  serialise through one place that starts from the whole row and subtracts, so
  the next migration cannot silently drop a column — a test fails first.
- The model detail page applied no per-backend field visibility, so a llama.cpp
  model advertised a `Tensor parallel size` and a `gpu_memory_utilization` it
  has no concept of, and printed a blank `n_gpu_layers`. The table that already
  did this on the model settings page is now shared by both pages.
- The Try stack panel stated two engine versions at once: its capability note
  read the real one while the version field's placeholder was a hardcoded
  `0.20.0` left over from the built-in template default — and uncorrectable,
  since the field is disabled on the shipped driver.
- GPU **indices** were labelled "GPUs: 1", which reads as a count. Now "GPU
  index: 1" / "GPU indices: 0, 1", on both the detail page and the model card.

## [v2026.09.02.1] - 2026-09-02

### Added

- **llama.cpp is available as a second inference backend.** `llama-server` is
  baked into the warden image and launched by the existing subprocess driver, so
  no new infrastructure is required. Pick it per model in the Add Model wizard;
  selecting a `.gguf` file pre-selects it, and the choice is always overridable —
  a GGUF that vLLM can also serve stays available to vLLM for throughput. The
  persisted default is unchanged: existing models, and new models that do not
  choose, stay on vLLM.

  It is **not** a superset of vLLM. llama.cpp's multi-GPU mode is a layer split,
  not tensor parallelism; for a model too large for one card, vLLM is still the
  right answer. See `docs/backends.md`.

- GGUF models can now carry a multimodal projector (`mmproj_filename`), pulled
  alongside the weights and passed to llama.cpp as `--mmproj`, so vision GGUFs
  work. A vision model started without its projector loads, serves, and silently
  ignores every image, so a set-but-missing projector is refused before any
  process starts.
- `n_gpu_layers` allows deliberate partial CPU offload for a model that does not
  fit the card. It is slow, and it is never chosen for you: blank means
  llama.cpp sizes its own offload.
- `GET /api/system/backends` reports each backend's own version. It previously
  reported vLLM's for everything, which becomes a lie with two backends.
- The live-stats SSE frame carries a `backend` field, so the dashboard can say
  *which* engine does not report a metric rather than showing a blank tile of
  unknown provenance.

### Changed

- Live-stats metric parsing is **per backend**. The frame's shape is unchanged;
  a metric the running engine does not publish is now reported as absent and
  rendered as such, rather than as zero. Three panels previously showed `0` for a
  metric that was simply not measured: generation/prompt tokens-per-second fell
  back to `0` in the headline number *and* pushed a flat line into the sparkline;
  KV-cache usage rendered `0%` against an empty meter — "plenty of headroom" when
  the truth is "not measured"; and the sleep-state chip printed the literal
  string `sleep null`.
- Model settings hides, rather than disables, fields belonging to a backend other
  than the row's. A greyed-out box for a knob the engine has never heard of is an
  invitation to wonder what it does.
- `extra_env` on a llama.cpp model accepts `GGML_*` only. `LLAMA_*` is refused:
  every llama.cpp flag has an environment twin, so allowing the prefix would let
  a model row re-bind the engine's unauthenticated API, swap its weights, move
  its port out from under the health probe, or switch metrics off. The five worst
  names are hard-locked globally, so they are refused loudly at write time rather
  than dropped silently.

### Fixed

- Requests to a model whose HuggingFace repo contains **only** GGUF files no
  longer fail with a 500. That repo shape ships no `tokenizer.json`, so token
  accounting raised on the proxy's hot path — before the request ever reached an
  engine that was ready to answer it. Accounting now uses `tokenizer_repo` when
  set (exact, local, free) and falls back to a character estimate rather than
  raising. The estimate is logged once per repo and reported through
  `TokenizerCache.estimating()`, because it also drives the per-token rate limit.

### Notes

- Warden image size: 9,174,592,443 B → 9,221,778,635 B — **+47.2 MB (+0.51%)** as
  docker reports it, **+85.3 MB (+0.93%)** counting the files the new layer adds.
  Build time 4m02s, up from 2m51s. Measured on `bonus` against the real base
  image, 2026-09-02.
- llama.cpp's version is fixed by the warden image (`b10731`); bumping it is a
  warden release, not a per-model choice.
- Acceptance: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` (IQ3_XXS + mmproj) serves
  on a single RTX A4000 with verdict **`supported`** — coherent multi-turn,
  correct arithmetic, exact format compliance, and a correct description of a
  known test image. Evidence in `docs/tested-stacks.md`.

### Changed (sub-project B, which the above builds on)

> The next entry says vLLM is the only backend. That was true when B landed;
> sub-project C above is the second one.

- **Backend abstraction (sub-project B).** `app/runtime/backends/` introduces a
  `Backend` axis above the existing engine driver: a driver answers *where* a
  process runs, a backend answers *which program, with what argv, env, health
  probe, log grammar and capabilities*. vLLM is the only backend and **runtime
  behaviour is unchanged** — the same argv, the same env, the same bind host,
  the same diagnoses, proven by a 25-case golden corpus captured before the
  refactor, three of whose cases are transcribed from live production rows and
  checked against the argv those installs actually produced. `cmd_builder`,
  `env_builder`, `log_diagnostics`, `templates/resolver` and
  `templates/engine_versions` moved under `app/runtime/backends/vllm/` as pure
  renames.
- `GET /api/models/{id}/effective-argv` now returns the **full** argv including
  `["vllm", "serve"]`. It previously omitted the leading pair; the field
  description said so and now describes what is actually returned.

### Added

- `GET /api/system/backends` — available backends, their advertised
  capabilities, and what the active driver lets each of them do.
  `GET /api/system/engine` is unchanged and kept as a deprecated alias.
- `models.backend` (migration 0027) — nullable, no backfill. `NULL` means
  `vllm`, so no existing row changes behaviour.
- `ModelCreate.backend`, defaulted to `"vllm"`. A client that never sends it
  behaves exactly as before.
- `PYTHONPATH` and `LD_PRELOAD` join `HARD_LOCKED_ENV_KEYS`. Neither was on any
  allowlist, so nothing an operator can do changes; the lock documents why they
  can never be added.
- `docs/backends.md` — the backend/driver matrix, how to add a backend, and why
  version pinning is reported as two separate fields.

## [v2026.09.01.1] - 2026-09-01

### Fixed
- **Chat: typing can no longer go nowhere.** Scrolling or clicking the thread
  moved focus to the scroller or `<body>`, after which keystrokes landed in no
  visible field. Bump `@podwarden/chat-ui` to 0.1.5: the composer takes focus
  on mount / chat switch / re-enable (never stealing from the sidebar filter
  or another editable field), and stray printable keys and Backspace are
  routed to the message box unless a dialog is open or a modifier is held.

## [v2026.08.31.1] - 2026-08-31

### Fixed
- **First-run setup: a browser Back out of the wizard no longer strands the
  install.** Reaching a later step and then pressing Back (or reloading)
  rendered the welcome page whose only button was rejected with `not at
  welcome step`, with no control anywhere to move forward. `POST
  /api/setup/welcome` is now idempotent past welcome (returns the real current
  step with 200), and the wizard layout syncs the URL to the server's
  canonical step on every load, so Back and reload self-heal on every step.
  Adds `frontend/src/lib/setup-steps.ts` mapping server steps to route
  segments; the login funnel now points at the current step instead of a
  hard-coded `/setup/welcome`. The forward-only state machine and its
  data-carrying guards are unchanged.
- **Auth: accept the origin this deployment is actually served on.** Behind a
  reverse proxy with `VW_FRONTEND_ORIGIN` unset, `load_settings` fell back to
  `http://localhost:3000`, so `require_matching_origin` 403'd every request
  carrying a real browser Origin — every page reload logged the operator out
  because `POST /api/auth/refresh` was rejected before it could read the
  cookie. The origin is now derived from the request
  (`derive_origin_from_request` in `app/config.py` + `app/auth/origin.py`),
  the same Host-vs-Origin argument PodWarden Core uses for its OIDC redirect
  base. ADDITIVE: the localhost default stays, because local development is
  genuinely cross-origin (frontend dev server on :3000, API on :8080).

## [v2026.08.26.1] - 2026-08-26

### Changed
- Bump `@podwarden/chat-ui` to 0.1.4 (additive: URLs inside code blocks and
  inline code are now clickable, plus prose autolink fixes; no behaviour
  change for Warden).

## [v2026.08.25.1] - 2026-08-25

### Changed
- **`/chat2` is now the shared `@podwarden/chat-ui` component (0.1.2), consumed
  over the GitLab package registry; the in-tree copy is gone and 19 private
  dependencies are no longer direct dependencies (still installed transitively
  through the package). No behaviour change.** The route is a thin host
  shell: Warden supplies auth (`authFetch`), theme, capabilities and the
  initial chat id from `?c=`; the package owns the rest. 60 source and test
  files deleted. `react-virtuoso` deliberately STAYS a direct dependency — it
  was never chat2-only, `models/log-stream.tsx` and `godmode/godmode-viewer.tsx`
  both render their rows through it. Chat-ui theme tokens are mapped onto
  retro/retro-dark.

### Added
- `GET /api/chat2/_whoami` reports `contract`; `settings.scope` is accepted
  opaque; the budget window is now an object.
- the chat-ui conformance kit runs against the backend in CI. `make
  conformance` (and the `conformance:chat2` job) boots the real app with only
  the model upstream faked and runs `@podwarden/chat-ui`'s own contract suite
  against it over HTTP; `tests/conformance/test_contract_routes.py` checks the
  same package's OpenAPI slice against our route table in the same job.

## [v2026.08.23.5] - 2026-08-24

### Changed
- **chat2 lightbox: natural zoom & pan gestures.** Plain vertical scroll
  (wheel / two-finger trackpad) zooms continuously around the cursor; holding
  the left button and moving pans in both axes. The +/- buttons still snap to
  fixed levels, and the viewport stays keyboard-scrollable.

### Fixed
- **chat2 layout: several utility classes silently generated no CSS.** The
  Tailwind content globs only scanned `src/components` and `src/app`, so any
  class used exclusively under `src/features/` (all of chat2) was purged from
  the build. Casualties: the chat root's `-m-6` (the page scrolled and the
  composer sat below the fold instead of the chat being exactly
  viewport-height), jump-to-latest's `-translate-x-1/2` (the button rendered
  on the left instead of centered), and the sidebar list's own scrollbar. The
  globs now cover `src/features` and `src/lib`, with a contract test pinning
  them.

## [v2026.08.23.4] - 2026-08-24

### Added
- **chat2 turns keep streaming server-side if you navigate away or reload;
  reopening the chat re-attaches to the live answer.** The turn now runs on a
  detached backend task registered in an in-process live-turn registry
  (`app/chat2/live.py`); the HTTP response is just a replayable subscriber, so
  a dropped connection no longer aborts generation — the full assistant row,
  usage, ledger entry and titles are always persisted. `GET /chats/{id}`
  exposes `live_turn`, `GET /chats/{id}/turn/live` replays the stream
  byte-identically and follows it, and stopping an answer is now an explicit
  `POST /chats/{id}/turn/abort` (which the Stop button and the
  send-while-streaming path call).

## [v2026.08.23.3] - 2026-08-23

### Changed
- **The old `/chat` playground page is retired.** `/chat` now permanently
  redirects to `/chat2`; the nav has a single "Chat" entry pointing there. The
  playground backend endpoints (`/api/chat/playground/ensure`,
  `/api/chat/completions`) are unchanged — only the UI is gone.

### Fixed
- **chat2: streaming no longer makes code blocks flicker.** Fenced code was
  re-highlighted asynchronously on every token, flashing an unstyled frame
  (font/colour swap) many times a second; blocks now highlight synchronously
  in render once the highlighter has loaded, so the only unstyled frame left
  is the very first code block of a session while Shiki loads.
- **chat2 sidebar: a long previous auto-title could hide the current one.**
  Title and the "(was: …)" hint now share one clamping container — up to two
  lines, title first — so clipping can only ever cut the hint's tail.
- **chat2: a just-pasted image no longer flashes “image expired”.** The
  optimistic user row referenced the attachment immediately, but the thread's
  attachment map only filled from the post-turn reload — so the image showed
  the expired placeholder for the whole stream (reopening the chat “fixed”
  it). `send()` now seeds the map from the composer's uploaded rows.
- **chat2: the attachment lightbox gained zoom.** Zoom buttons and
  ctrl/cmd+wheel step through 100–400%; when zoomed, scrolling pans the image.

## [v2026.08.23.2] — 2026-08-23

### Added
- **chat2 auto-titles are reassessed as a chat grows.** A chat used to be
  named once, from its first exchange, so a thread that wandered kept its
  opening-question title forever and the sidebar stopped being a usable index.
  The title is now re-derived on the first user turn and then every fourth one
  while the title is still `auto`, from the last six user/assistant messages
  rather than just the opening pair, asking the chat's own model for a 2–5 word
  topic title at the lowest proxy priority. Migration `0026` adds
  `chats.title_prev`, which records the title a reassessment displaced (and is
  cleared when the user renames the chat) so the rename can be surfaced and
  undone rather than silently swapping the sidebar entry. The swap is a single
  `UPDATE ... AND title_source = 'auto'`, so a user rename racing the
  (queued, lowest-priority) title call always wins.
- **Per-chat `enable_thinking` setting.** Reasoning models spend most of a
  turn's token budget on a preamble; chat2 had no way to decline. Setting it to
  `false` sends vLLM `chat_template_kwargs = {"enable_thinking": false}`. `true`
  or absent sends nothing at all, which is the only safe "on" for chat
  templates that have never heard of the flag. The internal title call now
  always disables thinking, since a 16-token budget spent on a preamble
  produced no title at all on exactly the models people most want titles for.
- **`supports_vision` is auto-detected from the on-disk HF config.** A
  genuinely multimodal model served every pasted image as `[image omitted]`
  because nothing had ever written `models.supports_vision` and the chat2
  catalog read NULL and 0 as the same `False`. The column is tri-state: 1/0 is
  an operator's explicit answer and always wins, NULL now means "ask the
  config" — a `vision_config` key or a `...ForConditionalGeneration`
  architecture counts as evidence, `hf_config_repo` is honoured when set, and
  an unreadable config still reads as `False`. Configs are cached per
  (path, mtime, size) so the 60s models poll stops re-parsing them.
  `supports_tools` / `supports_reasoning` stay manual.

### Changed
- **`PATCH /api/models/{id}/settings` documents and enforces the capability-flag
  contract.** `supports_tools` / `supports_vision` / `supports_reasoning`
  accept `true` / `false` (and their case-insensitive string spellings) or
  `null` to reset to auto-detection; an omitted key is left untouched. Anything
  else is now a 400 instead of being coerced — `"no"` used to be stored as an
  explicit *yes*. A body containing only capability flags is also exempt from
  the endpoint's unload-first 409: those columns are inert metadata, and the
  loaded model is exactly the one an operator is looking at when they notice a
  flag is wrong.


## [v2026.08.23.1] — 2026-08-23

### Added
- **chat2 page (#232).** New `/chat2` full-page chat on the chat2 backend:
  persisted per-user chats with a date-grouped sidebar (rename, fork, delete
  one/all, keyboard navigation), markdown + sanitized inline HTML with KaTeX
  and Shiki code blocks, collapsed reasoning and tool-call blocks, option
  buttons (`present_options` tool and a narrowed trailing-list heuristic),
  image attachments by drop/paste with signed URLs and expired placeholders,
  context-window / response-token / budget bars, per-chat settings with
  "save as my defaults", fork-at-message and edit-and-fork. The old `/chat`
  playground is unchanged. Renderer: `react-markdown` + Shiki, chosen over a
  spiked `streamdown` alternative — **Streamdown rejected**, it failed two of
  the spike's six portability checks under this codebase's Tailwind 3.4 /
  React 19 constraints (see
  `docs/superpowers/specs/2026-08-23-chat2-renderer-decision.md`). New
  dependencies: `react-markdown`, `shiki`, `remark-gfm`, `remark-math`,
  `rehype-katex`, `rehype-sanitize`, `rehype-raw` (added after the spike so
  sanitized inline HTML has raw markup to sanitize in the first place).
- **chat2 backend (#232).** `/api/chat2/*`: per-user persisted chats with
  linear messages and fork-to-new-chat, image attachments (PNG/JPEG/WebP,
  re-encoded, sha256-deduplicated on `/data/chat2`, per-user quota + free-space
  floor, TTL/LRU eviction that keeps rows and marks them expired), per-user
  defaults, a model catalog that resolves the context window and capability
  flags, an append-only usage ledger with rate cards, and a streaming turn
  endpoint that normalises the proxy's SSE into typed `ChatEvent`s (reasoning,
  tool calls, usage, guard/timeout outcomes) with one-turn-in-flight and
  idempotent `request_id`. Spec: `docs/superpowers/specs/2026-08-23-chat2-design.md`.
  The old `/chat` playground is unchanged.
- Per-user defaults include a `tool_policy` setting alongside model/sampling
  defaults, and the idempotent `request_id` used to dedupe a retried turn is
  stored namespaced per user, so two different users can never collide on
  the same client-generated id.
- New env knobs for chat2 attachment storage: `VW_CHAT_QUOTA_USER_BYTES`
  (default 2147483648 — 2 GiB per user across distinct `(user_id, sha256)`
  files), `VW_CHAT_QUOTA_FREE_FLOOR_BYTES` (default 5368709120 — 5 GiB of free
  space on `VW_DATA_DIR`; **below the floor every upload is refused with 413
  `quota_exceeded`, by design**, so the warden can never fill the disk it also
  writes its DB to) and `VW_CHAT_ATTACHMENT_TTL_DAYS` (default 90 — the
  collector unlinks attachment files whose newest referencing row is older
  than this, keeping the rows so chats stay readable).
- New runtime dependency: **Pillow 11.0.0** (`requirements.txt`), used to
  sniff, size-check and re-encode uploaded images so EXIF, ancillary chunks
  and polyglot payloads never reach disk.

## [v2026.08.22.2] — 2026-08-22

### Security
- **Engine processes no longer bind `0.0.0.0` under the local driver (#211).**
  `_engine_bind_host()` returned `0.0.0.0` unconditionally and it was passed as
  `--host` to every engine, so each loaded model listened on all interfaces on its
  port (10000-10999). Anything that could route to the pod IP reached
  `/v1/chat/completions` **directly** — bypassing `require_bearer`, the per-token
  rate limiter, the priority scheduler and all token accounting. In Kubernetes that
  is every other pod in the cluster.

  The bind host is now driver-dependent: `0.0.0.0` only for the docker driver,
  whose engine runs in its own network namespace and must be reachable across it,
  and `127.0.0.1` for the local-subprocess driver, where the engine shares the
  warden's netns. An unrecognised driver name also binds loopback — an out-of-tree
  driver that genuinely needs a wider bind now fails loudly at the health probe
  rather than silently republishing an unauthenticated API. `VW_ENGINE_BIND_HOST`
  still overrides both, so the escape hatch survives; it is simply no longer the
  thing that decides the safe default. `build_vllm_args(..., driver=...)` defaults
  to the loopback bind, so a caller that forgets the kwarg gets an unreachable
  engine rather than an exposed one.

  The docstring that made this look intentional is also fixed. It described the
  local-subprocess driver as "the legacy in-process subprocess driver" and framed
  loopback as an opt-in — but `app/config.py:59` defaults `engine_driver` to
  `"local"`, so the "legacy" path is the one production runs.

  **Behaviour change worth reading before upgrading:** under the local driver the
  engine ports are no longer reachable off-host. Anything that was dialling one
  directly — an external Prometheus scrape of the engine, a debug `curl` from a
  neighbouring pod, a sidecar — breaks at the next load, and will surface as "the
  metrics stopped" rather than as a security fix. Set `VW_ENGINE_BIND_HOST=0.0.0.0`
  to restore the old behaviour for a deployment that needs it. Related: a blank
  `VW_ENGINE_BIND_HOST=""` used to produce `--host ''` and kill the engine at
  argparse; it is now treated as unset.

### Fixed
- **The proxy scheduler leaked an admission slot when a queued request was
  cancelled at the wrong moment (#217).** `_release` hands a slot to the next
  waiter with `event.set()` and deliberately does *not* decrement `_inflight`, on
  the assumption that the woken waiter releases in its turn. If the cancellation
  arrived after `event.set()` but before that coroutine was rescheduled,
  `event.wait()` raised `CancelledError` against an already-set event: the waiter
  owned a slot it would never enter the `yield` for, nobody else could see it, and
  `_inflight` stayed one too high forever.

  Client disconnect while queued is routine under load, and `app/proxy/routes.py`
  drives the context manager by hand around the whole upstream call, so the window
  is fully reachable from the product path. After `VW_PROXY_MAX_INFLIGHT` (default
  16) such disconnects on one `engine_key`, every later request for that model
  blocked forever in an untimed `event.wait()` — clients hung rather than erroring,
  and the model looked healthy from every angle except its proxy path.

  The waiter is the only party that can tell the two cases apart at the moment of
  cancellation, so the fix lives there: if the event is already set it owns a slot
  and releases it exactly as a normal holder would. Both release sites now run
  under `asyncio.shield`, because both can be entered while the task is already
  unwinding a cancellation and `_release` awaits a process-wide lock — unshielded,
  a second cancellation delivered while waiting on that lock would abandon the
  release half-done and leak the very slot it was there to save. The class
  docstring now states the invariant the whole scheduler rests on, and the untimed
  wait carries a comment explaining that it is only safe because of it.

- **The orphan GPU reaper attributed processes by a substring and killed other
  tenants' work (#218).** `reap_orphan_gpu_holders` filtered candidates with
  `"VLLM::" not in cmd and "vllm" not in cmd.lower()`, under a comment reading
  "never touch a foreign GPU process". A substring is not an ownership signal: a
  second vllm-warden on the same box, another team's vLLM container, and a
  researcher's `python -m vllm.entrypoints.openai.api_server` all matched, and all
  were `SIGKILL`ed the moment this warden recovered a crashed model.

  Attribution is now by **session id** — the pid of the `vllm serve` wrapper that
  fathered the process — read from the supervisor's handle table via the new
  `engine_ownership()`. A process this warden did not fork is never a candidate,
  whatever its cmdline says, and anything the reaper declines to kill is logged
  with the reason so an operator staring at a failed reload can still see that
  something else on the box is sitting on the VRAM. The comment and the code now
  agree.

  This was a prerequisite for #209: the natural fix there puts the reaper on the
  unload path, which would have escalated a rare cross-tenant kill into one firing
  on every operator-initiated unload. `engine_ownership(sup, also_sessions=...)`
  exists for that caller — the wrapper pid must be sampled **before**
  `Supervisor.unload`, which pops the handle in a `finally`, after which the
  session id is unrecoverable.

- **Schema migrations are now atomic (#231).** `apply_migrations` ran each file
  through `db.executescript(sql)` and recorded the `schema_migrations` row
  afterwards. `executescript` issues an implicit COMMIT *before* executing, so the
  bookkeeping INSERT necessarily landed in a second transaction. Anything that
  killed the process between the two left the DDL half-applied while the migration
  still looked unapplied — the next boot replayed it, hit "duplicate column name",
  and the pod never came up again. Recovery meant hand-editing the SQLite file on
  the `/data` volume: the only failure mode in this tree with no in-band recovery.

  The trigger was not exotic. A pod evicted mid-migration does it, and so does
  ENOSPC on `/data` — the same volume the engine logs share, and a failure this
  project has already hit once (2026-06-15).

  Each file's statements and its `schema_migrations` row now commit together inside
  one `BEGIN IMMEDIATE ... COMMIT`. An interruption either commits the whole
  migration or none of it. A guard rejects `PRAGMA`, `VACUUM` and explicit
  transaction control inside a migration file — several pragmas are silent no-ops
  inside a transaction, `VACUUM` errors outright, and a migration managing its own
  transaction would reintroduce exactly the split-commit window this closes. No
  migration in `sql/` contains any of them today; the guard is so the next author
  who needs one gets a loud error instead of a silent hole.

### Changed
- `GET /api/models/{id}/effective-argv` passes the configured driver into
  `build_vllm_args`, so the preview keeps its promise of showing the argv "exactly
  as the supervisor would at load time" now that the bind host depends on it
  (#211). Display-only.
- The orphan-reaper tests moved from `tests/unit/test_engine_watchdog.py` to
  `tests/unit/runtime/test_engine_watchdog.py` and were rewritten against the new
  `EngineOwnership` signature (#218), including the cases the substring match used
  to get wrong — a stranger's vLLM, a second warden's orphans, and a reaper that
  can attribute nothing.
## [v2026.08.22.1] — 2026-08-22

### Fixed
- **`/dev/shm` sizing existed only in the docker driver; production runs the
  other one (#210).** `docker_socket.py` has enlarged the engine container's
  shared memory since #160 (`ipc_mode=host` plus `shm_size=16g`), but
  `VW_ENGINE_DRIVER` defaults to `local` and production runs the
  local-subprocess driver inside a Kubernetes pod — where the engine is a
  child of the warden process and inherits the pod's 64 MiB default `/dev/shm`.
  vLLM's `shm_broadcast` ring wants 160 MiB of that at the stock 16 MB chunk
  size. tmpfs backs pages lazily, so `ftruncate()` and `mmap()` both succeed
  and the engine only fails when a write touches a page tmpfs won't allocate —
  an uncatchable `SIGBUS` inside `shm_broadcast.enqueue`, 78 s to 3 minutes
  into serving, killing every in-flight request. The only thing holding
  production up was a hand-added `VLLM_MQ_MAX_CHUNK_BYTES_MB=4` row in each
  model's `extra_env`: a 40 MiB ring that fits inside 64 MiB, at a throughput
  cost (payloads over the limit take vLLM's slower zmq overflow copy path,
  which multimodal requests hit routinely) and without ever reaching the
  ~1.1 GB of `ShmRingBuffer` segments a TP=4 engine holds.

  The fix is a real `/dev/shm` on the warden pod — an `emptyDir` with
  `medium: Memory` and an explicit `sizeLimit` — which podwarden-core #2417
  renders from `shm_size:` on a compose service. The production stack now
  carries `shm_size: 2g` on the `api` service, the chunk-size override has
  been removed from every model row, and the warden deliberately ships **no**
  `VLLM_MQ_MAX_CHUNK_BYTES_MB` default (it remains an allowed `extra_env` key
  for a deployment that cannot get real shared memory). Note for operators:
  the *stack-level* `shm_size` field applies to the primary service only,
  which for this stack is Caddy — put it on the `api` service in compose.

### Added
- **CI publishes the release images on the CalVer tag pipeline.** A new
  `publish:images` job builds both `vllm-warden` and `vllm-warden-ui` with the
  tag baked in as `VW_BUILD_VERSION`, pushes `<tag>` / `production` / `latest`
  to `registry.podwarden.com`, and re-reads each tag from the registry so a
  silently failed push fails the job. Until now every release was pushed by
  hand from a workstation (`docs/releasing.md`), which is how v2026.08.21.1
  and .2 ended up as git tags with no images. `docs/releasing.md` now
  describes the tag-driven flow and keeps the manual commands as a fallback.
- **Startup warning when the local driver has too little `/dev/shm` (#210).**
  Lifespan reads the real size via `os.statvfs` and, when the driver is `local`
  and `/dev/shm` is under 2 GiB, logs a `WARNING` naming the measured size, the
  minimum, and the SIGBUS symptom — so the failure mode is diagnosable from the
  log instead of from an engine that died without a traceback. The 2 GiB floor
  is a TP=4 engine's observed ~1.1 GB of `ShmRingBuffer` segments plus one
  160 MiB message-queue ring, with headroom. A `/dev/shm` that cannot be read
  at all warns about *that* rather than claiming a size it never measured, and
  the docker driver is skipped (the warden's own `/dev/shm` says nothing about
  a sibling engine container's).

### Changed
- **`docs/operating.md` gains an "Engine shared memory (`/dev/shm`)" section**
  — the 2 GiB requirement, the SIGBUS-in-`shm_broadcast.enqueue` symptom, how
  each driver obtains its shared memory (compose `shm_size:` on the `api`
  service under PodWarden, and why the stack-level field is the wrong place),
  and the `VLLM_MQ_MAX_CHUNK_BYTES_MB` stop-gap.

## [v2026.08.21.2] — 2026-08-21

### Fixed
- **The GitHub publish shipped CI's test report.** `v2026.08.21.1` put
  `pytest-unit.xml` on GitHub. The publish job declared no `dependencies:`,
  and a GitLab job without that key downloads the artifacts of **every** job
  in every earlier stage into its build dir — `unit-tests` publishes
  `pytest-unit.xml` as a `paths:` artifact, not merely a `reports:` one.
  `GIT_STRATEGY: clone` does not help: the artifact download happens *after*
  the clone. The publish then rsyncs the build directory, so the report was
  staged as if it were repository content. The RFC1918 sanitation warning
  counting 5 files on CI against 4 locally was the tell — the report embeds
  test fixture strings — and it was read as noise rather than as evidence.

  `publish:github` now declares `dependencies: []`, which downloads nothing
  and keeps holding for artifacts added later by someone who never thinks
  about this job. `publish/exclude.txt` also names the report and the common
  coverage outputs, but that only covers the artifact we already know about;
  the empty dependency list is what covers the next one.

- **`publish/exclude.txt` now excludes untracked local build products**
  (`openapi.json`, `docker-compose.uitest.yml`). The publish stages the
  *working directory* with rsync and knows nothing about `.gitignore`, so a
  local dry run staged two gitignored files that CI's fresh clone never had.
  Harmless here — and the direction of the discrepancy meant CI published
  *less* than the local run, not more — but it means a workstation publish and
  a CI publish did not produce the same tree, which defeats the purpose of
  dry-running it.

  Verified by planting `pytest-unit.xml` in the build dir inside
  `alpine:latest` to reproduce the artifact download, then confirming it does
  not reach the stage and the RFC1918 warn count drops back to 4.

## [v2026.08.21.1] — 2026-08-21

### Fixed
- **The GitHub publish could never have succeeded on CI.** The changelog
  preflight in `publish/publish-github.sh` read
  `${CI_PROJECT_DIR}/CHANGELOG.md`, but the file in this repo is
  `changelog.md` — lowercase. macOS resolves both spellings, so every local
  dry run passed; the GitLab runner's filesystem is case-sensitive, so the
  `grep` matched nothing and the job aborted with "CHANGELOG.md has no
  '## [v...]' release heading" before touching the network. The read is now
  spelled to match the file on disk, and a missing/empty changelog fails with
  a message that names the path it actually looked at instead of a heading
  complaint. Verified by running the whole script inside a Debian container
  with the tree copied onto the container's own (case-sensitive) overlayfs —
  the old path is `No such file or directory` there, the new one is not.

### Changed
- **Screenshots moved `docs/img/` -> `assets/screenshots/`.** `publish/exclude.txt`
  strips all of `/docs/` from the public snapshot, so every image the README
  embedded would have 404'd on GitHub the moment the sanitizing publish ran.
  `assets/` is not excluded. Every relative link and image target in the
  staged README is now checked to resolve inside the stage.
- **README rewritten for the GitHub audience.** It no longer links to six
  `docs/*.md` files that the publish excludes, it leads with what vLLM Warden
  adds on top of vLLM rather than with internal topology, and its claims are
  checked against the code: the first run is the `/setup` wizard
  (welcome -> gpus -> hf_token -> admin), not credentials in `.env`; the
  installer flags (`--dir`, `--gpus`, `--origin`, `--no-generate-secrets`) and
  the generated Makefile targets match `podwarden-hub`'s generator; the base
  image is v0.26.0, not the stale v0.25.1 the old text named.
- **`deploy/hub/README-hub.md` no longer links to `docs/operating.md`** on
  GitHub. `deploy/` is published but `docs/` is not, so that link was the same
  404 in a second published file. It now points at the public docs site.
- **`publish/denylist-warn.txt` is no longer empty.** It watches `pw_prod` and
  RFC1918 `10.10.x.x` addresses, both of which survive in code comments and
  changelog narrative. The node codename `d5` is deliberately not listed: the
  scan is case-insensitive and covers PNGs, so it reports ~23 files against
  ~13 real prose hits, and a warn nobody trusts is worse than no warn.

### Notes
- The public snapshot now stages the changelog as `CHANGELOG.md` (uppercase),
  the spelling GitHub surfaces in its repo and release UI. `rsync --delete`
  against the release repo retires the lowercase path left by earlier
  publishes.

## [v2026.08.19.1] — 2026-08-19

### Added
- **`PYTHONFAULTHANDLER=1` in the engine subprocess env.** On 2026-08-18
  `EngineCore` died 20 times in one night and left nothing behind: no Python
  traceback, no exit code, no core file. The only trace was the TP workers
  logging `Parent process exited` — they outlive the engine, so the death is
  invisible from the process the driver spawned. Cores were being discarded
  silently too: the host's `core_pattern` pipes to `/usr/share/apport/apport`,
  which does not exist in the engine image. faulthandler turns a fatal signal
  (SIGSEGV, SIGBUS, SIGILL, SIGABRT, SIGFPE) into a printed C-level Python
  stack on stderr, which the supervisor already captures into the engine log.
  It costs one signal-handler installation, so it is a default rather than an
  opt-in — a production engine should never be able to die silently.

  `PYTHONFAULTHANDLER` is added to `ALLOWED_ENV_EXACT` **by name** rather than
  via a `PYTHON` prefix, so an operator can set it to `0` without
  `PYTHONUNBUFFERED=0` also becoming reachable — the exact regression the
  hard-lock comment warns a future hand about.

### Fixed
- **A restart that fails to spawn no longer strands the model forever.** Two
  gaps combined to keep prod down for 11.5 hours on 2026-08-18/19 even with
  auto-restart working: `_restart` marks the row `loading` before calling
  `start_engine`, and when that spawn produced nothing, no path ever reset it
  — `on_exit` never fires (no process was supervised) and no recovery path
  inspected `loading`. Hours later the pod restarted, boot reconciliation
  stamped `prior_status='loading'`, and the restart predicate excluded it by
  design, making the row permanently unrecoverable.

  `_restart` now returns the row to `failed` and restores its restart flag if
  the spawn throws (releasing the port too), and `wants_restart` accepts
  `prior_status='loading'`. The latter is safe because the only writer of that
  value is boot reconciliation — "the warden restarted mid-load". `on_exit`
  sets the flag exclusively for rows that were `loaded`, so a model that failed
  on its own merits during a user-initiated load still carries no flag and is
  still left for a human. `pulling` and `unloading` remain excluded.

- **NCCL bumped 2.28.9 -> 2.30.4 to stop the TP worker hang.** This targets the
  crash itself rather than recovery from it. NCCL 2.28.9 deadlocks on CUDA-graph
  replay containing NCCL collectives: the collective never returns, the TP
  worker stops answering, and vLLM's watchdog fires `TimeoutError: RPC call to
  sample_tokens timed out` -> `EngineDeadError`. Workers stay alive with no
  traceback, no CUDA error and node-wide `oom_kill` at 0, which is why every
  memory theory read clean for three separate incidents (2026-08-02, and twice
  on 2026-08-18).

  Upstream vllm#52504 verified deadlock on 2.28.9 and clean on 2.30.4 with
  nothing else changed, and `--enforce-eager` avoiding it isolated the hang to
  captured-graph collectives. pw_prod ran exactly 2.28.9 with
  `cudagraph_mode=FULL_AND_PIECEWISE` and TP=4. The `sample_tokens` in the
  message is a red herring — the sampler is innocent.

  torch hard-pins `nvidia-nccl-cu13==2.28.9`, so pip warns; that is
  metadata-only (ABI-compatible within NCCL major 2, and torch dlopens the lib
  from the pip package directory, not the apt copy). Verified by a build-time
  assertion on the wheel version AND on the path torch actually maps —
  deliberately NOT via `torch.cuda.nccl.version()`, which returns the
  COMPILE-time version and reports 2.28.9 even after a successful upgrade.

- **A crashed engine now restarts itself.** Recovery had two paths and neither
  covered the common case. `check_once` only probes rows the DB believes are
  `loaded`; when the engine's wrapper exits, the supervisor's `_watch_exit`
  flips the row to `failed` first, after which the probe loop never looks at it
  again. Boot restore did not catch it either, because it matched `last_error`
  against an exact sentinel string — and the crash path writes a *diagnosis*
  there instead, displacing the sentinel. Prod died twice this way on
  2026-08-18 (09:15 and 13:11), the second time sitting dead for 1h43m while
  advertising a completely unrelated "requires trust_remote_code" message.

  Recovery now keys on a dedicated `prior_status` column (migration 0024)
  instead of on prose, and a new `restart_crashed_models` sweep restarts
  engines that died with their wrapper. Both paths share one `RestartBudget`,
  so a model flapping between them still hits a single ceiling rather than
  drawing a fresh allowance from each. Evidence is captured before the restart
  overwrites the live engine log — the exit path never did this, which is why
  `/data/crashes` was empty after a real crash. Only a row that was actually
  *serving* is restarted: a model that died while `loading` is a config problem
  for a human, not a retry loop.

- **"This model requires trust_remote_code" is no longer the default
  diagnosis for every crash.** The matcher was the bare substring
  `trust_remote_code`, and vLLM echoes the full engine config — including
  `trust_remote_code=False` — in its startup banner and in every
  `dump_input.py` ERROR line. Sitting last in the chain, it caught essentially
  any unrecognised failure and told the operator to enable arbitrary remote
  code execution to fix it. It now matches only the prose transformers/vLLM
  actually emit ("requires you to execute...", "set the option
  trust_remote_code=True"). Matching `trust_remote_code=True` would have
  reintroduced the bug for anyone who legitimately enables the flag.

- **`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` is now settable per model.** This is
  how long EngineCore waits for a TP worker before declaring the engine dead;
  vLLM defaults to 300s, tuned for "a slow batch" rather than "a worker is
  wedged". On 2026-08-18 a worker stopped answering and the engine served
  nothing for the full five minutes before erroring out. With automatic
  restart in place a *shorter* timeout is strictly better — it turns a silent
  five-minute hang into a fast detect-and-restart. Deliberately left at vLLM's
  default: lowering it globally would abort legitimately long batches on a slow
  GPU, so it is an operator dial rather than a new default.

### Changed
- **Engine-log disk usage is now bounded.** Crash dirs copied the *whole*
  engine log, making `/data` occupancy `watchdog_crash_keep x current log
  size` — a moving target that grows with uptime rather than a ceiling.
  Measured 2026-08-18: 20 crash dirs x a 6.8 MB log = 131 MB, all of the
  volume's usage. Only the tail is copied now (`VW_WATCHDOG_CRASH_LOG_BYTES`,
  4 MiB) since the crash is at the END of the file, and the live log rotates
  once past `VW_ENGINE_LOG_MAX_BYTES` (32 MiB) keeping one `.1` alongside.
  `/data` also holds the SQLite DB, so filling it takes the database down with
  it — the 2026-06-15 ENOSPC incident.

- **The header badge now distinguishes loading and failed from idle.** The
  metrics stream only surfaced a model at `status='loaded'`, and required a
  `model_runtime` sibling — which `loading` and `failed` rows never have. Both
  therefore fell through to "idle": a 27B load showed "idle" for the several
  minutes it took, and a crashed engine showed "idle" indefinitely,
  indistinguishable from a clean box with nothing loaded. The frame carries a
  new `active_model_status` (`loaded` | `loading` | `failed` | null) and the
  badge reads "loading" (sky) or "error" (red) accordingly. A failed engine
  deliberately outranks the amber probe-error tint — a degraded `nvidia-smi`
  must never mask a dead engine. Precedence is `loaded` > `loading` >
  `failed`, so a live engine is never repainted by an unrelated stale failure.
  The field is optional client-side: the UI and API ship as separate images,
  so a UI newer than its API still renders the old contract.

## [v2026.08.17.1] — 2026-08-17

### Added
- **Engine watchdog with crash-evidence capture.** The exit watcher awaits
  `handle.wait()` on the process the driver spawned — for the local driver that
  is the `vllm serve` **wrapper**, not the engine. vLLM v1 runs `EngineCore` and
  the TP workers underneath it, and on 2026-08-17 EngineCore died four times
  while the wrapper stayed alive. `handle.wait()` never returned, so the model
  row kept `status: loaded` with `last_error: None` while every generation
  returned 500. `GET /v1/models` answered 200 from the DB and `/healthz`
  reported on the warden, so every endpoint an operator would check said
  healthy; only a real generation revealed it. Each occurrence needed a manual
  force-unload and load.

  The watchdog probes each loaded engine's **own** `/health` instead — the same
  signal the load path already trusts — every 30s, acting after 3 consecutive
  failures. It does not watch processes at all, because the wrapper is
  unreliable in both directions: at 07:39 it exited, at 09:03 it was still alive
  seven minutes later.
- **Crash evidence, written before recovery.** A restart destroys the scene, so
  `{data_dir}/crashes/{model_id}/{timestamp}/` is populated first: the engine
  log copied out of the path the next load reuses, the process table, `/proc`
  meminfo plus the node-wide `oom_kill` counter, `nvidia-smi`, and a
  `report.json` carrying the wrapper return code — a negative value names the
  fatal signal. Newest 20 kept per model. Every capture step is individually
  guarded; collecting evidence must never block a restart.
- **Orphaned-worker reaping.** `unload(force=True)` reaps only the wrapper; the
  TP workers are its grandchildren and survive it, still holding all the VRAM.
  Measured: four `VLLM::Worker_TP*` processes holding 10550 MiB each after both
  EngineCore and the wrapper were gone, with the reload then failing on
  `Free memory on device cuda:0 (4.98/15.6 GiB)`. Recovery now kills orphaned
  GPU holders — only processes with no living engine ancestor, so a second
  healthy model is never touched — and waits for the VRAM to actually come back
  before reloading.
- **Restore after a warden restart.** The engine is a child of the warden
  process, so restarting the warden takes the engine with it and boot
  reconciliation marks the row failed with nothing to bring it back. Models that
  were serving are now reloaded with their stored settings; models that were
  mid-pull, deliberately unloaded, or failed on their own merits are left alone.
- **Watchdog settings under Settings → Maintenance.** `watchdog_enabled`,
  `watchdog_restore_on_boot`, `watchdog_interval_s`,
  `watchdog_failure_threshold`, `watchdog_max_restarts` (0 = detect and record,
  never auto-restart). Re-read from the settings KV every tick, so a change
  applies on the next pass with no restart.

### Changed
- `start_engine()` is extracted from the load route so the watchdog restarts a
  model through the exact path an operator's load takes, rather than a copy that
  would drift.
- `mark_runtime_dead_on_startup` records which prior status was interrupted, so
  a model that was serving can be told from one that was mid-pull.

### Known issue
- **Why EngineCore dies is still unexplained.** It vanishes mid-serving with no
  traceback, no CUDA error and no OOM message. The node-wide `oom_kill` counter
  read 0 across an uptime covering all four deaths, so the OOM hypothesis is
  refuted and a memory limit would not have helped. The leading remaining
  hypothesis is a native crash in CUDA/NCCL or the MTP speculative-decoding
  path. `report.json` captures the exit signal so the next occurrence is
  diagnosable rather than a guess.

## [v2026.08.12.1] — 2026-08-12

### Added
- **Per-token content logging (diagnostic, off by default).** Writes one JSONL
  record per request — prompt, completion, finish reason — to
  `content_log_path`. Hard-scoped to an explicit token allowlist so it can run
  on a shared server without ever logging all traffic: a request is logged only
  when `VW_CONTENT_LOG_ENABLED` is set **and** its token id is in
  `VW_CONTENT_LOG_TOKENS`. Prompt/completion fields are capped at
  `VW_CONTENT_LOG_MAX_CHARS` (default 40000) so the file cannot grow unbounded
  on very long generations. When disabled the proxy forward path is
  byte-identical to a build without it.
- **Runaway-generation detector (off by default).** Watches decoded tokens for
  a pathological generation — an unclosed `<think>` block past
  `VW_RUNAWAY_THINK_BUDGET`, a phrase repeating more than
  `VW_RUNAWAY_REPEAT_MAX` times, or a hard ceiling at `VW_RUNAWAY_HARD_MAX`.
  `VW_RUNAWAY_MODE=log` records a would-trip incident via the content-log sink
  without interrupting; `enforce` tears the generation down and emits a clean
  terminal SSE frame carrying a `runaway` finish reason. Reasoning-parser
  models (`--reasoning-parser qwen3`) stream the chain as
  `delta.reasoning_content`, so the detector synthesizes the think markers that
  the parser strips — otherwise the budget signal would be dead on exactly the
  models the pathology affects. When the mode is `off` no detector is built and
  no upstream stream-forcing happens.
  See `docs/superpowers/specs/2026-07-21-runaway-detector-design.md`.

## [v2026.08.03.1] — 2026-08-03

### Added
- **God Mode — inline images from vision requests.** The viewer now captures
  `image_url` parts from vision requests into a bounded in-memory media store
  (never persisted, dies on restart) and renders them as thumbnails with
  click-to-expand lightbox. Remote image URLs are hotlinked; stored images are
  served lazily via `GET /api/admin/godmode/media/{media_id}` (JWT-gated).
  New env knobs: `VW_GODMODE_MEDIA_STORE_CHARS` (default 64000000),
  `VW_GODMODE_MAX_IMAGE_CHARS` (default 14000000),
  `VW_GODMODE_MAX_IMAGES_PER_REQ` (default 16). Inert unless
  `VW_GODMODE_ENABLED=true`.
- **God Mode — live prompt/output stream (admin-only, off by default).** A new
  `/godmode` page, reachable from the stats page, streams real-time prompts and
  model output (both `content` and qwen3 `reasoning_content` channels) over SSE.
  Reuses the model-log live/explore scroll pattern: auto-follows at the bottom,
  scroll up to explore history, scroll back to the bottom to re-attach to live.
  Each request/session is rendered in a deterministic distinct color for visual
  filtering of interleaved streams. Backed by an in-memory `GodModeHub` ring
  buffer (dual-bounded on event count + char budget; nothing persisted, lost on
  restart). Gated by `VW_GODMODE_ENABLED` (default off) and the `require_jwt` UI
  session; when disabled, the `POST /api/auth/sse-ticket` mint returns 409 for
  the god-mode path so the viewer renders a clear "disabled" placeholder rather
  than spinning (EventSource `onerror` carries no HTTP status, so the disabled
  signal must come from the ticket mint the browser can read). Captures **all**
  tokens for the privileged admin view — a deliberate, scoped choice for this
  operator-only diagnostic surface.
- **Model-log fast-load scroll fix** verified and regression-tested: a model
  that finishes loading quickly no longer auto-scrolls up into exploration mode
  (the stick/free latch is decoupled from `followOutput`).
- **God Mode — collapse repeated system prompts.** When a token replays the same
  long system-prompt head request after request, the god-mode viewer now folds
  the unchanged head behind a small `(system prompt [+])` toggle and shows only
  the new turn by default; the head is always one click away (`aria-expanded`).
  The diff is line-aware and per-token (a prompt is only compared against the
  previous prompt from the **same** token), guarded by minimum length/ratio
  thresholds so short or genuinely-divergent prompts render in full. Backend:
  the god-mode capture window is now head **+ tail** (`VW_GODMODE_PROMPT_TAIL_CHARS`,
  default `4000`) instead of a pure head slice, so a huge repeated head no longer
  clips the genuinely-new tail turn; the elision marker keeps the collapse honest
  (it never diffs across an elided gap). Display-only — the proxied request body
  is still forwarded byte-identical.
- **Token rotation can now revoke the old token immediately (#185).** The rotate
  dialog asks what should happen to the predecessor: keep it working for a grace
  period (default, still 24h) or **revoke it immediately**, which sends
  `grace_hours: 0` and sets the old row's `revoked_at` to *now* so it is rejected
  with `401 token revoked` on its very next request — bearer auth reads the DB
  per request, so there is no cache to wait out. This is for the case rotation is
  most often reached for: a leaked or abused key, where the old 24h window kept
  the very secret you were trying to kill alive and admitting traffic for a full
  day. `grace_hours: 0` has always been accepted by the API (`0..720`, outside is
  422) — what was missing was any way to choose it, or to see that you had. The
  dialog states plainly that a hard cut stops **future** admissions only: a
  generation already running on the engine is not cancelled (vLLM has no concept
  of warden tokens) and runs to completion or to the read timeout.
  `POST /api/tokens/{id}/rotate` now echoes the applied `grace_hours` in its
  response, so scripted callers get a confirmation and the success modal
  describes what the server did rather than what the client asked for.

### Fixed
- **The "Test" button reported revocation wrongly in both directions (#185).**
  `POST /api/tokens/{id}/test` derived `revoked` from `revoked_at is not None`,
  but `revoked_at` is a *future* timestamp for the whole of a rotation grace
  window. A predecessor that authenticated fine was reported as revoked, and a
  hard-revoked one produced the identical message — so pressing Test proved
  nothing either way. It now makes the same `revoked_at <= now` comparison
  `require_bearer` makes. Note this flips the answer for tokens inside a healthy
  grace window from "revoked" to "not revoked"; that is the correct answer.
- **A rotated token's status badge never lapsed (#185).** `deriveStatus` gated
  its only revoked branch on `rotated_at == null`, so a rotated row was painted
  amber "Rotated (grace)" forever — including after the window closed, and
  including immediately after a hard revoke. Rotated rows whose `revoked_at` has
  passed now show a red **"Rotated (revoked)"**; both rotated badges carry the
  exact timestamp as a tooltip. The list payload gained a server-computed
  `is_revoked` (beside the existing `is_expired`) so the badge does not depend on
  the operator's workstation clock.
- **Clearing the rotate dialog's grace field silently hard-revoked the token
  (#185).** `Number("") === 0` passed the `>= 0` validation, so blanking the box
  — the natural gesture for "never mind, leave the default" — POSTed
  `grace_hours: 0` and cut the predecessor off immediately, with nothing on
  screen saying so. The field now only exists in grace mode, and a blank value
  there is a validation error instead of a zero.

## [v2026.07.19.3] — 2026-07-19

### Added
- **Server-side request reaper for the proxy** (`VW_REQUEST_MAX_WALL_S`,
  default `0.0` = disabled). When set `> 0`, a proxied request that runs longer
  than this many seconds is torn down: the proxy stops draining the upstream,
  `aclose()`s the upstream response + httpx client (so vLLM aborts the generation
  and frees its KV blocks), and releases the scheduler slot. Covers both paths —
  streaming (per-chunk wall-clock check in the SSE loop) and non-streaming
  (`asyncio.wait_for` around the upstream read, returning `504` on expiry). Closes
  the "abandoned but transport-alive client" hole where the downstream connection
  keeps draining tokens (or a hung upstream keeps the socket alive), no
  `http.disconnect` ever reaches uvicorn, Starlette never cancels the body
  iterator, and the request pins its slot + KV indefinitely. Recommended
  production value: `600`. Priority scheduling and the auth/token model are
  unchanged.

### Fixed
- **Streaming proxy now detects downstream disconnect during the reasoning
  phase.** The `request.is_disconnected()` check was nested inside `if delta:`
  (content deltas only), so it was unreachable while the engine streamed
  `reasoning_content` — a request abandoned mid-reasoning was never noticed. The
  disconnect and wall-clock checks now run every loop iteration, independent of
  whether the chunk carried a parseable content delta.
- **Live-stats dashboard no longer mislabels reasoning as "prefill 0%".** The
  phase flips to `decode` on any SSE `data:` frame (content *or*
  `reasoning_content`), not only content deltas; token accounting stays
  content-only so billing is unaffected.

## [v2026.07.19.2] — 2026-07-19

### Added
- **New realtime stats dashboard at `/ui/stats/live`** (the old `/stats` page is
  unchanged, gaining only a "Live view" link). Surfaces live vLLM engine +
  per-request telemetry the aggregate `/stats` page could not: requests
  running/waiting (by wait reason), KV-cache pressure with absolute used/total
  tokens and preemptions/s, throughput and latency percentiles, prefix-cache hit
  rate and MFU, finished-reason mix, and a live per-request table (token · client
  IP · model · phase · context-window fill vs `max_model_len` · elapsed · orphan
  flag) aggregated by warden token and client IP. Built as two independent data
  planes behind a fixed contract (`docs/live-stats-spec.md`):
  - **Engine plane** — `GET /api/stats/live` (SSE, ticket-auth): scrapes the vLLM
    aggregate Prometheus `/metrics`, ~1.5s cache, rates as frame deltas, histogram
    percentiles from `_bucket`, absolute KV tokens from `cache_config_info`.
  - **Proxy plane** — `GET /api/stats/requests` (JWT poll): in-process
    `RequestRegistry` hooked fail-open into the proxy `_forward` path; the only
    source of per-request context (aggregate `/metrics` cannot provide it).

## [v2026.07.19.1] — 2026-07-19

### Fixed
- **Non-streaming proxy path now closes the upstream vLLM socket on client
  disconnect (#184).** In `app/proxy/routes.py` the `resp.aclose()` /
  `client.aclose()` calls sat after `await resp.aread()` inside the `try`, so a
  client disconnect (which raises `CancelledError` out of `aread()`) skipped
  them — the warden→vLLM socket stayed open until GC, leaving vLLM generating
  and holding KV blocks for a request nobody was reading. Moved both `aclose()`
  calls into the `finally` (mirroring the already-correct streaming path) so the
  abort reaches vLLM synchronously and KV blocks free immediately. A latent KV
  amplifier under load on VRAM-constrained engines; no behaviour change on the
  success path (`aclose()` is idempotent).

## [v2026.07.18.4] — 2026-07-18

### Fixed
- **All model loads on 0.25.1 fixed — `vllm-gguf-plugin` 0.0.4
  `override_quantization_method` signature patched.** vLLM 0.25.1's
  `_verify_quantization` (`vllm/config/model.py`) calls
  `override_quantization_method(quant_cfg, user_quant, hf_config=...)` for
  **every registered quantization method** during ModelConfig init — core's
  base signature grew a third `hf_config` param. The out-of-tree plugin (0.0.4
  is the latest on PyPI) still had the old 2-arg classmethod signature, so the
  `hf_config=` kwarg raised `TypeError` and broke **100%** of loads, even the
  non-GGUF FP8 production row. This was a second, pre-existing blocker masked
  behind the nixl_ep segfault below — it only surfaced (as a catchable `rc=1`)
  once the segfault was fixed. The Dockerfile now patches the plugin
  classmethod's signature at build time to accept and ignore the extra kwarg
  (its body already discards its inputs, so this is behaviour-preserving). The
  patch is applied in one build step and **verified in a separate `python3`
  process**: resolving the module path imports the plugin package (loading the
  config module at the old source into `sys.modules`), so an in-process
  re-import returns a stale compiled code object and the signature check fails
  spuriously — a fresh interpreter compiles the patched source, exactly as the
  runtime `vllm serve` subprocess does. Keep the `s2 != s` guard: any base-image
  or plugin bump can re-break the match and must fail the build loudly.

## [v2026.07.18.3] — 2026-07-18

### Fixed
- **gguf-plugin `override_quantization_method` signature patched for vLLM
  0.25.1 — superseded the same day by v2026.07.18.4.** Intermediate hotfix: it
  applied the signature patch (see v2026.07.18.4) but validated it in the *same*
  `python3` interpreter that had already imported the plugin, so the build-time
  check read a stale compiled code object and passed spuriously — the shipped
  image did not actually carry the working patch. Corrected in v2026.07.18.4 by
  running the verification in a separate interpreter.

## [v2026.07.18.2] — 2026-07-18

### Fixed
- **Model loads no longer SIGSEGV (rc=-11) at engine startup on non-AVX CPUs.**
  The v0.25.1 base image bundles NVIDIA's `nixl_ep` (cross-node expert-parallel
  all2all) packages, whose compiled CUDA extension dlopens an AVX-built UCX
  library at *import* time. On a CPU without AVX (the pw_prod GPU node runs a
  QEMU vCPU with none) UCX's load-time feature check aborts the process —
  `FATAL: UCX library was compiled with avx but CPU does not support it` — a
  C-level segfault, not a catchable exception, so *every* model load crashed on
  0.25.1. vLLM guards this import only with `has_nixl_ep()` (a bare
  `importlib.find_spec` presence check), so the Dockerfile now removes the
  `nixl_ep*` packages: the guard goes False and the AVX-UCX path is never
  touched. `nixl_ep` is cross-node expert-parallel, irrelevant to our
  single-node TP deployments; base `nixl` is left intact.

## [v2026.07.18.1] — 2026-07-18

### Changed
- **Base image bumped `vllm/vllm-openai` v0.20.0 → v0.25.1** (digest-pinned).
  Motivation: the 2026-07-17 production outage — vLLM 0.20.0's ngram
  speculative-decoding + prefix-caching block-accounting bug
  (`AssertionError: num_required_blocks < len(req_blocks)` in
  `single_type_kv_cache_manager.py`) crash-looped the engine; 0.25.1 carries
  the upstream fix, so `--speculative-config` can be re-enabled on affected
  model rows after this ships.
- **GGUF support now comes from the `vllm-gguf-plugin` (pinned 0.0.4)**
  instead of vLLM core. Upstream moved GGUF loading out of core in the 0.25.x
  line (`model_loader/gguf_loader.py` no longer exists), so the three
  sed-patches we carried against the 0.20.0 base for Qwen3.5/3.6 GGUF tensor
  renames (#107/#108/#115, upstream vllm PR #38140 — still open/unmerged) are
  removed with it. The plugin uses an adapter-based loader with no rename
  table; **Qwen3.5/3.6 GGUF loadability under the plugin is unverified** — if
  a Qwen3.x GGUF row regresses after this release, start there. Safetensors
  rows (including the production FP8 model) are unaffected.
- Curated template engine pins (`app/templates/registry.py`,
  `docs/tested-stacks.md`) deliberately stay on their tested 0.20.0 stacks —
  they describe docker-driver try-stack combos that were validated on 0.20.0,
  not the baked engine of this image.

### Fixed
- **Live log tail no longer unsticks mid-stream during a fast vLLM startup
  burst.** The per-model log viewer (`models/log-stream.tsx`) drives
  react-virtuoso's `followOutput` through the `useStickyBottom` hook, which
  previously gated auto-follow on its own stick/free latch. The latch flips to
  "free" on *any* `atBottomStateChange(false)`, and under a ~200 line/s append
  burst Virtuoso emits a *transient* `atBottom=false` (the freshly-appended row
  sits below the viewport for a frame before the follow-scroll lands — and the
  old `"smooth"` scroll could not animate fast enough to keep pace). That
  transient permanently disabled auto-follow, forcing the operator to click
  "Jump to latest" over and over. `followOutput` is now always the instant
  `"auto"` behavior, decoupled from the latch: react-virtuoso self-gates to
  only auto-scroll when the list is already at the bottom, so the tail stays
  pinned through bursts yet never yanks an operator who has scrolled up to read.
  The `mode` latch survives but now only drives "Jump to latest" visibility. A
  defensive `atBottomThreshold={64}` on the `<Virtuoso>` swallows sub-row
  transients. Pinned by `tests/component/use-sticky-bottom.test.ts` (4 tests).

## [v2026.06.16.1] — 2026-06-16

### Changed
- **`open_db` now sets an explicit `PRAGMA busy_timeout = 30000` (was the
  implicit aiosqlite default of 5000 ms).** During a user-triggered HuggingFace
  model pull the process can saturate CPU/IO and briefly starve the asyncio
  event loop; the short-lived connections opened by background writers (stats
  sampler, pull poller) could then collide with the foreground writer and fail
  fast with `database is locked`. A generous 30 s busy_timeout makes them wait
  out the contention window instead. (`app/db/database.py`; covered by
  `tests/unit/db/test_database_pragmas.py`.) Note: this hardens against lock
  contention but is not the master cause of probe-kills during a pull — that is
  whole-container event-loop starvation, addressed at the deployment-probe layer
  in the PodWarden Hub stack template (lenient liveness + a `startupProbe`, and
  the Caddy front-door health probe decoupled from the api via a Caddy-local
  `/caddy-healthz` route).

## [v2026.06.15.1] — 2026-06-15

### Fixed
- **Engine now writes downloaded weights to the HF cache volume, not `/data`
  (ENOSPC crash fix).** The vLLM engine subprocess resolved its model cache via
  `HF_HOME`, whose on-disk layout (`$HF_HOME/hub/models--…`) did not match where
  the pull agent had placed the snapshot, so the engine re-downloaded the full
  model into the small `vllm-warden-data` volume (`/data`, 10 GiB), filling it
  to 0 bytes and taking the SQLite database down with it (`Fatal Python error:
  _PyImport_Init`). `app/runtime/env_builder.py` now sets `HF_HUB_CACHE` (whose
  layout `$HF_HUB_CACHE/models--…` matches `snapshot_download(cache_dir=…)`)
  instead of `HF_HOME`, so the engine and the pull agent agree on one path —
  the 100 GiB `vllm-warden-hfcache` volume. `HF_HOME` is no longer exported.

### Removed
- **The non-functional "HF cache directory" setting has been removed entirely.**
  The `hf_cache_dir` key was seeded into the settings KV table and exposed as an
  editable field in the General settings tab and the `/api/settings/runtime`
  surface, but its stored value was never read — the real cache path is
  env-driven only (`VW_HF_CACHE_DIR` → `Settings.hf_cache_dir`), so editing it in
  the UI silently did nothing. Removed the seed (migration `0010`), the runtime
  API key + coercer (`/api/settings/runtime` now rejects `hf_cache_dir` as an
  unknown key with `400`), and the General-tab field. New migration `0023`
  deletes any `hf_cache_dir` row left in existing databases (idempotent).

### Changed
- **`VW_FRONTEND_ORIGIN` is now optional in the PodWarden Hub stack template.**
  The hub compose previously hard-required `VW_FRONTEND_ORIGIN` via Compose's
  `${…:?}` operator, so a fresh install refused to start until the operator set
  it. The API already treats an explicitly-empty value as the localhost-default
  fallback (so a blank var never locks admins out), so the compose now uses
  `${WARDEN_FRONTEND_ORIGIN:-}` and the stack boots without it. Set it to your
  public UI origin to harden the CSRF/Origin check on `/api/auth/*`.

## [v2026.05.28.1] — 2026-05-28

### Fixed
- **Engine bind-host default flipped from `127.0.0.1` to `0.0.0.0`** so loads
  succeed on the docker-socket driver out of the box (#180). The docker driver
  (production default) runs each engine as a separate container; binding to
  loopback inside that container made `/health` and `/v1/*` unreachable from
  the api container and from docker-proxy's published port, so the supervisor
  health probe timed out and the row flipped to `failed` despite the engine
  emitting `Application startup complete.` Operators on the legacy in-process
  subprocess driver (engine shares the api container's netns) can opt back
  into loopback by exporting `VW_ENGINE_BIND_HOST=127.0.0.1` at deploy time.

## [v2026.05.27.6] — 2026-05-27

### Changed
- **Try-stack: "Try combo" button replaced by one-click "Load" that pins the
  combo and starts the engine in a single action.** Picking a channel + vLLM
  version on the per-model Try-stack panel and clicking Load now POSTs
  `/api/models/{id}/try-stack` (pin + record pending attempt) and then
  `/api/models/{id}/load` (start the engine) back-to-back, instead of forcing
  the operator to bounce to the model card's Load button after picking a
  combo. The button is now disabled while the model is `loading`, `loaded`,
  or `unloading`, with an inline note instructing the operator to unload
  first to try a different combo. Backend endpoints are unchanged.

### Added
- **Try-stack: warn before re-submitting a (channel, vLLM version) combo that
  previously failed.** When the picker's selection matches a historical
  attempt with `result === "failed"`, a yellow banner above the Load button
  surfaces the past error + timestamp; the first click on Load shows a
  "Click Load again to confirm" prompt instead of submitting, and the second
  click submits. Changing channel or version clears the banner and the
  confirm gate.
- **Try-stack: Retry link on each history row pre-fills the picker** with
  that row's `channel` + `vllm_version`. The Retry link is hidden on the
  most recent attempt while it is still pending (no signal in re-pinning a
  not-yet-reported attempt). When the prefilled combo matches a failed row,
  the duplicate-failed-attempt banner above fires naturally on the next Load
  click.

## [v2026.05.27.5] — 2026-05-27

### Added
- **Fit-preview now warns when the detected GPUs can't run the chosen
  quant/dtype well (#176).** The live GPU probe (`GET /api/system/gpus` and the
  fit-preview internals) now captures each card's CUDA compute capability
  (`compute_cap`, e.g. `8.6`), degrading gracefully to `null` when the driver
  reports `[Not Supported]`/`[N/A]`. `POST /api/models/fit-preview` crosses the
  weakest selected GPU's `compute_cap` with the candidate's quant/dtype (from
  `config.json` `torch_dtype` + `quantization_config.quant_method`; GGUF is
  treated as bf16-compute since it dequantizes at load) and appends actionable,
  human-readable strings to the existing `warnings` list — e.g. FP8 on an
  Ampere card (cc &lt; 8.9) is flagged as emulated (slow/inaccurate), bf16 below
  cc 8.0, NVFP4/FP4 below Blackwell (cc 10.0), and AWQ/GPTQ INT4 below the
  Marlin floor (cc 7.5). These are warnings only — fit-preview never blocks. The
  rule matrix lives in the pure module `app/models/gpu_capability.py`; when no
  GPU reports a compute capability, no capability warnings are produced.
- **The try-stack vLLM-version field is now a typeable dropdown backed by the
  published image-resolving versions (#177).** The free-text "vLLM version"
  input on the per-model Try-stack panel becomes a combobox populated from a new
  `GET /api/templates/engine-versions?channel=` endpoint, which lists the
  `vllm/vllm-openai` semver tags that actually resolve to an image for the
  selected channel (resolvable CUDA channels only; rocm/cpu/xpu/unknown return
  an empty list). Versions come from a 6h in-process cache keyed by image family
  so Docker Hub's anonymous rate limit is never hit per request, and a Docker
  Hub hiccup serves stale-or-empty rather than failing the page. The field stays
  free-text, so an operator can still enter an unpublished version or a pinned
  digest the catalog won't list.
- **Origin guard now supports a comma-separated allow-list and an opt-in
  reverse-proxy trust mode.** `VW_FRONTEND_ORIGIN` accepts a comma-separated
  list of exact origins (whitespace trimmed, trailing slashes stripped) instead
  of a single value, so more than one front-door origin can be accepted by the
  `/api/auth/refresh` and `/api/auth/logout` origin check. A new
  `VW_TRUST_PROXY_ORIGIN` env var (default off) additionally accepts a request
  whose `Origin` equals the proxy-derived `X-Forwarded-Proto://X-Forwarded-Host`
  — only enable it behind a trusted reverse proxy that sets those headers. This
  fixes reloads logging users out when the served origin isn't pinned ahead of
  time. The check remains fail-closed: an origin matching neither the allow-list
  nor (when enabled) the proxy-derived origin is rejected with `403`.
- **GPU selection is now checkbox-driven from live system inventory (#175).**
  The default GPU indices field on the Settings → General tab and the per-model
  settings page replaces the comma-text field with a `GpuChecklist` component
  populated from a live `GET /api/system/gpus` probe. Each GPU renders as a
  labelled checkbox (index, name, VRAM). Configured GPU indices that are no
  longer present in the live probe render as a removable amber "not present" row
  with a warning banner so operators can identify and clean up stale assignments
  without losing the value before they decide what to do. The per-model
  `gpu_indices` field now enforces a client-side minimum of one selection:
  emptying it disables Save and renders an inline "Select at least one GPU."
  validation message next to the field (the Settings → General
  `default_gpu_indices` field may still be empty, meaning "no preset").
- **Model load fails fast with `422 gpu_index_missing` when a configured GPU
  is absent (#175).** The load pre-flight now cross-checks the model's
  `gpu_indices` against the current `nvidia-smi` probe and rejects immediately
  with a structured error (`detail.error_code = "gpu_index_missing"`, a
  human-readable `detail.message` naming the absent indices, and
  `detail.available` listing the GPUs the probe did see) rather than letting
  vLLM start and crash with a cryptic NVML error several seconds later. The
  pre-flight now **fails closed**: a probe error (e.g. `nvidia-smi`
  unavailable) leaves no ground truth confirming any configured GPU, so the
  load also 422s `gpu_index_missing` (with `detail.available = []` and the
  probe's error in `detail.probe_error`) instead of optimistically proceeding.
  The descriptive message is surfaced in the UI's load-error toast so the
  operator knows which index to fix.
- **Proxy throughput: per-engine concurrency + native vLLM priority (#173).**
  The `/v1` proxy's `PriorityScheduler` no longer serializes the whole engine
  behind a single global in-flight slot — the historic cause of the product
  path flat-lining at ~40 tok/s while the engines themselves sustained 1000+
  tok/s aggregate. It now admits up to `VW_PROXY_MAX_INFLIGHT` (default 16)
  concurrent requests **per engine** (keyed on the model id), so a busy engine
  never blocks a request bound for an idle one, and continuous batching inside
  vLLM is actually exercised. Token priority (0..9, 9 = highest) is preserved
  as the admission ordering under contention **and** pushed into the engine:
  every engine launches with `--scheduling-policy priority` and the proxy
  injects a per-request `priority = -token.priority` into the forwarded
  chat/completions body (vLLM orders its waiting queue ascending, so the
  negation puts high-priority traffic ahead of default-0 traffic). Warden
  priority 0 maps to vLLM's inert default and is not injected; a client that
  sets `priority` explicitly is left untouched.
- **Docs: `docs/tested-stacks.md` — validated model + engine combinations.**
  A living record of model / vLLM-engine stacks driven end-to-end through the
  Chat Playground UI and human-confirmed coherent on sm_86 Ampere hardware
  (4× RTX A4000 on host d5). Documents six validated combos (gpt-oss-20b,
  Qwen2.5-3B, Nemotron-Nano-8B, Mistral-7B, Llama-3.1-8B-AWQ, Qwen2.5-14B-AWQ)
  plus the Ampere quantization rules that drove the choices: bf16 8B OOMs on a
  single 16 GiB card, `--quantization fp8` is emulated and produces gibberish,
  AWQ INT4 is the working single-GPU path, and `--enforce-eager` avoids the
  inductor compile-time OOM. Also records the "Save working combo as template"
  gap (it drops `extra_args` and the `gpu_memory_utilization` override).
- **Open-source release: Apache-2.0 LICENSE + GitHub publish flow.** vLLM
  Warden is now open source under Apache-2.0. A manual `publish:github` CI
  job (on `main` push) mirrors the tracked tree to the public
  [github.com/Podwarden/vllm-warden](https://github.com/Podwarden/vllm-warden)
  repo as a single squashed "Initial public release" commit via
  `scripts/publish-github.sh`. Ships with a public-facing README (install,
  usage, build-from-source) and UI screenshots.

### Fixed
- **Live engine logs now stream under the docker engine driver (#177
  follow-up).** Under `DockerSocketDriver` the engine runs as a sibling
  container whose vLLM output goes only to `docker logs <engine-container>`;
  no per-model log file was ever written. But the SSE log endpoint
  (`app/models/routes_logs.py`) tail-follows `<data_dir>/logs/<model_id>.log`
  exclusively, so the UI "Live logs" panel showed stale pre-switch content
  (or nothing) — on one host this made a freshly-loaded `v0.21.0` engine look
  like it was still running `v0.20.0`. `DockerSocketDriver` now mirrors the
  engine container's combined stdout+stderr into that same per-model log file:
  on `spawn()` it truncates the file (clearing any stale subprocess-driver
  content) and pumps `container.logs(stream=True, follow=True, ...)` — a
  blocking generator — on a dedicated daemon thread, so `routes_logs.py` and
  the SSE pipeline stream live output unchanged. The no-log-file behavior is
  preserved when no log dir is configured.
- **Engine-version pins are no longer silently discarded under the in-container
  subprocess driver (#177).** The Try-stack version selector pins a vLLM engine
  image onto a model, but under the default `LocalSubprocessDriver` the warden
  runs the vLLM binary baked into its own image and never reads `EngineSpec.image`
  — so a pinned version (e.g. `v0.21.0`) was quietly ignored and the in-image
  version launched instead. Drivers now declare a `supports_engine_image`
  capability (`False` for subprocess, `True` for the docker driver), and
  `Supervisor.load` refuses an unsupported pin up front (raising
  `EnginePinUnsupported` before claiming any GPU) so it surfaces as a failed load
  with an actionable message instead of a wrong-version launch. A new
  JWT-protected `GET /api/system/engine` reports the active driver
  (`subprocess`/`docker`), whether version selection is supported, and the
  baked-in vLLM version; the Try-stack panel reads it to disable the channel /
  version / Try controls and show an amber note explaining that version selection
  requires the docker engine driver when the deployment can't honor a pin.
- **Settings inputs for GPU indices, extra args, and extra env no longer wipe
  in-progress text (#175).** The `int-list`, `string-list`, and `kv-map`
  `SettingField` kinds were using uncontrolled-to-controlled input patterns that
  reset the cursor whenever the parent re-rendered with a round-tripped API
  value — making it impossible to type a trailing comma or newline before the
  next value. These fields are now fully controlled: the local draft state is the
  single source of truth during editing and only flushes to the server on Save,
  so more than one GPU index (or extra arg, or env entry) can be entered without
  the field wiping itself mid-keystroke.
- **Docker driver: engines pinned to a non-zero GPU index no longer crash at
  startup (#172).** When the `DockerSocketDriver` spawns a sibling engine
  container it pins the model's physical GPUs via `device_requests`; the NVIDIA
  container runtime then renumbers those passed-through devices to `0..N-1`
  *inside* the container. `env_builder` sets `CUDA_VISIBLE_DEVICES` to the
  model's **host** indices (correct for the in-process `LocalSubprocessDriver`,
  whose parent container sees every GPU), so a model on host GPU 1/2/3 received
  `CUDA_VISIBLE_DEVICES=1/2/3` against a container whose NVML view only had
  index 0 — vLLM's `nvmlDeviceGetHandleByIndex(host_idx)` raised
  `NVMLError_InvalidArgument` and the engine exited `rc=1`. Only GPU 0 happened
  to work (host index 0 == relative 0), silently masking the bug on
  single-engine hosts. The driver now remaps `CUDA_VISIBLE_DEVICES` to the
  container-relative range (`0..N-1`) while `device_requests` still pins the
  correct physical GPUs — unblocking multi-engine / full-cluster utilization.
- **"Save working combo as template" now captures `extra_args` +
  `gpu_memory_utilization` (#170).** The try-stack save-as-template flow only
  persisted the engine axis, repo, `max_model_len` and `tensor_parallel_size`,
  silently dropping the live model's `extra_args` (came back `[]`) and its
  `gpu_memory_utilization` override (came back as the `0.9` default). An AWQ
  single-GPU template saved with `--enforce-eager` +
  `gpu_memory_utilization=0.92` therefore re-instantiated without them and
  OOM'd. `TemplateCreate` now accepts the live `model_id`; the backend sources
  `extra_args` + `gpu_memory_utilization` from that model row (the same source
  of truth the create form uses, with explicit body fields still winning), and
  the try-stack panel passes `model_id` so the saved template reproduces the
  combo that actually worked.
- **`terminate()` reaps a cancelled engine wait instead of 500-ing (#171,
  root cause behind #166).** `DockerHandle.wait()` memoizes one
  `_wait_task` shared between the exit-watcher and `terminate()`. When
  `Supervisor.unload()` cancels the watcher it cancels that shared task; a
  later `terminate()` awaiting the same task re-raised
  `asyncio.CancelledError` — a `BaseException` since py3.8, so it escaped
  `terminate()`'s `except Exception` reap guard, propagated through
  `unload()`, and collapsed in the CSRF middleware to
  `RuntimeError("No response returned.")` → HTTP 500. Fixed in two places for
  robustness: the reap guard now also catches `asyncio.CancelledError` (treats
  the container as exited, returncode 0), and `wait()` no longer reuses a
  poisoned (already-cancelled) memoized task — it recreates the reap against
  the live container. This addresses the underlying defect that #166's
  background-teardown change worked around.
- **Unload no longer 500s / strands the row in `unloading` (#166).** The
  unload route used to `await` the full engine teardown inline. A large
  multi-GPU engine takes many seconds to terminate, so the client/proxy
  disconnected mid-request and the CSRF `BaseHTTPMiddleware` raised
  Starlette's `RuntimeError("No response returned.")` → HTTP 500 — leaving the
  model stuck in the transient `unloading` state with the only recovery being
  a control-plane restart. Teardown now runs in a background task (mirroring
  the load path): the fast refusal check is still surfaced synchronously as
  409, the route returns 202 immediately, and the row always reaches a
  terminal state — even if teardown raises — so recovery never requires a
  restart. New `Supervisor.ensure_unloadable()` pre-flight separates the
  instant state check from the slow teardown.
- **Supervisor.unload: never leak the in-memory GPU claim when teardown
  raises.** `unload()` released GPU ownership only as its trailing statement,
  so if `driver.terminate()` raised — or the await was cancelled by a
  client/proxy disconnect mid-teardown — `self.gpus.release()` was skipped and
  the GPUs stayed permanently `already claimed` by a model that no longer
  exists. Every subsequent load onto those GPUs then failed with
  `GPUs [...] already claimed`, recoverable only by restarting the control
  plane. Teardown now runs in a `try/finally` so the GPU claim and lifecycle
  bookkeeping are always released past the refusal gate; a refused unload
  (transient state, no `force`) still correctly retains the claim. Observed on
  d5 after a `#166` client-disconnect unload stranded a multi-GPU engine's
  claim. Complements `#166` (which decouples teardown from the request so the
  primary cancellation path no longer triggers) with defense-in-depth for any
  other teardown exception.
- **DockerSocketDriver: give engine siblings adequate shared memory for
  tensor-parallel.** The sibling engine container inherited docker's default
  64MB `/dev/shm`, so vLLM `tensor_parallel_size > 1` hung indefinitely at
  the distributed-init barrier (workers communicate over POSIX shared memory
  via the shm_broadcast message queue + custom-all-reduce CUDA-IPC handles) —
  the model sat in `loading` forever with no crash and no `last_error`. The
  driver now spawns engines with `ipc_mode=host` (matching the control-plane
  compose override) plus an explicit `shm_size` fallback. Both are
  env-overridable (`VLLM_ENGINE_IPC_MODE`, `VLLM_ENGINE_SHM_SIZE`); set
  `VLLM_ENGINE_IPC_MODE=""` for a private but enlarged `/dev/shm`. Surfaced
  loading gpt-oss-20b at TP=2 on the docker driver.
- **try-stack (#162): wire the missing report (ok/failed) control.** The
  model-detail try-stack panel recorded a "pending" attempt and offered
  "Save working combo as template" (gated on `result === "ok"`), but had no
  control to report whether the model actually came up — so a pending attempt
  could never become "ok" through the UI and save-as-template was unreachable.
  The panel now renders "Mark working" / "Mark failed" (with an optional
  failure-detail field) that POST to `/api/models/{id}/try-stack/{attempt_id}`,
  completing the try → report → save round-trip the panel already documented.
- **publish:github: normalize the SSH deploy-key file to a trailing
  newline before use.** GitLab stores File-variable values without a
  trailing newline, so the mounted key tripped OpenSSH's
  `Load key "...": error in libcrypto` and the GitHub mirror push failed
  with `Permission denied (publickey)` despite a valid, write-enabled
  deploy key. `scripts/publish-github.sh` now rewrites the key to exactly
  one trailing newline (`printf '%s\n' "$(cat ...)"`) regardless of how
  the variable was stored.
- **chat: send `served_model_name` in completion requests instead of
  internal model id.** Fixes `model '<hash>' is not loaded` 404 surfaced
  2026-05-22 by Qwen3.6-27B's distinct served name on
  `https://vllm.example.com/ui/chat`. The bug was latent for the
  entire history of the chat playground because every prior model on
  the fleet had `served_model_name == id`; Qwen3.6-27B is the first
  deployment whose served name differs from its internal id, and every
  chat send started 404'ing the moment it landed. The picker still
  keys on the row's `id` — only the wire field changes. Both
  `handleSubmit` and `handleRegenerate` (the two chat-completion call
  sites in `frontend/src/app/chat/page.tsx`) now resolve
  `modelId → served_model_name` via the already-loaded `loadedModels`
  list before placing it on the request. Regression test
  (`frontend/tests/component/chat-page-served-model-name.test.tsx`)
  pins the wire contract.

### Added
- **Caddy: 308-redirect v1-era bare paths to /ui/\*.** Old bookmarks /
  browser-typed paths (`/login`, `/models`, `/settings`, `/setup`,
  `/stats`, `/tokens`, `/cache`, `/chat`) now 308-redirect to their
  basePath-prefixed equivalent under `/ui/*` instead of hitting Caddy's
  404 fallthrough. 308 preserves request method. Also: `/favicon.ico`
  is aliased (301) to `/ui/icon.svg` so browser-auto requests for a
  favicon resolve against the icon the Next.js UI already ships.

- **New `public_url` runtime setting (#154, subsumes #151).** Sets the
  external base URL used in client-facing snippets (curl examples,
  OpenAI client configs) when the warden sits behind a reverse proxy
  whose external URL differs from the browser's address bar. Leave
  blank to use `window.location.origin` — that's the right default
  for direct-access deployments. Backend validates http/https scheme,
  non-empty netloc, ≤2048 chars, and strips any trailing slash before
  persisting. New frontend helper `getPublicBaseUrl()` (and its
  pure-function inner `_resolvePublicBaseUrl()`) is the single source
  of truth for snippet rendering; defends in depth by falling back to
  origin if a malformed value somehow lands in the row. Migration
  0021 is doc-only (no seed — the key is intentionally unset by
  default).

- **Per-tab key membership contract test (#154).** The frontend
  contract test `settings-tab-membership.test.ts` fails CI if a new
  runtime key is added to `RUNTIME_HINTS` without being placed in
  exactly one of the four tab arrays — guards against fields silently
  disappearing from the UI on future additions.

### Changed
- **Landing page (/_landing) redesigned to podwarden.com visual identity.**
  Retro-dark palette (slate-950 background, emerald-400 accent,
  slate-100 text), DM Sans typography via Google Fonts CDN, Lucide
  shield + Simple Icons GitHub marks inlined as SVG (no external JS).
  Copy rewritten around operator-console positioning ("Self-host vLLM,
  with the operator surface you actually wanted") with a four-card
  feature grid. Load-bearing entry-points (`/ui/`, GitHub repo,
  podwarden.com) preserved; existing contract test still passes,
  plus a new redesign-surface test pins the new copy and visual
  identity tokens.

- **/settings page reorganised into five purpose-grouped tabs (#154).**
  The previous 13-field "Runtime" flat list (one 581-line component) is
  replaced with five purpose-grouped tabs — General, Networking,
  Sessions & Tokens, Maintenance, Model — each composed of titled
  section cards. Each runtime sub-tab owns its own Edit/Save lifecycle
  scoped to that tab's keys, so editing one tab no longer marks the
  entire form dirty. The underlying `GET /api/settings/runtime` fetch
  is deduplicated by SWR across tabs, so the page still issues exactly
  one read per visit. Spec lives at
  `docs/superpowers/specs/2026-05-24-settings-redesign-design.md`.

- **Token rotation renames old row, mints new with original name (#150).**
  Operators previously had to rename-then-mint by hand (three steps,
  collision risk on the typed-back name). Now a single `Rotate` click:
  - Renames the existing row to `"{name} (old N)"` (N = next free slot;
    starts at 1, cascades to `(old 2)`, `(old 3)` … on subsequent
    rotations — gaps in N are NOT reused, so a deleted `(old 1)` does not
    make slot 1 available again).
  - Mints a brand-new row that keeps the ORIGINAL name, inheriting
    `rate_limit_tps` and `priority` from the predecessor.
  - Returns both `name` (active) and `renamed_to` (predecessor's new
    name) so the UI's success modal can tell the operator exactly where
    each token landed.
  - Rejects with **409 Conflict** if the requested row was already
    rotated — the UI disables the button on those rows, but the
    server-side guard defends against direct API callers and racing
    double-clicks. Single SQLite transaction wraps SELECT + UPDATE
    (rename) + INSERT (successor) + UPDATE (rotated_at / revoked_at), so
    a crash mid-flight cannot leave rotation half-applied.
  - **`app/db/repos/tokens.py`** — `rotate()` signature changed: dropped
    the `new_name` parameter (repo now derives names internally) and now
    returns `(new_id, new_plaintext, renamed_to)`. Added private
    `_next_old_suffix()` helper with strict regex parsing and LIKE-
    escape handling so token names containing `_` or `%` cannot
    false-match into someone else's `(old N)` ladder.
  - **`app/tokens/routes_api.py`** — rotate endpoint returns `name` +
    `renamed_to` fields and raises 409 on already-rotated rows. The
    public response shape gained two fields; existing callers reading
    `id` / `plaintext` / `prefix` / `rotated_from` are unaffected.
  - **`frontend/src/components/tokens/rotate-token-dialog.tsx`** —
    success-state footer now reads "New active token: `<name>`. The
    previous token was renamed to `<renamed_to>` …" instead of the
    stale "Rotated from `<old_id>`" copy.
  - **Tests** — repo-layer (`tests/unit/db/test_token_rotate_naming.py`,
    6 cases covering happy-path naming, cascading `(old 2)`, MAX+1 with
    gaps, 409-on-already-rotated, predecessor secret intact during
    grace, and SQL-wildcard escaping). Endpoint-layer additions in
    `tests/unit/tokens/test_rotate_endpoint.py` (3 cases) pin the
    response shape and the 409. Existing rotate tests (4 files)
    updated to drop the `new_name=` argument and assert the new
    naming contract.
  - **Docs** — `docs/operating.md` rotate-token section rewritten to
    show the new response fields and document the 409 case.

### Added
- **Unified-port architecture: single host port `:8080` fronted by Caddy (#155).** A new `caddy:2-alpine` service joins the compose stack as the only host-published port. It fans out `/` → FastAPI `/_landing`, `/ui/*` + `/_next/*` → ui:3000, `/api/*` + `/v1/*` + `/healthz` → api:8080. The `api` and `ui` containers move to `expose:` only (no host mappings) so operators no longer have to remember "did I open 8080 or 3000?" — there's only one port. SSE routes (`/api/chat/*`, `/api/models/{id}/logs/stream`, `/api/stats/stream`) use `flush_interval -1` so chunks flush immediately. Caddy admin API is `admin off` — no port 2019 to lock down. See `deploy/caddy/Caddyfile` for the routing map and `docs/operating.md#unified-port-topology-155` for the operator notes.
- **Public landing page at `/_landing` with opt-out (#155).** New JWT-exempt route returns a single-file HTML page with the project title, a deep link to `/ui`, the source repo, and `podwarden.com`. Caddy rewrites `/` to `/_landing` so an anonymous browser hitting the unified-port root sees something useful. New runtime setting `landing_page_enabled` (boolean, default `true`, restart kind `none`) lets operators disable the page for private deployments; with the toggle off both `/` and `/_landing` return `404 landing page disabled`. UI surfaces the toggle as a new field on `/settings/runtime`. Migration `0020_landing_page_setting.sql` seeds the default.
- **`make smoke` target (#155).** Five-curl liveness check against the live unified-port front-door: `/`, `/_landing`, `/ui/`, `/api/csrf`, `/healthz` — all must return 200. Intended for post-`docker compose up -d` sanity in operator runbooks and CI smoke jobs.
- **System Configuration panel on `/stats` (#148).** New section at the bottom of the stats page surfaces host inventory the operator wants when interpreting the live metrics above it: CPU model + physical/thread counts, total RAM, OS release + kernel, Docker server version + default runtime, and one card per GPU (name, VRAM, driver version, CUDA version). Backed by a new JWT-gated `GET /api/system/info` endpoint that composes from `/proc/cpuinfo`, `/proc/meminfo`, `/etc/os-release`, `uname -r`, `nvidia-smi`, and `docker info`; each source degrades independently (no NVIDIA → `gpus: []`, no docker socket → `available: false`). 60s in-process cache absorbs the page's 30s SWR poll so shell-outs happen at most once per minute.
- **Favicon (#152).** Browser tabs and iOS home-screen now render a
  branded icon instead of the default globe. Ships two assets in
  `frontend/src/app/`: a static `icon.svg` (filled emerald Shield
  matching the navbar wordmark, white "W" centred — Next App Router
  auto-emits `<link rel="icon" type="image/svg+xml">`) and a
  programmatic `apple-icon.tsx` that uses `next/og`'s `ImageResponse`
  to render the same composition as a 180×180 PNG at build time. No
  raster asset is shipped in the repo — the apple-icon is statically
  optimized and cached by Next. Colour pinned to Tailwind `emerald-500`
  (`#10b981`) for consistency with the navbar Shield.

### Changed
- **Next.js UI now lives under `basePath: '/ui'` (#155).** Required so a single Caddy listener can fan out `/` → landing, `/ui/*` → Next, `/api`/`/v1` → FastAPI without path conflicts. `basePath` is baked into the standalone build at image-build time. SSR `BACKEND_URL` for the Next.js server still points at `api:8080` over the compose network; client-side fetches now use relative paths because the browser is same-origin with Caddy. `VW_FRONTEND_ORIGIN` default changes from `http://localhost:3000` to `http://localhost:8080` to match the new front-door.

### Removed
- **Next.js `/api/:path*` and `/v1/:path*` rewrites (#155).** Caddy owns these routes in the unified-port topology and proxies them directly to FastAPI; double-proxying through Next would (a) re-engage the gzip path that #52 had to disable for SSE and (b) bake a hardcoded backend URL into the standalone build. The `/healthz` → `/api/health` Next-internal alias (#48) is retained.
- **Stale "Storage moved to Cache" toast on `/stats` (#153).** The migration banner (carried over from S6 #105) was retired now that the cache surface has been on its own `/cache` route for multiple releases. The toast, its `vw.storage.toast.dismissed` localStorage key, and the lingering `next/link` + `lucide-react` imports it pulled in have all been removed.

### Fixed
- **Token-modal "Copy" button works on non-secure (HTTP) origins (#149).**
  Operator on d5 (production, served as raw HTTP over Tailscale at
  `http://10.10.0.187:8080`) hit *"Copy failed — select and copy the
  token manually"* every time on freshly-minted tokens, because
  `navigator.clipboard.writeText()` is undefined outside HTTPS/localhost.
  - **`copyToClipboard` (`frontend/src/lib/utils.ts`)** — now tries the
    async Clipboard API only when `window.isSecureContext` is true, then
    falls back to a hidden-`<textarea>` + `document.execCommand("copy")`.
    `execCommand`'s boolean return value is now checked (it was
    previously swallowed), and synchronous throws from hardened browsers
    are caught — the helper rejects only when BOTH paths fail, so the
    "select manually" hint is reserved for actual double-failure.
  - **All copy buttons refactored to use the shared helper** —
    `create-token-dialog.tsx`, `rotate-token-dialog.tsx`,
    `chat/message-list.tsx`, and the Effective-argv panel
    (`app/models/[id]/settings/page.tsx`). Previously each call site
    inlined its own `navigator.clipboard?.writeText(…)` with no
    fallback; now all four benefit from the secure-context check + the
    execCommand fallback in one place.
  - **Tests (`frontend/tests/component/copy-to-clipboard.test.ts`,
    additions in `tokens.test.tsx`)** — pin three branches called out by
    the issue AC: (a) clipboard available + `writeText` succeeds, (b)
    clipboard undefined + `execCommand` returns true, (c) clipboard
    undefined + `execCommand` returns false. Plus dialog-level tests
    that drive the success/failure toast paths end-to-end through both
    Create and Rotate modals to prevent a regression that strips the
    fallback.

## [v2026.05.23.1] — 2026-05-23

**Highlights — `2026-05 overhaul` epic close-out (S1-S9).** Stats v2
replaces the v1 GPU table with an operator cockpit (four current-value
tiles + 2×2 chart grid for VRAM / GPU util / power / tokens) plus a
per-key tokens table, fed by a new power-samples pipeline (5s sampler,
minute-bucket accumulator). New `/chat` playground proxies streaming
completions through the warden's JWT-authed SSE path; supervisor gains
a deterministic pre-spawn health-wait so cold loads no longer race a
half-up vLLM. API token rotation gains a configurable grace window so
callers can swap credentials without downtime. Bench v2 removed (CTO
decision; #117). First release tag of the epic — successor of
v2026.05.20.1.

### Added
- **Stats v2 frontend — rebuilt `/stats` page (epic/overhaul S7, #124).**
  Frontend half of the stats redesign. Consumes the new `/api/stats/v2/*`
  contract documented below.
  - **Page rebuild (`frontend/src/app/stats/page.tsx`)** — replaces the
    legacy v1 GPU table + ad-hoc cards with one operator-cockpit layout:
    a current-row of four tiles (VRAM used / total, GPU util max, summed
    power, TPS over the last full minute) sitting above a 2×2 chart grid
    (VRAM, GPU util, power, prompt+completion tokens) and a sortable
    per-key tokens table. Range selector (1h / 6h / 24h / 7d) drives both
    `/overview` and `/tokens-per-key` in one render pass; selection
    persists to `localStorage` under `vw.stats.range` and survives a
    reload. Active-model strip renders as emerald chips when at least one
    model is `loaded`; hidden entirely otherwise. SWR polls at 30s and
    pauses when the tab is hidden. The S6 "Storage moved to Cache" toast
    is preserved verbatim.
  - **`StatCard` (`frontend/src/components/stat-card.tsx`)** — reusable
    tile (label + value + optional unit + optional hint + optional
    `title` tooltip). Single source of truth so the current-row stays
    visually consistent with future surfaces (next slice's overview
    rail). Test IDs `stat-card`, `stat-card-unit`, `stat-card-hint`.
  - **`usePersistedRange` (`frontend/src/lib/use-persisted-range.ts`)** —
    SSR-safe `localStorage`-backed range hook. Returns the fallback
    until first effect-tick so server and first-paint markup agree
    (no hydration mismatch), then upgrades to the stored value on
    mount. Validates the stored string against the `StatsRange` union
    and silently swallows `localStorage`-unavailable errors.
  - **`v2-charts.tsx`** — four Recharts panels (`VramChart`, `UtilChart`,
    `PowerChart`, `TokensChart`) sharing a `ChartShell` empty-state
    wrapper. Time-scaled XAxis pinned to the range window so a sparse
    series doesn't collapse to a fake-wide chart. Animations disabled
    everywhere — a 30s poll on a flickering chart is exhausting. The
    `PowerChart` `supported` prop distinguishes "no samples yet" from
    "GPU can't report" so the operator stops waiting on a virtualised
    host.
  - **`lib/stats-v2.ts`** — frontend types mirroring the v2 docstrings
    plus formatters (`mibToGib`, `formatWatts`, `formatTps`) and the
    `withTs` helper that attaches an epoch-ms column for Recharts'
    time-scaled axis.
  - **API types regenerated** — `frontend/src/lib/api-types.generated.ts`
    picks up the two new v2 paths (response bodies arrive as bare
    objects since the backend routes return raw dicts; the page locks
    the inner shape via component tests).
- **Stats v2 backend — power telemetry + per-key tokens (epic/overhaul S7, #124).**
  Backend half of the stats redesign. The frontend half lands separately in
  dev-2's slice (same epic) and consumes the contract documented below.
  - **New migration `0019_power_samples.sql`** — per-(gpu, minute) watts
    accumulator. Schema is a write-path aggregator (`watts_sum` + `samples`
    counter) keyed on `(gpu_idx, minute)`; the read side recovers a true
    minute-average via `watts_sum / NULLIF(samples, 0)` rather than
    last-write-wins, so a 12-tick minute yields a representative number
    rather than "whichever 5s tick landed last". Retention follows
    `gpu_samples` (7d) and is handled by the existing stats pruner.
    Manual rollback recipe documented in the migration header.
  - **GPU sampler — single-pass util + mem + power** (`app/system/gpu.py`,
    `app/runtime/stats_sampler.py`). The existing `nvidia-smi` query
    now also fetches `power.draw`; one subprocess invocation per tick
    produces util, mem, and power together (no second pass). Sampler
    cadence drops from 60s to 5s (override via `VW_STATS_SAMPLER_INTERVAL_S`
    env var) so each minute bucket sees ~12 samples, then writes
    `gpu_samples` (last-write-wins) and `power_samples` (accumulator).
    Cards that report `[Not Supported]` / `[N/A]` for `power.draw` flow
    through as `power_w=None` and the sampler skips the power write for
    that tick — util/mem still land normally; no zero-row corruption.
  - **`GET /api/stats/v2/overview?range=1h|6h|24h|7d`** — JWT-gated.
    Returns the dashboard payload in one round-trip: a `current` snapshot
    (most-recent-minute VRAM%, util%, summed power W, TPS over the last
    full minute), `active_models` (only `status='loaded'`), and `series`
    with per-minute `vram` / `util` / `power` / `tokens` arrays.
    Power series sums per-GPU averages across the box; degenerate
    `samples=0` rows are filtered.
  - **`GET /api/stats/v2/tokens-per-key?range=...`** — JWT-gated.
    LEFT JOINs `token_usage_minute` onto `api_tokens` so the response
    carries the human-readable `name` + `prefix` alongside `token_id`.
    Tokens with no usage in the window are omitted; orphan rows
    (usage exists but `api_tokens` row was deleted) surface as
    `"(unknown)"` rather than being dropped, preserving historical
    visibility. Rows sorted by `total_tokens DESC`.
  - **CTO decision #7 honoured** — v1 endpoints (`/api/stats/models`,
    `/api/stats/gpus`) coexist with v2 untouched; both response shapes
    are stable.

### Fixed
- **Tokenizer cache no longer leaks on unload (epic/overhaul S7, #124).**
  `TokenizerCache` (in `app/proxy/tokenizers.py`) held one fully-loaded
  `AutoTokenizer` per `(hf_repo, trust_remote_code)` tuple with no
  eviction path — a long-lived warden that cycled models would accumulate
  one entry per ever-seen repo and eventually OOM. Now exposes
  `evict(hf_repo)` (drops both trust-remote-code variants) and `size()`;
  the model unload route (`POST /api/models/{id}/unload`) wires the
  eviction into its success tail, so unload reliably frees the entry.

- **`/cache` page + chained model delete (#105, #117).**
  Frontend-only re-home of the Storage section.
  - New top-level `/cache` route lifts the HF cache surface out of `/stats`
    so it sits alongside Models / Tokens. Reuses `CacheTable` and
    `CacheGcButton` verbatim; the page adds a header, summary tiles
    (`Total cached` / `Repos` / `Orphans`), and a first-class empty-state
    hero card. SWR polls `GET /api/cache/models` every 30s and pauses when
    the tab is hidden. No new backend endpoints.
  - **Delete-model confirm modal** with an opt-in "Also free cache"
    checkbox. When ticked, the modal chains
    `DELETE /api/models/{id}` → `DELETE /api/cache/models/{repo}?force=true`
    in that order. Step 2 is best-effort; a 404 is treated as success
    (cache was already gone), and any other failure surfaces an inline
    notice ("Model removed; cache delete failed — visit /cache to retry.")
    while still notifying the parent that the row was deleted. If step 1
    fails, step 2 never runs. Closes #105.
  - Nav-bar gains a single Cache entry (HardDrive icon) between Stats
    and Settings.
  - One-time toast on `/stats` ("Storage moved to Cache") with a Dismiss
    button — persisted in `localStorage` as `vw.storage.toast.dismissed`
    so operators see the signpost exactly once per browser.

- **Chat playground at `/chat` (epic/overhaul S8, #117, closes #125).** New
  session-only chat surface for probing loaded models:
  - Model picker (filtered to `status === 'loaded'`, auto-selects first
    available and resets when the current selection unloads),
    temperature (0..2) and max_tokens (1..8192) sliders, Send/Stop
    button that flips role based on phase. Enter sends, Shift+Enter
    inserts a newline, Esc aborts an in-flight stream.
  - Streaming via a new server-side proxy at `POST /api/chat/completions`
    (JWT- and CSRF-gated by the existing middleware) which forwards to
    the local `/v1/chat/completions` over httpx + SSE; the proxy
    forges the `Authorization: Bearer …` header server-side from a
    plaintext-cached `vw-playground` system token so the bearer never
    enters browser memory.
  - `POST /api/chat/playground/ensure` mints and caches the
    `vw-playground` token on first visit (idempotent — handles
    fresh-DB, stale-cache, and DB-row-without-plaintext-cache
    recovery paths by sweeping stale rows and re-minting).
  - Playwright happy-path polls the new admin counter (below) to
    verify abort cleanup — the server-side finally block in the
    streaming generator decrements the counter on socket close,
    fencing against the SSE-leak risk called out in the slice plan.
  - Operator-cockpit aesthetic per the `frontend-design` skill output:
    monospace body, emerald-400 accents on slate-900, role rails
    instead of bubbles, blinking emerald cursor block while streaming,
    hover-only Copy + Regenerate-last on assistant messages.
  - Conversation history is React state only — refresh discards it.
    Neither localStorage nor the DB are touched. (Explicit S8
    contract — playground is a probe, not a journal.)
- **`GET /api/admin/active-requests` endpoint (epic/overhaul S8, #117, closes
  #125).** JWT-gated live counter of in-flight chat-completion streams —
  returns `{count: int}`. Operator-facing diagnostic for triaging stuck SSE
  proxies in production, and the polling target the Playwright happy-path
  uses to assert abort cleanup converges back to 0.
- **Add-model + Tune UX overhaul (epic/overhaul S4, #117).** Frontend-led
  rework of the two highest-traffic pages, plus thin backend support:
  - Settings page rebuilt as a sectioned instrument panel (Identity /
    Memory / Compute / Advanced) with collapsible discrete sections —
    each section's subtitle shows a live "N unsaved" count, surfacing
    dirty state at a glance instead of buried per-field.
  - Empty-state hero on `/models`. When no models are registered the
    page now centres a dashed-border card with a short explanation of
    what adding a model does, replacing the one-line placeholder. Part
    of #117 (S4 design principle #2).
  - New **Presets** strip: four curated chips
    (`a4000-tight-awq`, `h100-single-shot`, `dev-tiny`, `moe-balanced`)
    served from `GET /api/presets` (auth-required, JSON-backed by
    `app/presets/builtin.json`). Clicking a chip opens a confirm
    popover showing a per-key diff against the current draft. Apply
    merges the sparse preset into the draft — never auto-saves.
  - New **Suggest values** inline panel in the Memory section. Reuses
    the existing `GET /api/models/{id}/suggest-config` (S3) and renders
    the suggestion as a diff list. Apply / Dismiss are explicit; the
    `disclaimer` field is shown above the diff as rationale. Null
    fields are filtered out so they don't render as bogus "→ null"
    diff rows.
  - New **Effective argv** faux-terminal panel at the bottom of the
    settings page, backed by a new `GET /api/models/{id}/effective-argv`
    endpoint that returns the exact `vllm serve` argv that would be
    used for the *persisted* settings (preview port 10000). Re-fetched
    after every Save so the operator gets a one-second feedback loop:
    edit → save → see argv update. Changed tokens flash emerald via
    the existing `row-flash-emerald` keyframe; honours
    `prefers-reduced-motion`.
  - Add-model modal **shard grouping (#112).** Sharded weights
    (`model-00001-of-NNNNN.safetensors`) collapse into a single
    disclosure-triangle family row showing the total shard count.
    Single-shard and loose single-file weights stay as regular rows
    (no disclosure noise). Members appear when the row is expanded.
  - Add-model modal **per-file GGUF arch warnings (#101).**
    `discover_repo_files` now emits `gguf_arch_unknown` /
    `gguf_arch_unsupported` warnings against the new `KNOWN_GGUF_ARCHES`
    allowlist (21 families — see operating.md
    `#supported-gguf-architectures`). Arch is inferred from
    `config.general.architecture` first, then `model_type`, then
    `architectures[0]`, then a filename heuristic with longest-match-
    first ordering. The modal renders the warning inline against the
    offending file row with the inferred arch surfaced for
    debuggability.
  - Closes #101 (GGUF arch warnings) and #112 (shard grouping). Part
    of #117 (overhaul epic S4).
- **Live header-metrics widget (epic/overhaul S2).** Compact instrument
  cluster in the nav bar — VRAM%, GPU%, and the currently-loaded
  model name — refreshed via a single shared SSE stream
  (`GET /api/header/metrics/stream`, default 2s cadence, configurable
  through `VW_HEADER_METRICS_INTERVAL_S`, floored at 0.5s). The widget
  uses one `EventSource` per browser tab regardless of how many times
  `<NavBar />` re-mounts (React StrictMode, hot reload, route change),
  and shares `app.state.gpu_probe_cache` with `/api/system/gpus` so
  multi-tab operators don't double `nvidia-smi` load. Accent colour
  encodes status: slate=idle, emerald=model loaded, amber=probe error
  or reconnecting, red=terminal/401. Hidden on `/login` and `/setup`
  (server-side, via the same exact-match guard NavBar uses) plus on
  viewports below md (`hidden md:inline-flex`). Closes #67 (UI
  build-args), part of #117 (overhaul epic S2).
- **`build:ui` CI job and Dockerfile `ARG VW_BUILD_VERSION` /
  `ARG VW_BUILD_SHA`.** The frontend image is now built with the same
  identity build-args as the backend so the nav footer never falls
  back to "dev" / "unknown" on CI builds. Both images carry matching
  tags for a given commit. Closes #67.
- **`GET /api/models/{id}/suggest-config` (epic/overhaul S3, #113).** New
  endpoint returns a STARTING-POINT configuration blob for a registered
  model — `gpu_memory_utilization`, `max_model_len`, and `kv_cache_dtype`
  — explicitly NEVER auto-applied. Heuristics in `app/models/suggest.py`:
  `gpu_memory_utilization=0.92` (per-process util fraction);
  `max_model_len` from `config.max_position_embeddings` (None when
  absent — no guessing); `kv_cache_dtype='fp8'` iff the model looks
  AWQ-quantized (any of `config.quantization_config.quant_method=='awq'`,
  hf_repo name marker, or filename marker). Payload always carries a
  `disclaimer` field with the phrasing "starting points only and are
  never auto-applied" so a downstream consumer that auto-applies the
  values ignores a contract red flag. Closes #113.

### Changed
- **`/stats` Storage section removed (#105).** The HF
  cache table and GC button no longer render on `/stats`; visit `/cache`
  instead. A one-time `localStorage`-keyed toast on `/stats` is the
  bridge for operators who land on the old location out of muscle
  memory. Closes #105 (frontend half — the backend endpoints were
  already in place).
- **`/stats` page rebuilt on the v2 contract (epic/overhaul S7, #124).**
  Replaces the v1 GPU table and ad-hoc cards with persistent range, four
  time-series charts, four current-value tiles, active-models strip, and
  tokens-per-key table. See the corresponding `### Added` bullet above
  for the full feature breakdown.
- **Settings PATCH allowlist is now derived from `ModelRow` (epic/overhaul
  S3, #110).** `app/settings/routes_api.py` no longer hand-maintains the
  `_PATCHABLE_MODEL_FIELDS` set. The allowlist is computed once at
  import time as `dataclasses.fields(ModelRow) - _NEVER_PATCH`, where
  `_NEVER_PATCH` is the explicit blocklist of lifecycle-owned columns
  (`id`, `status`, `pulled_bytes`, `pulled_total`, `last_error`,
  `updated_at`, `created_at`). A startup-time drift guard raises if any
  name in `_NEVER_PATCH` is not present on `ModelRow` (modulo the
  documented `created_at` SQL-only exception). Side effect, intended:
  `filename`, `parallelism_strategy`, `max_batch_size`, `hf_config_repo`,
  and `tokenizer_repo` (added by #85 / #106) are now patchable; the
  hand-maintained list never picked them up. Plan called for
  `ModelRow.__fields__` (Pydantic accessor) — actual dataclass shape
  used `dataclasses.fields()`. Closes #110.
- **`wait_for_health` timeout injectable via module constant (epic/overhaul
  S3, #99).** `app/runtime/supervisor.py` now exports
  `DEFAULT_HEALTH_TIMEOUT_S: float = 600.0` and `wait_for_health` takes
  `timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S` so callers can override
  without parroting the magic number. The production load runner already
  passes `settings.load_timeout_s` explicitly; this only changes
  ad-hoc callers (integration tests, scripts) and centralizes the
  single source of truth.

### Fixed
- Quarantined flaky `test_on_exit_callback_flips_status_to_failed_and_releases_port` until supervisor state-leak fix lands. Develop pipeline #6621 blocker. Related: #127.
- Quarantined 2 sibling `test_supervisor_real_subprocess.py` flakes (`test_supervisor_spawns_real_fake_process` + `test_supervisor_extra_env_reaches_subprocess`) — same `UnloadRefused: state is LOADING, not READY` family as the sibling above. Pre-emptive on the same MR per the 2026-05-23 "don't ship known-failing siblings" lesson. Tracked under #143; restore once the supervisor state-leak fix lands.
- **Nav-bar double "v" version prefix (#83).** When the backend
  returned a CalVer tag that already included a leading `v` (e.g.
  `v2026.05.19.1`), the nav footer prefixed its own `v`, producing
  `vv2026.05.19.1`. `formatVersion` now emits the `version` field
  verbatim — the fallback string retains the literal `v? · ?` for the
  "could not load" case so the shape stays unambiguous. Regression
  pinned in `frontend/tests/component/nav-bar.test.tsx`.
- **Exact-match unauthenticated-route guard (#39).** NavBar previously
  used `path.startsWith('/login')` to suppress the nav chrome on
  unauthenticated routes, which silently swallowed any path beginning
  with `/login` (a hypothetical `/login-help` marketing page would
  have rendered as unauthenticated). The guard now exact-matches
  `/login`, `/login/`, `/setup`, `/setup/` and retains the
  `/setup/*` wizard subtree carve-out. `auth-fetch.ts` got the
  matching exact-match check on its 401 redirect path so it doesn't
  loop-redirect from a hypothetical `/login-help` either. Pinned in
  `nav-bar.test.tsx` (5 new cases).
- **Flaky `401 invalid credentials` in JWT login fixtures (#55).**
  `tests/conftest.py` now ships `seed_admin_user` and `jwt_login`
  shared helpers that close the WAL frame propagation race between
  the sync `sqlite3` admin seed and the async `aiosqlite` route
  handler. Primary fix is S5's `seed_admin_user()` (adopted verbatim
  during the S2→develop reconcile): `PRAGMA journal_mode = WAL`
  matching the app pool, an explicit `BEGIN IMMEDIATE` for the INSERT,
  and a `PRAGMA wal_checkpoint(FULL)` before close so the frame is
  visible to every subsequent reader. Defence in depth is `jwt_login`'s
  bounded retry-on-401 (`LOGIN_RETRY_MAX = 5`, exponential backoff,
  returns immediately on any non-401). 18 unit tests + 1 integration
  test migrated to the shared helpers (4 token-auth and seed-only
  tests skipped — no race).
- **`mark_runtime_dead_on_startup` now wipes `pulling` rows + zeros pull
  progress (epic/overhaul S3, #11).** Before this change, a process
  killed mid-pull would leave the row stuck at `status='pulling'`
  forever with stale `pulled_bytes` / `pulled_total` — the only recovery
  was a manual SQL update. The startup sweeper now flips `pulling` rows
  to `failed` alongside the existing `loaded`/`loading`/`unloading`
  sweep, and uses `CASE WHEN status = 'pulling' THEN 0 ELSE pulled_*
  END` so progress counters are zeroed for `pulling` rows only — the
  `pulled_total` on a `loaded` row is the persisted weights size and
  must NOT be wiped. Closes #11.
- **`health_ok` column now written after warmup probe succeeds
  (epic/overhaul S3, #29).** The load runner in
  `app/models/routes_api.py` calls `RuntimeRepo.update_health(model_id,
  True, <utcnow>)` immediately after `Supervisor.mark_ready()` so the
  `runtime` row reflects the actual health state. Before, `health_ok`
  stayed at its previous value (frequently `0`) even when the model
  was successfully serving; the column is read by the live status
  badge on `/models/[id]`, so the UI was silently degraded.
- **Operator guidance: `VW_CONTAINER_GPU_COUNT` (epic/overhaul S3, #46).**
  New 'Container GPU isolation' chapter in `docs/operating.md` covers
  what the variable does (the container's stated GPU-slice size that
  the wizard, validators, and probes index against), how to set it for
  whole-host / split / single-GPU deployments, a worked docker compose
  example, and common misconfig symptoms (count > exposed → silent
  until pull/load CUDA-error; count < exposed → wizard hides GPUs).
  Explicit decoupling note: warden never reads
  `NVIDIA_VISIBLE_DEVICES`; operators reconcile the two manually.

### Removed
- **Bench v2 subsystem fully excised (epic/overhaul S1).** Every
  Bench-v2 source file, persistence table, HTTP route, UI surface,
  and test is gone. Specifically: `app/bench/` (supervisor, runner,
  cli, harness, events, internal_routes, best, repos.bench), the
  `/api/bench/*` and `/api/models/{id}/benchmark` endpoints, the
  `/benchmarks` UI page + `/models/[id]` Benchmark tab,
  `FromBenchmarkChip` and its consumers on the settings page, the
  `BENCH_HEALTH_WAIT_S` env var + `bench_health_wait_s` setting,
  `_is_bench_internal_path` in CSRF middleware, the
  `integration-tests` CI job, `scripts/publish_to_hub.py` (dead
  publisher), `scripts/build_long_corpus.py` (bench fixture
  generator), and the bench-v2 design spec at
  `docs/superpowers/specs/2026-05-13-vllm-warden-benchmark-v2-design.md`.
  Migration `0017_drop_bench.sql` drops the `bench_run`,
  `bench_load_config_attempt`, and `bench_cell` tables (children
  first; idempotent via `DROP TABLE IF EXISTS`). The 5xx envelope
  enricher (`app/proxy/envelope_hint.py`) was rewritten as a slim
  ~80-LOC stub that attaches `models.last_error` to OpenAI-style
  error envelopes when the upstream returns 5xx — replacing the
  bench-coupled hint engine. Closes #14, #15, #16, #17, #20, #24,
  #25, #31, #47. (~23k source / ~11.6k test LOC removed; rationale:
  external benchmarking tool, see overhaul plan §S1)
- **JWT re-login required after this rollout.** No auth-cookie
  schema change, but the bench WS-event channel previously held
  long-lived browser sessions open; operators with stale tabs
  pointing at `/benchmarks` will see 404 on next click and should
  refresh. New nav menu drops the Benchmarks entry (Models /
  Tokens / Stats / Settings remain).
- **Migration 0017 caveat.** The migration uses
  `DROP TABLE IF EXISTS` so it is idempotent and safe against
  pre-overhaul DBs that may have only some of the three tables
  (e.g. a partially-failed prior install). Downgrading past 0017
  is not supported — the bench code is gone, so there is nothing
  to recreate the tables for. Migration 0016 is intentionally
  reserved (the overhaul plan slot it lived in moved to a later
  slice; numbering stays monotonic).

### Changed
- **Pin `transformers <5.7` and `huggingface_hub <1.16`** in
  `requirements-dev.txt`. Transformers 5.7.0 + hub 1.16.x broke
  pytest-asyncio fixture collection (AttributeError on
  `tests/unit/system/test_hf.py`: `'function' object has no attribute
  '__func__'`). Locked to versions matching the vllm/vllm-openai 0.20.0
  base image (5.6.2 / 1.12.0). Unrelated to bench removal — surfaced by
  fresh pip install on this branch's CI re-run.
- **`app/proxy/envelope_hint.py` replaced with `models.last_error` stub
  (~80 LOC).** The bench-coupled 5xx hint engine was rewritten as a slim
  stub that attaches `models.last_error` from the model row to OpenAI-style
  error envelopes when the upstream vLLM returns 5xx. No regression to bare
  500s — operators see the last known error from the model row.
- **Supervisor refuses unload during model load/warmup.** The vLLM
  subprocess lifecycle now goes `LOADING → WARMING → READY → UNLOADING`
  inside the supervisor. Calling `POST /api/models/{id}/unload` while
  the model is not yet `READY` returns 409 with the current state in
  the body. To force-terminate a stuck or still-warming model, use
  `POST /api/models/{id}/unload?force=true`. A warmup verification
  probe (`POST /v1/completions max_tokens=1`) runs after vLLM's
  `/health` returns 200 — DB status only flips to `loaded` once that
  probe succeeds, closing the race window where Qwen3-VL's
  `_warmup_mm_processor` was still running but the row looked
  serviceable. When the probe or health-wait fails, the row is marked
  `failed` and the subprocess is **left running** holding its GPUs
  until an explicit `?force=true` unload — operators must check
  `last_error` (contains `subprocess still holding`) and decide
  whether to retry or release. The `warmup_probe_timeout_s` setting
  (default 60s, env `VW_WARMUP_PROBE_TIMEOUT_S`) controls the probe
  budget. See spec
  `docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md`.

### Fixed
- **#104 — Flaky `tmp_data_dir` fixture race in test infra.** `tests/conftest.py:seed_admin_user()`
  pre-creates the `data/` parent directory before yielding the bind-mount path, so concurrent
  test-worker fixture instantiation can no longer race the `mkdir`. Pinned by a 10-iteration
  parametrized stability test (`tests/unit/tokens/test_rotate_endpoint_104_stability.py`)
  so the flake cannot silently return. (S5 scope — surfaced while extending the token test suite.)
- **Qwen3-VL crash loop during multimodal warmup.** Three independent bugs
  surfaced loading `Qwen3.6-35B-A3B-AWQ-4bit` on a fresh warden host: (1)
  `extra_args` from the model row was defined in the schema but never
  forwarded to `vllm serve` argv, so per-model flags like
  `--enable-auto-tool-choice --tool-call-parser qwen3_xml` were silently
  dropped (`app/runtime/cmd_builder.py`); (2) `_extract_prompt` in the
  proxy assumed `message.content` was always a string, raising on the
  OpenAI multimodal shape where `content` is a list of
  `{"type":"text"|"image_url", ...}` blocks (`app/proxy/routes.py`); (3)
  HuggingFace Hub's revalidation HEAD requests were unauthenticated
  because the new `HF_TOKEN` env var (the `huggingface_hub` >=0.23 name)
  was not set in the subprocess env — only the legacy
  `HUGGING_FACE_HUB_TOKEN` was — making cold multimodal warmup slow
  enough to expand the SIGTERM race window (`app/runtime/env_builder.py`,
  also locked against extra_env override).

### Added
- **Per-token rate limit, priority, and 24h usage rollup (S5, closes #104).**
  Migration `0018_tokens_rate_priority.sql` adds two columns to `api_tokens`:
  `rate_limit_tps INTEGER NULL` (NULL = unlimited; positive integer = sliding
  10-second budget in tokens/sec) and `priority INTEGER NOT NULL DEFAULT 5`
  bounded 0..9 via BEFORE INSERT/UPDATE triggers. A new `token_usage_minute`
  table holds per-token, per-minute request/prompt/completion counters, and
  the proxy success path now writes one row per request alongside the
  existing `/counters` + `/model_samples` writes. New endpoints:
  `PATCH /api/tokens/{id}` (update rate/priority), `GET /api/tokens/{id}/usage?range={1h|24h|7d}`
  (minute-bucketed rollup + totals), and `POST /api/tokens/{id}/test` (lookup
  + scope projection + proxy liveness check for the UI "Test token" button).
  The proxy adds a sliding-window `TokenRateLimiter` (rejected requests do
  NOT consume budget — otherwise oversized prompts lock the token out
  indefinitely) and a STRICT `PriorityScheduler` (priority-9 always served
  before priority-0). Rate-limited requests get **HTTP 429** to match
  OpenAI client retry semantics. The window length is operator-tunable via
  `VW_RATE_LIMIT_WINDOW_S`. **Starvation is by design under STRICT
  scheduling** — priority-0 tokens can wait indefinitely under sustained
  priority-9 pressure; documented in `docs/operating.md` and surfaced in
  the create-dialog form helper text + the priority column tooltip. The
  `/tokens` UI page grew three columns (Rate / Priority / Last 24h) and
  a Test button per row.
- **HF cache management endpoints + /stats Storage section (#114).** New
  Storage section on `/stats` lists every cached HuggingFace repo under the
  configured cache root with size, last-used age, and the model rows that
  own it (or "orphan" if none). Per-row Delete refuses if the repo backs an
  active model (`loaded`/`loading`/`unloading`/`pulling`) and requires
  `?force=true` when the row is `pulled`/`idle` but not active. A
  "Garbage-collect" button shows a dry-run preview (orphans + failed rows
  older than 24h by default) before the destructive call, so the operator
  confirms the exact bytes/repos that will be freed. Three new endpoints:
  `GET /api/cache/models`, `DELETE /api/cache/models/{repo:path}`,
  `POST /api/cache/models/gc?dry_run={bool}&failed_older_than_hours={int}`.
  All gated by `require_jwt`; destructive ones additionally validate
  `repo` against `^[\w.-]+/[\w.-]+$` (no `..`, single slash) at the
  boundary and 400 on traversal-shaped input. Filesystem walks run on
  `asyncio.to_thread` so a multi-GiB rmtree never blocks the event loop.

### Fixed
- **#115** Qwen3.5 / Qwen3.6 MoE GGUFs (e.g. `unsloth/Qwen3.6-35B-A3B-GGUF`) now
  load. Companion to #107: that earlier patch handled the dense `qwen3_5 → qwen35`
  rename, but the MoE variant declares `model_type: "qwen3_5_moe"` /
  `architectures: ["Qwen3_5MoeForConditionalGeneration"]` and hit the same
  `Unknown gguf model_type: qwen3_5_moe` failure at
  `vllm/model_executor/model_loader/gguf_loader.py:208`. vLLM v0.20.0 already
  ships the `Qwen3_5MoeForConditionalGeneration` model class and the bundled
  gguf-py already has `MODEL_ARCH.QWEN35MOE='qwen35moe'` with the correct fused
  `experts.gate_up_proj` / `experts.down_proj` tensor map that the model class
  expects — only the rename was missing. Dockerfile now runs a third `sed -i`
  (right after the #107 RUN) inserting `if model_type == "qwen3_5_moe":
  model_type = "qwen35moe"`. We intentionally do NOT install a manual
  `gguf_to_hf_name_map` block (the existing `qwen2_moe`/`qwen3_moe` branch
  targets the older split-expert layout `experts.X.gate_proj/up_proj/down_proj`
  which the new fused qwen3_5 MoE doesn't use; `gguf.get_tensor_name_map(
  MODEL_ARCH.QWEN35MOE, ...)` covers the new layout natively). The RUN is
  guarded by `grep -q` on the post-patch marker and `py_compile`s the file so a
  broken sed fails the image build loudly. No upstream PR yet — vLLM PR #38140
  covers `qwen3_5` dense only and is still OPEN. Remove this RUN + #107 + #108
  RUNs when we bump the vLLM base past a release that adds the MoE rename.
- **#111** Sharded HF safetensors pull only fetched the first shard. Picking
  a `model-NNNNN-of-NNNNN.safetensors` member in the Add Model wizard now
  expands the HF `allow_patterns` to cover all sibling shards plus
  `model.safetensors.index.json` (the vLLM weight_map). The shard-set regex
  that fit-preview already used (`app/models/routes_api.py` `_shard_glob_for`)
  is lifted to a shared module `app/models/sharding.py` and reused by
  `allow_patterns_for` so both code paths agree on what "the whole shard
  family" means. Affects every sharded repo — AWQ/GPTQ/AutoRound/FP8/BF16/GGUF
  — broken since v2026.05.19.3 (#85). Fixes #111.
- **#108** Multimodal Qwen3.5 / Qwen3.6 GGUFs (e.g. `unsloth/Qwen3.6-27B-GGUF`) now load fully without crashing.
  After #107 renamed the architecture so `Qwen3_5ForConditionalGeneration`
  resolves, the next line of vLLM's `gguf_loader.py` (`vision_num_layers =
  config.vision_config.num_hidden_layers`, line 214 in vLLM 0.20.0) crashed
  with `AttributeError: 'Qwen3_5VisionConfig' object has no attribute
  'num_hidden_layers'`. The new `Qwen3_5VisionConfig` (transformers 5.6.2,
  `transformers/models/qwen3_5/configuration_qwen3_5.py:118-140`) follows the
  Qwen2.5-VL precedent and stores the vision layer count as `depth: int = 27`
  with no `num_hidden_layers` shim. The Dockerfile now runs a second `sed -i`
  (right after the #107 RUN) that rewrites that one line to
  `vision_num_layers = getattr(config.vision_config, "num_hidden_layers",
  None) or config.vision_config.depth` so legacy vision configs that DO
  expose `num_hidden_layers` are unaffected and the new Qwen3.5 family
  picks up `depth` as the fallback. The RUN is guarded by a `grep -q` on
  the post-patch marker so rebuilds re-run the step idempotently, and
  `py_compile` runs at the end so a broken sed fails the image build
  loudly. Same upstream PR ([vllm#38140](https://github.com/vllm-project/vllm/pull/38140))
  is the long-term fix — drop both RUN steps when we bump the vLLM base
  image past that merge.
- **#107** Qwen3.5 / Qwen3.6 GGUFs (e.g. `unsloth/Qwen3.6-27B-GGUF`) now load.
  vLLM 0.20.0's `gguf_loader.py` ships a model_type translation map for
  `qwen2_moe` / `qwen3_moe` / `gemma3_text` / `cohere` / `deepseek_v3` etc. but
  is missing the `qwen3_5` → `qwen35` rename, so any GGUF whose HF config
  declares `model_type: "qwen3_5"` died at load with `Unknown gguf model_type:
  qwen3_5`. Upstream fix lives in
  [vllm#38140](https://github.com/vllm-project/vllm/pull/38140) but the PR is
  OPEN — no released vLLM version contains it. As a workaround the Dockerfile
  now runs a one-line `sed -i` against the installed
  `vllm/model_executor/model_loader/gguf_loader.py` at image build time,
  inserting an `if model_type == "qwen3_5": model_type = "qwen35"` block right
  after the existing `gemma3_text` rename. The RUN step also greps the result
  and `py_compile`s the patched file so a broken sed fails the image build
  loudly. Stale `v0.6.3` comment on the `FROM` line was corrected to `v0.20.0`
  to match the pinned digest. Remove the patch + RUN once upstream PR is
  merged and we bump the vLLM base image to a release that contains it.
- **#106** GGUF deployments whose quantized repo omits `config.json` (the
  common unsloth republish shape) now boot. `app/runtime/cmd_builder.py` emits
  `--hf-config-path <hf_config_repo>` and `--tokenizer <tokenizer_repo>` when
  the new columns are set; migration `app/db/sql/0015_models_hf_config_tokenizer.sql`
  adds nullable `hf_config_repo` and `tokenizer_repo` to `models`; the
  Pydantic schema (`app/models/schemas.py`) validates them with the same
  `^[\w.-]+/[\w.-]+$` owner/name pattern as `hf_repo` and a
  `mode="before"` field validator coerces empty strings to `None` so a
  cleared FE field round-trips. The Add Model wizard (`frontend/src/components/models/add-model-modal.tsx`)
  grows a "Base repo (config + tokenizer)" Input in the Advanced section
  that fans out to both fields by default, plus a nested `<details>`
  "Override tokenizer separately" for the rare split case. Legacy / non-GGUF
  rows are unaffected — both columns stay `NULL` and the cmd builder skips
  the flags. This closes the `--tokenizer` / `--hf-config-path` follow-up
  explicitly deferred from #100.
- **#100** GGUF deployments now boot. `app/runtime/cmd_builder.py` derives the
  vLLM-required `:quant_type` suffix from `model.filename` (regex
  `-(Q\d[_A-Za-z0-9]*)\.gguf$`) and emits `--model <hf_repo>:<quant>`. Without
  this every GGUF model added since #85 (v17.13) failed at vllm subprocess
  startup with rc=1 because vLLM 0.20.0 requires the colon form. Safetensors /
  non-quantized rows are unchanged — a missing filename or non-matching
  pattern falls back to the bare `hf_repo`. The `--tokenizer` /
  `--hf-config-path` follow-up landed in #106.

### Added
- **Frontend: Add Model wizard advanced section + tooltip math + parallelism
  wiring (#87).** `frontend/src/components/models/add-model-modal.tsx` grows
  a collapsible "Advanced" disclosure (native `<details>`, collapsed by
  default) housing three new inputs: `parallelism_strategy` (auto/tp/pp),
  `max_batch_size` (1–64), and an explicit `max_model_len` override. All
  three flow through to `POST /api/models` so the backend's #88 wiring sees
  them. Single-host pipeline-parallel is explicitly allowed — no client-side
  guard against `pp` when only one host is configured, mirroring the
  cmd_builder's behaviour. The fit-badge tooltip already surfaced the four
  primitives (`bytes_per_token`, `kv_reserve`, `weights_budget`, `ratio`);
  `orange` ("tight") rows now also surface a "Recommended max_model_len"
  hint below the badge, preferring the backend's `recommended_max_model_len`
  field and falling back to a client-side solver in `frontend/src/lib/fit.ts`
  (`recommendMaxModelLen`) when the backend value is null. New
  `RECOMMENDATION_TARGET_RATIO=0.70` constant mirrors `app/models/fit.py`
  so the two solvers agree. Vitest component tests pin the collapse
  behaviour, the input wiring, the no-PP-block contract, the four-primitive
  tooltip, and the verdict-gated recommendation hint. **CR fix-up:**
  `fetchFitPreview()` now threads `max_batch_size` and `max_model_len`
  through the `POST /api/models/fit-preview` body whenever the operator
  has set them in the Advanced section, and a 300 ms-debounced effect
  refetches on each numeric edit so the tooltip's `kv_reserve` / verdict /
  recommended-L stay live as the operator tunes batch & context. A new
  per-request `fitSeqRef` short-circuits stale writes when a slow earlier
  fetch lands after a fresh override. Unit coverage for
  `recommendMaxModelLen()` lives in
  `frontend/tests/contract/fit-classifier.test.ts` (happy path against the
  backend formula + every null-return branch); component coverage in
  `frontend/tests/component/add-model-modal.test.tsx` pins the FE-fallback
  hint on orange rows when the BE returns `recommended_max_model_len: null`,
  the debounced refetch carrying overrides, and a consequence check that
  asserts `kv_reserve` scales ~16× with `max_batch_size=16` vs the baseline.
- **Backend: parallelism strategy + gated-repo polish + shard aggregation
  finalise (#88).** `app/runtime/cmd_builder.py` now consults
  `ModelRow.parallelism_strategy` (from migration 0014): `auto`/`tp` keep
  emitting `--tensor-parallel-size=N` (legacy behaviour, default), `pp`
  swaps in `--pipeline-parallel-size=N` instead. Single-host PP is not
  blocked at the builder layer — vLLM accepts it and operators benchmark
  both strategies in the wizard. Benchmark-v2's `tensor_parallel_size`
  override still drives the parallelism *degree* regardless of which
  flag is emitted, so existing sweep semantics keep working when an
  operator flips strategy mid-experiment. `app/models/pull_task.py`
  surfaces gated/private HF repos as a typed `PullAuthRequired` sentinel
  (mirrors `DiscoveryAuthRequired` from #84): both the metadata fetch
  (`estimate_repo_bytes`) and the actual `snapshot_download` route 401/403
  refusals through `_classify_hf_auth_error`, and `run_pull` writes
  `last_error = "auth_required: <hint with /setup/hf-token> (<exc>)"` —
  same prefix the FE keys off in the discovery stage. 5xx HF outages
  still fall through to the generic `pull error:` envelope so operators
  don't chase a token rotation for a server-side problem. Fit-preview
  shard aggregation in `app/models/routes_api.py` now covers GGUF splits
  (`*-NNNNN-of-NNNNN.gguf`) alongside the safetensors set shipped in #85;
  picking part 1-of-3 of a Llama 70B GGUF classifies against the full
  ~40 GB, not 13 GB.
- **Frontend: Add Model wizard rebuilt as a 4-state machine (#86).**
  `frontend/src/components/models/add-model-modal.tsx` now drives a
  `enter-repo → discovering → select-file → submitting` flow against the new
  `GET /api/models/discover` and `POST /api/models/fit-preview` endpoints.
  The select-file stage renders the repo's files (filename, size, kind
  badge, quant, fit verdict) with a per-row colored fit badge; ticking the
  GPU checkboxes recomputes the verdict client-side via the mirrored
  classifier in `frontend/src/lib/fit.ts` (constants locked to
  `app/models/fit.py:21-23`). The badge tooltip exposes the underlying
  math (`bytes_per_token`, `kv_reserve`, `weights_budget`, `ratio`).
  GGUF rows surface a soft "vLLM GGUF serving not yet supported" warning
  banner but still allow submit (operator override). Gated/private repos
  return `401 auth_required` → the modal swaps the file table for a CTA
  linking to `/setup/hf-token`. Vitest component + contract tests plus a
  route-mocked Playwright spec lock the new contract.
- **Backend: VRAM-fit math + per-file download + schema bump (#85).**
  New `app/models/fit.py` module with pure functions
  (`dtype_bytes_from_torch_dtype`, `kv_reserve_bytes`, `weights_budget_bytes`,
  `classify_fit`, `recommend_max_model_len`) — the source-of-truth for the
  4-way fit verdict (green < 0.55 ≤ yellow < 0.80 ≤ orange < 1.0 ≤ red).
  New `POST /api/models/fit-preview` route — accepts the discover-time
  config + the operator's GPU pick + max_model_len, and returns
  `verdict`, `ratio`, `kv_reserve`, `weights_budget`,
  `file_size`, plus a `recommended_max_model_len` for orange/red.
  Multi-part safetensors shards (`prefix-NNNNN-of-NNNNN.safetensors`)
  are aggregated to their full set before classification, so picking
  shard 1-of-4 of a 30 GB model classifies against 30 GB, not 7.5 GB.
  `ModelCreate` schema gained three persisted fields: `filename`
  (optional pinned weights file), `parallelism_strategy`
  (`tp`/`pp`/`auto`), `max_batch_size` (1-64). Migration 0014 extends
  the `models` table with matching columns + defaults so legacy rows
  decode unchanged. `app/models/pull_task.py` plumbs an
  `allow_patterns = [filename, config.json, tokenizer*, *.txt, *.md]`
  through both `estimate_repo_bytes` and `snapshot_download` when a
  filename is pinned — the disk-shortage pre-check now sizes against
  the filtered set, fixing a false shortage on 19.8 GB single-file
  pulls of repos with 200 GB total weights.

### Changed
- **SSE wire alignment.** Renamed/added fields on request_started,
  request_completed, request_summary, bench_progress to match the FE
  bench-event contract (resolves Phase 3 CR + QA blockers).
- **FE wire alignment with BE Bundle 2.** Reducer + types updated to
  consume the canonical BE wire shape (`status` string,
  `elapsed_ms`, `tokens_in`/`tokens_out`, `dropped_*`, `*_count`,
  `bench_progress.phase`). Dropped four event-type constants the BE
  does not emit (`RESUMED`, `LOAD_DONE`, `PHASE1_STEP`,
  `CELL_STARTED`) and their reducer branches. Regenerated
  `api-types.generated.ts` to pick up the new
  `MatrixOverrides.phase1_request_timeout_s` field. Added
  `tests/contract/bench-sse-wire.test.ts` — a 5-test suite that
  replays frozen BE-shipped JSON fixtures through `benchReducer` and
  fails loudly if the wire shape drifts from what the FE consumes.

### Added
- **Bench FE — virtualized Events tab with filter chips + per-request rows
  (#77, #75).** Replaces the prior `EventLog` single-feed component with a
  Virtuoso-backed list that interleaves request lifecycle rows
  (`request_started` → `request_completed` upsert keyed by `request_id`)
  with the raw event tail. Five-way filter (`system` / `cell_result` /
  `phase1_envelope` / `request` / `other`); empty selection = "show
  everything" (Gmail-pill UX). Backpressure summary banner appears when
  `request_summary.dropped_started + dropped_completed > 0`. Status chip
  variants: streaming → info, ok → success, fail / http_error → error,
  truncated / timeout → warning. Request grid: `req_id | status |
  elapsed | ttft | tok_out | fail_reason`. FE ring-buffer cap is
  `MAX_REQUEST_ROWS = 2000` (mirrors BE `REQUEST_EVENT_BUDGET_PER_S × 40
  s` design horizon — a regression test in
  `tests/component/virtualization-smoke.test.tsx` pins the FE constant
  so silent drift trips at test time). Shared `useStickyBottom` hook in
  `src/components/shared/use-sticky-bottom.ts` keeps log-stream and
  events-tab in lockstep on the "scroll up → free; jumpToLatest → stick"
  behavior so the two streaming surfaces can't drift on the first bug
  fix. Mounted in both `/benchmarks/[runId]` and the model-detail
  Benchmark tab.
- **Bench FE — virtualized model log tail with elided-count marker
  (#75).** `LogStream` switched from `<pre>` + `AnsiLog` to Virtuoso
  rendering with stable monotonic ids per line so FIFO eviction
  (`MAX_LINES = 5000`) doesn't re-mount surviving rows. An elided-count
  banner ("… N older line(s) elided") renders above the scroll region
  when the buffer has wrapped, so a long-running run doesn't silently
  appear "stuck" once it crosses 5k log lines. Vitest setup gained a
  `vi.mock('react-virtuoso')` shim that renders every item as a plain
  div: jsdom can't drive ResizeObserver, so the real library renders
  zero rows and every assertion downstream of it fails. The shim
  passes through `data` / `itemContent` / `computeItemKey` / `style` /
  `components.List` so role="log" semantics survive virtualization.
- **Bench FE — live Phase 1 probe view in `EnvelopePanel` (#76).** Until
  `phase1_envelope` lands, the panel renders the running probe state
  from `phase1_progress` instead of a 30–60 s "measuring…" skeleton.
  Shows probes_done/total + progress bar, current concurrency × max_new,
  pass-rate %, p50/p95 latency, truncated/failed/timeout counts.
  Selection order is `final` → `running` → `empty`/skeleton so the
  settled answer always wins when both slices are populated. The
  `EnvelopeLimitedBy` union widened to include `"timeout"`,
  `"ceiling"`, `"truncation"` (badge variants: timeout/oom → warning,
  rest → info).
- **Bench FE — sticky run-progress banner with phase chip + ETA (#78).**
  New `ProgressHeader` mounted at the top of both bench detail surfaces.
  Self-hides until the cli emits the first `bench_progress`; resets
  cleanly on `load_attempt` because the reducer wipes the slice when a
  new lc starts (so a fresh attempt's bar restarts at 0% rather than
  inheriting the previous attempt's tail). Shows phase chip (phase1 =
  info, phase2 = success), progress bar, %-complete, cells_done/total,
  elapsed, ETA. ETA renders "—" until the cli has ≥3 samples.
- **Bench FE — operator-tunable Phase 1 request timeout in the start-run
  form (#79).** `ModelBenchCard` on `/benchmarks` gained a numeric
  input (range 1..600, empty ⇒ 30 s server default) wired to
  `matrix_overrides.phase1_request_timeout_s` in the POST body. Helper
  text under the field surfaces the range + default so operators
  don't have to dig into the matrix-overrides schema to find the knob.
- **Bench FE reducer + types — Bundle 2 surface (#76, #77, #78, #80).**
  `useBenchSSE` grew six new state slices — `envelope_running`,
  `requests`, `requests_summary`, `progress`, `system_warning` — fed
  by the new event types `phase1_progress`, `request_started`,
  `request_completed`, `request_summary`, `bench_progress`,
  `system_warning`. `load_attempt` now resets every lc-scoped slice
  (envelope, envelope_running, requests, requests_summary, progress)
  so a new load-config never displays the previous one's tail. New
  reducer tests in `tests/component/bench-reducer-bundle2.test.tsx`
  pin the dispatch behavior for every Bundle 2 event type + the
  load_attempt reset. Event-type strings live in
  `src/lib/bench-event-types.ts` as `EVENT_TYPES_ALL` so no caller
  hand-rolls a literal.
- **Bench Phase 1 per-request wall-clock timeout (`phase1_request_timeout_s`).**
  Before this change a stalled vLLM server (queue saturated past the
  per-request budget) let the bench loop hang on a single
  `client.stream(...)` call until httpx's pool-level read timeout (120 s)
  fired, *then* surfaced as `http_error=True` (a 5xx-style ceiling). The
  envelope finder couldn't distinguish "server stalled" from "model
  produced garbage" because both ended up in the same `fail_count` /
  `pass_rate=0` bucket. `_stream_one` now wraps the entire stream in
  `asyncio.wait_for(..., timeout=request_timeout_s)`; on expiry the
  returned `_ReqResult` carries `timed_out=True` and
  `latency_ms == request_timeout_s * 1000`. The aggregator buckets those
  into a new `timeout_count` (excluded from p50/p95/ttft/pass_rate) and
  emits `limited_by="timeout"` when timeouts dominate the failure modes
  for the cell. `matrix_overrides.phase1_request_timeout_s` (1..600,
  default 30) flows through `_drive_run` to every Phase 1 probe and
  Phase 2 load cell, replacing the implicit p95-latency cap that
  `_classify` used to apply. Closes #79.
- **Bench `/internal/event` passthrough enabler.** The existing
  `/internal/cell` and `/internal/status` routes are tied to specific
  `bench_cell` / `bench_run` side-effects; any *additional* event type
  the cli wants the SSE relay to see (fine-grained `cell_progress`
  ticks, the `populated_buckets` warning, future hooks) flows through
  the new `POST /api/bench/runs/{run_id}/internal/event`. Body cap is
  4 KB (matches `MAX_EVENT_BYTES`; checked on `content-length` header
  THEN raw body so an unannounced upload can't OOM the parser); per-run
  rate cap is 1000 eps via a token bucket (`BENCH_EVENT_RATE_LIMIT_EPS`,
  refilled on monotonic clock). Structural fields `ts` / `run_id` /
  `seq` are stripped silently — the supervisor + EventWriter are the
  sole source of truth. Auth: same bench-internal token scheme as the
  other internal routes. Status codes: 200 ok, 400 bad payload (missing
  `type`, non-object, malformed JSON), 413 oversize, 429
  rate-limit-exceeded (with `Retry-After: 1`). Bucket lifecycle is
  reset on run cancel and run-failed paths so a fixture-reused
  `run_id` starts with a full bucket. Sync 1 of v17.11 Bundle 2 —
  unblocks Task C and dev-2's UI surfacing.
- **Bench corpus stub detection (`populated_buckets`).** v1 ships
  `app/bench/corpus/128k` and `1m_plus` as README-only stubs; before
  this change the matrix sweep would still enumerate cells against
  them and every request would fail "no prompts for size" — N empty
  cells per load-config with no operator-visible reason. The new
  `matrix.populated_buckets(corpus_root) -> (buckets, skipped)` scans
  `<corpus_root>/<bucket>/*.json` non-recursively and splits the
  canonical bucket set into populated (a `dict[str, int]` that drops
  straight into `cells_for_envelope(buckets=...)`) and skipped (the
  unpopulated names in canonical order). `_drive_run` calls the
  scanner once at the top and emits a single `system_warning` event
  (`type=system_warning`, `kind=populated_buckets`, `skipped=[...]`,
  `populated=[...]`) via the new `post_event` passthrough when the
  skipped list is non-empty. The populated subset feeds
  `cells_for_envelope` so Phase 2 only iterates real content. Failure
  to emit the warning is non-fatal (log-and-continue) — the event is
  informational. Closes #80.
- **Bench corpus: long-context tiers populated (64k / 128k / 256k /
  512k / 1m_plus).** v1 shipped `128k` and `1m_plus` as README-only
  stubs (the trigger for the #80 `system_warning`); this change adds
  real prompt files for those two and introduces three new tiers —
  `64k`, `256k`, `512k` — bringing the bench length axis to nine
  buckets. `matrix.DEFAULT_BUCKET_TOKEN_BUDGETS` extends from six to
  nine entries (`1k`, `4k`, `16k`, `31k`, `64k`, `128k`, `256k`,
  `512k`, `1m_plus`) so the sweep enumerates the new tiers without
  any matrix-override. Per-tier prompt counts: 6 for
  `64k`/`128k`/`256k`/`512k`, 4 for `1m_plus` (million-token prompts
  already cost real GPU minutes per request — four is enough to spot
  a regression without making a sweep prohibitive). Each prompt is
  one of three task families wired to the existing bench graders:
  `needle` (literal-string recall), `keyword_in` (case-insensitive
  substring presence), `json_extract` (single-field JSON output);
  this is the same grader contract `populated_buckets` and the
  Phase 2 cell renderer already speak. Token sizes are measured with
  tiktoken `cl100k_base` and held to a ±10% band per tier
  (`test_long_buckets_are_sized_appropriately` extended for all five
  new tiers). The corpus is generated by a new deterministic CLI —
  `python -m app.bench.corpus._generate <tier>` — which materialises
  one tier from a fixed manifest plus
  `app/bench/corpus/_haystack.SEEDS` (28 license-clean synthetic
  technical paragraphs written for this benchmark, no third-party
  content); two runs with the same args produce byte-identical JSON,
  so the files are checked in rather than rebuilt at sweep time. Per-tier
  `README.md` in each new directory documents prompt mix, tokenizer
  choice, content source, and regeneration command; the top-level
  `app/bench/corpus/README.md` gains a "Regenerating long tiers"
  section pointing at `_generate.py`. `tiktoken>=0.7` is added to
  `requirements-dev.txt` only — the bench runtime path never imports
  it. Closes #81.
- **Bench SSE event-type wire constants.** `app/bench/events.py` now
  publishes one `EVENT_TYPE_*` constant per canonical event ``type``
  the supervisor or cli stamps onto the SSE stream (`run_started`,
  `run_done`, `cancelled`, `paused`, `error`, `load_attempt`,
  `load_result`, `phase1_envelope`, `cell_result`, `cell_progress`,
  `system_warning`) plus a `EVENT_TYPES_ALL` frozen set covering them.
  Producers can now import the symbol instead of hard-coding the
  string; a regex-over-source integrity test guards against rogue
  type literals leaking into the supervisor or cli. Sync 1 of v17.11
  Bundle 2 — wire-name compat with dev-2's SSE relay.
- **Bench `load_config_grid` Cartesian shorthand.**
  `matrix_overrides.load_config_grid` accepts per-axis arrays
  (e.g. `{"gpu_memory_utilization": [0.7, 0.8, 0.9], "max_model_len": [4096, 8192]}`)
  and `resolve_matrix` expands them into the Cartesian product as N
  `LoadConfig` rows (first axis varies slowest — matches
  `itertools.product` default order). Cap of 25 products; rejects
  oversize grids, empty grids, empty axes, unknown axes (typo guard),
  and mixing with the explicit `load_configs` list with a 400 at the
  schema layer. Sets up (#62) 3D sweep chart input — the natural
  shape for "sweep across each axis of a 3D plot". Closes #59.
- **Bench: real per-LoadConfig vLLM reload (`apply_load_config`).** The
  bench `Supervisor.apply_load_config()` was previously a logging-only
  stub — supplying multiple LoadConfigs in `matrix_overrides.load_configs`
  did nothing, so the 3D sweep chart (gpu_memory_utilization × concurrency
  × max_model_length) was unreachable. The supervisor now performs the
  real model swap for every LoadConfig the cli requests: it unloads the
  current vLLM process (SIGTERM → 30s → SIGKILL), releases the port,
  allocates a fresh port, calls `RuntimeSupervisor.load(model, port,
  overrides=lc.to_overrides())`, then waits up to
  `settings.bench_health_wait_s` for `/health` to go green. On success
  it returns `(load_ok=True, load_ms, load_error=None)` and updates the
  `model_runtime` row with the new pid/port. On failure it classifies
  the cause: `"oom"` (CUDA OOM marker found by scanning the last 64 KB
  of the vllm log), `"crashed"` (subprocess died without an OOM marker),
  `"health_timeout"` (subprocess alive but `/health` never green), or
  `"<ExcType>:<truncated msg>"` for anything else. A failed reload on
  LoadConfig A does NOT abort the run — the cli records the failure on
  the attempt row (`load_ok=0`, `load_error`, `ended_at`) and proceeds
  to LoadConfig B. Wired via a new internal endpoint
  `POST /api/bench/runs/{run_id}/internal/apply_load_config` (bench-scope
  token, same CSRF bypass as the rest of the internal callback
  perimeter); the cli's `_drive_run` calls it once per LoadConfig before
  `find_envelope`. On run end (`_reap_on_exit` or `cancel`) the
  supervisor restores the model to its row-default overrides via
  `restore_original_config`, so a cancelled or completed bench run
  leaves the model in the same shape the operator booted it in. Tests:
  7 unit tests cover every error-classification branch + missing-wiring
  guard + restore-noop; 3 cli-side unit tests pin the multi-LoadConfig
  isolation contract (apply-failure on A → run continues on B; RPC
  exceptions treated as `load_failed`; full 5-field LoadConfig dict on
  the wire); 1 integration test posts two LoadConfigs through the
  internal-callback perimeter and asserts 2 attempt rows, 2
  `phase1_envelope` events, 2 `phase1_probe` cells, and the run row
  stays `running` after LC-A's quality-floor failure. Closes #58.
  - **Fix-up from code review (`4d2dc3c` → next).** Addresses 7
    review findings without changing the public contract: (C1) every
    bench-spawned vLLM now ships with an `on_exit` callback that marks
    the model row `failed`, clears the runtime row, and releases the
    port — mirrors the production `/load` route so a vLLM that dies
    mid-bench no longer leaves the runtime in a zombie state. (C2)
    `apply_load_config` snapshots the LIVE overrides via the new
    public `RuntimeSupervisor.get_overrides(model_id)` accessor (no
    more reading `_overrides` from the runtime supervisor's
    private state); operator-tuned configs (e.g. a pre-bench load
    with `gpu_memory_utilization=0.85`) now survive a bench run
    exactly as the operator left them, instead of silently regressing
    to row defaults. (I1) replaced the private `sup._processes[id].pid`
    read with a public `RuntimeSupervisor.get_pid(model_id)` accessor
    that returns `None` when the process is gone, classified as
    `crashed` (with OOM-scan fallback) instead of raising `KeyError`.
    (I2) `bench/cli.py` derives the `apply_load_config` HTTP timeout
    from `BENCH_HEALTH_WAIT_S` (`max(180s, wait_s + 60s)`) so slow
    loaders no longer trip a hard 180s RPC ceiling. (I6) `cancel`'s
    no-active-run branch now also invokes `restore_original_config`,
    matching the with-active-run branch — an operator-cancelled
    bench run no longer leaves the model swapped to the last bench
    LoadConfig. (M1) OOM scan now reads from a per-model offset
    bookmark recorded at the start of `apply_load_config`, so LC #1's
    OOM marker can never mis-classify LC #3's plain crash on the
    append-only vLLM log. (M3) timestamps in `RuntimeRepo.upsert`
    are ISO-8601 via `datetime.now(UTC).isoformat()`, matching the
    production `/load` route. Tests: 4 new unit tests pin the C1/C2/I6
    contracts (snapshot-captures-live-overrides, on_exit-is-wired,
    cancel-invokes-restore, reap-on-exit-invokes-restore); the
    pre-existing 8 tests continue to pass. Full suite: 639 passed.
- **Bench: Phase-1 envelope probe steps now persist as `bench_cell`
  rows.** Each per-step probe the envelope finder runs (`c=1, 2, 4, …`
  at `4k` / `max_new=256`, then the `max_new` sweep at the discovered
  `max_concurrency`) lands in `bench_cell` with `status="phase1_probe"`
  and the full signal set (agg_tps, p50/p95 ttft, p50/p95 latency,
  pass_rate, ok/fail counts, error). Pre-issue-#57 every probe was
  discarded on the floor — operators could see the final
  `Envelope(max_concurrency, max_new, limited_by)` but not *why* that
  decision was made, so a zero envelope from a bad-quality grader was
  indistinguishable from one caused by an actual model failure.
  `GET /api/bench/runs/{id}/cells` returns the probe rows alongside
  Phase-2 cells; the chart endpoint (issue #60) can filter on
  `status="phase1_probe"` to compute per-row `limited_by`.
  Schema-shape choice: the sentinel rides on the existing
  `bench_cell.status` column rather than introducing a new `phase`
  column — `BenchCellRepo.list_ok_for_resume` and `best_by_metric` both
  filter on `status='ok'`, so probe rows are automatically excluded from
  resume and best-of queries without a migration. Also: probe-drain
  failures from the internal callback are now logged-and-continued
  rather than fatal (a single bad `post_cell` no longer sinks the
  Phase-2 status emit); the cells table renders probe rows with a
  neutral `info` badge labelled "probe" instead of the red error
  variant. Closes #57.
- **Live GPU/VRAM probe endpoint (`GET /api/system/gpus`).** Returns a
  fresh nvidia-smi snapshot — per-GPU memory total / used / free / utilisation,
  GPU model name, and a list of `holders` (PIDs currently using GPU memory).
  Holders are attributed to vLLM Warden models when the supervisor recognises
  the PID (or its process group leader, so tensor-parallel workers map back
  to their parent model). Unknown PIDs are labelled `kind: external` with
  the bare process name. Response includes a `probed_at` ISO timestamp and a
  `probe_error` field so the UI can render a clean empty state on dev boxes
  without an NVIDIA driver. Burst polling is absorbed by an in-process 2 s
  cache to keep the call cheap for 5–10 s fast polling. Backend-only — the
  frontend gauges are Phase 2.

  **Deployment requirement.** Holder attribution requires the api container
  to share the host PID namespace — `pid: host` in docker-compose,
  `hostPID: true` in K8s pod spec. nvidia-smi reports host PIDs while the
  supervisor tracks in-container PIDs, so without this flag the namespaces
  diverge and every model holder silently degrades to `kind: external`
  (no crash, no error — just attribution loss). The bundled
  `docker-compose.yml` and `deploy/hub/compose.yaml` already set
  `pid: host`. Caveat: PodWarden's compose-stack → K8s translation does
  not currently honour `pid: host` (a follow-up bug is filed against
  PodWarden); K8s deployments via the PodWarden Hub catalog stay in the
  degraded-but-functional state until that lands.
- **GPU model name on `gpu_samples`.** Migration `0013` adds a nullable
  `name` column to `gpu_samples`; the per-minute stats sampler now writes the
  card model (e.g. "NVIDIA RTX A4000") on each insert and `GET /api/stats/gpus`
  returns it alongside the existing memory/utilisation fields so the
  historical chart can label gauges without a second round-trip. Pre-0013 rows
  keep `name = NULL` and the API surfaces it as such; UI must fall back to
  "GPU N".

### Fixed
- **CI: pin transformers below 5.8 to dodge upstream pip metadata bug.**
  `transformers-5.8.1` shipped on PyPI with a `Version` object (not a
  string) in its `requires-python` metadata, which crashes pip's
  specifier parser with `TypeError: expected string or bytes-like
  object, got 'Version'` during `pip install` in the integration-tests
  job. Capped `requirements-dev.txt` to `transformers>=4.55,<5.8` to
  exclude 5.8.x until upstream republishes; production unaffected
  (transformers comes from the vllm/vllm-openai base image, not this
  pin).
- #91 — Disable vLLM async output processor on `gpt_oss` family models to
  work around concurrent `RefCell::borrow_mut` panic in the
  `openai_harmony` Rust extension. The model template registry entry for
  `gpt-oss-20b` now defaults `VLLM_USE_ASYNC_OUTPUT_PROC=0` in
  `extra_env`; the synchronous output path serializes harmony access
  per-engine so two overlapping `POST /v1/chat/completions` calls no
  longer race and the second caller no longer surfaces
  `RuntimeError: Already borrowed` as HTTP 500. Root cause is a shared
  module-level `_harmony_encoding` singleton in
  `vllm/entrypoints/openai/harmony_utils.py` — lazy-initialized once and
  returned by reference, with no lock or per-request isolation. The
  singleton is still present in upstream v0.21.0 (verified by reading
  the file on the published tag); PRs #41181 and #40059 touch
  `vllm/tokenizers/hf.py` / `vllm/renderers/hf.py` /
  `tool_parsers/*.py` but do not modify `harmony_utils.py`, so a base-image
  bump alone is not sufficient. Scope is intentionally narrow: the flip
  applies only to newly-created models from the `gpt-oss-20b` template
  via the FE wizard, leaving other model families on the async output
  path. **Follow-up TODO:** remove `VLLM_USE_ASYNC_OUTPUT_PROC=0` from the
  template once upstream vLLM lands per-request isolation in
  `harmony_utils.py` (tracking upstream issue). Operators with an
  already-deployed gpt-oss model must edit the model row's `extra_env`
  via `PUT /api/models/{id}` (or re-create from the updated template) to
  pick up the workaround. Closes #91.
- **Add Model modal crash on file-picker stage.** `/api/system/gpus` returns `{ gpus: [...] }` but the modal treated the response as a bare array, throwing `TypeError: gpus.filter is not a function` the moment the user clicked Discover. Unwrap `.gpus` and add defensive `Array.isArray` guard. (#98)
- **Frontend: silent refresh + transient/terminal classification — no more
  "logged out every few minutes" (#97).** Two narrowly-scoped changes in
  `frontend/src/lib/auth-fetch.ts`. (1) Every successful login or refresh
  now schedules a proactive refresh at 80% of `expires_in` (12 min for the
  default 15 min access TTL), so the in-memory access token is rotated
  *before* it expires — operators no longer get punted to /login at the
  hard 15-minute boundary. (2) `refresh()` now distinguishes "the backend
  rejected the refresh cookie" (HTTP 401/403 → terminal, redirect to
  /login under the existing single-shot guard) from "the backend
  hiccupped" (HTTP 5xx, 429, network error, malformed body → transient,
  surface the original 401 to the caller, do NOT redirect). Pre-fix a
  single 502 from /api/auth/refresh during a backend bounce evicted the
  user; now the next user action retries. Login page reads `expires_in`
  from the /api/auth/login body and passes it through `setAccessToken`.
- **Bench: `BENCH_HEALTH_WAIT_S` default bumped 30s → 180s for 20B+
  cold-loads.** Cold-starting a `gpt-oss-20b` (or any 20B+ model) under
  `apply_load_config` reliably exceeds the prior 30s health-wait budget
  off NVMe — the supervisor would abort the load, the cli would then
  mis-attribute the supervisor's health-wait timeout as
  `apply_load_config_rpc:ReadTimeout` on the wire, and operators got a
  misleading "RPC timed out" status even though the model was still
  loading correctly. New default (`Settings.bench_health_wait_s = 180.0`,
  and the `BENCH_HEALTH_WAIT_S` env fallback in `load_settings` now
  matches) gives the model 3 minutes to register healthy; the cli's
  `_apply_load_config_timeout_s` likewise mirrors the new 180s floor so
  it can't ever undershoot the supervisor. Operators with small / fast
  models can still shrink the wait via `BENCH_HEALTH_WAIT_S=<seconds>`.
  3 unit tests pin the dataclass default + the env override + the
  unset-env path. Closes #69.
- **Bench Phase-1: probes honour the corpus's `expected_max_new`
  floor.** The envelope finder hard-codes `max_new=256` on its sweep,
  but the bundled `4k` corpus has prompts that ask for 384–512 tokens
  of structured output. At 256 those prompts were truncated mid-answer,
  the grader scored every one of them as a fail, and Phase 1 saw an
  artificial "concurrency=1 fails quality" wall it couldn't get past.
  `run_cell_http` now accepts `respect_prompt_max_new=True` (set
  automatically by `envelope_adapter`) and promotes the per-cell
  `max_new` to `max(p.expected_max_new for p in prompts)` (capped at
  2048 to bound any single bad corpus tag — unannotated prompts keep
  the legacy 256 default so existing corpora are not penalised). The
  envelope finder's own `max_new` sweep (256 → 4096) still stacks on
  top of the floor: the requested value wins when it's higher than
  the prompt-derived ask. Phase 2 (matrix sweep) leaves the flag
  False — the matrix runner already chose `max_new` deliberately and
  must not be silently promoted. 7 unit tests pin the helper's
  per-axis behaviour (largest-wins, requested-above-floor-wins,
  ceiling-caps, missing-annotation-default, mixed annotated/un, zero/
  null defensive) plus 2 end-to-end tests on `run_cell_http` that
  verify outbound request bodies carry the floored `max_tokens` only
  when the flag is on. Closes #70.
- **Bench: truncated responses (`finish_reason="length"`) no longer
  drag pass_rate down.** A request that vLLM cuts off via the
  `max_tokens` budget is a *capacity* outcome — the model didn't fail,
  the budget did — but the grader saw an incomplete answer and counted
  it as a quality fail, so cells whose `max_new` was even slightly
  short showed a "the model is terrible" pass_rate when really the
  bench's own knob was wrong. `_ReqResult.truncated` is now set when
  the final stream chunk carries `finish_reason="length"`; the cell
  aggregator buckets these under `truncated_count` and EXCLUDES them
  from the `pass_rate` denominator (`pass_rate = ok / (ok + fail)`,
  not `ok / total`). If *every* completed request was truncated,
  `limited_by="truncation"` rides alongside on the `cell_result` SSE
  event payload so the supervisor / UI can show "this cell's max_new
  was the bottleneck, not the model" rather than "this cell scored
  zero". Both fields are additive on the event payload only — no
  `bench_cell` schema migration, the supervisor reads them off the
  raw cli POST and includes them in the emit. 3 unit tests cover
  the three regimes (mixed ok/fail/truncated → pass_rate honours the
  exclusion; all-truncated → `limited_by="truncation"`; no
  truncation → legacy behaviour unchanged). Closes #71.
- **Bench Phase-1: probe cells now pin sampling for reproducibility.**
  Probe runs used to feed the model with whatever temperature / top_p
  the runtime defaulted to and an unseeded RNG, so two reruns of the
  same probe against the same model build could produce different
  text — making the grader's pass_rate a function of "what RNG state
  did vLLM pick this morning" rather than "is this concurrency
  workable". `run_cell_http` now accepts `pinned_sampling=True`
  (set automatically by `envelope_adapter`); when on, every outbound
  /v1/completions body carries `temperature=0.0`, `top_p=1.0`, and a
  deterministic 31-bit `seed = blake2b(cell_id|attempt_id|prompt_idx)`
  so two probe runs against the same model with the same
  coordinates yield identical text. `attempt_id` is forwarded by the
  cli's `_drive_run` so the seed is also stable across re-attempts of
  the same load-config; when not supplied the seed falls back to
  coordinates-only (still deterministic within a run). Phase 2 load
  cells deliberately leave the flag False — they're meant to measure
  real-workload variance. 8 unit tests pin the seed function
  (same-inputs-same-seed, prompt-idx / cell-id / attempt-id all
  perturb it, 31-bit mask is honoured) plus the
  `run_cell_http`-level integration (body fields when pinned, body
  fields absent when not pinned, two runs same coords → same seed,
  envelope_adapter wires all three flags). Closes #72.
- **Bench `json_path` grader: leading preambles no longer shadow the
  real structured answer.** `_extract_json_blob` used to find the
  FIRST balanced `{...}` / `[...]` substring and return it — so an
  output like `{plan: lots of reasoning... {"category":"auth"}}`
  (unclosed pseudo-JSON preamble + real answer) tripped the depth
  scanner: depth stayed >0 past the inner object, the scanner fell
  through to array search, and the grader returned the inner
  `key_phrases` list or a `no JSON` failure instead of the real
  answer. The scanner now starts a fresh nesting frame at every
  top-level opener (so a never-closed preamble doesn't block the rest
  of the string) and `_extract_json_candidates` collects EVERY
  successful `json.loads` candidate in source order. `_grade_json_path`
  walks the candidate list in reverse and accepts the first one
  whose path resolves — i.e. the LAST candidate where the requested
  key exists, matching operator intent ("the final structured output
  is the real answer; preambles are commentary"). `_extract_json_blob`
  itself stays as the public "give me the last parseable JSON" thin
  wrapper for non-path callers. 7 new unit tests cover the
  preamble-doesn't-block-inner-json bug, last-candidate-wins,
  last-resolving-candidate-wins, missing-key surfaces a path
  reason (not a JSON-extraction error), and the truly-no-JSON
  legacy reason is preserved. Closes #73.
- **Bench: phase-1 envelope failures no longer surface as silent
  `load_result load_ok=false load_error=null`.** When `find_envelope`
  returns a zero envelope (e.g. concurrency=1 probe fails the quality
  grader), the SSE `load_result` event now carries an explanatory
  `load_error="phase1_no_envelope: limited_by=<reason>"` so operators can
  tell envelope-failed-quality apart from a real vLLM load failure. Also
  re-ordered the supervisor's SSE emits so `load_result` precedes
  `phase1_envelope` in the per-POST emit sequence. Closes #56.
- **CI `integration-tests` no longer wedges for 25+ minutes after pytest
  passes.** Pipelines 5867 / 5871 all showed `47 passed in ~25s` followed
  by a multi-minute hang the runner couldn't even cancel cleanly.
  Root cause (proven via `faulthandler.dump_traceback_later` inside the
  CI image): `aiosqlite.Connection` extends `threading.Thread` and is
  constructed with the default `daemon=False`. Its worker `run()` loop
  parks on `self._tx.get()` waiting for the stop sentinel that
  `Connection.close()` enqueues. When pytest-asyncio tears down the
  per-test event loop, occasional close paths fail mid-flight
  (`RuntimeError: Event loop is closed` surfacing inside the worker's
  `loop.call_soon_threadsafe`), the sentinel is never delivered, the
  non-daemon thread stays parked, and `threading._shutdown` blocks
  interpreter exit indefinitely — so container PID 1 never exits,
  `docker run --rm` never returns, and the runner waits on docker
  forever. Fix: monkey-patch `aiosqlite.core.Connection.__init__` in
  `tests/integration/conftest.py` to flip `self.daemon = True` after the
  base `Thread.__init__` runs. Daemon worker threads do not block
  interpreter shutdown — Python reaps them on `main` exit. Test
  semantics are unchanged (close still runs on the happy path); only
  the wedge mode is reachable. Conftest also leaves an armed
  `faulthandler.dump_traceback_later(30)` in `pytest_unconfigure` as a
  regression canary: any future re-emergence will print the live-thread
  dump 30s after pytest finishes. Production code is untouched.
  Closes #43.
- **Bench CLI: `--corpus-root` default now points at the bundled corpus
  directory.** The CLI fell back to `/data/bench/corpus` when the supervisor
  spawned it without `VW_BENCH_CORPUS_ROOT` set (which is the steady-state
  case — the supervisor never injected it), producing
  `corpus bucket missing: /data/bench/corpus/4k` and failing every bench run
  within ~10ms before any model load. Default is now
  `str(Path(__file__).parent / "corpus")`, which resolves to
  `/app/app/bench/corpus` in the production image and to the repo-relative
  corpus directory in dev/tests. Closes #54.
- **Live logs panel now streams in real time.** Next.js standalone's
  default `compression` middleware was silently gzipping the
  `text/event-stream` SSE response; because the stream is sparse,
  the gzip encoder buffered indefinitely waiting for a flush block
  and bytes never reached the browser. The panel connected and
  stayed blank. Disabled compression at the Next.js layer
  (`compress: false` in `next.config.ts`) — SSE must be uncompressed,
  and Caddy/Traefik already handles ingress-side compression for
  bulk HTML/JSON. As a secondary cleanup we also removed the v17.2
  status-keyed `<LogStream>` remount that amplified the problem via
  proxy churn (it now keys on model id alone and consumes `status`
  as a prop), retuned the SSE backoff from 1s/30s to 250ms/5s so
  transient blips resolve sub-second, and skip opening an
  EventSource entirely while the model is in the `registered` state
  (no log file content exists yet — a placeholder explains the
  wait). Closes #52, closes #53.
- **Live logs panel no longer goes dead on model status transitions.**
  Same-second SSE ticket re-mints for the same `(sub, path)` produced
  byte-identical HMAC strings; the single-use deny-list treated the
  second consume as a replay and rejected it. A per-mint 8-byte `jti`
  nonce in the payload ensures distinct strings across same-second
  mints. Closes #51.
- **First-load no longer logs three `401 Unauthorized` errors in the
  console.** `authFetch` now eager-refreshes the access token before
  the first request when its in-memory copy is null (hard reload, new
  tab, deep link); the post-401 replay path remains for mid-session
  expiry. Part of #50.
- **Live-logs panel: re-subscribes when the model status changes.**
  `<LogStream>` now keys on status so it fully remounts on every
  `unloaded → pulled → loading → loaded` transition, opening a fresh
  SSE connection (with backend backfill) instead of holding the same
  stream across all four states. Part of #50.
- **Release procedure: `docker buildx --push` now passes `VW_BUILD_VERSION` and `VW_BUILD_SHA` build-args** to both the backend and UI buildx commands so the version banner shows the real tag instead of `vdev · unknown`. Also added a runtime override path via env on the `api` service in `deploy/hub/compose.yaml`. Closes #45.
- **`/healthz` now returns 200 instead of 404 for Kubernetes liveness probes and uptime monitors.** Added a Next.js rewrite that aliases `/healthz → /api/health` so the operator-convention path works without reconfiguring probe manifests. Closes #48.
- **Live-logs panel: "(no log lines yet)" placeholder for connected
  empty streams.** `LogStream` previously rendered an empty
  `<div role="log">` when the SSE handshake had succeeded but the
  subprocess hadn't produced any stdout — operators saw a blank panel
  and couldn't distinguish a healthy-but-quiet stream from a wedged
  one. The component now shows an explicit `role="status"` placeholder
  in that window; once the first line arrives it falls back to the
  normal log container. Part of v2026.05.15.5 hotfix bundle.
- **`GET /api/models/{id}/logs/stream` now opens-or-creates the log
  file.** Previously returned HTTP 404 when the log path didn't exist
  yet, which surfaced through `useEventSource`'s ticket-mint preflight
  as a terminal-error — the UI stuck on "stream unavailable" even
  though the supervisor would create the file moments later. Matches
  the supervisor's `O_CREAT | O_APPEND` behaviour. A missing parent
  directory still 500s (real deployment bug, not a race).
- **SSE log stream emits keepalive comment lines every 15s during
  idle.** Without `: keepalive\n\n` comments, intermediate proxies
  (nginx/Caddy default 60s idle timeout) and EventSource's own
  heuristics could declare the connection dead during long pre-first-
  line silences — e.g. vLLM loading a 20GB weight set before
  producing any stdout. Comment lines are silently ignored by the
  EventSource client but traverse proxies, so the connection stays
  healthy.
- **Subprocess env: `PYTHONUNBUFFERED=1` set on every vLLM child.**
  CPython block-buffers stdout when stdout is a file fd, so a fast-
  crashing subprocess (rc=1 within <1s) could exit before its
  traceback was flushed — operators saw nothing in the log file even
  though Python printed the error. Defence-in-depth alongside the
  open-or-create fix above.
- **`/setup` root URL no longer 404s.** Added a server-side redirect
  page so `https://vllm.example.com/setup` lands on
  `/setup/welcome` instead of returning a Next.js 404. The /setup
  segment had a layout.tsx and five child pages but was missing the
  root `page.tsx`. Closes #41.
- **`docs/releasing.md`: UI image publish step documented.** The
  release runbook only described the backend `vllm-warden` image
  buildx command; operators were silently dropping the
  `vllm-warden-ui` push, leading to deploy-host pulls of stale UI
  tags. Added a dedicated UI section with digest-verification.
  Closes #40.
- **Login page no longer enters an infinite reload loop.** On
  v2026.05.15.3, opening the login page caused a continuous page
  reload: the NavBar's version-check request returned 401 (no session
  yet), which triggered a redirect back to /login, resetting all
  in-flight state and starting the cycle again. The version check is
  now suppressed on /login and /setup, and the redirect logic has a
  guard that prevents /login → /login navigation under any
  circumstances. Closes #38.
- **Frontend live-log panel: renders blank when other authenticated
  endpoints simultaneously 401.** The login-redirect fallback in
  `auth-fetch` was firing for every concurrent 401 (e.g. parallel
  `/api/version` + `/api/models` failures during session-token expiry), and the
  overlapping `window.location.replace('/login')` calls stranded the
  SSE preflight (`POST /api/auth/sse-ticket`) before it could mint a
  ticket — `useEventSource` saw `authFetch` resolve to a value it
  didn't classify and `LogStream` stayed on the "Connecting…"
  placeholder. The redirect is now de-duplicated via a module-level
  `loginRedirectInFlight` flag, the 401 Response is still returned to
  the caller so its error handling runs, and `useEventSource` checks
  the flag (before and after preflight) — transitioning to terminal
  `"session expired"` instead of looping into reconnects that would
  all fail the same way during the unload window. Closes #37.
- **Bench v2: load step no longer crashes with
  `asyncio.run() cannot be called from a running event loop`.** The cli's
  `_drive_run` is itself an async coroutine, and the supervisor invokes
  `find_envelope` with a sync `run_cell` closure that called
  `asyncio.run(run_cell_http(...))` directly. Switching that to
  `await asyncio.to_thread(find_envelope, envelope_adapter(http_args), …)`
  hands the synchronous probe off to a worker thread so the inner
  `asyncio.run` is no longer nested inside the cli's own loop. The bench
  wizard now reaches phase-2 grid without the loader silently aborting.
  Closes #28.
- **Frontend /benchmarks: completed-run timestamps are no longer off by
  a factor of 1 000.** `fmtTs` was multiplying its input by `1000`, but the
  API already returns epoch-ms (`finished_at`), so a run finished at
  16:42 UTC rendered as a year-58341 date. Removed the multiplication;
  added a regression test pinning the contract. Closes #33.
- **Frontend /stats: chart X-axis now actually honours the range selector.**
  The throughput and GPU-util charts relied on recharts' default
  `["dataMin","dataMax"]` domain, so switching from 1h → 7d resized the
  chart container but kept the axis pinned to the data window (which is
  often only a few minutes when the rollup table is sparse). New
  `rangeBounds(range, now)` helper derives `[startMs, endMs]` from the
  selector; both charts now pass it into `<XAxis domain>` with
  `allowDataOverflow` so the axis covers the requested window even when
  data is sparse, and the tick formatter switches to "MMM dd" for the
  7d view so the label row doesn't smear into illegibility. Closes #34.
- **Frontend SSE: terminal stream failures no longer hide behind an
  unbounded "Connecting…" placeholder.** `useEventSource` previously
  exp-backoff'd forever on every error class — a 404 / 401 ticket-mint
  looked identical to a transient network blip, so operators saw a stuck
  spinner with no actionable signal. The hook now mints the SSE ticket
  via `POST /api/auth/sse-ticket` before each connect and uses that
  request's HTTP status to classify failures: 4xx (except 429) is
  terminal and stops the loop; 5xx / 429 / network / EventSource onerror
  is transient with capped exp-backoff (MAX_RECONNECT = 5). The hook
  surfaces `{status, errorCode, attempts}` so `LogStream` can render
  distinct UI for 401 ("session expired"), 404 ("Log stream not found"),
  502+/HTTP-coded ("Stream unavailable (HTTP …)"), and post-retry
  exhaustion ("after N retries"). Terminal banners use `role="alert"`;
  transient retry banners use `role="status"`. Closes #35.
- **CI: integration tests** — set `VW_HF_CACHE_DIR` in 5 tests that previously
  crashed on `PermissionError: /root/.cache/huggingface` under non-root CI
  runner. Closes #30.
- **Bench v2: SSE event stream now populated; UI no longer stuck on
  "Events failed".** The bench supervisor was driving DB rows directly but
  never opening the per-run jsonl event log that `app/bench/sse.py` tails,
  so every wizard session that started a run saw the SSE relay give up
  after its 10 s wait-for-file window and surface "Events failed" with no
  way to recover except a hard refresh. Supervisor now opens an
  `EventWriter` at run start and emits `run_started`, `load_attempt`,
  `phase1_envelope`, `load_result`, `cell_result`, `paused`, `cancelled`,
  `error`, and `run_done` events at the correct lifecycle points (all
  types match `frontend/src/components/bench/use-bench-sse.ts` VALID_TYPES).
  The first event fires *before* the health-wait so the SSE relay's
  10 s discovery window always succeeds. Issue #26.
- **Bench v2: cli no longer races the model loader.** A bench POST that
  arrived before `model_runtime` reported healthy made the cli's first
  `/v1/chat/completions` call against a not-yet-listening port; the cli
  exited rc≠0, the supervisor caught the error but never finalised the
  run (no `ended_at`, no summary), and the UI sat on "running" until
  manual refresh. Supervisor now polls `runtime.supervisor.wait_for_health`
  with a configurable budget (`BENCH_HEALTH_WAIT_S`, default 30 s) before
  spawning the cli; if the port is missing it marks the run failed with
  `runtime_not_loaded`, and if the health check times out it marks it
  failed with `health_timeout`. The cli's fast-fail status payload was
  also extended: it now includes `load_error`, `load_ok=False`,
  `ended_at`, and either `load_config_id` (for failures inside an attempt)
  or `pre_load: True` (for failures before the first attempt), so the
  supervisor's `write_status` handler can finalise both the attempt row
  and the run row in one POST. Issue #27.

### Notes
- **After upgrading on the bonus deployment: re-run the initial setup
  wizard.** The v2026.05.15.3 deadlock recovery on 2026-05-17 required
  recreating the data PVC, which deleted all user accounts. No backup
  existed. Navigate to /setup to create the admin account again before
  using the UI.

### Changed
- New `BENCH_HEALTH_WAIT_S` env var (default `30.0`) controls the
  supervisor's pre-spawn health-wait budget. Bump for slow loaders.

## [v2026.05.19.2] — 2026-05-19

### Added
- **HF discovery service + `GET /api/models/discover` endpoint (#84).**
  New JWT-required route that lists a HuggingFace repo's files (with
  per-file `kind` / `quant` / `params` hints derived from filename)
  alongside the six `config.json` keys the Add Model wizard needs
  (`hidden_size`, `num_hidden_layers`, `num_attention_heads`,
  `num_key_value_heads`, `max_position_embeddings`, `torch_dtype`).
  Pure-helper triplet (`_classify` / `_parse_quant` / `_parse_params`)
  lives in `app/models/discovery.py` so the same classification logic
  can be reused by the FE wizard without recomputing buckets. Gated /
  private repos silently reuse `data_dir/hf-token` (no inline
  re-confirmation prompt — per CTO decision in #84) and surface as a
  typed `auth_required` envelope when the stored token is insufficient.
  HF 404 maps to `repo_not_found`; transport errors to `discovery_failed`
  (sanitised — no raw HF stack trace crosses the wire). In-process
  `(repo_id, revision)`-keyed cache with 60 s TTL mirrors
  `_ProbeCache` in `app/system/routes_gpus.py` so the wizard can
  refetch on selection changes without thrashing huggingface.co.
  Wire shape pinned by `tests/fixtures/discovery/qwen-gguf.json` —
  dev-2 mirrors this fixture in #86 for FE mocking; if the wire shape
  drifts the contract snapshot test fails loudly. Blocks #82.2 and
  #82.3.

## [2026.05.15.0] — 2026-05-15

### Fixed
- Non-streaming `/v1/chat/completions` with large `max_tokens` (e.g. 32768)
  no longer returns HTTP 500 / ECONNRESET after exactly 30 s. Root cause:
  the request path is `client → Caddy → ui (Next.js rewrites) → api → vLLM`,
  and Next.js's `experimental.proxyTimeout` defaults to **30 000 ms** for
  rewrites. When the upstream vLLM generation took longer than 30 s, Next.js
  cut the socket and returned a stock 500, while the api saw the socket
  close mid-request (no uvicorn access-log entry for the failure). Three-
  layer bypass test (api podIP 114 ms ✓, api Service 100 ms ✓, ui Service
  with 32k repro hangs >28 s ✗) localised the cut to Next.js. Set
  `experimental.proxyTimeout: 600_000` in `frontend/next.config.ts` so
  long completions get a 10-minute budget through the UI server. Issue #13.
- Removed unused `IDX_STRIDE` import from `tests/unit/bench/test_events_writer.py`
  (ruff F401). Unblocks the `develop` lint job (pipeline 5581).

## [2026.05.12.1] — 2026-05-12

### Fixed
- HuggingFace model cache now lives on the dedicated `vllm-warden-hfcache`
  PVC (mounted at `/root/.cache/huggingface`) instead of the small data
  PVC subdirectory `/data/hf-cache`. The old path was a `@property` on
  `Settings`, so even though `deploy/hub/compose.yaml` already mounted the
  dedicated PVC at `/root/.cache/huggingface`, the pull task wrote to a
  directory living on the data PVC — which `shutil.disk_usage(...).free`
  reported as 0 GiB for any non-trivial model. `Settings.hf_cache_dir` is
  now a real dataclass field populated from `VW_HF_CACHE_DIR` (default
  `/root/.cache/huggingface`). The Hub template now requires
  `VW_COOKIE_SECRET` (the api refused to boot without it) and surfaces
  `VW_CONTAINER_GPU_COUNT`. The api service also publishes container
  port 8080 so the K8s compose-bridge emits a ClusterIP Service for
  the ui→api hop.
- vLLM subprocess no longer dies at import time with
  `ImportError: cannot import name 'Gemma3Config' from 'transformers'`.
  Root cause: `requirements.txt` pinned `transformers==4.47.1`, which
  downgraded the version that ships with the `vllm/vllm-openai` base
  image. The base image's `vllm==0.20.0` imports `Gemma3Config` (added
  in transformers 4.50), so every model load failed with `rc=1` before
  the model config was even parsed. Pin removed; the base image's vLLM
  dependency now constrains transformers transitively. Our only direct
  use is `from transformers import AutoTokenizer` in
  `app/proxy/tokenizers.py`, which is API-stable across versions.

## [2026.05.12.0] — 2026-05-12

### Fixed
- **Auth: duplicate `Authorization` header on the wire.**
  `auth-fetch.ts:headersToObject` emitted the header twice — once
  lowercase (from `Headers.forEach`, per WHATWG spec) and once
  canonical-cased. fetch's case-insensitive header merge concatenated
  them into `Authorization: Bearer xxx, Bearer xxx`; the API parsed the
  comma-suffixed string as a malformed token and returned 401, putting
  the entire v2 UI into an infinite refresh loop the moment a user
  logged in. Same treatment applied to `X-CSRF-Token`.
  **Production-breaking.**
- **vLLM subprocess: `--dtype` / `--max-model-len` emitted as None.**
  `cmd_builder.build_vllm_args` appended both flags unconditionally. When
  the model row had `dtype IS NULL` or `max_model_len IS NULL`,
  `asyncio.create_subprocess_exec` raised `expected str, bytes or
  os.PathLike object, not NoneType` — the subprocess never started, and
  the failure surfaced only as a cryptic `last_error` on the row. Both
  flags now omitted when unset (vLLM picks safe defaults).
- **Supervisor: missing `/data/hf-token` killed every load.** Previously
  `Path(hf_token_path).read_text()` raised `FileNotFoundError` whenever
  the operator had not seeded the token file. Now treated as an empty
  string, mirroring the graceful handling already present in
  `pull_task._read_hf_token`. Public HF repos load without a token;
  private repos still need it to be present.
- **Auto-pull on register.** Backend exposed `POST /api/models/{id}/pull`
  but the v2 UI had no caller anywhere, leaving freshly-registered models
  stuck at `status="registered"` forever. `AddModelModal` now
  fire-and-forgets the pull request immediately after a successful POST
  /api/models; pull failures surface on the detail page via the row's
  `last_error`.
- **Frontend: `BACKEND_URL` baked into the standalone build.** Next 15
  freezes `rewrites()` into `routes-manifest.json` at build time, so
  setting `BACKEND_URL` only at runtime was a no-op — every `/api/*`
  request 404'd at the UI tier before reaching the API. Wired in the
  Dockerfile build stage; works for both the local compose stack and
  `deploy/hub/compose.yaml` because both expose the API as `api:8080`.
- **Playwright selector drift.** Cleaned up login, create-token,
  token-row, and the modal Add/Create button scoping to match the v2
  redesign so the happy-path spec can drive the UI without
  pointer-events fights.

### Tests
- `tests/e2e/happy-path.spec.ts` rewritten with stale-state cleanup,
  CSRF cookie warmup, dialog-scoped clicks, `waitForResponse(201)` on
  register, and a 600s test budget that accommodates the 120s pull +
  60s load + vLLM warmup windows. Full path verified GREEN against
  the local compose stack: login → add opt-125m → auto-pull (753 MB)
  → load → /v1/completions → mint+rotate token → unload → delete,
  ~32s warm.

## [2026.05.11.2] — 2026-05-11

### Fixed
- HuggingFace model cache now lives on the dedicated `vllm-warden-hfcache`
  PVC (mounted at `/root/.cache/huggingface`) instead of the small data
  PVC subdirectory `/data/hf-cache`. The old path was a `@property` on
  `Settings`, so even though `deploy/hub/compose.yaml` already mounted the
  dedicated PVC at `/root/.cache/huggingface`, the pull task wrote to a
  directory living on the data PVC — which `shutil.disk_usage(...).free`
  reported as 0 GiB for any non-trivial model. `Settings.hf_cache_dir` is
  now a real dataclass field populated from `VW_HF_CACHE_DIR` (default
  `/root/.cache/huggingface`). The Hub template now requires
  `VW_COOKIE_SECRET` (the api refused to boot without it) and surfaces
  `VW_CONTAINER_GPU_COUNT`. The api service also publishes container
  port 8080 so the K8s compose-bridge emits a ClusterIP Service for
  the ui→api hop.
- vLLM subprocess no longer dies at import time with
  `ImportError: cannot import name 'Gemma3Config' from 'transformers'`.
  Root cause: `requirements.txt` pinned `transformers==4.47.1`, which
  downgraded the version that ships with the `vllm/vllm-openai` base
  image. The base image's `vllm==0.20.0` imports `Gemma3Config` (added
  in transformers 4.50), so every model load failed with `rc=1` before
  the model config was even parsed. Pin removed; the base image's vLLM
  dependency now constrains transformers transitively. Our only direct
  use is `from transformers import AutoTokenizer` in
  `app/proxy/tokenizers.py`, which is API-stable across versions.

## [2026.05.11.0] — 2026-05-11

### Added
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` is now baked into the per-model
  subprocess env baseline (`app/runtime/env_builder.py`) and added to
  the hard-locked keys list. Fixes a latent footgun on heterogeneous
  GPU hosts (e.g. pw_prod `bonus` mixes Quadro RTX 4000 on slot 0
  with A4000s on 1–2): without `PCI_BUS_ID`, NVML may reorder devices
  by compute capability, so `gpu_indices=[1,2]` could land on the
  wrong physical GPUs. vLLM warned about this at load time.
- gpt-oss-20b preset template + `extra_env` plumbing + cmd_builder
  typo cure (bundle landed via `feat/model-templates`).

## [2026.05.10.2] — 2026-05-10

### Fixed
- Pull progress on `/models` now actually advances. `run_pull` was
  inserting models with `pulled_bytes=0, pulled_total=NULL` and never
  writing to either column, so the static `<progress>` element rendered
  nothing useful for the entire download. The new `run_pull` (1) calls
  `HfApi.repo_info(files_metadata=True)` upfront to persist
  `pulled_total` (the bar's denominator), (2) spawns a background poller
  that walks `{hf_cache_dir}/models--{org}--{repo}/blobs/` once per
  second and writes the summed file size to `pulled_bytes`, and (3)
  writes a final `pulled_bytes` after `snapshot_download` returns
  (`app/models/pull_task.py`). Regression tests in
  `tests/unit/models/test_pull_progress.py` pin total/final-bytes
  persistence and the `_snapshot_dir_size` walk.

### Added
- `GET /api/models/{id}/pull/progress` streams pull state as SSE.
  Emits one JSON event per second with `{status, bytes, total,
  last_error}` and terminates when the row leaves
  `pulling`/`registered` (or when the row is missing — emits
  `{"status": "missing"}` and closes, since EventSource doesn't
  surface HTTP errors usefully). `app/web/models/_card.html` now
  consumes this stream with `EventSource`, computing a rolling
  download rate and ETA from successive byte deltas, and reloads
  the page when the row reaches a terminal status. Auth required
  via `require_session_json` (401 not redirect, so the FE stream
  doesn't silently swallow a session expiry).

## [2026.05.10.1] — 2026-05-10

### Fixed
- Pull/Load/Unload buttons on `/models` now reload the page on success
  and surface the server's error body via `alert()` on failure
  (`hx-on::after-request` handlers in `app/web/models/_card.html`) —
  previously `hx-swap="none"` ate the response and gave the operator
  no visible feedback.
- `GET /models/{id}/logs` now resolves: route was missing (only the
  card linked to it). New page streams stderr via `EventSource` against
  the existing `/api/models/{id}/logs/stream` endpoint
  (`app/web/models/logs.html`, route added in `app/models/routes_web.py`).
- Welcome page no longer flashes raw JSON into the DOM after admin
  bootstrap. `app/web/setup/welcome.html` was using `hx-swap="innerHTML"`
  on `<body>` against a JSON endpoint; replaced with `fetch` +
  `window.location.href` redirect.

### Added
- Disk-space safeguard is now overridable. `POST /api/models/{id}/pull`
  accepts `?force=true` which threads through to `run_pull(force=True)`
  and skips the cache-free-bytes precheck. `_card.html` exposes this as
  a "Force pull" button (with `hx-confirm`) for `registered` and
  `failed` models. Regression test
  `tests/unit/models/test_pull_endpoint.py::test_pull_endpoint_passes_force_flag`
  pins the wiring.
- `/models` now renders an "Available capacity" banner showing combined
  VRAM across allowed GPUs and free space in the HF cache directory.
  Per-model card adds a hint with `Weights: X GiB · Assigned VRAM: Y GiB
  (GPUs [...])` and a red warning when weights exceed assigned VRAM
  (`app/models/routes_web.py` builds the rows; `_card.html` and
  `index.html` render them).

### Added
- `internal/web/`: wired the setup wizard's HTTP surface on top of
  the Phase-1 server. New options `WithWizard(launcher, drafts, bus)`
  and `WithAuth(mw, sessions, csrf, handlers, hasAdminPassword)` are
  applied independently; both panic on nil collaborators because
  `cmd/sidekick` constructs them together at boot. Routes land in
  sibling files: `auth_routes.go` mounts `GET/POST /login` and
  `POST /logout` (login skips CSRF on first POST since no session
  exists yet); `wizard_routes.go` mounts `GET/POST /setup/step/{n}`,
  `POST /api/v1/wizard/launch`, `GET /api/v1/wizard/progress` (SSE
  with snapshot-then-delta replay, drop-oldest ring, 25s
  `:keepalive` heartbeat), and `POST /api/v1/wizard/reset` behind
  `LoadSession → RequireAuth → RequireCSRF` (the only wizard
  endpoint that lives on the protected surface — for an admin
  re-running the wizard on a fresh box). `middleware.go` adds
  `wizardOnlyWhenUnsetup` which 303s `/setup/*` to `/login` once
  the admin password is set. `handleRoot` now branches on
  `hasAdminPassword`: 303 to `/setup/step/1` when unset, to
  `/login` when set, falling back to the Phase-1 placeholder when
  no auth is wired (preserves existing tests). Templates: 9 new
  files under `web/templates/{auth,setup,partials}/` — full HTML
  page for login, plus a step chrome (`step.html`) that branches
  on `StepName` to dispatch to the five step bodies (GPU pick,
  HF token, model repo+revision, runtime args, review+launch).
  Step 5 wires Alpine inline against the SSE endpoint to walk
  through phase events and follow the launcher's `redirect` on
  `done`. New dep: `github.com/google/shlex` for `extra_args`
  parsing on step 4. New integration test
  (`wizard_integration_test.go`) walks a real `Server` (with
  real `DraftStore` + `Bus` + auth middleware) end-to-end:
  GET `/` → POST steps 1-4 → POST `/launch` (202) → subscribe
  to `/progress` until `done` → GET `/` now redirects to
  `/login` (the unsetup gate flipped) → anonymous reset is
  rejected. Only HF fetch, vLLM subprocess+prober, and
  `ConfigSaver` are faked. Closes #771.

- `internal/wizard/`: new package implementing the five-step setup
  wizard's domain layer. Contents: `Step`/`Draft`/`FieldError`
  state types with `Validate` cross-field rules; `DraftStore` —
  schema-versioned (`DraftSchemaVersion=1`) JSON persistence at
  `<dataDir>/wizard-draft.json` with mode 0o600 atomic writes via
  temp-file rename; `Bus` — bounded ring buffer (cap 64) +
  drop-oldest pub/sub for SSE progress events with monotonic
  sequence numbers and replay-from-cursor semantics; `Promote` /
  `BuildConfig` — converts a healthy draft to `config.File` with
  bcrypt-hashed admin password and canonical-order vLLM args;
  `Launcher` — single-flight orchestration that calls the
  injected `Fetcher` (HF model fetch), `Process` (vLLM start +
  stop) and `Prober` (HTTP health) under a goroutine bound to
  `context.Background()` (per #9 regression guard) with
  10-minute health timeout, 2-second poll interval, panic
  recovery, LIFO cleanup deferral, and benign-tolerant draft
  delete on success. `launchertest/` exposes `FakeFetcher` /
  `FakeVLLM` fakes plus `NewHTTPHealthServer` for downstream
  tests in #771. 38 tests cover happy-path promote, save error
  propagation, draft round-trip, schema-version rejection,
  bus replay/seq/time-override/drop-oldest semantics, and the
  full launcher state machine including health timeout
  (deterministic via injected clock), single-flight rejection,
  panic recovery, kill-window enforcement, request-context
  isolation, and benign delete-failure. No HTTP routes yet —
  those land in #771. Closes #770.

- `internal/auth/`: new package providing the auth primitives
  consumed by the setup wizard and protected routes. Contents:
  bcrypt password hashing (`Hash`/`Verify`, cost 12); in-memory
  session store with 7-day absolute and 24-hour idle TTLs and a
  background sweeper; per-session double-submit CSRF tokens with
  constant-time comparison; per-IP sliding-window login rate limit
  (5/minute); chi-compatible middleware bundle (`LoadSession`,
  `RequireAuth`, `RequireCSRF`); login/logout HTTP handlers with
  `next=` open-redirect defence (`SafeNextParam`). 26 unit tests
  cover happy-path, TTL eviction, header/form CSRF acceptance,
  rate-limit window slide and per-IP isolation, redirect
  sanitization (8 cases), and middleware context propagation. No
  routes are wired yet — that lands with the wizard handlers
  (#771). Closes #769.

### Docs
- `docs/superpowers/specs/2026-05-02-441-setup-wizard-design.md`:
  added the design spec for the 5-step first-run setup wizard
  (#441). Covers state model, auth (bcrypt + session + CSRF +
  rate-limit), wizard backend (draft store, progress.Bus,
  single-flight launcher), HTMX/Alpine/Tailwind frontend, per-step
  contracts, and a 5-phase implementation plan that splits into
  children #769–#773. Refs #441.

### Test
- `internal/gpu/detect_test.go`: added
  `TestAutoDetector_Poll_NoBackendReportsGPU` mirroring the existing
  `TestAutoDetector_NoBackendReportsGPU` so `AutoDetector.Poll`'s
  final `noGPUInfo` return is exercised. Closes the 1/7-statements
  (14.3%) coverage gap flagged in QA of MR !7. Closes #6.

### Removed
- `internal/vllm/fakes_test.go`: dropped three never-called test
  helpers — `(*fakeRunner).lastChild`, `(*fakeChild).writeStdout`,
  and `(*captureLogger).all`. Flagged by golangci-lint v1.61
  `unused` linter during QA of MR !10. No callers in the repo;
  receiver fields they touched (`created`, `stdout`, `lines`) are
  retained because other helpers/methods still use them. Closes #7.

### Test fixtures
- `internal/gpu/testdata/bonus-3gpu.csv`: replaced the synthesized
  three-GPU fixture (Quadro RTX 4000 + 2× RTX A4000) with a live
  capture from the `bonus` host. Column count, format, and static
  fields (name, memory.total, power.limit, compute_cap) all match
  the synthesized values exactly; only runtime-state columns
  (memory.free/used, power.draw, temperature.gpu) refresh to the
  observed values. The live capture exercises a sub-1W
  `power.draw` reading (`0.53`) and a 1 MiB `memory.used` reading
  that the synthesized fixture didn't hit. Closes #4.

### Polish
- `internal/config/config.go`: `Manager.Save` now fsyncs the parent
  directory after the temp-file rename so the directory-entry
  update is durable, not just the file contents. Best-effort —
  errors are swallowed because the user-visible save has already
  succeeded. Closes a durability gap on NFS / overlayfs under
  power loss; ext4 and Docker bind-mounts (the common deployment
  target) were already safe. Item 2 of #5.
- `internal/db/sqlite.go` + `internal/db/sqlite_test.go`: introduced
  a tiny `nowFunc` clock seam for `CreateAPIKey` and rewrote
  `TestListAPIKeys_OrderingAndRevoked` to drive it from a
  fixed-timestamp pair instead of `time.Sleep(1100ms)`. Drops
  ~1.1s from the `internal/db` package suite. Item 3 of #5.
- CI polish from MR !4 review (#2): pinned `golangci/golangci-lint`
  image from `latest-alpine` to `v2.12.1-alpine` so an upstream
  default-linter change can no longer break the pipeline
  unannounced; added a one-line comment in the build job's CalVer
  block explaining that `TAGS` is *reset* (not appended) on a
  CalVer tag push to keep `:production` / `:latest` from a prior
  main build from bleeding onto the immutable tag.
- `internal/web/server.go`: corrected the `NewServer` docstring to
  match reality — Options are applied BEFORE `s.routes()`, not
  after. The inline comment and the code already agreed; only the
  docstring lagged. No behavioural change. Spotted in post-merge
  code-review of MR !10.
- `internal/web/vllm_routes.go`: extracted the SSE keepalive
  interval to a package-level `sseKeepaliveInterval` var (default
  unchanged at 25s) so a unit test can shorten it. Added
  `TestLogsStreamEmitsKeepaliveComments` in `vllm_routes_test.go`
  — overrides the interval to 50ms, asserts >=2 `:keepalive`
  comment frames arrive on an idle SSE stream within ~250ms, and
  cancels the request context before the test exits to avoid a
  goroutine leak. Closes #8.

### Added
- Phase 2 vLLM process manager (#440):
  - `internal/vllm`: child-process supervisor for the vLLM server.
    `Manager` (constructed via `NewManager(Options{...})`) owns a
    state machine — `stopped → starting → running → stopping →
    stopped`, plus `crashed` / `failed` for unexpected exit and
    failed-startup respectively. `Start(ctx, LaunchSpec)` /
    `Stop(ctx)` / `Restart(ctx)` / `Status()` are the public API,
    safe for concurrent use. `LaunchSpec` decouples per-launch
    settings (model id, binary path, argv, env, pip extras, health
    URL) from the long-lived Manager so a single Manager can serve
    successive different models.
  - Lifecycle: SIGTERM-then-SIGKILL escalation on Stop with a 30s
    grace window (configurable via `Options.ShutdownTimeout`); HTTP
    `/health` probe every 5s while in `starting` or `running`;
    `Options.StartupTimeout` (default 5min) caps the wait for the
    first 200 before declaring the launch a `failed` and tearing
    the child down. Crash detection auto-restarts with exponential
    backoff (1s → 60s by default, doubling per crash, reset on
    successful `running`).
  - `Runner`/`Child`/`HealthChecker`/`PipRunner` interfaces let
    tests swap in deterministic fakes; production
    implementations (`execRunner`, `httpHealthChecker`,
    `PipExecRunner`) live in `runner.go` and `health.go`. The
    production `execRunner` sets `Setpgid: true` and signals via
    `syscall.Kill(-pid, sig)` so SIGTERM propagates to vLLM's
    tensor-parallel worker children when the warden is PID 1.
  - `internal/vllm/logs.go`: `LogBuffer` is the in-memory + on-disk
    + SSE fan-out for the child's stdout/stderr. 5000-line ring
    buffer (per #440 spec), optional on-disk file at
    `{LogDir}/vllm-{unix-nano}.log` (raw bytes persisted before
    line-splitting so a partial trailing line survives a crash),
    bounded per-subscriber channels with detach-on-slow-consumer
    so a stalled SSE listener never blocks the writer.
  - `internal/web`: `NewServer` now accepts functional `Option`s
    so independent feature packages can attach routes without
    conflicting on the constructor signature. `WithVLLM(sup,
    specProvider)` registers `/api/v1/vllm/{status,start,stop,
    restart,logs/tail,logs/stream}`. `/start` returns 409 when the
    supervisor is already running; `/stop` is idempotent (200 even
    when already stopped); `/logs/stream` is SSE with a 25s
    keepalive comment.
  - `cmd/sidekick/main.go`: constructs the supervisor at boot,
    auto-starts vLLM if `config.json` has a model AND the model is
    `ready` in the SQLite store, stops the supervisor (with up to
    35s of slack) before HTTP shutdown on SIGTERM/SIGINT.
- Phase 2 curated model catalog + GPU fit verdicts (#439):
  - `internal/models/curated_models.json`: 18 editorial entries
    covering the families called out in the issue — gpt-oss-20b
    (featured) and gpt-oss-120b, Llama 3 / 3.1 / 3.2, Mistral 7B,
    Mixtral 8x7B, Qwen2.5 (7B/14B/32B/Coder), DeepSeek V2 Lite
    (Chat/Coder), Phi-3.5-mini and Phi-3-medium, Gemma 2 9B/27B.
    Each row carries the full schema from the issue
    (`id`, `name`, `arch`, `params_total`, `params_active`,
    `min_vram_gb`, `recommended_tp`, `quantizations`,
    `default_max_len`, `disk_gb`, `vllm_extra_args`,
    `vllm_pip_extra`, `hf_gated`, `description`, `tags`).
  - `internal/models/curated.go`: embeds the JSON via `go:embed`,
    decodes it once behind `sync.Once`, and validates every row
    against the architecture invariants (dense ⇒ active==total,
    moe ⇒ active<total, all required fields populated). Public
    surface: `LoadCurated()`, `LookupCurated(id)`, plus the
    `Architecture` enum with a guarded `UnmarshalJSON` so a typo'd
    `arch` value fails the build instead of silently misclassifying.
  - `internal/models/compatibility.go`: GPU fit calculator. Adds
    `FitStatus` (green/yellow/red), `FitReport`, and two entry
    points: `FitCurated(model, gpu.GPUInfo)` for catalog rows and
    `EstimateFit(paramCount, precision, info)` for the
    paste-arbitrary-repo path. Aggregates per-GPU memory across a
    multi-GPU host, honours `recommended_tp`, applies a 20%
    headroom for KV-cache + activations, degrades gated repos
    from green to yellow with a "request access" note, and
    surfaces a yellow-with-quantization hint when the model only
    fits at sub-FP16 precision. `CuratedWithFit(info)` pairs every
    catalog row with its verdict in one call for the API.
  - `internal/web/catalog_routes.go`: new HTTP surface.
    `GET /api/v1/catalog` returns the curated list with per-row
    fit verdicts and a host-GPU summary; `GET /api/v1/catalog/{id}`
    returns one row by exact id (URL-encoded slashes accepted),
    404 on miss, 405 on non-GET, 400 on empty id. The handler
    swallows GPU-detection errors so the wizard remains usable
    when `nvidia-smi` is hung or absent — no GPU collapses every
    verdict to red rather than 500-ing the request.
  - `internal/web/server.go`: `NewServer` now takes functional
    options. `WithGPUDetector(gpu.Detector)` wires the detector
    used by the catalog route. `cmd/sidekick/main.go` passes
    `gpu.NewAutoDetector()` so the live binary serves real
    verdicts.
  - Tests: 22 new test functions across `internal/models` and
    `internal/web`. Coverage: `internal/models` 83.6%,
    `internal/web` 87.3%. Acceptance criterion from the issue
    (gpt-oss-20b is curated with the expected MoE shape) is
    pinned by `TestCuratedIncludesGPTOSS20B`.

- Phase 2 HuggingFace client + model downloader (#438):
  - `internal/hfclient`: pure-Go HTTP client for the Hub. No third-party
    HF SDK, no CGO. `Client` (constructed via `New(Options{...})`)
    exposes `Whoami` (`/api/whoami-v2`), `Search` (`/api/models`),
    `ModelInfo` (`/api/models/{repo}`), `ListFiles`
    (`/api/models/{repo}/tree/{rev}?recursive=true`) and `Download`
    (`/{repo}/resolve/{rev}/{path}`). Sentinel errors `ErrUnauthorized`
    (folds 401+403), `ErrNotFound`, `ErrRateLimited` for typed
    branching in callers. `User-Agent: vllm-warden/0.1
    (+https://podwarden.com)` and a 30s per-call HTTP timeout (vs.
    Go's no-timeout default) on every request.
  - `Download` semantics: HEAD probe reads `Content-Length` and
    `Accept-Ranges`, GET sends `Range: bytes=N-` when a `<file>.part`
    is present and the server advertises range support. Server
    returning `200 OK` to a Range request triggers a truncate-and-
    restart. `.part` is only deleted when the server cannot resume,
    fsync'd before rename, atomically renamed onto the final path.
    Size mismatch leaves the `.part` in place for retry. Throttled
    progress callback at 200ms cadence; final 100% callback fired
    unconditionally.
  - LFS handling: `ListFiles` flattens the LFS sub-object so
    `RepoFile.Size` is the canonical weight size (not the
    `.gitattributes` pointer's ~130 bytes) and `RepoFile.SHA256` is
    the LFS oid for non-empty LFS rows.
  - `ModelInfoResponse.IsGated()` normalises HF's quirky
    boolean-or-string `gated` field.
  - `internal/models`: `Manager` orchestrator composes `hfclient` +
    `db.Store` + on-disk layout. `LocalPathFor(repo, rev)` returns
    `{baseDir}/{owner}/{name}/{rev}`. `EnsureDiskSpace` runs
    `syscall.Statfs` against `baseDir` and rejects allocations that
    would leave less than 2 GiB headroom (`ErrInsufficientDiskSpace`).
    `EstimateVRAMBytes(params, precision)` returns the FP16 / INT8 /
    INT4 weight footprint for the wizard's GPU-fit check.
    `DownloadModel` ties it together: `ModelInfo` → resolve sha →
    create row → `ListFiles` → space check → per-file
    `UpsertModelFile` + `Download` + aggregate `UpdateModelProgress`
    → `UpdateModelStatus(ready)`. Gated repos surface `ErrGatedRepo`
    up front.
  - `internal/db/migrations/0002_models.sql`: `models` table
    (`id`, `repo`, `revision`, `status` enum
    `pending|downloading|ready|failed`, `local_path`,
    `total_size_bytes`, `downloaded_size_bytes`, `created_at`,
    `updated_at`, `error_message`; `UNIQUE(repo, revision)`,
    `idx_models_status`) and `model_files` table with
    `FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE`,
    `UNIQUE(model_id, path)`, `idx_model_files_model`. All
    timestamps unix seconds matching the existing convention.
  - `Store` extended with eight model-state methods (`CreateModel`
    is a `INSERT ... ON CONFLICT DO UPDATE ... RETURNING` upsert so
    re-entry from the wizard returns the existing row rather than
    failing; `UpdateModelStatus` clears `error_message` on every
    non-failed transition; `DeleteModel` is idempotent;
    `UpsertModelFile` keys on `(model_id, path)`). Tests cover
    insert, conflict-on-repo-revision, status transitions, progress
    updates preserving `total` when zero is passed, FK cascade, and
    the SQLite `CHECK` constraint on illegal status strings.
- Phase 2 config & SQLite data layer (#437):
  - `internal/config`: env-driven `Runtime` (`VLLM_WARDEN_DATA_DIR`,
    `VLLM_WARDEN_LISTEN_ADDR`, `VLLM_WARDEN_LOG_LEVEL`) plus a JSON
    `Manager` for `/data/config.json` (`model`, `vllm_args`,
    `admin_password_hash`, `hf_token`, `trust_proxy_auth`). Atomic
    write (temp-file + rename + fsync), 0o600 perms, RWMutex-guarded
    cache, race-clean tests.
  - `internal/db`: pure-Go SQLite store on
    `modernc.org/sqlite v1.38.2` (`CGO_ENABLED=0` build still
    static), embedded `migrations/*.sql` via `go:embed`, idempotent
    migration runner with `schema_version` bookkeeping.
  - Initial schema (migration `0001_init.sql`): `api_keys`
    (id, name, key_hash, key_prefix, rate_limit, created_at,
    last_used_at, revoked + idx_api_keys_hash), `usage_log`
    (id, key_id, timestamp, endpoint, prompt_tokens,
    completion_tokens, model, latency_ms + idx by ts and (key, ts)),
    `usage_hourly` (key_id, hour, request_count, prompt_tokens,
    completion_tokens; PK (key_id, hour)).
  - `Store` interface + `*SQLiteStore` impl: `CreateAPIKey` (mints
    `pwk_<64hex>` with 256-bit entropy, SHA-256 hash stored,
    plaintext returned once), `ListAPIKeys`, `GetAPIKeyByPlaintext`
    (rejects revoked keys as `ErrKeyNotFound`, bumps last_used_at),
    `RevokeAPIKey` (idempotent), `RecordUsage` (single-tx
    log+rollup upsert), `HourlyForKey`, `PruneUsageLogOlderThan`.
  - `cmd/sidekick/main.go` wires the new pieces in minimally:
    `LoadRuntime` → `db.Open(rt.DataDir)` → `config.Manager.Load`
    (missing file is the expected first-boot state); `--addr`
    flag retained as an explicit override above
    `VLLM_WARDEN_LISTEN_ADDR`.

### Changed
- GPU detect contract refinements (#3): clarified `GPU.MemoryUsedBytes` docstring (populated by both `Detect` and `Poll`); normalised `ComputeCapability` parser-boundary sentinels (`[N/A]`, `N/A`, `-`, `[Not Supported]`, case-insensitive) to empty string via new `normalizeOptional` helper; added `gpu.DefaultDetectTimeout = 10 * time.Second` constant with `Detector` interface guidance for callers; new `[N/A]` testdata fixture (`old-driver-na-compute-cap.csv`) and unit tests covering the normaliser, the `ComputeCapability` round-trip, and `AutoDetector.Poll` dispatch / fall-through / error-propagation paths.
- Replaced Phase 0 CI echo-skeleton with real lint+test+build+push pipeline (F-3, issue #1).
  - **lint** stage: `golangci-lint` (staticcheck, govet, errcheck, ineffassign, gofmt) via `golangci/golangci-lint:latest-alpine`.
  - **test** stage: `CGO_ENABLED=1 go test -race -coverprofile=coverage.out ./...` with `coverage.out` artifact (7-day TTL); gcc installed on Alpine to enable the race detector.
  - **build** stage: `docker buildx` → `registry.podwarden.com/podwarden/apps/vllm-warden`; tag scheme aligned with workspace standard (`sha-{commit}` always, `{branch-slug}` always, `:staging` on develop, `:production`+`:latest` on main (manual gate), `v{YYYY.MM.DD.N}` on CalVer tag push).
  - Added `.golangci.yml` (staticcheck, govet, errcheck, ineffassign, gofmt; no stylistic noise).
- Fixed data race in `internal/web.Server`: `httpSrv` field now guarded by `sync.Mutex` so concurrent `ListenAndServe` and `Shutdown` calls do not race. Detected by `-race` when wiring up the CI test stage.
- Hardened HTTP server with `ReadHeaderTimeout` (gosec G112).
- Added graceful shutdown on SIGINT/SIGTERM with a 10s drain budget so
  `docker stop` (and systemd / k8s) exit cleanly with status 0 instead
  of being SIGKILL'd after the orchestrator's grace window.

### Added
- GPU detection module (#436): pure-Go `internal/gpu` package built around a
  vendor-neutral `Detector` interface and a `GPUInfo`/`GPU` data shape consumed
  by `internal/models/fit.go` (issue #439).
  - `NvidiaDetector` shells out to `nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,power.draw,power.limit,temperature.gpu,compute_cap --format=csv,noheader,nounits`
    and a second host-wide `driver_version,cuda_version` query; CSV parser is
    tolerant of `[N/A]` / `[Not Supported]` (coerce to zero) and short rows
    (degrades to per-field zero, never aborts the batch).
  - VRAM is normalised to bytes at the parser boundary; `GPUInfo.TotalVRAMBytes`
    is pre-summed for the fit logic.
  - "No GPU" is a non-error result (`Count: 0`, `Vendor: VendorNone`), not an
    error: matches both nvidia-smi missing and nvidia-smi exiting with
    "No devices were found" (exit 9). `Detect` only errors on a real failure.
  - `Detect` (one-time static facts) and `Poll` (live utilisation/power/
    temperature/used-VRAM) accept a `context.Context` honoured by
    `exec.CommandContext`.
  - `AutoDetector` composes per-vendor backends (NVIDIA only in v1) and routes
    to whichever reports a GPU; structured for ROCm/Intel additions later.
  - Test seam: `NvidiaDetector.runner` accepts a fake exec function. Tests
    drive committed `testdata/` fixtures (3-GPU bonus shape, single RTX 3090,
    Tesla M40 with `[Not Supported]` power readings) plus a real `sh -c` exec
    to verify the no-devices stderr-matching path.
- Phase 1 scaffolding (#435): `go mod init`, full directory tree per design
  spec §Project Structure, `go:embed` for `web/templates/` and `web/static/`,
  pinned Dockerfile (`vllm/vllm-openai@sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d`),
  Makefile (`build`, `test`, `vet`, `fmt-check`, `docker-build`, `docker-run`),
  stub `cmd/sidekick/main.go` serving a Tailwind-CDN placeholder on `:8080`,
  `internal/web` server with `httptest` smoke coverage.
- HTMX 2.0.4 and Alpine.js 3.14.8 wired into `web/templates/layout.html` via
  pinned CDN scripts (foundation for #441 wizard interactivity); regression
  tripwire test in `internal/web/layout_libs_test.go` asserts both pinned
  URLs survive future edits.
- Phase 0 kickoff: repo created, CI skeleton green, bonus-node GPU prerequisite verified.
