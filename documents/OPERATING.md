# Operating LM Warden day to day

Everything here assumes an install that already passes `make smoke`.
Getting there is [INSTALL.md](INSTALL.md); the things to read before the
first model is loaded are [HAZARDS.md](HAZARDS.md).

## Day-to-day

| Command | What it does |
|---|---|
| `make start` | start the stack (detached) |
| `make stop` | stop and remove the containers; volumes and data stay |
| `make restart` | stop + start, re-reading `.env` and the override |
| `make logs` | follow live logs (`make logs S=api` for one service) |
| `make status` | container state and health |
| `make smoke` | check the front door end to end (`/`, `/_landing`, `/ui/`, `/api/csrf`, `/healthz`) |
| `make pull` | pull the release named by `VERSION` in `.env` and restart on it |
| `make config` | print the fully merged compose config — what actually runs |
| `make preflight` | re-run the installer's host checks |
| `make save-images` / `make load-images` | move a release between hosts as a tarball |
| `make export-hf-cache` / `make import-hf-cache` | move the model cache the same way |
| `make uninstall` | stop and delete the data volumes (asks first) |
| `make help` | list every target |

**Upgrading:** set `VERSION` in `.env` to the new release (or
`./install.sh --version vX`) and `make pull`. When the stack files themselves
changed, `git pull && ./install.sh` refreshes `docker-compose.yml` and the
override and re-pins `VERSION`; `.env` is kept.

**Upgrading across the rename to `lm-warden`.** The product is now called
LM Warden (formerly LLM Warden and, before that, vLLM Warden). The repository
moved from `Podwarden/vllm-warden` (and briefly `Podwarden/llm-warden`) to
`Podwarden/lm-warden`, and the images from
`registry.podwarden.com/podwarden/apps/vllm-warden{,-ui}` (and briefly
`.../llm-warden{,-ui}`) to `.../lm-warden{,-ui}`. Nothing an install stores
was renamed: the volumes, container names and the `/data/vllm-warden.db`
database keep their `vllm-warden` names, so an existing install upgrades in
place.

- Every release is still published under both old image names as well (same
  digest) until **2026-12-31**, so an install that keeps pulling
  `.../vllm-warden:${VERSION:-latest}` or `.../llm-warden:${VERSION:-latest}`
  keeps getting updates until then. Re-run
  `./install.sh` from an updated checkout at your leisure before that date: it
  rewrites `docker-compose.override.yml` to the new image names and keeps `.env`.
- `git pull` keeps working (GitHub redirects the old URL); point the remote at
  the new one when convenient:
  `git remote set-url origin https://github.com/Podwarden/lm-warden.git`.
- **Keep your existing checkout directory.** The volumes belong to the Compose
  project name. `install.sh` writes `COMPOSE_PROJECT_NAME=vllm-warden` into
  `.env`, which pins the names (`vllm-warden_vw-data`, `vllm-warden_vw-hfcache`,
  `vllm-warden_caddy-*`) whatever the directory is called. Without that line --
  an `.env` written by hand, or plain `docker compose up` in a checkout with no
  `.env` -- Compose names the project after the directory, and a fresh
  `git clone` now lands in `lm-warden/`: that stack would start on new, empty
  volumes. Check with `docker compose config | head -1` (it prints
  `name: vllm-warden`), and if it does not, add `COMPOSE_PROJECT_NAME=vllm-warden`
  to `.env` before the first `make start`.

Once running:

- **UI** — `http://YOUR-HOST:8080/ui/`
- **OpenAI API** — `http://YOUR-HOST:8080/v1/chat/completions`
- **Control API** — `http://YOUR-HOST:8080/api/` (JWT-gated)
- **Health** — `http://YOUR-HOST:8080/healthz`

For gated models (Llama, Mistral, gpt-oss) you need a HuggingFace token. The
first-run wizard asks for one, and you can change it later under
**Settings → General**.

**On HTTP vs HTTPS.** Plain `http://` works — the session cookies follow the
scheme the browser actually used, so a LAN install stays signed in. It is still
an evaluation posture: the API key and the session both cross the network in the
clear. For anything shared, terminate TLS in front of the `:8080` listener and
tell the warden its public URL with
`./install.sh --origin https://llm.example.com`; the cookies then carry
`Secure`. If your proxy sets `X-Forwarded-Proto`, set `VW_TRUST_PROXY_ORIGIN=1`
in `.env` so the warden believes it.

**The page at `/`.** Every install serves a public landing page at its root,
with its own video, screenshots and fonts under `/_landing/assets/` and a
`/robots.txt`, `/llms.txt` and `/llms-full.txt` beside it. It is private by
default: the page is marked `noindex` and `robots.txt` disallows everything.
Turn it off entirely with the `landing_page_enabled` setting (**Settings**),
and every one of those routes 404s except `robots.txt`, which stays
disallow-all. Only if this instance should be found by search engines, set
`VW_LANDING_CANONICAL_URL` in `.env` to its public origin, e.g.
`VW_LANDING_CANONICAL_URL=https://llm.example.com`, and re-run `./install.sh`
or `make restart`: the page then carries a canonical link and `index,follow`,
and `robots.txt` points at a `/sitemap.xml`. A value with a path, query or
credentials is ignored rather than half-applied.

---

## Managing API keys

**API tokens** lists every key; click a name — there or on the stats page —
to open that key's page at `/ui/tokens/{id}`. Everything you can do to a key
is on it:

- **Rename** it with the pencil next to the name (Enter saves, Esc cancels).
  Names need not be unique.
- **Pause** it. A paused key's next request gets
  `403 {"detail":"token paused"}`, so a client can tell "switched off" from
  "wrong key" (401).
  Requests already running finish and nothing queued is aborted. **Resume**
  lets the next request through. An expired key, or one revoked past its
  grace window, cannot be paused — it is already refused. An old key still in
  its rotation grace window can be, which cuts it off early.
- **Priority**, P0 to P9. Scheduling is strict: a waiting P9 request always
  goes before a P8 one, so a low priority can wait indefinitely on a busy box.
  If you want fair sharing, give every key the same priority. Priority is the
  only per-key fairness control: per-key rate limits were removed in
  v2026.09.18.2, because they counted prompt tokens only. An API body that
  still sends `rate_limit_tps` is refused with 422, and
  `VW_RATE_LIMIT_WINDOW_S` is no longer read — a leftover line in `.env` is
  ignored.
- **Test** it: whether the key is live, paused, expired or revoked, and
  whether the proxy answers.
- **Rotate** it. A new key keeps the name; the old one becomes
  `<name> (old N)` and keeps working through a grace window you choose
  (24 hours by default). The page then moves to the new key.
- **Delete** it.
- **Rotation history** lists the keys this one replaced, oldest first, each
  linking to its own page with when it was rotated and whether its grace
  window is still open.

**Usage.** The charts show tokens per minute (prompt or completion), requests
per minute, queue wait, and latency (first token or full response) — median
solid, 95th percentile dashed — over 1h, 6h, 24h or 7d, or a custom period:
drag the window on the history strip, which covers the key's whole life with
each rotation marked, or type **From** and **To** and press **Apply**. The
period is kept in the URL, so a reload or a shared link shows the same view.
**Include earlier keys** (on by default) adds the keys this one was rotated
from. Bins are chosen on the server, at most about 360 per chart, and a
minute with no traffic counts as zero rather than being skipped. Presets
refresh on their own; a custom period does not.

Queue wait and latency come from per-request timings, which are recorded per
key from v2026.09.18.1 on and kept as long as the rest of request history —
30 days by default (`VW_REQUEST_HISTORY_RETENTION_DAYS`). A period before that is
hatched and labelled "Not recorded", so it does not read as idle. On a very
busy key the timings are computed from an even sample of its requests, and
the chart says so; request and token counts are always exact.

The same actions and series are in the API — see
[Managing a key from the API](API.md#managing-a-key-from-the-api).

---

## Where request content can end up

By default no prompt or completion text is stored anywhere. `request_history`
— the table behind the requests chart and the per-key rollups — has no column
that holds it; it carries identifiers, token counts, timings and finish
reasons, and that is the whole of the metadata path. Two diagnostic features
can capture content. Both are off by default, and with both off the proxy's
forward path is the code it would be in a build that never had them.

**God mode** (`VW_GODMODE_ENABLED`) streams prompts, completions and any
inline images to the admin session — the way to settle "the model said X"
when the client is somebody else's code. You watch it one key at a time: with
the flag on, each key's page has a **God mode** dock along its bottom edge.
Opening the dock replays that key's recent requests from memory and then
streams its new ones live, including the keys it was rotated from when
**Include earlier keys** is on. Closing the dock closes the stream; a closed
dock holds no connection, and the dock always starts closed. With the flag
off the dock is not shown at all. There is no separate god-mode page and no
all-keys view in the UI. The dock decides only what you watch, not what is
captured: while the flag is on, every key's traffic goes into the ring whether
or not a dock is open. What it keeps lives in a bounded
in-memory ring: 2000 events and ~4 million characters by default
(`VW_GODMODE_RING_EVENTS`, `VW_GODMODE_RING_CHARS`), oldest evicted first,
with inline images in a separate bounded store beside it. Nothing reaches the
database or the disk, and all of it is gone when the container restarts. A
prompt longer than `VW_GODMODE_MAX_PROMPT_CHARS` (16000) is captured as its
head plus a `VW_GODMODE_PROMPT_TAIL_CHARS` tail, so the newest turn survives a
large repeated system prompt — display-only, and never a change to what is
forwarded. The bearer token never reaches the viewer; only the key's name
does. While it is on, that text sits in the api container's memory and any
holder of the admin session can read it: there is no separate role that
cannot, because the project has no roles.

**The content log** (`VW_CONTENT_LOG_ENABLED`) is the one that writes to disk.
Each logged request appends a JSON line — prompt, completion, token counts,
finish reason — to `VW_CONTENT_LOG_PATH`, which defaults to
`/data/logs/content.jsonl`. That is the persistent data volume, the same one
that holds the SQLite database, so the file is inside whatever backs that
volume up. It is scoped as well as gated: a request is logged only when the
flag is on **and** its token id appears in `VW_CONTENT_LOG_TOKENS`, an
explicit comma-separated allowlist that is empty by default. There is no "log
everything" mode. `VW_CONTENT_LOG_MAX_CHARS` (40000) caps the prompt and the
completion within each record.

Three properties of that file to know before enabling it:

- **It has a size ceiling, not rotation.** `VW_CONTENT_LOG_MAX_CHARS` bounds
  each record, not the file; `VW_CONTENT_LOG_MAX_BYTES` (512 MiB) bounds the
  file. On reaching it the writer stops and logs one warning naming the file
  and the limit — deliberately, so the incidents the log was armed to capture
  are not deleted to make room for newer ones. Nothing prunes, truncates or
  rotates it; archive or remove the file, or turn the flag off, to resume.
  `0` or a negative value switches the ceiling off, the same convention
  `VW_ENGINE_LOG_MAX_BYTES` uses.
- **It shares a volume with the database.** The ceiling exists because
  letting it grow until that volume is full is an outage, not merely a large
  file.
- **It is created owner-only.** The `logs` directory is created `0700` and
  the file `0600`, both applied at creation rather than left to the process
  umask. An existing file keeps whatever mode it has — the product never
  `chmod`s it — so a content log written by a build before v2026.09.07.1 keeps
  its old, umask-derived mode until you `chmod 600` it once.

Retention past the ceiling, and shipping the file anywhere, are yours to
arrange.

`VW_RUNAWAY_MODE=log` has no sink of its own. It attaches its trip signal to a
record the content log was already going to write, so it captures nothing
unless content logging is enabled *and* the request's token is on that
allowlist.

`VW_GODMODE_ENABLED` and all five content-log variables are in `.env.example`,
each with what its default does; set them in `.env`, or in the environment of
whatever runs the api container.
