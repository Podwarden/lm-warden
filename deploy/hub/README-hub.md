# LM Warden

LM Warden was formerly called LLM Warden (and, before that, vLLM Warden). Website: [lmwarden.com](https://lmwarden.com).

Self-hosted, OpenAI-compatible LLM inference with a guided setup wizard. Deploy any model from HuggingFace and expose it on your network in minutes — no command-line tuning required.

## Migrating from v1

v1 shipped a single `vllm-warden` service on port 8080 that served both the HTML setup wizard and the OpenAI-compatible API.

v2 splits this into two services:

- **`api`** (port 8080, internal only) — FastAPI backend: OpenAI `/v1/*` endpoints, JWT auth, model management.
- **`ui`** (port 3000, public) — Next.js frontend: dashboard, setup wizard, admin UI. All browser traffic goes here; the UI proxies `/api/*` and `/v1/*` requests to `api` via Docker DNS (`http://api:8080`).

### Steps to migrate an existing v1 install

1. **(Optional) Set `VW_FRONTEND_ORIGIN`** — set this env var to your public UI origin (e.g. `https://vllm.example.com`) to harden CSRF/Origin enforcement on auth endpoints. It is **optional**: if left blank the API falls back to the localhost default so a fresh install boots without it — set it once you front the UI with a public origin.

2. **Rebind ingress** — point your PodWarden ingress rule from `service: vllm-warden, port: 8080` to `service: ui, port: 3000`. The API port (8080) is now internal-only and should not be published directly.

3. **Re-deploy via Hub UI** — use "Update to v2.0.0" in the PodWarden stack view. PodWarden's volume-reuse logic detects matching volume names (`vllm-warden-data`, `vllm-warden-hfcache`) and preserves all operator data (config, credentials) and HF model cache automatically.

4. **Re-log-in once** — v2 replaces the old SessionMiddleware with JWT auth. All active browser sessions are invalidated on first boot. Existing API/MCP bearer tokens (`sk-...`) remain valid.

Full operator runbook: [documents/OPERATING.md](https://github.com/Podwarden/lm-warden/blob/main/documents/OPERATING.md)

## What you get

- **Browser-based setup wizard** — pick a model, paste your HuggingFace token, set an admin password, and go. No `vllm serve …` flags to memorise.
- **OpenAI-compatible API** at `http://<host>:8080/v1` — works with the OpenAI Python/Node SDKs, LangChain, LlamaIndex, OpenWebUI, Continue, and anything else that speaks the OpenAI protocol.
- **Persistent state** — `data` volume holds your config and admin credentials; `hfcache` volume holds downloaded model weights so restarts don't re-download.
- **GPU-aware** — reserves one NVIDIA GPU by default; bump `gpu_count` and `vram_request` in PodWarden before deploy if you have bigger hardware.

## Hardware floor

The default reservation (1 GPU, 24 GiB VRAM, 2 CPU, 16 GiB RAM) is sized for a 7B-class model in `bfloat16`. For 13B/30B/70B models, raise `gpu_count` and `vram_request` in the PodWarden UI before deploying — the wizard otherwise fails to load the model.

## After install

1. Open `http://<host>:8080` — you'll land on `/setup`.
2. Walk the wizard: model repo, HuggingFace token, admin password, GPU/memory tunables.
3. The wizard launches vLLM, polls `/health` until ready, then forwards you to the dashboard.
4. The OpenAI endpoint at `/v1/chat/completions` is now live.

## Documentation

Full docs: [github.com/Podwarden/lm-warden/documents](https://github.com/Podwarden/lm-warden/tree/main/documents)
Source: [github.com/Podwarden/lm-warden](https://github.com/Podwarden/lm-warden)
