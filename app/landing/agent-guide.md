# Driving LM Warden from a coding agent

You are an agent with shell access, asked to choose, load and tune models on an LM Warden install. This guide is the short version of {{raw}}/documents/API.md; read that when you need detail.

## Credentials

- The operator gives you an **admin token** (`vwa_…`) in `VW_ADMIN_TOKEN` and the warden's address in `W` (for example `http://gpu-host:8080`).
- Send it on every control-API call: `-H "Authorization: Bearer $VW_ADMIN_TOKEN"`. With an admin token there is no login, no cookie and no CSRF header.
- The whole API is described at `curl -s -H "Authorization: Bearer $VW_ADMIN_TOKEN" $W/api/openapi.json`. Read it before guessing a path or a field.
- An admin token is not an inference key. To send test traffic to `/v1`, issue an inference key with POST /api/tokens (body `{"name": "agent-bench", "expires_in_days": 1}`), use its `plaintext` (`vw_…`) as `Authorization: Bearer` on `/v1/*`, and revoke it with DELETE /api/tokens/{token_id} when you finish.

## Look before you load

1. **Hardware.** GET /api/system/gpus lists the cards: index, name, total, used and free VRAM, and which processes hold memory. (/api/setup/gpus is for first-run setup only and returns 404 afterwards.)
2. **What is already there.** GET /api/models lists registered models with their `status`.
3. **Will it fit?** POST /api/models/fit-preview answers before anything is downloaded, with a `green` / `yellow` / `orange` / `red` verdict and the arithmetic. Body: `hf_repo` (or a GGUF `filename`), `gpu_indices`, and optionally `backend` (`vllm` or `llamacpp`), `max_model_len`, `data_parallel_size`. Rule candidates out here, cheaply.

## The model lifecycle

1. **Register.** POST /api/models with `served_model_name`, `hf_repo`, `gpu_indices` (required, a subset of the allowed cards), and optionally `backend` (`vllm`, the default, or `llamacpp` with a GGUF `filename`), `max_model_len`, `data_parallel_size`. Tensor parallel size is derived: `tensor_parallel_size × data_parallel_size == len(gpu_indices)`.
2. **A starting point.** GET /api/models/{model_id}/suggest-config returns a suggested configuration for the registered row. Start there rather than from nothing.
3. **Pull.** POST /api/models/{model_id}/pull (202). Progress is a server-sent event stream at GET /api/models/{model_id}/pull/progress: read the `data: {…}` lines one at a time; it is not one JSON document.
4. **Load.** POST /api/models/{model_id}/load (202). Poll GET /api/models/{model_id} until `status` is `loaded`, or `failed` with `last_error`. Large models take minutes. A refusal (for example, the card is busy) also arrives this way, after the 202.
5. **Unload.** POST /api/models/{model_id}/unload frees the cards. One loaded model per GPU: unload before loading another on the same cards.
6. **Change settings.** PATCH /api/models/{model_id}/settings on an unloaded row (GET /api/models/{model_id}/settings shows every field), then load again.

## Measuring

- **Stress run.** POST /api/models/{model_id}/stress with `{"mode": "quick", "acknowledge_disruption": true}` measures the context length the loaded model can really serve. It deliberately pushes the engine until it fails, so the engine restarts during the run. Never start one on a warden that other people are using. `thorough` takes longer and measures more.
- **What it found.** GET /api/models/{model_id}/capabilities is the full record: limits with where each came from, history and advice.
- **Apply.** POST /api/models/{model_id}/stress/apply writes the measured `max_model_len` and reloads the engine.
- **Your own benchmark.** Send real prompts to `/v1/chat/completions` with the inference key, streaming, and time the first token and tokens per second yourself. Vary concurrency (1, 4, 16) and prompt length. Use prompts like the operator's real work.
- **The warden's view.** GET /api/stats/v2/throughput and GET /api/stats/v2/latency report prefill and generation tokens per second, time to first token and durations as the warden measured them; GET /api/stats/live streams the requests in flight.

## Rules

- Rule candidates out with fit-preview before pulling: downloads are tens of GB.
- Change one thing at a time and keep a table of every configuration and its numbers.
- If a load fails, read `last_error` before retrying. Do not loop.
- Do not edit files on the host or restart containers; everything goes through the API.
- Finish by leaving the best configuration loaded, unloading everything else you loaded, revoking the inference key you issued, and reporting your table.
