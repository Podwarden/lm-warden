# Let your coding agent pick the model

Choosing a model for your cards means choosing a quantisation, an engine (vLLM or llama.cpp), how to split it across GPUs, how much context to give it, the KV-cache type and whether speculative decoding helps. Each guess costs a download and a load. A coding agent can run that search while you do something else, because everything the console does is an API call.

## Set it up in three steps

1. **Install LM Warden** on the GPU host (see [INSTALL.md](INSTALL.md)) and finish the setup wizard.
2. **Issue an admin token** in Settings → Access Tokens, with an expiry of 30 days. Export it as `VW_ADMIN_TOKEN`, and the warden's address as `W`, in the shell your agent uses.
3. **Start your agent**: Claude Code, Codex CLI or another one that can run shell commands. It can use a hosted model for this job; the model it is tuning is the one on your cards.

## What the agent uses

| Step | Endpoint |
| --- | --- |
| See the cards | `GET /api/system/gpus` |
| Check a model fits, before downloading | `POST /api/models/fit-preview` |
| Register, get a starting config | `POST /api/models`, `GET /api/models/{model_id}/suggest-config` |
| Pull and load | `POST /api/models/{model_id}/pull`, `POST /api/models/{model_id}/load` |
| Measure the real context ceiling | `POST /api/models/{model_id}/stress`, then `POST /api/models/{model_id}/stress/apply` |
| Read the warden's numbers | `GET /api/stats/v2/throughput`, `GET /api/stats/v2/latency` |
| Send test traffic | an inference key from `POST /api/tokens`, on `/v1/chat/completions` |

The agent reads [AGENT-GUIDE.md](AGENT-GUIDE.md) first. Your warden serves it at `/agent-guide.md`: a short description of these calls written for a language model, with the mistakes to avoid.

## Prompts to paste

Replace `https://your-warden` with your warden's address, and the capitalised names with models you have in mind.

### 1. Survey the hardware and shortlist models

```
Read https://your-warden/agent-guide.md and follow it. My admin token is in $VW_ADMIN_TOKEN and the warden is at $W.
List the GPUs this warden may use. Then shortlist three open-weight models from Hugging Face that fit them, for agentic coding with tool calling.
Check each with POST /api/models/fit-preview and keep only those that fit with at least 65536 tokens of context.
Show me a table: model, quantisation, engine, GPUs, context, and why. Do not download anything yet.
```

### 2. Choose the engine and layout for one model

```
Read https://your-warden/agent-guide.md and follow it. Credentials are in $VW_ADMIN_TOKEN and $W.
For the model MODEL_REPO, decide between vLLM and llama.cpp, and between tensor parallel and data parallel replicas on my GPUs.
Use fit-preview for every layout you consider. Register and load the best two layouts one after the other, send each the same 20 streaming requests through an inference key you issue, and compare time to first token and tokens per second at concurrency 1 and 8.
Keep the winner loaded, unload the other, revoke the key, and report the numbers.
```

### 3. Find the real limits and apply them

```
Read https://your-warden/agent-guide.md and follow it. Credentials are in $VW_ADMIN_TOKEN and $W.
The loaded model is MODEL_NAME. Nobody else is using this warden right now.
Run a quick stress run on it, wait for it to finish, read its capabilities and apply the measured context.
Then benchmark it at concurrency 1, 4 and 16 with prompts of about 2k and 30k tokens, and give me a table of time to first token and tokens per second.
```

### 4. Compare two candidates on my workload

```
Read https://your-warden/agent-guide.md and follow it. Credentials are in $VW_ADMIN_TOKEN and $W.
Compare MODEL_A and MODEL_B for my workload: the prompts in ./workload/*.txt.
Load each in turn with its best layout, replay the prompts at concurrency 4, and record latency, tokens per second, and whether each answer is correct by your judgement.
Write the results to ./model-comparison.md with a recommendation, and leave the recommended model loaded.
```

## Before you start

- An admin token has full admin rights over the warden: models, keys and settings. There are no narrower scopes yet. Give it a short expiry.
- A stress run deliberately crashes an engine to find its limit. Run these prompts on a warden nobody else depends on.
- Downloads are large. The prompts ask the agent to check fit before pulling, but watch your disk.
- Revoke the admin token in Settings → Access Tokens when the job is done.

## Related

- [CODING-AGENTS.md](CODING-AGENTS.md): connect a coding agent.
- [API-KEYS.md](API-KEYS.md): one key per app.
