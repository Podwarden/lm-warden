# Connect a coding agent

LM Warden speaks the three protocols coding agents use:

- OpenAI Chat Completions (`/v1/chat/completions`)
- the Anthropic Messages API (`/v1/messages`)
- the OpenAI Responses API (`/v1/responses`)

So Claude Code, the Anthropic and OpenAI SDKs, OpenCode, Aider, Grok CLI, Continue, Cline, Cursor and Codex CLI connect with a base URL and a key. Their requests show up per key, per session, with the phase each one is in.

## Use the Connect page

You do not write the config by hand. In the console, open **Connect**: it picks your client, your model and your key and writes the files to paste. Each client's setup there shows the same files this guide describes, with your values filled in, and a test request that proves the path works.

## Clients and what works

| Client | Id in Connect | Protocol | What works |
| --- | --- | --- | --- |
| Claude Code | `claude-code` | Anthropic Messages | Full support |
| Anthropic SDK | `anthropic-sdk` | Anthropic Messages | Full support |
| OpenCode | `opencode` | OpenAI Chat Completions | Local models only |
| Aider | `aider` | OpenAI Chat Completions | Local models only |
| Grok CLI | `grok-cli` | OpenAI Chat Completions | Local models only |
| OpenAI SDK | `openai-sdk` | OpenAI Chat Completions | Local models only |
| Continue | `continue` | OpenAI Chat Completions | Documented, not run: local models only |
| Cline | `cline` | OpenAI Chat Completions | Documented, not run: local models only |
| Cursor | `cursor` | OpenAI Chat Completions | Documented, not run: chat models only, and your warden must be reachable from the internet (Cursor sends the requests from its own servers) |
| Codex CLI | `codex-cli` | OpenAI Responses | Local models only |

"Full support" means the client can use a model you loaded or pass through to its hosted provider. "Local models only" means it uses models loaded on this warden. "Documented, not run" means the setup follows the client's own documentation and the request it sends was tested against the warden, but the editor itself was not run; the Connect page shows each one's limits. Codex CLI speaks only the Responses API, which the warden translates to Chat Completions for the served model.

## What every client needs

- **Base URL.** The warden's address. OpenAI-style clients take it with a `/v1` suffix (`http://gpu-box:8080/v1`). Claude Code takes the root URL in `ANTHROPIC_BASE_URL` and adds `/v1/messages` itself.
- **Key.** An API key from **API Keys** (a `vw_` key). See [API-KEYS.md](API-KEYS.md).
- **Model.** The served name of a model you have loaded, as listed by `GET /v1/models`. That list also reports each model's `max_model_len`.

## Related

- [API-KEYS.md](API-KEYS.md): one key per app, priorities, pausing and rotating.
- [AGENT-TUNING.md](AGENT-TUNING.md): let your agent pick and tune the model.
- Missing a client? Ask in [GitHub Discussions](https://github.com/Podwarden/lm-warden/discussions/new?category=ideas).
