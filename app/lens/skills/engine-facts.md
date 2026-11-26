---
name: engine-facts
description: What each engine field means for prompt caching - block size rounding, the match unit, the dropped last block - and how to use the engine_facts tool.
needs: engine
---

# Engine facts

`engine_facts` (REST: `GET /api/me/engine`) reports static facts about the engine behind each model your key may use. They explain cache behaviour that no prompt change can fix, and they let you size a prompt for the engine. The answer has no traffic figures, no other keys and no host names. A value the warden cannot determine right now (the model is not loaded, the engine has not reported yet) is `null` and the field is listed in `unknown`; it is never guessed. Your key needs the engine-facts permission.

| Field | Meaning |
|---|---|
| `model` | The served model name, the one you send in `model`. |
| `backend` | The engine family serving it (`vllm`, `llamacpp`). |
| `running` | The model is loaded now. An unloaded model reports only its configuration. |
| `prefix_caching` | Prefix caching is on in the running engine. If false, nothing is reused whatever the prompt does. |
| `engine_block_tokens` | The KV cache block size in tokens. The engine reuses a prompt in **whole blocks**: the tail after the last full block is recomputed on every request. A classic page is 16 tokens; a hybrid model can use 1,584 or more, so a prompt shorter than two blocks is never served from cache. |
| `engine_match_unit_tokens` | How many tokens a prefix match steps by (`--prefix-match-unit`). If none is set it equals the block size. A smaller unit lets reuse follow the prompt more closely than the block. |
| `engine_drops_last_block` | With speculative decoding that reads the target model's hidden states (EAGLE, MTP and similar), one more unit at the end of every cache hit is recomputed. |
| `speculative` | The speculative method (`mtp`, `eagle`, ...), or `null` when off. |
| `max_model_len` | The longest context the model accepts, in tokens. |
| `replicas` | How many engine copies serve the model. |
| `cache_aware_routing` | The warden keeps a conversation on the replica that holds its cache (only meaningful with more than one replica). |
| `unknown` | The fields above that are `null` because the warden cannot say. |

## Using them

- **Expect rounding.** With block size B and match unit U, the most a request can reuse is its stable prefix rounded down to a multiple of U (or of B when U equals B), minus one more unit when `engine_drops_last_block` is true. A prompt whose stable prefix is shorter than that unit reuses nothing. This is not a finding against your prompt: Prompt Lens counts block rounding as the engine's, not yours.
- **Size the stable head.** Put everything that does not change per request first, and make it longer than a block or two, so most of the prompt is whole blocks that stay cached.
- **Read cache figures with it.** `cached_tokens` below the stable prefix by less than one block is rounding. A bigger gap is a prompt change; use `lens_explain_request`.
- **leanprompt.** `leanprompt profile detect --lmwarden <url>` writes `block_tokens`, `match_unit_tokens`, `speculative` and the context length from these fields into `leanprompt.yaml` (see the `leanprompt` skill).
