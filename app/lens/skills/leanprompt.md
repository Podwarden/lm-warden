---
name: leanprompt
description: What leanprompt is, how its rules map to Prompt Lens findings, and how leanprompt profile detect reads LM Warden's engine facts.
---

# leanprompt

[leanprompt](https://github.com/Podwarden/leanprompt) is a set of skills that teach AI coding assistants (Claude Code, Codex, OpenCode, Cursor, anything that reads `AGENTS.md`) to make apps that call LLMs cheaper and faster. Its promise: audit an app, find the places where it pays for the same tokens twice, give each a fix and a way to prove the fix helped. It is not a proxy, not a prompt compressor and not a hosted service. It is by the PodWarden Team; install with `npx leanprompt init`.

## The skills

- `llm-call-audit`: scans a codebase for LLM calls and reports findings, each with a rule id, file and line, the fix and the verify step, plus a score.
- `prefill-cache`: makes prompts the provider's or engine's cache can reuse (rule ids `LP-PC-01` to `LP-PC-12`).
- `generation-economy`: stops paying to generate or prefill again what one call could have produced (`LP-GE-01` to `LP-GE-10`).
- `measure-loop`: measure, change, measure; no fix is claimed without numbers. It is the loop of the `prompt-lens` skill here.
- `lmwarden`: self-hosting on LM Warden, and the engine settings that change cache reuse.

## How the rules map to Prompt Lens findings

Prompt Lens findings name the leanprompt rule that explains the fix (the `leanprompt` column of the `prompt-lens` skill's table, without the `LP-` prefix). The ones you meet most:

| Prompt Lens | leanprompt rule | In short |
|---|---|---|
| E-01, C-06 | LP-PC-03, LP-PC-02 | No ids, timestamps, versions or paths at the top of the prompt; stable content first, without hurting meaning |
| E-02 | LP-PC-04 | Fixed tool order and JSON key order |
| E-03, C-01 | LP-PC-05 | Tool lists that grow only at the end |
| E-04, C-02, C-07 | LP-PC-06 | Append-only history: never rewrite a message already sent |
| E-05 | LP-PC-09 | Keep or drop reasoning the same way every turn |
| E-06 | LP-PC-01 | Put cache marks where the provider needs them |
| E-07, C-05, R-02 | LP-PC-08 | Cache-aware compaction: keep system prompt and tools identical, append the summary |
| E-08 | LP-PC-11 | Keep pauses inside the cache lifetime, or expect a cold prefix |
| E-09 | LP-PC-10 | Stable image and attachment placement |
| R-01, R-03, R-07, R-08 | LP-GE-03 | Output and thinking budgets that fit the task |
| R-04 | LP-GE-10 | Structured, typed tool arguments |
| R-05, R-06 | LP-GE-05 | Retry only on retryable errors; return errors that say what to change |

leanprompt's rule files carry a "Verify" section: the provider's usage field across two requests, and kvsig ids (see the `kvsig` skill).

## The score

leanprompt scores an app from measured calls: `leanprompt score = floor(100 x (0.6 x prefill efficiency x stability + 0.4 x (1 - generation waste)))`, with at least 50 calls counted. The `lens_score` tool (and `GET /api/lens/me/score`) returns these inputs for the calling key, computed from LM Warden's stored request figures.

## Fill leanprompt.yaml's engine section from LM Warden

leanprompt keeps a target profile in `leanprompt.yaml`; for a self-hosted engine it has an `engine` section: cache block size, prefix match unit, speculative decoding on or off, and context length. leanprompt can read these from LM Warden:

```
export LMWARDEN_API_KEY=<your LM Warden key>   # never print it or commit it
leanprompt profile detect --lmwarden https://<your lm warden host> [--model <name>]
```

The command calls LM Warden's engine endpoint (`GET /api/me/engine`, the `engine_facts` tool here) with the key from `LMWARDEN_API_KEY` and writes the `engine:` section: `block_tokens`, `match_unit_tokens`, `speculative` (true when the engine runs a speculative method) and the context length from `max_model_len`. When the key may use several models, pass `--model`. The key needs the engine-facts permission; see the `engine-facts` skill. If the command is not available in the installed leanprompt, ask the user to update it; do not guess the values.
