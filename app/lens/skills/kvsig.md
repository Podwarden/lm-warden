---
name: kvsig
description: What kvsig is, how Prompt Lens uses it to group requests into sessions and find a changed system prompt or tool list, and the CI test that fails when a prompt's stable head changes.
---

# kvsig

[kvsig](https://github.com/Podwarden/kvsig) is a specification and reference library for keyed, content-derived ids of LLM chat conversations, so a gateway or router can tell which conversation a stateless request belongs to, without any client cooperation. Install the reference libraries with `pip install kvsig` (Python) or `npm install @kvsig/core` (TypeScript for Node, browsers, Deno, Bun and edge runtimes). The scheme comes from LM Warden's conversation-identity layer; kvsig is its vendor-neutral home.

Ids are HMAC-keyed per deployment: without the key they reveal nothing about content. They are message-level identity, not token-block hashing (that stays with the engines).

## What each id identifies

| Id | Identifies | Changes when |
|---|---|---|
| `sys` | the system and developer messages | the system prompt changes |
| `tools` | the tool list | a tool is added, removed or edited, or tools are reordered |
| `head` | `sys` and `tools` together | either of them changes |
| `msgs[i]` | message `i` of the body | that message changes |
| `nodes[i]` | the whole prompt up to and including message `i`, like a git commit | anything before or at `i` changes |
| `conv` | the conversation: loose system messages and the first real message | a different system prompt or opening message |

Every id except `conv` also has a loose form (`loose_sys` and so on) that ignores harness noise: reminder blocks, date-only lines, whitespace, reasoning, member order in tool arguments. Use exact ids for anything that depends on the bytes the model sees, such as prompt caching; use loose ids for identity.

## How Prompt Lens uses it

- **Sessions.** Requests of one key with the same `conv` and a chain of `nodes` that continue each other belong to one conversation; LM Warden joins them into a session (and links a compaction or a forked subagent to its parent when it can). That is why a session needs no session id from your client, though one is used when sent.
- **A changed head.** When a request does not extend the previous one's prefix, the ids say where it broke: `sys` differs while `loose_sys` is equal means the system prompt changed only in noise (a date line, a reminder); `tools` differs with the same `conv` means the tool list changed (a tool added or reordered); a shorter shared `nodes` prefix means history was rewritten. Prompt Lens turns this into the first broken part (`sys`, `tools` or a message), the cause and, for the system prompt, the line number and the kind of value on it (a date, a time, a path, an id). That is findings E-01 to E-04.
- Everything Prompt Lens stores about this is codes, counts and keyed hashes, never text.

## The CI test: fail when the stable head changes

kvsig lets a test assert what the model's cache needs: build the request your app would send for two fixed inputs (two users, two days, two working directories), compute the ids with the reference library, and fail when the stable head differs. The same test catches a tool reordered by a refactor, a timestamp added to a system prompt, or a history store that re-serialises old messages. Which id to assert on:

- the system prompt must not change between inputs: `sys` equal (read a failure as: `sys` differs but `loose_sys` is equal, so the prompt changed only in noise);
- the tool list must be the same: `tools` equal;
- history must be append-only: for the second turn, `nodes` of the first request must be a prefix of the second request's `nodes`.

```python
from kvsig import Key, canonicalize, ids

key = Key.generate(epoch=1)

def head_of(body):
    canon = canonicalize(body, "anthropic.messages")  # or "openai.chat" or "openai.responses"; None means no ids
    assert canon is not None
    return ids(canon, key)

def test_the_stable_head_does_not_change():
    a = head_of(build_request(user="ann", today="2026-11-01"))
    b = head_of(build_request(user="bo", today="2026-11-02"))
    assert a.sys == b.sys and a.tools == b.tools
```

`build_request` is your app's own function that builds the request body. Check the calls against the kvsig README for the version you install. Under the spec label `kvsig/1` and the loose profile `loose/1` the ids are pinned by test vectors and never change in a later release; `loose/2` is experimental, so do not assert on it. leanprompt's `measure-loop` skill uses this test (see the `leanprompt` skill).
