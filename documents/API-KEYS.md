# API keys

Give each colleague, laptop, CI job and agent its own key. A key's page shows what it spent and how long its requests waited in the queue (median and p95) over any window you drag or type.

## What you can do with a key

- **Rotate** a key with a grace window, so clients using the old value keep working while you switch them over.
- **Set a priority** from P0 to P9. A waiting P9 request always goes before a P8 one.
- **Pause** a key. New requests are refused while the ones already running finish:

```
$ curl -i http://gpu-box:8080/v1/chat/completions \
    -H "Authorization: Bearer vw_…" -d '{…}'
HTTP/1.1 403 Forbidden
content-type: application/json

{"detail":"token paused"}
```

- **Watch** a key's requests and tokens per minute over the last hour, its whole history, and (opt-in, off by default) a live view of its prompts and replies in Prompt Lens (the page `/ui/prompt-lens`, opened from the key's page with this key and its earlier keys preset).

The Stats page also draws every key's requests on one chart: a mark per request, coloured by key, sized by the tokens it generated, at the height of its duration.

## What someone signed in with a key sees

A colleague who signs in with their own key (API key tab on the sign-in page) gets their own area, with a **Stats** page for that key alone:

- **Your speed**: time to first token, decode speed, prompt speed and duration over the window, and a chart with a dot per request. Prompt speed is prompt tokens divided by the time to first token, so it includes cache effects.
- **Your tokens / second**, prompt and completion per minute.
- **Your cache**: how much of their prompts was reused from their own earlier prompts.
- **Your requests**: a chart of their finished requests and a table of the ones running now, with an estimated wait for a request that has not started.

A key that has not been used lately does not get empty panels: when the chosen window (1h, 6h, 24h or 7d) holds none of its requests, the page shows the same length of time up to the key's last request instead, and the line under the model tabs says so ("No requests in the last hour · showing the hour up to your last request, 3 d ago"), with **Back to now**. A key that was never used shows the empty page.

It shows nothing fleet-wide. No other key's requests, names, sessions or prompts, no client IP (the session identifies a request), no install, server load, engine or host figures, and no count of requests ahead of theirs. The wait estimate comes from their own requests only. No cache figure on the page can tell them that someone else sent the same text.

**Prompt Lens for a key holder** (called god mode before 2026-11-22) follows the platform. When it is on (the switch in Settings, General, Privacy, or `VW_PROMPT_LENS_ENABLED`), everyone signed in with their own key gets a **Prompt Lens** page (`/ui/key/prompt-lens`, in their area's nav, full screen), and an eye on each of their own rows in the Stats Requests table opens that request there; when it is off, neither is there. The page lists their own requests, live and recent, and shows each one's prompt part by part (which parts match their own earlier requests), its thinking, reply and tool calls as they stream, the timing, the tokens and the finish reason. A section opens whole in a large view with search. Captures stay in server memory for a short time, not on disk, unless an admin set the key to keep its last 50 requests on disk, and the page tells them so and that admins can see the captures too. It shows nothing about other keys: no client IP, no other key's requests, and a prompt part another key made cached reads "new". Their stream follows their sign-in: it ends when they sign out, the sign-in expires, or the key is paused, rotated or revoked.

## Leaving is a `base_url` change

LM Warden speaks `/v1/chat/completions` and `/v1/models`, the Anthropic Messages API at `/v1/messages` and the OpenAI Responses API at `/v1/responses`. A client changes two lines:

```diff
  import os
  from openai import OpenAI

  client = OpenAI(
-     base_url="https://api.openai.com/v1",
-     api_key=os.environ["OPENAI_API_KEY"],
+     base_url="http://gpu-box:8080/v1",
+     api_key=os.environ["LM_WARDEN_KEY"],   # a vw_ key from API Keys
  )

  reply = client.chat.completions.create(
-     model="gpt-4o-mini",
+     model="gpt-oss-20b",   # the served name of a model you loaded
      messages=[{"role": "user", "content": "What is a KV cache?"}],
  )
```

The weights sit in an ordinary HuggingFace cache that `make export-hf-cache` hands you as a tarball, and the exact argv each engine was started with is one GET away, so a working setup can be rebuilt without LM Warden. Read the diff backwards and that is how you leave.

## Related

- [CODING-AGENTS.md](CODING-AGENTS.md): connect a coding agent.
- [AGENT-TUNING.md](AGENT-TUNING.md): let your agent pick and tune the model.
