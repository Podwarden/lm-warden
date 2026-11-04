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

- **Watch** a key's requests and tokens per minute over the last hour, its whole history, and (opt-in, off by default) a live view of its prompts and replies in the god-mode dock.

The Stats page also draws every key's requests on one chart: a mark per request, coloured by key, sized by the tokens it generated, at the height of its duration.

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
