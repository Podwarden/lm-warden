"""A fake OpenAI-compatible engine, with a knob for every outcome class.

This started as a two-endpoint stub for the proxy integration tests. The stress
harness needs much more from it: the whole point of that design is that the
seven outcome classes, the crash/recover path and the wedge path are all
testable **with no GPU**, and the only way that is true is if a fake engine can
be told to produce each of them on demand and deterministically.

Two rules govern what is faked here.

**The shapes are recorded, not invented.** ``tests/fixtures/stress/llamacpp_recorded.json``
holds verbatim captures from llama-server b10731, and the error envelope, the
usage frame and the unknown-model behaviour below are copied from it rather
than guessed. In particular:

* a context refusal is ``error.type = "exceed_context_size_error"`` with exact
  ``n_prompt_tokens``/``n_ctx``, which is *structured* evidence a classifier can
  act on without parsing prose;
* usage arrives in a **final frame whose ``choices`` array is empty**, carrying
  ``prompt_tokens_details.cached_tokens``;
* an **unknown model name is not an error** — it returns 200 with empty content
  and ``finish_reason: "length"``. That is the nastiest trap in the whole
  feature, because it trips the REASONING_ONLY and ABRUPT gates and so publishes
  a harness bug as model degradation. It is on by default here precisely so a
  harness that forgets to send the real ``served_model_name`` fails its own
  tests.

**The engine answers the probes correctly when it is healthy.** A fake that
returned canned text would make every graded probe WRONG, every outcome
DEGRADED, and every end-to-end runner test vacuous. :func:`_answer` therefore
understands the six probes in ``app/stress/probes.py`` and answers them, so PASS
is reachable and a degradation knob means something.

Knobs resolve in three layers, widest first: environment (``VW_FAKE_*``, for the
subprocess form), the app's mutable knob object (``POST /_fake/knobs`` or
``FakeEngine.set``), then per-request query parameters. The last is what lets one
server serve a healthy probe and a refusing probe in the same test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields

from aiohttp import web

# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------

#: Every mode a probe can be answered in. The names match the outcome class or
#: quality gate they are there to produce, so a test reads as a statement about
#: the taxonomy rather than about this file.
MODES = (
    "ok",  # -> PASS
    "empty",  # -> DEGRADED(empty)
    "reasoning_only",  # -> DEGRADED(reasoning_only)
    "short",  # -> DEGRADED(short)
    "repeat",  # -> DEGRADED(repeating)
    "abrupt",  # -> DEGRADED(abrupt): finish_reason=length
    "truncated",  # -> DEGRADED(abrupt): stream stops with no terminal frame
    "wrong",  # -> DEGRADED(wrong)
    "rate_limit",  # -> BACKPRESSURE (429)
    "error",  # -> ERRORED (envelope, but no refusal confirmation)
    "hang",  # -> TIMEOUT
    "wedge",  # -> TIMEOUT forever, /health still 200
    "exit",  # -> CRASHED
)

#: Roughly the ratio ``app/stress/probes.py`` aims with. Keeping the same
#: constant here means a probe asking for ~4096 tokens is reported by this
#: engine as ~4096 prompt tokens, so a context ceiling knob is expressed in the
#: same unit the search reasons in.
CHARS_PER_TOKEN = 4


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int | None) -> int | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None else float(v)


@dataclass
class Knobs:
    """What this engine does next. Every field is independently settable."""

    mode: str = "ok"

    #: Prompt-token count above which the engine refuses. This is the context
    #: ceiling; a search that never reaches it never sees a refusal.
    refuse_above_tokens: int | None = None
    #: Both are configurable because the two engines disagree: vLLM refuses at
    #: 400, llama.cpp at 500, and ``refusal_confidence`` is derived from which.
    refuse_status: int = 400
    refuse_error_type: str = "exceed_context_size_error"
    #: Reported as ``n_ctx`` in the refusal envelope. Defaults to the ceiling.
    n_ctx: int | None = None

    #: Die above a prompt size, with no refusal and no degradation first. This
    #: is the measured 2026-09-03 Quadro behaviour -- clean answers, then a dead
    #: process at ~2.2k tokens -- and it is why the crash cannot be demoted to
    #: an exception path.
    crash_above_tokens: int | None = None
    #: Degrade above a prompt size, which is the COMMON failure: still 200,
    #: still healthy, answers no longer usable.
    degrade_above_tokens: int | None = None
    degrade_mode: str = "short"

    #: Model a REASONING model: this many tokens of chain-of-thought are
    #: emitted before the answer can start. When the request's ``max_tokens``
    #: does not clear it, the engine returns 200 with reasoning only, empty
    #: content and ``finish_reason: "length"`` -- the measured gpt-oss-20b-gguf
    #: behaviour that made the first production run fail every probe on a
    #: healthy model (2026-09-04). Verbatim, on port 10007:
    #:     max_tokens=16 -> content='', reasoning truncated, finish=length
    #:     max_tokens=64 -> content='42', finish=stop
    reasoning_tokens: int | None = None

    #: For ``mode="error"``: an envelope that is NOT a context refusal, so the
    #: classifier has to fall through to ERRORED.
    error_status: int = 500
    error_type: str = "server_error"

    #: For ``mode="hang"``. ``wedge`` ignores this and hangs until cancelled.
    hang_s: float = 30.0
    #: Delay before the first token, for deadline arithmetic.
    first_token_delay_s: float = 0.0
    #: Paused before EVERY SSE content frame, so one response stretches over
    #: seconds — what lets the proxy's wall-clock reaper and the
    #: client-disconnect teardown be driven deterministically from a test.
    chunk_delay_s: float = 0.0

    #: Emit ``usage`` in a final ``choices: []`` frame. The recorded llama.cpp
    #: capture does this unconditionally; OpenAI gates it on stream_options.
    always_usage: bool = True
    #: Model the prefix cache. A repeated prompt prefix comes back with a high
    #: ``cached_tokens``, which is what makes the unique-preamble defeat
    #: checkable rather than a precaution taken on faith.
    prefix_cache: bool = True

    #: 200 for a live engine, 503 for a dead one. ``wedge`` deliberately leaves
    #: this at 200 while completions never return.
    health_status: int = 200

    #: An unknown model name returns 200 + empty content, as recorded. Turning
    #: this off is only ever for asserting what it costs.
    unknown_model_trap: bool = True

    #: Non-stream JSON responses carry a usage block. Some real engines answer
    #: 200 without one (older vLLM builds, some llama.cpp configurations) —
    #: the shape the proxy's tokenizer fallback has to survive.
    non_stream_usage: bool = True

    #: Non-stream JSON responses carry an EXPLICIT ``"usage": null``. Some
    #: OpenAI-compatible engines send the key present-but-null, which the
    #: proxy's usage read has to treat like an absent block (R36).
    non_stream_usage_null: bool = False

    #: Ollama-shaped /v1/models, as llama-server actually serves it.
    models_shape: str = "openai"

    #: Standalone only: the exit code passed to ``os._exit``.
    exit_code: int = 1

    #: Every request served, newest last. Lets a test assert what was SENT —
    #: the model name above all.
    seen: list[dict] = field(default_factory=list, repr=False)

    @classmethod
    def from_env(cls) -> Knobs:
        return cls(
            mode=os.environ.get("VW_FAKE_MODE", "ok"),
            refuse_above_tokens=_env_int("VW_FAKE_REFUSE_ABOVE_TOKENS", None),
            refuse_status=_env_int("VW_FAKE_REFUSE_STATUS", 400) or 400,
            refuse_error_type=os.environ.get(
                "VW_FAKE_REFUSE_ERROR_TYPE", "exceed_context_size_error"
            ),
            n_ctx=_env_int("VW_FAKE_N_CTX", None),
            crash_above_tokens=_env_int("VW_FAKE_CRASH_ABOVE_TOKENS", None),
            degrade_above_tokens=_env_int("VW_FAKE_DEGRADE_ABOVE_TOKENS", None),
            degrade_mode=os.environ.get("VW_FAKE_DEGRADE_MODE", "short"),
            error_status=_env_int("VW_FAKE_ERROR_STATUS", 500) or 500,
            error_type=os.environ.get("VW_FAKE_ERROR_TYPE", "server_error"),
            hang_s=_env_float("VW_FAKE_HANG_S", 30.0),
            first_token_delay_s=_env_float("VW_FAKE_FIRST_TOKEN_DELAY_S", 0.0),
            chunk_delay_s=_env_float("VW_FAKE_CHUNK_DELAY_S", 0.0),
            always_usage=_env_bool("VW_FAKE_ALWAYS_USAGE", True),
            prefix_cache=_env_bool("VW_FAKE_PREFIX_CACHE", True),
            health_status=_env_int("VW_FAKE_HEALTH_STATUS", 200) or 200,
            unknown_model_trap=_env_bool("VW_FAKE_UNKNOWN_MODEL_TRAP", True),
            non_stream_usage=_env_bool("VW_FAKE_NON_STREAM_USAGE", True),
            non_stream_usage_null=_env_bool("VW_FAKE_NON_STREAM_USAGE_NULL", False),
            models_shape=os.environ.get("VW_FAKE_MODELS_SHAPE", "openai"),
            exit_code=_env_int("VW_FAKE_EXIT_CODE", 1) or 1,
        )

    def apply(self, patch: dict) -> None:
        names = {f.name for f in fields(self)} - {"seen"}
        for k, v in patch.items():
            if k in names:
                setattr(self, k, v)


_SETTABLE = {f.name for f in fields(Knobs)} - {"seen"}
_INT_FIELDS = {
    "refuse_above_tokens",
    "refuse_status",
    "n_ctx",
    "error_status",
    "health_status",
    "exit_code",
    "crash_above_tokens",
    "degrade_above_tokens",
}
_FLOAT_FIELDS = {"hang_s", "first_token_delay_s", "chunk_delay_s"}
_BOOL_FIELDS = {
    "always_usage",
    "prefix_cache",
    "unknown_model_trap",
    "non_stream_usage",
    "non_stream_usage_null",
}


def _query_overrides(req) -> dict:
    """Per-request knobs from the query string.

    Coerced by field type rather than guessed, so ``?refuse_above_tokens=100``
    is an int and ``?prefix_cache=0`` is False rather than a truthy string.
    """
    out: dict = {}
    for k, v in req.query.items():
        if k not in _SETTABLE:
            continue
        if k in _INT_FIELDS:
            out[k] = None if v == "" else int(v)
        elif k in _FLOAT_FIELDS:
            out[k] = float(v)
        elif k in _BOOL_FIELDS:
            out[k] = v.strip().lower() in ("1", "true", "yes", "on")
        else:
            out[k] = v
    return out


def _effective(req) -> Knobs:
    base: Knobs = req.app["knobs"]
    over = _query_overrides(req)
    if not over:
        return base
    merged = Knobs(**{k: v for k, v in asdict(base).items() if k != "seen"})
    merged.apply(over)
    return merged


# ---------------------------------------------------------------------------
# answering the actual probe suite
# ---------------------------------------------------------------------------

_ARITH = re.compile(r"What is (\d+) plus (\d+)\?")
_ECHO = re.compile(r"Repeat exactly, with no other text:\s*(\S+)")
_VAULT = re.compile(r"the vault code is ([A-Z0-9]+)")

_SUMMARY = (
    "The survey team recorded ambient conditions at each station before dawn, "
    "logging sediment cores by depth and cross-checking every reading. "
    "Wind held steady while the northern transect was rerouted around flooding."
)


def _answer(prompt: str) -> str:
    """The correct answer to whichever probe this is.

    Ordered most-specific first: the summary probe embeds the whole needle
    prompt, so a needle match would fire on it and answer the wrong question.
    """
    if "summarise what the survey team recorded" in prompt:
        return _SUMMARY
    if "What is the vault code?" in prompt:
        m = _VAULT.search(prompt)
        if m:
            return m.group(1)
    if (m := _ARITH.search(prompt)) is not None:
        return str(int(m.group(1)) + int(m.group(2)))
    if (m := _ECHO.search(prompt)) is not None:
        return m.group(1)
    if "exactly one word: yes or no" in prompt:
        return "yes"
    if "Output exactly this JSON and nothing else" in prompt:
        return '{"a": 1}'
    # Unrecognised: the pre-stress behaviour, which the proxy integration
    # tests assert on.
    return f"echo: {prompt}"


def _degrade(mode: str, answer: str) -> tuple[str, str, str]:
    """(content, reasoning_content, finish_reason) for a degradation mode."""
    if mode == "empty":
        return "", "", "stop"
    if mode == "reasoning_only":
        # Verbatim from the recorded unknown-model response: all output went to
        # reasoning, none to content, and it ran to the length cap.
        return "", "////", "length"
    if mode == "short":
        return answer[:2] or ".", "", "stop"
    if mode == "wrong":
        return "definitely not the answer", "", "stop"
    if mode == "abrupt":
        return answer, "", "length"
    if mode == "repeat":
        # >48 chars per cycle and many cycles, so a 48-char shingle recurs well
        # past repeat_max inside the detector's 4096-char window.
        return answer + " " + ("the loop continues and the loop continues. " * 24), "", "stop"
    return answer, "", "stop"


def _prompt_text(body: dict) -> str:
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        parts = [str(m.get("content") or "") for m in msgs if isinstance(m, dict)]
        return "\n".join(parts)
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        return str(prompt[0]) if prompt else ""
    return str(prompt)


def _count(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _cached_for(app, knobs: Knobs, prompt: str, prompt_tokens: int) -> int:
    """Simulated prefix-cache reuse, keyed on the head of the prompt.

    Block-aligned from position 0, like the real thing — which is exactly why
    the harness has to put fresh entropy at position 0 rather than merely
    somewhere in the prompt.
    """
    if not knobs.prefix_cache:
        return 0
    key = prompt[:128]
    seen = app["_prefixes"]
    hit = key in seen
    seen.add(key)
    return max(0, prompt_tokens - 4) if hit else 0


# ---------------------------------------------------------------------------
# error envelopes
# ---------------------------------------------------------------------------


def _refusal(knobs: Knobs, prompt_tokens: int) -> web.Response:
    """The recorded llama.cpp over-context refusal, field for field."""
    n_ctx = knobs.n_ctx or knobs.refuse_above_tokens or 0
    return web.json_response(
        {
            "error": {
                "code": knobs.refuse_status,
                "message": (
                    f"request ({prompt_tokens} tokens) exceeds the available "
                    f"context size ({n_ctx} tokens), try increasing it"
                ),
                "type": knobs.refuse_error_type,
                "n_prompt_tokens": prompt_tokens,
                "n_ctx": n_ctx,
            }
        },
        status=knobs.refuse_status,
    )


def _envelope(status: int, err_type: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"code": status, "message": message, "type": err_type}},
        status=status,
    )


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------


async def health(req):
    knobs = _effective(req)
    if knobs.health_status == 200:
        return web.Response(status=200, text="ok")
    return web.Response(status=knobs.health_status, text="unhealthy")


async def models(req):
    name = req.app["served_name"]
    knobs = _effective(req)
    if knobs.models_shape == "llamacpp":
        # Ollama-shaped, as recorded. A uniform /v1/models parser breaks here,
        # which is the point of being able to serve it.
        return web.json_response(
            {
                "models": [
                    {
                        "name": name,
                        "model": name,
                        "type": "model",
                        "capabilities": ["completion"],
                        "details": {"format": "gguf"},
                    }
                ]
            }
        )
    return web.json_response({"data": [{"id": name, "object": "model"}]})


async def set_knobs(req):
    """Retune a running engine without restarting it.

    A stress run changes what the engine does *mid-run* — it degrades, then
    refuses, then dies — so a test needs to move the knobs between probes
    against the same port.
    """
    patch = await req.json()
    req.app["knobs"].apply(patch)
    return web.json_response({"ok": True})


async def _die(req, knobs: Knobs, resp=None):
    """Stop serving, the way a crashed engine stops serving.

    Standalone, that is a real process exit — the subprocess integration path
    needs the port to actually go away. In-process it cannot be, because that
    would take pytest with it, so the connection is aborted and ``/health``
    flipped to 503: from the harness's side those are the same two observables
    (transport error, engine unhealthy) that classify CRASHED.
    """
    if req.app.get("standalone"):
        os._exit(knobs.exit_code)
    req.app["knobs"].health_status = 503
    transport = req.transport
    if transport is not None:
        transport.abort()
    raise ConnectionResetError("fake engine exited mid-request")


async def _stream(
    req, knobs: Knobs, *, content, reasoning, finish_reason, prompt_tokens, cached, chat: bool
):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(req)

    async def frame(obj):
        await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

    name = req.app["served_name"]
    if knobs.first_token_delay_s:
        await asyncio.sleep(knobs.first_token_delay_s)

    if chat:
        # Recorded first frame: role only, content explicitly null.
        await frame(
            {
                "choices": [
                    {
                        "finish_reason": None,
                        "index": 0,
                        "delta": {"role": "assistant", "content": None},
                    }
                ],
                "object": "chat.completion.chunk",
                "model": name,
            }
        )

    pieces = [("reasoning_content", reasoning)] if reasoning else []
    pieces += [("content", c) for c in _chunks(content)] if content else []

    for i, (key, text) in enumerate(pieces):
        if knobs.mode == "exit" and i == 1:
            await _die(req, knobs, resp)
        if knobs.chunk_delay_s:
            await asyncio.sleep(knobs.chunk_delay_s)
        if chat:
            await frame(
                {
                    "choices": [{"finish_reason": None, "index": 0, "delta": {key: text}}],
                    "object": "chat.completion.chunk",
                    "model": name,
                }
            )
        else:
            await frame({"choices": [{"text": text, "index": 0}], "model": name})

    if knobs.mode == "truncated":
        # No terminal frame, no finish_reason, no [DONE] — what the wall-clock
        # reaper and the disconnect path actually leave behind.
        return resp

    if chat:
        await frame(
            {
                "choices": [{"finish_reason": finish_reason, "index": 0, "delta": {}}],
                "object": "chat.completion.chunk",
                "model": name,
            }
        )
    else:
        await frame(
            {"choices": [{"text": "", "index": 0, "finish_reason": finish_reason}], "model": name}
        )

    if knobs.always_usage:
        completion_tokens = _count(content or reasoning or "")
        await frame(
            {
                "choices": [],
                "object": "chat.completion.chunk" if chat else "text_completion",
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }
        )
    await resp.write(b"data: [DONE]\n\n")
    return resp


def _chunks(text: str, size: int = 8) -> list[str]:
    """One SSE frame per few characters, so delta counts mean something."""
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


async def _serve(req, *, chat: bool):
    knobs = _effective(req)
    body = await req.json()
    prompt = _prompt_text(body)
    prompt_tokens = _count(prompt)
    requested_model = body.get("model")
    req.app["knobs"].seen.append(
        {
            "model": requested_model,
            "prompt_tokens": prompt_tokens,
            "stream": bool(body.get("stream")),
            "max_tokens": body.get("max_tokens"),
        }
    )

    if knobs.mode == "wedge":
        # /health stays 200 — that is the whole pathology. Sleep forever and
        # let the harness's own deadline be the thing that ends it.
        await asyncio.sleep(3600)

    if knobs.mode == "hang":
        await asyncio.sleep(knobs.hang_s)

    if knobs.mode == "rate_limit":
        return _envelope(429, "rate_limit_error", "server is busy, retry later")

    if knobs.mode == "error":
        return _envelope(knobs.error_status, knobs.error_type, "internal failure")

    if knobs.refuse_above_tokens is not None and prompt_tokens > knobs.refuse_above_tokens:
        return _refusal(knobs, prompt_tokens)

    if knobs.crash_above_tokens is not None and prompt_tokens > knobs.crash_above_tokens:
        await _die(req, knobs)

    if knobs.mode == "exit" and not body.get("stream"):
        await _die(req, knobs)

    # THE TRAP, recorded verbatim: an unknown model name is not an error. The
    # engine serves whatever is loaded and returns 200 with empty content and
    # finish_reason=length, which reads as REASONING_ONLY + ABRUPT.
    if (
        knobs.unknown_model_trap
        and requested_model is not None
        and requested_model != req.app["served_name"]
    ):
        content, reasoning, finish = "", "////", "length"
    else:
        mode = knobs.mode
        if knobs.degrade_above_tokens is not None and prompt_tokens > knobs.degrade_above_tokens:
            mode = knobs.degrade_mode
        content, reasoning, finish = _degrade(mode, _answer(prompt))

    # A reasoning model spends its budget thinking first. If the caller did not
    # allow room for the chain AND the answer, the answer never starts -- the
    # engine is behaving correctly and the harness is the one at fault.
    if knobs.reasoning_tokens is not None and content:
        budget = body.get("max_tokens")
        chain = " ".join(["think"] * knobs.reasoning_tokens)
        if budget is not None and budget <= knobs.reasoning_tokens:
            content, reasoning, finish = "", " ".join(["think"] * budget), "length"
        else:
            reasoning = chain

    cached = _cached_for(req.app, knobs, prompt, prompt_tokens)

    if body.get("stream"):
        return await _stream(
            req,
            knobs,
            content=content,
            reasoning=reasoning,
            finish_reason=finish,
            prompt_tokens=prompt_tokens,
            cached=cached,
            chat=chat,
        )

    completion_tokens = _count(content or reasoning or "")
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    if chat:
        message = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        payload = {
            "id": "fake-1",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.app["served_name"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        }
    else:
        payload = {
            "id": "fake-1",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.app["served_name"],
            "choices": [{"index": 0, "text": content, "finish_reason": finish}],
        }
    if knobs.non_stream_usage_null:
        payload["usage"] = None
    elif knobs.non_stream_usage:
        payload["usage"] = usage
    return web.json_response(payload)


async def chat_completions(req):
    return await _serve(req, chat=True)


async def completions(req):
    return await _serve(req, chat=False)


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def make_app(
    *, served_model_name: str, knobs: Knobs | None = None, standalone: bool = False
) -> web.Application:
    app = web.Application()
    app["served_name"] = served_model_name
    app["knobs"] = knobs or Knobs.from_env()
    app["standalone"] = standalone
    app["_prefixes"] = set()
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/_fake/knobs", set_knobs)
    return app


class FakeEngine:
    """The fake, in-process, on an ephemeral port.

    In-process because the stress runner's hardest paths are crash → recover →
    **re-resolve the port**, and reproducing that with subprocesses would make a
    unit test slow, flaky and platform-dependent. :meth:`crash` closes the
    listener the way a dead engine closes it, and :meth:`restart` deliberately
    comes back on a *different* port — a runner that cached the old one fails
    here rather than in production.
    """

    def __init__(
        self, *, served_model_name: str = "fake-model", knobs: Knobs | None = None
    ) -> None:
        self.served_model_name = served_model_name
        self.knobs = knobs or Knobs()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self.port: int | None = None

    async def start(self) -> int:
        app = make_app(served_model_name=self.served_model_name, knobs=self.knobs)
        # A wedged engine's handler sleeps for an hour on purpose; the default
        # 60s graceful-shutdown wait would make every wedge test take a minute.
        self._runner = web.AppRunner(app, shutdown_timeout=0.1)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        sock = next(iter(self._site._server.sockets))  # noqa: SLF001
        self.port = sock.getsockname()[1]
        return self.port

    async def crash(self) -> None:
        """Stop serving without any tidy shutdown handshake."""
        await self.stop()
        self.knobs.health_status = 503

    async def restart(self) -> int:
        """Come back on a NEW port, as ``watchdog._restart`` does."""
        await self.stop()
        self.knobs.health_status = 200
        self.knobs.mode = "ok"
        return await self.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        self._runner = None
        self._site = None
        self.port = None

    def set(self, **patch) -> None:
        self.knobs.apply(patch)

    async def __aenter__(self) -> FakeEngine:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18001)
    ap.add_argument("--served-model-name", default="fake-model")
    args = ap.parse_args()

    app = make_app(
        served_model_name=args.served_model_name,
        knobs=Knobs.from_env(),
        standalone=True,
    )
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
