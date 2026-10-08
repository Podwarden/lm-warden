import asyncio
import json
import logging
import secrets
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.db.database import open_db
from app.db.repos.models import ModelRow
from app.db.repos.tokens import TokenRow
from app.models.context_window import effective_context_window
from app.models.parallelism import spill_threshold_for
from app.proxy import content_log, forest_fields
from app.proxy.auth import require_bearer, token_allows
from app.proxy.dp_affinity import KV_MAX_AGE_S, RANK_HEADER, client_rank
from app.proxy.dp_affinity import affinity_key as _dp_affinity_key
from app.proxy.envelope_hint import enrich_5xx_from_db
from app.proxy.model_cache import all_models, model_by_served_name
from app.proxy.reaggregate import StreamAggregator, parse_sse_event
from app.proxy.request_registry import LiveRequest, finished_record
from app.proxy.runaway import RunawayDetector
from app.proxy.scheduler import model_admission_cap
from app.proxy.session_id import SessionInfo, extract_session

# Aliased: the forward handler already binds a LOCAL `registry` (the Plane-B
# live-request registry off app.state), which would shadow an unaliased import
# for the whole function -- a real F823 that ruff caught rather than a style nit.
from app.runtime.backends import registry as backend_registry
from app.runtime.variants import (
    Variant,
    running_variant,
)
from app.utils.client_ip import client_ip as _client_ip
from app.utils.sse import delta_reasoning

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["proxy"])

# God mode taps log-and-swallow: a broadcast failure must NEVER break, delay,
# or backpressure a proxied response.
_gm_log = logging.getLogger("vllm_warden.godmode")

# The only paths the runaway detector guards — the two streamable completion
# endpoints. Every other proxied path is forwarded untouched even when armed.
_RUNAWAY_PATHS = ("/v1/chat/completions", "/v1/completions")

# Live-stats registry throttle: refresh a streaming request's coarse
# completion-token estimate at most this often, to keep the hot path cheap
# (no per-delta tokenize — see docs/live-stats-spec.md § "Plane B").
_LIVE_UPDATE_INTERVAL_S = 0.5

# The single clock `_forward` reads for queue-wait / duration / TTFT timing
# (request_start, the admission-gate span, started_monotonic, ttft, and the
# `now` passed into `finished_record`). It is `time.monotonic` by another
# name in production. Tests patch THIS name rather than `time.monotonic`
# itself: monkeypatching the real `time.monotonic` also freezes asyncio's own
# event-loop clock (loop.time() reads it too), which can wedge any coroutine
# with a real scheduled callback (background writers, retries, ...) — see
# tests/unit/proxy/test_queue_wait_recorded.py.
_monotonic = time.monotonic

# #296: the upstream send is retried once only if it failed within this many
# seconds of starting. A stale pooled keep-alive (the engine closed it while
# idle) fails at once: the write or the first read hits a closed socket in
# milliseconds. A send that fails later most likely reached the engine, which
# then spent the time on it (a long prefill, then a reset); retrying that
# would run the same work twice and double the latency. 1 s sits far above
# the stale-socket case and far below any real prefill worth protecting.
STALE_KEEPALIVE_FAIL_S = 1.0


async def _resolve_target(request: Request, served_name: str):
    # From the cached models table (#293): every ModelRepo write drops it, so a
    # status change decides the very next request's 404.
    model = await model_by_served_name(
        getattr(request.app.state, "model_cache", None),
        request.app.state.settings.db_path,
        served_name,
    )
    return _resolve_loaded(request, model, served_name)


def _resolve_loaded(
    request: Request, model: ModelRow | None, served_name: str
) -> tuple[ModelRow, str, int, Variant]:
    """The engine behind an already-fetched model row (#287 reuses this for
    rule targets fetched by id); 404 unless it is loaded and has a port."""
    sup = request.app.state.supervisor
    if not model or model.status != "loaded":
        raise HTTPException(404, f"model '{served_name}' is not loaded")
    port = sup.get_port(model.id)
    if port is None:
        raise HTTPException(404, f"model '{served_name}' not running")
    # The driver owns where the engine listens: loopback for the
    # in-container subprocess, the engine container's DNS name for the
    # docker driver. Loopback fallback covers a driver without get_host.
    host = sup.get_host(model.id) or "127.0.0.1"
    # The variant the running engine was launched as (0036), read where the
    # engine itself is resolved, so the usage rollup and request_history name
    # the same one.
    variant = running_variant(request.app.state, model)
    return model, host, port, variant


def _extract_prompt(body_json) -> str:
    def _text(c):
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text", ""))
            return "\n".join(parts)
        return ""

    if "messages" in body_json:
        return "\n".join(
            _text(m.get("content", "")) for m in body_json["messages"] if isinstance(m, dict)
        )
    return body_json.get("prompt", "") or ""


def _godmode_prompt_window(text: str, max_chars: int, tail_chars: int) -> tuple[str, bool]:
    """Display-only capture window for the god-mode tap.

    NEVER touches the forwarded request — only the tapped copy. If the prompt
    fits within ``max_chars`` it is returned verbatim. Otherwise keep the head
    ``(max_chars - tail_chars)`` chars PLUS the last ``tail_chars`` chars,
    joined by the marker ``\\n\\n…[{n} chars elided]…\\n\\n`` where ``n`` is the
    elided count. This guarantees the newest turn (which lives at the tail)
    survives even under a giant repeated system prompt, instead of the old
    head-slice spending the whole budget on the system prompt.

    Returns ``(windowed_text, elided)`` where ``elided`` is True iff the middle
    was dropped. O(len) — pure slices, no extra copies.
    """
    n = len(text)
    if n <= max_chars:
        return text, False
    # Clamp the tail so a degenerate config (tail >= max) can't make the head
    # length negative; the head is whatever budget remains after the tail.
    tail = max(0, min(tail_chars, max_chars))
    head_len = max_chars - tail
    head = text[:head_len]
    tail_str = text[n - tail :] if tail > 0 else ""
    elided = n - head_len - tail
    return f"{head}\n\n…[{elided} chars elided]…\n\n{tail_str}", True


def _extract_godmode_media(body_json, store, settings) -> list[dict]:
    """Display-only capture of ``image_url`` content parts for god mode
    (spec 2026-08-03). NEVER touches the forwarded request. Never raises —
    a malformed body yields a short (possibly empty) list.

    data URIs -> payload sliced into ``store`` (base64 string, no decode on
    the hot path), event carries {media_id, mime, chars}. Remote http(s)
    URLs -> {url} (the admin browser hotlinks them; the warden never
    fetches). Oversized -> {dropped: "too_large"}. More than
    ``godmode_max_images_per_req`` -> one trailing {dropped: "count"}.
    """
    media: list[dict] = []
    try:
        if not isinstance(body_json, dict):
            return media
        messages = body_json.get("messages")
        if not isinstance(messages, list):
            return media
        max_items = settings.godmode_max_images_per_req
        max_chars = settings.godmode_max_image_chars
        overflow = 0
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for blk in content:
                if not (isinstance(blk, dict) and blk.get("type") == "image_url"):
                    continue
                url = blk.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                if not isinstance(url, str):
                    continue
                # Cap checked BEFORE the entry is extracted (which is where
                # store.put() happens for a data URI): otherwise every image
                # past max_items still gets sliced into the store as an
                # orphan blob nothing ever references, evicting legitimate
                # images via the store's oldest-first budget. A malformed
                # part past the cap now also counts toward overflow — an
                # acceptable trade since it's never checked otherwise.
                if len(media) >= max_items:
                    overflow += 1
                    continue
                entry = _godmode_media_entry(url, store, max_chars)
                if entry is None:
                    continue
                media.append(entry)
        if overflow:
            media.append({"kind": "image", "dropped": "count", "count": overflow})
    except Exception:
        logger.warning("godmode media extraction failed", exc_info=True)
    return media


def _godmode_media_entry(url: str, store, max_chars: int) -> dict | None:
    """One image_url string -> media entry dict, or None to skip. Split from
    _extract_godmode_media so the per-item paths stay readable."""
    if url.startswith("data:"):
        sep = url.find(";base64,")
        if sep < 0:
            return None
        # Strip any params (e.g. ";charset=utf-8") that may sit between the
        # mime and ";base64,", and normalize case, so a legitimate param'd
        # data URI is captured instead of failing the allowlist and showing
        # up mislabeled as "too_large" with the raw param string as its mime.
        mime = url[5:sep].split(";", 1)[0].strip().lower()
        payload = url[sep + len(";base64,") :]
        if not payload or not mime.startswith("image/"):
            return None
        if len(payload) > max_chars:
            return {
                "kind": "image",
                "mime": mime,
                "chars": len(payload),
                "dropped": "too_large",
            }
        media_id = secrets.token_hex(8)
        if not store.put(media_id, mime, payload):
            # Store-side rejection (e.g. mime fails the strict allowlist).
            return {
                "kind": "image",
                "mime": mime,
                "chars": len(payload),
                "dropped": "too_large",
            }
        return {"kind": "image", "media_id": media_id, "mime": mime, "chars": len(payload)}
    if url.startswith("http://") or url.startswith("https://"):
        return {"kind": "image", "url": url}
    return None


def _parse_sse_delta(line: bytes) -> str | None:
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return None
    try:
        ev = json.loads(payload)
        return ev["choices"][0].get("delta", {}).get("content") or None
    except Exception:
        return None


def _parse_sse_text(line: bytes) -> str | None:
    """Legacy ``/v1/completions`` variant of ``_parse_sse_delta`` (R13).

    That endpoint streams ``choices[0].text`` chunks instead of
    ``choices[0].delta.content``; without this parse a streamed
    completions request is metered as 0 completion tokens. The text
    joins ``accumulated`` (content log) and the end-of-stream completion
    count exactly as chat content does. Chat frames carry no ``text``
    field, so this returns ``None`` for them.
    """
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return None
    try:
        ev = json.loads(payload)
        return ev["choices"][0].get("text") or None
    except Exception:
        return None


def _parse_sse_godmode(line: bytes) -> tuple[str | None, str | None, str | None]:
    """God-mode-local SSE frame parse → ``(content, reasoning, finish_reason)``.

    Kept SEPARATE from ``_parse_sse_delta`` (whose single-string return the
    counter path depends on) so god mode can also surface the reasoning
    channel and ``finish_reason`` without touching that contract. With
    vLLM's ``--reasoning-parser qwen3`` the thinking stream arrives in
    ``delta.reasoning_content`` — or ``delta.reasoning`` on newer builds
    (R22) — NOT inline ``<think>`` tags, so ``delta_reasoning`` reads
    both names. Any part may be ``None``.
    """
    if not line.startswith(b"data:"):
        return None, None, None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return None, None, None
    try:
        choice = json.loads(payload)["choices"][0]
        delta = choice.get("delta", {}) or {}
        return (
            delta.get("content"),
            delta_reasoning(delta),
            choice.get("finish_reason"),
        )
    except Exception:
        return None, None, None


def _parse_sse_frame(
    line: bytes,
) -> tuple[int | None, str, int | None, int | None, str | None]:
    """Completion-accounting parse of one SSE frame →
    ``(usage, extra, cached, prompt, kind)``. ``kind`` is the live phase the
    frame's delta implies (``tool_call`` > ``answering`` > ``thinking``), or None.

    Kept SEPARATE from ``_parse_sse_delta``, whose content-only return the
    content log (via ``accumulated``) and the live ``delta_count`` estimate
    depend on.

    ``usage`` is the frame's ``usage.completion_tokens`` when present: a
    client that asked for ``stream_options.include_usage`` receives the
    engine's own count on the final frame (vLLM sends ``"usage": null`` on
    the token frames before it), and it wins over the end-of-stream
    tokenizer estimate. ``extra`` is the choice-0 delta text
    ``_parse_sse_delta`` does NOT return: the reasoning channel (the
    ``--reasoning-parser`` channel — ``reasoning_content``, or ``reasoning``
    on newer builds, via ``delta_reasoning``) plus each tool call's ``function``
    ``name`` and ``arguments`` deltas — the channels agents emit almost
    exclusively, which is why the old content-only count metered them as
    ~1 completion token per request. Either may be empty/None.
    """
    if not line.startswith(b"data:"):
        return None, "", None, None, None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return None, "", None, None, None
    try:
        ev = json.loads(payload)
    except Exception:
        return None, "", None, None, None
    usage = None
    u = ev.get("usage")
    if isinstance(u, dict):
        ct = u.get("completion_tokens")
        if isinstance(ct, int) and not isinstance(ct, bool) and ct >= 0:
            usage = ct
    extra = ""
    kind: str | None = None
    choices = ev.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            if isinstance(first.get("text"), str) and first["text"]:
                kind = "answering"  # legacy /v1/completions
            delta = first.get("delta")
            if isinstance(delta, dict):
                reasoning = delta_reasoning(delta)
                if isinstance(reasoning, str):
                    extra += reasoning
                    if reasoning:
                        kind = "thinking"
                content = delta.get("content")
                if isinstance(content, str) and content:
                    kind = "answering"
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    kind = "tool_call"
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function")
                        if not isinstance(fn, dict):
                            continue
                        name = fn.get("name")
                        if isinstance(name, str):
                            extra += name
                        args = fn.get("arguments")
                        if isinstance(args, str):
                            extra += args
    return usage, extra, _cached_tokens_of(ev), _prompt_tokens_of(ev), kind


def _gm_publish(hub, event: dict) -> None:
    """Publish an event, swallowing+logging any failure. God mode is strictly
    best-effort observation; it must never surface an error into ``_forward``."""
    try:
        hub.publish(event)
    except Exception:
        _gm_log.warning("godmode publish failed for %s", event.get("type"), exc_info=True)


def _feed_detector(detector: RunawayDetector, synth: dict, ev: dict, is_chat: bool) -> None:
    """Feed the primary choice's decoded text to the runaway detector.

    Reasoning-parser models (``--reasoning-parser qwen3``) strip the
    ``<think>``/``</think>`` tags and stream the reasoning chain as
    ``delta.reasoning_content`` — or ``delta.reasoning`` on newer builds
    (R22), read via ``delta_reasoning`` — rather than ``delta.content``. The detector's
    unclosed-think signal is purely textual, so we synthesize a single
    ``<think>`` on the first reasoning delta and a ``</think>`` when content
    finally begins — otherwise an over-reasoning generation (the exact
    pathology we hunt) would never open a think block and the budget signal
    would be dead. Only the choice at index 0 is fed so delta-count ≈
    token-count stays true for the budget metric.
    """
    choices = ev.get("choices")
    if not isinstance(choices, list):
        return
    prim = None
    for c in choices:
        if isinstance(c, dict) and c.get("index", 0) == 0:
            prim = c
            break
    if prim is None:
        return
    if is_chat:
        delta = prim.get("delta")
        if not isinstance(delta, dict):
            return
        reasoning = delta_reasoning(delta)
        content = delta.get("content")
        if reasoning:
            if not synth["think_open"]:
                detector.feed("<think>")
                synth["think_open"] = True
            detector.feed(reasoning)
        if content:
            if synth["think_open"]:
                detector.feed("</think>")
                synth["think_open"] = False
            detector.feed(content)
    else:
        text = prim.get("text")
        if text:
            detector.feed(text)


def _runaway_terminal_chunk(is_chat: bool, meta: dict, finish_reason: str) -> bytes:
    """Build the terminal SSE frame injected on an enforce trip: one final
    chunk carrying the runaway ``finish_reason`` plus the ``[DONE]`` sentinel,
    so a streaming client sees a clean, schema-valid end to the generation."""
    if is_chat:
        ev = {
            "id": meta.get("id") or "chatcmpl-runaway",
            "object": "chat.completion.chunk",
            "created": meta.get("created") or int(time.time()),
            "model": meta.get("model") or "",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
    else:
        ev = {
            "id": meta.get("id") or "cmpl-runaway",
            "object": "text_completion",
            "created": meta.get("created") or int(time.time()),
            "model": meta.get("model") or "",
            "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}],
        }
    return b"data: " + json.dumps(ev).encode() + b"\n\ndata: [DONE]\n\n"


def _is_sse_stream(resp) -> bool:
    """True only when the upstream response is an actual SSE token stream.

    We force ``stream=true`` upstream when the detector is armed, but an
    upstream ERROR (4xx/5xx) comes back as a plain JSON envelope, not SSE.
    Re-aggregating that non-stream body finds zero ``data:`` frames and
    yields a degenerate empty ``chat.completion`` skeleton — discarding the
    real error body and bypassing the 5xx DB-hint enrichment. Gate the
    re-aggregation branch on this so an error response falls through to the
    normal non-stream path (which buffers, enriches, and returns it intact).
    """
    return (
        resp.status_code == 200
        and "text/event-stream" in resp.headers.get("content-type", "").lower()
    )


def _completion_text(out: dict, is_chat: bool) -> str:
    """Concatenate the reconstructed completion text across all choices — for
    token accounting and the incident record."""
    parts = []
    for c in out.get("choices", []):
        if is_chat:
            parts.append((c.get("message") or {}).get("content") or "")
        else:
            parts.append(c.get("text") or "")
    return "".join(parts)


_record_warned_at = -60.0  # monotonic time of the last record-failure warning


async def _record_counters(
    request: Request,
    model,
    token_id: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    variant: Variant | None = None,
    cached_tokens: int | None = None,
) -> None:
    # #279 stage 3: accumulate only; app/proxy/ledger.py writes once a second.
    if variant is None and token_id is not None:
        # Best effort, in its own try: a failed lookup only loses the
        # per-variant rollup, never the counters, sample and usage.
        try:
            variant = running_variant(request.app.state, model)
        except Exception:  # noqa: BLE001
            logger.debug("ledger: could not resolve the running variant", exc_info=True)
            variant = None
    try:
        request.app.state.ledger.record_request(
            model_id=model.id,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            variant_id=variant.id if variant is not None else None,
            variant=variant,
        )
    except Exception:  # noqa: BLE001 -- accounting must never fail a request
        global _record_warned_at
        now = time.monotonic()
        if now - _record_warned_at >= 60.0:
            _record_warned_at = now
            logger.warning("ledger: could not record a request (rate-limited)", exc_info=True)


class _UnstartedStreamGuard:
    """Body iterator that releases a stream's resources if it is never started.

    ``gen`` is an async generator; closing or discarding one that never ran
    skips its ``finally``. If the client disconnects before the first read
    (or the response start fails to send) that leaked the admission slot and
    the replica in-flight count (#286 review). ``abandon`` runs ``on_abandon``
    at most once, and only while the generator has not started; after the
    first ``__anext__`` the generator's own ``finally`` owns cleanup.
    """

    def __init__(
        self, gen: AsyncGenerator[bytes, None], on_abandon: Callable[[], Awaitable[None]]
    ) -> None:
        self._gen = gen
        self._on_abandon = on_abandon
        self._started = False
        self._abandoned = False

    def __aiter__(self) -> "_UnstartedStreamGuard":
        return self

    async def __anext__(self) -> bytes:
        if self._abandoned:
            raise StopAsyncIteration
        self._started = True
        return await self._gen.__anext__()

    async def abandon(self) -> None:
        if self._started or self._abandoned:
            return
        self._abandoned = True
        await asyncio.shield(self._on_abandon())
        await self._gen.aclose()

    async def aclose(self) -> None:
        if self._started:
            await self._gen.aclose()
        else:
            await self.abandon()


def _abandon_unstarted_stream(
    deregister: Callable[[], Awaitable[None]],
    resp: httpx.Response,
    release_slot: Callable[[], Awaitable[None]],
) -> Callable[[], Awaitable[None]]:
    async def _run() -> None:
        try:
            await deregister()
        finally:
            try:
                await resp.aclose()
            finally:
                await release_slot()

    return _run


def _parse_sse_accounting(line: bytes) -> tuple[int | None, str]:
    """``(usage, extra)`` of ``_parse_sse_frame``, for callers that only account."""
    usage, extra, _cached, _prompt, _kind = _parse_sse_frame(line)
    return usage, extra


def _nonneg_int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def _is_sse_error_frame(line: bytes) -> bool:
    """True for a ``data:`` frame whose top-level JSON has an ``error`` object --
    how llama-server reports a failure (e.g. "Context size has been exceeded")
    inside an HTTP 200 stream. Callers gate on the cheap substring first."""
    if not line.lstrip().startswith(b"data:"):
        return False
    try:
        ev = json.loads(line.split(b"data:", 1)[1].strip())
    except Exception:
        return False
    return isinstance(ev, dict) and bool(ev.get("error"))


def _prompt_tokens_of(obj: Any) -> int | None:
    """The ENGINE's ``usage.prompt_tokens`` in a frame / body, or None. It wins
    over the warden's own count for accounting and display (the warden's is a
    character estimate for a GGUF without a tokenizer)."""
    usage = obj.get("usage") if isinstance(obj, dict) else None
    return _nonneg_int(usage.get("prompt_tokens")) if isinstance(usage, dict) else None


def _cached_tokens_of(obj: Any) -> int | None:
    """The engine's MEASURED prefix-cache hit from a final frame / response body.

    First present of: ``usage.prompt_tokens_details.cached_tokens`` (vLLM with
    ``--enable-prompt-tokens-details``, recent llama-server), llama.cpp's
    ``timings.cache_n``, llama.cpp's legacy top-level ``tokens_cached``.
    None = not reported ("not measured", never 0).
    """
    if not isinstance(obj, dict):
        return None
    usage = obj.get("usage")
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    if isinstance(details, dict):
        ct = _nonneg_int(details.get("cached_tokens"))
        if ct is not None:
            return ct
    timings = obj.get("timings")
    if isinstance(timings, dict):
        ct = _nonneg_int(timings.get("cache_n"))
        if ct is not None:
            return ct
    return _nonneg_int(obj.get("tokens_cached"))


#: model id -> monotonic time of the last "usage backend sent no cached_tokens"
#: debug line; one line per model per minute is enough to spot a coverage gap.
_NO_CACHED_LOGGED: dict[str, float] = {}
_NO_CACHED_LOG_EVERY_S = 60.0


def _note_no_cached(model_id: str) -> None:
    now = _monotonic()
    last = _NO_CACHED_LOGGED.get(model_id)
    if last is not None and now - last < _NO_CACHED_LOG_EVERY_S:
        return
    _NO_CACHED_LOGGED[model_id] = now
    logger.debug(
        "cache-obs: model=%s backend reports usage but sent no cached_tokens; "
        "its requests stay out of the cache figures",
        model_id,
    )


def _cached_reporting(model: Any) -> str:
    """The backend's cached_tokens_reporting; "usage" when unknown (fail-open)."""
    try:
        return backend_registry.get(
            getattr(model, "backend", None)
        ).capabilities.cached_tokens_reporting
    except Exception:  # noqa: BLE001
        return "usage"


async def _forward(
    request: Request,
    model: ModelRow,
    host: str,
    port: int,
    path: str,
    token: TokenRow,
    *,
    variant: Variant | None = None,
    affinity_key: str | None = None,
    affinity_source: str | None = None,
    session_id: str | None = None,
) -> Response:
    """Run one request through admission, the engine and the bookkeeping.

    ``affinity_source`` and ``session_id`` are for routes that swap the request
    body for a translated one (``/v1/responses``): the translated chat body no
    longer carries the client's ``prompt_cache_key``, so the route derives
    (key, source, session id) from the ORIGINAL body and hands them in. With
    ``affinity_source`` set, ``affinity_key`` is used verbatim as the replica
    key and the extractor is not run on the (translated) body; ``session_id``
    is what the live row shows. Both default to None: no behaviour change.
    """
    body = await request.body()
    body_json = json.loads(body) if body else {}
    # #286: the replica-affinity key comes from the body as the client sent it,
    # before anything below may re-serialise it. Pure and read-only, but it is
    # still on the hot path: a failure means "no key", never a failed request.
    dp_key: str | None = None
    dp_key_source = "none"
    try:
        if affinity_source is not None:
            dp_key, dp_key_source = affinity_key, affinity_source
        else:
            dp_key, dp_key_source = _dp_affinity_key(
                body_json, request.headers, token.id, explicit=affinity_key
            )
    except Exception:  # noqa: BLE001 — routing must never fail a request
        logger.debug("dp-routing: could not derive an affinity key", exc_info=True)
    is_stream = bool(body_json.get("stream"))
    # Runaway detector arming. When VW_RUNAWAY_MODE != "off" (and this is one of
    # the two guarded completion paths) we must watch every decoded token, so we
    # force the upstream call to stream even when the client asked for a
    # non-streaming response — then re-aggregate the SSE back into the non-stream
    # JSON vLLM would have produced (see reaggregate.py). When the mode is "off"
    # none of this runs and the body + forward stay byte-identical to a build
    # without the detector (hard regression guard in test_runaway_forward.py).
    settings = request.app.state.settings
    runaway_mode = getattr(settings, "runaway_mode", "off")
    runaway_active = runaway_mode != "off" and path in _RUNAWAY_PATHS
    client_wants_stream = is_stream
    is_chat_path = path.endswith("/chat/completions")
    if runaway_active and not client_wants_stream:
        # Force upstream streaming + request the usage tail so the re-aggregated
        # non-stream response still carries an accurate usage block. A streaming
        # client needs neither mutation (its stream is already true and it did
        # not ask for include_usage), so we leave its body untouched.
        body_json["stream"] = True
        body_json.setdefault("stream_options", {})["include_usage"] = True
        body = json.dumps(body_json).encode()
        is_stream = True
    tok_cache = request.app.state.tokenizers
    token_id = token.id
    if variant is None:
        variant = running_variant(request.app.state, model)
    # Wall-clock origin for the server-side reaper (see settings.request_max_wall_s).
    request_start = _monotonic()

    prompt_text = _extract_prompt(body_json)
    # fallback_repo: a GGUF-only repo ships no tokenizer.json, so counting
    # against it raises -- on the hot path, with no try/except above -- and
    # 500s a request the engine could have served. tokenizer_repo (migration
    # 0015) points at the sibling that does have one; count() also fails open
    # to a character estimate when even that is unavailable.
    #
    # #264: the row's `trust_remote_code` is deliberately NOT passed. This call
    # runs in the warden's own process, on data-plane traffic, for accounting
    # only -- it was the one place in the product that executed a model repo's
    # Python, and no engine ever received the flag. The cache now hard-codes
    # False; a repo whose tokenizer needs custom code is estimated instead.
    prompt_tokens = await tok_cache.count(
        model.hf_repo,
        prompt_text,
        fallback_repo=getattr(model, "tokenizer_repo", None),
    )

    # God mode tap (off by default). The single cheap gate below is the ONLY
    # cost on the hot path when disabled: no hub call, byte-identical forward.
    settings = request.app.state.settings
    hub = getattr(request.app.state, "godmode_hub", None)
    gm_on = hub is not None and settings.godmode_enabled
    req_id = None
    if gm_on:
        req_id = secrets.token_hex(8)
        client_ip = _client_ip(request)
        gm_prompt, gm_prompt_elided = _godmode_prompt_window(
            prompt_text,
            settings.godmode_max_prompt_chars,
            settings.godmode_prompt_tail_chars,
        )
        gm_media = _extract_godmode_media(
            body_json, getattr(request.app.state, "godmode_media", None), settings
        )
        _gm_publish(
            hub,
            {
                "type": "request_start",
                "req_id": req_id,
                "ts": time.time(),
                # token label = name if present else id prefix. NEVER the raw
                # bearer token or Authorization header.
                "token_label": token.name or (token.id[:8] if token.id else None),
                "token_id": token.id,
                "model": model.id,
                "served_name": model.served_model_name,
                "client_ip": client_ip,
                "stream": is_stream,
                "prompt": gm_prompt,
                "prompt_elided": gm_prompt_elided,
                **({"media": gm_media} if gm_media else {}),
            },
        )

    # S5 — STRICT priority scheduler in front of vLLM. Acquired here so the
    # slot is held for the full duration of the upstream call (including the
    # streaming-response body), and released in the StreamingResponse's
    # finally block via the `_release` callback we pass through.
    scheduler = request.app.state.scheduler
    # Per-engine admission (#173 part A): key on the model id so a hot engine
    # never blocks a request bound for an idle one. The cap is the engine's
    # real capacity, computed from the model's own flags (#277) instead of a
    # fixed 16, unless the operator sets VW_PROXY_MAX_INFLIGHT globally.
    # Priority is preserved as the admission ordering within the engine's
    # queue, and pushed into vLLM itself (#173 part B) via the per-request
    # priority field.
    # Live-stats registry (Plane B): register this in-flight request so
    # GET /api/stats/requests can surface it -- BEFORE the admission wait, as
    # phase "queued", so a request stuck behind others is visible. FAIL-OPEN --
    # any registry error is swallowed so it can never break the proxied
    # request. ``live_req`` is None if registration failed; every later hook
    # guards on that.
    registry = getattr(request.app.state, "request_registry", None)
    live_req: LiveRequest | None = None
    session_source: str | None = None
    parent_session_id: str | None = None
    try:
        if affinity_source is not None:
            # The route already derived the session from the original body.
            parent_session_id = extract_session({}, request.headers).parent
            sess = SessionInfo(session_id, affinity_source if session_id else None)
        else:
            sess = extract_session(body_json, request.headers, affinity_key)
            session_id, parent_session_id = sess.id, sess.parent
        # A prompt-hash key is not a session id (the cell stays "—"), but the
        # source is still worth showing: the conversation was inferred.
        session_source = sess.source or ("prompt_hash" if dp_key_source == "prompt_hash" else None)
    except Exception:  # noqa: BLE001 -- display metadata must never fail a request
        logger.debug("live-stats: could not derive session id", exc_info=True)
    forest_kw: dict[str, Any] = {}
    try:
        _secret = request.app.state.settings.cookie_secret
        _skey = forest_fields.hkey(_secret, session_id)
        _turns = getattr(request.app.state, "session_turns", None)
        forest_kw = {
            "session_key": _skey,
            "parent_session_key": forest_fields.hkey(_secret, parent_session_id),
            "turn_index": _turns.next(_skey) if _turns is not None else None,
            "batch_id": forest_fields.clean_batch_id(request.headers.get("x-batch-id")),
            "tools_in": forest_fields.tools_in_from_body(body_json),
        }
    except Exception:  # noqa: BLE001 -- display metadata must never fail a request
        logger.debug("forest: could not derive fields", exc_info=True)
    try:
        if registry is not None:
            live_req = LiveRequest(
                id=uuid4().hex,
                token_id=token.id,
                token_name=token.name,
                client_ip=_client_ip(request),
                model=model.served_model_name,
                model_row_id=model.id,
                variant_id=variant.id,
                path=path,
                prompt_tokens=prompt_tokens,
                max_model_len=model.max_model_len,
                # Re-read once the slot is held: TTFT and elapsed are measured
                # from admission, as before.
                started_monotonic=_monotonic(),
                started_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                session_id=session_id,
                session_source=session_source,
                parent_session_id=parent_session_id,
                **forest_kw,
                phase="queued",
                is_stream=client_wants_stream,
            )
            await registry.register(live_req)
    except Exception:
        live_req = None
    except BaseException:
        if live_req is not None and registry is not None:
            await asyncio.shield(registry.deregister(live_req.id))
        raise
    # Time the admission gate ALONE. Tokenization happens above and is
    # deliberately outside this span, so the recorded
    # figure means "waited for a slot" and nothing else.
    #
    # This wait is NOT inside ttft_s or duration_s: `started_monotonic` below
    # is read once the slot is held, so it was never added in and must not be
    # subtracted from either. It is the WARDEN's queue only -- once admitted,
    # an engine keeps its own waiting queue (vLLM's continuous batching), and
    # THAT wait is inside ttft_s and invisible from here.
    _queue_start = _monotonic()
    try:
        # Constructed inside the try: a synchronous raise here must not leak the
        # already-registered "queued" row.
        slot_cm = scheduler.acquire(
            priority=token.priority,
            engine_key=model.id,
            cap=model_admission_cap(
                model.extra_args,
                data_parallel_size=getattr(model, "data_parallel_size", 1) or 1,
            ),
        )
        await slot_cm.__aenter__()
    except BaseException:
        # Cancelled or refused while waiting: nothing downstream owns the row.
        if live_req is not None and registry is not None:
            try:
                await asyncio.shield(registry.deregister(live_req.id))
            except Exception:  # noqa: BLE001
                pass
        raise
    queued_s = _monotonic() - _queue_start
    slot_released = False

    # #286 — cache-affine replica routing. After admission, so the in-flight
    # count only ever covers requests that hold a slot. FAIL-OPEN: any error
    # leaves dp_rank None (unrouted, vLLM's own balancer decides). A client
    # that already carries X-data-parallel-rank is never overridden.
    dp_rank: int | None = None
    dp_state = None
    client_pin_header = False
    try:
        dp = getattr(model, "data_parallel_size", 1) or 1
        if dp > 1:
            dp_state = getattr(request.app.state, "dp_routing", None)
            client_pin_header = RANK_HEADER.lower() in {k.lower() for k in request.headers.keys()}
            pinned = client_rank(request.headers, dp)
            # A header that does not parse to a valid rank is still the
            # client's: forwarded untouched, counted nowhere.
            if dp_state is not None and (not client_pin_header or pinned is not None):
                threshold, _src = spill_threshold_for(
                    model.extra_args, getattr(model, "dp_spill_threshold", None)
                )
                # Placement inputs, all cheap reads: the engine run (so a restart
                # forgets placements) and the latest cached KV scrape if fresh.
                place_epoch = ""
                place_kv = None
                try:
                    gen = getattr(request.app.state.supervisor, "get_generation", None)
                    place_epoch = f"{variant.id}:{gen(model.id) if gen else 0}"
                    scrapes = getattr(request.app.state, "dp_rank_scrape_cache", None)
                    if scrapes is not None:
                        place_kv = scrapes.recent_kv(model.id, KV_MAX_AGE_S, place_epoch)
                except Exception:  # noqa: BLE001
                    pass
                dp_rank, dp_decision = dp_state.route(
                    model.id,
                    dp=dp,
                    key=dp_key,
                    threshold=threshold,
                    pinned=pinned,
                    affinity_enabled=bool(getattr(model, "dp_affinity_enabled", 1)),
                    epoch=place_epoch,
                    kv=place_kv,
                )
                logger.debug(
                    "dp-routing model=%s rank=%s decision=%s source=%s",
                    model.id,
                    dp_rank,
                    dp_decision,
                    dp_key_source,
                )
    except Exception:  # noqa: BLE001 — routing must never fail a request
        dp_rank = None
        logger.debug("dp-routing: routing failed; forwarding unrouted", exc_info=True)
    dp_released = False

    # Dashboard-only metadata: the client's session id and an ESTIMATE of the
    # prefix-cache hit (see PrefixMemory). FAIL-OPEN: any error leaves both None.
    cache_est_tokens: int | None = None
    cache_est_raw_tokens: int | None = None
    est_gap_s: float | None = None
    est_decayed = False
    prefix_memory = getattr(request.app.state, "prefix_memory", None)
    # Remember/estimate only with a trustworthy basis: a real conversation key
    # (a prompt-hash key is a guess at "same conversation"), and a known
    # replica -- a data-parallel request that went unrouted could have landed
    # on any rank, so rank None is only meaningful for a single-replica model.
    mem_key: str | None = None
    mem_epoch = ""
    try:
        dp_n = getattr(model, "data_parallel_size", 1) or 1
        if (
            prefix_memory is not None
            and dp_key is not None
            and dp_key_source != "prompt_hash"
            and (dp_n <= 1 or dp_rank is not None)
        ):
            mem_key = dp_key
            # One engine run: the variant plus the supervisor's load counter, so
            # a restart or variant swap starts from an empty memory.
            gen = getattr(request.app.state.supervisor, "get_generation", None)
            mem_epoch = f"{variant.id if variant else ''}:{gen(model.id) if gen else 0}"
            # Assumes ONE shared prefix cache per replica. llama.cpp slots or an
            # MLX cache of limited size can evict or split it, so this may be
            # optimistic.
            cache_est_tokens, est_gap_s = prefix_memory.estimate_with_gap(
                model.id, mem_key, dp_rank, prompt_tokens, epoch=mem_epoch
            )
            cache_est_raw_tokens = cache_est_tokens
            # Learned (off-path) idle-gap decay: where the estimate is mostly
            # wrong for this gap, scale it down. A dictionary read.
            learned = getattr(request.app.state, "prefill_model", None)
            if cache_est_tokens and learned is not None:
                factor = learned.decay_for(model.id, est_gap_s)
                if factor is not None:
                    cache_est_tokens = int(cache_est_tokens * factor)
                    est_decayed = True
    except Exception:  # noqa: BLE001 — display metadata must never fail a request
        mem_key = None
        logger.debug("live-stats: could not derive session/cache estimate", exc_info=True)
    # Cache observation (app/cache_obs): match BEFORE forward so a request never
    # matches itself; observed in _deregister. FAIL-OPEN like the block above.
    cache_index = getattr(request.app.state, "cache_index", None)
    obs_chain = None
    obs_epoch = ""
    obs_all = obs_own = None
    try:
        if cache_index is not None:
            from app.cache_obs.canonical import chain_of

            obs_chain = chain_of(body_json)
            if obs_chain is not None:
                gen = getattr(request.app.state.supervisor, "get_generation", None)
                obs_epoch = f"{variant.id if variant else ''}:{gen(model.id) if gen else 0}"
                obs_all = cache_index.match(model.id, obs_epoch, obs_chain)
                obs_own = cache_index.match(model.id, obs_epoch, obs_chain, token_id=token.id)
    except Exception:  # noqa: BLE001 — observation must never fail a request
        obs_chain = obs_all = obs_own = None
        logger.debug("cache-obs: match failed", exc_info=True)
    # Set once the engine has actually processed the prompt (a 2xx, and either
    # a finished non-stream body or a first streamed frame). Anything earlier --
    # refused connection, 4xx/5xx, cancelled before the engine ran -- leaves the
    # previous memory entry alone.
    upstream_ok = False
    prefix_processed = False
    # An engine error frame inside a 200 stream: not served, never remembered.
    prefix_failed = False

    # #173 part B — push the token's priority into the engine itself. vLLM's
    # priority scheduler orders the waiting queue by (priority, arrival) ASCENDING
    # — LOWER value is scheduled first — while the warden convention is 9=highest.
    # Map ``vllm_priority = -warden_priority`` so warden's default 0 stays at
    # vLLM's default 0 (inert: behaves like FCFS), and higher-priority tokens
    # sort ahead of default traffic. Only inject when non-zero so unprioritised
    # requests forward a byte-identical body. Re-serialize so the upstream send
    # carries the field. Never override a priority the client set explicitly.
    # Sub-project C makes the last condition a CAPABILITY question. A backend
    # without a priority scheduler advertises supports_request_priority=False and
    # we stop injecting the field: llama-server would silently ignore it (its
    # parser looks up the fields it knows and never iterates the request's keys),
    # so this is not a bug fix -- it is refusing to re-serialise a body to send a
    # hint nothing reads, and refusing to imply a capability the engine does not
    # have. The warden's OWN admission scheduler above still orders by priority
    # for every backend; only the engine-side hint is backend-dependent.
    if (
        token.priority
        and "priority" not in body_json
        and backend_registry.get(
            getattr(model, "backend", None)
        ).capabilities.supports_request_priority
    ):
        body_json["priority"] = -token.priority
        body = json.dumps(body_json).encode()

    async def _release_slot() -> None:
        nonlocal slot_released
        if slot_released:
            return
        slot_released = True
        # Mirror the contextmanager's exit: pass no exception (we already
        # propagated upstream errors before reaching here).
        await slot_cm.__aexit__(None, None, None)

    def _release_dp_rank() -> None:
        # Exactly once, on every exit path (all of them end in _deregister).
        nonlocal dp_released
        if dp_released:
            return
        dp_released = True
        if dp_rank is None or dp_state is None:
            return
        try:
            dp_state.release(model.id, dp_rank)
        except Exception:  # noqa: BLE001 — bookkeeping must never fail a request
            logger.debug("dp-routing: release failed", exc_info=True)

    # The engine's own prompt count, once any usage carries it. Accounting and
    # display switch to it; everything the warden's pre-forward count GATES
    # (context limit, admission, max_tokens clamp) already ran on that count.
    engine_prompt: int | None = None

    def _note_engine_prompt(n: int) -> None:
        nonlocal engine_prompt
        engine_prompt = n
        if live_req is not None:
            live_req.prompt_tokens = n

    def _acct_prompt() -> int:
        return engine_prompt if engine_prompt is not None else prompt_tokens

    def _acct_cached() -> int | None:
        c = live_req.cached_tokens if live_req is not None else None
        # Both engine-reported: a cache hit cannot exceed the prompt.
        if c is not None and engine_prompt is not None:
            c = min(c, engine_prompt)
        return c

    def _set_tools_out(resp_body: dict[str, Any]) -> None:
        # Fail-open: the forest column must never affect the response.
        if live_req is None:
            return
        try:
            _msg = ((resp_body.get("choices") or [{}])[0] or {}).get("message")
            live_req.tools_out = forest_fields.tools_out_from_message(
                _msg, request.app.state.settings.cookie_secret
            )
        except Exception:  # noqa: BLE001
            logger.debug("forest: tools_out failed", exc_info=True)

    async def _deregister() -> None:
        _release_dp_rank()
        if live_req is not None:
            live_req.cached_tokens = _acct_cached()
        if live_req is None or registry is None:
            return
        # Record it BEFORE the registry forgets it. Duration, TTFT and finish
        # reason exist only at this moment; until now they were discarded here,
        # which is why the dashboard went blank the instant anything
        # interesting ended. `record` enqueues for a background writer and
        # never touches the DB here: this runs before the slot is released.
        try:
            if (
                prefix_memory is not None
                and mem_key is not None
                and upstream_ok
                and prefix_processed
                and not prefix_failed
            ):
                prefix_memory.remember(
                    model.id,
                    mem_key,
                    dp_rank,
                    prompt_tokens,
                    engine_prompt,
                    epoch=mem_epoch,
                )
        except Exception:  # noqa: BLE001 — bookkeeping must never fail a request
            logger.debug("stats: could not remember prefix", exc_info=True)
        try:
            if cache_index is not None and obs_chain is not None:
                served = upstream_ok and prefix_processed and not prefix_failed
                dp_n = getattr(model, "data_parallel_size", 1) or 1
                rank_known = dp_n <= 1 or dp_rank is not None
                # An unrouted data-parallel request is recorded under rank
                # None; record.py leaves that bit out of a routed request's
                # R_fleet, so it can never make a routed miss read "misrouted".
                if served:
                    cache_index.observe(model.id, obs_epoch, dp_rank, obs_chain, token.id)
                reporting = _cached_reporting(model)
                if served and reporting == "usage" and live_req.cached_tokens is None:
                    _note_no_cached(model.id)
                from app.cache_obs.record import observe_fields

                ttft = (
                    live_req.first_token_monotonic - live_req.started_monotonic
                    if live_req.first_token_monotonic is not None
                    else None
                )
                learned = getattr(request.app.state, "prefill_model", None)
                live_req.cache_obs = observe_fields(
                    prompt=_acct_prompt(),
                    cached=live_req.cached_tokens,
                    reporting=reporting,
                    ttft_s=ttft if live_req.is_stream else None,
                    cold_rate=learned.cold_rate_for(model.id) if learned is not None else None,
                    match_all=obs_all if served else None,
                    match_own=obs_own if served else None,
                    rank=dp_rank,
                    rank_known=rank_known,
                )
        except Exception:  # noqa: BLE001 — bookkeeping must never fail a request
            logger.debug("cache-obs: observe failed", exc_info=True)
        try:
            history = getattr(request.app.state, "request_history", None)
            if history is not None:
                history.record(finished_record(live_req, now=_monotonic()))
        except Exception:  # noqa: BLE001 — bookkeeping must never fail a request
            logger.debug("stats: could not record finished request", exc_info=True)
        try:
            await registry.deregister(live_req.id)
        except Exception:
            pass

    try:
        if live_req is not None:
            # Admitted: the clock for elapsed / TTFT starts now.
            live_req.started_monotonic = _monotonic()
            # Same instant as started_monotonic, so history's started_iso and
            # duration_s agree; the wait is queued_s, separately.
            live_req.started_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            live_req.queued_s = queued_s
            live_req.cache_est_tokens = cache_est_tokens
            live_req.cache_est_raw_tokens = cache_est_raw_tokens
            live_req.est_decayed = est_decayed
            live_req.gap_s = est_gap_s
            live_req.dp_rank = dp_rank
            # Other admitted requests on this model + replica right now: 0 means
            # this one prefills alone, the only case that measures the engine.
            # A snapshot read; fail-open to "not measured".
            try:
                if registry is not None:
                    live_req.inflight_same_rank_at_start = sum(
                        1
                        for o in registry.snapshot()
                        if o is not live_req
                        and o.model_row_id == model.id
                        and o.dp_rank == dp_rank
                        and o.phase != "queued"
                    )
            except Exception:  # noqa: BLE001
                pass
            # A non-streaming reply reports nothing until it is complete, so
            # "prefill" would be a claim the warden cannot back.
            live_req.phase = "prefill" if client_wants_stream else "waiting"
        elif registry is not None:
            live_req = LiveRequest(
                id=uuid4().hex,
                token_id=token.id,
                token_name=token.name,
                client_ip=_client_ip(request),
                model=model.served_model_name,
                model_row_id=model.id,
                variant_id=variant.id,
                path=path,
                prompt_tokens=prompt_tokens,
                max_model_len=model.max_model_len,
                started_monotonic=_monotonic(),
                started_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                queued_s=queued_s,
                session_id=session_id,
                session_source=session_source,
                parent_session_id=parent_session_id,
                **forest_kw,
                cache_est_tokens=cache_est_tokens,
                cache_est_raw_tokens=cache_est_raw_tokens,
                est_decayed=est_decayed,
                gap_s=est_gap_s,
                dp_rank=dp_rank,
                phase="prefill" if client_wants_stream else "waiting",
                is_stream=client_wants_stream,
            )
            await registry.register(live_req)
    except Exception:
        live_req = None
    except BaseException:
        # Cancelled (client gone) while registering: nothing downstream owns
        # the rank count or the admission slot yet, so release both here,
        # shielded from the cancellation that is already in flight (#286).
        async def _abort() -> None:
            await _deregister()
            await _release_slot()

        await asyncio.shield(_abort())
        raise

    try:
        url = f"http://{host}:{port}{path}"
        _gen = getattr(request.app.state.supervisor, "get_generation", None)
        client = request.app.state.upstream_clients.get(
            host, port, _gen(model.id) if _gen else 0, model_id=model.id
        )
        up_headers = {
            k: v
            for k, v in request.headers.items()
            # x-api-key: the inference key when an Anthropic client sent
            # it there (/v1/messages, #281) -- a credential, never forwarded.
            # x-lmwarden-key (#287) is the same key in its dedicated header;
            # cookie could carry the warden's own session. Neither belongs on
            # the engine, and with the router on authorization / x-api-key
            # are the client's ANTHROPIC credential -- also never the engine's.
            if k.lower()
            not in (
                "host",
                "authorization",
                "content-length",
                "x-api-key",
                "x-lmwarden-key",
                "cookie",
            )
        }
        if dp_rank is not None and not client_pin_header:
            up_headers[RANK_HEADER] = str(dp_rank)
        upstream = client.build_request(
            request.method,
            url,
            content=body,
            headers=up_headers,
        )
        # Always send with stream=True, even for non-streaming clients. With
        # stream=False httpx reads the ENTIRE upstream body inside send() — and
        # send() here runs under the client's timeout=None, so a runaway
        # non-stream generation blocks in send() forever, *before* the
        # non-stream wall-clock reaper below (which only wraps resp.aread()).
        # For a stream=False response aread() returns the already-buffered body
        # instantly, so asyncio.wait_for(resp.aread(), ...) bounds nothing and
        # the slot pins until max_model_len. stream=True defers the body read to
        # resp.aread()/aiter_bytes(), which the reaper genuinely bounds. The
        # non-stream branch still buffers the full body via aread() and returns
        # one Response, so downstream clients are byte-for-byte unchanged.
        send_start = _monotonic()
        try:
            resp = await client.send(upstream, stream=True)
        except (httpx.RemoteProtocolError, httpx.ReadError):
            # #279: the pooled keep-alive connection was closed by the engine
            # while idle (uvicorn drops idle ones after 5 s; the pool expires
            # them at 2 s, but the two clocks can still race). send() raised,
            # so no response was received; retry ONCE on a fresh request,
            # which only works because the body is bytes already in memory.
            # Never after a response has started: that is the stream's code.
            # #296: and only when the send failed fast (STALE_KEEPALIVE_FAIL_S);
            # a slow failure means the engine probably worked on the request.
            if not isinstance(body, bytes | bytearray):
                raise
            if _monotonic() - send_start >= STALE_KEEPALIVE_FAIL_S:
                raise
            logger.debug("upstream send failed before a response; retrying once", exc_info=True)
            upstream = client.build_request(
                request.method,
                url,
                content=body,
                headers=up_headers,
            )
            resp = await client.send(upstream, stream=True)
        upstream_ok = 200 <= resp.status_code < 300
    except httpx.ConnectError as exc:
        # The engine's port refused the connection (crashed, or not yet
        # restarted after a crash). That is an UPSTREAM availability fault,
        # not a warden bug: a 500 claims the warden itself broke and tells
        # SDK clients to stop, while a 502 is what OpenAI-compatible
        # gateways return for "upstream unreachable" — the status SDK retry
        # loops act on. Nothing was served, so nothing is billed (no
        # counters row); release the slot and deregister as on any send
        # failure so the next request for this engine is admitted.
        await _deregister()
        await _release_slot()
        raise HTTPException(status_code=502, detail="upstream engine unreachable") from exc
    except BaseException:
        # On send failure release the slot immediately so the next priority
        # waiter is not blocked. Re-raise to FastAPI.
        await _deregister()
        await _release_slot()
        raise

    if client_wants_stream:
        # Server-side reaper budget (0.0 = disabled). Guarantees no single
        # request pins its scheduler slot + KV blocks indefinitely when the
        # downstream client abandons the request but its TCP connection stays
        # transport-alive (writes keep draining at the token rate, so no
        # http.disconnect ever reaches uvicorn and Starlette never cancels
        # this body iterator). See config.Settings.request_max_wall_s.
        max_wall = getattr(request.app.state.settings, "request_max_wall_s", 0.0) or 0.0

        # Content logging (diagnostic, disabled by default; see content_log.py).
        # Gate resolved ONCE here so the disabled path adds nothing per-chunk.
        do_content_log = content_log.should_log(request.app.state.settings, token_id)

        async def gen():
            nonlocal prefix_processed, prefix_failed
            buf = b""
            accumulated = ""
            # R11: the text tokenized at the end — content PLUS the channels
            # _parse_sse_delta does not return (reasoning, tool-call
            # name/arguments). `accumulated` stays the visible output only
            # (delta.content, or choices[0].text on the legacy
            # /v1/completions path, R13): the content log records it.
            completion_text = ""
            # The engine's own completion count when the stream carries
            # usage.completion_tokens (stream_options.include_usage); wins
            # over the tokenizer estimate.
            stream_usage_completion: int | None = None
            gm_finish_reason = None
            # Last non-null finish_reason seen in the SSE stream, for the content
            # log record. Only tracked when do_content_log is set.
            sse_finish_reason = None
            # Coarse completion-token estimate for the live registry: count SSE
            # content deltas (vLLM streams ~1 token/delta) rather than
            # tokenizing on the hot path. Pushed into the registry
            # at most every _LIVE_UPDATE_INTERVAL_S. The exact count is still
            # recomputed once at the end for accounting (below).
            delta_count = 0
            last_live_update = 0.0
            last_disconnect_check = 0.0
            # Runaway detector (armed only when runaway_active). Feeds every
            # decoded delta — synthesizing the <think>/</think> boundary the
            # reasoning-parser strips — and, in enforce mode, tears the
            # generation down with a terminal runaway chunk on the first trip.
            detector = RunawayDetector.from_settings(settings) if runaway_active else None
            synth = {"think_open": False}
            first_meta: dict = {}
            enforce = runaway_mode == "enforce"
            enforce_trip = False

            async def _settle() -> None:
                # All end-of-stream bookkeeping for this request, in order.
                # The final count must land on ``live_req`` BEFORE
                # ``_deregister`` snapshots it into the request-history row:
                # deregistering first is what left history with the coarse
                # live ``delta_count`` estimate (content deltas only) while
                # the counters below got the true number.
                if stream_usage_completion is not None:
                    # The engine's own count: it covers every channel (content,
                    # reasoning, tool calls) across all choices, which our
                    # tokenizer estimate of the visible text cannot.
                    completion_tokens = stream_usage_completion
                else:
                    completion_tokens = await tok_cache.count(
                        model.hf_repo,
                        completion_text,
                        fallback_repo=getattr(model, "tokenizer_repo", None),
                    )
                # Same shape as the non-stream paths: reflect the final count
                # for the brief window before deregister.
                if live_req is not None:
                    try:
                        live_req.completion_tokens = completion_tokens
                    except Exception:
                        pass
                if live_req is not None:
                    try:
                        live_req.tools_out = tools_acc.result(
                            request.app.state.settings.cookie_secret
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug("forest: tools_out failed", exc_info=True)
                await _deregister()
                await resp.aclose()
                await _record_counters(
                    request,
                    model,
                    token_id,
                    _acct_prompt(),
                    completion_tokens,
                    variant,
                    _acct_cached(),
                )
                if gm_on:
                    _gm_publish(
                        hub,
                        {
                            "type": "request_end",
                            "req_id": req_id,
                            "token_id": token_id,
                            "ts": time.time(),
                            "finish_reason": gm_finish_reason,
                            "prompt_tokens": _acct_prompt(),
                            "completion_tokens": completion_tokens,
                        },
                    )
                # Release the scheduler slot only after the body is fully
                # drained — otherwise we'd hand the slot to the next waiter
                # while still hammering vLLM with our SSE chunks.
                await _release_slot()
                # Content logging (diagnostic): write AFTER the slot is released
                # so file I/O never holds the priority slot. Best-effort inside.
                # On a runaway trip the record carries the runaway finish_reason
                # plus the trip signal + think-token count as an incident.
                if do_content_log:
                    if detector is not None and detector.tripped:
                        await content_log.write_entry(
                            request.app.state.settings,
                            token_id=token_id,
                            model_id=model.id,
                            served_name=model.served_model_name,
                            stream=True,
                            max_tokens=body_json.get("max_tokens"),
                            prompt_tokens=_acct_prompt(),
                            completion_tokens=completion_tokens,
                            finish_reason=detector.finish_reason,
                            prompt=prompt_text,
                            completion=accumulated,
                            extra={
                                "signal": detector.finish_reason,
                                "think_tokens": detector.think_tokens,
                                "runaway": True,
                            },
                        )
                    else:
                        await content_log.write_entry(
                            request.app.state.settings,
                            token_id=token_id,
                            model_id=model.id,
                            served_name=model.served_model_name,
                            stream=True,
                            max_tokens=body_json.get("max_tokens"),
                            prompt_tokens=_acct_prompt(),
                            completion_tokens=completion_tokens,
                            finish_reason=sse_finish_reason,
                            prompt=prompt_text,
                            completion=accumulated,
                        )

            tools_acc = forest_fields.ToolsOutAccumulator()
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
                    buf += chunk

                    # --- Reaper + disconnect propagation --------------------
                    # Run EVERY iteration, independent of whether this chunk
                    # carried a parseable content delta (the reasoning phase
                    # emits reasoning_content deltas that _parse_sse_delta
                    # returns None for — the old check lived inside `if delta:`
                    # and was unreachable for that entire phase). On trigger,
                    # break out: the `finally` below aclose()s the upstream
                    # response + httpx client, which makes vLLM abort the
                    # generation and free its KV blocks, and releases the slot.
                    now = _monotonic()
                    # (a) wall-clock backstop — pure arithmetic, checked always.
                    if max_wall > 0.0 and (now - request_start) >= max_wall:
                        if live_req is not None:
                            try:
                                live_req.orphan = True
                            except Exception:
                                pass
                        break
                    # (b) downstream client gone — throttled receive-poll.
                    if now - last_disconnect_check >= _LIVE_UPDATE_INTERVAL_S:
                        last_disconnect_check = now
                        try:
                            if await request.is_disconnected():
                                if live_req is not None:
                                    try:
                                        live_req.orphan = True
                                    except Exception:
                                        pass
                                break
                        except Exception:
                            pass

                    while b"\n\n" in buf:
                        line, buf = buf.split(b"\n\n", 1)
                        # Any SSE `data:` frame means the engine has left prefill
                        # and is emitting tokens (content OR reasoning_content),
                        # so flip the dashboard phase to decode here rather than
                        # only on content deltas — otherwise a request that is
                        # still streaming reasoning renders as "prefill 0%".
                        # An engine error frame is not a token: it must not end
                        # prefill, set TTFT or count the request as served.
                        error_frame = b'"error"' in line and _is_sse_error_frame(line)
                        if error_frame:
                            prefix_failed = True
                            if live_req is not None:
                                live_req.finish_reason = "error"
                        if (
                            live_req is not None
                            and not error_frame
                            and live_req.first_token_monotonic is None
                            and line.lstrip().startswith(b"data:")
                        ):
                            try:
                                # First frame out of the engine: this is TTFT,
                                # and the only place every backend can be
                                # measured the same way.
                                live_req.first_token_monotonic = _monotonic()
                            except Exception:
                                pass
                        # Forest tool names: feed() has its own byte gate.
                        tools_acc.feed(line)
                        # Cheap substring gate first: the hot path must not
                        # parse JSON on every frame to learn something that
                        # appears once, in the last one.
                        if live_req is not None and b'"finish_reason"' in line:
                            try:
                                fr_seen = content_log.parse_sse_finish(line)
                                if fr_seen:
                                    live_req.finish_reason = fr_seen
                            except Exception:
                                pass
                        if do_content_log:
                            fr = content_log.parse_sse_finish(line)
                            if fr is not None:
                                sse_finish_reason = fr
                        if detector is not None and not detector.tripped:
                            ev = parse_sse_event(line)
                            if ev is not None:
                                if not first_meta:
                                    for _k in ("id", "model", "created"):
                                        if ev.get(_k) is not None:
                                            first_meta[_k] = ev[_k]
                                _feed_detector(detector, synth, ev, is_chat_path)
                                if detector.tripped and enforce:
                                    enforce_trip = True
                                    break
                        # Legacy /v1/completions frames carry choices[0].text,
                        # not delta.content (R13): parse that channel instead
                        # so the streamed text reaches accumulated (content
                        # log) and the end-of-stream count like chat content.
                        delta = _parse_sse_delta(line) if is_chat_path else _parse_sse_text(line)
                        if delta:
                            accumulated += delta
                            completion_text += delta
                            delta_count += 1
                            if live_req is not None:
                                try:
                                    if now - last_live_update >= _LIVE_UPDATE_INTERVAL_S:
                                        live_req.completion_tokens = delta_count
                                        last_live_update = now
                                except Exception:
                                    pass
                        # R11 completion accounting, separate parse (the
                        # contract above is untouched): the engine's usage
                        # frame wins over our estimate; until it arrives,
                        # reasoning + tool-call text joins the counted text.
                        frame_usage, extra, cached, frame_prompt, frame_kind = _parse_sse_frame(
                            line
                        )
                        if frame_prompt is not None:
                            _note_engine_prompt(frame_prompt)
                        if frame_kind is not None and live_req is not None:
                            live_req.phase = frame_kind
                        if frame_usage is not None:
                            stream_usage_completion = frame_usage
                        if delta or extra or (frame_usage or 0) > 0:
                            prefix_processed = True
                        if live_req is not None and cached is not None:
                            live_req.cached_tokens = cached
                        if stream_usage_completion is None:
                            completion_text += extra
                        # God-mode tap: emit content + reasoning channels and
                        # track the last non-null finish_reason. Separate parse
                        # so the counter path (accumulated) is untouched. Runs
                        # every line (independent of `delta`) so reasoning-only
                        # frames still surface on the reasoning channel.
                        if gm_on:
                            gm_content, gm_reasoning, gm_fr = _parse_sse_godmode(line)
                            if gm_content:
                                _gm_publish(
                                    hub,
                                    {
                                        "type": "delta",
                                        "req_id": req_id,
                                        "token_id": token_id,
                                        "ts": time.time(),
                                        "channel": "content",
                                        "text": gm_content,
                                    },
                                )
                            if gm_reasoning:
                                _gm_publish(
                                    hub,
                                    {
                                        "type": "delta",
                                        "req_id": req_id,
                                        "token_id": token_id,
                                        "ts": time.time(),
                                        "channel": "reasoning",
                                        "text": gm_reasoning,
                                    },
                                )
                            if gm_fr is not None:
                                gm_finish_reason = gm_fr

                    if enforce_trip:
                        # Detector tripped in enforce mode: stop forwarding
                        # upstream and let the terminal chunk below close the
                        # stream. The finally tears down the upstream socket so
                        # vLLM aborts and frees its KV blocks.
                        break
                # Enforce trip: append a schema-valid terminal chunk carrying the
                # runaway finish_reason + [DONE] so the client sees a clean end.
                if enforce_trip and detector is not None:
                    yield _runaway_terminal_chunk(is_chat_path, first_meta, detector.finish_reason)
            finally:
                # Settle in a task shielded from this task's cancellation. On
                # a client disconnect Starlette cancels this body task, and
                # while the anyio scope is still cancelled every genuine
                # suspension point in the bookkeeping below would be
                # re-cancelled: the teardown is interrupted mid-way, and ``_release_slot`` is
                # never reached — pinning one admission slot per disconnect
                # until VW_PROXY_MAX_INFLIGHT of them and the engine's proxy
                # path is dead while the model looks healthy (#217 failure
                # mode). The shielded task is a bare asyncio task outside the
                # cancelled anyio scope, so it runs to completion even if
                # this task's cancellation re-lands on the await below; the
                # scheduler's own release sites are shielded for exactly this
                # reason. On the normal path nothing is cancelled, so the
                # await simply runs _settle to completion inline as before.
                await asyncio.shield(_settle())

        # A never-started async generator skips its ``finally`` when closed or
        # discarded, so a client that vanishes before the first chunk would
        # leak the admission slot and the replica in-flight count. The guard
        # releases both exactly once in that case; once ``gen`` has started,
        # its own ``_settle`` owns the cleanup and the guard stays out of it.
        guard = _UnstartedStreamGuard(
            gen(),
            _abandon_unstarted_stream(_deregister, resp, _release_slot),
        )
        return StreamingResponse(
            guard,
            background=BackgroundTask(guard.abandon),
            status_code=resp.status_code,
            headers={"content-type": resp.headers.get("content-type", "text/event-stream")},
        )

    if runaway_active and _is_sse_stream(resp):
        # Non-streaming client, but we forced upstream stream=true so the
        # detector could watch every token. Re-aggregate the SSE back into the
        # non-stream JSON vLLM would have produced (reaggregate.py). Defensive by
        # construction — an unrecognized chunk is skipped, never fatal — because
        # the hard rule is never to 500 a request that generated fine. On an
        # enforce trip we override every choice's finish_reason with the runaway
        # signal so the client sees it on the partial content it did receive.
        do_content_log = content_log.should_log(request.app.state.settings, token_id)
        agg = StreamAggregator(is_chat=is_chat_path)
        detector = RunawayDetector.from_settings(settings)
        synth = {"think_open": False}
        enforce = runaway_mode == "enforce"
        tripped_enforce = False
        out: dict = {}
        completion = ""
        completion_tokens = 0
        runaway_cached: int | None = None
        buf = b""
        try:
            async for chunk in resp.aiter_bytes():
                buf += chunk
                while b"\n\n" in buf:
                    line, buf = buf.split(b"\n\n", 1)
                    ev = parse_sse_event(line)
                    if ev is None:
                        continue
                    agg.feed_event(ev)
                    # Keep the last reported hit: llama.cpp's timings.cache_n
                    # rides on a frame the aggregator does not keep.
                    ev_cached = _cached_tokens_of(ev)
                    if ev_cached is not None:
                        runaway_cached = ev_cached
                    if not detector.tripped:
                        _feed_detector(detector, synth, ev, is_chat_path)
                        if detector.tripped and enforce:
                            tripped_enforce = True
                            break
                if tripped_enforce:
                    break
            override = detector.finish_reason if (detector.tripped and enforce) else None
            out = agg.build(finish_reason_override=override)
            _set_tools_out(out)
            completion = _completion_text(out, is_chat_path)
            usage = out.get("usage") or {}
            prefix_processed = True
            run_prompt = _prompt_tokens_of(out)
            if run_prompt is not None:
                _note_engine_prompt(run_prompt)
            if live_req is not None:
                live_req.cached_tokens = (
                    runaway_cached if runaway_cached is not None else _cached_tokens_of(out)
                )
            completion_tokens = usage.get("completion_tokens")
            if completion_tokens is None:
                completion_tokens = await tok_cache.count(
                    model.hf_repo,
                    completion,
                    fallback_repo=getattr(model, "tokenizer_repo", None),
                )
            if live_req is not None:
                try:
                    live_req.completion_tokens = completion_tokens
                except Exception:
                    pass
            await _record_counters(
                request,
                model,
                token_id,
                _acct_prompt(),
                completion_tokens,
                variant,
                _acct_cached(),
            )
        finally:
            await _deregister()
            await resp.aclose()
            await _release_slot()
            if do_content_log:
                if detector.tripped:
                    await content_log.write_entry(
                        request.app.state.settings,
                        token_id=token_id,
                        model_id=model.id,
                        served_name=model.served_model_name,
                        stream=False,
                        max_tokens=body_json.get("max_tokens"),
                        prompt_tokens=_acct_prompt(),
                        completion_tokens=completion_tokens,
                        finish_reason=detector.finish_reason,
                        prompt=prompt_text,
                        completion=completion,
                        extra={
                            "signal": detector.finish_reason,
                            "think_tokens": detector.think_tokens,
                            "runaway": True,
                        },
                    )
                else:
                    nat_finish = None
                    if out.get("choices"):
                        nat_finish = out["choices"][0].get("finish_reason")
                    await content_log.write_entry(
                        request.app.state.settings,
                        token_id=token_id,
                        model_id=model.id,
                        served_name=model.served_model_name,
                        stream=False,
                        max_tokens=body_json.get("max_tokens"),
                        prompt_tokens=_acct_prompt(),
                        completion_tokens=completion_tokens,
                        finish_reason=nat_finish,
                        prompt=prompt_text,
                        completion=completion,
                    )
        return JSONResponse(out, status_code=resp.status_code)

    try:
        # Server-side wall-clock backstop for the non-stream path too (the
        # config contract is "streaming or not"). httpx runs with timeout=None,
        # so a hung upstream that keeps the socket transport-alive would pin the
        # slot forever. When request_max_wall_s > 0, bound the read by the time
        # already spent since request_start; on expiry asyncio.TimeoutError
        # unwinds into the `finally` below, which aclose()s the upstream socket
        # (vLLM aborts + frees KV) and releases the slot. 0.0 = disabled.
        max_wall = getattr(request.app.state.settings, "request_max_wall_s", 0.0) or 0.0
        if max_wall > 0.0:
            remaining = max_wall - (_monotonic() - request_start)
            content = await asyncio.wait_for(resp.aread(), timeout=max(remaining, 0.0))
        else:
            content = await resp.aread()
        # The engine's usage block wins when it carries a count; some engines
        # answer 200 without one, in which case the served completion is still
        # billed via the tokenizer — the same fallback the two streaming paths
        # use (a served completion billing zero would silently undercharge the
        # token's rollup). A non-JSON body counts as an empty completion, as
        # before.
        body_obj = None
        try:
            body_obj = json.loads(content)
        except Exception:
            pass
        # ``or {}`` not ``get("usage", {})``: an engine may answer with the
        # usage key PRESENT BUT ``null`` (some OpenAI-compatible builds), and
        # ``.get``'s default only covers the absent key — the ``None`` would
        # then raise AttributeError and 500 a completion the engine served
        # (R36). Treat null like absent: fall back to the tokenizer.
        engine_completion = (
            (body_obj.get("usage") or {}).get("completion_tokens")
            if isinstance(body_obj, dict)
            else None
        )
        prefix_processed = resp.status_code < 300
        body_prompt = _prompt_tokens_of(body_obj)
        if body_prompt is not None:
            _note_engine_prompt(body_prompt)
        if live_req is not None and isinstance(body_obj, dict):
            live_req.cached_tokens = _cached_tokens_of(body_obj)
            _set_tools_out(body_obj)
        if engine_completion is None:
            completion_tokens = await tok_cache.count(
                model.hf_repo,
                _completion_text(body_obj, is_chat_path) if isinstance(body_obj, dict) else "",
                fallback_repo=getattr(model, "tokenizer_repo", None),
            )
        else:
            completion_tokens = engine_completion
        # Live registry: reflect the final completion count for the brief
        # window before deregister (non-stream has no observable decode phase).
        if live_req is not None:
            try:
                live_req.completion_tokens = completion_tokens
            except Exception:
                pass
        await _record_counters(
            request,
            model,
            token_id,
            _acct_prompt(),
            completion_tokens,
            variant,
            _acct_cached(),
        )
    except TimeoutError:
        # Wall-clock backstop fired: the upstream read outlived
        # request_max_wall_s. The ``finally`` below tears down the upstream
        # socket (vLLM aborts + frees KV) and releases the slot; surface a
        # clean 504 to the still-connected client rather than a raw 500.
        raise HTTPException(status_code=504, detail="upstream request timed out") from None
    finally:
        # Close the upstream response/client in ``finally`` so a client
        # disconnect — which raises CancelledError out of ``resp.aread()``
        # above — still tears down the warden->vLLM socket synchronously. That
        # makes vLLM abort the generation and free its KV blocks immediately,
        # rather than leaving them pinned until the httpx objects are GC'd.
        # ``aclose()`` is idempotent, so calling it on the success path is
        # safe; this mirrors the streaming path's teardown. (#184)
        await _deregister()
        await resp.aclose()
        await _release_slot()

    # God-mode tap (non-stream): no upstream streaming is forced here, so the
    # client sees ONE synthetic delta per channel on completion plus a
    # request_end — a documented limitation (non-stream clients appear as one
    # block, not live token-by-token).
    if gm_on:
        try:
            parsed = json.loads(content)
            choice0 = (parsed.get("choices") or [{}])[0]
            message = choice0.get("message", {}) or {}
            gm_content = message.get("content")
            # R22: the message carries the reasoning channel under whichever
            # name the engine uses — `reasoning_content` or `reasoning`.
            gm_reasoning = delta_reasoning(message)
            if gm_content:
                _gm_publish(
                    hub,
                    {
                        "type": "delta",
                        "req_id": req_id,
                        "token_id": token_id,
                        "ts": time.time(),
                        "channel": "content",
                        "text": gm_content,
                    },
                )
            if gm_reasoning:
                _gm_publish(
                    hub,
                    {
                        "type": "delta",
                        "req_id": req_id,
                        "token_id": token_id,
                        "ts": time.time(),
                        "channel": "reasoning",
                        "text": gm_reasoning,
                    },
                )
            _gm_publish(
                hub,
                {
                    "type": "request_end",
                    "req_id": req_id,
                    "token_id": token_id,
                    "ts": time.time(),
                    "finish_reason": choice0.get("finish_reason"),
                    "prompt_tokens": _acct_prompt(),
                    "completion_tokens": completion_tokens,
                },
            )
        except Exception:
            _gm_log.warning("godmode non-stream tap failed", exc_info=True)

    # Error enrichment: on upstream 5xx, attempt to attach a last_error
    # hint from the models row to the JSON error envelope. Pure-function
    # gates inside ``enrich_5xx_from_db`` (status >= 500, JSON envelope
    # shape, last_error present) keep this cheap when nothing matches.
    if resp.status_code >= 500:
        try:
            gpu_set = ",".join(str(i) for i in sorted(model.gpu_indices or []))
            asked = {
                "concurrency": body_json.get("n") or body_json.get("best_of"),
                "max_new": body_json.get("max_tokens"),
            }
            enriched = await enrich_5xx_from_db(
                db_path=request.app.state.settings.db_path,
                model_cache=getattr(request.app.state, "model_cache", None),
                model_id=model.id,
                gpu_set=gpu_set,
                status_code=resp.status_code,
                body_bytes=content,
                asked=asked,
            )
            if enriched is not None:
                content = enriched
        except Exception:
            pass

    # Content logging (diagnostic, disabled by default; see content_log.py).
    # Runs after the slot is released (finally above) so file I/O never holds
    # the priority slot; best-effort inside write_entry.
    if content_log.should_log(request.app.state.settings, token_id):
        completion, finish_reason = content_log.parse_nonstream(
            content, path.endswith("/chat/completions")
        )
        await content_log.write_entry(
            request.app.state.settings,
            token_id=token_id,
            model_id=model.id,
            served_name=model.served_model_name,
            stream=False,
            max_tokens=body_json.get("max_tokens"),
            prompt_tokens=_acct_prompt(),
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            prompt=prompt_text,
            completion=completion,
        )

    return Response(
        content=content,
        status_code=resp.status_code,
        headers={"content-type": resp.headers.get("content-type", "application/json")},
    )


@router.post("/chat/completions")
async def chat_completions(request: Request, token: TokenRow = Depends(require_bearer)):
    body_bytes = await request.body()
    body_json = json.loads(body_bytes) if body_bytes else {}
    served_name = body_json.get("model")
    if not served_name:
        raise HTTPException(400, "missing 'model' field")
    if not token_allows(token, served_name):
        raise HTTPException(403, f"token not allowed for model '{served_name}'")
    model, host, port, variant = await _resolve_target(request, served_name)

    # Re-set the body on the request so _forward can re-read it.
    async def _receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    request._receive = _receive

    return await _forward(
        request, model, host, port, "/v1/chat/completions", token, variant=variant
    )


@router.post("/completions")
async def completions(request: Request, token: TokenRow = Depends(require_bearer)):
    body_bytes = await request.body()
    body_json = json.loads(body_bytes) if body_bytes else {}
    served_name = body_json.get("model")
    if not served_name:
        raise HTTPException(400, "missing 'model' field")
    if not token_allows(token, served_name):
        raise HTTPException(403, f"token not allowed for model '{served_name}'")
    model, host, port, variant = await _resolve_target(request, served_name)

    async def _receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    request._receive = _receive

    return await _forward(request, model, host, port, "/v1/completions", token, variant=variant)


@router.get("/models")
async def list_models(request: Request, token: TokenRow = Depends(require_bearer)):
    settings = request.app.state.settings
    from app.stress.routes_api import warden_blocks_for

    # The model list comes from the cached table (#293), which every ModelRepo
    # write drops, so it is never a superseded generation of a row.
    cache = getattr(request.app.state, "model_cache", None)
    rows = await all_models(cache, settings.db_path)
    loaded = [r for r in rows if r.status == "loaded" and token_allows(token, r.served_model_name)]
    blocks: dict[str, dict[str, Any]] = {}
    # The stress records still need the database, but only when something is
    # listed: a token that sees no loaded model costs no connection at all.
    if loaded:
        async with open_db(settings.db_path) as db:
            # Additive only. `id`/`object`/`owned_by` keep their exact meaning:
            # OpenAI clients ignore unknown fields, so `warden` costs nothing to a
            # client that does not know about it, and renaming or dropping any of
            # the three would break every client that does not.
            #
            # `warden` carries MEASURED limits for this exact configuration (or a
            # flagged stale ceiling) and never `recommended_config`, which names a
            # configuration this engine is not running -- a client told it may send
            # 96k-token prompts to an engine allocated for 32k would get a refusal
            # it has no way to interpret. See app/stress/routes_api.py.
            blocks = await warden_blocks_for(request.app.state, db, loaded, all_models=rows)
    sup = getattr(request.app.state, "supervisor", None)
    data = []
    for r in loaded:
        entry: dict[str, Any] = {
            "id": r.served_model_name,
            "object": "model",
            "owned_by": "vllm-warden",
        }
        # `max_model_len` is vLLM's own spelling on this endpoint (vllm-project
        # /vllm#4643) and OpenAI has never specified a context field, so a
        # client that knows vLLM already reads this key and one that does not
        # ignores it. It sits at the top level rather than inside `warden`
        # because it is CONFIGURED, not measured: it is known the moment the
        # engine loads, whereas the `warden` block is gated on a stress run
        # that most wardens never have. Gating a known ceiling behind an
        # optional measurement is what left clients guessing.
        window = effective_context_window(settings, sup, r)
        if window:
            entry["max_model_len"] = window
        block = blocks.get(r.id)
        if block is not None:
            entry["warden"] = block
        data.append(entry)
    return {"object": "list", "data": data}
