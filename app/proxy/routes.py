import asyncio
import json
import logging
import secrets
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.db.database import open_db
from app.db.repos.counters import CountersRepo
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.samples import SamplesRepo
from app.db.repos.tokens import TokenModelUsageRepo, TokenRow, TokenUsageRepo
from app.models.context_window import effective_context_window
from app.proxy import content_log
from app.proxy.auth import require_bearer, token_allows
from app.proxy.envelope_hint import enrich_5xx_from_db
from app.proxy.reaggregate import StreamAggregator, parse_sse_event
from app.proxy.request_registry import LiveRequest, finished_record
from app.proxy.runaway import RunawayDetector

# Aliased: the forward handler already binds a LOCAL `registry` (the Plane-B
# live-request registry off app.state), which would shadow an unaliased import
# for the whole function -- a real F823 that ruff caught rather than a style nit.
from app.runtime.backends import registry as backend_registry
from app.runtime.variants import (
    ModelVariantRepo,
    Variant,
    is_recorded,
    mark_recorded,
    running_variant,
)
from app.utils.client_ip import client_ip as _client_ip

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


async def _resolve_target(request: Request, served_name: str):
    settings = request.app.state.settings
    sup = request.app.state.supervisor
    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get_by_served_name(served_name)
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
    tail_str = text[n - tail:] if tail > 0 else ""
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
        payload = url[sep + len(";base64,"):]
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


def _parse_sse_godmode(line: bytes) -> tuple[str | None, str | None, str | None]:
    """God-mode-local SSE frame parse → ``(content, reasoning, finish_reason)``.

    Kept SEPARATE from ``_parse_sse_delta`` (whose single-string return the
    counter path depends on) so god mode can also surface ``reasoning_content``
    and ``finish_reason`` without touching that contract. With vLLM's
    ``--reasoning-parser qwen3`` the thinking stream arrives in
    ``delta.reasoning_content`` (NOT inline ``<think>`` tags), so both channels
    must be read to capture reasoning. Any part may be ``None``.
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
            delta.get("reasoning_content"),
            choice.get("finish_reason"),
        )
    except Exception:
        return None, None, None


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
    ``delta.reasoning_content`` rather than ``delta.content``. The detector's
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
        reasoning = delta.get("reasoning_content")
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
    return resp.status_code == 200 and "text/event-stream" in resp.headers.get("content-type", "").lower()


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


async def _record_counters(
    request: Request,
    model,
    token_id: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    variant: Variant | None = None,
) -> None:
    settings = request.app.state.settings
    minute = int(time.time() // 60)
    async with open_db(settings.db_path) as db:
        await CountersRepo(db).increment(
            model_id=model.id,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        await SamplesRepo(db).add_model_sample(
            model_id=model.id,
            minute=minute,
            delta_requests=1,
            delta_prompt=prompt_tokens,
            delta_completion=completion_tokens,
        )
        # S5 (#104) — per-token minute-bucket rollup feeding /api/tokens/{id}/usage.
        # Only count when token_id is set (skip system-token-less requests so the
        # NULL key doesn't accumulate orphan rows). Same minute integer as
        # SamplesRepo above for cross-table joins later (Stats v2).
        if token_id is not None:
            # Committed together with the per-model row below (one commit, not
            # two), and on its own if that part fails.
            await TokenUsageRepo(db).add(
                token_id=token_id,
                minute=minute,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                commit=False,
            )
            # 0036 -- the same minute split by model VARIANT, for the token
            # page's "Usage by model" card and per-model tokens chart. The
            # variant is the RUNNING engine's (app/runtime/variants.py), not
            # the row's current settings. Its descriptor is INSERT OR IGNOREd
            # the first time this process sees it, in case the engine start did
            # not record it. Bookkeeping: this runs after the upstream replied,
            # so a failure here is logged and must never fail the response.
            recorded: str | None = None
            try:
                if variant is None:
                    variant = running_variant(request.app.state, model)
                if not is_recorded(settings.db_path, variant.id):
                    await ModelVariantRepo(db).ensure(variant, commit=False)
                    recorded = variant.id
                await TokenModelUsageRepo(db).add(
                    token_id=token_id,
                    variant_id=variant.id,
                    model_id=model.id,
                    minute=minute,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    commit=False,
                )
            except Exception:  # noqa: BLE001 -- see above
                recorded = None
                logger.warning(
                    "could not record per-model usage for %s", model.id, exc_info=True
                )
            await db.commit()
            if recorded is not None:
                mark_recorded(settings.db_path, recorded)


async def _forward(
    request: Request,
    model: ModelRow,
    host: str,
    port: int,
    path: str,
    token: TokenRow,
    *,
    variant: Variant | None = None,
) -> Response:
    body = await request.body()
    body_json = json.loads(body) if body else {}
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
        _gm_publish(hub, {
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
        })

    # S5 — STRICT priority scheduler in front of vLLM. Acquired here so the
    # slot is held for the full duration of the upstream call (including the
    # streaming-response body), and released in the StreamingResponse's
    # finally block via the `_release` callback we pass through.
    scheduler = request.app.state.scheduler
    # Per-engine admission (#173 part A): key on the model id so each engine
    # admits up to VW_PROXY_MAX_INFLIGHT requests concurrently and a hot
    # engine never blocks a request bound for an idle one. Priority is
    # preserved as the admission ordering within the engine's queue, and
    # pushed into vLLM itself (#173 part B) via the per-request priority field.
    slot_cm = scheduler.acquire(priority=token.priority, engine_key=model.id)
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
    await slot_cm.__aenter__()
    queued_s = _monotonic() - _queue_start
    slot_released = False

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

    # Live-stats registry (Plane B): register this in-flight request so
    # GET /api/stats/requests can surface it. FAIL-OPEN — any registry error
    # is swallowed so it can never break the proxied request. ``live_req`` is
    # None if registration failed; every later hook guards on that.
    registry = getattr(request.app.state, "request_registry", None)
    live_req: LiveRequest | None = None
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
                started_monotonic=_monotonic(),
                started_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                queued_s=queued_s,
            )
            await registry.register(live_req)
    except Exception:
        live_req = None

    async def _deregister() -> None:
        if live_req is None or registry is None:
            return
        # Record it BEFORE the registry forgets it. Duration, TTFT and finish
        # reason exist only at this moment; until now they were discarded here,
        # which is why the dashboard went blank the instant anything
        # interesting ended. `record` enqueues for a background writer and
        # never touches the DB here: this runs before the slot is released.
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
        url = f"http://{host}:{port}{path}"
        client = httpx.AsyncClient(timeout=None)
        upstream = client.build_request(
            request.method, url, content=body,
            headers={k: v for k, v in request.headers.items()
                     if k.lower() not in ("host", "authorization", "content-length")},
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
        resp = await client.send(upstream, stream=True)
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
        do_content_log = content_log.should_log(
            request.app.state.settings, token_id
        )

        async def gen():
            buf = b""
            accumulated = ""
            gm_finish_reason = None
            # Last non-null finish_reason seen in the SSE stream, for the content
            # log record. Only tracked when do_content_log is set.
            sse_finish_reason = None
            # Coarse completion-token estimate for the live registry: count SSE
            # content deltas (vLLM streams ~1 token/delta for chat completions)
            # rather than tokenizing on the hot path. Pushed into the registry
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
                        if (
                            live_req is not None
                            and live_req.phase != "decode"
                            and line.lstrip().startswith(b"data:")
                        ):
                            try:
                                live_req.phase = "decode"
                                # First frame out of the engine: this is TTFT,
                                # and the only place every backend can be
                                # measured the same way.
                                if live_req.first_token_monotonic is None:
                                    live_req.first_token_monotonic = _monotonic()
                            except Exception:
                                pass
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
                        delta = _parse_sse_delta(line)
                        if delta:
                            accumulated += delta
                            delta_count += 1
                            if live_req is not None:
                                try:
                                    if now - last_live_update >= _LIVE_UPDATE_INTERVAL_S:
                                        live_req.completion_tokens = delta_count
                                        last_live_update = now
                                except Exception:
                                    pass
                        # God-mode tap: emit content + reasoning channels and
                        # track the last non-null finish_reason. Separate parse
                        # so the counter path (accumulated) is untouched. Runs
                        # every line (independent of `delta`) so reasoning-only
                        # frames still surface on the reasoning channel.
                        if gm_on:
                            gm_content, gm_reasoning, gm_fr = _parse_sse_godmode(line)
                            if gm_content:
                                _gm_publish(hub, {
                                    "type": "delta", "req_id": req_id,
                                    "token_id": token_id,
                                    "ts": time.time(), "channel": "content",
                                    "text": gm_content,
                                })
                            if gm_reasoning:
                                _gm_publish(hub, {
                                    "type": "delta", "req_id": req_id,
                                    "token_id": token_id,
                                    "ts": time.time(), "channel": "reasoning",
                                    "text": gm_reasoning,
                                })
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
                await _deregister()
                await resp.aclose()
                await client.aclose()
                completion_tokens = await tok_cache.count(
                    model.hf_repo,
                    accumulated,
                    fallback_repo=getattr(model, "tokenizer_repo", None),
                )
                await _record_counters(
                    request, model, token_id, prompt_tokens, completion_tokens, variant
                )
                if gm_on:
                    _gm_publish(hub, {
                        "type": "request_end", "req_id": req_id,
                        "token_id": token_id,
                        "ts": time.time(), "finish_reason": gm_finish_reason,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    })
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
                            prompt_tokens=prompt_tokens,
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
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            finish_reason=sse_finish_reason,
                            prompt=prompt_text,
                            completion=accumulated,
                        )

        return StreamingResponse(
            gen(), status_code=resp.status_code,
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
                    if not detector.tripped:
                        _feed_detector(detector, synth, ev, is_chat_path)
                        if detector.tripped and enforce:
                            tripped_enforce = True
                            break
                if tripped_enforce:
                    break
            override = detector.finish_reason if (detector.tripped and enforce) else None
            out = agg.build(finish_reason_override=override)
            completion = _completion_text(out, is_chat_path)
            usage = out.get("usage") or {}
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
                request, model, token_id, prompt_tokens, completion_tokens, variant
            )
        finally:
            await _deregister()
            await resp.aclose()
            await client.aclose()
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
                        prompt_tokens=prompt_tokens,
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
                        prompt_tokens=prompt_tokens,
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
        completion_tokens = 0
        try:
            completion_tokens = json.loads(content).get("usage", {}).get("completion_tokens", 0)
        except Exception:
            pass
        # Live registry: reflect the final completion count for the brief
        # window before deregister (non-stream has no observable decode phase).
        if live_req is not None:
            try:
                live_req.completion_tokens = completion_tokens
            except Exception:
                pass
        await _record_counters(
            request, model, token_id, prompt_tokens, completion_tokens, variant
        )
    except TimeoutError:
        # Wall-clock backstop fired: the upstream read outlived
        # request_max_wall_s. The ``finally`` below tears down the upstream
        # socket (vLLM aborts + frees KV) and releases the slot; surface a
        # clean 504 to the still-connected client rather than a raw 500.
        raise HTTPException(
            status_code=504, detail="upstream request timed out"
        ) from None
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
        await client.aclose()
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
            gm_reasoning = message.get("reasoning_content")
            if gm_content:
                _gm_publish(hub, {
                    "type": "delta", "req_id": req_id, "token_id": token_id,
                    "ts": time.time(),
                    "channel": "content", "text": gm_content,
                })
            if gm_reasoning:
                _gm_publish(hub, {
                    "type": "delta", "req_id": req_id, "token_id": token_id,
                    "ts": time.time(),
                    "channel": "reasoning", "text": gm_reasoning,
                })
            _gm_publish(hub, {
                "type": "request_end", "req_id": req_id, "token_id": token_id,
                "ts": time.time(),
                "finish_reason": choice0.get("finish_reason"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })
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
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            prompt=prompt_text,
            completion=completion,
        )

    return Response(
        content=content, status_code=resp.status_code,
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
    # The stress record is read on the SAME connection as the model list so a
    # client never sees a model listed against a measurement taken for a
    # different generation of the row.
    from app.stress.routes_api import warden_blocks_for

    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()
        loaded = [
            r for r in rows if r.status == "loaded" and token_allows(token, r.served_model_name)
        ]
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
            "id": r.served_model_name, "object": "model", "owned_by": "vllm-warden",
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
