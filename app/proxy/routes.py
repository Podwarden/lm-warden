import asyncio
import functools
import json
import logging
import secrets
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.cache_obs import canonical as _cobs_canonical
from app.cobs2.canonical import _HEAD_ROLES, Canon, canonicalize, loose_message, loose_text
from app.cobs2.ids import HeadPartIds, Ids, conv_id, head_part_ids, ids_for, scope_salt
from app.cobs2.keys import NetworkKey
from app.cobs2.lineage import Lineage, Observation, key_scope
from app.cobs2.reply import ReplyCapture
from app.cobs2.shape import Toolset, break_index, dead_weight_of, shape_json, shape_of, toolset_of
from app.db.database import open_db
from app.db.repos.models import ModelRow
from app.db.repos.tokens import TokenRow
from app.guard import proxy as guard_proxy
from app.guard.placeholders import RestoreMap
from app.lens import hot as lens_hot
from app.lens import reply_verdicts
from app.lens import versions as lens_versions
from app.lens.causes import LensVerdict, RenderOrder
from app.lens.heads import RecentHeads
from app.models.context_window import effective_context_window
from app.models.parallelism import spill_threshold_for
from app.proxy import content_log, forest_fields, prompt_lens_switch, remote_media
from app.proxy.alias_overrides import apply_alias_params
from app.proxy.aliases import first_loaded_target, resolve_alias
from app.proxy.auth import require_bearer, token_allows
from app.proxy.capture import capturing
from app.proxy.dp_affinity import KV_MAX_AGE_S, RANK_HEADER, client_rank
from app.proxy.dp_affinity import conversation_key as _dp_affinity_key
from app.proxy.envelope_hint import enrich_5xx_from_db
from app.proxy.kv import kv_client, record_finished, set_kv_params, strip_client_kv_params
from app.proxy.kv_scope import (
    CACHE_SALT,
    ISOLATION_UNAVAILABLE,
    UNKEYED_SHARING,
    CacheSaltRefused,
    KvIdentity,
    can_isolate_backend,
    can_isolate_running,
    cannot_isolate_message,
    check_cache_salt,
    gpu_salt,
    is_isolated,
    kv_identity,
    reserved_salt,
    rewrite_cache_salt,
    running_backend,
)
from app.proxy.live_tokens import LiveOutCounter
from app.proxy.model_cache import all_aliases, all_models, model_by_served_name
from app.proxy.prompt_diet import apply_diet, parse_diet
from app.proxy.prompt_lens import PromptLensHub
from app.proxy.prompt_lens_capture import (
    BodyLayout,
    GmFrame,
    ToolCallBudget,
    ToolCallDelta,
    _prompt_lens_prompt_window,
    body_layout,
    structured_prompt,
    tool_call_delta,
)
from app.proxy.prompt_lens_kvsig import build_kvsig, build_kvsig_pair
from app.proxy.reaggregate import StreamAggregator, parse_sse_event
from app.proxy.request_id import REQUEST_ID_HEADER, with_request_id
from app.proxy.request_registry import LiveRequest, finished_record
from app.proxy.runaway import RunawayDetector
from app.proxy.scheduler import model_admission_cap
from app.proxy.session_id import SessionInfo, extract_session
from app.proxy.thinking_policy import (
    apply_thinking_off,
    apply_thinking_on,
    effective_policy,
)

# Aliased: the forward handler already binds a LOCAL `registry` (the Plane-B
# live-request registry off app.state), which would shadow an unaliased import
# for the whole function -- a real F823 that ruff caught rather than a style nit.
from app.runtime.backends import registry as backend_registry
from app.runtime.kv_store import kv_enabled
from app.runtime.variants import (
    Variant,
    running_variant,
)
from app.sessions.refs import sess_ref
from app.stats import speed
from app.utils.client_ip import client_ip as _client_ip
from app.utils.sse import delta_reasoning

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["proxy"])

# Prompt Lens taps log-and-swallow: a broadcast failure must NEVER break, delay,
# or backpressure a proxied response.
_gm_log = logging.getLogger("vllm_warden.prompt_lens")

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


async def _resolve_target(request: Request, served_name: str) -> tuple[ModelRow, str, int, Variant]:
    # From the cached models table (#293): every ModelRepo write drops it, so a
    # status change decides the very next request's 404.
    cache = getattr(request.app.state, "model_cache", None)
    db_path = request.app.state.settings.db_path
    # Aliases first (D4a): an enabled alias wins over a model with the same served name.
    alias = await resolve_alias(cache, db_path, served_name)
    if alias is not None:
        # D4b: the first LOADED target in the ordered list serves the request.
        model = first_loaded_target(alias, await all_models(cache, db_path))
        request.state.lmw_alias = alias
        if model is None:
            raise HTTPException(404, f"no model behind alias '{alias.name}' is loaded")
        return _resolve_loaded(request, model, served_name)
    request.state.lmw_alias = None
    model = await model_by_served_name(cache, db_path, served_name)
    return _resolve_loaded(request, model, served_name)


async def _resolve_for_token(
    request: Request, token: TokenRow, served_name: str
) -> tuple[ModelRow, str, int, Variant]:
    """Authorize ``token`` for ``served_name`` and resolve the engine behind it.

    Every route that takes a model name from the body (the OpenAI routes and
    /v1/messages) goes through here, so an alias is handled the same way
    everywhere: it wins over a model of the same name, the first loaded target
    serves, and the key's allow-list may name the alias OR the serving model
    (D4d). Sets ``request.state.lmw_alias`` for the alias settings _forward applies.
    """
    alias = await resolve_alias(
        getattr(request.app.state, "model_cache", None),
        request.app.state.settings.db_path,
        served_name,
    )
    if alias is None and not token_allows(token, served_name):
        # Non-alias request: the original refusal, before the lookup that 404s.
        raise HTTPException(403, f"token not allowed for model '{served_name}'")
    target = await _resolve_target(request, served_name)
    if alias is not None and not (
        token_allows(token, served_name) or token_allows(token, target[0].served_model_name)
    ):
        # D4d: a request through an alias is allowed when the key's allow-list
        # contains the alias name OR the served name of the model that serves it.
        raise HTTPException(403, f"token not allowed for model '{served_name}'")
    return target


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


def _extract_prompt_lens_media(body_json, store, settings) -> list[dict]:
    """Display-only capture of ``image_url`` content parts for Prompt Lens
    (spec 2026-08-03). NEVER touches the forwarded request. Never raises —
    a malformed body yields a short (possibly empty) list.

    data URIs -> payload sliced into ``store`` (base64 string, no decode on
    the hot path), event carries {media_id, mime, chars}. Remote http(s)
    URLs -> {url} (the admin browser hotlinks them; the warden never
    fetches). Oversized -> {dropped: "too_large"}. More than
    ``prompt_lens_max_images_per_req`` -> one trailing {dropped: "count"}.
    """
    media: list[dict] = []
    try:
        if not isinstance(body_json, dict):
            return media
        messages = body_json.get("messages")
        if not isinstance(messages, list):
            return media
        max_items = settings.prompt_lens_max_images_per_req
        max_chars = settings.prompt_lens_max_image_chars
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
                entry = _prompt_lens_media_entry(url, store, max_chars)
                if entry is None:
                    continue
                media.append(entry)
        if overflow:
            media.append({"kind": "image", "dropped": "count", "count": overflow})
    except Exception:
        logger.warning("prompt_lens media extraction failed", exc_info=True)
    return media


def _prompt_lens_media_entry(url: str, store, max_chars: int) -> dict | None:
    """One image_url string -> media entry dict, or None to skip. Split from
    _extract_prompt_lens_media so the per-item paths stay readable."""
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


def _parse_sse_prompt_lens(line: bytes) -> GmFrame:
    """Prompt Lens-local SSE frame parse → ``GmFrame(content, reasoning,
    finish_reason, tool_calls)`` of choice 0.

    Kept SEPARATE from ``_parse_sse_delta`` (whose single-string return the
    counter path depends on) so Prompt Lens can also surface the reasoning
    channel, the tool calls and ``finish_reason`` without touching that
    contract. With vLLM's ``--reasoning-parser qwen3`` the thinking stream
    arrives in ``delta.reasoning_content`` — or ``delta.reasoning`` on newer
    builds (R22) — NOT inline ``<think>`` tags, so ``delta_reasoning`` reads
    both names. ``tool_calls`` holds one ``ToolCallDelta`` per
    ``delta.tool_calls`` entry in frame order (parallel calls interleave by
    ``index``); a bad entry is skipped and never costs the other channels.
    """
    if not line.startswith(b"data:"):
        return _GM_EMPTY
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return _GM_EMPTY
    try:
        choice = json.loads(payload)["choices"][0]
        delta = choice.get("delta", {}) or {}
        content = delta.get("content")
        reasoning = delta_reasoning(delta)
        finish = choice.get("finish_reason")
    except Exception:
        return _GM_EMPTY
    calls: tuple[ToolCallDelta, ...] = ()
    raw = delta.get("tool_calls")
    if isinstance(raw, list) and raw:
        calls = tuple(
            d for d in (tool_call_delta(tc, pos) for pos, tc in enumerate(raw)) if d is not None
        )
    return GmFrame(content, reasoning, finish, calls)


_GM_EMPTY = GmFrame(None, None, None, ())


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


def _gm_prompt_captures(
    body_json: Any, narrow: int, wide: int | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The stream event's structured prompt and, when ``wide`` is set, the
    section store's wide one (G9): every window as wide as its budget."""
    parts = structured_prompt(body_json, budget_chars=narrow)
    if wide is None:
        return parts, None
    return parts, structured_prompt(body_json, budget_chars=wide, sys_chars=wide, msg_chars=wide)


def _gm_publish(hub: Any, event: dict[str, Any], *, keep: bool = True) -> None:
    """Publish an event, swallowing+logging any failure. Prompt Lens is strictly
    best-effort observation; it must never surface an error into ``_forward``.
    ``keep=False`` is a live-only event (``progress``): not kept in the ring."""
    try:
        if keep:
            hub.publish(event)
        else:
            hub.publish(event, keep=False)
    except Exception:
        _gm_log.warning("prompt_lens publish failed for %s", event.get("type"), exc_info=True)


def _gm_publish_tool_calls(
    hub: Any,
    req_id: str | None,
    token_id: str | None,
    calls: Iterable[ToolCallDelta],
    budget: ToolCallBudget,
) -> None:
    """Publish one ``delta{channel: "tool_call"}`` per ``ToolCallDelta`` in
    ``calls``, in order, within ``budget``. ``call_id``/``name`` appear only
    when the fragment carries them. A call is never marked done here: the
    stream gives no per-call end, so a call is done at ``request_end``."""
    for d in calls:
        fields = budget.take(d)
        if fields is None:
            continue
        _gm_publish(
            hub,
            {
                "type": "delta",
                "req_id": req_id,
                "token_id": token_id,
                "ts": time.time(),
                "channel": "tool_call",
                **fields,
            },
        )


def _reasoning_tokens_of(obj: Any) -> int | None:
    """``usage.completion_tokens_details.reasoning_tokens``, when the engine reports it."""
    usage = obj.get("usage") if isinstance(obj, dict) else None
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    return _nonneg_int(details.get("reasoning_tokens")) if isinstance(details, dict) else None


def _s3(v: float | None) -> float | None:
    return round(v, 3) if v is not None else None


def _gm_end(
    hub: Any,
    *,
    req_id: str | None,
    token_id: str | None,
    finish_reason: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    live_req: LiveRequest | None,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
    t0: float | None = None,
) -> None:
    """Publish one request's ``request_end`` (G5): usage, timings and cache.

    Timings are the request-history row's (``finished_record``): ``ttft_s`` and
    ``duration_s`` from admission, ``queued_s`` the admission wait; a request that never
    left the queue spent all of ``duration_s`` there. ``decode_tps`` is the stats page's
    (``speed.row_metrics``). ``cache`` is ``live_req.cache_obs`` (admin only: the key
    holder's stream strips it). ``t0`` times a request with no live row. Never raises.
    """
    try:
        now = _monotonic()
        queued = ttft = dur = None
        cache: dict[str, Any] | None = None
        if live_req is not None:
            dur = now - live_req.started_monotonic
            queued = live_req.queued_s
            if queued is None and live_req.phase == "queued":
                queued = dur
            if live_req.first_token_monotonic is not None:
                ttft = live_req.first_token_monotonic - live_req.started_monotonic
            if isinstance(live_req.cache_obs, dict):
                cache = dict(live_req.cache_obs)
        elif t0 is not None:
            dur = now - t0
        queued, ttft, dur = _s3(queued), _s3(ttft), _s3(dur)
        decode = speed.row_metrics(
            {
                "queued_s": queued,
                "ttft_s": ttft,
                "duration_s": dur,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )["decode"]
        own = _nonneg_int(cache.get("reusable_tokens_own")) if cache is not None else None
        event: dict[str, Any] = {
            "type": "request_end",
            "req_id": req_id,
            "token_id": token_id,
            "ts": time.time(),
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
            "queued_s": queued,
            "ttft_s": ttft,
            "duration_s": dur,
            "decode_tps": decode,
            "own_reused_tokens": (
                min(cached_tokens, own) if cached_tokens is not None and own is not None else None
            ),
            "cache": cache,
            **(extra or {}),
        }
    except Exception:  # noqa: BLE001 -- display only, never into _forward
        _gm_log.warning("prompt_lens request_end failed", exc_info=True)
        return
    _gm_publish(hub, event)


#: The fastest a request's live ``progress`` event repeats (G6), whatever the
#: live tick (``_LIVE_UPDATE_INTERVAL_S``) is.
_GM_PROGRESS_MIN_S = 1.0


class _GmProgress:
    """One streamed request's per-channel frame counts (G6), published live-only.

    ``frames`` counts SSE frames with text on each channel, one per frame: a cheap
    stand-in for a per-section rate (approximate, Q6), with no per-token tokenizing.
    ``tick`` publishes at most once per ``max(_LIVE_UPDATE_INTERVAL_S, 1 s)``; the
    request's ``_GmEnd.end`` publishes the final counts once, right before
    ``request_end`` (ruling G-F5). Exists only with Prompt Lens on, streamed requests
    only. Never raises into ``_forward``.
    """

    __slots__ = ("hub", "req_id", "token_id", "gm", "frames", "last")

    def __init__(self, gm: "_GmEnd") -> None:
        self.hub = gm.hub
        self.req_id = gm.req_id
        self.token_id = gm.token_id
        self.gm = gm
        self.frames = {"reasoning": 0, "content": 0, "tool_call": 0}
        self.last = float("-inf")

    def count(self, content: str | None, reasoning: str | None, calls: Any, now: float) -> None:
        """Count one SSE frame's channels; publish when the throttle allows."""
        grew = False
        if reasoning:
            self.frames["reasoning"] += 1
            grew = True
        if content:
            self.frames["content"] += 1
            grew = True
        if calls:
            self.frames["tool_call"] += 1
            grew = True
        if grew and now - self.last >= max(_LIVE_UPDATE_INTERVAL_S, _GM_PROGRESS_MIN_S):
            self.last = now
            lr = self.gm.live_req
            self.publish(lr.completion_tokens if lr is not None else None)

    def publish(self, completion_tokens: int | None) -> None:
        lr = self.gm.live_req
        _gm_publish(
            self.hub,
            {
                "type": "progress",
                "req_id": self.req_id,
                "token_id": self.token_id,
                "ts": time.time(),
                "phase": lr.phase if lr is not None else "running",
                "frames": dict(self.frames),
                "completion_tokens": completion_tokens,
            },
            keep=False,
        )


class _GmEnd:
    """One request's ``request_end``, published at most once (G5, ruling G-F1).

    Created only with Prompt Lens on, right after ``request_start``. The exit that knows the
    outcome calls ``end``; ``_forward``'s outer ``finally`` calls ``fallback`` for every
    other exit (queue refusal, cancelled while queued or registering, send failures, ...),
    unless the response was handed to a stream, whose generator then owns the end.
    """

    __slots__ = (
        "hub",
        "req_id",
        "token_id",
        "prompt_tokens",
        "t0",
        "live_req",
        "done",
        "handed_off",
        "progress",
    )

    def __init__(self, hub: Any, req_id: str | None, token_id: str | None, prompt_tokens: int):
        self.hub = hub
        self.req_id = req_id
        self.token_id = token_id
        self.prompt_tokens = prompt_tokens
        self.t0 = _monotonic()
        self.live_req: LiveRequest | None = None
        self.done = False
        self.handed_off = False
        #: Set by a streamed request's generator (G6); its final counts go out here.
        self.progress: _GmProgress | None = None

    def end(
        self,
        finish_reason: str | None,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.done:
            return
        self.done = True
        if self.progress is not None:
            # G-F5: the final counts, unthrottled, right before request_end.
            # Guarded: a progress failure must never cost the request its end.
            try:
                self.progress.publish(completion_tokens)
            except Exception:  # noqa: BLE001 -- display only
                _gm_log.warning("prompt_lens final progress failed", exc_info=True)
        _gm_end(
            self.hub,
            req_id=self.req_id,
            token_id=self.token_id,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            live_req=self.live_req,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            extra=extra,
            t0=self.t0,
        )

    def fallback(self, finish_reason: str | None) -> None:
        """The end of an exit that published none, from what the live row holds."""
        lr = self.live_req
        self.end(
            finish_reason,
            prompt_tokens=lr.prompt_tokens if lr is not None else self.prompt_tokens,
            completion_tokens=lr.completion_tokens if lr is not None else 0,
            cached_tokens=lr.cached_tokens if lr is not None else None,
        )


def _gm_after_settle(gm: "_GmEnd", _task: Any) -> None:
    """The settle task's done-callback. ``_settle`` publishes the end itself, so this
    only acts when it did not reach that point: it raised (or was cancelled), an
    ``error``. It never reads the task's exception, so an unawaited failure still
    logs as it did before."""
    gm.fallback("error")


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


def _produced_s(live_req: Any, now: float) -> float | None:
    """How long the request was producing tokens, for per-minute booking (B7, D3).

    Streamed: from the first token to now (the prompt books at the first-token
    minute). Not streamed, the warden sees no first token: from the moment the
    slot was held to now, so the prompt books at the slot's minute and the
    reply is spread over the whole slot (prefill included, the closest the
    proxy can see). None without a live row: everything books at the finish.
    """
    if live_req is None:
        return None
    try:
        t0 = live_req.first_token_monotonic
        if t0 is None:
            t0 = live_req.started_monotonic
        return max(0.0, now - float(t0))
    except Exception:  # noqa: BLE001 -- accounting must never fail a request
        return None


async def _record_counters(
    request: Request,
    model,
    token_id: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    variant: Variant | None = None,
    cached_tokens: int | None = None,
    produced_s: float | None = None,
) -> None:
    # #279 stage 3: accumulate only; app/proxy/ledger.py writes once a second.
    # ``produced_s``: first token to now, so the per-minute rollups book the
    # tokens to the minutes they were produced in (B7, D3).
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
            produced_s=produced_s,
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


def _honours_cache_salt(model: Any) -> bool:
    """Whether the backend isolates its prefix cache by cache_salt (#300).
    True when unknown: the vLLM default, and a salted chain errs to cold."""
    try:
        return backend_registry.get(getattr(model, "backend", None)).capabilities.honours_cache_salt
    except Exception:  # noqa: BLE001
        return True


def _cached_reporting(model: Any) -> str:
    """The backend's cached_tokens_reporting; "usage" when unknown (fail-open)."""
    try:
        return backend_registry.get(
            getattr(model, "backend", None)
        ).capabilities.cached_tokens_reporting
    except Exception:  # noqa: BLE001
        return "usage"


#: cobs2 hot-path bounds (task 5 fix round 1). Huge contexts are the product:
#: the cap stays MAX_CHAIN_BYTES (2 MiB), and the costs that scale with one big
#: message are bounded instead.
#: Bodies over this are canonicalised and hashed in a worker thread.
_COBS2_THREAD_BYTES = 256 * 1024
#: The compaction-fingerprint scan reads this much of each end of the last
#: user message (the summarize prompt is short and sits at one end).
_COBS2_SCAN_CHARS = 32 * 1024
#: Sketch input prefix: the sketch reads the first 2000 shingles (~2000 words).
_COBS2_SKETCH_CHARS = 64 * 1024
#: The reply's prefix, roomier: a leading <analysis> block is stripped first.
_COBS2_REPLY_SKETCH_CHARS = 256 * 1024


def _cobs2_raw_text(msg: dict[str, Any]) -> str:
    return " ".join(p["v"] for p in msg["parts"] if p.get("t") == "text")


def _cobs2_text(msg: dict[str, Any], limit: int | None = None) -> str:
    """The text parts of a canonical message, loose-transformed, without
    reasoning. With ``limit``, a message longer than that is cut to its first
    ``limit`` characters BEFORE the (linear-time) loose transform."""
    if limit is not None:
        raw = _cobs2_raw_text(msg)
        if len(raw) > limit:
            return loose_text(raw[:limit])
    return " ".join(p["v"] for p in loose_message(msg)["parts"] if p.get("t") == "text")


def _cobs2_scan_text(msg: dict[str, Any]) -> str:
    """Loose text for the compaction-fingerprint scan: the whole message when
    short, else only its first and last ``_COBS2_SCAN_CHARS`` characters."""
    raw = _cobs2_raw_text(msg)
    if len(raw) <= 2 * _COBS2_SCAN_CHARS:
        return _cobs2_text(msg)
    return loose_text(raw[:_COBS2_SCAN_CHARS]) + " " + loose_text(raw[-_COBS2_SCAN_CHARS:])


def _cobs2_size_over(canon: Canon, limit: int) -> bool:
    """True when the canonical form's strings exceed ``limit`` UTF-8 bytes.

    Characters first (a lower bound on bytes, and a UTF-8 character is at most
    4 bytes): only a form between ``limit / 4`` and ``limit`` characters is
    encoded to count its bytes."""
    strs: list[str] = [t for t in canon.tools if isinstance(t, str)]
    media = 0

    def walk(ps: list[dict[str, Any]]) -> None:
        nonlocal media
        for p in ps:
            t = p.get("t")
            if t in ("text", "reasoning"):
                strs.append(p["v"])
            elif t == "tool_call":
                strs.append(p["name"])
                strs.append(p["args"])
            elif t == "tool_result":
                walk(p["parts"])
            else:
                media += 64

    for m in (*canon.head, *canon.messages):
        walk(m["parts"])
    chars = media + sum(len(x) for x in strs)
    if chars > limit:
        return True
    if chars * 4 <= limit:
        return False
    return media + sum(len(x.encode("utf-8", "surrogatepass")) for x in strs) > limit


class _Cobs2Pre:
    """The pure part of the cobs2 pass (safe in a worker thread)."""

    __slots__ = ("ids", "salt", "scan_text", "first", "conv", "shape", "toolset", "gm", "lens")

    def __init__(
        self,
        ids: Ids,
        salt: bytes,
        scan_text: str,
        first: dict[str, Any] | None,
        conv: str | None = None,
        shape: dict[str, Any] | None = None,
        toolset: Toolset | None = None,
        gm: tuple[HeadPartIds, BodyLayout] | None = None,
        lens: lens_hot.LensPre | None = None,
    ) -> None:
        self.ids = ids
        self.salt = salt
        self.scan_text = scan_text
        self.first = first
        # Sessions pages: conversation id, sizes/counts, offered toolset.
        self.conv = conv
        self.shape = shape
        self.toolset = toolset
        # Prompt Lens only (G3): the head part ids and the body layout the kvsig
        # event reads. None when Prompt Lens is off: no canon is kept either way.
        self.gm = gm
        # Prompt Lens layer 1 (analysis plan Task 1.3): digests, hashes, sizes
        # and the memoised probes; it holds the body until the link releases it.
        self.lens = lens


class _Cobs2Seen:
    """What ``_forward`` keeps from the cobs2 pass for the finish. ``salt`` is the
    LINEAGE scope (``key_scope``: cache_salt digest narrowed to the API key), so
    ``remember_reply`` lands in the same per-key scope as the observe."""

    __slots__ = ("ids", "obs", "salt", "compaction", "lens")

    def __init__(self, ids: Ids, obs: Observation | None, salt: bytes, compaction: bool) -> None:
        self.ids = ids
        self.obs = obs
        self.salt = salt
        self.compaction = compaction
        # Prompt Lens layer 1: the lineage-side inputs (None: not computed).
        self.lens: lens_hot.LensState | None = None


def _cobs2_compute(
    body_json: Any,
    key: NetworkKey | None,
    secret: str | None = None,
    prompt_lens: bool = False,
    lens: bool = False,
) -> _Cobs2Pre | None:
    """Canonical form, size cap, ids and the bounded fingerprint text. Pure;
    runs in a worker thread for a big body (the id memo is locked).

    ``prompt_lens`` (on only while Prompt Lens is): also the head part ids and the
    body layout for the kvsig event, in this same pass (``gm``; None when
    either fails, which fails only the event).

    ``lens``: also the Prompt Lens layer 1 inputs (``app.lens.hot.prepare``),
    in this same pass; None when they fail, which fails only the lens fields.

    None when there is nothing to hash, no key, or the canonical form is over
    ``MAX_CHAIN_BYTES`` UTF-8 bytes (ids_for is then never run). Raises on a
    bug; the caller fails open."""
    if key is None or not isinstance(body_json, dict):
        return None
    canon = canonicalize(body_json)
    if canon is None or _cobs2_size_over(canon, _cobs_canonical.MAX_CHAIN_BYTES):
        return None
    salt = scope_salt(body_json)
    ids = ids_for(canon, key, salt)
    msgs = canon.messages
    # Only a request whose LAST message is the user's can be the summarize
    # call: in a tool loop the last user message is an earlier human prompt.
    last_user = msgs[-1] if msgs and msgs[-1]["role"] == "user" else None
    scan = _cobs2_scan_text(last_user) if last_user is not None else ""
    # Sessions pages: each field fails alone (None), never the ids.
    conv: str | None = None
    shape: dict[str, Any] | None = None
    toolset: Toolset | None = None
    try:
        conv = conv_id(ids, key, salt)
    except Exception:  # noqa: BLE001
        logger.debug("cobs2: conv id failed", exc_info=True)
    try:
        shape = shape_of(canon)
    except Exception:  # noqa: BLE001
        logger.debug("cobs2: shape failed", exc_info=True)
    if shape is not None:
        # Dead weight (sessions pages): its own field; a failure drops only it.
        try:
            shape["dw"] = dead_weight_of(canon, ids)
        except Exception:  # noqa: BLE001
            logger.debug("cobs2: dead weight failed", exc_info=True)
    if secret is not None:
        try:
            toolset = toolset_of(body_json.get("tools"), canon.tools, secret)
        except Exception:  # noqa: BLE001
            logger.debug("cobs2: toolset failed", exc_info=True)
    gm: tuple[HeadPartIds, BodyLayout] | None = None
    if prompt_lens:
        try:
            layout = body_layout(body_json)
            if layout is not None:
                gm = (head_part_ids(canon, key, salt, ids.head), layout)
        except Exception as exc:  # noqa: BLE001 -- display only
            # The type only: an exception's text could quote the body.
            _gm_log.warning("prompt_lens kvsig layout failed: %s", type(exc).__name__)
    lens_pre: lens_hot.LensPre | None = None
    if lens:
        try:
            lens_pre = lens_hot.prepare(body_json, canon, ids, key, salt)
        except Exception as exc:  # noqa: BLE001 -- the lens never fails the pass
            # The type only: an exception's text could quote the body.
            _gm_log.warning("lens: prepare failed: %s", type(exc).__name__)
    return _Cobs2Pre(ids, salt, scan, msgs[0] if msgs else None, conv, shape, toolset, gm, lens_pre)


def _gm_kvsig(
    body_json: Any,
    key: NetworkKey | None,
    pre: _Cobs2Pre | None,
    seen: _Cobs2Seen | None,
    failed: str | None,
    *,
    chain: Any,
    match_all: Any,
    match_own: Any,
    prompt_tokens: int,
    lens: LensVerdict | None = None,
) -> dict[str, Any]:
    """The kvsig event body for ``_forward`` (G3), or why there is none:
    ``not_chat`` (no ``messages`` list), ``no_key`` (no cobs2 key), ``error``
    (the compute or the layout failed), ``over_cap`` (the canonical form is
    over the chain cap, so no ids were hashed)."""
    reason: str | None = None
    if not isinstance(body_json, dict) or not isinstance(body_json.get("messages"), list):
        reason = "not_chat"
    elif key is None:
        reason = "no_key"
    elif failed is not None:
        reason = failed
    elif pre is None:
        reason = "over_cap"
    elif pre.gm is None:
        reason = "error"
    if reason is not None or pre is None or pre.gm is None:
        return build_kvsig(
            ids=None,
            head_parts=None,
            obs=None,
            chain=None,
            match_all=None,
            match_own=None,
            prompt_tokens=prompt_tokens,
            raw_body_index=[],
            parts_chars=[],
            sys_chars=0,
            tools_chars=0,
            roles=[],
            reason=reason or "error",
        )
    head_parts, lay = pre.gm
    return build_kvsig_pair(
        ids=pre.ids,
        head_parts=head_parts,
        obs=seen.obs if seen is not None else None,
        chain=chain,
        match_all=match_all,
        match_own=match_own,
        prompt_tokens=prompt_tokens,
        raw_body_index=lay.raw_body_index,
        parts_chars=lay.parts_chars,
        sys_chars=lay.sys_chars,
        tools_chars=lay.tools_chars,
        roles=lay.roles,
        reason=None,
        raw_head_index=lay.raw_head_index,
        head_roles=lay.head_roles,
        lens=lens,
    )


def _cobs2_key_ids(body_json: Any, key: NetworkKey | None) -> Ids | None:
    """Ids over the head, the tools and the FIRST body message only: what the
    replica key reads (``loose_head``, ``loose_nodes[0]``), whatever the body's
    size. Equal to the full compute's values: the head is every system /
    developer message, and nothing before the first body message sets a
    tool-call reference. Pure; None when there is nothing to hash."""
    if key is None or not isinstance(body_json, dict):
        return None
    msgs = body_json.get("messages")
    if not isinstance(msgs, list):
        return None
    keep: list[Any] = []
    have_first = False
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") in _HEAD_ROLES:
            keep.append(m)
        elif not have_first:
            keep.append(m)
            have_first = True
    prefix: dict[str, Any] = {"messages": keep}
    if "tools" in body_json:
        prefix["tools"] = body_json["tools"]
    canon = canonicalize(prefix)
    if canon is None:
        return None
    return ids_for(canon, key, scope_salt(body_json))


def _cobs2_link(
    pre: _Cobs2Pre,
    *,
    lineage: Lineage | None,
    model_id: str,
    session_key: str | None,
    parent_session_key: str | None,
    token_scope: str | None,
    inflight_heads: Any,
    request_id: str | None = None,
    lens_heads: RecentHeads | None = None,
    lens_parents: lens_hot.ParentHeads | None = None,
    render_order: RenderOrder = "sys_first",
) -> _Cobs2Seen:
    """The lineage step, on the event loop. The first message's sketch text is
    built lazily, only when the body collapsed.

    With ``pre.lens`` (Prompt Lens layer 1): the lineage records each new
    node's probes from its memo, then ``lens_hot.link`` finds the parent's head
    and feeds ``lens_heads``; any fault leaves ``seen.lens`` None (fail-open)."""
    compaction = Lineage.is_compaction_request(pre.scan_text)
    # Per API key: the lineage never links one key's request to another's.
    scope = key_scope(pre.salt, token_scope)
    obs = None
    lp = pre.lens
    if lineage is not None:
        first = pre.first
        obs = lineage.observe(
            model_id=model_id,
            scope=scope,
            session_key=session_key,
            parent_session_key=parent_session_key,
            ids=pre.ids,
            last_user_text_loose=pre.scan_text,
            first_body_text_loose=(
                (lambda: _cobs2_text(first, _COBS2_SKETCH_CHARS)) if first is not None else ""
            ),
            token_scope=token_scope,
            inflight_heads=inflight_heads,
            request_id=request_id,
            conv=pre.conv,
            probes_for=lp.probes_at if lp is not None else None,
        )
    seen = _Cobs2Seen(pre.ids, obs, scope, compaction)
    if lp is not None:
        pre.lens = None
        if obs is not None:
            try:
                seen.lens = lens_hot.link(
                    lp,
                    obs,
                    scope=scope,
                    model_id=model_id,
                    request_id=request_id,
                    heads=lens_heads,
                    parents=lens_parents,
                    now=time.monotonic(),
                    render_order=render_order,
                )
            except Exception as exc:  # noqa: BLE001 -- the lens never fails the link
                _gm_log.warning("lens: link failed: %s", type(exc).__name__)
        if seen.lens is None:
            lp.release()
    return seen


def _cobs2_observe(
    body_json: Any,
    *,
    key: NetworkKey | None,
    lineage: Lineage | None,
    model_id: str,
    session_key: str | None,
    parent_session_key: str | None,
    token_scope: str | None,
    inflight_heads: Any,
) -> _Cobs2Seen | None:
    """``_cobs2_compute`` then ``_cobs2_link``, in the calling thread."""
    pre = _cobs2_compute(body_json, key)
    if pre is None:
        return None
    return _cobs2_link(
        pre,
        lineage=lineage,
        model_id=model_id,
        session_key=session_key,
        parent_session_key=parent_session_key,
        token_scope=token_scope,
        inflight_heads=inflight_heads,
    )


async def _idle_bounded(it: Any, idle_s: float, live_req: Any) -> Any:
    """``it``'s chunks, ending when the upstream sends nothing for ``idle_s`` (verified finding
    #4): the caller's ``finally`` then closes the upstream (the engine aborts and frees its KV
    blocks) and releases the slot, as the wall-clock reaper does."""
    ait = it.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(ait.__anext__(), timeout=idle_s)
        except StopAsyncIteration:
            return
        except TimeoutError:
            logger.warning("upstream sent nothing for %.0f s; ending the stream", idle_s)
            if live_req is not None:
                try:
                    live_req.orphan = True
                except Exception:  # noqa: BLE001
                    pass
            return
        yield chunk


async def _guard_inspect(
    request: Request,
    guard: Any,
    token: TokenRow,
    body_json: dict[str, Any],
    raw_len: int,
    affinity_key: str | None,
) -> guard_proxy.Outcome:
    """Run the key's guard on the client's body (off the loop for big bodies, like cobs2)."""
    state = request.app.state
    secret = guard_proxy.shared_secret(state)
    seen = guard_proxy.shared_seen(state)
    try:
        session = extract_session(body_json, request.headers, affinity_key).id
    except Exception:  # noqa: BLE001 -- no session id: the key is the session
        session = None
    run = functools.partial(
        guard_proxy.run_guard,
        body_json,
        "openai",
        guard,
        token_id=token.id,
        secret=secret,
        session=session or token.id,
        seen=seen,
        raw_len=raw_len,
    )
    # Big bodies, and any custom pattern (an admin's regex is not ours to trust with the event
    # loop), are scanned on a worker thread.
    if raw_len > 256 * 1024 or any(r.pack == "custom" for r in guard.rules):
        return await asyncio.to_thread(run)
    return run()


def _guard_log(request: Request, events: list[Any], request_id: str) -> None:
    guard_proxy.log_events(request.app.state, events, request_id)


async def _record_blocked(
    request: Request,
    model: ModelRow,
    token: TokenRow,
    path: str,
    body_json: dict[str, Any],
    outcome: guard_proxy.Outcome,
    request_id: str,
) -> None:
    """A history row for a request the guard answered itself: no slot, no live row, no counters
    as served, the prompt counted, no completion."""
    try:
        history = getattr(request.app.state, "request_history", None)
        if history is None:
            return
        prompt_tokens = await request.app.state.tokenizers.count(
            model.hf_repo,
            _extract_prompt(body_json),
            fallback_repo=getattr(model, "tokenizer_repo", None),
        )
        now = _monotonic()
        variant = running_variant(request.app.state, model)
        req = LiveRequest(
            id=request_id,
            token_id=token.id,
            token_name=token.name,
            client_ip=_client_ip(request),
            model=model.served_model_name,
            model_row_id=model.id,
            variant_id=variant.id,
            path=path,
            prompt_tokens=prompt_tokens,
            max_model_len=model.max_model_len,
            started_monotonic=now,
            started_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            phase="blocked",
            is_stream=bool(body_json.get("stream")),
        )
        req.finish_reason = "guard_blocked"
        req.completion_tokens = 0
        req.guard = outcome.history_json
        history.record(finished_record(req, now=now))
    except Exception:  # noqa: BLE001 -- bookkeeping must never fail the block reply
        logger.debug("guard: could not record the blocked request", exc_info=True)


async def _guard_frames(it: Any, trailer: str | None) -> Any:
    """The engine's SSE with the guard's trailer put before the finish chunk (engine view: no
    value restored). Without a trailer the chunks pass through untouched."""
    if not trailer:
        async for c in it:
            yield c
        return
    r = guard_proxy.ChatStreamRestorer(RestoreMap(), trailer=trailer)
    async for c in it:
        out = r.feed(c)
        if out:
            yield out
    tail = r.flush()
    if tail:
        yield tail


#: ``_redacted_engine_body``: the client sent no cache_salt / keep the body's own.
_NO_SALT = object()
_KEEP_SALT = object()


#: Whether the missing-network-key WARNING was logged since the key was last seen.
_warned_no_network_key = False


def _kv_isolation_preflight(
    request: Request, model: Any, token: TokenRow, body_json: Any, variant: Any = None
) -> tuple[KvIdentity, bytes | None, bool]:
    """The request's KV identity, the network key, and whether the engine's cache_salt is
    rewritten (every backend that honours it: vLLM). Raises the client's refusal.

    Fail closed for an isolated key: a model whose backend cannot isolate its cache
    (llama.cpp, MLX, an unknown backend) is a 403, decided first (O4: the answer, never a
    failure to retry or fall back from); no network key, or a scope that cannot be derived
    (no tenant id, a bad epoch), is a 503; a cache_salt that cannot be used is a 400. A
    sharing key without the network key stays unattributed, as in step 1, except that a
    reserved ``lmw1:`` salt, which cannot be rewritten then, is a 503 too.
    """
    global _warned_no_network_key
    isolated = is_isolated(token)
    # The engine as launched decides, not the row (which may have been edited since), both
    # the refusal and the salt rewrite, so an admitted isolated request is always salted.
    launched = running_backend(request.app.state, model, variant)
    if isolated and (launched is None or not can_isolate_backend(launched)):
        raise HTTPException(403, cannot_isolate_message(getattr(model, "served_model_name", "")))
    salt_rewrite = (
        can_isolate_backend(launched) if launched is not None else _honours_cache_salt(model)
    )
    key_obj = getattr(request.app.state, "cobs2_key", None)
    nk = getattr(key_obj, "key", None) if key_obj is not None else None
    if isinstance(nk, bytes):
        _warned_no_network_key = False
    elif not _warned_no_network_key:
        _warned_no_network_key = True
        logger.warning(
            "kv: the network key is unavailable; isolated keys get 503 and sharing keys "
            "run unattributed until it is back"
        )
    ident: KvIdentity | None = None
    if isinstance(nk, bytes):
        try:
            ident = kv_identity(token, nk)
        except ValueError:  # never the value: these are secrets of the key holder
            ident = None
    if ident is None:
        nk = None
        if isolated:
            raise HTTPException(503, ISOLATION_UNAVAILABLE)
        ident = UNKEYED_SHARING
    if ident.isolated and not isinstance(body_json, dict):
        raise HTTPException(400, "the request body must be a JSON object")
    try:
        check_cache_salt(body_json, ident)
    except CacheSaltRefused as e:
        raise HTTPException(400, str(e)) from None
    if nk is None and salt_rewrite and reserved_salt(body_json):
        raise HTTPException(503, ISOLATION_UNAVAILABLE)
    return ident, nk, salt_rewrite


def _salt_secrets(salt: Any) -> list[bytes]:
    """The derived salt's secret part (its hex), as bytes; [] for anything else."""
    if isinstance(salt, str) and salt.startswith("lmw1:"):
        return [salt[len("lmw1:") :].encode()]
    return []


def _apply_cache_salt(body_json: Any, ident: KvIdentity, nk: bytes | None) -> bool:
    """The late half: the engine's cache_salt for ``ident``; True when the body changed.

    An isolated request never reaches the engine without its derived salt (ruling R7):
    should the rewrite fail, it gets the salt of an empty client salt; should even that
    fail, this raises a 503 and the caller's send path releases the slot and the live row.
    """
    if nk is None or not isinstance(body_json, dict):
        if ident.isolated:  # the pre-flight refuses these; never forward one unsalted
            raise HTTPException(503, ISOLATION_UNAVAILABLE)
        return False
    try:
        return rewrite_cache_salt(body_json, ident, nk)
    except Exception as e:  # noqa: BLE001 -- never the value, only its type
        logger.warning("kv: cache_salt rewrite failed (%s)", type(e).__name__)
        if not ident.isolated and not reserved_salt(body_json):
            return False
        body_json[CACHE_SALT] = gpu_salt(nk, ident.scope, "")
        return True


#: What replaces a per-key secret in an engine response the proxy relays or captures.
_SECRET_PLACEHOLDER = b"<redacted>"


def _scrub_secrets(data: bytes, secrets: tuple[bytes, ...]) -> bytes:
    """``data`` with every per-key secret of this request replaced by the placeholder.

    An engine error can echo the request (vLLM's RequestValidationError puts the whole
    request dict in a 400), and that request carries the derived cache_salt and the
    lmw_tenant / lmw_scope hex. The values are hex, which JSON never escapes.
    """
    for s in secrets:
        if s in data:
            data = data.replace(s, _SECRET_PLACEHOLDER)
    return data


class _ScrubbedResponse:
    """An upstream error response, read whole and scrubbed (``_scrub_secrets``); the rest
    of ``_forward`` reads it through ``aread`` / ``aiter_bytes`` as before."""

    def __init__(self, resp: Any, content: bytes) -> None:
        self._resp = resp
        self._content = content
        self.status_code = resp.status_code
        self.headers = resp.headers

    async def aread(self) -> bytes:
        return self._content

    async def aiter_bytes(self) -> Any:
        if self._content:
            yield self._content

    async def aclose(self) -> None:
        await self._resp.aclose()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resp, name)


def _redacted_engine_body(body: bytes, client_salt: Any) -> bytes:
    """The engine body as the debug capture keeps it: the client's own cache_salt and
    the lmw_tenant / lmw_scope values blanked (secrets of the key holder)."""
    try:
        view = json.loads(body)
    except ValueError:
        return b""
    if not isinstance(view, dict):
        return b""
    if client_salt is _NO_SALT:
        view.pop(CACHE_SALT, None)
    elif client_salt is not _KEEP_SALT:
        view[CACHE_SALT] = client_salt
    kvp = view.get("kv_transfer_params")
    if isinstance(kvp, dict):
        for k in ("lmw_tenant", "lmw_scope"):
            if kvp.get(k):
                kvp[k] = "<redacted>"
    return json.dumps(view).encode()


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
    wire_body: bytes | None = None,
    wire_path: str | None = None,
) -> Response:
    """Run one request through admission, the engine and the bookkeeping.

    ``wire_body`` / ``wire_path`` are for the routes that translate (``/v1/messages``,
    ``/v1/responses``): the bytes the client sent and the path it called, which only the debug
    capture uses (it records them as ``request_body`` / ``path``, and the translated chat body as
    ``translated_body``). Both default to None: the request body is then the wire body.

    ``affinity_source`` and ``session_id`` are for routes that swap the request
    body for a translated one (``/v1/responses``): the translated chat body no
    longer carries the client's ``prompt_cache_key``, so the route derives
    (key, source, session id) from the ORIGINAL body and hands them in. With
    ``affinity_source`` set, ``affinity_key`` is used verbatim as the replica
    key and the extractor is not run on the (translated) body; ``session_id``
    is what the live row shows. Both default to None: no behaviour change.

    G-F1: every request whose ``request_start`` Prompt Lens published gets exactly one
    ``request_end``. The exit that knows the outcome publishes it; this outer ``finally``
    publishes it for every other exit (``abort`` when cancelled, ``error`` on a raise or
    an error status). With Prompt Lens off ``gm_slot`` stays empty: nothing else runs.
    """
    gm_slot: list[_GmEnd] = []
    gm_exit: str | None = None
    try:
        out = await _forward_body(
            request,
            model,
            host,
            port,
            path,
            token,
            gm_slot,
            variant=variant,
            affinity_key=affinity_key,
            affinity_source=affinity_source,
            session_id=session_id,
            wire_body=wire_body,
            wire_path=wire_path,
        )
        if gm_slot and out.status_code >= 400:
            gm_exit = "error"
        return out
    except BaseException as exc:
        if gm_slot:
            gm_exit = "abort" if isinstance(exc, asyncio.CancelledError) else "error"
        raise
    finally:
        if gm_slot and not gm_slot[0].handed_off:
            gm_slot[0].fallback(gm_exit)


async def _forward_body(
    request: Request,
    model: ModelRow,
    host: str,
    port: int,
    path: str,
    token: TokenRow,
    gm_slot: list[_GmEnd],
    *,
    variant: Variant | None,
    affinity_key: str | None,
    affinity_source: str | None,
    session_id: str | None,
    wire_body: bytes | None,
    wire_path: str | None,
) -> Response:
    """``_forward``'s work; ``gm_slot`` receives the request's ``_GmEnd`` (Prompt Lens on)."""
    wire_sanitized = False
    body = await request.body()
    client_body = body  # debug capture: the request as _forward got it (translated, if it was)
    body_json = json.loads(body) if body else {}
    # KV-cache tiers: with kv_store on, a client's own kv_transfer_params never reach the
    # engine (vLLM acts on do_remote_prefill / do_remote_decode, and the connector on lmw_*).
    # With kv_store off they pass through: the engine's connector, if any, is the operator's
    # (NIXL or disaggregated prefill through extra_args) and needs them.
    if kv_enabled(model) and strip_client_kv_params(body_json):
        body = json.dumps(body_json).encode()
    # Per-key KV isolation (app/proxy/kv_scope.py), first half: the request's identity and
    # every refusal, decided HERE, before the live row and the admission slot exist, so a
    # refused request holds neither. The engine's cache_salt is rewritten just before the
    # upstream send, after cobs2 and cache observation read the client's own salt.
    kv_ident, kv_nk, salt_rewrite = _kv_isolation_preflight(
        request, model, token, body_json, variant
    )
    kv_redact = False  # the capture's engine body hides the per-key values
    # The per-key secrets this request carries to the engine (derived salt, tenant and
    # scope hex): scrubbed from a relayed error body and from the captured response.
    kv_secrets: list[bytes] = []
    kv_capture_salt: Any = _KEEP_SALT
    # Sensitive data guard (app/guard/proxy.py), before anything reads the body: everything below
    # (Prompt Lens, cobs2, capture, history, the engine) sees the sanitized body. Off for this key:
    # one attribute read, and the path is byte-identical.
    g_out: guard_proxy.Outcome | None = None
    # The router (app/router/data_plane.py) already scanned and sanitized a /v1/messages body:
    # its outcome comes with the request, and scanning the translation again would find nothing.
    g_handed = getattr(request.state, "guard", None)
    g_guard = (
        None
        if g_handed is not None
        else guard_proxy.guard_for(
            getattr(request.app.state, "guard_platform", None),
            token,
            getattr(request.state, "lmw_alias", None),
        )
    )
    if g_handed is not None:
        g_out = g_handed
    elif g_guard is not None:
        try:
            g_out = await _guard_inspect(
                request, g_guard, token, body_json, len(body), affinity_key
            )
        except guard_proxy.GuardFailed as e:
            return guard_proxy.failed_reply(e.reason)
        if g_out.action == "block":
            blocked_id = uuid4().hex
            _guard_log(request, g_out.events, blocked_id)
            await _record_blocked(request, model, token, path, body_json, g_out, blocked_id)
            return with_request_id(
                guard_proxy.block_reply(g_out, stream=bool(body_json.get("stream")), path=path),
                blocked_id,
            )
        if g_out.action == "sanitize":
            body_json = g_out.body
            body = json.dumps(body_json).encode()
            client_body = body
            # The guard rewrote the translated body; the wire bytes still hold what it removed,
            # and the capture must not (the guard's promise is that nothing downstream sees it).
            wire_sanitized = wire_body is not None
            wire_body = None
        else:
            g_out = None
    # Restoring (client side) and the trailer (engine side, so reply ids, capture and the content
    # log see what the model's next turn will carry back). Both None without findings.
    g_restore: RestoreMap | None = g_out.restore if g_out is not None and g_out.restore else None
    g_trailer: str | None = g_out.trailer if g_out is not None else None
    g_restored = [0]
    # The id the engine serves (applied to the body after the #173 priority
    # block below). Resolved HERE, before any live row or admission slot is
    # taken, so a raise cannot leak either. The supervisor's launch-time record
    # first (MLX's id is the resolved local path, which only the launch knows);
    # the row's answer only for an engine this supervisor did not launch -- MLX
    # raises ValueError there rather than name an id the engine would download.
    _sup_engine_id = getattr(
        getattr(request.app.state, "supervisor", None), "get_engine_model_id", None
    )
    engine_id = _sup_engine_id(model.id) if _sup_engine_id is not None else None
    if not isinstance(engine_id, str):
        engine_id = backend_registry.get(getattr(model, "backend", None)).engine_model_id(model)
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

    # Prompt Lens tap (off by default). The single cheap gate below is the ONLY
    # cost on the hot path when disabled: no hub call, byte-identical forward.
    settings = request.app.state.settings
    hub = getattr(request.app.state, "prompt_lens_hub", None)
    gm_on = hub is not None and prompt_lens_switch.enabled(request.app)
    if gm_on and isinstance(hub, PromptLensHub):
        # Bound to the hub's epoch now: if Prompt Lens goes off (and on again)
        # while this request runs, nothing more of it is published.
        hub = hub.bound()
    # The request's one request_end (G5, G-F1); set once request_start is published.
    gm: _GmEnd | None = None
    gm_sections: Any = None
    gm_wide: dict[str, Any] | None = None
    req_id = None
    # One id for the live row, the history row, the response header and the
    # Prompt Lens capture (its request_start names it, so a capture can be found
    # by the request it belongs to -- the key holder's View, stats redesign B11).
    request_id = uuid4().hex
    if gm_on:
        req_id = secrets.token_hex(8)
        client_ip = _client_ip(request)
        gm_prompt, gm_prompt_elided = _prompt_lens_prompt_window(
            prompt_text,
            settings.prompt_lens_max_prompt_chars,
            settings.prompt_lens_prompt_tail_chars,
        )
        gm_media = _extract_prompt_lens_media(
            body_json, getattr(request.app.state, "prompt_lens_media", None), settings
        )
        # G1: the structured capture (roles, tools, tool calls and results),
        # windowed and bounded. A big body is walked off the loop, like the
        # cobs2 ids; nothing is held yet, so a cancellation here leaks nothing.
        # G9: the WIDE capture for the section store (tool results and the
        # prompt on expand), in the same pass: one window per message as wide
        # as the whole budget, so a tool result is whole up to it.
        gm_sections = getattr(request.app.state, "prompt_lens_sections", None)
        gm_parts_fn = functools.partial(
            _gm_prompt_captures,
            body_json,
            settings.prompt_lens_max_prompt_chars,
            settings.prompt_lens_section_prompt_chars if gm_sections is not None else None,
        )
        if len(body) > _COBS2_THREAD_BYTES:
            gm_parts, gm_wide = await asyncio.to_thread(gm_parts_fn)
        else:
            gm_parts, gm_wide = gm_parts_fn()

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
    # Prompt Lens layer 3 (analysis plan Task 1.4) runs only with kvsig on (a
    # network key): with it off the lens makes zero calls and its fields stay None.
    lens_l3 = getattr(request.app.state, "cobs2_key", None) is not None
    try:
        _secret = request.app.state.settings.cookie_secret
        _skey = forest_fields.hkey(_secret, session_id)
        _turns = getattr(request.app.state, "session_turns", None)
        forest_kw = {
            "session_key": _skey,
            "parent_session_key": forest_fields.hkey(_secret, parent_session_id),
            "turn_index": _turns.next(_skey) if _turns is not None else None,
            "batch_id": forest_fields.clean_batch_id(request.headers.get("x-batch-id")),
            # The error heuristic for OpenAI-format results only: an Anthropic
            # tool_result says is_error itself (translated to an "Error" prefix).
            "tools_in": forest_fields.tools_in_from_body(
                body_json, heuristic=lens_l3 and wire_path != "/v1/messages"
            ),
        }
    except Exception:  # noqa: BLE001 -- display metadata must never fail a request
        logger.debug("forest: could not derive fields", exc_info=True)
    if lens_l3:
        # Layer 3 (spec 8.1, length cut): the cap this request asked for.
        try:
            _mt = _nonneg_int(body_json.get("max_tokens"))
            if _mt is None:
                _mt = _nonneg_int(body_json.get("max_completion_tokens"))
            forest_kw["max_tokens_req"] = _mt
        except Exception as exc:  # noqa: BLE001 -- the lens never fails a request
            _gm_log.warning("lens: max_tokens failed: %s", type(exc).__name__)
        # Prompt version (Task 3.1): the client's own name wins over detection.
        try:
            forest_kw["lens_version"] = lens_versions.client_version(
                request.headers.get("x-prompt-version")
            )
        except Exception as exc:  # noqa: BLE001 -- the lens never fails a request
            _gm_log.warning("lens: version header failed: %s", type(exc).__name__)
    if gm_on:
        # Published here, once the session and the turn are known, still before
        # the scheduler acquire (a queued request is visible) and with no await
        # between the forest fields and the publish.
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
                "request_id": request_id,
                "model": model.id,
                "served_name": model.served_model_name,
                "client_ip": client_ip,
                "stream": is_stream,
                # The flat prompt stays for one release (scripts read it).
                "prompt": gm_prompt,
                "prompt_elided": gm_prompt_elided,
                "prompt_parts": gm_parts,
                "session_id": session_id,
                "turn": forest_kw.get("turn_index"),
                **({"media": gm_media} if gm_media else {}),
            },
        )
        if gm_sections is not None:
            # Right after its request_start reached the store, no await between.
            gm_sections.put_prompt(
                req_id, token.id, gm_wide, keep=int(getattr(token, "keep_captures", 0) or 0)
            )
        gm = _GmEnd(hub, req_id, token.id, prompt_tokens)
        gm_slot.append(gm)
    if g_out is not None and g_out.events and g_handed is None:
        _guard_log(request, g_out.events, request_id)
    try:
        if registry is not None:
            live_req = LiveRequest(
                id=request_id,
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
    if gm is not None:
        gm.live_req = live_req
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

    # cobs2 L1 (spec section 6): content ids over the body as the client sent
    # it (the runaway mutation above only touches stream fields; the ones below
    # touch none of these). Hashed HERE, after admission and before the replica
    # routing below, so the conversation key can come from the ids (spec 8.1).
    # A big body is hashed in a worker thread. FAIL-OPEN: any error means no
    # ids. A cancellation while waiting on the thread releases the slot and
    # the queued row (no rank is held yet). Ids and digests only.
    cobs2_pre: _Cobs2Pre | None = None
    # The ids the replica key reads. The full compute's when it ran, else
    # (size cap, or an error) the same values from the head, the tools and
    # message 0 alone, so the key never flips with the body's size.
    cobs2_key_ids: Ids | None = None
    # G3: why the kvsig event has no parts when the compute gave none.
    gm_kv_reason: str | None = None
    cobs2_key = None
    try:
        try:
            cobs2_key = getattr(request.app.state, "cobs2_key", None)
            if (
                cobs2_key is not None
                and isinstance(body_json, dict)
                and isinstance(body_json.get("messages"), list)
            ):
                cobs2_snap = {
                    k: body_json[k] for k in ("messages", "tools", "cache_salt") if k in body_json
                }
                cobs2_secret = request.app.state.settings.cookie_secret
                # Message 0 alone can only be over the thread bound if the body is.
                cobs2_off_loop = len(body) > _COBS2_THREAD_BYTES
                try:
                    if cobs2_off_loop:
                        cobs2_pre = await asyncio.to_thread(
                            _cobs2_compute, cobs2_snap, cobs2_key, cobs2_secret, gm_on, True
                        )
                    else:
                        cobs2_pre = _cobs2_compute(cobs2_snap, cobs2_key, cobs2_secret, gm_on, True)
                except Exception:  # noqa: BLE001 — identity must never fail a request
                    cobs2_pre = None
                    gm_kv_reason = "error"
                    logger.debug("cobs2: ids failed", exc_info=True)
                if cobs2_pre is not None:
                    cobs2_key_ids = cobs2_pre.ids
                elif (
                    affinity_source is not None
                    or (getattr(model, "data_parallel_size", 1) or 1) <= 1
                ):
                    pass  # no replica key to derive: never hash a capped body
                elif cobs2_off_loop:
                    cobs2_key_ids = await asyncio.to_thread(_cobs2_key_ids, cobs2_snap, cobs2_key)
                else:
                    cobs2_key_ids = _cobs2_key_ids(cobs2_snap, cobs2_key)
        except Exception:  # noqa: BLE001 — routing must never fail a request
            cobs2_key_ids = None
            logger.debug("cobs2: key ids failed; today's key", exc_info=True)
    except BaseException:
        # Cancelled while waiting on the hashing thread. Accepted: the request
        # never reaches request history (it was not served), only the slot and
        # the queued live row are released.

        async def _abort_ids() -> None:
            try:
                await slot_cm.__aexit__(None, None, None)
            finally:
                if live_req is not None and registry is not None:
                    await registry.deregister(live_req.id)

        slot_released = True
        try:
            await asyncio.shield(_abort_ids())
        except Exception:  # noqa: BLE001
            pass
        raise
    # The replica key from the ids: the conversation keeps its rank when a
    # reminder or date changes in the head or first message. Only when the
    # key was derived here (a route-supplied key is used verbatim).
    if cobs2_key_ids is not None and affinity_source is None:
        try:
            dp_key, dp_key_source = _dp_affinity_key(
                body_json,
                request.headers,
                token.id,
                explicit=affinity_key,
                cobs2_ids=cobs2_key_ids,
            )
        except Exception:  # noqa: BLE001 — routing must never fail a request
            logger.debug("dp-routing: could not derive an ids affinity key", exc_info=True)

    # #286 — cache-affine replica routing. After admission, so the in-flight
    # count only ever covers requests that hold a slot. FAIL-OPEN: any error
    # leaves dp_rank None (unrouted, vLLM's own balancer decides). A client
    # that already carries X-data-parallel-rank is never overridden.
    dp_rank: int | None = None
    dp_state = None
    place_epoch = ""
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
                place_kv = None
                place_pool = None
                try:
                    gen = getattr(request.app.state.supervisor, "get_generation", None)
                    place_epoch = f"{variant.id}:{gen(model.id) if gen else 0}"
                    scrapes = getattr(request.app.state, "dp_rank_scrape_cache", None)
                    if scrapes is not None:
                        place_kv = scrapes.recent_kv(model.id, KV_MAX_AGE_S, place_epoch)
                        place_pool = scrapes.recent_pool_tokens(model.id, KV_MAX_AGE_S, place_epoch)
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
                    pool_tokens=place_pool,
                )
                logger.debug(
                    "dp-routing model=%s rank=%s decision=%s source=%s reason=%s",
                    model.id,
                    dp_rank,
                    dp_decision,
                    dp_key_source,
                    getattr(dp_state, "last_reason", None),
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
            # Per API key (key_scope): another key's prompts with the same
            # client session id never feed this key's estimate.
            mem_key = key_scope(
                dp_key.encode("utf-8", "surrogatepass"),
                forest_fields.hkey(request.app.state.settings.cookie_secret, token.id),
            ).hex()
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
    iso_star_frac: float | None = None
    try:
        if cache_index is not None:
            from app.cache_obs.canonical import chain_of

            obs_chain = chain_of(body_json, honour_salt=_honours_cache_salt(model))
            if obs_chain is not None:
                gen = getattr(request.app.state.supervisor, "get_generation", None)
                obs_epoch = f"{variant.id if variant else ''}:{gen(model.id) if gen else 0}"
                obs_all = cache_index.match(model.id, obs_epoch, obs_chain)
                obs_own = cache_index.match(model.id, obs_epoch, obs_chain, token_id=token.id)
                if kv_ident.isolated:
                    # An isolated key can be served by its own blocks only: its fleet
                    # figure is its own lens. The operator lens (what sharing would have
                    # served) is kept for the admin-only isolation cost.
                    if obs_chain.total_bytes > 0:
                        iso_star_frac = obs_all.best_bytes / obs_chain.total_bytes
                    obs_all = obs_own
    except Exception:  # noqa: BLE001 — observation must never fail a request
        obs_chain = obs_all = obs_own = None
        iso_star_frac = None
        logger.debug("cache-obs: match failed", exc_info=True)
    cobs2_seen: _Cobs2Seen | None = None
    cobs2_kw: dict[str, Any] = {}
    # Prompt Lens layer 1 (analysis plan Task 1.3): set after the link, read by
    # the kvsig event and, with the outcome, at the finish. None = no lens.
    lens_state: lens_hot.LensState | None = None
    cobs2_lineage = getattr(request.app.state, "cobs2_lineage", None)
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

    # Pin the model id the ENGINE actually serves this row under (the Backend's
    # engine_model_id: vLLM/llama.cpp the --served-model-name/--alias plan()
    # passes, MLX the local --model path — it has no alias at all). A body
    # naming any other id reaches the engine as an unknown name; for MLX that
    # triggers a Hub download of ANOTHER model (Metal design s5 A). Re-serialise
    # ONLY when it differs from what the client sent, so a body already naming
    # the served id forwards byte-identical (same discipline as the priority
    # injection above, which it composes with: one re-serialisation carries
    # both fields).
    if body_json.get("model") != engine_id:
        body_json["model"] = engine_id
        body = json.dumps(body_json).encode()

    # Thinking policy (spec 2026-10-25 §17.3): a token or model set to "off"
    # forces chat_template_kwargs.enable_thinking=false. An alias's policy takes
    # the model's place (D4c); the key still wins. Runs on the final (already
    # translated, for /v1/messages and /v1/responses) chat body. "client"
    # forwards the bytes untouched. Fail open: any error forwards the body as
    # it was.
    alias = getattr(request.state, "lmw_alias", None)
    thinking_forced_off = False
    try:
        policy = effective_policy(
            getattr(token, "thinking_policy", None),
            getattr(model, "thinking_policy", None),
            getattr(alias, "thinking_policy", None),
        )
        if policy == "off":
            if apply_thinking_off(body_json):
                body = json.dumps(body_json).encode()
            # Marked for every chat body under an effective "off", even when the
            # body already carried enable_thinking=false (translated
            # /v1/messages and /v1/responses do): the flag means "policy forced
            # it", so the post-deploy check reads 1 on every turn.
            if isinstance(body_json.get("messages"), list):
                thinking_forced_off = True
        elif policy == "on":
            if apply_thinking_on(body_json):
                body = json.dumps(body_json).encode()
    except Exception:
        logger.debug("thinking policy: could not apply, forwarding unchanged", exc_info=True)

    # Alias parameter overrides (D4c): applied on the final chat/completions body
    # right after the thinking policy, inside the same fail-open try (an error
    # forwards the body unchanged and logs at debug).
    try:
        if alias is not None and alias.params:
            if apply_alias_params(body_json, alias.params):
                body = json.dumps(body_json).encode()
    except Exception:
        logger.debug("alias params: could not apply, forwarding unchanged", exc_info=True)

    # Prompt diet (2026-10-28): the key's opt-in cut of unused tools and carried
    # reasoning, on the final chat body. Lineage and the recorded shape were
    # computed on the client's body above. Fail open: the body as it was.
    diet_cut: str | None = None
    try:
        diet = parse_diet(getattr(token, "prompt_diet", None))
        if diet is not None:
            trial = json.loads(body)
            cut = apply_diet(trial, diet)
            if cut is not None:
                body_json = trial
                body = json.dumps(body_json).encode()
                diet_cut = cut.json()
    except Exception:
        logger.debug("prompt diet: could not apply, forwarding unchanged", exc_info=True)

    # Debug capture (app/proxy/capture.py): only while the key's window is open.
    cap_store = getattr(request.app.state, "capture_store", None)
    cap_on = False
    try:
        cap_on = cap_store is not None and capturing(
            getattr(token, "capture_until", None), time.time()
        )
    except Exception:  # noqa: BLE001
        cap_on = False
    cap_max = int(getattr(request.app.state.settings, "capture_resp_max_bytes", 8 * 1024**2))
    cap_resp = bytearray()
    cap_done = False

    def _cap_feed(chunk: bytes) -> None:
        if cap_on and len(cap_resp) < cap_max:
            cap_resp.extend(chunk[: cap_max - len(cap_resp)])

    def _capture_engine_body() -> str | None:
        # What reached the engine, minus the per-key secrets: the client's own cache_salt
        # and no lmw_tenant / lmw_scope values (app/proxy/kv_scope.py).
        if body == client_body:
            return None
        if not kv_redact:
            return body.decode("utf-8", "replace")
        view = _redacted_engine_body(body, kv_capture_salt)
        return None if view == client_body else view.decode("utf-8", "replace")

    async def _capture_done(content: bytes | None = None) -> None:
        nonlocal cap_done
        if not cap_on or cap_done or cap_store is None:
            return
        cap_done = True
        try:
            resp_b = bytes(content[:cap_max]) if content is not None else bytes(cap_resp)
            if kv_secrets:
                resp_b = _scrub_secrets(resp_b, tuple(kv_secrets))
            await cap_store.put(
                {
                    "id": live_req.id if live_req is not None else secrets.token_hex(16),
                    "ts": time.time(),
                    "token_id": getattr(token, "id", None),
                    "model_id": model.id,
                    # ``path`` / ``request_body`` are what the client called and sent;
                    # ``translated_body`` is the OpenAI chat body the translator made from it
                    # (None when the route did not translate); ``engine_body`` is what reached
                    # the engine when that differs. Records written before 2026-11-21 hold the
                    # translated body in ``request_body`` and have no ``translated_body``.
                    "path": wire_path or path,
                    "request_body": (
                        None
                        if wire_sanitized
                        else (wire_body if wire_body is not None else client_body).decode(
                            "utf-8", "replace"
                        )
                    ),
                    "translated_body": (
                        client_body.decode("utf-8", "replace")
                        if wire_body is not None or wire_sanitized
                        else None
                    ),
                    "engine_body": _capture_engine_body(),
                    "response": resp_b.decode("utf-8", "replace"),
                    "response_truncated": (len(content) if content is not None else len(cap_resp))
                    >= cap_max,
                    "prompt_diet": diet_cut,
                }
            )
        except Exception:  # noqa: BLE001 — capture must never fail a request
            logger.debug("capture: failed", exc_info=True)

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
                _msg, request.app.state.settings.cookie_secret, args=lens_l3
            )
        except Exception:  # noqa: BLE001
            logger.debug("forest: tools_out failed", exc_info=True)
        if lens_l3:
            # Layer 3 (spec 8.1, long thinking): the engine's reasoning count.
            try:
                live_req.reasoning_tokens = _reasoning_tokens_of(resp_body)
            except Exception as exc:  # noqa: BLE001 -- the lens never fails a request
                _gm_log.warning("lens: reasoning tokens failed: %s", type(exc).__name__)

    reply_cap = (
        ReplyCapture() if is_chat_path and isinstance(body_json.get("messages"), list) else None
    )

    def _set_cobs2_reply() -> None:
        # Fail-open: hashing the generated reply must never affect the response.
        if reply_cap is None or live_req is None or live_req.cobs2_reply_msg is not None:
            return
        # A truncated reply (reaped, disconnected, upstream error, engine error
        # frame) must not get an id: it was not the message the client keeps.
        if live_req.orphan or live_req.finish_reason == "error" or not reply_cap.complete:
            return
        try:
            key = getattr(request.app.state, "cobs2_key", None)
            msg = reply_cap.message()
            if key is None or msg is None:
                return
            canon = canonicalize({"messages": [msg]})
            if canon is None:
                return
            ids = ids_for(canon, key, scope_salt(body_json))
            if ids.msgs:
                live_req.cobs2_reply_msg = ids.msgs[0]
                live_req.cobs2_reply_loose_msg = ids.loose_msgs[0]
        except Exception:  # noqa: BLE001
            logger.debug("cobs2: reply ids failed", exc_info=True)
            return
        # Link the reply to the request's last node, in the request's session,
        # with its loose text sketched in memory (never stored). Fail-open.
        try:
            seen = cobs2_seen
            if (
                ids.msgs
                and seen is not None
                and seen.obs is not None
                and seen.ids.nodes
                and cobs2_lineage is not None
            ):
                cobs2_lineage.remember_reply(
                    model_id=model.id,
                    scope=seen.salt,
                    session=seen.obs.session,
                    node=seen.ids.nodes[-1],
                    reply_ids=ids,
                    reply_text_loose=_cobs2_text(canon.messages[0], _COBS2_REPLY_SKETCH_CHARS),
                )
        except Exception:  # noqa: BLE001
            logger.debug("cobs2: remember reply failed", exc_info=True)

    def _set_lens_reply() -> None:
        # Prompt Lens layer 3 (spec 8.1): the reply's counts and a verdict per
        # tool call, on the message ReplyCapture assembled once the response
        # is complete (the proxied bytes are already out or fixed). FAIL-OPEN,
        # each field on its own; codes and counts only, the type in the log.
        if not lens_l3 or reply_cap is None or live_req is None:
            return
        if live_req.orphan or live_req.finish_reason == "error" or not reply_cap.complete:
            return
        msg = reply_cap.message()
        if msg is None:
            return
        try:
            live_req.reply_chars, live_req.calls_n = reply_verdicts.reply_counts(msg)
        except Exception as exc:  # noqa: BLE001 -- the lens never fails a request
            _gm_log.warning("lens: reply counts failed: %s", type(exc).__name__)
        try:
            verdicts = reply_verdicts.call_verdicts(
                msg,
                reply_verdicts.offered_of(body_json.get("tools")),
                request.app.state.settings.cookie_secret,
            )
            live_req.calls_bad = verdicts  # [k, hash8, code], the non-ok calls only
        except Exception as exc:  # noqa: BLE001
            _gm_log.warning("lens: call verdicts failed: %s", type(exc).__name__)

    async def _deregister() -> None:
        _set_cobs2_reply()
        _set_lens_reply()
        _release_dp_rank()
        if live_req is not None:
            live_req.cached_tokens = _acct_cached()
            if g_out is not None:
                live_req.guard = guard_proxy.with_restored(g_out.history_json, g_restored[0])
                if g_out.restore.restored:
                    _guard_log(request, g_out.restored_events(now=time.time()), request_id)
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
            if (
                dp_state is not None
                and dp_key is not None
                and dp_rank is not None
                and dp_decision != "client_pinned"
                and upstream_ok
                and prefix_processed
                and not prefix_failed
            ):
                dp_state.placement.note_prompt(
                    model.id, place_epoch, dp_key, engine_prompt or prompt_tokens
                )
        except Exception:  # noqa: BLE001 — bookkeeping must never fail a request
            logger.debug("dp: could not note prompt size", exc_info=True)
        try:
            if cache_index is not None and obs_chain is not None:
                served = upstream_ok and prefix_processed and not prefix_failed
                dp_n = getattr(model, "data_parallel_size", 1) or 1
                rank_known = dp_n <= 1 or dp_rank is not None
                # An unrouted data-parallel request is recorded under rank
                # None; record.py leaves that bit out of a routed request's
                # R_fleet, so it can never make a routed miss read "misrouted".
                if served:
                    cache_index.observe(
                        model.id,
                        obs_epoch,
                        dp_rank,
                        obs_chain,
                        token.id,
                        operator_lens=not kv_ident.isolated,
                    )
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
                # #301: the engine's KV block size for this run, a dict read
                # (learned off-path by app/cache_obs/block_size.py).
                sizes = getattr(request.app.state, "engine_block_sizes", None)
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
                    block_size=sizes.get(model.id, obs_epoch) if sizes is not None else None,
                    drops_last_block=(
                        sizes.drops_last_block(model.id, obs_epoch) if sizes is not None else False
                    ),
                    match_unit=sizes.match_unit(model.id, obs_epoch) if sizes is not None else None,
                    cobs2_edge=live_req.cobs2_edge,
                    loose_match_beyond_exact=live_req.cobs2_loose_beyond_exact,
                )
        except Exception:  # noqa: BLE001 — bookkeeping must never fail a request
            logger.debug("cache-obs: observe failed", exc_info=True)
        # Prompt Lens layer 1: the row's verdict, with this request's own
        # outcome and reuse. FAIL-OPEN: a fault leaves the lens fields None.
        try:
            if lens_state is not None and live_req is not None:
                served = upstream_ok and prefix_processed and not prefix_failed
                co = live_req.cache_obs or {}
                lens_hot.apply(
                    live_req,
                    lens_state.final(
                        prompt_tokens=_acct_prompt(),
                        own_outcome=co.get("cache_outcome_own") if served else None,
                        actual_tokens=(
                            lens_hot.own_reuse(_acct_cached(), co.get("reusable_tokens_own"))
                            if served
                            else None
                        ),
                        gap_s=live_req.gap_s,
                    ),
                )
        except Exception as exc:  # noqa: BLE001 -- the lens never fails a request
            _gm_log.warning("lens: finish failed: %s", type(exc).__name__)
        # Layer 2/3 fields the kvsig pass prepared: the repeated tool results
        # (at this request's tokens per char) and the toolset's lint. FAIL-OPEN.
        try:
            if lens_state is not None and live_req is not None:
                live_req.lens_repeat_results = lens_state.repeat_tokens(_acct_prompt())
                if lens_state.tools_lint is not None:
                    live_req.tools_lint = list(lens_state.tools_lint)
                if live_req.lens_version is None:  # a header's name, set at admission, wins
                    live_req.lens_version = lens_state.version
        except Exception as exc:  # noqa: BLE001
            _gm_log.warning("lens: repeat results failed: %s", type(exc).__name__)
        try:
            history = getattr(request.app.state, "request_history", None)
            if history is not None:
                # With a KV tier the row waits (in a background task, never this one) for
                # the hits the connector recorded under the request id.
                record_finished(
                    history,
                    live_req,
                    kv_client(request.app.state, model),
                    now=_monotonic(),
                    iso_star_frac=iso_star_frac,
                )
        except Exception:  # noqa: BLE001 — bookkeeping must never fail a request
            logger.debug("stats: could not record finished request", exc_info=True)
        try:
            await registry.deregister(live_req.id)
        except Exception:
            pass

    try:
        # cobs2: the lineage step on the loop, over the ids hashed right after
        # admission. FAIL-OPEN: any error leaves every cobs2 field None.
        try:
            if cobs2_pre is not None:
                cobs2_seen = _cobs2_link(
                    cobs2_pre,
                    lineage=cobs2_lineage,
                    model_id=model.id,
                    session_key=forest_kw.get("session_key"),
                    parent_session_key=forest_kw.get("parent_session_key"),
                    token_scope=forest_fields.hkey(
                        request.app.state.settings.cookie_secret, token.id
                    ),
                    inflight_heads=[
                        o.cobs2_head
                        for o in (registry.snapshot() if registry is not None else ())
                        if o is not live_req
                        and o.token_id == token.id
                        and o.model_row_id == model.id
                        and o.cobs2_head
                    ],
                    request_id=request_id,
                    lens_heads=getattr(request.app.state, "lens_heads", None),
                    lens_parents=getattr(request.app.state, "lens_parents", None),
                )
                c_ids, c_obs = cobs2_seen.ids, cobs2_seen.obs
                cobs2_kw = {
                    "cobs2_head": c_ids.head,
                    "cobs2_node": c_ids.nodes[-1] if c_ids.nodes else None,
                    "cobs2_parent_node": c_obs.parent if c_obs is not None else None,
                    "cobs2_edge": c_obs.edge if c_obs is not None else None,
                    "cobs2_compaction_request": cobs2_seen.compaction,
                    "cobs2_loose_beyond_exact": bool(
                        c_obs is not None and c_obs.loose_match_beyond_exact
                    ),
                    # The lane's conv: inherited from the parent when this
                    # continues one (a harness may rewrite the first message).
                    "cobs2_conv": (
                        c_obs.conv if c_obs is not None and c_obs.conv else cobs2_pre.conv
                    ),
                    "cobs2_toolset": cobs2_pre.toolset,
                    "cobs2_parent_req": c_obs.parent_req if c_obs is not None else None,
                }
                if c_obs is not None:
                    cobs2_kw["cobs2_session"] = c_obs.session
                if cobs2_pre.shape is not None:
                    try:
                        cobs2_kw["cobs2_shape"] = shape_json(cobs2_pre.shape, break_index(c_obs))
                    except Exception:  # noqa: BLE001 -- one field, never the row
                        logger.debug("cobs2: shape json failed", exc_info=True)
            # R7: no cobs2 lineage (over the cap, an error, a non-chat path):
            # the row still joins its harness session, as a lineage-unknown turn.
            if not cobs2_kw.get("cobs2_session") and forest_kw.get("session_key"):
                cobs2_kw["cobs2_session"] = forest_kw["session_key"]
            if cobs2_kw.get("cobs2_session"):
                cobs2_kw["cobs2_sess_ref"] = sess_ref(
                    request.app.state.settings.cookie_secret, token.id, cobs2_kw["cobs2_session"]
                )
            if cobs2_kw:
                if live_req is not None:
                    for _k, _v in cobs2_kw.items():
                        setattr(live_req, _k, _v)
        except Exception:  # noqa: BLE001 — identity must never fail a request
            cobs2_seen = None
            cobs2_kw = {}
            logger.debug("cobs2: ids or lineage failed", exc_info=True)
        # Prompt Lens layer 1: the structural verdict at admission (the kvsig
        # event's lens block). FAIL-OPEN, and the type only in the log.
        try:
            lens_state = cobs2_seen.lens if cobs2_seen is not None else None
            if lens_state is not None:
                lens_state.admit(prompt_tokens)
        except Exception as exc:  # noqa: BLE001 -- the lens never fails a request
            lens_state = None
            _gm_log.warning("lens: admission failed: %s", type(exc).__name__)
        if gm_on:
            # G3: the prompt map, once the request is admitted (a queued
            # request shows none yet). Ids and sizes only: the content passes
            # ran in the cobs2 compute; this one walks ids. Fail open.
            try:
                _gm_publish(
                    hub,
                    {
                        "type": "kvsig",
                        "req_id": req_id,
                        "token_id": token.id,
                        "ts": time.time(),
                        "session_id": session_id,
                        "turn": forest_kw.get("turn_index"),
                        # G-R14: the admission wait, so the dock shows "queued"
                        # from here on (request_end repeats it). Own timing: a
                        # key holder may see it.
                        "queued_s": _s3(queued_s),
                        **_gm_kvsig(
                            body_json,
                            cobs2_key,
                            cobs2_pre,
                            cobs2_seen,
                            gm_kv_reason,
                            chain=obs_chain,
                            match_all=obs_all,
                            match_own=obs_own,
                            prompt_tokens=prompt_tokens,
                            lens=lens_state.adm if lens_state is not None else None,
                        ),
                    },
                )
            except Exception as exc:  # noqa: BLE001 -- display only
                _gm_log.warning("prompt_lens kvsig failed: %s", type(exc).__name__)
            if cobs2_pre is not None:
                cobs2_pre.gm = None  # published: the request holds no layout
        if live_req is not None:
            # Admitted: the clock for elapsed / TTFT starts now.
            live_req.started_monotonic = _monotonic()
            # Same instant as started_monotonic, so history's started_iso and
            # duration_s agree; the wait is queued_s, separately.
            live_req.started_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            live_req.queued_s = queued_s
            live_req.thinking_forced_off = thinking_forced_off
            live_req.prompt_diet = diet_cut
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
            # "prefill" would be a claim the warden cannot back. "waiting"
            # lasts until the body is forwarded (then ``running_no_stream``).
            live_req.phase = "prefill" if client_wants_stream else "waiting"
        elif registry is not None:
            live_req = LiveRequest(
                id=request_id,
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
                thinking_forced_off=thinking_forced_off,
                prompt_diet=diet_cut,
                **cobs2_kw,
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
    if gm is not None:
        gm.live_req = live_req

    if live_req is not None and salt_rewrite:
        live_req.kv_isolated = kv_ident.isolated

    try:
        # KV-cache tiers: LM Warden's own kv_transfer_params, for a model with kv_store on
        # only, and the engine's cache_salt (per-key isolation, every vLLM model). Set here,
        # after the cobs2 ids, the lineage link and the cache-obs chain were computed over
        # the client's body, so none of them can change; lmw_req is the request id the
        # connector records its hits under. Inside this try: any raise releases the slot
        # and the live row. One re-serialisation, only when something changed.
        kv_changed = False
        if kv_enabled(model):
            try:
                set_kv_params(
                    body_json,
                    req_id=request_id,
                    conv=cobs2_kw.get("cobs2_conv"),
                    session=cobs2_kw.get("cobs2_session"),
                    tenant=kv_ident.tenant.hex(),
                    scope=kv_ident.scope.hex(),
                )
                kv_changed = True
            except Exception:  # noqa: BLE001 — the tier is an optimisation
                logger.debug("kv: could not set kv_transfer_params", exc_info=True)
        if salt_rewrite:
            kv_client_salt = (
                body_json.get(CACHE_SALT, _NO_SALT) if isinstance(body_json, dict) else _KEEP_SALT
            )
            if _apply_cache_salt(body_json, kv_ident, kv_nk):
                kv_changed = True
                kv_capture_salt = kv_client_salt
                kv_secrets.extend(_salt_secrets(body_json.get(CACHE_SALT)))
            elif (isinstance(body_json, dict) and CACHE_SALT in body_json) or (
                b"cache_salt" in body
            ):
                # Forward exactly what the proxy parsed: a raw body with a duplicate key
                # could make an engine that keeps the first value read another salt.
                kv_changed = True
        if kv_changed:
            body = json.dumps(body_json).encode()
            kv_redact = True
            if kv_enabled(model):
                kv_secrets.extend(h.encode() for h in (kv_ident.tenant.hex(), kv_ident.scope.hex()))
            kv_secrets[:] = [x for x in kv_secrets if x]
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
        # How long to wait for the engine's response to start (verified finding #4): a stream's
        # headers come at once, so the idle limit applies; a non-stream answer starts only when
        # generation ends, so the whole request deadline applies.
        _idle = float(getattr(request.app.state.settings, "upstream_idle_s", 300.0) or 300.0)
        _wall = float(getattr(request.app.state.settings, "request_max_wall_s", 0.0) or 0.0)
        _nonstream = _wall or float(
            getattr(request.app.state.settings, "nonstream_max_s", 1800.0) or 1800.0
        )
        send_deadline = _idle if client_wants_stream else _nonstream
        idle_s = _idle
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
        # A non-streaming reply's headers arrive only when generation is done,
        # so from here on the engine holds the request: the row is
        # ``running_no_stream``, not ``waiting``. vLLM exports no per-request
        # series, so whether its own scheduler has admitted it yet cannot be
        # told per request; "forwarded" is the honest boundary.
        if live_req is not None and not client_wants_stream:
            live_req.phase = "running_no_stream"
        send_start = _monotonic()
        try:
            resp = await asyncio.wait_for(client.send(upstream, stream=True), timeout=send_deadline)
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
            resp = await asyncio.wait_for(client.send(upstream, stream=True), timeout=send_deadline)
        upstream_ok = 200 <= resp.status_code < 300
        if kv_secrets and resp.status_code >= 400:
            # An error body is small; read it whole so a secret split across two
            # chunks is still found. Bounded like the response start.
            raw_err = await asyncio.wait_for(resp.aread(), timeout=idle_s)
            resp = _ScrubbedResponse(resp, _scrub_secrets(raw_err, tuple(kv_secrets)))
    except TimeoutError as exc:
        # Verified finding #4: the engine accepted the request and then never answered.
        await _deregister()
        await _release_slot()
        raise HTTPException(status_code=504, detail="upstream engine stalled") from exc
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
            # G5: how the stream ended, for request_end (Prompt Lens only reads them).
            gm_exit: str | None = None  # "abort" / "error" when the body raised
            gm_client_gone = False
            gm_wall = False
            gm_reasoning_tokens: int | None = None
            # G2: per-request bound on the tool-call text Prompt Lens publishes.
            gm_tc = ToolCallBudget(settings.prompt_lens_section_prompt_chars) if gm_on else None
            # G6: per-channel frame counts, published live-only (Prompt Lens on only).
            gm_prog: _GmProgress | None = None
            if gm is not None:
                gm_prog = gm.progress = _GmProgress(gm)
            # Last non-null finish_reason seen in the SSE stream, for the content
            # log record. Only tracked when do_content_log is set.
            sse_finish_reason = None
            # Coarse completion-token estimate for the live registry: count SSE
            # content deltas (vLLM streams ~1 token/delta) rather than
            # tokenizing on the hot path. Pushed into the registry
            # at most every _LIVE_UPDATE_INTERVAL_S. The exact count is still
            # recomputed once at the end for accounting (below).
            delta_count = 0
            # B1: tokens of every generated channel, counted every live tick.
            live_out = LiveOutCounter()

            async def _live_count(text: str) -> int:
                n = await tok_cache.count(
                    model.hf_repo, text, fallback_repo=getattr(model, "tokenizer_repo", None)
                )
                return int(n)

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
                produced_s = _produced_s(live_req, _monotonic())
                await _record_counters(
                    request,
                    model,
                    token_id,
                    _acct_prompt(),
                    completion_tokens,
                    variant,
                    _acct_cached(),
                    produced_s=produced_s,
                )
                if gm is not None:
                    if enforce_trip:
                        gm_reason: str | None = "runaway"
                    elif gm_exit is not None:
                        gm_reason = gm_exit
                    elif gm_client_gone:
                        gm_reason = "abort"
                    elif (
                        resp.status_code >= 400
                        or prefix_failed
                        or gm_wall
                        # The upstream idle limit ended it (_idle_bounded marks the row).
                        or (live_req is not None and live_req.orphan)
                    ):
                        gm_reason = "error"
                    else:
                        gm_reason = gm_finish_reason
                    gm.end(
                        gm_reason,
                        prompt_tokens=_acct_prompt(),
                        completion_tokens=completion_tokens,
                        cached_tokens=_acct_cached(),
                        reasoning_tokens=gm_reasoning_tokens,
                        extra=gm_tc.end_fields() if gm_tc is not None else None,
                    )
                # Release the scheduler slot only after the body is fully
                # drained — otherwise we'd hand the slot to the next waiter
                # while still hammering vLLM with our SSE chunks.
                await _release_slot()
                await _capture_done()
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

            tools_acc = forest_fields.ToolsOutAccumulator(args=lens_l3)
            client_r = (
                guard_proxy.ChatStreamRestorer(
                    g_restore, trailer=None, restore_mode=g_out.restore_mode
                )
                if g_restore is not None and g_out is not None
                else None
            )
            try:
                async for chunk in _guard_frames(
                    _idle_bounded(resp.aiter_bytes(), idle_s, live_req), g_trailer
                ):
                    if client_r is None:
                        yield chunk
                    else:
                        out_chunk = client_r.feed(chunk)
                        g_restored[0] = client_r.restored
                        if out_chunk:
                            yield out_chunk
                    buf += chunk
                    _cap_feed(chunk)

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
                        gm_wall = True
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
                                gm_client_gone = True
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
                        reply_parsed = False
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
                            if reply_cap is not None:
                                reply_parsed = True
                                if ev is not None:
                                    reply_cap.feed_event(ev, len(line))
                            if ev is not None:
                                if not first_meta:
                                    for _k in ("id", "model", "created"):
                                        if ev.get(_k) is not None:
                                            first_meta[_k] = ev[_k]
                                _feed_detector(detector, synth, ev, is_chat_path)
                                if detector.tripped and enforce:
                                    enforce_trip = True
                                    break
                        if reply_cap is not None and not reply_parsed:
                            reply_cap.feed_sse(line)
                        # Legacy /v1/completions frames carry choices[0].text,
                        # not delta.content (R13): parse that channel instead
                        # so the streamed text reaches accumulated (content
                        # log) and the end-of-stream count like chat content.
                        delta = _parse_sse_delta(line) if is_chat_path else _parse_sse_text(line)
                        if delta:
                            accumulated += delta
                            completion_text += delta
                            delta_count += 1
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
                        if lens_l3 and live_req is not None and b'"reasoning_tokens"' in line:
                            # Layer 3: the engine's reasoning count, on its usage
                            # frame (one parse a stream, behind the byte gate).
                            try:
                                _rt = _reasoning_tokens_of(parse_sse_event(line))
                                if _rt is not None:
                                    live_req.reasoning_tokens = _rt
                            except Exception as exc:  # noqa: BLE001
                                _gm_log.warning(
                                    "lens: reasoning tokens failed: %s", type(exc).__name__
                                )
                        if delta or extra or (frame_usage or 0) > 0:
                            prefix_processed = True
                        if live_req is not None and cached is not None:
                            live_req.cached_tokens = cached
                        if stream_usage_completion is None:
                            completion_text += extra
                        if live_req is not None:
                            # B1: the live row counts TOKENS of every channel
                            # (content, reasoning, tool calls), not content
                            # frames; tokenized once per live tick, new text only.
                            try:
                                live_out.feed((delta or "") + extra, frame_usage)
                                if (
                                    live_out.dirty or frame_usage is not None
                                ) and now - last_live_update >= _LIVE_UPDATE_INTERVAL_S:
                                    n_out = await live_out.count(_live_count)
                                    live_req.completion_tokens = n_out
                                    live_req.out_samples.append((now, n_out))
                                    last_live_update = now
                            except Exception:  # noqa: BLE001 -- display only
                                pass
                        # Prompt Lens tap: emit content, reasoning and tool-call
                        # channels and track the last non-null finish_reason.
                        # Separate parse so the counter path (accumulated) is
                        # untouched. Runs every line (independent of `delta`)
                        # so reasoning-only and tool-call-only frames surface.
                        if gm_on:
                            gm_content, gm_reasoning, gm_fr, gm_calls = _parse_sse_prompt_lens(line)
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
                            if gm_calls and gm_tc is not None:
                                _gm_publish_tool_calls(hub, req_id, token_id, gm_calls, gm_tc)
                            if gm_prog is not None:
                                gm_prog.count(gm_content, gm_reasoning, gm_calls, now)
                            if gm_fr is not None:
                                gm_finish_reason = gm_fr
                            # G5: the engine's reasoning count, on its usage frame.
                            if b'"reasoning_tokens"' in line:
                                gm_rt = _reasoning_tokens_of(parse_sse_event(line))
                                if gm_rt is not None:
                                    gm_reasoning_tokens = gm_rt

                    if enforce_trip:
                        # Detector tripped in enforce mode: stop forwarding
                        # upstream and let the terminal chunk below close the
                        # stream. The finally tears down the upstream socket so
                        # vLLM aborts and frees its KV blocks.
                        break
                if client_r is not None:
                    tail = client_r.flush()
                    g_restored[0] = client_r.restored
                    if tail:
                        yield tail
                # Enforce trip: append a schema-valid terminal chunk carrying the
                # runaway finish_reason + [DONE] so the client sees a clean end.
                if enforce_trip and detector is not None:
                    yield _runaway_terminal_chunk(is_chat_path, first_meta, detector.finish_reason)
            except BaseException as exc:
                if gm is not None:
                    # Cancelled, or the client closed the body: abort. Anything else: error.
                    gm_exit = (
                        "abort"
                        if isinstance(exc, asyncio.CancelledError | GeneratorExit)
                        else "error"
                    )
                raise
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
                if gm is None:
                    await asyncio.shield(_settle())
                else:
                    # G-F1: _settle publishes the end. When it cannot (it raised) the
                    # settle task's done-callback does, so a cancellation that re-lands
                    # on this await (a real client disconnect) never pre-empts the
                    # settled figures: the end waits for the task, not for this frame.
                    settle_task = asyncio.ensure_future(_settle())
                    settle_task.add_done_callback(functools.partial(_gm_after_settle, gm))
                    await asyncio.shield(settle_task)

        # A never-started async generator skips its ``finally`` when closed or
        # discarded, so a client that vanishes before the first chunk would
        # leak the admission slot and the replica in-flight count. The guard
        # releases both exactly once in that case; once ``gen`` has started,
        # its own ``_settle`` owns the cleanup and the guard stays out of it.
        on_deregister = _deregister
        if gm is not None:
            gm_unstarted = gm

            async def _deregister_unstarted() -> None:
                try:
                    await _deregister()
                finally:
                    gm_unstarted.fallback("error" if resp.status_code >= 400 else "abort")

            on_deregister = _deregister_unstarted
        guard = _UnstartedStreamGuard(
            gen(),
            _abandon_unstarted_stream(on_deregister, resp, _release_slot),
        )
        streaming = StreamingResponse(
            guard,
            background=BackgroundTask(guard.abandon),
            status_code=resp.status_code,
            headers={
                "content-type": resp.headers.get("content-type", "text/event-stream"),
                REQUEST_ID_HEADER: request_id,
            },
        )
        if gm is not None:
            # From here the stream owns the end: gen() or the unstarted guard.
            gm.handed_off = True
        return streaming

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
            async for chunk in _guard_frames(
                _idle_bounded(resp.aiter_bytes(), idle_s, live_req), g_trailer
            ):
                buf += chunk
                _cap_feed(chunk)
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
            if reply_cap is not None:
                reply_cap.set_json(out)
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
                produced_s=_produced_s(live_req, _monotonic()),
            )
        finally:
            await _deregister()
            await resp.aclose()
            await _release_slot()
            await _capture_done()
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
        if gm is not None:
            # This path publishes no deltas (it never did); the request still ends.
            # Guarded: never 500 a request that generated fine.
            try:
                choices0 = (out.get("choices") or [{}])[0]
                if tripped_enforce:
                    gm_agg_finish: str | None = "runaway"
                elif resp.status_code >= 400:
                    gm_agg_finish = "error"
                else:
                    gm_agg_finish = (
                        choices0.get("finish_reason") if isinstance(choices0, dict) else None
                    )
                gm.end(
                    gm_agg_finish,
                    prompt_tokens=_acct_prompt(),
                    completion_tokens=completion_tokens,
                    cached_tokens=_acct_cached(),
                    reasoning_tokens=_reasoning_tokens_of(out),
                )
            except Exception:  # noqa: BLE001 -- display only
                _gm_log.warning("prompt_lens re-aggregated end failed", exc_info=True)
                gm.fallback(None)
        if g_restore is not None and g_out is not None:
            raw_out, g_restored[0] = guard_proxy.restore_json(
                json.dumps(out).encode(), g_restore, trailer=None, restore_mode=g_out.restore_mode
            )
            out = json.loads(raw_out)
            # The history row was written before this restore; the action log still records it.
            if g_restore.restored:
                _guard_log(request, g_out.restored_events(now=time.time()), request_id)
        return JSONResponse(
            out, status_code=resp.status_code, headers={REQUEST_ID_HEADER: request_id}
        )

    try:
        # Server-side wall-clock backstop for the non-stream path too (the
        # config contract is "streaming or not"). httpx runs with timeout=None,
        # so a hung upstream that keeps the socket transport-alive would pin the
        # slot forever. When request_max_wall_s > 0, bound the read by the time
        # already spent since request_start; on expiry asyncio.TimeoutError
        # unwinds into the `finally` below, which aclose()s the upstream socket
        # (vLLM aborts + frees KV) and releases the slot. 0.0 = disabled.
        max_wall = getattr(request.app.state.settings, "request_max_wall_s", 0.0) or 0.0
        max_wall = max_wall or float(
            getattr(request.app.state.settings, "nonstream_max_s", 1800.0) or 1800.0
        )
        remaining = max_wall - (_monotonic() - request_start)
        content = await asyncio.wait_for(resp.aread(), timeout=max(remaining, 0.0))
        if g_trailer and resp.status_code < 300:
            content, _ = guard_proxy.restore_json(content, RestoreMap(), trailer=g_trailer)
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
            if reply_cap is not None:
                reply_cap.set_json(body_obj)
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
            produced_s=_produced_s(live_req, _monotonic()),
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

    # Prompt Lens tap (non-stream): no upstream streaming is forced here, so the
    # client sees ONE synthetic delta per channel on completion plus a
    # request_end — a documented limitation (non-stream clients appear as one
    # block, not live token-by-token).
    if gm is not None:
        gm_ns_finish: str | None = None
        gm_ns_reasoning: int | None = None
        nonstream_tc = ToolCallBudget(settings.prompt_lens_section_prompt_chars)
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
            # G2: one tool_call delta per message.tool_calls entry, whole
            # arguments, index = its position in the list.
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, list) and raw_calls:
                _gm_publish_tool_calls(
                    hub,
                    req_id,
                    token_id,
                    [
                        d._replace(index=pos)
                        for pos, c in enumerate(raw_calls)
                        if (d := tool_call_delta(c, pos)) is not None
                    ],
                    nonstream_tc,
                )
            gm_ns_finish = choice0.get("finish_reason")
            gm_ns_reasoning = _reasoning_tokens_of(parsed)
        except Exception:
            _gm_log.warning("prompt_lens non-stream tap failed", exc_info=True)
        gm.end(
            "error" if resp.status_code >= 400 else gm_ns_finish,
            prompt_tokens=_acct_prompt(),
            completion_tokens=completion_tokens,
            cached_tokens=_acct_cached(),
            reasoning_tokens=gm_ns_reasoning,
            extra=nonstream_tc.end_fields(),
        )

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

    await _capture_done(content)
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

    if g_restore is not None and g_out is not None and resp.status_code < 300:
        content, g_restored[0] = guard_proxy.restore_json(
            content, g_restore, trailer=None, restore_mode=g_out.restore_mode
        )
        # The history row was written before this restore; the action log still records it.
        if g_restore.restored:
            _guard_log(request, g_out.restored_events(now=time.time()), request_id)
    return Response(
        content=content,
        status_code=resp.status_code,
        headers={
            "content-type": resp.headers.get("content-type", "application/json"),
            REQUEST_ID_HEADER: request_id,
        },
    )


@router.post("/chat/completions")
async def chat_completions(request: Request, token: TokenRow = Depends(require_bearer)):
    body_bytes = await request.body()
    body_json = json.loads(body_bytes) if body_bytes else {}
    served_name = body_json.get("model")
    if not served_name:
        raise HTTPException(400, "missing 'model' field")
    # #324: the engine fetches any remote media URL it is handed, so refuse
    # them unless the operator allow-listed the host.
    media_why = remote_media.refusal(body_json)
    if media_why:
        return JSONResponse(
            {
                "error": {
                    "message": media_why,
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "remote_media_not_allowed",
                }
            },
            status_code=400,
        )
    model, host, port, variant = await _resolve_for_token(request, token, served_name)

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
    model, host, port, variant = await _resolve_for_token(request, token, served_name)

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
    aliases = await all_aliases(cache, settings.db_path)
    # D4e: an enabled alias shadows a model with the same served name; that
    # model is hidden and the alias is listed instead.
    alias_names = {a.name for a in aliases if a.enabled}
    loaded = [
        r
        for r in rows
        if r.status == "loaded"
        and r.served_model_name not in alias_names
        and token_allows(token, r.served_model_name)
    ]
    # D4e: each enabled alias with a loaded target the key may use (by the
    # alias name or the serving model's served name) is listed under the alias
    # name with the SAME fields as that model's own entry: clients size their
    # context from max_model_len.
    alias_targets = []
    for alias in aliases:
        if not alias.enabled:
            continue
        target = first_loaded_target(alias, rows)
        if target is None:
            continue
        if not (token_allows(token, alias.name) or token_allows(token, target.served_model_name)):
            continue
        alias_targets.append((alias, target))
    listed = list(loaded)
    seen = {r.id for r in loaded}
    for _alias, target in alias_targets:
        if target.id not in seen:
            seen.add(target.id)
            listed.append(target)
    blocks: dict[str, dict[str, Any]] = {}
    # The stress records still need the database, but only when something is
    # listed: a token that sees no loaded model costs no connection at all.
    if listed:
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
            blocks = await warden_blocks_for(request.app.state, db, listed, all_models=rows)
    sup = getattr(request.app.state, "supervisor", None)
    isolated = is_isolated(token)
    entries: dict[str, dict[str, Any]] = {}
    for r in listed:
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
        if isolated:
            # O4: which models this isolated key can use. A sharing key's list is unchanged.
            entry["lmw_cache_isolation"] = (
                "isolated" if can_isolate_running(request.app.state, r) else "unsupported"
            )
        entries[r.id] = entry
    data = [entries[r.id] for r in loaded]
    for alias, target in alias_targets:
        data.append({**entries[target.id], "id": alias.name})
    return {"object": "list", "data": data}
