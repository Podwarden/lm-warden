"""The streaming turn endpoints — `POST /api/chat2/chats/{id}/turns` plus the
detached-turn surface (`GET .../turn/live`, `POST .../turn/abort`) (spec §4.2).

The turn is DETACHED from its HTTP response: after the pre-flight checks and
the user/tool-row persist, the whole generation pipeline — the loopback call
to the warden's own OpenAI-compatible proxy (so the real priority /
usage-rollup pipeline runs unmodified), the `ChatEvent`
normalisation, the assistant-row/ledger/title persistence — runs on its own
`asyncio.Task` (`_run_turn`, the *runner*), which appends every SSE frame to
an `app.chat2.live.LiveTurn` entry. The HTTP response is merely a replayable
*subscriber* over those frames. Consequences:

* a client disconnect cancels only the subscriber generator — the runner
  streams on to completion and persists the full assistant row
  (finish_reason stop/length/...), its usage, the ledger row and the titles;
* navigation/reload can re-attach: `GET /chats/{id}` exposes `live_turn`, and
  `GET /chats/{id}/turn/live` replays the frames byte-identically and then
  follows the live tail;
* stopping a turn is now an explicit `POST /chats/{id}/turn/abort`, which
  cancels the runner; the runner's CancelledError path persists the partial
  tail as `aborted` (spec §4.2 step 8) and emits a terminal
  `done {finishReason: "aborted"}` frame for any remaining subscribers.

Ordering guarantees inside the runner (spec §4.2 steps 7-9), unchanged from
the pre-detached implementation:

* the upstream response and client are closed **first** in the `finally`, so
  the proxy releases its scheduler slot exactly as the playground does today;
* the active-request counter is exited next (the Playwright leak test polls it);
* persistence is spawned as its own task and awaited through
  `asyncio.shield`, so cancelling the runner (explicit abort, app shutdown)
  still lands an `aborted` assistant row and its ledger entry;
* the lock is released from that task's completion callback, so "released
  last" holds on the abort path too. The lock is acquired before the runner
  is spawned and released only by it — a second POST while the runner lives
  is a 409 `turn_in_flight` regardless of any client's connection state.

The runner is a plain task, so cancellation is single-shot — but app shutdown
(`asyncio.wait_for` in the lifespan) can cancel a second time, and the anyio
quirk of re-delivered cancellation still applies wherever a scope is
involved, so every teardown await stays wrapped in `_close_safely` /
`asyncio.shield` exactly as before.

There is deliberately no wall-clock timeout around the upstream call here:
the loopback proxy enforces `settings.request_max_wall_s`
(`VW_REQUEST_MAX_WALL_S`) server-side and ends the stream, which the runner
sees as a normal upstream EOF.

Failure surfacing (spec §4.5): pre-flight failures (chat gone, context full,
tools unsupported, budget) are `HTTPException`s carrying the shared chat2
`{code, message}` envelope (`errors.api_error`); anything that happens once we
have started talking to the model is an `error` event **and** a persisted
assistant row carrying `error_json`, so the failure is visible in history with
a Retry affordance.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.auth.deps import stream_key
from app.auth.stream_guard import guarded
from app.chat import routes_api as playground
from app.chat.playground_store import PLAYGROUND_TOKEN_NAME, PlaygroundSecret
from app.chat2 import catalog, storage
from app.chat2.clock import now_iso
from app.chat2.context import ContextState, derive_context
from app.chat2.errors import api_error
from app.chat2.events import StreamState, frame, normalize
from app.chat2.history import PRESENT_OPTIONS_TOOL, build_messages
from app.chat2.identity import current_user_id
from app.chat2.ids import new_id
from app.chat2.ledger import LedgerRepo, Usage
from app.chat2.limits import ALLOWED_IMAGE_MIMES, MAX_IMAGES_PER_MESSAGE
from app.chat2.live import LiveTurn, LiveTurns
from app.chat2.repo import Chat2Repo
from app.chat2.schemas import TurnBody
from app.chat2.titles import (
    first_line_title,
    generate_llm_title,
    normalized_title,
    title_context,
)
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo
from app.utils.sse import sse_headers

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat2", tags=["chat2"])

# Tools the *client* fulfils (the UI renders buttons and posts `tool_results`);
# advertised only to models whose catalog row says `supports_tools` (spec §4.4).
CLIENT_TOOLS: dict[str, dict[str, Any]] = {"present_options": PRESENT_OPTIONS_TOOL}
PROVIDER = "vllm"

# Upstream HTTP status -> ChatEvent error code (spec §4.2 step 6). 409 is the
# proxy's "playground token not initialised" shape, i.e. our own bug -> internal.
_STATUS_CODES = {429: "rate_limited", 404: "model_not_loaded", 409: "internal", 504: "timeout"}
_RETRYABLE = {"rate_limited", "timeout", "upstream"}
# outcome -> finish_reason on the persisted assistant row (and on a replayed
# `done`). "ok" is absent on purpose: it defers to the upstream finish_reason,
# falling back to "stop".
_OUTCOME_FINISH = {"guard": "guard", "timeout": "timeout", "error": "error",
                   "aborted": "aborted", "interrupted": "interrupted"}

# Auto-title cadence. Turn 1 names a brand-new chat; after that the title is
# re-derived every fourth user turn, because a thread that has run for a while
# is usually no longer about its opening question and a stale sidebar entry
# makes the whole list unusable as an index. Every reassessment costs a real
# (lowest-priority, user-billed) completion, so the interval is deliberately
# coarse rather than per-turn.
TITLE_REASSESS_EVERY = 4


class _Persisted(NamedTuple):
    """What the persistence task hands back to the streaming generator."""

    context: ContextState | None
    title_source: str | None
    user_turns: int
    context_text: str


def _should_title(title_source: str | None, user_turns: int) -> bool:
    return title_source == "auto" and user_turns > 0 and (
        user_turns == 1 or user_turns % TITLE_REASSESS_EVERY == 0
    )

# asyncio only holds *weak* references to scheduled tasks, so a fire-and-forget
# task can be garbage-collected mid-flight. Keep a strong ref until it settles.
_BACKGROUND: set[asyncio.Task[Any]] = set()
_T = TypeVar("_T")


def _spawn(coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
    task = asyncio.ensure_future(coro)
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    return task


async def _close_safely(what: str, coro: Coroutine[Any, Any, Any]) -> None:
    """Run one teardown step so a re-delivered cancellation cannot skip it.

    Starlette wraps the response in an anyio task group, and an anyio cancel
    scope re-delivers `CancelledError` at *every* checkpoint while it is
    cancelled — so a bare `await resp.aclose()` inside the generator's
    `finally` is itself cancelled on the abort path, leaving the upstream
    response and the loopback client open and the proxy's slot held until GC.
    Running the close on its own task and awaiting it through `asyncio.shield`
    means the cancellation only reaches the *waiter*: the close itself always
    runs to completion.
    """
    task = _spawn(coro)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        pass  # the close continues on its own task, which is the point
    except BaseException:
        logger.exception("chat2 turn teardown failed: %s", what)


def _ledger_request_id(user_id: int, request_id: str) -> str:
    """Namespace the client-supplied `request_id` by user before it is stored.

    `usage_ledger.request_id` is declared globally UNIQUE (0025_chat2.sql), but
    the id is minted by the client and is only meaningfully unique per user.
    Without the namespace, user B replaying user A's id would either be handed
    A's recorded outcome and message id (an id leak, and B's turn silently
    dropped) or -- once the lookup is scoped -- run the turn and then lose it to
    an `IntegrityError` on the ledger insert. The composite key gives per-user
    idempotency on a globally unique column without touching the migration.
    """
    return f"{user_id}:{request_id}"


def _read_bytes(path: Path) -> bytes | None:
    """Blocking read, hopped onto a worker thread by the caller."""
    try:
        return path.read_bytes()
    except OSError:
        return None


async def _ensure_playground_secret(request: Request) -> PlaygroundSecret:
    """Mirror `POST /api/chat/playground/ensure` without the HTTP round-trip.

    The turn endpoint is JWT-authed but the loopback target (`/v1/chat/
    completions`) is bearer-gated, so we forge the `vw-playground` bearer
    server-side exactly as the playground proxy does — the browser never sees
    it. Cache hit + live DB row wins; otherwise sweep and mint a fresh row.
    """
    store = playground._get_store(request)
    cached = await store.get()
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        if cached is not None:
            existing = await repo.get(cached.token_id)
            if existing is not None and existing.name == PLAYGROUND_TOKEN_NAME:
                return cached
            await store.clear()
        await playground._delete_other_playground_rows(repo, keep_id=None)
        return await playground._mint_playground_token(request, repo)


def _estimate_usage(messages: list[dict[str, Any]], output_text: str, image_tokens: int) -> Usage:
    """chars/4 + a flat per-image constant — the spec's fallback when the
    upstream never sent a `usage` block (aborted / guarded / failed turns)."""
    chars = 0
    images = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for p in c:
                if p.get("type") == "text":
                    chars += len(p.get("text", ""))
                else:
                    images += 1
    return Usage(prompt=chars // 4 + images * image_tokens, completion=len(output_text) // 4,
                 estimated=True)


async def _once(payload: bytes) -> AsyncIterator[bytes]:
    yield payload


@router.post("/chats/{chat_id}/turns")
async def turn(
    chat_id: str, body: TurnBody, request: Request, user_id: int = Depends(current_user_id)
) -> StreamingResponse:
    locks = request.app.state.chat2_turn_locks
    if not locks.try_acquire(chat_id):
        raise api_error(409, "turn_in_flight", "another turn is in flight for this chat")
    try:
        return await _turn_locked(chat_id, body, request, user_id)
    except BaseException:
        # Only the pre-stream path lands here; once `_turn_locked` returns a
        # StreamingResponse the persist task's done-callback owns the release.
        locks.release(chat_id)
        raise


async def _turn_locked(
    chat_id: str, body: TurnBody, request: Request, user_id: int
) -> StreamingResponse:
    settings = request.app.state.settings
    locks = request.app.state.chat2_turn_locks

    async with open_db(settings.db_path) as db:
        repo, ledger = Chat2Repo(db), LedgerRepo(db)
        recorded = await ledger.find(
            _ledger_request_id(user_id, body.request_id), user_id=user_id
        )
        if recorded is not None:
            # Idempotent retry (spec §4.2 step 1): the turn already ran, so
            # replay its *recorded* outcome rather than charging the model
            # twice — a turn that ended in `guard`/`error` must not come back
            # as a clean `stop`.
            outcome, message_id = recorded
            locks.release(chat_id)
            # Guarded like every other credentialed stream, even though there is
            # nothing here to cancel (one pre-computed frame): it is the only
            # exception to "every stream the app opens under a credential runs
            # under guard_stream", and one line is cheaper than a caveat.
            return StreamingResponse(
                guarded(request, stream_key(request), _once(
                    frame({"type": "done", "messageId": message_id or "",
                           "finishReason": _OUTCOME_FINISH.get(outcome, "stop")}))),
                media_type="text/event-stream", headers=sse_headers(),
            )
        decision = await request.app.state.chat2_budget.check(user_id, None)
        if not decision.allowed:
            # A human sentence, not JSON: this message is rendered verbatim by the
            # client's error part. `blocked_until` stays available inside it.
            until = decision.blocked_until
            raise api_error(429, "budget_blocked",
                            f"budget exhausted until {until}" if until else "budget exhausted")
        chat = await repo.get_chat(user_id, chat_id)
        if chat is None:
            raise api_error(404, "not_found", "chat not found")
        if not chat.model:
            raise api_error(404, "model_not_loaded", "no model selected")
        info = await catalog.get_model_info(settings, chat.model)
        if info is None:
            raise api_error(404, "model_not_loaded", f"model '{chat.model}' is not loaded")
        msgs = await repo.list_messages(chat_id)
        ctx = derive_context(msgs, chat.settings, info.context_window)
        if ctx.full and not body.regenerate:
            raise api_error(409, "context_full", "context window is full")
        if body.tool_results and not info.supports_tools:
            raise api_error(422, "tools_unsupported", "model does not support tools")
        if not body.regenerate and not body.user_parts and not body.tool_results:
            raise api_error(422, "invalid", "nothing to send")
        if len(body.attachment_ids) > MAX_IMAGES_PER_MESSAGE:
            # The upload route caps each *file*; this is the only place a
            # message's image *count* can be capped (spec §2 limits).
            raise api_error(422, "too_many_images",
                       f"at most {MAX_IMAGES_PER_MESSAGE} images per message")

        # --- validate everything BEFORE the first append_message ------------
        # Step 4 is meant to be one atomic persist; a 409/422 discovered after
        # the user row is committed would leave an orphan row in history.
        for aid in body.attachment_ids:
            att = await repo.get_attachment(user_id, aid)
            if att is None or att.chat_id != chat_id or att.message_id is not None:
                raise api_error(422, "attachment_missing", f"attachment {aid} is gone")
        if body.tool_results:
            answered = {r.call_id for r in body.tool_results}
            for m in reversed(msgs):
                if m.role == "tool" and any(p.get("callId") in answered for p in m.parts):
                    raise api_error(409, "duplicate_tool_result", "tool call already answered")

        secret = await _ensure_playground_secret(request)

        # --- persist the incoming turn --------------------------------------
        persisted: list[dict[str, Any]] = []
        if body.regenerate:
            # Walks back over trailing tool rows to the owning assistant row;
            # a no-op when the tail is a user row (nothing to regenerate over).
            await repo.delete_tail_assistant(chat_id)
        if body.user_parts:
            parts: list[dict[str, Any]] = [{"type": "text", "text": p.text}
                                           for p in body.user_parts]
            parts += [{"type": "image", "attachmentId": a} for a in body.attachment_ids]
            row = await repo.append_message(chat_id, role="user", parts=parts,
                                            settings_snapshot=chat.settings)
            try:
                await repo.link_attachments(user_id, chat_id, row.id, body.attachment_ids)
            except KeyError as exc:  # pragma: no cover - pre-checked above
                raise api_error(422, "attachment_missing", f"attachment {exc.args[0]} is gone") from exc
            persisted.append({"type": "message-persisted", "role": "user", "messageId": row.id,
                              "seq": row.seq, "attachmentIds": body.attachment_ids})
        if body.tool_results:
            tool_parts: list[dict[str, Any]] = [
                {"type": "tool_result", "callId": r.call_id, "result": r.result, "ok": True}
                for r in body.tool_results
            ]
            row = await repo.append_message(chat_id, role="tool", parts=tool_parts,
                                            settings_snapshot=chat.settings)
            for r in body.tool_results:
                # The owner is the assistant row that actually made *this*
                # call, not merely the most recent one (a later assistant row
                # may have been appended since).
                owner = next(
                    (m for m in reversed(msgs) if m.role == "assistant" and any(
                        p.get("type") == "tool_call" and p.get("id") == r.call_id
                        for p in m.parts)),
                    None,
                )
                if owner is None:
                    continue
                # Same request as the tool row: a reload can never re-enable
                # the option buttons (spec §4.4).
                sel = r.result.get("selected", []) if isinstance(r.result, dict) else []
                await repo.set_options_answered(owner.id, r.call_id, [str(s) for s in sel])
            persisted.append({"type": "message-persisted", "role": "tool", "messageId": row.id,
                              "seq": row.seq, "attachmentIds": []})
        msgs = await repo.list_messages(chat_id)

        async def load_image(aid: str) -> tuple[str, bytes] | None:
            a = await repo.get_attachment(user_id, aid)
            if a is None or a.evicted_at is not None:
                return None
            p = storage.file_path(settings.data_dir, user_id, a.sha256, ALLOWED_IMAGE_MIMES[a.mime])
            data = await run_in_threadpool(_read_bytes, p)
            return (a.mime, data) if data is not None else None

        oa_messages = await build_messages(msgs, chat.settings, supports_vision=info.supports_vision,
                                           load_image=load_image)

    # --- everything trivial, computed before the counter is taken -----------
    payload: dict[str, Any] = {
        "model": chat.model, "messages": oa_messages, "stream": True,
        # The proxy injects `include_usage` only when it has to force-stream a
        # NON-streaming client (app/proxy/routes.py). A streaming client is
        # left untouched, so we must ask for the usage tail ourselves —
        # otherwise every turn falls back to estimated usage.
        "stream_options": {"include_usage": True},
        "temperature": chat.settings.get("temperature", 0.7),
        "max_tokens": chat.settings.get("max_tokens", 1024),
        "top_p": chat.settings.get("top_p", 1.0),
    }
    # Only the tools this chat has enabled are advertised (spec §2.4): the
    # settings panel's Tools checkbox writes `enabled_tools`, and unchecking it
    # has to actually stop the model being offered `present_options`. Unknown
    # names in the list are ignored rather than trusted into the payload.
    enabled = chat.settings.get("enabled_tools") or []
    tools = [CLIENT_TOOLS[n] for n in enabled if n in CLIENT_TOOLS]
    if info.supports_tools and tools:
        payload["tools"] = tools
    # Thinking is a chat-template decision, not a sampling one: vLLM only
    # exposes it through `chat_template_kwargs`, so there is no OpenAI field to
    # set. Only the OFF case is sent — a template that has never heard of the
    # flag must keep its own default, which is what "on" has to mean here.
    template_kwargs: dict[str, Any] = {}
    if chat.settings.get("enable_thinking") is False:
        template_kwargs["enable_thinking"] = False
    else:
        # `reasoning_effort` (#241) rides the same kwarg and is only meaningful
        # while thinking is on. It is forwarded verbatim, in the template's own
        # vocabulary, and ONLY when this model's template is known to accept
        # that exact value: a validating template `raise_exception`s on an
        # unknown one, so a stored level that this model does not list (the
        # chat was pinned to a different model when it was chosen, or the
        # template could not be read) is dropped and the engine default applies.
        effort = chat.settings.get("reasoning_effort")
        if effort and effort in info.reasoning_efforts:
            template_kwargs["reasoning_effort"] = effort
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs

    state = StreamState()
    assistant_id = new_id()
    # The top-of-turn `context` frame (spec §3: context state is "computed on
    # GET /chats/{id} and at the top of every turn"). It is the same state the
    # `context_full` 409 above was decided from, so the client can render the
    # window gauge from the first frames instead of waiting for the end-of-turn
    # `context` that follows the reply.
    ctx_before = ctx
    chat_model: str = chat.model
    chat_settings = chat.settings
    window = info.context_window
    image_tokens = info.image_tokens_estimate
    user_text = "".join(p.text for p in (body.user_parts or []))
    upstream_url = f"http://127.0.0.1:{settings.bind_port}/v1/chat/completions"
    upstream_headers = {"Authorization": f"Bearer {secret.plaintext}",
                        "Content-Type": "application/json", "Accept": "text/event-stream"}
    content = json.dumps(payload).encode("utf-8")

    # --- detach: register the live turn, spawn the runner, subscribe ---------
    # The lock is held by the runner from here on: it was acquired in `turn`
    # before this spawn and is released only by the runner's persist-task
    # callback. Nothing below may raise (registry insert + task spawn), so the
    # `turn` wrapper's release-on-raise cannot double-fire once the runner owns
    # the lock.
    registry: LiveTurns = request.app.state.chat2_live_turns
    live = registry.start(chat_id, request_id=body.request_id, message_id=assistant_id)
    live.task = _spawn(_run_turn(
        app=request.app, registry=registry, live=live, settings=settings, locks=locks,
        persisted=persisted, ctx_before=ctx_before, state=state, assistant_id=assistant_id,
        chat_id=chat_id, user_id=user_id,
        ledger_request_id=_ledger_request_id(user_id, body.request_id),
        chat_model=chat_model, chat_settings=chat_settings, window=window,
        image_tokens=image_tokens, user_text=user_text, oa_messages=oa_messages,
        upstream_url=upstream_url, upstream_headers=upstream_headers, content=content,
        secret_plaintext=secret.plaintext,
    ))
    # Guarded: registered for logout / revoke and an admin token re-checked
    # while open (app/auth/stream_guard.py). Ending it ends only this
    # subscriber; the detached turn runs on.
    return StreamingResponse(
        guarded(request, stream_key(request), registry.subscribe(live)),
        media_type="text/event-stream", headers=sse_headers(),
    )


async def _run_turn(
    *,
    app: Any,  # noqa: ANN401
    registry: LiveTurns,
    live: LiveTurn,
    settings: Any,  # noqa: ANN401
    locks: Any,  # noqa: ANN401
    persisted: list[dict[str, Any]],
    ctx_before: ContextState,
    state: StreamState,
    assistant_id: str,
    chat_id: str,
    user_id: int,
    ledger_request_id: str,
    chat_model: str,
    chat_settings: dict[str, Any],
    window: int | None,
    image_tokens: int,
    user_text: str,
    oa_messages: list[dict[str, Any]],
    upstream_url: str,
    upstream_headers: dict[str, str],
    content: bytes,
    secret_plaintext: str,
) -> None:
    """The detached runner: the whole former response generator, appending to
    the live-turn registry instead of yielding to one HTTP response.

    Cancellation reaches this task from exactly two places — the explicit
    abort endpoint and the lifespan shutdown — never from a client disconnect
    (that cancels only a subscriber). The CancelledError is swallowed rather
    than re-raised: the task is detached, nobody consumes its result, and
    completing normally lets the abort endpoint `gather` it to know the
    `aborted` tail is persisted and the lock released.
    """
    counter = app.state.chat_active_requests
    outcome = "ok"
    pre_stream_failure = False
    error_event: dict[str, Any] | None = None
    counter_token: Any = None
    created: httpx.AsyncClient | None = None
    opened: httpx.Response | None = None

    async def emit(event: dict[str, Any]) -> None:
        await registry.append(live, frame(event))

    try:
        counter_token = await counter.enter()
        http = httpx.AsyncClient(timeout=None)
        created = http
        # No wall-clock timeout here on purpose: the loopback proxy enforces
        # settings.request_max_wall_s (VW_REQUEST_MAX_WALL_S) and ends the
        # stream server-side, which this loop sees as upstream EOF.
        resp = await http.send(
            http.build_request("POST", upstream_url, content=content,
                               headers=upstream_headers),
            stream=True,
        )
        opened = resp
        for ev in persisted:
            await emit(ev)
        await emit({"type": "message-start", "messageId": assistant_id, "seq": -1,
                    "model": chat_model})
        await emit(ctx_before.as_event())
        if resp.status_code != 200:
            # Pre-stream failure: no ChatEvents will ever arrive, so the
            # `error` event is the terminal frame (spec §4.2 step 6 /
            # §4.5 — step 7's context+done applies "on stream end").
            pre_stream_failure = True
            raw = await resp.aread()
            code = _STATUS_CODES.get(resp.status_code, "upstream")
            outcome = "timeout" if code == "timeout" else "error"
            error_event = {"type": "error", "code": code,
                           "message": raw.decode("utf-8", "replace")[:500],
                           "retryable": code in _RETRYABLE}
            await emit(error_event)
        else:
            async for ev in normalize(resp.aiter_raw(), state, set(CLIENT_TOOLS)):
                await emit(ev)
            if state.finish_reason == "guard":
                outcome = "guard"
                error_event = {"type": "error", "code": "guard",
                               "message": "stopped by the runaway guard",
                               "retryable": True}
                await emit(error_event)
    except asyncio.CancelledError:
        # Explicit abort (POST .../turn/abort) or app shutdown — NOT a client
        # disconnect, which never reaches this task. Record an `aborted` turn;
        # the `finally` persists the partial tail and emits the terminal
        # `done {finishReason: "aborted"}` for any subscribers still attached.
        outcome = "aborted"
    except Exception as exc:  # upstream dropped mid-stream
        outcome = "interrupted"
        error_event = {"type": "error", "code": "upstream", "message": str(exc)[:500],
                       "retryable": True}
        logger.exception("chat2 turn interrupted")
        try:
            await emit(error_event)
        except Exception:  # registry append should never fail; belt-and-braces
            pass
    finally:
        # 1. close upstream FIRST so the proxy releases its slot. Kept
        #    non-fatal so a teardown error cannot skip persistence (and
        #    with it the lock release).
        # Order is load-bearing: response, then client, then counter.
        # Each step is cancellation-safe (see `_close_safely`), so a second
        # cancellation (shutdown's wait_for timeout) can neither leave the
        # proxy's slot held nor skip the persist spawn and the lock release.
        if opened is not None:
            await _close_safely("upstream response", opened.aclose())
        if created is not None:
            await _close_safely("loopback client", created.aclose())
        if counter_token is not None:
            await _close_safely("active-request counter", counter.exit(counter_token))
        # 2. persist — in its own task behind a shield, so cancelling the
        #    runner cannot leave the turn unrecorded (spec §4.2 step 8).
        finish = _OUTCOME_FINISH.get(outcome) or state.finish_reason or "stop"
        usage = state.usage or _estimate_usage(oa_messages, state.text, image_tokens)
        persist = _spawn(_persist_turn(
            settings=settings, state=state, chat_id=chat_id, user_id=user_id,
            assistant_id=assistant_id, chat_model=chat_model,
            chat_settings=chat_settings, window=window,
            request_id=ledger_request_id,
            outcome=outcome, finish=finish, usage=usage, error_event=error_event,
            user_text=user_text,
        ))
        # 3. the lock is released *last* — from the persist task's own
        #    completion, so the guarantee also holds on the abort path
        #    where nobody is left waiting on the shield below.
        persist.add_done_callback(lambda _t: locks.release(chat_id))
        done_state = _Persisted(None, None, 0, "")
        try:
            done_state = await asyncio.shield(persist)
        except asyncio.CancelledError:
            pass  # the shielded task runs to completion on its own
        # `user_text` keeps a tool-result-only follow-up from re-firing
        # the title on a user_turns count it did not move.
        if outcome == "ok" and user_text and _should_title(done_state.title_source,
                                                           done_state.user_turns):
            # Fire-and-forget, after the turn is recorded (spec §4.2 step 9).
            _spawn(_llm_title(app, user_id, chat_id, chat_model,
                              done_state.context_text, secret_plaintext,
                              first=done_state.user_turns == 1))
        # 4. terminal frames — appended even when the original client is long
        #    gone (that is the point of detaching). After a pre-stream failure
        #    the `error` frame above is terminal, exactly as before. An aborted
        #    turn ends with `done {finishReason: "aborted"}` so a reattached
        #    subscriber commits the partial tail the same way the aborting
        #    client does locally.
        try:
            ctx_after = done_state.context
            if outcome == "aborted":
                await emit({"type": "done", "messageId": assistant_id,
                            "finishReason": finish})
            elif not pre_stream_failure:
                if ctx_after is not None:
                    await emit(ctx_after.as_event())
                await emit({"type": "done", "messageId": assistant_id,
                            "finishReason": finish})
        finally:
            await registry.finish(live)


@router.post("/chats/{chat_id}/turn/abort")
async def turn_abort(
    chat_id: str, request: Request, user_id: int = Depends(current_user_id)
) -> dict[str, Any]:
    """Explicitly stop the chat's detached turn.

    Cancels the runner and WAITS for it: when this returns, the partial tail
    is persisted as `aborted` (ledger row included) and the per-chat turn lock
    is released, so the caller can immediately start a new turn without racing
    a 409 `turn_in_flight`.
    """
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        if await Chat2Repo(db).get_chat(user_id, chat_id) is None:
            raise api_error(404, "not_found", "chat not found")
    registry: LiveTurns = request.app.state.chat2_live_turns
    live = registry.get(chat_id)
    if live is None or live.done:
        raise api_error(404, "not_found", "no live turn for this chat")
    task = live.task
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    if not live.done:
        # The runner was cancelled before its first tick ever ran (a window of
        # one event-loop step): its finally never executed, so nothing else
        # will release the lock or close the entry. No frames were produced
        # and no assistant row exists — release the minimum by hand.
        request.app.state.chat2_turn_locks.release(chat_id)
        await registry.finish(live)
    return {"aborted": True, "message_id": live.message_id}


@router.get("/chats/{chat_id}/turn/live")
async def turn_live(
    chat_id: str, request: Request, user_id: int = Depends(current_user_id)
) -> StreamingResponse:
    """Re-attach to the chat's detached turn.

    Replays every frame the original subscriber saw — byte-identical, so the
    frontend reducer runs the exact same path — then follows the live tail
    until the turn is done. A finished turn still replays (and terminates
    immediately) for `live.DONE_TTL_S` after `done`, which covers the race
    where the turn finishes between `GET /chats/{id}` and this request; after
    the reap it is a 404 with the shared error envelope.
    """
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        if await Chat2Repo(db).get_chat(user_id, chat_id) is None:
            raise api_error(404, "not_found", "chat not found")
    registry: LiveTurns = request.app.state.chat2_live_turns
    live = registry.get(chat_id)
    if live is None:
        raise api_error(404, "not_found", "no live turn for this chat")
    # Guarded: registered for logout / revoke and an admin token re-checked
    # while open (app/auth/stream_guard.py). Ending it ends only this
    # subscriber; the detached turn runs on.
    return StreamingResponse(
        guarded(request, stream_key(request), registry.subscribe(live)),
        media_type="text/event-stream", headers=sse_headers(),
    )


async def _persist_turn(
    *,
    settings: Any,  # noqa: ANN401
    state: StreamState,
    chat_id: str,
    user_id: int,
    assistant_id: str,
    chat_model: str,
    chat_settings: dict[str, Any],
    window: int | None,
    request_id: str,
    outcome: str,
    finish: str,
    usage: Usage,
    error_event: dict[str, Any] | None,
    user_text: str,
) -> _Persisted:
    """Write the assistant row + ledger row + chat rollup; return the fresh
    context state plus everything the LLM-title decision needs (the chat's
    `title_source`, the user-turn count, and the prompt context built from the
    rows this turn just committed).

    Runs in its own task so cancellation of the streaming response cannot
    abandon it half-written, and swallows its own failures: on the abort path
    nobody awaits the result, so a raised exception would surface only as an
    "exception was never retrieved" warning at GC time.
    """
    try:
        async with open_db(settings.db_path) as db:
            repo, ledger = Chat2Repo(db), LedgerRepo(db)
            parts = state.parts(set(CLIENT_TOOLS))
            if error_event:
                parts.append({"type": "error", "code": error_event["code"],
                              "message": error_event["message"]})
            await repo.append_message(
                chat_id, role="assistant", parts=parts, settings_snapshot=chat_settings,
                model=chat_model, usage=usage.as_dict(), finish_reason=finish,
                error=error_event, message_id=assistant_id,
            )
            _lid, cost = await ledger.record(
                request_id=request_id, user_id=user_id, org_id=None, chat_id=chat_id,
                message_id=assistant_id, purpose="turn", provider=PROVIDER, model=chat_model,
                usage=usage, outcome=outcome, provider_request_id=state.upstream_id,
            )
            # `touch_chat` has no user scope by design — safe here because
            # `get_chat(user_id, chat_id)` already authorised this turn.
            await repo.touch_chat(chat_id, add_cost_micros=cost.cost_micros or 0)
            chat_after = await repo.get_chat(user_id, chat_id)
            msgs_after = await repo.list_messages(chat_id)
            user_turns = sum(1 for m in msgs_after if m.role == "user")
            if (chat_after is not None and chat_after.title_source == "auto" and user_text
                    and user_turns == 1):
                # First completed exchange: a synchronous first-line title so
                # the sidebar is never stuck on "New chat" (spec §4.2 step 9).
                await repo.update_chat(user_id, chat_id, title=first_line_title(user_text))
            return _Persisted(
                context=derive_context(msgs_after, chat_settings, window),
                title_source=chat_after.title_source if chat_after else None,
                user_turns=user_turns,
                context_text=title_context(msgs_after),
            )
    except Exception:
        logger.exception("chat2 turn persistence failed")
        return _Persisted(None, None, 0, "")


async def _llm_title(app: Any, user_id: int, chat_id: str, model: str,  # noqa: ANN401
                     context_text: str, bearer: str, *, first: bool = False) -> None:
    settings = app.state.settings

    async def post_json(payload: dict[str, Any]) -> dict[str, Any] | None:
        decision = await app.state.chat2_budget.check(user_id, None)
        if not decision.allowed:
            return None
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"http://127.0.0.1:{settings.bind_port}/v1/chat/completions",
                             json=payload, headers={"Authorization": f"Bearer {bearer}"})
        if r.status_code != 200:
            return None
        data: dict[str, Any] = r.json()
        u = data.get("usage") or {}
        async with open_db(settings.db_path) as db:
            await LedgerRepo(db).record(
                request_id=f"title-{chat_id}-{now_iso()}", user_id=user_id, org_id=None,
                chat_id=chat_id, message_id=None, purpose="title", provider=PROVIDER, model=model,
                usage=Usage(prompt=int(u.get("prompt_tokens", 0)),
                            completion=int(u.get("completion_tokens", 0))),
                outcome="ok", provider_request_id=data.get("id"),
            )
        return data

    try:
        title = await generate_llm_title(post_json=post_json, model=model,
                                         context_text=context_text)
        if title:
            async with open_db(settings.db_path) as db:
                repo = Chat2Repo(db)
                chat = await repo.get_chat(user_id, chat_id)
                # The cheap guard: an unchanged verdict must not churn
                # `title_prev` into a self-reference. The authoritative
                # "is this chat still auto-titled?" check is the
                # `title_source = 'auto'` predicate inside
                # `replace_auto_title` -- the user may rename the chat while
                # the (lowest-priority) title call is queued, and their title
                # has to win even though that race is far too narrow to see
                # from a read here.
                if (chat and chat.title_source == "auto"
                        and normalized_title(title) != normalized_title(chat.title)):
                    await repo.replace_auto_title(user_id, chat_id, title=title,
                                                  keep_prev=not first)
    except Exception:
        logger.exception("chat2 llm title failed")
