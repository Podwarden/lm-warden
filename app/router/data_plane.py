"""The router data plane (#287): decide, serve locally, fall back, relay.

``routes_messages`` calls ``messages_entry`` / ``catch_all_entry`` only when
the router is enabled. Nothing in here may log or store a header or a body;
the one ``Decision`` per request carries names, a route, a reason and timings.

Security invariant (decision 2): the ONLY calls to ``passthrough.relay`` are
in ``_relay_to_anthropic``, and it is reached only through ``_relay_gate``,
which refuses unless the key may relay, the request carried ``X-LMWarden-Key``
and an upstream credential that is not itself a warden key.
"""

import asyncio
import base64
import binascii
import dataclasses
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException

from app.db.repos.models import ModelRow
from app.db.repos.tokens import TokenRow
from app.guard import anthropic as guard_anthropic
from app.guard import proxy as guard_proxy
from app.proxy.auth import token_allows, upstream_credential_headers
from app.proxy.messages_clamp import clamp_max_tokens
from app.proxy.messages_translate import error_body
from app.proxy.model_cache import model_by_id, model_by_served_name
from app.proxy.request_id import REQUEST_ID_HEADER, copy_request_id
from app.proxy.session_id import extract_session
from app.router.passthrough import RelayError, relay
from app.router.relay_record import RelayRecorder
from app.router.rules import RuleSet, RuleSpec, match_rule
from app.router.state import Decision, Route
from app.utils.sse import sse_headers

logger = logging.getLogger(__name__)

_STREAM_ERROR_FRAME = (
    b"\n\nevent: error\n"
    b'data: {"type":"error","error":{"type":"api_error",'
    b'"message":"router: upstream stream interrupted"}}\n\n'
)


@dataclass
class Outcome:
    """What to do about a local leg that did not produce a usable answer.

    ``route``: ``fallback`` (go to Anthropic), ``refused`` (the router answers
    with ``status``) or ``local`` (the engine's own answer goes to the client).
    """

    route: str
    reason: str | None
    status: int
    breaker_failure: bool


def local_failure_outcome(status: int, *, fallback_rule: bool, path: str) -> Outcome:
    """Decision 13, row by row, for an engine/translation status."""
    reason = f"status_{status}"
    if path.endswith("/count_tokens"):
        # It generates nothing: always falls back, never trips the breaker.
        return Outcome("fallback", reason, status, False)
    breaker = status >= 500 or status == 429
    bad_request = status in (400, 413)
    if not (breaker or bad_request):
        return Outcome("local", None, status, False)
    if fallback_rule:
        return Outcome("fallback", reason, status, breaker)
    return Outcome("refused", reason, 400 if bad_request else 529, breaker)


def _soft_outcome(reason: str, rule: RuleSpec, *, count_tokens: bool) -> Outcome:
    """Failures the warden sees coming (target missing/unloaded, breaker open)."""
    if rule.fallback or count_tokens:
        return Outcome("fallback", reason, 529, False)
    return Outcome("refused", reason, 529, False)


class _Fail(Exception):
    def __init__(self, outcome: Outcome, response: Response | None = None) -> None:
        super().__init__(outcome.reason)
        self.outcome = outcome
        self.response = response


def _err(status: int, message: str) -> JSONResponse:
    return JSONResponse(error_body(status, message), status_code=status)


class _Rec:
    """Accumulates the single Decision of one request; ``finish`` is idempotent."""

    def __init__(self, request: Request, token: TokenRow, path: str, model_in: str | None) -> None:
        self.state = request.app.state.router
        self.t0 = time.monotonic()
        self.ts = time.time()
        self.path = path
        self.model_in = model_in
        self.model_out: str | None = None
        self.rule: RuleSpec | None = None
        self.route: Route = "error"
        self.reason: str | None = None
        self.status = 500
        self.ttfb_ms: int | None = None
        self.stream = False
        self.token_name = token.name
        self._done = False

    def mark_ttfb(self) -> None:
        if self.ttfb_ms is None:
            self.ttfb_ms = int((time.monotonic() - self.t0) * 1000)

    def set(self, route: Route, reason: str | None, status: int) -> None:
        self.route, self.reason, self.status = route, reason, status

    def finish(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            d = Decision(
                ts=self.ts,
                path=self.path,
                model_in=self.model_in,
                model_out=self.model_out,
                rule_id=self.rule.id if self.rule else None,
                route=self.route,
                reason=self.reason,
                status=self.status,
                latency_ms=int((time.monotonic() - self.t0) * 1000),
                ttfb_ms=self.ttfb_ms,
                stream=self.stream,
                token_name=self.token_name,
            )
            self.state.record(d, rule_pattern=self.rule.pattern if self.rule else None)
            logger.debug(
                "router path=%s model_in=%r model_out=%r route=%s reason=%s status=%s rule=%s",
                d.path,
                (d.model_in or "")[:128],
                d.model_out,
                d.route,
                d.reason,
                d.status,
                d.rule_id,
            )
        except Exception:  # noqa: BLE001 - never fails a request
            logger.debug("router: decision logging failed", exc_info=True)


# -- decision 2: the only road to Anthropic -----------------------------------

#: A warden inference (``vw_``) or admin (``vwa_``) key, wherever it sits in a
#: credential value, whatever the scheme -- but only at the start of a token, so
#: an Anthropic key that happens to contain "vw_" mid-token is not mistaken for one.
_WARDEN_KEY = re.compile(r"(?<![A-Za-z0-9])vwa?_", re.IGNORECASE)


def _is_warden_key_value(value: str) -> bool:
    if _WARDEN_KEY.search(value):
        return True
    # HTTP Basic (or any scheme) wrapping a warden key in base64.
    for part in re.split(r"[\s,]+", value):
        if len(part) < 8:
            continue
        try:
            decoded = base64.b64decode(part + "=" * (-len(part) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        if _WARDEN_KEY.search(decoded.decode("latin-1")):
            return True
    return False


def _relay_gate(request: Request, token: TokenRow) -> tuple[str, str] | None:
    """None when this request may be relayed; else (reason, message) to refuse.

    The router being enabled and the key being valid are guaranteed by the
    callers (``require_bearer`` ran; the gate in ``routes_messages`` checked).
    """
    if not token.anthropic_relay:
        return "relay_not_allowed", "router: this API key may not relay to Anthropic"
    creds = upstream_credential_headers(request)
    if not creds:
        return (
            "no_upstream_credential",
            "router: no Anthropic credential on the request "
            "(send the warden key in X-LMWarden-Key)",
        )
    for value in creds.values():
        if _is_warden_key_value(value):
            # A warden key in the Anthropic credential slot is never relayed.
            return "no_upstream_credential", "router: the upstream credential is a warden key"
    return None


async def _relay_to_anthropic(
    request: Request,
    token: TokenRow,
    rec: _Rec,
    cfg_upstream: str,
    *,
    body: bytes | None,
    route: Route,
    reason: str | None,
) -> Response:
    state = rec.state
    refusal = _relay_gate(request, token)
    if refusal is not None:  # defence in depth: every caller has already gated
        return _refuse(rec, 403, refusal[0], refusal[1])
    rec.route = route
    rec.reason = reason

    def on_close(resp: Response) -> None:
        rec.status = resp.status_code
        rec.finish()

    # Sessions pages: a relayed /v1/messages request gets a history row,
    # recorded off the stream path (relay_record). Fail-open.
    recorder: RelayRecorder | None = None
    if body is not None and rec.path == "/v1/messages":
        try:
            recorder = RelayRecorder(request, token, body, upstream=cfg_upstream)
            recorder.start()
        except Exception:  # noqa: BLE001
            recorder = None

    def on_usage(i: int | None, o: int | None) -> None:
        state.add_passthrough_tokens(i, o)
        if recorder is not None:
            recorder.note_usage(i, o)

    def on_ttfb() -> None:
        rec.mark_ttfb()
        if recorder is not None:
            recorder.on_ttfb()

    def on_close_all(resp: Response) -> None:
        on_close(resp)
        if recorder is not None:
            recorder.on_close(resp)

    guarded = getattr(request.state, "guard", None)
    try:
        resp = await relay(
            request,
            upstream_base=cfg_upstream,
            body=body,
            credentials=upstream_credential_headers(request),
            on_usage=on_usage,
            on_ttfb=on_ttfb,
            on_close=on_close_all,
            # Guarded: the recorder is fed the engine view (trailer added) by the wrapper below.
            on_chunk=recorder.on_chunk if recorder is not None and guarded is None else None,
            extra_headers=(
                {REQUEST_ID_HEADER: recorder.request_id} if recorder is not None else None
            ),
            # Only for recorded relays: the recorder must be able to read the reply.
            narrow_encoding=recorder is not None,
            identity_encoding=guarded is not None,
        )
    except RelayError as exc:
        if exc.reason is not None:  # the request itself was refused; Anthropic was not tried
            return _refuse(rec, exc.status, exc.reason, exc.message)
        rec.set("error", exc.error_reason, exc.status)
        rec.finish()
        return _err(exc.status, exc.message)
    rec.status = resp.status_code
    rec.stream = resp.headers.get("content-type", "").startswith("text/event-stream")
    if guarded is not None and isinstance(resp, StreamingResponse):
        resp = guard_anthropic.wrap_relay(
            resp, guarded, on_engine_chunk=recorder.on_chunk if recorder is not None else None
        )
    return resp


def _refuse(rec: _Rec, status: int, reason: str, message: str) -> Response:
    rec.set("refused", reason, status)
    rec.finish()
    return _err(status, f"{message} ({reason})")


# -- entry points -----------------------------------------------------------------


async def _read_capped(request: Request, cap: int) -> bytes | None:
    """The request body, or None once it exceeds ``cap`` bytes.

    Reads the stream chunk by chunk and stops at the cap, so a chunked upload
    (no Content-Length to check) is never buffered whole. On success the body
    is cached on the request, as ``request.body()`` would have.
    """
    parts: list[bytes] = []
    size = 0
    async for part in request.stream():
        size += len(part)
        if size > cap:
            return None
        parts.append(part)
    raw = b"".join(parts)
    request._body = raw
    return raw


def _loads(raw: bytes) -> dict[str, Any] | None:
    try:
        data = json.loads(raw) if raw else None
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


async def messages_entry(request: Request, token: TokenRow, *, count_tokens: bool) -> Response:
    """The whole ``/v1/messages[/count_tokens]`` handler while the router is on."""

    state = request.app.state.router
    rs: RuleSet = await state.ruleset(request.app.state.settings.db_path)
    cfg = rs.config
    path = "/v1/messages/count_tokens" if count_tokens else "/v1/messages"

    cap = cfg.max_body_mb << 20
    declared = request.headers.get("content-length", "")
    too_big = declared.isdigit() and int(declared) > cap
    raw = None if too_big else await _read_capped(request, cap)
    if raw is None:
        rec = _Rec(request, token, path, None)
        return _refuse(rec, 413, "request_too_large", f"router: body exceeds {cfg.max_body_mb} MiB")

    data = _loads(raw)
    if data is None:
        # Not a JSON object: today's handler answers (400), nothing to route.
        return await _legacy(request, token, count_tokens, _Rec(request, token, path, None))
    model_in = data.get("model")
    rec = _Rec(request, token, path, model_in if isinstance(model_in, str) else None)
    rec.stream = bool(data.get("stream")) and not count_tokens

    # Sensitive data guard: one scan for both legs, before any rule is matched, so the local
    # leg, a fallback and a passthrough all only ever see the sanitized body.
    guarded = await _guard_entry(request, token, data, rec, count_tokens=count_tokens)
    if isinstance(guarded, Response):
        return guarded
    if guarded is not None:
        raw = guarded

    rule = match_rule(model_in, rs.rules)
    if rule is not None:
        rec.rule = rule
        return await _rule_flow(request, token, rs, rule, raw, rec, count_tokens=count_tokens)

    if isinstance(model_in, str) and await _is_served_name(request, model_in):
        return await _legacy(request, token, count_tokens, rec)

    if not cfg.passthrough_unmatched:
        return await _legacy(request, token, count_tokens, rec)

    refusal = _relay_gate(request, token)
    if refusal is not None:
        return _refuse(rec, 403, refusal[0], refusal[1])
    if not isinstance(model_in, str) or not token_allows(token, model_in):
        return _refuse(rec, 403, "token_not_allowed", "router: token not allowed for this model")
    return await _relay_to_anthropic(
        request, token, rec, cfg.upstream_url, body=raw, route="passthrough", reason=None
    )


async def catch_all_entry(request: Request, token: TokenRow, rest: str) -> Response:
    """Any ``/v1/*`` path no warden route owns, while the router is on."""
    state = request.app.state.router
    rs: RuleSet = await state.ruleset(request.app.state.settings.db_path)
    rec = _Rec(request, token, request.url.path[:200], None)
    if guard_proxy.guard_for(getattr(request.app.state, "guard_platform", None), token):
        return _refuse(
            rec,
            403,
            "guard_no_passthrough",
            "router: this key's sensitive data guard does not cover unparsed passthrough",
        )
    if not rs.config.passthrough_unmatched:
        return _refuse(rec, 404, "passthrough_disabled", "router: passthrough is off")
    if token.allowed_models is not None:
        # The model of such a request is not ours to check (batches carry many,
        # a body we do not parse): a model-restricted key does not relay at all.
        return _refuse(rec, 403, "token_not_allowed", "router: a model-restricted key cannot relay")
    refusal = _relay_gate(request, token)
    if refusal is not None:
        return _refuse(rec, 403, refusal[0], refusal[1])
    return await _relay_to_anthropic(
        request, token, rec, rs.config.upstream_url, body=None, route="passthrough", reason=None
    )


async def _guard_entry(
    request: Request, token: TokenRow, data: dict[str, Any], rec: _Rec, *, count_tokens: bool
) -> Response | bytes | None:
    """The key's guard on an Anthropic body: None (nothing to do), the sanitized body (also put
    on the request for the local leg, with the outcome on ``request.state.guard``), or LM
    Warden's own answer (blocked, or the guard could not check it)."""
    app_state = request.app.state
    guard = guard_proxy.guard_for(getattr(app_state, "guard_platform", None), token)
    if guard is None:
        return None
    seen = guard_proxy.shared_seen(app_state)
    if count_tokens:
        # A count is never answered by the model: replace, never block, and leave the
        # session's block-once state to the real request.
        guard = dataclasses.replace(guard, mode="sanitize", overrides=())
        seen = guard_proxy.SeenSet()
    try:
        session = extract_session(data, request.headers, None).id
    except Exception:  # noqa: BLE001 -- no session id: the key is the session
        session = None
    try:
        out = await asyncio.to_thread(
            guard_proxy.run_guard,
            data,
            "anthropic",
            guard,
            token_id=token.id,
            secret=guard_proxy.shared_secret(app_state),
            session=session or token.id,
            seen=seen,
        )
    except guard_proxy.GuardFailed as e:
        rec.set("refused", "guard_failed", 422)
        rec.finish()
        return guard_anthropic.failed_reply(e.reason)
    if out.action == "pass":
        return None
    blocked_id = uuid.uuid4().hex if out.action == "block" else None
    guard_proxy.log_events(app_state, out.events, blocked_id)
    if out.action == "block":
        rec.set("refused", "guard_blocked", 200)
        rec.finish()
        reply = guard_anthropic.block_reply(out, stream=rec.stream)
        reply.headers[REQUEST_ID_HEADER] = blocked_id or ""
        return reply
    if count_tokens:
        out.trailer = None
    new_raw = json.dumps(out.body, ensure_ascii=False).encode()
    request._body = new_raw
    request.state.guard = out
    return new_raw


async def _is_served_name(request: Request, name: str) -> bool:
    state = request.app.state
    cache = getattr(state, "model_cache", None)
    return await model_by_served_name(cache, state.settings.db_path, name) is not None


async def _legacy(request: Request, token: TokenRow, count_tokens: bool, rec: _Rec) -> Response:
    """Today's handler, unchanged (served names, malformed bodies, 404s)."""
    from app.proxy import routes_messages as rm

    try:
        resp = await (rm.count_tokens_legacy if count_tokens else rm.messages_legacy)(
            request, token
        )
    except HTTPException as exc:
        rec.set("local", f"status_{exc.status_code}", exc.status_code)
        rec.finish()
        raise
    ok = resp.status_code < 400
    rec.set("local", None if ok else f"status_{resp.status_code}", resp.status_code)
    rec.mark_ttfb()
    rec.finish()
    return resp


# -- the rule flow: local leg, fallback, refusal --------------------------------------


async def _rule_flow(
    request: Request,
    token: TokenRow,
    rs: RuleSet,
    rule: RuleSpec,
    raw: bytes,
    rec: _Rec,
    *,
    count_tokens: bool,
) -> Response:
    cfg = rs.config
    holder: dict[str, Any] = {}
    timeout = cfg.local_header_timeout_s if rec.stream else cfg.local_nonstream_timeout_s
    try:
        try:
            # An asyncio.timeout (not wait_for) so the leg can take the clamp's
            # token count out of the timed region: a cold tokenizer load must
            # not read as an engine that is slow to answer.
            async with asyncio.timeout(timeout) as deadline:
                holder["deadline"] = (deadline, timeout)
                resp = await _local_leg(
                    request, token, rs, rule, rec, holder, count_tokens=count_tokens
                )
            rec.mark_ttfb()
            if not isinstance(resp, StreamingResponse):
                rec.finish()  # a streamed answer finishes when its body does
            return resp
        except TimeoutError:
            await _discard(holder.get("resp"))
            reason = "first_byte_timeout" if "resp" in holder else "header_timeout"
            raise _Fail(Outcome("fallback", reason, 529, True)) from None
    except _Fail as fail:
        _release_probe(rec, holder)
        return await _handle_failure(
            request, token, rs, rule, raw, rec, fail, count_tokens=count_tokens
        )
    except HTTPException as exc:  # _forward's own refusals (502 unreachable, admission, ...)
        _release_probe(rec, holder)
        outcome = local_failure_outcome(
            exc.status_code,
            fallback_rule=rule.fallback,
            path="/v1/messages/count_tokens" if count_tokens else "/v1/messages",
        )
        if outcome.route == "local":
            rec.set("local", f"status_{exc.status_code}", exc.status_code)
            rec.finish()
            raise
        return await _handle_failure(
            request, token, rs, rule, raw, rec, _Fail(outcome), count_tokens=count_tokens
        )
    except Exception as exc:  # noqa: BLE001 - a bug or a DB/tokenizer error in OUR leg
        # Not a client cancel (that is a BaseException): a local failure like
        # any other -- falls back per the rule, counts toward the breaker, and
        # the client never sees the exception text.
        # Type only: an exception's text may quote request content.
        logger.warning("router: unexpected local-leg error (%s)", type(exc).__name__)
        _release_probe(rec, holder)
        await _discard(holder.get("resp"))
        if count_tokens:
            outcome = Outcome("fallback", "local_error", 529, False)
        else:
            outcome = Outcome("fallback" if rule.fallback else "refused", "local_error", 529, True)
        return await _handle_failure(
            request, token, rs, rule, raw, rec, _Fail(outcome), count_tokens=count_tokens
        )
    except BaseException:
        # Client gone mid-leg (cancel): never a failure, nothing to answer.
        await _discard(holder.get("resp"))
        rec.set("local", "client_closed", 499)
        rec.finish()
        raise
    finally:
        # A probe that ended without an outcome (4xx, cancel, token refusal)
        # must not keep the half-open breaker closed to everybody else.
        probe = holder.pop("probe", None)
        if probe is not None:
            rec.state.release_probe(*probe)


def _release_probe(rec: _Rec, holder: dict[str, Any]) -> None:
    """Hand the half-open probe slot on. Called before a fallback relay so the
    claim is not held while Anthropic answers; a failure that counts re-opens
    the breaker afterwards anyway (record_failure clears the claim)."""
    probe = holder.pop("probe", None)
    if probe is not None:
        rec.state.release_probe(*probe)


async def _discard(resp: Response | None) -> None:
    """Close a local response that will never reach the client (#286 guarantees:
    the iterator's finally settles a started stream; ``background`` is the
    unstarted-stream guard's abandon)."""
    if resp is None:
        return

    async def _run() -> None:
        it = getattr(resp, "body_iterator", None)
        aclose = getattr(it, "aclose", None)
        try:
            if aclose is not None:
                await aclose()
        finally:
            if resp.background is not None:
                await resp.background()

    try:
        await asyncio.shield(_run())
    except Exception:  # noqa: BLE001
        logger.debug("router: discarding a local response failed", exc_info=True)


async def _local_leg(
    request: Request,
    token: TokenRow,
    rs: RuleSet,
    rule: RuleSpec,
    rec: _Rec,
    holder: dict[str, Any],
    *,
    count_tokens: bool,
) -> Response:
    from app.proxy import routes_messages as rm
    from app.proxy.messages_schemas import CountTokensRequest, MessagesRequest
    from app.proxy.routes import _resolve_loaded

    state = rec.state
    path = rec.path
    target: ModelRow | None = await model_by_id(
        getattr(request.app.state, "model_cache", None),
        request.app.state.settings.db_path,
        rule.target_model_id,
    )
    if target is None:
        raise _Fail(_soft_outcome("target_missing", rule, count_tokens=count_tokens))
    served = target.served_model_name
    rec.model_out = served
    try:
        resolved = _resolve_loaded(request, target, served)
    except HTTPException:
        raise _Fail(_soft_outcome("target_not_loaded", rule, count_tokens=count_tokens)) from None
    if not token_allows(token, served):
        raise _Fail(
            Outcome("refused", "token_not_allowed", 403, False),
        )
    if not count_tokens:
        cfg = rs.config
        claim = state.breaker_admit(
            target.id,
            timeout_s=cfg.local_header_timeout_s if rec.stream else cfg.local_nonstream_timeout_s,
        )
        if claim is None:
            raise _Fail(_soft_outcome("breaker_open", rule, count_tokens=False))
        if claim:  # we are the half-open probe; _rule_flow releases it on any exit
            holder["probe"] = (target.id, claim)

    requested = rec.model_in or ""
    try:
        if count_tokens:
            creq = await rm._parse(request, CountTokensRequest)
            creq.model = served
            oai = rm._translate(creq)
        else:
            req = await rm._parse(request, MessagesRequest)
            req.model = served
            if rule.strip_thinking:
                req.thinking = None
            if rule.min_max_tokens > 0:
                req.max_tokens = max(req.max_tokens, rule.min_max_tokens)
            oai = rm._translate(req)
            # After the min_max_tokens floor: the context wins over the floor.
            # Pause the header-timeout clock while tokenizing (a cold tokenizer
            # load can take seconds), then restart the full window.
            timed = holder.get("deadline")
            if timed is not None:
                timed[0].reschedule(None)
            try:
                await clamp_max_tokens(request, resolved, oai)
            finally:
                if timed is not None:
                    timed[0].reschedule(asyncio.get_running_loop().time() + timed[1])
    except HTTPException as exc:
        raise _Fail(
            local_failure_outcome(exc.status_code, fallback_rule=rule.fallback, path=path)
        ) from None

    if count_tokens:
        resp: Response = await rm._count_local(request, creq, oai, resolved)
        if resp.status_code != 200:
            raise _Fail(local_failure_outcome(resp.status_code, fallback_rule=True, path=path))
        rec.set("local", None, 200)
        return resp

    resp = await rm._serve_local(request, token, req, oai, resolved, response_model_name=requested)
    if not isinstance(resp, StreamingResponse) or resp.status_code != 200:
        if resp.status_code != 200:
            outcome = local_failure_outcome(
                resp.status_code, fallback_rule=rule.fallback, path=path
            )
            raise _Fail(outcome, resp)
        state.record_success(target.id)
        rec.set("local", None, 200)
        return resp

    # Streamed: the first translated frame proves the engine is generating.
    holder["resp"] = resp
    it = resp.body_iterator.__aiter__()
    try:
        first = await it.__anext__()
    except StopAsyncIteration:
        await _discard(resp)
        raise _Fail(
            Outcome("fallback" if rule.fallback else "refused", "first_frame_error", 529, True)
        ) from None
    except Exception:  # noqa: BLE001 - the engine died before its first frame
        await _discard(resp)
        raise _Fail(
            Outcome("fallback" if rule.fallback else "refused", "first_frame_error", 529, True)
        ) from None
    if _as_bytes(first).startswith(b"event: error"):
        await _discard(resp)
        raise _Fail(
            Outcome("fallback" if rule.fallback else "refused", "first_frame_error", 529, True)
        )
    rec.mark_ttfb()
    rec.set("local", None, 200)

    def on_end(kind: str) -> None:
        if kind == "failed":
            state.record_failure(
                target.id,
                "stream_interrupted",
                threshold=rs.config.breaker_threshold,
                open_s=rs.config.breaker_open_s,
            )
            rec.reason = "stream_interrupted"
        rec.finish()

    holder.pop("resp", None)  # handed to the client wrapper below
    holder.pop("probe", None)
    # The first frame proves the engine generates: close the breaker now (this
    # also frees a probe claim). A later break still counts as a failure.
    state.record_success(target.id)
    return copy_request_id(
        resp,
        StreamingResponse(
            _chain(first, it, on_end),
            status_code=200,
            media_type="text/event-stream",
            headers=sse_headers(),
            background=resp.background,
        ),
    )


def _as_bytes(chunk: Any) -> bytes:
    return chunk.encode() if isinstance(chunk, str) else bytes(chunk)


async def _chain(first: Any, it: Any, on_end: Any):  # type: ignore[no-untyped-def]
    """First frame, the rest, and an in-band error if the stream breaks.

    Never replays after the first byte (decision 14): an exception ends the
    stream with an ``event: error`` frame and counts toward the breaker. A
    client cancel (GeneratorExit/CancelledError) is not a failure.
    """
    kind = "cancelled"
    try:
        yield first
        error_seen = False
        async for frame in it:
            if not error_seen and _as_bytes(frame).startswith(b"event: error"):
                error_seen = True
            yield frame
        kind = "failed" if error_seen else "ok"
    except Exception:  # noqa: BLE001 - upstream broke after the first byte
        kind = "failed"
        yield _STREAM_ERROR_FRAME
    finally:
        try:
            aclose = getattr(it, "aclose", None)
            if aclose is not None:
                await asyncio.shield(aclose())
        finally:
            on_end(kind)


async def _handle_failure(
    request: Request,
    token: TokenRow,
    rs: RuleSet,
    rule: RuleSpec,
    raw: bytes,
    rec: _Rec,
    fail: _Fail,
    *,
    count_tokens: bool,
) -> Response:
    outcome = fail.outcome
    state = rec.state
    cfg = rs.config
    if outcome.breaker_failure and outcome.reason:
        state.record_failure(
            rule.target_model_id,
            outcome.reason,
            threshold=cfg.breaker_threshold,
            open_s=cfg.breaker_open_s,
        )
    if outcome.route == "local":  # another 4xx: the engine's answer goes to the client
        if fail.response is not None:
            rec.set("local", f"status_{fail.response.status_code}", fail.response.status_code)
            rec.finish()
            return fail.response
        raise HTTPException(outcome.status)
    if outcome.route == "refused":
        if outcome.reason == "token_not_allowed":
            return _refuse(rec, 403, "token_not_allowed", "router: token not allowed for the model")
        return _refuse(
            rec,
            outcome.status,
            outcome.reason or "refused",
            "router: local model unavailable; fallback is off for this rule",
        )
    # fallback to Anthropic, original request untouched
    refusal = _relay_gate(request, token)
    if refusal is None and (
        not isinstance(rec.model_in, str) or not token_allows(token, rec.model_in)
    ):
        # The same allow-list a plain passthrough of this model would meet.
        refusal = ("token_not_allowed", "router: token not allowed for this model")
    if refusal is not None:
        if count_tokens:
            # Decision 13: count_tokens only *prefers* Anthropic when local fails;
            # when it may not relay, the local error is the answer.
            return _refuse(
                rec,
                outcome.status,
                outcome.reason or "refused",
                "router: local model unavailable and the request may not be relayed",
            )
        return _refuse(rec, 403, refusal[0], refusal[1])
    if fail.response is not None:
        await _discard(fail.response)
    return await _relay_to_anthropic(
        request, token, rec, cfg.upstream_url, body=raw, route="fallback", reason=outcome.reason
    )
