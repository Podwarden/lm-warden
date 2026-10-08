"""Anthropic Messages API passthrough: ``POST /v1/messages`` (#281).

Lets Claude Code and the Anthropic SDKs use a warden as their base URL. The
route owns no forwarding of its own: it validates the Anthropic body, turns
it into an OpenAI chat body (messages_translate.py), and runs it through the
same ``_forward`` as ``/v1/chat/completions`` -- so admission, priority,
accounting, the live registry, god mode, the runaway detector and the reaper
behave exactly as they do for OpenAI traffic. Auth is ``require_bearer``
(it also reads ``x-api-key`` on these routes, see auth.py) and routing is
``_resolve_target``, both unchanged.

Every refusal, including the ones ``require_bearer`` raises before the
handler runs, answers in Anthropic's envelope
``{"type": "error", "error": {"type": ..., "message": ...}}`` -- that is what
``_AnthropicErrorRoute`` is for: FastAPI resolves dependencies inside the
route handler, so wrapping the handler is the one place that sees them all
without an app-wide exception handler.
"""

import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable, Coroutine, Mapping
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import RedirectResponse
from starlette.routing import Match

from app.db.repos.tokens import TokenRow
from app.proxy.auth import require_bearer, token_allows
from app.proxy.messages_clamp import clamp_max_tokens
from app.proxy.messages_schemas import CountTokensRequest, MessagesRequest
from app.proxy.messages_translate import (
    AnthropicStreamTranslator,
    TranslationError,
    count_text,
    error_body,
    thinking_requested,
    to_anthropic_message,
    to_openai_request,
    upstream_error_message,
)
from app.proxy.routes import _forward, _resolve_target
from app.router.data_plane import catch_all_entry, messages_entry
from app.utils.sse import sse_headers

logger = logging.getLogger(__name__)


def anthropic_affinity_key(req: MessagesRequest) -> str | None:
    """Replica-affinity key (#286): the Anthropic ``metadata.user_id``.

    Read from the ORIGINAL body: the translated OpenAI body has no ``user``.
    """
    meta = (req.model_extra or {}).get("metadata")
    uid = meta.get("user_id") if isinstance(meta, dict) else None
    return uid if isinstance(uid, str) and uid else None


def anthropic_error(
    status: int, message: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(error_body(status, message), status_code=status, headers=headers)


def _detail_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        for key in ("message", "detail", "error_code"):
            value = detail.get(key)
            if isinstance(value, str):
                return value
    return json.dumps(detail)


class _AnthropicErrorRoute(APIRoute):
    """An APIRoute whose HTTPExceptions -- dependencies' included -- answer
    in the Anthropic error envelope, status and headers (Retry-After) kept."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except StarletteHTTPException as exc:
                return anthropic_error(exc.status_code, _detail_text(exc.detail), exc.headers)

        return route_handler


router = APIRouter(prefix="/v1", tags=["proxy"], route_class=_AnthropicErrorRoute)

_Req = TypeVar("_Req", bound=BaseModel)


async def _parse(request: Request, model: type[_Req]) -> _Req:
    raw = await request.body()
    try:
        data = json.loads(raw) if raw else None
    except ValueError:
        raise HTTPException(400, "request body is not valid JSON") from None
    if not isinstance(data, dict):
        raise HTTPException(400, "request body must be a JSON object")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise HTTPException(400, f"{where}: {first['msg']}") from None


def _translate(req: MessagesRequest | CountTokensRequest) -> dict[str, Any]:
    try:
        return to_openai_request(req)
    except TranslationError as exc:
        raise HTTPException(400, str(exc)) from None


async def _router_enabled(request: Request) -> bool:
    """True when the #287 router is on. One in-memory check on the hot path:
    ``ruleset()`` reads the DB only while dirty (boot, control-API writes).
    Fail-open: any error means "off", i.e. today's behaviour."""
    state = getattr(request.app.state, "router", None)
    if state is None:
        return False
    try:
        rs = await state.ruleset(request.app.state.settings.db_path)
    except Exception:  # noqa: BLE001 - the router must never break a request
        logger.debug("router: rule set unavailable; serving locally", exc_info=True)
        return False
    return bool(rs.config.enabled)


@router.post("/messages")
async def messages(request: Request, token: TokenRow = Depends(require_bearer)) -> Response:
    if await _router_enabled(request):
        return await messages_entry(request, token, count_tokens=False)
    return await messages_legacy(request, token)


async def messages_legacy(request: Request, token: TokenRow) -> Response:
    """The handler as it was before the router: served names only."""
    req = await _parse(request, MessagesRequest)
    if not token_allows(token, req.model):
        raise HTTPException(403, f"token not allowed for model '{req.model}'")
    oai = _translate(req)
    target = await _resolve_target(request, req.model)
    await clamp_max_tokens(request, target, oai)
    return await _serve_local(request, token, req, oai, target, response_model_name=req.model)


async def _serve_local(
    request: Request,
    token: TokenRow,
    req: MessagesRequest,
    oai: dict[str, Any],
    target: tuple[Any, str, int, Any],
    *,
    response_model_name: str,
) -> Response:
    """Run a translated request through the shared ``_forward`` and translate
    the answer back. The legacy path and the router's local leg share this one
    implementation; ``response_model_name`` is what the client is told."""
    model, host, port, variant = target

    # _forward reads the body from the request: hand it the translated one.
    # `_body` is Starlette's cache of an already-read body; `_receive` covers
    # a reader that goes to the stream (the same re-set chat_completions does).
    body = json.dumps(oai).encode()
    request._body = body

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = _receive

    resp = await _forward(
        request,
        model,
        host,
        port,
        "/v1/chat/completions",
        token,
        variant=variant,
        affinity_key=anthropic_affinity_key(req),
    )
    emit_thinking = thinking_requested(req)

    if isinstance(resp, StreamingResponse):
        if resp.status_code != 200:
            # An engine error on a streamed request is one JSON body, not SSE.
            # Draining it runs _forward's own end-of-stream bookkeeping.
            raw = b"".join([_as_bytes(c) async for c in resp.body_iterator])
            return anthropic_error(resp.status_code, upstream_error_message(raw))
        translator = AnthropicStreamTranslator(
            model=response_model_name,
            stop_sequences=req.stop_sequences,
            emit_thinking=emit_thinking,
        )
        return StreamingResponse(
            _anthropic_sse(resp.body_iterator, translator),
            status_code=200,
            media_type="text/event-stream",
            headers=sse_headers(),
            # _forward's unstarted-stream guard releases the rank and the
            # admission slot from this task when the client leaves early.
            background=resp.background,
        )

    if resp.status_code != 200:
        return anthropic_error(resp.status_code, upstream_error_message(bytes(resp.body)))
    try:
        completion = json.loads(resp.body)
    except ValueError:
        return anthropic_error(502, "upstream engine returned a non-JSON body")
    if not isinstance(completion, dict):
        return anthropic_error(502, "upstream engine returned an unexpected body")
    return JSONResponse(
        to_anthropic_message(
            completion,
            model=response_model_name,
            stop_sequences=req.stop_sequences,
            emit_thinking=emit_thinking,
        )
    )


def _as_bytes(chunk: Any) -> bytes:
    return chunk.encode() if isinstance(chunk, str) else bytes(chunk)


async def _anthropic_sse(
    upstream: AsyncIterable[Any], translator: AnthropicStreamTranslator
) -> AsyncIterator[bytes]:
    try:
        async for chunk in upstream:
            for frame in translator.feed_bytes(_as_bytes(chunk)):
                yield frame
            if translator.done:
                break
        for frame in translator.finish():
            yield frame
    finally:
        # Close _forward's generator whichever way we leave -- normal end,
        # an upstream error frame, or a client disconnect cancelling us --
        # so its shielded settle (counters, slot release, history) runs now.
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()


@router.post("/messages/count_tokens")
async def count_tokens(request: Request, token: TokenRow = Depends(require_bearer)) -> Response:
    """The prompt's size in the served model's own tokenizer.

    An estimate of what the engine will count: the chat template's framing
    tokens are not included, the same as the warden's own prompt accounting.
    """
    if await _router_enabled(request):
        return await messages_entry(request, token, count_tokens=True)
    return await count_tokens_legacy(request, token)


async def count_tokens_legacy(request: Request, token: TokenRow) -> Response:
    req = await _parse(request, CountTokensRequest)
    if not token_allows(token, req.model):
        raise HTTPException(403, f"token not allowed for model '{req.model}'")
    oai = _translate(req)
    target = await _resolve_target(request, req.model)
    return await _count_local(request, req, oai, target)


async def _count_local(
    request: Request, req: CountTokensRequest, oai: dict[str, Any], target: tuple[Any, ...]
) -> Response:
    model = target[0]
    n = await request.app.state.tokenizers.count(
        model.hf_repo,
        count_text(oai),
        fallback_repo=getattr(model, "tokenizer_repo", None),
    )
    return JSONResponse({"input_tokens": n})


def _owned_path_response(request: Request) -> Response | None:
    """What Starlette would have answered for a /v1 path the warden owns.

    The catch-all matches every /v1 path, so the two answers the router would
    otherwise swallow are reproduced here, router on or off: 405 for a known
    path with the wrong method, and the 307 trailing-slash redirect.
    """
    scope = request.scope
    me = scope.get("route")
    routes = [r for r in request.app.router.routes if r is not me]
    allow: set[str] = set()
    for route in routes:
        methods = getattr(route, "methods", None)
        match, _ = route.matches(scope)
        if match == Match.PARTIAL and methods:
            allow |= set(methods)
    if allow:
        return JSONResponse(
            {"detail": "Method Not Allowed"},
            status_code=405,
            headers={"Allow": ", ".join(sorted(allow))},
        )
    path = scope.get("path", "")
    if request.app.router.redirect_slashes and path != "/":
        redirect_scope = dict(scope)
        redirect_scope["path"] = path.rstrip("/") if path.endswith("/") else path + "/"
        for route in routes:
            match, _ = route.matches(redirect_scope)
            if match != Match.NONE:
                from starlette.datastructures import URL

                return RedirectResponse(url=str(URL(scope=redirect_scope)))
    return None


class _CatchAllRoute(_AnthropicErrorRoute):
    """The /v1 catch-all. What the warden itself answers is decided BEFORE the
    route's dependencies (``require_bearer``) run, so an unauthenticated caller
    sees exactly what it saw without this route: 405 with ``Allow`` and the 307
    trailing-slash redirect on a path the warden owns, and a plain 404 for
    ``OPTIONS`` to an unknown path (which is never relayed)."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            owned = _owned_path_response(request)
            if owned is not None:
                return owned
            return await handler(request)

        return route_handler

    async def handle(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["method"] not in (self.methods or ()):
            # Reached only when no warden route owns the path (owned routes
            # come first and answer their own 405): nothing here for this verb.
            await JSONResponse({"detail": "Not Found"}, status_code=404)(scope, receive, send)
            return
        await super().handle(scope, receive, send)


async def v1_catch_all(
    rest: str, request: Request, token: TokenRow = Depends(require_bearer)
) -> Response:
    """Any other /v1 path: Anthropic's, relayed -- or today's 404 (router off).

    Registered last on the router mounted after the OpenAI routes, so it only
    sees paths no other /v1 route owns.
    """
    if not await _router_enabled(request):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    return await catch_all_entry(request, token, rest)


router.add_api_route(
    "/{rest:path}",
    v1_catch_all,
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
    include_in_schema=False,
    route_class_override=_CatchAllRoute,
)


def find_routes_after_catch_all(app: Any) -> list[str]:
    """Paths of /v1 routes registered after the /v1 catch-all (they would be
    shadowed by it). Empty when the order is right."""
    routes = list(app.router.routes)
    idx = next(
        (i for i, r in enumerate(routes) if getattr(r, "path", "") == "/v1/{rest:path}"), None
    )
    if idx is None:
        return []
    return [
        r.path
        for r in routes[idx + 1 :]
        if str(getattr(r, "path", "")).startswith("/v1/") or getattr(r, "path", "") == "/v1"
    ]
