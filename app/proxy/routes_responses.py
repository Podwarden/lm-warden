"""OpenAI Responses API: ``POST /v1/responses`` (for Codex CLI).

The same shape as routes_messages.py: the route owns no forwarding of its own.
It validates the Responses body, turns it into an OpenAI chat body
(responses_translate.py), and runs it through the shared ``_forward`` as
``/v1/chat/completions`` -- so admission, priority, accounting, the live
registry, prefix-cache estimate, replica affinity, god mode and the runaway
detector behave exactly as they do for OpenAI traffic. Auth is
``require_bearer`` (``Authorization: Bearer``, ``X-LMWarden-Key``), routing is
``_resolve_target``: SERVED model names only, router on or off (router rules
are Claude-name globs with an Anthropic leg; there is nothing to relay to).

Every refusal, including the ones ``require_bearer`` raises before the handler
runs, answers in OpenAI's error envelope -- that is what ``_OpenAIErrorRoute``
is for (see routes_messages._AnthropicErrorRoute).

Registered BEFORE routes_messages.router in main.py, so the ``/v1/{rest:path}``
catch-all never sees this path.
"""

import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable, Coroutine, Mapping
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.db.repos.tokens import TokenRow
from app.proxy.auth import require_bearer, token_allows
from app.proxy.messages_clamp import clamp_max_tokens
from app.proxy.messages_translate import TranslationError, upstream_error_message
from app.proxy.responses_schemas import ResponsesRequest
from app.proxy.responses_translate import (
    ResponsesStreamTranslator,
    Translation,
    error_body,
    to_openai_request,
    to_response_object,
)
from app.proxy.routes import _forward, _resolve_target
from app.proxy.routes_messages import _as_bytes, _detail_text
from app.proxy.session_id import PROMPT_HASH_CHARS, extract_session, responses_first_user_text
from app.utils.sse import sse_headers

logger = logging.getLogger(__name__)


def openai_error(
    status: int, message: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(error_body(status, message), status_code=status, headers=headers)


class _OpenAIErrorRoute(APIRoute):
    """An APIRoute whose HTTPExceptions -- dependencies' included -- answer in
    the OpenAI error envelope, status and headers (Retry-After) kept."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except StarletteHTTPException as exc:
                return openai_error(exc.status_code, _detail_text(exc.detail), exc.headers)

        return route_handler


router = APIRouter(prefix="/v1", tags=["proxy"], route_class=_OpenAIErrorRoute)


def responses_affinity(
    body: dict[str, Any], headers: Mapping[str, str], token_id: str
) -> tuple[str | None, str, str | None]:
    """``(replica key, key source, session id)`` from the ORIGINAL body and
    headers (the translated chat body no longer carries ``prompt_cache_key``).

    The shared extractor first (Codex: ``session-id`` header, ``prompt_cache_key``);
    else a hash of the first real user turn -- ``<environment_context>``,
    developer and system items are skipped -- which is a guess at "same
    conversation", never a session id, and never feeds the prefix memory.
    """
    info = extract_session(body, headers)
    if info.id and info.source:
        return info.id, info.source, info.id
    text = responses_first_user_text(body.get("input"))[:PROMPT_HASH_CHARS]
    if text:
        return f"{token_id}\x00{text}", "prompt_hash", None
    return None, "none", None


async def _parse(request: Request) -> tuple[dict[str, Any], ResponsesRequest]:
    raw = await request.body()
    try:
        data = json.loads(raw) if raw else None
    except ValueError:
        raise HTTPException(400, "request body is not valid JSON") from None
    if not isinstance(data, dict):
        raise HTTPException(400, "request body must be a JSON object")
    try:
        return data, ResponsesRequest.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise HTTPException(400, f"{where}: {first['msg']}") from None


@router.post("/responses")
async def responses(request: Request, token: TokenRow = Depends(require_bearer)) -> Response:
    data, req = await _parse(request)
    if not token_allows(token, req.model):
        raise HTTPException(403, f"token not allowed for model '{req.model}'")
    try:
        tr = to_openai_request(req)
    except TranslationError as exc:
        raise HTTPException(400, str(exc)) from None
    if tr.dropped:
        logger.debug("responses: dropped %s", ",".join(tr.dropped))
    target = await _resolve_target(request, req.model)
    await clamp_max_tokens(request, target, tr.oai)
    key, source, session_id = responses_affinity(data, request.headers, token.id)
    return await _serve_local(request, token, req, tr, target, affinity=(key, source, session_id))


async def _serve_local(
    request: Request,
    token: TokenRow,
    req: ResponsesRequest,
    tr: Translation,
    target: tuple[Any, str, int, Any],
    *,
    affinity: tuple[str | None, str, str | None],
) -> Response:
    model, host, port, variant = target

    # _forward reads the body from the request: hand it the translated one.
    # `_body` is Starlette's cache of an already-read body; `_receive` covers
    # a reader that goes to the stream (the same re-set chat_completions does).
    body = json.dumps(tr.oai).encode()
    request._body = body

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = _receive
    key, source, session_id = affinity

    resp = await _forward(
        request,
        model,
        host,
        port,
        "/v1/chat/completions",
        token,
        variant=variant,
        affinity_key=key,
        affinity_source=source,
        session_id=session_id,
    )

    if isinstance(resp, StreamingResponse):
        if resp.status_code != 200:
            # An engine error on a streamed request is one JSON body, not SSE.
            # Draining it runs _forward's own end-of-stream bookkeeping.
            raw = b"".join([_as_bytes(c) async for c in resp.body_iterator])
            return openai_error(resp.status_code, upstream_error_message(raw))
        translator = ResponsesStreamTranslator(model=req.model, tr=tr)
        return StreamingResponse(
            _responses_sse(resp.body_iterator, translator),
            status_code=200,
            media_type="text/event-stream",
            headers=sse_headers(),
            # _forward's unstarted-stream guard releases the rank and the
            # admission slot from this task when the client leaves early.
            background=resp.background,
        )

    if resp.status_code != 200:
        return openai_error(resp.status_code, upstream_error_message(bytes(resp.body)))
    try:
        completion = json.loads(resp.body)
    except ValueError:
        return openai_error(502, "upstream engine returned a non-JSON body")
    if not isinstance(completion, dict):
        return openai_error(502, "upstream engine returned an unexpected body")
    return JSONResponse(to_response_object(completion, model=req.model, tr=tr))


async def _responses_sse(
    upstream: AsyncIterable[Any], translator: ResponsesStreamTranslator
) -> AsyncIterator[bytes]:
    try:
        try:
            async for chunk in upstream:
                for frame in translator.feed_bytes(_as_bytes(chunk)):
                    yield frame
                if translator.done:
                    break
            for frame in translator.finish():
                yield frame
        except Exception as exc:  # noqa: BLE001 -- the stream must end with one terminal event
            # Type only: the message could quote request or reply content.
            logger.warning("responses: stream translation failed (%s)", type(exc).__name__)
            for frame in translator.fail("response translation failed"):
                yield frame
    finally:
        # Close _forward's generator whichever way we leave -- normal end, an
        # upstream error frame, or a client disconnect cancelling us -- so its
        # shielded settle (counters, slot release, history) runs now.
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()
