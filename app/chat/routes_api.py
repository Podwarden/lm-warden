"""Chat-playground HTTP surface — JWT-authed wrapper around `/v1/chat/completions`.

Three endpoints land here:

* ``POST /api/chat/playground/ensure``  — idempotently mint the
  ``vw-playground`` system token; cache the plaintext server-side.
* ``POST /api/chat/completions``        — JWT-authed SSE proxy that
  forges the playground Bearer header server-side and streams upstream
  tokens back to the browser.
* ``GET  /api/admin/active-requests``   — counter-only diagnostic for
  the Playwright happy-path leak test (MED risk in the S8 plan).

The proxy deliberately does **not** ship the bearer plaintext to the
browser (the "do not put a bearer token in browser memory" risk in the
S8 plan); the playground UI uses its session JWT to call this endpoint
and the server does the rest.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth.bearer import generate_bearer_token
from app.auth.deps import require_jwt
from app.chat.playground_store import (
    PLAYGROUND_TOKEN_NAME,
    PlaygroundSecret,
    PlaygroundStore,
)
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo

router = APIRouter(tags=["chat"])


# --- Ensure-token ------------------------------------------------------------


class PlaygroundEnsureResponse(BaseModel):
    token_id: str
    """The DB id of the `vw-playground` row currently backing the playground."""

    created: bool
    """True iff this call minted a new row (or recreated one after cache miss)."""


def _get_store(request: Request) -> PlaygroundStore:
    return request.app.state.playground_store  # type: ignore[no-any-return]


async def _delete_other_playground_rows(repo: TokenRepo, keep_id: str | None) -> None:
    """Drop any extra rows named `vw-playground` so the table stays single-row.

    The ensure flow is called once per chat page mount; a race between two
    parallel visits could in theory produce two rows. We don't lock on the
    DB layer (no UNIQUE constraint on name — `tokens` is supposed to allow
    duplicate display names) so we sweep stale duplicates here.
    """
    rows = await repo.list_all()
    for row in rows:
        if row.name == PLAYGROUND_TOKEN_NAME and row.id != keep_id:
            await repo.delete(row.id)


async def _mint_playground_token(
    request: Request, repo: TokenRepo,
) -> PlaygroundSecret:
    """Create a fresh vw-playground row, cache the plaintext, return both."""
    plaintext = generate_bearer_token()
    token_id = secrets.token_hex(16)
    # 0 = never expires; the operator can rotate manually from /tokens if
    # they want a hard cap. The playground is meant to be persistent.
    await repo.create(
        token_id=token_id,
        name=PLAYGROUND_TOKEN_NAME,
        plaintext=plaintext,
        expires_in_days=0,
    )
    secret = PlaygroundSecret(token_id=token_id, plaintext=plaintext)
    await _get_store(request).set(token_id, plaintext)
    return secret


@router.post(
    "/api/chat/playground/ensure",
    response_model=PlaygroundEnsureResponse,
    status_code=status.HTTP_200_OK,
)
async def ensure_playground_token(
    request: Request, _user: str = Depends(require_jwt),
) -> PlaygroundEnsureResponse:
    """Idempotently mint the `vw-playground` system token.

    Cases:
      1. Plaintext cached in-process AND DB row still present → return as-is.
      2. Plaintext cached but DB row gone (operator deleted it from /tokens)
         → mint a new one, replace the cache.
      3. Plaintext NOT cached (fresh container) AND DB row present → we have
         no way to recover the plaintext, so drop the stale row and recreate.
      4. Nothing exists at all → mint a fresh row.

    All four paths converge on "one row named `vw-playground` exists AND its
    plaintext is in the cache" before returning. ``created`` distinguishes
    paths 1 from the rest so the UI can hint at "first visit" telemetry —
    not currently used in copy but cheap to expose.
    """
    store = _get_store(request)
    cached = await store.get()
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)

        if cached is not None:
            existing = await repo.get(cached.token_id)
            if existing is not None and existing.name == PLAYGROUND_TOKEN_NAME:
                # Case 1 — cache + DB agree. Best path.
                await _delete_other_playground_rows(repo, keep_id=cached.token_id)
                return PlaygroundEnsureResponse(
                    token_id=cached.token_id, created=False,
                )
            # Case 2 — cache is stale. Fall through to a fresh mint.
            await store.clear()

        # Cases 3 + 4 — sweep any DB row(s) named `vw-playground` (we can't
        # recover their plaintext) and mint a fresh one.
        await _delete_other_playground_rows(repo, keep_id=None)
        secret = await _mint_playground_token(request, repo)
    return PlaygroundEnsureResponse(token_id=secret.token_id, created=True)


# --- Chat completions proxy --------------------------------------------------


class ChatMessage(BaseModel):
    role: str = Field(min_length=1, max_length=32)
    content: str


class ChatCompletionsRequest(BaseModel):
    """Subset of the OpenAI chat-completions schema the playground supports.

    We deliberately accept only the knobs the UI exposes (model, messages,
    temperature, max_tokens, stream). Extra fields are dropped — if the
    operator wants the full schema they should hit /v1 directly.
    """

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=32_768)
    stream: bool = True


@router.post("/api/chat/completions")
async def chat_completions(
    body: ChatCompletionsRequest,
    request: Request,
    _user: str = Depends(require_jwt),
) -> StreamingResponse:
    """SSE-stream completions through the proxy using the playground token.

    The browser holds NO bearer secret — we read the cached plaintext
    server-side, forge the Authorization header, and stream the upstream
    response back. Cancellation: when the browser aborts via
    ``AbortController``, FastAPI propagates ``asyncio.CancelledError`` into
    the streaming generator; the ``finally`` block closes both the upstream
    response and the httpx client, which sends a TCP FIN to vLLM and frees
    the priority slot via the regular proxy machinery.

    Calls the internal `/v1/chat/completions` route via loopback so the
    real proxy pipeline (priority scheduler, token usage
    rollup, error enrichment) runs unmodified. We don't shortcut into
    ``_forward`` because that would skip the bearer validation we want
    as defence-in-depth — if the playground token is somehow stale we'd
    rather see a clean 401 than a bypass.
    """
    secret = await _get_store(request).get()
    if secret is None:
        # The UI must call /api/chat/playground/ensure on mount; missing
        # cache here means either the UI skipped that step (bug) or the
        # container restarted between ensure and send (race). The 409 is
        # actionable — the FE retries ensure() and replays the request.
        raise HTTPException(
            status_code=409,
            detail=(
                "playground token not initialised; "
                "POST /api/chat/playground/ensure first"
            ),
        )

    settings = request.app.state.settings
    upstream_url = (
        f"http://127.0.0.1:{settings.bind_port}/v1/chat/completions"
    )

    # Force-stream true regardless of payload — the playground UI only
    # supports streaming and the UX assumes incremental tokens. Disabling
    # streaming via the wire format would bypass our cancellation path.
    payload = body.model_dump()
    payload["stream"] = True

    counter = request.app.state.chat_active_requests
    counter_token = await counter.enter()

    client = httpx.AsyncClient(timeout=None)
    upstream_req = client.build_request(
        "POST",
        upstream_url,
        content=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {secret.plaintext}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    try:
        resp = await client.send(upstream_req, stream=True)
    except BaseException:
        await client.aclose()
        await counter.exit(counter_token)
        raise

    if resp.status_code >= 400:
        # Upstream rejected the request before any SSE bytes — drain the
        # body once, close, and surface the status to the UI. We don't
        # try to translate vLLM error envelopes; the playground tolerates
        # a generic detail string here.
        try:
            err_bytes = await resp.aread()
        except Exception:
            err_bytes = b""
        await resp.aclose()
        await client.aclose()
        await counter.exit(counter_token)
        detail: Any
        try:
            detail = json.loads(err_bytes)
        except Exception:
            detail = err_bytes.decode("utf-8", errors="replace")
        raise HTTPException(status_code=resp.status_code, detail=detail)

    async def _stream() -> Any:
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            # Always run, even when the consumer aborts (CancelledError)
            # or the connection is dropped. Closing the response sends a
            # TCP FIN to the upstream proxy which in turn releases the
            # scheduler slot in app/proxy/routes.py:_forward's finally.
            await resp.aclose()
            await client.aclose()
            await counter.exit(counter_token)

    return StreamingResponse(
        _stream(),
        status_code=resp.status_code,
        headers={
            "content-type": resp.headers.get(
                "content-type", "text/event-stream",
            ),
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
        },
    )


# --- Diagnostic --------------------------------------------------------------


class ActiveRequestsResponse(BaseModel):
    count: int


@router.get(
    "/api/admin/active-requests",
    response_model=ActiveRequestsResponse,
)
async def active_requests(
    request: Request, _user: str = Depends(require_jwt),
) -> ActiveRequestsResponse:
    """Read-only counter of in-flight playground streams.

    Plaintext counter, no per-request data. Used by the S8 Playwright
    happy-path to assert "abort during stream → counter returns to 0"
    so a future regression in the cancellation path is caught in CI.
    """
    return ActiveRequestsResponse(
        count=request.app.state.chat_active_requests.count(),
    )
