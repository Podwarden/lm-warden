"""The guard every token-authenticated SSE stream runs inside.

Every stream an admin token can open runs under ``guard_stream(request,
key)``, where ``key`` is the request's stream-registry key
(app/auth/deps.py::stream_key): ``admin_token:<id>`` for an admin token (spec
2026-09-19, decision 9), ``session:<username>`` for a browser session. That is
the four ticket streams behind ``require_sse_ticket`` -- god mode, model logs,
live stats and header metrics -- whose dependency returns the key, plus model
pull progress, the chat2 turn streams (``guarded``) and the chat playground's
completion proxy (app/chat/routes_api.py, added #258), which authenticate with
``require_jwt`` and read the key off the request. Seven streams, and every
stream the app opens under a credential is one of them -- including the chat2
turn route's replay of an already-recorded turn, which is a single pre-computed
frame with nothing to cancel and is guarded anyway, so that sentence needs no
footnote. The guard does two things:

* **Registers the stream's task** in app.state.stream_registry under ``key``
  for as long as the stream is open, so ``cancel_user(key)`` ends it: a
  logout ends that session's streams, and revoking an admin token -- or
  refreshing it with no grace -- ends exactly that token's streams. Keyed by
  token, not by owner, so a browser logout does not cut a script's stream.

* **Re-checks an admin token** every ``TOKEN_RECHECK_INTERVAL_S`` while the
  stream is open, and ends the stream when the token has expired, been
  revoked, had its rotation grace run out, been paused or been deleted.
  Authentication happens once, at connect; without the re-check a stream
  opened with a root-equivalent credential would outlive the credential's
  term. The stream ends cleanly: the generator stops and the response is
  completed, as if the server had nothing more to say. (A failed re-check
  read -- a locked database, say -- is logged and retried at the next
  interval rather than cutting a healthy stream.) A session's ticket stream
  is not re-checked: a ticket is single-use and minted by a live session,
  and logout ends it.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Request

from app.auth.deps import admin_token_id, token_refusal
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo, sqlite_utc_now

logger = logging.getLogger(__name__)

#: How often an open admin-token stream re-reads its token. Read at every
#: wait, so a test can monkeypatch a sub-second value.
TOKEN_RECHECK_INTERVAL_S: float = 60.0


async def _recheck(app: Any, token_id: str, stop: Callable[[], None]) -> None:
    """Every TOKEN_RECHECK_INTERVAL_S: is the token still usable? If not, stop."""
    while True:
        await asyncio.sleep(TOKEN_RECHECK_INTERVAL_S)
        try:
            async with open_db(app.state.settings.db_path) as db:
                row = await TokenRepo(db).get(token_id)
        except Exception:
            logger.warning(
                "stream re-check of admin token %s failed; retrying", token_id, exc_info=True
            )
            continue
        if row is None or token_refusal(row, sqlite_utc_now()) is not None:
            logger.info("admin token %s is no longer valid; ending its stream", token_id)
            stop()
            return


@asynccontextmanager
async def guard_stream(request: Request, key: str) -> AsyncIterator[None]:
    """Run an SSE generator's body under ``key`` (module docstring).

    Enter it inside the generator, so the task registered is the one that
    iterates it.
    """
    task = asyncio.current_task()
    if task is None:  # pragma: no cover -- a generator always runs in a task
        raise RuntimeError("guard_stream needs a running task")
    registry = request.app.state.stream_registry
    registry.register(key, task)
    stopped = False

    def stop() -> None:
        nonlocal stopped
        stopped = True
        task.cancel()

    token_id = admin_token_id(key)
    watchdog = (
        asyncio.create_task(_recheck(request.app, token_id, stop))
        if token_id is not None
        else None
    )
    try:
        yield
    except asyncio.CancelledError:
        # Ours, and nobody else's (a revoke, a disconnect, a shutdown): end
        # the stream normally so the response is completed. Anyone else's
        # cancellation propagates as before.
        if stopped and task.uncancel() == 0:
            return
        raise
    else:
        # The body swallowed our cancellation (a generator that returns on
        # CancelledError during its sleep): balance it all the same.
        if stopped and task.cancelling():
            task.uncancel()
    finally:
        if watchdog is not None:
            watchdog.cancel()
        registry.unregister(key, task)


async def guarded(
    request: Request, key: str, body: AsyncIterator[bytes]
) -> AsyncIterator[bytes]:
    """``body`` under ``guard_stream(request, key)`` -- for a stream whose
    generator is not the route's own (the chat2 turn subscribers)."""
    async with guard_stream(request, key):
        try:
            async for chunk in body:
                yield chunk
        finally:
            aclose = getattr(body, "aclose", None)
            if aclose is not None:
                await aclose()
