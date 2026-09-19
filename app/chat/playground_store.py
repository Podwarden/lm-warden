"""In-process cache for the `vw-playground` bearer-token plaintext.

The chat playground (S8 of the vllm-warden overhaul, see plan §S8) needs a
real bearer token to call `/v1/chat/completions` — that path is locked
behind ``require_bearer`` and we don't want the playground to invent a
JWT-only shortcut around the proxy's priority / accounting machinery.

The trade-off this module exists to negotiate:

* Tokens are stored as SHA-256 hashes in SQLite; the plaintext is only
  visible at create time (this is by design — see ``app/auth/bearer.py``
  and the rotate flow in ``app/db/repos/tokens.py``).
* The playground needs the plaintext to forge a Bearer header on the
  user's behalf without exposing the secret to the browser (one of the
  locked S8 risks: "do not put a bearer token in browser memory").

The compromise is this in-process singleton: when the chat page first
mounts, ``POST /api/chat/playground/ensure`` either reuses a previously
minted plaintext (cache hit) or rotates / creates a fresh `vw-playground`
token and stashes the plaintext here. The cache lives in
``app.state.playground_store`` so a container restart wipes it — the
ensure handler treats "row exists in DB but plaintext absent from cache"
as a recreate signal so a restart doesn't strand the playground.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

PLAYGROUND_TOKEN_NAME = "vw-playground"


@dataclass(frozen=True)
class PlaygroundSecret:
    """A `(token_id, plaintext)` pair recovered at create / rotate time."""

    token_id: str
    plaintext: str


class PlaygroundStore:
    """Thread-safe singleton for the playground plaintext.

    Storage is intentionally process-local: if we run more than one
    uvicorn worker in future (we don't today — see ``app/main.py``
    scheduler comment) this store must be swapped for something
    shared (Redis, mounted file, etc.). For the single-worker reality of
    the warden, an asyncio.Lock + dict is the right amount of machinery.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._secret: PlaygroundSecret | None = None

    async def get(self) -> PlaygroundSecret | None:
        async with self._lock:
            return self._secret

    async def set(self, token_id: str, plaintext: str) -> None:
        async with self._lock:
            self._secret = PlaygroundSecret(token_id=token_id, plaintext=plaintext)

    async def clear(self) -> None:
        async with self._lock:
            self._secret = None
