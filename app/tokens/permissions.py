"""Per-key permissions (owner decision 2026-11-26, migration 0066).

``prompt_lens``: the key's holder may use Prompt Lens on their own traffic (the key area's
pages, the Prompt Lens API, the Prompt Lens MCP tools). Prompt Lens for a holder also needs
the platform switch (app/proxy/prompt_lens_switch.py). ``engine_facts``: the key may read the
static engine facts of the models it may use. Both are set by the admin, on create and edit,
and carried to a rotated key. A holder sees their state read-only.
"""

from __future__ import annotations

import time

from fastapi import Request

from app.auth.admin_audit import route_template
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo
from app.utils.client_ip import client_ip

#: The refusal of a key without the permission: the one 404 of the Prompt Lens routes.
FLAGS = ("prompt_lens", "engine_facts")


async def prompt_lens_allowed(request: Request, token_id: str) -> bool:
    """The platform switch is on AND this key may use Prompt Lens. A missing key is False."""
    from app.proxy import prompt_lens_switch

    if not prompt_lens_switch.enabled(request.app):
        return False
    async with open_db(request.app.state.settings.db_path) as db:
        row = await TokenRepo(db).get(token_id)
    return row is not None and bool(row.prompt_lens)


def audit_change(request: Request, *, token_id: str, username: str, flag: str, on: bool) -> None:
    """One audit row for a permission an admin switched: the key, the actor, and the flag with
    its new value after the route template. Written for an admin-token request too: the
    middleware's own row for it names only the route."""
    writer = getattr(request.app.state, "admin_audit", None)
    if writer is None:
        return
    writer.record(
        ts=time.time(),
        token_id=token_id,
        username=username,
        method=request.method,
        path=f"{route_template(request.scope)} key.{flag}:{'on' if on else 'off'}"[:300],
        status=200,
        duration_ms=0,
        client_ip=client_ip(request),
        peer_ip=request.client.host if request.client else None,
    )
