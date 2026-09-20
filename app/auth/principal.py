"""Who is asking, as a value rather than as request state.

``request.state.principal`` is a string set by ``app/auth/deps.py``
(``"session"`` or ``"admin_token:<id>"``) and ``is_session_principal(request)``
reads it. That works for a route and only for a route, which is the shape of the
problem #256/#260/#261 kept running into: the rule being enforced is "who
authored the dangerous fields on this row", and a row is written from places
where no ``Request`` exists (the watchdog's restart sweep, boot reconciliation,
the pull task).

``Principal`` is that answer as a value, so ``app/models/writer.py`` can take it
as an argument and the no-principal callers can be named -- ``SYSTEM`` -- and
refused by construction rather than by inspection.
"""
from dataclasses import dataclass
from typing import Literal

from fastapi import Request

from app.auth.deps import SESSION_PRINCIPAL, admin_token_id


@dataclass(frozen=True)
class Principal:
    """The credential behind a write.

    ``kind`` is the whole space: the admin-token spec names two principals and
    calls scopes/RBAC a non-goal
    (docs/superpowers/specs/2026-09-19-admin-tokens-design.md), and ``system``
    is not a credential at all -- it is the absence of one.
    """

    kind: Literal["session", "admin_token", "system"]
    #: The token id for ``admin_token``; None otherwise.
    id: str | None = None

    @property
    def is_session(self) -> bool:
        return self.kind == "session"


#: The watchdog, boot reconciliation and the pull task. Deliberately NOT
#: accepted by ``apply_model_change``: these callers write only the state-machine
#: columns, through the named ``ModelRepo`` methods that own them.
SYSTEM = Principal("system")


def principal_of(request: Request) -> Principal:
    """The ``Principal`` behind an authenticated ``/api`` request.

    Raises rather than defaulting: every caller sits behind ``require_jwt`` /
    ``require_session``, which set ``request.state.principal`` before any route
    body runs, so an unset value means the route lost its dependency. Defaulting
    would silently pick one of the two answers, and both are wrong.
    """
    raw = getattr(request.state, "principal", None)
    if raw == SESSION_PRINCIPAL:
        return Principal("session")
    token_id = admin_token_id(raw or "")
    if token_id:
        return Principal("admin_token", id=token_id)
    raise RuntimeError(
        "no principal on the request: this route must depend on require_jwt "
        "or require_session, which set request.state.principal"
    )
