"""``X-LMW-Request-Id``: the id of the request's history row, on every proxied
response, so a client (the test agent) can read its turn back from the
sessions API."""

from __future__ import annotations

from starlette.responses import Response

REQUEST_ID_HEADER = "X-LMW-Request-Id"


def with_request_id(resp: Response, request_id: str | None) -> Response:
    """Set the header in place (when there is an id) and return ``resp``."""
    if request_id:
        resp.headers[REQUEST_ID_HEADER] = request_id
    return resp


#: The sensitive data guard's verdict (app/guard/proxy.py), carried with the request id so a
#: translated answer (/v1/messages, /v1/responses) keeps it.
GUARD_HEADER = "X-LMWarden-Guard"


def copy_request_id(src: Response, dst: Response) -> Response:
    """Carry ``src``'s request id and guard headers (if any) onto a response built from it."""
    guard = src.headers.get(GUARD_HEADER)
    if guard:
        dst.headers[GUARD_HEADER] = guard
    return with_request_id(dst, src.headers.get(REQUEST_ID_HEADER))


def guard_blocked(resp: Response) -> bool:
    """The answer is LM Warden's own block reply, not the model's."""
    return (resp.headers.get(GUARD_HEADER) or "").startswith("blocked")


#: The model name a guard block reply carries in every dialect.
GUARD_MODEL = "lm-warden-guard"
