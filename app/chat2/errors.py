"""The single HTTP error envelope for `/api/chat2/*` (spec §4.5).

Every non-2xx response under the chat2 prefix carries the same body shape::

    {"detail": {"code": "<machine code>", "message": "<human text>"}}

so the frontend can branch on `detail.code` and never has to parse prose or
guess whether `detail` is a string or an object. The codes in use:

* pre-flight/route errors -- `not_found`, `invalid`, `too_large`,
  `unsupported_media`, `quota_exceeded`, `bad_token`;
* turn-specific errors -- `turn_in_flight`, `budget_blocked`, `context_full`,
  `model_not_loaded`, `tools_unsupported`, `too_many_images`,
  `attachment_missing`, `duplicate_tool_result`.

`app/chat2/body_limit.py` writes the same envelope by hand (`too_large`): it
is a raw-ASGI middleware that runs ahead of routing, so it cannot raise an
`HTTPException`. Keep the two in step.
"""

from __future__ import annotations

from fastapi import HTTPException


def api_error(status: int, code: str, message: str) -> HTTPException:
    """Build an `HTTPException` whose detail is the chat2 `{code, message}` envelope."""
    return HTTPException(status, {"code": code, "message": message})
