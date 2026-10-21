"""One redactor for 422 validation errors, shared by every route.

FastAPI's default ``RequestValidationError`` handler returns
``jsonable_encoder(exc.errors())``, and each error carries ``input`` -- the
offending value. For a model-level ``model_validator`` that is the WHOLE body
(``extra_args`` / ``extra_env`` values that may hold secrets), so a 422 answer
echoes what the client sent. The OVERSEER decision (planner iter 40) is to keep
exactly ``loc`` / ``msg`` / ``type`` per error -- in the input order -- and drop
``input``, ``ctx`` and ``url`` and every other key. ``loc`` + ``msg`` still say
which field failed and why; validator messages name keys and flags, not values.

This is the shared redactor (``redact_validation_errors``) and the app-wide
handler built on it (``redacted_validation_exception_handler``). Registering the
handler in ``app/main.py`` is a separate task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def redact_validation_errors(errors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``loc`` / ``msg`` / ``type`` per error, in order; everything else is dropped."""
    return [{"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]} for err in errors]


async def redacted_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """FastAPI's 422 shape (``{'detail': [...]}``), redacted: no ``input`` /
    ``ctx`` / ``url`` and no other key survives, so the request body is never
    echoed back."""
    return JSONResponse(
        status_code=422,
        content={"detail": redact_validation_errors(exc.errors())},
    )
