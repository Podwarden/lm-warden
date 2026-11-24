"""Alias parameter overrides on the final request body (D4c).

Pure: no app, no engine, no DB. The settings an alias carries are applied
here (params) and beside this module (guard / thinking) on the final
chat/completions body, right after the thinking-policy block.

D4c modes:
- ``default``: used only when the client sent nothing usable for that
  parameter. A wrong-typed client value (string max_tokens, null temperature)
  counts as absent, never crashes.
- ``force``: always replaces.
- ``cap``: numeric only (min(client, value), or ``value`` when the client
  sent nothing usable). On a non-numeric parameter it behaves conservatively:
  like ``default`` (never overrides a valid client value). ``max_tokens`` also
  honours ``max_completion_tokens`` when the client sent that instead.
"""

from __future__ import annotations

import math
from typing import Any

_KWARGS_PREFIX = "chat_template_kwargs."

# Mirrors the D4c table in app/aliases/schemas.py: the runtime only needs the
# numeric/non-numeric split (cap is numeric-only); the write-time validator has
# already refused anything else.
_NUMERIC_PARAMS = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
)
_INT_PARAMS = ("max_tokens", "top_k", "seed")


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _entry(value: Any) -> tuple[Any, str]:
    if isinstance(value, dict):
        return value.get("value"), str(value.get("mode") or "default")
    return value, "default"


def _coerce(key: str, v: Any) -> Any:
    if key in _INT_PARAMS and _is_number(v):
        try:
            return int(v)
        except (OverflowError, ValueError):
            return v
    return v


def _apply_max_tokens(body: dict[str, Any], value: Any, mode: str) -> bool:
    # ``max_tokens`` honours ``max_completion_tokens`` when the client sent
    # that instead: the client's cap is whatever it sent under either name.
    if not _is_number(body.get("max_tokens")) and "max_completion_tokens" in body:
        field = "max_completion_tokens"
    else:
        field = "max_tokens"
    client_val = body.get(field)
    if not _is_number(client_val):
        client_val = None  # absent, or a non-number: treated as absent
    value = _coerce("max_tokens", value)

    if mode == "default":
        if client_val is not None:
            return False
        new = value
    elif mode == "force":
        new = value
    else:  # cap
        new = _coerce("max_tokens", min(client_val, value)) if client_val is not None else value

    if body.get(field) == new:
        return False
    body[field] = new
    return True


def _apply_kwarg(body: dict[str, Any], suffix: str, value: Any, mode: str) -> bool:
    if not suffix:
        return False
    if mode not in ("default", "force"):
        mode = "default"  # cap is not for kwarg entries (write-time refuses them too)
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    if mode == "default" and suffix in kwargs:
        return False
    if kwargs.get(suffix) == value and isinstance(body.get("chat_template_kwargs"), dict):
        return False
    kwargs[suffix] = value
    body["chat_template_kwargs"] = kwargs
    return True


def _apply_param(body: dict[str, Any], key: str, value: Any, mode: str) -> bool:
    client_val = body.get(key)
    if mode not in ("default", "force", "cap"):
        mode = "default"

    if mode == "force":
        new = _coerce(key, value)
        if body.get(key) == new:
            return False
        body[key] = new
        return True

    if key in _NUMERIC_PARAMS:
        usable = _is_number(client_val)
        if mode == "default":
            if usable:
                return False
            new = _coerce(key, value)
        else:  # cap
            new = _coerce(key, min(client_val, value)) if usable else _coerce(key, value)
    else:
        # reasoning_effort and anything else non-numeric: default only (cap is
        # numeric-only and already refused at write time)
        if client_val is not None and (
            isinstance(client_val, str) or isinstance(client_val, bool) or _is_number(client_val)
        ):
            return False  # the client sent a usable value
        new = value

    if body.get(key) == new:
        return False
    body[key] = new
    return True


def apply_alias_params(body_json: dict[str, Any], params: dict[str, Any]) -> bool:
    """Apply the alias's parameter overrides to the final body, in place.

    ``params`` is the stored map ``{key: {"value": ..., "mode": ...}}``. Returns
    True when anything changed. Raises nothing.
    """
    if not isinstance(body_json, dict) or not params:
        return False
    changed = False
    for key, entry in params.items():
        if not isinstance(key, str) or not key:
            continue
        value, mode = _entry(entry)
        if key == "max_tokens":
            changed = _apply_max_tokens(body_json, value, mode) or changed
        elif key.startswith(_KWARGS_PREFIX):
            changed = _apply_kwarg(body_json, key[len(_KWARGS_PREFIX) :], value, mode) or changed
        else:
            changed = _apply_param(body_json, key, value, mode) or changed
    return changed
