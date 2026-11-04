"""Platform default + per-key override → a key's effective guard settings.

A key's stored ``guard``: None = inherit the platform; ``{"off": true}`` = no guard for this key;
``{"override": {...}}`` = the platform settings with these fields replaced (a key can also turn
the guard on when the platform has none). ``source`` says where each field came from."""

from __future__ import annotations

from typing import Any

from app.guard.config import validate

FIELDS = ("mode", "packs", "rules", "overrides", "restore", "trailer", "fail")


def validate_key(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError('guard must be null, {"off": true} or {"override": {...}}')
    if raw.get("off") is True and set(raw) == {"off"}:
        return {"off": True}
    if set(raw) == {"override"} and isinstance(raw["override"], dict):
        unknown = set(raw["override"]) - set(FIELDS)
        if unknown:
            raise ValueError(f"unknown guard fields: {sorted(unknown)}")
        validate({**raw["override"]})
        return {"override": dict(raw["override"])}
    raise ValueError('guard must be null, {"off": true} or {"override": {...}}')


def effective(platform: dict[str, Any] | None, key: dict[str, Any] | None) -> dict[str, Any] | None:
    if key and key.get("off"):
        return None
    over = (key or {}).get("override") or {}
    if platform is None and not over:
        return None
    base = dict(platform or {})
    out = validate({**base, **over})  # every field in force, defaults filled in
    src = {k: ("key" if k in over else "platform" if k in base else "default") for k in FIELDS}
    out["source"] = src
    return out


def compiled_settings(eff: dict[str, Any] | None) -> dict[str, Any] | None:
    """The effective settings without ``source``, for compile_guard."""
    if eff is None:
        return None
    return {k: v for k, v in eff.items() if k != "source"}
