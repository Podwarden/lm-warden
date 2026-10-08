"""HTTP routes for built-in tuning presets (S4).

Single read-only endpoint: ``GET /api/presets``. Auth-gated like every
other ``/api/*`` route. The response shape is deliberately small —
``{presets: [{id, name, description, target_archetype, settings}]}`` —
because the FE renders the dropdown directly from this payload and
applies the ``settings`` dict via the existing
``PATCH /api/models/{id}/settings`` endpoint.

Presets are not user-editable today; we ship four built-ins and add to
them via a JSON edit + redeploy. If/when operator-owned presets become a
requirement, this module is where the POST/DELETE handlers land — they
will write to a new ``presets`` table without touching this read path.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.auth.deps import require_jwt
from app.presets import Preset, load_presets

router = APIRouter(prefix="/api/presets", tags=["presets"])


class PresetEntry(BaseModel):
    """One preset row, exactly the JSON shape of :class:`app.presets.Preset`."""

    id: str = Field(..., description="Stable slug, e.g. 'a4000-tight-awq'.")
    name: str = Field(..., description="Short human label.")
    description: str = Field(..., description="One-line rationale for the popover.")
    target_archetype: str = Field(..., description="Free-text hardware tag.")
    settings: dict[str, Any] = Field(
        ...,
        description=(
            "Map of PATCHable ModelRow field names to values. The FE spreads "
            "this dict into the body of PATCH /api/models/{id}/settings."
        ),
    )


class PresetsResponse(BaseModel):
    """Envelope for ``GET /api/presets``."""

    presets: list[PresetEntry]


# Cache the loaded list at import time — the JSON file is shipped with the
# package and never changes at runtime. Re-reading on every request would
# be cheap but pointless.
_BUILTIN_PRESETS: list[Preset] = load_presets()


@router.get("", response_model=PresetsResponse)
async def list_presets(_user: str = Depends(require_jwt)) -> PresetsResponse:
    """Return the list of built-in tuning presets.

    The list is loaded once at process start from ``app/presets/builtin.json``
    and held in memory; this handler always returns the same payload until
    the process is restarted. Auth is required because the response leaks
    nothing sensitive but every ``/api/*`` route is JWT-gated by policy.
    """
    return PresetsResponse(
        presets=[PresetEntry(**p.to_dict()) for p in _BUILTIN_PRESETS],
    )
