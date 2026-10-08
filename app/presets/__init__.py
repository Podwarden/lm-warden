"""Built-in tuning presets for the model settings page (S4).

A preset is a named bundle of PATCHable ModelRow fields that gets applied
in one shot to a model's settings (after operator confirmation) — the FE
calls ``GET /api/presets`` to populate the dropdown, then issues a normal
``PATCH /api/models/{id}/settings`` with the preset's ``settings`` dict
spread into the body. The backend stays dumb about what a preset is for;
preset *application* is a frontend concern.

We ship a small curated set of four presets keyed to common GPU
archetypes that we've seen in the field. Adding more is a 1-line JSON
edit; no schema migration required.

The list is loaded once at import time from ``builtin.json`` next to this
module. Tests can pass an explicit ``presets_path`` to ``load_presets``
for fixture coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Preset:
    """A named bundle of model-settings overrides.

    Attributes:
        id: Stable slug used as the React key + PATCH dropdown value.
        name: Short human label shown in the dropdown.
        description: One-line rationale shown in the confirm-and-diff popover.
        target_archetype: Free-text hardware tag (e.g. "2x A4000 24GB").
        settings: Dict of PATCHable ModelRow field names → values. Only
            keys that exist in ``_PATCHABLE_MODEL_FIELDS`` survive a PATCH;
            the FE filters out unknown keys before sending the request,
            but the backend's allowlist is the final guard.
    """

    id: str
    name: str
    description: str
    target_archetype: str
    settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "target_archetype": self.target_archetype,
            "settings": dict(self.settings),
        }


def load_presets(presets_path: Path | None = None) -> list[Preset]:
    """Load the built-in preset list from JSON.

    Returns a fresh list on every call so the route handler can hand back
    a non-shared payload (the per-request response is mutated by FastAPI's
    JSON encoder; sharing a frozen list would still be safe but the copy
    is cheap and removes a class of accidental coupling).

    Raises ``RuntimeError`` if the JSON is missing or malformed — the
    file is checked into the repo so a missing file is a packaging bug,
    not a runtime condition.
    """
    if presets_path is None:
        presets_path = Path(__file__).resolve().parent / "builtin.json"
    if not presets_path.exists():
        raise RuntimeError(f"presets file not found: {presets_path}")
    with open(presets_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise RuntimeError(f"presets file malformed: expected list, got {type(raw).__name__}")
    out: list[Preset] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise RuntimeError(f"preset entry malformed: {entry!r}")
        out.append(
            Preset(
                id=str(entry["id"]),
                name=str(entry["name"]),
                description=str(entry["description"]),
                target_archetype=str(entry["target_archetype"]),
                settings=dict(entry.get("settings") or {}),
            )
        )
    return out
