"""Tests for ``app.presets.load_presets`` — the JSON loader behind the
built-in preset list shipped with the warden.

These are pure unit tests: no fixtures, no HTTP, no auth. They pin:

* the built-in JSON is well-formed and yields exactly four presets,
* every required field is non-empty on every entry,
* the loader raises ``RuntimeError`` for missing/malformed inputs,
* the ``Preset.to_dict()`` shape matches the API contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.presets import Preset, load_presets


def test_builtin_presets_load() -> None:
    """``load_presets()`` returns the four curated entries with required keys.

    Adding a fifth (or removing one) is a deliberate product decision —
    update this assertion together with ``app/presets/builtin.json``.
    """
    presets = load_presets()
    assert len(presets) == 4
    ids = [p.id for p in presets]
    # Lock the four shipping IDs so a typo in the JSON gets caught.
    assert set(ids) == {
        "a4000-tight-awq",
        "h100-single-shot",
        "dev-tiny",
        "moe-balanced",
    }
    for p in presets:
        assert p.id
        assert p.name
        assert p.description
        assert p.target_archetype
        # Each preset must apply at least one knob — an empty settings dict
        # would be a no-op the FE would render as "Apply preset" → "No
        # changes — your draft already matches", which is a usability bug.
        assert p.settings, f"preset {p.id!r} has empty settings"


def test_load_presets_to_dict_round_trip() -> None:
    """``Preset.to_dict()`` returns a JSON-compatible mapping with all fields.

    The FE route handler echoes this dict shape verbatim; the test pins that
    no field is silently dropped during serialization.
    """
    p = Preset(
        id="x",
        name="X",
        description="d",
        target_archetype="t",
        settings={"gpu_memory_utilization": 0.9},
    )
    d = p.to_dict()
    assert d == {
        "id": "x",
        "name": "X",
        "description": "d",
        "target_archetype": "t",
        "settings": {"gpu_memory_utilization": 0.9},
    }


def test_load_presets_missing_file(tmp_path: Path) -> None:
    """A missing JSON file is a packaging bug, surfaced as ``RuntimeError``."""
    p = tmp_path / "nope.json"
    with pytest.raises(RuntimeError, match="not found"):
        load_presets(p)


def test_load_presets_malformed_root(tmp_path: Path) -> None:
    """A non-list JSON root raises ``RuntimeError`` — strict shape, no auto-coerce."""
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(RuntimeError, match="malformed"):
        load_presets(p)


def test_load_presets_malformed_entry(tmp_path: Path) -> None:
    """An entry that isn't a dict raises ``RuntimeError`` mid-load."""
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(["not-a-dict"]))
    with pytest.raises(RuntimeError, match="entry malformed"):
        load_presets(p)


def test_load_presets_custom_file(tmp_path: Path) -> None:
    """The loader accepts an explicit path — used by tests + future endpoints."""
    p = tmp_path / "custom.json"
    p.write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "name": "X",
                    "description": "d",
                    "target_archetype": "t",
                    "settings": {"max_model_len": 4096},
                }
            ]
        )
    )
    got = load_presets(p)
    assert len(got) == 1
    assert got[0].id == "x"
    assert got[0].settings == {"max_model_len": 4096}
