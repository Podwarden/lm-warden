"""Unit tests for the bundled vLLM engine registry.

The registry answers two questions from the image's baked-in installs:
  - which version ids are present (``list_bundled_engines``), and
  - which filesystem binary a given id points at (``bundled_engine_bin``).

The root is read from ``VW_ENGINES_ROOT`` at call time, so tests point it at
a tmp dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.runtime.backends.vllm import engines


def _make_engine(
    root: Path, name: str, *, executable: bool = True, bin_exists: bool = True
) -> Path:
    eng = root / name
    if bin_exists:
        bin_dir = eng / "bin"
        bin_dir.mkdir(parents=True)
        vllm = bin_dir / "vllm"
        vllm.write_text("#!/bin/sh\n")
        if executable:
            vllm.chmod(0o755)
        else:
            vllm.chmod(0o644)
    else:
        eng.mkdir(parents=True)
    return eng


@pytest.fixture
def engines_root(tmp_path, monkeypatch):
    root = tmp_path / "engines"
    monkeypatch.setenv("VW_ENGINES_ROOT", str(root))
    return root


# ---- list_bundled_engines --------------------------------------------------


def test_lists_only_dirs_with_executable_vllm(engines_root):
    _make_engine(engines_root, "a", executable=True)
    _make_engine(engines_root, "b", executable=False)
    (engines_root / "c").mkdir(parents=True)

    assert engines.list_bundled_engines() == ["a"]


def test_bin_path_for_present_engine(engines_root):
    _make_engine(engines_root, "0.34.0rc1.dev1202-pr69165")

    bin_path = engines.bundled_engine_bin("0.34.0rc1.dev1202-pr69165")

    assert bin_path == engines_root / "0.34.0rc1.dev1202-pr69165" / "bin" / "vllm"


def test_missing_root_lists_nothing(engines_root):
    assert engines.list_bundled_engines() == []


def test_missing_engine_bin_returns_none(engines_root):
    assert engines.bundled_engine_bin("nope-1") is None


# ---- validate_version_id ----------------------------------------------------


@pytest.mark.parametrize("bad", ["../x", "a/b", "/abs", "", ".", "x" * 65, "abc\n"])
def test_bad_version_ids_are_refused(bad):
    with pytest.raises(ValueError):
        engines.validate_version_id(bad)
    with pytest.raises(ValueError):
        engines.bundled_engine_bin(bad)


@pytest.mark.parametrize(
    "good",
    ["0.34.0rc1.dev1202-pr69165", "0.20.0", "1.0.0+build.42", "a", "x" * 64],
)
def test_good_version_ids_are_accepted(good):
    assert engines.validate_version_id(good) == good
