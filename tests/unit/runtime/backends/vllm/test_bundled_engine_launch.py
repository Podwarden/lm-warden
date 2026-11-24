"""Launching a model on its bundled vLLM engine.

A row carrying a bare ``engine_vllm_version`` (no ``engine_channel`` /
``engine_image``) names an install baked into the image; ``plan()`` then
launches that install's ``vllm`` binary instead of the one on PATH. The
root is read from ``VW_ENGINES_ROOT`` at call time, so tests point it at a
tmp dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.runtime.backends.vllm import VllmBackend
from app.runtime.backends.vllm.engines import bundled_pin
from tests.unit.runtime.backends.corpus import StubModel

BACKEND = VllmBackend()
ENGINE_ID = "0.34.0rc1.dev1202-pr69165"


def _make_engine(root: Path, name: str) -> Path:
    eng = root / name
    (eng / "bin").mkdir(parents=True)
    vllm = eng / "bin" / "vllm"
    vllm.write_text("#!/bin/sh\n")
    vllm.chmod(0o755)
    return vllm


@pytest.fixture
def engines_root(tmp_path, monkeypatch):
    root = tmp_path / "engines"
    monkeypatch.setenv("VW_ENGINES_ROOT", str(root))
    return root


def test_bundled_pin_uses_bundled_vllm_binary(engines_root):
    bin_path = _make_engine(engines_root, ENGINE_ID)

    model = StubModel()
    model.engine_vllm_version = ENGINE_ID

    plan = BACKEND.plan(model, port=10001, bind_host="127.0.0.1", engine_driver="local")

    assert plan.argv[0] == str(bin_path)
    assert plan.argv[1] == "serve"


def test_no_pin_keeps_launch_head():
    model = StubModel()

    plan = BACKEND.plan(model, port=10001, bind_host="127.0.0.1", engine_driver="local")

    assert plan.argv[:2] == ["vllm", "serve"]


def test_channel_pin_is_not_treated_as_bundled(engines_root):
    _make_engine(engines_root, "0.20.0")

    model = StubModel()
    model.engine_channel = "upstream"
    model.engine_vllm_version = "0.20.0"

    assert bundled_pin(model) is None
    plan = BACKEND.plan(model, port=10001, bind_host="127.0.0.1", engine_driver="local")
    assert plan.argv[:2] == ["vllm", "serve"]
