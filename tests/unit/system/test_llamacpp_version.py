"""The baked llama.cpp build id is declared, not discovered.

Read from VW_LLAMACPP_BUILD, which the Dockerfile sets from its
LLAMACPP_BUILD ARG -- the same single-source-of-truth discipline VW_BASE_DIGEST
already has, and for the same reason: CI's post-build cleanup greps the
Dockerfile, and a cleanup that reads a stale value deletes a base you are still
using.

Deliberately NOT ``llama-server --version``: the control plane answers capability
questions from a declared value (the pattern app/system/routes_engine.py set),
and an import-time subprocess is both a startup cost and a new failure mode for
a metadata read. tests/integration/test_llamacpp_smoke.py checks the declared
value against the real binary, where a subprocess is free.

The plan sketched this as a module-level ``BUILD`` constant. It is a FUNCTION
instead, for one concrete reason: a constant is bound at import time, so any
test -- or any caller -- that changes the environment has to reload the module
to see it, and a stale import elsewhere in the process would silently keep the
old value. An env read is cheap enough to do per call.
"""

from __future__ import annotations

from app.runtime.backends.llamacpp.version import baked_build, version_string


def test_build_is_none_outside_a_built_image(monkeypatch):
    monkeypatch.delenv("VW_LLAMACPP_BUILD", raising=False)
    assert baked_build() is None
    assert version_string() is None


def test_build_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("VW_LLAMACPP_BUILD", "b10731")
    assert baked_build() == "b10731"
    assert version_string() == "b10731"


def test_empty_env_reads_as_none(monkeypatch):
    """An exported-but-blank build arg must not surface as an empty-string
    version in the API -- the same 'empty counts as unset' rule
    engine_bind_host applies to VW_ENGINE_BIND_HOST."""
    monkeypatch.setenv("VW_LLAMACPP_BUILD", "")
    assert baked_build() is None


def test_whitespace_only_env_reads_as_none(monkeypatch):
    monkeypatch.setenv("VW_LLAMACPP_BUILD", "   ")
    assert baked_build() is None


def test_the_dockerfile_pins_the_same_tag_the_fixtures_were_captured_from():
    """The fixtures in tests/fixtures/llamacpp/ are what every parser in this
    sub-project was written against. If the Dockerfile's pin drifts from them,
    the argv, the log grammar and the metric names are all being tested against
    a build the product does not ship."""
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    dockerfile = (repo / "Dockerfile").read_text()
    m = re.search(r"^ARG LLAMACPP_BUILD=(\S+)", dockerfile, re.M)
    assert m, "Dockerfile must declare ARG LLAMACPP_BUILD as a greppable line"
    pinned = m.group(1)

    readme = (repo / "tests" / "fixtures" / "llamacpp" / "README.md").read_text()
    assert pinned in readme, (
        f"Dockerfile pins llama.cpp {pinned}, which the fixture README does not "
        f"mention. Re-capture tests/fixtures/llamacpp/ against {pinned}, or "
        f"revert the pin."
    )
