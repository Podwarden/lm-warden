"""``engine_model_id`` -- the id the proxy puts in ``body['model']``.

Pinned BEFORE the method exists (BACKLOG-macos P5, Metal design s5 A; OVERSEER
protocol-additions decision, planner iter 22). The proxy will re-point a request
``body["model"]`` at the string the ENGINE actually serves, so the method's
answer has to be byte-identical to what ``plan()`` passes to
``--served-model-name`` (vLLM) / ``--alias`` (llama.cpp) for the same row. The
consistency leg below compares the two directly so they cannot drift: if one
changes and the other does not, the proxy would hand the engine a name it never
registered and every request would 404.

The method is PURE and takes ``resolved`` only so that, later, ``Supervisor`` /
the proxy can call it through ONE call site without branching on the backend's
name. vLLM and llama.cpp ignore it; MLX (when it lands) will read its local
model path from it. It is deliberately NOT yet a ``Backend`` Protocol member --
that addition lands the moment MLX has the method too (adding it first would
make ``isinstance(x, Backend)`` false for every backend that lacks it).
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("served_model_name", ["qwen3.8-27b", "a", ""])
def test_vllm_engine_model_id_is_the_served_name(served_model_name):
    from app.runtime.backends.vllm import VllmBackend
    from tests.unit.runtime.backends.corpus import StubModel

    row = StubModel(served_model_name=served_model_name)
    # The value the proxy must put in body['model'].
    assert VllmBackend().engine_model_id(row) == served_model_name


@pytest.mark.parametrize("served_model_name", ["qwen3.8-27b", "a", ""])
def test_llamacpp_engine_model_id_is_the_alias(served_model_name):
    from app.runtime.backends.llamacpp import LlamaCppBackend
    from tests.unit.runtime.backends.test_llamacpp_plan import StubModel

    row = StubModel(served_model_name=served_model_name)
    # The value the proxy must put in body['model'].
    assert LlamaCppBackend().engine_model_id(row) == served_model_name


@pytest.mark.parametrize(
    "name",
    [
        "qwen3.8-27b",  # a normal printable served name
        "a",  # the shortest possible non-empty name
        "",  # an empty name: still the exact string plan() passes
    ],
)
def test_vllm_matches_the_argv_served_model_name(name):
    """Consistency leg: engine_model_id == the argv element after
    ``--served-model-name`` in plan() for the same row, so the two can never
    drift."""
    from app.runtime.backends.vllm import VllmBackend
    from tests.unit.runtime.backends.corpus import StubModel

    row = StubModel(served_model_name=name)
    backend = VllmBackend()
    argv = backend.plan(row, port=1, bind_host="127.0.0.1").argv
    assert "--served-model-name" in argv
    flag = "--served-model-name"
    assert argv[argv.index(flag) + 1] == backend.engine_model_id(row)


@pytest.mark.parametrize(
    "name",
    [
        "qwen3.8-27b",  # a normal printable served name
        "a",  # the shortest possible non-empty name
        "",  # an empty name: still the exact string plan() passes
    ],
)
def test_llamacpp_matches_the_argv_alias(name):
    """Consistency leg: engine_model_id == the argv element after ``--alias``
    in plan() for the same row, so the two can never drift."""
    from app.runtime.backends.llamacpp import LlamaCppBackend
    from app.runtime.backends.paths import ResolvedModelPaths
    from tests.unit.runtime.backends.test_llamacpp_plan import StubModel

    row = StubModel(served_model_name=name)
    resolved = ResolvedModelPaths(
        model_path="/cache/snap/model-IQ3_XXS.gguf",
        snapshot_dir="/cache/snap",
    )
    backend = LlamaCppBackend()
    argv = backend.plan(row, port=1, bind_host="127.0.0.1", resolved=resolved).argv
    assert "--alias" in argv
    flag = "--alias"
    assert argv[argv.index(flag) + 1] == backend.engine_model_id(row)


@pytest.mark.parametrize("name", ["qwen3.8-27b", "a", ""])
def test_vllm_resolved_is_ignored(name):
    """``resolved=None`` and a concrete ``ResolvedModelPaths`` stub give the
    same answer: the method's output does not depend on the resolved path."""
    from app.runtime.backends.paths import ResolvedModelPaths
    from app.runtime.backends.vllm import VllmBackend
    from tests.unit.runtime.backends.corpus import StubModel

    row = StubModel(served_model_name=name)
    backend = VllmBackend()
    assert backend.engine_model_id(row) == name
    assert backend.engine_model_id(row, resolved=None) == name
    assert (
        backend.engine_model_id(
            row,
            resolved=ResolvedModelPaths(
                model_path="/cache/snap/model-IQ3_XXS.gguf",
                snapshot_dir="/cache/snap",
            ),
        )
        == name
    )


@pytest.mark.parametrize("name", ["qwen3.8-27b", "a", ""])
def test_llamacpp_resolved_is_ignored(name):
    """``resolved=None`` and a concrete ``ResolvedModelPaths`` stub give the
    same answer: the method's output does not depend on the resolved path."""
    from app.runtime.backends.llamacpp import LlamaCppBackend
    from app.runtime.backends.paths import ResolvedModelPaths
    from tests.unit.runtime.backends.test_llamacpp_plan import StubModel

    row = StubModel(served_model_name=name)
    backend = LlamaCppBackend()
    assert backend.engine_model_id(row) == name
    assert backend.engine_model_id(row, resolved=None) == name
    assert (
        backend.engine_model_id(
            row,
            resolved=ResolvedModelPaths(
                model_path="/cache/snap/model-IQ3_XXS.gguf",
                snapshot_dir="/cache/snap",
            ),
        )
        == name
    )
