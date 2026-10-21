"""MLX registers in the apple profile, and the schemas accept it (Metal design s5,
OVERSEER DEFAULT_BACKEND decision, planner iter 22; OVERSEER schema-default
decision, planner iter 32).

The ``apple`` profile's backend set gains MLX and its default becomes ``mlx``;
the two ``backend`` Literals in ``app/models/schemas.py`` widen to admit it so a
register can name it; and ``ModelCreate.backend``'s default is the registry's
profile default (a plain default resolved at import, not a ``default_factory``)
so the OpenAPI ``default`` stays and reads the profile. ``linux`` is unchanged
in both the set and the default, and on ``linux`` a ``backend='mlx'`` body is
refused -- never silently accepted -- because the registry cannot serve it.

Every name introduced by this change is imported INSIDE the test functions (or
read through a module the file imports on the base), so the file collects on a
tree that has not made the change: a module-level import of a name that does not
exist yet would make the whole file an ImportError, which is red for the wrong
reason.
"""

import os
import subprocess
import sys

import pytest

from app.runtime.backends import registry


def test_apple_backend_set_is_llamacpp_and_mlx():
    """The red-first assertion: 'mlx' is in backends_for('apple').

    (Red on the base tree as an assertion failure: the apple set is
    ``{'llamacpp'}`` and has no ``'mlx'``.) Pinning the exact two-member set
    keeps the test red if the registration is added under a misspelled key."""
    from app.runtime.backends.mlx import MlxBackend

    backends = registry.backends_for("apple")
    assert set(backends) == {"llamacpp", "mlx"}
    assert isinstance(backends["mlx"], MlxBackend)


def test_apple_default_is_mlx():
    """Once MLX registers, the apple NULL-row default points at it (the comment
    in registry.py named this change). A default that is not in the set would
    make get(None) raise on every legacy NULL row."""
    assert registry.default_for("apple") == "mlx"
    assert registry.default_for("apple") in registry.backends_for("apple")


def test_linux_backend_set_and_default_unchanged():
    """Registration is the apple seam; linux serves exactly what it did before,
    and its default is still vllm -- the set the whole unit suite runs under."""
    from app.runtime.backends.llamacpp import LlamaCppBackend
    from app.runtime.backends.vllm import VllmBackend

    backends = registry.backends_for("linux")
    assert set(backends) == {"llamacpp", "vllm"}
    assert isinstance(backends["vllm"], VllmBackend)
    assert isinstance(backends["llamacpp"], LlamaCppBackend)
    assert "mlx" not in backends
    assert registry.default_for("linux") == "vllm"


def test_model_spec_backend_literal_includes_mlx():
    """The ROW schema admits 'mlx' so a legacy / decoded row can hold it."""
    from typing import get_args

    from app.models.schemas import ModelSpec

    field = ModelSpec.model_fields["backend"]
    literal = next(
        arg for arg in get_args(field.annotation) if getattr(arg, "__args__", None) is not None
    )
    assert set(get_args(literal)) == {"vllm", "llamacpp", "mlx"}


def test_model_create_backend_literal_includes_mlx():
    """The register schema admits 'mlx', so an operator can name it on create.
    Read off the annotation (not by constructing): on the linux profile
    constructing ``backend='mlx'`` raises, so the annotation is the only stable
    way to pin that the Literal widened. A bare Literal (no None), so its args
    are the backend names directly."""
    from typing import get_args

    from app.models.schemas import ModelCreate

    field = ModelCreate.model_fields["backend"]
    assert set(get_args(field.annotation)) == {"vllm", "llamacpp", "mlx"}


def test_linux_profile_refuses_an_mlx_row_naming_the_backend():
    """The acceptance's 422 leg: on the linux profile a ``backend='mlx'`` row is
    refused and never silently accepted. The refusal is the registry --
    ``ModelSpec._layout_consistent`` calls ``registry.get``, which raises
    UnknownBackendError naming the build's backends; FastAPI turns that
    validation error into a 422. Constructing the row must RAISE a ValidationError
    (wrapping the UnknownBackendError), not succeed and not fall back to the
    default (vllm). This is what keeps a linux deploy from writing a row it
    cannot serve."""
    from pydantic import ValidationError

    from app.models.schemas import ModelCreate

    with pytest.raises(ValidationError) as e:
        ModelCreate(served_model_name="x", hf_repo="a/b", gpu_indices=[0], backend="mlx")
    # The underlying refusal is the registry naming the backend, not a silent
    # default: the pydantic error's context carries the UnknownBackendError.
    assert isinstance(e.value.errors()[0]["ctx"]["error"], registry.UnknownBackendError)
    assert "mlx" in str(e.value)
    # And the registry's own answer names it too.
    assert not registry.is_known("mlx")


def _model_create_probe():
    return (
        "from app.models.schemas import ModelCreate as C\n"
        "m = C(served_model_name='x', hf_repo='a/b', gpu_indices=[0])\n"
        "print(m.backend, C.model_json_schema()['properties']['backend']['default'])\n"
    )


def _run_model_create_probe(profile):
    env = {k: v for k, v in os.environ.items() if k != "LMW_PLATFORM_PROFILE"}
    if profile is not None:
        env["LMW_PLATFORM_PROFILE"] = profile
    return subprocess.run(
        [sys.executable, "-c", _model_create_probe()],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_model_create_default_is_profile_default_in_a_fresh_interpreter():
    """Acceptance (e): a ``ModelCreate`` built without a backend carries the
    profile's registry default -- 'vllm' when the profile is unset/linux, 'mlx'
    under apple -- and the OpenAPI ``default`` follows it. A fresh interpreter
    per profile, because ``registry.DEFAULT_BACKEND`` is bound once at import."""
    for profile, expected in ((None, "vllm vllm"), ("linux", "vllm vllm"), ("apple", "mlx mlx")):
        proc = _run_model_create_probe(profile)
        assert proc.returncode == 0, (profile, proc.stderr)
        assert proc.stdout == expected + "\n", (profile, repr(proc.stdout))


def test_model_create_openapi_default_is_vllm_on_linux():
    """In-process (this suite runs under the linux profile): the OpenAPI
    ``default`` stays 'vllm' and the field keeps its documented default, so the
    generated api-types are unchanged on linux."""
    from app.models.schemas import ModelCreate

    assert ModelCreate(served_model_name="x", hf_repo="a/b", gpu_indices=[0]).backend == "vllm"
    assert ModelCreate.model_json_schema()["properties"]["backend"]["default"] == "vllm"


def test_published_backend_enum_is_the_profile_set_in_a_fresh_interpreter():
    """The OpenAPI enum lists only the running profile's backends, in the
    Literal's order: byte-identical on linux (``vllm, llamacpp`` -- the
    generated api-types stay unchanged) and ``llamacpp, mlx`` on apple, where
    vllm is absent."""
    probe = (
        "from app.models.schemas import ModelCreate as C\n"
        "print(','.join(C.model_json_schema()['properties']['backend']['enum']))\n"
    )
    for profile, expected in (
        (None, "vllm,llamacpp"),
        ("linux", "vllm,llamacpp"),
        ("apple", "llamacpp,mlx"),
    ):
        env = {k: v for k, v in os.environ.items() if k != "LMW_PLATFORM_PROFILE"}
        if profile is not None:
            env["LMW_PLATFORM_PROFILE"] = profile
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, (profile, proc.stderr)
        assert proc.stdout == expected + "\n", (profile, repr(proc.stdout))
