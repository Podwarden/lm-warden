"""MlxBackend.plan() must refuse an ``extra_args`` that sets ``--model``.

The engine's model id is the LOCAL MODEL PATH (Metal design s5 A): ``plan()``
emits it via the flag it itself builds, and the proxy re-points a request
``body["model"]`` at whatever the engine loaded (``engine_model_id`` == the
``--model`` value). An ``extra_args`` token that re-points ``--model`` would
make the engine serve an id the proxy never asks for. The load-time layer
(OVERSEER model-id decision, planner iter 32) refuses such a row at plan time,
with ``app.models.model_id_flags`` as the shared detector -- the same module
the write-time validator and the other backends' argv builders use, so the
abbreviation floor and the refusal text cannot drift.

Every import of a NEW name happens INSIDE a test function, never at module
level: a module-level import of ``model_id_refusal`` would make this whole
file an ImportError on the base tree, which runs nothing and is red for the
wrong reason.
"""

from __future__ import annotations

import pytest

from tests.unit.runtime.backends.test_mlx_plan import StubModel, _resolved


def _row(extra_args: list[str]):
    return StubModel(extra_args=extra_args)


def test_mlx_plan_refuses_model_flag_as_two_tokens():
    """``--model x`` as separate tokens: the red-first leg. Before the guard,
    the argv carried TWO ``--model`` (this one plus plan's own) and mlx_lm
    would load ``/elsewhere`` -- an id the proxy never asks for."""
    from app.models.model_id_flags import model_id_refusal
    from app.runtime.backends.mlx import MlxBackend

    with pytest.raises(ValueError) as exc:
        MlxBackend().plan(
            _row(["--model", "/elsewhere"]), port=1234, bind_host="127.0.0.1", resolved=_resolved()
        )
    # the message names the OFFENDING TOKEN (the flag as written), not the value
    assert model_id_refusal("mlx", "--model") in str(exc.value)


def test_mlx_plan_refuses_model_flag_equals_form():
    """``--model=x`` (the ``=value`` spelling of the same flag) is refused too."""
    from app.models.model_id_flags import model_id_refusal
    from app.runtime.backends.mlx import MlxBackend

    with pytest.raises(ValueError) as exc:
        MlxBackend().plan(
            _row(["--model=/elsewhere"]), port=1234, bind_host="127.0.0.1", resolved=_resolved()
        )
    assert model_id_refusal("mlx", "--model=/elsewhere") in str(exc.value)


def test_mlx_plan_refuses_model_flag_abbreviated_to_the_floor():
    """``--mo`` is the shortest prefix mlx_lm's ``allow_abbrev`` still resolves
    to ``--model`` (model_id_flags._ABBREV_FLOORS). The floor: refused."""
    from app.models.model_id_flags import model_id_refusal
    from app.runtime.backends.mlx import MlxBackend

    with pytest.raises(ValueError) as exc:
        MlxBackend().plan(
            _row(["--mo", "/elsewhere"]), port=1234, bind_host="127.0.0.1", resolved=_resolved()
        )
    assert model_id_refusal("mlx", "--mo") in str(exc.value)


def test_mlx_plan_accepts_the_prefix_just_below_the_floor():
    """``--m`` is one character under the floor: ambiguous with ``--max-tokens``
    (mlx_lm's parser rejects it), so the guard must NOT refuse it -- refusing
    here would be the over-refusal the floor is there to prevent."""
    from app.runtime.backends.mlx import MlxBackend

    plan = MlxBackend().plan(
        _row(["--m", "1000"]), port=1234, bind_host="127.0.0.1", resolved=_resolved()
    )
    assert "--m" in plan.argv


def test_mlx_plan_accepts_other_extra_args_unchanged():
    """``--draft-model`` is not a prefix of ``--model`` (it has a ``-`` after
    ``mo``) and other args pass through: the row still plans, with both the
    original ``--model`` (plan's own) and the extra arg in the argv."""
    from app.runtime.backends.mlx import MlxBackend

    plan = MlxBackend().plan(
        _row(["--draft-model", "/draft", "--max-tokens", "2000"]),
        port=1234,
        bind_host="127.0.0.1",
        resolved=_resolved(),
    )
    argv = plan.argv
    # plan's own --model is intact and still carries the resolved snapshot path
    assert argv.count("--model") == 1
    assert argv[argv.index("--model") + 1] == "/cache/models--org--repo/snapshots/abc123"
    assert "--draft-model" in argv
    assert "--max-tokens" in argv


def test_mlx_plan_without_flag_is_byte_identical():
    """A row without the flag gets the exact same plan as one with no
    ``extra_args`` at all -- the guard must not alter anything it does not
    refuse."""
    from app.runtime.backends.mlx import MlxBackend

    backend = MlxBackend()
    clean = backend.plan(_row([]), port=1234, bind_host="127.0.0.1", resolved=_resolved())
    other = backend.plan(
        _row(["--max-tokens", "2000"]),
        port=1234,
        bind_host="127.0.0.1",
        resolved=_resolved(),
    )
    # the --max-tokens pair is the ONLY difference; everything else is identical
    assert other.argv[: len(clean.argv)] == clean.argv
    assert other.argv[len(clean.argv) :] == ["--max-tokens", "2000"]
    assert other.env == clean.env
    assert other.image == clean.image
    assert other.port == clean.port
    assert other.gpu_indices == clean.gpu_indices


def test_mlx_plan_refusal_does_not_echo_other_extra_args():
    """extra_args may hold secrets (an API key): the refusal names only the
    offending token, never the rest of extra_args."""
    from app.runtime.backends.mlx import MlxBackend

    secret = "sk-reviewer-sentinel-must-not-echo"
    with pytest.raises(ValueError) as exc:
        MlxBackend().plan(
            _row(["--api-key", secret, "--model", "/elsewhere"]),
            port=1234,
            bind_host="127.0.0.1",
            resolved=_resolved(),
        )
    assert "--model" in str(exc.value)
    assert secret not in str(exc.value)
    assert "--api-key" not in str(exc.value)
