"""``ModelSpec`` -- the full-row validator (#262, #265).

Until this existed, the rules a ``ModelRow`` had to satisfy were attached to a
REQUEST: the full set on ``ModelCreate``, a thinner copy on ``TemplateCreate``,
a hand-written subset in ``patch_model_settings``, and none at all in
``try_stack`` and stress-apply. The runtime trusts the ROW -- ``backend.plan``
turns it into argv and env, ``_resolve_engine_image`` into a container image,
and the watchdog re-reads it and restarts it with nobody at the keyboard -- so
every write path has to be able to validate a whole row, not one request shape.

``ModelSpec`` is that validator, and ``ModelCreate`` is now a request-shaped
subclass of it. The tripwire below is the second half of the policy table's
completeness guarantee: the table says a column may be written by an operator,
and this says the validator has a rule for it. A new column can therefore be
unclassified (the app will not boot) but it cannot be writable-and-unvalidated.
"""
import dataclasses

import pytest
from pydantic import ValidationError

from app.db.repos.models import ModelRow
from app.models.policy import OPERATOR_FIELDS
from app.models.schemas import ModelCreate, ModelSpec


def _row_kwargs(**over):
    """A minimal, valid full-row dict the way the writer merges one."""
    base = dict(
        served_model_name="m1",
        hf_repo="org/repo",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# The tripwire
# ---------------------------------------------------------------------------


def test_model_spec_covers_exactly_the_operator_fields():
    """Every operator-writable column has a rule, and nothing else is on the
    spec.

    If this fails, a ``MODEL_FIELD_POLICY`` entry and ``ModelSpec`` have
    drifted: either a new ``OPERATOR``/``SESSION_TO_SET`` column arrived with
    no validation (add it to ``ModelSpec``), or ``ModelSpec`` grew a field that
    is not a ``ModelRow`` column an operator may write (it does not belong
    there -- a request-shaped field like ``template_id`` goes on
    ``ModelCreate``).
    """
    assert set(ModelSpec.model_fields) == OPERATOR_FIELDS


def test_model_spec_field_names_are_model_row_columns():
    columns = {f.name for f in dataclasses.fields(ModelRow)}
    assert set(ModelSpec.model_fields) <= columns


def test_model_create_is_a_model_spec():
    """The request model delegates rather than copying: one ruleset, two
    shapes."""
    assert issubclass(ModelCreate, ModelSpec)


def test_model_create_adds_only_request_shaped_fields():
    extra = set(ModelCreate.model_fields) - set(ModelSpec.model_fields)
    assert extra == {"template_id"}


# ---------------------------------------------------------------------------
# A whole row validates, and the bounds the PATCH used to skip apply to it
# ---------------------------------------------------------------------------


def test_a_minimal_row_validates():
    spec = ModelSpec(**_row_kwargs())
    assert spec.served_model_name == "m1"
    assert spec.gpu_indices == [0]


def test_hf_repo_is_required_on_a_row():
    """``ModelCreate`` may omit it (a template supplies it); a ROW may not."""
    kwargs = _row_kwargs()
    del kwargs["hf_repo"]
    with pytest.raises(ValidationError):
        ModelSpec(**kwargs)
    assert ModelCreate(**kwargs).hf_repo is None


@pytest.mark.parametrize(
    "over",
    [
        {"served_model_name": "bad name!"},
        {"served_model_name": ""},
        {"served_model_name": "x" * 101},
        {"hf_repo": "not-a-slug"},
        {"filename": ""},
        {"filename": "f" * 513},
        {"mmproj_filename": ""},
        {"mmproj_filename": "f" * 513},
        {"hf_config_repo": "not-a-slug"},
        {"tokenizer_repo": "not-a-slug"},
        {"max_model_len": 0},
        {"max_model_len": -1},
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": 1.5},
        {"max_batch_size": 0},
        {"max_batch_size": 65},
        {"n_gpu_layers": -1},
        {"n_gpu_layers": 1000},
        {"parallelism_strategy": "sideways"},
        {"gpu_indices": []},
        {"gpu_indices": [0, 0]},
        {"gpu_indices": [-1]},
        {"gpu_indices": "0"},
        {"extra_args": "--enforce-eager"},
        {"extra_args": [1, 2]},
        {"extra_env": "VLLM_USE_V1=1"},
        {"extra_env": {"LD_PRELOAD": "/tmp/x.so"}},
        {"extra_env": {"FOO_BAR": "1"}},
        {"backend": "sglang"},
        {"gpu_indices": [0, 1], "tensor_parallel_size": 1},
        {"supports_vision": 2},
        {"supports_vision": "no"},
    ],
)
def test_a_row_that_breaks_a_rule_is_refused(over):
    with pytest.raises(ValidationError):
        ModelSpec(**_row_kwargs(**over))


def test_tensor_parallel_size_defaults_to_the_gpu_count():
    spec = ModelSpec(**_row_kwargs(gpu_indices=[1, 2], tensor_parallel_size=None))
    assert spec.tensor_parallel_size == 2


def test_gpu_indices_are_canonicalised_to_sorted_order():
    """Register sorted them in the route; the rule belongs on the row, so every
    write path produces the same canonical list."""
    assert ModelSpec(**_row_kwargs(gpu_indices=[2, 0], tensor_parallel_size=2)).gpu_indices == [0, 2]


def test_extra_env_is_checked_against_the_rows_backend():
    ModelSpec(**_row_kwargs(backend="llamacpp", extra_env={"GGML_CUDA_P2P": "1"}))
    with pytest.raises(ValidationError, match="allowlist"):
        ModelSpec(**_row_kwargs(backend="llamacpp",
                                extra_env={"VLLM_LOGGING_LEVEL": "DEBUG"}))


def test_a_null_backend_is_the_registry_default_for_the_allowlist():
    """``models.backend`` is NULL on every pre-sub-project-B row and NULL is how
    an operator puts a row back on the default, so a row must validate with it
    -- against the default backend's allowlist."""
    ModelSpec(**_row_kwargs(backend=None, extra_env={"VLLM_USE_V1": "1"}))
    with pytest.raises(ValidationError, match="allowlist"):
        ModelSpec(**_row_kwargs(backend=None, extra_env={"GGML_CUDA_P2P": "1"}))


@pytest.mark.parametrize("value,expected", [
    (True, 1), (False, 0), (None, None), ("true", 1), ("False", 0), (1, 1), (0, 0),
])
def test_capability_flags_are_tri_state_integers(value, expected):
    """The row stores 1 / 0 / NULL and the third state is load-bearing, so the
    spec keeps the raw integer rather than collapsing NULL into False."""
    spec = ModelSpec(**_row_kwargs(supports_vision=value))
    assert spec.supports_vision == expected


def test_empty_strings_normalise_to_none_on_a_row():
    spec = ModelSpec(**_row_kwargs(hf_config_repo="", tokenizer_repo="   "))
    assert spec.hf_config_repo is None and spec.tokenizer_repo is None


# ---------------------------------------------------------------------------
# Rows written before these rules existed
# ---------------------------------------------------------------------------


def test_a_row_written_before_the_rules_still_decodes():
    """Validation is a WRITE-path rule. A row already in the database that
    would now be refused must still read back -- ``GET /api/models`` and the
    settings GET decode the dataclass and never touch ``ModelSpec``.
    """
    row = ModelRow(
        id="legacy",
        served_model_name="an invalid name!",
        hf_repo="not-a-slug",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype=None,
        max_model_len=0,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status="pulled",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={"FOO_BAR": "1"},
        max_batch_size=999,
    )
    assert row.served_model_name == "an invalid name!"
    with pytest.raises(ValidationError):
        ModelSpec(**{k: getattr(row, k) for k in OPERATOR_FIELDS})
