"""A template may not hold a shared column to weaker rules than a row does.

``TemplateCreate`` keeps its own wire shape -- it has an ``id``, a ``label``, an
``engine`` and no GPU axis, so it is not a ``ModelSpec`` -- but a template is a
register-time prefill: every value it stores is a value that lands on a
``ModelRow``. When the two schemas were allowed to keep separate copies of the
same rule, a template could store an ``extra_env`` key ``POST /api/models``
refused, which #261 had to close as its own door.

This is the drift guard for the remaining overlap. It compares the *rules*, not
the defaults: a template states a value where a row may leave one unset, so
``max_model_len`` defaults to 8192 on a template and to None on a row, and that
difference is deliberate.
"""
from typing import Annotated, get_args, get_origin

from app.models.schemas import ModelSpec, TemplateCreate

#: Columns both schemas carry. ``dtype``, ``hf_revision``, ``extra_args``,
#: ``trust_remote_code`` and ``tensor_parallel_size`` are here too but differ in
#: OPTIONALITY (a row may leave them unset), so only their base rules are
#: comparable -- see ``_constraints``.
SHARED = ("hf_repo", "max_model_len", "gpu_memory_utilization", "extra_args",
          "extra_env", "trust_remote_code", "hf_revision", "dtype")


def _constraints(model, name):
    """``(type, constraints)`` for a field, with optionality and defaults
    stripped: what a value must satisfy once it is present.

    Pydantic hoists ``Annotated`` metadata onto ``FieldInfo.metadata`` for a
    plain field but leaves it inside the annotation under an ``Optional``, so
    both places have to be collected before the two schemas are comparable.
    """
    field = model.model_fields[name]
    ann, meta = field.annotation, list(field.metadata)
    args = [a for a in get_args(ann) if a is not type(None)]
    if len(args) == 1:  # ``X | None`` -> X
        ann = args[0]
    if get_origin(ann) is Annotated:
        inner = get_args(ann)
        ann = inner[0]
        for m in inner[1:]:
            meta.extend(getattr(m, "metadata", None) or [m])
    return ann, tuple(sorted(repr(m) for m in meta))


def test_shared_columns_carry_identical_rules():
    for name in SHARED:
        assert name in ModelSpec.model_fields, name
        assert name in TemplateCreate.model_fields, name
        assert _constraints(ModelSpec, name) == _constraints(TemplateCreate, name), name


def test_the_shared_set_is_the_whole_overlap():
    """If a template gains a model-shaped column, it has to be listed above and
    hence held to the row's rule."""
    overlap = set(ModelSpec.model_fields) & set(TemplateCreate.model_fields)
    assert overlap == {
        *SHARED,
        # tensor_parallel_size is shared but genuinely differs: a template's is
        # a standalone `ge=1` count, while a row's is pinned to
        # len(gpu_indices) by ModelSpec's cross-field check, which a template
        # cannot run because it has no gpu_indices.
        "tensor_parallel_size",
    }
