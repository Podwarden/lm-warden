"""Structural guard against the hand-built-dict drift in the models read path.

The operator-visible symptom was "every model reports Engine: vllm", including
one demonstrably running ``llama-server``. The cause was not a wrong value: it
was that ``GET /api/models/{id}`` built its response as a dict literal, so every
column added by a migration (0027's ``backend``, 0028's ``mmproj_filename`` /
``n_gpu_layers``) had to be remembered at a second site — and was not.

``test_get_model_response_covers_every_model_row_field`` is the test that makes
that class of drift impossible rather than merely fixed: it enumerates
``ModelRow``'s dataclass fields and fails if the detail response cannot account
for one. A future migration that appends a column now fails HERE, at the point
where someone can still decide whether it is public, instead of silently
reaching the UI as ``undefined``.

#237 found the second-order version of the same bug: ``GET /api/models``
(list) hand-maintained its OWN narrower allowlist, so ``max_model_len`` /
``gpu_memory_utilization`` / ``extra_args`` and all three ``supports_*``
capability flags reached detail but not list -- and none of the three
capability flags reached EITHER response, because they were explicitly
withheld on the theory that ``GET /api/models/{id}/settings`` was their only
publish surface. ``test_detail_and_list_agree_field_for_field`` and the
tri-state tests below guard against both regressing.
"""

import dataclasses

from app.db.repos.models import ModelRow
from app.models.serialisation import (
    _CAPABILITY_FIELDS,
    _ENGINE_FIELDS,
    _PRIVATE_FIELDS,
    model_detail,
    model_summary,
)


def _row(**overrides) -> ModelRow:
    base = dict(
        id="abc123",
        served_model_name="qwen-gguf",
        hf_repo="o/r-GGUF",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype=None,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status="loaded",
        pulled_bytes=1,
        pulled_total=1,
        last_error=None,
        extra_env={},
    )
    base.update(overrides)
    return ModelRow(**base)


def test_get_model_response_covers_every_model_row_field():
    """Every persisted column is either published or explicitly withheld.

    Not "the response has these 24 keys" — that assertion would have passed
    happily while ``backend`` was missing. The point is the *closure*: the union
    of what we publish, what we fold into ``engine``, and what we deliberately
    keep private must equal ``ModelRow``. Nothing may fall through the gap.
    """
    row_fields = {f.name for f in dataclasses.fields(ModelRow)}
    published = set(model_detail(_row()))
    accounted = (published - {"engine"}) | _ENGINE_FIELDS | _PRIVATE_FIELDS
    assert row_fields - accounted == set(), (
        "ModelRow gained a column that GET /api/models/{id} neither publishes "
        "nor explicitly withholds. Add it to app/models/serialisation.py."
    )


def test_detail_and_list_agree_field_for_field():
    """#237: the list endpoint used to drop engine settings AND capability
    flags that the detail endpoint already returned correctly. Both routes
    call the same serialiser now, so a model's list row and detail row must
    be byte-for-byte identical (same keys, same values) for the same row.
    """
    row = _row(
        supports_tools=1,
        supports_vision=0,
        supports_reasoning=None,
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        extra_args=["--enforce-eager"],
    )
    detail = model_detail(row)
    summary = model_summary(row)
    assert summary == detail
    # And, concretely, none of the previously-dropped fields are missing.
    for key in (
        "supports_tools", "supports_vision", "supports_reasoning",
        "max_model_len", "gpu_memory_utilization", "extra_args",
    ):
        assert key in summary, f"{key} missing from GET /api/models row"
        assert key in detail, f"{key} missing from GET /api/models/{{id}}"


def test_capability_flags_preserve_the_tristate():
    """NULL means "nobody has said" and must serialise as ``null``, never
    coerced to ``false`` -- ``app/chat2/catalog.py`` depends on being able to
    tell the two apart to auto-detect vision. ``0`` means an explicit "no" and
    must come back as JSON ``false``, not the raw SQLite ``0``.
    """
    row = _row(supports_tools=0, supports_vision=1, supports_reasoning=None)
    for fn in (model_detail, model_summary):
        body = fn(row)
        assert body["supports_tools"] is False
        assert body["supports_vision"] is True
        assert body["supports_reasoning"] is None


def test_summary_fields_are_real_model_row_attributes_or_computed():
    """The list response publishes every ``ModelRow`` field it is allowed to,
    same as detail -- see ``_CAPABILITY_FIELDS`` and ``_ENGINE_FIELDS`` for the
    two kinds of field that are computed/reshaped rather than passed through
    verbatim.
    """
    row_fields = {f.name for f in dataclasses.fields(ModelRow)}
    published = set(model_summary(_row()))
    accounted = (published - {"engine"}) | _ENGINE_FIELDS | _PRIVATE_FIELDS
    assert row_fields - accounted == set()
    assert _CAPABILITY_FIELDS <= published


def test_detail_reports_the_backend_the_row_carries():
    assert model_detail(_row(backend="llamacpp"))["backend"] == "llamacpp"
    assert model_detail(_row(backend="vllm"))["backend"] == "vllm"


def test_detail_defaults_a_null_backend_to_vllm():
    """A pre-0027 row decodes to vLLM (D6) rather than leaking NULL to the UI."""
    assert model_detail(_row(backend=None))["backend"] == "vllm"
    assert model_summary(_row(backend=None))["backend"] == "vllm"


def test_engine_block_stays_none_until_a_channel_is_pinned():
    assert model_detail(_row())["engine"] is None
    pinned = model_detail(_row(
        engine_channel="cuda-stable",
        engine_vllm_version="0.26.0",
        engine_image="vllm/vllm-openai:v0.26.0",
    ))["engine"]
    assert pinned == {
        "channel": "cuda-stable",
        "vllm_version": "0.26.0",
        "image": "vllm/vllm-openai:v0.26.0",
    }


def test_the_response_model_declares_every_serialised_field():
    """ModelOut types GET /api/models(/{id}) in the OpenAPI spec. It must name
    exactly what _serialise returns: a column added to ModelRow and published
    by _serialise fails HERE until ModelOut declares it (extra="allow" keeps it
    in the response meanwhile, so nothing is silently dropped)."""
    from app.models.schemas import ModelOut

    assert set(model_detail(_row()).keys()) == set(ModelOut.model_fields)
