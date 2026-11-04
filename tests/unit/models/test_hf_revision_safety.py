"""hf_revision cannot read files outside the HF cache (verified finding #13)."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from app.models.schemas import HfRevision, ModelSpec
from app.runtime.variants import resolve_hf_commit


def _cache(tmp_path):
    refs = tmp_path / "models--org--m" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("a" * 40)
    (tmp_path / "secret").write_text("TOP-SECRET-CONTENT")
    return tmp_path


def test_resolver_reads_refs_only_and_returns_only_a_commit(tmp_path):
    root = _cache(tmp_path)
    assert resolve_hf_commit(root, "org/m", "main") == "a" * 40
    for bad in (str(root / "secret"), "../../secret", "/etc/hostname"):
        assert "TOP-SECRET" not in (resolve_hf_commit(root, "org/m", bad) or "")
    (root / "models--org--m" / "refs" / "weird").write_text("not a commit")
    assert resolve_hf_commit(root, "org/m", "weird") == "weird"


@pytest.mark.parametrize(
    "bad", ["/data/jwt_secret", "../../jwt_secret", "main/../../x", "a" * 200, "has space"]
)
def test_model_spec_refuses_unsafe_revisions(bad):
    with pytest.raises(ValidationError):
        TypeAdapter(HfRevision).validate_python(bad)


@pytest.mark.parametrize("ok", ["main", "v1.2.3", "refs/pr/12", "a" * 40, "release_2026-10"])
def test_model_spec_accepts_normal_revisions(ok):
    assert TypeAdapter(HfRevision).validate_python(ok) == ok


def test_model_spec_uses_the_safe_revision_type():
    assert (
        ModelSpec.model_fields["hf_revision"].annotation is str
    )  # the Annotated metadata carries the rules
    assert any(
        getattr(m, "func", None) is not None for m in ModelSpec.model_fields["hf_revision"].metadata
    )
