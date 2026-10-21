"""The stress subject's backend resolves NULL to the profile default.

D6: ``models.backend`` is nullable and NULL means the profile's default —
vLLM on linux, MLX on apple. A hardcoded ``or "vllm"`` at the subject
builder is wrong on the apple profile, where a NULL row's MLX engine
version never gets reported.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace


def test_null_backend_is_profile_default_apple(monkeypatch):
    from app.runtime.backends import registry
    from app.stress.routes_api import _subject_backend

    monkeypatch.setattr(registry, "DEFAULT_BACKEND", "mlx")
    assert _subject_backend(SimpleNamespace(backend=None)) == "mlx"


def test_empty_backend_is_profile_default(monkeypatch):
    from app.runtime.backends import registry
    from app.stress.routes_api import _subject_backend

    monkeypatch.setattr(registry, "DEFAULT_BACKEND", "mlx")
    assert _subject_backend(SimpleNamespace(backend="")) == "mlx"


def test_null_backend_on_linux_default_is_vllm(monkeypatch):
    from app.runtime.backends import registry
    from app.stress.routes_api import _subject_backend

    monkeypatch.setattr(registry, "DEFAULT_BACKEND", "vllm")
    assert _subject_backend(SimpleNamespace(backend=None)) == "vllm"


def test_explicit_backend_wins(monkeypatch):
    from app.runtime.backends import registry
    from app.stress.routes_api import _subject_backend

    monkeypatch.setattr(registry, "DEFAULT_BACKEND", "mlx")
    assert _subject_backend(SimpleNamespace(backend="llamacpp")) == "llamacpp"


def test_build_subject_has_no_hardcoded_vllm():
    from app.stress.routes_api import build_subject

    assert 'or "vllm"' not in inspect.getsource(build_subject)
