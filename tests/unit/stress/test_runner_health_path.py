"""The stress runner's health re-check polls the backend's health path.

Drives the real ``WardenEngineControl.wait_for_health`` (app/stress/runner.py):
the supervisor seam is faked (port map + liveness), but the health wait is
patched to RECORD the kwargs, so the test pins what path the runner asks the
supervisor to poll -- the model row's backend capability, not a hardcoded
``/health`` (the pre-existing bug a future Metal backend with ``/v1/ready``
would hit).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config import Settings
from app.stress import runner as runner_mod


class _FakeSup:
    """Just the supervisor surface WardenEngineControl.target()/wait_for_health read."""

    port: int | None = 4321

    def get_port(self, model_id):
        return self.port

    def get_host(self, model_id):
        return "127.0.0.1"

    def is_running(self, model_id):
        return True


def _settings(tmp_path: Path, **kw) -> Settings:
    base = dict(
        data_dir=tmp_path,
        hf_cache_dir=tmp_path,
        cookie_secret="x" * 32,
        container_gpu_count=0,
        load_timeout_s=0.05,
        load_max_s=30.0,
    )
    base.update(kw)
    return Settings(**base)


def _control(settings: Settings, backend: str | None) -> runner_mod.WardenEngineControl:
    row = SimpleNamespace(id="m1", served_model_name="m1", backend=backend)
    app_state = SimpleNamespace(supervisor=_FakeSup())
    return runner_mod.WardenEngineControl(
        settings,
        app_state,
        row,
        subject=None,
        caps=None,
    )


async def _captured_paths(control) -> list[str]:
    """Patch the supervisor's wait_for_health to record the ``path`` kwarg
    (absent means the runner did not pass one), then run the control's
    wait. Returns the recorded ``path`` values (or None for the default)."""
    from app.runtime import supervisor as sup_mod

    seen: list[str | None] = []

    async def spy(**kwargs):
        seen.append(kwargs.get("path"))
        return True

    try:
        with patch.object(sup_mod, "wait_for_health", new=spy):
            await control.wait_for_health()
    finally:
        pass
    return seen


async def test_wait_for_health_passes_the_backend_health_path(tmp_path):
    """A row whose backend advertises a non-default health path must be
    polled at THAT path, not ``/health``."""
    from app.runtime.backends import registry

    backend = SimpleNamespace(
        capabilities=SimpleNamespace(health_path="/v1/ready"),
    )
    settings = _settings(tmp_path)
    control = _control(settings, "metal")

    real_get = registry.get

    def fake_get(name):
        # Answer /v1/ready ONLY for the row's own backend name: a runner that
        # resolved some other name (e.g. the default) gets the real /health.
        return backend if name == "metal" else real_get(name)

    with patch.object(registry, "get", side_effect=fake_get):
        seen = await _captured_paths(control)

    assert seen == ["/v1/ready"], (
        "wait_for_health must poll the backend's capabilities.health_path, "
        f"not the hardcoded default; saw {seen!r}"
    )


async def test_wait_for_health_falls_back_to_health_when_backend_lookup_raises(tmp_path):
    """A row naming a backend this build does not have must still get a
    health re-check at ``/health`` rather than no re-check at all: the
    registry failure is non-fatal, exactly like the load path."""
    from app.runtime.backends import registry
    from app.runtime.backends.registry import UnknownBackendError

    settings = _settings(tmp_path)
    control = _control(settings, "metal")

    with patch.object(
        registry,
        "get",
        side_effect=UnknownBackendError("unknown backend 'metal'"),
    ):
        seen = await _captured_paths(control)

    assert seen == ["/health"], (
        "an unresolvable backend must fall back to /health, not skip the " f"re-check; saw {seen!r}"
    )


async def test_wait_for_health_uses_health_for_the_default_backends(tmp_path):
    """vLLM and llama.cpp both advertise ``/health`` today; the runner must
    keep polling that path for them (control: the change must not move an
    existing backend)."""
    settings = _settings(tmp_path)
    for backend_name in ("vllm", "llamacpp", None):
        control = _control(settings, backend_name)
        seen = await _captured_paths(control)
        assert seen == ["/health"], (
            f"backend {backend_name!r} must be polled at its own health " f"path; saw {seen!r}"
        )
