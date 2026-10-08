"""Config parsing for VW_LOAD_TIMEOUT_S (app/config.py).

#275: Settings.load_timeout_s had no environment override, unlike the
otherwise-identical warmup_probe_timeout_s -- a first load that legitimately
needs longer than the 600s default (NVFP4/MXFP8 on Blackwell doing FlashInfer
GEMM autotuning, multiplied by a data-parallel process count) could never
succeed: the engine was still warming up, not crashed, when the warden gave
up. VW_LOAD_TIMEOUT_S now overrides it, same pattern as
VW_WARMUP_PROBE_TIMEOUT_S.
"""

import pytest

from app.config import load_settings


@pytest.fixture
def _base_env(monkeypatch):
    # Minimum env for load_settings() to succeed.
    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "0")
    monkeypatch.delenv("VW_LOAD_TIMEOUT_S", raising=False)
    return monkeypatch


def test_load_timeout_defaults_to_600s_when_unset(_base_env):
    s = load_settings()
    assert s.load_timeout_s == 600.0


def test_load_timeout_env_override_wins(_base_env):
    # Deliberately not the default: an override test that sets the value the
    # code would have picked anyway proves nothing, and silently stops proving
    # anything the moment the default moves.
    _base_env.setenv("VW_LOAD_TIMEOUT_S", "1800")
    s = load_settings()
    assert s.load_timeout_s == 1800.0
