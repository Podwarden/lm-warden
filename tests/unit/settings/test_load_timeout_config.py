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


def test_load_timeout_empty_string_means_default(_base_env):
    # A compose file forwards the variable as "${VW_LOAD_TIMEOUT_S:-}", so an
    # unset deployment value arrives as "" -- that must not crash startup.
    _base_env.setenv("VW_LOAD_TIMEOUT_S", "")
    assert load_settings().load_timeout_s == 600.0
    _base_env.setenv("VW_LOAD_TIMEOUT_S", "   ")
    assert load_settings().load_timeout_s == 600.0


@pytest.mark.parametrize(
    "name,attr,default",
    [
        ("VW_WARMUP_PROBE_TIMEOUT_S", "warmup_probe_timeout_s", 600.0),
        ("VW_LOAD_STALL_WARN_S", "load_stall_warn_s", 300.0),
        ("VW_WATCHDOG_FAILURE_THRESHOLD", "watchdog_failure_threshold", 3),
        ("VW_REQUEST_MAX_WALL_S", "request_max_wall_s", 0.0),
        ("VW_CONTENT_LOG_MAX_CHARS", "content_log_max_chars", 40000),
        ("VW_MAX_REQUEST_BODY_MB", "max_request_body_mb", 64),
    ],
)
def test_numeric_env_empty_string_means_default(_base_env, name, attr, default):
    _base_env.setenv(name, "")
    assert getattr(load_settings(), attr) == default


def test_garbage_numeric_env_still_fails_loudly(_base_env):
    _base_env.setenv("VW_LOAD_TIMEOUT_S", "soon")
    with pytest.raises(ValueError):
        load_settings()


def test_compose_forwards_load_timeout_with_a_non_empty_default():
    # Compose gives a container only the variables its environment block
    # names; a deployment value for VW_LOAD_TIMEOUT_S is otherwise dropped.
    from pathlib import Path

    text = (Path(__file__).resolve().parents[3] / "docker-compose.yml").read_text()
    assert 'VW_LOAD_TIMEOUT_S: "${VW_LOAD_TIMEOUT_S:-600}"' in text
