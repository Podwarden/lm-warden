"""Config parsing for the runaway-detector env vars (app/config.py)."""

import pytest

from app.config import load_settings
from app.proxy.runaway import RunawayDetector


@pytest.fixture
def _base_env(monkeypatch):
    # Minimum env for load_settings() to succeed.
    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "0")
    # Ensure a clean slate for the vars under test.
    for k in (
        "VW_RUNAWAY_MODE",
        "VW_RUNAWAY_THINK_BUDGET",
        "VW_RUNAWAY_REPEAT_MAX",
        "VW_RUNAWAY_HARD_MAX",
    ):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_runaway_defaults_are_conservative_and_off(_base_env):
    s = load_settings()
    assert s.runaway_mode == "off"
    assert s.runaway_think_budget == 24000
    assert s.runaway_repeat_max == 6
    assert s.runaway_hard_max == 96000


def test_runaway_mode_parses_valid_values(_base_env):
    for val in ("off", "log", "enforce"):
        _base_env.setenv("VW_RUNAWAY_MODE", val)
        assert load_settings().runaway_mode == val


def test_runaway_mode_is_case_insensitive_and_trimmed(_base_env):
    _base_env.setenv("VW_RUNAWAY_MODE", "  ENFORCE ")
    assert load_settings().runaway_mode == "enforce"


def test_runaway_mode_unknown_value_falls_back_to_off(_base_env):
    _base_env.setenv("VW_RUNAWAY_MODE", "loud")
    assert load_settings().runaway_mode == "off"


def test_runaway_thresholds_override_from_env(_base_env):
    _base_env.setenv("VW_RUNAWAY_THINK_BUDGET", "1000")
    _base_env.setenv("VW_RUNAWAY_REPEAT_MAX", "3")
    _base_env.setenv("VW_RUNAWAY_HARD_MAX", "5000")
    s = load_settings()
    assert s.runaway_think_budget == 1000
    assert s.runaway_repeat_max == 3
    assert s.runaway_hard_max == 5000


def test_detector_from_settings_uses_configured_thresholds(_base_env):
    _base_env.setenv("VW_RUNAWAY_THINK_BUDGET", "42")
    _base_env.setenv("VW_RUNAWAY_REPEAT_MAX", "7")
    _base_env.setenv("VW_RUNAWAY_HARD_MAX", "999")
    d = RunawayDetector.from_settings(load_settings())
    assert d.think_budget == 42
    assert d.repeat_max == 7
    assert d.hard_max == 999
