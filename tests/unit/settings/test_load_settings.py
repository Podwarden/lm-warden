"""Config parsing for the load-progress stall policy settings (app/config.py).

Tests the three new fields:

- ``VW_LOAD_STALL_WARN_S``  -> ``load_stall_warn_s``   (default 300.0)
- ``VW_LOAD_STALL_TIMEOUT_S`` -> ``load_stall_timeout_s`` (default 1200.0)
- ``VW_LOAD_MAX_S``          -> ``load_max_s``
  unset  -> ``max(7200.0, 3 * load_timeout_s)``
  set    -> the set value

The ``load_max_s`` default is deliberately *coupled* to ``load_timeout_s``: the
cap is at least two hours, or three times the per-load health timeout, whichever
is larger. An explicit ``VW_LOAD_MAX_S`` always wins.
"""

import pytest

from app.config import load_settings


@pytest.fixture
def _base_env(monkeypatch):
    # Minimum env for load_settings() to succeed.
    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "0")
    monkeypatch.delenv("VW_LOAD_TIMEOUT_S", raising=False)
    monkeypatch.delenv("VW_LOAD_STALL_WARN_S", raising=False)
    monkeypatch.delenv("VW_LOAD_STALL_TIMEOUT_S", raising=False)
    monkeypatch.delenv("VW_LOAD_MAX_S", raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# load_stall_warn_s
# ---------------------------------------------------------------------------


def test_stall_warn_defaults_to_300s_when_unset(_base_env):
    s = load_settings()
    assert s.load_stall_warn_s == 300.0


def test_stall_warn_env_override_wins(_base_env):
    _base_env.setenv("VW_LOAD_STALL_WARN_S", "120")
    s = load_settings()
    assert s.load_stall_warn_s == 120.0


# ---------------------------------------------------------------------------
# load_stall_timeout_s
# ---------------------------------------------------------------------------


def test_stall_timeout_defaults_to_1200s_when_unset(_base_env):
    s = load_settings()
    assert s.load_stall_timeout_s == 1200.0


def test_stall_timeout_env_override_wins(_base_env):
    _base_env.setenv("VW_LOAD_STALL_TIMEOUT_S", "600")
    s = load_settings()
    assert s.load_stall_timeout_s == 600.0


# ---------------------------------------------------------------------------
# load_max_s — the brief's named tests
# ---------------------------------------------------------------------------


def test_max_defaults_to_two_hours(_base_env):
    """Unset VW_LOAD_MAX_S and default load_timeout_s (600) -> max(7200, 1800) = 7200."""
    s = load_settings()
    assert s.load_max_s == 7200.0


def test_max_never_below_3x_load_timeout(_base_env):
    """VW_LOAD_TIMEOUT_S=3600 -> max(7200, 10800) = 10800."""
    _base_env.setenv("VW_LOAD_TIMEOUT_S", "3600")
    s = load_settings()
    assert s.load_max_s == 10800.0


def test_explicit_max_wins(_base_env):
    """VW_LOAD_MAX_S set -> that value, regardless of load_timeout_s."""
    _base_env.setenv("VW_LOAD_MAX_S", "999")
    s = load_settings()
    assert s.load_max_s == 999.0


def test_explicit_max_wins_even_below_default_floor(_base_env):
    """An explicit max below 7200 is honoured — the operator wins."""
    _base_env.setenv("VW_LOAD_MAX_S", "100")
    s = load_settings()
    assert s.load_max_s == 100.0
