"""Prompt Lens env settings (app/config.py).

``VW_PROMPT_LENS_*`` are the names; the ``VW_GODMODE_*`` names they replaced
keep working as deprecated aliases (installs set them), each logged once per
process. The new name wins when both are set. ``prompt_lens_env_var`` says
which variable set ``prompt_lens_enabled`` (None: neither), so the Settings
page can show where the value in effect comes from.
"""

import logging

import pytest

from app import config
from app.config import load_settings

_SUFFIXES = (
    "ENABLED",
    "RING_EVENTS",
    "RING_CHARS",
    "MAX_PROMPT_CHARS",
    "PROMPT_TAIL_CHARS",
    "MEDIA_STORE_CHARS",
    "MAX_IMAGE_CHARS",
    "MAX_IMAGES_PER_REQ",
    "SECTION_PROMPT_CHARS",
    "SECTION_CHARS",
    "SECTION_MAX_CHARS",
    "SECTION_TTL_S",
)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "0")
    for s in _SUFFIXES:
        monkeypatch.delenv(f"VW_PROMPT_LENS_{s}", raising=False)
        monkeypatch.delenv(f"VW_GODMODE_{s}", raising=False)
    monkeypatch.setattr(config, "_DEPRECATED_LOGGED", set())
    return monkeypatch


def test_default_off_and_no_env_source(env):
    s = load_settings()
    assert s.prompt_lens_enabled is False
    assert s.prompt_lens_env_var is None


def test_new_name_turns_it_on(env):
    env.setenv("VW_PROMPT_LENS_ENABLED", "true")
    s = load_settings()
    assert s.prompt_lens_enabled is True
    assert s.prompt_lens_env_var == "VW_PROMPT_LENS_ENABLED"


def test_new_name_false_is_still_a_source(env):
    env.setenv("VW_PROMPT_LENS_ENABLED", "0")
    s = load_settings()
    assert s.prompt_lens_enabled is False
    assert s.prompt_lens_env_var == "VW_PROMPT_LENS_ENABLED"


def test_old_name_is_a_deprecated_alias(env, caplog):
    env.setenv("VW_GODMODE_ENABLED", "true")
    with caplog.at_level(logging.WARNING, logger="vllm_warden.config"):
        s = load_settings()
        load_settings()  # a second load logs nothing more
    assert s.prompt_lens_enabled is True
    assert s.prompt_lens_env_var == "VW_GODMODE_ENABLED"
    warnings = [r.getMessage() for r in caplog.records if "VW_GODMODE_ENABLED" in r.getMessage()]
    assert len(warnings) == 1
    assert "deprecated" in warnings[0] and "VW_PROMPT_LENS_ENABLED" in warnings[0]


def test_new_name_wins_over_old(env):
    env.setenv("VW_PROMPT_LENS_ENABLED", "false")
    env.setenv("VW_GODMODE_ENABLED", "true")
    s = load_settings()
    assert s.prompt_lens_enabled is False
    assert s.prompt_lens_env_var == "VW_PROMPT_LENS_ENABLED"


def test_old_bound_names_still_read(env):
    env.setenv("VW_GODMODE_RING_EVENTS", "123")
    env.setenv("VW_GODMODE_SECTION_TTL_S", "42")
    env.setenv("VW_PROMPT_LENS_RING_CHARS", "999")
    env.setenv("VW_GODMODE_RING_CHARS", "1")
    s = load_settings()
    assert s.prompt_lens_ring_events == 123
    assert s.prompt_lens_section_ttl_s == 42.0
    assert s.prompt_lens_ring_chars == 999


def test_no_warning_without_old_names(env, caplog):
    env.setenv("VW_PROMPT_LENS_ENABLED", "1")
    with caplog.at_level(logging.WARNING, logger="vllm_warden.config"):
        load_settings()
    assert not [r for r in caplog.records if "deprecated" in r.getMessage()]
