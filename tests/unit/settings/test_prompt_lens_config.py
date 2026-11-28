"""Prompt Lens env settings (app/config.py).

``VW_PROMPT_LENS_*`` are the names. The former-name prefix is no longer read:
one startup warning names any such variable still set (a non-empty value).
``prompt_lens_env_var`` says which variable set ``prompt_lens_enabled`` (None:
not set), so the Settings page can show where the value in effect comes from.
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
        monkeypatch.delenv(f"{config.REMOVED_ENV_PREFIX}{s}", raising=False)
    return monkeypatch


def test_default_on_and_no_env_source(env):
    s = load_settings()
    assert s.prompt_lens_enabled is True
    assert s.prompt_lens_env_var is None


def test_new_name_turns_it_on(env):
    env.setenv("VW_PROMPT_LENS_ENABLED", "true")
    s = load_settings()
    assert s.prompt_lens_enabled is True
    assert s.prompt_lens_env_var == "VW_PROMPT_LENS_ENABLED"


def test_a_blank_env_is_not_a_source(env):
    env.setenv("VW_PROMPT_LENS_ENABLED", "  ")
    s = load_settings()
    assert s.prompt_lens_enabled is True
    assert s.prompt_lens_env_var is None


def test_new_name_false_is_still_a_source(env):
    env.setenv("VW_PROMPT_LENS_ENABLED", "0")
    s = load_settings()
    assert s.prompt_lens_enabled is False
    assert s.prompt_lens_env_var == "VW_PROMPT_LENS_ENABLED"


def test_removed_prefix_is_not_read(env):
    env.setenv(f"{config.REMOVED_ENV_PREFIX}ENABLED", "true")
    env.setenv(f"{config.REMOVED_ENV_PREFIX}RING_EVENTS", "123")
    s = load_settings()
    assert s.prompt_lens_enabled is True  # the removed name does not switch it off either
    assert s.prompt_lens_env_var is None
    assert s.prompt_lens_ring_events == 2000


def test_new_name_is_read_whatever_the_removed_prefix_holds(env):
    env.setenv("VW_PROMPT_LENS_RING_EVENTS", "7")
    env.setenv(f"{config.REMOVED_ENV_PREFIX}RING_EVENTS", "123")
    assert load_settings().prompt_lens_ring_events == 7


def test_startup_warning_names_each_removed_variable_once(env, caplog):
    env.setenv(f"{config.REMOVED_ENV_PREFIX}ENABLED", "true")
    env.setenv(f"{config.REMOVED_ENV_PREFIX}RING_CHARS", "1")
    env.setenv(f"{config.REMOVED_ENV_PREFIX}SECTION_TTL_S", "  ")  # blank: not set
    with caplog.at_level(logging.WARNING, logger="vllm_warden.config"):
        names = config.warn_removed_prompt_lens_env()
    assert names == ["VW_GODMODE_ENABLED", "VW_GODMODE_RING_CHARS"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0]
    assert "VW_GODMODE_ENABLED" in msg and "VW_GODMODE_RING_CHARS" in msg
    assert "VW_GODMODE_SECTION_TTL_S" not in msg
    assert "no longer read" in msg and "VW_PROMPT_LENS_ENABLED" in msg


def test_no_warning_without_removed_names(env, caplog):
    env.setenv("VW_PROMPT_LENS_ENABLED", "1")
    with caplog.at_level(logging.WARNING, logger="vllm_warden.config"):
        assert config.warn_removed_prompt_lens_env() == []
        load_settings()
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_the_app_logs_the_warning_at_startup(env, caplog, tmp_data_dir):
    from fastapi.testclient import TestClient

    from app.main import build_app

    env.setenv(f"{config.REMOVED_ENV_PREFIX}ENABLED", "true")
    with caplog.at_level(logging.WARNING, logger="vllm_warden.config"):
        with TestClient(build_app()):
            pass
    assert [r for r in caplog.records if "VW_GODMODE_ENABLED" in r.getMessage()]
