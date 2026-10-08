"""Config parsing for the content-logger env vars (app/config.py).

Three separate hazards are pinned here.

1. ``VW_CONTENT_LOG_PATH`` used to default to the literal
   ``/data/logs/content.jsonl`` while the volume it is meant to sit on comes
   from ``VW_DATA_DIR`` — a knob the PodWarden Hub template exposes and
   interpolates. Move the data dir and the content log kept writing to
   ``/data/logs/``, which is then no longer the mounted volume: the records
   land on the container's writable layer, fill the *node's* disk instead of
   the PVC, and vanish on restart. Silently, on all three counts.

2. ``VW_CONTENT_LOG_MAX_CHARS`` is a character count, and there is no value
   that means "no limit" — a negative one is clamped to 0 (capture nothing)
   rather than quietly disabling the only per-record bound the module has.

3. ``VW_CONTENT_LOG_MAX_BYTES`` bounds the file itself.
"""

from pathlib import Path

import pytest

from app.config import load_settings


@pytest.fixture
def _base_env(monkeypatch):
    # Minimum env for load_settings() to succeed.
    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "0")
    # Clean slate for the vars under test.
    for k in (
        "VW_DATA_DIR",
        "VW_CONTENT_LOG_PATH",
        "VW_CONTENT_LOG_MAX_CHARS",
        "VW_CONTENT_LOG_MAX_BYTES",
    ):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# VW_CONTENT_LOG_PATH follows VW_DATA_DIR
# ---------------------------------------------------------------------------


def test_content_log_path_defaults_under_the_default_data_dir(_base_env):
    """Unchanged behaviour for anyone who never moved VW_DATA_DIR."""
    assert load_settings().content_log_path == Path("/data/logs/content.jsonl")


def test_content_log_path_follows_a_moved_data_dir(_base_env):
    _base_env.setenv("VW_DATA_DIR", "/mnt/warden")
    assert load_settings().content_log_path == Path("/mnt/warden/logs/content.jsonl")


def test_explicit_content_log_path_wins_over_the_data_dir(_base_env):
    """An operator deliberately pointing it at another volume keeps that."""
    _base_env.setenv("VW_DATA_DIR", "/mnt/warden")
    _base_env.setenv("VW_CONTENT_LOG_PATH", "/scratch/audit/content.jsonl")
    assert load_settings().content_log_path == Path("/scratch/audit/content.jsonl")


def test_blank_content_log_path_is_treated_as_unset(_base_env):
    """The Hub compose renders ``"${VW_CONTENT_LOG_PATH:-}"``.

    An unset variable therefore reaches the container as the empty string, and
    ``Path("")`` is ``Path(".")`` — the process CWD, which is neither the
    volume nor anywhere an operator would look. Blank must mean "derive".
    """
    _base_env.setenv("VW_DATA_DIR", "/mnt/warden")
    _base_env.setenv("VW_CONTENT_LOG_PATH", "   ")
    assert load_settings().content_log_path == Path("/mnt/warden/logs/content.jsonl")


# ---------------------------------------------------------------------------
# VW_CONTENT_LOG_MAX_CHARS has no "unlimited" value
# ---------------------------------------------------------------------------


def test_content_log_max_chars_default_unchanged(_base_env):
    assert load_settings().content_log_max_chars == 40000


def test_negative_max_chars_is_clamped_to_zero(_base_env):
    """``-1`` reads like "no limit" and used to *be* one. It is not one."""
    _base_env.setenv("VW_CONTENT_LOG_MAX_CHARS", "-1")
    assert load_settings().content_log_max_chars == 0


def test_negative_max_chars_warns_naming_the_variable(_base_env, caplog):
    _base_env.setenv("VW_CONTENT_LOG_MAX_CHARS", "-1")
    with caplog.at_level("WARNING"):
        load_settings()
    assert any("VW_CONTENT_LOG_MAX_CHARS" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# VW_CONTENT_LOG_MAX_BYTES bounds the file
# ---------------------------------------------------------------------------


def test_content_log_max_bytes_defaults_to_512_mib(_base_env):
    assert load_settings().content_log_max_bytes == 512 * 1024 * 1024


def test_content_log_max_bytes_overrides_from_env(_base_env):
    _base_env.setenv("VW_CONTENT_LOG_MAX_BYTES", "1048576")
    assert load_settings().content_log_max_bytes == 1048576


def test_content_log_max_bytes_zero_or_negative_means_unbounded(_base_env):
    """Same shape as VW_ENGINE_LOG_MAX_BYTES: ``<= 0`` switches the cap off."""
    _base_env.setenv("VW_CONTENT_LOG_MAX_BYTES", "0")
    assert load_settings().content_log_max_bytes == 0
    _base_env.setenv("VW_CONTENT_LOG_MAX_BYTES", "-1")
    assert load_settings().content_log_max_bytes <= 0
