"""Config parsing for request_history defaults (app/config.py).

The session forest needs 30 complete days of history to compute rolling
statistics over the full span. Each row is ~300 bytes (request metadata +
timings + forest fields), so 1M rows = ~300 MB at the cap.
"""

import pytest

from app.config import Settings, load_settings


@pytest.fixture
def _base_env(monkeypatch):
    # Minimum env for load_settings() to succeed.
    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "0")
    monkeypatch.delenv("VW_REQUEST_HISTORY_MAX_ROWS", raising=False)
    monkeypatch.delenv("VW_REQUEST_HISTORY_RETENTION_DAYS", raising=False)
    return monkeypatch


def test_history_defaults_cover_the_forest(_base_env, tmp_path):
    _base_env.setenv("VW_DATA_DIR", str(tmp_path))
    s = load_settings()
    assert s.request_history_retention_days == 30
    assert s.request_history_max_rows == 1_000_000
    assert Settings.__dataclass_fields__["request_history_max_rows"].default == 1_000_000
