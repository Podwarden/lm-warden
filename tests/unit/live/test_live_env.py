"""Pure-logic unit tests for tests/live/_env.py -- no network, no real env."""

from __future__ import annotations

from tests.live._env import load_config, missing_vars, redact_base_url

_FULL_ENV = {
    "VW_LIVE_BASE_URL": "https://internal-host.example:8443/",
    "VW_LIVE_ADMIN_USER": "admin",
    "VW_LIVE_ADMIN_PASSWORD": "hunter2",
    "VW_LIVE_MODEL": "some-model",
    "VW_LIVE_ACK_PAUSE_ALL": "1",
}


def test_missing_vars_reports_everything_absent():
    missing = missing_vars({})
    assert "VW_LIVE_BASE_URL" in missing
    assert "VW_LIVE_ADMIN_USER" in missing
    assert "VW_LIVE_ADMIN_PASSWORD" in missing
    assert "VW_LIVE_MODEL" in missing
    assert any("ACK_PAUSE_ALL" in m for m in missing)


def test_missing_vars_empty_when_contract_satisfied():
    assert missing_vars(_FULL_ENV) == []


def test_missing_vars_requires_ack_pause_all_exactly_1():
    env = dict(_FULL_ENV)
    env["VW_LIVE_ACK_PAUSE_ALL"] = "yes"
    missing = missing_vars(env)
    assert any("ACK_PAUSE_ALL" in m for m in missing)


def test_load_config_strips_trailing_slash():
    cfg = load_config(_FULL_ENV)
    assert cfg.base_url == "https://internal-host.example:8443"


def test_state_dir_default_is_not_tempdir(monkeypatch):
    """I4: the crash-recovery state file must survive a reboot -- a bare
    tempfile.gettempdir() (often tmpfs) is exactly what it must not use.

    HOME is pinned: CI runs the suite with HOME=/tmp, where the (correct)
    ~/.local/state default would itself sit under the temp dir."""
    import tempfile

    monkeypatch.setenv("HOME", "/home/tester")
    cfg = load_config(_FULL_ENV)
    assert cfg.state_dir == "/home/tester/.local/state/vw-live"
    assert not cfg.state_dir.startswith(tempfile.gettempdir())


def test_state_dir_env_override_still_wins():
    env = dict(_FULL_ENV)
    env["VW_LIVE_STATE_DIR"] = "/custom/state/dir"
    cfg = load_config(env)
    assert cfg.state_dir == "/custom/state/dir"


def test_load_config_defaults():
    cfg = load_config(_FULL_ENV)
    assert cfg.max_inflight == 16
    assert cfg.priorities == (0, 0, 2, 3, 4, 4, 6, 7, 9, 9)
    assert cfg.probe_labels == ("p0a", "p0b", "p2", "p3", "p4a", "p4b", "p6", "p7", "p9a", "p9b")
    assert cfg.filler_priority == 1
    assert cfg.dry_run is False
    assert cfg.assert_ttft is False


def test_load_config_overrides():
    env = dict(_FULL_ENV)
    env["VW_LIVE_MAX_INFLIGHT"] = "8"
    env["VW_LIVE_PRIORITIES"] = "0,9"
    env["VW_LIVE_DRY_RUN"] = "1"
    env["VW_LIVE_HOLD_S"] = "2.5"
    cfg = load_config(env)
    assert cfg.max_inflight == 8
    assert cfg.priorities == (0, 9)
    assert cfg.probe_labels == ("p0a", "p0b")
    assert cfg.dry_run is True
    assert cfg.hold_s == 2.5


def test_repr_never_leaks_credentials():
    cfg = load_config(_FULL_ENV)
    text = repr(cfg)
    assert "hunter2" not in text
    assert "admin_password" not in text or "redacted" in text
    assert _FULL_ENV["VW_LIVE_ADMIN_USER"] not in text or "redacted" in text


def test_redact_base_url_drops_hostname():
    out = redact_base_url("https://internal-host.example:8443")
    assert "internal-host" not in out
    assert out == "https://<redacted>"


def test_httpx_verify_prefers_ca_bundle():
    env = dict(_FULL_ENV)
    env["VW_LIVE_CA_BUNDLE"] = "/path/to/ca.pem"
    cfg = load_config(env)
    assert cfg.httpx_verify == "/path/to/ca.pem"


def test_httpx_verify_falls_back_to_verify_tls_flag():
    env = dict(_FULL_ENV)
    env["VW_LIVE_VERIFY_TLS"] = "0"
    cfg = load_config(env)
    assert cfg.httpx_verify is False
