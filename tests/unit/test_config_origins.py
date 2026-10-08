import pytest

from app.config import _parse_origins, _truthy, load_settings


def test_parse_single_origin():
    assert _parse_origins("https://vllm.example.com") == ("https://vllm.example.com",)


def test_parse_comma_list_trims_and_strips_trailing_slash():
    raw = " https://a.example.com/ , https://b.example.com ,"
    assert _parse_origins(raw) == ("https://a.example.com", "https://b.example.com")


def test_parse_empty_string_is_empty_tuple():
    assert _parse_origins("") == ()
    assert _parse_origins("   ") == ()


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "On"])
def test_truthy_accepts(raw):
    assert _truthy(raw) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "maybe"])
def test_truthy_rejects(raw):
    assert _truthy(raw) is False


def test_load_settings_defaults_origins(monkeypatch):
    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    monkeypatch.delenv("VW_FRONTEND_ORIGIN", raising=False)
    monkeypatch.delenv("VW_TRUST_PROXY_ORIGIN", raising=False)
    s = load_settings()
    assert s.allowed_origins == ("http://localhost:3000",)
    assert s.trust_proxy_origin is False


def test_load_settings_parses_env(monkeypatch):
    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    monkeypatch.setenv("VW_FRONTEND_ORIGIN", "https://a.example.com/,https://b.example.com")
    monkeypatch.setenv("VW_TRUST_PROXY_ORIGIN", "1")
    s = load_settings()
    assert s.allowed_origins == ("https://a.example.com", "https://b.example.com")
    assert s.trust_proxy_origin is True


def test_load_settings_explicit_empty_origin_falls_back_to_localhost(monkeypatch):
    # I-1: an explicitly-empty VW_FRONTEND_ORIGIN falls back to the localhost
    # default by design (avoid locking admins out on misconfig). The
    # fail-closed contract lives at the Settings level — see config.py comment.
    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    monkeypatch.setenv("VW_FRONTEND_ORIGIN", "")
    monkeypatch.delenv("VW_TRUST_PROXY_ORIGIN", raising=False)
    s = load_settings()
    assert s.allowed_origins == ("http://localhost:3000",)
