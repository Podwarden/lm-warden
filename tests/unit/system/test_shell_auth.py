"""Guard for the loopback-only shell status feed (GET /api/shell/status).

The shell mints a per-launch bearer (VW_SHELL_TOKEN); the backend admits only
a loopback, un-forwarded client that presents the exact token, compared in
constant time. A non-loopback caller must get a 404 indistinguishable from a
disabled endpoint (no oracle), and the token value must never leak into logs,
exception messages, response bodies or headers.
"""

import asyncio
import hmac
from types import SimpleNamespace

import pytest

TOKEN = "A" * 32


def _scope(*, client=("127.0.0.1", 50000), headers=(), token=TOKEN):
    """A raw ASGI http scope; headers are lowercase-bytes (name, value) pairs."""
    app = SimpleNamespace(state=SimpleNamespace(shell_token=token))
    return {
        "type": "http",
        "method": "GET",
        "path": "/api/shell/status",
        "headers": list(headers),
        "client": client,
        "app": app,
    }


def _run(coro):
    """Drive the async dependency to completion on a fresh loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# parse_shell_token
# ---------------------------------------------------------------------------


def test_parse_unset_returns_none():
    from app.system.shell_auth import parse_shell_token

    assert parse_shell_token({}) is None


def test_parse_empty_returns_none():
    from app.system.shell_auth import parse_shell_token

    assert parse_shell_token({"VW_SHELL_TOKEN": ""}) is None


def test_parse_exactly_32_valid_chars_returns_value():
    from app.system.shell_auth import parse_shell_token

    value = "a" * 32
    assert parse_shell_token({"VW_SHELL_TOKEN": value}) == value


def test_parse_31_chars_raises():
    from app.system.shell_auth import parse_shell_token

    with pytest.raises(ValueError) as exc:
        parse_shell_token({"VW_SHELL_TOKEN": "a" * 31})
    assert "VW_SHELL_TOKEN" in str(exc.value)
    assert "a" * 31 not in str(exc.value)


def test_parse_exactly_256_valid_chars_returns_value():
    from app.system.shell_auth import parse_shell_token

    value = "Z" * 256
    assert parse_shell_token({"VW_SHELL_TOKEN": value}) == value


def test_parse_257_chars_raises():
    from app.system.shell_auth import parse_shell_token

    with pytest.raises(ValueError) as exc:
        parse_shell_token({"VW_SHELL_TOKEN": "b" * 257})
    assert "VW_SHELL_TOKEN" in str(exc.value)
    assert "b" * 257 not in str(exc.value)


def test_parse_space_raises():
    from app.system.shell_auth import parse_shell_token

    value = "a" * 16 + " " + "a" * 15
    with pytest.raises(ValueError) as exc:
        parse_shell_token({"VW_SHELL_TOKEN": value})
    assert "VW_SHELL_TOKEN" in str(exc.value)
    assert value not in str(exc.value)


def test_parse_control_char_raises():
    from app.system.shell_auth import parse_shell_token

    value = "a" * 16 + "\t" + "a" * 15
    with pytest.raises(ValueError) as exc:
        parse_shell_token({"VW_SHELL_TOKEN": value})
    assert "VW_SHELL_TOKEN" in str(exc.value)
    assert value not in str(exc.value)


def test_parse_plus_raises():
    from app.system.shell_auth import parse_shell_token

    value = "a" * 16 + "+" + "a" * 15
    with pytest.raises(ValueError) as exc:
        parse_shell_token({"VW_SHELL_TOKEN": value})
    assert "VW_SHELL_TOKEN" in str(exc.value)
    assert value not in str(exc.value)


def test_parse_slash_raises():
    from app.system.shell_auth import parse_shell_token

    value = "a" * 16 + "/" + "a" * 15
    with pytest.raises(ValueError) as exc:
        parse_shell_token({"VW_SHELL_TOKEN": value})
    assert "VW_SHELL_TOKEN" in str(exc.value)
    assert value not in str(exc.value)


def test_parse_equals_raises():
    from app.system.shell_auth import parse_shell_token

    value = "a" * 16 + "=" + "a" * 15
    with pytest.raises(ValueError) as exc:
        parse_shell_token({"VW_SHELL_TOKEN": value})
    assert "VW_SHELL_TOKEN" in str(exc.value)
    assert value not in str(exc.value)


def test_parse_non_ascii_letter_raises():
    from app.system.shell_auth import parse_shell_token

    value = "a" * 16 + "é" + "a" * 15
    with pytest.raises(ValueError) as exc:
        parse_shell_token({"VW_SHELL_TOKEN": value})
    assert "VW_SHELL_TOKEN" in str(exc.value)
    assert value not in str(exc.value)


def test_parse_shell_token_env_constant():
    from app.system.shell_auth import SHELL_TOKEN_ENV

    assert SHELL_TOKEN_ENV == "VW_SHELL_TOKEN"


# ---------------------------------------------------------------------------
# is_loopback_client
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.255.255.254", "::1", "::ffff:127.0.0.1"],
)
def test_is_loopback_client_true(host):
    from starlette.requests import Request

    from app.system.shell_auth import is_loopback_client

    assert is_loopback_client(Request(_scope(client=(host, 1)))) is True


@pytest.mark.parametrize(
    "client",
    [
        ("10.0.0.5", 1),
        ("192.168.1.2", 1),
        ("::ffff:10.0.0.5", 1),
        ("0.0.0.0", 1),
        ("testclient", 1),
        ("localhost", 1),
        ("", 1),
        None,
    ],
)
def test_is_loopback_client_false(client):
    from starlette.requests import Request

    from app.system.shell_auth import is_loopback_client

    assert is_loopback_client(Request(_scope(client=client))) is False


# ---------------------------------------------------------------------------
# is_forwarded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [b"forwarded", b"x-forwarded-for", b"x-forwarded-host", b"x-real-ip"],
)
@pytest.mark.parametrize("value", [b"10.0.0.5", b""])
def test_is_forwarded_true_each_header_each_value(name, value):
    from starlette.requests import Request

    from app.system.shell_auth import is_forwarded

    assert is_forwarded(Request(_scope(headers=[(name, value)]))) is True


def test_is_forwarded_false_when_absent():
    from starlette.requests import Request

    from app.system.shell_auth import is_forwarded

    assert (
        is_forwarded(Request(_scope(headers=[(b"authorization", b"Bearer " + TOKEN.encode())])))
        is False
    )


# ---------------------------------------------------------------------------
# require_shell_client — refusal order
# ---------------------------------------------------------------------------


def _dep_exception(fn, *args):
    from starlette.exceptions import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _run(fn(*args))
    return excinfo


def test_dependency_token_unset_404_even_correct_header_loopback():
    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    scope = _scope(token=None, headers=[(b"authorization", b"Bearer " + TOKEN.encode())])
    exc = _dep_exception(require_shell_client, Request(scope))
    assert exc.value.status_code == 404
    assert exc.value.detail == "Not Found"


def test_dependency_empty_token_404_even_correct_header_loopback():
    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    scope = _scope(token="", headers=[(b"authorization", b"Bearer " + TOKEN.encode())])
    exc = _dep_exception(require_shell_client, Request(scope))
    assert exc.value.status_code == 404
    assert exc.value.detail == "Not Found"


def test_dependency_non_loopback_with_correct_token_404_not_401():
    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    scope = _scope(
        client=("192.168.1.2", 1),
        headers=[(b"authorization", b"Bearer " + TOKEN.encode())],
    )
    exc = _dep_exception(require_shell_client, Request(scope))
    assert exc.value.status_code == 404
    assert exc.value.detail == "Not Found"


def test_dependency_loopback_forwarded_with_correct_token_404():
    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    scope = _scope(
        client=("127.0.0.1", 1),
        headers=[
            (b"authorization", b"Bearer " + TOKEN.encode()),
            (b"x-forwarded-for", b"10.0.0.5"),
        ],
    )
    exc = _dep_exception(require_shell_client, Request(scope))
    assert exc.value.status_code == 404
    assert exc.value.detail == "Not Found"


def test_dependency_loopback_correct_token_returns_none():
    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    scope = _scope(headers=[(b"authorization", b"Bearer " + TOKEN.encode())])
    assert _run(require_shell_client(Request(scope))) is None


@pytest.mark.parametrize(
    "client,headers",
    [
        (("192.168.1.2", 1), []),  # LAN, no Authorization header
        (("192.168.1.2", 1), [(b"authorization", b"Bearer " + b"x" * 32)]),  # LAN, wrong token
        (("192.168.1.2", 1), [(b"authorization", b"Basic abc")]),  # LAN, malformed header
        (("127.0.0.1", 1), [(b"x-real-ip", b"10.0.0.5")]),  # relayed, no Authorization
        (("127.0.0.1", 1), [(b"forwarded", b""), (b"authorization", b"Bearer " + b"x" * 32)]),
        (None, []),  # no peer at all
    ],
)
def test_dependency_non_shell_peer_404_before_any_auth_check(client, headers):
    """A non-loopback or relayed caller gets the disabled-endpoint 404 whatever its
    Authorization header says -- a 401 here would be an oracle for a LAN scanner."""
    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    exc = _dep_exception(require_shell_client, Request(_scope(client=client, headers=headers)))
    assert exc.value.status_code == 404
    assert exc.value.detail == "Not Found"


def test_dependency_token_unset_404_without_header():
    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    exc = _dep_exception(require_shell_client, Request(_scope(token=None, headers=[])))
    assert exc.value.status_code == 404
    assert exc.value.detail == "Not Found"


def test_dependency_lowercase_scheme_accepted():
    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    scope = _scope(headers=[(b"authorization", b"bearer " + TOKEN.encode())])
    assert _run(require_shell_client(Request(scope))) is None


# ---------------------------------------------------------------------------
# require_shell_client — 401 legs (detail + WWW-Authenticate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "auth_value",
    [
        b"Basic " + TOKEN.encode(),  # wrong scheme
        b"Bearer",  # no token
        b"Bearer  " + TOKEN.encode(),  # two spaces
        b"Bearer " + TOKEN[:-1].encode() + b"b",  # last char changed (same length)
        b"Bearer " + TOKEN.encode() + b"c",  # token + one extra char
        b"Bearer " + TOKEN[:-1].encode(),  # token minus its last char
        b"Bearer ",  # empty token
    ],
)
def test_dependency_401_legs(auth_value):
    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    scope = _scope(headers=[(b"authorization", auth_value)])
    exc = _dep_exception(require_shell_client, Request(scope))
    assert exc.value.status_code == 401
    assert exc.value.detail == "unauthorized"
    assert exc.value.headers.get("WWW-Authenticate") == "Bearer"


def test_dependency_401_header_missing():
    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    scope = _scope(headers=[])
    exc = _dep_exception(require_shell_client, Request(scope))
    assert exc.value.status_code == 401
    assert exc.value.detail == "unauthorized"
    assert exc.value.headers.get("WWW-Authenticate") == "Bearer"


# ---------------------------------------------------------------------------
# constant-time compare: no length short-circuit
# ---------------------------------------------------------------------------


def test_compare_digest_called_once_same_length_wrong_token(monkeypatch):
    from starlette.requests import Request

    import app.system.shell_auth as mod

    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(mod.hmac, "compare_digest", spy)
    scope = _scope(headers=[(b"authorization", b"Bearer " + TOKEN[:-1].encode() + b"b")])
    exc = _dep_exception(mod.require_shell_client, Request(scope))
    assert exc.value.status_code == 401
    assert len(calls) == 1


def test_compare_digest_called_once_different_length_wrong_token(monkeypatch):
    from starlette.requests import Request

    import app.system.shell_auth as mod

    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(mod.hmac, "compare_digest", spy)
    scope = _scope(headers=[(b"authorization", b"Bearer " + TOKEN.encode() + b"c")])
    exc = _dep_exception(mod.require_shell_client, Request(scope))
    assert exc.value.status_code == 401
    assert len(calls) == 1


@pytest.mark.parametrize(
    "headers",
    [
        [(b"authorization", b"Basic " + TOKEN.encode())],  # wrong scheme
        [(b"authorization", b"Bearer")],  # no token
        [(b"authorization", b"Bearer ")],  # empty token
        [(b"authorization", b"Bearer  " + TOKEN.encode())],  # two spaces
        [],  # header missing
    ],
)
def test_compare_digest_not_called_on_malformed_header(monkeypatch, headers):
    from starlette.requests import Request

    import app.system.shell_auth as mod

    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(mod.hmac, "compare_digest", spy)
    # Malformed or missing: must 401 without a constant-time compare.
    scope = _scope(headers=headers)
    exc = _dep_exception(mod.require_shell_client, Request(scope))
    assert exc.value.status_code == 401
    assert len(calls) == 0


# ---------------------------------------------------------------------------
# no secret leakage into logs / exceptions / responses
# ---------------------------------------------------------------------------


def test_parse_never_leaks_value(caplog):
    import logging

    from app.system.shell_auth import parse_shell_token

    for value in ["a" * 31, " " * 32, "\x01" * 32, "é" * 32, "a" * 257]:
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError):
                parse_shell_token({"VW_SHELL_TOKEN": value})
        for record in caplog.records:
            assert value not in record.getMessage()
            for arg in record.args:
                assert value not in str(arg)


def test_dependency_success_leaks_nothing(caplog):
    import logging

    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    with caplog.at_level(logging.DEBUG):
        scope = _scope(headers=[(b"authorization", b"Bearer " + TOKEN.encode())])
        assert _run(require_shell_client(Request(scope))) is None
    for record in caplog.records:
        assert TOKEN not in record.getMessage()
        for arg in record.args:
            assert TOKEN not in str(arg)


def test_dependency_404_and_401_leak_nothing(caplog):
    import logging

    from starlette.exceptions import HTTPException
    from starlette.requests import Request

    from app.system.shell_auth import require_shell_client

    cases = [
        # token unset
        _scope(token=None, headers=[(b"authorization", b"Bearer " + TOKEN.encode())]),
        # non-loopback
        _scope(client=("10.0.0.5", 1), headers=[(b"authorization", b"Bearer " + TOKEN.encode())]),
        # forwarded
        _scope(client=("127.0.0.1", 1), headers=[(b"x-forwarded-for", b"10.0.0.5")]),
        # malformed / wrong tokens
        _scope(headers=[(b"authorization", b"Basic " + TOKEN.encode())]),
        _scope(headers=[(b"authorization", b"Bearer " + TOKEN.encode() + b"c")]),
        _scope(headers=[]),
    ]
    with caplog.at_level(logging.DEBUG):
        for scope in cases:
            try:
                _run(require_shell_client(Request(scope)))
            except HTTPException as exc:
                assert TOKEN not in str(exc.detail)
                assert TOKEN not in str(exc.headers)
    for record in caplog.records:
        assert TOKEN not in record.getMessage()
        for arg in record.args:
            assert TOKEN not in str(arg)
