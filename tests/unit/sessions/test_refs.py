"""Opaque session and request refs (sessions pages, task 3)."""

import pytest

from app.sessions.refs import (
    ParsedRef,
    is_fallback,
    parse_ref,
    request_ref,
    sess_ref,
    session_id,
)


def test_session_ids_are_opaque_and_parse_back() -> None:
    r = sess_ref("sec", "tok1", "abcd" * 4)
    assert r is not None and len(r) == 32 and "abcd" not in session_id(r)
    assert parse_ref(session_id(r)) == ParsedRef("session", r, None)
    assert parse_ref(session_id(r, "f" * 32)) == ParsedRef("session", r, "f" * 12)
    assert parse_ref(request_ref("a" * 32)) == ParsedRef("request", "a" * 32, None)


@pytest.mark.parametrize(
    "bad", ["", "s_", "s_xyz", "r_" + "a" * 31, "S_" + "a" * 32, "../etc", "s_" + "g" * 32]
)
def test_parse_ref_rejects(bad: str) -> None:
    assert parse_ref(bad) is None


def test_sess_ref_scopes_by_token() -> None:
    assert sess_ref("s", "t1", "x") != sess_ref("s", "t2", "x")
    assert sess_ref("s", "t1", None) is None


def test_is_fallback() -> None:
    assert is_fallback("f:abc") and not is_fallback("abc")
