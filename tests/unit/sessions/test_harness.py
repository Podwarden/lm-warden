"""Harness labels from session_source (sessions redesign, task 2, R8)."""

import pytest

from app.sessions.harness import HARNESSES, harness_of, session_harness


@pytest.mark.parametrize(
    "src,h",
    [
        ("claude_code_header", "claude_code"),
        ("anthropic_metadata", "claude_code"),
        ("x_session_id", "opencode"),
        ("x_session_affinity", "opencode"),
        ("session_id_header", "codex"),
        ("prompt_cache_key", "codex"),
        ("openai_user", "openai_sdk"),
        ("client_request_id", "other"),
        ("prompt_hash", "other"),
        (None, "other"),
    ],
)
def test_harness_of(src, h):
    assert harness_of(src) == h


def test_session_harness_is_the_majority():
    assert session_harness(["x_session_id", None, "x_session_id"]) == "opencode"


def test_session_harness_ties_follow_order_and_empty_is_other():
    assert session_harness(["openai_user", "claude_code_header"]) == "claude_code"
    assert session_harness([]) == "other"
    assert HARNESSES[0] == "claude_code"
