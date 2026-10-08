"""Session identity per harness (shapes only; every value is fake) and the shared
extractor behind both the dashboard's session column and the DP affinity key."""

import json

import pytest

from app.proxy.dp_affinity import affinity_key, session_id_of
from app.proxy.session_id import (
    extract_session,
    first_user_text,
    opaque_id,
    responses_first_user_text,
)

UUID = "0b1c2d3e-1111-4222-8333-444455556666"
ACCT = "aaaaaaaa-1111-4222-8333-999999999999"
OPENCODE = "ses_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3"  # ses_ + 26 alphanumerics
HERMES = "20260310_141516_a1b2c3"


def src(body=None, headers=None, explicit=None):
    info = extract_session(body or {}, headers or {}, explicit)
    return info.id, info.source


# ---- one test per harness ---------------------------------------------------------


def test_opencode_session_id_and_its_affinity_duplicate():
    assert len(OPENCODE) == 30
    h = {"x-session-id": OPENCODE, "x-session-affinity": OPENCODE}
    assert src(headers=h) == (OPENCODE, "x_session_id")
    assert src(headers={"x-session-affinity": OPENCODE}) == (OPENCODE, "x_session_affinity")


def test_opencode_subagent_parent_is_captured_for_display_only():
    info = extract_session(
        {}, {"x-session-id": OPENCODE, "x-parent-session-id": "ses_parentparent01"}
    )
    assert info.id == OPENCODE and info.parent == "ses_parentparent01"
    key, _ = affinity_key(
        {}, {"x-session-id": OPENCODE, "x-parent-session-id": "ses_parentparent01"}, "t"
    )
    assert key == OPENCODE  # the parent never changes routing


@pytest.mark.parametrize("sid", [HERMES, "cron_nightly-sync"])
def test_hermes_timestamp_and_cron_ids_via_x_session_id(sid):
    assert src(headers={"X-Session-Id": sid}) == (sid, "x_session_id")


def test_pi_underscore_header_and_affinity_header():
    assert src(headers={"session_id": UUID}) == (UUID, "session_id_header")
    assert src(headers={"x-session-affinity": UUID}) == (UUID, "x_session_affinity")


def test_pi_on_responses_prompt_cache_key_and_client_request_id():
    body = {"prompt_cache_key": UUID}
    assert src(body=body) == (UUID, "prompt_cache_key")
    assert src(headers={"x-client-request-id": UUID}) == (UUID, "client_request_id")


def test_codex_dash_header_and_prompt_cache_key_agree():
    h = {
        "session-id": UUID,
        "x-client-request-id": UUID,
        "x-codex-parent-thread-id": "thread-parent-01",
    }
    info = extract_session({"prompt_cache_key": UUID}, h)
    assert (info.id, info.source, info.parent) == (UUID, "session_id_header", "thread-parent-01")
    assert src(body={"prompt_cache_key": UUID}) == (UUID, "prompt_cache_key")


def test_aider_static_header():
    assert (
        src(headers={"X-Session-Id": "6f1d2c3b-aaaa-4bbb-8ccc-0123456789ab"})[1] == "x_session_id"
    )


# ---- precedence ----------------------------------------------------------------------


def test_precedence_runs_in_the_documented_order():
    values = {
        "claude_code_header": ("X-Claude-Code-Session-Id", "claude-hdr-0001"),
        "x_session_id": ("X-Session-Id", "xsess-000001"),
        "x_session_affinity": ("x-session-affinity", "affin-000001"),
        "session_id_header": ("session_id", "under-000001"),
    }
    headers = {name: v for name, v in values.values()}
    headers["session-id"] = "dash-0000001"
    headers["x-client-request-id"] = "reqid-000001"
    body = {"prompt_cache_key": "pck-00000001", "user": "user-000001"}
    uid = json.dumps({"session_id": "meta-0000001"})
    order = [
        ("claude_code_header", "claude-hdr-0001"),
        ("anthropic_metadata", "meta-0000001"),
        ("x_session_id", "xsess-000001"),
        ("x_session_affinity", "affin-000001"),
        ("session_id_header", "under-000001"),
        ("session_id_header", "dash-0000001"),
        ("prompt_cache_key", "pck-00000001"),
        ("openai_user", "user-000001"),
        ("client_request_id", "reqid-000001"),
    ]
    for i, (source, value) in enumerate(order):
        assert src(body, headers, uid) == (value, source), (i, source)
        # remove the winner and the next one must take over
        if i == 0:
            headers.pop("X-Claude-Code-Session-Id")
        elif i == 1:
            uid = None
        elif i == 2:
            headers.pop("X-Session-Id")
        elif i == 3:
            headers.pop("x-session-affinity")
        elif i == 4:
            headers.pop("session_id")
        elif i == 5:
            headers.pop("session-id")
        elif i == 6:
            body.pop("prompt_cache_key")
        elif i == 7:
            body.pop("user")
    assert src(body, headers, uid) == ("reqid-000001", "client_request_id")
    headers.pop("x-client-request-id")
    assert src(body, headers, uid) == (None, None)


def test_the_two_consumers_never_disagree():
    h = {"session-id": UUID}
    body = {"messages": [{"role": "user", "content": "hello"}]}
    assert session_id_of(body, h, None) == UUID
    assert affinity_key(body, h, "tok") == (UUID, "session_id_header")


# ---- shape rule ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "short",
        "has space in it",
        "someone@example.com",
        "x" * 129,
        "bad/slash/char",
        "",
        "ünïcode-id-001",
        7,
        None,
    ],
)
def test_values_that_are_not_opaque_ids_fall_through(bad):
    assert opaque_id(bad) is None
    h = {"X-Session-Id": bad} if isinstance(bad, str) else {}
    # a bad value is ignored and the next signal wins
    assert src(headers={**h, "session-id": UUID}) == (UUID, "session_id_header")
    assert src(body={"user": bad, "prompt_cache_key": UUID}) == (UUID, "prompt_cache_key")


@pytest.mark.parametrize("ok", ["abcdefgh", "a" * 128, "ses_A.b:c-d_1234"])
def test_opaque_ids_are_accepted(ok):
    assert opaque_id(ok) == ok


def test_account_bearing_values_never_leak_from_any_field():
    uid = f"user_deadbeef_account_{ACCT}_session_{UUID}"
    for h in ({"X-Session-Id": uid}, {"session-id": uid}, {"x-client-request-id": uid}):
        got = extract_session({}, h)
        assert got.id == UUID and ACCT not in (got.id or "")
    got = extract_session({"prompt_cache_key": uid}, {})
    assert got.id == UUID
    got = extract_session({"user": json.dumps({"session_id": UUID, "account_uuid": ACCT})}, {})
    assert got.id == UUID
    # an account value with no session part shows and routes on nothing
    nothing = f"user_x_account_{ACCT}"
    assert src(headers={"X-Session-Id": nothing}) == (None, None)
    assert affinity_key({}, {"X-Session-Id": nothing}, "t") == (None, "none")


def test_garbage_input_is_fail_open():
    assert extract_session("junk", {"x": object()}).id is None  # type: ignore[arg-type]
    assert extract_session(None, {}).source is None


# ---- the prompt-hash fallback: Aider preamble --------------------------------------------


AIDER_PREAMBLE = [
    {"role": "system", "content": "Act as an expert software developer."},
    {"role": "user", "content": "Change the greeting to be more casual"},
    {"role": "assistant", "content": "Ok, I will: 1. Switch the greeting."},
]


def aider(first_question, *, turns=0, repo_map=False):
    msgs = list(AIDER_PREAMBLE)
    if repo_map:
        msgs += [
            {
                "role": "user",
                "content": "Here are summaries of some files present in my git repository.",
            },
            {"role": "assistant", "content": "Ok, I won't edit those."},
        ]
    msgs += [{"role": "user", "content": first_question}]
    for i in range(turns):
        msgs += [
            {"role": "assistant", "content": f"answer {i}"},
            {"role": "user", "content": f"follow-up {i}"},
        ]
    return {"messages": msgs}


def test_aider_chats_with_different_first_questions_get_different_keys():
    a = affinity_key(aider("How do I parse the config file?"), {}, "tok")
    b = affinity_key(aider("Why does the build fail on CI?"), {}, "tok")
    assert a[1] == b[1] == "prompt_hash" and a[0] != b[0]


def test_aider_same_chat_keeps_its_key_across_turns():
    first = affinity_key(aider("How do I parse the config file?"), {}, "tok")[0]
    for n in (1, 2, 5):
        assert (
            affinity_key(aider("How do I parse the config file?", turns=n), {}, "tok")[0] == first
        )


def test_aider_reset_pair_is_skipped_too():
    body = {
        "messages": AIDER_PREAMBLE
        + [
            {
                "role": "user",
                "content": "I switched to a new code base. Please don't consider the above files.",
            },
            {"role": "assistant", "content": "Ok."},
            {"role": "user", "content": "Explain the retry logic"},
        ]
    }
    assert first_user_text(body) == "Explain the retry logic"


def test_aider_repo_map_is_combined_with_the_following_user_message():
    a = affinity_key(aider("Add a --verbose flag", repo_map=True), {}, "tok")[0]
    b = affinity_key(aider("Rename the module", repo_map=True), {}, "tok")[0]
    assert a != b  # the shared repo map alone would have pinned both chats together
    text = first_user_text(aider("Add a --verbose flag", repo_map=True))
    assert text.startswith("Here are summaries") and text.endswith("Add a --verbose flag")


def test_a_plain_chat_still_hashes_its_first_user_message():
    body = {
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hello there"}]
    }
    assert first_user_text(body) == "hello there"


# ---- /v1/responses input helper ---------------------------------------------------------------


def test_responses_input_skips_developer_system_and_environment_context():
    items = [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "rules"}],
        },
        {"role": "system", "content": "sys"},
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "<environment_context>\n<cwd>/x</cwd>\n</environment_context>",
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "fix the bug"}],
        },
        {"role": "user", "content": "second"},
    ]
    assert responses_first_user_text(items) == "fix the bug"
    assert first_user_text({"input": items}) == "fix the bug"
    assert responses_first_user_text("just a string") == "just a string"
    assert responses_first_user_text([{"role": "developer", "content": "x"}]) == ""
    assert responses_first_user_text(None) == ""
    key = affinity_key({"input": items}, {}, "tok")
    assert key[1] == "prompt_hash" and "fix the bug" in key[0]
