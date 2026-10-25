"""Conversation-keyed replica affinity: parallel agents sharing one harness
session id must not share a key."""

from app.proxy.dp_affinity import affinity_key, conversation_key, session_id_of
from app.proxy.routes_responses import responses_affinity

HDR = {"X-Claude-Code-Session-Id": "0b6f1d2e-1111-4222-8333-944455556666"}
SID = "0b6f1d2e-1111-4222-8333-944455556666"
REMINDER = "<system-reminder>" + "x" * 2000 + "</system-reminder>"


def chat(system, *users_and_replies):
    msgs = [{"role": "system", "content": system}]
    for i, t in enumerate(users_and_replies):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    return {"messages": msgs}


def test_parallel_agents_in_one_session_get_distinct_keys():
    a = conversation_key(chat("S", "task A"), HDR, "t")
    b = conversation_key(chat("S", "task B"), HDR, "t")
    assert a[1] == b[1] == "claude_code_header"
    assert a[0] != b[0] and a[0].startswith(SID + "\x00")


def test_turns_of_one_conversation_keep_one_key():
    k1 = conversation_key(chat("S", "task A"), HDR, "t")[0]
    k2 = conversation_key(chat("S", "task A", "ok", "next"), HDR, "t")[0]
    assert k1 == k2


def test_different_system_prompt_same_first_message_differs():
    assert (
        conversation_key(chat("S1", "go"), HDR, "t")[0]
        != conversation_key(chat("S2", "go"), HDR, "t")[0]
    )


def test_full_first_message_is_hashed_not_only_512():
    a = conversation_key(chat("S", REMINDER + "task A"), HDR, "t")[0]
    b = conversation_key(chat("S", REMINDER + "task B"), HDR, "t")[0]
    assert a != b


def test_prompt_hash_and_none_paths_are_unchanged():
    body = chat("S", "hello")
    assert conversation_key(body, {}, "t") == affinity_key(body, {}, "t")
    assert conversation_key({}, {}, "t") == (None, "none")


def test_no_user_text_falls_back_to_the_session_id():
    body = {"messages": [{"role": "system", "content": "S"}]}
    assert conversation_key(body, HDR, "t") == (SID, "claude_code_header")


def test_dashboard_session_id_is_still_the_raw_id():
    assert session_id_of(chat("S", "task A"), HDR, None) == SID


def test_responses_affinity_distinguishes_parallel_agents():
    h = {"session-id": "sess-00000001"}
    ka, _, sa = responses_affinity({"instructions": "I", "input": "task A"}, h, "t")
    kb, _, sb = responses_affinity({"instructions": "I", "input": "task B"}, h, "t")
    assert ka != kb and sa == sb == "sess-00000001"
