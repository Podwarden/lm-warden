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


# --- cobs2 L1 (plan 3 task 7): the discriminator comes from the LOOSE ids ----

import asyncio  # noqa: E402
import threading  # noqa: E402

import pytest  # noqa: E402

from app.cobs2.canonical import canonicalize  # noqa: E402
from app.cobs2.ids import ids_for  # noqa: E402
from app.cobs2.keys import NetworkKey  # noqa: E402
from app.proxy.session_id import conversation_discriminator  # noqa: E402

KEY = NetworkKey(epoch=1, key=b"k" * 32)


def ids(body):
    return ids_for(canonicalize(body), KEY, b"")


def test_cobs2_ids_drive_the_discriminator():
    body = chat("S", "task A")
    i = ids(body)
    key, source = conversation_key(body, HDR, "t", cobs2_ids=i)
    assert source == "claude_code_header"
    assert key == f"{SID}\x00{i.loose_head}:{i.loose_nodes[0]}"
    assert key != conversation_key(body, HDR, "t")[0]


def test_without_ids_behaviour_unchanged():
    for body in (chat("S", "task A"), chat("S", REMINDER + "go", "ok", "more")):
        today = f"{SID}\x00{conversation_discriminator(body)}"
        assert conversation_key(body, HDR, "t") == (today, "claude_code_header")
        assert conversation_key(body, HDR, "t", cobs2_ids=None) == (today, "claude_code_header")
    # prompt_hash and none ignore the ids
    body = chat("S", "hello")
    assert conversation_key(body, {}, "t", cobs2_ids=ids(body)) == affinity_key(body, {}, "t")
    assert conversation_key({}, {}, "t", cobs2_ids=ids(body)) == (None, "none")


def test_head_only_ids_fall_back_to_todays_discriminator():
    body = {"messages": [{"role": "system", "content": "S"}]}
    i = ids(body)
    assert i.loose_nodes == ()
    assert conversation_key(body, HDR, "t", cobs2_ids=i) == conversation_key(body, HDR, "t")


def test_parallel_agents_with_ids_get_distinct_keys():
    a_body, b_body = chat("S", "task A"), chat("S", "task B")
    a = conversation_key(a_body, HDR, "t", cobs2_ids=ids(a_body))
    b = conversation_key(b_body, HDR, "t", cobs2_ids=ids(b_body))
    assert a[1] == b[1] == "claude_code_header"
    assert a[0] != b[0] and a[0].startswith(SID + "\x00")


def test_reminder_change_keeps_the_same_key():
    one = chat("S", "<system-reminder>today is Monday</system-reminder>\ntask A")
    two = chat("S", "<system-reminder>today is Tuesday, files changed</system-reminder>\ntask A")
    # today's discriminator re-homes the conversation on a cosmetic change ...
    assert conversation_key(one, HDR, "t")[0] != conversation_key(two, HDR, "t")[0]
    # ... the loose ids do not
    assert (
        conversation_key(one, HDR, "t", cobs2_ids=ids(one))[0]
        == conversation_key(two, HDR, "t", cobs2_ids=ids(two))[0]
    )


def test_turns_of_one_conversation_keep_the_key():
    t1 = chat("S", "task A")
    t3 = chat("S", "task A", "ok", "next", "done", "and more")
    assert (
        conversation_key(t1, HDR, "t", cobs2_ids=ids(t1))[0]
        == conversation_key(t3, HDR, "t", cobs2_ids=ids(t3))[0]
    )


# --- end to end through _forward with dp > 1 --------------------------------


def _turn(first_user, *rest):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": first_user}]
    for i, t in enumerate(rest):
        msgs.append({"role": "assistant" if i % 2 == 0 else "user", "content": t})
    return {"model": "qwen", "messages": msgs}


@pytest.mark.parametrize("threaded", [False, True])
def test_conversation_stays_on_one_rank_across_turns_with_ids(
    tmp_data_dir, client, monkeypatch, threaded
):
    from app.proxy import routes
    from tests.unit.proxy.test_dp_routing_forward import RANK_HEADER, _post, _ready, _snap

    if threaded:
        monkeypatch.setattr(routes, "_COBS2_THREAD_BYTES", 0)
    seen_ids = []
    real = routes._dp_affinity_key

    def spy(*a, **kw):
        seen_ids.append(kw.get("cobs2_ids"))
        return real(*a, **kw)

    monkeypatch.setattr(routes, "_dp_affinity_key", spy)
    _ready(client, tmp_data_dir)
    assert client.app.state.cobs2_key is not None
    # Two other conversations of the same session first, so a re-home would show.
    _post(client, _turn("other task 1"), headers=HDR)
    _post(client, _turn("other task 2"), headers=HDR)
    r1 = "<system-reminder>Monday</system-reminder>\ntask A"
    r3 = "<system-reminder>Tuesday, 3 files changed</system-reminder>\ntask A"
    _, s1 = _post(client, _turn(r1), headers=HDR)
    _, s2 = _post(client, _turn(r1, "ok", "next"), headers=HDR)
    _, s3 = _post(client, _turn(r3, "ok", "next", "done", "more"), headers=HDR)
    ranks = {s[0].headers[RANK_HEADER] for s in (s1, s2, s3)}
    assert len(ranks) == 1
    assert any(i is not None for i in seen_ids)
    totals = _snap(client)["totals"]
    assert totals["placed"] == 3 and totals["sticky"] == 2


def test_cancel_during_threaded_ids_releases_slot_and_row(tmp_data_dir, client, monkeypatch):
    from app.proxy import routes
    from tests.unit.proxy.test_dp_routing_forward import (
        _direct_forward,
        _engine_inflight,
        _ready,
        _resp,
        _snap,
    )

    monkeypatch.setattr(routes, "_COBS2_THREAD_BYTES", 0)
    entered, release = threading.Event(), threading.Event()
    real = routes._cobs2_compute

    def slow(body, key, *rest):
        entered.set()
        release.wait(5)
        return real(body, key, *rest)

    monkeypatch.setattr(routes, "_cobs2_compute", slow)
    _ready(client, tmp_data_dir)
    go = _direct_forward(client, _turn("task A"), _resp())

    async def run():
        task = asyncio.create_task(go())
        assert await asyncio.to_thread(entered.wait, 5)
        assert _engine_inflight(client) == 1
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        release.set()

    try:
        client.portal.call(run)
    finally:
        release.set()
    assert _engine_inflight(client) == 0
    assert _snap(client)["totals"]["in_flight"] == 0
    assert client.app.state.request_registry.snapshot() == []


# --- fix round 1: the key's ids do not depend on body size (I1, M1, M4) -------


def _route_keys(client):
    """Wrap the dp state's ``route`` and collect the key it is handed."""
    keys = []
    st = client.app.state.dp_routing
    real = st.route

    def route(*a, **kw):
        keys.append(kw.get("key"))
        return real(*a, **kw)

    st.route = route
    return keys


def test_prefix_ids_equal_the_full_compute_ids():
    from app.proxy import routes

    body = {
        "messages": [
            {"role": "system", "content": "S <system-reminder>r</system-reminder>"},
            {"role": "developer", "content": "D"},
            {"role": "user", "content": "task A\n2026-10-25"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
            {"role": "system", "content": "late system note"},
            {"role": "user", "content": "more"},
        ],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "cache_salt": "salt-1",
    }
    full = routes._cobs2_compute(body, KEY).ids
    pre = routes._cobs2_key_ids(body, KEY)
    assert pre.loose_head == full.loose_head
    assert pre.loose_nodes[0] == full.loose_nodes[0]
    assert pre.head == full.head and pre.nodes[0] == full.nodes[0]
    first = {"messages": [{"role": "tool", "tool_call_id": "x", "content": "o"}]}
    assert (
        routes._cobs2_key_ids(first, KEY).loose_nodes
        == routes._cobs2_compute(first, KEY).ids.loose_nodes
    )


def test_key_is_the_same_just_under_and_just_over_the_cap(tmp_data_dir, client, monkeypatch):
    from app.cache_obs import canonical as cobs_canonical
    from app.proxy import routes
    from tests.unit.proxy.test_dp_routing_forward import RANK_HEADER, _post, _ready

    monkeypatch.setattr(cobs_canonical, "MAX_CHAIN_BYTES", 4000)
    computed = []
    real = routes._cobs2_compute

    def spy(body, key, *rest):
        out = real(body, key, *rest)
        computed.append(out)
        return out

    monkeypatch.setattr(routes, "_cobs2_compute", spy)
    _ready(client, tmp_data_dir)
    keys = _route_keys(client)
    _post(client, _turn("other task"), headers=HDR)
    _, s1 = _post(client, _turn("task A", "ok", "x" * 3000), headers=HDR)  # under
    _, s2 = _post(client, _turn("task A", "ok", "x" * 3000, "fine", "y" * 3000), headers=HDR)
    assert computed[1] is not None and computed[2] is None  # crossed the cap
    assert keys[1] == keys[2] and keys[1] != conversation_key(_turn("task A"), HDR, "tok1")[0]
    assert s1[0].headers[RANK_HEADER] == s2[0].headers[RANK_HEADER]


def test_one_off_compute_exception_keeps_the_key(tmp_data_dir, client, monkeypatch):
    from app.proxy import routes
    from tests.unit.proxy.test_dp_routing_forward import RANK_HEADER, _post, _ready

    real = routes._cobs2_compute
    calls = []

    def flaky(body, key, *rest):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("one-off")
        return real(body, key, *rest)

    monkeypatch.setattr(routes, "_cobs2_compute", flaky)
    _ready(client, tmp_data_dir)
    keys = _route_keys(client)
    seen = [
        _post(client, _turn("task A", *rest), headers=HDR)[1]
        for rest in ((), ("ok", "next"), ("ok", "next", "done", "more"))
    ]
    assert keys[0] == keys[1] == keys[2]
    assert len({s[0].headers[RANK_HEADER] for s in seen}) == 1


def test_key_is_todays_when_even_the_prefix_ids_fail(tmp_data_dir, client, monkeypatch):
    from app.proxy import routes
    from tests.unit.proxy.test_dp_routing_forward import _post, _ready

    def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(routes, "_cobs2_compute", boom)
    monkeypatch.setattr(routes, "_cobs2_key_ids", boom)
    _ready(client, tmp_data_dir)
    keys = _route_keys(client)
    body = _turn("task A", "ok", "next")
    r, _ = _post(client, body, headers=HDR)
    assert r.status_code == 200
    assert keys == [conversation_key(body, HDR, "tok1")[0]]


def test_route_supplied_key_is_used_verbatim_with_ids(tmp_data_dir, client, monkeypatch):
    from app.proxy import routes
    from tests.unit.proxy.test_dp_routing_forward import _direct_forward, _ready, _resp

    computed = []
    real = routes._cobs2_compute

    def spy(body, key, *rest):
        out = real(body, key, *rest)
        computed.append(out)
        return out

    monkeypatch.setattr(routes, "_cobs2_compute", spy)
    _ready(client, tmp_data_dir)
    keys = _route_keys(client)
    go = _direct_forward(
        client,
        _turn("task A"),
        _resp(),
        affinity_key="route-key",
        affinity_source="x_session_id",
        session_id="route-key",
    )
    client.portal.call(go)
    assert computed and computed[0] is not None  # ids were present
    assert keys == ["route-key"]
