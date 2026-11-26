"""G7: what a key holder's Prompt Lens stream may carry, and the shared SSE generator.

* ``for_key_holder`` -- the server-side strip (spec 2026-11-22 isolation rule 2):
  identity and fleet fields out, the parts built as if no other key had
  cached anything (ruling G-R11: no ``other`` state, boundary or size), the
  global ``seq`` out. Never mutates the ring.
* the key's own replay window -- what a key holder replays, and which of its
  captures survive, depends on its own traffic only (review I1).
* ``PromptLensHub.subscribe(token_ids=)`` -- a filtered subscriber's queue only
  ever holds its keys' events, so another key's traffic can neither wake it
  nor push it into "lagged".
* ``prompt_lens_sse`` -- ``lagged`` when the client fell behind its queue, the
  timer-based isolation re-check (independent of the keepalive, also under busy
  traffic), and the admin's ``ring`` frames.
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from app.auth.stream_registry import StreamRegistry
from app.cache_obs.canonical import Chain
from app.cache_obs.index import Match
from app.cobs2.ids import HeadPartIds, Ids
from app.cobs2.lineage import Observation
from app.proxy import prompt_lens_stream, routes_prompt_lens
from app.proxy.prompt_lens import PromptLensHub
from app.proxy.prompt_lens_kvsig import build_kvsig, build_kvsig_pair
from app.proxy.prompt_lens_scope import for_key_holder
from app.proxy.prompt_lens_stream import admin_view, prompt_lens_sse
from tests.conftest import wait_until_async

# ---- for_key_holder -----------------------------------------------------------


def test_strips_identity_and_fleet_fields():
    start = {
        "type": "request_start",
        "client_ip": "1.2.3.4",
        "token_label": "k",
        "prompt": "p",
        "token_id": "t",
    }
    assert for_key_holder(start) == {"type": "request_start", "prompt": "p", "token_id": "t"}
    end = {
        "type": "request_end",
        "cached_tokens": 9,
        "cache": {"reusable_tokens": 1},
        "own_reused_tokens": 3,
    }
    assert for_key_holder(end) == {"type": "request_end", "own_reused_tokens": 3}
    kv = {
        "type": "kvsig",
        "reusable_any": 5,
        "reusable_own": 2,
        "parts": [{"state": "other"}, {"state": "exact"}, {"state": "new"}],
    }
    # Ruling G-R11: a stray "other" reads "new" (the real event carries
    # parts_key, below).
    assert for_key_holder(kv) == {
        "type": "kvsig",
        "reusable_own": 2,
        "parts": [{"state": "new"}, {"state": "exact"}, {"state": "new"}],
    }


def test_fleet_figures_are_dropped_from_any_event():
    ev = {
        "type": "delta",
        "token_id": "t",
        "text": "x",
        "reusable_tokens": 1,
        "reusable_tokens_fleet": 2,
        "cache_outcome": "hit",
        "divergence_cause": "other",
        "other_key_label": "bob",
        "reusable_any": 3,
        "cached_tokens": 4,
        "cache": {},
    }
    assert for_key_holder(ev) == {"type": "delta", "token_id": "t", "text": "x"}


def test_does_not_mutate_input():
    e = {
        "type": "kvsig",
        "client_ip": "x",
        "seq": 7,
        "reusable_any": 1,
        "parts": [{"state": "other", "chars": 3}],
    }
    before = copy.deepcopy(e)
    out = for_key_holder(e)
    assert e == before
    assert out["parts"][0] is not e["parts"][0]


def test_global_seq_is_dropped():
    # The hub's seq counts EVERY key's events: the gap between two of a key's
    # own events would tell it how busy the other keys are. The key stream
    # numbers its own frames (below).
    assert for_key_holder({"type": "delta", "seq": 901}) == {"type": "delta"}


# ---- C1 (ruling G-R11): no "other" oracle, not even in part boundaries -----------


def _kvkw(n: int, *, match_all: int | None, match_own: int | None = 0) -> dict[str, Any]:
    step = 300

    def h(tag: str, i: int) -> str:
        return "01" + f"{i:02x}"[-2:] * 15 + tag

    ids = Ids(
        head=h("aa", 0),
        msgs=tuple(h("bb", i) for i in range(n)),
        nodes=tuple("01" + f"{i:02x}"[-2:] * 16 for i in range(n)),
        loose_head=h("cc", 0),
        loose_msgs=tuple(h("dd", i) for i in range(n)),
        loose_nodes=tuple(h("ee", i) for i in range(n)),
    )
    k = n + 2
    return {
        "ids": ids,
        "head_parts": HeadPartIds(
            sys=h("01", 1), tools=h("02", 2), loose_sys=h("03", 3), loose_tools=h("04", 4)
        ),
        "obs": Observation("new", None),
        "chain": Chain((), k * step, tuple(j * step for j in range(k)), 1, ()),
        "match_all": None
        if match_all is None
        else Match({None: match_all}, match_all, 10**9, None),
        "match_own": None
        if match_own is None
        else Match({None: match_own}, match_own, 10**9, None),
        "prompt_tokens": 1000,
        "raw_body_index": [i + 1 for i in range(n)],
        "parts_chars": [100] * n,
        "sys_chars": 100,
        "tools_chars": 100,
        "roles": ["user" if i % 2 == 0 else "assistant" for i in range(n)],
        "reason": None,
    }


def test_key_holder_parts_are_the_parts_without_any_other_key():
    """Exploit (review C1): key B guesses another tenant's prompt and reads
    which parts another key cached. Whatever the other key's span, B's parts
    must be exactly those of a world where no other key sent anything: same
    states, same boundaries, same sizes -- also under the 64-part cap, where
    an ``other`` run moves the cuts."""
    for n, span in ((6, 1500), (6, 2900), (500, 60_000), (500, 140_000)):
        kw = _kvkw(n, match_all=span)
        event = {"type": "kvsig", "token_id": "b", **build_kvsig_pair(**kw)}
        assert any(p["state"] == "other" for p in event["parts"]), (n, span)  # admin sees it
        nobody = build_kvsig(**{**kw, "match_all": None})["parts"]
        assert for_key_holder(event)["parts"] == nobody, (n, span)
        # The guess is unconfirmed: the same view as with no other key at all.
        alone = {"type": "kvsig", "token_id": "b", **build_kvsig_pair(**{**kw, "match_all": None})}
        assert for_key_holder(event) == for_key_holder(alone)


def test_the_admin_keeps_other_and_never_sees_parts_key():
    kw = _kvkw(6, match_all=1500)
    event = {"type": "kvsig", **build_kvsig_pair(**kw)}
    out = admin_view(event)
    assert "parts_key" not in out and "parts_key" in event  # a copy
    assert any(p["state"] == "other" for p in out["parts"])
    assert "parts_key" not in for_key_holder(event)


def test_no_other_part_no_second_build():
    kw = _kvkw(6, match_all=None)
    assert "parts_key" not in build_kvsig_pair(**kw)


# ---- the hub -------------------------------------------------------------------


def _ev(kind: str, tok: str | None, **kw: Any) -> dict[str, Any]:
    e: dict[str, Any] = {"type": kind, "req_id": f"r-{tok}", **kw}
    if tok is not None:
        e["token_id"] = tok
    return e


def test_publish_adds_nothing_but_seq_to_the_shared_dict():
    # Review M5: the ring's dicts are what admin readers see (hub.events());
    # the hub stamps seq, as it always has, and nothing else.
    hub = PromptLensHub()
    for keep in (True, False):
        e = _ev("delta", "a", text="x")
        before = dict(e)
        hub.publish(e, keep=keep)
        assert set(e) == set(before) | {"seq"}


def test_replay_depth_depends_on_the_keys_own_traffic_only():
    """Exploit (review I1): B publishes a burst, other keys publish N events,
    B reconnects and counts what survived to infer N. B's replay must be the
    same whatever N is."""

    def replay_after(n_other: int) -> list[str]:
        hub = PromptLensHub(ring_events=2000)
        for i in range(300):
            hub.publish(_ev("delta", "b", text=f"b{i}"))
        for i in range(n_other):
            hub.publish(_ev("delta", "v", text=f"v{i}"))
        _q, snap = hub.subscribe_key("b")
        return [e["text"] for e in snap]

    quiet, busy, flood = replay_after(0), replay_after(1074), replay_after(50_000)
    assert quiet == busy == flood == [f"b{i}" for i in range(300)]


def test_a_keys_window_is_bounded_by_its_own_traffic():
    from app.proxy import prompt_lens

    hub = PromptLensHub()
    for i in range(prompt_lens.KEY_RING_EVENTS + 50):
        hub.publish(_ev("delta", "b", text=f"b{i}"))
    snap = hub.key_events("b")
    assert len(snap) == prompt_lens.KEY_RING_EVENTS
    assert snap[-1]["text"] == f"b{prompt_lens.KEY_RING_EVENTS + 49}"
    # Starts by request id: per key too, so other keys' starts never push
    # this key's out.
    hub.publish({"type": "request_start", "token_id": "b", "request_id": "mine", "req_id": "g"})
    for i in range(5000):
        hub.publish(
            {"type": "request_start", "token_id": "v", "request_id": f"v{i}", "req_id": f"v{i}"}
        )
    assert hub.key_start_of("b", "mine") is not None


async def test_a_filtered_subscriber_queues_only_its_keys_events():
    hub = PromptLensHub(queue_maxsize=3)
    q, _ = hub.subscribe(token_ids=frozenset({"a"}))
    for _ in range(50):
        hub.publish(_ev("delta", "b", text="other"))
    hub.publish(_ev("delta", "a", text="mine"))
    # Fifty of b's events neither filled a's queue nor marked it lagged.
    assert q.qsize() == 1 and hub.is_lagged(q) is False
    assert q.get_nowait()["text"] == "mine"
    unfiltered, _ = hub.subscribe()
    hub.publish(_ev("delta", "b"))
    assert unfiltered.qsize() == 1 and q.qsize() == 0


# ---- prompt_lens_sse ----------------------------------------------------------------


def _request(hub: PromptLensHub) -> MagicMock:
    request = MagicMock()
    request.app.state.prompt_lens_hub = hub
    request.app.state.stream_registry = StreamRegistry()
    request.is_disconnected = AsyncMock(return_value=False)
    return request


def _data(chunks: list[str]) -> list[dict[str, Any]]:
    return [json.loads(c[len("data: ") : -2]) for c in chunks if c.startswith("data: ")]


async def _drain(resp: Any, *, until: Any, deadline_s: float = 5.0) -> tuple[list[str], bool]:
    """``(chunks, ended)``: ended is True when the generator finished by itself."""
    seen: list[str] = []
    ended = False

    async def run() -> None:
        nonlocal ended
        async for chunk in resp.body_iterator:
            seen.append(chunk)
            if until(seen):
                return
        ended = True

    try:
        await asyncio.wait_for(run(), timeout=deadline_s)
    finally:
        await resp.body_iterator.aclose()
    return seen, ended


async def _always() -> bool:
    return True


async def _key_stream(hub: PromptLensHub, request: Any, recheck: Any = None) -> Any:
    return await prompt_lens_sse(
        request,
        hub,
        user="key:a",
        wanted=frozenset({"a"}),
        transform=for_key_holder,
        ring_meta=False,
        recheck=recheck or _always,
        key_id="a",
    )


async def test_key_stream_replays_and_streams_only_its_key_stripped():
    hub = PromptLensHub()
    hub.publish(_ev("request_start", "a", prompt="mine", client_ip="1.1.1.1", token_label="me"))
    hub.publish(_ev("request_start", "b", prompt="theirs", client_ip="2.2.2.2"))
    request = _request(hub)
    resp = await _key_stream(hub, request)

    def live(seen: list[str]) -> bool:
        if len(seen) == 1:
            hub.publish(_ev("delta", "b", text="b-live"))
            hub.publish(_ev("delta", "a", text="a-live"))
        return any("a-live" in c for c in seen)

    seen, _ = await _drain(resp, until=live)
    body = "".join(seen)
    for leak in ("theirs", "b-live", "r-b", "2.2.2.2", "1.1.1.1", "client_ip", "token_label"):
        assert leak not in body, leak
    events = _data(seen)
    assert [e["type"] for e in events] == ["request_start", "delta"]
    assert [e["seq"] for e in events] == [1, 2]  # the key's own sequence
    assert hub.subscriber_count() == 0


async def test_lagged_frame_when_the_client_fell_behind():
    # G-F15: the server says so; the queue stays bounded (drop-oldest).
    hub = PromptLensHub(queue_maxsize=2)
    resp = await _key_stream(hub, _request(hub))
    for i in range(5):
        hub.publish(_ev("delta", "a", text=f"t{i}"))
    seen, _ = await _drain(resp, until=lambda s: any("t4" in c for c in s))
    events = _data(seen)
    assert events[0] == {"type": "lagged"}
    assert [e.get("text") for e in events[1:]] == ["t3", "t4"]
    # Reported once per gap, not on every later frame.
    assert sum(1 for e in events if e["type"] == "lagged") == 1


async def test_recheck_ends_the_stream_under_busy_traffic(monkeypatch):
    # G-F10: the re-check runs on its own timer. Here the key's own events
    # arrive every 5 ms and the keepalive is a minute away, so a check tied to
    # the keepalive would never run.
    monkeypatch.setattr(routes_prompt_lens, "KEEPALIVE_INTERVAL_S", 60.0)
    monkeypatch.setattr(prompt_lens_stream, "RECHECK_INTERVAL_S", 0.05)
    hub = PromptLensHub()
    calls = 0
    allowed = True

    async def recheck() -> bool:
        nonlocal calls
        calls += 1
        return allowed

    request = _request(hub)
    resp = await _key_stream(hub, request, recheck)
    stop = asyncio.Event()

    async def pump() -> None:
        while not stop.is_set():
            hub.publish(_ev("delta", "a", text="x"))
            await asyncio.sleep(0.005)

    pumping = asyncio.create_task(pump())
    loop = asyncio.get_running_loop()
    t0 = loop.time()

    def revoke_later(seen: list[str]) -> bool:
        nonlocal allowed
        if len(seen) == 20:
            allowed = False
        return False

    try:
        seen, ended = await _drain(resp, until=revoke_later, deadline_s=5.0)
    finally:
        stop.set()
        await pumping
    assert ended, "the stream did not end after the re-check failed"
    assert loop.time() - t0 < 2.0
    assert calls >= 2
    assert hub.subscriber_count() == 0
    assert request.app.state.stream_registry.count("key:a") == 0


async def test_a_failing_recheck_ends_the_stream(monkeypatch):
    # Fail closed: a re-check that cannot answer ends a key holder's stream
    # (the client mints a fresh ticket, which checks again).
    monkeypatch.setattr(prompt_lens_stream, "RECHECK_INTERVAL_S", 0.02)
    hub = PromptLensHub()

    async def recheck() -> bool:
        raise RuntimeError("db locked")

    resp = await _key_stream(hub, _request(hub), recheck)
    _seen, ended = await _drain(resp, until=lambda s: False, deadline_s=3.0)
    assert ended


async def test_recheck_interval_follows_a_patched_keepalive(monkeypatch):
    # The interval is read from routes_prompt_lens at call time (the existing
    # tests patch KEEPALIVE_INTERVAL_S there).
    monkeypatch.setattr(routes_prompt_lens, "KEEPALIVE_INTERVAL_S", 0.02)
    hub = PromptLensHub()

    async def recheck() -> bool:
        return False

    resp = await _key_stream(hub, _request(hub), recheck)
    _seen, ended = await _drain(resp, until=lambda s: False, deadline_s=3.0)
    assert ended


async def test_cancel_user_ends_the_key_stream_by_itself_even_when_busy(monkeypatch):
    # A revoke / rotate / pause / sign-out: cancel_user("key:a"). The stream's
    # generator RETURNS (the response completes) rather than having its task
    # cancelled mid-send, also while the key's own events keep coming.
    monkeypatch.setattr(prompt_lens_stream, "RECHECK_INTERVAL_S", 600.0)
    hub = PromptLensHub()
    request = _request(hub)

    async def ok() -> bool:
        return True

    resp = await _key_stream(hub, request, ok)
    registry = request.app.state.stream_registry
    drained = asyncio.create_task(_drain(resp, until=lambda s: False, deadline_s=5.0))
    await wait_until_async(lambda: registry.count("key:a") == 1, what="the stream to register")
    for i in range(3000):  # more than the queue holds
        hub.publish(_ev("delta", "a", text=f"t{i}"))
    assert registry.cancel_user("key:a") == 1
    seen, ended = await asyncio.wait_for(drained, 3.0)
    assert ended
    assert hub.subscriber_count() == 0
    assert registry.count("key:a") == 0
    # Nothing published after the cancel was sent.
    hub.publish(_ev("delta", "a", text="after"))
    assert not any("after" in c for c in seen)


async def test_admin_stream_sends_ring_frames(monkeypatch):
    monkeypatch.setattr(prompt_lens_stream, "RING_INTERVAL_S", 0.05)
    hub = PromptLensHub()
    hub.publish({"type": "request_start", "req_id": "r1", "ts": 5.0, "token_id": "a"})
    resp = await prompt_lens_sse(
        _request(hub),
        hub,
        user="session:admin",
        wanted=None,
        transform=prompt_lens_stream.admin_view,
        ring_meta=True,
        recheck=None,
    )
    seen, _ = await _drain(
        resp, until=lambda s: sum(1 for e in _data(s) if e["type"] == "ring") >= 2
    )
    assert seen[0] == routes_prompt_lens.CONNECTED_FRAME
    events = _data(seen)
    assert events[0] == {"type": "ring", **hub.ring_span()}
    assert events[1]["req_id"] == "r1"
    assert events[2]["type"] == "ring"


async def test_key_stream_never_sends_ring_frames(monkeypatch):
    monkeypatch.setattr(prompt_lens_stream, "RING_INTERVAL_S", 0.01)
    monkeypatch.setattr(routes_prompt_lens, "KEEPALIVE_INTERVAL_S", 0.05)
    hub = PromptLensHub()
    resp = await _key_stream(hub, _request(hub))
    seen, _ = await _drain(resp, until=lambda s: s.count(": keepalive\n\n") >= 3)
    assert _data(seen) == []


# ---- Task 6 carry: the final progress never costs a request its end ----------------


def test_a_failing_final_progress_still_publishes_request_end():
    from app.proxy.routes import _GmEnd

    hub = PromptLensHub()
    gm = _GmEnd(hub, "r1", "a", 3)

    class Boom:
        def publish(self, _completion_tokens: Any) -> None:
            raise RuntimeError("progress failed")

    gm.progress = Boom()  # type: ignore[assignment]
    gm.end("stop", prompt_tokens=3, completion_tokens=2)
    ends = [e for e in hub.events() if e["type"] == "request_end"]
    assert len(ends) == 1 and ends[0]["finish_reason"] == "stop"


async def test_seq_is_per_connection_and_says_nothing_about_other_keys():
    # Review M1: 1, 2, 3, ... on each connection, whatever other keys publish
    # in between, and never repeated within a connection.
    hub = PromptLensHub()
    hub.publish(_ev("delta", "a", text="old"))
    for _ in range(700):
        hub.publish(_ev("delta", "b", text="x"))
    resp = await _key_stream(hub, _request(hub))

    def live(seen: list[str]) -> bool:
        if len(seen) == 2:
            for i in range(3):
                hub.publish(_ev("delta", "b", text="x"))
                hub.publish(_ev("delta", "a", text=f"a{i}"))
        return any("a2" in c for c in seen)

    seen, _ = await _drain(resp, until=live)
    assert [e["seq"] for e in _data(seen)] == [1, 2, 3, 4]
    resp = await _key_stream(hub, _request(hub))
    seen, _ = await _drain(resp, until=lambda s: any("a2" in c for c in s))
    assert [e["seq"] for e in _data(seen)] == [1, 2, 3, 4]


async def test_an_unstarted_stream_gives_back_its_subscription(monkeypatch):
    # A client that leaves before the body starts must not hold a slot of the
    # key's stream cap forever.
    monkeypatch.setattr(prompt_lens_stream, "START_GRACE_S", 0.05)
    hub = PromptLensHub()
    request = _request(hub)
    await _key_stream(hub, request)
    registry = request.app.state.stream_registry
    assert registry.count("key:a") == 1 and hub.subscriber_count() == 1
    await wait_until_async(
        lambda: registry.count("key:a") == 0 and hub.subscriber_count() == 0,
        timeout_s=3.0,
        what="the abandoned stream to let go",
    )


def test_access_log_never_carries_a_ticket():
    """Review M3: uvicorn's access line is ``'%s - "%s %s HTTP/%s" %d'`` with
    the full path; the warden's filter rewrites it before any handler runs."""
    import logging

    from app.auth.sse_tickets import RedactTicketFilter, install_access_log_redaction

    install_access_log_redaction()
    install_access_log_redaction()  # once
    log = logging.getLogger("uvicorn.access")
    assert sum(isinstance(f, RedactTicketFilter) for f in log.filters) == 1
    rec = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:5", "GET", "/api/me/prompt-lens/stream?a=1&ticket=eyJzdWIi.c2ln&b=2", "1.1", 200),
        None,
    )
    for f in log.filters:
        f.filter(rec)
    line = rec.getMessage()
    assert "eyJzdWIi" not in line and "c2ln" not in line
    assert "ticket=[redacted]&b=2" in line and "a=1" in line


# ---- fix round 2 -------------------------------------------------------------------


def _world(other_cached: bool, fill: int) -> PromptLensHub:
    """Key B's guessed prefix P, then ``fill`` own deltas, in a hub where
    another key did (or did not) cache P. Everything B could observe must be
    the same in both worlds."""
    hub = PromptLensHub(key_ring_chars=3000)
    kw = _kvkw(6, match_all=2900 if other_cached else None)
    hub.publish(
        {
            "type": "request_start",
            "req_id": "gP",
            "token_id": "b",
            "request_id": "P",
            "prompt": "p" * 400,
        }
    )
    hub.publish({"type": "kvsig", "req_id": "gP", "token_id": "b", **build_kvsig_pair(**kw)})
    end: dict[str, Any] = {
        "type": "request_end",
        "req_id": "gP",
        "token_id": "b",
        "finish_reason": "stop",
    }
    if other_cached:
        end["cached_tokens"] = 900
        end["cache"] = {
            "cache_outcome": "hit-other-key",
            "divergence_cause": "another-tenant-prefix",
            "other_key_label": "zz-other-tenant-name" * 5,
        }
    hub.publish(end)
    for i in range(fill):
        hub.publish({"type": "delta", "req_id": "gQ", "token_id": "b", "text": f"{i:04d}" * 10})
    return hub


def test_the_key_window_is_the_same_whether_or_not_another_key_cached_the_guess():
    """Exploit (review N1): B tops its window up to just under the budget in
    the no-other-key world and watches whether P's start survives (replay or
    the capture's ``partial``). Charged with the raw events, the other key's
    fields tipped the eviction. Now every fill gives identical windows."""
    from app.tokens.routes_me_stats import _capture_events

    for fill in range(0, 80):
        a, b = _world(True, fill), _world(False, fill)
        assert a.key_chars("b") == b.key_chars("b"), fill
        assert a.key_events("b") == b.key_events("b"), fill
        assert (a.key_start_of("b", "P") is None) == (b.key_start_of("b", "P") is None), fill
        assert _capture_events(a, "P", "b") == _capture_events(b, "P", "b"), fill
    # The window keeps the key's view only: nothing the key could not see.
    text = json.dumps(_world(True, 0).key_events("b"))
    for hidden in ("other", "parts_key", "cache_outcome", "zz-other-tenant", "cached_tokens"):
        assert hidden not in text, hidden


def test_kept_starts_count_against_the_window_budget():
    # Review N2: a start that left the events is still memory.
    hub = PromptLensHub(key_ring_chars=20_000)
    for i in range(64):
        hub.publish(
            {
                "type": "request_start",
                "req_id": f"g{i}",
                "token_id": "b",
                "request_id": f"r{i}",
                "prompt": "p" * 2000,
            }
        )
    for _ in range(500):
        hub.publish({"type": "delta", "req_id": "gx", "token_id": "b", "text": "d" * 50})
        assert hub.key_chars("b") <= 20_000
    w = hub._keys["b"]
    assert w.chars == sum(c for _e, c in w.events)
    assert w.orphan == sum(c for _e, c, o in w.starts.values() if o)
    # The newest starts survive longest.
    kept = [r for r in w.starts]
    assert kept == sorted(kept, key=lambda r: int(r[1:])) and (not kept or kept[-1] == "r63")


async def test_a_stream_that_starts_after_it_was_given_up_sends_nothing(monkeypatch):
    # Review N3: past the grace its slot, registration and re-check are gone.
    monkeypatch.setattr(prompt_lens_stream, "START_GRACE_S", 0.02)
    hub = PromptLensHub()
    hub.publish(_ev("delta", "a", text="old"))
    request = _request(hub)
    resp = await _key_stream(hub, request)
    registry = request.app.state.stream_registry
    await wait_until_async(lambda: registry.count("key:a") == 0, timeout_s=3.0, what="give-up")
    seen, ended = await _drain(resp, until=lambda s: False, deadline_s=2.0)
    assert (seen, ended) == ([], True)
    assert hub.subscriber_count() == 0 and registry.count("key:a") == 0


# ---- restored starts: a dock that connects mid-request (Task 17 concern 1) ----------


def _long_request(
    hub: PromptLensHub, tok: str, rid: str, *, deltas: int, other_cached: bool = False
) -> None:
    hub.publish(
        {
            "type": "request_start",
            "req_id": f"g-{rid}",
            "token_id": tok,
            "request_id": rid,
            "model": "m2",
            "served_name": "qwen",
            "stream": True,
            "prompt": f"prompt of {rid}",
            "client_ip": "9.9.9.9",
            "token_label": "lbl",
        }
    )
    kw = _kvkw(6, match_all=2900 if other_cached else None)
    hub.publish({"type": "kvsig", "req_id": f"g-{rid}", "token_id": tok, **build_kvsig_pair(**kw)})
    for i in range(deltas):  # one token per delta: the start leaves the replay
        hub.publish(
            {
                "type": "delta",
                "req_id": f"g-{rid}",
                "token_id": tok,
                "channel": "content",
                "text": f"t{i} ",
            }
        )


def test_admin_replay_restores_the_start_of_a_live_request():
    hub = PromptLensHub(ring_events=200)
    _long_request(hub, "a", "R1", deltas=500)
    q, snap = hub.subscribe()
    assert not any(e["type"] == "request_start" for e in snap)  # the defect
    restored = hub.restore(snap)
    assert [e["type"] for e in restored] == ["request_start", "kvsig"]
    assert all(e["restored"] is True for e in restored)
    assert restored[0]["prompt"] == "prompt of R1" and restored[0]["served_name"] == "qwen"
    # The ring's own dicts are untouched (copies).
    assert "restored" not in hub._live["g-R1"][0]
    # Finished: nothing to restore. A filter applies as on the stream.
    assert hub.restore(snap, frozenset({"b"})) == []
    hub.publish({"type": "request_end", "req_id": "g-R1", "token_id": "a", "finish_reason": "stop"})
    assert hub.restore(hub.subscribe()[1]) == []


def test_a_start_still_in_the_replay_is_not_restored():
    hub = PromptLensHub()
    _long_request(hub, "a", "R1", deltas=3)
    _q, snap = hub.subscribe()
    assert hub.restore(snap) == []
    _q, ksnap = hub.subscribe_key("a")
    assert hub.key_restore("a", ksnap) == []


def test_key_replay_restores_its_own_live_starts_in_its_view_only():
    from app.proxy import prompt_lens

    hub = PromptLensHub()
    _long_request(hub, "b", "MINE", deltas=prompt_lens.KEY_RING_EVENTS + 100, other_cached=True)
    _long_request(hub, "v", "THEIRS", deltas=prompt_lens.KEY_RING_EVENTS + 100)
    _q, snap = hub.subscribe_key("b")
    restored = hub.key_restore("b", snap)
    assert [(e["type"], e["req_id"]) for e in restored] == [
        ("request_start", "g-MINE"),
        ("kvsig", "g-MINE"),
    ]
    text = json.dumps(restored)
    for hidden in ("THEIRS", "9.9.9.9", "lbl", '"other"', "parts_key", "reusable_any"):
        assert hidden not in text, hidden
    # The kvsig parts are the no-other-key world's.
    nobody = build_kvsig(**_kvkw(6, match_all=None))["parts"]
    assert restored[1]["parts"] == nobody


def test_restoring_changes_nothing_a_key_holder_could_measure():
    """Two worlds (another key cached the guess, or not; other keys busy or
    idle): identical restored frames and identical window accounting."""
    from app.proxy import prompt_lens

    def world(other: bool) -> PromptLensHub:
        hub = PromptLensHub()
        _long_request(hub, "b", "P", deltas=prompt_lens.KEY_RING_EVENTS + 10, other_cached=other)
        if other:
            for i in range(40):
                _long_request(hub, "v", f"V{i}", deltas=50)
        return hub

    a, b = world(True), world(False)
    sa, sb = a.subscribe_key("b")[1], b.subscribe_key("b")[1]
    assert a.key_restore("b", sa) == b.key_restore("b", sb)
    assert a.key_chars("b") == b.key_chars("b") and sa == sb


async def test_restored_frames_lead_the_replay_and_are_numbered():
    from app.proxy import prompt_lens

    hub = PromptLensHub()
    _long_request(hub, "a", "R1", deltas=prompt_lens.KEY_RING_EVENTS + 5)
    resp = await _key_stream(hub, _request(hub))
    seen, _ = await _drain(resp, until=lambda s: len(_data(s)) >= 4)
    ev = _data(seen)
    assert [e["type"] for e in ev[:3]] == ["request_start", "kvsig", "delta"]
    assert ev[0]["restored"] is True and ev[0]["prompt"] == "prompt of R1"
    assert "client_ip" not in ev[0] and "token_label" not in ev[0]
    assert [e["seq"] for e in ev] == list(range(1, len(ev) + 1))


# ---- the lens block (analysis plan, Task 1.3) ------------------------------------


def _lens(cause):
    return {
        "brk": None,
        "cause": cause,
        "span_class": None,
        "span_line": None,
        "cf_tokens": 900,
        "lost_tokens": 400,
        "other_breaks": 0,
    }


def test_key_holder_never_sees_misrouted():
    """``misrouted`` is a fleet cause (A-1): the key holder's lens says null."""
    ev = {"type": "kvsig", "available": True, "parts": [], "lens": _lens("misrouted")}
    before = copy.deepcopy(ev)
    out = for_key_holder(ev)
    assert out["lens"]["cause"] is None
    assert {k: v for k, v in out["lens"].items() if k != "cause"} == {
        k: v for k, v in _lens(None).items() if k != "cause"
    }
    assert ev == before  # the ring's own dict is never changed
    assert admin_view(ev)["lens"]["cause"] == "misrouted"


def test_key_holder_keeps_its_own_causes():
    for cause in ("sys_changed", "server_evicted", "provider_no_marks", None):
        ev = {"type": "kvsig", "available": True, "parts": [], "lens": _lens(cause)}
        assert for_key_holder(ev)["lens"] == _lens(cause)
    ev = {"type": "kvsig", "available": True, "parts": [], "lens": None}
    assert for_key_holder(ev)["lens"] is None


def test_build_kvsig_lens_block_from_a_verdict():
    from app.lens.causes import LensVerdict
    from app.proxy.prompt_lens_kvsig import lens_block

    v = LensVerdict("sys", "sys_changed", "path", 2, 7920, None, 1)
    assert lens_block(v) == {
        "brk": "sys",
        "cause": "sys_changed",
        "span_class": "path",
        "span_line": 2,
        "cf_tokens": 7920,
        "lost_tokens": None,
        "other_breaks": 1,
    }
    assert lens_block(None) is None
