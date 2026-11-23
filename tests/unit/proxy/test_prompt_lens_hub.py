"""Unit tests for the PromptLensHub broadcast hub — ring eviction (both bounds),
replay snapshot, fan-out, drop-oldest on a full subscriber, and monotonic seq.
"""

import asyncio

from app.proxy.prompt_lens import PromptLensHub


def _event(text="x", channel="content"):
    return {"type": "delta", "req_id": "r", "channel": channel, "text": text}


def test_publish_assigns_monotonic_seq_and_stamps_event():
    hub = PromptLensHub()
    ev = _event()
    seq = hub.publish(ev)
    assert seq == 1
    assert ev["seq"] == 1
    assert hub.publish(_event()) == 2


def test_ring_evicts_oldest_past_event_count():
    hub = PromptLensHub(ring_events=3, ring_chars=10**9)
    for i in range(10):
        hub.publish(_event(text=str(i)))
    _q, snap = hub.subscribe()
    assert len(snap) == 3
    # Only the three newest survive, in order.
    assert [e["text"] for e in snap] == ["7", "8", "9"]


def test_ring_evicts_oldest_past_char_budget():
    # Budget fits ~2 events of 100 chars; a third pushes past and evicts.
    hub = PromptLensHub(ring_events=10**6, ring_chars=250)
    for _ in range(10):
        hub.publish(_event(text="a" * 100))
    _q, snap = hub.subscribe()
    total = sum(len(e["text"]) for e in snap)
    assert total <= 250
    assert len(snap) <= 3  # each event carries a little non-text overhead too


def test_single_oversized_event_is_kept_not_self_evicted():
    # A single runaway completion bigger than the whole budget must remain
    # visible rather than evicting itself into nothing.
    hub = PromptLensHub(ring_events=10, ring_chars=50)
    hub.publish(_event(text="a" * 5000))
    _q, snap = hub.subscribe()
    assert len(snap) == 1
    assert snap[0]["text"] == "a" * 5000


def test_seq_is_monotonic_across_eviction():
    hub = PromptLensHub(ring_events=2, ring_chars=10**9)
    seqs = [hub.publish(_event()) for _ in range(5)]
    assert seqs == [1, 2, 3, 4, 5]
    _q, snap = hub.subscribe()
    # Ring only holds the last two, but their seqs reflect the true position.
    assert [e["seq"] for e in snap] == [4, 5]


def test_subscribe_returns_current_ring_snapshot_copy():
    hub = PromptLensHub()
    hub.publish(_event(text="one"))
    hub.publish(_event(text="two"))
    _q, snap = hub.subscribe()
    assert [e["text"] for e in snap] == ["one", "two"]
    # The returned snapshot is a copy — a later publish must not mutate it.
    hub.publish(_event(text="three"))
    assert [e["text"] for e in snap] == ["one", "two"]


async def test_publish_fans_out_to_all_subscribers():
    hub = PromptLensHub()
    q1, _ = hub.subscribe()
    q2, _ = hub.subscribe()
    hub.publish(_event(text="hello"))
    assert q1.get_nowait()["text"] == "hello"
    assert q2.get_nowait()["text"] == "hello"


async def test_full_subscriber_drops_oldest_and_publish_never_blocks():
    hub = PromptLensHub(queue_maxsize=2)
    q, _ = hub.subscribe()
    # Publish 5 into a maxsize-2 queue; publish must never block or raise.
    for i in range(5):
        hub.publish(_event(text=str(i)))
    drained = []
    while not q.empty():
        drained.append(q.get_nowait()["text"])
    # Only the two newest survive; the subscriber is flagged lagged.
    assert drained == ["3", "4"]
    assert hub.is_lagged(q) is True


async def test_unsubscribe_stops_delivery():
    hub = PromptLensHub()
    q, _ = hub.subscribe()
    assert hub.subscriber_count() == 1
    hub.unsubscribe(q)
    assert hub.subscriber_count() == 0
    hub.publish(_event(text="after"))
    assert q.empty()


async def test_publish_does_not_raise_when_no_subscribers():
    hub = PromptLensHub()
    # Purely exercises the "no subscriber" fan-out path.
    assert hub.publish(_event()) == 1
    assert not isinstance(hub.subscribe()[0], type(None))
    # sanity: a healthy subscriber that never lags reports not-lagged
    q, _ = hub.subscribe()
    hub.publish(_event())
    assert hub.is_lagged(q) is False
    assert isinstance(q, asyncio.Queue)


# --------------------------------------------------------------------------
# G6: live-only events (keep=False) and the ring span
# --------------------------------------------------------------------------
def test_keep_false_is_not_replayed():
    hub = PromptLensHub()
    hub.publish({"type": "progress"}, keep=False)
    _q, snap = hub.subscribe()
    assert snap == []
    assert hub.events() == []
    assert hub.ring_span() == {"oldest_ts": None, "events": 0, "chars": 0, "chars_max": 4_000_000}


def test_keep_false_reaches_live_subscribers():
    hub = PromptLensHub()
    q, _ = hub.subscribe()
    seq = hub.publish({"type": "progress"}, keep=False)
    got = q.get_nowait()
    assert got["seq"] == seq == 1
    assert got["type"] == "progress"
    # The seq stays monotonic across kept and live-only events.
    assert hub.publish(_event()) == 2


def test_keep_false_request_start_is_not_kept_by_id():
    hub = PromptLensHub()
    hub.publish(
        {"type": "request_start", "request_id": "r1", "req_id": "g1", "token_id": "a"},
        keep=False,
    )
    assert hub.key_start_of("a", "r1") is None
    assert hub.restore([]) == []


def test_keep_false_does_not_evict_the_ring():
    hub = PromptLensHub(ring_events=1, ring_chars=10)
    hub.publish(_event(text="kept"))
    hub.publish({"type": "progress", "phase": "x" * 100}, keep=False)
    _q, snap = hub.subscribe()
    assert [e["text"] for e in snap] == ["kept"]


def test_ring_span_reports_oldest_ts():
    hub = PromptLensHub(ring_chars=20)
    hub.publish({"type": "delta", "ts": 1.0, "text": "a" * 8})
    hub.publish({"type": "delta", "ts": 2.0, "text": "b" * 8})
    # "delta" + 8 chars = 13 per event; the first is evicted past 20.
    assert hub.ring_span() == {"oldest_ts": 2.0, "events": 1, "chars": 13, "chars_max": 20}


def test_ring_span_oldest_ts_none_when_oldest_has_no_ts():
    hub = PromptLensHub()
    hub.publish({"type": "delta", "text": "x"})
    assert hub.ring_span()["oldest_ts"] is None


def test_event_chars_counts_nested_text():
    # G-F8: prompt_parts (messages, older), tool-call deltas and kvsig parts
    # must cost their text, or the char cap understates the heap.
    hub = PromptLensHub(ring_chars=10**9)
    hub.publish(
        {
            "type": "request_start",  # 13
            "request_id": "r1",  # 2
            "prompt_parts": {
                "sys": {"text": "s" * 10},  # 10
                "messages": [
                    {"role": "user", "text": "m" * 20, "tool_calls": [{"name": "ls"}]},  # 4+20+2
                ],
                "older": {"from": 0, "roles": "uat", "chars": [5, 6, 7]},  # 3
            },
        }
    )
    assert hub.ring_span()["chars"] == 13 + 2 + 10 + 4 + 20 + 2 + 3
    hub.publish({"type": "kvsig", "parts": [{"state": "exact", "roles": "ua"}] * 3})
    assert hub.ring_span()["chars"] == 54 + 5 + 3 * (5 + 2)


def test_nested_text_counts_toward_the_char_cap():
    hub = PromptLensHub(ring_events=10**6, ring_chars=250)
    for _ in range(10):
        hub.publish({"type": "kvsig", "parts": [{"label": "a" * 100}]})
    span = hub.ring_span()
    assert span["events"] == 2 and span["chars"] == 2 * 105
