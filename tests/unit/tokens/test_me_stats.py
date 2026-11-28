"""The key holder's Stats view: /api/me/stats/* and /api/me/requests/{id}/capture.

Owner decisions D2, D5, D6, D8: a key holder sees ONLY their own requests and
nothing fleet-wide -- no other key's figures, names, sessions, prompts or IPs,
no install, engine or host data, no "requests ahead", and no figure derived
from other keys (a fleet-learned rate, a shared-cache measurement). Every route
binds the key from the sign-in; none takes a token id.

Each test seeds TWO keys with distinctive data and asserts that nothing of the
other key (its ids, its name, its IP, its numbers) reaches the response.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import time

import pytest

from app.proxy.request_registry import LiveRequest
from tests.unit.stats.test_forest_routes import _forest_login, _ready_key

OTHER_NAME = "zz-other-key-name"
OTHER_IP = "203.0.113.77"
OTHER_PROMPT = 777_777
OTHER_SESSION = "other-session-secret"

_HCOLS = (
    "id, finished_at, model_id, model, token_id, token_name, client_ip, prompt_tokens, "
    "completion_tokens, duration_s, ttft_s, queued_s, cached_tokens, reusable_tokens, "
    "reusable_tokens_own, cache_outcome, cache_outcome_own, inflight_same_rank_at_start, "
    "finish_reason, orphan, started_iso"
)


def _hist(i, tok, *, at, prompt, own_reuse=None, cached=None, name="mine", ip="10.0.0.5"):
    return (
        f"{tok}-r{i}",
        at,
        "qwen",
        "qwen",
        tok,
        name,
        ip,
        prompt,
        201,
        6.0,
        1.0,
        0.25,
        cached,
        cached,
        own_reuse,
        "hit",
        "hit" if own_reuse else "cold",
        0,
        "stop",
        0,
        "",
    )


def _seed_two_keys(db_path):
    now = time.time()
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES "
            "('tok2', ?, 'vw_zzzz', 'hash-other', 'inference')",
            (OTHER_NAME,),
        )
        rows = [
            _hist(i, "tok1", at=now - 120 - i, prompt=1000, own_reuse=400, cached=600)
            for i in range(6)
        ]
        rows += [
            _hist(
                i,
                "tok2",
                at=now - 100 - i,
                prompt=OTHER_PROMPT,
                cached=OTHER_PROMPT - 1,
                name=OTHER_NAME,
                ip=OTHER_IP,
            )
            for i in range(30)
        ]
        db.executemany(f"INSERT INTO request_history({_HCOLS}) VALUES ({','.join('?' * 21)})", rows)
        db.commit()


def _live(rid, tok, *, phase="answering", prompt=1000, session=None, started_ago=5.0):
    now = time.monotonic()
    r = LiveRequest(
        id=rid,
        token_id=tok,
        token_name=OTHER_NAME if tok == "tok2" else "mine",
        client_ip=OTHER_IP if tok == "tok2" else "10.0.0.5",
        model="qwen",
        model_row_id="qwen",
        path="/v1/chat/completions",
        prompt_tokens=prompt,
        max_model_len=4096,
        started_monotonic=now - started_ago,
        started_iso="",
    )
    r.phase = phase
    r.session_id = session
    if phase != "prefill" and phase != "queued":
        r.first_token_monotonic = now - started_ago + 1.0
        r.completion_tokens = 50
        r.out_samples.append((now - 0.5, 50))
    return r


def _assert_no_other(body):
    text = json.dumps(body)
    for leak in (
        "tok2",
        OTHER_NAME,
        OTHER_IP,
        str(OTHER_PROMPT),
        OTHER_SESSION,
        "client_ip",
        "token_name",
        "prefill_tok_s",
        "est_accuracy",
        "ahead_count",
        "top_diverging",
    ):
        assert leak not in text, leak


@pytest.fixture
def key(tmp_data_dir, client):
    db_path, plaintext, admin = _ready_key(client, tmp_data_dir)
    _seed_two_keys(db_path)
    return {"k": _forest_login(client, plaintext), "admin": admin, "db": db_path}


def test_speed_is_own_rows_only(client, key):
    r = client.get("/api/me/stats/speed?range=1h&model=qwen", headers=key["k"])
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_no_other(body)
    assert body["count"] == 6
    assert body["summary"]["ttft"]["p50"] == 1.25
    assert "prefill" not in body["summary"] and body["summary"]["prompt"]["p50"] == 1000.0
    assert all("cached_tokens" not in p and "prefill" not in p for p in body["points"])
    assert [m["model_id"] for m in body["models"]] == ["qwen"]
    # Without a model: the key's models only, no figures yet.
    assert client.get("/api/me/stats/speed?range=1h", headers=key["k"]).json()["count"] == 6


def test_a_token_id_parameter_is_ignored(client, key):
    body = client.get(
        "/api/me/stats/speed?range=1h&model=qwen&token_id=tok2", headers=key["k"]
    ).json()
    _assert_no_other(body)
    assert body["count"] == 6


def test_tokens_per_minute_own_only(client, key):
    body = client.get("/api/me/stats/tokens?range=1h&model=qwen", headers=key["k"]).json()
    _assert_no_other(body)
    assert sum(m["prompt_tokens"] for m in body["minutes"]) == 6000
    assert sum(m["completion_tokens"] for m in body["minutes"]) == 6 * 201
    assert all(
        set(m) == {"minute", "requests", "prompt_tokens", "completion_tokens"}
        for m in body["minutes"]
    )


def test_cache_is_the_own_lens_without_shared_cache_measurements(client, key):
    body = client.get("/api/me/stats/cache?range=1h&model=qwen", headers=key["k"]).json()
    _assert_no_other(body)
    c = body["cache"]
    assert c["own_reused_tokens"] == 6 * 400
    assert c["own_known_prompt_tokens"] == 6000
    for fleet in (
        "cached_tokens",
        "measured_cached_tokens",
        "reusable_tokens",
        "by_rank",
        "known_prompt_tokens",
        "reused_tokens",
    ):
        assert fleet not in c, fleet
    assert "misrouted" not in c["outcomes"]
    assert c["prefill_saved_basis"] == "configured"


def test_requests_now_own_rows_no_fleet_values(client, key):
    reg = client.app.state.request_registry
    import asyncio

    async def put():
        await reg.register(_live("mine-1", "tok1", session="my-session"))
        await reg.register(_live("mine-2", "tok1", phase="queued", started_ago=1.0))
        await reg.register(_live("other-1", "tok2", phase="prefill", session=OTHER_SESSION))
        await reg.register(_live("other-2", "tok2", phase="prefill", prompt=OTHER_PROMPT))

    asyncio.run(put())
    body = client.get("/api/me/stats/requests", headers=key["k"]).json()
    _assert_no_other(body)
    ids = {r["id"] for r in body["requests"]}
    assert ids == {"mine-1", "mine-2"}
    q = next(r for r in body["requests"] if r["id"] == "mine-2")
    # Waiting estimate from the key's own requests only (own prompt speed 1000
    # tok/s from history; own earlier work ahead: none prefilling). Never a count.
    assert q["expected_start_basis"] == "own"
    assert q["expected_start_s"] == 1.0
    assert body["state"] == {"qwen": "waiting"}


def test_capture_follows_platform_prompt_lens_and_is_own_only(client, key):
    hub = client.app.state.prompt_lens_hub
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=False
    )
    for rid, tok, prompt in (("h-mine", "tok1", "my prompt"), ("h-other", "tok2", "their secret")):
        req_id = "gm-" + rid
        hub.publish(
            {
                "type": "request_start",
                "req_id": req_id,
                "ts": time.time(),
                "token_id": tok,
                "request_id": rid,
                "token_label": "x",
                "client_ip": "1.2.3.4",
                "model": "qwen",
                "served_name": "qwen",
                "stream": True,
                "prompt": prompt,
                "prompt_elided": 0,
            }
        )
        hub.publish(
            {
                "type": "delta",
                "req_id": req_id,
                "token_id": tok,
                "ts": time.time(),
                "channel": "content",
                "text": "reply of " + tok,
            }
        )
        hub.publish(
            {
                "type": "request_end",
                "req_id": req_id,
                "token_id": tok,
                "ts": time.time(),
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 3,
            }
        )
    k = key["k"]
    # Platform Prompt Lens off: 404 (the same answer as for any unknown id).
    assert client.get("/api/me/requests/h-mine/capture", headers=k).status_code == 404
    # On for the platform: the key holder opens their own capture (owner
    # decision 2026-11-22: no per-key switch).
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=True
    )
    r = client.get("/api/me/requests/h-mine/capture", headers=k)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prompt"] == "my prompt" and body["reply"] == "reply of tok1"
    assert body["finish_reason"] == "stop" and body["live"] is False
    assert "admins can see them" in body["note"]
    assert "client_ip" not in body and "token_label" not in body
    # Foreign and unknown ids get the SAME answer.
    foreign = client.get("/api/me/requests/h-other/capture", headers=k)
    unknown = client.get("/api/me/requests/nope/capture", headers=k)
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json()
    assert "their secret" not in foreign.text
    # An admin session is refused (this is the key holder's own area).
    assert client.get("/api/me/requests/h-mine/capture", headers=key["admin"]).status_code == 403


def test_capture_note_says_so_when_the_key_keeps_captures_on_disk(client, key):
    """G-R14: a key with keep_captures on has its last 50 captures on disk, so
    the note must not say "not on disk" for it."""
    from app.tokens.routes_me_stats import CAPTURE_NOTE, CAPTURE_NOTE_KEPT

    hub = client.app.state.prompt_lens_hub
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=True
    )
    for t, kind in ((time.time(), "request_start"), (time.time(), "request_end")):
        ev = {"type": kind, "req_id": "gm-n1", "ts": t, "token_id": "tok1"}
        if kind == "request_start":
            ev.update(request_id="n1", model="qwen", served_name="qwen", stream=True, prompt="p")
        else:
            ev.update(finish_reason="stop", prompt_tokens=1, completion_tokens=1)
        hub.publish(ev)
    k = key["k"]
    body = client.get("/api/me/requests/n1/capture", headers=k).json()
    assert body["note"] == CAPTURE_NOTE and "not on disk" in CAPTURE_NOTE

    with sqlite3.connect(client.app.state.settings.db_path) as db:
        db.execute("UPDATE api_tokens SET keep_captures = 7 WHERE id = 'tok1'")
        db.commit()
    body = client.get("/api/me/requests/n1/capture", headers=k).json()
    assert body["note"] == CAPTURE_NOTE_KEPT
    assert "last 50 captures are kept on disk for this key" in CAPTURE_NOTE_KEPT
    assert "not on disk" not in CAPTURE_NOTE_KEPT and "admins can see them" in CAPTURE_NOTE_KEPT


def test_capture_survives_ring_eviction_of_its_start(client, key):
    """Review finding 18: on a busy box the ring evicts by size, oldest first, so
    a long request's big request_start left memory while it still ran and the
    View said 404. The start is kept in a small index; the reply is marked
    partial when its earliest deltas were evicted."""
    from app.proxy.prompt_lens import PromptLensHub

    hub = PromptLensHub(ring_events=50, ring_chars=2000)
    client.app.state.prompt_lens_hub = hub
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=True
    )
    hub.publish(
        {
            "type": "request_start",
            "req_id": "g1",
            "ts": time.time(),
            "token_id": "tok1",
            "request_id": "long-1",
            "model": "qwen",
            "served_name": "qwen",
            "stream": True,
            "prompt": "p" * 1500,
            "prompt_elided": 0,
        }
    )
    # Another key's traffic leaves the key's own capture alone (Task 7 review
    # I1: it is read from the key's own window, so neither its presence nor
    # its "partial" counts the other keys' events).
    for i in range(40):
        hub.publish(
            {
                "type": "delta",
                "req_id": f"b{i}",
                "token_id": "tok2",
                "ts": time.time(),
                "channel": "content",
                "text": "x" * 100,
            }
        )
    assert (
        client.get("/api/me/requests/long-1/capture", headers=key["k"]).json()["partial"] is False
    )
    for i in range(40):  # the key's OWN later traffic pushes the start out of its window
        hub.publish(
            {
                "type": "delta",
                "req_id": f"o{i}",
                "token_id": "tok1",
                "ts": time.time(),
                "channel": "content",
                "text": "x" * 100,
            }
        )
    hub.publish(
        {
            "type": "delta",
            "req_id": "g1",
            "token_id": "tok1",
            "ts": time.time(),
            "channel": "content",
            "text": "tail",
        }
    )
    assert not any(e.get("request_id") == "long-1" for e in hub.events())
    r = client.get("/api/me/requests/long-1/capture", headers=key["k"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prompt"] == "p" * 1500 and body["reply"] == "tail" and body["partial"] is True
    # Still never another key's: tok2 cannot read it.
    assert client.get("/api/me/requests/o1/capture", headers=key["k"]).status_code == 404


def test_capture_route_is_deprecated_in_the_api_schema(client):
    """The dock reads sections and the key stream instead; the old capture
    route stays one release, marked deprecated in the schema (Task 17)."""
    op = client.app.openapi()["paths"]["/api/me/requests/{request_id}/capture"]["get"]
    assert op.get("deprecated") is True


# ---- anchored window (owner request 2026-11-22) ---------------------------------
#
# A key not used recently must not show empty screens: /api/me/stats/* take an
# optional `until` (ISO time, clamped to now, validated) and the window ends
# there; the speed response carries the key's own `last_request_at`.


def _old_rows(db_path, *, tok="tok1", days=3.0, n=4, prompt=1000):
    at = time.time() - days * 86400
    with sqlite3.connect(db_path) as db:
        db.executemany(
            f"INSERT INTO request_history({_HCOLS}) VALUES ({','.join('?' * 21)})",
            [
                _hist(100 + i, tok, at=at - i * 60, prompt=prompt, own_reuse=400, cached=600)
                for i in range(n)
            ],
        )
        db.commit()
    return at


def _iso(t):
    from datetime import UTC, datetime

    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


def test_idle_key_reports_its_last_request_and_until_anchors_the_window(tmp_data_dir, client):
    db_path, plaintext, _ = _ready_key(client, tmp_data_dir)
    last = _old_rows(db_path)
    k = _forest_login(client, plaintext)
    now = client.get("/api/me/stats/speed?range=1h", headers=k).json()
    assert now["count"] == 0
    assert abs(now["last_request_at"] - last) < 1
    q = f"range=1h&model=qwen&until={_iso(last + 1)}"
    sp = client.get(f"/api/me/stats/speed?{q}", headers=k).json()
    assert sp["count"] == 4 and sp["until"] is not None
    assert [
        m["model_id"]
        for m in client.get(
            f"/api/me/stats/speed?range=1h&until={_iso(last + 1)}", headers=k
        ).json()["models"]
    ] == ["qwen"]
    tk = client.get(f"/api/me/stats/tokens?{q}", headers=k).json()
    assert sum(m["prompt_tokens"] for m in tk["minutes"]) == 4000
    assert tk["now_minute"] == int((last + 1) // 60)
    c = client.get(f"/api/me/stats/cache?{q}", headers=k).json()["cache"]
    assert c["requests"] == 4 and c["own_reused_tokens"] == 1600


def test_never_used_key_has_no_last_request(tmp_data_dir, client):
    _db, plaintext, _ = _ready_key(client, tmp_data_dir)
    body = client.get(
        "/api/me/stats/speed?range=1h", headers=_forest_login(client, plaintext)
    ).json()
    assert body["count"] == 0 and body["last_request_at"] is None and body["models"] == []


def test_until_is_validated_and_clamped(tmp_data_dir, client):
    _db, plaintext, _ = _ready_key(client, tmp_data_dir)
    k = _forest_login(client, plaintext)
    assert client.get("/api/me/stats/speed?range=1h&until=yesterday", headers=k).status_code == 422
    future = client.get(
        f"/api/me/stats/speed?range=1h&until={_iso(time.time() + 86400)}", headers=k
    ).json()
    assert future["until"] <= time.time() + 1


def test_until_never_reveals_another_keys_data(client, key):
    """Anchoring the window on the other key's activity shows nothing of it,
    and last_request_at is the caller's own."""
    other_at = _old_rows(key["db"], tok="tok2", days=5.0, n=6, prompt=OTHER_PROMPT)
    k = key["k"]
    q = f"range=1h&model=qwen&until={_iso(other_at + 1)}"
    for path in ("speed", "tokens", "cache"):
        body = client.get(f"/api/me/stats/{path}?{q}", headers=k).json()
        _assert_no_other(body)
    assert client.get(f"/api/me/stats/speed?{q}", headers=k).json()["count"] == 0
    own = client.get("/api/me/stats/speed?range=1h", headers=k).json()["last_request_at"]
    assert own > other_at + 86400  # tok1's own recent rows, never tok2's
