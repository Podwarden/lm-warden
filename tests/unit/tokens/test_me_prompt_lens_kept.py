"""Prompt Lens keep captures (ruling G-R4): a key's last 50 finished requests on disk.

Two keys, A (``tok1``) and B (``tok2``), both with interleaved traffic. A key
with ``keep_captures`` on has its finished requests' structured capture and
whole sections written to ``prompt_lens_kept``; they outlive the in-memory store
and a restart, and the same key-filtered routes serve them. Nothing of A's is
ever readable as B, the key holder always gets the key-holder view, deleting
the key or turning the switch off deletes the rows.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from app.proxy import prompt_lens_kept
from app.proxy.prompt_lens import PromptLensHub
from app.proxy.prompt_lens_kept import KEPT_PER_KEY, KeptCaptures
from app.proxy.prompt_lens_sections import PromptLensSections
from tests.unit.tokens.test_me_prompt_lens_stream import IP, NAME, A, B, Env, _set_token, env

__all__ = ["env"]  # the fixture, re-used


def _kept_rows(db: Path, tid: str | None = None) -> list[str]:
    with sqlite3.connect(db) as c:
        if tid is None:
            rows = c.execute("SELECT req_id FROM prompt_lens_kept ORDER BY ended_ts").fetchall()
        else:
            rows = c.execute(
                "SELECT req_id FROM prompt_lens_kept WHERE token_id = ? ORDER BY ended_ts", (tid,)
            ).fetchall()
    return [r[0] for r in rows]


def _gen(db: Path, tid: str) -> int:
    """The key's keep_captures generation now (0 = off), as the proxy reads it."""
    with sqlite3.connect(db) as c:
        return int(
            c.execute("SELECT keep_captures FROM api_tokens WHERE id = ?", (tid,)).fetchone()[0]
        )


def _request(env: Env, tid: str, n: int, *, keep: bool) -> str:
    """One request of key ``tid`` through the hub, as the proxy would."""
    rid = f"gm-{tid}-k-{n}"
    secret = f"secret-of-{tid}-{n}"
    hub = env.hub
    hub.publish(
        {
            "type": "request_start",
            "req_id": rid,
            "ts": time.time(),
            "token_label": NAME[tid],
            "token_id": tid,
            "request_id": f"req-{tid}-k-{n}",
            "model": "qwen",
            "served_name": "qwen",
            "client_ip": IP[tid],
            "stream": True,
            "prompt": f"prompt {secret}",
            "prompt_parts": {"messages": [{"i": 0, "role": "user", "text": secret}]},
            "session_id": f"sess-{tid}",
        }
    )
    env.app.state.prompt_lens_sections.put_prompt(
        rid,
        tid,
        {
            "sys": {"chars": 3, "text": "sys", "elided": False},
            "tools": {"count": 0, "chars": 0, "names": []},
            "messages": [
                {"i": 0, "role": "user", "text": secret},
                {"i": 1, "role": "tool", "text": f"tool result {secret}"},
            ],
            "older": {"from": 0, "roles": "", "chars": []},
            "collapsed": 0,
            "omitted": 0,
            "total_chars": 40,
        },
        keep=_gen(env.db, tid) if keep else 0,
    )
    hub.publish(
        {
            "type": "kvsig",
            "req_id": rid,
            "token_id": tid,
            "ts": time.time(),
            "reusable_any": 99,
            "parts": [{"p": "sys", "state": "other"}, {"p": "nodes", "state": "new"}],
            "parts_key": [{"p": "sys", "state": "new"}, {"p": "nodes", "state": "new"}],
        }
    )
    hub.publish(
        {
            "type": "delta",
            "req_id": rid,
            "token_id": tid,
            "channel": "reasoning",
            "text": f"think {secret}",
        }
    )
    hub.publish(
        {
            "type": "delta",
            "req_id": rid,
            "token_id": tid,
            "channel": "content",
            "text": f"reply {secret}",
        }
    )
    hub.publish(
        {
            "type": "delta",
            "req_id": rid,
            "token_id": tid,
            "channel": "tool_call",
            "index": 0,
            "call_id": "c0",
            "name": "bash",
            "text": "{}",
        }
    )
    hub.publish(
        {
            "type": "request_end",
            "req_id": rid,
            "token_id": tid,
            "ts": time.time() + n * 0.001,
            "finish_reason": "stop",
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "cached_tokens": 987654321,
            "cache": {"reusable_tokens": 876543219, "cache_outcome": "hit"},
            "own_reused_tokens": 3,
        }
    )
    return rid


async def _keep(env: Env, tid: str, on: bool) -> None:
    await asyncio.to_thread(_set_token, env.db, tid, keep_captures=str(int(on)))


async def _flush(env: Env) -> None:
    await env.app.state.prompt_lens_kept.flush()


def _forget_memory(env: Env) -> None:
    """Memory loses everything -- the ring, the key windows and the section
    store (ring pressure, the TTL, a restart)."""
    fresh = PromptLensSections(on_kept=env.app.state.prompt_lens_kept.offer)
    env.app.state.prompt_lens_sections = fresh
    env.app.state.prompt_lens_hub = PromptLensHub(sections=fresh)


_STRIPPED = (
    "client_ip",
    "token_label",
    "cached_tokens",
    "cache",
    "reusable_any",
    "parts_key",
    "seq",
)


async def test_a_keeping_key_writes_its_finished_requests_and_nothing_else(env: Env) -> None:
    await _keep(env, A, True)
    a = _request(env, A, 0, keep=True)
    b = _request(env, B, 0, keep=False)
    await _flush(env)
    assert _kept_rows(env.db) == [a]
    assert b not in _kept_rows(env.db)


async def test_kept_rows_are_pruned_to_fifty_per_key(env: Env) -> None:
    await _keep(env, A, True)
    await _keep(env, B, True)
    rids = []
    for n in range(KEPT_PER_KEY + 5):
        rids.append(_request(env, A, n, keep=True))
        await _flush(env)  # the writer runs as requests finish
    _request(env, B, 0, keep=True)
    await _flush(env)
    assert _kept_rows(env.db, A) == rids[-KEPT_PER_KEY:]
    assert len(_kept_rows(env.db, B)) == 1


async def test_kept_sections_are_served_after_memory_is_gone(env: Env) -> None:
    await _keep(env, A, True)
    rid = _request(env, A, 0, keep=True)
    await _flush(env)
    _forget_memory(env)
    for kind, check in (
        ("reply", lambda b: b["text"] == f"reply secret-of-{A}-0"),
        ("thinking", lambda b: b["text"] == f"think secret-of-{A}-0"),
        ("tool_calls", lambda b: b["calls"][0]["name"] == "bash"),
        ("tool_results", lambda b: b["messages"][0]["text"] == f"tool result secret-of-{A}-0"),
        (
            "prompt",
            lambda b: [m["role"] for m in b["messages"]] == ["sys", "tools", "user", "tool"],
        ),
    ):
        mine = await env.http.get(
            f"/api/me/prompt-lens/requests/{rid}/section/{kind}", headers=env.key[A]
        )
        assert mine.status_code == 200, (kind, mine.text)
        body = mine.json()
        assert body["kept"] is True and body["live"] is False and body["seq"] is None
        assert check(body), (kind, body)
        assert "token_id" not in body
        adm = await env.http.get(
            f"/api/admin/prompt-lens/requests/{rid}/section/{kind}", headers=env.admin
        )
        assert adm.status_code == 200 and adm.json()["kept"] is True
        assert isinstance(adm.json()["seq"], int)
    r = await env.http.get(
        f"/api/me/prompt-lens/requests/{rid}/section/reply?from=6", headers=env.key[A]
    )
    assert (r.json()["offset"], r.json()["text"]) == (6, f"secret-of-{A}-0")


async def test_kept_captures_of_a_are_never_visible_to_b(env: Env) -> None:
    await _keep(env, A, True)
    await _keep(env, B, True)
    a = _request(env, A, 0, keep=True)
    b = _request(env, B, 0, keep=True)
    await _flush(env)
    _forget_memory(env)
    lst = await env.http.get("/api/me/prompt-lens/kept", headers=env.key[B])
    assert lst.status_code == 200
    assert [k["req_id"] for k in lst.json()["kept"]] == [b]
    assert f"secret-of-{A}" not in lst.text and A not in lst.text
    for path in (
        f"/api/me/prompt-lens/kept/{a}",
        f"/api/me/prompt-lens/requests/{a}/section/reply",
        f"/api/me/prompt-lens/requests/{a}/section/prompt",
        f"/api/me/requests/req-{A}-k-0/capture",
    ):
        r = await env.http.get(path, headers=env.key[B])
        unknown = await env.http.get(
            path.replace(a, "gm-nope").replace(f"req-{A}", "req-nope"), headers=env.key[B]
        )
        assert r.status_code == unknown.status_code == 404, path
        assert r.json() == unknown.json() == {"detail": "no capture of yours with this id"}
        assert f"secret-of-{A}" not in r.text


async def test_key_holder_reads_of_kept_captures_are_stripped(env: Env) -> None:
    await _keep(env, A, True)
    rid = _request(env, A, 0, keep=True)
    await _flush(env)
    _forget_memory(env)
    # Stored as the admin sees it...
    adm = await env.http.get(f"/api/admin/prompt-lens/kept/{rid}", headers=env.admin)
    assert adm.status_code == 200, adm.text
    a_events = adm.json()["events"]
    assert a_events[0]["client_ip"] == IP[A] and a_events[2]["cache"]["cache_outcome"] == "hit"
    assert a_events[1]["parts"][0]["state"] == "other"
    # ...and every key-holder read strips it.
    mine = await env.http.get(f"/api/me/prompt-lens/kept/{rid}", headers=env.key[A])
    assert mine.status_code == 200, mine.text
    body = mine.json()
    assert body["kept"] is True and "token_id" not in body
    events = body["events"]
    assert [e["type"] for e in events] == ["request_start", "kvsig", "request_end"]
    assert not any(k in e for e in events for k in _STRIPPED)
    assert [p["state"] for p in events[1]["parts"]] == ["new", "new"]
    for leak in (IP[A], NAME[A], '"other"', "987654321", "876543219"):
        assert leak not in mine.text, leak
    assert body["sections"]["reply"]["chars"] == len(f"reply secret-of-{A}-0")
    lst = await env.http.get("/api/me/prompt-lens/kept", headers=env.key[A])
    assert [k["req_id"] for k in lst.json()["kept"]] == [rid]
    assert IP[A] not in lst.text and NAME[A] not in lst.text and "token_id" not in lst.text
    cap = await env.http.get(f"/api/me/requests/req-{A}-k-0/capture", headers=env.key[A])
    assert cap.status_code == 200, cap.text
    assert cap.json()["reply"] == f"reply secret-of-{A}-0"
    assert cap.json()["reasoning"] == f"think secret-of-{A}-0"
    assert cap.json()["kept"] is True
    assert IP[A] not in cap.text and NAME[A] not in cap.text


async def test_the_admin_kept_route_sends_what_the_admin_stream_sends(env: Env) -> None:
    """Final review 2: the same kvsig has the same bytes on both admin routes
    -- the admin view, without ``parts_key`` (the key holder's parts, which
    never leave the server)."""
    from app.proxy.prompt_lens_scope import admin_view

    await _keep(env, A, True)
    rid = _request(env, A, 0, keep=True)
    live_kv = next(e for e in env.hub.events() if e["type"] == "kvsig" and e["req_id"] == rid)
    await _flush(env)
    _forget_memory(env)
    adm = await env.http.get(f"/api/admin/prompt-lens/kept/{rid}", headers=env.admin)
    assert adm.status_code == 200, adm.text
    assert "parts_key" not in adm.text
    kept_kv = next(e for e in adm.json()["events"] if e["type"] == "kvsig")
    assert kept_kv["parts"] == admin_view(live_kv)["parts"]
    assert set(kept_kv) <= set(admin_view(live_kv))


async def test_off_or_unknown_or_foreign_kept_ids_get_one_404(env: Env) -> None:
    await _keep(env, A, True)
    rid = _request(env, A, 0, keep=True)
    await _flush(env)
    _forget_memory(env)
    env.prompt_lens(False)
    off = await env.http.get(f"/api/me/prompt-lens/kept/{rid}", headers=env.key[A])
    off_list = await env.http.get("/api/me/prompt-lens/kept", headers=env.key[A])
    off_sec = await env.http.get(
        f"/api/me/prompt-lens/requests/{rid}/section/reply", headers=env.key[A]
    )
    assert off.status_code == off_list.status_code == off_sec.status_code == 404
    assert off.json() == off_sec.json() == {"detail": "no capture of yours with this id"}
    # The admin still reads kept captures while Prompt Lens is off (read-only).
    adm = await env.http.get(f"/api/admin/prompt-lens/kept/{rid}", headers=env.admin)
    assert adm.status_code == 200, adm.text
    assert adm.json()["req_id"] == rid
    adm_list = await env.http.get("/api/admin/prompt-lens/kept", headers=env.admin)
    assert [k["req_id"] for k in adm_list.json()["kept"]] == [rid]
    adm_sec = await env.http.get(
        f"/api/admin/prompt-lens/requests/{rid}/section/reply", headers=env.admin
    )
    assert adm_sec.status_code == 200 and adm_sec.json()["kept"] is True


async def test_deleting_a_key_deletes_its_kept_rows(env: Env) -> None:
    await _keep(env, A, True)
    await _keep(env, B, True)
    _request(env, A, 0, keep=True)
    b = _request(env, B, 0, keep=True)
    await _flush(env)
    r = await env.http.delete(f"/api/tokens/{A}", headers=await env.admin_csrf())
    assert r.status_code in (200, 204), r.text
    assert _kept_rows(env.db, A) == [] and _kept_rows(env.db) == [b]


async def test_turning_it_off_purges_and_stops_new_writes(env: Env) -> None:
    await _keep(env, A, True)
    _request(env, A, 0, keep=True)
    await _flush(env)
    assert len(_kept_rows(env.db, A)) == 1
    # A request still running when the switch goes off: it began with keep on.
    late = "gm-late"
    env.hub.publish({"type": "request_start", "req_id": late, "token_id": A, "request_id": "q"})
    env.app.state.prompt_lens_sections.put_prompt(late, A, None, keep=_gen(env.db, A))
    r = await env.http.patch(
        f"/api/tokens/{A}", json={"keep_captures": False}, headers=await env.admin_csrf()
    )
    assert r.status_code == 200 and r.json()["keep_captures"] is False
    assert _kept_rows(env.db, A) == []
    env.hub.publish({"type": "request_end", "req_id": late, "token_id": A, "ts": time.time()})
    await _flush(env)
    assert _kept_rows(env.db, A) == []
    # Back on: new requests are kept again (via the PATCH, as the admin page does).
    r = await env.http.patch(
        f"/api/tokens/{A}", json={"keep_captures": True}, headers=await env.admin_csrf()
    )
    assert r.json()["keep_captures"] is True
    _request(env, A, 1, keep=True)
    await _flush(env)
    assert len(_kept_rows(env.db, A)) == 1


async def test_the_key_holder_cannot_switch_it(env: Env) -> None:
    r = await env.http.patch("/api/me/key", json={"keep_captures": True}, headers=env.key[A])
    assert r.status_code == 422
    with sqlite3.connect(env.db) as c:
        assert (
            c.execute("SELECT keep_captures FROM api_tokens WHERE id = ?", (A,)).fetchone()[0] == 0
        )


async def test_kept_rows_survive_a_restart(env: Env, tmp_data_dir: Path) -> None:
    """A fresh writer and store over the same database (a restart) still serve them."""
    await _keep(env, A, True)
    rid = _request(env, A, 0, keep=True)
    await _flush(env)
    env.app.state.prompt_lens_kept = KeptCaptures(env.db)
    _forget_memory(env)
    r = await env.http.get(f"/api/me/prompt-lens/requests/{rid}/section/reply", headers=env.key[A])
    assert r.status_code == 200 and r.json()["text"] == f"reply secret-of-{A}-0"


async def test_admin_lists_kept_by_key(env: Env) -> None:
    await _keep(env, A, True)
    await _keep(env, B, True)
    a = _request(env, A, 0, keep=True)
    b = _request(env, B, 0, keep=True)
    await _flush(env)
    r = await env.http.get(f"/api/admin/prompt-lens/kept?token_ids={A}", headers=env.admin)
    assert r.status_code == 200, r.text
    assert [(k["req_id"], k["token_id"]) for k in r.json()["kept"]] == [(a, A)]
    r = await env.http.get("/api/admin/prompt-lens/kept", headers=env.admin)
    assert {k["req_id"] for k in r.json()["kept"]} == {a, b}
    r = await env.http.get("/api/admin/prompt-lens/kept", headers=env.key[A])
    assert r.status_code in (401, 403)


async def test_a_failed_write_logs_no_content(
    env: Env, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    writer = KeptCaptures(tmp_path / "missing" / "nope.db")
    rec: dict[str, Any] = {
        "req_id": "r",
        "token_id": A,
        "keep_gen": 1,
        "start": {"prompt": "secret-in-the-prompt"},
        "end": {"ts": 1.0},
        "sections": {},
    }
    writer.offer(rec)
    with caplog.at_level(logging.DEBUG, logger=prompt_lens_kept.__name__):
        await writer.flush()
    assert writer.written == 0
    assert "secret-in-the-prompt" not in caplog.text


async def test_a_full_queue_drops_and_counts(tmp_path: Path) -> None:
    writer = KeptCaptures(tmp_path / "x.db", per_key=1)
    assert writer.offer({"req_id": "a", "token_id": B}) is True
    assert writer.offer({"req_id": "b", "token_id": B}) is False
    assert writer.dropped == 1
    # Another key's record is never dropped for B's burst (review M7).
    assert writer.offer({"req_id": "c", "token_id": A}) is True


async def test_the_switch_reads_back_on_the_admin_and_key_pages(env: Env) -> None:
    r = await env.http.get(f"/api/tokens/{A}", headers=env.admin)
    assert r.json()["keep_captures"] is False
    r = await env.http.patch(
        f"/api/tokens/{A}", json={"keep_captures": True}, headers=await env.admin_csrf()
    )
    assert r.status_code == 200 and r.json()["keep_captures"] is True
    # The holder can see that their prompts are kept (they cannot change it).
    me = await env.http.get("/api/me/key", headers=env.key[A])
    assert me.json()["keep_captures"] is True
    # null leaves it as it is.
    r = await env.http.patch(
        f"/api/tokens/{A}", json={"keep_captures": None}, headers=await env.admin_csrf()
    )
    assert r.json()["keep_captures"] is True


# -- fix round 1 ------------------------------------------------------------


def _rows_of(db: Path, *tids: str) -> list[str]:
    with sqlite3.connect(db) as c:
        marks = ",".join("?" * len(tids))
        return [
            r[0]
            for r in c.execute(
                f"SELECT token_id FROM prompt_lens_kept WHERE token_id IN ({marks})", tids
            ).fetchall()
        ]


async def _rotate(env: Env, tid: str) -> str:
    r = await env.http.post(
        f"/api/tokens/{tid}/rotate", json={"grace_hours": 24}, headers=await env.admin_csrf()
    )
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


async def test_switch_off_on_the_successor_purges_the_whole_lineage(env: Env) -> None:
    """Review I4: A keeps, A is rotated to A2, the admin turns A2 off: A's rows go too."""
    r = await env.http.patch(
        f"/api/tokens/{A}", json={"keep_captures": True}, headers=await env.admin_csrf()
    )
    assert r.status_code == 200
    _request(env, A, 0, keep=True)
    await _flush(env)
    a2 = await _rotate(env, A)
    assert _gen(env.db, a2) == _gen(env.db, A) != 0  # the successor keeps the switch
    r = await env.http.patch(
        f"/api/tokens/{a2}", json={"keep_captures": False}, headers=await env.admin_csrf()
    )
    assert r.status_code == 200
    assert _rows_of(env.db, A, a2) == []
    assert _gen(env.db, A) == _gen(env.db, a2) == 0  # A, still in grace, stops keeping too
    lst = await env.http.get(f"/api/admin/prompt-lens/kept?token_ids={A},{a2}", headers=env.admin)
    assert lst.json()["kept"] == []


async def test_deleting_the_newest_key_deletes_its_predecessors_rows(env: Env) -> None:
    await _keep(env, A, True)
    _request(env, A, 0, keep=True)
    await _flush(env)
    a2 = await _rotate(env, A)
    assert _rows_of(env.db, A) == [A]
    r = await env.http.delete(f"/api/tokens/{a2}", headers=await env.admin_csrf())
    assert r.status_code in (200, 204), r.text
    assert _rows_of(env.db, A, a2) == []


async def test_off_and_on_again_never_brings_a_queued_capture_back(env: Env) -> None:
    """Review M8: a record queued under the old generation is not written."""
    await env.http.patch(
        f"/api/tokens/{A}", json={"keep_captures": True}, headers=await env.admin_csrf()
    )
    old_gen = _gen(env.db, A)
    writer = KeptCaptures(env.db)  # no writer task: the record stays queued
    env.app.state.prompt_lens_sections._on_kept = writer.offer
    _request(env, A, 0, keep=True)
    for on in (False, True):
        await env.http.patch(
            f"/api/tokens/{A}", json={"keep_captures": on}, headers=await env.admin_csrf()
        )
    assert _gen(env.db, A) not in (0, old_gen)
    await writer.flush()
    assert writer.written == 0 and _kept_rows(env.db, A) == []


async def test_turning_it_on_again_keeps_its_generation(env: Env) -> None:
    await env.http.patch(
        f"/api/tokens/{A}", json={"keep_captures": True}, headers=await env.admin_csrf()
    )
    g = _gen(env.db, A)
    await env.http.patch(
        f"/api/tokens/{A}", json={"keep_captures": True}, headers=await env.admin_csrf()
    )
    assert _gen(env.db, A) == g != 0


async def test_kept_rows_are_capped_by_chars_per_key(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G-R12: oldest rows go past the per-key char cap; the newest always stays."""
    await _keep(env, A, True)
    await _keep(env, B, True)
    rids = []
    for n in range(4):
        rids.append(_request(env, A, n, keep=True))
        await _flush(env)
    _request(env, B, 0, keep=True)
    await _flush(env)
    with sqlite3.connect(env.db) as c:
        size = c.execute("SELECT MAX(chars) FROM prompt_lens_kept").fetchone()[0]
    monkeypatch.setattr(prompt_lens_kept, "KEPT_CHARS_PER_KEY", int(size * 2.5))
    rids.append(_request(env, A, 4, keep=True))
    await _flush(env)
    assert _kept_rows(env.db, A) == rids[-2:]
    assert len(_kept_rows(env.db, B)) == 1  # another key's rows are untouched
    monkeypatch.setattr(prompt_lens_kept, "KEPT_CHARS_PER_KEY", 1)
    rids.append(_request(env, A, 5, keep=True))
    await _flush(env)
    assert _kept_rows(env.db, A) == rids[-1:]


async def test_kept_rows_do_not_repeat_the_narrow_prompt(env: Env) -> None:
    await _keep(env, A, True)
    rid = _request(env, A, 0, keep=True)
    await _flush(env)
    with sqlite3.connect(env.db) as c:
        events = c.execute(
            "SELECT events FROM prompt_lens_kept WHERE req_id = ?", (rid,)
        ).fetchone()
    assert "prompt_parts" not in events[0]


async def test_a_text_section_read_parses_nothing(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review I3: a reply refetch reads the reply column from the offset only."""
    await _keep(env, A, True)
    rid = _request(env, A, 0, keep=True)
    await _flush(env)
    _forget_memory(env)

    async def no_parse(_text: str) -> Any:
        raise AssertionError("parsed a kept row for a text section")

    monkeypatch.setattr(prompt_lens_kept, "_loads", no_parse)
    r = await env.http.get(
        f"/api/me/prompt-lens/requests/{rid}/section/reply?from=6", headers=env.key[A]
    )
    assert r.status_code == 200 and r.json()["text"] == f"secret-of-{A}-0"


async def test_kept_reads_are_rate_limited_per_key(env: Env) -> None:
    await _keep(env, A, True)
    rid = _request(env, A, 0, keep=True)
    await _flush(env)
    _forget_memory(env)
    path = f"/api/me/prompt-lens/requests/{rid}/section/reply"
    for _ in range(prompt_lens_kept.READS_PER_MINUTE):
        assert (await env.http.get(path, headers=env.key[A])).status_code == 200
    r = await env.http.get(path, headers=env.key[A])
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    # Unknown ids count the same (no oracle) and B has its own budget.
    assert (await env.http.get(path.replace(rid, "gm-x"), headers=env.key[A])).status_code == 429
    assert (await env.http.get(path, headers=env.key[B])).status_code == 404


async def test_a_kept_row_without_events_is_a_404(env: Env) -> None:
    """Review M9."""
    await _keep(env, A, True)
    with sqlite3.connect(env.db) as c:
        c.execute(
            "INSERT INTO prompt_lens_kept(req_id, token_id, request_id, ended_ts, chars, summary, events)"
            " VALUES ('gm-empty', ?, 'req-empty', 1.0, 0, '{}', '[]')",
            (A,),
        )
    for path in ("/api/me/requests/req-empty/capture", "/api/me/prompt-lens/kept/gm-empty"):
        r = await env.http.get(path, headers=env.key[A])
        assert r.status_code == 404 and r.json() == {"detail": "no capture of yours with this id"}
