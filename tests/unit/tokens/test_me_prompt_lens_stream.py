"""G7: the key holder's live Prompt Lens stream, bound to the key on the server.

``POST /api/me/prompt-lens/ticket`` (the key's forest sign-in) mints a single-use
ticket whose subject is ``key:<token id>``; ``GET /api/me/prompt-lens/stream``
consumes it. Every test here runs two keys, A (``tok1``) and B (``tok2``), with
interleaved traffic, and checks that nothing of one reaches the other: not its
request ids, prompts, deltas, kvsig, progress or end events, and no count or
timing of it either (the key stream carries its own ``seq``).

Driven over raw ASGI (tests/asgi_stream.py) so a real endless stream opens.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sqlite3
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.auth.deps import key_stream_key
from app.db.repos.tokens import hash_token, sqlite_utc_now
from app.proxy import prompt_lens_stream, routes_prompt_lens
from app.proxy.routes_prompt_lens import STREAM_PATH as ADMIN_STREAM_PATH
from app.tokens.routes_me_prompt_lens import ME_STREAM_PATH, TICKET_PATH
from tests.asgi_stream import OpenStream, open_stream
from tests.conftest import bearer, wait_until_async

A, B = "tok1", "tok2"
PLAIN = {A: "vw_keyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", B: "vw_keyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"}
NAME = {A: "alpha-key", B: "zz-bravo-secret-name"}
IP = {A: "10.0.0.5", B: "203.0.113.77"}


def _seed_keys(db_path: Path) -> None:
    with sqlite3.connect(db_path) as db:
        for tid in (A, B):
            db.execute(
                "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
                (tid, NAME[tid], PLAIN[tid][:8], hash_token(PLAIN[tid]), "inference"),
            )
        db.commit()


def _set_token(db_path: Path, tid: str, **cols: str) -> None:
    with sqlite3.connect(db_path) as db:
        sets = ", ".join(f"{k} = ?" for k in cols)
        db.execute(f"UPDATE api_tokens SET {sets} WHERE id = ?", (*cols.values(), tid))


@dataclasses.dataclass
class Env:
    app: Any
    http: httpx.AsyncClient
    db: Path
    key: dict[str, dict[str, str]]  # forest sign-in headers per key
    admin: dict[str, str]  # an admin session

    @property
    def hub(self) -> Any:
        return self.app.state.prompt_lens_hub

    def prompt_lens(self, on: bool) -> None:
        self.app.state.settings = dataclasses.replace(
            self.app.state.settings, prompt_lens_enabled=on
        )

    async def admin_csrf(self) -> dict[str, str]:
        r = await self.http.get("/api/csrf")
        return {**self.admin, "X-CSRF-Token": r.json()["csrf"]}

    async def mint(self, tid: str) -> str:
        r = await self.http.post(TICKET_PATH, headers=self.key[tid])
        assert r.status_code == 200, r.text
        return str(r.json()["ticket"])


@pytest.fixture
async def env(live_app: Any, tmp_data_dir: Path) -> AsyncIterator[Env]:
    db = tmp_data_dir / "vllm-warden.db"
    await asyncio.to_thread(_seed_keys, db)
    transport = httpx.ASGITransport(app=live_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        keys = {}
        for tid in (A, B):
            r = await http.post("/api/forest/login", headers=bearer(PLAIN[tid]))
            assert r.status_code == 200, r.text
            keys[tid] = bearer(r.json()["token"])
        r = await http.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
        assert r.status_code == 200, r.text
        admin = bearer(r.json()["access_token"])
        # Bearer-only from here: no cookie, so no CSRF token is needed.
        http.cookies.clear()
        e = Env(app=live_app, http=http, db=db, key=keys, admin=admin)
        e.prompt_lens(True)
        yield e


def _frames(stream: OpenStream) -> list[dict[str, Any]]:
    out = []
    for block in stream.body.decode().split("\n\n"):
        if block.startswith("data: "):
            out.append(json.loads(block[len("data: ") :]))
    return out


async def _until(stream: OpenStream, pred: Any, what: str) -> None:
    await wait_until_async(lambda: pred(_frames(stream)), timeout_s=5.0, what=what)


def _traffic(tid: str, n: int, *, phase: str) -> list[tuple[dict[str, Any], bool]]:
    """One request's worth of every event kind, for key ``tid``: (event, keep)."""
    rid = f"gm-{tid}-{phase}-{n}"
    secret = f"secret-of-{tid}-{phase}-{n}"
    return [
        (
            {
                "type": "request_start",
                "req_id": rid,
                "ts": time.time(),
                "token_label": NAME[tid],
                "token_id": tid,
                "request_id": f"req-{tid}-{phase}-{n}",
                "model": "qwen",
                "client_ip": IP[tid],
                "stream": True,
                "prompt": f"prompt {secret}",
                "prompt_parts": {"messages": [{"i": 0, "role": "user", "text": secret}]},
                "session_id": f"sess-{tid}",
            },
            True,
        ),
        (
            {
                "type": "kvsig",
                "req_id": rid,
                "token_id": tid,
                "ts": time.time(),
                "available": True,
                "head8": f"h{tid}",
                "reusable_own": 10,
                "reusable_any": 99,
                "parts": [
                    {"p": "sys", "state": "other", "exact8": f"x{tid}"},
                    {"p": "nodes", "state": "new", "exact8": f"y{tid}"},
                ],
            },
            True,
        ),
        (
            {
                "type": "delta",
                "req_id": rid,
                "token_id": tid,
                "ts": time.time(),
                "channel": "content",
                "text": f"reply {secret}",
            },
            True,
        ),
        (
            {
                "type": "progress",
                "req_id": rid,
                "token_id": tid,
                "ts": time.time(),
                "phase": "answering",
                "frames": {"content": n + 1},
                "completion_tokens": n + 1,
            },
            False,
        ),
        (
            {
                "type": "request_end",
                "req_id": rid,
                "token_id": tid,
                "ts": time.time(),
                "finish_reason": "stop",
                "cached_tokens": 777,
                "cache": {"reusable_tokens": 555, "cache_outcome": "hit"},
                "own_reused_tokens": 5,
            },
            True,
        ),
    ]


def _publish_interleaved(hub: Any, *, phase: str, rounds: int = 3) -> None:
    for n in range(rounds):
        a, b = _traffic(A, n, phase=phase), _traffic(B, n, phase=phase)
        for (ea, ka), (eb, kb) in zip(a, b, strict=True):
            hub.publish(eb, keep=kb)
            hub.publish(ea, keep=ka)


_STRIPPED = (
    "client_ip",
    "token_label",
    "cached_tokens",
    "cache",
    "reusable_any",
    "reusable_tokens",
    "cache_outcome",
)


def _assert_only(stream: OpenStream, mine: str) -> None:
    other = B if mine == A else A
    body = stream.body.decode()
    for leak in (
        other,
        f"gm-{other}",
        f"req-{other}",
        f"secret-of-{other}",
        f"sess-{other}",
        NAME[other],
        IP[other],
        NAME[mine],  # the own token_label is stripped too
        IP[mine],
        '"other"',
    ):
        assert leak not in body, leak
    frames = _frames(stream)
    assert frames, "no frames"
    assert {f["token_id"] for f in frames} == {mine}
    assert not any(k in f for f in frames for k in _STRIPPED)
    assert not any(f["type"] in ("ring", "lagged") for f in frames)
    seqs = [f["seq"] for f in frames]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


async def test_ticket_404_when_prompt_lens_off(env: Env) -> None:
    env.prompt_lens(False)
    for tid in (A, B):
        r = await env.http.post(TICKET_PATH, headers=env.key[tid])
        assert r.status_code == 404
        assert r.json()["detail"] == "Prompt Lens is off"


async def test_ticket_refused_for_admin_session_and_anonymous(env: Env) -> None:
    assert (await env.http.post(TICKET_PATH, headers=env.admin)).status_code == 403
    assert (await env.http.post(TICKET_PATH)).status_code == 401


async def test_two_keys_with_interleaved_traffic_see_only_their_own(env: Env) -> None:
    # Before either connects: the ring holds both keys' earlier requests.
    _publish_interleaved(env.hub, phase="old")
    ta, tb = await env.mint(A), await env.mint(B)
    async with (
        open_stream(env.app, f"{ME_STREAM_PATH}?ticket={ta}", {}) as sa,
        open_stream(env.app, f"{ME_STREAM_PATH}?ticket={tb}", {}) as sb,
    ):
        assert sa.status == sb.status == 200
        # Replay at connect: the key's own kept events only (no progress, it
        # is live-only).
        for s, tid in ((sa, A), (sb, B)):
            await _until(
                s,
                lambda fs: sum(1 for f in fs if f["type"] == "request_end") == 3,
                f"{tid}'s replay",
            )
        _publish_interleaved(env.hub, phase="live")
        for s, tid in ((sa, A), (sb, B)):
            await _until(
                s,
                lambda fs: sum(1 for f in fs if f["type"] == "request_end") == 6,
                f"{tid}'s live events",
            )
    for s, tid in ((sa, A), (sb, B)):
        _assert_only(s, tid)
        frames = _frames(s)
        kinds = [f["type"] for f in frames]
        assert kinds.count("request_start") == 6 and kinds.count("progress") == 3
        assert kinds.count("kvsig") == 6 and kinds.count("delta") == 6
        kv = next(f for f in frames if f["type"] == "kvsig")
        # Ruling G-R11: never "other" (nor a stand-in name for it).
        assert [p["state"] for p in kv["parts"]] == ["new", "new"]
        assert kv["reusable_own"] == 10
        end = next(f for f in frames if f["type"] == "request_end")
        assert end["own_reused_tokens"] == 5
        assert f"secret-of-{tid}-live-2" in s.body.decode()
        # Numbered per connection: 12 replayed (live-only progress is not
        # kept), then 15 live -- nothing where the other key's events went.
        assert [f["seq"] for f in frames] == list(range(1, 28))


async def test_token_ids_parameter_is_ignored(env: Env) -> None:
    _publish_interleaved(env.hub, phase="old", rounds=1)
    t = await env.mint(A)
    url = f"{ME_STREAM_PATH}?token_ids={B}&ticket={t}"
    async with open_stream(env.app, url, {}) as s:
        assert s.status == 200
        await _until(s, lambda fs: any(f["type"] == "request_end" for f in fs), "A's replay")
    _assert_only(s, A)


async def test_a_ticket_is_single_use(env: Env) -> None:
    t = await env.mint(A)
    async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", {}) as s:
        assert s.status == 200
    async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", {}) as again:
        assert again.status == 401


async def test_an_expired_ticket_is_refused(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    t = await env.mint(A)
    # Older than any TTL: what a ticket presented after its minute reads as.
    monkeypatch.setattr(env.app.state.sse_tickets, "_ttl", -1)
    async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", {}) as s:
        assert s.status == 401


async def test_cross_key_and_forged_tickets_are_refused(env: Env) -> None:
    store = env.app.state.sse_tickets
    sessions = env.app.state.auth_sessions
    a_sid = next(s for s, (k, sub, _c) in sessions._active.items() if k == "key" and sub == A)
    exp = int(time.time()) + 3600
    forged = [
        # B's stream claimed with A's sign-in.
        store.mint(key_stream_key(B), ME_STREAM_PATH, sid=a_sid, exp=exp),
        # No sign-in at all.
        store.mint(key_stream_key(A), ME_STREAM_PATH, exp=exp),
        # A's sign-in, but no expiry carried: refused, never "forever".
        store.mint(key_stream_key(A), ME_STREAM_PATH, sid=a_sid),
        # An admin's (session or plain) subject on the key path.
        store.mint("admin", ME_STREAM_PATH),
        store.mint("session:admin", ME_STREAM_PATH),
        # The right subject, the wrong path.
        store.mint(key_stream_key(A), ADMIN_STREAM_PATH, sid=a_sid),
        "bogus.sig",
    ]
    for t in forged:
        async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", {}) as s:
            assert s.status == 401, t
    # A key ticket can never open the admin stream, minted for either path.
    for t in (
        await env.mint(A),
        store.mint(key_stream_key(A), ADMIN_STREAM_PATH, sid=a_sid),
    ):
        async with open_stream(env.app, f"{ADMIN_STREAM_PATH}?ticket={t}", {}) as s:
            assert s.status == 401, t


async def test_header_credentials_cannot_open_the_key_stream(env: Env) -> None:
    t = await env.mint(A)
    # A forest sign-in (even with a valid ticket beside it) or an inference
    # key: 401. An admin session: 403 (this is the key holder's own stream).
    for headers, code in (
        (env.key[A], 401),
        (bearer(PLAIN[A]), 401),
        (env.admin, 403),
    ):
        async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", headers) as s:
            assert s.status == code, headers
        t = await env.mint(A)


async def test_a_ticket_beside_a_header_is_spent_all_the_same(env: Env) -> None:
    """Review M3: a refused request still spends its ticket, so a ticket that
    reached a log line is dead at once."""
    t = await env.mint(A)
    async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", env.admin) as s:
        assert s.status == 403
    async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", {}) as s:
        assert s.status == 401


async def test_a_fifth_open_stream_is_429(env: Env) -> None:
    """Exploit (review I2): one key opening streams without end, each a hub
    subscriber on the proxy hot path and a re-check every few seconds."""
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        # Minted before the cap filled: the stream itself still refuses it.
        spare = await env.mint(A)
        for _ in range(4):
            s = await stack.enter_async_context(
                open_stream(env.app, f"{ME_STREAM_PATH}?ticket={await env.mint(A)}", {})
            )
            assert s.status == 200
        async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={spare}", {}) as s:
            assert s.status == 429
        start = next(m for m in s.messages if m["type"] == "http.response.start")
        assert any(k.lower() == b"retry-after" for k, _v in start["headers"])
        # Another key is not counted against this one.
        async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={await env.mint(B)}", {}) as s:
            assert s.status == 200
    # Closed: room again.
    async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={await env.mint(A)}", {}) as s:
        assert s.status == 200


async def test_a_mint_with_four_streams_open_is_429(env: Env) -> None:
    """Ruling G-R15: the browser cannot read the stream GET's 429, so the
    ticket mint refuses too, with Retry-After, and spends no mint."""
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        for _ in range(4):
            s = await stack.enter_async_context(
                open_stream(env.app, f"{ME_STREAM_PATH}?ticket={await env.mint(A)}", {})
            )
            assert s.status == 200
        book = env.app.state.key_ticket_mints
        before = len(book[A])
        r = await env.http.post(TICKET_PATH, headers=env.key[A])
        assert r.status_code == 429, r.text
        assert r.json()["detail"] == "this key already has the most live streams open"
        assert int(r.headers["Retry-After"]) >= 1
        assert len(book[A]) == before
        # Another key is not counted against this one.
        assert (await env.http.post(TICKET_PATH, headers=env.key[B])).status_code == 200
    # Closed: minting works again.
    assert (await env.http.post(TICKET_PATH, headers=env.key[A])).status_code == 200


async def test_ticket_mints_are_capped_per_minute(env: Env) -> None:
    from app.tokens.routes_me_prompt_lens import MINTS_PER_MINUTE

    for _ in range(MINTS_PER_MINUTE):
        assert (await env.http.post(TICKET_PATH, headers=env.key[A])).status_code == 200
    r = await env.http.post(TICKET_PATH, headers=env.key[A])
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert (await env.http.post(TICKET_PATH, headers=env.key[B])).status_code == 200


async def test_the_stream_ends_when_the_sign_in_expires(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review M2: the stream does not outlive the forest token that opened it."""
    from app.tokens import routes_me_prompt_lens

    monkeypatch.setattr(prompt_lens_stream, "RECHECK_INTERVAL_S", 0.05)
    t = await env.mint(A)
    real = time.time
    async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", {}) as s:
        assert s.status == 200
        assert not await s.ended(within_s=0.3)
        # Past the sign-in's exp (12 h ahead): the re-check ends it.
        monkeypatch.setattr(routes_me_prompt_lens, "_now", lambda: real() + 13 * 3600)
        assert await s.ended(within_s=3.0)
    # And a ticket presented after the sign-in expired is refused at connect.
    monkeypatch.setattr(routes_me_prompt_lens, "_now", real)
    t = await env.mint(A)
    monkeypatch.setattr(routes_me_prompt_lens, "_now", lambda: real() + 13 * 3600)
    async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", {}) as s:
        assert s.status == 401


async def test_capture_survival_does_not_count_other_keys(env: Env) -> None:
    """Exploit (review I1 on the capture route): A's capture -- found, and not
    partial -- whatever B publishes after it."""
    _publish_interleaved(env.hub, phase="old", rounds=1)
    url = f"/api/me/requests/req-{A}-old-0/capture"
    before = (await env.http.get(url, headers=env.key[A])).json()
    assert before["partial"] is False
    for n in range(800):  # past the shared ring and the shared start index
        for e, keep in _traffic(B, n, phase="flood"):
            env.hub.publish(e, keep=keep)
    after = await env.http.get(url, headers=env.key[A])
    assert after.status_code == 200, after.text
    assert after.json()["partial"] is False
    assert after.json()["reply"] == before["reply"]


async def test_an_admin_cannot_mint_a_ticket_for_a_key_path(env: Env) -> None:
    for path in (ME_STREAM_PATH, "//api/me/prompt-lens/stream", "/api/me/anything"):
        r = await env.http.post("/api/auth/sse-ticket", json={"path": path}, headers=env.admin)
        assert r.status_code == 403, path


async def test_stream_404_when_prompt_lens_off(env: Env) -> None:
    t = await env.mint(A)
    env.prompt_lens(False)
    async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", {}) as s:
        assert s.status == 404


async def _open_both(env: Env) -> tuple[str, str]:
    return await env.mint(A), await env.mint(B)


async def _act(env: Env, action: str) -> None:
    h = env.key[A]
    if action == "revoke":
        r = await env.http.post("/api/me/key/revoke", json={"confirm_name": NAME[A]}, headers=h)
        assert r.status_code == 204, r.text
    elif action.startswith("rotate"):
        grace = int(action.removeprefix("rotate"))
        r = await env.http.post("/api/me/key/rotate", json={"grace_hours": grace}, headers=h)
        assert r.status_code == 201, r.text
    elif action == "logout":
        r = await env.http.post("/api/forest/logout", headers=h)
        assert r.status_code == 204, r.text
    elif action == "admin-delete":
        r = await env.http.delete(f"/api/tokens/{A}", headers=await env.admin_csrf())
        assert r.status_code == 204, r.text
    elif action == "admin-pause":
        r = await env.http.patch(
            f"/api/tokens/{A}", json={"paused": True}, headers=await env.admin_csrf()
        )
        assert r.status_code == 200, r.text
    else:  # pragma: no cover
        raise AssertionError(action)


@pytest.mark.parametrize(
    "action", ["revoke", "rotate0", "rotate24", "logout", "admin-delete", "admin-pause"]
)
async def test_revoke_rotate_logout_end_the_stream_at_once(
    env: Env, action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No re-check and no keepalive within the test: only the registry's
    # cancel_user("key:tok1") can end the stream this fast.
    monkeypatch.setattr(prompt_lens_stream, "RECHECK_INTERVAL_S", 600.0)
    monkeypatch.setattr(routes_prompt_lens, "KEEPALIVE_INTERVAL_S", 600.0)
    registry = env.app.state.stream_registry
    ta, tb = await _open_both(env)
    async with (
        open_stream(env.app, f"{ME_STREAM_PATH}?ticket={ta}", {}) as sa,
        open_stream(env.app, f"{ME_STREAM_PATH}?ticket={tb}", {}) as sb,
    ):
        assert sa.status == sb.status == 200
        assert registry.count(key_stream_key(A)) == 1
        await _act(env, action)
        assert await sa.ended(within_s=3.0), action
        # The other key's stream is untouched.
        assert not await sb.ended(within_s=0.2)
        assert registry.count(key_stream_key(B)) == 1
        env.hub.publish(_traffic(A, 9, phase="after")[2][0])
    assert "after" not in sa.body.decode()
    assert registry.count(key_stream_key(A)) == 0


async def _pause_sql(env: Env) -> None:
    await asyncio.to_thread(_set_token, env.db, A, paused_at=sqlite_utc_now())


async def _revoke_sql(env: Env) -> None:
    await asyncio.to_thread(_set_token, env.db, A, revoked_at=sqlite_utc_now())


async def _end_session(env: Env) -> None:
    sessions = env.app.state.auth_sessions
    await sessions.end_subject("key", A)


async def _prompt_lens_off(env: Env) -> None:
    env.prompt_lens(False)


@pytest.mark.parametrize(
    "change",
    [_pause_sql, _revoke_sql, _end_session, _prompt_lens_off],
    ids=["paused", "revoked", "sign-in-ended", "platform-off"],
)
async def test_recheck_ends_the_stream_even_under_busy_traffic(
    env: Env, change: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing tells the registry (an SQL write, another process, a settings
    switch): the stream's own timer finds it. A's own events arrive every few
    ms and the keepalive is ten minutes away, so a check tied to the keepalive
    would never run (ruling G-F10)."""
    monkeypatch.setattr(prompt_lens_stream, "RECHECK_INTERVAL_S", 0.05)
    monkeypatch.setattr(routes_prompt_lens, "KEEPALIVE_INTERVAL_S", 600.0)
    ta, tb = await _open_both(env)
    stop = asyncio.Event()

    async def pump() -> None:
        n = 0
        while not stop.is_set():
            n += 1
            for tid in (A, B):
                env.hub.publish(_traffic(tid, n, phase="busy")[2][0])
            await asyncio.sleep(0.005)

    async with (
        open_stream(env.app, f"{ME_STREAM_PATH}?ticket={ta}", {}) as sa,
        open_stream(env.app, f"{ME_STREAM_PATH}?ticket={tb}", {}) as sb,
    ):
        assert sa.status == sb.status == 200
        pumping = asyncio.create_task(pump())
        try:
            assert not await sa.ended(within_s=0.3)  # several re-checks, all fine
            await change(env)
            assert await sa.ended(within_s=3.0)
            if change is not _prompt_lens_off:
                assert not await sb.ended(within_s=0.2)
        finally:
            stop.set()
            await pumping
    _assert_only(sa, A)
    _assert_only(sb, B)


async def test_admin_stream_sends_a_ring_frame_first(env: Env) -> None:
    r = await env.http.post(
        "/api/auth/sse-ticket", json={"path": ADMIN_STREAM_PATH}, headers=env.admin
    )
    assert r.status_code == 200, r.text
    async with open_stream(env.app, f"{ADMIN_STREAM_PATH}?ticket={r.json()['ticket']}", {}) as s:
        assert s.status == 200
        await _until(s, lambda fs: len(fs) >= 1, "the ring frame")
    assert _frames(s)[0]["type"] == "ring"


async def test_foreign_and_unknown_request_ids_get_the_same_404(env: Env) -> None:
    _publish_interleaved(env.hub, phase="old", rounds=1)
    mine = await env.http.get(f"/api/me/requests/req-{A}-old-0/capture", headers=env.key[A])
    assert mine.status_code == 200, mine.text
    foreign = await env.http.get(f"/api/me/requests/req-{B}-old-0/capture", headers=env.key[A])
    unknown = await env.http.get("/api/me/requests/nope/capture", headers=env.key[A])
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json()
    assert f"secret-of-{B}" not in foreign.text


# -- G9: GET /api/me/prompt-lens/requests/{req_id}/section/{kind} ---------------


def _sec(rid: str, kind: str = "reply") -> str:
    return f"/api/me/prompt-lens/requests/{rid}/section/{kind}"


async def test_section_own_only_and_same_404(env: Env) -> None:
    _publish_interleaved(env.hub, phase="sec", rounds=2)
    mine = await env.http.get(_sec(f"gm-{A}-sec-0"), headers=env.key[A])
    assert mine.status_code == 200, mine.text
    assert mine.json()["text"] == f"reply secret-of-{A}-sec-0"
    # The other key's id, an unknown id, a released id and Prompt Lens off: one 404.
    foreign = await env.http.get(_sec(f"gm-{B}-sec-0"), headers=env.key[A])
    unknown = await env.http.get(_sec("gm-nope"), headers=env.key[A])
    env.app.state.prompt_lens_sections._drop(f"gm-{A}-sec-1")  # released from memory
    released = await env.http.get(_sec(f"gm-{A}-sec-1"), headers=env.key[A])
    env.prompt_lens(False)
    off = await env.http.get(_sec(f"gm-{A}-sec-0"), headers=env.key[A])
    replies = (foreign, unknown, released, off)
    assert {r.status_code for r in replies} == {404}
    assert len({r.text for r in replies}) == 1
    assert foreign.json() == {"detail": "no capture of yours with this id"}
    assert f"secret-of-{B}" not in foreign.text
    # And key B sees its own, never A's.
    env.prompt_lens(True)
    b = await env.http.get(_sec(f"gm-{B}-sec-0"), headers=env.key[B])
    assert b.status_code == 200 and f"secret-of-{A}" not in b.text
    a_as_b = await env.http.get(_sec(f"gm-{A}-sec-0"), headers=env.key[B])
    assert a_as_b.status_code == 404


async def test_section_strips_nothing_secret(env: Env) -> None:
    _publish_interleaved(env.hub, phase="strip", rounds=1)
    for kind in ("reply", "thinking", "tool_calls", "tool_results", "prompt"):
        r = await env.http.get(_sec(f"gm-{A}-strip-0", kind), headers=env.key[A])
        assert r.status_code == 200, (kind, r.text)
        body = r.json()
        # No key id, and not the hub's seq: it counts every key's events.
        assert "token_id" not in body and body["seq"] is None
        for leak in (B, NAME[B], IP[B], IP[A], NAME[A]):
            assert leak not in r.text, (kind, leak)


async def test_section_from_offset(env: Env) -> None:
    _publish_interleaved(env.hub, phase="off", rounds=1)
    text = f"reply secret-of-{A}-off-0"
    r = await env.http.get(_sec(f"gm-{A}-off-0") + "?from=6", headers=env.key[A])
    assert r.status_code == 200
    assert (r.json()["offset"], r.json()["text"]) == (6, text[6:])
    r = await env.http.get(_sec(f"gm-{A}-off-0") + "?from=9999", headers=env.key[A])
    assert (r.json()["offset"], r.json()["text"]) == (len(text), "")


async def test_section_refuses_an_admin_session(env: Env) -> None:
    _publish_interleaved(env.hub, phase="adm", rounds=1)
    r = await env.http.get(_sec(f"gm-{A}-adm-0"), headers=env.admin)
    assert r.status_code == 403


async def test_a_dock_that_connects_mid_request_gets_its_start_back(env: Env) -> None:
    """Task 17 concern 1, under live traffic: two keys each stream one long
    request (one token per delta) while docks connect, disconnect and
    reconnect. Every connect opens with the start of each live request its
    scope may see -- prompt, model, stream -- marked restored, then the live
    text; never another key's for a key holder."""

    for tid in (A, B):
        start, keep = _traffic(tid, 0, phase="long")[0]
        env.hub.publish(start, keep=keep)
    stop = asyncio.Event()

    async def pump() -> None:
        n = 0
        while not stop.is_set():
            n += 1
            for tid in (A, B):
                env.hub.publish(
                    {
                        "type": "delta",
                        "req_id": f"gm-{tid}-long-0",
                        "token_id": tid,
                        "ts": time.time(),
                        "channel": "content",
                        "text": f"tok{n} ",
                    }
                )
            await asyncio.sleep(0.0005 if n < 1200 else 0.005)

    pumping = asyncio.create_task(pump())
    try:
        await wait_until_async(
            lambda: not any(e["type"] == "request_start" for e in env.hub.key_events(A))
            and not any(e["type"] == "request_start" for e in env.hub.events()),
            what="the starts to leave A's window and the shared ring",
        )
        for _round in range(2):  # a reconnect restores again
            t = await env.mint(A)
            async with open_stream(env.app, f"{ME_STREAM_PATH}?ticket={t}", {}) as s:
                await _until(s, lambda fs: len(fs) >= 3, "A's replay")
            first = _frames(s)[0]
            assert first["type"] == "request_start" and first["restored"] is True
            assert first["prompt"] == f"prompt secret-of-{A}-long-0" and first["stream"] is True
            assert first["req_id"] == f"gm-{A}-long-0"
            _assert_only(s, A)
        r = await env.http.post(
            "/api/auth/sse-ticket", json={"path": ADMIN_STREAM_PATH}, headers=env.admin
        )
        async with open_stream(
            env.app, f"{ADMIN_STREAM_PATH}?ticket={r.json()['ticket']}", {}
        ) as s:
            await _until(s, lambda fs: len(fs) >= 4, "the admin replay")
        restored = [f for f in _frames(s) if f.get("restored")]
        assert {f["token_id"] for f in restored} == {A, B}
        assert all(f["type"] == "request_start" and f["prompt"] for f in restored)
    finally:
        stop.set()
        await pumping
