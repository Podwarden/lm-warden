"""Changing a key's KV isolation: admin and holder switches, reset, revoke and delete.

Spec 2026-11-21 (Plan 2) sections 3.3, 3.4 and 4.5. A key leaves a scope when it goes
from isolated to sharing or an isolated key is reset: every live row of its family moves
to the next epoch and the old scope's tag is retired in the same transaction. Going
isolated changes no epoch and retires nothing (O3). A revoke or delete retires the scope
only when no live row of the family is left.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import KvChange, TokenRepo, sqlite_utc_in, sqlite_utc_now
from app.kv import isolation as kv_iso
from lmw_kv.connector_logic import fold_cache_salt
from lmw_kv.scopes import gpu_cache_salt, isolation_scope, scope_tag
from lmw_kv.store import Store
from tests.conftest import csrf_header, jwt_login
from tests.unit.lmw_kv.test_store import GROUPS, _cfg, _ns
from tests.unit.lmw_kv.test_store import opened as opened  # noqa: F401  (fixture)
from tests.unit.lmw_kv.test_store_isolation import _keys, _save_chain
from tests.unit.proxy.test_kv_params import _kv_on
from tests.unit.proxy.test_messages import _json_upstream
from tests.unit.stats.test_forest_routes import _forest_login, _ready_key

NK = bytes(range(32))
TAG_FOR = kv_iso.tag_for(NK)


def _tag(tenant: str, epoch: int) -> bytes:
    return scope_tag(isolation_scope(NK, tenant, epoch))


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def _kv(db, token_id: str) -> tuple[str, str, int]:
    cur = await db.execute(
        "SELECT kv_isolation, kv_tenant_id, kv_epoch FROM api_tokens WHERE id = ?", (token_id,)
    )
    row = await cur.fetchone()
    assert row is not None
    return row[0], row[1], row[2]


async def _retired(db) -> list[tuple[str, str]]:
    cur = await db.execute("SELECT scope_tag, kv_tenant_id FROM kv_retired_scopes ORDER BY 1")
    return [(r[0], r[1]) for r in await cur.fetchall()]


async def _key(db, token_id: str = "a", isolation: str = "sharing") -> TokenRepo:
    repo = TokenRepo(db)
    await repo.create(token_id, token_id, f"vw_{token_id}_" + "x" * 30, kv_isolation=isolation)
    return repo


# ---- update_kv -------------------------------------------------------------------------


async def test_isolated_to_sharing_retires_one_tag_and_bumps_the_epoch(db):
    repo = await _key(db, isolation="isolated")
    change = await repo.update_kv(
        "a", isolation="sharing", reset=False, tag_for=TAG_FOR, now=sqlite_utc_now()
    )
    assert change.family == ["a"]
    assert change.retired == [_tag("a", 0)]
    assert await _kv(db, "a") == ("sharing", "a", 1)
    assert await _retired(db) == [(_tag("a", 0).hex(), "a")]


async def test_reset_retires_the_old_epoch_and_bumps_it(db):
    repo = await _key(db, isolation="isolated")
    now = sqlite_utc_now()
    first = await repo.update_kv("a", isolation=None, reset=True, tag_for=TAG_FOR, now=now)
    second = await repo.update_kv("a", isolation=None, reset=True, tag_for=TAG_FOR, now=now)
    assert first.retired == [_tag("a", 0)] and second.retired == [_tag("a", 1)]
    assert await _kv(db, "a") == ("isolated", "a", 2)
    assert {t for t, _ in await _retired(db)} == {_tag("a", 0).hex(), _tag("a", 1).hex()}


async def test_sharing_to_isolated_retires_nothing_and_keeps_the_epoch(db):
    repo = await _key(db)
    change = await repo.update_kv(
        "a", isolation="isolated", reset=False, tag_for=TAG_FOR, now=sqlite_utc_now()
    )
    assert change.family == ["a"] and change.retired == []
    assert await _kv(db, "a") == ("isolated", "a", 0)
    assert await _retired(db) == []


async def test_reset_of_a_sharing_key_is_refused(db):
    repo = await _key(db)
    with pytest.raises(ValueError, match="sharing"):
        await repo.update_kv("a", isolation=None, reset=True, tag_for=TAG_FOR, now=sqlite_utc_now())
    assert await _kv(db, "a") == ("sharing", "a", 0)


async def test_the_same_setting_again_changes_nothing(db):
    repo = await _key(db, isolation="isolated")
    change = await repo.update_kv(
        "a", isolation="isolated", reset=False, tag_for=TAG_FOR, now=sqlite_utc_now()
    )
    assert change == KvChange(family=[], retired=[])
    assert await _kv(db, "a") == ("isolated", "a", 0)


async def test_kv_change_repr_names_no_tag(db):
    repo = await _key(db, isolation="isolated")
    change = await repo.update_kv(
        "a", isolation=None, reset=True, tag_for=TAG_FOR, now=sqlite_utc_now()
    )
    assert _tag("a", 0).hex() not in repr(change) and repr(_tag("a", 0)) not in repr(change)


async def test_isolated_sharing_isolated_round_trip(db, tmp_path: Path, opened):  # noqa: F811
    repo = await _key(db, isolation="isolated")
    now = sqlite_utc_now()
    for iso in ("sharing", "isolated"):
        await repo.update_kv("a", isolation=iso, reset=False, tag_for=TAG_FOR, now=now)
    iso_now, tenant, epoch = await _kv(db, "a")
    assert (iso_now, epoch) == ("isolated", 1)
    current = isolation_scope(NK, tenant, epoch)
    retired = [bytes.fromhex(t) for t, _ in await _retired(db)]
    assert scope_tag(current) not in retired and retired == [_tag("a", 0)]

    # A store fed the retired list still saves and finds the current scope's blocks.
    s = Store(_cfg(tmp_path, nvme=False))
    opened.append(s)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a")
    s.retire_scopes(retired)
    keys = _keys(ns_id, fold_cache_salt(current, gpu_cache_salt(NK, current, "")))
    _save_chain(s, ns_id, keys, current, tag=scope_tag(current))
    assert s.lookup(ns_id, keys) == len(keys)
    # ... and the retired scope's blocks are refused.
    old = isolation_scope(NK, tenant, 0)
    old_keys = _keys(ns_id, fold_cache_salt(old, gpu_cache_salt(NK, old, "")))
    assert s.begin_save(ns_id, old_keys[0], b"\x00" * 32, old, scope_tag=scope_tag(old)) is None


async def test_a_failure_after_the_update_leaves_neither(db):
    repo = await _key(db, isolation="isolated")

    def boom(tenant: str, epoch: int) -> bytes:
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        await repo.update_kv(
            "a", isolation="sharing", reset=False, tag_for=boom, now=sqlite_utc_now()
        )
    assert await _kv(db, "a") == ("isolated", "a", 0)
    assert await _retired(db) == []
    with pytest.raises(RuntimeError, match="injected"):
        await repo.delete("a", tag_for=boom, now=sqlite_utc_now())
    row = await repo.get("a")
    assert row is not None and row.revoked_at is None
    assert await _retired(db) == []


async def test_update_kv_of_a_dead_key_is_refused(db):
    repo = await _key(db, isolation="isolated")
    await repo.revoke("a")
    with pytest.raises(ValueError, match="dead"):
        await repo.update_kv("a", isolation=None, reset=True, tag_for=TAG_FOR, now=sqlite_utc_now())


# ---- revoke and delete ------------------------------------------------------------------


async def test_deleting_an_isolated_key_retires_its_scope(db):
    repo = await _key(db, isolation="isolated")
    ok, retired = await repo.delete_with_retired("a", tag_for=TAG_FOR, now=sqlite_utc_now())
    assert ok
    assert retired == [_tag("a", 0)]
    assert await _retired(db) == [(_tag("a", 0).hex(), "a")]


async def test_deleting_a_sharing_key_retires_nothing(db):
    repo = await _key(db)
    assert await repo.delete("a", tag_for=TAG_FOR, now=sqlite_utc_now()) is True
    assert await _retired(db) == []


async def test_revoke_retires_nothing(db):
    """Only admin tokens are revoked; the re-sync (app/kv/policy_sync.py) derives a dead
    isolated family's tag from its rows (tests/unit/kv/test_policy_sync.py)."""
    repo = await _key(db, isolation="isolated")
    await repo.revoke("a", now=sqlite_utc_now())
    assert await _retired(db) == []


async def test_deleting_an_expired_isolated_key_retires_its_scope(db):
    """Once the rows are gone nothing can derive the tag again: delete records it."""
    repo = await _key(db, isolation="isolated")
    await db.execute(
        "UPDATE api_tokens SET expires_at = ? WHERE id = 'a'",
        (sqlite_utc_in(timedelta(hours=-1)),),
    )
    await db.commit()
    ok, retired = await repo.delete_with_retired("a", tag_for=TAG_FOR, now=sqlite_utc_now())
    assert ok and retired == [_tag("a", 0)]
    assert await _retired(db) == [(_tag("a", 0).hex(), "a")]


async def test_deleting_a_family_s_last_row_retires_every_epoch(db):
    """Resets made without a network key recorded nothing; the delete records them all."""
    repo = await _key(db, isolation="isolated")
    now = sqlite_utc_now()
    await repo.update_kv("a", isolation=None, reset=True, tag_for=None, now=now)
    await repo.update_kv("a", isolation=None, reset=True, tag_for=None, now=now)
    ok, retired = await repo.delete_with_retired("a", tag_for=TAG_FOR, now=now)
    assert ok and sorted(retired) == sorted([_tag("a", 0), _tag("a", 1), _tag("a", 2)])
    assert {t for t, _ in await _retired(db)} == {_tag("a", e).hex() for e in range(3)}


async def test_deleting_a_sharing_key_that_was_isolated_retires_its_past_epoch(db):
    repo = await _key(db, isolation="isolated")
    now = sqlite_utc_now()
    await repo.update_kv("a", isolation="sharing", reset=False, tag_for=None, now=now)
    ok, retired = await repo.delete_with_retired("a", tag_for=TAG_FOR, now=now)
    assert ok and retired == [_tag("a", 0)]  # epoch 1 was never isolated


# ---- a rotated family (Review Focus 1) ---------------------------------------------------


async def _family(db) -> tuple[TokenRepo, str]:
    """An isolated key "a" rotated with a 24 h grace: "a" is in grace, the successor live."""
    repo = await _key(db, isolation="isolated")
    new_id, _plain, _renamed = await repo.rotate("a", grace_hours=24)
    pred = await repo.get("a")
    assert pred is not None and pred.revoked_at is not None and pred.revoked_at > sqlite_utc_now()
    assert await _kv(db, new_id) == ("isolated", "a", 0)
    return repo, new_id


async def test_revoking_then_deleting_the_predecessor_in_grace_retires_nothing(db):
    repo, succ = await _family(db)
    await repo.revoke("a", now=sqlite_utc_now())
    assert await repo.delete("a", tag_for=TAG_FOR, now=sqlite_utc_now()) is True
    assert await _retired(db) == []
    assert await _kv(db, succ) == ("isolated", "a", 0)


async def test_deleting_the_successor_while_the_predecessor_is_in_grace_retires_nothing(db):
    repo, succ = await _family(db)
    assert await repo.delete(succ, tag_for=TAG_FOR, now=sqlite_utc_now()) is True
    assert await _retired(db) == []
    assert await _kv(db, "a") == ("isolated", "a", 0)


async def test_reset_of_the_successor_moves_both_rows(db):
    repo, succ = await _family(db)
    change = await repo.update_kv(
        succ, isolation=None, reset=True, tag_for=TAG_FOR, now=sqlite_utc_now()
    )
    assert sorted(change.family) == sorted(["a", succ]) and change.retired == [_tag("a", 0)]
    assert await _kv(db, "a") == ("isolated", "a", 1)
    assert await _kv(db, succ) == ("isolated", "a", 1)


async def test_switching_the_successor_to_sharing_switches_both_rows(db):
    repo, succ = await _family(db)
    await repo.update_kv(
        succ, isolation="sharing", reset=False, tag_for=TAG_FOR, now=sqlite_utc_now()
    )
    assert await _kv(db, "a") == ("sharing", "a", 1)
    assert await _kv(db, succ) == ("sharing", "a", 1)
    assert await _retired(db) == [(_tag("a", 0).hex(), "a")]


async def test_a_dead_row_of_the_family_is_left_alone(db):
    repo, succ = await _family(db)
    await repo.revoke("a", now=sqlite_utc_now())
    await repo.update_kv(succ, isolation=None, reset=True, tag_for=TAG_FOR, now=sqlite_utc_now())
    assert await _kv(db, "a") == ("isolated", "a", 0)  # dead: its scope was not its to move
    assert await _kv(db, succ) == ("isolated", "a", 1)


async def test_an_expired_row_in_grace_is_not_live(db):
    """A predecessor whose ``expires_at`` passed is refused by require_bearer even inside
    its grace window, so it neither keeps the family's scope alive nor moves with it."""
    repo, succ = await _family(db)
    await db.execute("UPDATE api_tokens SET expires_at = ? WHERE id = 'a'", (sqlite_utc_now(),))
    await db.commit()
    later = sqlite_utc_in(timedelta(seconds=1))
    change = await repo.update_kv(succ, isolation=None, reset=True, tag_for=TAG_FOR, now=later)
    assert change.family == [succ]
    assert await _kv(db, "a") == ("isolated", "a", 0)


# ---- the routes --------------------------------------------------------------------------


def _ready(client, tmp_data_dir) -> tuple[Path, str, dict[str, str]]:
    """``_ready_key`` with the admin session's CSRF header."""
    db_path, plaintext, admin = _ready_key(client, tmp_data_dir)
    return db_path, plaintext, {**admin, **csrf_header(client)}


def _row(db_path: Path, tid: str = "tok1") -> tuple[str, str, int, Any, Any]:
    with sqlite3.connect(db_path) as db:
        return db.execute(
            "SELECT kv_isolation, kv_tenant_id, kv_epoch, kv_max_pct_ram, kv_max_pct_nvme "
            "FROM api_tokens WHERE id = ?",
            (tid,),
        ).fetchone()


def _retired_tags(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as db:
        return [r[0] for r in db.execute("SELECT scope_tag FROM kv_retired_scopes ORDER BY 1")]


def _app_tag(client, tenant: str, epoch: int) -> bytes:
    return kv_iso.tag_for(client.app.state.cobs2_key.key)(tenant, epoch)


@pytest.fixture
def pushed(monkeypatch) -> list[list[bytes]]:
    calls: list[list[bytes]] = []

    async def fake(app_state: Any, tags: list[bytes]) -> None:
        calls.append(list(tags))

    monkeypatch.setattr(kv_iso, "push_retired", fake)
    return calls


def test_admin_switches_both_ways(tmp_data_dir, client, pushed):
    db_path, _plain, admin = _ready(client, tmp_data_dir)
    r = client.patch("/api/tokens/tok1", json={"kv_isolation": "isolated"}, headers=admin)
    assert r.status_code == 200, r.text
    assert r.json()["kv_isolation"] == "isolated"
    assert _row(db_path)[:3] == ("isolated", "tok1", 0)
    r = client.patch("/api/tokens/tok1", json={"kv_isolation": "sharing"}, headers=admin)
    assert r.status_code == 200, r.text
    assert r.json()["kv_isolation"] == "sharing"
    assert _row(db_path)[:3] == ("sharing", "tok1", 1)
    tag = _app_tag(client, "tok1", 0)
    assert _retired_tags(db_path) == [tag.hex()]
    assert pushed == [[tag]]
    body = r.text
    assert tag.hex() not in body and "kv_tenant_id" not in body and "kv_epoch" not in body
    assert (
        client.patch("/api/tokens/tok1", json={"kv_isolation": None}, headers=admin).status_code
        == 422
    )
    assert (
        client.patch("/api/tokens/tok1", json={"kv_isolation": "x"}, headers=admin).status_code
        == 422
    )


def test_admin_reset(tmp_data_dir, client, pushed):
    db_path, _plain, admin = _ready(client, tmp_data_dir)
    r = client.post("/api/tokens/tok1/kv-reset", headers=admin)
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == "this key shares its cache; there is nothing of its own to reset"
    client.patch("/api/tokens/tok1", json={"kv_isolation": "isolated"}, headers=admin)
    r = client.post("/api/tokens/tok1/kv-reset", headers=admin)
    assert r.status_code == 200, r.text
    assert _row(db_path)[:3] == ("isolated", "tok1", 1)
    assert pushed == [[_app_tag(client, "tok1", 0)]]
    assert client.post("/api/tokens/nope/kv-reset", headers=admin).status_code == 404


def test_holder_goes_isolated_and_may_reset_but_not_go_back(tmp_data_dir, client, pushed):
    db_path, plaintext, _admin = _ready(client, tmp_data_dir)
    k = _forest_login(client, plaintext)
    r = client.post("/api/me/key/kv-reset", headers=k)
    assert r.status_code == 409
    assert r.json()["detail"] == "this key shares its cache; there is nothing of its own to reset"
    r = client.patch("/api/me/key", json={"kv_isolation": "isolated"}, headers=k)
    assert r.status_code == 200, r.text
    assert r.json()["kv_isolation"] == "isolated"
    r = client.patch("/api/me/key", json={"kv_isolation": "sharing"}, headers=k)
    assert r.status_code == 403
    assert r.json()["detail"] == "only an admin can switch this key back to sharing"
    assert _row(db_path)[:3] == ("isolated", "tok1", 0)
    r = client.post("/api/me/key/kv-reset", headers=k)
    assert r.status_code == 200, r.text
    assert _row(db_path)[:3] == ("isolated", "tok1", 1)
    assert pushed == [[_app_tag(client, "tok1", 0)]]
    # name stays optional and still works; caps are the admin's.
    assert client.patch("/api/me/key", json={"name": "lap"}, headers=k).json()["name"] == "lap"
    assert client.patch("/api/me/key", json={"kv_max_pct_ram": 5}, headers=k).status_code == 422


def test_holder_revoke_and_admin_delete_retire_an_isolated_key(tmp_data_dir, client, pushed):
    db_path, plaintext, admin = _ready(client, tmp_data_dir)
    client.patch("/api/tokens/tok1", json={"kv_isolation": "isolated"}, headers=admin)
    k = _forest_login(client, plaintext)
    name = client.get("/api/me/key", headers=k).json()["name"]
    r = client.post("/api/me/key/revoke", json={"confirm_name": name}, headers=k)
    assert r.status_code == 204, r.text
    tag = _app_tag(client, "tok1", 0)
    assert _retired_tags(db_path) == [tag.hex()]
    assert pushed == [[tag]]


_OK = {
    "id": "c1",
    "object": "chat.completion",
    "model": "qwen",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def test_each_kv_change_reaches_the_very_next_v1_request(tmp_data_dir, client, pushed):
    """Across the HTTP boundary, in one process: a holder's switch to isolated, a holder's
    reset, an admin's reset and an admin's switch each change the next /v1 request's
    engine ``cache_salt`` and ``lmw_scope`` (the token cache is invalidated, the epoch
    moves). Nothing is cached in between that could carry the old identity."""
    db_path, plaintext, admin = _ready(client, tmp_data_dir)
    client.app.state.supervisor._ports["qwen"] = 19099
    tokenizer = MagicMock()
    tokenizer.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: len(text))
    client.app.state.tokenizers = tokenizer
    _kv_on(client, tmp_data_dir)
    nk = client.app.state.cobs2_key.key
    key = {"Authorization": f"Bearer {plaintext}"}
    body = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}], "cache_salt": "s"}

    def next_request() -> tuple[str, str]:
        send = AsyncMock(return_value=_json_upstream(_OK))
        with patch("httpx.AsyncClient.send", new=send):
            r = client.post("/v1/chat/completions", headers=key, json=body)
        assert r.status_code == 200, r.text
        sent = json.loads(send.call_args.args[0].content)
        return sent["cache_salt"], sent["kv_transfer_params"]["lmw_scope"]

    def isolated_at(epoch: int) -> tuple[str, str]:
        scope = isolation_scope(nk, "tok1", epoch)
        return gpu_cache_salt(nk, scope, "s"), scope.hex()

    seen = [next_request()]
    assert seen[-1] == ("s", "")  # sharing: the client's salt, no scope
    holder = _forest_login(client, plaintext)
    steps = [
        (lambda: client.patch("/api/me/key", json={"kv_isolation": "isolated"}, headers=holder),
         isolated_at(0)),
        (lambda: client.post("/api/me/key/kv-reset", headers=holder), isolated_at(1)),
        (lambda: client.post("/api/tokens/tok1/kv-reset", headers=admin), isolated_at(2)),
        (lambda: client.patch("/api/tokens/tok1", json={"kv_isolation": "sharing"}, headers=admin),
         ("s", "")),
        (lambda: client.patch("/api/tokens/tok1", json={"kv_isolation": "isolated"}, headers=admin),
         isolated_at(3)),
    ]  # fmt: skip
    for change, want in steps:
        r = change()
        assert r.status_code == 200, r.text
        seen.append(next_request())
        assert seen[-1] == want
    assert len(set(seen)) == len(seen) - 1  # every isolated identity new; sharing twice
    assert _row(db_path)[:3] == ("isolated", "tok1", 3)


class _DownClient:
    def retire_scopes(self, tags: Any) -> int | None:
        raise OSError("daemon down")


class _Sup:
    def __init__(self, client: Any) -> None:
        self._client = client

    def client(self) -> Any:
        return self._client


@pytest.mark.parametrize("daemon", [None, _DownClient()])
def test_a_daemon_that_is_down_changes_nothing_for_the_request(tmp_data_dir, client, daemon):
    db_path, _plain, admin = _ready(client, tmp_data_dir)
    client.app.state.kv_store = _Sup(daemon)
    client.patch("/api/tokens/tok1", json={"kv_isolation": "isolated"}, headers=admin)
    r = client.post("/api/tokens/tok1/kv-reset", headers=admin)
    assert r.status_code == 200, r.text
    assert _row(db_path)[:3] == ("isolated", "tok1", 1)
    assert _retired_tags(db_path) == [_app_tag(client, "tok1", 0).hex()]
    r = client.delete("/api/tokens/tok1", headers=admin)
    assert r.status_code == 204, r.text
    assert len(_retired_tags(db_path)) == 2


async def test_push_retired_sends_the_tags_off_the_loop():
    seen: list[list[bytes]] = []

    class C:
        def retire_scopes(self, tags: list[bytes]) -> int:
            seen.append(list(tags))
            return len(tags)

    class State:
        kv_store = _Sup(C())

    await kv_iso.push_retired(State(), [b"\x01" * 16])
    await kv_iso.push_retired(State(), [])
    assert seen == [[b"\x01" * 16]]

    class NoStore:
        pass

    await kv_iso.push_retired(NoStore(), [b"\x01" * 16])  # no lmw_kv daemon: skipped


def test_the_default_for_new_keys(tmp_data_dir, client):
    db_path, _plain, admin = _ready(client, tmp_data_dir)
    r = client.patch("/api/settings/runtime", json={"kv_isolation_default": "bogus"}, headers=admin)
    assert r.status_code in (400, 422), r.text
    r = client.post("/api/tokens", json={"name": "before"}, headers=admin)
    assert _row(db_path, r.json()["id"])[0] == "sharing"
    r = client.patch(
        "/api/settings/runtime", json={"kv_isolation_default": "isolated"}, headers=admin
    )
    assert r.status_code == 200, r.text
    r = client.post("/api/tokens", json={"name": "after"}, headers=admin)
    assert r.status_code == 201, r.text
    new_id = r.json()["id"]
    assert _row(db_path, new_id)[:3] == ("isolated", new_id, 0)
    assert _row(db_path)[0] == "sharing"  # existing keys stay as they are


def test_the_ui_chat_key_ignores_the_default(tmp_data_dir, client):
    """The playground key stays sharing: isolating it would refuse the UI chat on every
    llama.cpp and MLX model (O4)."""
    db_path, _plain, admin = _ready(client, tmp_data_dir)
    r = client.patch(
        "/api/settings/runtime", json={"kv_isolation_default": "isolated"}, headers=admin
    )
    assert r.status_code == 200, r.text
    r = client.post("/api/chat/playground/ensure", headers=admin)
    assert r.status_code == 200, r.text
    with sqlite3.connect(db_path) as db:
        (iso,) = db.execute(
            "SELECT kv_isolation FROM api_tokens WHERE name = 'vw-playground'"
        ).fetchone()
    assert iso == "sharing"


@pytest.mark.parametrize("field", ["kv_max_pct_ram", "kv_max_pct_nvme"])
def test_caps(tmp_data_dir, client, field):
    db_path, _plain, admin = _ready(client, tmp_data_dir)
    col = 3 if field == "kv_max_pct_ram" else 4
    for ok in (1, 100):
        r = client.patch("/api/tokens/tok1", json={field: ok}, headers=admin)
        assert r.status_code == 200, r.text
        assert r.json()[field] == ok and _row(db_path)[col] == ok
    r = client.patch("/api/tokens/tok1", json={field: None}, headers=admin)
    assert r.status_code == 200 and _row(db_path)[col] is None
    for bad in (0, 101):
        assert client.patch("/api/tokens/tok1", json={field: bad}, headers=admin).status_code == 422


def _audit(db_path: Path) -> list[tuple[str, str, str, str]]:
    with sqlite3.connect(db_path) as db:
        return db.execute(
            "SELECT token_id, username, method, path FROM admin_audit ORDER BY id"
        ).fetchall()


def test_a_session_admin_change_writes_one_audit_row(tmp_data_dir, client, flush_audit, pushed):
    db_path, plaintext, admin = _ready(client, tmp_data_dir)
    client.patch("/api/tokens/tok1", json={"kv_isolation": "isolated"}, headers=admin)
    client.post("/api/tokens/tok1/kv-reset", headers=admin)
    client.patch("/api/tokens/tok1", json={"priority": 3}, headers=admin)  # no kv change
    k = _forest_login(client, plaintext)
    client.post("/api/me/key/kv-reset", headers=k)
    flush_audit()
    rows = _audit(db_path)
    assert rows == [
        ("tok1", "admin", "PATCH", "/api/tokens/{token_id} kv:isolated"),
        ("tok1", "admin", "POST", "/api/tokens/{token_id}/kv-reset kv:reset"),
        ("tok1", "key-holder", "POST", "/api/me/key/kv-reset kv:reset"),
    ]
    text = repr(rows)
    assert _app_tag(client, "tok1", 0).hex() not in text


def test_an_admin_token_change_is_audited_once_by_the_middleware(
    tmp_data_dir, client, flush_audit, pushed
):
    from tests.conftest import bearer, seed_admin_token

    db_path, _plain, _admin = _ready(client, tmp_data_dir)
    admin_tid, secret = seed_admin_token(db_path)
    r = client.patch("/api/tokens/tok1", json={"kv_isolation": "isolated"}, headers=bearer(secret))
    assert r.status_code == 200, r.text
    flush_audit()
    rows = _audit(db_path)
    assert [(t, p) for t, _u, _m, p in rows] == [(admin_tid, "/api/tokens/{token_id}")]


def test_holder_routes_are_key_holder_only(tmp_data_dir, client):
    _db, _plain, admin = _ready(client, tmp_data_dir)
    assert client.post("/api/me/key/kv-reset", headers=admin).status_code == 403
    assert jwt_login(client)  # the admin session still works


# ---- fix round 1 -------------------------------------------------------------------------


async def test_a_holder_change_back_to_sharing_is_refused_inside_the_transaction(db):
    repo = await _key(db, isolation="isolated")
    with pytest.raises(ValueError, match="to_sharing"):
        await repo.update_kv(
            "a",
            isolation="sharing",
            reset=False,
            tag_for=TAG_FOR,
            now=sqlite_utc_now(),
            allow_to_sharing=False,
        )
    assert await _kv(db, "a") == ("isolated", "a", 0)
    assert await _retired(db) == []


def test_holder_sharing_races_an_admin_switch_to_isolated(tmp_data_dir, client, monkeypatch):
    """The holder's route reads the key as sharing; an admin isolates it before the holder's
    write. The holder's "sharing" must not undo the admin's switch (O2)."""
    db_path, plaintext, _admin = _ready(client, tmp_data_dir)
    k = _forest_login(client, plaintext)
    real_get = TokenRepo.get
    calls = []

    async def get_then_admin_switches(self: TokenRepo, token_id: str) -> Any:
        row = await real_get(self, token_id)
        calls.append(token_id)
        if len(calls) == 2:  # 1: the sign-in check; 2: the route's own read
            with sqlite3.connect(db_path) as other:  # the admin's write lands now
                other.execute("UPDATE api_tokens SET kv_isolation = 'isolated' WHERE id = 'tok1'")
        return row

    monkeypatch.setattr(TokenRepo, "get", get_then_admin_switches)
    r = client.patch("/api/me/key", json={"kv_isolation": "sharing"}, headers=k)
    monkeypatch.undo()
    assert len(calls) >= 2 and calls[1] == "tok1"
    assert r.status_code == 403, r.text
    assert r.json()["detail"] == "only an admin can switch this key back to sharing"
    assert _row(db_path)[:3] == ("isolated", "tok1", 0)


def test_a_rotated_predecessor_cannot_change_the_family(tmp_data_dir, client, pushed):
    db_path, plaintext, admin = _ready(client, tmp_data_dir)
    r = client.post("/api/tokens/tok1/rotate", json={"grace_hours": 24}, headers=admin)
    assert r.status_code == 201, r.text
    succ = r.json()["id"]
    k = _forest_login(client, plaintext)  # the predecessor, inside its grace window
    r = client.patch("/api/me/key", json={"kv_isolation": "isolated"}, headers=k)
    assert r.status_code == 409, r.text
    assert "successor" in r.json()["detail"]
    client.patch(f"/api/tokens/{succ}", json={"kv_isolation": "isolated"}, headers=admin)
    assert client.post("/api/me/key/kv-reset", headers=k).status_code == 409
    assert _row(db_path)[:3] == ("isolated", "tok1", 0)
    assert _row(db_path, succ)[:3] == ("isolated", "tok1", 0)
    # An admin may still act on the predecessor.
    r = client.post("/api/tokens/tok1/kv-reset", headers=admin)
    assert r.status_code == 200, r.text
    assert _row(db_path, succ)[:3] == ("isolated", "tok1", 1)


def test_admin_patch_refuses_a_null_priority_before_the_kv_change(tmp_data_dir, client, pushed):
    db_path, _plain, admin = _ready(client, tmp_data_dir)
    r = client.patch(
        "/api/tokens/tok1", json={"kv_isolation": "isolated", "priority": None}, headers=admin
    )
    assert r.status_code == 422, r.text
    assert _row(db_path)[:3] == ("sharing", "tok1", 0)


def test_admin_patch_finishes_the_kv_change_when_a_later_write_fails(
    tmp_data_dir, client, flush_audit, pushed, monkeypatch
):
    db_path, _plain, admin = _ready(client, tmp_data_dir)
    client.patch("/api/tokens/tok1", json={"kv_isolation": "isolated"}, headers=admin)

    async def boom(self: TokenRepo, token_id: str, **kw: Any) -> bool:
        raise RuntimeError("injected")

    monkeypatch.setattr(TokenRepo, "update", boom)
    with pytest.raises(RuntimeError, match="injected"):
        client.patch("/api/tokens/tok1", json={"kv_isolation": "sharing"}, headers=admin)
    monkeypatch.undo()
    assert _row(db_path)[:3] == ("sharing", "tok1", 1)
    assert pushed == [[_app_tag(client, "tok1", 0)]]
    flush_audit()
    assert [p for _t, _u, _m, p in _audit(db_path)][-1] == "/api/tokens/{token_id} kv:sharing"


def test_a_retire_skipped_for_want_of_the_network_key_warns_once(tmp_data_dir, client, caplog):
    db_path, _plain, admin = _ready(client, tmp_data_dir)
    client.patch("/api/tokens/tok1", json={"kv_isolation": "isolated"}, headers=admin)
    client.app.state.cobs2_key = None
    with caplog.at_level("WARNING", logger="app.db.repos.tokens"):
        r = client.post("/api/tokens/tok1/kv-reset", headers=admin)
    assert r.status_code == 200, r.text
    assert _row(db_path)[:3] == ("isolated", "tok1", 1)
    assert _retired_tags(db_path) == []
    warns = [x for x in caplog.records if x.name == "app.db.repos.tokens"]
    assert len(warns) == 1 and warns[0].levelname == "WARNING"
    assert "tok1" not in warns[0].getMessage()


def test_admin_sees_the_models_a_key_may_use(tmp_data_dir, client, pushed):
    """The token details page lists an isolated key's models with their marks; it needs the
    key's allowed models (null: every model) to leave out the ones the key may not use."""
    db_path, _plain, admin = _ready(client, tmp_data_dir)
    assert client.get("/api/tokens/tok1", headers=admin).json()["allowed_models"] is None
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE api_tokens SET allowed_models = 'qwen, llama' WHERE id = 'tok1'")
        conn.commit()
    assert client.get("/api/tokens/tok1", headers=admin).json()["allowed_models"] == [
        "qwen",
        "llama",
    ]
    items = client.get("/api/tokens", headers=admin).json()["items"]
    assert next(i for i in items if i["id"] == "tok1")["allowed_models"] == ["qwen", "llama"]
