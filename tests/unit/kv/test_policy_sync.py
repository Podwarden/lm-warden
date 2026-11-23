"""The api keeps the ``lmw_kv`` daemon in sync: eviction policy, weights, retired scopes.

Spec 2026-11-21 (Plan 2) sections 4.5, 5.3 and 5.4. ``build_policy`` is the ``set_policy``
document; ``retired_tags`` is every scope tag nobody may use again, from
``kv_retired_scopes`` and derived from the key rows; ``sync_daemon`` sends both;
``request_sync`` coalesces bursts into one sync.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.tokens import TokenRepo, sqlite_utc_in, sqlite_utc_now
from app.kv import isolation as kv_iso
from app.kv import policy_sync
from app.kv.policy_sync import PolicySync, build_policy, request_sync, retired_tags, sync_daemon
from lmw_kv.fair import WIRE_KEYS, WIRE_TENANT_KEYS, Policy
from lmw_kv.scopes import isolation_scope, scope_tag, tenant_tag
from tests.conftest import csrf_header
from tests.unit.stats.test_forest_routes import _ready_key

NK = bytes(range(32))
TAG_FOR = kv_iso.tag_for(NK)


def _tt(tenant: str) -> str:
    return tenant_tag(NK, tenant).hex()


def _tag(tenant: str, epoch: int) -> bytes:
    return scope_tag(isolation_scope(NK, tenant, epoch))


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def _key(db, tid: str, *, priority: int = 5, isolation: str = "sharing") -> TokenRepo:
    repo = TokenRepo(db)
    await repo.create(tid, tid, f"vw_{tid}_" + "x" * 30, priority=priority, kv_isolation=isolation)
    return repo


async def _model(db, mid: str, kv: str, weight: int = 1) -> None:
    await ModelRepo(db).insert(
        ModelRow(
            id=mid,
            served_model_name=mid,
            hf_repo="org/m",
            hf_revision="main",
            gpu_indices=[0],
            tensor_parallel_size=1,
            dtype=None,
            max_model_len=None,
            gpu_memory_utilization=0.9,
            trust_remote_code=False,
            extra_args=[],
            extra_env={},
            status="registered",
            pulled_bytes=0,
            pulled_total=None,
            last_error=None,
            kv_store=kv,
            kv_store_weight=weight,
        )
    )


async def _set(db, sql: str, *params: Any) -> None:
    await db.execute(sql, params)
    await db.commit()


# ---- build_policy -------------------------------------------------------------------------


async def test_build_policy_exact_output(db):
    await _key(db, "hi", priority=9)
    await _key(db, "lo", priority=0)
    repo = await _key(db, "cap")
    await repo.update("cap", kv_max_pct_ram=10, kv_max_pct_nvme=20)
    # A rotation family inside its grace window: one tenant, the newest live row wins.
    await _key(db, "rot")
    succ, _plain, _renamed = await repo.rotate("rot", grace_hours=24)
    await repo.update(succ, priority=3)
    await _key(db, "gone")
    await repo.revoke("gone", now=sqlite_utc_now())
    await _key(db, "old")
    await _set(db, "UPDATE api_tokens SET expires_at = ? WHERE id = 'old'", sqlite_utc_now())
    await _model(db, "m1", "ram", 1)
    await _model(db, "m2", "ram+nvme", 3)
    await _model(db, "m3", "off", 5)
    # An admin token is not a tenant: it never runs inference.
    await repo.create("adm", "adm", "vw_adm_" + "x" * 30, scope="admin")

    doc = await build_policy(db, {"kv_evict_policy": "fair"}, NK, sqlite_utc_in(timedelta(0)))

    none = {"max_ram_pct": None, "max_nvme_pct": None}
    assert doc == {
        "mode": "fair",
        "idle_ram_s": 7200,
        "idle_nvme_s": 604800,
        "owners": {"m1": 1, "m2": 3},
        "tenants": {
            _tt("hi"): {"weight": 10, **none},
            _tt("lo"): {"weight": 1, **none},
            _tt("cap"): {"weight": 6, "max_ram_pct": 10, "max_nvme_pct": 20},
            _tt("rot"): {"weight": 4, **none},
        },
    }
    # The daemon reads exactly these keys (fair.py) and accepts the document.
    assert set(doc) == WIRE_KEYS
    assert all(set(t) == WIRE_TENANT_KEYS for t in doc["tenants"].values())
    pol = Policy.from_wire(doc)
    assert pol.mode == "fair" and dict(pol.owner_weights) == {"m1": 1, "m2": 3}
    assert pol.tenant(bytes.fromhex(_tt("cap"))).max_ram_pct == 10


async def test_build_policy_defaults_and_idle_settings(db):
    await _key(db, "a")
    doc = await build_policy(db, {}, NK, sqlite_utc_now())
    assert (doc["mode"], doc["idle_ram_s"], doc["idle_nvme_s"]) == ("lru", 7200, 604800)
    doc = await build_policy(
        db,
        {"kv_evict_policy": "lru", "kv_idle_ram_min": "30", "kv_idle_nvme_h": "2"},
        NK,
        sqlite_utc_now(),
    )
    assert (doc["idle_ram_s"], doc["idle_nvme_s"]) == (1800, 7200)
    # A value the settings PATCH could never have written falls back to the default.
    doc = await build_policy(
        db, {"kv_evict_policy": "bogus", "kv_idle_ram_min": "x"}, NK, sqlite_utc_now()
    )
    assert (doc["mode"], doc["idle_ram_s"]) == ("lru", 7200)
    Policy.from_wire(doc)


async def test_build_policy_without_a_network_key_sends_no_tenants(db):
    await _key(db, "a")
    doc = await build_policy(db, {}, b"", sqlite_utc_now())
    assert doc["tenants"] == {}


# ---- retired_tags ---------------------------------------------------------------------------


async def test_retired_tags_derive_every_past_epoch_and_dead_families(db):
    now = sqlite_utc_now()
    # Reset twice with no network key: nothing was recorded, epochs 0 and 1 are past.
    repo = await _key(db, "x", isolation="isolated")
    await repo.update_kv("x", isolation=None, reset=True, tag_for=None, now=now)
    await repo.update_kv("x", isolation=None, reset=True, tag_for=None, now=now)
    # Isolated, then back to sharing with no network key: epoch 0 is past.
    await _key(db, "s2", isolation="isolated")
    await repo.update_kv("s2", isolation="sharing", reset=False, tag_for=None, now=now)
    # An expired isolated key retired nothing when it expired.
    await _key(db, "e", isolation="isolated")
    await _set(db, "UPDATE api_tokens SET expires_at = ? WHERE id = 'e'", now)
    # A family whose last grace window ran out: the successor was revoked while the
    # predecessor was in grace (nothing retired then), and the grace is now over.
    await _key(db, "r", isolation="isolated")
    succ, _p, _n = await repo.rotate("r", grace_hours=24)
    await repo.revoke(succ, now=now)
    await _set(db, "UPDATE api_tokens SET revoked_at = ? WHERE id = 'r'", now)
    # Not retired: a live isolated key's current scope, a dead sharing key.
    await _key(db, "live", isolation="isolated")
    await _key(db, "sh")
    await repo.revoke("sh", now=now)
    # A tag recorded by an earlier change whose rows are gone.
    await _set(
        db,
        "INSERT INTO kv_retired_scopes(scope_tag, kv_tenant_id, retired_at) VALUES (?, ?, ?)",
        _tag("deleted", 0).hex(),
        "deleted",
        now,
    )

    tags = await retired_tags(db, TAG_FOR, sqlite_utc_in(timedelta(seconds=1)))

    want = {
        _tag("x", 0),
        _tag("x", 1),
        _tag("s2", 0),
        _tag("e", 0),
        _tag("r", 0),
        _tag("deleted", 0),
    }
    assert set(tags) == want and len(tags) == len(want)
    # The derived tags are recorded, so they outlive the rows they came from.
    cur = await db.execute("SELECT scope_tag FROM kv_retired_scopes")
    assert {bytes.fromhex(r[0]) for r in await cur.fetchall()} == want


async def test_retired_tags_without_a_network_key_are_the_table(db):
    now = sqlite_utc_now()
    repo = await _key(db, "x", isolation="isolated")
    await repo.update_kv("x", isolation=None, reset=True, tag_for=None, now=now)
    await _set(
        db,
        "INSERT INTO kv_retired_scopes(scope_tag, kv_tenant_id, retired_at) VALUES (?, ?, ?)",
        _tag("t", 0).hex(),
        "t",
        now,
    )
    assert await retired_tags(db, None, now) == [_tag("t", 0)]


# ---- sync_daemon and request_sync ------------------------------------------------------------


class FakeClient:
    def __init__(
        self,
        *,
        fail: bool = False,
        fail_policy: int = 0,
        on_policy: Any = None,
        hello: dict[str, Any] | None = None,
    ) -> None:
        self.hello_reply: dict[str, Any] | None = (
            {"protocol": 1, "minor": 1} if hello is None else hello
        )
        self.hellos = 0
        self.calls: list[tuple[str, Any]] = []
        self.threads: set[int] = set()
        self.fail = fail
        self.fail_policy = fail_policy  # refuse the first N set_policy calls
        self.on_policy = on_policy  # called with the call number, before answering

    def _note(self, op: str, arg: Any) -> None:
        self.threads.add(threading.get_ident())
        self.calls.append((op, arg))

    def hello(self) -> dict[str, Any] | None:
        self.hellos += 1  # not in ``calls``: the tests there count the sync's own ops
        return self.hello_reply

    def set_policy(self, policy: dict[str, Any]) -> bool | None:
        self._note("set_policy", policy)
        n = sum(1 for op, _ in self.calls if op == "set_policy")
        if self.on_policy is not None:
            self.on_policy(n)
        return None if self.fail or n <= self.fail_policy else True

    def retire_scopes(self, tags: list[bytes]) -> int | None:
        self._note("retire_scopes", list(tags))
        return None if self.fail else 0

    def release_owner(self, owner: str) -> int | None:
        self._note("release_owner", owner)
        if self.fail:
            raise OSError("daemon down")
        return 0


class FakeKvStore:
    def __init__(self, client: FakeClient | None) -> None:
        self._client = client
        self.gen = 1
        self.synced: list[int] = []

    def client(self) -> FakeClient | None:
        return self._client

    def healthy(self) -> bool:
        return self._client is not None

    def generation(self) -> int:
        return self.gen

    def mark_synced(self, generation: int) -> None:
        self.synced.append(generation)


def _state(tmp_data_dir: Path, client: FakeClient | None, window_s: float = 0.05):
    return SimpleNamespace(
        kv_store=FakeKvStore(client),
        settings=SimpleNamespace(db_path=tmp_data_dir / "vllm-warden.db"),
        cobs2_key=SimpleNamespace(key=NK),
        kv_sync=PolicySync(window_s=window_s),
    )


async def test_sync_sends_the_policy_then_every_retired_tag_off_the_loop(db, tmp_data_dir, caplog):
    caplog.set_level(logging.DEBUG, logger="app")
    repo = await _key(db, "x", isolation="isolated", priority=2)
    await repo.update_kv("x", isolation=None, reset=True, tag_for=None, now=sqlite_utc_now())
    await _set(db, "INSERT INTO settings(key, value) VALUES ('kv_evict_policy', 'fair')")
    fake = FakeClient()
    st = _state(tmp_data_dir, fake)
    st.kv_store.gen = 3

    assert await sync_daemon(st) is True

    assert [op for op, _ in fake.calls] == ["set_policy", "retire_scopes"]
    policy = fake.calls[0][1]
    assert policy["mode"] == "fair"
    assert policy["tenants"] == {_tt("x"): {"weight": 3, "max_ram_pct": None, "max_nvme_pct": None}}
    assert fake.calls[1][1] == [_tag("x", 0)]
    assert threading.get_ident() not in fake.threads
    assert st.kv_store.synced == [3]
    # Neither a tenant tag nor a scope tag is ever logged by the api.
    text = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app"))
    assert _tt("x") not in text and _tag("x", 0).hex() not in text


async def test_a_minor_0_daemon_gets_no_policy_and_one_warning(db, tmp_data_dir, caplog):
    """An older daemon answers unknown_op to set_policy: skip it instead of retrying."""
    caplog.set_level(logging.WARNING, logger="app.kv.policy_sync")
    fake = FakeClient(hello={"protocol": 1, "ram_path": "/x", "ram_bytes": 0})  # no minor
    st = _state(tmp_data_dir, fake)
    assert await sync_daemon(st) is True  # done: there is nothing to retry
    assert await sync_daemon(st) is True
    assert fake.calls == [] and fake.hellos == 1
    warned = [r for r in caplog.records if "protocol minor 0" in r.getMessage()]
    assert len(warned) == 1
    assert st.kv_store.synced == [1, 1]  # a load waiting on the generation goes ahead
    st.kv_store.gen = 2  # a new daemon is asked again
    fake.hello_reply = {"protocol": 1, "minor": 1}
    assert await sync_daemon(st) is True
    assert [op for op, _ in fake.calls] == ["set_policy"]  # no retired tag to send
    assert fake.hellos == 2


async def test_a_daemon_that_does_not_answer_hello_is_synced_anyway(db, tmp_data_dir):
    fake = FakeClient()
    fake.hello_reply = None
    st = _state(tmp_data_dir, fake)
    st.kv_store.gen = 7
    assert await sync_daemon(st) is True
    assert [op for op, _ in fake.calls] == ["set_policy"]  # no retired tag to send


async def test_sync_with_no_daemon_or_a_failing_one_does_not_raise(db, tmp_data_dir):
    assert await sync_daemon(_state(tmp_data_dir, None)) is False
    st = SimpleNamespace(settings=SimpleNamespace(db_path=tmp_data_dir / "vllm-warden.db"))
    assert await sync_daemon(st) is False
    failing = _state(tmp_data_dir, FakeClient(fail=True))
    assert await sync_daemon(failing) is False
    # The attempt still counts: a load waiting on this generation is let through.
    assert failing.kv_store.synced == [1]


async def test_a_burst_of_ten_changes_sends_one_set_policy(db, tmp_data_dir):
    fake = FakeClient()
    st = _state(tmp_data_dir, fake)
    for _ in range(10):
        assert request_sync(st) is None  # schedules only, never blocks
    assert fake.calls == []
    await st.kv_sync.drain()
    assert [op for op, _ in fake.calls].count("set_policy") == 1
    request_sync(st)
    await st.kv_sync.drain()
    assert [op for op, _ in fake.calls].count("set_policy") == 2


async def test_a_change_during_a_sync_runs_one_more_sync(db, tmp_data_dir, monkeypatch):
    st = _state(tmp_data_dir, FakeClient())
    started = asyncio.Event()
    release = asyncio.Event()
    runs: list[int] = []

    async def slow(app_state: Any) -> bool:
        runs.append(1)
        started.set()
        await release.wait()
        return True

    monkeypatch.setattr(policy_sync, "sync_daemon", slow)
    request_sync(st)
    await asyncio.wait_for(started.wait(), 5)
    for _ in range(5):
        request_sync(st)
    release.set()
    await st.kv_sync.drain()
    assert len(runs) == 2


def _retrying(tmp_data_dir: Path, client: FakeClient, **kw: float):
    st = _state(tmp_data_dir, client)
    st.kv_sync = PolicySync(window_s=0.01, retry_min_s=0.02, retry_max_s=0.05, **kw)
    return st


def _policies(client: FakeClient) -> int:
    return sum(1 for op, _ in client.calls if op == "set_policy")


async def test_a_failed_sync_is_retried_with_back_off_until_it_succeeds(db, tmp_data_dir):
    fake = FakeClient(fail_policy=2)
    st = _retrying(tmp_data_dir, fake)
    request_sync(st)  # one request, three attempts
    await asyncio.wait_for(st.kv_sync.drain(), 10)
    assert _policies(fake) == 3
    assert st.kv_sync.retry_delays == [0.02, 0.04]
    await asyncio.sleep(0.15)
    assert _policies(fake) == 3  # a success stops the retries


async def test_back_off_is_capped_reset_by_a_new_generation_and_stops_with_the_daemon(
    db, tmp_data_dir
):
    st: Any = None

    def on_policy(n: int) -> None:
        if n == 4:
            st.kv_store.gen = 2  # the daemon restarted during this attempt
        if n == 5:
            st.kv_store._client = None  # and then stopped

    fake = FakeClient(fail=True, on_policy=on_policy)
    st = _retrying(tmp_data_dir, fake)
    request_sync(st)
    await asyncio.wait_for(st.kv_sync.drain(), 10)
    assert _policies(fake) == 5
    assert st.kv_sync.retry_delays == [0.02, 0.04, 0.05, 0.02]


async def test_a_change_during_the_back_off_syncs_at_once(db, tmp_data_dir):
    fake = FakeClient(fail_policy=1)
    st = _state(tmp_data_dir, fake)
    st.kv_sync = PolicySync(window_s=0.01, retry_min_s=60, retry_max_s=60)
    request_sync(st)
    loop = asyncio.get_running_loop()
    end = loop.time() + 5
    while not st.kv_sync.retry_delays:
        assert loop.time() < end
        await asyncio.sleep(0.01)
    request_sync(st)  # e.g. a new generation's on_healthy
    await asyncio.wait_for(st.kv_sync.drain(), 5)
    assert _policies(fake) == 2


async def test_request_sync_outside_a_loop_is_a_no_op(tmp_data_dir):
    st = _state(tmp_data_dir, FakeClient())
    await asyncio.to_thread(request_sync, st)  # no running loop in the worker thread
    await st.kv_sync.drain()
    assert st.kv_store.client().calls == []


async def test_release_owner_is_best_effort(tmp_data_dir):
    fake = FakeClient()
    await policy_sync.release_owner(FakeKvStore(fake), "m1")
    assert fake.calls == [("release_owner", "m1")]
    await policy_sync.release_owner(FakeKvStore(FakeClient(fail=True)), "m1")  # raises inside
    await policy_sync.release_owner(FakeKvStore(None), "m1")  # daemon down
    await policy_sync.release_owner(None, "m1")  # no supervisor


# ---- the routes request a sync ---------------------------------------------------------------


@pytest.fixture
def syncs(monkeypatch) -> list[Any]:
    calls: list[Any] = []
    monkeypatch.setattr(policy_sync, "request_sync", lambda app_state: calls.append(app_state))
    return calls


def _ready(client, tmp_data_dir):
    db_path, plaintext, admin = _ready_key(client, tmp_data_dir)
    return db_path, plaintext, {**admin, **csrf_header(client)}


def test_key_changes_request_a_sync(tmp_data_dir, client, syncs):
    _db, _plain, admin = _ready(client, tmp_data_dir)
    r = client.patch("/api/tokens/tok1", json={"priority": 7}, headers=admin)
    assert r.status_code == 200, r.text
    assert len(syncs) == 1
    r = client.patch("/api/tokens/tok1", json={"kv_max_pct_ram": 5}, headers=admin)
    assert r.status_code == 200, r.text
    assert len(syncs) == 2
    r = client.post("/api/tokens", json={"name": "new"}, headers=admin)
    assert r.status_code == 201, r.text
    assert len(syncs) == 3
    new_id = r.json()["id"]
    r = client.post(f"/api/tokens/{new_id}/rotate", json={}, headers=admin)
    assert r.status_code == 201, r.text
    assert len(syncs) == 4
    client.patch("/api/tokens/tok1", json={"kv_isolation": "isolated"}, headers=admin)
    r = client.post("/api/tokens/tok1/kv-reset", headers=admin)
    assert r.status_code == 200, r.text
    assert len(syncs) == 6
    assert client.delete(f"/api/tokens/{new_id}", headers=admin).status_code == 204
    assert len(syncs) == 7


def _model_row(db_path: Path, mid: str, kv: str, status: str) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error, "
            "kv_store) VALUES (?, ?, 'org/m', 'main', '[0]', 1, 'auto', 4096, 0.9, 0, '[]', ?, "
            "0, NULL, NULL, ?)",
            (mid, mid, status, kv),
        )
        db.commit()


def test_kv_store_weight_is_editable_while_loaded_and_requests_a_sync(tmp_data_dir, client, syncs):
    db_path, _plain, admin = _ready(client, tmp_data_dir)  # model "qwen" is loaded
    r = client.patch("/api/models/qwen/settings", json={"kv_store_weight": 3}, headers=admin)
    assert r.status_code == 200, r.text
    assert len(syncs) == 1
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT kv_store_weight FROM models WHERE id='qwen'").fetchone() == (3,)
    assert client.get("/api/models/qwen", headers=admin).json()["kv_store_weight"] == 3
    for bad in (0, 101, "x", None):
        r = client.patch("/api/models/qwen/settings", json={"kv_store_weight": bad}, headers=admin)
        assert r.status_code in (400, 422), (bad, r.text)
    assert len(syncs) == 1
    # A setting the daemon does not hear about asks for nothing.
    r = client.patch("/api/models/qwen/settings", json={"supports_tools": True}, headers=admin)
    assert r.status_code == 200, r.text
    assert len(syncs) == 1


def test_a_model_registered_with_a_tier_keeps_its_weight_and_requests_a_sync(
    tmp_data_dir, client, syncs
):
    db_path, _plain, admin = _ready(client, tmp_data_dir)
    body = {"served_model_name": "kv", "hf_repo": "o/m", "gpu_indices": [0], "kv_store": "ram"}
    r = client.post("/api/models", json={**body, "kv_store_weight": 4}, headers=admin)
    assert r.status_code == 201, r.text
    assert client.get(f"/api/models/{r.json()['id']}", headers=admin).json()["kv_store_weight"] == 4
    assert len(syncs) == 1
    r = client.post(
        "/api/models", json={**body, "served_model_name": "plain", "kv_store": "off"}, headers=admin
    )
    assert r.status_code == 201, r.text
    assert client.get(f"/api/models/{r.json()['id']}", headers=admin).json()["kv_store_weight"] == 1
    assert len(syncs) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"kv_evict_policy": "fair"},
        {"kv_idle_ram_min": 30},
        {"kv_idle_nvme_h": 48},
        {"kv_isolation_default": "isolated"},
    ],
)
def test_kv_settings_request_a_sync(tmp_data_dir, client, syncs, body):
    _db, _plain, admin = _ready(client, tmp_data_dir)
    r = client.patch("/api/settings/runtime", json=body, headers=admin)
    assert r.status_code == 200, r.text
    assert r.json()["requires_restart_kinds"] == []
    assert len(syncs) == 1


def test_non_kv_settings_request_no_sync(tmp_data_dir, client, syncs):
    _db, _plain, admin = _ready(client, tmp_data_dir)
    r = client.patch("/api/settings/runtime", json={"log_retention_lines": 500}, headers=admin)
    assert r.status_code == 200, r.text
    assert syncs == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("kv_evict_policy", "random"),
        ("kv_idle_ram_min", 0),
        ("kv_idle_ram_min", 10081),
        ("kv_idle_nvme_h", 0),
        ("kv_idle_nvme_h", 8761),
    ],
)
def test_kv_settings_bounds(tmp_data_dir, client, syncs, key, value):
    _db, _plain, admin = _ready(client, tmp_data_dir)
    r = client.patch("/api/settings/runtime", json={key: value}, headers=admin)
    assert r.status_code == 422, r.text
    assert syncs == []


@pytest.mark.parametrize(
    ("key", "value"),
    [("kv_idle_ram_min", 1), ("kv_idle_ram_min", 10080), ("kv_idle_nvme_h", 8760)],
)
def test_kv_settings_bounds_accept_the_edges(tmp_data_dir, client, syncs, key, value):
    _db, _plain, admin = _ready(client, tmp_data_dir)
    r = client.patch("/api/settings/runtime", json={key: value}, headers=admin)
    assert r.status_code == 200, r.text
