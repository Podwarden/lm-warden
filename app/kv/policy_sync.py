"""The api keeps the ``lmw_kv`` daemon in sync: eviction policy, weights, retired scopes.

The daemon starts with ``{"mode": "lru"}``, no weights and only the retired tags it kept
on NVMe. Every daemon generation (the first included) gets, from the api:

- ``set_policy``: the runtime settings ``kv_evict_policy``, ``kv_idle_ram_min`` and
  ``kv_idle_nvme_h``, each ``kv_store`` model's ``kv_store_weight`` (the model row id is
  the engine's owner name) and each live key family's weight (``priority + 1``) and caps.
- ``retire_scopes``: every scope tag nobody may use again (spec 4.5). That is the
  ``kv_retired_scopes`` table plus what the key rows imply: every epoch before a
  family's current one, and the current one of an isolated family with no live row left
  (expired, or revoked with the last grace window over). Those cover a change made
  while the network key was unavailable, and keys that died without a revoke.

``request_sync`` is what everything calls: after a key, model or ``kv_`` setting change,
and when a daemon turns healthy. It only schedules one ``sync_daemon`` within
``SYNC_WINDOW_S``, so a burst of changes sends one policy.

Fail open: no daemon, a slow one or one that refuses changes nothing for a request. The
policy holds tenant tags and the retired list holds scope tags: neither is ever logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
from collections.abc import Mapping
from typing import Any

import aiosqlite

from app.kv.isolation import TagFor, tag_for

log = logging.getLogger(__name__)

#: Runtime settings the policy reads, and their defaults (spec 5.4).
EVICT_POLICY_KEY = "kv_evict_policy"
IDLE_RAM_KEY = "kv_idle_ram_min"
IDLE_NVME_KEY = "kv_idle_nvme_h"
POLICY_SETTING_KEYS = (EVICT_POLICY_KEY, IDLE_RAM_KEY, IDLE_NVME_KEY)
EVICT_POLICIES = ("lru", "fair")
DEFAULT_EVICT_POLICY = "lru"
DEFAULT_IDLE_RAM_MIN = 120
DEFAULT_IDLE_NVME_H = 168
IDLE_RAM_MIN_BOUNDS = (1, 10080)
IDLE_NVME_H_BOUNDS = (1, 8760)

#: A burst of changes inside this window sends one sync.
SYNC_WINDOW_S = 1.0
#: A failed sync is tried again while the daemon stays healthy: after 5 s, doubling up
#: to 5 min. A success or a new daemon generation starts the back-off over.
RETRY_MIN_S = 5.0
RETRY_MAX_S = 300.0
#: Scope tags per ``retire_scopes`` request: about 0.4 MB of JSON, well under a line.
RETIRE_CHUNK = 10_000
#: ``set_policy`` refuses a longer owner name (``lmw_kv.fair.OWNER_MAX``) or a weight
#: outside ``WIRE_MIN..WIRE_MAX``; the api never sends one.
_OWNER_MAX = 255
_WIRE_MIN, _WIRE_MAX = 1, 100

#: The live rows of a key family: the rule ``require_bearer`` applies.
_LIVE_SQL = (
    "(revoked_at IS NULL OR revoked_at > :now) AND (expires_at IS NULL OR expires_at > :now)"
)


def _bounded(raw: str | None, bounds: tuple[int, int], default: int) -> int:
    try:
        v = int(raw) if raw is not None else default
    except ValueError:
        return default
    return v if bounds[0] <= v <= bounds[1] else default


def _wire_int(v: Any) -> int:
    return max(_WIRE_MIN, min(_WIRE_MAX, int(v)))


async def build_policy(
    db: aiosqlite.Connection, runtime: Mapping[str, str], network_key: bytes, now: str
) -> dict[str, Any]:
    """The ``set_policy`` document (``lmw_kv/PROTOCOL.md``; the keys ``lmw_kv.fair`` reads).

    ``owners``: every model row with ``kv_store`` on, its ``kv_store_weight``. ``tenants``:
    the tenant tag of every live inference key family, ``{"weight": priority + 1, "max_ram_pct",
    "max_nvme_pct"}`` from the family's newest live row. Idle values in seconds. Without
    a network key (``b""``) no tenant tag can be computed and ``tenants`` is empty: every
    key then gets the daemon's default weight.
    """
    from app.runtime.kv_store import KV_TIERS

    mode = runtime.get(EVICT_POLICY_KEY)
    if mode not in EVICT_POLICIES:
        mode = DEFAULT_EVICT_POLICY
    idle_ram = _bounded(runtime.get(IDLE_RAM_KEY), IDLE_RAM_MIN_BOUNDS, DEFAULT_IDLE_RAM_MIN)
    idle_nvme = _bounded(runtime.get(IDLE_NVME_KEY), IDLE_NVME_H_BOUNDS, DEFAULT_IDLE_NVME_H)

    marks = ", ".join("?" for _ in KV_TIERS)
    cur = await db.execute(
        f"SELECT id, kv_store_weight FROM models WHERE kv_store IN ({marks}) ORDER BY id",
        KV_TIERS,
    )
    owners = {
        mid: _wire_int(w if w is not None else 1)
        for mid, w in await cur.fetchall()
        if len(str(mid).encode("utf-8")) <= _OWNER_MAX
    }

    tenants: dict[str, dict[str, Any]] = {}
    if network_key:
        scopes = importlib.import_module("lmw_kv.scopes")
        # Newest first: the family head (never rotated), then the latest insert.
        cur = await db.execute(
            "SELECT kv_tenant_id, priority, kv_max_pct_ram, kv_max_pct_nvme FROM api_tokens "
            f"WHERE scope = 'inference' AND kv_tenant_id != '' AND {_LIVE_SQL} "
            "ORDER BY rotated_at IS NOT NULL, created_at DESC, rowid DESC",
            {"now": now},
        )
        for tenant, priority, max_ram, max_nvme in await cur.fetchall():
            tag = scopes.tenant_tag(network_key, tenant).hex()
            if tag in tenants:
                continue
            tenants[tag] = {
                "weight": _wire_int(priority + 1),
                "max_ram_pct": None if max_ram is None else _wire_int(max_ram),
                "max_nvme_pct": None if max_nvme is None else _wire_int(max_nvme),
            }
    return {
        "mode": mode,
        "idle_ram_s": idle_ram * 60,
        "idle_nvme_s": idle_nvme * 3600,
        "owners": owners,
        "tenants": tenants,
    }


async def retired_tags(db: aiosqlite.Connection, tag_for: TagFor | None, now: str) -> list[bytes]:
    """Every retired scope tag: ``kv_retired_scopes`` plus the tags the key rows imply.

    Per key family (``kv_tenant_id``): every epoch below the family's highest, and that
    one too when no row of the family is live any more and a row at it is isolated. An
    epoch is only ever left (a reset or a switch to sharing bumps it), so a past epoch's
    scope is never used again; a dead family's current scope neither. The implied tags
    missing from the table are added to it, so they outlive the rows. Without
    ``tag_for`` (no network key) only the table is read.
    """
    if tag_for is not None:
        cur = await db.execute(
            "SELECT kv_tenant_id, kv_epoch, kv_isolation, "
            f"CASE WHEN {_LIVE_SQL} THEN 1 ELSE 0 END FROM api_tokens WHERE kv_tenant_id != ''",
            {"now": now},
        )
        families: dict[str, list[tuple[int, str, int]]] = {}
        for tenant, epoch, isolation, live in await cur.fetchall():
            families.setdefault(tenant, []).append((epoch, isolation, live))
        cur = await db.execute("SELECT scope_tag FROM kv_retired_scopes")
        known = {r[0] for r in await cur.fetchall()}
        implied: list[tuple[str, str]] = []
        for tenant, rows in families.items():
            top = max(e for e, _, _ in rows)
            epochs = list(range(top))
            if not any(live for _, _, live in rows) and any(
                e == top and iso == "isolated" for e, iso, _ in rows
            ):
                epochs.append(top)
            for e in epochs:
                hex_tag = tag_for(tenant, e).hex()
                if hex_tag not in known:
                    known.add(hex_tag)
                    implied.append((hex_tag, tenant))
        if implied:
            await db.executemany(
                "INSERT OR IGNORE INTO kv_retired_scopes(scope_tag, kv_tenant_id, retired_at) "
                "VALUES (?, ?, ?)",
                [(t, tenant, now) for t, tenant in implied],
            )
            await db.commit()
    cur = await db.execute("SELECT scope_tag FROM kv_retired_scopes ORDER BY retired_at, rowid")
    out: list[bytes] = []
    for (hex_tag,) in await cur.fetchall():
        try:
            out.append(bytes.fromhex(hex_tag))
        except (TypeError, ValueError):
            continue  # a hand-edited row; never the value in a log line
    return out


#: Where the supervisor remembers ``(generation, minor)`` of its daemon.
_MINOR_ATTR = "_kv_sync_minor"


def _daemon_minor(sup: Any, client: Any, generation: int) -> int | None:
    """The protocol minor of this daemon generation, asked once (``hello`` has no
    ``minor`` before minor 1, which reads as 0). None when the daemon did not answer: the
    caller then tries the sync anyway, and asks again next time."""
    seen = getattr(sup, _MINOR_ATTR, None)
    if isinstance(seen, tuple) and seen[0] == generation:
        return int(seen[1])
    hello = getattr(client, "hello", None)
    h = hello() if callable(hello) else None
    if not isinstance(h, dict):
        return None
    raw = h.get("minor", 0)  # a minor 0 daemon's hello has no "minor" field
    minor: int = raw if isinstance(raw, int) and not isinstance(raw, bool) else 0
    with contextlib.suppress(AttributeError, TypeError):
        setattr(sup, _MINOR_ATTR, (generation, minor))
    if minor < 1:
        log.warning(
            "kv: the store daemon speaks protocol minor %d, older than this api; the "
            "eviction policy and retired scopes are not sent to it (it keeps lru) until a "
            "daemon of the same version starts",
            minor,
        )
    return minor


def _network_key(app_state: Any) -> bytes | None:
    key_obj = getattr(app_state, "cobs2_key", None)
    nk = getattr(key_obj, "key", None) if key_obj is not None else None
    return nk if isinstance(nk, bytes) else None


async def sync_daemon(app_state: Any) -> bool:
    """Send ``set_policy``, then ``retire_scopes`` with every retired tag; True when the
    daemon took both. The client calls run off the event loop.

    Returns False at once when no daemon is healthy. Whatever the outcome, the attempt is
    reported to the supervisor (``mark_synced``) so a load waiting on this generation
    goes ahead: a later generation, or the next change, syncs again.
    """
    from app.db.database import open_db
    from app.db.repos.settings import SettingsRepo
    from app.db.repos.tokens import sqlite_utc_now

    sup = getattr(app_state, "kv_store", None)
    client_of = getattr(sup, "client", None)
    if not callable(client_of):
        return False
    client = client_of()
    if client is None:
        return False
    gen_of = getattr(sup, "generation", None)
    generation = gen_of() if callable(gen_of) else 0
    ok = False
    try:
        minor = await asyncio.to_thread(_daemon_minor, sup, client, generation)
        if minor is not None and minor < 1:
            return True  # nothing this daemon can take; warned once for its generation
        nk = _network_key(app_state)
        tf: TagFor | None = None
        if nk is not None:
            try:
                tf = tag_for(nk)
            except ImportError:
                nk = None
        now = sqlite_utc_now()
        async with open_db(app_state.settings.db_path) as db:
            runtime = await SettingsRepo(db).get_many(list(POLICY_SETTING_KEYS))
            policy = await build_policy(db, runtime, nk or b"", now)
            tags = await retired_tags(db, tf, now)
        applied = await asyncio.to_thread(client.set_policy, policy)
        retired_ok = True
        for i in range(0, len(tags), RETIRE_CHUNK):
            added = await asyncio.to_thread(client.retire_scopes, tags[i : i + RETIRE_CHUNK])
            retired_ok = retired_ok and added is not None
        ok = applied is True and retired_ok
        if not ok:
            log.warning(
                "kv: the store did not take the eviction policy or the retired scopes; "
                "it is tried again"
            )
    except Exception as exc:  # noqa: BLE001 -- fail open; the type only, never a value
        log.warning("kv: syncing the store failed (%s); it is tried again", type(exc).__name__)
    finally:
        mark = getattr(sup, "mark_synced", None)
        if callable(mark):
            with contextlib.suppress(Exception):
                mark(generation)
    return ok


def _daemon_healthy(app_state: Any) -> bool:
    sup = getattr(app_state, "kv_store", None)
    healthy = getattr(sup, "healthy", None)
    if callable(healthy):
        return bool(healthy())
    client_of = getattr(sup, "client", None)
    return callable(client_of) and client_of() is not None


def _daemon_generation(app_state: Any) -> int:
    gen_of = getattr(getattr(app_state, "kv_store", None), "generation", None)
    return gen_of() if callable(gen_of) else 0


class PolicySync:
    """``request_sync``'s state on ``app.state.kv_sync``: at most one sync pending, one
    more after a sync that a request arrived during, and retries with back-off after a
    failed sync while the daemon stays healthy."""

    def __init__(
        self,
        window_s: float = SYNC_WINDOW_S,
        retry_min_s: float = RETRY_MIN_S,
        retry_max_s: float = RETRY_MAX_S,
    ) -> None:
        self.window_s = window_s
        self.retry_min_s = retry_min_s
        self.retry_max_s = retry_max_s
        #: Every back-off scheduled, in seconds (for tests and diagnosis).
        self.retry_delays: list[float] = []
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._again = False
        self._backing_off = False
        self._wake = asyncio.Event()

    def request(self, app_state: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # only the event loop schedules a sync
        if self._task is not None and not self._task.done():
            if self._running:
                self._again = True  # the running sync read the state before this change
            elif self._backing_off:
                self._wake.set()  # a change (or a new daemon) does not wait out a back-off
            return
        self._task = loop.create_task(self._run(app_state), name="kv-policy-sync")

    async def _run(self, app_state: Any) -> None:
        delay = self.window_s
        retry: float | None = None
        retry_gen: int | None = None
        while True:
            if retry is None:
                await asyncio.sleep(delay)
            else:
                self._wake.clear()
                self._backing_off = True
                try:
                    with contextlib.suppress(TimeoutError):
                        async with asyncio.timeout(delay):
                            await self._wake.wait()
                finally:
                    self._backing_off = False
                if self._wake.is_set():
                    await asyncio.sleep(self.window_s)  # coalesce with what woke us
            self._again = False
            self._running = True
            ok = False
            try:
                ok = await sync_daemon(app_state)
            except Exception as exc:  # noqa: BLE001 -- never ends a caller
                log.warning("kv: the store sync failed (%s)", type(exc).__name__)
            finally:
                self._running = False
            if self._again:
                delay = self.window_s
                continue
            if ok or not _daemon_healthy(app_state):
                return  # done; a stopped daemon syncs again when it turns healthy
            gen = _daemon_generation(app_state)
            if retry is None or gen != retry_gen:
                retry = self.retry_min_s
            else:
                retry = min(retry * 2, self.retry_max_s)
            retry_gen = gen
            self.retry_delays.append(retry)
            delay = retry

    async def drain(self) -> None:
        """Wait until no sync is pending (tests, shutdown)."""
        while self._task is not None and not self._task.done():
            await asyncio.gather(self._task, return_exceptions=True)

    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def request_sync(app_state: Any) -> None:
    """Schedule one ``sync_daemon`` within ``SYNC_WINDOW_S``, coalescing bursts. Never
    blocks; a no-op outside the event loop."""
    state = getattr(app_state, "kv_sync", None)
    if state is None:
        state = PolicySync()
        try:
            app_state.kv_sync = state
        except AttributeError:
            return
    state.request(app_state)


async def release_owner(kv_store: Any, owner: str) -> None:
    """Tell the daemon ``owner`` (a model row id) holds none of its namespaces any more.
    Best effort, off the event loop: with the daemon down nothing happens, and a fresh
    daemon (every api start) has no owners anyway."""
    client_of = getattr(kv_store, "client", None)
    if not callable(client_of):
        return
    try:
        client = client_of()
        if client is None:
            return
        await asyncio.to_thread(client.release_owner, owner)
    except Exception:  # noqa: BLE001 -- fail open
        log.info("kv: could not release a model from the store; it is released on restart")


#: Model columns the policy carries: a change to one asks for a sync.
KV_MODEL_FIELDS = frozenset({"kv_store", "kv_store_weight"})


async def after_model_change(
    app_state: Any, model_id: str, changes: Mapping[str, Any], *, was_on: bool
) -> None:
    """After a model row change committed: release the row's owner when ``kv_store``
    went ``off`` (a missed release can only keep NVMe off, never turn it on), and ask
    for a sync when a column the policy carries changed."""
    if was_on and changes.get("kv_store") == "off":
        await release_owner(getattr(app_state, "kv_store", None), model_id)
    if KV_MODEL_FIELDS.intersection(changes):
        request_sync(app_state)
