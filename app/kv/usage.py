"""Per-key and per-model KV cache reports (spec 2026-11-21, Plan 2, section 6).

Two sources:

- ``request_history``: tokens each turn was served from the GPU, RAM and NVMe, so a key's
  windows (1 h, 24 h, 7 days) are exact sums over its rows. A row whose three tier counts
  are not all known (no store on that turn) is left out.
- The ``lmw_kv`` daemon's ``stats`` reply: bytes held per tier, fair shares and evictions,
  per tenant and per (model, tenant). These are counted since the store started
  (``started_at``), which every store figure carries as ``since``.

The store knows tenants only by their 16-byte tenant tag; the api names them by computing
the tag of every key family (``tenant_names``). A tag no family has (a deleted key, or an
eviction with no taker) is ``unknown``.

Privacy: the key holder's report has the key's own totals only. Who evicted its blocks
(``evicted_by``), fair-share bytes (they reveal the tier size) and the isolation cost (it
reveals what other keys cached) are admin only. Tenant tags and the stats reply are never
logged.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import Iterable, Mapping
from typing import Any

import aiosqlite

from app.db.repos.tokens import TokenRepo, TokenRow

#: The key report's windows, in seconds.
WINDOWS: dict[str, int] = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}

#: The tenant of internal requests (``app.proxy.kv_scope.INTERNAL_TENANT_ID``).
INTERNAL = "internal"
#: A tenant tag no key family has.
UNKNOWN = ("unknown", "unknown")

#: How long a stats reply is reused, so a page that polls costs the daemon one call.
STATS_TTL_S = 5.0

_TIERS = ("ram", "nvme")
# A row counts in a window when its GPU tokens are known: the store's GPU share, or with
# no store on that turn the engine's cached count (R12). RAM and NVMe come from the store
# only, so they are summed over the rows where the store reported them.
_GPU = "COALESCE(served_gpu_tokens, cached_tokens)"
_TIERS_KNOWN = "served_ram_tokens IS NOT NULL AND served_nvme_tokens IS NOT NULL"
_SERVED = (
    f"MAX(MIN({_GPU} + COALESCE(served_ram_tokens, 0) + COALESCE(served_nvme_tokens, 0), "
    "prompt_tokens), 0)"
)


def _scopes() -> Any:
    return importlib.import_module("lmw_kv.scopes")


def _int(v: Any) -> int:
    """A non-negative count from the daemon's reply; anything else is 0."""
    if isinstance(v, bool) or not isinstance(v, int | float) or not 0 <= v < 1e18:
        return 0  # NaN fails the range check too
    return int(round(v))


def _map(v: Any) -> Mapping[str, Any]:
    return v if isinstance(v, Mapping) else {}


def _since(stats: Mapping[str, Any]) -> str | None:
    """``started_at`` as a SQLite UTC string, the format of every other api timestamp."""
    v = stats.get("started_at")
    if isinstance(v, bool) or not isinstance(v, int | float) or not 0 < v < 1e11:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(v))


def tenant_names(rows: Iterable[TokenRow], network_key: bytes) -> dict[bytes, tuple[str, str]]:
    """Tenant tag to (the newest token id of the family, its name).

    The internal tenant maps to ``("internal", "internal")``. A tag missing from the
    result is reported as ``unknown`` by the callers.
    """
    scopes = _scopes()
    out: dict[bytes, tuple[str, str]] = {}
    rows = list(rows)
    # A rotation's successor can share its predecessor's created_at second: the newest
    # row is one nothing was rotated from, then the latest created.
    rotated = {r.rotated_from for r in rows if r.rotated_from}
    newest: dict[str, TokenRow] = {}
    for r in rows:
        fam = r.kv_tenant_id
        if not fam:
            continue
        cur = newest.get(fam)
        if cur is None or _newer(r, cur, rotated):
            newest[fam] = r
    for fam, r in newest.items():
        try:
            out[scopes.tenant_tag(network_key, fam)] = (r.id, r.name)
        except ValueError:
            continue  # a hand-edited tenant id; never the value in a log line
    out[scopes.tenant_tag(network_key, INTERNAL)] = (INTERNAL, INTERNAL)
    return out


def _newer(a: TokenRow, b: TokenRow, rotated: set[str]) -> bool:
    return (a.id not in rotated, a.created_at or "", a.id) > (
        b.id not in rotated,
        b.created_at or "",
        b.id,
    )


def _name(names: Mapping[bytes, tuple[str, str]], hex_tag: Any) -> tuple[str, str]:
    if not isinstance(hex_tag, str):
        return UNKNOWN
    try:
        tag = bytes.fromhex(hex_tag)
    except ValueError:
        return UNKNOWN
    return names.get(tag, UNKNOWN)


def _evicted_by(raw: Any, names: Mapping[bytes, tuple[str, str]]) -> list[dict[str, Any]]:
    """``{tag: count}`` to ``[{"token_id", "name", "count"}]``, most first, then by id; the
    tags that name no key are summed into one ``unknown`` entry."""
    counts: dict[tuple[str, str], int] = {}
    for hex_tag, n in _map(raw).items():
        who = _name(names, hex_tag)
        counts[who] = counts.get(who, 0) + _int(n)
    return [
        {"token_id": tid, "name": name, "count": n}
        for (tid, name), n in sorted(counts.items(), key=lambda e: (-e[1], e[0]))
        if n > 0
    ]


class _StatsCache:
    """The last stats reply of one supervisor, and the lock that keeps one call in flight."""

    def __init__(self, sup: Any) -> None:
        self.sup = sup
        self.at = float("-inf")
        self.value: dict[str, Any] | None = None
        self.lock = asyncio.Lock()


async def _ask(sup: Any) -> dict[str, Any] | None:
    client_of = getattr(sup, "stats_client", None)
    if not callable(client_of):
        return None  # never the hot-path client: a slow reply would cost live turns
    try:
        client = client_of()
        if client is None:
            return None
        st = await asyncio.to_thread(client.stats)
    except Exception:  # noqa: BLE001 -- fail open; the reply is never logged
        return None
    return st if isinstance(st, dict) else None


async def fetch_stats(app_state: Any) -> dict[str, Any] | None:
    """The daemon's ``stats`` reply, or None when there is no daemon or it does not answer
    (fail open: the reports then have ``store: null``).

    It is asked on the supervisor's ``stats_client`` (its own connection, 2 s), off the
    event loop, at most once per ``STATS_TTL_S`` with one call in flight; a None is
    reused as well. Never logged.
    """
    sup = getattr(app_state, "kv_store", None)
    cache = getattr(app_state, "kv_stats_cache", None)
    if not isinstance(cache, _StatsCache) or cache.sup is not sup:
        cache = _StatsCache(sup)
        try:
            app_state.kv_stats_cache = cache
        except AttributeError:
            pass  # a state that takes no attributes: no cache
    async with cache.lock:
        if time.monotonic() - cache.at < STATS_TTL_S:
            return cache.value
        cache.value = await _ask(sup)
        cache.at = time.monotonic()
        return cache.value


async def _windows(db: aiosqlite.Connection, token_id: str, now: float) -> dict[str, Any]:
    """The three windows in one query over the 7-day range (the token index)."""
    cols: list[str] = []
    args: list[Any] = []
    for span in WINDOWS.values():
        w = f"finished_at >= ? AND {_GPU} IS NOT NULL"
        cols += [
            f"COALESCE(SUM(CASE WHEN {w} THEN {_GPU} END), 0)",
            f"SUM(CASE WHEN {w} AND {_TIERS_KNOWN} THEN served_ram_tokens END)",
            f"SUM(CASE WHEN {w} AND {_TIERS_KNOWN} THEN served_nvme_tokens END)",
            f"COALESCE(SUM(CASE WHEN {w} THEN prompt_tokens END), 0)",
            f"COALESCE(SUM(CASE WHEN {w} THEN {_SERVED} END), 0)",
        ]
        args += [now - span] * 5
    cur = await db.execute(
        f"SELECT {', '.join(cols)} FROM request_history WHERE token_id = ? AND finished_at >= ?",
        (*args, token_id, now - max(WINDOWS.values())),
    )
    row = await cur.fetchone()
    vals = list(row) if row else [0, None, None, 0, 0] * len(WINDOWS)
    out: dict[str, Any] = {}
    for i, label in enumerate(WINDOWS):
        gpu, ram, nvme, prompt, served = vals[5 * i : 5 * i + 5]
        out[label] = {
            "gpu": int(gpu),
            "ram": None if ram is None else int(ram),
            "nvme": None if nvme is None else int(nvme),
            "recomputed": int(prompt) - int(served),
            "hit_rate": int(served) / int(prompt) if prompt else None,
        }
    return out


async def _isolation_cost_24h(db: aiosqlite.Connection, token_id: str, now: float) -> int:
    cur = await db.execute(
        "SELECT COALESCE(SUM(kv_iso_cost_tokens), 0) FROM request_history "
        "WHERE token_id = ? AND finished_at >= ?",
        (token_id, now - WINDOWS["24h"]),
    )
    row = await cur.fetchone()
    return int(row[0]) if row else 0


def _store_block(
    row: TokenRow,
    stats: Mapping[str, Any],
    network_key: bytes,
    names: Mapping[bytes, tuple[str, str]] | None,
) -> dict[str, Any]:
    tag = _scopes().tenant_tag(network_key, row.kv_tenant_id).hex()
    t = _map(_map(stats.get("by_tenant")).get(tag))
    cap = {"ram": _int(stats.get("ram_bytes")), "nvme": _int(stats.get("nvme_cap_bytes"))}
    out: dict[str, Any] = {}
    for tier in _TIERS:
        out[f"{tier}_bytes"] = _int(t.get(f"{tier}_bytes"))
    for tier in _TIERS:
        share = _int(t.get(f"{tier}_share"))
        out[f"{tier}_share"] = share / cap[tier] if cap[tier] > 0 else 0.0
    if names is not None:
        for tier in _TIERS:
            out[f"{tier}_share_bytes"] = _int(t.get(f"{tier}_share"))
    for tier in _TIERS:
        out[f"{tier}_evictions"] = _int(t.get(f"{tier}_evictions"))
    out["since"] = _since(stats)
    if names is not None:
        out["evicted_by"] = _evicted_by(t.get("evicted_by"), names)
    return out


async def key_cache_report(
    db: aiosqlite.Connection,
    token_id: str,
    stats: dict[str, Any] | None,
    network_key: bytes | None,
    *,
    admin: bool,
    now: float | None = None,
) -> dict[str, Any]:
    """One key's KV report. ``store`` is None when the daemon is down or there is no
    network key (no tenant tag can be computed). ``admin`` adds ``store.evicted_by``,
    the fair-share bytes and ``isolation_cost_tokens``; the holder gets neither.

    The caller checks the key exists.
    """
    repo = TokenRepo(db)
    row = await repo.get(token_id)
    if row is None:
        raise LookupError(token_id)
    now = time.time() if now is None else now
    out: dict[str, Any] = {
        "isolation": row.kv_isolation,
        "windows": await _windows(db, token_id, now),
        "store": None,
    }
    if isinstance(stats, dict) and network_key is not None and row.kv_tenant_id:
        try:
            names = tenant_names(await repo.list_all(), network_key) if admin else None
            out["store"] = _store_block(row, stats, network_key, names)
        except (ValueError, ImportError):
            out["store"] = None
    if admin:
        out["isolation_cost_tokens"] = {"24h": await _isolation_cost_24h(db, token_id, now)}
    return out


def kv_usage(stats: Any, names: Mapping[bytes, tuple[str, str]]) -> dict[str, Any]:
    """``GET /api/kv/usage``: the tiers, then each model's part and use, then each key's
    use, share and evictions inside the model (admin only). ``available`` is False with
    no daemon, or one that does not answer."""
    if not isinstance(stats, Mapping):
        return {"available": False, "policy": None, "since": None, "tiers": None, "models": []}
    dead = _map(stats.get("dead_bytes"))
    policy = stats.get("policy")
    models: list[dict[str, Any]] = []
    for raw in _map(stats.get("by_model")).values():
        m = _map(raw)
        owners = sorted(o for o in (m.get("owners") or []) if isinstance(o, str))
        last = m.get("last_owner")
        label = owners[0] if owners else (last if isinstance(last, str) and last else "unknown")
        used = {tier: _int(m.get(f"{tier}_bytes")) for tier in _TIERS}
        tenants: list[dict[str, Any]] = []
        for hex_tag, traw in _map(m.get("tenants")).items():
            t = _map(traw)
            tid, name = _name(names, hex_tag)
            t_used = {tier: _int(t.get(f"{tier}_bytes")) for tier in _TIERS}
            tenants.append(
                {
                    "token_id": tid,
                    "name": name,
                    "used": {**t_used, "total": t_used["ram"] + t_used["nvme"]},
                    "share": {tier: _int(t.get(f"{tier}_share")) for tier in _TIERS},
                    "evictions": {tier: _int(t.get(f"{tier}_evictions")) for tier in _TIERS},
                    "evicted_by": _evicted_by(t.get("evicted_by"), names),
                }
            )
        tenants.sort(key=lambda e: (-e["used"]["total"], e["token_id"]))
        models.append(
            {
                "model_id": label,
                "model_ids": owners,
                "weight": _int(m.get("weight")),
                "part": {tier: _int(m.get(f"{tier}_part")) for tier in _TIERS},
                "used": {**used, "total": used["ram"] + used["nvme"]},
                "tenants": tenants,
            }
        )
    models.sort(key=lambda e: (-e["used"]["total"], e["model_id"]))
    return {
        "available": True,
        "policy": policy if isinstance(policy, str) else None,
        "since": _since(stats),
        "tiers": {
            "ram": {
                "capacity": _int(stats.get("ram_bytes")),
                "used": _int(stats.get("ram_used")),
                "dead": _int(dead.get("ram")),
            },
            "nvme": {
                "capacity": _int(stats.get("nvme_cap_bytes")),
                "used": _int(stats.get("nvme_used")),
                "dead": _int(dead.get("nvme")),
            },
        },
        "models": models,
    }


def network_key(app_state: Any) -> bytes | None:
    """The app's network key, or None (the reports then have ``store: null``)."""
    key_obj = getattr(app_state, "cobs2_key", None)
    nk = getattr(key_obj, "key", None) if key_obj is not None else None
    return nk if isinstance(nk, bytes) else None
