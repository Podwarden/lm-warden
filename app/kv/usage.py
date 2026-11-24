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

- The daemon's ``top_reused`` op (protocol minor 2): the most reloaded stored positions,
  by label. A minor 1 daemon answers ``unknown_op``: the top lists are None.

Reuse (disk-only spec 6, counters design 6.2): the daemon keeps two views of every
stored position, ``charged`` (the key paying for it now, every key's reloads) and
``saver`` (the key that saved it, its own reloads only). The holder's figures and top
list read only the saver view and the holder's own top list, so another key's reloads
never move them. The admin figures add the charged view, regret and the histogram of
reloads at eviction. A ``reuse`` key is in a report only when the daemon sends
``stats.reuse``; the response models read it as null otherwise.

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


def _utc(v: Any) -> str | None:
    """Unix seconds as a SQLite UTC string, the format of every other api timestamp; 0 or
    anything else is None."""
    if isinstance(v, bool) or not isinstance(v, int | float) or not 0 < v < 1e11:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(v))


def _since(stats: Mapping[str, Any]) -> str | None:
    """``started_at`` as a SQLite UTC string."""
    return _utc(stats.get("started_at"))


def _float(v: Any, default: float = 0.0) -> float:
    """A non-negative number from the daemon's reply; anything else is ``default``."""
    if isinstance(v, bool) or not isinstance(v, int | float) or not 0 <= v < 1e18:
        return default
    return float(v)


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


# ---- reuse (disk-only spec 6) ----------------------------------------------------------------

#: The daemon's fixed decay half-life in stage 1, used when its reply has none.
HALF_LIFE_S = 21600.0
#: Rows in each model's admin top list, and in the holder's own.
TOP_N = 10
#: The buckets of reloads at eviction (``lmw_kv`` PROTOCOL.md, ``evicted_hist``).
HIST_BUCKETS = ("0", "1", "2-3", "4-7", "8+")
#: From this many distinct reloading keys on, a top row says "4 or more".
KEYS_SEEN_CAP = 4

_CHARGED_COUNTS = ("reloads", "ram_reloads", "nvme_reloads")
_SAVER_COUNTS = (
    "positions",
    "positions_reloaded",
    "saver_reloads",
    "nvme_bytes_written",
    "nvme_bytes_read",
    "wasted_nvme_bytes",
    "regret",
)


def _hist(raw: Any) -> dict[str, dict[str, dict[str, int]]]:
    h = _map(raw)
    out: dict[str, dict[str, dict[str, int]]] = {}
    for tier in _TIERS:
        row = _map(h.get(tier))
        out[tier] = {}
        for b in HIST_BUCKETS:
            cell = _map(row.get(b))
            out[tier][b] = {
                "positions": _int(cell.get("positions")),
                "bytes": _int(cell.get("bytes")),
            }
    return out


def _reuse_norm(raw: Any) -> dict[str, Any]:
    """A daemon ``reuse`` map read safely: every figure an int, missing ones 0."""
    r = _map(raw)
    ch, sv = _map(r.get("charged")), _map(r.get("saver"))
    charged: dict[str, Any] = {k: _int(ch.get(k)) for k in _CHARGED_COUNTS}
    for k in ("positions", "positions_reloaded"):
        charged[k] = {tier: _int(_map(ch.get(k)).get(tier)) for tier in _TIERS}
    charged["evicted_hist"] = _hist(ch.get("evicted_hist"))
    saver: dict[str, Any] = {k: _int(sv.get(k)) for k in _SAVER_COUNTS}
    saver["evicted_hist"] = _hist(sv.get("evicted_hist"))
    return {"charged": charged, "saver": saver}


def _reuse_add(dst: dict[str, Any], src: Mapping[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, Mapping):
            _reuse_add(dst[k], v)
        else:
            dst[k] += v


def reuse_holder(raw: Mapping[str, Any], half_life_s: float) -> dict[str, Any]:
    """A key's own reuse (``KvReuseHolder``), from the saver view only: reloads by other
    keys of positions it saved never count here."""
    sv = _reuse_norm(raw)["saver"]
    n, reused = sv["positions"], sv["positions_reloaded"]
    return {
        "positions": n,
        "positions_reused": reused,
        "reused_pct": 100.0 * reused / n if n else None,
        "reloads": sv["saver_reloads"],
        "nvme_bytes_written": sv["nvme_bytes_written"],
        "never_reused_nvme_bytes": sv["wasted_nvme_bytes"],
        "half_life_s": half_life_s,
    }


def reuse_admin(raw: Mapping[str, Any], half_life_s: float) -> dict[str, Any]:
    """``reuse_holder`` plus the charged view, disk reads, the histogram of reloads at
    eviction (charged view: every key's reloads) and regret (``KvReuseAdmin``)."""
    r = _reuse_norm(raw)
    ch, sv = r["charged"], r["saver"]
    return {
        **reuse_holder(r, half_life_s),
        "charged_positions": ch["positions"],
        "charged_positions_reused": ch["positions_reloaded"],
        "all_reloads": ch["reloads"],
        "ram_reloads": ch["ram_reloads"],
        "nvme_reloads": ch["nvme_reloads"],
        "nvme_bytes_read": sv["nvme_bytes_read"],
        "evicted_hist": ch["evicted_hist"],
        "regret": sv["regret"],
    }


def _half_life(stats: Mapping[str, Any]) -> float | None:
    """The decay half-life, or None when the daemon reports no reuse (minor 1)."""
    store = stats.get("reuse")
    if not isinstance(store, Mapping):
        return None
    hl = _float(store.get("half_life_s"), HALF_LIFE_S)
    return hl if hl > 0 else HALF_LIFE_S


def _block_tokens(stats: Mapping[str, Any]) -> dict[str, int]:
    """Model (hex) to tokens per block, from the ``by_namespace`` rows that carry a
    ``block_size``; a model whose namespaces disagree is left out (unknown)."""
    seen: dict[str, set[int]] = {}
    for raw in _map(stats.get("by_namespace")).values():
        ns = _map(raw)
        model, bs = ns.get("model"), ns.get("block_size")
        if isinstance(model, str) and isinstance(bs, int) and not isinstance(bs, bool) and bs > 0:
            seen.setdefault(model, set()).add(bs)
    return {m: next(iter(v)) for m, v in seen.items() if len(v) == 1}


def _depth_tokens(depth: Any, block: int | None) -> int | None:
    """``depth * block size``; None when either is unknown (the daemon sends -1)."""
    if block is None or isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        return None
    return int(depth) * block


def _holder_row(r: Mapping[str, Any], blocks: Mapping[str, int]) -> dict[str, Any] | None:
    lab = r.get("label")
    if not isinstance(lab, str):
        return None
    return {
        "label": lab,
        "depth_tokens": _depth_tokens(r.get("depth"), blocks.get(str(r.get("model")))),
        "reloads": _int(r.get("reloads")),
    }


def _top_holder(rows: list[dict[str, Any]], blocks: Mapping[str, int]) -> list[dict[str, Any]]:
    """Holder rows: label, depth as a token offset, the key's own reloads. Nothing else of
    the daemon's row reaches the report."""
    out = (_holder_row(_map(r), blocks) for r in rows)
    return [row for row in out if row is not None]


def _top_admin(
    rows: list[dict[str, Any]],
    blocks: Mapping[str, int],
    names: Mapping[bytes, tuple[str, str]],
) -> list[dict[str, Any]]:
    """Admin rows: the holder fields plus decay, times, tiers, the number of keys that
    reloaded it (capped) and the charged key's name. Never a tenant tag."""
    out: list[dict[str, Any]] = []
    for raw in rows:
        r = _map(raw)
        row = _holder_row(r, blocks)
        if row is None:
            continue
        seen = _int(r.get("tenants_seen"))
        many = r.get("many") is True or seen >= KEYS_SEEN_CAP
        raw_tiers = r.get("tiers")
        tiers: list[Any] = raw_tiers if isinstance(raw_tiers, list) else []
        row.update(
            decayed=_float(r.get("decayed")),
            last_reload=_utc(r.get("last_reload")),
            first_saved=_utc(r.get("first_saved")),
            tiers=[t for t in _TIERS if t in tiers],
            keys_seen=f"{KEYS_SEEN_CAP} or more" if many else str(seen),
            charged_name=_name(names, r.get("charged"))[1],
        )
        out.append(row)
    return out


def nvme_capacity(stats: Mapping[str, Any]) -> int:
    """The NVMe cap in force: ``nvme_cap_effective_bytes`` (minor 2, the operator's cap
    clamped to the disk's room), else ``nvme_cap_bytes`` (an older daemon)."""
    if "nvme_cap_effective_bytes" in stats:
        return _int(stats.get("nvme_cap_effective_bytes"))
    return _int(stats.get("nvme_cap_bytes"))


def _reason(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None


def store_errors(stats: Any, sup: Any) -> dict[str, str | None]:
    """Why the store keeps a tier off (its start guards): ``pool_error`` (no RAM pool) and
    ``nvme_error`` (no disk tier), each None when fine.

    From the daemon's ``stats``; with no reply, the pool error is the supervisor's own (the
    daemon's ``hello``, or why it started no daemon). Never a reason to fail a request.
    """
    if isinstance(stats, Mapping):
        return {
            "pool_error": _reason(stats.get("pool_error")),
            "nvme_error": _reason(stats.get("nvme_error")),
        }
    fn = getattr(sup, "pool_error", None)
    try:
        pool = _reason(fn()) if callable(fn) else None
    except Exception:  # noqa: BLE001 -- fail open
        pool = None
    return {"pool_error": pool, "nvme_error": None}


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


class _TopCache:
    """The last ``top_reused`` replies of one supervisor, per request, and one lock."""

    def __init__(self, sup: Any) -> None:
        self.sup = sup
        self.entries: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]] | None]] = {}
        self.lock = asyncio.Lock()


async def _ask_top(
    sup: Any, model: bytes | None, tenant: bytes | None, view: str, n: int
) -> list[dict[str, Any]] | None:
    client_of = getattr(sup, "stats_client", None)
    if not callable(client_of):
        return None
    try:
        client = client_of()
        top = getattr(client, "top_reused", None) if client is not None else None
        if not callable(top):
            return None
        rows = await asyncio.to_thread(top, model=model, tenant=tenant, view=view, n=n)
    except Exception:  # noqa: BLE001 -- fail open; the reply is never logged
        return None
    if not isinstance(rows, list) or not all(isinstance(r, Mapping) for r in rows):
        return None
    return rows


async def fetch_top(
    app_state: Any, *, model: bytes | None, tenant: bytes | None, view: str, n: int
) -> list[dict[str, Any]] | None:
    """The daemon's ``top_reused`` rows, or None with no daemon, a timeout, an error or
    a minor 1 daemon (``unknown_op``). Asked like ``fetch_stats``: the report connection,
    off the event loop, each request at most once per ``STATS_TTL_S``. Never logged."""
    sup = getattr(app_state, "kv_store", None)
    cache = getattr(app_state, "kv_top_cache", None)
    if not isinstance(cache, _TopCache) or cache.sup is not sup:
        cache = _TopCache(sup)
        try:
            app_state.kv_top_cache = cache
        except AttributeError:
            pass
    key = (model, tenant, view, n)
    async with cache.lock:
        now = time.monotonic()
        hit = cache.entries.get(key)
        if hit is not None and now - hit[0] < STATS_TTL_S:
            return hit[1]
        cache.entries = {k: v for k, v in cache.entries.items() if now - v[0] < STATS_TTL_S}
        value = await _ask_top(sup, model, tenant, view, n)
        cache.entries[key] = (time.monotonic(), value)
        return value


async def fetch_tops(app_state: Any, stats: Any) -> dict[str, list[dict[str, Any]] | None]:
    """Each model's admin top list (``TOP_N`` rows), by model hex, for ``kv_usage``; empty
    when the daemon reports no reuse (a minor 1 daemon has no ``top_reused``)."""
    if not isinstance(stats, Mapping) or _half_life(stats) is None:
        return {}
    out: dict[str, list[dict[str, Any]] | None] = {}
    for hex_model in _map(stats.get("by_model")):
        try:
            model = bytes.fromhex(hex_model)
        except (TypeError, ValueError):
            continue
        out[hex_model] = await fetch_top(app_state, model=model, tenant=None, view="admin", n=TOP_N)
    return out


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
    cap = {"ram": _int(stats.get("ram_bytes")), "nvme": nvme_capacity(stats)}
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
    hl = _half_life(stats)
    if hl is not None:
        raw = _map(t.get("reuse"))
        out["reuse"] = reuse_admin(raw, hl) if names is not None else reuse_holder(raw, hl)
    return out


async def key_cache_report(
    db: aiosqlite.Connection,
    token_id: str,
    stats: dict[str, Any] | None,
    network_key: bytes | None,
    *,
    admin: bool,
    now: float | None = None,
    app_state: Any = None,
) -> dict[str, Any]:
    """One key's KV report. ``store`` is None when the daemon is down or there is no
    network key (no tenant tag can be computed). ``admin`` adds ``store.evicted_by``,
    the fair-share bytes and ``isolation_cost_tokens``; the holder gets neither.
    ``store.reuse`` is the admin or the holder reuse figures. For a holder, with
    ``app_state``, ``top_reused`` is the key's own top list (``fetch_top``, holder view),
    None when the store cannot say.

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
    tag: bytes | None = None
    if isinstance(stats, dict) and network_key is not None and row.kv_tenant_id:
        try:
            names = tenant_names(await repo.list_all(), network_key) if admin else None
            out["store"] = _store_block(row, stats, network_key, names)
            tag = _scopes().tenant_tag(network_key, row.kv_tenant_id)
        except (ValueError, ImportError):
            out["store"] = None
    if not admin and app_state is not None:
        out["top_reused"] = None
        if tag is not None and stats is not None and _half_life(stats) is not None:
            rows = await fetch_top(app_state, model=None, tenant=tag, view="holder", n=TOP_N)
            if rows is not None:
                out["top_reused"] = _top_holder(rows, _block_tokens(stats))
    if admin:
        out["isolation_cost_tokens"] = {"24h": await _isolation_cost_24h(db, token_id, now)}
    return out


def kv_usage(
    stats: Any,
    names: Mapping[bytes, tuple[str, str]],
    top: Mapping[str, list[dict[str, Any]] | None] | None = None,
) -> dict[str, Any]:
    """``GET /api/kv/usage``: the tiers, then each model's part and use, then each key's
    use, share and evictions inside the model (admin only). ``available`` is False with
    no daemon, or one that does not answer.

    With ``stats.reuse``, each model and each key in it has ``reuse`` (admin figures; the
    model's are its keys' summed). With ``top`` (``fetch_tops``), each model has
    ``top_reused``: its admin rows, or None when the store could not say."""
    if not isinstance(stats, Mapping):
        return {"available": False, "policy": None, "since": None, "tiers": None, "models": []}
    dead = _map(stats.get("dead_bytes"))
    policy = stats.get("policy")
    hl = _half_life(stats)
    blocks = _block_tokens(stats)
    models: list[dict[str, Any]] = []
    for hex_model, raw in _map(stats.get("by_model")).items():
        m = _map(raw)
        pair_reuse = _map(m.get("reuse"))
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
            if hl is not None:
                tenants[-1]["reuse"] = reuse_admin(_map(pair_reuse.get(hex_tag)), hl)
        tenants.sort(key=lambda e: (-e["used"]["total"], e["token_id"]))
        entry: dict[str, Any] = {
            "model_id": label,
            "model_ids": owners,
            "weight": _int(m.get("weight")),
            "part": {tier: _int(m.get(f"{tier}_part")) for tier in _TIERS},
            "used": {**used, "total": used["ram"] + used["nvme"]},
            "tenants": tenants,
        }
        if hl is not None:
            total = _reuse_norm({})
            for r in pair_reuse.values():
                _reuse_add(total, _reuse_norm(r))
            entry["reuse"] = reuse_admin(total, hl)
        if top is not None:
            rows = top.get(hex_model)
            entry["top_reused"] = None if rows is None else _top_admin(rows, blocks, names)
        models.append(entry)
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
                "capacity": nvme_capacity(stats),
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
