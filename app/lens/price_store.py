"""Price catalogue, refresh and overrides: the database side of the Prices tab (#347).

``price_catalogue`` holds public list prices (app/lens/price_sources.py): seeded from
the snapshot shipped with the release, then refreshed from the network. A refresh
never deletes what it cannot replace, and a failed one changes nothing but the status.
An admin's price is a ``rate_cards`` row (org_id NULL, dated today) that wins over the
catalogue (app/lens/prices.py); every set, clear and interval change is a
``price_audit`` row.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import httpx

from app.db.database import open_db
from app.lens import price_sources as src
from app.lens.price_sources import PriceRow, model_key

log = logging.getLogger("vllm_warden.prices")

SNAPSHOT_PATH = Path(__file__).with_name("price_snapshot.json")
STATUS_KEY = "price_refresh_status"
INTERVAL_KEY = "price_refresh_interval_hours"
DEFAULT_INTERVAL_H = 24
MAX_INTERVAL_H = 24 * 30
RETRY_AFTER_FAILURE_S = 3600.0
#: A catalogue row not refreshed for this long is dropped (the model left every list).
PRUNE_AFTER_DAYS = 120
_FALSE = {"0", "false", "no", "off"}


def network_allowed() -> bool:
    """False on an offline install (``HF_HUB_OFFLINE=1``) or with ``VW_PRICE_REFRESH=0``."""
    if os.environ.get("VW_PRICE_REFRESH", "").strip().lower() in _FALSE:
        return False
    return os.environ.get("HF_HUB_OFFLINE", "").strip() != "1"


# -- the bundled snapshot ------------------------------------------------------------


def load_snapshot() -> tuple[dict[str, Any], list[PriceRow]]:
    """``(meta, rows)`` of the snapshot shipped with the release; empty when unreadable."""
    try:
        d = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, []
    if not isinstance(d, dict) or not isinstance(d.get("rows"), list):
        return {}, []
    at = str(d.get("fetched_at") or "")
    rows: list[PriceRow] = []
    for r in d["rows"]:
        try:
            host, name, inp, out, read, write = r
            rows.append(
                PriceRow(host, model_key(name), name, inp, out, read, write, "snapshot", at)
            )
        except (TypeError, ValueError):
            continue
    meta = {k: d.get(k) for k in ("source", "fetched_at", "note")}
    return meta, rows


def snapshot_json(rows: list[PriceRow], source: str, at: str, note: str) -> str:
    """The text of a snapshot file for ``rows`` (scripts/refresh_price_snapshot.py)."""
    body = {
        "note": note,
        "source": source,
        "fetched_at": at,
        "rows": [
            [r.host, r.model, r.input_per_m_micros, r.output_per_m_micros,
             r.cache_read_per_m_micros, r.cache_write_per_m_micros]
            for r in sorted(rows, key=lambda r: (r.host, r.model_key))
        ],
    }  # fmt: skip
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False) + "\n"


# -- the catalogue ---------------------------------------------------------------------

_COLS = (
    "host, model_key, model, currency, input_per_m_micros, output_per_m_micros, "
    "cache_read_per_m_micros, cache_write_per_m_micros, source, fetched_at"
)


def _values(r: PriceRow) -> tuple[Any, ...]:
    return (
        r.host,
        r.model_key,
        r.model,
        r.currency,
        r.input_per_m_micros,
        r.output_per_m_micros,
        r.cache_read_per_m_micros,
        r.cache_write_per_m_micros,
        r.source,
        r.fetched_at,
    )


async def upsert_rows(db: aiosqlite.Connection, rows: list[PriceRow]) -> None:
    """Insert or replace ``rows`` (a newer answer wins) and drop rows nobody refreshed
    for ``PRUNE_AFTER_DAYS``. One transaction."""
    await db.executemany(
        f"INSERT INTO price_catalogue ({_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(host, model_key) DO UPDATE SET model = excluded.model, "
        "currency = excluded.currency, input_per_m_micros = excluded.input_per_m_micros, "
        "output_per_m_micros = excluded.output_per_m_micros, "
        "cache_read_per_m_micros = excluded.cache_read_per_m_micros, "
        "cache_write_per_m_micros = excluded.cache_write_per_m_micros, "
        "source = excluded.source, fetched_at = excluded.fetched_at",
        [_values(r) for r in rows],
    )
    cutoff = (datetime.now(UTC) - timedelta(days=PRUNE_AFTER_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    await db.execute(
        "DELETE FROM price_catalogue WHERE source != 'snapshot' AND fetched_at < ?", (cutoff,)
    )
    await db.commit()


async def seed_snapshot(db: aiosqlite.Connection) -> int:
    """Put the bundled snapshot in the catalogue: rows it has that the catalogue lacks,
    and rows the catalogue itself took from an older snapshot. Never replaces a row
    fetched from the network. Returns the number of rows written."""
    _meta, rows = load_snapshot()
    if not rows:
        return 0
    cur = await db.execute("SELECT host, model_key, source, fetched_at FROM price_catalogue")
    have = {(h, k): (s, f) for h, k, s, f in await cur.fetchall()}
    todo = []
    for r in rows:
        cur_row = have.get((r.host, r.model_key))
        if cur_row is None or (cur_row[0] == "snapshot" and cur_row[1] < r.fetched_at):
            todo.append(r)
    if todo:
        await db.executemany(
            f"INSERT OR REPLACE INTO price_catalogue ({_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [_values(r) for r in todo],
        )
        await db.commit()
    return len(todo)


# -- refresh settings and status -------------------------------------------------------------


async def get_interval_hours(db: aiosqlite.Connection) -> int:
    """The refresh interval in hours (0: no automatic refresh)."""
    cur = await db.execute("SELECT value FROM settings WHERE key = ?", (INTERVAL_KEY,))
    row = await cur.fetchone()
    try:
        h = int(str(row[0]).strip()) if row else DEFAULT_INTERVAL_H
    except ValueError:
        h = DEFAULT_INTERVAL_H
    return min(max(h, 0), MAX_INTERVAL_H)


async def set_interval_hours(db: aiosqlite.Connection, hours: int, *, actor: str) -> int:
    old = await get_interval_hours(db)
    await db.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (INTERVAL_KEY, str(hours)),
    )
    await audit(db, actor, "interval", None, None, old, hours)
    await db.commit()
    return old


async def get_status(db: aiosqlite.Connection) -> dict[str, Any]:
    cur = await db.execute("SELECT value FROM settings WHERE key = ?", (STATUS_KEY,))
    row = await cur.fetchone()
    try:
        d = json.loads(row[0]) if row else {}
    except ValueError:
        d = {}
    return d if isinstance(d, dict) else {}


async def _put_status(db: aiosqlite.Connection, status: dict[str, Any]) -> None:
    await db.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (STATUS_KEY, json.dumps(status)),
    )
    await db.commit()


async def refresh_once(
    db_path: Path, *, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, Any]:
    """One refresh: fetch (OpenRouter, else LiteLLM), merge into the catalogue, record the
    outcome. A failure leaves the catalogue as it was. Never raises."""
    now = time.time()
    try:
        if not network_allowed():
            raise RuntimeError("network refresh is off (offline install or VW_PRICE_REFRESH=0)")
        rows, source, note = await src.fetch_rows(transport=transport)
    except Exception as exc:  # noqa: BLE001 -- the last good table stays
        err = f"{type(exc).__name__}: {str(exc)[:300]}"
        log.info("price refresh failed, keeping the last good table: %s", err)
        async with open_db(db_path) as db:
            st = await get_status(db)
            st.update(last_attempt=now, last_error=err)
            await _put_status(db, st)
            return st
    async with open_db(db_path) as db:
        await upsert_rows(db, rows)
        st = await get_status(db)
        st.update(
            last_attempt=now,
            last_success=now,
            last_error=None,
            source=source,
            rows=len(rows),
            note=note,
        )
        await _put_status(db, st)
        return st


def due(status: dict[str, Any], interval_h: int, now: float | None = None) -> bool:
    """Whether a refresh is due: the interval has passed since the last success, and a
    failed attempt is retried after an hour."""
    if interval_h <= 0:
        return False
    now = time.time() if now is None else now
    last_ok = float(status.get("last_success") or 0)
    last_try = float(status.get("last_attempt") or 0)
    if now - last_ok < interval_h * 3600:
        return False
    return now - last_try >= RETRY_AFTER_FAILURE_S


#: The catalogue is seeded this long after startup (a short write transaction kept off the
#: startup path); until then Prompt Lens prices from the snapshot held in memory
#: (app/lens/prices.py), so nothing waits for it.
SEED_DELAY_S = 15.0
FIRST_DELAY_S = 30.0
POLL_S = 900.0


async def run_forever(db_path: Path) -> None:
    """The background refresher: seeds the catalogue from the snapshot at once (a local
    file), then refreshes on the settings interval. Never blocks startup, never raises."""
    import asyncio

    await asyncio.sleep(SEED_DELAY_S)
    try:
        async with open_db(db_path) as db:
            await seed_snapshot(db)
    except Exception:  # noqa: BLE001
        log.warning("price snapshot seed failed", exc_info=True)
    await asyncio.sleep(FIRST_DELAY_S)
    while True:
        try:
            async with open_db(db_path) as db:
                interval = await get_interval_hours(db)
                status = await get_status(db)
            if network_allowed() and due(status, interval):
                await refresh_once(db_path)
        except Exception:  # noqa: BLE001 -- the loop must outlive any one pass
            log.debug("price refresh pass failed", exc_info=True)
        await asyncio.sleep(POLL_S)


# -- audit and overrides -------------------------------------------------------------------


async def audit(
    db: aiosqlite.Connection,
    actor: str,
    action: str,
    provider: str | None,
    model: str | None,
    old: Any,
    new: Any,
) -> None:
    """One ``price_audit`` row (the caller commits)."""
    await db.execute(
        "INSERT INTO price_audit (ts, actor, action, provider, model, old_value, new_value)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            time.time(),
            actor,
            action,
            provider,
            model,
            None if old is None else json.dumps(old),
            None if new is None else json.dumps(new),
        ),
    )


_PRICE_FIELDS = (
    "currency",
    "input_per_m_micros",
    "output_per_m_micros",
    "cache_read_per_m_micros",
    "cache_write_per_m_micros",
)


async def _override_row(
    db: aiosqlite.Connection, provider: str, model: str
) -> tuple[str, dict[str, Any], str] | None:
    """``(id, prices, effective_from)`` of the newest platform card with exactly this key."""
    cur = await db.execute(
        f"SELECT id, {', '.join(_PRICE_FIELDS)}, effective_from FROM rate_cards "
        "WHERE org_id IS NULL AND provider = ? AND model = ? "
        "ORDER BY effective_from DESC, id DESC LIMIT 1",
        (provider, model),
    )
    r = await cur.fetchone()
    if r is None:
        return None
    return r[0], dict(zip(_PRICE_FIELDS, r[1:-1], strict=True)), r[-1]


async def set_override(
    db: aiosqlite.Connection,
    *,
    actor: str,
    provider: str,
    model: str,
    prices: dict[str, Any],
    effective_from: str | None = None,
) -> None:
    """Make ``prices`` the admin's price for ``(provider, model)``: a ``rate_cards`` row
    dated ``effective_from`` (today), replacing a row with the same date. Audited. Commits."""
    day = effective_from or datetime.now(UTC).strftime("%Y-%m-%d")
    cur = await db.execute(
        "SELECT id FROM rate_cards WHERE org_id IS NULL AND provider = ? AND model = ? "
        "AND effective_from = ?",
        (provider, model, day),
    )
    same = await cur.fetchone()
    before = await _override_row(db, provider, model)
    vals = tuple(prices.get(f) for f in _PRICE_FIELDS)
    if same is not None:
        await db.execute(
            "UPDATE rate_cards SET currency = ?, input_per_m_micros = ?, output_per_m_micros = ?, "
            "cache_read_per_m_micros = ?, cache_write_per_m_micros = ? WHERE id = ?",
            (*vals, same[0]),
        )
    else:
        await db.execute(
            "INSERT INTO rate_cards (id, org_id, provider, model, currency, input_per_m_micros, "
            "output_per_m_micros, cache_read_per_m_micros, cache_write_per_m_micros, markup_pct, "
            "effective_from) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (uuid.uuid4().hex, provider, model, *vals, day),
        )
    new = {f: prices.get(f) for f in _PRICE_FIELDS}
    await audit(db, actor, "override_set", provider, model, before[1] if before else None, new)
    await db.commit()


async def clear_override(
    db: aiosqlite.Connection, *, actor: str, provider: str, model: str
) -> bool:
    """Remove every platform card for ``(provider, model)`` so the catalogue price applies
    again. Audited. Returns False when there was none. Commits."""
    before = await _override_row(db, provider, model)
    if before is None:
        return False
    await db.execute(
        "DELETE FROM rate_cards WHERE org_id IS NULL AND provider = ? AND model = ?",
        (provider, model),
    )
    await audit(db, actor, "override_clear", provider, model, before[1], None)
    await db.commit()
    return True


async def recent_audit(db: aiosqlite.Connection, limit: int = 50) -> list[dict[str, Any]]:
    cur = await db.execute(
        "SELECT ts, actor, action, provider, model, old_value, new_value FROM price_audit "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    out = []
    for ts, actor, action, provider, model, old, new in await cur.fetchall():
        out.append(
            {
                "ts": float(ts),
                "actor": actor,
                "action": action,
                "provider": provider,
                "model": model,
                "old": json.loads(old) if old else None,
                "new": json.loads(new) if new else None,
            }
        )
    return out


async def set_reference(
    db: aiosqlite.Connection,
    *,
    actor: str,
    model_id: str,
    provider: str | None,
    model: str | None,
) -> None:
    """Price own model ``model_id`` as ``provider``/``model`` (both None clears it). Audited.
    Commits. The caller has checked the model exists."""
    from app.lens.prices import REFERENCE_PREFIX

    key = REFERENCE_PREFIX + model_id
    cur = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = await cur.fetchone()
    old = json.loads(row[0]) if row else None
    if provider is None:
        await db.execute("DELETE FROM settings WHERE key = ?", (key,))
        new = None
    else:
        new = {"provider": provider, "model": model}
        await db.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(new)),
        )
    await audit(db, actor, "reference", None, model_id, old, new)
    await db.commit()
