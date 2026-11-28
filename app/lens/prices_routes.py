"""``/api/admin/prices``: the Prices tab (#347).

Providers are derived from what the install uses: the router's upstream, every upstream
seen in the request history, and every provider an admin priced by hand. Each model shows
the price in effect and where it comes from (the admin's override, else the catalogue:
OpenRouter, LiteLLM or the snapshot shipped with the release) and the date it was read.
A model with no known price has none here, and Prompt Lens shows tokens for it.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlsplit

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.lens import price_store as store
from app.lens.price_sources import HOST_LABELS, model_key
from app.lens.prices import REFERENCE_PREFIX
from app.router.rules import DEFAULT_UPSTREAM_URL

router = APIRouter(prefix="/api/admin/prices", tags=["prompt-lens"])

SEEN_DAYS = 90
_PRICE_COLS = (
    "currency, input_per_m_micros, output_per_m_micros, cache_read_per_m_micros, "
    "cache_write_per_m_micros"
)


class PriceBody(BaseModel):
    """Prices in micro-units of the currency per million tokens (100000 = 0.10)."""

    currency: str = Field(default="USD", min_length=3, max_length=3)
    input_per_m_micros: int = Field(ge=0, le=10**12)
    output_per_m_micros: int | None = Field(default=None, ge=0, le=10**12)
    cache_read_per_m_micros: int | None = Field(default=None, ge=0, le=10**12)
    cache_write_per_m_micros: int | None = Field(default=None, ge=0, le=10**12)


class PriceOverrideBody(PriceBody):
    provider: str = Field(min_length=1, max_length=120)
    model: str = Field(min_length=1, max_length=200)


class PriceSettingsBody(BaseModel):
    refresh_interval_hours: int = Field(ge=0, le=store.MAX_INTERVAL_H)


def _label(host: str) -> str:
    return HOST_LABELS.get(host, host)


def _price(r: Any) -> dict[str, Any]:
    cur, inp, out, read, write = r
    return {
        "currency": cur,
        "input_per_m_micros": inp,
        "output_per_m_micros": out,
        "cache_read_per_m_micros": read,
        "cache_write_per_m_micros": write,
    }


async def _catalogue(db: aiosqlite.Connection, host: str, model: str) -> dict[str, Any] | None:
    cur = await db.execute(
        f"SELECT {_PRICE_COLS}, source, fetched_at, model FROM price_catalogue "
        "WHERE host = ? AND model_key = ?",
        (host, model_key(model)),
    )
    r = await cur.fetchone()
    if r is None:
        return None
    return {**_price(r[:5]), "source": r[5], "fetched_at": r[6], "catalogue_model": r[7]}


async def _override(db: aiosqlite.Connection, host: str, model: str) -> dict[str, Any] | None:
    cur = await db.execute(
        f"SELECT model, {_PRICE_COLS}, effective_from FROM rate_cards "
        "WHERE org_id IS NULL AND provider = ? ORDER BY effective_from DESC, id DESC",
        (host,),
    )
    rows = await cur.fetchall()
    pick = next((r for r in rows if r[0] == model), None) or next(
        (r for r in rows if model_key(r[0]) == model_key(model)), None
    )
    if pick is None:
        return None
    cur = await db.execute(
        "SELECT ts, actor FROM price_audit WHERE action = 'override_set' AND provider = ? "
        "AND model = ? ORDER BY id DESC LIMIT 1",
        (host, pick[0]),
    )
    a = await cur.fetchone()
    return {
        **_price(pick[1:6]),
        "effective_from": pick[6],
        "model": pick[0],
        "by": a[1] if a else None,
        "at": float(a[0]) if a else None,
    }


async def entry(db: aiosqlite.Connection, host: str, model: str) -> dict[str, Any]:
    """The price in effect for ``(host, model)`` and its provenance."""
    ov = await _override(db, host, model)
    cat = await _catalogue(db, host, model)
    out: dict[str, Any] = {"model": model, "price": None, "source": None, "fetched_at": None}
    if ov is not None:
        out["price"] = {k: v for k, v in ov.items() if k in _PRICE_KEYS}
        out["source"] = "override"
        out["fetched_at"] = ov["effective_from"]
        out["override"] = {"by": ov["by"], "at": ov["at"]}
        out["list_price"] = {k: v for k, v in cat.items() if k in _PRICE_KEYS} if cat else None
        out["list_source"] = cat["source"] if cat else None
    elif cat is not None:
        out["price"] = {k: v for k, v in cat.items() if k in _PRICE_KEYS}
        out["source"] = cat["source"]
        out["fetched_at"] = cat["fetched_at"]
    return out


_PRICE_KEYS = (
    "currency",
    "input_per_m_micros",
    "output_per_m_micros",
    "cache_read_per_m_micros",
    "cache_write_per_m_micros",
)


async def _router_host(db: aiosqlite.Connection) -> str | None:
    cur = await db.execute("SELECT value FROM settings WHERE key = 'router_upstream_url'")
    row = await cur.fetchone()
    try:
        return urlsplit(row[0] if row and row[0] else DEFAULT_UPSTREAM_URL).hostname
    except ValueError:
        return None


async def _providers(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    reasons: dict[str, list[str]] = {}
    models: dict[str, dict[str, int]] = {}

    def add(host: str | None, why: str) -> None:
        if host:
            r = reasons.setdefault(host, [])
            if why not in r:
                r.append(why)
            models.setdefault(host, {})

    add(await _router_host(db), "router upstream")
    cur = await db.execute(
        "SELECT upstream, model, COUNT(*) FROM request_history WHERE upstream IS NOT NULL "
        "AND finished_at > ? GROUP BY upstream, model ORDER BY COUNT(*) DESC LIMIT 300",
        (time.time() - SEEN_DAYS * 86400,),
    )
    for host, model, n in await cur.fetchall():
        add(host, "relayed requests")
        models[host][model] = n
    cur = await db.execute(
        "SELECT DISTINCT provider, model FROM rate_cards WHERE org_id IS NULL "
        "ORDER BY provider, model"
    )
    for host, model in await cur.fetchall():
        add(host, "price set by hand")
        models[host].setdefault(model, 0)
    out = []
    for host in sorted(reasons, key=lambda h: (h not in HOST_LABELS, _label(h).lower())):
        cur = await db.execute("SELECT COUNT(*) FROM price_catalogue WHERE host = ?", (host,))
        row = await cur.fetchone()
        count = row[0] if row else 0
        in_use = [
            {**await entry(db, host, m), "requests": n}
            for m, n in sorted(models[host].items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        out.append(
            {
                "host": host,
                "label": _label(host),
                "reasons": reasons[host],
                "catalogue_models": count,
                "models": in_use,
            }
        )
    return out


async def _local(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    cur = await db.execute("SELECT id, served_model_name FROM models ORDER BY served_model_name")
    local = await cur.fetchall()
    cur = await db.execute(
        "SELECT key, value FROM settings WHERE key LIKE ?", (REFERENCE_PREFIX + "%",)
    )
    refs: dict[str, dict[str, Any]] = {}
    for key, value in await cur.fetchall():
        try:
            d = json.loads(value)
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get("provider"), str):
            refs[key[len(REFERENCE_PREFIX) :]] = d
    out = []
    for mid, name in local:
        ref = refs.get(mid)
        out.append(
            {
                "id": mid,
                "name": name,
                "reference": (
                    {"provider": ref["provider"], "model": ref.get("model")} if ref else None
                ),
                "priced_as": (
                    await entry(db, ref["provider"], str(ref.get("model"))) if ref else None
                ),
            }
        )
    return out


async def _refresh_block(db: aiosqlite.Connection) -> dict[str, Any]:
    st = await store.get_status(db)
    meta, snap_rows = store.load_snapshot()
    cur = await db.execute("SELECT COUNT(*), MAX(fetched_at) FROM price_catalogue")
    n, newest = (await cur.fetchone()) or (0, None)
    return {
        "interval_hours": await store.get_interval_hours(db),
        "network_allowed": store.network_allowed(),
        "last_attempt": st.get("last_attempt"),
        "last_success": st.get("last_success"),
        "last_error": st.get("last_error"),
        "last_source": st.get("source"),
        "catalogue_rows": n,
        "catalogue_newest": newest,
        "snapshot": {
            "source": meta.get("source"),
            "fetched_at": meta.get("fetched_at"),
            "rows": len(snap_rows),
        },
        "sources": ["openrouter.ai (primary)", "raw.githubusercontent.com, LiteLLM (fallback)"],
    }


@router.get("")
async def get_prices(request: Request, _user: str = Depends(require_jwt)) -> dict[str, Any]:
    """Every provider the install uses with its models and prices, the local models and
    what each is priced as, and the refresh state."""
    async with open_db(request.app.state.settings.db_path) as db:
        return {
            "providers": await _providers(db),
            "local": await _local(db),
            "refresh": await _refresh_block(db),
        }


@router.get("/models")
async def search_models(
    request: Request,
    provider: str = Query(min_length=1, max_length=120),
    q: str = Query(default="", max_length=80),
    limit: int = Query(default=50, ge=1, le=200),
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """The catalogue's models of one provider whose name contains ``q``."""
    like = "%" + q.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    async with open_db(request.app.state.settings.db_path) as db:
        cur = await db.execute(
            "SELECT model FROM price_catalogue WHERE host = ? AND lower(model) LIKE ? ESCAPE '\\' "
            "ORDER BY model LIMIT ?",
            (provider, like, limit),
        )
        names = [r[0] for r in await cur.fetchall()]
        return {"provider": provider, "models": [await entry(db, provider, n) for n in names]}


@router.put("/override")
async def put_override(
    body: PriceOverrideBody, request: Request, user: str = Depends(require_jwt)
) -> dict[str, Any]:
    """Set the admin's price for one model of one provider. Audited."""
    async with open_db(request.app.state.settings.db_path) as db:
        await store.set_override(
            db,
            actor=user,
            provider=body.provider,
            model=body.model,
            prices={
                **body.model_dump(exclude={"provider", "model"}),
                "currency": body.currency.upper(),
            },
        )
        return await entry(db, body.provider, body.model)


@router.delete("/override")
async def delete_override(
    request: Request,
    provider: str = Query(min_length=1, max_length=120),
    model: str = Query(min_length=1, max_length=200),
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Remove the admin's price so the catalogue's applies again. Audited."""
    async with open_db(request.app.state.settings.db_path) as db:
        if not await store.clear_override(db, actor=user, provider=provider, model=model):
            raise HTTPException(status_code=404, detail="no override for that model")
        return await entry(db, provider, model)


@router.post("/refresh")
async def refresh_now(request: Request, _user: str = Depends(require_jwt)) -> dict[str, Any]:
    """Fetch the public price lists now (bounded by the fetch timeouts). A failure keeps the
    last good table and is reported as ``last_error``."""
    await store.refresh_once(request.app.state.settings.db_path)
    async with open_db(request.app.state.settings.db_path) as db:
        return await _refresh_block(db)


@router.put("/settings")
async def put_settings(
    body: PriceSettingsBody, request: Request, user: str = Depends(require_jwt)
) -> dict[str, Any]:
    """The refresh interval in hours (0 turns the automatic refresh off). Audited."""
    async with open_db(request.app.state.settings.db_path) as db:
        await store.set_interval_hours(db, body.refresh_interval_hours, actor=user)
        return await _refresh_block(db)


@router.get("/audit")
async def get_audit(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """The newest changes of prices, reference cards and the refresh interval."""
    async with open_db(request.app.state.settings.db_path) as db:
        return {"entries": await store.recent_audit(db, limit)}
