"""Prompt Lens prices: which card prices a session, and what a fix saves in money
(analysis spec 5.2, 5.4).

Money is shown only with a price: a relayed request uses the provider's price; an
own engine uses the price of the model the admin picked as its reference
(``settings`` key ``lens_reference_card:<model_id>``); with none the lens shows tokens
and seconds only. A price is, in order: the admin's override (a ``rate_cards`` row,
migration 0025), else the public list price in ``price_catalogue`` (refreshed daily,
seeded from the snapshot shipped with the release: app/lens/price_store.py). A card's ``effective_from``
date is shown next to every money figure. Reads here are synchronous (the lens
services run in a worker thread). A card holds prices, never text (A-10).
"""

from __future__ import annotations

import functools
import json
import sqlite3
from datetime import UTC, datetime
from typing import Any, Literal

from app.chat2.ledger import _CARD_COLS, RateCard
from app.lens.catalogue import Saving
from app.lens.price_sources import PriceRow, model_key
from app.lens.price_store import load_snapshot

REFERENCE_PREFIX = "lens_reference_card:"
_PER_M = 1_000_000.0


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def find_card(
    db: sqlite3.Connection, provider: str, model: str, at: str | None = None
) -> RateCard | None:
    """The price of ``(provider, model)``: the admin's newest card effective by ``at``
    (exact model name first, else a name that folds the same), else the catalogue's."""
    try:
        rows = db.execute(
            f"SELECT {_CARD_COLS} FROM rate_cards WHERE provider = ? "
            "AND org_id IS NULL AND effective_from <= ? "
            "ORDER BY effective_from DESC, id DESC",
            (provider, at or _today()),
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    for r in rows:
        if r[3] == model:
            return RateCard(*r)
    key = model_key(model)
    for r in rows:
        if model_key(r[3]) == key:
            return RateCard(*r)
    return _catalogue_card(db, provider, model)


def _catalogue_card(db: sqlite3.Connection, provider: str, model: str) -> RateCard | None:
    try:
        r = db.execute(
            "SELECT host, model, currency, input_per_m_micros, output_per_m_micros, "
            "cache_read_per_m_micros, cache_write_per_m_micros, fetched_at FROM price_catalogue "
            "WHERE host = ? AND model_key = ?",
            (provider, model_key(model)),
        ).fetchone()
    except sqlite3.OperationalError:
        return _snapshot_card(provider, model)
    if r is None:
        return _snapshot_card(provider, model)
    host, name, cur, inp, out, read, write, at = r
    # ``out`` may be unknown: 0 makes the output-side figure absent (price_saving needs > 0).
    return RateCard(
        f"catalogue:{host}:{name}",
        None,
        host,
        name,
        inp,
        out or 0,
        None,
        read,
        write,
        0,
        cur,
        at[:10],
    )


def reference_of(db: sqlite3.Connection, model_id: str) -> tuple[str, str] | None:
    """The ``(provider, model)`` the admin priced own model ``model_id`` as."""
    try:
        r = db.execute(
            "SELECT value FROM settings WHERE key = ?", (REFERENCE_PREFIX + model_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if r is None:
        return None
    try:
        d = json.loads(r[0])
    except (TypeError, ValueError):
        return None
    if (
        isinstance(d, dict)
        and isinstance(d.get("provider"), str)
        and isinstance(d.get("model"), str)
    ):
        return d["provider"], d["model"]
    return None


def card_for(
    db: sqlite3.Connection,
    *,
    relayed: bool,
    upstream: str | None,
    model: str,
    model_id: str,
    at: str | None = None,
) -> RateCard | None:
    """Relays: the provider's card. Own engines: the reference card, or None."""
    if relayed:
        return find_card(db, upstream, model, at) if upstream else None
    ref = reference_of(db, model_id)
    return find_card(db, ref[0], ref[1], at) if ref is not None else None


@functools.lru_cache(maxsize=1)
def _snapshot_index() -> dict[tuple[str, str], PriceRow]:
    return {(r.host, r.model_key): r for r in load_snapshot()[1]}


def _snapshot_card(provider: str, model: str) -> RateCard | None:
    """The price in the snapshot shipped with the release, for the moments before the
    catalogue is seeded and for a catalogue that could not be read."""
    r = _snapshot_index().get((provider, model_key(model)))
    if r is None:
        return None
    return RateCard(
        f"snapshot:{r.host}:{r.model}", None, r.host, r.model, r.input_per_m_micros,
        r.output_per_m_micros or 0, None, r.cache_read_per_m_micros, r.cache_write_per_m_micros,
        0, r.currency, r.fetched_at[:10],
    )  # fmt: skip


def _micros(v: int | None) -> float | None:
    return None if v is None else v / _PER_M / _PER_M  # micros per M tokens -> currency per token


def price_saving(
    s: Saving,
    card: RateCard | None,
    *,
    cached_share: float,
    kind: Literal["reuse", "waste", "output"] = "reuse",
) -> Saving:
    """``s`` with ``money`` filled: the currency saved over the session, in the card's currency.

    ``reuse`` (E): ``s.tokens`` are billed at the input price today and would be
    read at the cache-read price after the fix. ``cached_share`` is the share of
    the session's prompt those tokens are; the rest of the prompt is written
    once, and a card with a write price charges its premium on that (spec 5.2's
    worked example). ``waste`` (C): dead weight that is not sent after the fix,
    billed today at the read price for the share of the prompt the cache served
    and at the input price for the rest. ``output`` (R): the wasted completion
    tokens at the output price. No card, or a card without the price the figure
    needs: money stays None, never a guess."""
    if card is None or s.tokens <= 0:
        return s
    if kind == "output":
        price = _micros(card.output_per_m_micros)
        return Saving(s.tokens, s.prefill_s, s.decode_s, s.tokens * price if price else None)
    inp, read = _micros(card.input_per_m_micros), _micros(card.cache_read_per_m_micros)
    if not inp:
        return s
    if kind == "waste":
        cs = min(max(cached_share, 0.0), 1.0)
        rd = inp if read is None else min(read, inp)
        return Saving(s.tokens, s.prefill_s, s.decode_s, s.tokens * (cs * rd + (1 - cs) * inp))
    if read is None:
        return s
    money = s.tokens * (inp - min(read, inp))
    write = _micros(card.cache_write_per_m_micros)
    if write is not None and write > inp and 0 < cached_share < 1:
        written = s.tokens * (1 - cached_share) / cached_share
        money -= written * (write - inp)
    return Saving(s.tokens, s.prefill_s, s.decode_s, max(money, 0.0))


def session_input_cost(prompt_tokens: int, card: RateCard | None) -> float | None:
    """What a session's prompt cost as billed, with nothing read from a cache."""
    inp = _micros(card.input_per_m_micros) if card is not None else None
    return prompt_tokens * inp if inp else None


def load_defaults() -> list[dict[str, Any]]:
    """Retired with #347: the shipped list prices are now the catalogue (Settings, Prices);
    ``GET /api/admin/lens/rate-cards`` still answers a ``defaults`` list, always empty."""
    return []
