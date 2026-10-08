from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import aiosqlite

from app.chat2.clock import now_iso
from app.chat2.ids import new_id


@dataclass(frozen=True)
class Usage:
    prompt: int
    completion: int
    reasoning: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None
    estimated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "reasoning": self.reasoning,
            "cacheRead": self.cache_read,
            "cacheWrite": self.cache_write,
            "estimated": self.estimated,
        }


@dataclass(frozen=True)
class RateCard:
    id: str
    org_id: str | None
    provider: str
    model: str
    input_per_m_micros: int
    output_per_m_micros: int
    reasoning_per_m_micros: int | None
    cache_read_per_m_micros: int | None
    cache_write_per_m_micros: int | None
    markup_pct: int
    currency: str
    effective_from: str


@dataclass(frozen=True)
class CostResult:
    cost_micros: int | None
    cost_status: str  # priced | unpriced | incomplete_card


def compute_cost(usage: Usage, card: RateCard | None) -> CostResult:
    if card is None:
        return CostResult(None, "unpriced")
    parts: list[tuple[int | None, int | None]] = [
        (usage.prompt, card.input_per_m_micros),
        (usage.completion, card.output_per_m_micros),
        (usage.reasoning, card.reasoning_per_m_micros),
        (usage.cache_read, card.cache_read_per_m_micros),
        (usage.cache_write, card.cache_write_per_m_micros),
    ]
    total = Decimal(0)
    for tokens, rate in parts:
        if not tokens:
            continue
        if rate is None:
            return CostResult(None, "incomplete_card")
        total += Decimal(tokens) * Decimal(rate) / Decimal(1_000_000)
    total = total * (Decimal(100 + card.markup_pct) / Decimal(100))
    return CostResult(int(total.quantize(Decimal(1), rounding=ROUND_HALF_UP)), "priced")


_CARD_COLS = (
    "id, org_id, provider, model, input_per_m_micros, output_per_m_micros, "
    "reasoning_per_m_micros, cache_read_per_m_micros, cache_write_per_m_micros, markup_pct, "
    "currency, effective_from"
)


class LedgerRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def find_card(
        self, provider: str, model: str, org_id: str | None, at: str
    ) -> RateCard | None:
        for scope in ([org_id] if org_id else []) + [None]:
            cur = await self.db.execute(
                f"SELECT {_CARD_COLS} FROM rate_cards WHERE provider = ? AND model = ? "
                f"AND org_id {'= ?' if scope else 'IS NULL'} AND effective_from <= ? "
                "ORDER BY effective_from DESC, id DESC LIMIT 1",
                (provider, model, *([scope] if scope else []), at),
            )
            r = await cur.fetchone()
            if r:
                return RateCard(*r)
        return None

    async def find(
        self, request_id: str, *, user_id: int, purpose: str = "turn"
    ) -> tuple[str, str | None] | None:
        """`(outcome, message_id)` of an already-recorded request, else None.

        The turn endpoint replays a *truthful* `done` for an idempotent retry
        (a turn that failed must not replay as `stop`), so it needs the
        recorded outcome and not merely existence.

        Scoped by `user_id` **and** `purpose`: `request_id` is client-supplied
        and carries no uniqueness guarantee across users, so an unscoped lookup
        would let one user's id collide with another's and silently drop the
        second user's turn (replaying the first user's message_id back at them).
        """
        cur = await self.db.execute(
            "SELECT outcome, message_id FROM usage_ledger WHERE request_id = ? "
            "AND user_id = ? AND purpose = ? ORDER BY created_at DESC, id DESC LIMIT 1",
            (request_id, user_id, purpose),
        )
        r = await cur.fetchone()
        return (str(r[0]), r[1]) if r else None

    async def record(
        self,
        *,
        request_id: str,
        user_id: int,
        org_id: str | None,
        chat_id: str | None,
        message_id: str | None,
        purpose: str,
        provider: str,
        model: str,
        usage: Usage,
        outcome: str,
        provider_request_id: str | None = None,
    ) -> tuple[str, CostResult]:
        ts = now_iso()
        card = await self.find_card(provider, model, org_id, ts)
        cost = compute_cost(usage, card)
        lid = new_id()
        await self.db.execute(
            "INSERT INTO usage_ledger(id, request_id, provider_request_id, user_id, org_id, "
            "chat_id, message_id, purpose, provider, model, prompt_tokens, completion_tokens, "
            "reasoning_tokens, cache_read_tokens, cache_write_tokens, estimated, outcome, "
            "rate_card_id, cost_micros, currency, cost_status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                lid,
                request_id,
                provider_request_id,
                user_id,
                org_id,
                chat_id,
                message_id,
                purpose,
                provider,
                model,
                usage.prompt,
                usage.completion,
                usage.reasoning,
                usage.cache_read,
                usage.cache_write,
                int(usage.estimated),
                outcome,
                card.id if card else None,
                cost.cost_micros,
                card.currency if card else None,
                cost.cost_status,
                ts,
            ),
        )
        await self.db.commit()
        return lid, cost
