from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BudgetWindow:
    used_micros: int
    limit_micros: int
    resets_at: str


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    blocked_until: str | None = None
    window: BudgetWindow | None = None


class BudgetPolicy(Protocol):
    async def check(self, user_id: int, org_id: str | None) -> BudgetDecision: ...


class AlwaysAllow:
    """LM Warden never charges; the Hub wires its rolling windows here."""

    async def check(self, user_id: int, org_id: str | None) -> BudgetDecision:
        return BudgetDecision(allowed=True)
