"""``router_rules`` repository (#287) -- the ONLY writer of the table."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import aiosqlite

_COLS = (
    "id, position, pattern, target_model_id, enabled, fallback, strip_thinking, "
    "min_max_tokens, created_at, updated_at"
)
_UPDATABLE = frozenset(
    {"pattern", "target_model_id", "enabled", "fallback", "strip_thinking", "min_max_tokens"}
)
_BOOL_COLS = frozenset({"enabled", "fallback", "strip_thinking"})


@dataclass
class RouterRuleRow:
    id: str
    position: int
    pattern: str
    target_model_id: str
    enabled: bool
    fallback: bool
    strip_thinking: bool
    min_max_tokens: int
    created_at: str
    updated_at: str


def _decode(r: aiosqlite.Row) -> RouterRuleRow:
    return RouterRuleRow(
        id=r[0],
        position=int(r[1]),
        pattern=r[2],
        target_model_id=r[3],
        enabled=bool(r[4]),
        fallback=bool(r[5]),
        strip_thinking=bool(r[6]),
        min_max_tokens=int(r[7]),
        created_at=r[8],
        updated_at=r[9],
    )


class RouterRuleRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def list_all(self) -> list[RouterRuleRow]:
        cur = await self.db.execute(f"SELECT {_COLS} FROM router_rules ORDER BY position, id")
        return [_decode(r) for r in await cur.fetchall()]

    async def get(self, rule_id: str) -> RouterRuleRow | None:
        cur = await self.db.execute(f"SELECT {_COLS} FROM router_rules WHERE id = ?", (rule_id,))
        r = await cur.fetchone()
        return _decode(r) if r else None

    async def create(
        self,
        pattern: str,
        target_model_id: str,
        enabled: bool = True,
        fallback: bool = True,
        strip_thinking: bool = True,
        min_max_tokens: int = 0,
    ) -> RouterRuleRow:
        """Append a rule (position = max + 1) and commit."""
        rule_id = uuid.uuid4().hex
        await self.db.execute(
            "INSERT INTO router_rules(id, position, pattern, target_model_id, enabled, "
            " fallback, strip_thinking, min_max_tokens) "
            "SELECT ?, COALESCE(MAX(position), -1) + 1, ?, ?, ?, ?, ?, ? FROM router_rules",
            (
                rule_id,
                pattern,
                target_model_id,
                int(enabled),
                int(fallback),
                int(strip_thinking),
                min_max_tokens,
            ),
        )
        await self.db.commit()
        row = await self.get(rule_id)
        assert row is not None
        return row

    async def update(self, rule_id: str, **fields: object) -> RouterRuleRow | None:
        """Set the given fields (and bump ``updated_at``); None if unknown id."""
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"cannot update: {', '.join(sorted(unknown))}")
        sets = [f"{k} = ?" for k in fields]
        params: list[object] = [int(v) if k in _BOOL_COLS else v for k, v in fields.items()]  # type: ignore[call-overload]
        sets.append("updated_at = datetime('now')")
        params.append(rule_id)
        cur = await self.db.execute(
            f"UPDATE router_rules SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        await self.db.commit()
        if cur.rowcount == 0:
            return None
        return await self.get(rule_id)

    async def delete(self, rule_id: str) -> bool:
        cur = await self.db.execute("DELETE FROM router_rules WHERE id = ?", (rule_id,))
        await self.db.commit()
        return cur.rowcount > 0

    async def reorder(self, ids: Sequence[str]) -> None:
        """Rewrite positions 0..n-1 in ``ids`` order, in one transaction.

        ValueError unless ``ids`` is exactly the set of existing rule ids.
        """
        cur = await self.db.execute("SELECT id FROM router_rules")
        existing = {r[0] for r in await cur.fetchall()}
        if len(ids) != len(set(ids)) or set(ids) != existing:
            raise ValueError("ids must be exactly the set of existing rule ids")
        try:
            for pos, rule_id in enumerate(ids):
                await self.db.execute(
                    "UPDATE router_rules SET position = ? WHERE id = ?", (pos, rule_id)
                )
            await self.db.commit()
        except BaseException:
            await self.db.rollback()
            raise
