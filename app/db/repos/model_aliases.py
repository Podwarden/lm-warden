"""``model_aliases`` repository -- the ONLY writer of the table."""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import aiosqlite

from app.proxy.model_cache import invalidate_all

_COLS = "id, name, targets, guard, thinking_policy, params, enabled, created_at, updated_at"
_UPDATABLE = frozenset({"name", "targets", "guard", "thinking_policy", "params", "enabled"})


@dataclass
class AliasRow:
    id: str
    name: str
    targets: list[str] = field(default_factory=list)
    guard: str = "inherit"
    thinking_policy: str = "inherit"
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""


def _decode(r: aiosqlite.Row) -> AliasRow:
    targets = r[2]
    params = r[5]
    return AliasRow(
        id=r[0],
        name=r[1],
        targets=json.loads(targets) if isinstance(targets, str) else list(targets or []),
        guard=r[3] if r[3] is not None else "inherit",
        thinking_policy=r[4] if r[4] is not None else "inherit",
        params=json.loads(params) if isinstance(params, str) else dict(params or {}),
        enabled=bool(r[6]),
        created_at=r[7] or "",
        updated_at=r[8] or "",
    )


class ModelAliasRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def _commit(self) -> None:
        # Every write to `model_aliases` ends here. Dropping the /v1 model
        # cache AFTER the commit is what makes an alias or model change visible
        # to the very next request (app/proxy/model_cache.py).
        await self.db.commit()
        invalidate_all()

    async def list(self) -> list[AliasRow]:
        cur = await self.db.execute(f"SELECT {_COLS} FROM model_aliases ORDER BY created_at, name")
        return [_decode(r) for r in await cur.fetchall()]

    async def get(self, alias_id: str) -> AliasRow | None:
        cur = await self.db.execute(f"SELECT {_COLS} FROM model_aliases WHERE id = ?", (alias_id,))
        r = await cur.fetchone()
        return _decode(r) if r else None

    async def get_by_name(self, name: str) -> AliasRow | None:
        cur = await self.db.execute(f"SELECT {_COLS} FROM model_aliases WHERE name = ?", (name,))
        r = await cur.fetchone()
        return _decode(r) if r else None

    async def create(self, row: AliasRow) -> AliasRow:
        alias_id = row.id or uuid.uuid4().hex
        await self.db.execute(
            "INSERT INTO model_aliases(id, name, targets, guard, thinking_policy, params, enabled) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                alias_id,
                row.name,
                json.dumps(row.targets),
                row.guard,
                row.thinking_policy,
                json.dumps(row.params),
                int(row.enabled),
            ),
        )
        await self._commit()
        new_row = await self.get(alias_id)
        assert new_row is not None
        return new_row

    async def update(self, alias_id: str, changes: dict[str, Any]) -> AliasRow | None:
        """Set the given fields (and bump ``updated_at``); None if unknown id."""
        unknown = set(changes) - _UPDATABLE
        if unknown:
            raise ValueError(f"cannot update: {', '.join(sorted(unknown))}")
        if not changes:
            return await self.get(alias_id)
        sets = []
        params: list[Any] = []
        for key, value in changes.items():
            if key == "targets":
                sets.append("targets = ?")
                params.append(json.dumps(value))
            elif key == "params":
                sets.append("params = ?")
                params.append(json.dumps(value))
            elif key == "enabled":
                sets.append("enabled = ?")
                params.append(int(value))
            else:
                sets.append(f"{key} = ?")
                params.append(value)
        sets.append("updated_at = datetime('now')")
        params.append(alias_id)
        cur = await self.db.execute(
            f"UPDATE model_aliases SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        await self._commit()
        if cur.rowcount == 0:
            return None
        return await self.get(alias_id)

    async def delete(self, alias_id: str) -> bool:
        cur = await self.db.execute("DELETE FROM model_aliases WHERE id = ?", (alias_id,))
        await self._commit()
        return cur.rowcount > 0

    async def drop_target(self, model_id: str) -> int:
        """Remove ``model_id`` from every alias's target list. Returns the
        number of aliases touched."""
        changed = 0
        for row in await self.list():
            if model_id not in row.targets:
                continue
            new_targets = [t for t in row.targets if t != model_id]
            await self.db.execute(
                "UPDATE model_aliases SET targets = ?, updated_at = datetime('now') WHERE id = ?",
                (json.dumps(new_targets), row.id),
            )
            changed += 1
        if changed:
            await self._commit()
        return changed
