from dataclasses import dataclass

import aiosqlite


@dataclass
class RuntimeRow:
    model_id: str
    pid: int | None
    port: int | None
    started_at: str | None
    health_ok: bool
    last_health_at: str | None


class RuntimeRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def upsert(
        self, model_id: str, pid: int | None, port: int | None, started_at: str | None
    ) -> None:
        await self.db.execute(
            "INSERT INTO model_runtime(model_id, pid, port, started_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(model_id) DO UPDATE SET pid=excluded.pid, "
            "port=excluded.port, started_at=excluded.started_at",
            (model_id, pid, port, started_at),
        )
        await self.db.commit()

    async def update_health(self, model_id: str, ok: bool, when: str) -> None:
        await self.db.execute(
            "UPDATE model_runtime SET health_ok = ?, last_health_at = ? WHERE model_id = ?",
            (1 if ok else 0, when, model_id),
        )
        await self.db.commit()

    async def get(self, model_id: str) -> RuntimeRow | None:
        cur = await self.db.execute(
            "SELECT model_id, pid, port, started_at, health_ok, last_health_at "
            "FROM model_runtime WHERE model_id = ?",
            (model_id,),
        )
        r = await cur.fetchone()
        return RuntimeRow(r[0], r[1], r[2], r[3], bool(r[4]), r[5]) if r else None

    async def clear(self, model_id: str) -> None:
        await self.db.execute("DELETE FROM model_runtime WHERE model_id = ?", (model_id,))
        await self.db.commit()

    async def clear_all(self) -> None:
        """Called at startup so no stale runtime rows survive."""
        await self.db.execute("DELETE FROM model_runtime")
        await self.db.commit()
