import json
from dataclasses import dataclass

import aiosqlite


@dataclass
class StackAttemptRow:
    id: str
    model_id: str
    channel: str
    vllm_version: str
    image: str | None
    result: str  # 'pending' | 'ok' | 'failed'
    error: str | None
    category: str | None
    suggested_next: dict | None
    created_at: str | None = None


_COLS = (
    "id, model_id, channel, vllm_version, image, result, error, "
    "category, suggested_next, created_at"
)


def _decode(row: tuple) -> StackAttemptRow:
    return StackAttemptRow(
        id=row[0],
        model_id=row[1],
        channel=row[2],
        vllm_version=row[3],
        image=row[4],
        result=row[5],
        error=row[6],
        category=row[7],
        suggested_next=json.loads(row[8]) if row[8] else None,
        created_at=row[9],
    )


class StackAttemptRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def insert(self, row: StackAttemptRow) -> None:
        await self.db.execute(
            "INSERT INTO stack_attempts(id, model_id, channel, vllm_version, "
            "image, result, error, category, suggested_next) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                row.id,
                row.model_id,
                row.channel,
                row.vllm_version,
                row.image,
                row.result,
                row.error,
                row.category,
                json.dumps(row.suggested_next) if row.suggested_next else None,
            ),
        )
        await self.db.commit()

    async def get(self, attempt_id: str) -> StackAttemptRow | None:
        cur = await self.db.execute(
            f"SELECT {_COLS} FROM stack_attempts WHERE id = ?", (attempt_id,)
        )
        row = await cur.fetchone()
        return _decode(row) if row else None

    async def list_for_model(self, model_id: str) -> list[StackAttemptRow]:
        cur = await self.db.execute(
            f"SELECT {_COLS} FROM stack_attempts WHERE model_id = ? " "ORDER BY created_at",
            (model_id,),
        )
        return [_decode(r) for r in await cur.fetchall()]

    async def set_result(
        self,
        attempt_id: str,
        result: str,
        error: str | None,
        category: str | None,
        suggested_next: dict | None,
    ) -> None:
        await self.db.execute(
            "UPDATE stack_attempts SET result = ?, error = ?, category = ?, "
            "suggested_next = ? WHERE id = ?",
            (
                result,
                error,
                category,
                json.dumps(suggested_next) if suggested_next else None,
                attempt_id,
            ),
        )
        await self.db.commit()
