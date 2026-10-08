"""Repository for the `settings` kv table introduced by migration 0010.

Style note: mirrors the simple async wrapper pattern used in
`app/db/repos/tokens.py` — no caching, every method commits when it writes.
Values are stored as TEXT and callers parse/serialize as needed.
"""

import aiosqlite


class SettingsRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def get(self, key: str) -> str | None:
        cur = await self.db.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cur.fetchone()
        return row[0] if row else None

    async def get_many(self, keys: list[str]) -> dict[str, str]:
        """Return only keys that exist; absent keys are omitted from the result.

        Empty input short-circuits without hitting the DB.
        """
        if not keys:
            return {}
        placeholders = ",".join("?" * len(keys))
        cur = await self.db.execute(
            f"SELECT key, value FROM settings WHERE key IN ({placeholders})",
            tuple(keys),
        )
        return {k: v for k, v in await cur.fetchall()}

    async def set(self, key: str, value: str) -> None:
        """Upsert a single key. Commits on success."""
        await self.db.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await self.db.commit()
