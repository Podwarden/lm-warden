from dataclasses import dataclass

import aiosqlite


@dataclass(frozen=True)
class UserRow:
    id: int
    username: str
    password_hash: str


class UserRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def create(self, username: str, password_hash: str) -> int:
        cur = await self.db.execute(
            "INSERT INTO users(username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        await self.db.commit()
        return cur.lastrowid

    async def get_by_username(self, username: str) -> UserRow | None:
        cur = await self.db.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        )
        row = await cur.fetchone()
        return UserRow(*row) if row else None

    async def count(self) -> int:
        cur = await self.db.execute("SELECT COUNT(*) FROM users")
        (n,) = await cur.fetchone()
        return n
