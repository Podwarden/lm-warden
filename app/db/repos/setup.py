import json
from dataclasses import dataclass

import aiosqlite


@dataclass
class SetupState:
    step: str
    draft: dict


class SetupRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def get(self) -> SetupState:
        cur = await self.db.execute("SELECT step, draft FROM setup_state WHERE id = 1")
        r = await cur.fetchone()
        return SetupState(step=r[0], draft=json.loads(r[1]))

    async def set_step(self, step: str) -> None:
        await self.db.execute(
            "UPDATE setup_state SET step = ?, updated_at = datetime('now') WHERE id = 1",
            (step,),
        )
        await self.db.commit()

    async def merge_draft(self, **kwargs) -> dict:
        cur = await self.db.execute("SELECT draft FROM setup_state WHERE id = 1")
        (draft_json,) = await cur.fetchone()
        draft = json.loads(draft_json)
        draft.update(kwargs)
        await self.db.execute(
            "UPDATE setup_state SET draft = ?, updated_at = datetime('now') WHERE id = 1",
            (json.dumps(draft),),
        )
        await self.db.commit()
        return draft

    async def is_done(self) -> bool:
        s = await self.get()
        return s.step == "done"
