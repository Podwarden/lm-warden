import aiosqlite

# The two conflict targets: the composite PRIMARY KEY for a keyed row, and the
# partial unique index from 0047 for the NULL token, which the PRIMARY KEY does
# not cover (SQLite treats NULLs as distinct in a composite key, #298).
_KEYED_TARGET = "(model_id, token_id)"
_NULL_TOKEN_TARGET = "(model_id) WHERE token_id IS NULL"


class CountersRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def increment(
        self,
        model_id: str,
        token_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        # One UPSERT, so two concurrent first requests cannot both INSERT: the
        # old SELECT-then-INSERT could, and made a duplicate NULL-token row.
        target = _NULL_TOKEN_TARGET if token_id is None else _KEYED_TARGET
        await self.db.execute(
            "INSERT INTO counters(model_id, token_id, requests, prompt_tokens, completion_tokens) "
            f"VALUES (?, ?, 1, ?, ?) ON CONFLICT{target} DO UPDATE SET "
            "requests = requests + 1, "
            "prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
            "completion_tokens = completion_tokens + excluded.completion_tokens",
            (model_id, token_id, prompt_tokens, completion_tokens),
        )
        await self.db.commit()
