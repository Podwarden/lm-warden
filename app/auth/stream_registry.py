import asyncio
from collections import defaultdict


class StreamRegistry:
    def __init__(self) -> None:
        self._by_user: dict[str, set[asyncio.Task]] = defaultdict(set)

    def register(self, user_id: str, task: asyncio.Task) -> asyncio.Task:
        self._by_user[user_id].add(task)
        return task

    def unregister(self, user_id: str, task: asyncio.Task) -> None:
        bucket = self._by_user.get(user_id)
        if bucket:
            bucket.discard(task)
            if not bucket:
                self._by_user.pop(user_id, None)

    def cancel_user(self, user_id: str) -> int:
        tasks = list(self._by_user.get(user_id, ()))
        for t in tasks:
            t.cancel()
        self._by_user.pop(user_id, None)
        return len(tasks)

    def count(self, user_id: str) -> int:
        return len(self._by_user.get(user_id, ()))
