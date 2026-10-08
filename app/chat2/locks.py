from __future__ import annotations


class TurnLocks:
    """One turn in flight per chat (in-process; the app is single-replica)."""

    def __init__(self) -> None:
        self._held: set[str] = set()

    def try_acquire(self, chat_id: str) -> bool:
        if chat_id in self._held:
            return False
        self._held.add(chat_id)
        return True

    def release(self, chat_id: str) -> None:
        self._held.discard(chat_id)

    def held(self, chat_id: str) -> bool:
        return chat_id in self._held
