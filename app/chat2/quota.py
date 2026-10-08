from __future__ import annotations

import shutil
from typing import Any


class QuotaExceeded(Exception):
    pass


async def check_quota(repo: Any, settings: Any, user_id: int, incoming_bytes: int) -> None:
    used = await repo.user_bytes(user_id)
    if used + incoming_bytes > settings.chat_quota_user_bytes:
        raise QuotaExceeded(
            f"attachment quota exceeded ({used + incoming_bytes} > "
            f"{settings.chat_quota_user_bytes} bytes)"
        )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(settings.data_dir).free
    if free - incoming_bytes < settings.chat_quota_free_floor_bytes:
        raise QuotaExceeded("storage free-space floor reached; quota refused")
