"""The daily update check (spec 2026-11-20).

Once a day the install asks lmwarden.com whether a newer release or a
security advisory exists. The request carries the version, the edition,
the platform and a random install ID made on first use, and nothing else.
The answer is stored as a settings row; the console reads it from there.

Off when ``VW_UPDATE_CHECK`` is 0/false/no/off, when ``HF_HUB_OFFLINE=1``
(an offline install), when the runtime setting ``update_check_enabled`` is
false, and always in the ``vendor`` edition (the update server itself).
A dev build (version not CalVer) never checks: it is not an install.

Every failure is swallowed and logged at debug level: an unreachable or
broken update server must never cost the install anything but the banner.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.db.database import open_db
from app.db.repos.settings import SettingsRepo
from app.updates.version import parse_calver

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://lmwarden.com/api/update-check"
INSTALL_ID_KEY = "update_install_id"
RESULT_KEY = "update_last_result"
CHECKED_KEY = "update_last_checked_at"
DISMISSED_KEY = "update_dismissed_versions"
ENABLED_KEY = "update_check_enabled"

TIMEOUT_S = 10.0
FIRST_DELAY_S = 120.0
INTERVAL_S = 24 * 3600.0
JITTER_S = 3600.0

_OFF = {"0", "false", "no", "off"}


def check_url() -> str:
    return os.environ.get("VW_UPDATE_CHECK_URL") or DEFAULT_URL


def env_enabled() -> bool:
    if os.environ.get("VW_UPDATE_CHECK", "").strip().lower() in _OFF:
        return False
    return os.environ.get("HF_HUB_OFFLINE", "").strip() != "1"


def platform_name() -> str:
    return "macos" if sys.platform == "darwin" else "linux"


async def enabled(db_path: Path, edition: str) -> bool:
    if edition == "vendor" or not env_enabled():
        return False
    async with open_db(db_path) as db:
        row = await SettingsRepo(db).get(ENABLED_KEY)
    return row is None or row.strip().lower() not in _OFF


async def get_or_create_install_id(db_path: Path) -> str:
    async with open_db(db_path) as db:
        repo = SettingsRepo(db)
        existing = await repo.get(INSTALL_ID_KEY)
        if existing:
            return existing
        new = str(uuid.uuid4())
        await repo.set(INSTALL_ID_KEY, new)
        return new


def _valid_answer(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    latest = body.get("latest")
    if latest is not None and not (
        isinstance(latest, dict) and isinstance(latest.get("version"), str)
    ):
        return False
    return isinstance(body.get("advisories", []), list)


async def check_once(
    db_path: Path,
    edition: str,
    version: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any] | None:
    """One check. Returns the stored answer, or None when nothing was stored."""
    if parse_calver(version) is None:
        return None
    try:
        install_id = await get_or_create_install_id(db_path)
        payload = {
            "install_id": install_id,
            "version": version,
            "edition": edition,
            "platform": platform_name(),
        }
        async with httpx.AsyncClient(timeout=TIMEOUT_S, transport=transport) as http:
            r = await http.post(
                check_url(), json=payload, headers={"User-Agent": f"lm-warden/{version}"}
            )
        r.raise_for_status()
        body: dict[str, Any] = r.json()
        if not _valid_answer(body):
            raise ValueError("unexpected answer shape")
    except Exception as exc:  # noqa: BLE001 -- a failed check costs only the banner
        logger.debug("update check failed: %s", exc)
        return None
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    async with open_db(db_path) as db:
        repo = SettingsRepo(db)
        await repo.set(RESULT_KEY, json.dumps(body))
        await repo.set(CHECKED_KEY, now)
    return body


async def run_forever(db_path: Path, edition: str, version: str) -> None:
    await asyncio.sleep(FIRST_DELAY_S)
    while True:
        try:
            if await enabled(db_path, edition):
                await check_once(db_path, edition, version)
        except Exception:  # noqa: BLE001 -- the loop must outlive any one pass
            logger.debug("update check pass failed", exc_info=True)
        await asyncio.sleep(INTERVAL_S + random.uniform(-JITTER_S, JITTER_S))
