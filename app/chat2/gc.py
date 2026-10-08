"""Orphan collector: reconciles attachment rows against files on disk, expires
stale drafts, evicts attachment groups past the TTL, and evicts the oldest
groups when a user is near their storage quota. See spec §3.

Follows the ``app.runtime.stats_pruner`` pattern: a pure ``collect_once()``
for tests plus a ``run_gc_forever()`` loop that swallows and logs any
iteration failure so the background task never dies.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.chat2 import storage
from app.chat2.clock import now_iso
from app.chat2.limits import ALLOWED_IMAGE_MIMES, DRAFT_TTL_DAYS, GC_INTERVAL_S
from app.db.database import open_db

logger = logging.getLogger(__name__)


@dataclass
class GcReport:
    files_without_rows: int = 0
    rows_without_files: int = 0
    stale_drafts: int = 0
    evicted_ttl: int = 0
    evicted_lru: int = 0


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def _ext(mime: str, sha256: str) -> str | None:
    """Map a stored row's mime to its on-disk extension, or None + a log line.

    A row whose mime is not in `ALLOWED_IMAGE_MIMES` can only come from a
    narrowing of that whitelist (or a hand-edited DB row), and there is no
    path from it to a file name. `ALLOWED_IMAGE_MIMES[mime]` would raise a
    KeyError that aborts the WHOLE collector pass — every later row, every
    later user — over one bad row. Skip it loudly instead so the pass finishes.
    """
    ext = ALLOWED_IMAGE_MIMES.get(mime)
    if ext is None:
        logger.warning("chat2 gc: skipping attachment %s with unknown mime %r", sha256, mime)
    return ext


async def _evict_group(
    db: Any, data_dir: Path, user_id: int, sha256: str, mime: str, now: str
) -> bool:
    ext = _ext(mime, sha256)
    if ext is None:
        return False
    storage.unlink_quiet(storage.file_path(data_dir, user_id, sha256, ext))
    await db.execute(
        "UPDATE attachments SET evicted_at = ? WHERE user_id = ? AND sha256 = ? AND evicted_at IS NULL",
        (now, user_id, sha256),
    )
    return True


async def collect_once(settings: Any, *, now: str | None = None) -> GcReport:
    now = now or now_iso()
    now_dt = _parse(now)
    rep = GcReport()
    root = settings.data_dir / "chat2"
    async with open_db(settings.db_path) as db:
        cur = await db.execute(
            "SELECT id, user_id, message_id, mime, sha256, created_at, evicted_at FROM attachments"
        )
        rows = await cur.fetchall()
        live_keys = {(r[1], r[4]) for r in rows if r[6] is None}
        # 1. files with no live row (or stray .tmp leftovers from a crashed write)
        if root.is_dir():
            for user_dir in root.iterdir():
                if not user_dir.is_dir() or not user_dir.name.isdigit():
                    continue
                for f in user_dir.iterdir():
                    if f.suffix == ".tmp" or (int(user_dir.name), f.stem) not in live_keys:
                        if storage.unlink_quiet(f):
                            rep.files_without_rows += 1
        # 2. live rows whose file is missing
        for rid, uid, _mid, mime, sha, _created, ev in rows:
            ext = _ext(mime, sha) if ev is None else None
            if (
                ev is None
                and ext is not None
                and not storage.file_path(settings.data_dir, uid, sha, ext).exists()
            ):
                await db.execute("DELETE FROM attachments WHERE id = ?", (rid,))
                rep.rows_without_files += 1
        # 3. stale drafts (never linked to a message) past DRAFT_TTL_DAYS
        draft_cutoff = now_dt - timedelta(days=DRAFT_TTL_DAYS)
        for rid, uid, mid, mime, sha, created, ev in rows:
            if mid is None and ev is None and _parse(created) < draft_cutoff:
                await db.execute("DELETE FROM attachments WHERE id = ?", (rid,))
                rep.stale_drafts += 1
                cur = await db.execute(
                    "SELECT COUNT(*) FROM attachments WHERE user_id = ? AND sha256 = ?", (uid, sha)
                )
                row = await cur.fetchone()
                ext = _ext(mime, sha)
                if row is not None and row[0] == 0 and ext is not None:
                    storage.unlink_quiet(storage.file_path(settings.data_dir, uid, sha, ext))
        await db.commit()
        # 4. TTL eviction per (user, sha256): newest row older than the
        # authoritative settings TTL (defaults to limits.ATTACHMENT_TTL_DAYS
        # but the runtime setting always wins — see controller ruling).
        ttl_cutoff = now_dt - timedelta(days=settings.chat_attachment_ttl_days)
        cur = await db.execute(
            "SELECT user_id, sha256, MAX(created_at), MIN(mime) FROM attachments "
            "WHERE evicted_at IS NULL GROUP BY user_id, sha256"
        )
        groups = await cur.fetchall()
        for uid, sha, newest, mime in groups:
            if _parse(newest) < ttl_cutoff and await _evict_group(
                db, settings.data_dir, uid, sha, mime, now
            ):
                rep.evicted_ttl += 1
        await db.commit()
        # 5. LRU eviction: while a user is over 90% of quota, evict their
        # oldest (user_id, sha256) groups until back under 80%. A group that
        # still holds a LIVE DRAFT (message_id IS NULL, younger than
        # DRAFT_TTL_DAYS) is never evicted — spec §3: "Eviction never touches
        # drafts younger than the draft TTL". The user is still composing that
        # message; evicting it would blank the composer thumbnail and make the
        # turn fail with `attachment_missing`. Its bytes still count towards
        # `used`, so a user whose quota is entirely fresh drafts simply stops
        # freeing space here (the upload quota check is what pushes back).
        cur = await db.execute(
            "SELECT user_id, sha256, MIN(mime), MAX(size_bytes), "
            "MAX(CASE WHEN message_id IS NULL THEN created_at END) FROM attachments "
            "WHERE evicted_at IS NULL GROUP BY user_id, sha256 ORDER BY user_id, MAX(created_at)"
        )
        per_user: dict[int, list[tuple[str, str, int, str | None]]] = {}
        for uid, sha, mime, size, newest_draft in await cur.fetchall():
            per_user.setdefault(uid, []).append((sha, mime, int(size), newest_draft))
        quota = settings.chat_quota_user_bytes
        for uid, items in per_user.items():
            used = sum(i[2] for i in items)
            if used <= 0.9 * quota:
                continue
            for sha, mime, size, newest_draft in items:  # oldest group first
                if used < 0.8 * quota:
                    break
                if newest_draft is not None and _parse(newest_draft) >= draft_cutoff:
                    continue  # live draft — skip, its bytes stay counted
                if not await _evict_group(db, settings.data_dir, uid, sha, mime, now):
                    continue
                used -= size
                rep.evicted_lru += 1
        await db.commit()
    return rep


async def run_gc_forever(settings: Any) -> None:
    while True:
        try:
            rep = await collect_once(settings)
            logger.info("chat2 gc: %s", rep)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("chat2 gc iteration failed; continuing")
        await asyncio.sleep(GC_INTERVAL_S)
