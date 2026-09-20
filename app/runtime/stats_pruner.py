import asyncio
import logging
import time

from app.db.database import open_db
from app.db.repos import admin_audit
from app.stats import request_history

logger = logging.getLogger(__name__)
RETENTION_MINUTES = 7 * 24 * 60
PRUNE_INTERVAL_SECONDS = 3600


async def prune_once(settings) -> dict[str, int]:
    cutoff_minute = int(time.time() // 60) - RETENTION_MINUTES
    # Per-request history keeps its own, longer, configurable retention (30
    # days by default, plus a row cap) -- see Settings.request_history_*.
    history_days = int(getattr(settings, "request_history_retention_days", 30))
    history_max_rows = int(getattr(settings, "request_history_max_rows", 200_000))
    history_cutoff = time.time() - history_days * 86400.0
    async with open_db(settings.db_path) as db:
        cur1 = await db.execute("DELETE FROM model_samples WHERE minute < ?", (cutoff_minute,))
        cur2 = await db.execute("DELETE FROM gpu_samples WHERE minute < ?", (cutoff_minute,))
        # S7 (#124) — power_samples share retention with gpu_samples.
        cur3 = await db.execute("DELETE FROM power_samples WHERE minute < ?", (cutoff_minute,))
        history = await request_history.prune(
            db, cutoff=history_cutoff, max_rows=history_max_rows
        )
        # Admin-token audit (0037): 90 days, at most 200k rows total, and (#257)
        # at most 20k rows per token so one noisy/leaked token can only evict
        # its own history. Read through the module so a test can lower them.
        audit = await admin_audit.prune(
            db,
            cutoff=time.time() - admin_audit.RETENTION_DAYS * 86400.0,
            max_rows=admin_audit.MAX_ROWS,
            per_token_max_rows=admin_audit.PER_TOKEN_MAX_ROWS,
        )
        await db.commit()
        return {
            "model_samples": cur1.rowcount or 0,
            "gpu_samples": cur2.rowcount or 0,
            "power_samples": cur3.rowcount or 0,
            "request_history": history["by_age"] + history["by_count"],
            "admin_audit": audit,
        }


async def run_pruner_forever(settings) -> None:
    while True:
        try:
            await prune_once(settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("stats pruner iteration failed; continuing")
        await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
