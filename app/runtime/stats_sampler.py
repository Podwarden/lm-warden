"""Background GPU stats sampler.

S7 (#124) reworks the cadence to 5s per tick (was 60s) so the per-minute
``power_samples`` bucket gets ~12 averaging samples per minute — that's
what produces a representative minute-average for the new ``/api/stats/v2``
chart instead of "whichever 5s tick happened to land last".

CTO decision #6 — every tick performs a SINGLE ``nvidia-smi`` acquisition
that returns util, mem, and power.draw in one CSV row per GPU; the sampler
then writes:

  * ``gpu_samples`` (util/mem/name) — UPSERT last-write-wins inside the
    minute bucket (the existing semantics; multiple 5s writes per minute
    just overwrite each other with the freshest reading)
  * ``power_samples`` (watts_sum/samples) — UPSERT accumulator (every 5s
    write adds one to ``samples`` and watts to ``watts_sum``, so the read
    side recovers a true average for the minute)

Cards that report ``[Not Supported]`` for power.draw produce
``power_w=None`` rows, and the sampler skips the power write for those —
util/mem still flow normally. The probe is bounded by the existing 5s
nvidia-smi timeout (app/system/gpu.py::_run_nvidia_smi); if the entire
probe times out the sampler swallows the failure and continues to the
next tick (same shape as before S7 — degraded but not fatal).
"""

import asyncio
import logging
import os
import time

from app.db.database import open_db
from app.db.repos.samples import SamplesRepo
from app.system.gpu import query_gpus

logger = logging.getLogger(__name__)

# S7 (#124) — 5s tick. Was 60s before this slice; rationale in module
# docstring. Operators can override via env for low-resource environments
# (e.g. CI smoke fixtures) without rebuilding the image.
DEFAULT_SAMPLE_INTERVAL_SECONDS = 5.0


def _interval_seconds() -> float:
    """Tick cadence (seconds). Defaults to ``DEFAULT_SAMPLE_INTERVAL_SECONDS``.
    Override via ``VW_STATS_SAMPLER_INTERVAL_S``. Values <= 0 fall back to
    the default so a misconfigured env can't pin a CPU."""
    raw = os.environ.get("VW_STATS_SAMPLER_INTERVAL_S")
    if not raw:
        return DEFAULT_SAMPLE_INTERVAL_SECONDS
    try:
        v = float(raw)
    except ValueError:
        return DEFAULT_SAMPLE_INTERVAL_SECONDS
    return v if v > 0 else DEFAULT_SAMPLE_INTERVAL_SECONDS


# Back-compat re-export: ``SAMPLE_INTERVAL_SECONDS`` was a module-level
# constant in S6; kept around so any external monkey-patch (tests, docs)
# doesn't silently break. Use ``_interval_seconds()`` for new code.
SAMPLE_INTERVAL_SECONDS = DEFAULT_SAMPLE_INTERVAL_SECONDS


async def sample_gpus_once(settings, *, now: float | None = None) -> int:
    """Probe GPUs once and write util/mem + power into the per-minute tables.

    Returns the number of GPUs sampled. Swallows nvidia-smi failures
    (logs and returns 0) so a transient probe error never tears down the
    long-running sampler task. Single ``nvidia-smi`` acquisition per call
    per CTO decision #6.
    """
    minute = int((now if now is not None else time.time()) // 60)
    try:
        gpus = await query_gpus()
    except Exception:
        logger.exception("query_gpus failed in stats sampler; skipping minute %d", minute)
        return 0
    async with open_db(settings.db_path) as db:
        repo = SamplesRepo(db)
        for g in gpus:
            await repo.add_gpu_sample(
                gpu_index=g.index,
                minute=minute,
                utilization_pct=g.utilization_pct,
                memory_used_mib=g.memory_used_mib,
                memory_total_mib=g.memory_total_mib,
                name=g.name,
            )
            # S7 (#124) — power.draw write only when the driver reported a
            # number. ``[Not Supported]`` / ``[N/A]`` cards land here with
            # ``power_w=None``; we skip rather than corrupt the bucket with
            # a zero (a 0.0 W average would be misleading on the chart).
            if g.power_w is not None:
                await repo.add_power_sample(
                    gpu_idx=g.index,
                    minute=minute,
                    watts=g.power_w,
                )
    return len(gpus)


async def run_sampler_forever(settings) -> None:
    """Long-lived background task. Cancellation-safe."""
    interval = _interval_seconds()
    while True:
        try:
            await sample_gpus_once(settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("stats sampler iteration failed; continuing")
        await asyncio.sleep(interval)
