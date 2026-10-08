"""A learned model of how this box prefills and caches, built OFF the request path.

Everything here is derived from finished rows in ``request_history`` by a
background task (``run_forever``, ~60 s). The proxy only does dictionary reads
of the result (``PrefillModelState``), so nothing here can slow or fail a
request, and every read is fail-open: no data means "fall back to the hint".

Four things are learned per model:

* prefill rate  -- median of (prompt - cached) / ttft over streamed requests
  that prefilled ALONE (nothing else admitted on the same replica at start),
  with a measured cache hit and >= MIN_FRESH_TOKENS of real work. Contended or
  tiny requests measure the queue or the fixed overhead, not the engine.
* estimate calibration -- how often the in-flight cache estimate was right
  (measured >= 0.8 x estimated).
* idle-gap decay -- hit ratio of the estimate by how long the session sat idle
  before the request. A gap bucket where the estimate is mostly wrong means the
  engine evicts idle sessions that fast; the live estimate is scaled down.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from app.db.database import open_db

logger = logging.getLogger(__name__)

WINDOW_S = 24 * 3600
MIN_SAMPLES = 20
MIN_FRESH_TOKENS = 512
REFRESH_S = 60.0
#: measured >= this fraction of the estimate counts as the estimate being right.
CALIBRATION_FRACTION = 0.8
#: a gap bucket needs this many rows before it may decay anything.
MIN_BUCKET_SAMPLES = 10
#: (label, lo_s inclusive, hi_s exclusive)
GAP_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-30s", 0.0, 30.0),
    ("30-120s", 30.0, 120.0),
    ("2-5m", 120.0, 300.0),
    ("5-15m", 300.0, 900.0),
)
#: below this hit ratio a bucket is considered an eviction zone.
DECAY_BELOW = 0.5


def _num(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def learned_rate(
    rows: Iterable[dict[str, Any]], *, min_samples: int = MIN_SAMPLES
) -> tuple[float | None, int]:
    """``(median tok/s, samples)``; the rate is None below ``min_samples``."""
    rates: list[float] = []
    for r in rows:
        ttft = _num(r.get("ttft_s"))
        prompt = _num(r.get("prompt_tokens"))
        cached = _num(r.get("cached_tokens"))
        if ttft is None or ttft <= 0 or prompt is None or cached is None:
            continue
        # Only a request that prefilled alone measures the engine.
        if r.get("inflight_same_rank_at_start") != 0:
            continue
        fresh = prompt - cached
        if fresh < MIN_FRESH_TOKENS:
            continue
        rates.append(fresh / ttft)
    if len(rates) < min_samples:
        return None, len(rates)
    return statistics.median(rates), len(rates)


#: Same floor as the cache-obs index: below this nothing was reusable.
COLD_MAX_REUSABLE = 256


def cold_rate(
    rows: Iterable[dict[str, Any]], *, min_samples: int = MIN_SAMPLES
) -> tuple[float | None, int]:
    """Prefill tok/s of requests with NOTHING reusable, for engines that do not
    report cached tokens: there, prompt / ttft is the engine's raw rate only
    when no prefix could have been served (spec section 3.2)."""
    rates: list[float] = []
    for r in rows:
        ttft = _num(r.get("ttft_s"))
        prompt = _num(r.get("prompt_tokens"))
        reusable = _num(r.get("reusable_tokens"))
        if r.get("cached_tokens") is not None or reusable is None or reusable >= COLD_MAX_REUSABLE:
            continue
        if ttft is None or ttft <= 0 or prompt is None or prompt < MIN_FRESH_TOKENS:
            continue
        if r.get("inflight_same_rank_at_start") != 0:
            continue
        rates.append(prompt / ttft)
    if len(rates) < min_samples:
        return None, len(rates)
    return statistics.median(rates), len(rates)


def calibration(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fraction of estimates that were right: measured >= 0.8 x estimated."""
    n = right = 0
    for r in rows:
        est = _num(r.get("cache_est_tokens"))
        cached = _num(r.get("cached_tokens"))
        if est is None or est <= 0 or cached is None:
            continue
        n += 1
        if cached >= CALIBRATION_FRACTION * est:
            right += 1
    return {"samples": n, "hit_ratio": (right / n) if n else None}


def bucket_of(gap_s: float) -> str | None:
    for label, lo, hi in GAP_BUCKETS:
        if lo <= gap_s < hi:
            return label
    return None


def gap_buckets(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per idle-gap bucket: how often at least half the estimate was real."""
    tally: dict[str, list[int]] = {label: [0, 0] for label, _, _ in GAP_BUCKETS}
    for r in rows:
        gap = _num(r.get("gap_s"))
        est = _num(r.get("cache_est_tokens"))
        cached = _num(r.get("cached_tokens"))
        if gap is None or est is None or est <= 0 or cached is None:
            continue
        label = bucket_of(gap)
        if label is None:
            continue
        tally[label][0] += 1
        if cached >= 0.5 * est:
            tally[label][1] += 1
    out = []
    for label, lo, hi in GAP_BUCKETS:
        n, good = tally[label]
        out.append(
            {
                "label": label,
                "from_s": lo,
                "to_s": hi,
                "samples": n,
                "hit_ratio": (good / n) if n else None,
            }
        )
    return out


@dataclass
class ModelStats:
    rate_tok_s: float | None = None
    rate_samples: int = 0
    calibration: dict[str, Any] = field(default_factory=lambda: {"samples": 0, "hit_ratio": None})
    gaps: list[dict[str, Any]] = field(default_factory=list)
    cold_rate_tok_s: float | None = None


def build(rows: Iterable[dict[str, Any]]) -> dict[str, ModelStats]:
    """Group finished rows by model row id and learn each model's stats."""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        mid = r.get("model_id")
        if mid:
            by_model.setdefault(str(mid), []).append(r)
    out: dict[str, ModelStats] = {}
    for mid, rs in by_model.items():
        rate, n = learned_rate(rs)
        cold, _ = cold_rate(rs)
        out[mid] = ModelStats(
            rate_tok_s=rate,
            rate_samples=n,
            calibration=calibration(rs),
            gaps=gap_buckets(rs),
            cold_rate_tok_s=cold,
        )
    return out


class PrefillModelState:
    """The latest learned stats, read by the proxy and the stats endpoints.

    Reads are plain dict lookups and never raise; ``replace`` swaps the whole
    mapping atomically, so a reader sees one consistent generation.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._models: dict[str, ModelStats] = {}
        self.updated_at: float | None = None

    def replace(self, models: dict[str, ModelStats]) -> None:
        self._models = models
        self.updated_at = self._clock()

    def get(self, model_id: str) -> ModelStats | None:
        return self._models.get(model_id)

    def rate_for(self, model_id: str, hint: float) -> tuple[float, str]:
        """``(tok/s, "learned" | "hint")``."""
        st = self._models.get(model_id)
        if st is not None and st.rate_tok_s:
            return st.rate_tok_s, "learned"
        return hint, "hint"

    def cold_rate_for(self, model_id: str) -> float | None:
        st = self._models.get(model_id)
        return st.cold_rate_tok_s if st is not None else None

    def accuracy_for(self, model_id: str) -> float | None:
        st = self._models.get(model_id)
        return st.calibration.get("hit_ratio") if st is not None else None

    def decay_for(self, model_id: str, gap_s: float | None) -> float | None:
        """Factor (< 1) to scale an estimate by when its idle gap falls in a
        bucket where the estimate is mostly wrong; None = leave it alone."""
        st = self._models.get(model_id)
        if st is None or gap_s is None:
            return None
        label = bucket_of(gap_s)
        for b in st.gaps:
            if b["label"] == label:
                ratio = b["hit_ratio"]
                if ratio is not None and b["samples"] >= MIN_BUCKET_SAMPLES and ratio < DECAY_BELOW:
                    return float(ratio)
        return None

    def snapshot(self, hint: float) -> dict[str, Any]:
        return {
            "updated_at": self.updated_at,
            "hint_tok_s": hint,
            "window_s": WINDOW_S,
            "min_samples": MIN_SAMPLES,
            "models": {
                mid: {
                    "rate_tok_s": st.rate_tok_s,
                    "rate_source": "learned" if st.rate_tok_s else "hint",
                    "samples": st.rate_samples,
                    "estimate_accuracy": st.calibration,
                    "gap_buckets": st.gaps,
                    "cold_rate_tok_s": st.cold_rate_tok_s,
                }
                for mid, st in self._models.items()
            },
        }


async def refresh(state: PrefillModelState, db_path: Any, *, now: float | None = None) -> int:
    """Recompute from the last 24 h of request_history. Returns rows read."""
    since = (now if now is not None else time.time()) - WINDOW_S
    async with open_db(db_path) as db:
        cur = await db.execute(
            "SELECT model_id, prompt_tokens, cached_tokens, cache_est_tokens, ttft_s, "
            "inflight_same_rank_at_start, gap_s, reusable_tokens FROM request_history "
            "WHERE finished_at >= ? ORDER BY finished_at DESC LIMIT 50000",
            (since,),
        )
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r, strict=True)) for r in await cur.fetchall()]
    state.replace(build(rows))
    return len(rows)


async def run_forever(state: PrefillModelState, db_path: Any, interval: float = REFRESH_S) -> None:
    """Background loop: log and keep going, never raise."""
    while True:
        try:
            await refresh(state, db_path)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- learning is best-effort
            logger.debug("prefill model refresh failed", exc_info=True)
        await asyncio.sleep(interval)
