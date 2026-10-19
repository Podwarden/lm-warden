"""Persistence for model load history (Task 7).

After a model loads successfully, the warden records how long each load step
took, so the next load of the same model with the same settings can show
"usually ~6m" bars and a ready-by time.

Rows are keyed on ``(model_id, fingerprint)``: the fingerprint is the identity
of what was loaded — backend, engine image/pin, repo, revision, filename,
dtype, quantization-relevant extra_args, GPU/parallel layout, context, and
``VLLM_`` env vars. A change to any of those starts a fresh history, so an
engine bump (e.g. vLLM 0.26 -> 0.32) does not pollute the "usual" estimate
with timings from the old engine.

The repo keeps only the newest 20 rows per model (not per fingerprint): a model
that is loaded often should not accumulate hundreds of rows, and 20 recent
loads is plenty for a 5-sample median. ``usual`` takes the median of the last
``n`` rows matching the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

import aiosqlite

from app.runtime.load_progress.parse import Step

if TYPE_CHECKING:  # pragma: no cover - type-only, never imported at runtime
    from app.db.repos.models import ModelRow

logger = logging.getLogger(__name__)

__all__ = ["LoadHistoryRepo", "load_fingerprint", "median"]

#: Keep the newest N rows per model so the table stays bounded.
_KEEP_NEWEST = 20

#: How many recent loads the "usual" median is computed from.
_USUAL_N = 5


def load_fingerprint(
    model: ModelRow,
    image: str | None,
    *,
    baked_engine_version: str | None = None,
) -> str:
    """Return the sha256 fingerprint of a model's load configuration.

    ``image`` is the engine image/pin the model will run with (from
    ``_resolve_engine_image``). When ``image`` is None the model runs the
    in-container engine, and ``baked_engine_version`` (the vLLM/llama.cpp
    version compiled into this warden image) is folded in instead — so an
    engine bump starts a fresh history.

    The fingerprint covers: backend, image (or baked engine version), hf_repo,
    hf_revision, filename, dtype, quantization-relevant extra_args,
    gpu_indices, tensor_parallel_size, data_parallel size, max_model_len,
    gpu_memory_utilization, and extra_env keys starting with ``VLLM_``.
    It deliberately ignores ``served_model_name`` (cosmetic), ``status``,
    pull progress, and non-``VLLM_`` env vars (which hold secrets such as
    ``HF_TOKEN``).
    """
    extra_env = getattr(model, "extra_env", None) or {}
    vllm_env = {k: v for k, v in extra_env.items() if k.startswith("VLLM_")}
    # Quantization-relevant extra_args: flags that change the weights format
    # (--quantization, --fp8, etc.). We include the full extra_args list
    # because any flag that changes how the engine loads weights changes the
    # load timing; secret-looking flags are rare in extra_args (they live in
    # extra_env) and the fingerprint is not stored alongside secrets.
    extra_args = getattr(model, "extra_args", None) or []
    payload = {
        "backend": getattr(model, "backend", None),
        "image": image,
        "baked_engine_version": baked_engine_version if image is None else None,
        "hf_repo": getattr(model, "hf_repo", None),
        "hf_revision": getattr(model, "hf_revision", None),
        "filename": getattr(model, "filename", None),
        "dtype": getattr(model, "dtype", None),
        # Order-sensitive: the argv builder passes these verbatim, so a
        # reordered list can change vLLM's behaviour (e.g. last-wins flags).
        "extra_args": list(extra_args),
        "gpu_indices": list(getattr(model, "gpu_indices", None) or []),
        "tensor_parallel_size": getattr(model, "tensor_parallel_size", None),
        "data_parallel_size": getattr(model, "data_parallel_size", None),
        "max_model_len": getattr(model, "max_model_len", None),
        "gpu_memory_utilization": getattr(model, "gpu_memory_utilization", None),
        "vllm_env": dict(sorted(vllm_env.items())),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LoadHistoryRepo:
    """Record and read model load timings."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def record(
        self,
        model_id: str,
        fingerprint: str,
        total_s: float,
        steps: dict[Step, float],
    ) -> None:
        """Record one successful load's timings.

        Keeps only the newest ``_KEEP_NEWEST`` rows per model. Never raises:
        a recording failure (e.g. a stale model_id) must not fail the load —
        the caller's ``on_loaded`` hook already wraps this in a try/except, but
        the repo is defensive too so a direct caller is safe.
        """
        steps_json = json.dumps(
            {step.value: dur for step, dur in steps.items()},
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            await self.db.execute(
                "INSERT INTO model_load_history"
                "(model_id, fingerprint, total_s, steps_json) "
                "VALUES (?, ?, ?, ?)",
                (model_id, fingerprint, total_s, steps_json),
            )
            await self.db.commit()
            await self._trim(model_id)
        except Exception:  # noqa: BLE001 — recording must never fail the load
            logger.warning(
                "load-history record failed for %s; load is unaffected",
                model_id,
                exc_info=True,
            )

    async def _trim(self, model_id: str) -> None:
        """Keep only the newest ``_KEEP_NEWEST`` rows for ``model_id``."""
        await self.db.execute(
            "DELETE FROM model_load_history "
            "WHERE model_id = ? AND id NOT IN ("
            "  SELECT id FROM model_load_history "
            "  WHERE model_id = ? "
            "  ORDER BY finished_at DESC, id DESC "
            "  LIMIT ?"
            ")",
            (model_id, model_id, _KEEP_NEWEST),
        )
        await self.db.commit()

    async def usual(
        self,
        model_id: str,
        fingerprint: str,
        n: int = _USUAL_N,
    ) -> tuple[float | None, dict[Step, float]]:
        """Return the median total and per-step durations from the last ``n``
        loads matching ``(model_id, fingerprint)``.

        Returns ``(None, {})`` when no history exists.
        """
        cur = await self.db.execute(
            "SELECT total_s, steps_json FROM model_load_history "
            "WHERE model_id = ? AND fingerprint = ? "
            "ORDER BY finished_at DESC, id DESC LIMIT ?",
            (model_id, fingerprint, n),
        )
        rows = await cur.fetchall()
        if not rows:
            return None, {}
        totals = [r[0] for r in rows]
        step_values: dict[Step, list[float]] = {}
        for _, steps_json in rows:
            try:
                raw = json.loads(steps_json)
            except (json.JSONDecodeError, TypeError):
                continue
            for key, val in raw.items():
                try:
                    step = Step(key)
                except ValueError:
                    continue
                step_values.setdefault(step, []).append(float(val))
        usual_total = median(totals)
        usual_steps = {step: median(vals) for step, vals in step_values.items() if vals}
        return usual_total, usual_steps


def median(values: list[float]) -> float:
    """Return the median of a non-empty list of floats."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0
