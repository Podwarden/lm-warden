"""Proxy 5xx → ``models.last_error`` hint enrichment.

Slice S1 of the 2026-05 overhaul removed the benchmark-v2 subsystem
that previously fed this layer with a structured envelope hint. Until
the S3/S4 "Suggest values" preset button lands and reintroduces a
proper hint source, this module is a slim stub: on upstream 5xx
responses, attach the model's last_error message (set by the load
runner when a vLLM subprocess fails its warmup probe) to the JSON
error envelope so callers see *some* recent operator-facing detail.

The signature of :func:`enrich_5xx_from_db` is preserved so
``app/proxy/routes.py`` doesn't need to be refactored — extra args
that no longer matter are accepted and silently ignored.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.proxy.model_cache import ModelCache, model_by_id

logger = logging.getLogger(__name__)


def _is_error_envelope(parsed: Any) -> bool:
    """True if ``parsed`` looks like an OpenAI-style ``{"error": {...}}``."""
    return isinstance(parsed, dict) and isinstance(parsed.get("error"), dict)


async def enrich_5xx_from_db(
    *,
    db_path,
    model_id: str,
    status_code: int,
    body_bytes: bytes,
    asked: dict | None = None,
    gpu_set: str | None = None,
    model_cache: ModelCache | None = None,
) -> bytes | None:
    """Return enriched body bytes, or ``None`` if no enrichment was done.

    The proxy hot path catches any exception we raise, so this helper
    can be defensive: return ``None`` whenever the inputs don't match
    the narrow happy path (5xx + JSON envelope shape + last_error set).
    """
    if status_code < 500:
        return None
    try:
        parsed = json.loads(body_bytes)
    except (ValueError, TypeError):
        return None
    if not _is_error_envelope(parsed):
        return None

    try:
        # The cached models table (#293); a DB read only on a cold cache.
        row = await model_by_id(model_cache, db_path, model_id)
    except Exception:  # noqa: BLE001
        logger.exception("envelope_hint: DB lookup failed for %s", model_id)
        return None
    last_error = getattr(row, "last_error", None) if row else None
    if not last_error:
        return None

    err_obj = parsed["error"]
    # Don't clobber a pre-existing ``hint`` field — keep the original
    # message and route ours into ``last_error_hint`` instead.
    target_key = "last_error_hint" if "hint" in err_obj else "hint"
    err_obj[target_key] = {"model_id": model_id, "last_error": last_error}

    try:
        return json.dumps(parsed).encode("utf-8")
    except (TypeError, ValueError):
        logger.exception("envelope_hint: re-encode failed")
        return None
