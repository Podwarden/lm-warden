"""Verify that a vLLM subprocess is actually serving before flipping
DB status to ``loaded``.

vLLM's ``/health`` returns 200 once the engine reports up, which can
happen BEFORE multimodal warmup (``_warmup_mm_processor``) completes.
A unload arriving in that window SIGTERMs an actively-warming
subprocess and aborts the load. Sending a cheap completion request
forces the engine to actually serve, closing the race window.

Spec: docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    detail: str | None


async def warmup_probe(
    *, port: int, served_model_name: str, timeout_s: float, host: str = "127.0.0.1"
) -> ProbeResult:
    """Send one POST /v1/completions to host:port and report success.

    ``host`` is loopback for the in-container subprocess driver and the
    engine container's DNS name for the docker driver (the supervisor
    supplies it via ``get_host``).

    Success = HTTP 200 with a non-empty ``choices`` array in the response
    body. Any other outcome (non-2xx, timeout, malformed body, network
    error) returns ``ok=False`` with a short ``detail`` string suitable
    for ``models.last_error``.

    Does NOT kill the subprocess on failure — the caller decides cleanup.
    """
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": served_model_name,
        "prompt": " ",
        "max_tokens": 1,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, json=payload)
    except httpx.ReadTimeout:
        return ProbeResult(False, "warmup probe timeout")
    except httpx.ConnectError as e:
        return ProbeResult(False, f"warmup probe connect error: {e}")
    except Exception as e:  # noqa: BLE001 — best-effort classification
        return ProbeResult(False, f"warmup probe error: {e!r}")

    if r.status_code != 200:
        return ProbeResult(
            False,
            f"warmup probe HTTP {r.status_code}: {r.text[:200]}",
        )
    try:
        body = r.json()
    except Exception as e:  # noqa: BLE001
        return ProbeResult(False, f"warmup probe non-JSON body: {e!r}")
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices:
        return ProbeResult(False, "warmup probe response missing 'choices' array")
    return ProbeResult(True, None)
