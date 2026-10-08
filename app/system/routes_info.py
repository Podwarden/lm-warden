"""GET /api/system/info — static system inventory (#148).

Hardware + OS + Docker context that helps an operator interpret the live
numbers on the /stats page. Probes shell out to ``nvidia-smi`` and
``docker info``; an in-process 60s cache (``SystemInfoCache``) absorbs the
30s-interval polling from the stats page so the typical request path is a
dict lookup.

See ``app/system/system_info.py`` for the per-source collector helpers and
the cache implementation; this module is purely the FastAPI wiring.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.auth.deps import require_jwt
from app.system.system_info import SystemInfoCache

router = APIRouter()


def _get_cache(request: Request) -> SystemInfoCache:
    """Lazy-install the cache on ``app.state``.

    Mirrors the pattern in routes_gpus.py — tests inject their own cache
    instance with a stub clock/collector before the first request.
    """
    cache = getattr(request.app.state, "system_info_cache", None)
    if cache is None:
        cache = SystemInfoCache()
        request.app.state.system_info_cache = cache
    return cache


@router.get("/api/system/info")
async def system_info(request: Request, _user: str = Depends(require_jwt)) -> dict[str, Any]:
    """Static system inventory: CPU, RAM, GPUs, OS, Docker.

    Response shape (locked for issue #148 frontend consumption)::

        {
          "cpu":  {"model": "...", "physical_cores": int, "threads": int},
          "ram":  {"total_mb": int},
          "gpus": [
            {
              "index": int,
              "name": "NVIDIA RTX A4000",
              "vram_total_mb": 16376,
              "driver_version": "550.54.15",
              "cuda_version": "12.4" | null
            }, ...
          ],
          "os":     {"name": str, "version": str, "kernel": str},
          "docker": {
            "version":   str | null,
            "runtime":   str | null,
            "available": bool       # false when docker info couldn't run
          }
        }

    When ``nvidia-smi`` is not on PATH (dev box without NVIDIA) the
    endpoint still returns HTTP 200 with ``gpus: []`` so the UI can
    render an empty state rather than crash.

    When ``docker info`` fails (e.g. the API container has no docker
    socket mounted, which is the common case in production) the docker
    slot reports ``available: false`` instead of erroring the whole
    endpoint.
    """
    return _get_cache(request).get()
