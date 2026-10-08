"""HTTP routes for the engine-version dropdown (#177).

Single read-only endpoint: ``GET /api/templates/engine-versions?channel=...``.
Backs the try-stack panel's vLLM-version field with the published
``vllm/vllm-openai`` semver tags that actually resolve to an image.

The channel → image family map (``app/runtime/backends/vllm/images.py``) decides
resolvability: every CUDA channel → ``vllm/vllm-openai``; rocm/cpu/xpu/unknown
have no upstream tag scheme today and return an empty list (they already 400
on try-stack, so the dropdown simply offers nothing). Docker Hub's anonymous
tag listing is rate-limited hard, so the versions come from a 6h in-process
TTL cache keyed by image family — never a per-request fetch — and a Docker Hub
hiccup serves stale-or-empty rather than 500ing the page.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.auth.deps import require_jwt
from app.runtime.backends.vllm.versions import EngineVersionsCache, resolve_family

router = APIRouter(prefix="/api/templates", tags=["templates"])


class EngineVersionsResponse(BaseModel):
    """Envelope for ``GET /api/templates/engine-versions``."""

    channel: str = Field(..., description="The channel the versions were resolved for.")
    family: str | None = Field(
        None,
        description=(
            "The Docker image family the channel maps to (e.g. "
            "'vllm/vllm-openai'), or null when the channel is not resolvable "
            "to an upstream image today (rocm/cpu/xpu/unknown)."
        ),
    )
    versions: list[str] = Field(
        default_factory=list,
        description=(
            "Published vLLM semver versions (newest first, no 'v' prefix) that "
            "resolve to a real image. Empty for non-resolvable channels or when "
            "Docker Hub is unavailable with no cached value."
        ),
    )
    error: str | None = Field(
        None,
        description=(
            "Set to a short sanitised code (e.g. 'docker_hub_unavailable') when "
            "the upstream tag fetch failed; versions then come from stale cache "
            "or are empty. The field never causes the request to fail."
        ),
    )


def _get_engine_versions_cache(request: Request) -> EngineVersionsCache:
    cache = getattr(request.app.state, "engine_versions_cache", None)
    if cache is None:
        cache = EngineVersionsCache()
        request.app.state.engine_versions_cache = cache
    return cache


@router.get("/engine-versions", response_model=EngineVersionsResponse)
async def engine_versions(
    request: Request,
    channel: str = "cuda-stable",
    _user: str = Depends(require_jwt),
) -> EngineVersionsResponse:
    """Return the published vLLM versions that resolve to an image for ``channel``.

    Non-resolvable channels short-circuit to an empty list (200) — no Docker
    Hub call. Resolvable channels read from the 6h family-keyed TTL cache.
    """
    family = resolve_family(channel)
    if family is None:
        return EngineVersionsResponse(channel=channel, family=None, versions=[], error=None)
    cache = _get_engine_versions_cache(request)
    versions, error = await cache.get(family)
    return EngineVersionsResponse(channel=channel, family=family, versions=versions, error=error)
