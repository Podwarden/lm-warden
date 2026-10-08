"""GET /api/version — build-time version + commit sha for the nav footer.

VW_BUILD_VERSION and VW_BUILD_SHA are baked into the container at image-build
time by the `.gitlab-ci.yml` build job and exposed as ENV in the final stage
of the Dockerfile. They never change at runtime, so we cache them at import
time and return the cached values on every request.

When the binary is run locally without those env vars (dev shell, pytest,
local docker without `--build-arg`), both fall back to literal "dev" /
"unknown" rather than empty strings — the nav footer renders them as
"v? — sha ?" via the frontend fallback in that case.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends

from app.auth.deps import require_jwt

# Read once at import. Cheap, but guards against any future code path that
# might monkeypatch os.environ mid-process and produce inconsistent answers.
_VERSION: str = os.environ.get("VW_BUILD_VERSION") or "dev"
_SHA: str = os.environ.get("VW_BUILD_SHA") or "unknown"

router = APIRouter()


@router.get("/api/version")
async def get_version(_user: str = Depends(require_jwt)) -> dict[str, str]:
    return {"version": _VERSION, "sha": _SHA}
