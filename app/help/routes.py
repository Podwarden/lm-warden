"""GET/HEAD /agent-guide.md: the control API written for a coding agent.

Core serves it in every home-page mode (website on or off), so the prompts
that say "read <warden>/agent-guide.md" work on any install. The text is
app/help/agent-guide.md, inside the package so every packaging (Docker, the
macOS bundle that ships only `app`) carries it; documents/AGENT-GUIDE.md is an
identical copy for readers on GitHub. It is read once at import, so a broken
package fails at boot, not on a visitor's request. The route is public and
mints no cookie (app/auth/policy.py, app/auth/csrf.py).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, Response

router = APIRouter()

GUIDE_PATH = Path(__file__).resolve().parent / "agent-guide.md"
_GUIDE: str = GUIDE_PATH.read_text(encoding="utf-8")


@router.api_route("/agent-guide.md", methods=["GET", "HEAD"], include_in_schema=False)
async def agent_guide() -> Response:
    return PlainTextResponse(
        _GUIDE,
        media_type="text/markdown; charset=utf-8",
        headers={"X-Robots-Tag": "noindex"},
    )
