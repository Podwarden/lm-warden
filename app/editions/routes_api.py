"""GET /api/edition: which edition this build is and what it carries."""

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.auth.deps import require_jwt

router = APIRouter()


@router.get("/api/edition")
async def get_edition(request: Request, _user: str = Depends(require_jwt)) -> dict[str, Any]:
    reg = request.app.state.ext
    return {
        "edition": reg.edition,
        "features": sorted(reg.features),
        "load_errors": list(reg.load_errors),
        "console_links": [{"label": c.label, "href": c.href} for c in reg.console_links],
    }
