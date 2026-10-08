"""Every route the package contract names must exist on this backend.

The static half of the conformance story: the shipped OpenAPI slice
(`dist/contract/openapi.chat.json`) is the route map, so a path or method
that has drifted out of `app/` shows up here in the ordinary unit run —
long before the vitest kit gets a chance to fail against a live server.
Skips when the npm package is not installed (`npm ci` in `frontend/`).
"""

import json
from pathlib import Path

import pytest

# An OpenAPI path item also carries non-operation keys -- `parameters`,
# `summary`, `servers`, `$ref`. Comparing those against our route table
# would invent a missing "PARAMETERS /api/chat2/chats" the moment the
# package hoists a shared parameter.
HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options", "trace"})

SLICE = (
    Path(__file__).resolve().parents[2]
    / "frontend/node_modules/@podwarden/chat-ui/dist/contract/openapi.chat.json"
)


@pytest.mark.skipif(
    not SLICE.exists(), reason="@podwarden/chat-ui not installed (run npm ci in frontend/)"
)
def test_backend_serves_every_contract_route(tmp_data_dir: Path) -> None:
    from app.main import build_app

    contract = json.loads(SLICE.read_text())
    ours = build_app().openapi()["paths"]
    missing = [
        f"{method.upper()} {path}"
        for path, ops in contract["paths"].items()
        for method in sorted(HTTP_METHODS & ops.keys())
        if path not in ours or method not in ours[path]
    ]
    assert missing == []
