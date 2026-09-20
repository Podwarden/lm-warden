"""The API description is not public (spec 2026-09-19, decision 11):
FastAPI's /docs, /redoc and /openapi.json are gone; GET /api/openapi.json
serves the spec to a session or an admin token. The spec declares the bearer
scheme globally and exempts the public routes and /v1."""

import pytest

from app.main import app as module_app
from tests.conftest import jwt_login, seed_admin_token, seed_admin_user


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"])
def test_fastapis_own_docs_are_gone(client, path):
    assert client.get(path).status_code == 404


def test_the_spec_needs_credentials(client, tmp_data_dir):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    assert client.get("/api/openapi.json").status_code == 401
    _, secret = seed_admin_token(db_path)
    for headers in (jwt_login(client), {"Authorization": f"Bearer {secret}"}):
        r = client.get("/api/openapi.json", headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["openapi"].startswith("3.")
        assert "/api/admin-tokens" in r.json()["paths"]


def test_the_spec_declares_the_bearer_scheme_globally():
    spec = module_app.openapi()
    assert spec["components"]["securitySchemes"]["bearerAuth"]["type"] == "http"
    assert spec["components"]["securitySchemes"]["bearerAuth"]["scheme"] == "bearer"
    assert spec["security"] == [{"bearerAuth": []}]
    # A credentialed route inherits the global requirement.
    assert "security" not in spec["paths"]["/api/models"]["get"]


@pytest.mark.parametrize(
    ("path", "method"),
    [("/v1/models", "get"), ("/v1/chat/completions", "post"), ("/healthz", "get"),
     ("/api/setup/state", "get"), ("/api/setup/admin", "post"), ("/api/auth/login", "post"),
     ("/api/auth/refresh", "post"), ("/api/csrf", "get"),
     ("/api/chat2/attachments/{attachment_id}", "get")],
)
def test_public_routes_and_v1_opt_out(path, method):
    assert module_app.openapi()["paths"][path][method]["security"] == []


def test_the_spec_types_the_admin_token_and_model_responses():
    schemas = module_app.openapi()["components"]["schemas"]
    for name in ("AdminToken", "AdminTokenIssued", "AdminTokenList", "AdminAuditRow",
                 "AdminAuditPage", "ModelOut", "ModelList", "TokenListPage"):
        assert name in schemas, name
    ok = module_app.openapi()["paths"]["/api/models"]["get"]["responses"]["200"]
    assert ok["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/ModelList"}


def test_the_spec_is_built_once_and_hides_its_own_route():
    first = module_app.openapi()
    assert module_app.openapi() is first
    assert list(first["components"]["securitySchemes"]) == ["bearerAuth"]
    assert "/api/openapi.json" not in first["paths"]
