"""/v1's require_bearer parses Authorization with the control API's parser
(app/auth/bearer.py::parse_bearer_header): case-insensitive scheme, any
whitespace between scheme and credential."""

import pytest

from tests.conftest import csrf_header, jwt_login


@pytest.fixture
def inference_key(client, seeded_db) -> str:
    session = {**jwt_login(client), **csrf_header(client)}
    r = client.post("/api/tokens", json={"name": "k"}, headers=session)
    assert r.status_code == 201, r.text
    return str(r.json()["plaintext"])


@pytest.mark.parametrize("template", ["Bearer {}", "bearer {}", "Bearer\t{}", "Bearer   {}"])
def test_v1_accepts_the_bearer_forms_the_parser_accepts(client, inference_key, template):
    r = client.get("/v1/models", headers={"Authorization": template.format(inference_key)})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("value", ["", "Bearer", "Bearer ", "Token vw_x", "vw_x"])
def test_v1_refuses_a_header_without_a_bearer_credential(client, inference_key, value):
    r = client.get("/v1/models", headers={"Authorization": value})
    assert (r.status_code, r.json()["detail"]) == (401, "missing bearer token")
