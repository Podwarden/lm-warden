from fastapi.testclient import TestClient


def test_health_endpoint_returns_200(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    # The GPU probe's last-known state rides along (#255); liveness never
    # depends on it.
    assert body["gpu"]["state"] in ("unknown", "ok", "failing", "absent")
