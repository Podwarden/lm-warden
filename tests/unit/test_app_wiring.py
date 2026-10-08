from app.main import app


def test_all_expected_routes_registered():
    paths = {r.path for r in app.routes}

    expected_api = {
        "/api/setup/welcome",
        "/api/setup/gpus",
        "/api/setup/hf_token",
        "/api/setup/admin",
        "/api/models",
        "/api/models/{model_id}",
        "/api/models/{model_id}/pull",
        "/api/models/{model_id}/load",
        "/api/models/{model_id}/unload",
        "/api/models/{model_id}/logs/stream",
        "/api/tokens",
        "/api/router/rules",
        "/api/router/settings",
        "/api/router/stats",
        "/api/tokens/{token_id}",
        "/api/stats/models",
        "/api/stats/gpus",
        "/api/system/gpus",
        "/api/settings/runtime",
        "/api/version",
    }
    missing = expected_api - paths
    assert not missing, f"missing API routes: {missing}"

    expected_proxy = {"/v1/chat/completions", "/v1/completions", "/v1/models"}
    missing = expected_proxy - paths
    assert not missing, f"missing proxy routes: {missing}"


def test_no_duplicate_routes():
    seen = {}
    for r in app.routes:
        methods = tuple(sorted(getattr(r, "methods", set()) or set()))
        key = (r.path, methods)
        seen.setdefault(key, []).append(r)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not dupes, f"duplicate routes: {list(dupes.keys())}"
