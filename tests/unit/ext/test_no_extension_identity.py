"""The extension machinery must leave the Community app untouched.

(a) Route-table identity: the mounted APIRoute set equals the committed
    baseline plus GET /api/edition. route_table_baseline.json is the sorted
    [method, path] list of build_app() at base commit 5b69b5ee (before the
    machinery), minus the 450 website routes that left core with the site
    split (Task 5: removed exactly the routes of the website package's router,
    checked by set difference when the file was regenerated). Regenerate
    when a later task intentionally adds routes:
        git worktree add --detach /tmp/base <commit-before-your-change>
        cd /tmp/base && python - <<'PY' > baseline.json
        import json; from app.main import build_app
        from fastapi.routing import APIRoute
        print(json.dumps(sorted([m, r.path] for r in build_app().routes
              if isinstance(r, APIRoute) for m in r.methods)))
        PY
    or simply add the new (method, path) pairs to EXPECTED_ADDED below.

(b) Behaviour identity: every parameterless, non-streaming GET answers the
    same with no entry points at all as with a real entry point whose
    register() does nothing, anonymously and with admin headers. It does NOT
    reach routes with path parameters or streaming routes (they are covered
    by the auth matrix / their own tests), nor POST/PATCH/DELETE routes.
    Status-only bodies (nondeterministic): see STATUS_ONLY.
"""

import json
import re
from pathlib import Path
from typing import Any

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.ext import Registry
from tests.conftest import FakeEntryPoint, jwt_login, seed_admin_user

BASELINE = Path(__file__).parent / "route_table_baseline.json"
EXPECTED_ADDED = {
    ("GET", "/api/edition"),
    # Update check (spec 2026-11-20).
    ("GET", "/api/system/update-status"),
    ("POST", "/api/system/update-status/dismiss"),
    # Model aliases admin API.
    ("GET", "/api/aliases"),
    ("POST", "/api/aliases"),
    ("GET", "/api/aliases/{alias_id}"),
    ("PATCH", "/api/aliases/{alias_id}"),
    ("DELETE", "/api/aliases/{alias_id}"),
    # Prices tab (#347).
    ("GET", "/api/admin/prices"),
    ("GET", "/api/admin/prices/models"),
    ("GET", "/api/admin/prices/audit"),
    ("PUT", "/api/admin/prices/override"),
    ("DELETE", "/api/admin/prices/override"),
    ("POST", "/api/admin/prices/refresh"),
    ("PUT", "/api/admin/prices/settings"),
}

NONDETERMINISTIC_HEADERS = {"date", "set-cookie", "x-request-id", "content-length"}
# path regex -> why the body is not byte-comparable
STATUS_ONLY = {
    r"^/api/csrf$": "random CSRF token per call",
    r"^/api/system": "uptime / memory / clock readings",
    r"^/api/stats": "time-windowed counters",
    r"^/api/sessions$": "body carries scanned_at / range_start, derived from time.time()",
    r"^/healthz$": "GPU probe timestamps",
    r"^/metrics": "live counters",
}
STREAMING = re.compile(r"(/stream|/events|/logs|/sse|/watch)")


def _routes(app: Any) -> set[tuple[str, str]]:
    return {(m, r.path) for r in app.routes if isinstance(r, APIRoute) for m in r.methods}


def _gets(app: Any) -> list[str]:
    return sorted(
        {
            r.path
            for r in app.routes
            if isinstance(r, APIRoute)
            and "GET" in r.methods
            and "{" not in r.path
            and not STREAMING.search(r.path)
        }
    )


def _snapshot(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in _gets(client.app):
        r = client.get(path, headers=headers, follow_redirects=False)
        hdrs = {k: v for k, v in r.headers.items() if k.lower() not in NONDETERMINISTIC_HEADERS}
        status_only = any(re.match(p, path) for p in STATUS_ONLY)
        out[path] = (r.status_code, hdrs, None if status_only else r.content)
    return out


def test_route_table_is_baseline_plus_edition(ext_registry: Any, client: TestClient) -> None:
    baseline = {tuple(x) for x in json.loads(BASELINE.read_text())}
    assert _routes(client.app) == baseline | EXPECTED_ADDED
    assert client.app.state.ext.routers == []


def _diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Per path that differs: both snapshots (status, headers, short body)."""
    return {
        p: [(v[0], v[1], (v[2] or b"")[:300]) for v in (a.get(p), b.get(p)) if v]
        for p in sorted(set(a) | set(b))
        if a.get(p) != b.get(p)
    }


def _both(client: TestClient, tmp_data_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    return _snapshot(client, {}), _snapshot(client, jwt_login(client))


def test_noop_entry_point_changes_no_behaviour(
    ext_registry: Any, client_factory: Any, tmp_data_dir: Path
) -> None:
    ext_registry()
    with client_factory() as c:
        anon_a, admin_a = _both(c, tmp_data_dir)
    ext_registry(FakeEntryPoint("noop", lambda reg: None))
    with client_factory() as c:
        anon_b, admin_b = _both(c, tmp_data_dir)
        assert c.app.state.ext.load_errors == []
    assert len(admin_a) > 20  # the walk really covered the app
    assert _diff(anon_a, anon_b) == {}
    assert _diff(admin_a, admin_b) == {}


def test_registry_starts_empty() -> None:
    reg = Registry()
    assert reg.public_routes == [] and reg.settings_keys == [] and reg.home_providers == []
