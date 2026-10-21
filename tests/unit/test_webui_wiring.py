"""build_app wiring of the /ui static export (BACKLOG-macos P4 part c).

Part a (app/webui/mount.py) and part b (app/webui/redirects.py) exist but are
not registered; this pins the registration: ``build_app`` reads
``VW_WEBUI_DIR`` at build time and, only when it names a directory, includes
``build_redirect_router()`` and calls ``mount_webui`` LAST -- after /healthz,
after ``install_openapi`` and after the /v1 catch-all guard -- so the /ui
mount cannot shadow an API route and the API routes cannot shadow /ui.
Without the variable the route table is byte-identical to a Linux/CI build.

The mount's own serving behaviour is p4a's (tests/unit/webui/test_webui_mount.py)
and the redirect table's is p4b's; this file pins the registration and its
interaction with the api: the /ui/healthz alias answers the api's /healthz
JSON, /models' 308 is asserted against the table's own target, the UI CSP is
on /ui pages only and the api's sandbox CSP stays on api responses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# The export must satisfy what mount_webui checks at mount time: a directory
# holding an index.html. These names are the boundary of that rule.
EXPORT_ROOT_INDEX = "index.html"
EXPORT_SUBDIR = "models"


def _make_export(tmp_path: Path) -> Path:
    export = tmp_path / "out"
    (export / EXPORT_SUBDIR).mkdir(parents=True)
    (export / EXPORT_ROOT_INDEX).write_text("<!doctype html><title>home</title>", encoding="utf-8")
    (export / EXPORT_SUBDIR / EXPORT_ROOT_INDEX).write_text(
        "<!doctype html><title>models</title>", encoding="utf-8"
    )
    return export


def _route_keys(app) -> list:
    # (path, methods) for every route, in registration order: the control
    # compares the full table, not just the set of paths.
    return [(r.path, tuple(sorted(getattr(r, "methods", set()) or set()))) for r in app.routes]


@pytest.fixture
def webui_app(tmp_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A built app with VW_WEBUI_DIR pointing at a tmp export; the mount
    must have run at build time, so the variable is set before build_app."""
    from app.main import build_app

    monkeypatch.setenv("VW_WEBUI_DIR", str(_make_export(tmp_path)))
    return build_app()


def test_ui_index_serves_export_when_webui_dir_set(webui_app):
    # (a) the red-first leg: GET /ui/ -> 200 serving the export's index.html
    # with the UI CSP (red today: 404).
    from app.webui.mount import UI_CSP

    with TestClient(webui_app) as client:
        r = client.get("/ui/")
    assert r.status_code == 200
    assert "home" in r.text
    assert r.headers["content-security-policy"] == UI_CSP


def test_webui_dir_unset_route_table_unchanged(tmp_data_dir, monkeypatch):
    # (b) control: with VW_WEBUI_DIR unset the route table is identical to a
    # build with the feature code absent -- same (path, methods), same order.
    from app.main import build_app

    monkeypatch.delenv("VW_WEBUI_DIR", raising=False)
    a = build_app()

    # The baseline build: same env, but with the feature's imports unbound so
    # any call to them raises -- it is the route table a Linux/CI install has.
    monkeypatch.setattr("app.webui.mount.mount_webui", None)
    monkeypatch.setattr("app.webui.redirects.build_redirect_router", None)
    b = build_app()
    monkeypatch.undo()

    assert _route_keys(a) == _route_keys(b)


def test_ui_healthz_alias_matches_healthz(webui_app):
    # (c) /ui/healthz is aliased to the api's /healthz (P4 decision): same
    # status, same JSON body.
    with TestClient(webui_app) as client:
        r = client.get("/healthz")
        aliased = client.get("/ui/healthz")
    assert r.status_code == 200
    assert aliased.status_code == 200
    # checked_at is the wall clock of the last probe; a lifespan sampler tick
    # between the two GETs moves it, so compare the rest of the body.
    body, aliased_body = r.json(), aliased.json()
    body["gpu"].pop("checked_at", None)
    aliased_body["gpu"].pop("checked_at", None)
    assert aliased_body == body
    assert body["ok"] is True


def test_ui_healthz_alias_wins_over_an_export_healthz_page(tmp_data_dir, tmp_path, monkeypatch):
    # (c) boundary: even when the export carries a healthz/ page the alias is
    # the api's JSON, not the page -- the alias is registered before the mount.
    from app.main import build_app

    export = _make_export(tmp_path)
    (export / "healthz").mkdir()
    (export / "healthz" / "index.html").write_text(
        "<!doctype html><title>healthz-page</title>", encoding="utf-8"
    )
    monkeypatch.setenv("VW_WEBUI_DIR", str(export))
    with TestClient(build_app()) as client:
        aliased = client.get("/ui/healthz")
    assert aliased.status_code == 200
    assert aliased.json()["ok"] is True
    assert "healthz-page" not in aliased.text


def test_ui_healthz_alias_absent_when_unset(tmp_data_dir, monkeypatch):
    # (c) the alias exists only with the UI mounted.
    from app.main import build_app

    monkeypatch.delenv("VW_WEBUI_DIR", raising=False)
    with TestClient(build_app()) as client:
        assert client.get("/ui/healthz").status_code == 404


def test_api_and_v1_routes_reach_targets_with_ui_mounted(webui_app):
    # (d) the /ui mount cannot shadow the API: /healthz still answers JSON and
    # an /api route still answers like a route, not like the export.
    with TestClient(webui_app) as client:
        h = client.get("/healthz")
        m = client.get("/api/models")
    assert h.status_code == 200
    assert h.json()["ok"] is True
    # Not the export (a 200 HTML page); the API answers 401 without a session.
    assert m.status_code == 401
    assert "home" not in m.text


def test_webui_mount_is_last_route(webui_app):
    # (e) registration order: the /ui mount is the final app.routes entry,
    # after /healthz, install_openapi and the /v1 catch-all guard.
    last = webui_app.routes[-1]
    assert getattr(last, "path", None) == "/ui"
    assert getattr(last, "name", None) == "webui"
    paths = [getattr(r, "path", None) for r in webui_app.routes]
    assert paths.index("/healthz") < len(paths) - 1


def test_models_redirects_308_only_when_mounted(webui_app):
    # (f) wiring leg: the redirect router is registered by build_app and the
    # table's /models rule answers 308 to its own target. No-follow: the
    # TestClient otherwise follows the hop into the mount, which answers for
    # the target (p4a pins that leg) and hides the status we are asserting.
    from app.webui.redirects import REDIRECTS

    models_rule = next(rule for rule in REDIRECTS if rule[0] == "/models")
    assert models_rule[2] == 308

    with TestClient(webui_app) as client:
        r = client.get("/models", follow_redirects=False)
    assert r.status_code == models_rule[2]
    assert r.headers["location"] == models_rule[1]


def test_webui_dir_missing_raises_valueerror(tmp_data_dir, tmp_path, monkeypatch):
    # (g) a VW_WEBUI_DIR that names no directory fails loud at build time,
    # naming the path -- not a silent no-UI.
    from app.main import build_app

    missing = tmp_path / "no-such-export"
    monkeypatch.setenv("VW_WEBUI_DIR", str(missing))
    with pytest.raises(ValueError, match=str(missing)):
        build_app()


def test_webui_dir_unset_registers_no_ui_or_redirect_route(tmp_data_dir, monkeypatch):
    # (b) unset: nothing under /ui and none of the redirect table's sources is
    # registered, and /ui/ and /models answer 404 -- the Linux/CI table.
    from app.main import build_app
    from app.webui.redirects import REDIRECTS

    monkeypatch.delenv("VW_WEBUI_DIR", raising=False)
    app = build_app()
    paths = [getattr(r, "path", "") for r in app.routes]
    assert not [p for p in paths if p == "/ui" or p.startswith("/ui/")], paths
    sources = {src.removesuffix("/*") for src, _target, _status in REDIRECTS}
    assert not [p for p in paths if p in sources or p.split("/{")[0] in sources], paths
    with TestClient(app) as client:
        assert client.get("/ui/").status_code == 404
        assert client.get("/models", follow_redirects=False).status_code == 404


def test_ui_csp_on_ui_pages_only_api_csp_on_api(webui_app):
    # The UI's CSP is on /ui pages (root and a sub-page) and the api's sandbox
    # CSP is not; api responses keep the api's CSP -- the UI's does not leak.
    from app.utils.security_headers import API_CSP
    from app.webui.mount import UI_CSP

    with TestClient(webui_app) as client:
        pages = [client.get("/ui/"), client.get("/ui/models/")]
        api = [client.get("/healthz"), client.get("/api/models"), client.get("/ui/healthz")]
    for r in pages:
        assert r.status_code == 200
        assert r.headers["content-security-policy"] == UI_CSP
        assert "sandbox" not in r.headers["content-security-policy"]
    for r in api:
        assert r.headers["content-security-policy"] == API_CSP


def test_ui_unsafe_methods_refused(webui_app):
    # GET /ui/ needs no session; an unsafe method on /ui is refused: 403 from
    # the CSRF check without a token, 405 from the static mount with one.
    with TestClient(webui_app) as client:
        assert client.post("/ui/").status_code == 403
        token = client.get("/api/csrf").json()["csrf"]
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            r = client.request(method, "/ui/", headers={"X-CSRF-Token": token})
            assert r.status_code == 405, (method, r.status_code)
