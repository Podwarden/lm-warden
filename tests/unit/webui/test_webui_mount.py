"""The /ui static export mount (app/webui/mount.py) and its UI security headers.

Part a of BACKLOG-macos P4 (Metal design §1; OVERSEER P4 decision, planner
iter 8): the existing Next.js UI is served as a static export by the same
uvicorn process, behind ``mount_webui``. The UI must carry its own security
headers so its Content-Security-Policy wins over the api's
``default-src 'none'; sandbox`` default (which would otherwise make every page
inert) -- the same set the standalone UI sends in
frontend/src/lib/security-headers.ts.

The header order that makes the win possible is pinned here:
``SecurityHeadersMiddleware`` (outermost) sets the api defaults with
``headers.setdefault``, which only fills a header that is absent; the
inner UI layer's CSP is therefore already present when the middleware runs,
so the UI's policy survives.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.testclient import TestClient


def _make_export(tmp_path: Path) -> Path:
    """A minimal export in the trailingSlash layout: an index.html at the
    root and in a subdirectory, plus one static asset."""
    export = tmp_path / "out"
    (export / "models").mkdir(parents=True)
    (export / "_next" / "static").mkdir(parents=True)
    (export / "index.html").write_text("<!doctype html><title>home</title>", encoding="utf-8")
    (export / "models" / "index.html").write_text(
        "<!doctype html><title>models</title>", encoding="utf-8"
    )
    (export / "_next" / "static" / "x.js").write_text("console.log('x');", encoding="utf-8")
    return export


def _app_with_webui(export: Path) -> FastAPI:
    """A bare app with the api's SecurityHeadersMiddleware outermost and the
    /ui mount inside it -- the shape the queued wiring task will produce in
    build_app, without the rest of build_app's weight."""
    from app.utils.security_headers import SecurityHeadersMiddleware
    from app.webui.mount import mount_webui

    app = FastAPI()
    mount_webui(app, export)
    app.add_middleware(SecurityHeadersMiddleware)
    return app


def test_ui_index_serves_export_index_with_ui_csp_not_api_sandbox(tmp_path):
    # (a) GET /ui/ -> the export's index.html, with the UI CSP, not the api's
    # "sandbox" CSP, with SecurityHeadersMiddleware outermost.
    from app.utils.security_headers import API_CSP
    from app.webui.mount import UI_CSP

    app = _app_with_webui(_make_export(tmp_path))
    with TestClient(app) as client:
        r = client.get("/ui/")

    assert r.status_code == 200
    assert r.text == "<!doctype html><title>home</title>"
    assert r.headers["content-security-policy"] == UI_CSP
    assert "sandbox" not in r.headers["content-security-policy"]
    assert r.headers["content-security-policy"] != API_CSP


def test_ui_models_serves_directory_index(tmp_path):
    # (b) GET /ui/models/ -> models/index.html (html=True directory index,
    # trailingSlash export layout).
    app = _app_with_webui(_make_export(tmp_path))
    with TestClient(app) as client:
        r = client.get("/ui/models/")

    assert r.status_code == 200
    assert r.text == "<!doctype html><title>models</title>"


def test_ui_static_asset_served_with_ui_headers(tmp_path):
    # (c) GET /ui/_next/static/x.js -> the asset, same UI headers, a
    # javascript content-type.
    from app.webui.mount import UI_CSP

    app = _app_with_webui(_make_export(tmp_path))
    with TestClient(app) as client:
        r = client.get("/ui/_next/static/x.js")

    assert r.status_code == 200
    assert r.text == "console.log('x');"
    assert r.headers["content-security-policy"] == UI_CSP
    # starlette 0.41 serves .js as text/javascript (the standard type); pin
    # that it is a javascript content-type, not a generic one.
    assert "javascript" in r.headers["content-type"]


def test_ui_apple_icon_is_served_as_png(tmp_path):
    # The export holds the apple-icon metadata image as an extensionless
    # `apple-icon` (Next names the file after the route); mimetypes cannot
    # type it, and a text/plain icon under nosniff is unusable. A 304
    # revalidation still answers without a body.
    export = _make_export(tmp_path)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    (export / "apple-icon").write_bytes(png)
    app = _app_with_webui(export)
    with TestClient(app) as client:
        r = client.get("/ui/apple-icon")
        again = client.get("/ui/apple-icon", headers={"if-none-match": r.headers["etag"]})

    assert r.status_code == 200
    assert r.content == png
    assert r.headers["content-type"] == "image/png"
    assert r.headers["content-length"] == str(len(png))
    assert again.status_code == 304


def test_ui_export_404_page_keeps_its_404_status(tmp_path):
    # The real export carries a 404.html (frontend/src/app/not-found.tsx),
    # which StaticFiles(html=True) serves for a missing path. The mount's
    # StaticFiles subclass (the apple-icon typing) must leave that page a
    # 404 with its html content type.
    export = _make_export(tmp_path)
    (export / "404.html").write_text("<!doctype html><title>nf</title>", encoding="utf-8")
    app = _app_with_webui(export)
    with TestClient(app) as client:
        r = client.get("/ui/nope/")

    assert r.status_code == 404
    assert "<title>nf</title>" in r.text
    assert r.headers["content-type"].startswith("text/html")


def test_ui_missing_path_is_404(tmp_path):
    # (d) GET /ui/nope/ -> 404 (a path the export does not have).
    app = _app_with_webui(_make_export(tmp_path))
    with TestClient(app) as client:
        r = client.get("/ui/nope/")

    assert r.status_code == 404


def test_each_ui_header_is_set_by_name(tmp_path):
    # (e) one leg per header, asserted by name on a /ui response.
    from app.webui.mount import UI_CSP, UI_PERMISSIONS_POLICY

    app = _app_with_webui(_make_export(tmp_path))
    with TestClient(app) as client:
        r = client.get("/ui/")

    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["permissions-policy"] == UI_PERMISSIONS_POLICY
    assert r.headers["content-security-policy"] == UI_CSP


def test_ui_constants_match_security_headers_ts():
    # The two constants must port the values from
    # frontend/src/lib/security-headers.ts verbatim (that file is the source
    # they cite); pin the exact strings so a drift in either direction is
    # caught.
    from app.webui.mount import UI_CSP, UI_PERMISSIONS_POLICY

    assert UI_CSP == "frame-ancestors 'none'; base-uri 'self'; object-src 'none'"
    assert UI_PERMISSIONS_POLICY == (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), payment=(), usb=(), browsing-topics=()"
    )


def test_mount_missing_dir_raises_value_error(tmp_path):
    # (f) a directory that does not exist -> ValueError naming the path.
    from app.webui.mount import mount_webui

    missing = tmp_path / "does-not-exist"
    app = FastAPI()
    try:
        mount_webui(app, missing)
    except ValueError as e:
        assert str(missing) in str(e)
    else:
        raise AssertionError("expected ValueError for a missing export directory")


def test_mount_dir_without_index_html_raises_value_error(tmp_path):
    # (f) a directory that exists but has no index.html -> ValueError naming
    # the path. (The other arm of the same acceptance leg as the missing dir.)
    from app.webui.mount import mount_webui

    export = tmp_path / "no-index"
    export.mkdir()
    app = FastAPI()
    try:
        mount_webui(app, export)
    except ValueError as e:
        assert str(export) in str(e)
    else:
        raise AssertionError("expected ValueError for an export directory without index.html")


def test_non_ui_route_keeps_api_csp(tmp_path):
    # (g) control: a non-/ui route on the same app keeps the api's CSP (the
    # UI layer must not leak its headers onto the rest of the app).
    from app.utils.security_headers import API_CSP

    app = _app_with_webui(_make_export(tmp_path))

    @app.get("/api/whatever", response_class=Response)
    async def whatever() -> Response:
        return Response("ok", media_type="text/plain")

    with TestClient(app) as client:
        r = client.get("/api/whatever")

    assert r.status_code == 200
    assert r.headers["content-security-policy"] == API_CSP


def _export_with_secret_beside_it(tmp_path: Path) -> tuple[Path, Path]:
    """The export plus a file OUTSIDE it that no /ui URL may reach."""
    export = _make_export(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-OUTSIDE-EXPORT", encoding="utf-8")
    return export, secret


def test_dot_dot_traversal_out_of_export_is_404(tmp_path):
    # Security: an encoded ``..`` segment must not climb out of export_dir.
    # The secret sits one level above the export; the request is 404 and the
    # secret's bytes are not in the body.
    export, _ = _export_with_secret_beside_it(tmp_path)
    app = _app_with_webui(export)
    with TestClient(app) as client:
        for url in ("/ui/%2e%2e/secret.txt", "/ui/models/%2e%2e/%2e%2e/secret.txt"):
            r = client.get(url)
            assert r.status_code == 404, url
            assert "TOP-SECRET" not in r.text, url


def test_symlink_escaping_export_is_404(tmp_path):
    # Security: a symlink inside the export that points outside it is not
    # followed (StaticFiles' default follow_symlink=False resolves the real
    # path and refuses it when it leaves the directory).
    export, secret = _export_with_secret_beside_it(tmp_path)
    (export / "leak.txt").symlink_to(secret)
    (export / "leakdir").symlink_to(tmp_path, target_is_directory=True)
    app = _app_with_webui(export)
    with TestClient(app) as client:
        for url in ("/ui/leak.txt", "/ui/leakdir/secret.txt"):
            r = client.get(url)
            assert r.status_code == 404, url
            assert "TOP-SECRET" not in r.text, url


def test_mount_claims_only_the_ui_prefix(tmp_path):
    # The mount must not shadow any route outside /ui: a route whose path
    # merely starts with the letters "ui" (registered AFTER the mount, as the
    # wiring puts the mount last) still answers, and so does the root.
    app = _app_with_webui(_make_export(tmp_path))

    @app.get("/uikit", response_class=Response)
    async def uikit() -> Response:
        return Response("uikit", media_type="text/plain")

    @app.get("/", response_class=Response)
    async def root() -> Response:
        return Response("root", media_type="text/plain")

    with TestClient(app) as client:
        assert client.get("/uikit").text == "uikit"
        assert client.get("/").text == "root"
