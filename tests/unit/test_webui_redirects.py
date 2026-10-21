"""app/webui/redirects.py pins: the Caddyfile redirect table, ported verbatim,
plus the /ui/<x>/<id> -> /ui/<x>/detail/?id= bookmark routes.

The spec is deploy/caddy/Caddyfile (the redir block). Caddy is the source of
truth on Linux; on Mac the single uvicorn process serves the same table so
bookmarks keep working after the export-mode conversion (Metal design sec. 1).
The drift guard at the end re-reads the Caddyfile so a redir line added there
without a matching REDIRECTS entry fails this test.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[2]
CADDYFILE = REPO / "deploy" / "caddy" / "Caddyfile"


def _app():
    # Imported inside so the file collects on the base (red_first: the
    # ModuleNotFoundError is raised in the test body, not at import time).
    from app.webui.redirects import build_redirect_router

    app = FastAPI()
    app.include_router(build_redirect_router())
    return app


def _client() -> TestClient:
    return TestClient(_app(), follow_redirects=False)


# (Caddyfile source, expected Location for GET <source>)
REDIR_CASES = [
    ("/login", "/ui/login"),
    ("/models", "/ui/models"),
    ("/settings", "/ui/settings"),
    ("/setup", "/ui/setup"),
    ("/stats", "/ui/stats"),
    ("/tokens", "/ui/tokens"),
    ("/cache", "/ui/cache"),
    ("/chat", "/ui/chat"),
    ("/login/a", "/ui/login/a"),
    ("/login/a/b", "/ui/login/a/b"),
    ("/models/x", "/ui/models/x"),
    ("/settings/x", "/ui/settings/x"),
    ("/setup/x", "/ui/setup/x"),
    ("/stats/x", "/ui/stats/x"),
    ("/tokens/x", "/ui/tokens/x"),
    ("/cache/x", "/ui/cache/x"),
    ("/chat/x", "/ui/chat/x"),
]


@pytest.mark.parametrize(("path", "location"), REDIR_CASES)
def test_redir_table_matches_caddyfile(path: str, location: str) -> None:
    # The red_first leg: GET /models -> 308 Location /ui/models.
    # Bare-path rules use a literal target (Caddy's {uri} only applies to
    # the /* rules), so no query is appended here -- it is kept on the
    # sub-path leg below (/stats/x?a=1 -> /ui/stats/x?a=1).
    c = _client()
    r = c.get(path)
    assert r.status_code == 308, f"{path} -> {r.status_code}"
    assert r.headers["location"] == location


def test_favicon_redirects_301_to_icon_svg() -> None:
    c = _client()
    r = c.get("/favicon.ico")
    assert r.status_code == 301
    assert r.headers["location"] == "/ui/icon.svg"
    # The only 301 in the table: a bare 308 leg above proves the split.
    r308 = c.get("/chat")
    assert r308.status_code == 308


def test_query_string_preserved_on_subpath_redirect() -> None:
    c = _client()
    r = c.get("/stats/x?a=1")
    assert r.status_code == 308
    assert r.headers["location"] == "/ui/stats/x?a=1"
    # Multiple query params survive in order.
    r2 = c.get("/chat/room?b=2&a=1")
    assert r2.status_code == 308
    assert r2.headers["location"] == "/ui/chat/room?b=2&a=1"


@pytest.mark.parametrize(
    ("path", "location"),
    [
        # Caddy {uri} is the raw request URI: escapes pass through untouched,
        # never decoded into a query (%3F), a fragment (%23), a segment (%2F)
        # or a stray '%' (%25).
        ("/stats/x%3Fa=1", "/ui/stats/x%3Fa=1"),
        ("/stats/a%23b", "/ui/stats/a%23b"),
        ("/stats/a%2Fb", "/ui/stats/a%2Fb"),
        ("/stats/a%25b", "/ui/stats/a%25b"),
        ("/stats/a%20b?q=a%26b", "/ui/stats/a%20b?q=a%26b"),
        # '/<x>/' (empty rest) is under Caddy's /<x>/* too.
        ("/login/", "/ui/login/"),
        # A literal target carries no {uri}: Caddy drops the query.
        ("/login?next=/x", "/ui/login"),
        # A doubled slash stays under /ui -- never a protocol-relative //host.
        ("/login//evil.example", "/ui/login//evil.example"),
    ],
)
def test_uri_is_forwarded_raw(path: str, location: str) -> None:
    r = _client().get(path)
    assert r.status_code == 308, f"{path} -> {r.status_code}"
    assert r.headers["location"] == location


@pytest.mark.parametrize(
    "path",
    # Prefix look-alikes the Caddy smoke harness proxies to the api
    # (scripts/smoke-caddy-redirects.sh): /cache must not take
    # /cache-aware-routing, nor /login take /loginx.
    ["/cache-aware-routing", "/api-keys", "/loginx", "/chatty", "/favicon.icon"],
)
def test_prefix_lookalikes_not_redirected(path: str) -> None:
    assert _client().get(path).status_code == 404


def test_detail_id_query_metachars_are_encoded() -> None:
    # An id carrying '&', '=', '#' or '?' must not add a parameter to the
    # detail page's query (the id comes from the user's URL).
    r = _client().get("/ui/models/a%26x%3D1%23f%3Fg")
    assert r.status_code == 308
    assert r.headers["location"] == "/ui/models/detail/?id=a%26x%3D1%23f%3Fg"


@pytest.mark.parametrize(
    ("path", "location"),
    [
        ("/ui/models/abc", "/ui/models/detail/?id=abc"),
        ("/ui/models/abc/", "/ui/models/detail/?id=abc"),
        ("/ui/models/abc/settings", "/ui/models/detail/settings/?id=abc"),
        ("/ui/models/abc/settings/", "/ui/models/detail/settings/?id=abc"),
        ("/ui/tokens/t1", "/ui/tokens/detail/?id=t1"),
        ("/ui/tokens/t1/", "/ui/tokens/detail/?id=t1"),
    ],
)
def test_detail_bookmark_routes(path: str, location: str) -> None:
    c = _client()
    r = c.get(path)
    assert r.status_code == 308, f"{path} -> {r.status_code}"
    assert r.headers["location"] == location


@pytest.mark.parametrize(
    ("raw_id", "id"),
    [
        ("my model", "my%20model"),  # space percent-encoded
        ("modél", "mod%C3%A9l"),  # non-ascii (no slash) percent-encoded
    ],
)
def test_detail_id_percent_encoding(raw_id: str, id: str) -> None:
    c = _client()
    r = c.get("/ui/models/" + raw_id)
    assert r.status_code == 308
    assert r.headers["location"] == "/ui/models/detail/?id=" + id
    # quote() defaults to always_safe="/"; the id must not leak one.
    assert "/" not in quote(id, safe="")


@pytest.mark.parametrize(
    ("path", "location"),
    [
        ("/ui/tokens/t1?range=7d", "/ui/tokens/detail/?id=t1&range=7d"),
        ("/ui/tokens/t1/?from=1&to=2", "/ui/tokens/detail/?id=t1&from=1&to=2"),
        ("/ui/models/abc/settings?x=%2F", "/ui/models/detail/settings/?id=abc&x=%2F"),
        # The path id wins: a stray `id` in the old query is dropped.
        ("/ui/tokens/t1?id=other&range=24h", "/ui/tokens/detail/?id=t1&range=24h"),
    ],
)
def test_detail_bookmark_keeps_its_query(path: str, location: str) -> None:
    # A token page bookmark carries its range in the query; the 308 keeps it
    # (next.config's redirects() does the same on the Linux standalone build).
    r = _client().get(path)
    assert r.status_code == 308
    assert r.headers["location"] == location


def test_literal_detail_segment_is_not_redirected() -> None:
    # No-loop control: the new search-param page itself must not be re-redirected.
    c = _client()
    for path in ("/ui/models/detail", "/ui/models/detail/", "/ui/models/detail/?id=x"):
        r = c.get(path)
        assert r.status_code == 404, f"{path} -> {r.status_code}"
    assert c.get("/ui/tokens/detail/").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/ui/models/index.txt",
        "/ui/tokens/index.txt",
        "/ui/models/x.json",
        "/ui/models/__next.models.txt",
        "/ui/models/index.txt/settings",
    ],
)
def test_dotted_segments_are_the_uis_own_files_not_ids(path: str) -> None:
    # Without a mount they fall through to 404; they are never redirected.
    r = _client().get(path)
    assert r.status_code == 404, f"{path} -> {r.status_code}"
    assert "location" not in r.headers


def _export(tmp_path: Path) -> Path:
    # The trailingSlash export's shape for the three detail pages
    # (frontend/src/app/{models,tokens}/detail/**, BACKLOG-macos P4).
    for page in ("", "models/detail/", "models/detail/settings/", "tokens/detail/"):
        d = tmp_path / page
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(f"<html>page:/{page}</html>", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("bookmark", "page"),
    [
        ("/ui/models/abc", "models/detail/"),
        ("/ui/models/abc/settings", "models/detail/settings/"),
        ("/ui/tokens/t1", "tokens/detail/"),
    ],
)
def test_detail_pages_reach_the_mount_with_the_router_in_front(
    tmp_path: Path, bookmark: str, page: str
) -> None:
    # Wired as app/main.py does: the router first, the /ui mount after it. A
    # route that merely *answers* the literal `detail` segment (even with a
    # 404) shadows the mount, so the new page itself must not match a
    # bookmark route at all -- else every old bookmark lands on a 404.
    from app.webui.mount import mount_webui

    app = _app()
    mount_webui(app, _export(tmp_path))
    c = TestClient(app, follow_redirects=False)
    r = c.get(bookmark)
    assert r.status_code == 308
    landed = c.get(r.headers["location"])
    assert landed.status_code == 200, f"{r.headers['location']} -> {landed.status_code}"
    assert f"page:/{page}" in landed.text


def test_root_and_healthz_not_matched() -> None:
    # The landing page and the api own / and /healthz; the router must not claim them.
    c = _client()
    assert c.get("/").status_code == 404
    assert c.get("/healthz").status_code == 404
    # Unknown /ui/ paths are the mount's concern (p4a), not the router's.
    assert c.get("/ui/nope").status_code == 404


def test_caddyfile_redir_drift_guard() -> None:
    # Every `redir` line in the Caddyfile must have a matching REDIRECTS entry,
    # so adding a Caddy rule without porting it fails here.
    from app.webui.redirects import REDIRECTS  # imported inside so the file collects on the base

    lines = CADDYFILE.read_text(encoding="utf-8").splitlines()
    caddy_rules = [
        (m.group(1), m.group(2), int(m.group(3)))
        for m in (re.match(r"\s*redir\s+(\S+)\s+(\S+)\s+(\d+)\s*$", ln) for ln in lines)
        if m is not None
    ]
    # Verbatim, both ways and in order: no Caddy rule missing, no extra rule,
    # same targets and statuses -- and the regex must actually find the block.
    assert len(caddy_rules) == 17
    assert tuple(caddy_rules) == REDIRECTS
    # REDIRECTS is a tuple of (source, target, status) -- the Caddyfile
    # lines, in order. Build a lookup by source.
    table = {source: status for source, _target, status in REDIRECTS}

    for line in lines:
        m = re.match(r"\s*redir\s+(\S+)\s+(\S+)\s+(\d+)\s*$", line)
        if m is None:
            continue
        src, target, status = m.group(1), m.group(2), int(m.group(3))
        assert src in table, f"Caddyfile redir {src} has no REDIRECTS entry"
        assert table[src] == status, f"Caddyfile redir {src} status {status} != {table[src]}"
        c = _client()
        r = c.get(src)
        assert r.status_code == status, f"{src} -> {r.status_code} (Caddyfile says {status})"
        expected = target.replace("{uri}", src)
        assert (
            r.headers["location"] == expected
        ), f"{src} -> {r.headers['location']} (Caddyfile says {expected})"
