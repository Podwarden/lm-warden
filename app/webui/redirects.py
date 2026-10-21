"""The v1-bookmark redirect table, ported verbatim from deploy/caddy/Caddyfile.

The redir block (lines 125-160) is the spec: Caddy is the source of truth on
Linux, and on Mac the single uvicorn process serves the same table so
bookmarks keep working after the export-mode conversion (Metal design sec. 1,
OVERSEER P4 decision planner iter 8). The Caddyfile's ``{uri}`` placeholder
expands to the full request URI, so a wildcard rule like ``redir /login/*
/ui{uri} 308`` maps ``/login/foo`` -> ``/ui/login/foo`` -- path plus query.
The literal-segment rules use a fixed target. ``/favicon.ico`` is the table's
only 301.

Two extra rules carry the dynamic-route -> search-param conversion:
``/ui/models/<id>`` (and ``/<id>/settings``) and ``/ui/tokens/<id>`` redirect
to ``/ui/<x>/detail/?id=<id>`` so old bookmarks survive. The literal segment
``detail`` is never matched (no loop): the id segment's convertor excludes
it, so the request falls through to the /ui mount, which serves the new page.
(A route that matched it and answered 404 would shadow the mount -- the
router is registered in front of it.) ``/`` and ``/healthz`` are the landing
page and the api's -- the router does not claim them.
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Request
from starlette.convertors import Convertor, register_url_convertor
from starlette.responses import RedirectResponse, Response

# Each entry: (source, target, status). A bare source matches exactly;
# a source ending in '/*' matches the segment and everything under it,
# forwarding path + query (Caddy {uri} semantics).
REDIRECTS: tuple[tuple[str, str, int], ...] = (
    ("/login", "/ui/login", 308),
    ("/login/*", "/ui{uri}", 308),
    ("/models", "/ui/models", 308),
    ("/models/*", "/ui{uri}", 308),
    ("/settings", "/ui/settings", 308),
    ("/settings/*", "/ui{uri}", 308),
    ("/setup", "/ui/setup", 308),
    ("/setup/*", "/ui{uri}", 308),
    ("/stats", "/ui/stats", 308),
    ("/stats/*", "/ui{uri}", 308),
    ("/tokens", "/ui/tokens", 308),
    ("/tokens/*", "/ui{uri}", 308),
    ("/cache", "/ui/cache", 308),
    ("/cache/*", "/ui{uri}", 308),
    ("/chat", "/ui/chat", 308),
    ("/chat/*", "/ui{uri}", 308),
    ("/favicon.ico", "/ui/icon.svg", 301),
)


def _path_for(source: str) -> str:
    # Caddy's ``/*`` (segment + everything under it) becomes a Starlette
    # path template: ``/login/*`` -> ``/login/{seg:path}`` so
    # ``/login/foo/bar`` matches. The bare source is its own template.
    if source.endswith("/*"):
        return source.rstrip("/*") + "/{seg:path}"
    return source


def _table_target(rule_target: str, request: Request) -> str:
    # Caddy {uri} = the full request URI (path + query) exactly as the client
    # sent it, percent-escapes included. Forward the RAW path (scope
    # "raw_path"), not the decoded one: decoding would turn /stats/x%3Fa=1
    # into a real query and /stats/a%23b into a fragment. So /stats/x?a=1 ->
    # /ui/stats/x?a=1; a literal target passes through (query dropped, as
    # Caddy does for a target without {uri}).
    if "{uri}" not in rule_target:
        return rule_target
    raw = request.scope.get("raw_path")
    path = raw.decode("latin-1") if raw else request.url.path
    query = request.scope.get("query_string", b"").decode("latin-1")
    return rule_target.replace("{uri}", path + (f"?{query}" if query else ""))


class _BookmarkIdConvertor(Convertor[str]):
    """One path segment that is not the literal ``detail`` page segment.

    The id also contains no dot: model and token ids never do, and the Next
    static export keeps its own files beside the pages (``models/index.txt``,
    the RSC payload the client router fetches on every navigation, and
    similar ``*.txt`` / ``*.json`` files). A dotted segment is such a file, so
    it must fall through to the /ui mount instead of becoming
    ``detail/?id=index.txt`` ("Model not found").

    The lookahead stops at the segment end (``/`` or end of path) so
    ``/ui/models/detail/settings`` is spared too, while ``detailx`` is an id.
    """

    regex = "(?!detail(?:/|$))[^/.]+"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


#: Name under which the convertor is registered (Starlette's registry is
#: process-wide; the prefix keeps it clear of anyone else's).
_ID = "lmw_bookmark_id"
register_url_convertor(_ID, _BookmarkIdConvertor())


def build_redirect_router() -> APIRouter:
    """GET handlers for the REDIRECTS table plus the detail bookmark routes.

    ``include_in_schema=False``: these are bookmark redirects, not API
    surface. The detail routes are registered after the table; their id
    segment never matches the literal ``detail`` (the no-loop control), so
    the new search-param page falls through to the /ui mount.
    """

    router = APIRouter(include_in_schema=False)

    for source, target, status in REDIRECTS:

        @router.get(_path_for(source), include_in_schema=False)
        def _table_redirect(
            request: Request, source: str = source, target: str = target, status: int = status
        ) -> RedirectResponse:
            return RedirectResponse(url=_table_target(target, request), status_code=status)

    def _models_target(id: str, settings: bool) -> str:
        where = "detail/settings" if settings else "detail"
        return f"/ui/models/{where}/?id={quote(id)}"

    def _tokens_target(id: str) -> str:
        return f"/ui/tokens/detail/?id={quote(id)}"

    def _redirect(target: str, request: Request) -> Response:
        # Keep the bookmark's own query (a token page's range=/from=/to=), as
        # next.config's redirects() does on Linux. The path id stays first and
        # any `id` in the old query is dropped, so the page reads the path's.
        raw = request.scope.get("query_string", b"").decode("latin-1")
        rest = [kv for kv in raw.split("&") if kv and kv.split("=", 1)[0] != "id"]
        return RedirectResponse(url="&".join([target, *rest]), status_code=308)

    @router.get(f"/ui/models/{{id:{_ID}}}", include_in_schema=False)
    def _models_detail(id: str, request: Request) -> Response:
        return _redirect(_models_target(id, settings=False), request)

    @router.get(f"/ui/models/{{id:{_ID}}}/", include_in_schema=False)
    def _models_detail_slash(id: str, request: Request) -> Response:
        return _redirect(_models_target(id, settings=False), request)

    @router.get(f"/ui/models/{{id:{_ID}}}/settings", include_in_schema=False)
    def _models_settings(id: str, request: Request) -> Response:
        return _redirect(_models_target(id, settings=True), request)

    @router.get(f"/ui/models/{{id:{_ID}}}/settings/", include_in_schema=False)
    def _models_settings_slash(id: str, request: Request) -> Response:
        return _redirect(_models_target(id, settings=True), request)

    @router.get(f"/ui/tokens/{{id:{_ID}}}", include_in_schema=False)
    def _tokens_detail(id: str, request: Request) -> Response:
        return _redirect(_tokens_target(id), request)

    @router.get(f"/ui/tokens/{{id:{_ID}}}/", include_in_schema=False)
    def _tokens_detail_slash(id: str, request: Request) -> Response:
        return _redirect(_tokens_target(id), request)

    return router
