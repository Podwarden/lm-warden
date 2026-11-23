"""Which routes take no credential, and which of them skip the CSRF cookie.

Core owns the setup / auth / health entries; an extension (the website, an
edition package) declares its own through ``Registry.public_routes``. Core
never imports an extension: everything here reads ``app.state.ext``.

Matching is by (METHOD, route path template) for the auth policy and the
OpenAPI document (``is_public`` / ``public_routes``). ``prefix=True`` entries
exist for the CSRF no-mint check only (``mints_csrf``): they never name a
mounted route, so they are left out of ``public_routes``.
"""

from fastapi import FastAPI

from app.ext import PublicRoute, Registry

_CORE_PUBLIC: tuple[tuple[str, str], ...] = (
    ("GET", "/api/setup/state"),
    ("POST", "/api/setup/welcome"),
    ("GET", "/api/setup/gpus"),
    ("POST", "/api/setup/gpus"),
    ("POST", "/api/setup/hf_token"),
    ("POST", "/api/setup/admin"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/refresh"),
    ("GET", "/api/csrf"),
    ("GET", "/healthz"),
    ("GET", "/readyz"),
    # A chat2 attachment is fetched with its own signed URL (?t=).
    ("GET", "/api/chat2/attachments/{attachment_id}"),
)


def core_public_routes() -> list[PublicRoute]:
    """The routes core itself serves without a credential. They mint the CSRF
    cookie as ever, except /agent-guide.md (a cacheable static document)."""
    routes = [PublicRoute(m, p, mint_csrf=True) for m, p in _CORE_PUBLIC]
    routes.append(PublicRoute("GET", "/agent-guide.md", mint_csrf=False))
    return routes


def _registered(app: FastAPI) -> list[PublicRoute]:
    """Core's routes plus the extensions', merged once per app on first use
    (``mints_csrf`` runs per request). Built lazily so it sees what extensions
    registered at load; the entry count in the key rebuilds it if one is added
    later."""
    reg: Registry | None = getattr(app.state, "ext", None)
    ext = list(reg.public_routes) if reg else []
    cached: tuple[int, list[PublicRoute]] | None = getattr(app.state, "_public_routes_cache", None)
    if cached is not None and cached[0] == len(ext):
        return cached[1]
    merged = core_public_routes() + ext
    app.state._public_routes_cache = (len(ext), merged)
    return merged


def public_routes(app: FastAPI) -> frozenset[tuple[str, str]]:
    """(METHOD, route path template) of every credential-free route."""
    return frozenset((r.method, r.path) for r in _registered(app) if not r.prefix)


def is_public(app: FastAPI, method: str, path_template: str) -> bool:
    return (method, path_template) in public_routes(app)


def mints_csrf(app: FastAPI, path: str) -> bool:
    """Whether a response at ``path`` may carry a fresh anonymous CSRF cookie.
    False only for a registered route that opted out: an exact path, or a
    ``prefix=True`` entry the path starts with."""
    for r in _registered(app):
        if r.mint_csrf:
            continue
        if path == r.path or (r.prefix and path.startswith(r.path)):
            return False
    return True
