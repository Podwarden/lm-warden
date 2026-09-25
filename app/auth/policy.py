"""Which credential each route takes -- the control API's auth policy as data.

Read by tests/unit/auth/test_auth_matrix.py, which walks every mounted route,
sorts it into one of four buckets and fails when the route's dependencies do
not carry that bucket's guard (or a constant names a route that is gone), and
by app/openapi_spec.py, which marks PUBLIC_ROUTES and /v1 with `security: []`.

  inference  path starts with INFERENCE_PREFIX   require_bearer (vw_ keys)
  session    SESSION_ONLY_ROUTES, or under a     require_session
             SESSION_ONLY_PREFIXES path
  public     PUBLIC_ROUTES                       no credential
  admin      every other route                   require_jwt / require_sse_ticket /
                                                 require_operator: a session JWT
                                                 or an admin token (vwa_...)

Keys are (METHOD, route.path) -- the template as declared, converters included.
Spec: docs/superpowers/specs/2026-09-19-admin-tokens-design.md.
"""

#: The OpenAI-compatible data plane. Inference tokens only; never an admin
#: token (decision 2).
INFERENCE_PREFIX = "/v1/"

#: Session-only (decision 3): an admin token gets 403 `session_only`. It must
#: not sign a session out, mint SSE tickets, or issue, refresh, revoke or read
#: the audit of admin tokens -- so a leaked one cannot copy, extend or hide
#: itself.
SESSION_ONLY_ROUTES: frozenset[tuple[str, str]] = frozenset({
    ("POST", "/api/auth/logout"),
    ("POST", "/api/auth/sse-ticket"),
    ("GET", "/api/admin-tokens"),
    ("POST", "/api/admin-tokens"),
    ("POST", "/api/admin-tokens/{token_id}/rotate"),
    ("DELETE", "/api/admin-tokens/{token_id}"),
    ("GET", "/api/admin-tokens/{token_id}/audit"),
})

#: Session-only by prefix: EVERY route at or under one of these paths is
#: session-only, whether or not it is listed above -- so a new admin-token
#: management route cannot land in the admin bucket by being forgotten.
SESSION_ONLY_PREFIXES: tuple[str, ...] = ("/api/admin-tokens",)


def is_session_only(method: str, path: str) -> bool:
    """Whether (METHOD, route.path) is session-only: listed in
    SESSION_ONLY_ROUTES, or at/under a SESSION_ONLY_PREFIXES path."""
    return (method, path) in SESSION_ONLY_ROUTES or any(
        path == prefix or path.startswith(prefix + "/") for prefix in SESSION_ONLY_PREFIXES
    )


#: No control-plane credential at all. The setup wizard runs before an admin
#: exists; login and refresh are how a session starts; /api/csrf mints the
#: anonymous CSRF token; the landing page, its assets, robots.txt,
#: sitemap.xml and llms*.txt are for anonymous browsers and crawlers; a chat2
#: attachment is fetched with its own signed URL (?t=).
PUBLIC_ROUTES: frozenset[tuple[str, str]] = frozenset({
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
    ("GET", "/_landing"),
    ("GET", "/_landing/assets/{name}"),
    ("GET", "/robots.txt"),
    ("GET", "/sitemap.xml"),
    ("GET", "/llms.txt"),
    ("GET", "/llms-full.txt"),
    ("GET", "/api/chat2/attachments/{attachment_id}"),
})
