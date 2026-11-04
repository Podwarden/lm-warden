"""Which credential each route takes -- the control API's auth policy as data.

Read by tests/unit/auth/test_auth_matrix.py, which walks every mounted route,
sorts it into one of four buckets and fails when the route's dependencies do
not carry that bucket's guard (or a constant names a route that is gone), and
by app/openapi_spec.py, which marks the public routes and /v1 with `security: []`.

  inference  path starts with INFERENCE_PREFIX   require_bearer (vw_ keys)
  session    SESSION_ONLY_ROUTES, or under a     require_session
             SESSION_ONLY_PREFIXES path
  public     app/auth/public_routes.py           no credential
  admin      every other route                   require_jwt / require_sse_ticket /
                                                 require_operator: a session JWT
                                                 or an admin token (vwa_...)

Keys are (METHOD, route.path) -- the template as declared, converters included.
Spec: docs/superpowers/specs/2026-09-19-admin-tokens-design.md.
"""

#: The OpenAI-compatible data plane. Inference tokens only; never an admin
#: token (decision 2).
INFERENCE_PREFIX = "/v1/"

#: Control-plane routes that take an INFERENCE key (vw_...) rather than a
#: control-plane credential: the key holder's forest login trades one for a
#: forest-only JWT. Sorted into the inference bucket (require_bearer, no admin
#: token, no session JWT).
INFERENCE_KEY_ROUTES: frozenset[tuple[str, str]] = frozenset({("POST", "/api/forest/login")})


def is_inference(method: str, path: str) -> bool:
    """Whether (METHOD, route.path) takes an inference key: /v1, or INFERENCE_KEY_ROUTES."""
    return path.startswith(INFERENCE_PREFIX) or (method, path) in INFERENCE_KEY_ROUTES


#: The only routes a key holder's forest JWT (typ="forest") may open. Every
#: other route refuses it (tests/unit/auth/test_auth_matrix.py enforces both
#: directions).
FOREST_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/stats/forest"),
        ("GET", "/api/stats/requests"),
        # Sessions pages: read-only, scoped to the key's own token.
        ("GET", "/api/sessions"),
        ("GET", "/api/sessions/{session_id}"),
        ("GET", "/api/requests/{request_id}"),
        ("POST", "/api/forest/logout"),
    }
    | {
        # The key holder's own area (app/tokens/routes_me.py): see KEY_HOLDER_ROUTES.
        ("GET", "/api/me/key"),
        ("PATCH", "/api/me/key"),
        ("POST", "/api/me/key/rotate"),
        ("POST", "/api/me/key/revoke"),
        ("GET", "/api/me/usage"),
        ("GET", "/api/me/cache"),
        ("GET", "/api/me/guard/events"),
    }
)

#: Routes ONLY a key holder's sign-in opens: the key's own area. An admin session or admin token
#: gets 403 there (the admin manages keys under /api/tokens).
KEY_HOLDER_ROUTES: frozenset[tuple[str, str]] = frozenset(
    r for r in FOREST_ROUTES if r[1].startswith("/api/me/")
)

#: Session-only (decision 3): an admin token gets 403 `session_only`. It must
#: not sign a session out, mint SSE tickets, or issue, refresh, revoke or read
#: the audit of admin tokens -- so a leaked one cannot copy, extend or hide
#: itself.
SESSION_ONLY_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/api/auth/logout"),
        ("POST", "/api/auth/sse-ticket"),
        ("GET", "/api/admin-tokens"),
        ("POST", "/api/admin-tokens"),
        ("POST", "/api/admin-tokens/{token_id}/rotate"),
        ("DELETE", "/api/admin-tokens/{token_id}"),
        ("GET", "/api/admin-tokens/{token_id}/audit"),
    }
)

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
