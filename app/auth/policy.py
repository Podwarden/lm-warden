"""Which credential each route takes -- the control API's auth policy as data.

Read by tests/unit/auth/test_auth_matrix.py, which walks every mounted route,
sorts it into a bucket and fails when the route's dependencies do
not carry that bucket's guard (or a constant names a route that is gone), and
by app/openapi_spec.py, which marks the public routes and /v1 with `security: []`.

  inference  path starts with INFERENCE_PREFIX   require_bearer (vw_ keys)
  session    SESSION_ONLY_ROUTES, or under a     require_session
             SESSION_ONLY_PREFIXES path
  public     app/auth/public_routes.py           no credential
  key_stream KEY_STREAM_ROUTES                   require_key_ticket: a key holder's
                                                 single-use ticket, no header
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
        ("POST", "/api/me/key/kv-reset"),
        ("GET", "/api/me/key/cache"),
        ("POST", "/api/me/key/rotate"),
        ("POST", "/api/me/key/revoke"),
        ("GET", "/api/me/usage"),
        ("GET", "/api/me/cache"),
        ("GET", "/api/me/guard/events"),
        # The key holder's Stats view (app/tokens/routes_me_stats.py): own rows
        # only, bound to the signed-in key, nothing fleet-wide.
        ("GET", "/api/me/stats/speed"),
        ("GET", "/api/me/stats/tokens"),
        ("GET", "/api/me/stats/cache"),
        ("GET", "/api/me/stats/requests"),
        ("GET", "/api/me/requests/{request_id}/capture"),
        # The key holder's live Prompt Lens stream: the ticket mint (G7). The
        # stream itself is ticket-only, see KEY_STREAM_ROUTES.
        ("POST", "/api/me/prompt-lens/ticket"),
        # One whole section of an own request (G9), from memory or the key's
        # kept captures.
        ("GET", "/api/me/prompt-lens/requests/{req_id}/section/{kind}"),
        # The key's own kept captures (keep_captures, an admin's switch).
        ("GET", "/api/me/prompt-lens/kept"),
        ("GET", "/api/me/prompt-lens/kept/{req_id}"),
    }
)
# The same key-holder routes under their deprecated /api/me/godmode paths
# (app/proxy/prompt_lens_legacy.py): one release.
FOREST_ROUTES = FOREST_ROUTES | frozenset(
    (m, p.replace("/api/me/prompt-lens", "/api/me/godmode", 1))
    for m, p in FOREST_ROUTES
    if p.startswith("/api/me/prompt-lens/")
)

#: The key holder's ticket streams (ruling G-F2): a single-use ticket minted by
#: the key's sign-in (``POST /api/me/prompt-lens/ticket``) and nothing else --
#: ``require_key_ticket``. A header credential is refused: a forest sign-in or
#: an inference key 401, an admin session or admin token 403 (the admin
#: watches every key on /api/admin/prompt-lens/stream).
KEY_STREAM_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {("GET", "/api/me/prompt-lens/stream"), ("GET", "/api/me/godmode/stream")}  # old: deprecated
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
