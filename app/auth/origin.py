from fastapi import HTTPException, Request, status


def _normalize(origin: str) -> str:
    return origin.rstrip("/")


def _first(value: str | None) -> str | None:
    # X-Forwarded-* may be a comma list (proxy chain); take the client-most.
    if value is None:
        return None
    return value.split(",")[0].strip()


def _derived_origin(request: Request) -> str | None:
    """The origin this request was actually addressed to, or None.

    Prefers the proxy's forwarded view, falling back to the request's own
    Host and scheme so a proxy that sets no X-Forwarded-* still works.
    """
    proto = _first(request.headers.get("x-forwarded-proto")) or request.url.scheme
    host = _first(request.headers.get("x-forwarded-host")) or request.headers.get("host")
    if not proto or not host:
        return None
    return _normalize(f"{proto}://{host}")


def require_matching_origin(
    request: Request,
    allowed_origins: tuple[str, ...],
    trust_proxy_origin: bool = False,
    derive_from_request: bool = False,
) -> None:
    got = request.headers.get("origin")
    if got is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "origin missing")

    got_norm = _normalize(got)
    if any(got_norm == _normalize(a) for a in allowed_origins):
        return

    # Derive the origin this request was addressed to and compare. Two callers
    # want this: an operator who opted into trusting the proxy, and a
    # deployment that configured no origin at all.
    #
    # It is not a hole. A browser sets both Origin and Host, and a page cannot
    # forge Host on a cross-site fetch — so comparing them still answers the
    # question CSRF protection asks ("did this come from where it was sent
    # to?"). The alternative, a hardcoded localhost, answers a question nobody
    # asked and 403s every real deployment behind a proxy.
    if trust_proxy_origin or derive_from_request:
        derived = _derived_origin(request)
        if derived is not None and got_norm == derived:
            return

    raise HTTPException(status.HTTP_403_FORBIDDEN, "origin mismatch")


def origin_check_dep(request: Request) -> None:
    settings = request.app.state.settings
    require_matching_origin(
        request,
        settings.allowed_origins,
        trust_proxy_origin=settings.trust_proxy_origin,
        derive_from_request=settings.derive_origin_from_request,
    )
