"""Where the session cookies' ``Secure`` flag comes from.

A browser silently DISCARDS a ``Secure`` cookie that arrives over plain
``http://``. The refresh cookie used to be set with a hardcoded
``secure=True`` while the CSRF cookie was set with a hardcoded
``secure=False``, and the quick start tells operators to open
``http://YOUR-HOST:8080/ui/``. On that URL the browser stored the CSRF cookie
and threw the refresh cookie away, so ``POST /api/auth/refresh`` answered
``401 missing refresh cookie``: the access token lives only in memory, so
every full page load, reload or pasted deep link logged the operator out and
the session could never be refreshed. In-app navigation kept working, which
is what made it easy to miss.

Hardcoding ``False`` is not the fix. On a TLS deployment the flag is doing
real work -- it is what stops the cookie being replayed onto a plaintext
request to the same host. So the flag is *derived* from the scheme the
BROWSER actually used, in this order:

1. ``X-Forwarded-Proto``, but only when the operator set
   ``VW_TRUST_PROXY_ORIGIN=1``. Untrusted, that header is just something the
   client sent us, and honouring it would let any caller decide the flag.
   Trusted, it is the front door telling us what the browser did -- which is
   the only party that knows.
2. The scheme of the connection this process itself accepted. Real end-to-end
   HTTPS needs no configuration at all.
3. ``VW_FRONTEND_ORIGIN``. Setting it is a deliberate statement of the public
   URL, so an allowlist that is entirely ``https://`` means the browser is on
   HTTPS even when a TLS-terminating proxy forwards no header and was never
   trusted. Without this rule the common "Caddy in front, defaults otherwise"
   deployment would silently lose the flag.

Anything else is a plain-HTTP install: no ``Secure``, and the session works.

Both cookies call this one function. Their disagreement was the sharpest edge
of the original bug -- one cookie survived the round trip and the other did
not, so the session looked half-alive rather than plainly absent.
"""

from __future__ import annotations

from starlette.requests import Request


def _client_most(value: str | None) -> str | None:
    """First entry of a possibly comma-joined ``X-Forwarded-*`` chain."""
    if value is None:
        return None
    first = value.split(",")[0].strip()
    return first or None


def cookie_secure(request: Request) -> bool:
    """True when this request reached us over HTTPS, as the browser saw it."""
    settings = request.app.state.settings

    if settings.trust_proxy_origin:
        forwarded = _client_most(request.headers.get("x-forwarded-proto"))
        if forwarded is not None:
            # A trusted front door has spoken; it outranks our own socket,
            # which is the *inside* of the proxy hop either way.
            return forwarded.lower() == "https"

    if request.url.scheme == "https":
        return True

    # No trusted header and a plaintext socket -- but the operator may still
    # have told us the public URL. Every configured origin must be https for
    # this to be conclusive; a mixed allowlist means some browsers really are
    # on http, and dropping the flag is what keeps those sessions working.
    origins = settings.allowed_origins
    return bool(origins) and all(origin.lower().startswith("https://") for origin in origins)
