"""An unconfigured deployment must accept its own public origin, not localhost.

Behind any reverse proxy — which is how LM Warden is normally deployed —
`VW_FRONTEND_ORIGIN` is usually unset, because nothing in the deploy flow knows
the public URL at container-build time. `load_settings` then falls back to
`("http://localhost:3000",)`, which is never the browser's origin, so
`require_matching_origin` 403s every request that carries one.

Observed on a live install at https://ultra86.vxloc.com. Login succeeded (it
carries no refresh cookie yet), then every page reload logged the operator out,
because the reload's `POST /api/auth/refresh` was rejected before it could read
the cookie:

    Origin: https://ultra86.vxloc.com  ->  403   origin mismatch
    Origin: http://localhost:3000      ->  401   (reached the handler)

The existing code already knows how to derive the origin from the request, but
only under `trust_proxy_origin`, which defaults False and must be set by hand —
so it does not help the case where nothing was configured at all.

Deriving is safe, and is the same reasoning as PodWarden Core's OIDC redirect
base (core#2599): the browser sets both `Origin` and `Host`, and a page cannot
forge `Host` on a cross-site fetch. Comparing them therefore still answers the
question CSRF protection actually asks — "did this request originate from the
same place it was sent to?" — while a hardcoded localhost answers a question
nobody asked.

The Settings-level fail-closed guarantee is deliberately preserved: constructing
`Settings(allowed_origins=(), derive_origin_from_request=False)` still blocks
everything. Derivation is opt-in state that only `load_settings` turns on, and
only when the operator configured no origin.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.auth.origin import require_matching_origin
from app.config import load_settings


class _Req:
    """Minimal Request stand-in: headers plus a url.scheme."""

    def __init__(self, headers: dict[str, str], scheme: str = "http"):
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.url = type("U", (), {"scheme": scheme})()


# --------------------------------------------------------------------------
# The live failure
# --------------------------------------------------------------------------

def test_unconfigured_deployment_accepts_its_own_public_origin():
    req = _Req({
        "origin": "https://ultra86.vxloc.com",
        "host": "ultra86.vxloc.com",
        "x-forwarded-proto": "https",
        "x-forwarded-host": "ultra86.vxloc.com",
    }, scheme="http")
    # No configured allowlist — exactly what an un-templated deploy produces.
    require_matching_origin(req, (), derive_from_request=True)


def test_it_works_without_forwarded_headers_too():
    """Not every proxy sets X-Forwarded-*; the request's own Host still answers."""
    req = _Req({"origin": "https://ultra86.vxloc.com",
                "host": "ultra86.vxloc.com"}, scheme="https")
    require_matching_origin(req, (), derive_from_request=True)


# --------------------------------------------------------------------------
# Derivation must not become a hole
# --------------------------------------------------------------------------

def test_a_foreign_origin_is_still_rejected():
    """The whole point: an attacker's page cannot forge Host on a cross-site
    fetch, so Origin != Host and the request is refused."""
    req = _Req({"origin": "https://evil.example",
                "host": "ultra86.vxloc.com"}, scheme="https")
    with pytest.raises(HTTPException) as exc:
        require_matching_origin(req, (), derive_from_request=True)
    assert exc.value.status_code == 403


def test_a_missing_origin_is_still_rejected():
    req = _Req({"host": "ultra86.vxloc.com"}, scheme="https")
    with pytest.raises(HTTPException) as exc:
        require_matching_origin(req, (), derive_from_request=True)
    assert exc.value.status_code == 403


def test_scheme_mismatch_is_rejected():
    """http://host must not satisfy a request that arrived over https."""
    req = _Req({"origin": "http://ultra86.vxloc.com",
                "host": "ultra86.vxloc.com",
                "x-forwarded-proto": "https"}, scheme="https")
    with pytest.raises(HTTPException):
        require_matching_origin(req, (), derive_from_request=True)


# --------------------------------------------------------------------------
# The existing fail-closed contract is untouched
# --------------------------------------------------------------------------

def test_settings_level_fail_closed_still_blocks_everything():
    """`Settings(allowed_origins=())` with derivation off blocks all origins.

    config.py documents this as a deliberate guarantee, separate from
    env-parse leniency. Derivation must be opt-in state, not a new default
    that quietly reopens it.
    """
    req = _Req({"origin": "https://ultra86.vxloc.com",
                "host": "ultra86.vxloc.com"}, scheme="https")
    with pytest.raises(HTTPException):
        require_matching_origin(req, (), derive_from_request=False)


def test_an_explicit_allowlist_still_wins():
    req = _Req({"origin": "https://configured.example",
                "host": "ultra86.vxloc.com"}, scheme="https")
    require_matching_origin(req, ("https://configured.example",),
                            derive_from_request=True)


# --------------------------------------------------------------------------
# load_settings wiring
# --------------------------------------------------------------------------

def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 40)


def test_unset_origin_enables_derivation(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.delenv("VW_FRONTEND_ORIGIN", raising=False)
    s = load_settings()
    assert s.derive_origin_from_request is True
    # The localhost default is KEPT, not replaced: local development runs the
    # frontend on :3000 against this API on :8080, which is genuinely
    # cross-origin, so derivation alone would reject every dev request.
    assert s.allowed_origins == ("http://localhost:3000",)


def test_blank_origin_enables_derivation(monkeypatch, tmp_path):
    """A blank-but-set value is what the deploy template actually produces."""
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("VW_FRONTEND_ORIGIN", "")
    s = load_settings()
    assert s.derive_origin_from_request is True
    assert s.allowed_origins == ("http://localhost:3000",)


def test_an_explicit_origin_disables_derivation(monkeypatch, tmp_path):
    """Configuring an allowlist is a deliberate act — honour it exactly."""
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("VW_FRONTEND_ORIGIN", "https://configured.example")
    s = load_settings()
    assert s.allowed_origins == ("https://configured.example",)
    assert s.derive_origin_from_request is False


# --------------------------------------------------------------------------
# Many domains per workload — the case a single env var cannot express
# --------------------------------------------------------------------------

@pytest.mark.parametrize("domain", [
    "https://ultra86.vxloc.com",
    "https://flint88.ichinichi.ai",
    "https://a-third-name.example",
])
def test_every_attached_domain_is_accepted(domain):
    """A workload may answer on several domains, and the set changes over time.

    That is the case a configured allowlist handles worst: `VW_FRONTEND_ORIGIN`
    would have to be re-templated every time an operator attaches or removes a
    domain, and nothing in the deploy flow does that. Derivation needs no
    knowledge of the set at all — each request answers for the host it arrived
    on, so a domain attached tomorrow works without redeploying.
    """
    host = domain.split("://", 1)[1]
    req = _Req({"origin": domain, "host": host,
                "x-forwarded-proto": "https", "x-forwarded-host": host})
    require_matching_origin(req, (), derive_from_request=True)


def test_one_attached_domain_does_not_authorise_another():
    """Derivation is per-request, not a union: a request that arrived on one
    host still refuses an Origin claiming a different one."""
    req = _Req({"origin": "https://flint88.ichinichi.ai",
                "host": "ultra86.vxloc.com",
                "x-forwarded-proto": "https",
                "x-forwarded-host": "ultra86.vxloc.com"})
    with pytest.raises(HTTPException) as exc:
        require_matching_origin(req, (), derive_from_request=True)
    assert exc.value.status_code == 403
