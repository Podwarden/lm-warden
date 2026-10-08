import pytest
from fastapi import HTTPException, Request

from app.auth.origin import require_matching_origin

ALLOWED = ("https://vllm.example.com",)


def _req(origin_value=None, headers=None):
    raw = list(headers or [])
    if origin_value is not None:
        raw.append((b"origin", origin_value.encode()))
    scope = {
        "type": "http",
        "headers": raw,
        "method": "POST",
        "scheme": "http",
        "path": "/api/auth/refresh",
    }
    return Request(scope)


def test_missing_origin_rejected():
    with pytest.raises(HTTPException) as ei:
        require_matching_origin(_req(None), ALLOWED)
    assert ei.value.status_code == 403


def test_origin_in_single_element_allowlist_ok():
    require_matching_origin(_req("https://vllm.example.com"), ALLOWED)


def test_origin_in_multi_element_allowlist_ok():
    allowed = ("https://a.example.com", "https://b.example.com")
    require_matching_origin(_req("https://a.example.com"), allowed)
    require_matching_origin(_req("https://b.example.com"), allowed)


def test_third_origin_not_in_list_rejected():
    allowed = ("https://a.example.com", "https://b.example.com")
    with pytest.raises(HTTPException) as ei:
        require_matching_origin(_req("https://c.example.com"), allowed)
    assert ei.value.status_code == 403


def test_trailing_slash_normalized_match():
    # env value carried a trailing slash; browser Origin never does
    require_matching_origin(_req("https://vllm.example.com"), ("https://vllm.example.com/",))


def test_not_listed_trust_off_rejected():
    with pytest.raises(HTTPException) as ei:
        require_matching_origin(
            _req("https://other.example.com"), ALLOWED, trust_proxy_origin=False
        )
    assert ei.value.status_code == 403


def test_not_listed_trust_on_forwarded_match_ok():
    req = _req(
        "https://proxied.example.com",
        headers=[
            (b"x-forwarded-proto", b"https"),
            (b"x-forwarded-host", b"proxied.example.com"),
        ],
    )
    require_matching_origin(req, ALLOWED, trust_proxy_origin=True)


def test_trust_on_forwarded_host_mismatch_rejected():
    req = _req(
        "https://proxied.example.com",
        headers=[
            (b"x-forwarded-proto", b"https"),
            (b"x-forwarded-host", b"different.example.com"),
        ],
    )
    with pytest.raises(HTTPException) as ei:
        require_matching_origin(req, ALLOWED, trust_proxy_origin=True)
    assert ei.value.status_code == 403


def test_trust_on_host_fallback_ok():
    # no X-Forwarded-Host → fall back to Host header
    req = _req(
        "https://hosted.example.com",
        headers=[
            (b"x-forwarded-proto", b"https"),
            (b"host", b"hosted.example.com"),
        ],
    )
    require_matching_origin(req, ALLOWED, trust_proxy_origin=True)


def test_empty_allowlist_trust_off_rejected():
    with pytest.raises(HTTPException) as ei:
        require_matching_origin(_req("https://anything.example.com"), ())
    assert ei.value.status_code == 403


def test_trust_on_forwarded_host_comma_list_uses_first_value():
    # C-1: X-Forwarded-Host may carry a proxy-chain comma list. The derived
    # origin must use only the first (client-most) value, symmetric with
    # X-Forwarded-Proto handling. A request whose Origin matches the first
    # value is accepted; a request whose Origin matches a later (injected)
    # value is rejected.
    headers = [
        (b"x-forwarded-proto", b"https"),
        (b"x-forwarded-host", b"legit.com, evil.com"),
    ]
    require_matching_origin(
        _req("https://legit.com", headers=headers),
        ALLOWED,
        trust_proxy_origin=True,
    )
    with pytest.raises(HTTPException) as ei:
        require_matching_origin(
            _req("https://evil.com", headers=headers),
            ALLOWED,
            trust_proxy_origin=True,
        )
    assert ei.value.status_code == 403


def test_trust_on_no_forwarded_headers_rejected():
    # I-2: trust mode enabled but the request arrives with no forwarded
    # headers (direct connection). The Host fallback would derive an origin
    # from the test client's Host header, which is not the listed origin, so
    # the request must still 403 — fail-safe boundary, trust mode does not
    # blanket-allow.
    with pytest.raises(HTTPException) as ei:
        require_matching_origin(
            _req("https://other.example.com"),
            ALLOWED,
            trust_proxy_origin=True,
        )
    assert ei.value.status_code == 403
