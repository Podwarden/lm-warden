"""Unit tests for the `_url` coercer used by the `public_url` runtime
setting (#154).

The coercer is the only line of defence between user input and the
settings KV table; PATCH validation is all-or-nothing (one bad key
aborts the whole transaction), so the rules here are deliberately
strict:

  * must be a string (no JSON nulls, bools, numbers, lists);
  * stripped value must be non-empty;
  * length <= 2048 characters (matches RFC 7230 URI parser practicality);
  * scheme MUST be http or https — the snippets we render with this URL
    don't make sense for ftp:// or file://;
  * netloc MUST be non-empty (catches "http://", "http:///foo");
  * canonical form strips a single trailing slash so two operators
    typing `https://x/` and `https://x` produce the same stored value.
"""

import pytest

from app.settings.routes_api import _url


class TestUrlCoercerHappyPath:
    """Inputs the coercer accepts and the canonical TEXT it returns."""

    @pytest.mark.parametrize(
        "raw, canonical",
        [
            ("https://vllm.example.com", "https://vllm.example.com"),
            # Trailing slash gets stripped so duplicates collapse.
            ("https://vllm.example.com/", "https://vllm.example.com"),
            # http is allowed for LAN deployments.
            ("http://vllm.local", "http://vllm.local"),
            # Port is preserved.
            ("https://vllm.example.com:8443", "https://vllm.example.com:8443"),
            ("http://192.0.2.10:10000", "http://192.0.2.10:10000"),
            # Surrounding whitespace gets stripped before parse.
            ("  https://vllm.example.com  ", "https://vllm.example.com"),
            # Path is preserved (only the trailing slash on the whole URL
            # is normalised, not slashes inside the path).
            ("https://x.example.com/api/v1", "https://x.example.com/api/v1"),
        ],
    )
    def test_accepts_and_canonicalises(self, raw, canonical):
        assert _url(raw) == canonical


class TestUrlCoercerRejects:
    """Inputs the coercer must reject with ValueError. The route layer
    converts these into HTTP 422 with the key name in the detail."""

    @pytest.mark.parametrize(
        "raw",
        [
            "",  # empty string
            "   ",  # whitespace only
        ],
    )
    def test_rejects_empty(self, raw):
        with pytest.raises(ValueError, match="non-empty"):
            _url(raw)

    @pytest.mark.parametrize(
        "raw",
        [
            "ftp://example.com",
            "file:///etc/passwd",
            "ws://socket.example.com",
            "javascript:alert(1)",
            # No scheme at all — urlsplit treats it as path-only, netloc empty.
            "example.com",
        ],
    )
    def test_rejects_non_http(self, raw):
        with pytest.raises(ValueError):
            _url(raw)

    @pytest.mark.parametrize(
        "raw",
        [
            "http://",
            "https://",
            "http:///path-only",
        ],
    )
    def test_rejects_missing_netloc(self, raw):
        with pytest.raises(ValueError, match="host"):
            _url(raw)

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            True,
            False,
            123,
            12.5,
            ["https://x"],
            {"url": "https://x"},
        ],
    )
    def test_rejects_non_string(self, raw):
        with pytest.raises(ValueError, match="string"):
            _url(raw)

    def test_rejects_oversize(self):
        # 2048 is the boundary — 2049 must fail.
        too_long = "https://example.com/" + ("a" * (2049 - len("https://example.com/")))
        assert len(too_long) == 2049
        with pytest.raises(ValueError, match="2048"):
            _url(too_long)

    def test_accepts_at_boundary(self):
        """Exactly 2048 chars is allowed."""
        at_limit = "https://example.com/" + ("a" * (2048 - len("https://example.com/")))
        assert len(at_limit) == 2048
        out = _url(at_limit)
        assert out == at_limit
