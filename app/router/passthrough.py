"""The Anthropic relay (#287 decisions 3, 17, 18, 20).

Everything here is deliberately dumb: copy the request out, copy the response
back, byte for byte, streaming. No header, body or query string is ever
logged, and nothing is parsed except two integers of ``usage``.
"""

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from typing import Any
from urllib.parse import quote, unquote

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)

#: RFC 9110 hop-by-hop headers (plus the legacy proxy ones).
HOP_BY_HOP: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

# Never relayed to Anthropic, in addition to HOP_BY_HOP: the credentials are
# supplied separately by the caller (upstream_credential_headers), the cookie
# could carry the warden's own session, host/content-length are httpx's.
_REQUEST_DROP = frozenset({"host", "content-length", "cookie", "authorization", "x-api-key"})
_LMWARDEN_PREFIX = "x-lmwarden-"

# The warden sits behind a reverse proxy / CDN: its edge and client-identity
# headers describe OUR client, not something Anthropic should learn (and
# `cdn-loop` can trip Cloudflare's loop detection on Anthropic's side).
_EDGE_EXACT = frozenset({"x-real-ip", "forwarded", "cdn-loop", "true-client-ip"})
_EDGE_PREFIXES = ("x-forwarded-", "cf-")

# Anthropic's origin state must not be set on the warden's origin: cookies,
# HSTS, alternative services and the reporting endpoints.
_RESPONSE_DROP = frozenset(
    {
        "set-cookie",
        "set-cookie2",
        "alt-svc",
        "strict-transport-security",
        "report-to",
        "nel",
    }
)

#: Largest non-stream body we read ``usage`` from, and the longest SSE line we
#: buffer while looking for it.
USAGE_CAP = 1 << 20


#: Relay pool bounds: a burst beyond this waits at most ``POOL_TIMEOUT_S`` for a
#: connection and then gets a 529, instead of hanging until a stream ends.
MAX_CONNECTIONS = 256
MAX_KEEPALIVE = 32
POOL_TIMEOUT_S = 10.0


def make_relay_client() -> httpx.AsyncClient:
    """The ONE shared client of the relay. Long generations are legitimate, so
    only connect and pool-wait are bounded; no redirects, no proxy env."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=POOL_TIMEOUT_S),
        limits=httpx.Limits(
            max_connections=MAX_CONNECTIONS, max_keepalive_connections=MAX_KEEPALIVE
        ),
        follow_redirects=False,
        trust_env=False,
    )


class RelayError(Exception):
    """The relay could not (or must not) reach Anthropic; ``status`` is ours."""

    def __init__(
        self,
        status: int,
        message: str,
        reason: str | None = None,
        error_reason: str = "upstream_unreachable",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        #: The decision reason when this is an upstream failure (``route=error``).
        self.error_reason = error_reason
        #: Set when the request itself is refused (not an upstream failure).
        self.reason = reason


def _connection_tokens(items: Iterable[tuple[str, str]]) -> set[str]:
    named: set[str] = set()
    for k, v in items:
        if k.lower() == "connection":
            named.update(t.strip().lower() for t in v.split(",") if t.strip())
    return named


def request_headers_for_upstream(headers: Mapping[str, str]) -> dict[str, str]:
    """Decision 3: what may travel to Anthropic (lower-case names)."""
    items = [(k.lower(), v) for k, v in headers.items()]
    named = _connection_tokens(items)
    out: dict[str, str] = {}
    for k, v in items:
        if k in HOP_BY_HOP or k in _REQUEST_DROP or k in named or k.startswith(_LMWARDEN_PREFIX):
            continue
        if k in _EDGE_EXACT or k.startswith(_EDGE_PREFIXES):
            continue
        out[k] = v
    # httpx would otherwise add `gzip, deflate` and raw gzip bytes would reach
    # a client that never asked for them.
    out.setdefault("accept-encoding", "identity")
    return out


def _client_header_items(items: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    items = [(k.lower(), v) for k, v in items]
    named = _connection_tokens(items)
    return [
        (k, v)
        for k, v in items
        if k not in HOP_BY_HOP and k not in named and k not in _RESPONSE_DROP
    ]


def response_headers_for_client(headers: Mapping[str, str]) -> dict[str, str]:
    """Hop-by-hop headers and Anthropic's origin state (cookies, HSTS, alt-svc,
    reporting) are dropped; ``content-encoding`` / ``-length`` stay consistent
    with the raw bytes we relay."""
    return dict(_client_header_items(headers.items()))


class UsageSniffer:
    """Best-effort ``usage`` from Anthropic's own response (decision 20).

    Keeps two integers and nothing else; never raises.
    """

    def __init__(self) -> None:
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self._buf = b""

    def _take(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        i, o = usage.get("input_tokens"), usage.get("output_tokens")
        if isinstance(i, int) and not isinstance(i, bool):
            self.input_tokens = i
        if isinstance(o, int) and not isinstance(o, bool):
            self.output_tokens = o

    def feed_json(self, body: bytes) -> None:
        try:
            if len(body) > USAGE_CAP:
                return
            obj = json.loads(body)
            if isinstance(obj, dict):
                self._take(obj.get("usage"))
        except Exception:  # noqa: BLE001 - sniffing must never fail a relay
            return

    def feed_sse(self, chunk: bytes) -> None:
        try:
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                self._line(line.rstrip(b"\r"))
            if len(self._buf) > USAGE_CAP:
                self._buf = b""  # an endless line: drop it, keep the memory bounded
        except Exception:  # noqa: BLE001
            self._buf = b""

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        if b"message_start" not in line and b"message_delta" not in line:
            return
        try:
            ev = json.loads(line[5:])
        except ValueError:
            return
        if not isinstance(ev, dict):
            return
        if ev.get("type") == "message_start":
            msg = ev.get("message")
            if isinstance(msg, dict):
                self._take(msg.get("usage"))
        elif ev.get("type") == "message_delta":
            self._take(ev.get("usage"))


def _upstream_url(request: Request, upstream_base: str) -> str:
    raw = request.scope.get("raw_path")
    path = (
        raw.decode("latin-1").split("?", 1)[0]
        if isinstance(raw, bytes) and raw
        else quote(request.url.path)
    )
    for seg in path.split("/"):
        # Decoded, so every spelling of "." / ".." (%2e, .%2e, ..%2f ...) is
        # caught, and so is a segment that smuggles a separator.
        plain = unquote(seg)
        if plain in (".", "..") or "/" in plain or "\\" in plain:
            raise RelayError(400, "router: this path is not relayed", reason="bad_path")
    query = request.scope.get("query_string") or b""
    url = upstream_base.rstrip("/") + path
    return url + ("?" + query.decode("latin-1") if query else "")


def _combined_request_headers(request: Request) -> dict[str, str]:
    combined: dict[str, str] = {}
    for k, v in request.headers.items():
        k = k.lower()
        combined[k] = f"{combined[k]}, {v}" if k in combined else v
    return combined


def _has_request_body(request: Request) -> bool:
    if request.method in ("GET", "HEAD"):
        return "content-length" in request.headers and request.headers["content-length"] != "0"
    cl = request.headers.get("content-length")
    return "transfer-encoding" in request.headers or (cl is not None and cl != "0")


def _encode_value(value: str) -> bytes:
    try:
        return value.encode("latin-1")
    except UnicodeEncodeError:
        return value.encode("utf-8")


async def relay(
    request: Request,
    *,
    upstream_base: str,
    body: bytes | None,
    credentials: Mapping[str, str],
    on_usage: Callable[[int | None, int | None], None] | None = None,
    on_ttfb: Callable[[], None] | None = None,
    on_close: Callable[[Response], None] | None = None,
) -> Response:
    """Send ``request`` to Anthropic and stream the answer back.

    ``body`` is the original bytes (``/v1/messages*``), or ``None`` to stream
    the request body through without buffering (the catch-all). Raises
    ``RelayError`` when Anthropic cannot be reached. ``on_close`` runs once
    when the body is finished or abandoned, with the client response.
    """
    url = _upstream_url(request, upstream_base)
    headers = request_headers_for_upstream(_combined_request_headers(request))
    headers.update({k.lower(): v for k, v in credentials.items()})
    content: Any = None
    if body is not None:
        content = body
    elif _has_request_body(request):
        content = request.stream()
        if "content-length" in request.headers:
            # known length: keep it rather than let httpx switch to chunked
            headers["content-length"] = request.headers["content-length"]
    client: httpx.AsyncClient = request.app.state.router_http
    try:
        upstream_request = client.build_request(
            request.method, url, headers=headers, content=content
        )
        resp = await client.send(upstream_request, stream=True)
    except httpx.PoolTimeout:
        raise RelayError(
            529,
            "router: too many concurrent relayed requests; retry shortly",
            error_reason="relay_pool_exhausted",
        ) from None
    except httpx.HTTPError as exc:
        # The message never includes the URL, headers or exception text.
        raise RelayError(
            502, f"router: Anthropic upstream unreachable ({type(exc).__name__})"
        ) from None

    ctype = resp.headers.get("content-type", "")
    encoded = resp.headers.get("content-encoding", "identity").lower() not in ("", "identity")
    sniffer = UsageSniffer() if on_usage is not None and not encoded else None
    is_sse = ctype.startswith("text/event-stream")
    is_json = ctype.startswith("application/json") and 200 <= resp.status_code < 300
    json_parts: list[bytes] = []
    json_size = 0
    finished = False
    holder: dict[str, Response] = {}

    def finish() -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        try:
            if sniffer is not None and on_usage is not None:
                if is_json and json_parts:
                    sniffer.feed_json(b"".join(json_parts))
                on_usage(sniffer.input_tokens, sniffer.output_tokens)
            if on_close is not None and "r" in holder:
                on_close(holder["r"])
        except Exception:  # noqa: BLE001 - bookkeeping never fails a relay
            logger.debug("router: relay bookkeeping failed", exc_info=True)

    async def aclose() -> None:
        try:
            await resp.aclose()
        finally:
            finish()

    async def body_iter() -> AsyncIterator[bytes]:
        nonlocal json_size
        first = True
        try:
            async for chunk in resp.aiter_raw():
                if first:
                    first = False
                    if on_ttfb is not None:
                        try:
                            on_ttfb()
                        except Exception:  # noqa: BLE001
                            pass
                if sniffer is not None:
                    if is_sse:
                        sniffer.feed_sse(chunk)
                    elif is_json and json_size <= USAGE_CAP:
                        json_size += len(chunk)
                        if json_size <= USAGE_CAP:
                            json_parts.append(chunk)
                yield chunk
        finally:
            await aclose()

    out = StreamingResponse(
        body_iter(),
        status_code=resp.status_code,
        media_type=None,
        background=BackgroundTask(aclose),
    )
    out.raw_headers = [
        (k.encode("latin-1"), _encode_value(v))
        for k, v in _client_header_items(resp.headers.multi_items())
    ]
    holder["r"] = out
    return out


__all__ = [
    "HOP_BY_HOP",
    "RelayError",
    "UsageSniffer",
    "make_relay_client",
    "relay",
    "request_headers_for_upstream",
    "response_headers_for_client",
]
