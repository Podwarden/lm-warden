"""Pure parts of the Anthropic relay (#287 decisions 3, 13, 17, 20)."""

import json

import pytest

from app.router import passthrough as pt
from app.router.data_plane import local_failure_outcome

SENT = {
    "host": "warden.example",
    "content-length": "12",
    "cookie": "vw_session=abc",
    "connection": "x-foo, keep-alive",
    "x-foo": "bar",
    "keep-alive": "timeout=5",
    "proxy-authenticate": "x",
    "proxy-authorization": "x",
    "te": "trailers",
    "trailer": "x",
    "transfer-encoding": "chunked",
    "upgrade": "websocket",
    "x-lmwarden-key": "vw_SECRET",
    "x-lmwarden-trace": "t",
    "authorization": "Bearer sk-ant-FAKE",
    "x-api-key": "sk-ant-FAKE",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "a,b",
    "user-agent": "claude-cli/2",
    "x-stainless-os": "Linux",
    "accept": "application/json",
    "content-type": "application/json",
}


def test_request_filter_drops_everything_it_must():
    out = pt.request_headers_for_upstream(SENT)
    for gone in (
        "host",
        "content-length",
        "cookie",
        "connection",
        "x-foo",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-lmwarden-key",
        "x-lmwarden-trace",
        "authorization",
        "x-api-key",
    ):
        assert gone not in out, gone
    for kept in (
        "anthropic-version",
        "anthropic-beta",
        "user-agent",
        "x-stainless-os",
        "accept",
        "content-type",
    ):
        assert out[kept] == SENT[kept]


def test_request_filter_is_case_insensitive_on_input():
    out = pt.request_headers_for_upstream({"X-LMWarden-Key": "k", "Cookie": "c", "Accept": "x"})
    assert out == {"accept": "x", "accept-encoding": "identity"}


def test_hop_by_hop_set_is_complete():
    assert {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    } <= pt.HOP_BY_HOP


def test_accept_encoding_identity_when_absent_kept_when_present():
    assert pt.request_headers_for_upstream({})["accept-encoding"] == "identity"
    assert pt.request_headers_for_upstream({"accept-encoding": "gzip"})["accept-encoding"] == "gzip"


def test_response_filter_drops_hop_by_hop_only():
    out = pt.response_headers_for_client(
        {
            "Transfer-Encoding": "chunked",
            "Connection": "close, x-bar",
            "x-bar": "1",
            "Content-Encoding": "gzip",
            "Content-Length": "5",
            "Content-Type": "application/json",
            "request-id": "req_1",
        }
    )
    assert "transfer-encoding" not in out and "connection" not in out and "x-bar" not in out
    assert out["content-encoding"] == "gzip"
    assert out["content-length"] == "5"
    assert out["content-type"] == "application/json"
    assert out["request-id"] == "req_1"


def _sse(*events: dict) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def test_sniffer_sse_split_across_chunks():
    raw = _sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 11, "output_tokens": 1}}},
        {"type": "content_block_delta", "delta": {"text": "hi"}},
        {"type": "message_delta", "usage": {"output_tokens": 42}},
    )
    s = pt.UsageSniffer()
    for i in range(0, len(raw), 7):
        s.feed_sse(raw[i : i + 7])
    assert (s.input_tokens, s.output_tokens) == (11, 42)


def test_sniffer_json_body():
    s = pt.UsageSniffer()
    s.feed_json(json.dumps({"usage": {"input_tokens": 3, "output_tokens": 9}}).encode())
    assert (s.input_tokens, s.output_tokens) == (3, 9)


def test_sniffer_caps_and_garbage_never_raise():
    s = pt.UsageSniffer()
    big = json.dumps({"usage": {"input_tokens": 1, "output_tokens": 1}, "pad": "x" * (2 << 20)})
    s.feed_json(big.encode())
    assert (s.input_tokens, s.output_tokens) == (None, None)
    for junk in (b"\xff\xfe", b"data: {not json\n\n", b"data: [1,2]\n\n", b"\n\n\n"):
        s.feed_sse(junk)
        s.feed_json(junk)
    assert (s.input_tokens, s.output_tokens) == (None, None)
    s.feed_sse(b"data: " + b"x" * (3 << 20))  # an endless line must not grow memory unbounded
    s.feed_sse(b"\n\n")


@pytest.mark.parametrize(
    "status,fallback_rule,route,reason,breaker,out_status",
    [
        (500, True, "fallback", "status_500", True, None),
        (502, True, "fallback", "status_502", True, None),
        (503, True, "fallback", "status_503", True, None),
        (429, True, "fallback", "status_429", True, None),
        (400, True, "fallback", "status_400", False, None),
        (413, True, "fallback", "status_413", False, None),
        (500, False, "refused", "status_500", True, 529),
        (502, False, "refused", "status_502", True, 529),
        (429, False, "refused", "status_429", True, 529),
        (400, False, "refused", "status_400", False, 400),
        (413, False, "refused", "status_413", False, 400),
        (404, True, "local", None, False, 404),
        (404, False, "local", None, False, 404),
        (422, True, "local", None, False, 422),
    ],
)
def test_local_failure_outcome_table(status, fallback_rule, route, reason, breaker, out_status):
    o = local_failure_outcome(status, fallback_rule=fallback_rule, path="/v1/messages")
    assert (o.route, o.reason, o.breaker_failure) == (route, reason, breaker)
    if out_status is not None:
        assert o.status == out_status


@pytest.mark.parametrize("status", [400, 404, 413, 422, 429, 500, 502])
@pytest.mark.parametrize("fallback_rule", [True, False])
def test_count_tokens_always_falls_back_and_never_trips_breaker(status, fallback_rule):
    o = local_failure_outcome(status, fallback_rule=fallback_rule, path="/v1/messages/count_tokens")
    assert o.route == "fallback" and o.breaker_failure is False


# -- review #4: edge/proxy identity never reaches Anthropic, Anthropic's origin
# state never reaches the browser ----------------------------------------------

_EDGE = (
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-forwarded-port",
    "x-real-ip",
    "forwarded",
    "cf-connecting-ip",
    "cf-ray",
    "cf-ipcountry",
    "cf-visitor",
    "cf-worker",
    "cdn-loop",
    "true-client-ip",
)


@pytest.mark.parametrize("name", _EDGE)
def test_request_filter_drops_edge_and_proxy_identity(name):
    out = pt.request_headers_for_upstream({name: "1.2.3.4", "anthropic-version": "2023-06-01"})
    assert name not in out
    assert out["anthropic-version"] == "2023-06-01"


@pytest.mark.parametrize(
    "name",
    ["set-cookie", "set-cookie2", "alt-svc", "strict-transport-security", "report-to", "nel"],
)
def test_response_filter_drops_origin_state(name):
    out = pt.response_headers_for_client({name: "x", "request-id": "req_1"})
    assert name not in out
    assert out["request-id"] == "req_1"


def test_response_filter_drops_every_set_cookie_value():
    items = [("Set-Cookie", "a=1"), ("set-cookie", "b=2"), ("x-ok", "1")]
    assert pt._client_header_items(items) == [("x-ok", "1")]


# -- review #2: dot segments in any encoding -----------------------------------


class _Req:
    def __init__(self, raw: bytes, query: bytes = b"") -> None:
        self.scope = {"raw_path": raw, "query_string": query}

        class _U:
            path = raw.decode("latin-1")

        self.url = _U()


@pytest.mark.parametrize(
    "path",
    [
        b"/v1/../admin",
        b"/v1/./x",
        b"/v1/%2e%2e/admin",
        b"/v1/%2E%2e/admin",
        b"/v1/%2E%2e/admin",
        b"/v1/.%2e/x",
        b"/v1/%2e./x",
        b"/v1/..%2fx",
        b"/v1/..%2Fx",
        b"/v1/a%2fb",
        b"/v1/a%5cb",
        b"/v1/..%5cx",
        b"/v1/a\\b",
        b"/v1/%2e",
    ],
)
def test_upstream_url_refuses_dot_and_slash_segments(path):
    with pytest.raises(pt.RelayError) as ei:
        pt._upstream_url(_Req(path), "https://anthropic.example")
    assert ei.value.status == 400 and ei.value.reason == "bad_path"


@pytest.mark.parametrize(
    "path", [b"/v1/models/claude-x", b"/v1/messages/batches/msgbatch_01/results", b"/v1/a.b/c..d"]
)
def test_upstream_url_keeps_ordinary_paths_verbatim(path):
    url = pt._upstream_url(_Req(path, b"limit=2"), "https://anthropic.example/")
    assert url == "https://anthropic.example" + path.decode() + "?limit=2"
