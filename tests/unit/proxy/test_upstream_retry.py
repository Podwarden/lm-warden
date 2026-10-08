"""#279 final review I2: a pooled keep-alive connection the engine closed while
idle fails the first send with RemoteProtocolError/ReadError before any
response; the proxy retries that send once on a fresh request.

#296: only a FAST failure is the stale keep-alive case. A send that failed
after the engine had the request for a while is not retried, so the engine
does not run the same prefill twice. The clock is the module's injectable
``_monotonic``; the fake sender advances it, so no test sleeps."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.proxy import routes
from tests.unit.proxy.test_dp_routing_forward import AUTH, _ready, _resp

BODY = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}


class _FakeClock:
    """A monotonic clock that only moves when a test advances it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    fake = _FakeClock()
    with patch("app.proxy.routes._monotonic", new=fake):
        yield fake


def _flaky_sender(failures, clock=None, fail_after_s=0.0):
    """Raise each of ``failures`` in turn, then answer 200. Each failure first
    advances ``clock`` by ``fail_after_s``: how long the failed send took."""
    seen: list[httpx.Request] = []
    pending = list(failures)

    async def _send(request, *, stream=False, **kw):
        seen.append(request)
        if pending:
            if clock is not None:
                clock.now += fail_after_s
            raise pending.pop(0)
        return _resp()

    return AsyncMock(side_effect=_send), seen


def _inflight(client):
    return client.app.state.scheduler._inflight_for_test("qwen")


@pytest.mark.parametrize(
    "exc",
    [
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.ReadError("connection reset"),
    ],
)
def test_a_stale_keepalive_failure_is_retried_once(tmp_data_dir, client, exc):
    _ready(client, tmp_data_dir, dp=1, affinity=0)
    send, seen = _flaky_sender([exc])
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post("/v1/chat/completions", headers=AUTH, json=BODY)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok"
    assert len(seen) == 2
    assert seen[0] is not seen[1]  # a fresh request object, not a re-sent one
    assert seen[1].content == seen[0].content
    assert _inflight(client) == 0


def test_two_failures_keep_the_existing_error_mapping(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1, affinity=0)
    send, seen = _flaky_sender(
        [httpx.RemoteProtocolError("first"), httpx.RemoteProtocolError("second")]
    )
    with patch("httpx.AsyncClient.send", new=send):
        # Unchanged: the send error is not mapped; it propagates (a 500 under a
        # real server; the test client re-raises it).
        with pytest.raises(httpx.RemoteProtocolError, match="second"):
            client.post("/v1/chat/completions", headers=AUTH, json=BODY)
    assert len(seen) == 2
    assert _inflight(client) == 0


def test_a_connect_error_is_not_retried(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1, affinity=0)
    send, seen = _flaky_sender([httpx.ConnectError("refused")])
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post("/v1/chat/completions", headers=AUTH, json=BODY)
    assert r.status_code == 502
    assert len(seen) == 1


def test_a_fast_failure_is_retried_exactly_once(tmp_data_dir, client, clock):
    _ready(client, tmp_data_dir, dp=1, affinity=0)
    send, seen = _flaky_sender(
        [httpx.RemoteProtocolError("stale")],
        clock=clock,
        fail_after_s=routes.STALE_KEEPALIVE_FAIL_S / 2,
    )
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post("/v1/chat/completions", headers=AUTH, json=BODY)
    assert r.status_code == 200
    assert len(seen) == 2
    assert _inflight(client) == 0


@pytest.mark.parametrize(
    "exc",
    [
        httpx.RemoteProtocolError("reset mid-prefill"),
        httpx.ReadError("reset mid-prefill"),
    ],
)
def test_a_slow_failure_is_not_retried(tmp_data_dir, client, clock, exc):
    _ready(client, tmp_data_dir, dp=1, affinity=0)
    send, seen = _flaky_sender([exc], clock=clock, fail_after_s=routes.STALE_KEEPALIVE_FAIL_S + 0.5)
    with patch("httpx.AsyncClient.send", new=send):
        # The existing mapping: the send error propagates (a 500 under a real
        # server; the test client re-raises it).
        with pytest.raises(type(exc), match="reset mid-prefill"):
            client.post("/v1/chat/completions", headers=AUTH, json=BODY)
    assert len(seen) == 1
    assert _inflight(client) == 0


def test_a_connect_error_is_never_retried_even_when_fast(tmp_data_dir, client, clock):
    _ready(client, tmp_data_dir, dp=1, affinity=0)
    send, seen = _flaky_sender([httpx.ConnectError("refused")], clock=clock, fail_after_s=0.0)
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post("/v1/chat/completions", headers=AUTH, json=BODY)
    assert r.status_code == 502
    assert len(seen) == 1
