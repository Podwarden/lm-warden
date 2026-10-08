import httpx

from app.proxy.upstream import UpstreamClients


async def test_same_engine_run_reuses_one_client():
    u = UpstreamClients()
    a = u.get("127.0.0.1", 9000, 1, model_id="a")
    assert u.get("127.0.0.1", 9000, 1, model_id="a") is a
    assert isinstance(a, httpx.AsyncClient)
    await u.aclose_all()


async def test_new_generation_gets_new_client_and_closes_old():
    u = UpstreamClients()
    a = u.get("127.0.0.1", 9000, 1, model_id="a")
    b = u.get("127.0.0.1", 9000, 2, model_id="a")
    assert b is not a
    await u.aclose_all()
    assert a.is_closed and b.is_closed


async def test_two_models_on_same_port_and_generation_do_not_share():
    u = UpstreamClients()
    a = u.get("127.0.0.1", 9000, 1, model_id="a")
    b = u.get("127.0.0.1", 9000, 1, model_id="b")
    assert a is not b
    assert not a.is_closed and not b.is_closed
    await u.aclose_all()


async def test_stale_lookup_does_not_close_current_run():
    u = UpstreamClients()
    cur = u.get("127.0.0.1", 9000, 2, model_id="a")
    stale = u.get("127.0.0.1", 9000, 1, model_id="a")
    assert stale is not cur
    assert not cur.is_closed
    assert u.get("127.0.0.1", 9000, 2, model_id="a") is cur
    await u.aclose_all()


async def test_discard_model_closes_only_that_model():
    u = UpstreamClients()
    a = u.get("127.0.0.1", 9000, 1, model_id="a")
    b = u.get("127.0.0.1", 9001, 1, model_id="b")
    await u.discard_model("a")
    assert a.is_closed and not b.is_closed
    await u.aclose_all()


async def test_client_has_no_timeout_and_no_env_proxy():
    u = UpstreamClients()
    c = u.get("127.0.0.1", 9000, 1, model_id="a")
    assert c.timeout.read is None and c.timeout.connect is None
    # httpx exposes no public trust_env; the private attribute is the only handle.
    assert c._trust_env is False
    await u.aclose_all()


async def test_set_cookie_is_never_replayed():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"set-cookie": "sid=secret; Path=/"})

    u = UpstreamClients(transport=httpx.MockTransport(handler))
    c = u.get("127.0.0.1", 9000, 1, model_id="a")
    await c.get("http://127.0.0.1:9000/x")
    await c.get("http://127.0.0.1:9000/x")
    assert len(seen) == 2
    assert "cookie" not in seen[1].headers
    await u.aclose_all()


async def test_keepalive_expires_before_the_engines_idle_timeout():
    # uvicorn (the engine's server) closes an idle keep-alive connection after
    # 5 s; the pool must drop it first (#279 final review I2).
    u = UpstreamClients()
    c = u.get("127.0.0.1", 9000, 1, model_id="a")
    # httpx keeps Limits only on the transport's pool; this is the only handle.
    assert c._transport._pool._keepalive_expiry == 2.0
    await u.aclose_all()
