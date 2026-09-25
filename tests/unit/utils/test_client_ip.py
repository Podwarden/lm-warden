"""trusted_client_ip (app/utils/client_ip.py): X-Forwarded-For is believed only
from a trusted proxy, read right to left, and never past a hop the trusted
proxies did not vouch for."""

from __future__ import annotations

import pytest
from starlette.requests import Request

from app.config import _parse_trusted_proxies
from app.utils.client_ip import (
    DEFAULT_TRUSTED_PROXIES,
    client_ip,
    resolve_client,
    trusted_client_ip,
)


def _conn(peer: str | None, *headers: tuple[str, str]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
        "client": (peer, 1234) if peer else None,
    }
    return Request(scope)


T = DEFAULT_TRUSTED_PROXIES


def test_untrusted_peer_is_the_client_whatever_the_headers_say():
    c = _conn("203.0.113.7", ("X-Forwarded-For", "10.0.0.1"), ("X-Real-Ip", "10.0.0.2"))
    assert trusted_client_ip(c, T) == "203.0.113.7"
    # The display helper still reports the claim; that is its job.
    assert client_ip(c) == "10.0.0.1"


def test_trusted_proxy_reports_the_client():
    c = _conn("172.18.0.5", ("X-Forwarded-For", "203.0.113.7"))
    assert trusted_client_ip(c, T) == "203.0.113.7"


def test_client_written_hops_left_of_the_proxys_own_are_ignored():
    c = _conn("172.18.0.5", ("X-Forwarded-For", "198.51.100.1, 203.0.113.7"))
    assert trusted_client_ip(c, T) == "203.0.113.7"


def test_a_chain_of_trusted_proxies_is_walked():
    # client -> Traefik (10.42.0.9) -> Caddy (172.18.0.5) -> api
    c = _conn("172.18.0.5", ("X-Forwarded-For", "203.0.113.7, 10.42.0.9"))
    assert trusted_client_ip(c, T) == "203.0.113.7"


def test_repeated_headers_are_one_chain():
    c = _conn("172.18.0.5", ("X-Forwarded-For", "198.51.100.1"),
              ("X-Forwarded-For", "203.0.113.7"))
    assert trusted_client_ip(c, T) == "203.0.113.7"


def test_x_real_ip_only_without_x_forwarded_for():
    assert trusted_client_ip(_conn("127.0.0.1", ("X-Real-Ip", "203.0.113.7")), T) == "203.0.113.7"


def test_no_headers_means_the_peer():
    assert trusted_client_ip(_conn("172.18.0.5"), T) == "172.18.0.5"


def test_garbage_where_the_proxy_should_have_written_keys_on_the_proxy():
    c = _conn("172.18.0.5", ("X-Forwarded-For", "203.0.113.7, not-an-ip"))
    assert trusted_client_ip(c, T) == "172.18.0.5"


def test_ports_brackets_and_mapped_addresses():
    c = _conn("::ffff:172.18.0.5", ("X-Forwarded-For", "[2001:db8::1]:443"))
    assert trusted_client_ip(c, T) == "2001:db8::1"
    c = _conn("127.0.0.1", ("X-Forwarded-For", "203.0.113.7:5555"))
    assert trusted_client_ip(c, T) == "203.0.113.7"


def test_trusting_nobody_keys_on_the_socket():
    c = _conn("172.18.0.5", ("X-Forwarded-For", "203.0.113.7"))
    assert trusted_client_ip(c, ()) == "172.18.0.5"


def test_the_asgi_test_client_peer_is_kept_as_is():
    assert trusted_client_ip(_conn("testclient", ("X-Forwarded-For", "1.2.3.4")), T) == "testclient"
    assert trusted_client_ip(_conn(None), T) == "unknown"


def test_the_env_knob():
    assert _parse_trusted_proxies("") == DEFAULT_TRUSTED_PROXIES
    assert _parse_trusted_proxies("none") == ()
    assert _parse_trusted_proxies(" 10.1.0.0/16, 192.0.2.4 ") == ("10.1.0.0/16", "192.0.2.4")
    with pytest.raises(ValueError, match="VW_TRUSTED_PROXIES"):
        _parse_trusted_proxies("10.0.0.0/33")


# --- resolve_client: may a lock be keyed on this address? ----------------------


@pytest.mark.parametrize(
    "peer,headers",
    [
        pytest.param("203.0.113.7", (), id="public-peer-direct"),
        pytest.param("172.18.0.5", (("X-Forwarded-For", "203.0.113.7"),), id="public-via-proxy"),
        pytest.param("172.18.0.5", (("X-Forwarded-For", "203.0.113.7, 10.42.0.9"),),
                     id="public-via-proxy-chain"),
        pytest.param("127.0.0.1", (("X-Real-Ip", "2001:db8::1"),), id="public-x-real-ip"),
    ],
)
def test_a_public_address_the_chain_vouches_for_is_attributable(peer, headers):
    c = resolve_client(_conn(peer, *headers), T)
    assert c.attributable, c
    assert c.reason == ""


@pytest.mark.parametrize(
    "peer,headers,trusted",
    [
        pytest.param("172.18.0.5", (), T, id="trusted-proxy-no-xff"),
        pytest.param("172.18.0.5", (("X-Forwarded-For", "10.42.0.9"),), T, id="only-proxy-hops"),
        pytest.param("172.18.0.5", (("X-Forwarded-For", "100.64.1.2"),), T, id="cgnat-hop"),
        pytest.param("172.18.0.5", (("X-Forwarded-For", "169.254.1.1"),), T, id="link-local-hop"),
        pytest.param("172.18.0.5", (("X-Forwarded-For", "not-an-ip"),), T, id="garbage-hop"),
        pytest.param("172.18.0.5", (("X-Forwarded-For", "fe80::1"),), T, id="v6-link-local-hop"),
        pytest.param("192.168.1.20", (), (), id="untrusted-private-peer"),
        pytest.param("::1", (), (), id="untrusted-loopback-peer"),
        pytest.param("100.64.0.9", (), T, id="untrusted-cgnat-peer"),
        pytest.param("198.51.100.0", (), ("198.51.100.0/24",), id="public-but-trusted-peer-no-xff"),
        pytest.param("testclient", (), T, id="unparseable-peer"),
        pytest.param(None, (), T, id="no-peer"),
    ],
)
def test_anything_many_visitors_may_share_is_unattributable(peer, headers, trusted):
    c = resolve_client(_conn(peer, *headers), trusted)
    assert not c.attributable, c
    assert c.reason


def test_via_trusted_proxy_names_whether_a_proxy_is_in_front():
    assert resolve_client(_conn("172.18.0.5"), T).via_trusted_proxy
    assert not resolve_client(_conn("192.168.1.20"), ()).via_trusted_proxy
    assert not resolve_client(_conn("203.0.113.7"), T).via_trusted_proxy
