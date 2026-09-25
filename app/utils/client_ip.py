"""Who sent a request: the forgeable view and the trusted one.

Two functions, for two different jobs:

* ``client_ip`` -- the address the request CLAIMS to come from. Used where the
  value is only displayed or recorded next to the socket peer (live request
  stats, request history, the admin-token audit trail). It believes any
  X-Forwarded-For, which is fine for a label and wrong for a decision.

* ``trusted_client_ip`` -- the address a decision may be keyed on (the login
  and bearer throttles in app/auth/throttle.py). X-Forwarded-For is honoured
  only when the socket peer is a trusted proxy (VW_TRUSTED_PROXIES), and the
  chain is read right to left, stopping at the first hop that is not itself a
  trusted proxy. That hop is the one a trusted proxy actually saw; everything
  to its left was written by the client and is ignored. A client that talks to
  the api directly therefore cannot pick its own address by sending the
  header, and a client behind Caddy cannot either: Caddy appends the real peer
  on the right.

* ``resolve_client`` -- ``trusted_client_ip`` plus the verdict a lock needs:
  is the address ATTRIBUTABLE, i.e. a public address that the trusted-proxy
  chain actually vouches for? A private, loopback, link-local or CGNAT
  address, a trusted proxy's own address (X-Forwarded-For missing, or only
  proxy hops in it), an unparseable hop or no peer at all is not: many
  visitors can arrive as that one address, so a lock keyed on it is a button
  an attacker can press for everyone. See app/auth/throttle.py.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from functools import lru_cache

from starlette.requests import HTTPConnection

IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

#: Peers trusted to report the client's address when VW_TRUSTED_PROXIES is
#: unset: loopback and the private ranges. Every supported topology puts a
#: proxy on a private network in front of the api -- Caddy on the compose
#: network, Traefik on the cluster network -- and without trusting it every
#: visitor would share the proxy's address, so one attacker's failed logins
#: would lock out everybody. A deployment that exposes the api port directly
#: to a private network it does not control should set the variable.
DEFAULT_TRUSTED_PROXIES: tuple[str, ...] = (
    "127.0.0.0/8",
    "::1/128",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "fc00::/7",
)


@lru_cache(maxsize=16)
def parse_trusted_proxies(specs: tuple[str, ...]) -> tuple[IPNetwork, ...]:
    """CIDRs (or bare addresses) to networks. Raises ValueError on a bad one,
    so a typo in VW_TRUSTED_PROXIES fails the process at start-up instead of
    silently trusting nobody (or everybody)."""
    return tuple(ipaddress.ip_network(s.strip(), strict=False) for s in specs)


#: Addresses that do not name one client on the public internet: many
#: machines share each of them (NAT, a proxy, a container network), so none
#: of them may key a lock. The IETF documentation ranges (192.0.2.0/24 and
#: friends) are deliberately NOT here: nothing routes them, so no real
#: request carries one, and the tests use them to stand for public clients.
NON_PUBLIC_NETWORKS: tuple[IPNetwork, ...] = tuple(
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",  # "this network"
        "10.0.0.0/8",
        "100.64.0.0/10",  # carrier-grade NAT (RFC 6598)
        "127.0.0.0/8",
        "169.254.0.0/16",  # link-local
        "172.16.0.0/12",
        "192.0.0.0/24",  # IETF protocol assignments
        "192.168.0.0/16",
        "224.0.0.0/4",  # multicast
        "240.0.0.0/4",  # reserved, and the broadcast address
        "::/128",
        "::1/128",
        "fc00::/7",  # unique local
        "fe80::/10",  # link-local
        "ff00::/8",  # multicast
    )
)


@dataclass(frozen=True)
class ClientAddress:
    """Who a request came from, as far as a lock may rely on it."""

    #: ``trusted_client_ip``'s answer.
    address: str
    #: True only for a public address the trusted-proxy chain vouches for.
    attributable: bool
    #: The socket peer is a trusted proxy (VW_TRUSTED_PROXIES).
    via_trusted_proxy: bool
    #: Why the address is not attributable ("" when it is).
    reason: str = ""


def _parse_ip(raw: str | None) -> IPAddress | None:
    """An address from a socket peer or a forwarded hop, or None. Tolerates
    the ``1.2.3.4:5678`` and ``[2001:db8::1]:443`` spellings some proxies
    write, and unwraps IPv4-mapped IPv6."""
    if not raw:
        return None
    value = raw.strip().strip('"')
    if value.startswith("["):
        value = value[1:].split("]", 1)[0]
    elif value.count(":") == 1:
        value = value.split(":", 1)[0]
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return None
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def _trusted(addr: IPAddress, nets: tuple[IPNetwork, ...]) -> bool:
    return any(addr.version == net.version and addr in net for net in nets)


def _public(addr: IPAddress) -> bool:
    return not any(addr.version == net.version and addr in net for net in NON_PUBLIC_NETWORKS)


def client_ip(conn: HTTPConnection) -> str | None:
    """First X-Forwarded-For hop, else X-Real-Ip, else the socket peer.

    Shared by the proxy's live request registry and request history
    (app/proxy/routes.py) and the admin-token audit trail
    (app/auth/admin_audit.py), so both record the same address. The headers
    are whatever the reverse proxy (or the client) sent, so this can be
    forged; the audit trail keeps the socket peer next to it. Never key a
    security decision on it -- use ``trusted_client_ip``.
    """
    xff = conn.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip() or None
    xri = conn.headers.get("x-real-ip")
    if xri:
        return xri.strip() or None
    return conn.client.host if conn.client else None


def trusted_client_ip(conn: HTTPConnection, trusted: tuple[str, ...]) -> str:
    """The client's address as far as the trusted proxies vouch for it.

    * The socket peer is not a trusted proxy: the peer, whatever the headers
      say.
    * It is: walk X-Forwarded-For from the right, skipping trusted proxies;
      the first other hop is the client. If every hop is a trusted proxy, the
      left-most one. With no X-Forwarded-For, X-Real-Ip (a single value the
      proxy set) and then the peer.

    Always returns a string: an unparseable peer (the ASGI test client says
    "testclient") is returned as-is, and a missing one as "unknown".
    """
    return resolve_client(conn, trusted).address


def resolve_client(conn: HTTPConnection, trusted: tuple[str, ...]) -> ClientAddress:
    """``trusted_client_ip``'s address, and whether a lock may be keyed on it
    (see ``ClientAddress.attributable`` and the module docstring)."""
    peer_raw = conn.client.host if conn.client else None
    peer = _parse_ip(peer_raw)
    nets = parse_trusted_proxies(trusted)
    if peer is None:
        return ClientAddress(peer_raw or "unknown", False, False, "no usable socket peer")
    if not _trusted(peer, nets):
        if _public(peer):
            return ClientAddress(str(peer), True, False)
        return ClientAddress(
            str(peer), False, False,
            "the socket peer is a private address that is not in VW_TRUSTED_PROXIES",
        )

    def vouched(addr: IPAddress) -> ClientAddress:
        if _trusted(addr, nets):
            return ClientAddress(
                str(addr), False, True,
                "X-Forwarded-For names only trusted proxies",
            )
        if not _public(addr):
            return ClientAddress(
                str(addr), False, True,
                "the forwarded client address is a private address",
            )
        return ClientAddress(str(addr), True, True)

    hops = [
        h.strip()
        for value in conn.headers.getlist("x-forwarded-for")
        for h in value.split(",")
        if h.strip()
    ]
    if hops:
        for hop in reversed(hops):
            addr = _parse_ip(hop)
            if addr is None:
                # Garbage where a trusted proxy should have written the
                # address it saw. Do not guess past it: key on the proxy.
                return ClientAddress(
                    str(peer), False, True, "X-Forwarded-For has an unparseable hop"
                )
            if not _trusted(addr, nets):
                return vouched(addr)
        first = _parse_ip(hops[0])
        return vouched(first) if first is not None else vouched(peer)

    real = _parse_ip(conn.headers.get("x-real-ip"))
    if real is not None:
        return vouched(real)
    return ClientAddress(
        str(peer), False, True, "a trusted proxy sent no X-Forwarded-For"
    )
