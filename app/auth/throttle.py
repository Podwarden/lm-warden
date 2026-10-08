"""Brute-force throttles for the unauthenticated credential checks.

Two mechanisms, and the rule that picks between them:

* a LOCK refuses the attempt (``429`` + ``Retry-After``) before the password
  or secret is even checked;
* a THROTTLE checks the attempt as usual and, when it failed, holds the
  answer back for a moment (an escalating delay, capped at a few seconds).
  It never refuses a correct password: success is answered at once.

A lock is only ever keyed on an address an attacker cannot aim at somebody
else: a PUBLIC client address that the trusted-proxy chain vouches for
(``app/utils/client_ip.py::resolve_client``, "attributable"). When the api
cannot tell its visitors apart -- the address is a trusted proxy's own
(X-Forwarded-For missing or only proxy hops), private, loopback, link-local,
CGNAT, garbage or absent -- a lock on it would lock everybody behind that
proxy, which is exactly what v2026.09.23.2 did on a deployment whose outer
proxy dropped the client address. Such an attempt is "unattributable": it
is throttled, never locked. The same goes for usernames: anybody can type
the admin's name, so the username key is throttled, never locked.

``LoginThrottle`` -- ``POST /api/auth/login``:

* attributable address: 5 failures lock the address for 60s, and every
  further failure doubles the lock (120s, 240s, ...) up to 15 min.
* unattributable address: no lock. After 5 failures on that (shared) address
  every further failure is answered after 0.5s, doubling up to 3s.
* every username: no lock. After 3 failures on the name every further
  failure is answered after 0.5s, doubling up to 3s. The larger of the two
  delays applies.
* at most 32 failed attempts are held back at once (per process). A failure
  beyond that is answered at once with ``429`` + ``Retry-After`` instead of
  being held, so a flood cannot pile up sleeping sockets. Only a failed
  attempt is ever held or answered like this; a correct password never is.

A failure count is forgotten after 15 quiet minutes (no failure, and no lock
still running). A successful login clears the address and username counts
(a shared unattributable address's count is left alone: the success says
nothing about the other people behind it). A request refused by a lock is
not a failure: it neither extends nor escalates the lock.

``BearerThrottle`` -- ``vw_`` inference keys on /v1 and ``vwa_`` admin tokens
on /api. The secrets carry 280 random bits, so guessing one is hopeless and
this is not what stops that; it stops a client spraying garbage from costing
a database lookup per request. 20 secrets that match no row from one
attributable address lock it for 30s, doubling up to 15 min. An
unattributable address is never locked (it would take /v1 down for every
client behind that proxy). A secret that has matched a row since the process
started is never throttled (whatever the address), and a revoked key that
keeps being retried gets its normal 401, not a lock on its address.

Unattributable traffic behind a trusted proxy almost always means the outer
proxy does not pass the client address on; the first such request logs one
warning per process saying so.

Memory is bounded: every table is an LRU with a fixed number of keys, and an
IPv6 client is keyed by its /64 (one subscriber's allocation), so neither
random usernames nor random addresses grow it past a few MB. Under a flood
the least recently touched entries are forgotten first.

Not throttled, deliberately:

* ``POST /api/auth/refresh`` -- the cookie is an HMAC-signed JWT, so there is
  nothing to guess, and a browser with an expired cookie calls it on every
  page load; throttling it would only lock out returning users.
* session access JWTs and SSE tickets -- signed or random, same reasoning.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from fastapi import HTTPException, status
from starlette.requests import HTTPConnection

from app.utils.client_ip import DEFAULT_TRUSTED_PROXIES, ClientAddress, resolve_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LockPolicy:
    #: The failure that starts the first lock.
    threshold: int
    #: Length of the first lock; each further failure doubles it.
    base_lock_s: float
    max_lock_s: float
    #: A count is forgotten after this long with no failure and no lock.
    forget_after_s: float


@dataclass(frozen=True)
class DelayPolicy:
    #: Failures answered without any delay.
    grace: int
    #: Delay of the first failure past the grace; each further one doubles it.
    base_delay_s: float
    max_delay_s: float
    #: A count is forgotten after this long with no failure.
    forget_after_s: float


LOGIN_IP_POLICY = LockPolicy(threshold=5, base_lock_s=60, max_lock_s=900, forget_after_s=900)
LOGIN_USER_DELAY = DelayPolicy(grace=3, base_delay_s=0.5, max_delay_s=3.0, forget_after_s=900)
LOGIN_SHARED_DELAY = DelayPolicy(grace=5, base_delay_s=0.5, max_delay_s=3.0, forget_after_s=900)
BEARER_IP_POLICY = LockPolicy(threshold=20, base_lock_s=30, max_lock_s=900, forget_after_s=900)

#: Failed logins held back (sleeping) at once, per process.
MAX_DELAYED = 32
#: Keys per table. An entry is a few hundred bytes.
MAX_KEYS = 10_000
#: Digests of bearer secrets that matched a token row.
MAX_KNOWN_SECRETS = 4_096

BEARER_THROTTLED_MESSAGE = "too many requests with unknown tokens; retry later"
LOGIN_THROTTLED_MESSAGE = "too many failed login attempts; retry later"

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class _Entry:
    __slots__ = ("failures", "last_failure", "locked_until")

    def __init__(self) -> None:
        self.failures = 0
        self.last_failure = 0.0
        self.locked_until = 0.0


class _Counts:
    """Failure counts per key in a bounded LRU, forgotten after a quiet
    period."""

    def __init__(self, forget_after_s: float, *, max_keys: int, clock: Clock) -> None:
        self.forget_after_s = forget_after_s
        self.max_keys = max_keys
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def _live(self, key: str, now: float) -> _Entry | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        quiet_since = max(entry.last_failure, entry.locked_until)
        if now - quiet_since > self.forget_after_s:
            del self._entries[key]
            return None
        return entry

    def _record(self, key: str, now: float) -> _Entry:
        entry = self._live(key, now)
        if entry is None:
            entry = _Entry()
            self._entries[key] = entry
        entry.failures += 1
        entry.last_failure = now
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_keys:
            self._entries.popitem(last=False)
        return entry

    def failures(self, key: str) -> int:
        entry = self._live(key, self._clock())
        return 0 if entry is None else entry.failures

    def reset(self, key: str) -> None:
        self._entries.pop(key, None)


class FailureTracker(_Counts):
    """Failure counts and locks per key, with escalation, in a bounded LRU."""

    def __init__(
        self, policy: LockPolicy, *, max_keys: int = MAX_KEYS, clock: Clock = time.monotonic
    ) -> None:
        super().__init__(policy.forget_after_s, max_keys=max_keys, clock=clock)
        self.policy = policy

    def retry_after(self, key: str) -> float:
        """Seconds until ``key`` may try again; 0 when it is not locked."""
        now = self._clock()
        entry = self._live(key, now)
        if entry is None:
            return 0.0
        return max(0.0, entry.locked_until - now)

    def record_failure(self, key: str) -> None:
        now = self._clock()
        entry = self._record(key, now)
        over = entry.failures - self.policy.threshold
        if over >= 0:
            lock = self.policy.base_lock_s * (2 ** min(over, 32))
            entry.locked_until = now + min(lock, self.policy.max_lock_s)


class DelayTracker(_Counts):
    """Failure counts per key, turned into an escalating answer delay. Never
    a lock: there is no state in which this refuses anything."""

    def __init__(
        self, policy: DelayPolicy, *, max_keys: int = MAX_KEYS, clock: Clock = time.monotonic
    ) -> None:
        super().__init__(policy.forget_after_s, max_keys=max_keys, clock=clock)
        self.policy = policy

    def record_failure(self, key: str) -> float:
        """Count a failure; return how long to hold its answer back."""
        entry = self._record(key, self._clock())
        over = entry.failures - self.policy.grace
        if over <= 0:
            return 0.0
        delay = self.policy.base_delay_s * (2 ** min(over - 1, 32))
        return float(min(delay, self.policy.max_delay_s))


class DelayGate:
    """Holds failed attempts back, at most ``max_concurrent`` at once."""

    def __init__(self, max_concurrent: int = MAX_DELAYED, *, sleep: Sleep = asyncio.sleep) -> None:
        self.max_concurrent = max_concurrent
        self.active = 0
        self._sleep = sleep

    async def hold(self, seconds: float) -> bool:
        """Sleep ``seconds``; False (at once) when the gate is full."""
        if seconds <= 0:
            return True
        if self.active >= self.max_concurrent:
            return False
        self.active += 1
        try:
            await self._sleep(seconds)
        finally:
            self.active -= 1
        return True


def address_key(address: str) -> str:
    """The throttle key of a client address: IPv6 by /64, anything else as
    given (an unparseable peer, e.g. the ASGI test client, keys as itself)."""
    try:
        addr = ipaddress.ip_address(address)
    except ValueError:
        return address[:64]
    if isinstance(addr, ipaddress.IPv6Address):
        return str(ipaddress.IPv6Network((addr, 64), strict=False))
    return str(addr)


def username_key(username: str) -> str:
    return username.strip().casefold()[:64]


UNATTRIBUTABLE_WARNING = (
    "brute-force throttle: requests arrive through a trusted proxy (%s) "
    "without a usable client address (%s: %s). X-Forwarded-For missing or only "
    "private hops -- configure the outer proxy to pass the client address; see "
    "VW_TRUSTED_PROXIES. Until then failed logins are only slowed down, never "
    "locked, and API-key spraying is not locked either."
)
_unattributable_warned = False


def _warn_unattributable(conn: HTTPConnection, client: ClientAddress) -> None:
    """Once per process: behind a trusted proxy, and it told us nothing."""
    global _unattributable_warned
    if _unattributable_warned or client.attributable or not client.via_trusted_proxy:
        return
    _unattributable_warned = True
    peer = conn.client.host if conn.client else "?"
    logger.warning(UNATTRIBUTABLE_WARNING, peer, client.address, client.reason)


def throttle_client(conn: HTTPConnection) -> ClientAddress:
    """The client a throttle decision is keyed on (proxy-aware only for
    peers in VW_TRUSTED_PROXIES; see app/utils/client_ip.py), its address
    already turned into a throttle key."""
    settings = getattr(conn.app.state, "settings", None)
    trusted = getattr(settings, "trusted_proxies", DEFAULT_TRUSTED_PROXIES)
    client = resolve_client(conn, trusted)
    _warn_unattributable(conn, client)
    return replace(client, address=address_key(client.address))


def too_many_attempts(retry_after: float, message: str) -> HTTPException:
    """429 with a whole-second Retry-After (never 0: that reads as "now")."""
    seconds = max(1, math.ceil(retry_after))
    return HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        detail=message,
        headers={"Retry-After": str(seconds)},
    )


class LoginThrottle:
    def __init__(
        self,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        max_keys: int = MAX_KEYS,
        max_delayed: int = MAX_DELAYED,
    ) -> None:
        #: Locks, attributable addresses only.
        self.by_address = FailureTracker(LOGIN_IP_POLICY, max_keys=max_keys, clock=clock)
        #: Delays, unattributable (shared) addresses.
        self.by_shared_address = DelayTracker(LOGIN_SHARED_DELAY, max_keys=max_keys, clock=clock)
        #: Delays, every username.
        self.by_username = DelayTracker(LOGIN_USER_DELAY, max_keys=max_keys, clock=clock)
        self.gate = DelayGate(max_delayed, sleep=sleep)

    def retry_after(self, client: ClientAddress) -> float:
        """Seconds this attempt is locked out for; 0 when it may go ahead.
        Only an attributable address can be locked."""
        if not client.attributable:
            return 0.0
        return self.by_address.retry_after(client.address)

    def failure(self, client: ClientAddress, username: str) -> float:
        """Count a failed attempt; return how long to hold its answer back."""
        delay = self.by_username.record_failure(username_key(username))
        if client.attributable:
            self.by_address.record_failure(client.address)
        else:
            delay = max(delay, self.by_shared_address.record_failure(client.address))
        return delay

    async def failed(self, client: ClientAddress, username: str) -> None:
        """Count a failed attempt and hold its answer back. Raises 429 when
        too many failed attempts are already being held."""
        delay = self.failure(client, username)
        if not await self.gate.hold(delay):
            raise too_many_attempts(delay, LOGIN_THROTTLED_MESSAGE)

    def success(self, client: ClientAddress, username: str) -> None:
        self.by_username.reset(username_key(username))
        if client.attributable:
            self.by_address.reset(client.address)


def _secret_digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class BearerThrottle:
    def __init__(self, *, clock: Clock = time.monotonic, max_keys: int = MAX_KEYS) -> None:
        self.by_address = FailureTracker(BEARER_IP_POLICY, max_keys=max_keys, clock=clock)
        self._known: OrderedDict[str, None] = OrderedDict()

    def retry_after(self, address: str | None, secret: str) -> float:
        """``address`` is None for an unattributable client: never locked."""
        if address is None or _secret_digest(secret) in self._known:
            return 0.0
        return self.by_address.retry_after(address)

    def unknown(self, address: str | None) -> None:
        """A secret matched no token row (``address`` as ``retry_after``)."""
        if address is not None:
            self.by_address.record_failure(address)

    def matched(self, secret: str) -> None:
        """A secret matched a token row (valid or not: it was not a guess)."""
        digest = _secret_digest(secret)
        self._known[digest] = None
        self._known.move_to_end(digest)
        while len(self._known) > MAX_KNOWN_SECRETS:
            self._known.popitem(last=False)


def login_throttle(conn: HTTPConnection) -> LoginThrottle:
    state = conn.app.state
    throttle = getattr(state, "login_throttle", None)
    if throttle is None:
        throttle = state.login_throttle = LoginThrottle()
    return throttle


def bearer_throttle(conn: HTTPConnection) -> BearerThrottle:
    state = conn.app.state
    throttle = getattr(state, "bearer_throttle", None)
    if throttle is None:
        throttle = state.bearer_throttle = BearerThrottle()
    return throttle


def check_bearer(conn: HTTPConnection, secret: str) -> str | None:
    """Refuse with 429 before the token lookup when this address is locked
    for unknown secrets (and ``secret`` is not one already seen to match).
    Returns the address key for ``BearerThrottle.unknown`` -- None when the
    client is unattributable, which is never locked."""
    client = throttle_client(conn)
    address = client.address if client.attributable else None
    wait = bearer_throttle(conn).retry_after(address, secret)
    if wait > 0:
        raise too_many_attempts(wait, BEARER_THROTTLED_MESSAGE)
    return address
