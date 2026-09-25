"""Brute-force throttles (app/auth/throttle.py): the state machine with a fake
clock, then the login route and the two bearer checks end to end.

The rule under test: a LOCK (429 before the check) only ever keys on a public
client address the trusted proxies vouch for; everything else -- a username,
or an address many visitors may share -- is only THROTTLED (a failed attempt
is answered late), and a correct password is never refused by a throttle.

The ASGI test client's peer is the string "testclient", which is not an
address, so it is unattributable. Tests that need a real peer (a trusted
proxy, a public client, two different clients) go through ``_peer_client``,
which sets the socket peer from a test-only header before the app sees the
request. Every route test installs a LoginThrottle with a recording fake
sleep, so no test waits for real.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import pytest
from fastapi.testclient import TestClient

import app.auth.throttle as throttle_mod
from app.auth.throttle import (
    BEARER_IP_POLICY,
    LOGIN_IP_POLICY,
    LOGIN_SHARED_DELAY,
    LOGIN_USER_DELAY,
    MAX_DELAYED,
    BearerThrottle,
    DelayGate,
    DelayTracker,
    FailureTracker,
    LockPolicy,
    LoginThrottle,
    address_key,
)
from app.utils.client_ip import ClientAddress
from tests.conftest import bearer, seed_admin_token

PASSWORD = "hunter2"  # conftest.seed_admin_user's; predates the 12-char rule


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleep:
    """Records every hold instead of sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


PUBLIC = ClientAddress("203.0.113.7", True, False)
SHARED = ClientAddress("172.18.0.5", False, True, "a trusted proxy sent no X-Forwarded-For")


# --- FailureTracker ------------------------------------------------------------


def test_no_lock_below_the_threshold():
    clock = FakeClock()
    t = FailureTracker(LockPolicy(5, 60, 900, 900), clock=clock)
    for _ in range(4):
        t.record_failure("k")
    assert t.retry_after("k") == 0


def test_the_threshold_failure_locks_and_each_further_one_doubles():
    clock = FakeClock()
    t = FailureTracker(LockPolicy(5, 60, 900, 900), clock=clock)
    for _ in range(5):
        t.record_failure("k")
    assert t.retry_after("k") == 60
    clock.advance(60)
    assert t.retry_after("k") == 0
    locks = []
    for _ in range(5):
        t.record_failure("k")
        locks.append(t.retry_after("k"))
        clock.advance(locks[-1])
    assert locks == [120, 240, 480, 900, 900]  # capped at max_lock_s


def test_a_count_is_forgotten_after_a_quiet_period():
    clock = FakeClock()
    t = FailureTracker(LockPolicy(5, 60, 900, 900), clock=clock)
    for _ in range(5):
        t.record_failure("k")
    # Quiet is measured from the END of the lock, not the last failure.
    clock.advance(60 + 899)
    t.record_failure("k")
    assert t.retry_after("k") == 120, "still counting: 6th failure"
    clock.advance(120 + 901)
    t.record_failure("k")
    assert t.retry_after("k") == 0, "forgotten: this is failure #1 again"


def test_reset_clears_the_count():
    clock = FakeClock()
    t = FailureTracker(LockPolicy(5, 60, 900, 900), clock=clock)
    for _ in range(5):
        t.record_failure("k")
    t.reset("k")
    assert t.retry_after("k") == 0
    t.record_failure("k")
    assert t.retry_after("k") == 0


def test_memory_is_bounded_under_a_flood_of_keys():
    t = FailureTracker(LockPolicy(5, 60, 900, 900), max_keys=100, clock=FakeClock())
    for i in range(10_000):
        t.record_failure(f"random-user-{i}")
    assert len(t) == 100
    # The most recent keys survive, the oldest were evicted.
    assert "random-user-9999" in t._entries
    assert "random-user-0" not in t._entries


def test_login_throttle_tables_are_bounded():
    throttle = LoginThrottle(clock=FakeClock(), max_keys=50)
    for i in range(5_000):
        throttle.failure(ClientAddress(f"198.51.{i // 256}.{i % 256}", True, False), f"user{i}")
        throttle.failure(ClientAddress(f"10.0.{i // 256}.{i % 256}", False, False, "x"), f"u{i}")
    assert len(throttle.by_address) == 50
    assert len(throttle.by_shared_address) == 50
    assert len(throttle.by_username) == 50


def test_ipv6_clients_are_keyed_by_their_64():
    assert address_key("2001:db8:1:2::1") == address_key("2001:db8:1:2:ffff::9")
    assert address_key("2001:db8:1:2::1") != address_key("2001:db8:1:3::1")
    assert address_key("203.0.113.9") == "203.0.113.9"


# --- DelayTracker / DelayGate ------------------------------------------------------


def test_delays_escalate_after_the_grace_and_are_capped():
    t = DelayTracker(LOGIN_USER_DELAY, clock=FakeClock())
    delays = [t.record_failure("admin") for _ in range(10)]
    assert delays == [0, 0, 0, 0.5, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]


def test_a_delay_count_is_forgotten_after_a_quiet_period():
    clock = FakeClock()
    t = DelayTracker(LOGIN_USER_DELAY, clock=clock)
    for _ in range(6):
        t.record_failure("admin")
    clock.advance(LOGIN_USER_DELAY.forget_after_s + 1)
    assert t.record_failure("admin") == 0


async def test_the_gate_bounds_how_many_failures_are_held_at_once():
    release = asyncio.Event()

    async def blocking_sleep(_seconds: float) -> None:
        await release.wait()

    gate = DelayGate(3, sleep=blocking_sleep)
    held = [asyncio.create_task(gate.hold(1.0)) for _ in range(3)]
    await asyncio.sleep(0)
    assert gate.active == 3
    # The gate is full: the next failure is not held (the caller answers 429).
    assert await gate.hold(1.0) is False
    # A zero delay never needs the gate.
    assert await gate.hold(0) is True
    release.set()
    assert await asyncio.gather(*held) == [True, True, True]
    assert gate.active == 0
    assert await gate.hold(1.0) is True


# --- LoginThrottle policy ----------------------------------------------------------


def test_the_policies_are_the_documented_ones():
    assert (LOGIN_IP_POLICY.threshold, LOGIN_IP_POLICY.base_lock_s, LOGIN_IP_POLICY.max_lock_s) == (5, 60, 900)
    assert (LOGIN_USER_DELAY.grace, LOGIN_USER_DELAY.base_delay_s, LOGIN_USER_DELAY.max_delay_s) == (3, 0.5, 3.0)
    assert (LOGIN_SHARED_DELAY.grace, LOGIN_SHARED_DELAY.base_delay_s, LOGIN_SHARED_DELAY.max_delay_s) == (5, 0.5, 3.0)
    assert (BEARER_IP_POLICY.threshold, BEARER_IP_POLICY.max_lock_s) == (20, 900)
    assert MAX_DELAYED == 32
    # A throttle delay stays short: it is what the real operator waits too.
    assert max(LOGIN_USER_DELAY.max_delay_s, LOGIN_SHARED_DELAY.max_delay_s) <= 3


def test_only_an_attributable_address_is_ever_locked():
    throttle = LoginThrottle(clock=FakeClock())
    for _ in range(50):
        throttle.failure(SHARED, "admin")
        throttle.failure(PUBLIC, "admin")
    assert throttle.retry_after(PUBLIC) > 0
    assert throttle.retry_after(SHARED) == 0
    # A username is never locked: it is not even an input to the lock.
    assert throttle.retry_after(ClientAddress("198.51.100.1", True, False)) == 0


def test_a_distributed_guess_slows_the_username_down():
    throttle = LoginThrottle(clock=FakeClock())
    delays = [
        throttle.failure(ClientAddress(f"203.0.113.{i}", True, False), "admin")
        for i in range(6)
    ]
    assert delays == [0, 0, 0, 0.5, 1.0, 2.0]
    # Case and surrounding space do not dodge it; another name is untouched.
    assert throttle.failure(ClientAddress("198.51.100.1", True, False), " ADMIN ") == 3.0
    assert throttle.failure(ClientAddress("198.51.100.1", True, False), "someone") == 0


def test_an_unattributable_address_is_slowed_as_one_shared_key():
    throttle = LoginThrottle(clock=FakeClock())
    delays = [throttle.failure(SHARED, f"ghost{i}") for i in range(8)]
    assert delays == [0, 0, 0, 0, 0, 0.5, 1.0, 2.0]


def test_success_resets_the_address_and_username_counts():
    throttle = LoginThrottle(clock=FakeClock())
    for _ in range(4):
        throttle.failure(PUBLIC, "admin")
    throttle.success(PUBLIC, "admin")
    for _ in range(3):
        assert throttle.failure(PUBLIC, "admin") == 0
    assert throttle.retry_after(PUBLIC) == 0


def test_bearer_throttle_never_locks_a_secret_that_matched():
    clock = FakeClock()
    throttle = BearerThrottle(clock=clock)
    throttle.matched("vw_known")
    for _ in range(25):
        throttle.unknown("203.0.113.5")
    assert throttle.retry_after("203.0.113.5", "vw_guess") > 0
    assert throttle.retry_after("203.0.113.5", "vw_known") == 0


def test_bearer_throttle_never_locks_an_unattributable_client():
    throttle = BearerThrottle(clock=FakeClock())
    for _ in range(100):
        throttle.unknown(None)
    assert throttle.retry_after(None, "vw_guess") == 0
    assert len(throttle.by_address) == 0


# --- The login route ------------------------------------------------------------------


@pytest.fixture
def fake_sleep(client) -> FakeSleep:
    """The app's login throttle, with a fake clock-free sleep."""
    sleep = FakeSleep()
    client.app.state.login_throttle = LoginThrottle(sleep=sleep)
    return sleep


def _peer_client(client: TestClient) -> TestClient:
    """A client on the same (already started) app whose socket peer is the
    ``x-test-peer`` header, so a test can be a trusted proxy or a stranger."""
    inner = client.app

    async def with_peer(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            peer = headers.get(b"x-test-peer")
            if peer:
                scope = {**scope, "client": (peer.decode(), 40000)}
        await inner(scope, receive, send)

    return TestClient(with_peer)


def _login(c: TestClient, password: str, *, username: str = "admin", **headers: str):
    return c.post(
        "/api/auth/login",
        json={"username": username, "password": password},
        headers={k.replace("_", "-"): v for k, v in headers.items()},
    )


HOST_A = "203.0.113.7"  # a public client talking to the api directly
PROXY = "172.18.0.5"  # Caddy on the compose network (a private range)


def test_a_public_address_locks_after_five_failures(client, seeded_db, fake_sleep):
    c = _peer_client(client)
    for _ in range(5):
        assert _login(c, "wrong", x_test_peer=HOST_A).status_code == 401
    r = _login(c, "wrong", x_test_peer=HOST_A)
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "60"
    assert r.json()["detail"] == "too many failed login attempts; retry later"
    # Locked means locked for THAT address: the right password waits too...
    assert _login(c, PASSWORD, x_test_peer=HOST_A).status_code == 429
    # ...and nobody else is touched.
    assert _login(c, PASSWORD, x_test_peer="198.51.100.20").status_code == 200


def test_a_refused_attempt_runs_no_bcrypt_and_does_not_escalate(
    client, seeded_db, monkeypatch, fake_sleep
):
    import app.auth.routes as routes_mod

    c = _peer_client(client)
    for _ in range(5):
        _login(c, "wrong", x_test_peer=HOST_A)
    calls = {"n": 0}
    real = routes_mod.bcrypt.checkpw

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(routes_mod.bcrypt, "checkpw", counting)
    for _ in range(10):
        r = _login(c, "wrong", x_test_peer=HOST_A)
        assert r.status_code == 429
        assert r.headers["Retry-After"] == "60"  # not escalated by the retries
    assert calls["n"] == 0


def test_success_resets_the_address_count(client, seeded_db, fake_sleep):
    c = _peer_client(client)
    for _ in range(4):
        assert _login(c, "wrong", x_test_peer=HOST_A).status_code == 401
    assert _login(c, PASSWORD, x_test_peer=HOST_A).status_code == 200
    for _ in range(4):
        assert _login(c, "wrong", x_test_peer=HOST_A).status_code == 401
    assert _login(c, PASSWORD, x_test_peer=HOST_A).status_code == 200


def test_the_lock_expires(client, seeded_db):
    clock = FakeClock()
    client.app.state.login_throttle = LoginThrottle(clock=clock, sleep=FakeSleep())
    c = _peer_client(client)
    for _ in range(5):
        _login(c, "wrong", x_test_peer=HOST_A)
    assert _login(c, PASSWORD, x_test_peer=HOST_A).status_code == 429
    clock.advance(61)
    assert _login(c, PASSWORD, x_test_peer=HOST_A).status_code == 200


def test_unknown_username_still_runs_bcrypt_and_counts(
    client, seeded_db, monkeypatch, fake_sleep
):
    import app.auth.routes as routes_mod

    calls = {"n": 0}
    real = routes_mod.bcrypt.checkpw

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(routes_mod.bcrypt, "checkpw", counting)
    c = _peer_client(client)
    for _ in range(5):
        assert _login(c, "x", username="ghost", x_test_peer=HOST_A).status_code == 401
    assert calls["n"] == 5, "the timing equalizer must still run for every unknown user"
    # Counted like a real user: an unknown name is not a way around the lock.
    assert _login(c, "x", username="ghost", x_test_peer=HOST_A).status_code == 429


def test_the_dummy_hash_password_does_not_sign_in_an_unknown_user(client, seeded_db, fake_sleep):
    assert _login(client, "timing-equalizer", username="ghost").status_code == 401


def test_spoofed_x_forwarded_for_from_an_untrusted_peer_is_ignored(client, seeded_db, fake_sleep):
    c = _peer_client(client)
    for i in range(5):
        r = _login(c, "wrong", x_test_peer=HOST_A, x_forwarded_for=f"198.51.100.{i}")
        assert r.status_code == 401
    # A new forged address does not buy a new budget: the peer is the key.
    r = _login(c, "wrong", x_test_peer=HOST_A, x_forwarded_for="198.51.100.200")
    assert r.status_code == 429


def test_x_forwarded_for_from_a_trusted_proxy_separates_clients(client, seeded_db, fake_sleep):
    c = _peer_client(client)
    for _ in range(5):
        _login(c, "wrong", x_test_peer=PROXY, x_forwarded_for="203.0.113.7")
    assert _login(c, "wrong", x_test_peer=PROXY, x_forwarded_for="203.0.113.7").status_code == 429
    # The admin behind the same proxy is somebody else.
    r = _login(c, PASSWORD, x_test_peer=PROXY, x_forwarded_for="192.0.2.44")
    assert r.status_code == 200, r.text
    # A client-written entry to the LEFT of the proxy's own does not help.
    r = _login(c, "wrong", x_test_peer=PROXY, x_forwarded_for="198.51.100.9, 203.0.113.7")
    assert r.status_code == 429


# The 2026-09-23 incident: Traefik -> Caddy -> api, and every visitor reached
# the api as the same private address. These are the shapes of that.
UNATTRIBUTABLE = [
    pytest.param({"x_test_peer": PROXY}, id="trusted-proxy-no-xff"),
    pytest.param({"x_test_peer": PROXY, "x_forwarded_for": "10.42.0.9"}, id="only-private-hops"),
    pytest.param({"x_test_peer": PROXY, "x_forwarded_for": "100.64.3.4"}, id="cgnat-hop"),
    pytest.param({"x_test_peer": PROXY, "x_forwarded_for": "garbage"}, id="garbage-hop"),
    pytest.param({"x_test_peer": "192.168.1.20"}, id="untrusted-private-peer"),
    pytest.param({}, id="testclient"),
]


@pytest.mark.parametrize("via", UNATTRIBUTABLE)
def test_an_unattributable_flood_never_refuses_a_correct_password(
    client, seeded_db, fake_sleep, via
):
    if via.get("x_test_peer") == "192.168.1.20":
        # VW_TRUSTED_PROXIES=none: a private peer that is not a trusted proxy.
        state = client.app.state
        state.settings = dataclasses.replace(state.settings, trusted_proxies=())
    c = _peer_client(client)
    for _ in range(40):
        assert _login(c, "wrong", **via).status_code == 401
    # Guessing got slow...
    assert fake_sleep.calls and max(fake_sleep.calls) == 3.0
    # ...but the operator, behind the same address, gets straight in.
    assert _login(c, PASSWORD, **via).status_code == 200
    # And a different user's first attempt is an ordinary 401, not a 429.
    assert _login(c, "wrong", username="someone-else", **via).status_code == 401


def test_the_incident_bad_logins_from_one_host_do_not_lock_another(client, seeded_db, fake_sleep):
    c = _peer_client(client)
    # Host A and host B both arrive as Traefik's address in X-Forwarded-For.
    for _ in range(5):
        assert _login(c, "x", username="made-up-a", x_test_peer=PROXY,
                      x_forwarded_for="10.42.0.9").status_code == 401
    r = _login(c, "x", username="made-up-b", x_test_peer=PROXY, x_forwarded_for="10.42.0.9")
    assert r.status_code == 401


def test_a_throttled_username_still_takes_the_correct_password(client, seeded_db, fake_sleep):
    c = _peer_client(client)
    # Many public addresses guess "admin": the name is slowed, never locked.
    for i in range(20):
        assert _login(c, "wrong", x_test_peer=f"198.51.100.{i}").status_code == 401
    assert max(fake_sleep.calls) == 3.0
    before = len(fake_sleep.calls)
    r = _login(c, PASSWORD, x_test_peer="192.0.2.44")
    assert r.status_code == 200, r.text
    assert len(fake_sleep.calls) == before, "a correct password is answered at once"


def test_a_full_delay_gate_answers_only_failures_with_429(client, seeded_db, fake_sleep):
    throttle = client.app.state.login_throttle
    throttle.gate.active = throttle.gate.max_concurrent  # as if a flood held every slot
    for _ in range(3):  # within the username grace: nothing to hold
        assert _login(client, "wrong").status_code == 401
    r = _login(client, "wrong")
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "1"
    assert _login(client, PASSWORD).status_code == 200


def test_the_unattributable_warning_is_logged_once(client, seeded_db, fake_sleep, monkeypatch, caplog):
    monkeypatch.setattr(throttle_mod, "_unattributable_warned", False)
    c = _peer_client(client)
    with caplog.at_level(logging.WARNING, logger="app.auth.throttle"):
        for _ in range(5):
            _login(c, "wrong", x_test_peer=PROXY)
            _login(c, "wrong", x_test_peer=PROXY, x_forwarded_for="10.42.0.9")
    warnings = [r for r in caplog.records if r.name == "app.auth.throttle"]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "X-Forwarded-For missing or only private hops" in msg
    assert "VW_TRUSTED_PROXIES" in msg


def test_no_warning_for_an_attributable_client(client, seeded_db, fake_sleep, monkeypatch, caplog):
    monkeypatch.setattr(throttle_mod, "_unattributable_warned", False)
    c = _peer_client(client)
    with caplog.at_level(logging.WARNING, logger="app.auth.throttle"):
        _login(c, "wrong", x_test_peer=PROXY, x_forwarded_for=HOST_A)
        _login(c, "wrong", x_test_peer=HOST_A)
    assert not [r for r in caplog.records if r.name == "app.auth.throttle"]


# --- Bearer secrets -------------------------------------------------------------------


def _unknown(prefix: str, i: int) -> str:
    return f"{prefix}{'a' * 50}{i:06d}"


@pytest.mark.parametrize(
    "path,prefix",
    [("/v1/models", "vw_"), ("/api/tokens", "vwa_")],
)
def test_unknown_bearer_secrets_from_one_public_address_are_throttled(
    client, seeded_db, path, prefix
):
    c = _peer_client(client)
    for i in range(BEARER_IP_POLICY.threshold):
        r = c.get(path, headers={**bearer(_unknown(prefix, i)), "x-test-peer": HOST_A})
        assert r.status_code == 401, r.text
    r = c.get(path, headers={**bearer(_unknown(prefix, 999)), "x-test-peer": HOST_A})
    assert r.status_code == 429
    assert r.headers["Retry-After"] == str(int(BEARER_IP_POLICY.base_lock_s))
    # Another public caller is untouched.
    r = c.get(path, headers={**bearer(_unknown(prefix, 998)), "x-test-peer": "198.51.100.3"})
    assert r.status_code == 401


@pytest.mark.parametrize(
    "path,prefix",
    [("/v1/models", "vw_"), ("/api/tokens", "vwa_")],
)
def test_an_unattributable_address_never_locks_bearer_callers(client, seeded_db, path, prefix):
    _, known = seed_admin_token(
        seeded_db, name="app", scope="inference" if prefix == "vw_" else "admin",
        plaintext=prefix + "k" * 56,
    )
    c = _peer_client(client)
    via = {"x-test-peer": PROXY, "x-forwarded-for": "10.42.0.9"}
    for i in range(BEARER_IP_POLICY.threshold * 3):
        r = c.get(path, headers={**bearer(_unknown(prefix, i)), **via})
        assert r.status_code == 401, r.text
    # Other callers behind the same proxy: an unknown secret is still a 401...
    assert c.get(path, headers={**bearer(_unknown(prefix, 999)), **via}).status_code == 401
    # ...and a key never seen before in this process just works.
    assert c.get(path, headers={**bearer(known), **via}).status_code == 200


def test_a_known_admin_token_keeps_working_from_a_throttled_address(client, seeded_db):
    _, secret = seed_admin_token(seeded_db)
    c = _peer_client(client)
    peer = {"x-test-peer": HOST_A}
    assert c.get("/api/tokens", headers={**bearer(secret), **peer}).status_code == 200
    for i in range(BEARER_IP_POLICY.threshold):
        c.get("/api/tokens", headers={**bearer(_unknown("vwa_", i)), **peer})
    assert c.get("/api/tokens", headers={**bearer(_unknown("vwa_", 999)), **peer}).status_code == 429
    assert c.get("/api/tokens", headers={**bearer(secret), **peer}).status_code == 200


def test_a_known_inference_key_keeps_working_from_a_throttled_address(client, seeded_db):
    _, secret = seed_admin_token(seeded_db, name="app", scope="inference",
                                 plaintext="vw_" + "k" * 56)
    c = _peer_client(client)
    peer = {"x-test-peer": HOST_A}
    assert c.get("/v1/models", headers={**bearer(secret), **peer}).status_code == 200
    for i in range(BEARER_IP_POLICY.threshold):
        c.get("/v1/models", headers={**bearer(_unknown("vw_", i)), **peer})
    assert c.get("/v1/models", headers={**bearer(_unknown("vw_", 999)), **peer}).status_code == 429
    assert c.get("/v1/models", headers={**bearer(secret), **peer}).status_code == 200
