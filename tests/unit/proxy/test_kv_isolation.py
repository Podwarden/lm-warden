"""Per-key KV isolation in the proxy (plan 2, task 9).

An isolated key's requests reach a vLLM engine with a ``cache_salt`` derived from the
key's scope (GPU tier) and with ``lmw_tenant`` / ``lmw_scope`` in ``kv_transfer_params``
(RAM and NVMe tiers). A sharing key keeps today's body. The identity and every refusal
are decided before admission, so a refused request holds no slot and no live row.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.db.repos.tokens import TokenRow, hash_token
from app.proxy.kv_scope import (
    CacheSaltRefused,
    KvIdentity,
    check_cache_salt,
    kv_identity,
    rewrite_cache_salt,
)
from app.proxy.request_id import REQUEST_ID_HEADER
from app.proxy.token_cache import invalidate_all as invalidate_tokens
from lmw_kv.scopes import gpu_cache_salt, isolation_scope, scope_tag, tenant_tag
from tests.conftest import wait_until
from tests.unit.proxy.test_kv_params import _kv_on
from tests.unit.proxy.test_kv_tier_record import FakeKv
from tests.unit.proxy.test_messages import AUTH, _json_upstream, _ready

NK = bytes(range(32))

OK = {
    "id": "c1",
    "object": "chat.completion",
    "model": "qwen",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
CHAT = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}

# B is the seeded key (tok1, sharing). A and C are added isolated keys.
KEY_A = "vw_isolatedkeyA1234567890abcdef1234"
KEY_C = "vw_isolatedkeyC1234567890abcdef1234"
AUTH_B = AUTH
AUTH_A = {"Authorization": f"Bearer {KEY_A}"}
AUTH_C = {"Authorization": f"Bearer {KEY_C}"}


def _row(**kw: Any) -> TokenRow:
    base: dict[str, Any] = {
        "id": "t1",
        "name": "n",
        "prefix": "vw_x",
        "scope": "inference",
        "allowed_models": None,
        "rate_limit_rpm": None,
        "rate_limit_tpm": None,
        "revoked_at": None,
        "last_used_at": None,
        "created_at": "x",
        "expires_at": None,
        "rotated_at": None,
        "rotated_from": None,
    }
    base.update(kw)
    return TokenRow(**base)


# -- pure -----------------------------------------------------------------------


def test_identity_of_an_isolated_key() -> None:
    ident = kv_identity(_row(kv_isolation="isolated", kv_tenant_id="fam", kv_epoch=2), NK)
    scope = isolation_scope(NK, "fam", 2)
    assert ident.isolated is True
    assert ident.tenant == tenant_tag(NK, "fam")
    assert ident.scope == scope and ident.scope_tag == scope_tag(scope)


def test_identity_of_a_sharing_key_and_of_no_row() -> None:
    ident = kv_identity(_row(kv_tenant_id="fam", kv_epoch=7), NK)
    assert ident == KvIdentity(False, tenant_tag(NK, "fam"), b"", b"")
    assert kv_identity(_row(kv_tenant_id=""), NK).tenant == tenant_tag(NK, "internal")
    assert kv_identity(None, NK) == KvIdentity(False, tenant_tag(NK, "internal"), b"", b"")


def test_identity_is_memoised_per_family_and_epoch() -> None:
    a = kv_identity(_row(kv_isolation="isolated", kv_tenant_id="fam", kv_epoch=0), NK)
    assert kv_identity(_row(id="t2", kv_isolation="isolated", kv_tenant_id="fam"), NK) is a
    b = kv_identity(_row(kv_isolation="isolated", kv_tenant_id="fam", kv_epoch=1), NK)
    assert b.scope != a.scope  # a reset (new epoch) is a new scope


def test_an_isolated_key_without_a_tenant_is_refused_not_pooled() -> None:
    """Fail closed: two broken isolated rows must never share the "internal" scope."""
    with pytest.raises(ValueError):
        kv_identity(_row(kv_isolation="isolated", kv_tenant_id=""), NK)
    with pytest.raises(ValueError):
        kv_identity(_row(kv_isolation="isolated", kv_tenant_id="fam", kv_epoch="1"), NK)


def test_an_unknown_isolation_value_is_isolated() -> None:
    assert kv_identity(_row(kv_isolation="ISOLATED", kv_tenant_id="fam"), NK).isolated


def test_constants_match_lmw_kv() -> None:
    import app.proxy.kv_scope as ks
    import lmw_kv.scopes as ls

    assert ks.GPU_SALT_PREFIX == ls.GPU_SALT_PREFIX
    assert ks.INTERNAL_TENANT_ID == ls.INTERNAL_TENANT_ID


def test_without_lmw_kv_isolated_fails_closed_and_sharing_works(monkeypatch) -> None:
    import app.proxy.kv_scope as ks

    monkeypatch.setattr(ks, "_scopes", None)
    ks._derive.cache_clear()
    try:
        with pytest.raises(ValueError):
            kv_identity(_row(kv_isolation="isolated", kv_tenant_id="nolib"), NK)
        with pytest.raises(ValueError):  # the pre-flight falls back to UNKEYED_SHARING
            kv_identity(_row(kv_tenant_id="nolib"), NK)
    finally:
        ks._derive.cache_clear()


def test_identity_repr_is_redacted() -> None:
    ident = kv_identity(_row(kv_isolation="isolated", kv_tenant_id="fam"), NK)
    r = repr(ident)
    assert r == "KvIdentity(isolated=True, <redacted>)"
    for v in (ident.scope, ident.scope_tag, ident.tenant):
        assert v.hex() not in r and v.hex() not in str(ident)


ISO = KvIdentity(True, b"t" * 16, b"s" * 32, b"g" * 16)
SHR = KvIdentity(False, b"t" * 16, b"", b"")


@pytest.mark.parametrize("bad", [1, 1.5, True, [], {}, ["s"]])
def test_check_refuses_a_non_string_salt_for_an_isolated_key(bad: Any) -> None:
    with pytest.raises(CacheSaltRefused, match="cache_salt must be a string"):
        check_cache_salt({"cache_salt": bad}, ISO)
    check_cache_salt({"cache_salt": bad}, SHR)  # sharing: today's behaviour (vLLM decides)


def test_check_refuses_a_salt_that_is_not_text() -> None:
    with pytest.raises(CacheSaltRefused):
        check_cache_salt({"cache_salt": "\ud800"}, ISO)
    with pytest.raises(CacheSaltRefused):
        check_cache_salt({"cache_salt": "lmw1:\ud800"}, SHR)
    check_cache_salt({"cache_salt": "\ud800"}, SHR)  # not reserved: passes through as today


def test_check_accepts_absent_null_and_strings() -> None:
    for body in ({}, {"cache_salt": None}, {"cache_salt": ""}, {"cache_salt": "x" * 100_000}):
        check_cache_salt(body, ISO)
        check_cache_salt(body, SHR)


def test_rewrite_isolated() -> None:
    for body, cs in (({}, ""), ({"cache_salt": None}, ""), ({"cache_salt": "s"}, "s")):
        assert rewrite_cache_salt(body, ISO, NK) is True
        assert body["cache_salt"] == gpu_cache_salt(NK, ISO.scope, cs)


def test_rewrite_sharing_touches_only_the_reserved_prefix() -> None:
    for salt in ("s", "", "x" * 100_000, 7, None, "LMW1:abc", " lmw1:"):
        body: dict[str, Any] = {"cache_salt": salt}
        assert rewrite_cache_salt(body, SHR, NK) is False
        assert body == {"cache_salt": salt}
    body = {}
    assert rewrite_cache_salt(body, SHR, NK) is False and body == {}
    forged = gpu_cache_salt(NK, ISO.scope, "")
    body = {"cache_salt": forged}
    assert rewrite_cache_salt(body, SHR, NK) is True
    assert body["cache_salt"] == gpu_cache_salt(NK, b"", forged) != forged


def test_isolated_keys_stay_disjoint_whatever_the_client_salt() -> None:
    a = kv_identity(_row(kv_isolation="isolated", kv_tenant_id="a"), NK)
    c = kv_identity(_row(kv_isolation="isolated", kv_tenant_id="c"), NK)
    a_salt = gpu_cache_salt(NK, a.scope, "s")
    seen = set()
    for ident in (a, c, SHR):
        for cs in ("s", "", a_salt, "lmw1:", "x" * 100_000):
            body: dict[str, Any] = {"cache_salt": cs}
            rewrite_cache_salt(body, ident, NK)
            seen.add((ident.scope, body["cache_salt"]))
    # No two (key, client salt) pairs of different keys land on one engine salt.
    by_salt: dict[str, set[bytes]] = {}
    for scope, salt in seen:
        by_salt.setdefault(salt, set()).add(scope)
    assert all(len(v) == 1 for v in by_salt.values())
    # C replaying A's derived salt does not reach A's blocks.
    body = {"cache_salt": a_salt}
    rewrite_cache_salt(body, c, NK)
    assert body["cache_salt"] != a_salt


# -- forward --------------------------------------------------------------------


def _db(tmp_data_dir):
    return tmp_data_dir / "vllm-warden.db"


def _add_keys(tmp_data_dir) -> None:
    with sqlite3.connect(_db(tmp_data_dir)) as db:
        for tid, pt in (("tokA", KEY_A), ("tokC", KEY_C)):
            db.execute(
                "INSERT INTO api_tokens(id, name, prefix, hash, scope, kv_isolation) "
                "VALUES (?, ?, ?, ?, 'inference', 'isolated')",
                (tid, tid, pt[:8], hash_token(pt)),
            )
        db.commit()


def _setup(client, tmp_data_dir) -> None:
    _ready(client, tmp_data_dir)
    _add_keys(tmp_data_dir)


def _nk(client) -> bytes:
    return client.app.state.cobs2_key.key


ROUTES = {
    "chat": ("/v1/chat/completions", CHAT),
    "completions": ("/v1/completions", {"model": "qwen", "prompt": "hi"}),
    "messages": (
        "/v1/messages",
        {"model": "qwen", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
    ),
    "responses": ("/v1/responses", {"model": "qwen", "input": "hi", "max_output_tokens": 16}),
}


def _go(client, auth, body, path="/v1/chat/completions"):
    send = AsyncMock(return_value=_json_upstream(OK))
    # ASCII-escaped JSON, so a lone surrogate survives the test client's encoder.
    raw = json.dumps(body).encode()
    hdrs = {**auth, "content-type": "application/json"}
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(path, headers=hdrs, content=raw)
    return r, send


def _sent(send) -> bytes:
    return send.call_args.args[0].content


def _sent_json(send) -> dict[str, Any]:
    return json.loads(_sent(send))


def _client_salted(route: str, salt: Any) -> tuple[str, dict[str, Any]]:
    path, body = ROUTES[route]
    body = dict(body)
    if salt is not _ABSENT:
        body["cache_salt"] = salt
    return path, body


_ABSENT = object()


def _scope_of(client, tid: str) -> bytes:
    return isolation_scope(_nk(client), tid, 0)


@pytest.mark.parametrize("route", list(ROUTES))
def test_isolated_keys_get_their_own_stable_gpu_salt(tmp_data_dir, client, route):
    _setup(client, tmp_data_dir)
    path, body = _client_salted(route, "s")
    salts: dict[str, list[str]] = {"A": [], "C": []}
    for _ in range(2):
        for name, auth in (("A", AUTH_A), ("C", AUTH_C)):
            r, send = _go(client, auth, body, path)
            assert r.status_code == 200, r.text
            salts[name].append(_sent_json(send)["cache_salt"])
    a, c = salts["A"], salts["C"]
    assert a[0] == a[1] and c[0] == c[1]  # stable: a key hits its own GPU blocks
    assert a[0] != c[0] and a[0].startswith("lmw1:") and c[0].startswith("lmw1:")
    if route in ("chat", "completions"):  # the client's salt is what was folded
        assert a[0] == gpu_cache_salt(_nk(client), _scope_of(client, "tokA"), "s")
    else:  # the translators carry no cache_salt: the empty client salt
        assert a[0] == gpu_cache_salt(_nk(client), _scope_of(client, "tokA"), "")


@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize("kv", ["off", "on"])
@pytest.mark.parametrize("salt", [_ABSENT, None, "", "s", "lmw1:" + "0" * 64, "x" * 100_000])
def test_no_isolated_request_reaches_the_engine_without_its_salt(
    tmp_data_dir, client, route, kv, salt
):
    """R7: the connector never sees an isolated key's raw scope only because every
    isolated request carries its derived GPU salt. kv=on runs with the daemon down."""
    _setup(client, tmp_data_dir)
    if kv == "on":
        _kv_on(client, tmp_data_dir)
    path, body = _client_salted(route, salt)
    r, send = _go(client, AUTH_A, body, path)
    assert r.status_code == 200, r.text
    sent = _sent_json(send)
    cs = salt if isinstance(salt, str) and route in ("chat", "completions") else ""
    assert sent["cache_salt"] == gpu_cache_salt(_nk(client), _scope_of(client, "tokA"), cs)


def test_a_failing_rewrite_still_salts_an_isolated_request(tmp_data_dir, client, monkeypatch):
    import app.proxy.routes as routes

    def boom(*a, **k):
        raise RuntimeError("rewrite failed")

    _setup(client, tmp_data_dir)
    monkeypatch.setattr(routes, "rewrite_cache_salt", boom)
    r, send = _go(client, AUTH_A, {**CHAT, "cache_salt": "s"})
    assert r.status_code == 200
    salt = _sent_json(send)["cache_salt"]
    assert salt == gpu_cache_salt(_nk(client), _scope_of(client, "tokA"), "")


def test_sharing_body_is_byte_identical_with_kv_off(tmp_data_dir, client):
    """Golden: the seeded sharing key B forwards exactly today's body."""
    _setup(client, tmp_data_dir)
    for body in (CHAT, {**CHAT, "cache_salt": "s"}):
        r, send = _go(client, AUTH_B, body)
        assert r.status_code == 200
        assert _sent(send) == json.dumps({**body, "priority": -5}).encode()


def test_sharing_body_with_kv_on_adds_only_lmw_tenant(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    _kv_on(client, tmp_data_dir)
    for body in (CHAT, {**CHAT, "cache_salt": "s"}):
        r, send = _go(client, AUTH_B, body)
        assert r.status_code == 200
        sent = _sent_json(send)
        kv = sent.pop("kv_transfer_params")
        assert sent == {**body, "priority": -5}
        assert kv.pop("lmw_tenant") == tenant_tag(_nk(client), "tok1").hex()
        assert set(kv) == {"lmw_req", "lmw_scope", "lmw_conv", "lmw_session"}
        assert kv["lmw_scope"] == "" and kv["lmw_req"] == r.headers[REQUEST_ID_HEADER]


def test_isolated_kv_params_carry_tenant_and_scope(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    _kv_on(client, tmp_data_dir)
    r, send = _go(client, AUTH_A, CHAT)
    assert r.status_code == 200
    kv = _sent_json(send)["kv_transfer_params"]
    assert kv["lmw_tenant"] == tenant_tag(_nk(client), "tokA").hex()
    assert kv["lmw_scope"] == _scope_of(client, "tokA").hex()


@pytest.mark.parametrize("auth", ["B", "C"])
def test_replaying_a_derived_salt_never_reaches_its_owner(tmp_data_dir, client, auth):
    _setup(client, tmp_data_dir)
    a_salt = gpu_cache_salt(_nk(client), _scope_of(client, "tokA"), "s")
    r, send = _go(client, AUTH_B if auth == "B" else AUTH_C, {**CHAT, "cache_salt": a_salt})
    assert r.status_code == 200
    got = _sent_json(send)["cache_salt"]
    assert got != a_salt
    if auth == "B":
        assert got == gpu_cache_salt(_nk(client), b"", a_salt)


@pytest.mark.parametrize("salt", ["", "x" * 100_000, 7, ["s"]])
def test_sharing_odd_salts_pass_through_as_today(tmp_data_dir, client, salt):
    _setup(client, tmp_data_dir)
    r, send = _go(client, AUTH_B, {**CHAT, "cache_salt": salt})
    assert r.status_code == 200
    assert _sent(send) == json.dumps({**CHAT, "cache_salt": salt, "priority": -5}).encode()


def _slots(client) -> tuple[int, int]:
    return (
        client.app.state.scheduler._inflight.get("qwen", 0),
        client.app.state.request_registry.count(),
    )


def _refused(client, auth, body, path="/v1/chat/completions"):
    before = _slots(client)
    acquire = client.app.state.scheduler.acquire
    calls: list[Any] = []

    def spy(*a, **k):
        calls.append(1)
        return acquire(*a, **k)

    client.app.state.scheduler.acquire = spy
    try:
        r, send = _go(client, auth, body, path)
    finally:
        client.app.state.scheduler.acquire = acquire
    assert not send.called
    assert calls == []  # never admitted
    assert _slots(client) == before
    return r


@pytest.mark.parametrize("route", ["chat", "completions"])  # the translators drop the field
@pytest.mark.parametrize("salt", [7, ["s"], {"a": 1}, True])
def test_an_isolated_non_string_salt_is_refused_before_admission(tmp_data_dir, client, route, salt):
    _setup(client, tmp_data_dir)
    path, body = ROUTES[route]
    r = _refused(client, AUTH_A, {**body, "cache_salt": salt}, path)
    assert r.status_code == 400 and "cache_salt must be a string" in r.text


def test_a_lone_surrogate_salt_is_a_400_not_a_500(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    for auth, salt in ((AUTH_A, "\ud800"), (AUTH_B, "lmw1:\ud800")):
        r = _refused(client, auth, {**CHAT, "cache_salt": salt})
        assert r.status_code == 400, r.text


def test_isolated_without_the_network_key_is_a_503_before_admission(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    client.app.state.cobs2_key = None
    for path, body in ROUTES.values():
        r = _refused(client, AUTH_A, body, path)
        assert r.status_code == 503 and "cache isolation is unavailable right now" in r.text
    # A sharing key keeps working, without attribution.
    r, send = _go(client, AUTH_B, CHAT)
    assert r.status_code == 200
    # A reserved salt cannot be rewritten without the key: refused, never forwarded.
    r = _refused(client, AUTH_B, {**CHAT, "cache_salt": "lmw1:" + "0" * 64})
    assert r.status_code == 503


def test_isolated_key_without_a_tenant_fails_closed(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    with sqlite3.connect(_db(tmp_data_dir)) as db:
        db.execute("UPDATE api_tokens SET kv_epoch = -1 WHERE id = 'tokA'")
        db.commit()
    r = _refused(client, AUTH_A, CHAT)
    assert r.status_code == 503


def _hist(tmp_data_dir, rid: str, cols: str):
    def _get():
        with sqlite3.connect(_db(tmp_data_dir)) as db:
            return db.execute(f"SELECT {cols} FROM request_history WHERE id = ?", (rid,)).fetchone()

    return wait_until(_get, what="history row")


def test_cobs2_ids_and_cache_obs_chain_survive_a_switch(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    with sqlite3.connect(_db(tmp_data_dir)) as db:
        db.execute("UPDATE api_tokens SET kv_isolation = 'sharing' WHERE id = 'tokA'")
        db.commit()
    invalidate_tokens()
    chains: list[Any] = []
    idx = client.app.state.cache_index
    orig = idx.match

    def spy(model_id, epoch, chain, **kw):
        chains.append(chain)
        return orig(model_id, epoch, chain, **kw)

    idx.match = spy
    body = {**CHAT, "cache_salt": "salt-1"}
    r1, s1 = _go(client, AUTH_A, body)
    with sqlite3.connect(_db(tmp_data_dir)) as db:
        db.execute("UPDATE api_tokens SET kv_isolation = 'isolated' WHERE id = 'tokA'")
        db.commit()
    invalidate_tokens()
    r2, s2 = _go(client, AUTH_A, body)
    assert r1.status_code == r2.status_code == 200
    assert _sent_json(s1)["cache_salt"] == "salt-1"
    assert _sent_json(s2)["cache_salt"].startswith("lmw1:")
    cols = "cobs2_head, cobs2_node, cobs2_conv, kv_isolated"
    h1 = _hist(tmp_data_dir, r1.headers[REQUEST_ID_HEADER], cols)
    h2 = _hist(tmp_data_dir, r2.headers[REQUEST_ID_HEADER], cols)
    assert h1[0] is not None and h1[:3] == h2[:3]
    assert (h1[3], h2[3]) == (0, 1)
    assert chains and all(c == chains[0] for c in chains)


def test_an_isolated_request_is_observed_on_its_own_lens_only(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    body = {**CHAT, "messages": [{"role": "user", "content": "y" * 4000}]}
    r, _ = _go(client, AUTH_A, body)
    assert r.status_code == 200
    _hist(tmp_data_dir, r.headers[REQUEST_ID_HEADER], "id")
    from app.cache_obs.canonical import chain_of

    idx = client.app.state.cache_index
    chain = chain_of(body)
    keys = [k for k in idx._d if k[0] == "c"]
    assert keys and all(k[3] != b"*" for k in keys)
    # B with the same chain finds nothing of A's on the operator lens.
    assert chain is not None
    epoch = keys[0][2]
    assert idx.match("qwen", epoch, chain).best_bytes == 0
    assert idx.match("qwen", epoch, chain, token_id="tokA").best_bytes > 0


USAGE_BIG = {
    "prompt_tokens": 1000,
    "completion_tokens": 1,
    "total_tokens": 1001,
    "prompt_tokens_details": {"cached_tokens": 100},
}


def test_isolation_cost_is_what_sharing_would_have_served(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    _kv_on(client, tmp_data_dir)
    fake = FakeKv({"ram_tokens": 0, "nvme_tokens": 0, "load_s": 0.0})
    client.app.state.kv_store.client = lambda: fake
    body = {**CHAT, "messages": [{"role": "user", "content": "z" * 4000}]}
    ok = {**OK, "usage": USAGE_BIG}
    for auth in (AUTH_B, AUTH_A):
        send = AsyncMock(return_value=_json_upstream(ok))
        with patch("httpx.AsyncClient.send", new=send):
            r = client.post("/v1/chat/completions", headers=auth, json=body)
        assert r.status_code == 200
        rid = r.headers[REQUEST_ID_HEADER]
        cols = "kv_isolated, kv_iso_cost_tokens, served_gpu_tokens"

        def _tiers(rid=rid, cols=cols):
            x = _hist(tmp_data_dir, rid, cols)
            return x if x[2] is not None else None

        row = wait_until(_tiers, what="tier columns")
        if auth is AUTH_B:
            assert row[:2] == (0, None)
    # A recomputed 900 tokens; the operator lens saw B's whole prompt: cost 900.
    assert row[0] == 1 and row[1] == 900


def test_a_sharing_body_naming_cache_salt_is_re_serialised(tmp_data_dir, client):
    """Fix round 1: a duplicate key must not reach the engine raw, where a parser that
    keeps the first value would read a reserved salt the proxy never saw."""
    _setup(client, tmp_data_dir)
    with sqlite3.connect(_db(tmp_data_dir)) as db:
        db.execute("UPDATE api_tokens SET priority = 0 WHERE id = 'tok1'")
        db.commit()
    invalidate_tokens()
    forged = gpu_cache_salt(_nk(client), _scope_of(client, "tokA"), "")
    raw = (
        b'{"model": "qwen", "messages": [{"role": "user", "content": "hi"}], '
        b'"cache_salt": "' + forged.encode() + b'", "cache_salt": "x"}'
    )
    send = AsyncMock(return_value=_json_upstream(OK))
    hdrs = {**AUTH_B, "content-type": "application/json"}
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post("/v1/chat/completions", headers=hdrs, content=raw)
    assert r.status_code == 200
    sent = _sent(send)
    assert sent.count(b"cache_salt") == 1 and forged.encode() not in sent
    assert json.loads(sent)["cache_salt"] == "x"
    # A body without the field still forwards byte-identical.
    plain = b'{"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}'
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post("/v1/chat/completions", headers=hdrs, content=plain)
    assert r.status_code == 200 and _sent(send) == plain


def test_an_unsalted_isolated_request_is_a_503_not_a_500(tmp_data_dir, client, monkeypatch):
    import app.proxy.routes as routes

    _setup(client, tmp_data_dir)
    real = routes._kv_isolation_preflight

    def no_key(*a, **k):
        ident, _nk_, rewrite = real(*a, **k)
        return ident, None, rewrite  # the late half finds no network key

    monkeypatch.setattr(routes, "_kv_isolation_preflight", no_key)
    before = _slots(client)
    r, send = _go(client, AUTH_A, CHAT)
    assert r.status_code == 503 and not send.called
    assert _slots(client) == before
