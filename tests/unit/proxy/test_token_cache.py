"""Bearer token cache (#279 stage 3): unit behaviour and the /v1 forward path."""

from app.db.database import open_db
from app.db.repos.tokens import TokenRepo
from app.proxy import token_cache as tc
from tests.unit.proxy.test_dp_routing_forward import _post, _ready


def test_hit_within_ttl_and_expiry():
    now = [0.0]
    c = tc.TokenCache(ttl_s=5.0, clock=lambda: now[0])
    c.put("h", "ROW")
    assert c.get("h") == "ROW"
    now[0] = 5.1
    assert c.get("h") is tc.MISS


def test_invalidate_all_drops_every_instance():
    a, b = tc.TokenCache(), tc.TokenCache()
    a.put("h", "ROW")
    b.put("h", "ROW")
    tc.invalidate_all()
    assert a.get("h") is tc.MISS and b.get("h") is tc.MISS


def test_none_is_never_cached():
    c = tc.TokenCache()
    c.put("h", None)
    assert c.get("h") is tc.MISS


def _count_lookups(monkeypatch):
    calls = []
    real = TokenRepo.find_by_plaintext

    async def counting(self, plaintext):
        calls.append(plaintext)
        return await real(self, plaintext)

    monkeypatch.setattr(TokenRepo, "find_by_plaintext", counting)
    return calls


def _repo_call(client, tmp_data_dir, method, *args, **kw):
    async def go():
        async with open_db(str(tmp_data_dir / "vllm-warden.db")) as db:
            return await getattr(TokenRepo(db), method)(*args, **kw)

    return client.portal.call(go)


def test_second_request_does_not_read_the_token_row(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir)
    calls = _count_lookups(monkeypatch)
    assert _post(client)[0].status_code == 200
    assert _post(client)[0].status_code == 200
    assert len(calls) == 1


def test_revoke_invalidates_cache_immediately(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    assert _post(client)[0].status_code == 200
    _repo_call(client, tmp_data_dir, "revoke", "tok1")
    assert _post(client)[0].status_code == 401


def test_pause_via_update_invalidates(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    assert _post(client)[0].status_code == 200
    _repo_call(client, tmp_data_dir, "update", "tok1", paused=True)
    r = _post(client)[0]
    assert r.status_code == 403
    _repo_call(client, tmp_data_dir, "update", "tok1", paused=False)
    assert _post(client)[0].status_code == 200


def test_delete_invalidates(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    assert _post(client)[0].status_code == 200
    _repo_call(client, tmp_data_dir, "delete", "tok1")
    assert _post(client)[0].status_code == 401


def test_unknown_secret_is_not_cached(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir)
    calls = _count_lookups(monkeypatch)
    hdr = {"Authorization": "Bearer vw_unknownsecret1234567890abcdef12"}
    for _ in range(2):
        r, _ = _post(client, headers=hdr)
        assert r.status_code == 401
    assert len(calls) == 2


def test_generation_snapshot_before_read_makes_stale_put_dead():
    c = tc.TokenCache()
    gen = tc.generation()
    tc.invalidate_all()
    c.put("h", "STALE", gen)
    assert c.get("h") is tc.MISS


def test_get_returns_a_copy():
    from app.db.repos.tokens import TokenRow

    c = tc.TokenCache()
    row = TokenRow(*_row_args())
    c.put("h", row)
    got = c.get("h")
    assert got == row and got is not row


def _row_args():
    import dataclasses

    from app.db.repos.tokens import TokenRow

    return [None] * len(dataclasses.fields(TokenRow))


def test_revoke_racing_an_inflight_lookup_wins(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir)
    real = TokenRepo.find_by_plaintext

    async def read_then_revoke(self, plaintext):
        row = await real(self, plaintext)
        # The revoke commits after this request read the row, before it puts it.
        async with open_db(str(tmp_data_dir / "vllm-warden.db")) as db2:
            await TokenRepo(db2).revoke("tok1")
        return row

    monkeypatch.setattr(TokenRepo, "find_by_plaintext", read_then_revoke)
    _post(client)  # may be 200: it read the row before the revoke
    monkeypatch.setattr(TokenRepo, "find_by_plaintext", real)
    assert _post(client)[0].status_code == 401


def _clock_at(monkeypatch, value):
    monkeypatch.setattr("app.proxy.auth.sqlite_utc_now", lambda: value[0])


def test_rotate_grace_zero_refuses_cached_old_and_accepts_new(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    assert _post(client)[0].status_code == 200
    _, new_plain, _ = _repo_call(client, tmp_data_dir, "rotate", "tok1", grace_hours=0)
    assert _post(client)[0].status_code == 401
    r, _ = _post(client, headers={"Authorization": f"Bearer {new_plain}"})
    assert r.status_code == 200


def test_rotate_grace_window_keeps_old_secret(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    assert _post(client)[0].status_code == 200
    _, new_plain, _ = _repo_call(client, tmp_data_dir, "rotate", "tok1", grace_hours=24)
    assert _post(client)[0].status_code == 200
    r, _ = _post(client, headers={"Authorization": f"Bearer {new_plain}"})
    assert r.status_code == 200


def test_cached_row_expiring_is_refused_without_a_db_read(tmp_data_dir, client, monkeypatch):
    import sqlite3

    _ready(client, tmp_data_dir)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE api_tokens SET expires_at='2030-01-01 00:00:10' WHERE id='tok1'")
    now = ["2030-01-01 00:00:00"]
    _clock_at(monkeypatch, now)
    calls = _count_lookups(monkeypatch)
    assert _post(client)[0].status_code == 200
    now[0] = "2030-01-01 00:00:11"
    r = _post(client)[0]
    assert r.status_code == 401 and r.json()["detail"] == "token expired"
    assert len(calls) == 1


def test_cache_hit_still_runs_throttle_and_refusal(tmp_data_dir, client, monkeypatch):
    from app.auth.throttle import BearerThrottle

    _ready(client, tmp_data_dir)
    matched = []
    real = BearerThrottle.matched
    monkeypatch.setattr(
        BearerThrottle, "matched", lambda self, s: (matched.append(s), real(self, s))[1]
    )
    calls = _count_lookups(monkeypatch)
    assert _post(client)[0].status_code == 200
    assert _post(client)[0].status_code == 200
    assert len(calls) == 1 and len(matched) == 2
