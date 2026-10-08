"""#294 -- the bookkeeping drop counters leave the process.

``bookkeeping.dropped_total()`` used to be readable only from inside the
proxy. These tests pin the export: the per-site requests-lost totals the
ledger already logged, and the admin endpoint the Stats page reads.
"""

import pytest

from app.proxy import bookkeeping
from tests.conftest import jwt_login, seed_admin_user


@pytest.fixture
def clean_counters(monkeypatch):
    monkeypatch.setattr(bookkeeping, "_dropped", {})
    monkeypatch.setattr(bookkeeping, "_lost", {})


def test_requests_lost_accumulates_per_site(clean_counters):
    bookkeeping.note_dropped("ledger_key_dropped", RuntimeError("x"), requests=3, log=False)
    bookkeeping.note_dropped("ledger_key_dropped", RuntimeError("y"), requests=2, log=False)
    bookkeeping.note_dropped("ledger_batch_dropped", RuntimeError("z"), requests=7, log=False)
    assert bookkeeping.dropped_total() == {"ledger_key_dropped": 2, "ledger_batch_dropped": 1}
    snap = bookkeeping.snapshot()
    assert snap["sites"] == {
        "ledger_key_dropped": {"count": 2, "requests_lost": 5},
        "ledger_batch_dropped": {"count": 1, "requests_lost": 7},
    }
    assert snap["total"] == {"count": 3, "requests_lost": 12}


def test_unknown_requests_lost_is_null_not_zero(clean_counters):
    """A site that drops without knowing how many requests it lost reports
    null: NULL means unknown, never 0 -- and it makes the total unknown too."""
    bookkeeping.note_dropped("ledger_key_dropped", RuntimeError("x"), requests=3, log=False)
    bookkeeping.note_dropped("touch_last_used", RuntimeError("y"), log=False)
    snap = bookkeeping.snapshot()
    assert snap["sites"]["touch_last_used"] == {"count": 1, "requests_lost": None}
    assert snap["sites"]["ledger_key_dropped"] == {"count": 1, "requests_lost": 3}
    assert snap["total"] == {"count": 2, "requests_lost": None}


def test_empty_snapshot(clean_counters):
    snap = bookkeeping.snapshot()
    assert snap["sites"] == {}
    assert snap["total"] == {"count": 0, "requests_lost": 0}
    assert isinstance(snap["since_epoch"], float)


def test_endpoint_requires_a_session(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    assert client.get("/api/stats/v2/bookkeeping").status_code == 401


def test_endpoint_reports_the_counters(tmp_data_dir, client, clean_counters):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    bookkeeping.note_dropped("ledger_key_dropped", RuntimeError("x"), requests=4, log=False)
    r = client.get("/api/stats/v2/bookkeeping", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["sites"] == {"ledger_key_dropped": {"count": 1, "requests_lost": 4}}
    assert body["total"] == {"count": 1, "requests_lost": 4}
