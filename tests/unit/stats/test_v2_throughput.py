"""GET /api/stats/v2/throughput -- prefill and generation tokens per second.

The stats page used to show ONE "Tokens / sec" number: prompt plus completion
over the last full minute. That blends two different quantities (how fast the
engine prefills, how fast it generates) into a figure that is neither, so this
endpoint separates them and reports each as average / max / mode.

Two bases, because "how fast is the rig" and "how much work did the box do"
are different questions with different right answers:

  basis=request     per-request engine speed, derived from request_history.
                    Idle time is invisible to it.
  basis=wallclock   tokens counted per minute from model_samples, idle minutes
                    included as zero.
"""

import sqlite3
import time

import pytest

from tests.conftest import jwt_login, seed_admin_user


def _seed_models(db_path):
    seed_admin_user(db_path)
    with sqlite3.connect(db_path) as db:
        for mid, name in (("id-model-a", "model-a"), ("id-model-b", "model-b")):
            db.execute(
                "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
                "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
                "gpu_memory_utilization, trust_remote_code, extra_args, status) "
                f"VALUES ('{mid}', '{name}', 'o/r', 'main', '[0]', 1, NULL, NULL, "
                "0.9, 0, '[]', 'loaded')"
            )
        db.commit()


def _insert_requests(db_path, rows):
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "INSERT INTO request_history(id, finished_at, model_id, model, token_name, "
            "client_ip, prompt_tokens, completion_tokens, duration_s, ttft_s, "
            "finish_reason, orphan, started_iso) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        db.commit()


def _insert_minutes(db_path, rows):
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "INSERT INTO model_samples(model_id, minute, requests, prompt_tokens, "
            "completion_tokens) VALUES (?,?,?,?,?)",
            rows,
        )
        db.commit()


def _req(i, *, at, model="a", prompt=100, gen=50, dur=2.0, ttft=0.5):
    return (
        f"r{i}", at, f"id-model-{model}", f"model-{model}", "key-a", "10.0.0.1",
        prompt, gen, dur, ttft, "stop", 0, "2026-09-10T18:59:00Z",
    )


def _ready(client, tmp_data_dir):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_models(db_path)
    return db_path, jwt_login(client)


def _get(client, auth, qs=""):
    r = client.get(f"/api/stats/v2/throughput{qs}", headers=auth)
    assert r.status_code == 200, r.text
    return r.json()


# ---- contract ---------------------------------------------------------------


def test_throughput_requires_a_session(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    assert client.get("/api/stats/v2/throughput").status_code == 401


def test_throughput_rejects_an_unknown_basis_and_an_unknown_range(tmp_data_dir, client):
    _, auth = _ready(client, tmp_data_dir)
    r = client.get("/api/stats/v2/throughput?basis=guess", headers=auth)
    assert r.status_code == 400
    assert "basis" in r.json()["detail"]
    assert client.get("/api/stats/v2/throughput?range=3h", headers=auth).status_code == 400


# ---- basis=request ----------------------------------------------------------


def test_request_basis_derives_prefill_and_generation_from_each_request(
    tmp_data_dir, client
):
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    _insert_requests(db_path, [
        # 100 prompt tokens in 0.5s -> 200 tok/s prefill.
        # 49 gaps across 1.5s of decode -> 32.67 tok/s generation.
        _req(1, at=now - 10, prompt=100, ttft=0.5, dur=2.0, gen=50),
        # 100 prompt tokens in 1.0s -> 100 tok/s. 10 gaps across 2.0s -> 5 tok/s.
        _req(2, at=now - 20, prompt=100, ttft=1.0, dur=3.0, gen=11),
    ])
    body = _get(client, auth, "?range=1h&basis=request")

    assert body["basis"] == "request"
    assert body["range"] == "1h"
    assert body["prefill"]["count"] == 2
    assert body["prefill"]["avg"] == pytest.approx(150.0)
    assert body["prefill"]["max"] == pytest.approx(200.0)
    assert body["generation"]["count"] == 2
    assert body["generation"]["avg"] == pytest.approx((49 / 1.5 + 5.0) / 2)
    assert body["generation"]["max"] == pytest.approx(49 / 1.5)
    # Two samples is below the mode's minimum — absent, not invented.
    assert body["prefill"]["mode"] is None


def test_request_basis_skips_requests_whose_rate_cannot_be_computed(
    tmp_data_dir, client
):
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    _insert_requests(db_path, [
        _req(1, at=now - 10, prompt=100, ttft=0.5, dur=2.0, gen=50),
        # Non-streaming: no first frame to time, so neither rate exists.
        (
            "r2", now - 11, "id-model-a", "model-a", "key-a", "10.0.0.1",
            100, 50, 2.0, None, "stop", 0, "2026-09-10T18:59:00Z",
        ),
        # One token: a prefill rate, but no gaps to measure generation across.
        _req(3, at=now - 12, prompt=100, ttft=0.5, dur=2.0, gen=1),
    ])
    body = _get(client, auth, "?range=1h&basis=request")
    assert body["prefill"]["count"] == 2  # r1 and r3
    assert body["generation"]["count"] == 1  # r1 only


def test_request_basis_honours_the_window(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    _insert_requests(db_path, [
        _req(1, at=now - 60, ttft=0.5),          # inside 1h
        _req(2, at=now - 2 * 3600, ttft=0.25),   # outside 1h, inside 6h
    ])
    assert _get(client, auth, "?range=1h&basis=request")["prefill"]["count"] == 1
    assert _get(client, auth, "?range=6h&basis=request")["prefill"]["count"] == 2


def test_request_basis_honours_the_model_selection(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    _insert_requests(db_path, [
        _req(1, at=now - 10, model="a", prompt=100, ttft=0.5),   # 200 tok/s
        _req(2, at=now - 11, model="b", prompt=100, ttft=0.1),   # 1000 tok/s
    ])
    body = _get(client, auth, "?range=1h&basis=request&models=id-model-a")
    assert body["selected_model_ids"] == ["id-model-a"]
    assert body["prefill"]["count"] == 1
    assert body["prefill"]["max"] == pytest.approx(200.0)


def test_an_empty_window_reports_absent_rather_than_zero(tmp_data_dir, client):
    _, auth = _ready(client, tmp_data_dir)
    body = _get(client, auth, "?range=1h&basis=request")
    for series in ("prefill", "generation"):
        assert body[series] == {"count": 0, "avg": None, "max": None, "mode": None}


# ---- basis=wallclock --------------------------------------------------------


def test_wallclock_basis_counts_idle_minutes_as_zero_throughput(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now_min = int(time.time() // 60)
    # Five busy minutes at 600 prompt / 60 completion tokens -> 10 and 1 tok/s,
    # then six idle minutes with no row at all.
    _insert_minutes(db_path, [
        ("id-model-a", now_min - m, 1, 600, 60) for m in range(11, 6, -1)
    ])
    body = _get(client, auth, "?range=1h&basis=wallclock")

    assert body["basis"] == "wallclock"
    # Eleven minutes from the first sample to the last COMPLETE minute; the
    # clock may tick mid-test, which adds one more idle minute.
    n = body["prefill"]["count"]
    assert n in (11, 12), n
    # The divisor is every minute, not just the busy ones — that is the whole
    # point of the wall-clock basis.
    assert body["prefill"]["avg"] == pytest.approx(50.0 / n)
    assert body["prefill"]["max"] == pytest.approx(10.0)
    assert body["generation"]["max"] == pytest.approx(1.0)
    # Idle is the most common minute, and it is reported as such.
    assert body["prefill"]["mode"] == 0.0


def test_wallclock_basis_honours_the_model_selection(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now_min = int(time.time() // 60)
    _insert_minutes(db_path, [
        ("id-model-a", now_min - 2, 1, 600, 60),
        ("id-model-b", now_min - 2, 1, 6000, 600),
    ])
    body = _get(client, auth, "?range=1h&basis=wallclock&models=id-model-a")
    assert body["prefill"]["max"] == pytest.approx(10.0)


def test_wallclock_basis_invents_no_minutes_before_history_began(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now_min = int(time.time() // 60)
    _insert_minutes(db_path, [("id-model-a", now_min - 2, 1, 600, 60)])
    # A 7d window over a store that is two minutes old must not report 10,080
    # idle minutes it has no evidence for.
    body = _get(client, auth, "?range=7d&basis=wallclock")
    assert body["prefill"]["count"] <= 3
