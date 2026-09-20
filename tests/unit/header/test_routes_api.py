"""Tests for ``GET /api/header/metrics/stream`` — the live header
metrics SSE feeding the nav-bar instrument cluster.

We pin:

  * auth gate (SSE ticket required, JWT alone is rejected),
  * payload shape (the eight derived fields the FE consumes),
  * active-model lookup (DB join semantics),
  * cache reuse (the same ``_ProbeCache`` instance backs both this
    endpoint and ``/api/system/gpus``),
  * interval-seconds env clamp.

The SSE generator itself loops indefinitely (it's a live feed); rather
than coaxing the httpx TestClient into closing the stream mid-iter
(which previously hung the suite indefinitely — TestClient buffers
chunks until the response generator returns), we exercise the
generator's building blocks (``_payload``, ``_active_model``,
``_get_cache``, ``_interval_seconds``) as pure unit tests. This pins
the same observable contract — payload shape, active-model lookup,
cache reuse — without depending on TestClient streaming semantics.

Auth and ticket-mint paths are still driven through the real HTTP
client so the request-level gates are exercised end-to-end.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from app.header.routes_api import (
    _active_models,
    _get_cache,
    _interval_seconds,
    _payload,
)
from app.system.gpu import GpuLive, GpuSnapshot
from app.system.routes_gpus import _ProbeCache
from tests.conftest import jwt_login, seed_admin_user


def _seed_done(db_path) -> None:
    seed_admin_user(db_path, allowed_gpu_indices=[0, 1])


def _install_probe(client, snap: GpuSnapshot) -> dict[str, int]:
    counter: dict[str, int] = {"n": 0}

    async def fake_probe() -> GpuSnapshot:
        counter["n"] += 1
        return snap

    cache = _ProbeCache(ttl=2.0, clock=lambda: 0.0, probe=fake_probe)
    client.app.state.gpu_probe_cache = cache
    return counter


SNAP_TWO_GPUS = GpuSnapshot(
    gpus=[
        GpuLive(
            index=0, uuid="GPU-aaaa", name="NVIDIA RTX A4000",
            memory_total_mib=16376, memory_used_mib=12450,
            memory_free_mib=3926, utilization_pct=87,
        ),
        GpuLive(
            index=1, uuid="GPU-bbbb", name="NVIDIA RTX A4000",
            memory_total_mib=16376, memory_used_mib=100,
            memory_free_mib=16276, utilization_pct=0,
        ),
    ],
    apps=[],
    probe_error=None,
)


def test_header_metrics_stream_requires_ticket(tmp_data_dir, client):
    """JWT alone is not sufficient — the route requires a single-use
    SSE ticket (same as model log streams). Without a ticket query
    string the response should be rejected even with a valid Bearer.

    Without a ticket require_sse_ticket answers 401 `missing ticket` --
    a session JWT header does not replace the ticket (only an admin
    token's does). With an *invalid* ticket the response is 401. Either
    way the stream MUST NOT start — we assert both branches to pin the
    contract.
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    # 1) Missing ticket → 401 from require_sse_ticket (JWT header ignored).
    r = client.get("/api/header/metrics/stream", headers=auth)
    assert r.status_code == 401, r.text
    assert r.json()["detail"] == "missing ticket"
    # 2) Bogus ticket → 401 from require_sse_ticket consume().
    r = client.get(
        "/api/header/metrics/stream?ticket=not-a-real-ticket", headers=auth
    )
    assert r.status_code == 401, r.text


def test_header_metrics_stream_rejects_unauthenticated_ticket_mint(
    tmp_data_dir, client
):
    """No JWT → ticket mint 401s, so the operator can't even acquire
    the ticket needed to open the stream."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/api/auth/sse-ticket",
        json={"path": "/api/header/metrics/stream"},
    )
    assert r.status_code == 401


def test_header_metrics_ticket_mint_succeeds_with_jwt(tmp_data_dir, client):
    """The ticket-mint path itself (used by the FE singleton) must
    succeed when JWT-authenticated. This pins the contract the FE
    relies on without opening the streaming response."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    r = client.post(
        "/api/auth/sse-ticket",
        json={"path": "/api/header/metrics/stream"},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body.get("ticket"), str) and body["ticket"]


def test_payload_emits_all_required_fields_with_no_loaded_model():
    """``_payload`` derives the eight FE-consumed fields correctly and
    surfaces ``active_model: null`` when no model is loaded."""
    out = _payload(SNAP_TWO_GPUS, [])
    assert out["active_model"] is None
    assert out["active_model_id"] is None
    assert out["probe_error"] is None
    assert out["vram_used_mib"] == 12550  # 12450 + 100
    assert out["vram_total_mib"] == 32752  # 16376 * 2
    # max across GPUs.
    assert out["gpu_util_pct"] == 87
    # 12550 / 32752 ≈ 0.383 → 38 %.
    assert out["vram_pct"] == 38
    assert len(out["gpus"]) == 2
    assert out["gpus"][0]["name"] == "NVIDIA RTX A4000"
    assert out["gpus"][0]["memory_used_mib"] == 12450
    # The timestamp ends in Z (RFC-3339 UTC).
    assert out["ts"].endswith("Z")


def test_payload_surfaces_active_model_when_supplied():
    """When the active-model tuple is non-null, ``_payload`` echoes
    both ``active_model_id`` and ``active_model`` into the frame."""
    out = _payload(SNAP_TWO_GPUS, [("m-loaded", "gpt-oss-20b", "loaded")])
    assert out["active_model"] == "gpt-oss-20b"
    assert out["active_model_id"] == "m-loaded"


def test_payload_with_empty_snapshot_surfaces_probe_error():
    """An empty GPU snapshot with a probe_error string surfaces through
    the payload so the FE can degrade gracefully -- and reports NO numbers:
    a lost GPU probe beside a serving model must not read as an idle 0 %
    card (#255)."""
    snap = GpuSnapshot(gpus=[], apps=[], probe_error="nvidia-smi exit 255: Failed to initialize NVML: Unknown Error")
    out = _payload(snap, [("m1", "qwen", "loaded")])
    assert out["probe_error"] == "nvidia-smi exit 255: Failed to initialize NVML: Unknown Error"
    assert out["gpus"] == []
    assert out["vram_total_mib"] is None
    assert out["vram_used_mib"] is None
    assert out["vram_pct"] is None
    assert out["gpu_util_pct"] is None
    assert out["active_model_status"] == "loaded"


def test_payload_with_no_cards_and_no_error_reports_no_numbers():
    """No card answered and nothing failed (a CPU-only box): still null,
    not 0 % of nothing."""
    out = _payload(GpuSnapshot(gpus=[], apps=[], probe_error=None), [])
    assert out["vram_pct"] is None
    assert out["gpu_util_pct"] is None


async def test_active_model_returns_none_when_no_loaded_row(
    tmp_data_dir, client
):
    """``_active_model`` returns ``(None, None)`` when no models row has
    status='loaded'."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    settings = client.app.state.settings
    out = await _active_models(settings.db_path)
    assert out == []


async def test_active_model_returns_tuple_when_loaded_row_present(
    tmp_data_dir, client
):
    """When a models row is status='loaded' AND has a model_runtime
    sibling, ``_active_model`` returns ``(model_id, served_model_name)``.
    The join on ``model_runtime`` ensures the supervisor actually owns
    the loaded row (not just the DB)."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loaded', 'gpt-oss-20b', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loaded')"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) "
            "VALUES ('m-loaded', 99999, 11000)"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_models(settings.db_path)
    assert out == [("m-loaded", "gpt-oss-20b", "loaded")]


async def test_active_model_skips_loaded_row_without_runtime(
    tmp_data_dir, client
):
    """A models row marked 'loaded' but with no model_runtime sibling
    must NOT surface — the supervisor doesn't own it. Pins the JOIN
    semantics (LEFT JOIN would have leaked stale state)."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-orphan', 'orphaned-model', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loaded')"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_models(settings.db_path)
    assert out == []


async def test_get_cache_reuses_system_gpus_cache(client):
    """``_get_cache`` returns ``app.state.gpu_probe_cache`` so a tab
    open on both views does not double the nvidia-smi load. Drive
    /api/system/gpus first to populate the cache, then assert the
    helper hands back the same instance (not a fresh ``_ProbeCache``)."""
    client.get("/healthz")
    # Force-install a known cache instance.
    counter = _install_probe(client, SNAP_TWO_GPUS)
    installed = client.app.state.gpu_probe_cache

    # Fake request that points at the app state — _get_cache reads
    # ``request.app.state.gpu_probe_cache``.
    fake_request = SimpleNamespace(app=client.app)
    got = _get_cache(fake_request)  # type: ignore[arg-type]
    assert got is installed

    # Driving the cache from the helper must reuse the same probe
    # function — calling .get() twice within the TTL should run the
    # probe exactly once. Pins the "no double probe" contract.
    await got.get()
    await got.get()
    assert counter["n"] == 1, f"expected single probe, got {counter['n']}"


def test_get_cache_lazy_initialises_when_absent(client):
    """If ``app.state.gpu_probe_cache`` was never set, the helper
    installs a fresh ``_ProbeCache`` on first call (and reuses it on
    subsequent calls — same identity)."""
    client.get("/healthz")
    # Wipe any prior probe cache the test client may have lazily
    # installed during /healthz. Starlette's State raises ``KeyError``
    # (not AttributeError) when the key is absent.
    try:
        del client.app.state.gpu_probe_cache
    except (AttributeError, KeyError):
        pass

    fake_request = SimpleNamespace(app=client.app)
    first = _get_cache(fake_request)  # type: ignore[arg-type]
    second = _get_cache(fake_request)  # type: ignore[arg-type]
    assert isinstance(first, _ProbeCache)
    assert first is second


def test_header_metrics_interval_clamps_to_floor():
    """``_interval_seconds`` floors at 0.5s so a misconfigured env can't
    spin the SSE loop hot."""
    import os

    saved = os.environ.get("VW_HEADER_METRICS_INTERVAL_S")
    try:
        os.environ["VW_HEADER_METRICS_INTERVAL_S"] = "0.01"
        assert _interval_seconds() == 0.5
        os.environ["VW_HEADER_METRICS_INTERVAL_S"] = "not-a-number"
        assert _interval_seconds() == 2.0
        os.environ.pop("VW_HEADER_METRICS_INTERVAL_S", None)
        assert _interval_seconds() == 2.0
        os.environ["VW_HEADER_METRICS_INTERVAL_S"] = "5.0"
        assert _interval_seconds() == 5.0
    finally:
        if saved is None:
            os.environ.pop("VW_HEADER_METRICS_INTERVAL_S", None)
        else:
            os.environ["VW_HEADER_METRICS_INTERVAL_S"] = saved


async def test_active_model_surfaces_loading_row_without_runtime(
    tmp_data_dir, client
):
    """A row still in 'loading' must surface so the header badge can say
    "loading" rather than falling through to "idle".

    A loading row has NO ``model_runtime`` sibling — the supervisor writes
    that only once the engine answers — so this case is exactly the one the
    strict JOIN used to drop. A 27B load takes minutes; reading "idle" for
    that whole window told the operator the opposite of the truth.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loading', 'qwen3.8-27b', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loading')"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_models(settings.db_path)
    assert out == [("m-loading", "qwen3.8-27b", "loading")]


async def test_active_model_prefers_loaded_over_loading(tmp_data_dir, client):
    """With both a serving row and a loading row present, the serving one
    wins the identity slot — the badge must not demote a live engine to
    "loading" because a second model started loading beside it."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loading', 'incoming', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loading')"
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loaded', 'serving', 'o/r', 'main', '[1]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loaded')"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) "
            "VALUES ('m-loaded', 99999, 11000)"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_models(settings.db_path)
    assert out[0] == ("m-loaded", "serving", "loaded")
    # Both rows surface now -- the header shows every model, not the first.
    # What "prefers" means is ORDER: the serving row leads, so the legacy
    # singular fields (and any client that renders only one chip) still name
    # the live engine rather than a neighbour that just started loading.
    assert len(out) == 2


def test_payload_surfaces_active_model_status():
    """``_payload`` echoes the status so the frontend can pick the label
    and dot colour without re-deriving state from the name alone."""
    out = _payload(SNAP_TWO_GPUS, [("m-loading", "qwen3.8-27b", "loading")])
    assert out["active_model"] == "qwen3.8-27b"
    assert out["active_model_id"] == "m-loading"
    assert out["active_model_status"] == "loading"


def test_payload_active_model_status_is_null_when_idle():
    """No model at all → the status field is present but null, so the
    frontend never has to distinguish "absent key" from "nothing loaded"."""
    out = _payload(SNAP_TWO_GPUS, [])
    assert out["active_model"] is None
    assert out["active_model_status"] is None


async def test_active_model_surfaces_failed_row(tmp_data_dir, client):
    """A crashed model must surface as 'failed' so the badge can go red.

    This is the state that matters most operationally and was the most
    invisible: after an engine death the row sits in 'failed' indefinitely
    (nothing auto-restores it), while the header cheerfully read "idle" —
    indistinguishable from a clean box with nothing loaded.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-failed', 'crashed-model', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'failed')"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_models(settings.db_path)
    assert out == [("m-failed", "crashed-model", "failed")]


async def test_active_model_prefers_loading_over_failed(tmp_data_dir, client):
    """Retrying a crashed model must flip the badge to 'loading', not leave
    it red — the operator's retry has to be visibly acknowledged."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-failed', 'crashed-model', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'failed')"
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loading', 'incoming', 'o/r', 'main', '[1]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loading')"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_models(settings.db_path)
    assert out[0] == ("m-loading", "incoming", "loading")
    assert len(out) == 2


async def test_active_model_prefers_loaded_over_failed(tmp_data_dir, client):
    """A live engine outranks an unrelated stale 'failed' row — one bad row
    must never paint a serving box red."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-failed', 'crashed-model', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'failed')"
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loaded', 'serving', 'o/r', 'main', '[1]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loaded')"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) "
            "VALUES ('m-loaded', 99999, 11000)"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_models(settings.db_path)
    assert out[0] == ("m-loaded", "serving", "loaded")
    # Both rows surface now -- the header shows every model, not the first.
    # What "prefers" means is ORDER: the serving row leads, so the legacy
    # singular fields (and any client that renders only one chip) still name
    # the live engine rather than a neighbour that just started loading.
    assert len(out) == 2


# ===========================================================================
# N loaded models -> N status entries.
#
# The header chip named ONE model. ``_active_model`` ended in LIMIT 1, under a
# comment reading "the supervisor enforces single-model loading, so multiple
# 'loaded' rows would be a defect we don't paper over here". That stopped being
# true: the operator runs llama-3.1-8b on vLLM/GPU 1 and qwen3.8-27b on
# llama.cpp/GPU 0 at the same time, and the header showed only qwen3.8-27b.
#
# LIMIT 1 was not a display choice, it was a load-bearing assumption baked into
# SQL. Removing it is the fix; the ORDER BY is what stays, because with several
# rows the order decides which one the legacy singular fields name.
# ===========================================================================


def _insert_model(db, mid, name, status, gpu="[0]"):
    db.execute(
        "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
        "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
        "gpu_memory_utilization, trust_remote_code, extra_args, status) "
        f"VALUES ('{mid}', '{name}', 'o/r', 'main', '{gpu}', 1, "
        f"NULL, NULL, 0.9, 0, '[]', '{status}')"
    )


async def test_two_loaded_models_produce_two_entries(tmp_data_dir, client):
    """The operator's actual fleet. Two engines, two cards, two entries."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        _insert_model(db, "m-vllm", "llama-3.1-8b", "loaded", "[1]")
        _insert_model(db, "m-lcpp", "qwen3.8-27b", "loaded", "[0]")
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) "
            "VALUES ('m-vllm', 1, 11000)"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) "
            "VALUES ('m-lcpp', 2, 11001)"
        )
        db.commit()

    out = await _active_models(client.app.state.settings.db_path)
    assert len(out) == 2
    assert {row[1] for row in out} == {"llama-3.1-8b", "qwen3.8-27b"}


async def test_n_loaded_models_produce_n_entries(tmp_data_dir, client):
    """The same property at N, because two is not a design.

    A fix that returned "the first two" would pass the test above. The header
    has to be built for however many the operator loads, so the invariant under
    test is the count, not the number two.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    n = 5
    with sqlite3.connect(db_path) as db:
        for i in range(n):
            _insert_model(db, f"m{i}", f"model-{i}", "loaded", f"[{i}]")
            db.execute(
                "INSERT INTO model_runtime(model_id, pid, port) "
                f"VALUES ('m{i}', {100 + i}, {11000 + i})"
            )
        db.commit()

    out = await _active_models(client.app.state.settings.db_path)
    assert len(out) == n
    assert [row[1] for row in out] == [f"model-{i}" for i in range(n)]


async def test_active_models_order_is_stable_across_calls(tmp_data_dir, client):
    """The chip repaints every 2 seconds. If the order came out of the
    database's whim, a two-model header would visibly reshuffle on every tick.

    Within a status class the tiebreak is served_model_name, not updated_at:
    updated_at moves under a running fleet, and an order that changes when a
    model is merely touched is the same flicker with a slower period.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        _insert_model(db, "m-z", "zeta", "loaded", "[0]")
        _insert_model(db, "m-a", "alpha", "loaded", "[1]")
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) VALUES ('m-z', 1, 1)"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) VALUES ('m-a', 2, 2)"
        )
        db.commit()

    settings = client.app.state.settings
    first = await _active_models(settings.db_path)
    second = await _active_models(settings.db_path)
    assert first == second
    assert [row[1] for row in first] == ["alpha", "zeta"]


async def test_a_failed_model_does_not_hide_a_serving_one(tmp_data_dir, client):
    """Both are reported, and the serving one leads.

    With one slot this was a choice between two truths. With N slots it stops
    being a choice: the operator sees that one engine is serving AND that
    another is dead, which is what they need to act on.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        _insert_model(db, "m-bad", "crashed", "failed", "[0]")
        _insert_model(db, "m-ok", "serving", "loaded", "[1]")
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) VALUES ('m-ok', 1, 1)"
        )
        db.commit()

    out = await _active_models(client.app.state.settings.db_path)
    assert out == [("m-ok", "serving", "loaded"), ("m-bad", "crashed", "failed")]


def test_payload_carries_every_active_model():
    out = _payload(
        SNAP_TWO_GPUS,
        [
            ("m-vllm", "llama-3.1-8b", "loaded"),
            ("m-lcpp", "qwen3.8-27b", "loaded"),
        ],
    )
    assert out["active_models"] == [
        {"id": "m-vllm", "served_model_name": "llama-3.1-8b", "status": "loaded"},
        {"id": "m-lcpp", "served_model_name": "qwen3.8-27b", "status": "loaded"},
    ]


def test_payload_legacy_singular_fields_mirror_the_first_entry():
    """A UI pod can outrun its api pod, and vice versa.

    active_model / active_model_id / active_model_status predate this change
    and are still the only fields an older bundle reads. They keep naming the
    leading model rather than disappearing, so a stale UI shows one correct
    model instead of "idle" while two engines serve.
    """
    out = _payload(
        SNAP_TWO_GPUS,
        [
            ("m-vllm", "llama-3.1-8b", "loaded"),
            ("m-lcpp", "qwen3.8-27b", "loaded"),
        ],
    )
    assert out["active_model"] == "llama-3.1-8b"
    assert out["active_model_id"] == "m-vllm"
    assert out["active_model_status"] == "loaded"


def test_payload_active_models_is_an_empty_list_when_idle():
    """Never null. A client that maps over it must not have to null-check
    first -- the same reason active_model_status is always present."""
    out = _payload(SNAP_TWO_GPUS, [])
    assert out["active_models"] == []
    assert out["active_model"] is None


def test_vram_pct_pools_the_whole_box_and_stays_meaningful_for_n_models():
    """VRAM% is a question about the BOX, not about a model, so N models
    change nothing about what it means: used across all cards over total
    across all cards. Two models on two cards fill one pool."""
    out = _payload(SNAP_TWO_GPUS, [])
    assert out["vram_used_mib"] == 12550
    assert out["vram_total_mib"] == 32752
    assert out["vram_pct"] == 38


def test_gpu_util_is_the_busiest_card_never_an_average():
    """The readout is a MAX, and with several cards that has to stay a max.

    An average would report 50% for a box with one card pinned at 90 and one
    at 10 -- a number that describes neither card and reads as "comfortable"
    while a model saturates its GPU. The max answers the question a header
    chip is actually asked: is anything hot right now? The per-card breakdown
    is one hover away in the tooltip.
    """
    snap = GpuSnapshot(
        gpus=[
            GpuLive(
                index=0, uuid="GPU-a", name="A",
                memory_total_mib=100, memory_used_mib=1,
                memory_free_mib=99, utilization_pct=10,
            ),
            GpuLive(
                index=1, uuid="GPU-b", name="B",
                memory_total_mib=100, memory_used_mib=1,
                memory_free_mib=99, utilization_pct=90,
            ),
        ],
        apps=[],
        probe_error=None,
    )
    out = _payload(snap, [])
    assert out["gpu_util_pct"] == 90
