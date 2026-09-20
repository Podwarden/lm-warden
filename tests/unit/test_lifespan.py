import json
import sqlite3

import pytest

from tests.conftest import jwt_login, seed_admin_user


@pytest.mark.fresh_db
def test_lifespan_creates_db_with_schema(tmp_data_dir, client):
    """Calling any endpoint must trigger lifespan, which migrates the DB.

    ``fresh_db``: the ``client`` fixture normally hands the lifespan a
    pre-migrated copy, which would make this test pass even if the lifespan
    stopped migrating. This is the one test that must watch it happen.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    assert db_path.exists()
    with sqlite3.connect(db_path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "users" in tables
        assert "models" in tables


def test_lifespan_clears_runtime_table(tmp_data_dir, client):
    """Stale model_runtime rows must be wiped on startup."""
    client.get("/healthz")  # boot
    db_path = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        # Insert a model + runtime row, then re-boot via second client.
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes) VALUES "
            "('m1','m1','o/r','main','[0]',1,0.9,0,'[]','loaded',0)"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) VALUES ('m1', 9999, 10000)"
        )
        db.commit()

    # Reboot app
    from fastapi.testclient import TestClient

    from app.main import build_app
    with TestClient(build_app()) as c2:
        c2.get("/healthz")

    with sqlite3.connect(db_path) as db:
        (n,) = db.execute("SELECT COUNT(*) FROM model_runtime").fetchone()
        assert n == 0
        (status,) = db.execute("SELECT status FROM models WHERE id='m1'").fetchone()
        assert status == "failed"


def test_lifespan_reconciles_stranded_rows_before_marking_the_runtime_dead(
    tmp_data_dir, client
):
    """#236 -- the ORDER of the two boot sweeps is the contract.

    ``reconcile_stranded_models`` demotes a transient row with no backing
    process ('loading' with no was-serving flag: an interrupted first load)
    to something an operator can act on. ``mark_runtime_dead_on_startup``
    sweeps every transient row into 'failed' + prior_status. Run the second
    first and the reconcile finds nothing: the row lands in 'failed' with
    prior_status='loading', which ``wants_restart`` reads as a serving model
    to bring back -- a load that never came up, retried forever, instead of a
    'registered'/'pulled' row with a Load button. Every other test drives the
    two functions in isolation; only a real boot sees the order.
    """
    client.get("/healthz")  # first boot: migrated DB
    db_path = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes) VALUES "
            "('stranded','stranded','o/r','main','[0]',1,0.9,0,'[]','loading',0)"
        )
        db.commit()

    from fastapi.testclient import TestClient

    from app.main import build_app

    with TestClient(build_app()) as c2:
        c2.get("/healthz")

    with sqlite3.connect(db_path) as db:
        status, last_error, prior_status = db.execute(
            "SELECT status, last_error, prior_status FROM models WHERE id='stranded'"
        ).fetchone()
    # The HF cache is empty, so the operator-actionable status is 'registered'
    # ('pulled' would be the answer with weights on disk).
    assert status == "registered", (
        f"got {status!r} (last_error={last_error!r}): reconcile_stranded_models must "
        "run BEFORE mark_runtime_dead_on_startup, or the dead-marking sweep has already "
        "turned every transient row into 'failed' and the reconcile is a no-op"
    )
    assert "recovered from an interrupted loading" in (last_error or "")
    assert prior_status is None, (
        "an interrupted first load must not be flagged for automatic restart"
    )


class _DriverSettings:
    """Only the fields ``_build_engine_driver`` reads."""

    def __init__(self, *, engine_driver, hf_cache_dir, tmp_path):
        self.engine_driver = engine_driver
        self.hf_cache_dir = hf_cache_dir
        self.data_dir = tmp_path
        self.engine_log_max_bytes = 0


def _docker_driver(monkeypatch, *, hf_cache_dir, tmp_path):
    """Build the docker driver without a docker daemon."""
    import docker

    monkeypatch.setattr(docker, "from_env", lambda: object())
    monkeypatch.setenv("VLLM_ENGINE_IMAGE", "img:tag")
    from app.main import _build_engine_driver

    return _build_engine_driver(
        _DriverSettings(
            engine_driver="docker", hf_cache_dir=hf_cache_dir, tmp_path=tmp_path
        )
    )


def test_docker_driver_warns_when_hf_cache_dir_is_not_the_engine_mount(
    monkeypatch, caplog, tmp_path
):
    """#265 — the one configuration the dropped ``/data`` mount could break.

    The engine container now gets the model-cache volume and nothing else, at a
    FIXED path. A deployment that moved ``VW_HF_CACHE_DIR`` under ``/data`` was
    reaching its cache *through* the old data mount; without it the engine's
    ``HF_HUB_CACHE`` names a path no volume backs and every model re-downloads
    into the container's writable layer. Being unable to test that live is a
    reason to be loud at startup, not to guess.
    """
    from app.runtime.engine.docker_socket import HFCACHE_MOUNT

    with caplog.at_level("WARNING"):
        _docker_driver(monkeypatch, hf_cache_dir="/data/hf-cache", tmp_path=tmp_path)
    assert any(
        "VW_HF_CACHE_DIR" in r.getMessage() and HFCACHE_MOUNT in r.getMessage()
        for r in caplog.records
    ), caplog.text


def test_docker_driver_is_quiet_on_the_default_cache_dir(monkeypatch, caplog, tmp_path):
    """The warning must not fire on every shipped deployment shape."""
    from pathlib import Path

    from app.runtime.engine.docker_socket import HFCACHE_MOUNT

    with caplog.at_level("WARNING"):
        _docker_driver(
            monkeypatch, hf_cache_dir=Path(HFCACHE_MOUNT), tmp_path=tmp_path
        )
    assert "VW_HF_CACHE_DIR" not in caplog.text


def test_lifespan_cleans_a_hard_locked_extra_env_key_before_serving_a_request(
    tmp_data_dir, client, caplog
):
    """#266 -- a row poisoned by a pre-fix settings PATCH (an extra_env key
    POST /api/models already refused) must be cleaned, and the operator must
    be able to see which model and which key from the boot log alone.

    The log line has to land BEFORE the app can serve a request: an operator
    reading it while the model is still refusing to load, not after, is the
    entire point (a boot-order regression here is silent until someone tries
    to load the model and gets a raw environment error instead).
    """
    client.get("/healthz")  # first boot: migrated DB
    db_path = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, gpu_memory_utilization, trust_remote_code, extra_args, "
            "extra_env, status, pulled_bytes) VALUES "
            "('poisoned','poisoned','o/r','main','[0]',1,0.9,0,'[]',?,'pulled',0)",
            (json.dumps({"VLLM_LOGGING_LEVEL": "DEBUG", "LD_PRELOAD": "/evil.so"}),),
        )
        db.commit()

    from fastapi.testclient import TestClient

    from app.main import build_app

    with caplog.at_level("WARNING", logger="app.runtime.boot_reconcile"):
        # The lifespan runs to completion (poison cleaned, warning logged)
        # BEFORE ``with`` returns control here -- TestClient's context manager
        # drives startup synchronously, so this assertion point is already
        # "after boot, before any request this test sends".
        with TestClient(build_app()) as c2:
            # The log line must already be there even before this first
            # request -- boot reconciliation is not deferred to a request
            # handler.
            assert "poisoned" in caplog.text
            assert "LD_PRELOAD" in caplog.text
            c2.get("/healthz")

    with sqlite3.connect(db_path) as db:
        (extra_env,) = db.execute(
            "SELECT extra_env FROM models WHERE id='poisoned'"
        ).fetchone()
    assert json.loads(extra_env) == {"VLLM_LOGGING_LEVEL": "DEBUG"}


def test_lifespan_prevents_a_wrong_shaped_row_from_500ing_the_models_list(
    tmp_data_dir, client
):
    """#266 verification: this is the MORE urgent half of the two poisoning
    questions the issue raised. ``ModelOut`` types ``extra_args: list[str]``,
    so a row whose column holds a bare string fails FastAPI's response
    validation -- and because ``GET /api/models`` returns every row in ONE
    response, that one bad row would 500 the whole models list, hiding every
    other model on the install too, not just itself. Boot reconciliation
    must land before the first request so this never happens.
    """
    client.get("/healthz")  # first boot: migrated DB
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, gpu_memory_utilization, trust_remote_code, extra_args, "
            "extra_env, status, pulled_bytes) VALUES "
            "('shapeless','shapeless','o/r','main','[0]',1,0.9,0,?,'{}','pulled',0)",
            (json.dumps("--enforce-eager"),),  # a string, not a list -- what
            # a pre-#266-fix PATCH could write verbatim
        )
        db.commit()

    from fastapi.testclient import TestClient

    from app.main import build_app

    with TestClient(build_app()) as c2:
        headers = jwt_login(c2)
        resp = c2.get("/api/models", headers=headers)

    assert resp.status_code == 200, resp.text
    by_id = {m["id"]: m for m in resp.json()["models"]}
    assert by_id["shapeless"]["extra_args"] == []
