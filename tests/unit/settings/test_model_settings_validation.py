"""#262: the settings PATCH must refuse what register refuses.

``PATCH /api/models/{id}/settings`` derived its allowlist from ``ModelRow`` and
then wrote whatever arrived: ``json.dumps(v)`` for the JSON columns and the raw
JSON value for everything else. So a string ``extra_args`` round-tripped as a
string and ``args += extra_args`` appended it to argv one CHARACTER at a time; a
``max_batch_size`` of 999 went straight into the KV-reserve math; a
``tensor_parallel_size`` that disagreed with ``gpu_indices`` produced a row that
could not load; and a ``served_model_name`` that collided with another row --
the column is ``UNIQUE`` in SQL -- reached SQLite as an ``IntegrityError`` and
came back as a 500, where register pre-checks and answers 409.

The premise the allowlist always rested on is that a key is patchable only
because register already accepts it. These tests hold the PATCH to the other
half of that sentence: the same VALUES too.
"""
import asyncio
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from tests.conftest import csrf_header, jwt_login, seed_admin_user

URL = "/api/models/m1/settings"


def _row(**over) -> ModelRow:
    base = dict(
        id="m1",
        served_model_name="m1",
        hf_repo="org/repo",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status="pulled",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
    )
    base.update(over)
    return ModelRow(**base)


@pytest.fixture
def auth(client: TestClient, tmp_data_dir: Path) -> dict[str, str]:
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path, allowed_gpu_indices=[0, 1])

    async def _seed():
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).insert(_row())

    asyncio.run(_seed())
    return {**jwt_login(client), **csrf_header(client)}


def _stored(db_path: Path, column: str):
    with sqlite3.connect(db_path) as db:
        return db.execute(f"SELECT {column} FROM models WHERE id = 'm1'").fetchone()[0]


# ---------------------------------------------------------------------------
# Every bound register applies, now applied here too
# ---------------------------------------------------------------------------

#: ``(body, the column that must not have moved)``. Each of these was a 200 with
#: the value written before #262.
REFUSED = [
    ({"served_model_name": "bad name!"}, "served_model_name"),
    ({"served_model_name": ""}, "served_model_name"),
    ({"served_model_name": "x" * 101}, "served_model_name"),
    ({"hf_repo": "not-a-slug"}, "hf_repo"),
    ({"hf_repo": ""}, "hf_repo"),
    ({"hf_config_repo": "not-a-slug"}, "hf_config_repo"),
    ({"tokenizer_repo": "not-a-slug"}, "tokenizer_repo"),
    ({"filename": ""}, "filename"),
    ({"filename": "f" * 513}, "filename"),
    ({"mmproj_filename": ""}, "mmproj_filename"),
    ({"mmproj_filename": "f" * 513}, "mmproj_filename"),
    ({"max_model_len": 0}, "max_model_len"),
    ({"max_model_len": -1}, "max_model_len"),
    ({"max_model_len": "lots"}, "max_model_len"),
    ({"gpu_memory_utilization": 0}, "gpu_memory_utilization"),
    ({"gpu_memory_utilization": 1.5}, "gpu_memory_utilization"),
    ({"max_batch_size": 0}, "max_batch_size"),
    ({"max_batch_size": 999}, "max_batch_size"),
    ({"n_gpu_layers": -1}, "n_gpu_layers"),
    ({"n_gpu_layers": 1000}, "n_gpu_layers"),
    ({"parallelism_strategy": "sideways"}, "parallelism_strategy"),
    ({"tensor_parallel_size": 4}, "tensor_parallel_size"),
    # The one that reached argv: `args += extra_args` iterates a string.
    ({"extra_args": "--enforce-eager"}, "extra_args"),
    ({"extra_args": [1, 2]}, "extra_args"),
    ({"extra_args": {"a": "b"}}, "extra_args"),
    ({"gpu_indices": [0, 0]}, "gpu_indices"),
]


@pytest.mark.parametrize("body,column", REFUSED, ids=[
    f"{list(b)[0]}={list(b.values())[0]!r}"[:60] for b, _ in REFUSED
])
def test_a_value_register_refuses_is_refused_here(
    client: TestClient, auth, tmp_data_dir: Path, body, column
):
    before = _stored(tmp_data_dir / "vllm-warden.db", column)
    r = client.patch(URL, json=body, headers=auth)
    assert r.status_code == 422, r.text
    assert _stored(tmp_data_dir / "vllm-warden.db", column) == before


@pytest.mark.parametrize("body,_column", REFUSED, ids=[
    f"{list(b)[0]}"[:60] for b, _ in REFUSED
])
def test_register_refuses_the_same_values(
    client: TestClient, tmp_data_dir: Path, body, _column
):
    """The premise, stated as a test: a key is patchable only because register
    already accepts it, so the two routes must agree on the values as well."""
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0, 1])
    headers = {**jwt_login(client), **csrf_header(client)}
    r = client.post("/api/models", json={
        "served_model_name": "reg", "hf_repo": "org/repo", "gpu_indices": [0],
        **body,
    }, headers=headers)
    assert r.status_code == 422, r.text


def test_the_cross_field_check_runs_on_a_merged_row(
    client: TestClient, auth, tmp_data_dir: Path
):
    """Register ties ``tensor_parallel_size`` to ``len(gpu_indices)``; a partial
    body cannot express that, which is why the PATCH validates the MERGED row.
    """
    r = client.patch(URL, json={"gpu_indices": [0, 1]}, headers=auth)
    assert r.status_code == 422, r.text
    assert "tensor_parallel_size" in r.text
    assert _stored(tmp_data_dir / "vllm-warden.db", "gpu_indices") == "[0]"

    ok = client.patch(
        URL, json={"gpu_indices": [0, 1], "tensor_parallel_size": 2}, headers=auth
    )
    assert ok.status_code == 200, ok.text
    assert _stored(tmp_data_dir / "vllm-warden.db", "gpu_indices") == "[0, 1]"


def test_a_colliding_served_model_name_is_a_409_not_a_500(
    client: TestClient, auth, tmp_data_dir: Path
):
    """``served_model_name`` is UNIQUE in SQL. The PATCH never looked, so a
    collision surfaced as an uncaught ``sqlite3.IntegrityError``."""
    db_path = tmp_data_dir / "vllm-warden.db"

    async def _seed_second():
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).insert(_row(id="m2", served_model_name="taken"))

    asyncio.run(_seed_second())

    r = client.patch(URL, json={"served_model_name": "taken"}, headers=auth)
    assert r.status_code == 409, r.text
    assert "already exists" in r.text
    assert _stored(db_path, "served_model_name") == "m1"


def test_renaming_to_a_free_name_still_works(client: TestClient, auth, tmp_data_dir):
    r = client.patch(URL, json={"served_model_name": "renamed"}, headers=auth)
    assert r.status_code == 200, r.text
    assert _stored(tmp_data_dir / "vllm-warden.db", "served_model_name") == "renamed"


def test_a_no_op_rename_to_its_own_name_is_not_a_collision(client: TestClient, auth):
    assert client.patch(URL, json={"served_model_name": "m1"}, headers=auth).status_code == 200


# ---------------------------------------------------------------------------
# The refusals that already worked keep their status codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body,status", [
    ({"status": "loaded"}, 400),
    ({"prior_status": "loaded"}, 400),
    ({"definitely_not_a_column": 1}, 400),
    ({"backend": "sglang"}, 400),
    ({"backend": 7}, 400),
    ({"supports_vision": "no"}, 400),
    ({"supports_vision": 1}, 400),
    ({"gpu_indices": "0"}, 400),
    ({"gpu_indices": []}, 400),
    ({"gpu_indices": [0, 5], "tensor_parallel_size": 2}, 400),
    ({"extra_env": {"LD_PRELOAD": "/tmp/x.so"}}, 422),
    ({"extra_env": {"FOO_BAR": "1"}}, 422),
    ({"extra_env": "VLLM_USE_V1=1"}, 422),
])
def test_an_existing_refusal_keeps_its_status(client: TestClient, auth, body, status):
    assert client.patch(URL, json=body, headers=auth).status_code == status, body


def test_a_valid_patch_still_succeeds(client: TestClient, auth, tmp_data_dir: Path):
    r = client.patch(URL, json={
        "max_model_len": 4096,
        "extra_args": ["--enforce-eager"],
        "extra_env": {"VLLM_USE_V1": "1"},
        "dtype": "bfloat16",
    }, headers=auth)
    assert r.status_code == 200, r.text
    db_path = tmp_data_dir / "vllm-warden.db"
    assert _stored(db_path, "max_model_len") == 4096
    assert _stored(db_path, "extra_args") == '["--enforce-eager"]'
    assert _stored(db_path, "dtype") == "bfloat16"


def test_an_empty_patch_is_still_a_200(client: TestClient, auth):
    assert client.patch(URL, json={}, headers=auth).status_code == 200


# ---------------------------------------------------------------------------
# Rows written before these rules existed
# ---------------------------------------------------------------------------


def test_a_row_that_would_now_be_refused_still_reads(
    client: TestClient, auth, tmp_data_dir: Path
):
    """Validation is a WRITE-path rule.

    A row already in the database that `ModelSpec` would refuse -- written
    before the rule existed, or straight into SQLite -- must keep reading, on
    every read surface. Nothing on the read path validates, and this pins that
    it stays that way: a model an operator cannot see is a model they cannot
    fix.
    """
    db_path = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE models SET max_batch_size = 999, max_model_len = 0, "
            "extra_env = ?, extra_args = ? WHERE id = 'm1'",
            ('{"FOO_BAR": "1"}', '["--ok"]'),
        )
        db.commit()

    listing = client.get("/api/models", headers=auth)
    assert listing.status_code == 200, listing.text
    assert [m["id"] for m in listing.json()["models"]] == ["m1"]

    detail = client.get("/api/models/m1", headers=auth)
    assert detail.status_code == 200, detail.text

    settings = client.get("/api/models/m1/settings", headers=auth)
    assert settings.status_code == 200, settings.text
    assert settings.json()["max_batch_size"] == 999
    assert settings.json()["extra_env"] == {"FOO_BAR": "1"}


def test_a_row_that_would_now_be_refused_can_be_repaired_in_one_patch(
    client: TestClient, auth, tmp_data_dir: Path
):
    """...and a change to it is refused until the offending column is fixed,
    which the same request can do."""
    db_path = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE models SET extra_env = ? WHERE id = 'm1'", ('{"FOO_BAR": "1"}',)
        )
        db.commit()

    blocked = client.patch(URL, json={"dtype": "bfloat16"}, headers=auth)
    assert blocked.status_code == 422, blocked.text
    assert "FOO_BAR" in blocked.text
    assert _stored(db_path, "dtype") == "auto"

    fixed = client.patch(
        URL, json={"dtype": "bfloat16", "extra_env": {}}, headers=auth
    )
    assert fixed.status_code == 200, fixed.text
    assert _stored(db_path, "dtype") == "bfloat16"
