"""``ModelSpec`` refuses ``extra_args`` that set the engine's model-id flag at
WRITE time (OVERSEER model-id decision, planner iter 32, write-time layer).

The engine's model id must come from the ``served_model_name`` column: the proxy
routes and forwards by that column, so an ``extra_args`` flag that overrides the
engine's served name would serve an id the proxy never asks for. This is the
write-time twin of the load-time ``model_id_flag_in`` guards (#model-id-flags);
it mirrors ``ModelSpec._no_parallel_flags_in_extra_args``: the writer validates
the MERGED row, so a legacy row still carrying such a flag stays editable for
fields that do not touch ``extra_args`` (it passes
``context={"extra_args_written": False}``), while a direct construction / a
register stays strict.

New product names are imported INSIDE the test functions, never at module level,
so a base tree missing them fails the assertion, not the collection.
"""

import pytest
from pydantic import ValidationError

from app.models.schemas import ModelCreate, ModelSpec
from tests.conftest import csrf_header, jwt_login, seed_admin_user


def _row_kwargs(**over):
    """A minimal, valid full-row dict the way the writer merges one."""
    base = dict(
        served_model_name="m1",
        hf_repo="org/repo",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# (a) a vllm row + --served-model-name is refused, the flag named
# ---------------------------------------------------------------------------


def test_vllm_row_with_served_model_name_is_refused():
    with pytest.raises(ValidationError) as exc:
        ModelSpec(**_row_kwargs(backend="vllm", extra_args=["--served-model-name", "x"]))
    from app.models.model_id_flags import model_id_refusal

    assert "served_model_name" in str(exc.value)
    assert model_id_refusal("vllm", "--served-model-name") in str(exc.value)
    assert "--served-model-name" in str(exc.value)


# ---------------------------------------------------------------------------
# (b) a llamacpp row + -a (the short alias of --alias) is refused
# ---------------------------------------------------------------------------


def test_llamacpp_row_with_short_alias_is_refused():
    with pytest.raises(ValidationError) as exc:
        ModelSpec(**_row_kwargs(backend="llamacpp", extra_args=["-a", "x"]))
    from app.models.model_id_flags import model_id_refusal

    assert model_id_refusal("llamacpp", "-a") in str(exc.value)
    assert "-a" in str(exc.value)


def test_llamacpp_row_with_long_alias_is_refused():
    with pytest.raises(ValidationError) as exc:
        ModelSpec(**_row_kwargs(backend="llamacpp", extra_args=["--alias", "x"]))
    from app.models.model_id_flags import model_id_refusal

    assert model_id_refusal("llamacpp", "--alias") in str(exc.value)


# ---------------------------------------------------------------------------
# (c) backend None resolves to the registry default (vllm on linux) and is
#     checked
# ---------------------------------------------------------------------------


def test_null_backend_resolves_to_the_registry_default_and_is_checked():
    from app.runtime.backends import registry

    assert registry.DEFAULT_BACKEND == "vllm"
    with pytest.raises(ValidationError) as exc:
        ModelSpec(**_row_kwargs(backend=None, extra_args=["--served-model-name", "x"]))
    from app.models.model_id_flags import model_id_refusal

    assert model_id_refusal("vllm", "--served-model-name") in str(exc.value)


# ---------------------------------------------------------------------------
# (d) extra_args_written False skips the check: a legacy row stays editable for
#     fields that do not touch extra_args
# ---------------------------------------------------------------------------


def test_extra_args_written_false_skips_the_check():
    spec = ModelSpec.model_validate(
        _row_kwargs(backend="vllm", extra_args=["--served-model-name", "x"]),
        context={"extra_args_written": False},
    )
    assert spec.extra_args == ["--served-model-name", "x"]


def test_extra_args_written_true_is_refused():
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            _row_kwargs(backend="vllm", extra_args=["--served-model-name", "x"]),
            context={"extra_args_written": True},
        )


# ---------------------------------------------------------------------------
# (e) unrelated extra_args are accepted
# ---------------------------------------------------------------------------


def test_unrelated_extra_args_are_accepted():
    spec = ModelSpec(
        **_row_kwargs(backend="vllm", extra_args=["--enforce-eager", "--max-model-len", "8192"])
    )
    assert spec.extra_args == ["--enforce-eager", "--max-model-len", "8192"]


def test_empty_extra_args_are_accepted():
    spec = ModelSpec(**_row_kwargs(backend="vllm", extra_args=[]))
    assert spec.extra_args == []


def test_absent_extra_args_are_accepted():
    spec = ModelSpec(**_row_kwargs(backend="vllm"))
    assert spec.extra_args == []


# ---------------------------------------------------------------------------
# the model-id flag does NOT leak across backends: a flag one engine does not
# know is fine on another row (the check is per the row's own backend)
# ---------------------------------------------------------------------------


def test_a_flag_another_backend_uses_is_accepted_here():
    # -a is llamacpp's model-id alias, not vllm's; a vllm row may carry it.
    spec = ModelSpec(**_row_kwargs(backend="vllm", extra_args=["-a", "x"]))
    assert spec.extra_args == ["-a", "x"]


def test_a_vllm_flag_is_accepted_on_a_llamacpp_row():
    # --served-model-name is vllm's; llamacpp does not know it, so the check
    # (run against the row's own backend) does not refuse it.
    spec = ModelSpec(**_row_kwargs(backend="llamacpp", extra_args=["--served-model-name", "x"]))
    assert spec.extra_args == ["--served-model-name", "x"]


# ---------------------------------------------------------------------------
# (f) POST /api/models answers 422 naming the flag (via the existing
#     ValidationError path)
# ---------------------------------------------------------------------------


def test_post_api_models_refuses_served_model_name_with_422(tmp_data_dir, client):
    from app.models.model_id_flags import model_id_refusal

    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0])
    auth = jwt_login(client)
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "x",
            "hf_repo": "o/r",
            "gpu_indices": [0],
            "backend": "vllm",
            "extra_args": ["--served-model-name", "y"],
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 422, r.text
    assert "--served-model-name" in r.text
    assert model_id_refusal("vllm", "--served-model-name") in r.text


def test_post_api_models_accepts_unrelated_extra_args(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0])
    auth = jwt_login(client)
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "x",
            "hf_repo": "o/r",
            "gpu_indices": [0],
            "backend": "vllm",
            "extra_args": ["--enforce-eager"],
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 201, r.text


# ---------------------------------------------------------------------------
# the refusal lands on ModelSpec and therefore on its request-shaped subclass
# ---------------------------------------------------------------------------


def test_model_create_also_refuses():
    with pytest.raises(ValidationError):
        ModelCreate(
            served_model_name="m1",
            hf_repo="org/repo",
            hf_revision="main",
            gpu_indices=[0],
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            backend="vllm",
            extra_args=["--served-model-name", "x"],
        )


# ---------------------------------------------------------------------------
# a write that changes ONLY the backend re-checks the stored extra_args against
# the NEW backend (the flag is per backend, so the row's verdict can flip)
# ---------------------------------------------------------------------------


def test_backend_written_rechecks_unwritten_extra_args():
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            _row_kwargs(backend="llamacpp", extra_args=["-a", "x"]),
            context={"extra_args_written": False, "backend_written": True},
        )


def _seed_row(tmp_data_dir, **over):
    import asyncio

    from app.db.database import open_db
    from app.db.repos.models import ModelRepo, ModelRow

    row = dict(
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
    row.update(over)

    async def _seed():
        async with open_db(str(tmp_data_dir / "vllm-warden.db")) as db:
            await ModelRepo(db).insert(ModelRow(**row))

    asyncio.run(_seed())


def _stored(tmp_data_dir, column):
    import sqlite3

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        return db.execute(f"SELECT {column} FROM models WHERE id = 'm1'").fetchone()[0]


def test_patch_that_changes_only_the_backend_rechecks_extra_args(tmp_data_dir, client):
    from app.models.model_id_flags import model_id_refusal

    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0])
    _seed_row(tmp_data_dir, backend="vllm", extra_args=["-a", "x"])
    headers = {**jwt_login(client), **csrf_header(client)}
    r = client.patch("/api/models/m1/settings", json={"backend": "llamacpp"}, headers=headers)
    assert r.status_code == 422, r.text
    assert model_id_refusal("llamacpp", "-a") in r.text
    assert _stored(tmp_data_dir, "backend") == "vllm"


def test_patch_that_changes_only_the_backend_is_accepted_when_clean(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0])
    _seed_row(tmp_data_dir, backend="vllm", extra_args=["--served-model-name", "x"])
    headers = {**jwt_login(client), **csrf_header(client)}
    # --served-model-name is vllm's flag, not llamacpp's: moving the row to
    # llamacpp leaves nothing that overrides the new engine's id.
    r = client.patch("/api/models/m1/settings", json={"backend": "llamacpp"}, headers=headers)
    assert r.status_code == 200, r.text
    assert _stored(tmp_data_dir, "backend") == "llamacpp"


def test_legacy_row_stays_editable_for_other_fields(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0])
    _seed_row(tmp_data_dir, backend="vllm", extra_args=["--served-model-name", "x"])
    headers = {**jwt_login(client), **csrf_header(client)}
    r = client.patch(
        "/api/models/m1/settings", json={"gpu_memory_utilization": 0.8}, headers=headers
    )
    assert r.status_code == 200, r.text
    assert _stored(tmp_data_dir, "gpu_memory_utilization") == 0.8
    assert client.get("/api/models/m1", headers=headers).status_code == 200
