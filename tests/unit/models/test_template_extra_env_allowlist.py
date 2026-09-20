"""A template must not be a way round the ``extra_env`` allowlist (#261, F1).

#261's first pass closed the body key on ``POST /api/models`` and the settings
PATCH, and the changelog said the write paths could not drift again. They
could: ``TemplateCreate.extra_env`` had no validator and ``create_model`` takes
``tpl.extra_env`` verbatim whenever the body's is empty, so the exact payload
register refuses was still persistable in two authenticated calls --

    POST /api/models            {"extra_env": {"LD_PRELOAD": ...}}  -> 422
    POST /api/models/templates  {"id": "poison", "extra_env": {...}} -> 201
    POST /api/models            {"template_id": "poison", ...}       -> 201
    -> models.extra_env = {"LD_PRELOAD": "/tmp/evil.so"}

which is the same per-model denial of service #261 is about: ``filter_extra_env``
fails closed on a hard-locked key at launch, so the row cannot be loaded again
until an operator hand-edits it.

``trust_remote_code`` needed exactly this treatment in #256 and for the same
reason, so ``extra_env`` now gets the same three doors: the body key, a template
create, and the MERGED value at register. The merged check is not redundant --
it is the only one that covers a template that was stored before this fix, or
by any future code path that writes ``engine_templates`` without going through
``TemplateCreate``.
"""

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db.database import open_db
from app.templates import store as template_store
from app.templates.registry import ModelTemplate
from tests.conftest import bearer, csrf_header, jwt_login, seed_admin_token, seed_admin_user

MODELS_URL = "/api/models"
TEMPLATES_URL = "/api/models/templates"
POISON = {"LD_PRELOAD": "/tmp/evil.so"}


def _template_ids(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as db:
        return [r[0] for r in db.execute("SELECT id FROM engine_templates").fetchall()]


def _model_env(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path) as db:
        return [
            json.loads(r[0] or "{}")
            for r in db.execute("SELECT extra_env FROM models").fetchall()
        ]


def _store_template(db_path: Path, **over) -> None:
    """Put a template straight in the store, bypassing the API's validator.

    This is what a row saved before the fix looks like -- the case only the
    merged check at register can catch.
    """
    base = dict(
        id="poison",
        label="poison",
        hf_repo="o/r",
        hf_revision="main",
        dtype="auto",
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        extra_env=dict(POISON),
        engine=None,
        source="user",
    )
    base.update(over)

    async def _save() -> None:
        async with open_db(db_path) as db:
            await template_store.save_user_template(db, ModelTemplate(**base))

    asyncio.run(_save())


@pytest.fixture
def auth(client: TestClient, tmp_data_dir: Path) -> dict[str, str]:
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0, 1])
    return {**jwt_login(client), **csrf_header(client)}


def test_the_two_request_template_bypass_is_closed(
    client: TestClient, tmp_data_dir: Path, auth
):
    """The reviewer's reproduction, request for request."""
    db_path = tmp_data_dir / "vllm-warden.db"

    # 1. Register refuses the payload outright -- what #261 fixed.
    direct = client.post(
        MODELS_URL,
        json={
            "served_model_name": "direct",
            "hf_repo": "o/r",
            "gpu_indices": [0],
            "extra_env": POISON,
        },
        headers=auth,
    )
    assert direct.status_code == 422, direct.text

    # 2. The same payload as a template was a 201, and is now refused before
    #    any write.
    tpl = client.post(
        TEMPLATES_URL,
        json={"id": "poison", "label": "poison", "hf_repo": "o/r", "extra_env": POISON},
        headers=auth,
    )
    assert tpl.status_code == 422, tpl.text
    assert _template_ids(db_path) == []

    # 3. ...so step 3 has nothing to register against.
    viatpl = client.post(
        MODELS_URL,
        json={
            "served_model_name": "viatpl",
            "gpu_indices": [0],
            "template_id": "poison",
        },
        headers=auth,
    )
    assert viatpl.status_code == 404, viatpl.text
    assert _model_env(db_path) == []


def test_registering_from_an_already_stored_poisoned_template_is_refused(
    client: TestClient, tmp_data_dir: Path, auth
):
    """The merged check on its own: a template that predates the fix."""
    db_path = tmp_data_dir / "vllm-warden.db"
    _store_template(db_path)
    assert _template_ids(db_path) == ["poison"]

    r = client.post(
        MODELS_URL,
        json={
            "served_model_name": "viatpl",
            "gpu_indices": [0],
            "template_id": "poison",
        },
        headers=auth,
    )
    assert r.status_code == 422, r.text
    # The merged row is validated by ModelSpec now, so the refusal carries
    # FastAPI's list-of-field-errors detail rather than a bare string (#262).
    assert "LD_PRELOAD" in r.text
    assert "hard-locked" in r.text
    # Refused before the insert.
    assert _model_env(db_path) == []


@pytest.mark.parametrize(
    "extra_env",
    [
        {"LD_PRELOAD": "/tmp/evil.so", "HF_TOKEN": "x"},
        {"CUDA_VISIBLE_DEVICES": "0"},
        {"PATH": "/evil"},
        {"FOO_BAR": "1"},
    ],
)
def test_a_template_refuses_exactly_what_register_refuses(
    client: TestClient, tmp_data_dir: Path, auth, extra_env
):
    """Same table as ``test_post_and_patch_agree_on_refusal``, third door."""
    db_path = tmp_data_dir / "vllm-warden.db"
    r = client.post(
        TEMPLATES_URL,
        json={"id": "t", "label": "t", "hf_repo": "o/r", "extra_env": extra_env},
        headers=auth,
    )
    assert r.status_code == 422, r.text
    assert _template_ids(db_path) == []


def test_an_admin_token_cannot_store_one_either(client: TestClient, tmp_data_dir: Path):
    """The template door is open to admin tokens (it only gates
    ``trust_remote_code`` as session-only), so the allowlist has to hold there
    too -- and it answers 422, not the 403 a session-only refusal gives."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path, allowed_gpu_indices=[0])
    _, secret = seed_admin_token(db_path)

    r = client.post(
        TEMPLATES_URL,
        json={"id": "poison", "label": "poison", "hf_repo": "o/r", "extra_env": POISON},
        headers=bearer(secret),
    )
    assert r.status_code == 422, r.text
    assert _template_ids(db_path) == []


def test_an_allowlisted_template_still_works_end_to_end(
    client: TestClient, tmp_data_dir: Path, auth
):
    """The happy path is untouched: a legitimate template saves, and a model
    registered from it inherits its ``extra_env``."""
    db_path = tmp_data_dir / "vllm-warden.db"
    good = {"VLLM_USE_V1": "1", "NCCL_DEBUG": "INFO"}

    tpl = client.post(
        TEMPLATES_URL,
        json={"id": "good", "label": "good", "hf_repo": "o/r", "extra_env": good},
        headers=auth,
    )
    assert tpl.status_code == 201, tpl.text

    r = client.post(
        MODELS_URL,
        json={
            "served_model_name": "viatpl",
            "gpu_indices": [0],
            "template_id": "good",
        },
        headers=auth,
    )
    assert r.status_code == 201, r.text
    assert _model_env(db_path) == [good]


def test_a_vllm_template_is_refused_for_a_llamacpp_model(
    client: TestClient, tmp_data_dir: Path, auth
):
    """The merged check uses the MERGED backend, so a vLLM-only template is
    refused when the body asks for llamacpp rather than silently storing env
    that ``filter_extra_env`` would drop at launch."""
    db_path = tmp_data_dir / "vllm-warden.db"
    tpl = client.post(
        TEMPLATES_URL,
        json={
            "id": "vllmish",
            "label": "vllmish",
            "hf_repo": "o/r",
            "extra_env": {"VLLM_USE_V1": "1"},
        },
        headers=auth,
    )
    assert tpl.status_code == 201, tpl.text

    r = client.post(
        MODELS_URL,
        json={
            "served_model_name": "gguf",
            "gpu_indices": [0],
            "template_id": "vllmish",
            "backend": "llamacpp",
        },
        headers=auth,
    )
    assert r.status_code == 422, r.text
    assert "VLLM_USE_V1" in r.text
    assert _model_env(db_path) == []


# ---------------------------------------------------------------------------
# A template has no backend axis (R1)
# ---------------------------------------------------------------------------


def test_a_llamacpp_only_template_can_be_saved(
    client: TestClient, tmp_data_dir: Path, auth
):
    """The regression the first pass at this fix introduced.

    Validating a template against the *default* backend made llama.cpp's
    allowlist unreachable through templates -- the two lists are disjoint
    (``VLLM_``/``TRITON_``/``NCCL_``/... vs ``GGML_``), so every ``GGML_*``
    template 422'd, #170's "save the working combo" flow included. Which
    backend a template serves is decided by the register body, so the only
    answer available at template-create time is the union.
    """
    db_path = tmp_data_dir / "vllm-warden.db"
    r = client.post(
        TEMPLATES_URL,
        json={
            "id": "gguf",
            "label": "gguf",
            "hf_repo": "o/r",
            "extra_env": {"GGML_CUDA_FORCE_MMQ": "1"},
        },
        headers=auth,
    )
    assert r.status_code == 201, r.text
    assert _template_ids(db_path) == ["gguf"]


def test_a_llamacpp_template_is_held_to_the_backend_it_is_registered_with(
    client: TestClient, tmp_data_dir: Path, auth
):
    """The union is only about *storing* a template. The per-backend rule is
    still enforced at register, where the backend is finally known."""
    db_path = tmp_data_dir / "vllm-warden.db"
    client.post(
        TEMPLATES_URL,
        json={
            "id": "gguf",
            "label": "gguf",
            "hf_repo": "o/r",
            "extra_env": {"GGML_CUDA_FORCE_MMQ": "1"},
        },
        headers=auth,
    )

    as_vllm = client.post(
        MODELS_URL,
        json={"served_model_name": "v", "gpu_indices": [0], "template_id": "gguf"},
        headers=auth,
    )
    assert as_vllm.status_code == 422, as_vllm.text
    assert "GGML_CUDA_FORCE_MMQ" in as_vllm.text
    assert _model_env(db_path) == []

    as_llamacpp = client.post(
        MODELS_URL,
        json={
            "served_model_name": "g",
            "gpu_indices": [0],
            "template_id": "gguf",
            "backend": "llamacpp",
        },
        headers=auth,
    )
    assert as_llamacpp.status_code == 201, as_llamacpp.text
    assert _model_env(db_path) == [{"GGML_CUDA_FORCE_MMQ": "1"}]


def test_the_union_still_refuses_hard_locked_keys(
    client: TestClient, tmp_data_dir: Path, auth
):
    """What the union gives up is the per-backend rule, not the lock.
    HARD_LOCKED_ENV_KEYS is global, and it is the half that matters here."""
    db_path = tmp_data_dir / "vllm-warden.db"
    r = client.post(
        TEMPLATES_URL,
        json={
            "id": "mixed",
            "label": "mixed",
            "hf_repo": "o/r",
            # One key legal for each backend, and one that is locked for both.
            "extra_env": {
                "GGML_CUDA_FORCE_MMQ": "1",
                "VLLM_USE_V1": "1",
                "LD_PRELOAD": "/tmp/evil.so",
            },
        },
        headers=auth,
    )
    assert r.status_code == 422, r.text
    assert _template_ids(db_path) == []


# ---------------------------------------------------------------------------
# An explicit extra_env overrides the template's, including {} (R2)
# ---------------------------------------------------------------------------


def test_an_explicit_empty_extra_env_overrides_the_template(
    client: TestClient, tmp_data_dir: Path, auth
):
    """Without this the merged-value refusal has no operator escape: an empty
    dict fell through the truthiness test to the template's env, so the only
    way past a refusal was to supply a non-empty, backend-legal dict.

    The builtin ``gpt-oss-20b`` carries vLLM env, so this is reachable without
    anyone creating a template at all.
    """
    db_path = tmp_data_dir / "vllm-warden.db"
    r = client.post(
        MODELS_URL,
        json={
            "served_model_name": "gguf",
            "gpu_indices": [0],
            "template_id": "gpt-oss-20b",
            "backend": "llamacpp",
            "extra_env": {},
        },
        headers=auth,
    )
    assert r.status_code == 201, r.text
    assert _model_env(db_path) == [{}]


def test_the_builtin_template_without_an_override_is_still_refused(
    client: TestClient, tmp_data_dir: Path, auth
):
    """The other half of the pair: no override still means the template's env,
    and the merged check still holds it to the backend being registered."""
    db_path = tmp_data_dir / "vllm-warden.db"
    r = client.post(
        MODELS_URL,
        json={
            "served_model_name": "gguf",
            "gpu_indices": [0],
            "template_id": "gpt-oss-20b",
            "backend": "llamacpp",
        },
        headers=auth,
    )
    assert r.status_code == 422, r.text
    assert _model_env(db_path) == []


def test_an_explicit_extra_env_still_wins_over_the_template(
    client: TestClient, tmp_data_dir: Path, auth
):
    """And a non-empty override is not quietly merged with the template's."""
    db_path = tmp_data_dir / "vllm-warden.db"
    r = client.post(
        MODELS_URL,
        json={
            "served_model_name": "v",
            "gpu_indices": [0],
            "template_id": "gpt-oss-20b",
            "extra_env": {"VLLM_USE_V1": "0"},
        },
        headers=auth,
    )
    assert r.status_code == 201, r.text
    assert _model_env(db_path) == [{"VLLM_USE_V1": "0"}]
