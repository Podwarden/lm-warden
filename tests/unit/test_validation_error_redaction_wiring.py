"""The redacting 422 handler is registered app-wide in ``create_app``.

Part 2 of the OVERSEER validation-error decision (planner iter 40): part 1
shipped the redactor in ``app/utils/validation_errors.py``; this pins that
``build_app`` wires ``redacted_validation_exception_handler`` for
``RequestValidationError`` so NO route can echo the request body in a 422.
The legs hit real routes: ``POST /api/models`` (secret ``extra_args`` /
``extra_env`` values) and a second typed write route (``POST /api/tokens``).
The handler must not touch a valid write (leg e).

The ``redacted_validation_exception_handler`` import lives INSIDE the tests:
a module-level import of a name that does not exist yet makes this file an
ImportError on the base, which runs nothing (red for the wrong reason).
"""

import json

from tests.conftest import csrf_header, jwt_login, seed_admin_user


def _seed_done(client, tmp_data_dir, allowed=None):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=allowed)
    return jwt_login(client)


def _assert_redacted(detail):
    """Every detail entry carries only loc/msg/type; nothing else survives."""
    assert isinstance(detail, list) and detail
    for err in detail:
        assert set(err) == {"loc", "msg", "type"}, f"unexpected 422 keys: {sorted(err)}"
        assert err["loc"], "loc must name the failing field"
        assert err["msg"], "msg must be non-empty"


def test_wired_handler_is_the_redacting_one():
    # Red-first leg: the app registers the shared redacting handler for
    # RequestValidationError, not FastAPI's default (which echoes 'input').
    from fastapi.exceptions import RequestValidationError

    from app.main import build_app
    from app.utils.validation_errors import redacted_validation_exception_handler

    app = build_app()
    handler = app.exception_handlers.get(RequestValidationError)
    assert handler is redacted_validation_exception_handler


def test_post_models_redacts_extra_args_secret(tmp_data_dir, client):
    # Leg (a): a body-level type error on gpu_indices trips the parser, and the
    # 422 must not echo the secret extra_args value that rode in the same body.
    auth = _seed_done(client, tmp_data_dir, allowed=[1, 2])
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "x",
            "hf_repo": "a/b",
            "gpu_indices": "not-a-list",
            "extra_args": ["--api-key", "sk-SECRET"],
            "extra_env": {"HF_TOKEN": "hf_SECRET"},
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 422, r.text
    assert "sk-SECRET" not in r.text
    _assert_redacted(r.json()["detail"])


def test_post_models_redacts_extra_env_secret(tmp_data_dir, client):
    # Leg (b): the extra_env value must be redacted the same way.
    auth = _seed_done(client, tmp_data_dir, allowed=[1, 2])
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "x",
            "hf_repo": "a/b",
            "gpu_indices": "not-a-list",
            "extra_env": {"HF_TOKEN": "hf_SECRET"},
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 422, r.text
    assert "hf_SECRET" not in r.text
    _assert_redacted(r.json()["detail"])


def test_post_models_loc_names_field_and_msg_nonempty(tmp_data_dir, client):
    # Leg (c): even redacted, the 422 still says WHICH field failed.
    auth = _seed_done(client, tmp_data_dir, allowed=[1, 2])
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "x",
            "hf_repo": "a/b",
            "gpu_indices": "not-a-list",
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    _assert_redacted(detail)
    # gpu_indices is the failing field; some loc must name it.
    assert any("gpu_indices" in [str(p) for p in err["loc"]] for err in detail)


def test_tokens_redacts_wrong_typed_field(tmp_data_dir, client):
    # Leg (d): a second typed write route -- wrong-typed value absent, loc names
    # the field. expires_in_days must be an int.
    auth = _seed_done(client, tmp_data_dir, allowed=[1, 2])
    r = client.post(
        "/api/tokens",
        json={
            "name": "tok-a",
            "expires_in_days": "not-an-int",
            "priority": 5,
            "anthropic_relay": False,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 422, r.text
    assert "not-an-int" not in r.text
    detail = r.json()["detail"]
    _assert_redacted(detail)
    assert any("expires_in_days" in [str(p) for p in err["loc"]] for err in detail)


def test_valid_post_models_still_201(tmp_data_dir, client):
    # Leg (e): the redacting handler must not swallow a valid write.
    auth = _seed_done(client, tmp_data_dir, allowed=[1, 2])
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "qwen3.5-9b",
            "hf_repo": "Qwen/Qwen3.5-9B",
            "gpu_indices": [1, 2],
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 201, r.text
    body = json.loads(r.text)
    assert body.get("id")
