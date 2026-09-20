"""Issue #256: an admin token must not be able to make the warden execute
arbitrary code via ``trust_remote_code``.

A final security review found that an admin token could register a model with
``trust_remote_code: true`` and put it in front of the warden -- code running
with the warden's privileges can read ``VW_JWT_SECRET``, mint a session, or
edit ``api_tokens`` / ``admin_audit`` directly. Two refusals close that, both
403 ``session_only``, both only for an admin token (a session is unaffected by
either):

  1. Setting it -- ``POST /api/models`` refuses a body that sets
     ``trust_remote_code: true``, before any write.
  2. Loading it -- ``POST /api/models/{id}/load`` refuses when the target
     row's ``trust_remote_code`` is truthy (a row can predate this change).

#264 corrected the mechanism this file used to describe: the column is never
emitted to any engine, so nothing ever ran "at load time" or "in the engine
process". Its one consumer was the warden's own proxy tokenizer on ``/v1``
traffic, and #264 removed that. The refusals below are unchanged -- they fence
a stored grant an admin token must not be able to write -- and everything this
module asserts still holds.

A follow-up review (C1/I1, 2026-09-19) found three ways past those two, all
closed here and all covered below:

  3. ``PATCH /api/models/{id}/settings`` could set ``trust_remote_code`` on an
     existing row, and could set ``prior_status`` -- the column the watchdog's
     restart sweep keys on. Two requests on a ``failed`` row therefore started
     the engine with the flag on, with no ``/load`` call at all. The PATCH now
     refuses a truthy ``trust_remote_code`` for an admin token, and
     ``prior_status`` is off the patchable surface entirely.
  4. ``POST /api/models/templates`` could create a template carrying the flag.
  5. ``POST /api/models`` only inspected the literal body value, so a
     ``template_id`` whose template carries the flag persisted a true row --
     and the builtin ``gpt-oss-20b`` template carries it, so no template
     create was even needed. The refusal is now on the merged, effective
     value.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

from app.runtime.watchdog import wants_restart
from tests.conftest import bearer, csrf_header, jwt_login, seed_admin_token, seed_admin_user

MODELS_URL = "/api/models"
TEMPLATES_URL = "/api/models/templates"
#: A builtin template that carries ``trust_remote_code=True``
#: (app/templates/registry.py). Registering from it is the shortest bypass.
TRUST_BUILTIN_TEMPLATE = "gpt-oss-20b"


def _seed_done(db_path, allowed=None):
    seed_admin_user(db_path, allowed_gpu_indices=allowed)


def _insert_model(db_path, *, model_id, trust_remote_code, status="pulled", gpus=None):
    gpus = [0] if gpus is None else gpus
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes, pulled_total, last_error) "
            "VALUES (?, ?, 'o/r', 'main', ?, ?, NULL, NULL, 0.9, ?, '[]', ?, 0, NULL, NULL)",
            (model_id, model_id, json.dumps(gpus), len(gpus), int(trust_remote_code), status),
        )
        db.commit()


def _model_ids(db_path) -> list[str]:
    with sqlite3.connect(db_path) as db:
        return [r[0] for r in db.execute("SELECT id FROM models").fetchall()]


def _row(db_path, model_id) -> tuple:
    """``(status, prior_status, trust_remote_code)`` -- the three columns the
    C1 escalation writes."""
    with sqlite3.connect(db_path) as db:
        return db.execute(
            "SELECT status, prior_status, trust_remote_code FROM models WHERE id = ?",
            (model_id,),
        ).fetchone()


def _template_ids(db_path) -> list[str]:
    with sqlite3.connect(db_path) as db:
        return [r[0] for r in db.execute("SELECT id FROM engine_templates").fetchall()]


def _trust(db_path, served_model_name) -> int:
    with sqlite3.connect(db_path) as db:
        return db.execute(
            "SELECT trust_remote_code FROM models WHERE served_model_name = ?",
            (served_model_name,),
        ).fetchone()[0]


@dataclass
class _FakeGpuLive:
    index: int
    memory_total_mib: int = 16376


@dataclass
class _FakeSnap:
    gpus: list = field(default_factory=list)
    apps: list = field(default_factory=list)
    probe_error: str | None = None


class _FakeProbeCache:
    def __init__(self, snap):
        self._snap = snap

    async def get(self):
        return self._snap


def _stub_probe(client, indices):
    client.app.state.gpu_probe_cache = _FakeProbeCache(
        _FakeSnap(gpus=[_FakeGpuLive(index=i) for i in indices])
    )


def _stub_engine():
    """Patch the supervisor + health-check so a load 202s without touching a
    real process, same pattern as tests/unit/models/test_load_endpoint.py."""
    return (
        patch("app.runtime.supervisor.Supervisor.load", new=AsyncMock()),
        patch("app.models.routes_api.wait_for_health", new=AsyncMock(return_value=True)),
    )


# ---------------------------------------------------------------------------
# Setting it: POST /api/models
# ---------------------------------------------------------------------------


def test_admin_token_registering_with_trust_remote_code_true_is_refused(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _, secret = seed_admin_token(db_path)
    r = client.post(MODELS_URL, json={
        "served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0],
        "trust_remote_code": True,
    }, headers=bearer(secret))
    assert r.status_code == 403, r.text
    assert r.json()["detail"]["error_code"] == "session_only"
    # Refused before any write -- nothing was persisted.
    assert _model_ids(db_path) == []


def test_admin_token_registering_with_trust_remote_code_false_succeeds(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _, secret = seed_admin_token(db_path)
    r = client.post(MODELS_URL, json={
        "served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0],
        "trust_remote_code": False,
    }, headers=bearer(secret))
    assert r.status_code == 201, r.text


def test_admin_token_registering_with_trust_remote_code_omitted_succeeds(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _, secret = seed_admin_token(db_path)
    r = client.post(MODELS_URL, json={
        "served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0],
    }, headers=bearer(secret))
    assert r.status_code == 201, r.text


def test_session_can_register_with_trust_remote_code_true(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    auth = jwt_login(client)
    r = client.post(MODELS_URL, json={
        "served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0],
        "trust_remote_code": True,
    }, headers={**auth, **csrf_header(client)})
    assert r.status_code == 201, r.text


# ---------------------------------------------------------------------------
# Loading it: POST /api/models/{id}/load
# ---------------------------------------------------------------------------


def test_admin_token_loading_a_trust_remote_code_model_is_refused(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _insert_model(db_path, model_id="trm", trust_remote_code=True)
    _, secret = seed_admin_token(db_path)
    r = client.post(f"{MODELS_URL}/trm/load", headers=bearer(secret))
    assert r.status_code == 403, r.text
    assert r.json()["detail"]["error_code"] == "session_only"


def test_session_can_load_a_trust_remote_code_model(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _insert_model(db_path, model_id="trm", trust_remote_code=True)
    _stub_probe(client, [0])
    auth = jwt_login(client)
    p1, p2 = _stub_engine()
    with p1, p2:
        r = client.post(f"{MODELS_URL}/trm/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 202, r.text


def test_admin_token_can_still_load_an_ordinary_model(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _insert_model(db_path, model_id="ord", trust_remote_code=False)
    _stub_probe(client, [0])
    _, secret = seed_admin_token(db_path)
    p1, p2 = _stub_engine()
    with p1, p2:
        r = client.post(f"{MODELS_URL}/ord/load", headers=bearer(secret))
    assert r.status_code == 202, r.text


# ---------------------------------------------------------------------------
# C1 -- updating it: PATCH /api/models/{id}/settings
#
# The settings PATCH is an admin-bucket route whose allowlist is DERIVED from
# ModelRow minus a blocklist, so `trust_remote_code` was patchable by default
# and `prior_status` -- the watchdog's restart flag -- with it. Two requests on
# a `failed` row armed the restart sweep, which re-reads the row and calls
# start_engine: code execution without ever calling /load.
# ---------------------------------------------------------------------------

SETTINGS_URL = MODELS_URL + "/{}/settings"


def test_admin_token_patching_trust_remote_code_true_is_refused(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _insert_model(db_path, model_id="m1", trust_remote_code=False)
    _, secret = seed_admin_token(db_path)
    r = client.patch(
        SETTINGS_URL.format("m1"), json={"trust_remote_code": True}, headers=bearer(secret)
    )
    assert r.status_code == 403, r.text
    assert r.json()["detail"]["error_code"] == "session_only"
    # Refused before any write -- the row is untouched.
    assert _row(db_path, "m1") == ("pulled", None, 0)


def test_admin_token_patching_trust_remote_code_false_succeeds(tmp_data_dir, client):
    """Turning it OFF, and every other setting, stays open to an admin token."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _insert_model(db_path, model_id="m1", trust_remote_code=True)
    _, secret = seed_admin_token(db_path)
    r = client.patch(
        SETTINGS_URL.format("m1"),
        json={"trust_remote_code": False, "max_model_len": 4096},
        headers=bearer(secret),
    )
    assert r.status_code == 200, r.text
    assert _row(db_path, "m1") == ("pulled", None, 0)


def test_session_can_patch_trust_remote_code_true(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _insert_model(db_path, model_id="m1", trust_remote_code=False)
    auth = jwt_login(client)
    r = client.patch(
        SETTINGS_URL.format("m1"),
        json={"trust_remote_code": True},
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    assert _row(db_path, "m1") == ("pulled", None, 1)


def test_prior_status_is_not_patchable_by_anyone(tmp_data_dir, client):
    """`prior_status` is watchdog control state, not a user setting. Nothing
    PATCHes it -- not an admin token and not a session."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _insert_model(db_path, model_id="m1", trust_remote_code=False, status="failed")
    _, secret = seed_admin_token(db_path)
    r = client.patch(
        SETTINGS_URL.format("m1"), json={"prior_status": "loaded"}, headers=bearer(secret)
    )
    assert r.status_code == 400, r.text
    assert "prior_status" in r.json()["detail"]
    auth = jwt_login(client)
    r2 = client.patch(
        SETTINGS_URL.format("m1"),
        json={"prior_status": "loaded"},
        headers={**auth, **csrf_header(client)},
    )
    assert r2.status_code == 400, r2.text
    assert _row(db_path, "m1") == ("failed", None, 0)


def test_an_admin_token_cannot_arm_the_watchdog_restart_sweep(tmp_data_dir, client):
    """The C1 exploit chain end to end: a `failed` row plus
    `trust_remote_code` + `prior_status` makes `wants_restart` true, and the
    sweep then calls start_engine off the row with no /load request. Neither
    column may move on an admin token, in one request or two.
    """
    import types

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _insert_model(db_path, model_id="m1", trust_remote_code=False, status="failed")
    _, secret = seed_admin_token(db_path)

    both = client.patch(
        SETTINGS_URL.format("m1"),
        json={"trust_remote_code": True, "prior_status": "loaded"},
        headers=bearer(secret),
    )
    assert both.status_code == 403, both.text
    assert both.json()["detail"]["error_code"] == "session_only"

    # And separately, so a two-request sequence cannot assemble the same state.
    assert client.patch(
        SETTINGS_URL.format("m1"), json={"trust_remote_code": True}, headers=bearer(secret)
    ).status_code == 403
    assert client.patch(
        SETTINGS_URL.format("m1"), json={"prior_status": "loaded"}, headers=bearer(secret)
    ).status_code == 400

    status, prior_status, trust = _row(db_path, "m1")
    assert (status, prior_status, trust) == ("failed", None, 0)
    # `wants_restart` reads exactly these three attributes off the row.
    row = types.SimpleNamespace(id="m1", status=status, prior_status=prior_status)
    assert wants_restart(row, leases=None) is False


# ---------------------------------------------------------------------------
# I1 -- templates: POST /api/models/templates, and the merged value on
# POST /api/models
# ---------------------------------------------------------------------------


def test_admin_token_creating_a_trust_remote_code_template_is_refused(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _, secret = seed_admin_token(db_path)
    r = client.post(TEMPLATES_URL, json={
        "id": "evil", "label": "evil", "hf_repo": "o/r",
        "trust_remote_code": True,
    }, headers=bearer(secret))
    assert r.status_code == 403, r.text
    assert r.json()["detail"]["error_code"] == "session_only"
    # Refused before any write -- nothing was persisted.
    assert _template_ids(db_path) == []


def test_admin_token_creating_an_ordinary_template_succeeds(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _, secret = seed_admin_token(db_path)
    r = client.post(TEMPLATES_URL, json={
        "id": "fine", "label": "fine", "hf_repo": "o/r",
    }, headers=bearer(secret))
    assert r.status_code == 201, r.text
    assert _template_ids(db_path) == ["fine"]


def test_session_can_create_a_trust_remote_code_template(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    auth = jwt_login(client)
    r = client.post(TEMPLATES_URL, json={
        "id": "ok", "label": "ok", "hf_repo": "o/r", "trust_remote_code": True,
    }, headers={**auth, **csrf_header(client)})
    assert r.status_code == 201, r.text
    assert _template_ids(db_path) == ["ok"]


def test_admin_token_registering_via_a_builtin_trust_template_is_refused(tmp_data_dir, client):
    """No template create is even needed: a builtin carries the flag."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _, secret = seed_admin_token(db_path)
    r = client.post(MODELS_URL, json={
        "served_model_name": "viatpl", "gpu_indices": [0],
        "template_id": TRUST_BUILTIN_TEMPLATE,
    }, headers=bearer(secret))
    assert r.status_code == 403, r.text
    assert r.json()["detail"]["error_code"] == "session_only"
    assert _model_ids(db_path) == []


def test_admin_token_registering_via_a_user_trust_template_is_refused(tmp_data_dir, client):
    """A session-created template carrying the flag is refused on the register
    leg too -- the refusal is on the merged value, not on the body key."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    auth = jwt_login(client)
    assert client.post(TEMPLATES_URL, json={
        "id": "trusting", "label": "trusting", "hf_repo": "o/r",
        "trust_remote_code": True,
    }, headers={**auth, **csrf_header(client)}).status_code == 201
    _, secret = seed_admin_token(db_path)
    r = client.post(MODELS_URL, json={
        "served_model_name": "viauser", "gpu_indices": [0],
        "template_id": "trusting",
    }, headers=bearer(secret))
    assert r.status_code == 403, r.text
    assert _model_ids(db_path) == []


def test_an_explicit_false_still_overrides_a_trust_template_for_an_admin_token(
    tmp_data_dir, client
):
    """The merge rule is "explicit body wins", so `trust_remote_code: false`
    against a trusting template yields a false row -- and must still 201."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _, secret = seed_admin_token(db_path)
    r = client.post(MODELS_URL, json={
        "served_model_name": "safe", "gpu_indices": [0],
        "template_id": TRUST_BUILTIN_TEMPLATE, "trust_remote_code": False,
    }, headers=bearer(secret))
    assert r.status_code == 201, r.text
    assert _trust(db_path, "safe") == 0


def test_session_can_register_via_a_trust_template(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    auth = jwt_login(client)
    r = client.post(MODELS_URL, json={
        "served_model_name": "viatpl", "gpu_indices": [0],
        "template_id": TRUST_BUILTIN_TEMPLATE,
    }, headers={**auth, **csrf_header(client)})
    assert r.status_code == 201, r.text
    assert _trust(db_path, "viatpl") == 1


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def test_a_register_refusal_is_audited(tmp_data_dir, client, flush_audit):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    tid, secret = seed_admin_token(db_path)
    client.post(MODELS_URL, json={
        "served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0],
        "trust_remote_code": True,
    }, headers=bearer(secret))
    flush_audit()  # the audit writer batches (#258)
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "SELECT token_id, status FROM admin_audit ORDER BY id"
        ).fetchall()
    assert rows == [(tid, 403)]


def test_a_settings_patch_refusal_is_audited(tmp_data_dir, client, flush_audit):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _insert_model(db_path, model_id="m1", trust_remote_code=False)
    tid, secret = seed_admin_token(db_path)
    client.patch(
        SETTINGS_URL.format("m1"), json={"trust_remote_code": True}, headers=bearer(secret)
    )
    flush_audit()  # the audit writer batches (#258)
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "SELECT token_id, path, status FROM admin_audit ORDER BY id"
        ).fetchall()
    assert rows == [(tid, "/api/models/{model_id}/settings", 403)]


def test_a_template_create_refusal_is_audited(tmp_data_dir, client, flush_audit):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    tid, secret = seed_admin_token(db_path)
    client.post(TEMPLATES_URL, json={
        "id": "evil", "label": "evil", "hf_repo": "o/r", "trust_remote_code": True,
    }, headers=bearer(secret))
    flush_audit()  # the audit writer batches (#258)
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "SELECT token_id, path, status FROM admin_audit ORDER BY id"
        ).fetchall()
    assert rows == [(tid, "/api/models/templates", 403)]


def test_a_load_refusal_is_audited(tmp_data_dir, client, flush_audit):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path, allowed=[0])
    _insert_model(db_path, model_id="trm", trust_remote_code=True)
    tid, secret = seed_admin_token(db_path)
    client.post(f"{MODELS_URL}/trm/load", headers=bearer(secret))
    flush_audit()  # the audit writer batches (#258)
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "SELECT token_id, status FROM admin_audit ORDER BY id"
        ).fetchall()
    assert rows == [(tid, 403)]
