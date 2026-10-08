"""The capability-advertisement surface.

routes_engine.py is the pattern the Backend interface was modelled on:
capability-driven, self-explaining, refusing rather than misleading. This route
generalises it from a single driver capability to per-backend capabilities
resolved against the active driver.

The OLD route must stay byte-identical. The live deployment on `bonus`
(v2026.08.31.1, subprocess driver, vLLM 0.26.0) answers
{"driver":"subprocess","supports_version_select":false,"vllm_version":"0.26.0"}
and an old UI against a new API must not break mid-rollout.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import jwt_login, seed_admin_user


def _seed_done(db_path: Path) -> None:
    seed_admin_user(db_path, allowed_gpu_indices=[0])


class _DockerCapable:
    supports_engine_image = True


def _ready(tmp_data_dir, client):
    """Boot the app, seed an admin, and return the auth header.

    The default app is built with the in-container subprocess driver, whose
    _driver.supports_engine_image is False -- i.e. `bonus` as deployed.
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    return jwt_login(client)


def _make_docker(client):
    """Swap in a driver that CAN swap the engine image."""
    client.app.state.supervisor._driver = _DockerCapable()


def test_engine_route_response_shape_is_unchanged(tmp_data_dir, client):
    auth = _ready(tmp_data_dir, client)
    r = client.get("/api/system/engine", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert sorted(body) == ["driver", "supports_version_select", "vllm_version"]
    assert body["driver"] == "subprocess"
    assert body["supports_version_select"] is False


def test_backends_route_lists_vllm_with_its_capabilities(tmp_data_dir, client):
    auth = _ready(tmp_data_dir, client)
    r = client.get("/api/system/backends", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == "vllm"
    assert body["driver"] == "subprocess"
    # Sub-project C registered a second backend; available() sorts, so the list
    # is ["llamacpp", "vllm"] and this test can no longer index position 0.
    # Selecting by NAME is what it always meant.
    assert "vllm" in [b["name"] for b in body["backends"]]

    vllm = next(b for b in body["backends"] if b["name"] == "vllm")
    assert vllm["display_name"] == "vLLM"
    assert vllm["health_path"] == "/health"
    assert vllm["metrics_path"] == "/metrics"
    assert vllm["supports_request_priority"] is True
    assert vllm["supports_tensor_parallel"] is True
    assert vllm["env_prefixes"] == ["VLLM_", "TRITON_", "NCCL_", "PYTORCH_", "TORCH_", "OMP_"]
    # TWO FIELDS, and on this deployment they DISAGREE. That is the point.
    #   supports_version_pin  -- the ENGINE fact: vLLM can be pinned. True.
    #   version_pin_available -- the DEPLOYMENT fact: the subprocess driver
    #                            cannot swap the image. False.
    # A client that gates its version selector on the first one lights up a
    # control that cannot work.
    assert vllm["supports_version_pin"] is True
    assert vllm["version_pin_available"] is False
    assert vllm["version"] == body["engine_version"]


def test_version_pin_is_available_under_the_docker_driver(tmp_data_dir, client):
    """NOT redundant with the unit test in test_vllm_backend.py, and the
    distinction is worth stating because it looks redundant.

    The unit test proves the METHOD returns True for a docker driver. This
    proves the ROUTE is WIRED to it -- that it passes the active driver object
    through rather than a name, a stale default, or nothing.

    The subprocess assertion above cannot catch a wiring fault, because a
    mis-wired route returns False and False is the correct answer there. Only
    the docker fixture distinguishes "correct" from "accidentally correct" --
    which is exactly how E's driver-as-string defect stayed camouflaged.
    """
    auth = _ready(tmp_data_dir, client)
    _make_docker(client)
    body = client.get("/api/system/backends", headers=auth).json()
    assert body["driver"] == "docker"
    vllm = next(b for b in body["backends"] if b["name"] == "vllm")
    assert vllm["supports_version_pin"] is True
    assert vllm["version_pin_available"] is True


def test_the_deprecated_alias_reports_the_DEPLOYMENT_fact(tmp_data_dir, client):
    """/api/system/engine's supports_version_select has always meant "can this
    deployment honour a pin?" -- so it maps to version_pin_available, NOT to
    the new engine-fact field. Mapping it to supports_version_pin would flip
    the live answer on `bonus` from false to true and re-enable a dead control
    in any client still reading the old route."""
    auth = _ready(tmp_data_dir, client)

    for make_docker in (False, True):
        if make_docker:
            _make_docker(client)
        old = client.get("/api/system/engine", headers=auth).json()
        new = client.get("/api/system/backends", headers=auth).json()
        vllm = next(b for b in new["backends"] if b["name"] == "vllm")
        assert old["supports_version_select"] == vllm["version_pin_available"]


def test_both_routes_require_auth(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    assert client.get("/api/system/backends").status_code == 401
    assert client.get("/api/system/engine").status_code == 401


# ---------------------------------------------------------------------------
# Sub-project C: the second backend on the advertisement surface
# ---------------------------------------------------------------------------


def test_backends_route_lists_both(tmp_data_dir, client):
    auth = _ready(tmp_data_dir, client)
    body = client.get("/api/system/backends", headers=auth).json()
    names = [b["name"] for b in body["backends"]]
    assert names == ["llamacpp", "vllm"]
    assert body["default"] == "vllm"


def test_llamacpp_advertises_its_own_capabilities(tmp_data_dir, client):
    auth = _ready(tmp_data_dir, client)
    body = client.get("/api/system/backends", headers=auth).json()
    lc = next(b for b in body["backends"] if b["name"] == "llamacpp")
    assert lc["display_name"] == "llama.cpp"
    assert lc["supports_request_priority"] is False
    assert lc["supports_tensor_parallel"] is False
    assert lc["supports_pipeline_parallel"] is True
    assert lc["supports_vision"] is True
    assert lc["env_prefixes"] == ["GGML_"]
    # B's route emits BOTH keys. The engine fact is True; the deployment fact
    # is False, and it is the deployment fact a UI gates on.
    assert lc["supports_version_pin"] is True
    assert lc["version_pin_available"] is False


def test_llamacpp_version_is_its_own_not_vllms(tmp_data_dir, client):
    """B's route hard-coded `_VLLM_VERSION if name == "vllm" else None`. With a
    second backend that must become a per-backend lookup, or llama.cpp reports
    a vLLM version number -- which is worse than reporting none."""
    auth = _ready(tmp_data_dir, client)
    body = client.get("/api/system/backends", headers=auth).json()
    lc = next(b for b in body["backends"] if b["name"] == "llamacpp")
    vl = next(b for b in body["backends"] if b["name"] == "vllm")
    assert lc["version"] != vl["version"] or lc["version"] is None
    # Outside a built image nothing stamped VW_LLAMACPP_BUILD, so None is the
    # honest answer -- exactly as _VLLM_VERSION is None where vLLM is absent.
    assert lc["version"] is None


def test_llamacpp_version_comes_from_the_baked_build_stamp(tmp_data_dir, client, monkeypatch):
    monkeypatch.setenv("VW_LLAMACPP_BUILD", "b10731")
    auth = _ready(tmp_data_dir, client)
    body = client.get("/api/system/backends", headers=auth).json()
    lc = next(b for b in body["backends"] if b["name"] == "llamacpp")
    assert lc["version"] == "b10731"


def test_the_llamacpp_entry_never_offers_a_version_selector(tmp_data_dir, client):
    """The #177 failure mode, which the two-field split exists to prevent: a
    control that is enabled because the ENGINE supports pinning, on a deployment
    that cannot honour it. For llama.cpp that must hold under every driver,
    because there is no llama.cpp image resolver to honour it with."""
    auth = _ready(tmp_data_dir, client)
    _make_docker(client)
    body = client.get("/api/system/backends", headers=auth).json()
    lc = next(b for b in body["backends"] if b["name"] == "llamacpp")
    vl = next(b for b in body["backends"] if b["name"] == "vllm")
    assert vl["version_pin_available"] is True
    assert lc["version_pin_available"] is False


def test_a_disabled_version_pin_always_carries_a_reason(tmp_data_dir, client):
    """A disabled control the operator cannot explain is a bug report waiting.

    The Add-model dialog offers an engine-version field. On this deployment it
    must be disabled -- but "greyed out, no idea why" is precisely the defect
    class this route exists to prevent, and the sentence explaining it cannot be
    invented in the frontend: the frontend does not know whether the cause is
    the driver or the backend.

    So ``version_pin_reason`` is non-null exactly when ``version_pin_available``
    is false -- the invariant the route's docstring already states in prose.
    """
    auth = _ready(tmp_data_dir, client)
    body = client.get("/api/system/backends", headers=auth).json()
    for b in body["backends"]:
        if b["version_pin_available"]:
            assert b["version_pin_reason"] is None
        else:
            assert b["version_pin_reason"], f"{b['name']} disabled with no reason"


def test_the_reason_names_the_driver_when_the_driver_is_the_blocker(tmp_data_dir, client):
    """vLLM under the subprocess driver -- `bonus` as deployed.

    The engine CAN be pinned; this deployment cannot honour it. The operator's
    next question is "what would make it work?", so the answer has to name the
    docker engine driver rather than merely restate that the field is off.
    """
    auth = _ready(tmp_data_dir, client)
    body = client.get("/api/system/backends", headers=auth).json()
    vllm = next(b for b in body["backends"] if b["name"] == "vllm")
    assert vllm["supports_version_pin"] is True
    assert vllm["version_pin_available"] is False
    assert "docker" in vllm["version_pin_reason"]


def test_the_driver_reason_does_not_claim_a_pin_is_discarded(tmp_data_dir, client):
    """The sentence must describe what happens NOW, not the #177 bug.

    It used to say a version pin "would be silently discarded". Since #177 the
    supervisor raises EnginePinUnsupported before claiming a GPU -- a pin is
    refused, loudly, never discarded. Read literally, the old sentence said
    the control's input is thrown away, which makes the (deliberately visible,
    deliberately disabled) control look like one that does nothing at all
    rather than one this deployment could enable.
    """
    auth = _ready(tmp_data_dir, client)
    body = client.get("/api/system/backends", headers=auth).json()
    vllm = next(b for b in body["backends"] if b["name"] == "vllm")
    reason = vllm["version_pin_reason"]
    assert "discard" not in reason.lower()
    assert "refused" in reason.lower()
    # And it still names the way out, which is the whole point of the field.
    assert "VW_ENGINE_DRIVER=docker" in reason


def test_the_reason_names_the_backend_when_the_backend_is_the_blocker(tmp_data_dir, client):
    """llama.cpp under a driver that CAN swap images.

    The driver is no longer the obstacle, so blaming it would be false. There is
    no llama.cpp image resolver, which is a property of the backend and holds
    under every driver -- the reason must say so.
    """
    auth = _ready(tmp_data_dir, client)
    _make_docker(client)
    body = client.get("/api/system/backends", headers=auth).json()
    lc = next(b for b in body["backends"] if b["name"] == "llamacpp")
    assert lc["version_pin_available"] is False
    assert "docker" not in lc["version_pin_reason"]
    assert "llama.cpp" in lc["version_pin_reason"]


def test_deprecated_engine_alias_is_still_vllm_only(tmp_data_dir, client):
    """B kept /api/system/engine byte-identical for an old UI mid-rollout.
    Adding a backend must not change it -- try-stack-panel.tsx:179 is still the
    only caller and still reads vllm_version."""
    auth = _ready(tmp_data_dir, client)
    body = client.get("/api/system/engine", headers=auth).json()
    assert set(body) == {"driver", "supports_version_select", "vllm_version"}


# ---------------------------------------------------------------------------
# version_pin_reason_code -- the MACHINE-READABLE half of the same answer.
#
# ``version_pin_reason`` is a sentence for the operator. A client that has to
# choose between two DIFFERENT renderings -- hide the control, or show it
# disabled with the sentence beside it -- cannot make that choice from prose
# without string-matching it, and a UI that greps a human sentence for the word
# "docker" is one copy-edit away from silently changing behaviour.
#
# So the route emits both, out of ONE branch, and the code is what a client
# switches on:
#
#   null      the pin is available; there is nothing to explain.
#   "engine"  this engine cannot be version-pinned at all.
#   "driver"  the engine can; THIS deployment's driver cannot honour it.
#             Removable by the operator -> disable the control and say so.
#   "backend" neither the engine nor the driver is the obstacle: there is no
#             image catalogue for this backend to pin against. Not removable,
#             and not a concept this engine has -> hide the control.
# ---------------------------------------------------------------------------


def test_reason_code_is_null_exactly_when_the_pin_is_available(tmp_data_dir, client):
    auth = _ready(tmp_data_dir, client)
    _make_docker(client)
    body = client.get("/api/system/backends", headers=auth).json()
    for b in body["backends"]:
        if b["version_pin_available"]:
            assert b["version_pin_reason_code"] is None
        else:
            assert b["version_pin_reason_code"] in {"engine", "driver", "backend"}


def test_reason_code_says_driver_for_vllm_on_the_subprocess_driver(tmp_data_dir, client):
    """`bonus` as deployed. The obstacle is removable, so the control stays
    visible-but-disabled and the sentence says what to change."""
    auth = _ready(tmp_data_dir, client)
    body = client.get("/api/system/backends", headers=auth).json()
    vllm = next(b for b in body["backends"] if b["name"] == "vllm")
    assert vllm["version_pin_reason_code"] == "driver"


def test_reason_code_says_backend_for_llamacpp_on_every_driver(tmp_data_dir, client):
    """The code must NOT depend on the driver for llama.cpp.

    This is the whole point of the field. Under the subprocess driver the
    driver is *also* an obstacle, so an implementation that checked the driver
    first would answer "driver" here -- and a UI that trusted it would render a
    version selector no driver could ever make work, promising the operator
    that switching to docker would fix it. It would not.
    """
    auth = _ready(tmp_data_dir, client)
    for make_docker in (False, True):
        if make_docker:
            _make_docker(client)
        body = client.get("/api/system/backends", headers=auth).json()
        lc = next(b for b in body["backends"] if b["name"] == "llamacpp")
        assert lc["version_pin_reason_code"] == "backend", f"docker={make_docker}"


def test_reason_and_reason_code_come_from_the_same_branch(tmp_data_dir, client):
    """The sentence and the code must never disagree.

    They are two renderings of one decision; if they can drift, the UI and the
    text it renders eventually describe different worlds.
    """
    auth = _ready(tmp_data_dir, client)
    body = client.get("/api/system/backends", headers=auth).json()
    for b in body["backends"]:
        assert (b["version_pin_reason"] is None) == (b["version_pin_reason_code"] is None)
    vllm = next(b for b in body["backends"] if b["name"] == "vllm")
    assert "docker" in vllm["version_pin_reason"]
    assert vllm["version_pin_reason_code"] == "driver"
