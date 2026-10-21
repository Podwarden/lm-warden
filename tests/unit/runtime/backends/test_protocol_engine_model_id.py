"""``engine_model_id`` as a ``Backend`` Protocol member.

P5h landed the method on all three backends (vLLM and llama.cpp return
``served_model_name``; MLX the local ``--model`` path). This file is the step
the OVERSEER's protocol-additions decision (planner iter 22) holds in reserve:
the method becomes a ``Backend`` member only once EVERY backend has it, because
a ``@runtime_checkable`` Protocol is a membership check -- declaring a member no
registered backend has would flip ``isinstance(x, Backend)`` to False for each
of them. So this test pins that the member is declared AND that the flip has
not happened: every registered backend and MLX are still Backends, and a
stand-in that omits ``engine_model_id`` is not.
"""

import inspect

from app.runtime.backends import Backend, BackendCapabilities, LaunchPlan


def _caps() -> BackendCapabilities:
    return BackendCapabilities(
        name="x",
        display_name="X",
        supports_tensor_parallel=True,
        supports_pipeline_parallel=True,
        max_gpus=None,
        openai_paths=frozenset({"/v1/completions"}),
        health_path="/health",
        metrics_path="/metrics",
        supports_request_priority=True,
        supports_lora=False,
        supports_vision=False,
        supports_version_pin=False,
        needs_local_model_path=False,
        env_prefixes=("X_",),
        env_exact=frozenset(),
        vram_cap_fraction=None,
    )


def _stand_in_with_engine_model_id() -> type:
    """A class with EVERY Backend member including ``engine_model_id``.

    Built once and reused so the missing-member stand-in below is provably the
    same shape minus exactly one method -- the two legs differ by one member
    and nothing else.
    """

    class StandIn:
        capabilities = _caps()

        def version_string(self):
            return None

        def version_pin_available(self, driver):
            return False

        def bind_host(self, driver_name):
            return "127.0.0.1"

        def plan(self, model, *, port, bind_host, overrides=None, resolved=None):
            return LaunchPlan(argv=["x"], env={}, image=None, port=port)

        def health_url(self, host, port):
            return f"http://{host}:{port}/health"

        def diagnose(self, log_tail):
            return None

        def parse_metrics(self, body):
            return None

        def engine_model_id(self, model, resolved=None) -> str:
            return "x"

    return StandIn


def test_engine_model_id_is_a_declared_protocol_member():
    # Declared directly on the Protocol (not inherited): its own __dict__ has
    # it. The runtime-checkable membership tests below are the teeth -- this
    # just pins the member is here, not merely reachable from a base.
    assert "engine_model_id" in Backend.__dict__


def test_engine_model_id_signature_is_model_resolved_none():
    """One positional (``model``), one keyword with a ``None`` default
    (``resolved``), a ``str`` return: the exact shape the three
    implementations share. The module's ``from __future__ import
    annotations`` stores annotations as strings, and ``ResolvedModelPaths``
    is a TYPE_CHECKING-only import there (the module imports nothing from
    ``app`` at runtime), so the strings are compared rather than resolved."""
    from app.runtime.backends import Backend

    sig = inspect.signature(Backend.engine_model_id)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["self", "model", "resolved"]
    assert params[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[1].default is inspect.Parameter.empty
    assert params[2].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[2].default is None
    assert params[2].annotation == "ResolvedModelPaths | None"
    assert sig.return_annotation == "str"


def test_registered_backends_stay_backends_after_the_member_landed():
    """The OVERSEER guard: a ``@runtime_checkable`` Protocol is a membership
    check, so a member any registered backend lacked would make this False for
    that backend. All three backends have the method (P5h, and MLX's step 1),
    so the declaration must not flip any of them off."""
    from app.runtime.backends import registry
    from app.runtime.backends.llamacpp import LlamaCppBackend
    from app.runtime.backends.mlx import MlxBackend
    from app.runtime.backends.vllm import VllmBackend

    assert isinstance(VllmBackend(), Backend)
    assert isinstance(LlamaCppBackend(), Backend)
    assert isinstance(MlxBackend(), Backend)
    for name in registry.available():
        assert isinstance(registry.get(name), Backend), name


def test_stand_in_missing_only_engine_model_id_is_not_a_backend():
    """The red leg. The same stand-in as the positive leg, minus
    ``engine_model_id``: today (member not yet declared) this passes the
    check, which is why the test is red; once the member lands, the check
    fails for exactly this reason."""
    stand_in = _stand_in_with_engine_model_id()

    class MissingEngineModelId:
        __init__ = stand_in.__init__
        capabilities = stand_in.capabilities
        version_string = stand_in.version_string
        version_pin_available = stand_in.version_pin_available
        bind_host = stand_in.bind_host
        plan = stand_in.plan
        health_url = stand_in.health_url
        diagnose = stand_in.diagnose
        parse_metrics = stand_in.parse_metrics

    assert not isinstance(MissingEngineModelId(), Backend)


def test_stand_in_with_engine_model_id_is_a_backend():
    """The positive boundary of the same rule: a stand-in with the full member
    set (including ``engine_model_id``) is a Backend. Together with the missing
    leg this proves that leg failed for the right reason rather than for a
    broken stand-in."""
    stand_in = _stand_in_with_engine_model_id()
    # An INSTANCE, not the class object: the missing leg checks an instance,
    # so the positive leg must too, or the two differ by more than one member.
    assert isinstance(stand_in(), Backend)
    assert stand_in().engine_model_id(object()) == "x"
