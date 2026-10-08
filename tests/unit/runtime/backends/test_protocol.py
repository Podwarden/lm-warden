"""The Backend interface's structural invariants.

The important test here is not that the dataclasses exist — it is
test_launch_plan_has_no_entrypoint_field and its sibling source scan. Spec §2
forbids runtime mutation (no patched image, no pre-start hook, no
sitecustomize / PYTHONPATH / LD_PRELOAD injection), and §6.3 chose to enforce
that STRUCTURALLY: a launch is argv + env and nothing else, so there is no
place in the type for a backend to mutate its own runtime. These tests are what
makes that structural rather than advisory.
"""

import ast
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from app.runtime.backends import Backend, BackendCapabilities, LaunchPlan

BACKENDS_DIR = Path(__file__).resolve().parents[4] / "app" / "runtime" / "backends"

# Vocabulary that can only be there to mutate a runtime before it starts.
FORBIDDEN_TOKENS = ("entrypoint", "sitecustomize", "LD_PRELOAD", "PYTHONPATH")


def _executable_source(path: Path) -> str:
    """``path``'s source with prose, and the hard-lock DECLARATION, removed.

    DEVIATION FROM THE PLAN AS WRITTEN, and it is load-bearing. The plan's
    version of the scan below read the raw file, which cannot work against the
    plan's own code: ``app/runtime/backends/__init__.py``'s module docstring
    says "There is deliberately no ``entrypoint`` field" (that sentence is the
    whole point of the invariant), and Task 9 adds the string literals
    ``"PYTHONPATH"`` and ``"LD_PRELOAD"`` to ``HARD_LOCKED_ENV_KEYS`` in
    ``backends/vllm/env.py`` -- i.e. *under* this directory. Taken literally,
    the guard fails on the very lines that enforce what it is guarding.

    So the scan runs over EXECUTABLE CODE only, and skips two things that name
    a forbidden token precisely in order to forbid it:

      * docstrings and comments -- prose explaining the ban is the invariant
        being written down where the code lives, which is the guard working;
      * the ``HARD_LOCKED_ENV_KEYS`` assignment -- a list of keys that may
        never be set is a declaration, not a use.

    Everything else still counts. A ``LaunchPlan.entrypoint`` field, an
    ``entrypoint=`` kwarg, or ``env["PYTHONPATH"] = ...`` anywhere under
    ``app/runtime/backends/`` fails this test, which is the behaviour the
    constraint actually asks for.

    Known, accepted narrowness: comments are stripped by splitting on the
    first ``#``, so a forbidden token appearing AFTER a ``#`` inside a string
    literal would be missed. Nothing legitimate looks like that, and the
    alternative (full tokenisation) buys no real coverage.
    """
    src = path.read_text()
    tree = ast.parse(src)
    skip: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                skip.update(range(body[0].lineno, body[0].end_lineno + 1))

        targets: list = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(getattr(t, "id", None) == "HARD_LOCKED_ENV_KEYS" for t in targets):
            skip.update(range(node.lineno, node.end_lineno + 1))

    kept = [
        line.split("#", 1)[0]
        for lineno, line in enumerate(src.splitlines(), start=1)
        if lineno not in skip
    ]
    return "\n".join(kept)


def test_launch_plan_has_no_entrypoint_field():
    names = {f.name for f in fields(LaunchPlan)}
    assert names == {"argv", "env", "image", "port", "gpu_indices"}
    assert "entrypoint" not in names


def test_launch_plan_is_frozen():
    plan = LaunchPlan(argv=["vllm", "serve"], env={}, image=None, port=1, gpu_indices=[])
    with pytest.raises(FrozenInstanceError):
        plan.argv = []


def _caps(**over):
    base = dict(
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
        # Sub-project C. Every field on this dataclass is REQUIRED -- there are
        # no defaults, deliberately, so a new backend cannot forget to answer a
        # capability question and silently inherit someone else's answer. That
        # means adding a field is a visible edit here, which is the point.
        needs_local_model_path=False,
        env_prefixes=("X_",),
        env_exact=frozenset(),
        # None = "this backend honours the operator's gpu_memory_utilization",
        # which is vLLM's answer and the safe default for a stand-in.
        vram_cap_fraction=None,
    )
    base.update(over)
    return BackendCapabilities(**base)


def test_capabilities_is_frozen():
    with pytest.raises(FrozenInstanceError):
        _caps().name = "y"


def test_supports_version_pin_is_a_capabilities_field():
    """Operator ruling, 2026-09-01: "can you pick a version" is a fact about
    the ENGINE, so it lives on BackendCapabilities."""
    assert "supports_version_pin" in {f.name for f in fields(BackendCapabilities)}


def test_capabilities_is_a_plain_attribute():
    """§6.3's shape, honoured exactly.

    An earlier draft made this a capabilities_for(driver) method because
    supports_version_pin appeared to need the driver. E's two-field split
    (5e498cb) removed that need -- the field is driver-invariant -- so the
    attribute stands and the deviation is retired."""
    from app.runtime.backends.vllm import VllmBackend

    assert isinstance(VllmBackend().capabilities, BackendCapabilities)


def test_the_two_version_pin_questions_are_separate_members():
    """The engine fact is a FIELD; the deployment fact is a METHOD.

    That is the rule the protocol follows throughout -- driver-invariant facts
    are dataclass fields, driver-dependent answers are methods (see also
    bind_host). Collapsing these two back into one boolean is what made the
    earlier draft need capabilities_for()."""
    from app.runtime.backends.vllm import VllmBackend

    assert isinstance(VllmBackend().capabilities.supports_version_pin, bool)
    assert callable(VllmBackend().version_pin_available)


def test_backend_protocol_is_runtime_checkable():
    class NotABackend:
        pass

    assert not isinstance(NotABackend(), Backend)


@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_no_runtime_mutation_vocabulary_under_backends(token):
    """Spec §2, enforced by the type system AND by this scan.

    If a future change needs something that looks like an entrypoint, that is a
    signal to pin a different mainline version or add a backend -- not to add
    the field. Deleting this test is the change that has to be argued for.

    See ``_executable_source`` for what "appears in" means here, and why it
    has to mean something narrower than "appears in the file".
    """
    offenders = [p for p in sorted(BACKENDS_DIR.rglob("*.py")) if token in _executable_source(p)]
    assert offenders == [], (
        f"{token!r} appears in {[str(p) for p in offenders]}; spec §2 forbids "
        "runtime mutation and §6.3 enforces it structurally"
    )


def test_vram_cap_fraction_is_declared_by_every_backend():
    """The fit preview reads this instead of branching on the backend name.

    vLLM answers None because `--gpu-memory-utilization` is a real flag the
    operator sets and vLLM hard-caps itself there. llama.cpp answers 1.0
    because it has no equivalent flag at all -- `llama-server --help` in the
    shipped image offers only `-ngl/--n-gpu-layers`, a layer count -- so the
    operator's number corresponds to nothing and must not shrink its budget.

    Applying vLLM's 0.9 to a llama.cpp row hid 10% of the card and reported
    "won't fit" for models that fit.
    """
    from app.runtime.backends import registry

    assert registry.get("vllm").capabilities.vram_cap_fraction is None
    assert registry.get("llamacpp").capabilities.vram_cap_fraction == 1.0


def test_vram_cap_fraction_has_no_default():
    """Same contract as every other field here: answering is mandatory."""
    import dataclasses

    field = next(
        f for f in dataclasses.fields(BackendCapabilities) if f.name == "vram_cap_fraction"
    )
    assert field.default is dataclasses.MISSING
    assert field.default_factory is dataclasses.MISSING
