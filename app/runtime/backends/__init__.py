"""The Backend axis: which program serves a model, and how we talk to it.

A **driver** (``app/runtime/engine/``) answers *where a process runs* — an
in-container subprocess, a sibling container, a pod. A **backend** answers
*which program runs, with what argv and env, on what health path, in what log
grammar, and with which capabilities*. They are orthogonal and compose as a
matrix (design spec §5, §6.1, decision D1): adding a backend or a driver is
O(1), not O(n*m).

Everything here is types only. It imports nothing from ``app`` so it can never
participate in an import cycle, and so a backend implementation can be read
without reading the control plane.

THE ONE THING TO KNOW BEFORE EDITING THIS FILE
----------------------------------------------
``LaunchPlan`` is argv + env and NOTHING ELSE. There is deliberately no
``entrypoint`` field, no pre-start hook, no patch list. The product invariant
(spec §2) is "we ship mainline runtimes; no monkeypatching, ever", and §6.3
chose to enforce it structurally rather than by review discipline: if the type
offers no place to mutate a runtime, no backend can. A future need that looks
like it wants an entrypoint is a signal to pin a different mainline version or
add a backend. ``tests/unit/runtime/backends/test_protocol.py`` fails if the
field -- or the vocabulary around it -- reappears in executable code anywhere
under this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    # TYPE-ONLY. Never imported at runtime, so the "imports nothing from app"
    # property above -- and the no-import-cycle guarantee that rests on it --
    # is unaffected. It exists so ``version_pin_available``'s annotation names
    # a real type instead of an undefined forward reference (ruff F821).
    from app.runtime.engine import EngineDriver


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can do. ADVERTISED, never inferred.

    Modelled on ``app/system/routes_engine.py``: the control plane answers
    capability questions from a declared value and refuses rather than
    misleading, instead of discovering a limitation 40 seconds into a load.

    Sub-project E appends ``model_formats`` and ``quant_methods`` here when it
    builds the compatibility matrix; they are omitted now because nothing in
    this sub-project reads them and inventing their values would be fiction.

    EVERY FIELD HERE IS DRIVER-INVARIANT -- a fact about the engine alone.
    That is what lets this be a plain frozen constant per backend. Anything
    whose answer depends on where the driver puts the process is a METHOD on
    ``Backend`` instead (``bind_host``, ``version_pin_available``). Keep that
    line clean: a driver-dependent value smuggled in here would make the
    constant lie for every deployment but one.
    """

    name: str  # "vllm"
    display_name: str  # "vLLM"
    # --- parallelism ---------------------------------------------------
    supports_tensor_parallel: bool
    supports_pipeline_parallel: bool
    max_gpus: int | None  # None = unbounded
    # --- data plane ----------------------------------------------------
    openai_paths: frozenset[str]
    health_path: str
    metrics_path: str | None
    supports_request_priority: bool
    supports_lora: bool
    supports_vision: bool
    # Does this engine need the weights resolved to an on-disk PATH, or does it
    # take a repo id and do its own lookup? An engine fact, driver-invariant:
    # where the process runs does not change what its --model flag accepts.
    #
    # It exists because the answer is EXPENSIVE and FALLIBLE. Resolving means
    # scanning the HF cache, and Path.is_dir() propagates EACCES (it swallows
    # only ENOENT/ENOTDIR/EBADF/ELOOP), so an unreadable cache root raises. A
    # vLLM load that resolved anyway would gain a failure mode on a directory
    # it never opens -- which is exactly what sub-project C shipped and CI
    # caught: six pre-existing vLLM tests failed on a non-root runner that
    # cannot stat under /root/.cache/huggingface, while passing locally as root.
    #
    # Gating on THIS rather than on `capabilities.name == "vllm"` is the point.
    # A name check in Supervisor.load would put per-backend branching back into
    # the control plane -- the O(n*m) multiplication decision D1 exists to
    # remove -- and every future backend would need a new case there. Declaring
    # the fact keeps the supervisor at one call site with no engine name in it.
    needs_local_model_path: bool
    # --- control plane -------------------------------------------------
    # Can this backend's engine version be chosen per model AT ALL? A fact
    # about the engine, driver-invariant: vLLM publishes many versioned
    # upstream images, so this is True under every driver. Operator ruling,
    # 2026-09-01.
    #
    # THIS IS NOT THE FIELD A UI SHOULD GATE A CONTROL ON. Whether THIS
    # deployment can honour a pin is Backend.version_pin_available(driver) --
    # False on `bonus`, whose subprocess driver bakes the version into the
    # warden image. Reading supports_version_pin to decide whether to enable a
    # version selector lights up a control that cannot work, which is the #177
    # bug wearing a new hat.
    supports_version_pin: bool
    # The extra_env allowlist, per backend. Making the PREFIXES per-backend
    # NARROWS the accepted surface rather than widening it -- a llama.cpp row
    # today would accept VLLM_* keys that do nothing. The hard-locked key set
    # stays GLOBAL (backends/vllm/env.py:HARD_LOCKED_ENV_KEYS) because every
    # key in it is locked for reasons unrelated to which server runs.
    env_prefixes: tuple[str, ...]
    env_exact: frozenset[str]
    # --- memory model (fit preview) ------------------------------------
    # The fraction of physical VRAM the ENGINE will confine itself to, or
    # None when the engine takes that fraction from the operator instead.
    #
    # vLLM is None: `--gpu-memory-utilization` is a real vLLM flag the
    # operator sets per model, and vLLM hard-caps itself there, so the fit
    # preview must honour whatever they chose. llama.cpp is 1.0 because it
    # has NO equivalent flag -- `llama-server --help` in the shipped image
    # offers only `-ngl/--n-gpu-layers`, a layer COUNT, not a VRAM fraction
    # -- so it will use as much of the card as the weights and KV need.
    #
    # 1.0 is not "assume no headroom is required". The headroom judgement
    # lives in the verdict ladder, which already calls 0.80-1.0 "tight"
    # rather than "fits". Putting an invented safety fraction here as well
    # would apply that margin twice and be a number nobody could source.
    #
    # This is a field rather than `if capabilities.name == "llamacpp"` in
    # the route, for the reason `needs_local_model_path` spells out above:
    # a name check puts per-backend branching back into the control plane,
    # and every future backend then needs a new case in the fit route.
    vram_cap_fraction: float | None
    # --- data parallelism (#286) ---------------------------------------
    # Can this engine run N identical replicas behind one API server
    # (`--data-parallel-size`)? vLLM yes; llama.cpp has no such mode.
    # Defaulted and last so a backend that predates the field answers False.
    supports_data_parallel: bool = False
    # Does the engine put usage.prompt_tokens_details.cached_tokens (or
    # llama.cpp's timings.cache_n) on its responses? "none" (mlx_lm.server)
    # routes the cache fact through the TTFT estimate (app/cache_obs/record.py)
    # instead -- an estimate is labelled as one, never stored as measured.
    cached_tokens_reporting: Literal["usage", "none"] = "usage"


@dataclass(frozen=True)
class LaunchPlan:
    """A backend's answer to "how do I start this model?".

    argv + env, and NOTHING ELSE -- see the module docstring and spec §2.
    """

    argv: list[str]  # full argv INCLUDING argv[0]
    env: dict[str, str]
    image: str | None  # for image-swapping drivers only
    port: int
    gpu_indices: list[int] = field(default_factory=list)


@runtime_checkable
class Backend(Protocol):
    # Driver-invariant facts about the engine. A plain attribute, per §6.3.
    capabilities: BackendCapabilities

    def version_pin_available(self, driver: EngineDriver | None) -> bool:
        """Can THIS deployment actually honour a per-model version pin?

        ``driver`` IS A DRIVER OBJECT, NOT ITS NAME. Contrast ``bind_host``
        below, which takes ``driver_name: str``. The asymmetry is deliberate
        and is explained in "Two members, two kinds of driver" at the end of
        this Protocol -- read it before implementing either.

        The companion to ``capabilities.supports_version_pin``, and the split
        is load-bearing (sub-project E's resolution):

          * ``supports_version_pin`` -- *can this engine be pinned at all?*
            An engine fact. True for vLLM under every driver.
          * ``version_pin_available(driver)`` -- *can we honour it here?*
            A deployment fact. False under the in-container subprocess driver,
            whose engine version is baked into the warden image.

        **UI and API clients gate controls on THIS, never on the capability
        field.** Enabling a version selector because the engine "supports"
        pinning, on a deployment that cannot swap the image, offers a control
        that silently does nothing -- the #177 failure mode.
        ``/api/system/engine``'s long-standing ``supports_version_select`` is
        this value, not the capability field.

        ``driver`` is an ``EngineDriver`` instance, or ``None`` when the caller
        genuinely has none. ``None`` yields False: ADVERTISING fails closed,
        because claiming a pin we cannot honour is the #177 bug. Note this is
        the OPPOSITE default from ENFORCEMENT in ``Supervisor.load``, which
        keeps ``getattr(driver, "supports_engine_image", True)`` so an unknown
        or test stand-in driver is never wrongly BLOCKED from loading.
        Advertising a capability and refusing to act on one are different
        questions and get different defaults; both fail safe.
        """

    def bind_host(self, driver_name: str) -> str:
        """Interface this backend's HTTP server should bind, given the driver.

        ``driver_name`` IS A STRING -- ``settings.engine_driver``, i.e.
        ``"local"`` / ``"docker"``. Contrast ``version_pin_available`` above,
        which takes the driver OBJECT.

        A security boundary, not a connectivity knob: the engine's own OpenAI
        server is unauthenticated, so it must bind as narrowly as the driver
        allows. Backend-owned because it is the backend's server.
        """

    def plan(
        self,
        model,
        *,
        port: int,
        bind_host: str,
        overrides: dict | None = None,
        resolved=None,
    ) -> LaunchPlan:
        """Translate a model row + overrides into argv/env. PURE.

        No I/O, no clock, no environment reads beyond the backend's own
        documented escape hatches. ``overrides`` is the in-memory load-config
        dict; it never mutates the row.

        ``resolved`` is a ``ResolvedModelPaths``
        (``app/runtime/backends/paths.py``) or ``None``. It exists BECAUSE
        plan() is pure: llama.cpp needs ``-m /abs/path/to.gguf`` where vLLM
        needs ``--model <repo-id>``, and something has to touch the filesystem
        to turn one into the other. ``Supervisor.load`` does it once, up front,
        inside the try/except that releases the GPU claim -- so a mistyped
        filename is reported as itself rather than as a subprocess that exits
        rc=1 forty seconds later, and cannot leave a card reserved.

        EVERY backend takes the parameter; one that needs no path accepts and
        ignores it. That is what keeps ONE call site in the supervisor instead
        of a branch on the backend's name -- which is the whole point of the
        axis (D1, §6.1). Sub-project B deliberately left this parameter out
        because ``ResolvedModelPaths`` did not exist yet; sub-project C
        introduced the type and widened the signature here.

        Implementations additionally accept ``hf_token``, ``hf_cache_dir``,
        ``engine_driver`` and ``image`` -- launch context the supervisor has,
        left out of the Protocol because a backend that needs none of them
        should not have to declare them.
        """

    def health_url(self, host: str, port: int) -> str:
        """URL the control plane polls to decide the engine is up."""

    def diagnose(self, log_tail: str):
        """Backend-specific log grammar -> actionable operator message.

        Returns the backend's diagnosis object or ``None`` when nothing is
        recognised, so the caller keeps its generic message.
        """

    def parse_metrics(self, body: str):
        """Prometheus exposition text -> the backend's parsed metric set.

        Returns ``None`` when the body cannot be parsed. Sub-project C splits
        the vLLM metric-NAME table out of ``app/stats/live_engine.py`` when a
        second dialect exists to justify the seam.
        """


# ---------------------------------------------------------------------------
# Two members, two kinds of driver -- READ THIS BEFORE IMPLEMENTING A BACKEND
# ---------------------------------------------------------------------------
# ``bind_host(driver_name: str)``          takes a NAME  -- "local", "docker"
# ``version_pin_available(driver: obj)``   takes an OBJECT -- an EngineDriver
#
# The asymmetry is deliberate, and each side is the only shape that works:
#
#   bind_host must take a NAME because the caller only has one. It is
#   ``settings.engine_driver``, a config string, read on paths that have no
#   driver instance to hand (the effective-argv preview builds argv from a
#   persisted row with no supervisor involved). It also feeds a pure function
#   whose #211 docstring is an incident record we move verbatim.
#
#   version_pin_available must take an OBJECT because the answer is
#   ``getattr(driver, "supports_engine_image", False)`` -- a capability the
#   driver itself declares. Deriving it from a name would mean each backend
#   carrying a {"local": False, "docker": True} map, and sub-project D's k8s
#   driver would then need a new "k8s" entry in EVERY backend. That is the
#   O(n*m) multiplication decision D1 exists to remove; the object form makes
#   a new driver O(1) across all backends. (Sub-project E reached the same
#   conclusion independently and adopted this form.)
#
# THIS EXACT CONFUSION HAS ALREADY SHIPPED A DEFECT. E first typed
# ``version_pin_available(driver: str)`` and called it with "docker".
# ``getattr("docker", "supports_engine_image", False)`` is False, so the
# version selector was disabled on every docker deployment -- and it was
# CAMOUFLAGED, because the subprocess case returns the correct answer for the
# wrong reason. Only a docker-fixture assertion catches it.
#
# So the parameter names differ, the annotations differ, AND the vLLM
# implementation raises TypeError on the wrong kind rather than returning a
# plausible-but-wrong answer. Fail loud, not fail-closed-and-silent.
