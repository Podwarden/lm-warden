"""GET /api/system/engine — the active engine driver's capability.

The frontend Try-stack panel (#177) needs to know whether the running
deployment can swap the engine container image to a pinned vLLM version. Under
the default in-container subprocess driver it CANNOT — the vLLM version is
fixed by the warden image — so the version selector must be disabled and the
operator told why, instead of silently launching the warden-baked version.

``vllm_version`` is the vLLM package version baked into THIS image. We read it
cheaply via ``importlib.metadata`` (no ``import vllm`` — that drags in torch
and is far too heavy for a metadata read) and cache it at import time. It is
``None`` in environments where vLLM isn't installed (dev shell, pytest).

Sub-project B adds ``GET /api/system/backends``, which generalises this from
one driver capability to each backend's capabilities resolved against the
active driver. ``/api/system/engine``
is kept BYTE-IDENTICAL as a deprecated alias so an old UI against a new API
does not break mid-rollout; it is removed no earlier than the release after the
frontend stops calling it.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Depends, Request

from app.auth.deps import require_jwt
from app.runtime.backends import registry

try:
    _VLLM_VERSION: str | None = version("vllm")
except PackageNotFoundError:
    _VLLM_VERSION = None


def _backend_version(name: str) -> str | None:
    """The engine version baked into THIS warden image, per backend.

    Sub-project B wrote this as ``_VLLM_VERSION if name == "vllm" else None``,
    which was correct while there was one backend and becomes a lie the moment
    there are two: a llama.cpp entry reporting a vLLM version number is worse
    than one reporting none.

    Both halves are DECLARED values, not probes. vLLM's comes from its installed
    package metadata; llama.cpp's comes from the build tag the Dockerfile stamps
    into the environment (D2 -- the binary is compiled into the image, so its
    version is a property of the image). Neither runs a subprocess: this route
    answers a metadata question and must not acquire a failure mode for it.

    ``None`` outside a built image -- a dev shell, the test suite -- exactly as
    ``_VLLM_VERSION`` is None where vLLM is not installed.
    """
    if name == "vllm":
        return _VLLM_VERSION
    if name == "llamacpp":
        from app.runtime.backends.llamacpp.version import version_string

        return version_string()
    return None


class _ImageSwappingDriver:
    """A stand-in driver used to ask one hypothetical question, never to run.

    "Would a driver that CAN swap the engine image make this backend's version
    pin available?" is the question that separates a removable obstacle from a
    permanent one, and only the backend can answer it -- so we ask the backend,
    with the only driver property ``version_pin_available`` reads.

    It is not a fake in the test sense: nothing is loaded through it and it
    never reaches the supervisor. It exists so this route derives the answer
    from ``Backend.version_pin_available`` rather than re-deriving it from a
    backend NAME, which is the per-backend branching in the control plane that
    decision D1 removed.
    """

    supports_engine_image = True


_HYPOTHETICAL_IMAGE_SWAPPING_DRIVER = _ImageSwappingDriver()

# The machine-readable half of ``version_pin_reason``. A client that must
# choose between two DIFFERENT renderings -- hide the control, or show it
# disabled with the sentence beside it -- cannot choose from prose without
# string-matching it, and a UI that greps a human sentence for "docker" is one
# copy-edit away from silently changing behaviour.
VersionPinObstacle = str  # "engine" | "driver" | "backend"


def _version_pin_obstacle(
    *,
    display_name: str,
    engine_supports: bool,
    driver_swaps_images: bool,
    pinnable_on_image_swapping_driver: bool,
) -> tuple[VersionPinObstacle | None, str | None]:
    """``(code, sentence)`` for why this backend's pin is unavailable here.

    ``(None, None)`` when it IS available. The two are returned together, out
    of one branch, precisely so they can never disagree: the sentence is what
    the operator reads and the code is what the client switches on, and a UI
    whose behaviour and whose explanation describe different worlds is worse
    than either failure alone.

    A disabled control the operator cannot explain is the failure this route
    exists to prevent, and the explanation cannot be written in the frontend:
    the frontend sees one boolean and cannot tell whether the obstacle is the
    DRIVER (removable -- run the docker engine driver) or the BACKEND (not
    removable -- there is no image catalogue to pin against). Two different
    answers to "what would make this work?", so the server owns both.

    ORDER IS LOAD-BEARING. The backend check comes BEFORE the driver check.
    Under the in-container subprocess driver -- what `bonus` runs -- llama.cpp
    is blocked by BOTH, and the driver-first order this function shipped with
    answered "the driver": a true sentence about vLLM, and a false promise
    about llama.cpp, whose pin no driver can ever make available. An operator
    who acted on it would migrate a deployment to the docker driver to unlock
    a control that would still be dead.
    """
    if not engine_supports:
        return "engine", f"{display_name} does not support version pinning."
    if not pinnable_on_image_swapping_driver:
        # No driver helps: llama.cpp has no (channel, version) -> image
        # resolver (app/runtime/backends/vllm/images.py has no counterpart), so
        # honouring a pin would resolve a vLLM image and launch vLLM under the
        # operator's llama.cpp model name. Blaming the driver here would be
        # false whichever driver is running.
        return "backend", (
            f"{display_name} cannot be version-pinned on any driver: its binary "
            f"is compiled into the warden image and there is no {display_name} "
            f"image catalogue to pin against."
        )
    if not driver_swaps_images:
        # The old wording -- "a version pin would be silently discarded" --
        # described the pre-#177 bug, not what happens now: Supervisor.load
        # raises EnginePinUnsupported before claiming a GPU, precisely so a
        # pin can never be discarded silently. Left uncorrected it read as a
        # control confessing it does nothing, which is a fair thing to
        # conclude from a sentence saying the input is thrown away. It is a
        # deployment choice, and the sentence should say so.
        return "driver", (
            "This deployment runs the in-container engine driver: the engine "
            "version is fixed by the warden image, so a pin is refused rather "
            "than applied. Version selection requires the docker engine "
            "driver (VW_ENGINE_DRIVER=docker)."
        )
    return None, None


router = APIRouter()


@router.get("/api/system/engine")
async def get_engine(request: Request, _user: str = Depends(require_jwt)) -> dict:
    driver = getattr(request.app.state, "supervisor", None)
    driver = getattr(driver, "_driver", None)
    # Unknown/test stand-in drivers default to capable so we never wrongly
    # block them — mirrors the defensive read in Supervisor.load.
    supports = getattr(driver, "supports_engine_image", True)
    return {
        "driver": "docker" if supports else "subprocess",
        "supports_version_select": supports,
        "vllm_version": _VLLM_VERSION,
    }


@router.get("/api/system/backends")
async def get_backends(request: Request, _user: str = Depends(require_jwt)) -> dict:
    """Which backends this build has, what each can do, and what the ACTIVE
    driver lets them do.

    Backend and driver compose as a product, not an enumeration, so this route
    answers for the deployment as it actually runs. Version pinning is reported
    as TWO fields, and on the deployment we run they disagree:

      supports_version_pin   the ENGINE fact -- can vLLM be pinned at all?
                             True, driver-invariant (operator ruling,
                             2026-09-01).
      version_pin_available  the DEPLOYMENT fact -- can this driver honour it?
                             False under the in-container subprocess driver.

    CLIENTS GATE CONTROLS ON version_pin_available. Enabling a version selector
    because the engine "supports" pinning, on a box that cannot swap the image,
    offers a control that silently does nothing. Advertising both is what lets
    the UI disable the control AND explain why, instead of a load failing 40
    seconds in.

    A third field carries the "why":

      version_pin_reason     non-null EXACTLY when version_pin_available is
                             false; rendered verbatim beside the disabled
                             control. The client cannot write this sentence
                             itself -- one boolean does not say whether the
                             obstacle is the driver (fixable: run the docker
                             engine driver) or the backend (llama.cpp has no
                             image catalogue to pin against).
    """
    driver = getattr(request.app.state, "supervisor", None)
    driver = getattr(driver, "_driver", None)
    supports_image = getattr(driver, "supports_engine_image", True)

    backends = []
    for name in registry.available():
        backend = registry.get(name)
        caps = backend.capabilities
        pin_available = backend.version_pin_available(driver)
        pin_code, pin_reason = (
            (None, None)
            if pin_available
            else _version_pin_obstacle(
                display_name=caps.display_name,
                engine_supports=caps.supports_version_pin,
                driver_swaps_images=supports_image,
                pinnable_on_image_swapping_driver=backend.version_pin_available(
                    _HYPOTHETICAL_IMAGE_SWAPPING_DRIVER
                ),
            )
        )
        backends.append(
            {
                "name": caps.name,
                "display_name": caps.display_name,
                # The engine version baked into THIS image. Only meaningful for a
                # backend that ships inside the warden image; None otherwise.
                "version": _backend_version(caps.name),
                "supports_tensor_parallel": caps.supports_tensor_parallel,
                "supports_pipeline_parallel": caps.supports_pipeline_parallel,
                "supports_data_parallel": caps.supports_data_parallel,
                "max_gpus": caps.max_gpus,
                "openai_paths": sorted(caps.openai_paths),
                "health_path": caps.health_path,
                "metrics_path": caps.metrics_path,
                "supports_request_priority": caps.supports_request_priority,
                "supports_lora": caps.supports_lora,
                "supports_vision": caps.supports_vision,
                "needs_local_model_path": caps.needs_local_model_path,
                "env_prefixes": list(caps.env_prefixes),
                "env_exact": sorted(caps.env_exact),
                # null = the operator's gpu_memory_utilization is the cap for this
                # backend. A number = the backend has no such flag and confines
                # itself to that fraction. The Add-model dialog recomputes the
                # weights budget client-side when the GPU selection changes, so it
                # needs the same fact the fit route uses or the two disagree.
                "vram_cap_fraction": caps.vram_cap_fraction,
                # The ENGINE fact (driver-invariant) and the DEPLOYMENT fact
                # (this driver). Clients gate controls on the SECOND one; see the
                # Backend protocol docstring for why both ship.
                "supports_version_pin": caps.supports_version_pin,
                "version_pin_available": pin_available,
                # Non-null exactly when version_pin_available is False. A client
                # renders it verbatim beside the disabled control; it never has to
                # guess which of the two obstacles applies.
                "version_pin_reason": pin_reason,
                # The same decision, machine-readable: null | "engine" | "driver" |
                # "backend". Clients that RENDER the reason use the sentence;
                # clients that BRANCH on it use this. "driver" is removable by the
                # operator, so the control stays visible and disabled; "backend"
                # and "engine" are not, and the control is a concept the engine
                # does not have, so it is hidden (frontend/src/lib/backend-fields.ts
                # states that rule for the field-level case and now owns this one).
                "version_pin_reason_code": pin_code,
            }
        )

    return {
        "default": registry.DEFAULT_BACKEND,
        "driver": "docker" if supports_image else "subprocess",
        "engine_version": _VLLM_VERSION,
        "backends": backends,
    }
