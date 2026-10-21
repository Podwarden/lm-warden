"""Pure builder for the macOS shell's status feed (planner iter 44).

``build_shell_status`` turns the supervisor's live engines and the proxy's
in-flight request registry into a ``ShellStatus`` value: loaded models with
their resident bytes, used/usable memory and a pressure level, and in-flight
request groups. It is the backend half of the loopback-only
``GET /api/shell/status`` route (the token guard and the route are separate
tasks); this module has no HTTP and no secrets.

The function is pure apart from calling the injected ``footprint`` probe
(``app.system.apple.proc.phys_footprint`` in production, a dict in tests), so
the whole contract is exercised on Linux with a duck-typed fake supervisor and
no live engine.

Privacy is the point of the shape: the output carries exactly the fields the
menu bar needs — model name/backend/state/bytes, used/usable bytes, a
pressure level, and (token name, model, count) groups — and nothing else. No
client IP, token id, request id, path, prompt or completion text, prompt or
completion token counts, timings, engine argv/descriptor, env, host or port.
"""

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.proxy.request_registry import LiveRequest


class ShellModel(BaseModel):
    """One loaded engine: its identity, backend, lifecycle state and resident bytes."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    backend: str
    state: str
    bytes: int | None


class ShellMemory(BaseModel):
    """Used and usable model memory, plus a numeric pressure level when the OS reports one."""

    model_config = ConfigDict(extra="forbid")

    used_bytes: int
    usable_bytes: int | None
    pressure_level: int | None


class ShellInFlightGroup(BaseModel):
    """A bundle of in-flight requests sharing the same caller and served model."""

    model_config = ConfigDict(extra="forbid")

    who: str
    model: str
    requests: int


class ShellInFlight(BaseModel):
    """The in-flight total and its per-(caller, model) breakdown."""

    model_config = ConfigDict(extra="forbid")

    count: int
    groups: list[ShellInFlightGroup]


class ShellStatus(BaseModel):
    """The whole feed: loaded models, memory, and in-flight requests."""

    model_config = ConfigDict(extra="forbid")

    models: list[ShellModel]
    memory: ShellMemory
    in_flight: ShellInFlight


def _model_bytes(pid: int | None, footprint: Callable[[int], int | None]) -> int | None:
    """``footprint(pid)`` when it yields a non-negative int, else None.

    ``pid`` must be a positive int (None, 0 or negative means "no known
    process"); a ``footprint`` that raises degrades to None rather than
    propagating, so a dead or unreadable process never takes the feed down."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        value = footprint(pid)
    except Exception:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def _variant_backend(descriptor: Any) -> str:
    """The variant descriptor's backend, or 'unknown' when absent or not a non-empty str."""
    backend = descriptor.get("backend") if isinstance(descriptor, dict) else None
    return backend if isinstance(backend, str) and backend else "unknown"


def build_shell_status(
    supervisor: Any,
    requests: Sequence[LiveRequest],
    *,
    footprint: Callable[[int], int | None],
    usable_bytes: int | None,
    pressure_level: int | None,
) -> ShellStatus:
    """Build the shell's ``ShellStatus`` from live engines and in-flight requests.

    ``supervisor`` is duck-typed and read for exactly four members:
    ``live_engine_ids()``, ``get_variant(id)``, ``get_state(id)`` and
    ``get_pid(id)``. ``footprint(pid)`` is called once per engine with a
    positive pid to get its resident bytes. ``usable_bytes`` and
    ``pressure_level`` pass through unchanged, except ``usable_bytes <= 0``
    and a negative ``pressure_level`` become None.
    """
    models: list[ShellModel] = []
    for model_id in sorted(supervisor.live_engine_ids()):
        variant = supervisor.get_variant(model_id)
        name = variant.served_model_name if variant is not None else model_id
        backend = _variant_backend(variant.descriptor) if variant is not None else "unknown"
        state_obj = supervisor.get_state(model_id)
        state = state_obj.value if state_obj is not None else "unknown"
        pid = supervisor.get_pid(model_id)
        models.append(
            ShellModel(
                id=model_id,
                name=name,
                backend=backend,
                state=state,
                bytes=_model_bytes(pid, footprint),
            )
        )
    models.sort(key=lambda m: (m.name, m.id))

    used_bytes = sum(m.bytes for m in models if m.bytes is not None)
    if usable_bytes is not None and usable_bytes <= 0:
        usable_bytes = None
    if pressure_level is not None and pressure_level < 0:
        pressure_level = None
    memory = ShellMemory(
        used_bytes=used_bytes,
        usable_bytes=usable_bytes,
        pressure_level=pressure_level,
    )

    counts: dict[tuple[str, str], int] = {}
    for req in requests:
        key = (req.token_name or "unknown", req.model)
        counts[key] = counts.get(key, 0) + 1
    groups = [
        ShellInFlightGroup(who=who, model=model, requests=n)
        for (who, model), n in sorted(
            counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    ]
    in_flight = ShellInFlight(count=len(requests), groups=groups)

    return ShellStatus(models=models, memory=memory, in_flight=in_flight)
