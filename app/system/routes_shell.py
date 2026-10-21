"""GET /api/shell/status — the loopback-only shell status feed (planner iter 44).

The HTTP seam over the two pure pieces already merged by the dependency tasks:
``require_shell_client`` (``shell_auth.py``) is a ``Depends`` and refuses any
non-loopback, forwarded or bearer-mismatched peer; ``build_shell_status``
(``shell_status.py``) turns the supervisor's live engines and the proxy's
in-flight registry into a ``ShellStatus``. This module only assembles them and
adds the apple-profile memory probes and the ``Cache-Control: no-store``
header.

``include_in_schema=False`` keeps the route out of the OpenAPI document, so
``api-types.generated.ts`` never changes. The route is mounted by ``build_app``
only when ``VW_SHELL_TOKEN`` is set — the wiring is a separate task
(``shell-status-wiring``); this file carries no wiring.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.system.apple.proc import phys_footprint
from app.system.shell_auth import require_shell_client
from app.system.shell_status import ShellStatus, build_shell_status

router = APIRouter()


def _run(argv: list[str]) -> str | None:
    """``subprocess.run(argv)`` -> stdout on returncode 0, else None.

    The runner ``read_memory_pressure`` expects; a raising spawn or a non-zero
    exit degrades to None (that probe then leaves the field None) and any
    exception is swallowed by the caller's per-probe try, never a 500."""
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=2, check=False)
    if completed.returncode != 0:
        return None
    return completed.stdout


@router.get(
    "/api/shell/status",
    include_in_schema=False,
    response_model=ShellStatus,
    dependencies=[Depends(require_shell_client)],
)
async def shell_status(request: Request) -> JSONResponse:
    """The shell's loaded models, memory and in-flight requests (loopback only).

    ``usable_bytes`` and ``pressure_level`` are the apple profile's memory
    probes: the unified-memory budget in MiB scaled to bytes, and
    ``kern.memorystatus_level``. On the linux profile both are None and no
    subprocess runs. Any probe exception degrades that one value to None — a
    wedged probe never takes the feed down with a 500.
    """
    supervisor: Any = request.app.state.supervisor
    registry: Any = getattr(request.app.state, "request_registry", None)
    requests = registry.snapshot() if registry is not None else []

    usable_bytes: int | None = None
    pressure_level: int | None = None
    from app import platform_profile
    from app.system import gpu
    from app.system.apple.memory import read_memory_pressure

    if platform_profile.PROFILE == "apple":
        try:
            budget_mib = await gpu._apple_budget_mib()
            usable_bytes = budget_mib * (2**20) if budget_mib else None
        except Exception:
            usable_bytes = None
        try:
            pressure_level = (await asyncio.to_thread(read_memory_pressure, _run)).pressure_level
        except Exception:
            pressure_level = None

    status = build_shell_status(
        supervisor,
        requests,
        footprint=phys_footprint,
        usable_bytes=usable_bytes,
        pressure_level=pressure_level,
    )
    return JSONResponse(status.model_dump(), headers={"Cache-Control": "no-store"})
