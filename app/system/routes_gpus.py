"""Live GPU probe: ``GET /api/system/gpus``.

Hits ``nvidia-smi`` directly each call (with a short in-process cache to
absorb burst polling) and returns per-GPU memory + a list of "holders" — the
PIDs currently using GPU memory, labelled with the owning model_id when the
supervisor recognises them.

The chart-friendly historical endpoint remains ``GET /api/stats/gpus`` in
``app/stats/routes_api.py``; this endpoint is the 5–10 s live gauge feed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.db.repos.setup import SetupRepo
from app.system.gpu import (
    GpuLive,
    GpuSnapshot,
    GpuTopology,
    architecture_for,
    query_gpu_snapshot,
)
from app.system.pid_attribution import attribute_pid_to_model

logger = logging.getLogger(__name__)
router = APIRouter()

# Burst-poll absorbtion. The frontend gauges poll every 5–10 s, but multiple
# tabs / fast scrubbing should not stack subprocesses. Two seconds is short
# enough to feel live and long enough to drop double-clicks.
PROBE_CACHE_TTL_SECONDS = 2.0


class _ProbeCache:
    """Single in-process cache for the nvidia-smi snapshot.

    A single asyncio.Lock guards both the cache read and the underlying probe,
    so concurrent requests inside the TTL window collapse to one nvidia-smi
    invocation. Lives on ``app.state.gpu_probe_cache`` so we can hand it a
    settable clock + probe fn for tests.
    """

    def __init__(
        self,
        *,
        ttl: float = PROBE_CACHE_TTL_SECONDS,
        clock=time.monotonic,
        probe=query_gpu_snapshot,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._probe = probe
        self._lock = asyncio.Lock()
        self._cached: GpuSnapshot | None = None
        self._cached_at: float = -1e9
        self.invocations = 0  # exposed for tests

    async def get(self) -> GpuSnapshot:
        async with self._lock:
            now = self._clock()
            if self._cached is not None and (now - self._cached_at) < self._ttl:
                return self._cached
            self.invocations += 1
            snap = await self._probe()
            self._cached = snap
            self._cached_at = now
            return snap


def _get_cache(request: Request) -> _ProbeCache:
    cache = getattr(request.app.state, "gpu_probe_cache", None)
    if cache is None:
        cache = _ProbeCache()
        request.app.state.gpu_probe_cache = cache
    return cache


async def _build_model_label_map(db_path: str) -> dict[str, str]:
    """Return {model_id: served_model_name} for every model currently in DB."""
    async with open_db(db_path) as db:
        rows = await ModelRepo(db).list_all()
    return {r.id: r.served_model_name for r in rows}


async def _allowed_gpu_indices(db_path: str) -> list[int] | None:
    """The setup wizard's GPU allowlist, or None if none was ever recorded.

    ``POST /api/models`` has always rejected a selection outside this list with
    a 400, but nothing read it back, so the Add-model dialog offered every card
    the box has and learned about the allowlist only by being refused -- after
    the operator had filled in the whole form. Offering a control that cannot
    work is the defect this codebase's capability plumbing exists to avoid.

    None and [] are DIFFERENT and clients must treat them so: None is "no
    allowlist recorded -- everything is selectable"; [] is "the operator allowed
    nothing". Collapsing them would lock an operator out of a box whose setup
    draft predates the key.
    """
    async with open_db(db_path) as db:
        state = await SetupRepo(db).get()
    allowed = state.draft.get("allowed_gpu_indices")
    if not isinstance(allowed, list):
        return None
    return sorted(int(i) for i in allowed)


def _holder_payload(
    app,
    *,
    parent_pid_to_model: dict[int, str],
    label_for: dict[str, str],
    pid: int,
    process_name: str,
    memory_mib: int,
) -> dict[str, Any]:
    model_id = attribute_pid_to_model(pid, parent_pid_to_model)
    if model_id is not None:
        return {
            "pid": pid,
            "memory_mib": memory_mib,
            "process": process_name,
            "kind": "model",
            "model_id": model_id,
            "label": label_for.get(model_id),
        }
    return {
        "pid": pid,
        "memory_mib": memory_mib,
        "process": process_name,
        "kind": "external",
        "model_id": None,
        "label": None,
    }


def _telemetry_payload(g: GpuLive) -> dict[str, Any]:
    """The per-card telemetry block. Every value is nullable and null means
    "the hardware did not report this" — the UI renders those as "not
    reported", never as 0 or blank, because a fan-less card, a card with no
    power sensor and a card drawing 0 W are three different things.

    ``nvlink`` is null only when the ``nvlink -s`` probe gave nothing for the
    card; a card with no NVLink is ``{"state": "unsupported"}``, which is a
    real answer and rendered as one. Likewise ``throttle`` is null only when
    the throttle-reason probe gave nothing; a card that is not throttling is
    an object with every flag false."""
    nvlink = (
        None
        if g.nvlink is None
        else {
            "state": g.nvlink.state,
            "active_links": g.nvlink.active_links,
            "total_links": g.nvlink.total_links,
            "link_speed_gbs": g.nvlink.link_speed_gbs,
        }
    )
    throttle = (
        None
        if g.throttle is None
        else {
            "mask": g.throttle.mask,
            "sw_thermal": g.throttle.sw_thermal,
            "hw_thermal": g.throttle.hw_thermal,
            "sw_power_cap": g.throttle.sw_power_cap,
            "hw_slowdown": g.throttle.hw_slowdown,
            "hw_power_brake": g.throttle.hw_power_brake,
            "idle": g.throttle.idle,
        }
    )
    return {
        "temperature_c": g.temperature_c,
        "temp_slowdown_c": g.temp_slowdown_c,
        "temp_shutdown_c": g.temp_shutdown_c,
        "temp_max_operating_c": g.temp_max_operating_c,
        "fan_pct": g.fan_pct,
        "power_w": g.power_w,
        "power_limit_w": g.power_limit_w,
        "ecc_enabled": g.ecc_enabled,
        "pstate": g.pstate,
        # Each clock against its own ceiling — the ceiling is what makes a
        # low reading readable (210 MHz idle is normal; 1200 of 2100 under
        # load is a card being held back).
        "sm_clock_mhz": g.sm_clock_mhz,
        "sm_clock_max_mhz": g.sm_clock_max_mhz,
        "mem_clock_mhz": g.mem_clock_mhz,
        "mem_clock_max_mhz": g.mem_clock_max_mhz,
        # The driver's own statement of why the clocks are held, never a
        # guess from a temperature ratio. Thermal, power cap and hardware
        # slowdown are different problems and stay separate flags.
        "throttle": throttle,
        "nvlink": nvlink,
        # gen_current is 1 on an idle card (PCIe power-saving) and is NOT a
        # fault; width_current < width_max is a permanent slot/riser limit.
        # gen_max is the negotiated ceiling; gen_gpu_max / gen_host_max are
        # what each endpoint could do, null on drivers that lack the fields.
        "pcie": {
            "gen_current": g.pcie_gen_current,
            "gen_max": g.pcie_gen_max,
            "width_current": g.pcie_width_current,
            "width_max": g.pcie_width_max,
            "gen_gpu_max": g.pcie_gen_gpu_max,
            "gen_host_max": g.pcie_gen_host_max,
        },
    }


def _topology_payload(topo: GpuTopology | None) -> dict[str, Any] | None:
    """The ``topo -m`` matrix as the panel draws it. ``indices[i]`` is the GPU
    number of row/column ``i``; cells are the legend codes verbatim ("X",
    "NV2", "PHB", ...). null = the probe gave nothing."""
    if topo is None:
        return None
    return {"indices": list(topo.indices), "matrix": [list(row) for row in topo.matrix]}


@router.get("/api/system/gpus")
async def system_gpus(request: Request, _user: str = Depends(require_jwt)) -> dict[str, Any]:
    """Live nvidia-smi snapshot with per-GPU memory and PID-attributed holders.

    Response shape (locked for Phase 2 frontend consumption)::

        {
          "probed_at": "<iso8601 UTC>",
          "probe_error": null | "nvidia-smi unavailable" | ...,
          "gpus": [
            {
              "index": 0,
              "name": "NVIDIA RTX A4000",
              "memory_total_mib": 16376,
              "memory_used_mib": 12450,
              "memory_free_mib": 3926,
              "utilization_pct": 87,
              "compute_cap": 8.6 | null,
              "architecture": "Ampere" | null,   # derived from compute_cap only
              "telemetry": {            # each null = "not reported"
                "temperature_c": 82,  "temp_slowdown_c": 100,
                "temp_shutdown_c": 103, "temp_max_operating_c": 98,
                "fan_pct": 80,        "power_w": 53.1,  "power_limit_w": 140.0,
                "ecc_enabled": false, "pstate": "P2",
                "sm_clock_mhz": 1350,  "sm_clock_max_mhz": 2100,
                "mem_clock_mhz": 6501, "mem_clock_max_mhz": 7001,
                "throttle": {"mask": 32, "sw_thermal": true, "hw_thermal": false,
                             "sw_power_cap": false, "hw_slowdown": false,
                             "hw_power_brake": false, "idle": false} | null,
                "nvlink": {"state": "unsupported" | "inactive" | "active",
                           "active_links": 0, "total_links": 0,
                           "link_speed_gbs": null} | null,
                "pcie": {"gen_current": 1, "gen_max": 3,      # gen 1 idle = normal
                         "width_current": 4, "width_max": 16, # x4 of x16 = a limit
                         "gen_gpu_max": 4, "gen_host_max": 4}
              },
              "holders": [
                {
                  "pid": 12345, "memory_mib": 12400,
                  "process": "vllm-worker",
                  "kind": "model",                 # we own this PID
                  "model_id": "61c82bbc55d5147b",
                  "label": "gpt-oss-20b"
                },
                {
                  "pid": 67890, "memory_mib": 50,
                  "process": "Xorg",
                  "kind": "external",              # unknown owner
                  "model_id": null,
                  "label": null
                }
              ]
            }
          ],
          "topology": {"indices": [0, 1, 2, 3],
                       "matrix": [["X", "PHB", "PHB", "PHB"], ...]} | null,
          "allowed_indices": [0, 1] | null
        }

    ``topology`` is the ``nvidia-smi topo -m`` matrix: ``X`` self, ``NV#`` a
    bond of # NVLinks, and PIX / PXB / PHB / NODE / SYS all "no direct link,
    routed through PCIe at increasing distance". null when the probe gave
    nothing.

    ``allowed_indices`` is the setup wizard's GPU allowlist -- the same list
    ``POST /api/models`` enforces with a 400. It is appended, never a
    replacement for ``gpus``: every physically present card is still reported,
    because hiding a card the operator can see in nvidia-smi answers a
    different question than the one they asked. ``null`` means no allowlist was
    recorded and is NOT the same as ``[]``.

    When ``nvidia-smi`` is not on PATH (dev box without NVIDIA) the endpoint
    still returns HTTP 200 with ``gpus: []`` and ``probe_error`` populated so
    the UI can render an empty state rather than crash.
    """
    snap = await _get_cache(request).get()
    parent_pid_to_model = request.app.state.supervisor.parent_pid_to_model()
    label_for = await _build_model_label_map(request.app.state.settings.db_path)

    uuid_to_index: dict[str, int] = {g.uuid: g.index for g in snap.gpus}
    holders_by_index: dict[int, list[dict[str, Any]]] = {g.index: [] for g in snap.gpus}
    for app in snap.apps:
        idx = uuid_to_index.get(app.gpu_uuid)
        if idx is None:
            # nvidia-smi returned a holder for a UUID we didn't see in the
            # per-GPU output — shouldn't happen but don't drop it on the
            # floor; log and skip.
            logger.warning("compute-app on unknown gpu_uuid=%s pid=%d", app.gpu_uuid, app.pid)
            continue
        holders_by_index[idx].append(
            _holder_payload(
                request.app,
                parent_pid_to_model=parent_pid_to_model,
                label_for=label_for,
                pid=app.pid,
                process_name=app.process_name,
                memory_mib=app.memory_mib,
            )
        )

    gpus_payload: list[dict[str, Any]] = []
    for g in snap.gpus:
        holders = sorted(holders_by_index[g.index], key=lambda h: h["memory_mib"], reverse=True)
        gpus_payload.append(
            {
                "index": g.index,
                "name": g.name,
                "memory_total_mib": g.memory_total_mib,
                "memory_used_mib": g.memory_used_mib,
                "memory_free_mib": g.memory_free_mib,
                "utilization_pct": g.utilization_pct,
                # #176 — CUDA compute capability (e.g. 8.6 / sm_86). null when the
                # driver doesn't report it. The Add Model wizard crosses this with
                # the candidate quant/dtype in fit-preview to warn on emulated
                # FP8 / unsupported builds.
                "compute_cap": g.compute_cap,
                # Derived from compute_cap and nothing else: nvidia-smi has no
                # architecture field. null = capability missing or not in the
                # fixed map, and the panel then shows the raw number.
                "architecture": architecture_for(g.compute_cap),
                "telemetry": _telemetry_payload(g),
                "holders": holders,
            }
        )

    return {
        "probed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "probe_error": snap.probe_error,
        "gpus": gpus_payload,
        "topology": _topology_payload(snap.topology),
        # The setup wizard's allowlist, so a client can stop offering GPUs that
        # POST /api/models would reject. Read from SQLite, so it survives a
        # failed probe: a box whose driver is missing still knows what it is
        # configured for. null != [] -- see _allowed_gpu_indices.
        "allowed_indices": await _allowed_gpu_indices(request.app.state.settings.db_path),
    }
