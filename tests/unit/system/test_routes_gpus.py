"""Tests for ``GET /api/system/gpus`` — the live nvidia-smi probe.

We exercise the route via TestClient against the real FastAPI app, with the
nvidia-smi probe stubbed at the function level and a fake supervisor PID map
installed on ``app.state``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app.system.gpu import (
    GpuComputeApp,
    GpuLive,
    GpuSnapshot,
    GpuTopology,
    NvlinkStatus,
    ThrottleReasons,
)
from tests.conftest import jwt_login, seed_admin_user


def _seed_done(db_path: Path, *, models: list[tuple[str, str]] | None = None) -> None:
    # #55 fix — admin user + setup_state via shared barrier-aware helper.
    seed_admin_user(db_path, allowed_gpu_indices=[0, 1])
    if not models:
        return
    with sqlite3.connect(db_path) as db:
        for model_id, served in models:
            db.execute(
                "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
                "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
                "gpu_memory_utilization, trust_remote_code, extra_args, status) "
                "VALUES (?, ?, 'o/r', 'main', '[0]', 1, NULL, NULL, 0.9, 0, '[]', 'loaded')",
                (model_id, served),
            )
        db.commit()


def _jwt_auth(client) -> dict[str, str]:
    # #55 fix — delegate to the barrier+retry helper.
    return jwt_login(client)


def _install_probe(client, snapshots: list[GpuSnapshot]) -> dict[str, int]:
    """Replace the cached probe on app.state with one returning queued snapshots.

    Returns a counter dict the test asserts on; ``counter['n']`` is the number
    of probe invocations (so we can prove the 2 s cache is collapsing them).
    """
    from app.system.routes_gpus import _ProbeCache

    counter: dict[str, int] = {"n": 0}

    queue = list(snapshots)

    async def fake_probe() -> GpuSnapshot:
        counter["n"] += 1
        if not queue:
            return snapshots[-1]
        return queue.pop(0)

    # Freeze the clock so cached value stays valid across the test calls.
    cache = _ProbeCache(ttl=2.0, clock=lambda: 0.0, probe=fake_probe)
    client.app.state.gpu_probe_cache = cache
    return counter


def _install_supervisor_pids(client, pid_to_model: dict[int, str]) -> None:
    """Patch the live supervisor's PID-to-model accessor."""
    client.app.state.supervisor.parent_pid_to_model = lambda: dict(pid_to_model)


def _patch_attribute(monkeypatch: pytest.MonkeyPatch, mapping: dict[int, int]) -> None:
    """Replace pgrp lookup so worker PIDs map to the supervisor parent PID."""
    from app.system import routes_gpus as routes_module

    def fake_attribute(pid: int, parent_pid_to_model: dict[int, str]) -> str | None:
        if pid in parent_pid_to_model:
            return parent_pid_to_model[pid]
        pgid = mapping.get(pid)
        if pgid is None:
            return None
        return parent_pid_to_model.get(pgid)

    monkeypatch.setattr(routes_module, "attribute_pid_to_model", fake_attribute)


SNAP_TWO_GPUS = GpuSnapshot(
    gpus=[
        GpuLive(
            index=0,
            uuid="GPU-aaaa",
            name="NVIDIA RTX A4000",
            memory_total_mib=16376,
            memory_used_mib=12450,
            memory_free_mib=3926,
            utilization_pct=87,
        ),
        GpuLive(
            index=1,
            uuid="GPU-bbbb",
            name="NVIDIA RTX A4000",
            memory_total_mib=16376,
            memory_used_mib=100,
            memory_free_mib=16276,
            utilization_pct=0,
        ),
    ],
    apps=[
        GpuComputeApp(pid=12345, gpu_uuid="GPU-aaaa", process_name="vllm-worker", memory_mib=12400),
        GpuComputeApp(pid=67890, gpu_uuid="GPU-aaaa", process_name="Xorg", memory_mib=50),
        GpuComputeApp(pid=22222, gpu_uuid="GPU-bbbb", process_name="python", memory_mib=100),
    ],
    probe_error=None,
)


def test_system_gpus_requires_auth(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/system/gpus")
    assert r.status_code == 401


def test_system_gpus_attributes_known_pids(tmp_data_dir, client, monkeypatch):
    """PID owned by supervisor → kind=model + model_id + label.
    PID unknown → kind=external."""
    client.get("/healthz")
    _seed_done(
        tmp_data_dir / "vllm-warden.db",
        models=[
            ("model-a", "gpt-oss-20b"),
        ],
    )
    _install_probe(client, [SNAP_TWO_GPUS])
    # Supervisor's tracked parent PID is 12345 → model-a.
    _install_supervisor_pids(client, {12345: "model-a"})
    # pid 12345 is direct hit; 67890 + 22222 are external (no pgrp mapping).
    _patch_attribute(monkeypatch, mapping={})

    auth = _jwt_auth(client)
    r = client.get("/api/system/gpus", headers=auth)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    assert body["probe_error"] is None
    assert len(body["gpus"]) == 2
    gpu0 = body["gpus"][0]
    assert gpu0 == {
        "index": 0,
        "name": "NVIDIA RTX A4000",
        "memory_total_mib": 16376,
        "memory_used_mib": 12450,
        "memory_free_mib": 3926,
        "utilization_pct": 87,
        # #176 — SNAP_TWO_GPUS rows don't set compute_cap, so it surfaces None.
        "compute_cap": None,
        "architecture": None,
        # Telemetry the fixture rows never set: every slot null = "not
        # reported", and nvlink null (probe gave nothing) is distinct from
        # {"state": "unsupported"}.
        "telemetry": {
            "temperature_c": None,
            "temp_slowdown_c": None,
            "temp_shutdown_c": None,
            "temp_max_operating_c": None,
            "fan_pct": None,
            "power_w": None,
            "power_limit_w": None,
            "ecc_enabled": None,
            "pstate": None,
            "sm_clock_mhz": None,
            "sm_clock_max_mhz": None,
            "mem_clock_mhz": None,
            "mem_clock_max_mhz": None,
            "throttle": None,
            "nvlink": None,
            "pcie": {
                "gen_current": None,
                "gen_max": None,
                "width_current": None,
                "width_max": None,
                "gen_gpu_max": None,
                "gen_host_max": None,
            },
        },
        "holders": [
            {
                "pid": 12345,
                "memory_mib": 12400,
                "process": "vllm-worker",
                "kind": "model",
                "model_id": "model-a",
                "label": "gpt-oss-20b",
            },
            {
                "pid": 67890,
                "memory_mib": 50,
                "process": "Xorg",
                "kind": "external",
                "model_id": None,
                "label": None,
            },
        ],
    }
    # Holder on the other GPU is external (pid 22222 — not in supervisor map).
    assert body["gpus"][1]["holders"] == [
        {
            "pid": 22222,
            "memory_mib": 100,
            "process": "python",
            "kind": "external",
            "model_id": None,
            "label": None,
        }
    ]


def test_system_gpus_surfaces_compute_cap(tmp_data_dir, client, monkeypatch):
    """#176 — a parsed compute_cap surfaces verbatim on the live shape; a row
    without it surfaces null."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    snap = GpuSnapshot(
        gpus=[
            GpuLive(
                index=0,
                uuid="GPU-aaaa",
                name="NVIDIA RTX A4000",
                memory_total_mib=16376,
                memory_used_mib=100,
                memory_free_mib=16276,
                utilization_pct=0,
                compute_cap=8.6,
            ),
            GpuLive(
                index=1,
                uuid="GPU-bbbb",
                name="Tesla V100",
                memory_total_mib=16376,
                memory_used_mib=100,
                memory_free_mib=16276,
                utilization_pct=0,
                compute_cap=None,
            ),
        ],
        apps=[],
        probe_error=None,
    )
    _install_probe(client, [snap])
    _install_supervisor_pids(client, {})
    _patch_attribute(monkeypatch, mapping={})

    auth = _jwt_auth(client)
    r = client.get("/api/system/gpus", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["gpus"][0]["compute_cap"] == 8.6
    assert body["gpus"][1]["compute_cap"] is None
    # The generation is derived from compute_cap and nothing else — the
    # V100 row has no capability, so it gets no label rather than a guess
    # from its name.
    assert body["gpus"][0]["architecture"] == "Ampere"
    assert body["gpus"][1]["architecture"] is None


def test_system_gpus_attributes_worker_pid_via_pgrp(tmp_data_dir, client, monkeypatch):
    """vLLM tensor-parallel worker (pid 99999) belongs to supervisor parent
    pid 12345 (pgrp == 12345). Must be labelled kind=model."""
    client.get("/healthz")
    _seed_done(
        tmp_data_dir / "vllm-warden.db",
        models=[
            ("model-a", "gpt-oss-20b"),
        ],
    )
    snap = GpuSnapshot(
        gpus=[
            GpuLive(
                index=0,
                uuid="GPU-aaaa",
                name="NVIDIA RTX A4000",
                memory_total_mib=16376,
                memory_used_mib=12450,
                memory_free_mib=3926,
                utilization_pct=87,
            )
        ],
        apps=[
            GpuComputeApp(
                pid=99999, gpu_uuid="GPU-aaaa", process_name="vllm-worker", memory_mib=12400
            )
        ],
        probe_error=None,
    )
    _install_probe(client, [snap])
    _install_supervisor_pids(client, {12345: "model-a"})
    # pgrp(99999) == 12345
    _patch_attribute(monkeypatch, mapping={99999: 12345})

    auth = _jwt_auth(client)
    r = client.get("/api/system/gpus", headers=auth)
    assert r.status_code == 200, r.text
    holder = r.json()["gpus"][0]["holders"][0]
    assert holder["kind"] == "model"
    assert holder["model_id"] == "model-a"
    assert holder["label"] == "gpt-oss-20b"


def test_system_gpus_cache_collapses_burst(tmp_data_dir, client, monkeypatch):
    """Two GETs within TTL → only one nvidia-smi invocation."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    counter = _install_probe(client, [SNAP_TWO_GPUS, SNAP_TWO_GPUS])
    _install_supervisor_pids(client, {})
    _patch_attribute(monkeypatch, mapping={})

    auth = _jwt_auth(client)
    r1 = client.get("/api/system/gpus", headers=auth)
    r2 = client.get("/api/system/gpus", headers=auth)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert counter["n"] == 1, f"expected 1 probe, got {counter['n']}"


def test_system_gpus_orphan_pid_falls_back_to_external(tmp_data_dir, client, monkeypatch):
    """Defence test for the supervisor-restart / pid-namespace-mismatch path.

    nvidia-smi reports pid 12345 holding GPU memory, but the supervisor's
    parent_pid_to_model map is empty — e.g. because the api container was
    restarted and orphaned the vLLM subprocess, or because nvidia-smi
    returned a *host* PID while the supervisor stored an *in-container* PID
    (the production failure mode #42 documents — requires `pid: host` /
    `hostPID: true` to keep the namespaces unified). Holder must degrade to
    ``kind: external`` cleanly — never crash, never mis-attribute.
    """
    client.get("/healthz")
    _seed_done(
        tmp_data_dir / "vllm-warden.db",
        models=[
            ("model-a", "gpt-oss-20b"),
        ],
    )
    _install_probe(client, [SNAP_TWO_GPUS])
    # Supervisor knows about model-a but its tracked PID (in-container or
    # restart-orphaned) doesn't match what nvidia-smi reports.
    _install_supervisor_pids(client, {99999: "model-a"})
    _patch_attribute(monkeypatch, mapping={})  # no pgrp resolution either

    auth = _jwt_auth(client)
    r = client.get("/api/system/gpus", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    # All three nvidia-smi compute apps must show as external; none should
    # be mis-attributed to model-a.
    all_holders = [h for g in body["gpus"] for h in g["holders"]]
    assert len(all_holders) == 3
    for h in all_holders:
        assert h["kind"] == "external", h
        assert h["model_id"] is None
        assert h["label"] is None


def test_system_gpus_probe_error_surfaces(tmp_data_dir, client, monkeypatch):
    """nvidia-smi missing → 200 with probe_error populated and empty gpus."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    _install_probe(client, [GpuSnapshot(gpus=[], apps=[], probe_error="nvidia-smi unavailable")])
    _install_supervisor_pids(client, {})
    _patch_attribute(monkeypatch, mapping={})

    auth = _jwt_auth(client)
    r = client.get("/api/system/gpus", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["gpus"] == []
    assert body["probe_error"] == "nvidia-smi unavailable"
    assert "probed_at" in body


# ---------------------------------------------------------------------------
# allowed_gpu_indices
#
# The setup wizard writes it into setup_state.draft and POST /api/models
# enforces it with a 400. Until now NO route read it back, so the Add-model
# dialog offered every physically present card and only learned about the
# allowlist from a rejection after the operator had filled in the whole form --
# a control that cannot work, offered anyway.
#
# It rides on this route rather than a new one: the wizard already calls
# /api/system/gpus for exactly this list, and the allowlist is a fact about the
# same GPUs.
# ---------------------------------------------------------------------------


def test_system_gpus_reports_the_configured_allowlist(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[1])
    _install_probe(client, [SNAP_TWO_GPUS])
    _install_supervisor_pids(client, {})
    _patch_attribute(monkeypatch, mapping={})

    auth = _jwt_auth(client)
    body = client.get("/api/system/gpus", headers=auth).json()
    # Both cards are still reported -- the allowlist narrows what may be
    # SELECTED; it does not hide hardware the operator can see in nvidia-smi.
    assert [g["index"] for g in body["gpus"]] == [0, 1]
    assert body["allowed_indices"] == [1]


def test_allowed_indices_is_null_when_setup_recorded_none(tmp_data_dir, client):
    """Null means "no allowlist recorded", which is NOT "none allowed".

    A client must read null as "every GPU is selectable". Collapsing the two
    would lock the operator out of a box whose setup draft predates the key.
    """
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE setup_state SET draft = '{}' WHERE id = 1")
        db.commit()

    auth = _jwt_auth(client)
    body = client.get("/api/system/gpus", headers=auth).json()
    assert body["allowed_indices"] is None


def test_allowed_indices_survives_a_probe_failure(tmp_data_dir, client, monkeypatch):
    """The allowlist comes from SQLite, not nvidia-smi. A box whose driver is
    missing still knows which GPUs it is configured for."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0, 3])
    _install_probe(client, [GpuSnapshot(gpus=[], apps=[], probe_error="nvidia-smi unavailable")])
    _install_supervisor_pids(client, {})
    _patch_attribute(monkeypatch, mapping={})

    auth = _jwt_auth(client)
    body = client.get("/api/system/gpus", headers=auth).json()
    assert body["gpus"] == []
    assert body["allowed_indices"] == [0, 3]


def test_system_gpus_surfaces_per_card_telemetry(tmp_data_dir, client, monkeypatch):
    """A heterogeneous two-card host: each card's own limits, ECC and NVLink
    standing surface verbatim, and a missing sensor surfaces as null rather
    than 0 — the difference between "no fan" and "fan at 0 %"."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    snap = GpuSnapshot(
        gpus=[
            GpuLive(
                index=0,
                uuid="GPU-aaaa",
                name="NVIDIA RTX A4000",
                memory_total_mib=16376,
                memory_used_mib=14137,
                memory_free_mib=2239,
                utilization_pct=0,
                compute_cap=8.6,
                bus_id="00000000:01:00.0",
                temperature_c=35,
                fan_pct=41,
                power_w=12.68,
                power_limit_w=140.0,
                ecc_enabled=False,
                pstate="P8",
                sm_clock_mhz=210,
                sm_clock_max_mhz=2100,
                mem_clock_mhz=405,
                mem_clock_max_mhz=7001,
                throttle=ThrottleReasons(
                    mask=1,
                    sw_thermal=False,
                    hw_thermal=False,
                    sw_power_cap=False,
                    hw_slowdown=False,
                    hw_power_brake=False,
                    idle=True,
                ),
                nvlink=NvlinkStatus(state="unsupported"),
                temp_slowdown_c=100,
                temp_shutdown_c=103,
                temp_max_operating_c=98,
                pcie_gen_current=1,
                pcie_gen_max=3,
                pcie_width_current=4,
                pcie_width_max=16,
                pcie_gen_gpu_max=4,
                pcie_gen_host_max=4,
            ),
            GpuLive(
                index=1,
                uuid="GPU-bbbb",
                name="Quadro RTX 5000",
                memory_total_mib=15360,
                memory_used_mib=13818,
                memory_free_mib=1542,
                utilization_pct=0,
                compute_cap=7.5,
                bus_id="00000000:02:00.0",
                temperature_c=29,
                fan_pct=None,
                power_w=0.0,
                power_limit_w=230.0,
                ecc_enabled=True,
                pstate="P8",
                sm_clock_mhz=300,
                nvlink=NvlinkStatus(state="inactive"),
            ),
        ],
        apps=[],
        probe_error=None,
    )
    _install_probe(client, [snap])
    _install_supervisor_pids(client, {})
    _patch_attribute(monkeypatch, mapping={})

    auth = _jwt_auth(client)
    r = client.get("/api/system/gpus", headers=auth)
    assert r.status_code == 200, r.text
    t0, t1 = (g["telemetry"] for g in r.json()["gpus"])
    assert t0 == {
        "temperature_c": 35,
        "temp_slowdown_c": 100,
        "temp_shutdown_c": 103,
        "temp_max_operating_c": 98,
        "fan_pct": 41,
        "power_w": 12.68,
        "power_limit_w": 140.0,
        "ecc_enabled": False,
        "pstate": "P8",
        "sm_clock_mhz": 210,
        "sm_clock_max_mhz": 2100,
        "mem_clock_mhz": 405,
        "mem_clock_max_mhz": 7001,
        # Idle: the driver says so (mask 0x1) and nothing is slowing it down.
        "throttle": {
            "mask": 1,
            "sw_thermal": False,
            "hw_thermal": False,
            "sw_power_cap": False,
            "hw_slowdown": False,
            "hw_power_brake": False,
            "idle": True,
        },
        "nvlink": {
            "state": "unsupported",
            "active_links": 0,
            "total_links": 0,
            "link_speed_gbs": None,
        },
        # Idle gen 1 against a negotiated max of 3, x4 of x16, both endpoints
        # Gen 4-capable: every number verbatim, no judgement applied here.
        "pcie": {
            "gen_current": 1,
            "gen_max": 3,
            "width_current": 4,
            "width_max": 16,
            "gen_gpu_max": 4,
            "gen_host_max": 4,
        },
    }
    # The second card never had PCIe fields (older row shape): all null.
    assert all(v is None for v in t1["pcie"].values())
    assert t1["power_limit_w"] == 230.0 and t1["ecc_enabled"] is True
    assert t1["fan_pct"] is None  # no fan sensor -> null, not 0
    assert t1["power_w"] == 0.0  # a real 0 W reading stays 0
    assert t1["temp_slowdown_c"] is None  # no thermal block for this card
    assert t1["nvlink"]["state"] == "inactive"
    # No throttle probe answer for card 1: null, which is not "not throttled".
    assert t1["throttle"] is None
    assert t1["sm_clock_max_mhz"] is None and t1["mem_clock_mhz"] is None
    # No topo -m in this snapshot either.
    assert r.json()["topology"] is None


def test_system_gpus_surfaces_topology_and_throttle_verdict(tmp_data_dir, client, monkeypatch):
    """The 4-card host as it is right now: every card in SW thermal slowdown
    holding 57-74 % of its SM ceiling, and a topo -m matrix with no NVLink
    anywhere. The matrix rides beside the list (it is a relationship BETWEEN
    cards), verbatim legend codes, and the throttle flags come from the
    driver — no temperature heuristic anywhere on the route."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    clocks = [1350, 1560, 1305, 1200]
    snap = GpuSnapshot(
        gpus=[
            GpuLive(
                index=i,
                uuid=f"GPU-{i}",
                name="NVIDIA RTX A4000",
                memory_total_mib=16376,
                memory_used_mib=15155,
                memory_free_mib=1221,
                utilization_pct=100,
                compute_cap=8.6,
                temperature_c=[93, 91, 95, 93][i],
                temp_slowdown_c=100,
                sm_clock_mhz=clocks[i],
                sm_clock_max_mhz=2100,
                mem_clock_mhz=6501,
                mem_clock_max_mhz=7001,
                throttle=ThrottleReasons(
                    mask=0x20,
                    sw_thermal=True,
                    hw_thermal=False,
                    sw_power_cap=False,
                    hw_slowdown=False,
                    hw_power_brake=False,
                    idle=False,
                ),
            )
            for i in range(4)
        ],
        apps=[],
        probe_error=None,
        topology=GpuTopology(
            indices=[0, 1, 2, 3],
            matrix=[
                ["X", "PHB", "PHB", "PHB"],
                ["PHB", "X", "PHB", "PHB"],
                ["PHB", "PHB", "X", "PHB"],
                ["PHB", "PHB", "PHB", "X"],
            ],
        ),
    )
    _install_probe(client, [snap])
    _install_supervisor_pids(client, {})
    _patch_attribute(monkeypatch, mapping={})

    auth = _jwt_auth(client)
    body = client.get("/api/system/gpus", headers=auth).json()
    assert body["topology"] == {
        "indices": [0, 1, 2, 3],
        "matrix": [
            ["X", "PHB", "PHB", "PHB"],
            ["PHB", "X", "PHB", "PHB"],
            ["PHB", "PHB", "X", "PHB"],
            ["PHB", "PHB", "PHB", "X"],
        ],
    }
    for i, g in enumerate(body["gpus"]):
        assert g["architecture"] == "Ampere"
        t = g["telemetry"]
        assert t["throttle"]["sw_thermal"] is True
        assert t["throttle"]["sw_power_cap"] is False
        assert (t["sm_clock_mhz"], t["sm_clock_max_mhz"]) == (clocks[i], 2100)
