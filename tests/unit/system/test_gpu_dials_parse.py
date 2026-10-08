"""Clocks, throttle reasons, the compute-cap -> architecture map and the
``nvidia-smi topo -m`` matrix behind the GPU dials on /stats.

Fixtures are recorded from real hosts where one existed (the 4x A4000 host
under thermal slowdown, the 2-card mixed host); the bridged-pair and switch
fabric matrices are the shapes ``topo -m`` documents for hardware we do not
have. Card names are what nvidia-smi prints for the hardware.
"""

from __future__ import annotations

import pytest

from app.system.gpu import (
    NVIDIA_SMI_THROTTLE_CMDS,
    GpuTopology,
    ThrottleReasons,
    architecture_for,
    parse_nvidia_smi_live_csv,
    parse_throttle_reasons_csv,
    parse_topology_matrix,
    query_gpu_snapshot,
    reset_static_facts_cache,
)

# ---------------------------------------------------------------------------
# compute capability -> generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cap", "arch"),
    [
        (5.0, "Maxwell"),
        (5.2, "Maxwell"),
        (5.3, "Maxwell"),
        (6.0, "Pascal"),
        (6.1, "Pascal"),
        (6.2, "Pascal"),
        (7.0, "Volta"),
        (7.2, "Volta"),
        (7.5, "Turing"),
        (8.0, "Ampere"),
        (8.6, "Ampere"),
        (8.7, "Ampere"),
        (8.9, "Ada Lovelace"),
        (9.0, "Hopper"),
        (10.0, "Blackwell"),
        (10.1, "Blackwell"),
        (10.3, "Blackwell"),
        (12.0, "Blackwell"),
    ],
)
def test_architecture_map_is_fixed(cap, arch):
    assert architecture_for(cap) == arch


@pytest.mark.parametrize("cap", [7.1, 7.3, 8.1, 8.8, 9.5, 11.0, 12.1, 13.0, 3.5, 0.0])
def test_unrecognised_capability_is_not_invented(cap):
    # A generation whose number is not public must NOT get a guessed label;
    # the panel shows the raw number and says it is not recognised.
    assert architecture_for(cap) is None


def test_architecture_of_missing_capability_is_none():
    assert architecture_for(None) is None
    assert architecture_for(-1.0) is None


# ---------------------------------------------------------------------------
# clocks: columns 21-23 of the live query
# ---------------------------------------------------------------------------

# The 4x A4000 host under load, thermally throttled: SM clocks well below
# the 2100 MHz ceiling, memory clock at its 7001 MHz ceiling minus the
# usual 500.
LIVE_THROTTLED = (
    "0, GPU-a, NVIDIA RTX A4000, 16376, 15155, 1221, 100, 8.6, "
    "00000000:01:00.0, 93, 89, 112.40, 140.00, Disabled, P2, 1350, 4, 4, 16, 16, 2100, 6501, 7001\n"
    "1, GPU-b, NVIDIA RTX A4000, 16376, 15155, 1221, 100, 8.6, "
    "00000000:02:00.0, 91, 77, 110.60, 140.00, Disabled, P2, 1560, 4, 4, 16, 16, 2100, 6501, 7001\n"
)


def test_parse_live_csv_clock_columns():
    g0, g1 = parse_nvidia_smi_live_csv(LIVE_THROTTLED)
    assert (g0.sm_clock_mhz, g0.sm_clock_max_mhz) == (1350, 2100)
    assert (g0.mem_clock_mhz, g0.mem_clock_max_mhz) == (6501, 7001)
    assert g1.sm_clock_mhz == 1560
    # The older columns still parse alongside.
    assert (g0.pcie_width_current, g0.pcie_width_max) == (16, 16)
    assert g0.temperature_c == 93 and g0.utilization_pct == 100


def test_parse_live_csv_20_col_rows_have_no_clock_ceilings():
    out = (
        "0, GPU-a, NVIDIA RTX A4000, 16376, 100, 16276, 0, 8.6, "
        "00000000:01:00.0, 41, 30, 20.0, 140.00, Disabled, P0, 1560, 1, 3, 16, 16\n"
    )
    (gpu,) = parse_nvidia_smi_live_csv(out)
    assert gpu.sm_clock_mhz == 1560
    assert gpu.sm_clock_max_mhz is None
    assert gpu.mem_clock_mhz is None and gpu.mem_clock_max_mhz is None


def test_parse_live_csv_clock_na_stays_none_not_zero():
    out = (
        "0, GPU-a, NVIDIA RTX A4000, 16376, 100, 16276, 0, 8.6, "
        "00000000:01:00.0, 41, 30, 20.0, 140.00, Disabled, P0, 210, 1, 3, 16, 16, "
        "[N/A], [Not Supported], [N/A]\n"
    )
    (gpu,) = parse_nvidia_smi_live_csv(out)
    assert gpu.sm_clock_mhz == 210
    assert gpu.sm_clock_max_mhz is None
    assert gpu.mem_clock_mhz is None
    assert gpu.mem_clock_max_mhz is None


# ---------------------------------------------------------------------------
# throttle reasons
# ---------------------------------------------------------------------------

# Verbatim from the 4-card host at 91-95 C: every card in SW thermal
# slowdown, nothing else active. mask 0x20 == sw_thermal_slowdown.
THROTTLE_THERMAL = (
    "0, 0x0000000000000020, Active, Not Active, Not Active\n"
    "1, 0x0000000000000020, Active, Not Active, Not Active\n"
    "2, 0x0000000000000020, Active, Not Active, Not Active\n"
    "3, 0x0000000000000020, Active, Not Active, Not Active\n"
)


def test_parse_throttle_reasons_thermal_host():
    reasons = parse_throttle_reasons_csv(THROTTLE_THERMAL)
    assert sorted(reasons) == [0, 1, 2, 3]
    assert reasons[0] == ThrottleReasons(
        mask=0x20,
        sw_thermal=True,
        hw_thermal=False,
        sw_power_cap=False,
        hw_slowdown=False,
        hw_power_brake=False,
        idle=False,
    )


def test_parse_throttle_reasons_idle_card_is_not_throttled():
    # mask 0x1 = gpu_idle: the clocks are low because there is nothing to
    # do, which is not a slowdown and must not read as one.
    (r,) = parse_throttle_reasons_csv(
        "0, 0x0000000000000001, Not Active, Not Active, Not Active\n"
    ).values()
    assert r.idle is True
    assert (r.sw_thermal, r.hw_thermal, r.sw_power_cap, r.hw_slowdown) == (
        False,
        False,
        False,
        False,
    )


def test_parse_throttle_reasons_distinguishes_power_cap_from_thermal():
    out = (
        "0, 0x0000000000000004, Not Active, Not Active, Active\n"  # power cap
        "1, 0x0000000000000048, Not Active, Active, Not Active\n"  # hw thermal + hw slowdown
        "2, 0x0000000000000080, Not Active, Not Active, Not Active\n"  # power brake, mask only
    )
    r = parse_throttle_reasons_csv(out)
    assert (r[0].sw_power_cap, r[0].sw_thermal, r[0].hw_thermal) == (True, False, False)
    assert (r[1].hw_thermal, r[1].hw_slowdown, r[1].sw_thermal) == (True, True, False)
    # The mask carries reasons the named columns do not.
    assert r[2].hw_power_brake is True
    assert (r[2].sw_thermal, r[2].sw_power_cap) == (False, False)


def test_parse_throttle_reasons_mask_fills_in_na_named_columns():
    r = parse_throttle_reasons_csv("0, 0x0000000000000020, [N/A], [N/A], [N/A]\n")[0]
    assert r.sw_thermal is True
    assert r.hw_thermal is False and r.sw_power_cap is False


def test_parse_throttle_reasons_everything_na_is_unknown_not_false():
    r = parse_throttle_reasons_csv("0, [N/A], [N/A], [N/A], [N/A]\n")[0]
    assert r.mask is None
    assert r.sw_thermal is None and r.hw_thermal is None and r.sw_power_cap is None
    assert r.hw_slowdown is None and r.idle is None


def test_parse_throttle_reasons_skips_malformed():
    r = parse_throttle_reasons_csv(
        "broken\n0, 0x20, Active, Not Active\n1, 0x0, Not Active, Not Active, Not Active\n"
    )
    assert sorted(r) == [1]


# ---------------------------------------------------------------------------
# nvidia-smi topo -m
# ---------------------------------------------------------------------------

# Verbatim from the 4-card host: no NVLink anywhere, every pair through the
# host bridge. Tab-separated, the self cell padded " X ".
TOPO_NO_NVLINK = (
    "\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\n"
    "GPU0\t X \tPHB\tPHB\tPHB\t0-27\t0\t\tN/A\n"
    "GPU1\tPHB\t X \tPHB\tPHB\t0-27\t0\t\tN/A\n"
    "GPU2\tPHB\tPHB\t X \tPHB\t0-27\t0\t\tN/A\n"
    "GPU3\tPHB\tPHB\tPHB\t X \t0-27\t0\t\tN/A\n"
    "\n"
    "Legend:\n"
    "\n"
    "  X    = Self\n"
    "  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)\n"
    "  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node\n"
    "  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)\n"
    "  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)\n"
    "  PIX  = Connection traversing at most a single PCIe bridge\n"
    "  NV#  = Connection traversing a bonded set of # NVLinks\n"
)

# Two pairs bridged (0-1, 2-3), the pairs reaching each other through the
# host bridge. The shape an NVLink-bridge workstation reports.
TOPO_BRIDGED_PAIRS = (
    "\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\n"
    "GPU0\t X \tNV2\tPHB\tPHB\t0-63\t0\n"
    "GPU1\tNV2\t X \tPHB\tPHB\t0-63\t0\n"
    "GPU2\tPHB\tPHB\t X \tNV2\t0-63\t0\n"
    "GPU3\tPHB\tPHB\tNV2\t X \t0-63\t0\n"
    "\nLegend:\n  X    = Self\n"
)

# Every pair NVLink: an NVSwitch fabric. Also carries the NIC columns and
# rows a DGX-class host prints, which must fall away.
TOPO_FABRIC = (
    "\tGPU0\tGPU1\tGPU2\tGPU3\tNIC0\tNIC1\tCPU Affinity\tNUMA Affinity\n"
    "GPU0\t X \tNV12\tNV12\tNV12\tPXB\tSYS\t0-31\t0\n"
    "GPU1\tNV12\t X \tNV12\tNV12\tPXB\tSYS\t0-31\t0\n"
    "GPU2\tNV12\tNV12\t X \tNV12\tSYS\tPXB\t32-63\t1\n"
    "GPU3\tNV12\tNV12\tNV12\t X \tSYS\tPXB\t32-63\t1\n"
    "NIC0\tPXB\tPXB\tSYS\tSYS\t X \tSYS\n"
    "NIC1\tSYS\tSYS\tPXB\tPXB\tSYS\t X \n"
    "\nLegend:\n  X    = Self\n"
    "\nNIC Legend:\n\n  NIC0: mlx5_0\n  NIC1: mlx5_1\n"
)


def test_parse_topology_no_nvlink_host():
    topo = parse_topology_matrix(TOPO_NO_NVLINK)
    assert topo == GpuTopology(
        indices=[0, 1, 2, 3],
        matrix=[
            ["X", "PHB", "PHB", "PHB"],
            ["PHB", "X", "PHB", "PHB"],
            ["PHB", "PHB", "X", "PHB"],
            ["PHB", "PHB", "PHB", "X"],
        ],
    )


def test_parse_topology_bridged_pairs():
    topo = parse_topology_matrix(TOPO_BRIDGED_PAIRS)
    assert topo is not None
    assert topo.matrix[0][1] == "NV2" and topo.matrix[1][0] == "NV2"
    assert topo.matrix[2][3] == "NV2" and topo.matrix[3][2] == "NV2"
    assert topo.matrix[0][2] == "PHB" and topo.matrix[1][3] == "PHB"


def test_parse_topology_fabric_drops_nic_rows_and_columns():
    topo = parse_topology_matrix(TOPO_FABRIC)
    assert topo is not None
    assert topo.indices == [0, 1, 2, 3]
    assert len(topo.matrix) == 4 and all(len(row) == 4 for row in topo.matrix)
    off_diagonal = [c for i, row in enumerate(topo.matrix) for j, c in enumerate(row) if i != j]
    assert set(off_diagonal) == {"NV12"}
    assert all(topo.matrix[i][i] == "X" for i in range(4))


def test_parse_topology_two_card_mixed_host():
    out = (
        "\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\n"
        "GPU0\t X \tPHB\t0-27\t0\t\tN/A\n"
        "GPU1\tPHB\t X \t0-27\t0\t\tN/A\n"
        "\nLegend:\n"
    )
    assert parse_topology_matrix(out) == GpuTopology(
        indices=[0, 1],
        matrix=[["X", "PHB"], ["PHB", "X"]],
    )


def test_parse_topology_space_separated_build():
    # A build that pads with spaces instead of tabs: the GPU columns lead,
    # so single tokens still align.
    out = (
        "        GPU0    GPU1    CPU Affinity    NUMA Affinity\n"
        "GPU0     X      NV4     0-15            0\n"
        "GPU1    NV4      X      0-15            0\n"
    )
    assert parse_topology_matrix(out) == GpuTopology(
        indices=[0, 1],
        matrix=[["X", "NV4"], ["NV4", "X"]],
    )


def test_parse_topology_sixteen_cards_is_not_capped():
    n = 16
    header = "\t" + "\t".join(f"GPU{i}" for i in range(n)) + "\tCPU Affinity\n"
    rows = "".join(
        f"GPU{i}\t" + "\t".join(" X " if i == j else "NV12" for j in range(n)) + "\t0-95\n"
        for i in range(n)
    )
    topo = parse_topology_matrix(header + rows)
    assert topo is not None and topo.indices == list(range(n))
    assert len(topo.matrix) == n and all(len(r) == n for r in topo.matrix)


def test_parse_topology_nothing_reported():
    assert parse_topology_matrix("") is None
    assert parse_topology_matrix("No devices were found\n") is None


def test_parse_topology_missing_row_is_not_reported():
    # Header names four cards, rows cover three: a half picture is worse
    # than none — the panel says "not reported".
    out = (
        "\tGPU0\tGPU1\tGPU2\tGPU3\n"
        "GPU0\t X \tPHB\tPHB\tPHB\n"
        "GPU1\tPHB\t X \tPHB\tPHB\n"
        "GPU2\tPHB\tPHB\t X \tPHB\n"
    )
    assert parse_topology_matrix(out) is None


# ---------------------------------------------------------------------------
# the snapshot probe: throttle spelling fallback + topology on the snapshot
# ---------------------------------------------------------------------------


async def test_query_gpu_snapshot_merges_throttle_and_topology(monkeypatch):
    from app.system import gpu as mod

    async def fake_run(args):
        if args is mod.NVIDIA_SMI_LIVE_CMD:
            return LIVE_THROTTLED
        if args is mod.NVIDIA_SMI_APPS_CMD:
            return ""
        if args is NVIDIA_SMI_THROTTLE_CMDS[0]:
            return THROTTLE_THERMAL
        if args is mod.NVIDIA_SMI_TOPO_CMD:
            return TOPO_NO_NVLINK
        return None

    monkeypatch.setattr(mod, "_run_nvidia_smi", fake_run)
    reset_static_facts_cache()
    try:
        snap = await query_gpu_snapshot()
        assert snap.probe_error is None
        assert snap.gpus[0].throttle is not None and snap.gpus[0].throttle.sw_thermal is True
        assert snap.gpus[1].throttle is not None and snap.gpus[1].throttle.sw_power_cap is False
        assert snap.topology is not None and snap.topology.indices == [0, 1, 2, 3]
        assert snap.topology.matrix[0][1] == "PHB"
    finally:
        reset_static_facts_cache()


async def test_driver_that_renamed_throttle_reasons_falls_back_to_event_spelling(monkeypatch):
    """A driver that dropped the ``clocks_throttle_reasons`` alias rejects
    that query outright. Only the throttle probe is affected: the live row
    keeps every other field, the ``clocks_event_reasons`` spelling is tried,
    and once it works it is tried FIRST on the next probe."""
    from app.system import gpu as mod

    calls: list[list[str]] = []

    async def fake_run(args):
        calls.append(args)
        if args is mod.NVIDIA_SMI_LIVE_CMD:
            return LIVE_THROTTLED
        if args is mod.NVIDIA_SMI_APPS_CMD:
            return ""
        if args is NVIDIA_SMI_THROTTLE_CMDS[0]:
            return None  # "Field ... is not a valid field to query" -> exit 1
        if args is NVIDIA_SMI_THROTTLE_CMDS[1]:
            return THROTTLE_THERMAL
        return None

    monkeypatch.setattr(mod, "_run_nvidia_smi", fake_run)
    reset_static_facts_cache()
    try:
        snap = await query_gpu_snapshot()
        assert snap.gpus[0].throttle is not None and snap.gpus[0].throttle.sw_thermal is True
        assert snap.gpus[0].sm_clock_max_mhz == 2100
        assert NVIDIA_SMI_THROTTLE_CMDS[0] in calls and NVIDIA_SMI_THROTTLE_CMDS[1] in calls

        calls.clear()
        await query_gpu_snapshot()
        throttle_calls = [a for a in calls if a in NVIDIA_SMI_THROTTLE_CMDS]
        assert throttle_calls == [NVIDIA_SMI_THROTTLE_CMDS[1]]
    finally:
        reset_static_facts_cache()


async def test_driver_without_any_throttle_field_leaves_verdict_not_reported(monkeypatch):
    from app.system import gpu as mod

    async def fake_run(args):
        if args is mod.NVIDIA_SMI_LIVE_CMD:
            return LIVE_THROTTLED
        if args is mod.NVIDIA_SMI_APPS_CMD:
            return ""
        return None

    monkeypatch.setattr(mod, "_run_nvidia_smi", fake_run)
    reset_static_facts_cache()
    try:
        snap = await query_gpu_snapshot()
        assert snap.probe_error is None
        assert snap.gpus[0].throttle is None  # not "not throttled"
        assert snap.gpus[0].sm_clock_mhz == 1350  # the rest is intact
        assert snap.topology is None
    finally:
        reset_static_facts_cache()


def test_underlined_header_does_not_swallow_gpu_zero() -> None:
    """nvidia-smi UNDERLINES the header, gluing an escape onto the first cell.

    Verbatim from a real four-card host: note the escape before ``GPU0`` and
    the reset closing the line. Matched naively, the first column is not
    ``GPU0`` but ``\x1b[4mGPU0``, so card 0 is dropped and the panel draws
    three cards on a four-card box — with a legend that counts three. That is
    what shipped in v2026.09.06.3.
    """
    raw = (
        "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
        "GPU0\t X \tPHB\tPHB\tPHB\t0-15\t0\t\tN/A\n"
        "GPU1\tPHB\t X \tPHB\tPHB\t0-15\t0\t\tN/A\n"
        "GPU2\tPHB\tPHB\t X \tPHB\t0-15\t0\t\tN/A\n"
        "GPU3\tPHB\tPHB\tPHB\t X \t0-15\t0\t\tN/A\n"
    )
    topo = parse_topology_matrix(raw)
    assert topo is not None
    assert topo.indices == [0, 1, 2, 3], "card 0 must survive the escape sequence"
    assert len(topo.matrix) == 4
    assert topo.matrix[0][1] == "PHB"
