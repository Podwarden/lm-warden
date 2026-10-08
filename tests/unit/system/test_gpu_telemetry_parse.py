"""Per-card telemetry parsing for the /stats GPU cards.

Covers the telemetry columns of NVIDIA_SMI_LIVE_CMD, the free-text
``nvidia-smi nvlink -s`` report (three distinct states), the ``-q -d
TEMPERATURE`` thresholds, and the merge that puts them on the live rows.

Fixtures are recorded from real hosts. Card names are what nvidia-smi
prints for the hardware; nothing here names a model.
"""

from __future__ import annotations

from app.system.gpu import (
    NvlinkStatus,
    StaticGpuFacts,
    TempLimits,
    merge_static_facts,
    parse_nvidia_smi_live_csv,
    parse_nvlink_status,
    parse_temperature_limits,
    query_gpu_snapshot,
    reset_static_facts_cache,
)

# A heterogeneous host: an RTX A4000 (16376 MiB, 140 W cap, ECC disabled)
# next to a Quadro RTX 5000 (15360 MiB, 230 W cap, ECC enabled). Nothing may
# be assumed shared between the two cards.
LIVE_HETERO = (
    "0, GPU-aaaa, NVIDIA RTX A4000, 16376, 14137, 2239, 0, 8.6, "
    "00000000:01:00.0, 35, 41, 12.68, 140.00, Disabled, P8, 210\n"
    "1, GPU-bbbb, Quadro RTX 5000, 15360, 13818, 1542, 0, 7.5, "
    "00000000:02:00.0, 29, 33, 12.23, 230.00, Enabled, P8, 300\n"
)


def test_parse_live_csv_heterogeneous_two_card_host():
    gpus = parse_nvidia_smi_live_csv(LIVE_HETERO)
    assert [g.name for g in gpus] == ["NVIDIA RTX A4000", "Quadro RTX 5000"]
    a4000, rtx5000 = gpus
    assert (a4000.memory_total_mib, rtx5000.memory_total_mib) == (16376, 15360)
    assert (a4000.power_limit_w, rtx5000.power_limit_w) == (140.0, 230.0)
    assert (a4000.ecc_enabled, rtx5000.ecc_enabled) == (False, True)
    assert (a4000.temperature_c, rtx5000.temperature_c) == (35, 29)
    assert (a4000.fan_pct, rtx5000.fan_pct) == (41, 33)
    assert (a4000.power_w, rtx5000.power_w) == (12.68, 12.23)
    assert (a4000.pstate, rtx5000.pstate) == ("P8", "P8")
    assert (a4000.sm_clock_mhz, rtx5000.sm_clock_mhz) == (210, 300)
    assert (a4000.bus_id, rtx5000.bus_id) == ("00000000:01:00.0", "00000000:02:00.0")
    # The static probes have not been merged at parse time.
    assert a4000.nvlink is None and a4000.temp_slowdown_c is None


def test_parse_live_csv_missing_sensors_stay_none_not_zero():
    # A fan-less card in a VM: no fan.speed, no power sensor, no ECC. Each
    # field degrades on its own and none of them becomes 0 — "0 W" is a
    # reading, "[N/A]" is the absence of one.
    out = (
        "0, GPU-a, NVIDIA RTX A4000, 16376, 100, 16276, 0, 8.6, "
        "00000000:01:00.0, 41, [N/A], [N/A], [N/A], [N/A], P0, 1560\n"
        "1, GPU-b, NVIDIA RTX A4000, 16376, 100, 16276, 0, 8.6, "
        "00000000:02:00.0, 40, 0, 0.00, 140.00, Disabled, P8, [Not Supported]\n"
    )
    vm_card, real_card = parse_nvidia_smi_live_csv(out)
    assert vm_card.fan_pct is None
    assert vm_card.power_w is None
    assert vm_card.power_limit_w is None
    assert vm_card.ecc_enabled is None
    assert vm_card.temperature_c == 41
    # ...whereas a genuine 0 % fan / 0 W reading is kept as 0.
    assert real_card.fan_pct == 0
    assert real_card.power_w == 0.0
    assert real_card.ecc_enabled is False
    assert real_card.sm_clock_mhz is None


def test_parse_live_csv_legacy_8_col_rows_keep_parsing():
    out = "0, GPU-a, NVIDIA RTX A4000, 16376, 100, 16276, 0, 8.6\n"
    (gpu,) = parse_nvidia_smi_live_csv(out)
    assert gpu.compute_cap == 8.6
    assert gpu.temperature_c is None and gpu.power_w is None and gpu.bus_id is None


def test_parse_live_csv_unrecognised_ecc_value_is_not_reported():
    out = (
        "0, GPU-a, NVIDIA RTX A4000, 16376, 100, 16276, 0, 8.6, "
        "00000000:01:00.0, 41, 30, 20.0, 140.00, Pending, P0, 1560\n"
    )
    (gpu,) = parse_nvidia_smi_live_csv(out)
    assert gpu.ecc_enabled is None


# ---------------------------------------------------------------------------
# nvidia-smi nvlink -s — three distinct states, never collapsed
# ---------------------------------------------------------------------------

NVLINK_THREE_STATES = """\
GPU 0: NVIDIA RTX A4000 (UUID: GPU-c70cc88f-417f-faf3-9238-292380156520)
Device does not have or support Nvlink
GPU 1: Quadro RTX 5000 (UUID: GPU-c01ce12b-7986-694b-9b2c-0da9306f5ded)
NVML: Unable to retrieve Nvlink information as all links are inActive
GPU 2: NVIDIA A100-SXM4-40GB (UUID: GPU-7fded661-d216-81c9-bd6d-4d3c0941a4a7)
\t Link 0: 25 GB/s
\t Link 1: 25 GB/s
\t Link 2: <inactive>
\t Link 3: 25 GB/s
"""


def test_parse_nvlink_three_states():
    status = parse_nvlink_status(NVLINK_THREE_STATES)
    assert status[0] == NvlinkStatus(state="unsupported")
    assert status[1] == NvlinkStatus(state="inactive")
    assert status[2] == NvlinkStatus(
        state="active",
        active_links=3,
        total_links=4,
        link_speed_gbs=25.0,
    )


def test_parse_nvlink_verdict_on_header_line():
    # Some builds print the verdict after the UUID on the same line.
    out = (
        "GPU 0: NVIDIA RTX A4000 (UUID: GPU-aaaa) Device does not have or support Nvlink\n"
        "GPU 1: Quadro RTX 5000 (UUID: GPU-bbbb) NVML: Unable to retrieve Nvlink "
        "information as all links are inActive\n"
    )
    status = parse_nvlink_status(out)
    assert status[0].state == "unsupported"
    assert status[1].state == "inactive"


def test_parse_nvlink_all_links_listed_inactive_is_inactive():
    out = "GPU 0: NVIDIA A100 (UUID: GPU-a)\n\t Link 0: <inactive>\n\t Link 1: <inactive>\n"
    status = parse_nvlink_status(out)
    assert status[0] == NvlinkStatus(state="inactive", active_links=0, total_links=2)


def test_parse_nvlink_header_with_nothing_after_it_is_not_reported():
    # A header the driver said nothing about must NOT be invented into one
    # of the three real states — the merge leaves nvlink=None.
    assert parse_nvlink_status("GPU 0: NVIDIA RTX A4000 (UUID: GPU-a)\n") == {}


def test_parse_nvlink_empty():
    assert parse_nvlink_status("") == {}


# ---------------------------------------------------------------------------
# nvidia-smi -q -d TEMPERATURE — thresholds keyed by bus id
# ---------------------------------------------------------------------------

TEMP_REPORT = """\

==============NVSMI LOG==============

Timestamp                                              : Sun Sep  6 08:21:38 2026
Driver Version                                         : 610.57.04 [Deprecated; will be removed in CUDA 14.0. Use KMD Version instead]
CUDA Version                                           : 13.3 [Deprecated; will be removed in CUDA 14.0. Use CUDA UMD Version instead]
KMD Version                                            : 610.57.04
CUDA UMD Version                                       : 13.3

Attached GPUs                                          : 2
GPU 00000000:01:00.0
    Temperature
        GPU Current Temp                               : 79 C
        GPU Current T.Limit Temp                       : N/A
        GPU Shutdown Temp                              : 103 C
        GPU Slowdown Temp                              : 100 C
        GPU Max Operating Temp                         : 98 C
        GPU Target Temperature Specification           : 90 C
        Memory Current Temp                            : N/A
        Memory Max Operating Temp                      : N/A

GPU 00000000:02:00.0
    Temperature
        GPU Current Temp                               : 29 C
        GPU Shutdown Temp                              : N/A
        GPU Slowdown Temp                              : N/A
        GPU Max Operating Temp                         : N/A

"""


def test_parse_temperature_limits_by_bus_id():
    limits = parse_temperature_limits(TEMP_REPORT)
    assert limits["00000000:01:00.0"] == TempLimits(
        slowdown_c=100,
        shutdown_c=103,
        max_operating_c=98,
    )
    # A card whose driver prints N/A for every threshold: present, all None.
    assert limits["00000000:02:00.0"] == TempLimits()


def test_parse_temperature_limits_empty():
    assert parse_temperature_limits("") == {}


# ---------------------------------------------------------------------------
# merge + the live snapshot probe
# ---------------------------------------------------------------------------


def test_merge_static_facts_joins_by_index_and_bus_id():
    gpus = parse_nvidia_smi_live_csv(LIVE_HETERO)
    facts = StaticGpuFacts(
        nvlink={0: NvlinkStatus(state="unsupported"), 1: NvlinkStatus(state="inactive")},
        temps={"00000000:02:00.0": TempLimits(slowdown_c=89, shutdown_c=94, max_operating_c=88)},
    )
    merge_static_facts(gpus, facts)
    assert gpus[0].nvlink == NvlinkStatus(state="unsupported")
    assert gpus[1].nvlink == NvlinkStatus(state="inactive")
    # Only card 1 had a thermal block; card 0 stays "not reported".
    assert gpus[0].temp_slowdown_c is None
    assert (gpus[1].temp_slowdown_c, gpus[1].temp_shutdown_c, gpus[1].temp_max_operating_c) == (
        89,
        94,
        88,
    )


async def test_query_gpu_snapshot_merges_static_probes(monkeypatch):
    from app.system import gpu as mod

    calls: list[list[str]] = []

    async def fake_run(args):
        calls.append(args)
        if args is mod.NVIDIA_SMI_LIVE_CMD:
            return LIVE_HETERO
        if args is mod.NVIDIA_SMI_APPS_CMD:
            return ""
        if args is mod.NVIDIA_SMI_NVLINK_CMD:
            return NVLINK_THREE_STATES
        if args is mod.NVIDIA_SMI_TEMP_CMD:
            return TEMP_REPORT
        if args is mod.NVIDIA_SMI_PCIE_CAPS_CMD:
            return "0, 4, 4\n1, 3, 4\n"
        if args is mod.NVIDIA_SMI_TOPO_CMD or args in mod.NVIDIA_SMI_THROTTLE_CMDS:
            return None  # covered in test_gpu_dials_parse.py
        raise AssertionError(args)

    monkeypatch.setattr(mod, "_run_nvidia_smi", fake_run)
    reset_static_facts_cache()
    try:
        snap = await query_gpu_snapshot()
        assert snap.probe_error is None
        assert snap.gpus[0].nvlink is not None and snap.gpus[0].nvlink.state == "unsupported"
        assert snap.gpus[1].nvlink is not None and snap.gpus[1].nvlink.state == "inactive"
        assert snap.gpus[0].temp_slowdown_c == 100
        assert snap.gpus[1].temp_slowdown_c is None

        # Second probe inside the TTL: live + apps run again, the static probes don't.
        calls.clear()
        await query_gpu_snapshot()
        assert mod.NVIDIA_SMI_LIVE_CMD in calls and mod.NVIDIA_SMI_APPS_CMD in calls
        assert mod.NVIDIA_SMI_NVLINK_CMD not in calls and mod.NVIDIA_SMI_TEMP_CMD not in calls
    finally:
        reset_static_facts_cache()


async def test_query_gpu_snapshot_static_probe_failure_is_not_reported(monkeypatch):
    # nvlink -s errors out (old driver) but the thermal report works: rows
    # carry nvlink=None (not "unsupported") and still get their limits.
    from app.system import gpu as mod

    async def fake_run(args):
        if args is mod.NVIDIA_SMI_LIVE_CMD:
            return LIVE_HETERO
        if args is mod.NVIDIA_SMI_TEMP_CMD:
            return TEMP_REPORT
        return None

    monkeypatch.setattr(mod, "_run_nvidia_smi", fake_run)
    reset_static_facts_cache()
    try:
        snap = await query_gpu_snapshot()
        assert snap.gpus[0].nvlink is None
        assert snap.gpus[0].temp_slowdown_c == 100
        assert snap.apps == []
    finally:
        reset_static_facts_cache()
