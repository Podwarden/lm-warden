"""PCIe link fields on the live rows + the endpoint-capability probe.

Recorded from a real host, idle: both cards sit at Gen 1 (PCIe power
saving — NOT a fault) and GPU 0 negotiated x4 of a possible x16 (a slot
or riser limit — a fault worth surfacing). The `gpumax`/`hostmax` pair is
a newer field family; a driver that lacks it fails only that probe.
"""

from __future__ import annotations

from app.system.gpu import (
    PcieCaps,
    StaticGpuFacts,
    merge_static_facts,
    parse_nvidia_smi_live_csv,
    parse_pcie_caps_csv,
    query_gpu_snapshot,
    reset_static_facts_cache,
)

LIVE_PCIE = (
    "0, GPU-aaaa, NVIDIA RTX A4000, 16376, 14137, 2239, 0, 8.6, "
    "00000000:01:00.0, 35, 41, 12.68, 140.00, Disabled, P8, 210, 1, 3, 4, 16\n"
    "1, GPU-bbbb, Quadro RTX 5000, 15360, 13818, 1542, 0, 7.5, "
    "00000000:02:00.0, 29, 33, 12.23, 230.00, Enabled, P8, 300, 1, 3, 16, 16\n"
)

PCIE_CAPS = "0, 4, 4\n1, 3, 4\n"


def test_parse_live_csv_pcie_columns():
    a4000, rtx5000 = parse_nvidia_smi_live_csv(LIVE_PCIE)
    # Idle downshift: gen 1 now, negotiated max 3 — both parsed verbatim,
    # the fault/no-fault judgement belongs to the UI, not the parser.
    assert (a4000.pcie_gen_current, a4000.pcie_gen_max) == (1, 3)
    assert (a4000.pcie_width_current, a4000.pcie_width_max) == (4, 16)
    assert (rtx5000.pcie_width_current, rtx5000.pcie_width_max) == (16, 16)
    # Capabilities are not on the live row.
    assert a4000.pcie_gen_gpu_max is None and a4000.pcie_gen_host_max is None


def test_parse_live_csv_16_col_rows_have_no_pcie():
    out = (
        "0, GPU-a, NVIDIA RTX A4000, 16376, 100, 16276, 0, 8.6, "
        "00000000:01:00.0, 41, 30, 20.0, 140.00, Disabled, P0, 1560\n"
    )
    (gpu,) = parse_nvidia_smi_live_csv(out)
    assert gpu.temperature_c == 41  # telemetry still parsed
    assert gpu.pcie_gen_current is None and gpu.pcie_width_max is None


def test_parse_live_csv_pcie_na_stays_none_not_zero():
    out = (
        "0, GPU-a, NVIDIA RTX A4000, 16376, 100, 16276, 0, 8.6, "
        "00000000:01:00.0, 41, 30, 20.0, 140.00, Disabled, P0, 1560, [N/A], [N/A], [N/A], [N/A]\n"
    )
    (gpu,) = parse_nvidia_smi_live_csv(out)
    assert gpu.pcie_gen_current is None
    assert gpu.pcie_gen_max is None
    assert gpu.pcie_width_current is None
    assert gpu.pcie_width_max is None


def test_parse_pcie_caps():
    caps = parse_pcie_caps_csv(PCIE_CAPS)
    assert caps == {
        0: PcieCaps(gpu_max_gen=4, host_max_gen=4),
        1: PcieCaps(gpu_max_gen=3, host_max_gen=4),
    }


def test_parse_pcie_caps_na_and_malformed():
    caps = parse_pcie_caps_csv("0, [N/A], 4\nbroken\n1, 3, [Not Supported]\n")
    assert caps == {
        0: PcieCaps(gpu_max_gen=None, host_max_gen=4),
        1: PcieCaps(gpu_max_gen=3, host_max_gen=None),
    }


def test_merge_attaches_pcie_caps_by_index():
    gpus = parse_nvidia_smi_live_csv(LIVE_PCIE)
    merge_static_facts(gpus, StaticGpuFacts(pcie_caps=parse_pcie_caps_csv(PCIE_CAPS)))
    assert (gpus[0].pcie_gen_gpu_max, gpus[0].pcie_gen_host_max) == (4, 4)
    assert (gpus[1].pcie_gen_gpu_max, gpus[1].pcie_gen_host_max) == (3, 4)
    # The negotiated max is untouched by the capabilities — it stays what
    # the link reported (3), not the flattering endpoint figure (4).
    assert gpus[0].pcie_gen_max == 3


async def test_driver_without_gpumax_family_degrades_to_not_reported(monkeypatch):
    """An old driver rejects the gpumax/hostmax query outright. Only that
    probe fails: the live row keeps its four classic PCIe fields and the
    capability pair is None — never 0."""
    from app.system import gpu as mod

    async def fake_run(args):
        if args is mod.NVIDIA_SMI_LIVE_CMD:
            return LIVE_PCIE
        if args is mod.NVIDIA_SMI_PCIE_CAPS_CMD:
            return None  # "Field ... is not a valid field to query" -> exit 1
        if args is mod.NVIDIA_SMI_APPS_CMD:
            return ""
        return None

    monkeypatch.setattr(mod, "_run_nvidia_smi", fake_run)
    reset_static_facts_cache()
    try:
        snap = await query_gpu_snapshot()
        assert snap.probe_error is None
        g0 = snap.gpus[0]
        assert (g0.pcie_gen_current, g0.pcie_gen_max) == (1, 3)
        assert (g0.pcie_width_current, g0.pcie_width_max) == (4, 16)
        assert g0.pcie_gen_gpu_max is None and g0.pcie_gen_host_max is None
    finally:
        reset_static_facts_cache()
