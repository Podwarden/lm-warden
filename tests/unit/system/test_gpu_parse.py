from app.system.gpu import (
    GpuComputeApp,
    GpuInfo,
    GpuLive,
    parse_compute_apps_csv,
    parse_nvidia_smi_csv,
    parse_nvidia_smi_live_csv,
)


def test_parse_nvidia_smi_csv_4_gpus():
    out = (
        "0, NVIDIA A100-SXM4-40GB, 40960, 1024, 5\n"
        "1, NVIDIA A100-SXM4-40GB, 40960, 0, 0\n"
        "2, NVIDIA A100-SXM4-40GB, 40960, 0, 0\n"
        "3, NVIDIA A100-SXM4-40GB, 40960, 24576, 95\n"
    )
    gpus = parse_nvidia_smi_csv(out)
    assert len(gpus) == 4
    assert gpus[0] == GpuInfo(
        index=0,
        name="NVIDIA A100-SXM4-40GB",
        memory_total_mib=40960,
        memory_used_mib=1024,
        utilization_pct=5,
    )
    assert gpus[3].utilization_pct == 95


def test_parse_nvidia_smi_csv_handles_empty():
    assert parse_nvidia_smi_csv("") == []


def test_parse_nvidia_smi_csv_skips_malformed():
    out = "0, NVIDIA, 40960, 1024, 5\nbroken\n1, NVIDIA, 40960, 0, 0\n"
    gpus = parse_nvidia_smi_csv(out)
    assert [g.index for g in gpus] == [0, 1]


def test_parse_nvidia_smi_live_csv_two_gpus():
    # index, uuid, name, memory.total, memory.used, memory.free, utilization, compute_cap
    out = (
        "0, GPU-aaaa-1111, NVIDIA RTX A4000, 16376, 12450, 3926, 87, 8.6\n"
        "1, GPU-bbbb-2222, NVIDIA RTX A4000, 16376, 100, 16276, 0, 8.6\n"
    )
    gpus = parse_nvidia_smi_live_csv(out)
    assert len(gpus) == 2
    assert gpus[0] == GpuLive(
        index=0,
        uuid="GPU-aaaa-1111",
        name="NVIDIA RTX A4000",
        memory_total_mib=16376,
        memory_used_mib=12450,
        memory_free_mib=3926,
        utilization_pct=87,
        compute_cap=8.6,
    )
    assert gpus[1].memory_free_mib == 16276
    assert gpus[1].compute_cap == 8.6


def test_parse_nvidia_smi_live_csv_compute_cap_not_supported():
    # Drivers/cards that don't report compute_cap print "[Not Supported]" /
    # "[N/A]" — same degradation as power.draw. Parser keeps the row with
    # compute_cap=None rather than dropping or crashing.
    out = (
        "0, GPU-a, NVIDIA RTX A4000, 16376, 100, 16276, 0, [Not Supported]\n"
        "1, GPU-b, NVIDIA RTX A4000, 16376, 100, 16276, 0, [N/A]\n"
        "2, GPU-c, NVIDIA RTX A4000, 16376, 0, 16376, 0, 7.5\n"
    )
    gpus = parse_nvidia_smi_live_csv(out)
    assert [g.index for g in gpus] == [0, 1, 2]
    assert gpus[0].compute_cap is None
    assert gpus[1].compute_cap is None
    assert gpus[2].compute_cap == 7.5


def test_parse_nvidia_smi_live_csv_skips_malformed():
    # Second line has only 6 fields (too few) → dropped; third line has a
    # non-numeric memory.total → ValueError, also dropped. Parser still
    # returns the two well-formed rows. (7- and 8-col rows are both valid.)
    out = (
        "0, GPU-a, NVIDIA RTX A4000, 16376, 100, 16276, 0, 8.6\n"
        "1, GPU-b, NVIDIA RTX A4000, 16376, 100, 16276\n"
        "2, GPU-c, NVIDIA RTX A4000, notanumber, 0, 16376, 0, 8.6\n"
        "3, GPU-d, NVIDIA RTX A4000, 16376, 0, 16376, 0, 8.6\n"
    )
    gpus = parse_nvidia_smi_live_csv(out)
    assert [g.index for g in gpus] == [0, 3]


def test_parse_compute_apps_csv_basic():
    out = "12345, GPU-aaaa-1111, vllm-worker, 12400\n" "67890, GPU-bbbb-2222, python, 5500\n"
    apps = parse_compute_apps_csv(out)
    assert apps == [
        GpuComputeApp(
            pid=12345, gpu_uuid="GPU-aaaa-1111", process_name="vllm-worker", memory_mib=12400
        ),
        GpuComputeApp(pid=67890, gpu_uuid="GPU-bbbb-2222", process_name="python", memory_mib=5500),
    ]


def test_parse_compute_apps_csv_skips_non_numeric_memory():
    # nvidia-smi prints "[Not Supported]" on some MIG / consumer cards for
    # used_memory. We must skip those rows, not crash.
    out = "12345, GPU-a, vllm-worker, [Not Supported]\n" "67890, GPU-b, python, 5500\n"
    apps = parse_compute_apps_csv(out)
    assert [a.pid for a in apps] == [67890]


def test_parse_compute_apps_csv_handles_empty():
    assert parse_compute_apps_csv("") == []


def test_parse_nvidia_smi_csv_4_gpus_unchanged_includes_name():
    # Smoke-checks the legacy 5-col parser still captures `name` — that's the
    # field flowing into the new `gpu_samples.name` column.
    out = "0, NVIDIA RTX A4000, 16376, 1024, 5\n"
    gpus = parse_nvidia_smi_csv(out)
    assert gpus == [
        GpuInfo(
            index=0,
            name="NVIDIA RTX A4000",
            memory_total_mib=16376,
            memory_used_mib=1024,
            utilization_pct=5,
        )
    ]
