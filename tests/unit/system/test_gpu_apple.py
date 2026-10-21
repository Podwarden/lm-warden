"""Parsers for `ioreg -r -c AGXAccelerator -d 2` (Apple GPU telemetry, Metal
design §8). Pure text -> GpuLive/GpuInfo; no subprocess here (the runner is
p6f). A missing optional field degrades to None; the non-optional int
readings (utilization, used memory) degrade to 0; nothing raises."""


def _entry(perf: str, name: str = "Apple M1") -> str:
    # A realistic ioreg -r -c AGXAccelerator -d 2 excerpt: one AGXAccelerator
    # object, then its dictionary. ``perf`` is the inner PerformanceStatistics
    # block; ``name`` is the "model" property line (or omitted).
    lines = [
        "+-o AGXAcceleratorG13X  <class AGXAccelerator, id 0x100000000, retain 3>",
        "| {",
    ]
    if name is not None:
        lines.append(f'|   "model" = "{name}"')
    lines.append('|   "gpu-core-count" = 8')
    lines.append('|   "IOSourceVersion" = "341.5.2"')
    if perf:
        lines.append(f'|   "PerformanceStatistics" = {{{perf}}}')
    lines.append("| }")
    return "\n".join(lines)


def test_parse_ioreg_agx_utilization():
    # Red-first leg: utilization_pct from PerformanceStatistics
    # "Device Utilization %". Imported inside the test so the file still
    # collects on the base tree (module-level import of a not-yet-existing
    # name would ImportError the whole file and run nothing).
    from app.system.gpu_apple import parse_ioreg_agx

    out = _entry('"Device Utilization %"=37,"In use system memory"=1395589120')
    gpus = parse_ioreg_agx(out, budget_mib=11264)
    assert gpus[0].utilization_pct == 37


def test_parse_ioreg_agx_full_entry():
    from app.system.gpu_apple import parse_ioreg_agx

    # 1395589120 bytes // 2**20 = 1330.07... -> 1330 MiB (floor)
    out = _entry('"Device Utilization %"=37,"In use system memory"=1395589120')
    gpus = parse_ioreg_agx(out, budget_mib=11264)
    assert len(gpus) == 1
    g = gpus[0]
    assert g.index == 0
    assert g.uuid == "apple-gpu-0"
    assert g.name == "Apple M1"
    assert g.utilization_pct == 37
    assert g.memory_used_mib == 1330
    # total is the §7 budget passed IN, not hw.memsize
    assert g.memory_total_mib == 11264
    assert g.memory_free_mib == 11264 - 1330
    # Apple has no CUDA compute capability; the warning layer keys off None.
    assert g.compute_cap is None
    # Every other optional field stays None, not 0 / "" / False.
    assert g.bus_id is None
    assert g.temperature_c is None
    assert g.fan_pct is None
    assert g.power_w is None
    assert g.ecc_enabled is None
    assert g.nvlink is None
    assert g.throttle is None


def test_parse_ioreg_agx_utilization_bounds():
    from app.system.gpu_apple import parse_ioreg_agx

    def util(value: str) -> object:
        return parse_ioreg_agx(_entry(f'"Device Utilization %"={value}'), budget_mib=1)[
            0
        ].utilization_pct

    # utilization_pct is a non-optional int on GpuLive, so the only readings
    # it can hold are whole numbers; a reading inside [0, 100] is taken as-is.
    assert util("0") == 0
    assert util("100") == 100
    assert util("57") == 57
    # Outside [0, 100] is not a utilization reading: the field is a non-
    # optional int (cannot be None), so an out-of-range value degrades to the
    # "not reported" value 0 rather than being clamped or raised on.
    assert util("101") == 0
    assert util("-5") == 0


def test_parse_ioreg_agx_used_bytes_floor_and_free_clamp():
    from app.system.gpu_apple import parse_ioreg_agx

    # used bytes -> MiB is a floor: 1 MiB - 1 byte = 0 MiB.
    g = parse_ioreg_agx(_entry('"In use system memory"=1048575'), budget_mib=1024)[0]
    assert g.memory_used_mib == 0
    assert g.memory_free_mib == 1024

    # used > budget: free clamps at 0, never negative.
    g = parse_ioreg_agx(_entry('"In use system memory"=2147483648'), budget_mib=1024)[0]
    assert g.memory_used_mib == 2048
    assert g.memory_total_mib == 1024
    assert g.memory_free_mib == 0


def test_parse_ioreg_agx_missing_model_default_name():
    from app.system.gpu_apple import parse_ioreg_agx

    g = parse_ioreg_agx(_entry("", name=None), budget_mib=1024)[0]
    assert g.name == "Apple GPU"


def test_parse_ioreg_agx_missing_performance_statistics():
    from app.system.gpu_apple import parse_ioreg_agx

    # No PerformanceStatistics key at all: utilization/used are non-optional
    # ints on GpuLive, so 0 is used (never a None that would break the type);
    # total stays the budget, so free == budget.
    g = parse_ioreg_agx(_entry(""), budget_mib=1024)[0]
    assert g.utilization_pct == 0
    assert g.memory_used_mib == 0
    assert g.memory_total_mib == 1024
    assert g.memory_free_mib == 1024


def test_parse_ioreg_agx_two_entries_indexed():
    from app.system.gpu_apple import parse_ioreg_agx

    out = (
        _entry('"Device Utilization %"=10', name="Apple M1")
        + "\n"
        + _entry('"Device Utilization %"=20', name="Apple M2")
    )
    gpus = parse_ioreg_agx(out, budget_mib=11264)
    assert [g.index for g in gpus] == [0, 1]
    assert [g.uuid for g in gpus] == ["apple-gpu-0", "apple-gpu-1"]
    assert [g.name for g in gpus] == ["Apple M1", "Apple M2"]
    assert [g.utilization_pct for g in gpus] == [10, 20]


def test_parse_ioreg_agx_empty_and_garbage():
    from app.system.gpu_apple import parse_ioreg_agx

    assert parse_ioreg_agx("", budget_mib=1) == []
    assert parse_ioreg_agx("not an ioreg dump at all", budget_mib=1) == []
    # Text with braces and quotes but no AGXAccelerator object.
    assert parse_ioreg_agx('| { "foo" = 1 }\n', budget_mib=1) == []


def test_parse_ioreg_agx_info():
    from app.system.gpu_apple import parse_ioreg_agx_info

    out = _entry('"Device Utilization %"=37,"In use system memory"=1395589120')
    infos = parse_ioreg_agx_info(out, budget_mib=11264)
    assert len(infos) == 1
    i = infos[0]
    assert i.index == 0
    assert i.name == "Apple M1"
    assert i.memory_total_mib == 11264
    assert i.memory_used_mib == 1330
    assert i.utilization_pct == 37


def test_parse_agx_facts():
    from app.system.gpu_apple import parse_agx_facts

    out = _entry('"Device Utilization %"=37,"In use system memory"=1395589120')
    facts = parse_agx_facts(out)
    assert len(facts) == 1
    assert facts[0]["gpu_core_count"] == 8
    assert facts[0]["iosource_version"] == "341.5.2"
    assert facts[0]["model"] == "Apple M1"


def test_parse_agx_facts_missing_fields_none():
    from app.system.gpu_apple import parse_agx_facts

    # An entry with only the PerformanceStatistics block: gpu-core-count and
    # IOSourceVersion are absent -> None, never 0 / "".
    out = "\n".join(
        [
            "+-o AGXAcceleratorG13X  <class AGXAccelerator, id 0x100000000, retain 3>",
            "| {",
            '|   "PerformanceStatistics" = {"Device Utilization %"=37}',
            "| }",
        ]
    )
    facts = parse_agx_facts(out)
    assert len(facts) == 1
    assert facts[0]["gpu_core_count"] is None
    assert facts[0]["iosource_version"] is None
    assert facts[0]["model"] is None


# Captured verbatim (trimmed to a subset of properties, lines unchanged) from
# a real Apple M4 Mac on macOS 26.1. Note the class is the concrete
# ``AGXAcceleratorG16G`` subclass, the "(driver)" sibling that precedes the
# real "In use system memory" key, and the two tree-art shapes: ``  | {`` when
# the object has children (``-d 2``), bare ``    {`` when it has none
# (``-d 1``).
_REAL_PERF = (
    '{"In use system memory (driver)"=0,"Alloc system memory"=6339903488,'
    '"Tiler Utilization %"=27,"recoveryCount"=0,"lastRecoveryTime"=0,'
    '"Renderer Utilization %"=27,"TiledSceneBytes"=1474560,'
    '"Device Utilization %"=27,"SplitSceneCount"=0,"Allocated PB Size"=139460608,'
    '"In use system memory"=287178752}'
)
_REAL_PROPS = [
    '"SchedulerState" = {"Stamps"=(),"BusyWorkQueues"=()}',
    '"IOMatchedAtBoot" = Yes',
    '"vendor-id" = <6b100000>',
    '"AGCInfo" = {"fLastSubmissionPID"=1431,"fSubmissionsSinceLastCheck"=0,"fBusyCount"=0}',
    '"PerformanceStatistics" = ' + _REAL_PERF,
    '"IOSourceVersion" = "341.11"',
    '"IOPowerManagement" = {"CurrentPowerState"=1,"CapabilityFlags"=2,"MaxPowerState"=1,"DriverPowerState"=1}',
    '"model" = "Apple M4"',
    '"gpu-core-count" = 8',
    '"IONameMatch" = ("gpu,t8015","gpu,t8027","gpu,t8030","gpu,t8103","gpu,t8122","gpu,t8132")',
    '"IOClass" = "AGXAcceleratorG16G"',
    '"IOReportLegend" = ({"IOReportChannels"=((1,6442450945,"Alloc system memory"),'
    '(2,6442450945,"In use system memory"),(3,6442450945,"GPU Restart Count"))})',
    '"SurfaceList" = ()',
]
_REAL_HEADER = (
    "+-o AGXAcceleratorG16G  <class AGXAcceleratorG16G, id 0x1000003c4, registered,"
    " matched, active, busy 0 (24091 ms), retain 72>"
)
# ioreg -r -c AGXAccelerator -d 2 -w 0 (design s8's command): children follow.
REAL_IOREG_D2 = "\n".join(
    [_REAL_HEADER, "  | {"]
    + ["  |   " + p for p in _REAL_PROPS]
    + [
        "  | }",
        "  | ",
        "  +-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x100000ce7,"
        " !registered, !matched, active, busy 0, retain 5>",
        "",
    ]
)
# ioreg -r -d 1 -w 0 -c IOAccelerator: no children, so no tree art at all.
REAL_IOREG_D1 = "\n".join(
    [_REAL_HEADER, "    {"] + ["      " + p for p in _REAL_PROPS] + ["    }", "    ", ""]
)


def test_parse_ioreg_agx_real_capture_both_depths():
    from app.system.gpu_apple import parse_agx_facts, parse_ioreg_agx

    for text in (REAL_IOREG_D2, REAL_IOREG_D1):
        gpus = parse_ioreg_agx(text, budget_mib=11264)
        assert len(gpus) == 1
        g = gpus[0]
        assert g.name == "Apple M4"
        assert g.utilization_pct == 27
        # 287178752 // 2**20 = 273 (the "(driver)"=0 sibling must not win)
        assert g.memory_used_mib == 273
        assert g.memory_total_mib == 11264
        assert g.memory_free_mib == 11264 - 273
        facts = parse_agx_facts(text)
        assert facts == [
            {
                "object": "AGXAcceleratorG16G",
                "model": "Apple M4",
                "gpu_core_count": 8,
                "iosource_version": "341.11",
            }
        ]


def test_parse_ioreg_agx_header_without_dict_does_not_borrow_next():
    from app.system.gpu_apple import parse_ioreg_agx

    # A header whose dictionary is missing must not adopt the dictionary of
    # the object printed after it (that would report one GPU twice).
    text = _REAL_HEADER + "\n" + REAL_IOREG_D2
    gpus = parse_ioreg_agx(text, budget_mib=11264)
    assert [g.utilization_pct for g in gpus] == [27]
