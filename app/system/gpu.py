import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

NVIDIA_SMI_CMD = [
    "nvidia-smi",
    # power.draw is the 6th field — added in S7 (#124) so the stats sampler
    # gets util/mem/power in a SINGLE nvidia-smi acquisition per tick
    # (CTO decision #6: no second pass). Driver versions that don't support
    # power.draw on the queried card report "[Not Supported]"/"[N/A]" — the
    # parser turns those into ``power_w=None`` and the sampler logs+skips
    # the power write rather than corrupting the bucket.
    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw",
    "--format=csv,noheader,nounits",
]

# Pull memory.free too for the live-probe endpoint (`/api/system/gpus`).
# Older `query_gpus()` shape stays the same so the existing DB-backed
# sampler does not change behaviour.
NVIDIA_SMI_LIVE_CMD = [
    "nvidia-smi",
    # compute_cap is the 8th field — added in #176 so fit-preview can cross
    # the detected GPU's CUDA compute capability (e.g. 8.6 for an A4000) with
    # the candidate's quant/dtype and warn on emulated/unsupported combos.
    # Drivers/cards that don't report it print "[Not Supported]"/"[N/A]"; the
    # parser turns those into ``compute_cap=None`` and keeps the row (same
    # degradation pattern as power.draw above).
    #
    # Columns 9-16 are the per-card telemetry the /stats GPU cards show:
    # bus id (the join key for the ``-q -d TEMPERATURE`` limits), temperature,
    # fan, power draw + the card's own limit, ECC mode, P-state and SM clock.
    # Every one of them may be "[N/A]" on a given card — a passively cooled
    # card has no fan.speed, a card in a VM often has no power sensor — and
    # each degrades to ``None`` on its own. ``None`` renders as "not reported";
    # it is never folded into 0, because "0 W" and "no power sensor" are
    # different facts about the hardware.
    #
    # Columns 17-20 are the PCIe link: generation now / negotiated max, width
    # now / max. Two traps, both seen on a real host. (a) gen.current is 1 on
    # an idle card — PCIe downshifts to save power and climbs under load, so
    # a low CURRENT generation is normal and must not be flagged. (b) Width
    # does not downshift: x4 against a max of x16 is a card in a x4 slot or
    # on a riser, a permanent ceiling, and on a host with no usable NVLink
    # it is the only path between cards — that one IS worth a warning.
    # These four fields are decades old; the newer gpumax/hostmax capability
    # pair lives in NVIDIA_SMI_PCIE_CAPS_CMD because an unknown field name
    # makes nvidia-smi reject the whole query.
    #
    # Columns 21-23 are the clock ceilings and the memory clock: clocks.max.sm,
    # clocks.mem, clocks.max.mem. A clock is only readable against its own
    # maximum — 210 MHz idle is normal, 1200 MHz of 2100 at full load is a
    # card being held back. These three are as old as clocks.sm itself
    # (nvidia-smi has listed them since the Kepler era), so they ride in the
    # main query; the throttle-reason family does NOT, see
    # NVIDIA_SMI_THROTTLE_CMDS.
    "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_cap,"
    "pci.bus_id,temperature.gpu,fan.speed,power.draw,power.limit,ecc.mode.current,pstate,clocks.sm,"
    "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,"
    "clocks.max.sm,clocks.mem,clocks.max.mem",
    "--format=csv,noheader,nounits",
]

# Column counts of NVIDIA_SMI_LIVE_CMD over time. The parser accepts every
# shape so fixtures recorded before a column was added still parse — the
# fields a row lacks simply stay None.
_LIVE_COLS_TELEMETRY = 16
_LIVE_COLS_PCIE = 20
_LIVE_COLS_FULL = 23
_LIVE_COL_COUNTS = (7, 8, _LIVE_COLS_TELEMETRY, _LIVE_COLS_PCIE, _LIVE_COLS_FULL)

# Why the clocks are throttled, from the driver rather than guessed from a
# temperature ratio. ``clocks_throttle_reasons.active`` is the NVML bitmask;
# the three named columns are the ones an operator acts on differently
# (thermal / power cap / hardware slowdown are different problems).
#
# Queried SEPARATELY from NVIDIA_SMI_LIVE_CMD because an unknown field name
# makes nvidia-smi reject the whole query: R535 renamed this family to
# ``clocks_event_reasons.*`` and kept the old spelling as an alias, and the
# alias is the one verified on the 610-era drivers this ships against. A
# driver that drops the alias fails only THIS probe, and the next spelling
# is tried; a driver that knows neither leaves the verdict "not reported"
# and takes nothing else down with it.
NVIDIA_SMI_THROTTLE_CMDS: tuple[list[str], ...] = (
    [
        "nvidia-smi",
        "--query-gpu=index,clocks_throttle_reasons.active,"
        "clocks_throttle_reasons.sw_thermal_slowdown,"
        "clocks_throttle_reasons.hw_thermal_slowdown,"
        "clocks_throttle_reasons.sw_power_cap",
        "--format=csv,noheader",
    ],
    [
        "nvidia-smi",
        "--query-gpu=index,clocks_event_reasons.active,"
        "clocks_event_reasons.sw_thermal_slowdown,"
        "clocks_event_reasons.hw_thermal_slowdown,"
        "clocks_event_reasons.sw_power_cap",
        "--format=csv,noheader",
    ],
)

# NVML's nvmlClocksThrottleReasons bit values, as printed in hex by the
# ``.active`` column. The mask carries reasons the named columns do not
# (a hardware slowdown that is not thermal, a power brake), so both are read.
THROTTLE_BIT_GPU_IDLE = 0x1
THROTTLE_BIT_APPLICATIONS_CLOCKS = 0x2
THROTTLE_BIT_SW_POWER_CAP = 0x4
THROTTLE_BIT_HW_SLOWDOWN = 0x8
THROTTLE_BIT_SYNC_BOOST = 0x10
THROTTLE_BIT_SW_THERMAL = 0x20
THROTTLE_BIT_HW_THERMAL = 0x40
THROTTLE_BIT_HW_POWER_BRAKE = 0x80

# Which card can reach which, and over what. ``nvidia-smi topo -m`` is its
# own sub-command printing a matrix keyed by GPU index; static for the life
# of the box, so it lives in the 60 s facts cache.
NVIDIA_SMI_TOPO_CMD = ["nvidia-smi", "topo", "-m"]

# What each PCIe endpoint is CAPABLE of, as opposed to what the link
# negotiated (pcie.link.gen.max above). On a real host these disagree: an
# RTX A4000 (Gen 4-capable) in a Gen 4-capable host negotiated a Gen 3 max,
# so the slot or riser is the limit. Newer field names (R5xx+); a driver
# that lacks them fails THIS probe only and the pair stays "not reported".
NVIDIA_SMI_PCIE_CAPS_CMD = [
    "nvidia-smi",
    "--query-gpu=index,pcie.link.gen.gpumax,pcie.link.gen.hostmax",
    "--format=csv,noheader,nounits",
]

# NVLink is not a --query-gpu field; ``nvidia-smi nvlink -s`` is its own
# sub-command with its own free-text format. Three DISTINCT outcomes:
#
#   GPU 0: NVIDIA RTX A4000 (UUID: ...)
#   Device does not have or support Nvlink                 -> "unsupported"
#   GPU 1: Quadro RTX 5000 (UUID: ...)
#   NVML: Unable to retrieve Nvlink information as all
#         links are inActive                               -> "inactive"
#   GPU 2: NVIDIA A100-SXM4-40GB (UUID: ...)
#            Link 0: 25 GB/s
#            Link 1: <inactive>                            -> "active" (1 of 2)
#
# The distinction matters to the operator: "this card cannot NVLink" and
# "this card could but nothing is bridged" call for different actions.
NVIDIA_SMI_NVLINK_CMD = ["nvidia-smi", "nvlink", "-s"]

# Thermal thresholds are only available from the ``-q`` report, keyed by PCI
# bus id rather than index. Static per card, so they're cached (see
# ``_StaticFactsCache``) rather than re-read on every 2 s probe.
NVIDIA_SMI_TEMP_CMD = ["nvidia-smi", "-q", "-d", "TEMPERATURE"]

NVIDIA_SMI_APPS_CMD = [
    "nvidia-smi",
    "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
    "--format=csv,noheader,nounits",
]


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_pct: int
    # S7 (#124) — current power draw in watts. ``None`` when the driver
    # reports "[Not Supported]" / "[N/A]" or the field can't be parsed.
    # New rows on cards that DO report power get a real float; the sampler
    # only writes to ``power_samples`` when this field is non-None.
    power_w: float | None = None


@dataclass
class GpuLive:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_pct: int
    # #176 — CUDA compute capability (e.g. 8.6 for sm_86). ``None`` when the
    # driver reports "[Not Supported]" / "[N/A]" or the field can't be parsed;
    # the capability-warning layer treats None as "can't classify" and emits
    # no warnings rather than fabricating one.
    compute_cap: float | None = None
    # ---- per-card telemetry (columns 9-16 of NVIDIA_SMI_LIVE_CMD) ----------
    # Each is None when the card/driver reports "[N/A]" / "[Not Supported]"
    # for that field, or the row predates the column. None means "not
    # reported" and is rendered as such; it is never a stand-in for 0.
    bus_id: str | None = None
    temperature_c: int | None = None
    fan_pct: int | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    # True/False when the driver reports "Enabled"/"Disabled"; None when the
    # card has no ECC at all ("[N/A]"). Consumer cards are the None case.
    ecc_enabled: bool | None = None
    pstate: str | None = None
    sm_clock_mhz: int | None = None
    # ---- clock ceilings + memory clock (columns 21-23) ---------------------
    # Each clock is shown against its own maximum. A missing maximum leaves
    # the reading standing alone; it is never invented from the other card.
    sm_clock_max_mhz: int | None = None
    mem_clock_mhz: int | None = None
    mem_clock_max_mhz: int | None = None
    # ---- PCIe link (columns 17-20) ------------------------------------------
    # gen_current sits at 1 on an idle card and is NOT a fault; width_current
    # below width_max is a permanent slot/riser limit and is. gen_max is the
    # NEGOTIATED ceiling — what the link will actually do.
    pcie_gen_current: int | None = None
    pcie_gen_max: int | None = None
    pcie_width_current: int | None = None
    pcie_width_max: int | None = None
    # ---- static facts merged from the separate probes ----------------------
    # Endpoint capabilities from NVIDIA_SMI_PCIE_CAPS_CMD. Shown beside
    # pcie_gen_max only when they exceed it — that is the honest reading of
    # "both ends can do Gen 4 but the link settled at Gen 3".
    pcie_gen_gpu_max: int | None = None
    pcie_gen_host_max: int | None = None
    # From ``nvidia-smi nvlink -s`` (see NvlinkStatus). None when that probe
    # failed or printed nothing for this index — distinct from "unsupported".
    nvlink: "NvlinkStatus | None" = None
    # From ``nvidia-smi -q -d TEMPERATURE``: the driver's own thresholds for
    # this card. ``temp_slowdown_c`` is the clock-throttle point — the limit
    # the temperature reading is shown against.
    temp_slowdown_c: int | None = None
    temp_shutdown_c: int | None = None
    temp_max_operating_c: int | None = None
    # From NVIDIA_SMI_THROTTLE_CMDS, merged by index on every probe (it is
    # live, not static). None when the probe gave nothing for this card —
    # distinct from "nothing is throttling", which is a ThrottleReasons with
    # every flag False.
    throttle: "ThrottleReasons | None" = None


@dataclass
class ThrottleReasons:
    """Why this card's clocks are where they are, as the driver states it.

    Each flag is True/False when known and None when the driver printed
    "[N/A]" for the column AND the bitmask could not fill it in. ``mask`` is
    the raw ``.active`` value. The named flags are what the panel words —
    thermal, power cap and hardware slowdown are different problems with
    different fixes, so they are never collapsed into one "throttled".
    """

    mask: int | None = None
    sw_thermal: bool | None = None
    hw_thermal: bool | None = None
    sw_power_cap: bool | None = None
    hw_slowdown: bool | None = None
    hw_power_brake: bool | None = None
    idle: bool | None = None


@dataclass
class GpuTopology:
    """The ``nvidia-smi topo -m`` matrix over the GPUs it listed.

    ``indices[i]`` is the GPU number of row/column ``i``; ``matrix[i][j]`` is
    the legend code between them: ``X`` self, ``NV#`` a bond of # NVLinks,
    and PIX / PXB / PHB / NODE / SYS all meaning *no direct link* — the pair
    is routed through PCIe at increasing distance. NIC rows and columns are
    dropped; only GPU-to-GPU cells are kept.
    """

    indices: list[int] = field(default_factory=list)
    matrix: list[list[str]] = field(default_factory=list)


NvlinkState = Literal["unsupported", "inactive", "active"]


@dataclass
class NvlinkStatus:
    """One card's NVLink standing, from ``nvidia-smi nvlink -s``.

    ``unsupported`` — the silicon has no NVLink (RTX A4000, consumer cards).
    ``inactive``    — the card supports it but every link is down (a lone
                      Quadro RTX 5000 with no bridge fitted).
    ``active``      — at least one link is up; ``active_links`` of
                      ``total_links`` carry traffic at ``link_speed_gbs``.
    """

    state: NvlinkState
    active_links: int = 0
    total_links: int = 0
    link_speed_gbs: float | None = None


@dataclass
class TempLimits:
    """Thermal thresholds for one card, keyed by PCI bus id in the -q report."""

    slowdown_c: int | None = None
    shutdown_c: int | None = None
    max_operating_c: int | None = None


@dataclass
class PcieCaps:
    """What the card and the host are each capable of (PCIe generation)."""

    gpu_max_gen: int | None = None
    host_max_gen: int | None = None


@dataclass
class GpuComputeApp:
    pid: int
    gpu_uuid: str
    process_name: str
    memory_mib: int


@dataclass
class GpuSnapshot:
    """Result of a live probe of nvidia-smi: per-GPU stats + compute holders."""

    gpus: list[GpuLive] = field(default_factory=list)
    apps: list[GpuComputeApp] = field(default_factory=list)
    probe_error: str | None = None
    # A relationship BETWEEN cards, so it sits beside the list rather than on
    # each row. None when ``topo -m`` gave nothing.
    topology: GpuTopology | None = None


# CUDA compute capability -> GPU generation. nvidia-smi has no architecture
# field (``--help-query-gpu`` lists none); ``compute_cap`` is the only
# queryable source, and the mapping is a fixed fact of the silicon. Keyed
# exactly: an unrecognised capability returns None and the panel shows the
# raw number and says the generation is not recognised, because the next
# generation's number is not public and inventing an entry would produce a
# confidently wrong label the day one ships.
_ARCH_BY_MAJOR: dict[int, str] = {5: "Maxwell", 6: "Pascal", 10: "Blackwell"}
_ARCH_BY_CAP: dict[tuple[int, int], str] = {
    (7, 0): "Volta",
    (7, 2): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 7): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
    (12, 0): "Blackwell",
}


def architecture_for(compute_cap: float | None) -> str | None:
    """GPU generation for a compute capability, or None when unrecognised."""
    if compute_cap is None or compute_cap < 0:
        return None
    major = int(compute_cap)
    minor = round((compute_cap - major) * 10)
    if (major, minor) in _ARCH_BY_CAP:
        return _ARCH_BY_CAP[(major, minor)]
    return _ARCH_BY_MAJOR.get(major)


def _parse_optional_float(raw: str) -> float | None:
    """Parse a numeric nvidia-smi field that may report "[Not Supported]" /
    "[N/A]" on cards or drivers without the capability. Returns the float
    on success, None on the documented sentinels, and None (with a debug
    log) on any other non-numeric value so a malformed row never crashes
    the sampler.
    """
    s = raw.strip()
    if not s or s.startswith("[") or s.upper() in {"N/A", "NOT SUPPORTED"}:
        return None
    try:
        return float(s)
    except ValueError:
        logger.debug("nvidia-smi non-numeric optional float skipped: %r", raw)
        return None


def parse_nvidia_smi_csv(stdout: str) -> list[GpuInfo]:
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        # S7 (#124) — accept either 5-col (pre-power) or 6-col (with power.draw)
        # rows. Older nvidia-smi builds in dev fixtures sometimes emit the
        # 5-col form; the runtime image's nvidia-smi reliably emits 6.
        if len(parts) not in (5, 6):
            continue
        try:
            power_w = _parse_optional_float(parts[5]) if len(parts) == 6 else None
            out.append(GpuInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mib=int(parts[2]),
                memory_used_mib=int(parts[3]),
                utilization_pct=int(parts[4]),
                power_w=power_w,
            ))
        except ValueError:
            logger.warning("nvidia-smi malformed row skipped: %r", line)
            continue
    return out


def _parse_optional_int(raw: str) -> int | None:
    """Integer flavour of ``_parse_optional_float`` — same sentinels, same
    "None rather than crash" contract. nvidia-smi prints whole numbers for
    temperature / fan / clocks but we go through float so "80.00" parses."""
    f = _parse_optional_float(raw)
    return None if f is None else int(f)


def _parse_optional_str(raw: str) -> str | None:
    """Free-text field ("P2", a bus id) that may still be an N/A sentinel."""
    s = raw.strip()
    if not s or s.startswith("[") or s.upper() in {"N/A", "NOT SUPPORTED"}:
        return None
    return s


def _parse_ecc_mode(raw: str) -> bool | None:
    """``ecc.mode.current`` prints "Enabled" / "Disabled", or "[N/A]" on a
    card that has no ECC memory at all. Three outcomes, three values: the
    None case is a card that cannot do ECC, not one that has it off."""
    s = _parse_optional_str(raw)
    if s is None:
        return None
    low = s.lower()
    if low == "enabled":
        return True
    if low == "disabled":
        return False
    logger.debug("nvidia-smi unrecognised ecc.mode.current skipped: %r", raw)
    return None


def parse_nvidia_smi_live_csv(stdout: str) -> list[GpuLive]:
    """Parser for NVIDIA_SMI_LIVE_CMD (16 cols; see the command for the list).

    Accepts the two older row shapes too — 7 cols (pre-compute_cap) and 8
    cols (pre-telemetry) — so fixtures recorded against older builds still
    parse; the fields those rows lack stay None. Every optional field
    degrades to None on "[Not Supported]"/"[N/A]"/unparseable on its own,
    without dropping the row: a card with no fan still has a temperature.
    """
    out: list[GpuLive] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) not in _LIVE_COL_COUNTS:
            continue
        try:
            compute_cap = _parse_optional_float(parts[7]) if len(parts) >= 8 else None
            gpu = GpuLive(
                index=int(parts[0]),
                uuid=parts[1],
                name=parts[2],
                memory_total_mib=int(parts[3]),
                memory_used_mib=int(parts[4]),
                memory_free_mib=int(parts[5]),
                utilization_pct=int(parts[6]),
                compute_cap=compute_cap,
            )
        except ValueError:
            logger.warning("nvidia-smi (live) malformed row skipped: %r", line)
            continue
        if len(parts) >= _LIVE_COLS_TELEMETRY:
            gpu.bus_id = _parse_optional_str(parts[8])
            gpu.temperature_c = _parse_optional_int(parts[9])
            gpu.fan_pct = _parse_optional_int(parts[10])
            gpu.power_w = _parse_optional_float(parts[11])
            gpu.power_limit_w = _parse_optional_float(parts[12])
            gpu.ecc_enabled = _parse_ecc_mode(parts[13])
            gpu.pstate = _parse_optional_str(parts[14])
            gpu.sm_clock_mhz = _parse_optional_int(parts[15])
        if len(parts) >= _LIVE_COLS_PCIE:
            gpu.pcie_gen_current = _parse_optional_int(parts[16])
            gpu.pcie_gen_max = _parse_optional_int(parts[17])
            gpu.pcie_width_current = _parse_optional_int(parts[18])
            gpu.pcie_width_max = _parse_optional_int(parts[19])
        if len(parts) >= _LIVE_COLS_FULL:
            gpu.sm_clock_max_mhz = _parse_optional_int(parts[20])
            gpu.mem_clock_mhz = _parse_optional_int(parts[21])
            gpu.mem_clock_max_mhz = _parse_optional_int(parts[22])
        out.append(gpu)
    return out


def _parse_active_flag(raw: str) -> bool | None:
    """"Active" / "Not Active" -> True / False; "[N/A]" -> None."""
    s = _parse_optional_str(raw)
    if s is None:
        return None
    low = s.lower()
    if low == "active":
        return True
    if low == "not active":
        return False
    logger.debug("nvidia-smi unrecognised throttle flag skipped: %r", raw)
    return None


def _parse_hex_mask(raw: str) -> int | None:
    """"0x0000000000000020" -> 32; "[N/A]" -> None."""
    s = _parse_optional_str(raw)
    if s is None:
        return None
    try:
        return int(s, 16)
    except ValueError:
        logger.debug("nvidia-smi non-hex throttle mask skipped: %r", raw)
        return None


def parse_throttle_reasons_csv(stdout: str) -> dict[int, ThrottleReasons]:
    """Parser for NVIDIA_SMI_THROTTLE_CMDS: {index: ThrottleReasons}.

    Columns: index, active (hex bitmask), sw_thermal_slowdown,
    hw_thermal_slowdown, sw_power_cap — the last three print "Active" /
    "Not Active". The named columns win; where one is "[N/A]" the bitmask
    fills it in, and the bitmask alone supplies the reasons that have no
    named column here (generic hardware slowdown, power brake, idle).
    """
    out: dict[int, ThrottleReasons] = {}
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != 5:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            logger.debug("nvidia-smi throttle malformed row skipped: %r", line)
            continue
        out[index] = _throttle_reasons_from_row(
            _parse_hex_mask(parts[1]), parts[2], parts[3], parts[4],
        )
    return out


def _throttle_reasons_from_row(
    mask: int | None, sw_thermal_raw: str, hw_thermal_raw: str, sw_power_cap_raw: str,
) -> ThrottleReasons:
    def from_mask(bit: int) -> bool | None:
        return None if mask is None else bool(mask & bit)

    def named(raw: str, bit: int) -> bool | None:
        flag = _parse_active_flag(raw)
        return flag if flag is not None else from_mask(bit)

    return ThrottleReasons(
        mask=mask,
        sw_thermal=named(sw_thermal_raw, THROTTLE_BIT_SW_THERMAL),
        hw_thermal=named(hw_thermal_raw, THROTTLE_BIT_HW_THERMAL),
        sw_power_cap=named(sw_power_cap_raw, THROTTLE_BIT_SW_POWER_CAP),
        hw_slowdown=from_mask(THROTTLE_BIT_HW_SLOWDOWN),
        hw_power_brake=from_mask(THROTTLE_BIT_HW_POWER_BRAKE),
        idle=from_mask(THROTTLE_BIT_GPU_IDLE),
    )


_TOPO_GPU_RE = re.compile(r"^GPU(\d+)$")


def _split_topo_line(line: str) -> list[str]:
    """Cells of one ``topo -m`` line. The report is tab-separated (a self
    cell prints as " X " with padding spaces, and "CPU Affinity" holds a
    space), so tabs are the delimiter when present; a build that prints
    spaces still parses because the GPU columns come first and single
    tokens are all the matrix needs."""
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    return line.split()


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def parse_topology_matrix(stdout: str) -> GpuTopology | None:
    """Parse ``nvidia-smi topo -m`` into the GPU-to-GPU legend matrix.

    Real output from a 4-card host::

                GPU0  GPU1  GPU2  GPU3  CPU Affinity  NUMA Affinity  GPU NUMA ID
        GPU0     X    PHB   PHB   PHB   0-27          0              N/A
        GPU1    PHB    X    PHB   PHB   0-27          0              N/A
        ...

        Legend: ...

    The header names the columns; only ``GPU<n>`` ones are kept, so NIC
    columns (``NIC0``, ``mlx5_0``) and the affinity columns fall away, and
    only ``GPU<n>`` rows are read. Returns None when no GPU header is found
    or a GPU row is missing — the panel then says the interconnect is not
    reported rather than drawing a partial picture.
    """
    # nvidia-smi UNDERLINES the header row, so the first cell arrives as
    # "\x1b[4mGPU0" and a bare GPU<n> match silently drops card 0 — the panel
    # then draws n-1 cards and its legend counts one card short. Seen on a real
    # 4-card host, which rendered as three. Strip the escapes before parsing.
    stdout = _ANSI_ESCAPE.sub("", stdout)
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    header_cells: list[str] | None = None
    header_at = 0
    for i, line in enumerate(lines):
        cells = _split_topo_line(line)
        if cells and cells[0] == "":
            cells = cells[1:]
        if any(_TOPO_GPU_RE.match(c) for c in cells):
            header_cells = cells
            header_at = i
            break
    if header_cells is None:
        return None

    gpu_cols: list[tuple[int, int]] = []  # (column position, gpu index)
    for pos, cell in enumerate(header_cells):
        m = _TOPO_GPU_RE.match(cell)
        if m is not None:
            gpu_cols.append((pos, int(m.group(1))))
    indices = [idx for _, idx in gpu_cols]

    rows: dict[int, list[str]] = {}
    for line in lines[header_at + 1:]:
        cells = _split_topo_line(line)
        if not cells:
            continue
        if line.strip().startswith("Legend"):
            break
        m = _TOPO_GPU_RE.match(cells[0])
        if m is None:
            continue
        values = cells[1:]
        if any(pos >= len(values) for pos, _ in gpu_cols):
            logger.debug("nvidia-smi topo row too short, skipped: %r", line)
            continue
        rows[int(m.group(1))] = [values[pos] for pos, _ in gpu_cols]

    if any(idx not in rows for idx in indices):
        logger.debug("nvidia-smi topo: header lists %s, rows cover %s", indices, sorted(rows))
        return None
    matrix = [rows[idx] for idx in indices]
    for i in range(len(indices)):
        # The self cell is padded " X " in the tab layout; normalise it.
        if matrix[i][i].strip() == "X":
            matrix[i][i] = "X"
    return GpuTopology(indices=indices, matrix=matrix)


def parse_pcie_caps_csv(stdout: str) -> dict[int, PcieCaps]:
    """Parser for NVIDIA_SMI_PCIE_CAPS_CMD: {index: PcieCaps}. Either
    capability may be "[N/A]" and stays None; a malformed row is skipped."""
    out: dict[int, PcieCaps] = {}
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            logger.debug("nvidia-smi pcie caps malformed row skipped: %r", line)
            continue
        out[index] = PcieCaps(
            gpu_max_gen=_parse_optional_int(parts[1]),
            host_max_gen=_parse_optional_int(parts[2]),
        )
    return out


_NVLINK_GPU_RE = re.compile(r"^GPU\s+(\d+)\s*:(.*)$")
_NVLINK_LINK_RE = re.compile(r"^Link\s+(\d+)\s*:\s*(.+?)\s*$")
_NVLINK_SPEED_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*GB/s", re.IGNORECASE)


def parse_nvlink_status(stdout: str) -> dict[int, NvlinkStatus]:
    """Parse ``nvidia-smi nvlink -s`` into {gpu_index: NvlinkStatus}.

    The output is a free-text report: a ``GPU N: <name> (UUID: ...)`` header
    per card, followed by either an explanation line or one ``Link N: ...``
    line per link. The explanation may also trail the header on the same
    line in some builds, so the header's remainder is scanned too. A header
    with nothing after it yields no entry — the caller renders that as
    "not reported", never as one of the three real states.
    """
    out: dict[int, NvlinkStatus] = {}
    current: int | None = None
    active = 0
    total = 0
    speed: float | None = None

    def flush() -> None:
        if current is None or total == 0:
            return
        out[current] = NvlinkStatus(
            state="active" if active > 0 else "inactive",
            active_links=active,
            total_links=total,
            link_speed_gbs=speed,
        )

    def classify(text: str) -> None:
        nonlocal active, total, speed
        if current is None:
            return
        low = text.lower()
        if "does not have or support" in low or "not supported" in low:
            out[current] = NvlinkStatus(state="unsupported")
            return
        if "all links are inactive" in low:
            out[current] = NvlinkStatus(state="inactive")
            return
        m = _NVLINK_LINK_RE.match(text)
        if m is None:
            return
        total += 1
        value = m.group(2)
        if "inactive" in value.lower():
            return
        active += 1
        sm = _NVLINK_SPEED_RE.search(value)
        if sm is not None:
            speed = float(sm.group(1))

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        gm = _NVLINK_GPU_RE.match(line)
        if gm is not None:
            flush()
            current = int(gm.group(1))
            active = total = 0
            speed = None
            # e.g. "GPU 0: NVIDIA RTX A4000 (UUID: ...) Device does not have
            # or support Nvlink" — the verdict on the header line itself.
            rest = gm.group(2)
            tail = rest.rsplit(")", 1)[1] if ")" in rest else ""
            if tail.strip():
                classify(tail.strip())
            continue
        classify(line)
    flush()
    return out


_TEMP_GPU_RE = re.compile(r"^GPU\s+([0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f])\s*$")
_TEMP_FIELD_RE = re.compile(r"^(GPU Shutdown Temp|GPU Slowdown Temp|GPU Max Operating Temp)\s*:\s*(.+?)\s*$")


def _parse_celsius(raw: str) -> int | None:
    """"100 C" -> 100; "N/A" -> None."""
    m = re.match(r"^([0-9]+)\s*C\b", raw.strip())
    return int(m.group(1)) if m else None


def parse_temperature_limits(stdout: str) -> dict[str, TempLimits]:
    """Parse ``nvidia-smi -q -d TEMPERATURE`` into {pci_bus_id: TempLimits}.

    Block headers are ``GPU 00000000:01:00.0`` (bus id, not index), which is
    why NVIDIA_SMI_LIVE_CMD also queries ``pci.bus_id`` — it is the join key.
    A threshold the driver prints as "N/A" stays None.
    """
    out: dict[str, TempLimits] = {}
    current: TempLimits | None = None
    for raw in stdout.splitlines():
        line = raw.strip()
        gm = _TEMP_GPU_RE.match(line)
        if gm is not None:
            current = TempLimits()
            out[gm.group(1)] = current
            continue
        if current is None:
            continue
        fm = _TEMP_FIELD_RE.match(line)
        if fm is None:
            continue
        value = _parse_celsius(fm.group(2))
        if fm.group(1) == "GPU Slowdown Temp":
            current.slowdown_c = value
        elif fm.group(1) == "GPU Shutdown Temp":
            current.shutdown_c = value
        else:
            current.max_operating_c = value
    return out


def parse_compute_apps_csv(stdout: str) -> list[GpuComputeApp]:
    """Parser for nvidia-smi --query-compute-apps. Memory may be reported as
    '[Not Supported]' or '[N/A]' on some MIG / consumer cards — those rows are
    skipped, not crashed on."""
    out: list[GpuComputeApp] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        pid_s, gpu_uuid, process_name, used_mem = parts
        try:
            pid = int(pid_s)
            memory_mib = int(used_mem)
        except ValueError:
            logger.debug("nvidia-smi compute-apps non-numeric row skipped: %r", line)
            continue
        out.append(GpuComputeApp(
            pid=pid,
            gpu_uuid=gpu_uuid,
            process_name=process_name,
            memory_mib=memory_mib,
        ))
    return out


GpuProbeState = Literal["unknown", "ok", "failing", "absent"]


@dataclass
class GpuProbeHealth:
    """Whether the warden can currently read its GPUs (#255).

    Tracks only the PRIMARY probes (the per-card query the stats sampler and
    the live snapshot both run). The optional ones -- NVLink, thermal limits,
    PCIe caps, topology, throttle reasons -- legitimately fail on some drivers
    and must not turn a healthy box into a failing one.

      unknown  no primary probe has run yet
      ok       the last primary probe answered
      failing  nvidia-smi is installed but did not answer -- e.g. "Failed to
               initialize NVML: Unknown Error" after a systemd reload revoked
               the container's GPU grant (#254). The engine already running
               keeps serving; the next model load fails.
      absent   no nvidia-smi on PATH: a CPU-only install, not a fault
    """

    state: GpuProbeState = "unknown"
    error: str | None = None
    # Wall-clock time (epoch s) the current state began, and of the last probe.
    since: float | None = None
    checked_at: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "error": self.error,
            "since": self.since,
            "checked_at": self.checked_at,
        }


_gpu_probe_health = GpuProbeHealth()


def gpu_probe_health() -> GpuProbeHealth:
    """The last-known GPU probe state. Reads memory only -- never runs
    nvidia-smi -- so /healthz can report it on every poll."""
    return _gpu_probe_health


def reset_gpu_probe_health() -> None:
    """Test hook."""
    global _gpu_probe_health
    _gpu_probe_health = GpuProbeHealth()


def _record_primary_probe(state: GpuProbeState, error: str | None) -> None:
    """Update the tracker, logging only when the state or its message
    changes. The probe runs every few seconds; a log line per probe is how
    'nvidia-smi exit 255:' reached ~1,300 lines an hour with nothing in it."""
    h = _gpu_probe_health
    now = time.time()
    h.checked_at = now
    if state == h.state and error == h.error:
        return
    previous = h.state
    h.state, h.error, h.since = state, error, now
    if state == "failing":
        logger.warning("GPU telemetry unavailable: %s", error)
    elif state == "absent":
        logger.info("GPU telemetry absent: %s (CPU-only host?)", error)
    elif state == "ok" and previous in ("failing", "absent"):
        logger.warning("GPU telemetry restored: nvidia-smi is answering again")


def _first_line(*streams: bytes) -> str:
    """The first non-empty line across the streams, trimmed. nvidia-smi
    prints NVML errors ('Failed to initialize NVML: Unknown Error') on
    STDOUT, not stderr, so both have to be read."""
    for raw in streams:
        for line in raw.decode(errors="replace").splitlines():
            line = line.strip()
            if line:
                return line[:200]
    return ""


async def _run_nvidia_smi(args: list[str], *, timeout: float = 5.0) -> str | None:  # noqa: ASYNC109
    """Run nvidia-smi with `args` and return stdout, or None if it failed.

    A failure of a primary probe (NVIDIA_SMI_CMD / NVIDIA_SMI_LIVE_CMD) is
    recorded in the GPU probe tracker with the reason nvidia-smi gave; an
    optional probe's failure is only a debug line.
    """
    primary = args is NVIDIA_SMI_CMD or args is NVIDIA_SMI_LIVE_CMD
    failure: tuple[GpuProbeState, str] | None = None
    stdout = b""
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except FileNotFoundError:
        failure = ("absent", "nvidia-smi not found")
    except TimeoutError:
        failure = ("failing", f"nvidia-smi did not answer within {timeout:g} s")
        # A wedged driver hangs every probe; without the kill each one
        # (every few seconds) would leave another nvidia-smi behind.
        if proc is not None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
    else:
        if proc.returncode != 0:
            detail = _first_line(stdout, stderr) or "no output"
            failure = ("failing", f"nvidia-smi exit {proc.returncode}: {detail}")
    if failure is not None:
        if primary:
            _record_primary_probe(*failure)
        else:
            logger.debug("optional nvidia-smi probe failed (%s): %s", args[1:3], failure[1])
        return None
    if primary:
        _record_primary_probe("ok", None)
    return stdout.decode()


async def query_gpus() -> list[GpuInfo]:
    """Run nvidia-smi and parse output. Returns [] if not available."""
    stdout = await _run_nvidia_smi(NVIDIA_SMI_CMD)
    if stdout is None:
        return []
    return parse_nvidia_smi_csv(stdout)


@dataclass
class StaticGpuFacts:
    """The per-card facts that don't change while the box is up: NVLink
    standing (by index) and thermal thresholds (by PCI bus id)."""

    nvlink: dict[int, NvlinkStatus] = field(default_factory=dict)
    temps: dict[str, TempLimits] = field(default_factory=dict)
    pcie_caps: dict[int, PcieCaps] = field(default_factory=dict)
    topology: GpuTopology | None = None


# Links don't come and go at runtime and thermal thresholds are firmware
# constants, so the two extra shell-outs are re-run once a minute rather than
# on every 2 s probe. The live endpoint's own cache is 2 s; this one sits
# behind it.
STATIC_FACTS_TTL_SECONDS = 60.0


class _StaticFactsCache:
    def __init__(
        self,
        *,
        ttl: float = STATIC_FACTS_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._cached: StaticGpuFacts | None = None
        self._cached_at: float = -1e9

    def reset(self) -> None:
        self._cached = None
        self._cached_at = -1e9

    async def get(self) -> StaticGpuFacts:
        now = self._clock()
        if self._cached is not None and (now - self._cached_at) < self._ttl:
            return self._cached
        facts = await _probe_static_facts()
        self._cached = facts
        self._cached_at = now
        return facts


async def _probe_static_facts() -> StaticGpuFacts:
    """Run the two static probes. Each degrades independently: a driver
    whose ``nvlink -s`` errors out still yields thermal limits, and vice
    versa. A failed probe leaves its dict empty, which the merge renders as
    "not reported" rather than any of the real states."""
    nvlink_out, temp_out, caps_out, topo_out = await asyncio.gather(
        _run_nvidia_smi(NVIDIA_SMI_NVLINK_CMD),
        _run_nvidia_smi(NVIDIA_SMI_TEMP_CMD),
        _run_nvidia_smi(NVIDIA_SMI_PCIE_CAPS_CMD),
        _run_nvidia_smi(NVIDIA_SMI_TOPO_CMD),
    )
    return StaticGpuFacts(
        nvlink=parse_nvlink_status(nvlink_out) if nvlink_out is not None else {},
        temps=parse_temperature_limits(temp_out) if temp_out is not None else {},
        pcie_caps=parse_pcie_caps_csv(caps_out) if caps_out is not None else {},
        topology=parse_topology_matrix(topo_out) if topo_out is not None else None,
    )


_static_facts_cache = _StaticFactsCache()


class _ThrottleProbe:
    """Runs the throttle-reason query, remembering which field spelling the
    driver accepts so a renamed family costs one failed shell-out ever,
    not one per probe. A driver that knows neither spelling yields {} and
    the rows keep ``throttle=None`` ("not reported")."""

    def __init__(self) -> None:
        self._preferred = 0

    def reset(self) -> None:
        self._preferred = 0

    async def get(self) -> dict[int, ThrottleReasons]:
        order = [self._preferred] + [
            i for i in range(len(NVIDIA_SMI_THROTTLE_CMDS)) if i != self._preferred
        ]
        for i in order:
            out = await _run_nvidia_smi(NVIDIA_SMI_THROTTLE_CMDS[i])
            if out is None:
                continue
            self._preferred = i
            return parse_throttle_reasons_csv(out)
        return {}


_throttle_probe = _ThrottleProbe()


def reset_static_facts_cache() -> None:
    """Test hook: forget the cached nvlink/thermal probes and the throttle
    spelling."""
    _static_facts_cache.reset()
    _throttle_probe.reset()


def merge_static_facts(gpus: list[GpuLive], facts: StaticGpuFacts) -> None:
    """Attach nvlink (by index) and thermal limits (by bus id) in place."""
    for g in gpus:
        g.nvlink = facts.nvlink.get(g.index)
        limits = facts.temps.get(g.bus_id) if g.bus_id is not None else None
        if limits is not None:
            g.temp_slowdown_c = limits.slowdown_c
            g.temp_shutdown_c = limits.shutdown_c
            g.temp_max_operating_c = limits.max_operating_c
        caps = facts.pcie_caps.get(g.index)
        if caps is not None:
            g.pcie_gen_gpu_max = caps.gpu_max_gen
            g.pcie_gen_host_max = caps.host_max_gen


async def query_gpu_snapshot() -> GpuSnapshot:
    """Live probe: per-GPU stats + telemetry + compute holders.

    If nvidia-smi is missing or fails, returns a snapshot with empty lists and
    a populated `probe_error` so the API can surface the cause without 500'ing.
    The static nvlink/thermal facts come from a 60 s cache and are merged onto
    each row; when their probes failed the rows simply keep None there.
    """
    live_out = await _run_nvidia_smi(NVIDIA_SMI_LIVE_CMD)
    if live_out is None:
        # The reason nvidia-smi gave (#255), e.g. "nvidia-smi exit 255: Failed
        # to initialize NVML: Unknown Error"; the generic text only when no
        # reason was recorded.
        return GpuSnapshot(
            gpus=[], apps=[], probe_error=_gpu_probe_health.error or "nvidia-smi unavailable"
        )
    gpus = parse_nvidia_smi_live_csv(live_out)
    apps_out, facts, throttle = await asyncio.gather(
        _run_nvidia_smi(NVIDIA_SMI_APPS_CMD),
        _static_facts_cache.get(),
        _throttle_probe.get(),
    )
    apps = parse_compute_apps_csv(apps_out) if apps_out is not None else []
    merge_static_facts(gpus, facts)
    for g in gpus:
        g.throttle = throttle.get(g.index)
    return GpuSnapshot(gpus=gpus, apps=apps, probe_error=None, topology=facts.topology)
