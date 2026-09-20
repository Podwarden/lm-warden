// Frontend types for the /api/system/info contract (#148).
//
// The endpoint returns a static-ish inventory used by the "System
// Configuration" panel on /stats: CPU model + core/thread counts,
// total RAM, per-GPU name / VRAM / driver / CUDA, OS release + kernel,
// and Docker version + runtime. See app/system/system_info.py for the
// authoritative shape; this file mirrors it so a drift fails typecheck
// in CI before it ships.
//
// Why hand-typed: FastAPI's OpenAPI schema for `dict[str, Any]` is
// opaque (the generated type collapses to `{ [key: string]: unknown }`),
// so openapi-typescript can't reach the inner shape. The contract is
// pinned by the backend unit tests + the page component test instead.
//
// Optional/null fields and their semantics:
//   * `cpu` — `null` when /proc/cpuinfo couldn't be read or yielded
//     no usable processor entries (e.g. exotic kernels).
//   * `ram` — `null` when /proc/meminfo couldn't be read or had no
//     MemTotal line. UI shows "—".
//   * `gpus` — always an array; `[]` when nvidia-smi is unavailable.
//   * `os` — always present; missing fields default to "unknown" rather
//     than null so the card always has something to render.
//   * `docker` — always present with `available: boolean`; when false
//     the `version`/`runtime` slots are null and the card shows the
//     "not available" hint.
//   * Per-GPU `cuda_version` — `null` on the rare host where nvidia-smi
//     prints a card row but no version banner (corrupt install / WSL).

export interface SystemInfoCpu {
  model: string;
  /** Distinct (physical_id, core_id) pairs in /proc/cpuinfo, or the
   *  total processor count when that metadata is absent. */
  physical_cores: number;
  /** Total processor entries in /proc/cpuinfo (== logical cores). */
  threads: number;
}

export interface SystemInfoRam {
  total_mb: number;
}

export interface SystemInfoGpu {
  index: number;
  name: string;
  vram_total_mb: number;
  driver_version: string;
  /** `null` when nvidia-smi --version didn't print a CUDA banner. */
  cuda_version: string | null;
}

export interface SystemInfoOs {
  /** Pretty OS name e.g. "Ubuntu". Defaults to "unknown". */
  name: string;
  /** Release version e.g. "24.04". Defaults to "unknown". */
  version: string;
  /** `uname -r` output e.g. "6.8.0-1008-nvidia". Defaults to "unknown". */
  kernel: string;
}

export interface SystemInfoDocker {
  /** Server version from `docker info`. `null` when unavailable. */
  version: string | null;
  /** Default runtime (typically "runc" or "nvidia"). `null` when
   *  unavailable. */
  runtime: string | null;
  /** False when the `docker info` shell-out failed (no socket / not
   *  installed). The card renders a "not available" placeholder. */
  available: boolean;
}

export interface SystemInfo {
  cpu: SystemInfoCpu | null;
  ram: SystemInfoRam | null;
  gpus: SystemInfoGpu[];
  os: SystemInfoOs;
  docker: SystemInfoDocker;
}

// ---- Live per-card telemetry: GET /api/system/gpus -----------------------
//
// The GPU cards join this onto the static inventory above by `index`. See
// app/system/routes_gpus.py for the authoritative shape. The rule every
// consumer must keep: a `null` here means THE HARDWARE DID NOT REPORT IT
// and renders as "not reported" — never as 0, never as a blank. A fan-less
// card, a card with no power sensor and a card drawing 0 W are three
// different things and must look different.

export type NvlinkState = "unsupported" | "inactive" | "active";

export interface GpuNvlink {
  /** `unsupported` = the silicon has no NVLink; `inactive` = it does but no
   *  link is up (no bridge fitted); `active` = at least one link carries
   *  traffic. Three facts, never collapsed into one. */
  state: NvlinkState;
  active_links: number;
  total_links: number;
  link_speed_gbs: number | null;
}

export interface GpuTelemetry {
  temperature_c: number | null;
  /** The driver's clock-throttle point for this card — the limit the
   *  temperature is shown against. */
  temp_slowdown_c: number | null;
  temp_shutdown_c: number | null;
  temp_max_operating_c: number | null;
  fan_pct: number | null;
  power_w: number | null;
  /** This card's own power cap. Cards in one box differ (140 W next to
   *  230 W on a real host), so never share one limit across the fleet. */
  power_limit_w: number | null;
  /** `null` = the card has no ECC memory at all, which is not "off". */
  ecc_enabled: boolean | null;
  pstate: string | null;
  /** Each clock is read against ITS OWN maximum — the maximum is what makes
   *  a low value readable: 210 MHz idle is normal, 1200 MHz of 2100 at
   *  full load is a card being held back. */
  sm_clock_mhz: number | null;
  sm_clock_max_mhz: number | null;
  mem_clock_mhz: number | null;
  mem_clock_max_mhz: number | null;
  /** Why the clocks are where they are, from the driver's
   *  `clocks_throttle_reasons` — never a guess from a temperature ratio.
   *  `null` only when that probe said nothing for this card, which is
   *  not the same as "nothing is throttling". */
  throttle: GpuThrottle | null;
  /** `null` only when the nvlink probe said nothing for this card. */
  nvlink: GpuNvlink | null;
  pcie: GpuPcie;
}

export interface GpuThrottle {
  /** The raw NVML bitmask, for the tooltip. */
  mask: number | null;
  sw_thermal: boolean | null;
  hw_thermal: boolean | null;
  sw_power_cap: boolean | null;
  hw_slowdown: boolean | null;
  hw_power_brake: boolean | null;
  /** Clocks are low because there is nothing to do. Not a slowdown. */
  idle: boolean | null;
}

/** The `nvidia-smi topo -m` matrix. `indices[i]` is the GPU number of
 *  row/column `i`; cells are legend codes verbatim. See gpu-topology.ts. */
export interface GpuTopologyMatrix {
  indices: number[];
  matrix: string[][];
}

export interface GpuPcie {
  /** 1 on an idle card is NORMAL — PCIe downshifts to save power and
   *  climbs under load. Never flag a low current generation. */
  gen_current: number | null;
  /** The NEGOTIATED ceiling: what the link will actually do. */
  gen_max: number | null;
  /** Width does not downshift. Below `width_max` = a x4 slot or a riser,
   *  a permanent limit, and worth a warning. */
  width_current: number | null;
  width_max: number | null;
  /** What the card / host could each do. Null on drivers that lack the
   *  `gpumax`/`hostmax` fields. Shown only when they exceed `gen_max`. */
  gen_gpu_max: number | null;
  gen_host_max: number | null;
}

/** Everything the PCIe row needs to say, decided in one place so the
 *  wording and the flags cannot drift apart.
 *
 *  Honesty rule for the generation: `gen_max` (negotiated) is the number
 *  shown as the ceiling, because it is what the link will do. The endpoint
 *  capabilities are added as a labelled note ONLY when both are known and
 *  both exceed it — "card Gen 4, host Gen 4, link settled at Gen 3" is a
 *  fact about the slot or riser, not a reason to print 4. */
export interface PcieReading {
  /** "Gen 1 now · up to Gen 3", or a partial when one side is missing. */
  generation: string | null;
  /** "x4 of x16", "x16 of x16", or a partial. */
  width: string | null;
  /** True only when width_current < width_max. Never for generation. */
  widthReduced: boolean;
  /** The endpoint-capability note, when it disagrees with the negotiated max. */
  capabilityNote: string | null;
}

export function pcieReading(p: GpuPcie | null | undefined): PcieReading | null {
  if (!p) return null;
  const { gen_current, gen_max, width_current, width_max, gen_gpu_max, gen_host_max } = p;
  if (gen_current === null && gen_max === null && width_current === null && width_max === null) {
    return null;
  }

  let generation: string | null = null;
  if (gen_current !== null && gen_max !== null) {
    // A link running at its ceiling is just "Gen 4"; the "now / up to"
    // form is for the idle downshift, where the two numbers differ.
    generation =
      gen_current === gen_max
        ? `Gen ${gen_max}`
        : `Gen ${gen_current} now · up to Gen ${gen_max}`;
  } else if (gen_current !== null) {
    generation = `Gen ${gen_current} now · max ${NOT_REPORTED}`;
  } else if (gen_max !== null) {
    generation = `up to Gen ${gen_max} · current ${NOT_REPORTED}`;
  }

  let width: string | null = null;
  if (width_current !== null && width_max !== null) {
    width = `x${width_current} of x${width_max}`;
  } else if (width_current !== null) {
    width = `x${width_current} · max ${NOT_REPORTED}`;
  } else if (width_max !== null) {
    width = `up to x${width_max} · current ${NOT_REPORTED}`;
  }

  const widthReduced =
    width_current !== null && width_max !== null && width_current < width_max;

  let capabilityNote: string | null = null;
  if (gen_max !== null && gen_gpu_max !== null && gen_host_max !== null) {
    const endpoints = Math.min(gen_gpu_max, gen_host_max);
    if (endpoints > gen_max) {
      capabilityNote = `card Gen ${gen_gpu_max}, host Gen ${gen_host_max} — link settled at Gen ${gen_max}`;
    }
  }

  return { generation, width, widthReduced, capabilityNote };
}

/** The Link line on the card: generation and width in one reading,
 *  "Gen 4 · x16 of x16" / "Gen 1 now · up to Gen 3 · x4 of x16". Null when
 *  the driver reported no PCIe fields at all. */
export function linkLabel(p: GpuPcie | null | undefined): string | null {
  const r = pcieReading(p);
  if (!r) return null;
  const parts = [r.generation, r.width].filter((s): s is string => s !== null);
  return parts.length > 0 ? parts.join(" · ") : null;
}

/** The throttle verdict, as the driver states it. Thermal, power cap and
 *  hardware slowdown are different problems with different fixes, so each
 *  active reason is named; several are joined. The clock-held figure is
 *  the SM clock against its own ceiling, when both are known.
 *
 *  Severity: a hardware slowdown (HW thermal, HW power brake, or the
 *  generic HW slowdown) is the silicon protecting itself — a genuine fault.
 *  Software thermal slowdown and the power cap are the driver pacing the
 *  card — a warning. In retro-dark, warn and positive share one amber, so
 *  the verdict is always words on a bordered chip, never colour alone. */
export interface ThrottleVerdict {
  reasons: string[];
  text: string;
  severity: "warn" | "fault";
  /** Round percentage of the SM ceiling, when both clocks are known. */
  heldPct: number | null;
}

export function throttleVerdict(
  throttle: GpuThrottle | null | undefined,
  sm_clock_mhz: number | null,
  sm_clock_max_mhz: number | null,
): ThrottleVerdict | null {
  if (!throttle) return null;
  const reasons: string[] = [];
  let fault = false;
  const idle = throttle.idle === true;
  if (throttle.hw_thermal) {
    reasons.push("HW thermal slowdown");
    fault = true;
  } else if (throttle.sw_thermal) {
    reasons.push("Thermal slowdown");
  }
  if (throttle.hw_power_brake) {
    reasons.push("HW power brake");
    fault = true;
  } else if (throttle.sw_power_cap && !idle) {
    // NVML raises sw_power_cap whenever the driver is holding clocks below
    // maximum -- including when the only reason is that there is no work to
    // do. It says so with a dedicated GpuIdle bit, which the backend already
    // parses. Without this guard a perfectly healthy idle card reads as
    // "Power cap - clock held at 6%", where the 6% is nothing but the idle
    // clock over the maximum. A soft power cap is a claim about a card under
    // load; the hardware faults above and below are reported either way.
    reasons.push("Power cap");
  }
  // The generic HW slowdown bit is set alongside its thermal / power-brake
  // sub-reasons; name it on its own only when neither explains it.
  if (throttle.hw_slowdown && !throttle.hw_thermal && !throttle.hw_power_brake) {
    reasons.push("HW slowdown");
    fault = true;
  }
  if (reasons.length === 0) return null;
  const heldPct =
    sm_clock_mhz !== null && sm_clock_max_mhz !== null && sm_clock_max_mhz > 0
      ? Math.round((sm_clock_mhz / sm_clock_max_mhz) * 100)
      : null;
  // "held at N%" implied something was capping the clock; the number is only
  // ever sm_clock/sm_clock_max, so state it as the ratio it is.
  const text =
    heldPct === null
      ? reasons.join(" + ")
      : `${reasons.join(" + ")} — clock at ${heldPct}% of max`;
  return { reasons, text, severity: fault ? "fault" : "warn", heldPct };
}

/** "Ampere · compute 8.6", or the honest fallbacks. The generation comes
 *  from the API, which derives it from compute_cap and nothing else;
 *  an unrecognised capability shows its raw number and says so, because
 *  the next generation's number is not public and a guessed label would be
 *  confidently wrong the day one ships. */
export function archLabel(
  compute_cap: number | null | undefined,
  architecture: string | null | undefined,
): string {
  if (compute_cap === null || compute_cap === undefined) {
    return "compute capability not reported";
  }
  const cap = compute_cap.toFixed(1);
  if (!architecture) return `compute ${cap} · generation not recognised`;
  return `${architecture} · compute ${cap}`;
}

/** A clock against its own ceiling: "1350 / 2100 MHz". */
export interface ClockReading {
  value: string;
  /** " / 2100 MHz", or " MHz · max not reported". */
  suffix: string;
  fraction: number | null;
}

export function clockReading(mhz: number | null, maxMhz: number | null): ClockReading | null {
  if (mhz === null) return null;
  if (maxMhz === null || maxMhz <= 0) {
    return { value: `${mhz}`, suffix: ` MHz · max ${NOT_REPORTED}`, fraction: null };
  }
  return { value: `${mhz}`, suffix: ` / ${maxMhz} MHz`, fraction: mhz / maxMhz };
}

export interface GpuLiveCard {
  index: number;
  name: string;
  memory_total_mib: number;
  memory_used_mib: number;
  memory_free_mib: number;
  utilization_pct: number;
  compute_cap: number | null;
  /** Derived from compute_cap by the API; null = unrecognised or missing. */
  architecture: string | null;
  telemetry: GpuTelemetry;
}

export interface GpuLiveResponse {
  probed_at: string;
  probe_error: string | null;
  gpus: GpuLiveCard[];
  /** `null` when `nvidia-smi topo -m` gave nothing. */
  topology: GpuTopologyMatrix | null;
  allowed_indices: number[] | null;
}

/** The one string every "hardware did not report this" slot renders. Kept
 *  as words, not a glyph, so it cannot be mistaken for a value of 0 and
 *  reads the same without colour. */
export const NOT_REPORTED = "not reported";

/** NVLink standing as a sentence. Each of the three real states gets its
 *  own wording so they differ by text, not by colour. */
export function nvlinkLabel(nvlink: GpuNvlink | null | undefined): string {
  if (!nvlink) return NOT_REPORTED;
  switch (nvlink.state) {
    case "unsupported":
      return "none on this card";
    case "inactive":
      return nvlink.total_links > 0
        ? `supported, 0 of ${nvlink.total_links} links active`
        : "supported, links inactive";
    case "active": {
      const base = `${nvlink.active_links} of ${nvlink.total_links} links active`;
      return nvlink.link_speed_gbs === null
        ? base
        : `${base} · ${nvlink.link_speed_gbs} GB/s each`;
    }
  }
}

export function eccLabel(ecc: boolean | null | undefined): string {
  if (ecc === null || ecc === undefined) return NOT_REPORTED;
  return ecc ? "on" : "off";
}

/** One decimal below 100, whole watts above — same rule as the Power tile
 *  so "12.7 W of 140 W" reads consistently with the host row. */
export function formatWattsPlain(w: number): string {
  return w < 100 ? w.toFixed(1) : Math.round(w).toString();
}

// ---- Formatters ----------------------------------------------------------

/** Render a MB count as a human-friendly GiB string, e.g.
 *  `64277` → `"62.8"`. One decimal place — matches the VRAM tile. */
export function mbToGib(mb: number): string {
  if (!mb) return "0";
  return (mb / 1024).toFixed(1);
}

/** Compose the OS line e.g. "Ubuntu 24.04". Hides "unknown" tokens so a
 *  partial /etc/os-release doesn't print "unknown unknown". */
export function formatOsName(os: SystemInfoOs): string {
  const parts: string[] = [];
  if (os.name && os.name !== "unknown") parts.push(os.name);
  if (os.version && os.version !== "unknown") parts.push(os.version);
  return parts.length > 0 ? parts.join(" ") : "unknown";
}
