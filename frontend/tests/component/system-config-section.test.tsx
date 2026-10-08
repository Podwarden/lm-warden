// Component tests for the System Configuration panel (#148) and its
// per-GPU dials.
//
// Drives the section through the SWR + authFetchJSON boundary with a
// stubbed `fetch`, the same pattern stats-page.test.tsx uses. Two feeds
// are stubbed: /api/system/info (static inventory) and /api/system/gpus
// (live per-card telemetry + the topo -m matrix), joined by GPU index.
//
// The fixtures are recorded from real hosts. Card names are what nvidia-smi
// prints for the hardware; nothing here names a model.
//
// We deliberately exercise the section in isolation (not via StatsPage)
// so a regression here doesn't have to wait for the full v2 overview
// fixture to also render — focused failure messages > integration soup.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  cleanup,
  waitFor,
  within,
} from "@testing-library/react";
import { SWRConfig } from "swr";
import { SystemConfigSection } from "@/components/stats/system-config-section";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";
import type {
  GpuLiveCard,
  GpuTelemetry,
  GpuThrottle,
  GpuTopologyMatrix,
} from "@/lib/system-info";

function renderSection() {
  return render(
    <SWRConfig
      value={{
        provider: () => new Map(),
        dedupingInterval: 0,
        revalidateOnFocus: false,
        revalidateOnReconnect: false,
      }}
    >
      <SystemConfigSection />
    </SWRConfig>,
  );
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

interface StubOptions {
  info: unknown;
  infoStatus?: number;
  /** Omit to answer /api/system/gpus with a 500, i.e. the live feed failed. */
  live?: unknown;
  liveStatus?: number;
}

function installFetchStub({ info, infoStatus = 200, live, liveStatus = 200 }: StubOptions) {
  const mock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url === "/api/auth/refresh") {
      return json({ access_token: "test-jwt-refreshed" });
    }
    if (url === "/api/csrf") {
      return json({ csrf: "test-csrf" });
    }
    if (url === "/api/system/info") {
      return json(info, infoStatus);
    }
    if (url === "/api/system/gpus") {
      return live === undefined ? json({ detail: "boom" }, 500) : json(live, liveStatus);
    }
    return json({}, 404);
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

// ---- fixtures ----------------------------------------------------------

const FIXTURE_FULL = {
  cpu: {
    model: "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz",
    physical_cores: 14,
    threads: 28,
  },
  ram: { total_mb: 64277 },
  gpus: [
    {
      index: 0,
      name: "NVIDIA RTX A4000",
      vram_total_mb: 16376,
      driver_version: "610.43.02",
      cuda_version: "13.3",
    },
    {
      index: 1,
      name: "Quadro RTX 5000",
      vram_total_mb: 15360,
      driver_version: "610.43.02",
      cuda_version: "13.3",
    },
  ],
  os: { name: "Ubuntu", version: "24.04", kernel: "6.8.0-1008-nvidia" },
  docker: { version: "28.1.1", runtime: "nvidia", available: true },
};

const FIXTURE_NO_GPU_NO_DOCKER = {
  cpu: { model: "AMD Ryzen 9 7950X", physical_cores: 16, threads: 32 },
  ram: { total_mb: 65536 },
  gpus: [],
  os: { name: "Debian", version: "12", kernel: "6.1.0-13-amd64" },
  docker: { version: null, runtime: null, available: false },
};

const FIXTURE_NULL_PROC = {
  cpu: null,
  ram: null,
  gpus: [],
  os: { name: "unknown", version: "unknown", kernel: "unknown" },
  docker: { version: null, runtime: null, available: false },
};

const NOT_THROTTLED: GpuThrottle = {
  mask: 0,
  sw_thermal: false,
  hw_thermal: false,
  sw_power_cap: false,
  hw_slowdown: false,
  hw_power_brake: false,
  idle: false,
};

const IDLE: GpuThrottle = { ...NOT_THROTTLED, mask: 1, idle: true };

const ALL_NOT_REPORTED: GpuTelemetry = {
  temperature_c: null,
  temp_slowdown_c: null,
  temp_shutdown_c: null,
  temp_max_operating_c: null,
  fan_pct: null,
  power_w: null,
  power_limit_w: null,
  ecc_enabled: null,
  pstate: null,
  sm_clock_mhz: null,
  sm_clock_max_mhz: null,
  mem_clock_mhz: null,
  mem_clock_max_mhz: null,
  throttle: null,
  nvlink: null,
  pcie: {
    gen_current: null,
    gen_max: null,
    width_current: null,
    width_max: null,
    gen_gpu_max: null,
    gen_host_max: null,
  },
};

// The real heterogeneous host, idle: an RTX A4000 (16376 MiB, 140 W, ECC
// off, no NVLink silicon, on a x4 riser) beside a Quadro RTX 5000 (15360
// MiB, 230 W, ECC on, NVLink-capable but unbridged). Nothing is shared
// between the two.
const LIVE_HETERO: GpuLiveCard[] = [
  {
    index: 0,
    name: "NVIDIA RTX A4000",
    memory_total_mib: 16376,
    memory_used_mib: 14137,
    memory_free_mib: 2239,
    utilization_pct: 0,
    compute_cap: 8.6,
    architecture: "Ampere",
    telemetry: {
      temperature_c: 35,
      temp_slowdown_c: 100,
      temp_shutdown_c: 103,
      temp_max_operating_c: 98,
      fan_pct: 41,
      power_w: 12.68,
      power_limit_w: 140,
      ecc_enabled: false,
      pstate: "P8",
      sm_clock_mhz: 210,
      sm_clock_max_mhz: 2100,
      mem_clock_mhz: 405,
      mem_clock_max_mhz: 7001,
      throttle: IDLE,
      nvlink: { state: "unsupported", active_links: 0, total_links: 0, link_speed_gbs: null },
      // Idle: gen 1 now (normal). x4 of x16: a x4 slot or riser (a fault).
      // Both endpoints Gen 4-capable, link settled at Gen 3.
      pcie: {
        gen_current: 1,
        gen_max: 3,
        width_current: 4,
        width_max: 16,
        gen_gpu_max: 4,
        gen_host_max: 4,
      },
    },
  },
  {
    index: 1,
    name: "Quadro RTX 5000",
    memory_total_mib: 15360,
    memory_used_mib: 13818,
    memory_free_mib: 1542,
    utilization_pct: 0,
    compute_cap: 7.5,
    architecture: "Turing",
    telemetry: {
      temperature_c: 29,
      temp_slowdown_c: 89,
      temp_shutdown_c: 94,
      temp_max_operating_c: 88,
      fan_pct: 33,
      power_w: 12.23,
      power_limit_w: 230,
      ecc_enabled: true,
      pstate: "P8",
      sm_clock_mhz: 300,
      sm_clock_max_mhz: 2100,
      mem_clock_mhz: 405,
      mem_clock_max_mhz: 7001,
      throttle: IDLE,
      nvlink: { state: "inactive", active_links: 0, total_links: 0, link_speed_gbs: null },
      // Idle gen 1 again, full x16, card Gen 3-capable on a Gen 4 host — the
      // negotiated max already equals the card's own ceiling.
      pcie: {
        gen_current: 1,
        gen_max: 3,
        width_current: 16,
        width_max: 16,
        gen_gpu_max: 3,
        gen_host_max: 4,
      },
    },
  },
];

const TOPO_TWO_PHB: GpuTopologyMatrix = {
  indices: [0, 1],
  matrix: [
    ["X", "PHB"],
    ["PHB", "X"],
  ],
};

// The real 4x A4000 host under load: every card in SW thermal slowdown at
// 91-95 °C, holding 57-74 % of its 2100 MHz ceiling.
const THERMAL: GpuThrottle = { ...NOT_THROTTLED, mask: 0x20, sw_thermal: true };
const A4000_INFO = (i: number) => ({
  index: i,
  name: "NVIDIA RTX A4000",
  vram_total_mb: 16376,
  driver_version: "610.57.04",
  cuda_version: "13.3",
});
const LIVE_THROTTLED: GpuLiveCard[] = [1350, 1560, 1305, 1200].map((sm, i) => ({
  index: i,
  name: "NVIDIA RTX A4000",
  memory_total_mib: 16376,
  memory_used_mib: 15155,
  memory_free_mib: 1221,
  utilization_pct: 100,
  compute_cap: 8.6,
  architecture: "Ampere",
  telemetry: {
    temperature_c: [93, 91, 95, 93][i],
    temp_slowdown_c: 100,
    temp_shutdown_c: 103,
    temp_max_operating_c: 98,
    fan_pct: [89, 77, 96, 90][i],
    power_w: [112.4, 110.6, 106.2, 111.0][i],
    power_limit_w: 140,
    ecc_enabled: false,
    pstate: "P2",
    sm_clock_mhz: sm,
    sm_clock_max_mhz: 2100,
    mem_clock_mhz: 6501,
    mem_clock_max_mhz: 7001,
    throttle: THERMAL,
    nvlink: { state: "unsupported", active_links: 0, total_links: 0, link_speed_gbs: null },
    pcie: {
      gen_current: 4,
      gen_max: 4,
      width_current: 16,
      width_max: 16,
      gen_gpu_max: 4,
      gen_host_max: 4,
    },
  },
}));
const TOPO_FOUR_PHB: GpuTopologyMatrix = {
  indices: [0, 1, 2, 3],
  matrix: [
    ["X", "PHB", "PHB", "PHB"],
    ["PHB", "X", "PHB", "PHB"],
    ["PHB", "PHB", "X", "PHB"],
    ["PHB", "PHB", "PHB", "X"],
  ],
};

function liveResponse(
  gpus: GpuLiveCard[],
  probe_error: string | null = null,
  topology: GpuTopologyMatrix | null = null,
) {
  return {
    probed_at: "2026-09-06T08:21:37Z",
    probe_error,
    gpus,
    topology,
    allowed_indices: null,
  };
}

function gpuCards() {
  return screen.getAllByTestId("system-gpu-card");
}

describe("SystemConfigSection", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders the top row (CPU + RAM + OS/Docker) and a card per GPU", async () => {
    installFetchStub({ info: FIXTURE_FULL, live: liveResponse(LIVE_HETERO, null, TOPO_TWO_PHB) });
    renderSection();

    await waitFor(() => {
      expect(screen.getByTestId("system-cpu-value")).toBeInTheDocument();
    });

    expect(screen.getByTestId("system-cpu-value").textContent).toContain("Xeon");
    // RAM card: 64277 MB → 62.8 GiB
    expect(screen.getByTestId("system-ram-value").textContent).toBe("62.8");

    // OS card line
    expect(screen.getByTestId("system-os-value").textContent).toBe("Ubuntu 24.04");
    expect(screen.getByTestId("system-os-kernel").textContent).toContain(
      "6.8.0-1008-nvidia",
    );

    // Docker card line
    const dockerLine = screen.getByTestId("system-docker-available").textContent;
    expect(dockerLine).toContain("28.1.1");
    expect(dockerLine).toContain("nvidia");

    // GPU cards — 2 total, indexes 0 and 1, driven by the static list.
    const cards = gpuCards();
    expect(cards).toHaveLength(2);
    expect(cards[0].getAttribute("data-gpu-index")).toBe("0");
    expect(cards[1].getAttribute("data-gpu-index")).toBe("1");

    const drivers = screen.getAllByTestId("system-gpu-driver");
    expect(drivers[0].textContent).toBe("610.43.02");
    const cudas = screen.getAllByTestId("system-gpu-cuda");
    expect(cudas[0].textContent).toBe("13.3");
  });

  it("shows each card's OWN VRAM, power limit, ECC and NVLink on a heterogeneous host", async () => {
    installFetchStub({ info: FIXTURE_FULL, live: liveResponse(LIVE_HETERO, null, TOPO_TWO_PHB) });
    renderSection();
    await waitFor(() => {
      expect(screen.getAllByTestId("system-gpu-pstate")).toHaveLength(2);
    });
    const [a4000, rtx5000] = gpuCards();

    // VRAM used / total per card, with a bar each — 14137/16376 vs 13818/15360.
    const vramA = within(a4000).getByTestId("system-gpu-vram");
    expect(vramA.textContent).toBe("13.8 / 16.0 GiB");
    expect(within(a4000).getByRole("meter", { name: "VRAM used of total" }).getAttribute("aria-valuenow")).toBe("86");
    const vramB = within(rtx5000).getByTestId("system-gpu-vram");
    expect(vramB.textContent).toBe("13.5 / 15.0 GiB");
    expect(within(rtx5000).getByRole("meter", { name: "VRAM used of total" }).getAttribute("aria-valuenow")).toBe("90");

    // Power against ITS card's limit — 140 W on one, 230 W on the other.
    expect(within(a4000).getByTestId("system-gpu-power").textContent).toBe("12.7 / 140 W");
    expect(
      within(a4000).getByRole("meter", { name: "power draw against this card's limit" }).getAttribute("aria-valuenow"),
    ).toBe("9");
    expect(within(rtx5000).getByTestId("system-gpu-power").textContent).toBe("12.2 / 230 W");

    // Temperature against its own throttle point, with the tick.
    const tempA = within(a4000).getByTestId("system-gpu-temp");
    expect(tempA.textContent).toContain("35 °C");
    expect(tempA.textContent).toContain("100° throttle");
    expect(within(tempA).getByTestId("system-gpu-temp-tick")).toBeInTheDocument();
    expect(within(rtx5000).getByTestId("system-gpu-temp").textContent).toContain("89° throttle");

    // Fan, ECC (genuinely differs per card), clocks, P-state.
    expect(within(a4000).getByTestId("system-gpu-fan").textContent).toBe("41% fan");
    expect(within(rtx5000).getByTestId("system-gpu-fan").textContent).toBe("33% fan");
    expect(within(a4000).getByTestId("system-gpu-ecc").textContent).toBe("off");
    expect(within(rtx5000).getByTestId("system-gpu-ecc").textContent).toBe("on");
    expect(within(a4000).getByTestId("system-gpu-clock").textContent).toBe("210 / 2100 MHz");
    expect(within(a4000).getByTestId("system-gpu-mem-clock").textContent).toBe("405 / 7001 MHz");
    expect(within(a4000).getByTestId("system-gpu-pstate").textContent).toBe("P8");
    expect(within(a4000).getByTestId("system-gpu-util").textContent).toBe("0%");

    // NVLink: "no silicon" and "silicon but no link" are different sentences.
    expect(within(a4000).getByTestId("system-gpu-nvlink").textContent).toBe("none on this card");
    expect(within(rtx5000).getByTestId("system-gpu-nvlink").textContent).toBe(
      "supported, links inactive",
    );
    expect(screen.queryByTestId("system-gpu-live-unavailable")).toBeNull();
  });

  it("marks a spec that disagrees with the other cards on the host as 'differs'", async () => {
    installFetchStub({ info: FIXTURE_FULL, live: liveResponse(LIVE_HETERO, null, TOPO_TWO_PHB) });
    renderSection();
    await waitFor(() => {
      expect(screen.getAllByTestId("system-gpu-pstate")).toHaveLength(2);
    });
    const [a4000, rtx5000] = gpuCards();
    // The mark goes on the card that disagrees with the host's majority; on
    // a 2-card host a tie resolves to the first card, so the Quadro's ECC
    // "on", name, generation and full-width link are the ones marked. The
    // shared driver / CUDA are not. The mark is a glyph plus a word,
    // outside the value span, so the value itself still reads clean.
    const eccMark = within(rtx5000).getByTestId("system-gpu-ecc-differs");
    expect(eccMark.textContent).toContain("differs");
    expect(eccMark.getAttribute("data-state")).toBe("differs");
    expect(within(rtx5000).getByTestId("system-gpu-ecc").textContent).toBe("on");
    expect(within(rtx5000).getByTestId("system-gpu-name-differs")).toBeInTheDocument();
    expect(within(rtx5000).getByTestId("system-gpu-arch-differs")).toBeInTheDocument();
    expect(within(rtx5000).getByTestId("system-gpu-nvlink-differs")).toBeInTheDocument();
    for (const card of [a4000, rtx5000]) {
      expect(within(card).queryByTestId("system-gpu-driver-differs")).toBeNull();
      expect(within(card).queryByTestId("system-gpu-cuda-differs")).toBeNull();
    }
    expect(within(a4000).queryByTestId("system-gpu-ecc-differs")).toBeNull();
  });

  it("puts the generation under the name, derived from compute capability, and says when it is not recognised", async () => {
    // A capability the fixed map does not know (the next generation's number
    // is not public) must NOT get a guessed label.
    const unknownCap: GpuLiveCard = {
      ...LIVE_HETERO[1],
      compute_cap: 11.4,
      architecture: null,
    };
    installFetchStub({
      info: FIXTURE_FULL,
      live: liveResponse([LIVE_HETERO[0], unknownCap], null, TOPO_TWO_PHB),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getAllByTestId("system-gpu-arch")).toHaveLength(2);
    });
    const [a4000, other] = gpuCards();
    expect(within(a4000).getByTestId("system-gpu-arch").textContent).toBe("Ampere · compute 8.6");
    const arch = within(other).getByTestId("system-gpu-arch");
    expect(arch.textContent).toBe("compute 11.4 · generation not recognised");
    expect(arch.getAttribute("data-state")).toBe("not-recognised");
  });

  it("words the third NVLink state as N of M links active", async () => {
    const bridged: GpuLiveCard = {
      ...LIVE_HETERO[0],
      telemetry: {
        ...LIVE_HETERO[0].telemetry,
        nvlink: { state: "active", active_links: 3, total_links: 4, link_speed_gbs: 25 },
      },
    };
    installFetchStub({
      info: { ...FIXTURE_FULL, gpus: [FIXTURE_FULL.gpus[0]] },
      live: liveResponse([bridged]),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getByTestId("system-gpu-nvlink")).toBeInTheDocument();
    });
    expect(screen.getByTestId("system-gpu-nvlink").textContent).toBe(
      "3 of 4 links active · 25 GB/s each",
    );
  });

  it("renders a metric the hardware did not report as 'not reported', never 0 or blank", async () => {
    // Card 0: a fan-less card in a VM with no power sensor, no ECC, no
    // nvlink answer, no thermal thresholds, no clock ceilings, no throttle
    // answer. Card 1: the same sensors present but reading genuinely 0 —
    // fan at 0 %, 0.0 W. The two must not look alike.
    const vmCard: GpuLiveCard = {
      ...LIVE_HETERO[0],
      telemetry: { ...ALL_NOT_REPORTED, temperature_c: 41 },
    };
    const idleCard: GpuLiveCard = {
      ...LIVE_HETERO[1],
      telemetry: {
        ...LIVE_HETERO[1].telemetry,
        fan_pct: 0,
        power_w: 0,
        temp_slowdown_c: null,
      },
    };
    installFetchStub({ info: FIXTURE_FULL, live: liveResponse([vmCard, idleCard], null, TOPO_TWO_PHB) });
    renderSection();
    await waitFor(() => {
      expect(screen.getAllByTestId("system-gpu-arch")).toHaveLength(2);
    });
    const [vm, idle] = gpuCards();

    for (const slot of [
      "system-gpu-fan", "system-gpu-power", "system-gpu-ecc", "system-gpu-nvlink",
      "system-gpu-clock", "system-gpu-mem-clock", "system-gpu-pcie",
    ]) {
      const dd = within(vm).getByTestId(slot);
      expect(dd.textContent, slot).toContain("not reported");
      expect(dd.querySelector('[data-state="not-reported"]'), slot).not.toBeNull();
      expect(dd.textContent, slot).not.toMatch(/\b0\b/);
    }
    // The temperature IS reported; only its limit is missing, and it says so
    // rather than drawing a tick at an invented point.
    const vmTemp = within(vm).getByTestId("system-gpu-temp");
    expect(vmTemp.textContent).toContain("41 °C");
    expect(vmTemp.textContent).toContain("throttle point not reported");
    expect(within(vmTemp).queryByTestId("system-gpu-temp-tick")).toBeNull();
    expect(within(vm).queryByTestId("system-gpu-pstate")).toBeNull();
    // No throttle answer is not "not throttled": it is said.
    expect(within(vm).getByTestId("system-gpu-throttle-not-reported").textContent).toContain(
      "throttle reasons not reported",
    );
    expect(within(vm).queryByTestId("system-gpu-throttle")).toBeNull();

    // A genuine zero is a value and renders as one.
    expect(within(idle).getByTestId("system-gpu-fan").textContent).toBe("0% fan");
    expect(within(idle).getByTestId("system-gpu-power").textContent).toBe("0.0 / 230 W");
    expect(within(idle).getByTestId("system-gpu-temp").textContent).toContain(
      "throttle point not reported",
    );
    expect(within(idle).queryByTestId("system-gpu-throttle-not-reported")).toBeNull();
  });

  it("throttle: the verdict comes from the driver, names the reason, and reads the clock against its ceiling", async () => {
    // The real 4-card host right now: sw_thermal_slowdown = Active on
    // every card at 91-95 °C, clocks at 57-74 % of max. No heuristic — the
    // banner says what the driver said, and the card border follows it.
    installFetchStub({
      info: { ...FIXTURE_FULL, gpus: [0, 1, 2, 3].map(A4000_INFO) },
      live: liveResponse(LIVE_THROTTLED, null, TOPO_FOUR_PHB),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getAllByTestId("system-gpu-throttle")).toHaveLength(4);
    });
    const cards = gpuCards();
    const banners = screen.getAllByTestId("system-gpu-throttle");
    expect(banners[0].textContent).toContain("Thermal slowdown — clock at 64% of max");
    expect(banners[1].textContent).toContain("Thermal slowdown — clock at 74% of max");
    expect(banners[2].textContent).toContain("Thermal slowdown — clock at 62% of max");
    expect(banners[3].textContent).toContain("Thermal slowdown — clock at 57% of max");
    // Shape, not colour: a role=status chip with a data-state, and the card
    // itself carries the state.
    expect(banners[0].getAttribute("role")).toBe("status");
    expect(banners[0].getAttribute("data-state")).toBe("warning");
    expect(cards[0].getAttribute("data-state")).toBe("warning");
    // SW thermal slowdown is the driver pacing the card — a warning, not the
    // fault colour, which stays reserved for genuine faults.
    expect(banners[0].className).not.toContain("negative");
    expect(within(cards[0]).getByTestId("system-gpu-clock").textContent).toBe("1350 / 2100 MHz");
    expect(within(cards[0]).getByTestId("system-gpu-util").textContent).toBe("100%");
    expect(within(cards[0]).getByTestId("system-gpu-temp").textContent).toContain("93 °C");
    // Full-width Gen 4 link, no fault banner.
    expect(within(cards[0]).getByTestId("system-gpu-pcie").textContent).toBe("Gen 4 · x16 of x16");
    expect(screen.queryByTestId("system-gpu-fault")).toBeNull();
  });

  it("throttle: a power cap and a hardware slowdown are named as what they are", async () => {
    const powerCapped: GpuLiveCard = {
      ...LIVE_THROTTLED[0],
      telemetry: {
        ...LIVE_THROTTLED[0].telemetry,
        throttle: { ...NOT_THROTTLED, mask: 0x4, sw_power_cap: true },
      },
    };
    const hwThermal: GpuLiveCard = {
      ...LIVE_THROTTLED[1],
      telemetry: {
        ...LIVE_THROTTLED[1].telemetry,
        throttle: { ...NOT_THROTTLED, mask: 0x68, sw_thermal: true, hw_thermal: true, hw_slowdown: true },
      },
    };
    installFetchStub({
      info: { ...FIXTURE_FULL, gpus: [0, 1].map(A4000_INFO) },
      live: liveResponse([powerCapped, hwThermal], null, TOPO_TWO_PHB),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getAllByTestId("system-gpu-throttle")).toHaveLength(2);
    });
    const [capped, hw] = gpuCards();
    expect(within(capped).getByTestId("system-gpu-throttle").textContent).toContain(
      "Power cap — clock at 64% of max",
    );
    expect(capped.getAttribute("data-state")).toBe("warning");
    // HW thermal slowdown is the silicon protecting itself: a genuine
    // fault. It subsumes the generic HW slowdown bit and outranks the SW
    // thermal one, so it is the single thermal reason named.
    const hwBanner = within(hw).getByTestId("system-gpu-throttle");
    expect(hwBanner.textContent).toContain("HW thermal slowdown — clock at 74% of max");
    expect(hwBanner.textContent).not.toContain("+");
    expect(hwBanner.getAttribute("data-state")).toBe("fault");
    expect(hw.getAttribute("data-state")).toBe("fault");
  });

  it("throttle: an idle card with nothing held shows no verdict and no flag", async () => {
    installFetchStub({
      info: { ...FIXTURE_FULL, gpus: [FIXTURE_FULL.gpus[1]] },
      live: liveResponse([LIVE_HETERO[1]]),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getByTestId("system-gpu-arch")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("system-gpu-throttle")).toBeNull();
    expect(screen.queryByTestId("system-gpu-throttle-not-reported")).toBeNull();
    expect(screen.queryByTestId("system-gpu-fault")).toBeNull();
    expect(gpuCards()[0].getAttribute("data-state")).toBeNull();
  });

  it("specs are paired onto three lines: Driver | CUDA, Link, NVLink | ECC", async () => {
    installFetchStub({
      info: { ...FIXTURE_FULL, gpus: [FIXTURE_FULL.gpus[0]] },
      live: liveResponse([LIVE_HETERO[0]]),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getByTestId("system-gpu-arch")).toBeInTheDocument();
    });
    const lines = screen.getAllByTestId("system-gpu-spec-line");
    expect(lines).toHaveLength(3);
    expect(within(lines[0]).getByTestId("system-gpu-driver").textContent).toBe("610.43.02");
    expect(within(lines[0]).getByTestId("system-gpu-cuda").textContent).toBe("13.3");
    expect(lines[0].textContent).not.toContain("|"); // a hairline border divides, not a glyph
    // Link owns its line because its value wraps.
    expect(within(lines[1]).getByTestId("system-gpu-pcie").textContent).toBe(
      "Gen 1 now · up to Gen 3 · x4 of x16",
    );
    expect(lines[1].querySelectorAll("[data-testid^='system-gpu-']")).toHaveLength(2); // value + caps note
    expect(within(lines[2]).getByTestId("system-gpu-nvlink").textContent).toBe("none on this card");
    expect(within(lines[2]).getByTestId("system-gpu-ecc").textContent).toBe("off");
    // Each pair wraps independently: the pairs are flex-wrap rows.
    for (const line of lines) expect(line.className).toContain("flex-wrap");
  });

  it("PCIe: a reduced link WIDTH is a fault banner; an idle downshift in generation is not", async () => {
    installFetchStub({ info: FIXTURE_FULL, live: liveResponse(LIVE_HETERO, null, TOPO_TWO_PHB) });
    renderSection();
    await waitFor(() => {
      expect(screen.getAllByTestId("system-gpu-pcie")).toHaveLength(2);
    });
    const [a4000, rtx5000] = gpuCards();

    // Both cards idle at gen 1 against a max of 3: shown together, no flag.
    expect(within(a4000).getByTestId("system-gpu-pcie").textContent).toBe(
      "Gen 1 now · up to Gen 3 · x4 of x16",
    );
    expect(within(rtx5000).getByTestId("system-gpu-pcie").textContent).toBe(
      "Gen 1 now · up to Gen 3 · x16 of x16",
    );
    // GPU 0: x4 of x16 — a permanent slot/riser limit, a genuine fault:
    // the fault banner, and the card border follows it.
    const fault = within(a4000).getByTestId("system-gpu-fault");
    expect(fault.textContent).toContain("Reduced link width — x4 of x16");
    expect(fault.getAttribute("role")).toBe("status");
    expect(fault.getAttribute("data-state")).toBe("fault");
    expect(a4000.getAttribute("data-state")).toBe("fault");
    // GPU 1: full width, no fault, and the gen-1 idle reading raised nothing.
    expect(within(rtx5000).queryByTestId("system-gpu-fault")).toBeNull();
    expect(rtx5000.getAttribute("data-state")).toBeNull();

    // Honesty about the ceiling: the negotiated max (3) is the number shown,
    // and the endpoint capabilities appear as a labelled note only where they
    // exceed it (GPU 0: card 4 / host 4 / link 3). GPU 1's card maxes at 3,
    // which equals the link, so it gets no note.
    expect(within(a4000).getByTestId("system-gpu-pcie-caps").textContent).toBe(
      "card Gen 4, host Gen 4 — link settled at Gen 3",
    );
    expect(within(rtx5000).queryByTestId("system-gpu-pcie-caps")).toBeNull();
  });

  it("PCIe: degrades cleanly when the driver lacks the gpumax/hostmax family", async () => {
    const oldDriver: GpuLiveCard = {
      ...LIVE_HETERO[0],
      telemetry: {
        ...LIVE_HETERO[0].telemetry,
        pcie: { ...LIVE_HETERO[0].telemetry.pcie, gen_gpu_max: null, gen_host_max: null },
      },
    };
    installFetchStub({
      info: { ...FIXTURE_FULL, gpus: [FIXTURE_FULL.gpus[0]] },
      live: liveResponse([oldDriver]),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getByTestId("system-gpu-pcie")).toBeInTheDocument();
    });
    // The four classic fields still render, the width fault still fires,
    // and there is simply no capability note — no "Gen 0", no "x0".
    expect(screen.getByTestId("system-gpu-pcie").textContent).toBe(
      "Gen 1 now · up to Gen 3 · x4 of x16",
    );
    expect(screen.getByTestId("system-gpu-fault")).toBeInTheDocument();
    expect(screen.queryByTestId("system-gpu-pcie-caps")).toBeNull();
    expect(screen.getByTestId("system-gpu-card").textContent).not.toMatch(/Gen 0|x0\b/);
  });

  it("PCIe: a partially reported link says which half is missing rather than printing 0", async () => {
    const partial: GpuLiveCard = {
      ...LIVE_HETERO[1],
      telemetry: {
        ...LIVE_HETERO[1].telemetry,
        pcie: {
          gen_current: 1,
          gen_max: null,
          width_current: null,
          width_max: 16,
          gen_gpu_max: null,
          gen_host_max: null,
        },
      },
    };
    installFetchStub({
      info: { ...FIXTURE_FULL, gpus: [FIXTURE_FULL.gpus[1]] },
      live: liveResponse([partial]),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getByTestId("system-gpu-pcie")).toBeInTheDocument();
    });
    expect(screen.getByTestId("system-gpu-pcie").textContent).toBe(
      "Gen 1 now · max not reported · up to x16 · current not reported",
    );
    // Unknown current width is not a "reduced width" — nothing to compare.
    expect(screen.queryByTestId("system-gpu-fault")).toBeNull();
  });

  it("clocks: a missing ceiling leaves the reading standing alone and says so", async () => {
    const noCeiling: GpuLiveCard = {
      ...LIVE_HETERO[0],
      telemetry: { ...LIVE_HETERO[0].telemetry, sm_clock_max_mhz: null, mem_clock_max_mhz: null },
    };
    installFetchStub({
      info: { ...FIXTURE_FULL, gpus: [FIXTURE_FULL.gpus[0]] },
      live: liveResponse([noCeiling]),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getByTestId("system-gpu-clock")).toBeInTheDocument();
    });
    expect(screen.getByTestId("system-gpu-clock").textContent).toBe("210 MHz · max not reported");
    expect(screen.getByTestId("system-gpu-mem-clock").textContent).toBe("405 MHz · max not reported");
  });

  it("draws the interconnect graph from topo -m for two or more cards", async () => {
    installFetchStub({ info: FIXTURE_FULL, live: liveResponse(LIVE_HETERO, null, TOPO_TWO_PHB) });
    renderSection();
    await waitFor(() => {
      expect(screen.getByTestId("system-gpu-interconnect-svg")).toBeInTheDocument();
    });
    const svg = screen.getByTestId("system-gpu-interconnect-svg");
    expect(svg.getAttribute("data-kind")).toBe("hub");
    // GPU 0 is on a x4 riser: its spoke is the dashed fault one.
    const edges = screen.getAllByTestId("system-gpu-interconnect-edge");
    expect(edges.map((e) => e.getAttribute("data-kind")).sort()).toEqual(["spoke", "spoke-narrow"]);
  });

  it("hides the interconnect section on a single-card host", async () => {
    installFetchStub({
      info: { ...FIXTURE_FULL, gpus: [FIXTURE_FULL.gpus[0]] },
      live: liveResponse([LIVE_HETERO[0]], null, { indices: [0], matrix: [["X"]] }),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getByTestId("system-gpu-arch")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("system-gpu-interconnect")).toBeNull();
  });

  it("says the interconnect is not reported when topo -m gave nothing", async () => {
    installFetchStub({ info: FIXTURE_FULL, live: liveResponse(LIVE_HETERO, null, null) });
    renderSection();
    await waitFor(() => {
      expect(screen.getByTestId("system-gpu-interconnect")).toBeInTheDocument();
    });
    expect(screen.getByTestId("system-gpu-interconnect-not-reported").textContent).toContain(
      "interconnect not reported",
    );
    expect(screen.queryByTestId("system-gpu-interconnect-svg")).toBeNull();
  });

  it("keeps the static rows and says so when the live feed fails", async () => {
    installFetchStub({ info: FIXTURE_FULL }); // /api/system/gpus → 500
    renderSection();
    await waitFor(() => {
      expect(screen.getAllByTestId("system-gpu-live-unavailable")).toHaveLength(2);
    });
    const [a4000] = gpuCards();
    expect(within(a4000).getByTestId("system-gpu-live-unavailable").textContent).toContain(
      "live telemetry unavailable",
    );
    // VRAM falls back to the static total and says usage is not reported —
    // it does not show "0 / 16.0".
    const vram = within(a4000).getByTestId("system-gpu-vram");
    expect(vram.textContent).toContain("16.0 GiB");
    expect(vram.textContent).toContain("usage not reported");
    expect(vram.textContent).not.toContain("0 /");
    expect(within(a4000).getByTestId("system-gpu-util").textContent).toBe("not reported");
    expect(within(a4000).getByTestId("system-gpu-temp").textContent).toContain("not reported");
    expect(within(a4000).getByTestId("system-gpu-arch").textContent).toBe("not reported");
    expect(within(a4000).getByTestId("system-gpu-driver").textContent).toBe("610.43.02");
    expect(within(a4000).getByTestId("system-gpu-cuda").textContent).toBe("13.3");
    // The graph section is there (two cards) but has nothing to draw, and
    // says why.
    expect(screen.getByTestId("system-gpu-interconnect-not-reported").textContent).toContain(
      "interconnect not reported",
    );
  });

  it("surfaces the probe's own error when nvidia-smi is missing on the live path", async () => {
    installFetchStub({
      info: FIXTURE_FULL,
      live: liveResponse([], "nvidia-smi unavailable"),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getAllByTestId("system-gpu-live-unavailable")).toHaveLength(2);
    });
    expect(screen.getAllByTestId("system-gpu-live-unavailable")[0].textContent).toContain(
      "nvidia-smi unavailable",
    );
  });

  it("renders CUDA as 'not reported' when the driver printed no version line", async () => {
    installFetchStub({
      info: {
        ...FIXTURE_FULL,
        gpus: [{ ...FIXTURE_FULL.gpus[0], cuda_version: null }],
      },
      live: liveResponse([LIVE_HETERO[0]]),
    });
    renderSection();
    await waitFor(() => {
      expect(screen.getByTestId("system-gpu-cuda")).toBeInTheDocument();
    });
    expect(screen.getByTestId("system-gpu-cuda").textContent).toBe("not reported");
  });

  it("shows the empty-state and 'not available' Docker line on a no-GPU host", async () => {
    installFetchStub({
      info: FIXTURE_NO_GPU_NO_DOCKER,
      live: liveResponse([], "nvidia-smi unavailable"),
    });
    renderSection();

    await waitFor(() => {
      expect(screen.getByTestId("system-cpu-value")).toBeInTheDocument();
    });

    expect(screen.getByTestId("system-cpu-value").textContent).toContain("Ryzen");
    expect(screen.getByTestId("system-gpus-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("system-gpu-card")).toBeNull();
    expect(screen.queryByTestId("system-gpu-interconnect")).toBeNull();
    expect(screen.getByTestId("system-docker-unavailable").textContent).toMatch(
      /not available/i,
    );
  });

  it("renders em-dash placeholders when CPU/RAM are null", async () => {
    installFetchStub({ info: FIXTURE_NULL_PROC, live: liveResponse([]) });
    renderSection();

    await waitFor(() => {
      expect(screen.getByTestId("system-cpu-value")).toBeInTheDocument();
    });

    expect(screen.getByTestId("system-cpu-value").textContent).toBe("—");
    expect(screen.getByTestId("system-ram-value").textContent).toBe("—");
    // OS reads "unknown" when all three fields are unknown — verifies
    // formatOsName collapses the all-unknown case.
    expect(screen.getByTestId("system-os-value").textContent).toBe("unknown");
  });

  it("renders an error message on fetch failure", async () => {
    installFetchStub({ info: { detail: "boom" }, infoStatus: 500 });
    renderSection();

    await waitFor(() => {
      expect(screen.getByTestId("system-config-error")).toBeInTheDocument();
    });
  });
});
