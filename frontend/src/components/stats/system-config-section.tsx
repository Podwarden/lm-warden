// System Configuration panel for /stats (#148), with per-card GPU dials.
//
// Static-ish inventory the operator wants when they're interpreting the
// live numbers above it — CPU model + core counts, total RAM, OS +
// Docker context, one card per GPU, and how the cards reach each other.
// Two feeds:
//
//   * GET /api/system/info  (30 s poll, 60 s server cache) — CPU, RAM, OS,
//     Docker, and per-GPU name / VRAM total / driver / CUDA.
//   * GET /api/system/gpus  (10 s poll, 2 s server cache) — per-card
//     telemetry: VRAM used, utilisation, temperature + throttle point, fan,
//     power draw + the card's own limit, clocks against their ceilings,
//     the driver's throttle reasons, ECC, NVLink, PCIe link, P-state,
//     compute capability + the generation derived from it — and the
//     `topo -m` matrix the interconnect graph is drawn from.
//
// The cards are driven by the static list and joined to the live feed by
// index, so a failed live probe degrades the cards rather than emptying
// the panel: the static rows stay and the card says the live feed is
// unavailable.
//
// The card's shape (approved mockup, v3 of this panel):
//   * Head: a large utilisation number beside two gauges — load, and
//     temperature with a TICK at the throttle point. Earlier versions drew
//     utilisation and temperature as concentric arcs; that was rejected
//     twice for being hard to read, spending ~120 px on two numbers and
//     drawing a meaningless stub at 0 %. No radial gauge here.
//   * A verdict chip when the DRIVER says the clocks are held — thermal,
//     power cap or hardware slowdown, each named, because they are
//     different problems. Never a guess from a temperature ratio.
//   * VRAM / power / fan as three bar rows, each against ITS card's total
//     or limit.
//   * Core and memory clock, each against its own maximum: 210 MHz idle
//     is normal; 1200 of 2100 at full load is a card being held back.
//   * Specs paired onto three lines — Driver | CUDA, Link, NVLink | ECC —
//     separated by a hairline border, wrapping independently when narrow.
//   * Architecture under the name, derived from compute capability only.
//
// Three rules this panel keeps:
//   * A metric the hardware does not report renders as the words "not
//     reported" — never 0, never a blank. A fan-less card, a card with no
//     power sensor and a card reading 0 W must look different.
//   * Cards are heterogeneous. Each row reads ITS card's total / limit /
//     ECC mode; nothing is shared across the fleet. A spec that disagrees
//     with the other cards on the host is marked "differs".
//   * Theme tokens only (chat-*). Every state also differs by wording or
//     shape (a tick, a dashed line, a bordered chip, the "not reported"
//     words), never by colour alone, so the table reads without colour.
//     The fault colour is reserved for genuine faults.

import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import {
  NOT_REPORTED,
  archLabel,
  clockReading,
  eccLabel,
  formatOsName,
  formatWattsPlain,
  linkLabel,
  mbToGib,
  nvlinkLabel,
  pcieReading,
  throttleVerdict,
  type GpuLiveCard,
  type GpuLiveResponse,
  type SystemInfo,
  type SystemInfoGpu,
} from "@/lib/system-info";
import { mibToGib } from "@/lib/stats-v2";
import { cn } from "@/lib/utils";
import { StatCard } from "@/components/stat-card";
import { GpuInterconnect, type InterconnectCard } from "@/components/stats/gpu-interconnect";
import { Skeleton } from "@/components/ui/skeleton";

// 30s cadence matches the rest of /stats so a single SWR refresh on
// page-focus revalidates everything together. Pause on hidden tabs.
const REFRESH_MS = 30_000;
// A plain number, not a function returning 0 while hidden: SWR stops its loop
// for good on a 0, so a tab hidden once never polled again. SWR already
// skips ticks while hidden (`refreshWhenHidden` defaults to false).
const refreshInterval = REFRESH_MS;

// Telemetry moves (temperature, power, fan); 10 s is live enough to watch a
// card warm up without stacking nvidia-smi shell-outs — the server side
// collapses bursts under its own 2 s cache anyway.
const LIVE_REFRESH_MS = 10_000;
const liveRefreshInterval = LIVE_REFRESH_MS;

const DASH = "—"; // em-dash — same glyph other tiles use for "missing"

/** Why the live feed has nothing for a card. `ok` means it does. */
type LiveStatus =
  | { kind: "ok" }
  | { kind: "loading" }
  | { kind: "unavailable"; reason: string };

export function SystemConfigSection() {
  const { data, error, isLoading } = useSWR<SystemInfo>(
    "/api/system/info",
    authFetchJSON,
    { refreshInterval },
  );
  const live = useSWR<GpuLiveResponse>("/api/system/gpus", authFetchJSON, {
    refreshInterval: liveRefreshInterval,
    keepPreviousData: true,
  });

  let liveStatus: LiveStatus;
  if (live.data) {
    liveStatus = live.data.probe_error
      ? { kind: "unavailable", reason: live.data.probe_error }
      : { kind: "ok" };
  } else if (live.error) {
    liveStatus = {
      kind: "unavailable",
      reason: live.error instanceof Error ? live.error.message : "request failed",
    };
  } else {
    liveStatus = { kind: "loading" };
  }

  return (
    <section
      aria-label="System configuration"
      data-testid="system-config-section"
      className="space-y-3"
    >
      <h2 className="text-sm font-semibold uppercase tracking-wider text-chat-muted">
        System configuration
      </h2>

      {error && !data ? (
        <p data-testid="system-config-error" className="text-sm text-chat-negative">
          Failed to load system info
          {error instanceof Error ? `: ${error.message}` : "."}
        </p>
      ) : isLoading || !data ? (
        <div className="space-y-3">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
          </div>
        </div>
      ) : (
        <SystemConfigCards
          data={data}
          liveGpus={live.data?.gpus ?? []}
          topology={live.data?.topology ?? null}
          liveStatus={liveStatus}
        />
      )}
    </section>
  );
}

// Split out so the test can render it with a known payload without
// needing to stub SWR — and so the loading/error branches above stay
// simple. Pure-presentational, no hooks.
function SystemConfigCards({
  data,
  liveGpus,
  topology,
  liveStatus,
}: {
  data: SystemInfo;
  liveGpus: GpuLiveCard[];
  topology: GpuLiveResponse["topology"];
  liveStatus: LiveStatus;
}) {
  const cpu = data.cpu;
  const ram = data.ram;

  // The CPU value is the model name itself — physical/threads go into
  // the `hint` line so the tile reads "Intel Xeon E5-2680v4 · 14C/28T"
  // at a glance.
  const cpuValue = cpu ? cpu.model : DASH;
  const cpuHint = cpu ? `${cpu.physical_cores}C / ${cpu.threads}T` : undefined;

  const ramValue = ram ? mbToGib(ram.total_mb) : DASH;
  const ramHint = ram ? `${ram.total_mb.toLocaleString()} MB total` : undefined;
  const ramUnit = ram ? "GiB" : undefined;

  const liveByIndex = new Map(liveGpus.map((g) => [g.index, g]));
  const specs = data.gpus.map((gpu) => cardSpecs(gpu, liveByIndex.get(gpu.index)));
  const majority = majoritySpecs(specs);
  const mixed = data.gpus.length > 1;

  const interconnectCards = new Map<number, InterconnectCard>(
    data.gpus.map((gpu) => {
      const l = liveByIndex.get(gpu.index);
      return [
        gpu.index,
        {
          memory_total_mib: l?.memory_total_mib ?? Math.round(gpu.vram_total_mb),
          width_current: l?.telemetry.pcie.width_current ?? null,
          width_max: l?.telemetry.pcie.width_max ?? null,
        },
      ];
    }),
  );

  return (
    <>
      <div
        data-testid="system-config-top-row"
        className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3"
      >
        <StatCard
          label="CPU"
          value={
            <span
              data-testid="system-cpu-value"
              className="text-base font-medium"
              title={cpu?.model}
            >
              {cpuValue}
            </span>
          }
          hint={cpuHint}
        />
        <StatCard
          label="RAM"
          value={<span data-testid="system-ram-value">{ramValue}</span>}
          unit={ramUnit}
          hint={ramHint}
        />
        <OsDockerCard data={data} />
      </div>

      {data.gpus.length === 0 ? (
        <div
          data-testid="system-gpus-empty"
          className="rounded-md border border-dashed border-chat-rule bg-chat-surface/30 p-4 text-center text-sm text-chat-muted"
        >
          No NVIDIA GPUs detected.
        </div>
      ) : (
        <div
          data-testid="system-gpus-grid"
          className="grid grid-cols-[repeat(auto-fit,minmax(250px,1fr))] gap-3"
        >
          {data.gpus.map((gpu, i) => (
            <GpuCard
              key={gpu.index}
              gpu={gpu}
              live={liveByIndex.get(gpu.index)}
              liveStatus={liveStatus}
              specs={specs[i]}
              majority={mixed ? majority : null}
            />
          ))}
        </div>
      )}

      {/* A relationship BETWEEN cards: only meaningful with two or more. */}
      {data.gpus.length >= 2 && (
        <GpuInterconnect
          topology={topology}
          cards={interconnectCards}
          unavailableReason={liveStatus.kind === "unavailable" ? liveStatus.reason : undefined}
        />
      )}
    </>
  );
}

// Combined OS + Docker card — keeps the top row at exactly 3 tiles so
// the layout doesn't reflow when docker is unavailable.
function OsDockerCard({ data }: { data: SystemInfo }) {
  const os = data.os;
  const docker = data.docker;
  const osLine = formatOsName(os);
  const kernelLine =
    os.kernel && os.kernel !== "unknown" ? `kernel ${os.kernel}` : undefined;

  return (
    <div
      data-testid="system-os-docker-card"
      className="min-w-0 rounded-lg border border-chat-rule bg-chat-surface/50 p-4"
    >
      <p className="text-xs font-semibold uppercase tracking-wider text-chat-dim">
        OS &amp; Docker
      </p>
      <p
        data-testid="system-os-value"
        className="mt-2 break-words text-base font-medium text-chat-fg"
      >
        {osLine}
      </p>
      {kernelLine && (
        <p className="break-words text-xs text-chat-dim" data-testid="system-os-kernel">
          {kernelLine}
        </p>
      )}
      <div className="mt-3 border-t border-chat-rule pt-2 text-xs">
        {docker.available ? (
          <p data-testid="system-docker-available" className="text-chat-muted">
            Docker{" "}
            <span className="font-mono text-chat-fg">{docker.version ?? DASH}</span>
            <span className="text-chat-dim"> · runtime </span>
            <span className="font-mono text-chat-fg">{docker.runtime ?? DASH}</span>
          </p>
        ) : (
          <p
            data-testid="system-docker-unavailable"
            className="text-chat-dim"
            title="docker info shell-out failed — no socket mounted or daemon unreachable"
          >
            Docker not available
          </p>
        )}
      </div>
    </div>
  );
}

// ---- per-card specs and the "differs" mark -------------------------------

/** The values a card is compared to its neighbours on. Every card carries
 *  its own — they are NOT fleet-wide: a mixed host has different models,
 *  memory, power caps and ECC per slot. A null is "not reported" and is
 *  compared as such. */
interface CardSpecs {
  name: string;
  arch: string;
  driver: string;
  cuda: string | null;
  link: string | null;
  nvlink: string | null;
  ecc: string | null;
}

type SpecKey = keyof CardSpecs;
const SPEC_KEYS: SpecKey[] = ["name", "arch", "driver", "cuda", "link", "nvlink", "ecc"];

function cardSpecs(gpu: SystemInfoGpu, live: GpuLiveCard | undefined): CardSpecs {
  const t = live?.telemetry;
  return {
    name: gpu.name,
    arch: live ? archLabel(live.compute_cap, live.architecture) : NOT_REPORTED,
    driver: gpu.driver_version,
    cuda: gpu.cuda_version,
    link: t ? linkLabel(t.pcie) : null,
    nvlink: t && t.nvlink !== null ? nvlinkLabel(t.nvlink) : null,
    ecc: t && t.ecc_enabled !== null ? eccLabel(t.ecc_enabled) : null,
  };
}

/** The most common value per spec across the host's cards. A spec is
 *  marked "differs" only when THIS card disagrees with that, which is the
 *  whole reason the specs live on the card and not in a host header. */
function majoritySpecs(all: CardSpecs[]): Record<SpecKey, string | null> {
  const out = {} as Record<SpecKey, string | null>;
  for (const key of SPEC_KEYS) {
    const counts = new Map<string | null, number>();
    for (const s of all) counts.set(s[key], (counts.get(s[key]) ?? 0) + 1);
    let best: string | null = null;
    let bestCount = -1;
    for (const [value, count] of counts) {
      if (count > bestCount) {
        best = value;
        bestCount = count;
      }
    }
    out[key] = best;
  }
  return out;
}

// ---- GPU card -------------------------------------------------------------

/** The words, not a glyph: a slot the hardware did not report. Carries a
 *  data attribute so tests (and a stylesheet) can find the state without
 *  reading colour. */
function NotReported({ hint }: { hint?: string }) {
  return (
    <span data-state="not-reported" className="italic text-chat-dim" title={hint}>
      {NOT_REPORTED}
    </span>
  );
}

/** " ◂ differs" — this card disagrees with the others on the host. A
 *  glyph plus a word, so it reads without colour. */
function Differs({ testid }: { testid: string }) {
  return (
    <span
      data-testid={testid}
      data-state="differs"
      className="ml-1 font-sans text-[10px] text-chat-warn"
      title="This card's value differs from the other cards on this host."
    >
      ◂ differs
    </span>
  );
}

/** A bordered chip with an icon and explicit words: reads as a flag
 *  without colour. `fault` is reserved for genuine faults. */
function Banner({
  severity,
  title,
  testid,
  children,
}: {
  severity: "warn" | "fault";
  title: string;
  testid: string;
  children: string;
}) {
  return (
    <div
      data-testid={testid}
      data-state={severity === "fault" ? "fault" : "warning"}
      role="status"
      title={title}
      className={cn(
        "flex items-center gap-1.5 rounded-md border bg-chat-surface-2 px-2 py-1 text-[11.5px]",
        severity === "fault"
          ? "border-chat-negative text-chat-negative"
          : "border-chat-warn text-chat-warn",
      )}
    >
      <WarnIcon />
      <span>{children}</span>
    </div>
  );
}

function WarnIcon() {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="flex-none"
    >
      <path d="M8 2.6 14 13H2L8 2.6Z" />
      <path d="M8 6.6v3M8 11.1v.1" />
    </svg>
  );
}

const ROW_ICON = {
  chip: (
    <>
      <rect x="4" y="4" width="8" height="8" rx="1.2" />
      <path d="M6.4 2v2M9.6 2v2M6.4 12v2M9.6 12v2M2 6.4h2M2 9.6h2M12 6.4h2M12 9.6h2" />
    </>
  ),
  bolt: <path d="M8.8 2 4 9h3.4l-.6 5L12 7H8.4l.4-5Z" />,
  fan: (
    <>
      <circle cx="8" cy="8" r="1.3" />
      <path d="M8 6.7c0-2.4.7-4 2.2-4 1 0 1.6.8 1.6 1.8 0 1.6-1.6 2.2-3.8 2.2" />
      <path d="M9.3 8c2.4 0 4 .7 4 2.2 0 1-.8 1.6-1.8 1.6-1.6 0-2.2-1.6-2.2-3.8" />
      <path d="M6.7 8c-2.4 0-4-.7-4-2.2 0-1 .8-1.6 1.8-1.6C6.1 4.2 6.7 5.8 6.7 8" />
    </>
  ),
} as const;

function RowIcon({ kind }: { kind: keyof typeof ROW_ICON }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="text-chat-dim"
    >
      {ROW_ICON[kind]}
    </svg>
  );
}

/** A plain bar against the card's own limit. `work` is the accent, `heat`
 *  the strong accent — the two tones the mockup uses for load and heat. */
function Bar({
  fraction,
  tone,
  label,
  className,
}: {
  fraction: number;
  tone: "work" | "heat";
  label: string;
  className?: string;
}) {
  const clamped = Math.max(0, Math.min(1, fraction));
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped * 100)}
      className={cn("h-1 w-full overflow-hidden rounded-sm bg-chat-surface-2", className)}
    >
      <div
        className={cn(
          "h-full rounded-sm transition-[width] duration-500 ease-out motion-reduce:transition-none",
          tone === "heat" ? "bg-chat-accent-strong" : "bg-chat-accent",
        )}
        style={{ width: `${clamped * 100}%` }}
      />
    </div>
  );
}

/** A metric row: icon, bar, "value / limit unit". Without a limit the bar
 *  is omitted and the value says the limit is not reported. */
function MetricRow({
  icon,
  testid,
  label,
  children,
  bar,
}: {
  icon: keyof typeof ROW_ICON;
  testid: string;
  label: string;
  children: React.ReactNode;
  bar: { fraction: number; tone: "work" | "heat" } | null;
}) {
  return (
    <div className="grid grid-cols-[15px_1fr_auto] items-center gap-2">
      <RowIcon kind={icon} />
      <span className="sr-only">{label}</span>
      {bar ? (
        <Bar fraction={bar.fraction} tone={bar.tone} label={label} />
      ) : (
        <span aria-hidden="true" />
      )}
      <span
        data-testid={testid}
        className="min-w-0 break-words text-right font-mono text-xs tabular-nums text-chat-fg"
      >
        {children}
      </span>
    </div>
  );
}

/** The temperature gauge: a track scaled so the throttle point sits inside
 *  it, with a TICK at that point — the one mark on this card that adds a
 *  fact a number cannot. Full scale is 110 °C unless the throttle point or
 *  the reading is higher. */
function TempGauge({
  temperature_c,
  slowdown_c,
}: {
  temperature_c: number;
  slowdown_c: number | null;
}) {
  const scale = Math.max(110, (slowdown_c ?? 0) + 10, temperature_c + 5);
  const fillPct = Math.min(100, (temperature_c / scale) * 100);
  const tickPct = slowdown_c === null ? null : Math.min(100, (slowdown_c / scale) * 100);
  return (
    <div className={cn("relative", tickPct !== null && "pb-3")}>
      <div
        role="meter"
        aria-label="temperature against this card's throttle point"
        aria-valuemin={0}
        aria-valuemax={scale}
        aria-valuenow={temperature_c}
        className="relative h-2 rounded-sm bg-chat-surface-2"
      >
        <div
          className="h-full rounded-sm bg-chat-accent-strong transition-[width] duration-500 ease-out motion-reduce:transition-none"
          style={{ width: `${fillPct}%` }}
        />
        {tickPct !== null && (
          <>
            <span
              data-testid="system-gpu-temp-tick"
              aria-hidden="true"
              className="absolute -bottom-[3px] -top-[3px] w-0.5 rounded-[1px] bg-chat-negative"
              style={{ left: `${tickPct}%` }}
            />
            <span
              aria-hidden="true"
              className="absolute top-[11px] -translate-x-1/2 whitespace-nowrap text-[9.5px] text-chat-negative"
              style={{ left: `${tickPct}%` }}
            >
              {slowdown_c}° throttle
            </span>
          </>
        )}
      </div>
    </div>
  );
}

/** One spec: label + monospaced value, with the "differs" mark when this
 *  card disagrees with the host's majority. Siblings on a line are divided
 *  by a hairline border, not a glyph. */
function Spec({
  label,
  testid,
  value,
  differs,
  hint,
  note,
}: {
  label: string;
  testid: string;
  value: string | null;
  differs: boolean;
  hint?: string;
  note?: React.ReactNode;
}) {
  return (
    <div className="flex min-w-0 items-baseline gap-1.5 [&+&]:border-l [&+&]:border-chat-rule [&+&]:pl-2.5">
      <span className="flex-none text-[10.5px] text-chat-dim">{label}</span>
      <span className="min-w-0 [overflow-wrap:anywhere] font-mono text-[11.5px] text-chat-muted">
        <span data-testid={testid} className={cn(differs && "text-chat-fg")}>
          {value === null ? <NotReported hint={hint} /> : value}
        </span>
        {differs && <Differs testid={`${testid}-differs`} />}
        {note}
      </span>
    </div>
  );
}

// One card per GPU. Static rows (name, VRAM total, driver, CUDA) come from
// /api/system/info; everything that moves comes from /api/system/gpus and
// is joined by index. Index is the label so a row of cards reads
// "GPU 0 / GPU 1" at a glance.
function GpuCard({
  gpu,
  live,
  liveStatus,
  specs,
  majority,
}: {
  gpu: SystemInfoGpu;
  live: GpuLiveCard | undefined;
  liveStatus: LiveStatus;
  specs: CardSpecs;
  /** null on a single-card host — nothing to differ from. */
  majority: Record<SpecKey, string | null> | null;
}) {
  const t = live?.telemetry;
  const differs = (key: SpecKey) => majority !== null && specs[key] !== majority[key];

  const vramFraction =
    live && live.memory_total_mib > 0 ? live.memory_used_mib / live.memory_total_mib : 0;

  const verdict = t ? throttleVerdict(t.throttle, t.sm_clock_mhz, t.sm_clock_max_mhz) : null;
  const pcie = pcieReading(t?.pcie);
  const widthFault =
    pcie?.widthReduced && t
      ? `Reduced link width — x${t.pcie.width_current} of x${t.pcie.width_max}`
      : null;
  const faulted = widthFault !== null || verdict?.severity === "fault";
  const flagged = !faulted && verdict !== null;

  const core = t ? clockReading(t.sm_clock_mhz, t.sm_clock_max_mhz) : null;
  const mem = t ? clockReading(t.mem_clock_mhz, t.mem_clock_max_mhz) : null;

  return (
    <div
      data-testid="system-gpu-card"
      data-gpu-index={gpu.index}
      data-state={faulted ? "fault" : flagged ? "warning" : undefined}
      className={cn(
        "flex min-w-0 flex-col gap-2 rounded-lg border bg-chat-surface/50 p-4",
        faulted ? "border-chat-negative" : flagged ? "border-chat-warn" : "border-chat-rule",
      )}
    >
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wider text-chat-dim">
          GPU {gpu.index}
        </p>
        {t?.pstate ? (
          <span
            data-testid="system-gpu-pstate"
            className="rounded border border-chat-rule px-1.5 py-0.5 font-mono text-[10px] text-chat-muted"
            title="Performance state — P0 is full clocks, P8 is idle"
          >
            {t.pstate}
          </span>
        ) : null}
      </div>
      <div>
        <p
          data-testid="system-gpu-name"
          className={cn(
            "break-words text-[13.5px] font-medium leading-tight",
            differs("name") ? "text-chat-accent-strong" : "text-chat-fg",
          )}
          title={gpu.name}
        >
          {gpu.name}
          {differs("name") && <Differs testid="system-gpu-name-differs" />}
        </p>
        <p className="mt-0.5 text-[11px] text-chat-dim">
          <span
            data-testid="system-gpu-arch"
            data-state={live && live.architecture === null ? "not-recognised" : undefined}
            className={cn(
              live?.architecture ? "text-chat-muted" : "italic",
              differs("arch") && "text-chat-accent-strong",
            )}
            title={
              live
                ? "Generation derived from the CUDA compute capability — nvidia-smi has no architecture field."
                : "Compute capability comes from the live feed."
            }
          >
            {specs.arch}
          </span>
          {differs("arch") && <Differs testid="system-gpu-arch-differs" />}
        </p>
      </div>

      {liveStatus.kind === "unavailable" && (
        <p
          data-testid="system-gpu-live-unavailable"
          className="text-xs italic text-chat-dim"
          title={liveStatus.reason}
        >
          live telemetry unavailable ({liveStatus.reason})
        </p>
      )}

      {/* Head: the big number beside the two gauges. */}
      <div className="grid grid-cols-[auto_1fr] items-center gap-3">
        <div>
          <div
            data-testid="system-gpu-util"
            className="font-mono text-[34px] font-medium leading-none tabular-nums text-chat-fg"
          >
            {live ? (
              <>
                {live.utilization_pct}
                <span className="text-base text-chat-dim">%</span>
              </>
            ) : (
              <span className="text-xs font-normal">
                <NotReported hint="Utilisation comes from the live feed." />
              </span>
            )}
          </div>
          <div className="mt-0.5 text-[10.5px] text-chat-dim">utilisation</div>
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <div>
            <div className="mb-1 flex items-baseline justify-between gap-2 text-[11px] text-chat-dim">
              <span>load</span>
              <b className="font-mono text-xs font-medium tabular-nums text-chat-fg">
                {live ? `${live.utilization_pct}%` : <NotReported />}
              </b>
            </div>
            {live && (
              <Bar
                fraction={live.utilization_pct / 100}
                tone="work"
                label="utilisation"
                className="h-2"
              />
            )}
          </div>
          <div data-testid="system-gpu-temp">
            <div className="mb-1 flex items-baseline justify-between gap-2 text-[11px] text-chat-dim">
              <span>temperature</span>
              <b className="font-mono text-xs font-medium tabular-nums text-chat-fg">
                {!t || t.temperature_c === null ? (
                  <NotReported hint="The driver reports no temperature sensor for this card." />
                ) : (
                  `${t.temperature_c} °C`
                )}
              </b>
            </div>
            {t && t.temperature_c !== null && (
              <>
                <TempGauge temperature_c={t.temperature_c} slowdown_c={t.temp_slowdown_c} />
                {t.temp_slowdown_c === null && (
                  <span className="block text-[9.5px] italic text-chat-dim">
                    throttle point {NOT_REPORTED}
                  </span>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* The verdict, from the driver. Fault first: it is the bigger problem. */}
      {widthFault !== null && (
        <Banner
          severity="fault"
          testid="system-gpu-fault"
          title="The card negotiated fewer lanes than it supports — a narrower slot or a riser. Width does not downshift at idle, so this is permanent, and without NVLink it is the only path between cards for tensor-parallel serving."
        >
          {widthFault}
        </Banner>
      )}
      {verdict !== null && (
        <Banner
          severity={verdict.severity}
          testid="system-gpu-throttle"
          title={throttleHint(verdict.reasons)}
        >
          {verdict.text}
        </Banner>
      )}
      {t && t.throttle === null && (
        <p
          data-testid="system-gpu-throttle-not-reported"
          className="text-[10.5px] italic text-chat-dim"
          title="The driver did not answer the clocks_throttle_reasons query — the clocks may still be held; nothing here can say."
        >
          throttle reasons {NOT_REPORTED}
        </p>
      )}

      {/* VRAM / power / fan, each against ITS card's total or limit. */}
      <div className="flex flex-col gap-1.5">
        <MetricRow
          icon="chip"
          testid="system-gpu-vram"
          label="VRAM used of total"
          bar={live ? { fraction: vramFraction, tone: "work" } : null}
        >
          {live ? (
            <>
              {mibToGib(live.memory_used_mib)}
              <span className="text-chat-dim"> / {mibToGib(live.memory_total_mib)} GiB</span>
            </>
          ) : (
            <>
              {mbToGib(gpu.vram_total_mb)} GiB
              <span className="text-chat-dim"> · usage {NOT_REPORTED}</span>
            </>
          )}
        </MetricRow>
        <MetricRow
          icon="bolt"
          testid="system-gpu-power"
          label="power draw against this card's limit"
          bar={
            t && t.power_w !== null && t.power_limit_w !== null && t.power_limit_w > 0
              ? { fraction: t.power_w / t.power_limit_w, tone: "heat" }
              : null
          }
        >
          {!t || t.power_w === null ? (
            <>
              <span className="text-chat-dim">power </span>
              <NotReported hint="No power sensor reported for this card." />
            </>
          ) : t.power_limit_w === null ? (
            <>
              {formatWattsPlain(t.power_w)} W
              <span className="text-chat-dim"> · limit {NOT_REPORTED}</span>
            </>
          ) : (
            <>
              {formatWattsPlain(t.power_w)}
              <span className="text-chat-dim"> / {formatWattsPlain(t.power_limit_w)} W</span>
            </>
          )}
        </MetricRow>
        <MetricRow
          icon="fan"
          testid="system-gpu-fan"
          label="fan speed"
          bar={t && t.fan_pct !== null ? { fraction: t.fan_pct / 100, tone: "work" } : null}
        >
          {!t || t.fan_pct === null ? (
            <>
              <span className="text-chat-dim">fan </span>
              <NotReported hint="No fan speed from the driver — a passively cooled card, a card in a VM, or no sensor." />
            </>
          ) : (
            <>
              {t.fan_pct}
              <span className="text-chat-dim">% fan</span>
            </>
          )}
        </MetricRow>
      </div>

      {/* Clocks against their own ceilings. */}
      <div className="grid grid-cols-2 gap-2 border-t border-chat-rule pt-2">
        <div>
          <div className="text-[10.5px] text-chat-dim">Core clock</div>
          <div
            data-testid="system-gpu-clock"
            className={cn(
              "font-mono text-xs tabular-nums",
              verdict !== null ? "text-chat-accent-strong" : "text-chat-fg",
            )}
          >
            {core === null ? (
              <NotReported hint="No SM clock reported for this card." />
            ) : (
              <>
                {core.value}
                <span className="text-[11px] text-chat-dim">{core.suffix}</span>
              </>
            )}
          </div>
        </div>
        <div>
          <div className="text-[10.5px] text-chat-dim">Memory clock</div>
          <div data-testid="system-gpu-mem-clock" className="font-mono text-xs tabular-nums text-chat-fg">
            {mem === null ? (
              <NotReported hint="No memory clock reported for this card." />
            ) : (
              <>
                {mem.value}
                <span className="text-[11px] text-chat-dim">{mem.suffix}</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Specs, paired: Driver | CUDA; Link on its own (its value wraps);
          NVLink | ECC. Each pair wraps independently on a narrow card. */}
      <div className="flex flex-col gap-1 border-t border-chat-rule pt-2">
        <div data-testid="system-gpu-spec-line" className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
          <Spec label="Driver" testid="system-gpu-driver" value={specs.driver} differs={differs("driver")} />
          <Spec
            label="CUDA"
            testid="system-gpu-cuda"
            value={specs.cuda}
            differs={differs("cuda")}
            hint="nvidia-smi printed no CUDA version line."
          />
        </div>
        <div data-testid="system-gpu-spec-line" className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
          <Spec
            label="Link"
            testid="system-gpu-pcie"
            value={specs.link}
            differs={differs("link")}
            hint="The driver reports no PCIe link fields for this card."
            note={
              pcie?.capabilityNote ? (
                <span
                  data-testid="system-gpu-pcie-caps"
                  className="block font-sans text-[10px] text-chat-dim"
                  title="Both endpoints can do more than the link negotiated — the slot or riser sets the ceiling. The negotiated figure is what the link will actually do."
                >
                  {pcie.capabilityNote}
                </span>
              ) : null
            }
          />
        </div>
        <div data-testid="system-gpu-spec-line" className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
          <Spec
            label="NVLink"
            testid="system-gpu-nvlink"
            value={specs.nvlink}
            differs={differs("nvlink")}
            hint="nvidia-smi nvlink -s said nothing about this card."
          />
          <Spec
            label="ECC"
            testid="system-gpu-ecc"
            value={specs.ecc}
            differs={differs("ecc")}
            hint="This card reports no ECC memory mode."
          />
        </div>
      </div>
    </div>
  );
}

function throttleHint(reasons: string[]): string {
  const parts: string[] = [];
  for (const r of reasons) {
    if (r === "Thermal slowdown") {
      parts.push("SW thermal slowdown: the driver is lowering clocks to hold temperature — airflow or ambient.");
    } else if (r === "HW thermal slowdown") {
      parts.push("HW thermal slowdown: the silicon itself cut clocks by half or more on temperature — check cooling now.");
    } else if (r === "Power cap") {
      parts.push("Power cap: clocks held to stay under this card's power limit — raise the limit or accept it.");
    } else if (r === "HW power brake") {
      parts.push("HW power brake: the board's external power-brake signal is asserted — PSU or board power delivery.");
    } else if (r === "HW slowdown") {
      parts.push("HW slowdown: the silicon cut clocks by half or more for a reason it did not name.");
    }
  }
  return parts.join(" ");
}
