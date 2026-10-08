"use client";

// Recent router decisions (GET /api/router/decisions), newest first, with the
// route filter of the Activity page (plan §4.8). The backend records the key's
// *name* only; no credential reaches this payload.
//
// Tables here and in router-stats.tsx stack under 760 px: the same DOM, rows
// become 2-column grids and each cell repeats its column header from
// `data-label` (CSS only).

import type { components } from "@/lib/api-types.generated";
import { fmtInt } from "@/lib/router";
import { FOCUS, ROUTE_LABEL, ROUTE_SWATCH, routeTone } from "./styles";

export type Decision = components["schemas"]["DecisionOut"];

export function fmtMs(v: number | null | undefined): string {
  return v == null ? "—" : `${Math.round(v).toLocaleString("en-US")} ms`;
}

/** "820 / 2,900 ms"; a missing side is "—"; both missing is a bare "—". */
export function fmtMsPair(p50: number | null | undefined, p95: number | null | undefined): string {
  if (p50 == null && p95 == null) return "—";
  const f = (v: number | null | undefined) => (v == null ? "—" : fmtInt(Math.round(v)));
  return `${f(p50)} / ${f(p95)} ms`;
}

function fmtTime(ts: string): string {
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleTimeString("en-GB", { hour12: false });
}

// ── Stacked tables (shared with router-stats.tsx) ───────────────────────────
// mockup: table.t (12.5px, th 11.5 uppercase dim, td 8px 10px, rule-soft
// hairlines) and table.t.stack under 760 px.
export const TABLE =
  "w-full border-collapse text-left text-[12.5px] max-[759px]:block max-[759px]:[&_thead]:hidden max-[759px]:[&_tbody]:block";
export const TH =
  "whitespace-nowrap border-b border-vw-rule-soft/60 px-2.5 py-2 text-left text-[11.5px] font-semibold uppercase tracking-[0.05em] text-chat-dim";
export const TH_N = `${TH} text-right`;
export const TR =
  "max-[759px]:grid max-[759px]:grid-cols-[auto_minmax(0,1fr)] max-[759px]:gap-x-2.5 max-[759px]:gap-y-0.5 max-[759px]:border-b max-[759px]:border-vw-rule-soft/45 max-[759px]:py-2.5";
export const TD =
  "border-b border-vw-rule-soft/35 px-2.5 py-2 align-top max-[759px]:border-0 max-[759px]:p-0 max-[759px]:text-left max-[759px]:before:mr-1.5 max-[759px]:before:text-chat-dim max-[759px]:before:content-[attr(data-label)]";
export const TD_N = `${TD} text-right tabular-nums`;
/** The row's title cell: full width when stacked, no label. */
export const TD_HEAD = `${TD} [overflow-wrap:anywhere] max-[759px]:col-span-full`;
export const TD_EMPTY = "px-2.5 py-3 text-[12.5px] text-chat-muted max-[759px]:block";
// Keep the wide tables usable between 760 px and their natural width.
export const TABLE_WRAP = "min-[760px]:overflow-x-auto";

// ── Route filter ────────────────────────────────────────────────────────────

export type RouteFilter = "all" | "local" | "passthrough" | "fallback" | "refused";

interface FilterDef {
  id: RouteFilter;
  label: string;
  /** Lower-case words for "No {word} decisions in the last 100." */
  word: string;
  match: (route: string) => boolean;
}

export const ROUTE_FILTERS: readonly FilterDef[] = [
  { id: "all", label: "All", word: "", match: () => true },
  { id: "local", label: "Local", word: "local", match: (r) => r === "local" },
  { id: "passthrough", label: "Anthropic", word: "Anthropic", match: (r) => r === "passthrough" },
  { id: "fallback", label: "Fell back", word: "fell back", match: (r) => r === "fallback" },
  { id: "refused", label: "Refused · errors", word: "refused or error", match: (r) => r === "refused" || r === "error" },
];

const FILTER_BY_ID = new Map(ROUTE_FILTERS.map((f) => [f.id, f]));

/** `?route=` → a filter; unknown values are "all". `error` folds into refused. */
export function parseRouteFilter(v: string | null | undefined): RouteFilter {
  if (v === "error") return "refused";
  return v && FILTER_BY_ID.has(v as RouteFilter) ? (v as RouteFilter) : "all";
}

export function filterDecisions(decisions: Decision[], filter: RouteFilter): Decision[] {
  const f = FILTER_BY_ID.get(filter) ?? ROUTE_FILTERS[0];
  return filter === "all" ? decisions : decisions.filter((d) => f.match(d.route));
}

// .fchip { border rule-soft, radius 999, padding 3px 11px, 12.5px }  [aria-pressed=true] surface-2, 600
const CHIP =
  `inline-flex min-h-8 items-center rounded-full border px-[11px] py-[3px] text-[12.5px] tabular-nums ${FOCUS}`;
const CHIP_OFF = "border-vw-rule-soft bg-transparent text-chat-fg hover:bg-chat-surface-2/60";
const CHIP_ON = "border-chat-muted/60 bg-chat-surface-2 font-semibold text-chat-fg";

export function RouteFilterChips({
  decisions,
  value,
  onChange,
}: {
  decisions: Decision[];
  value: RouteFilter;
  onChange: (f: RouteFilter) => void;
}) {
  return (
    <div role="group" aria-label="Filter by route" className="flex flex-wrap gap-2">
      {ROUTE_FILTERS.map((f) => {
        const n = f.id === "all" ? decisions.length : decisions.filter((d) => f.match(d.route)).length;
        const on = value === f.id;
        return (
          <button
            key={f.id}
            type="button"
            aria-pressed={on}
            onClick={() => onChange(f.id)}
            className={`${CHIP} ${on ? CHIP_ON : CHIP_OFF}`}
          >
            {`${f.label} ${fmtInt(n)}`}
          </button>
        );
      })}
    </div>
  );
}

// ── Table ───────────────────────────────────────────────────────────────────

export function RouteLabel({ route }: { route: string }) {
  const t = routeTone(route);
  return (
    <span className={`${ROUTE_LABEL} ${t.fg}`}>
      <span aria-hidden="true" className={ROUTE_SWATCH} />
      {t.label}
    </span>
  );
}

function modelText(d: Decision): string {
  if (!d.model_in) return d.model_out ?? "—";
  return d.model_out && d.model_out !== d.model_in ? `${d.model_in} → ${d.model_out}` : d.model_in;
}

const HEADERS: Array<[string, boolean]> = [
  ["Time", false], ["Model", false], ["Route", false], ["Reason", false],
  ["Status", true], ["Latency", true], ["TTFB", true], ["Key", false],
];

export function DecisionsTable({
  decisions,
  filter = "all",
}: {
  /** The full list; the filter applies here. */
  decisions: Decision[];
  filter?: RouteFilter;
}) {
  const rows = filterDecisions(decisions, filter);
  const word = FILTER_BY_ID.get(filter)?.word ?? "";
  return (
    <div className={TABLE_WRAP}>
      <table data-testid="router-decisions" className={TABLE}>
        <thead>
          <tr>
            {HEADERS.map(([h, n]) => (
              <th key={h} scope="col" className={n ? TH_N : TH}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr className="max-[759px]:block">
              <td colSpan={HEADERS.length} className={TD_EMPTY}>
                {decisions.length === 0 || filter === "all"
                  ? "No decisions yet."
                  : `No ${word} decisions in the last ${fmtInt(decisions.length)}.`}
              </td>
            </tr>
          ) : rows.map((d, i) => (
            <tr key={`${d.ts}-${i}`} className={TR}>
              <td className={`${TD} whitespace-nowrap tabular-nums`} title={d.ts}>
                {fmtTime(d.ts)}
              </td>
              <td className={`${TD} [overflow-wrap:anywhere]`}>
                <span className="!font-mono">{modelText(d)}</span>
                <span className="block text-[11.5px] text-chat-dim">
                  <span className="!font-mono">{d.path}</span>
                  {` · ${d.stream ? "stream" : "non-stream"}`}
                </span>
              </td>
              <td className={TD} data-label="Route"><RouteLabel route={d.route} /></td>
              <td className={TD} data-label="Reason">
                {d.reason ? <code className="font-mono">{d.reason}</code> : "—"}
              </td>
              <td className={TD_N} data-label="Status">{d.status}</td>
              <td className={TD_N} data-label="Latency">{fmtMs(d.latency_ms)}</td>
              <td className={TD_N} data-label="TTFB">{fmtMs(d.ttfb_ms)}</td>
              <td className={`${TD} [overflow-wrap:anywhere]`} data-label="Key">{d.token_name ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
