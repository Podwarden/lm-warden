"use client";

// The weekly (or daily, or per-version) medians chart (mockup `trends.html`,
// `dataviz` conventions of components/stats/v2-charts.tsx): one line per score
// on a fixed 0-100 axis, the overall's 25th-75th band, version boundaries as
// dashed vertical markers, direct labels at the line ends, a native tooltip on
// every point. A bucket with fewer than three scored sessions has null medians
// (A-7): the line breaks there and a dotted tick says "not scored", never a 0.
// The model review's own small chart is not here: it arrives with the review.

import { bucketLabel, METRICS, METRIC_NAME, type Metric, type TrendBy, type TrendPoint, type VersionInfo } from "@/lib/prompt-lens/trends";

const W = 760;
const H = 300;
const L = 36;
const R = 640;
const T = 16;
const B = 264;

const COLOR: Record<Metric, string> = {
  overall: "text-chat-accent",
  E: "text-vw-ttft",
  C: "text-vw-dur",
  R: "text-vw-completion",
};

export interface TrendChartProps {
  points: TrendPoint[];
  versions: VersionInfo[];
  by: TrendBy;
  /** Metrics drawn (default all four). */
  show?: readonly Metric[];
}

const yOf = (v: number) => B - (v / 100) * (B - T);

/** Runs of consecutive points for which `ok` holds (a gap ends a run). */
function runsOf(points: TrendPoint[], ok: (p: TrendPoint) => boolean): TrendPoint[][] {
  const runs: TrendPoint[][] = [];
  let cur: TrendPoint[] = [];
  for (const p of points) {
    if (ok(p)) cur.push(p);
    else if (cur.length) {
      runs.push(cur);
      cur = [];
    }
  }
  if (cur.length) runs.push(cur);
  return runs;
}

/** Where each line's end label sits: at its last point, pushed apart so two
 *  lines that end close together stay readable (at least `gap` px between). */
function dodge(ends: { m: Metric; y: number }[], gap = 14): Map<Metric, number> {
  const sorted = [...ends].sort((a, b) => a.y - b.y);
  const out = new Map<Metric, number>();
  let prev = -Infinity;
  for (const e of sorted) {
    const y = Math.max(e.y, prev + gap);
    out.set(e.m, y);
    prev = y;
  }
  return out;
}

export function TrendChart({ points, versions, by, show = METRICS }: TrendChartProps) {
  const t0 = points.length ? points[0].start : 0;
  const t1 = points.length ? points[points.length - 1].start : 0;
  const xOf = (t: number) => (t1 > t0 ? L + ((t - t0) / (t1 - t0)) * (R - L) : (L + R) / 2);
  const every = Math.max(1, Math.ceil(points.length / 9));

  const marks = versions
    .slice(1)
    .filter((v) => by !== "version" && v.since >= t0 && v.since <= t1)
    .map((v) => ({ id: v.id, label: v.label, x: xOf(v.since) }));
  const first = points.find((p) => p.medians.overall !== null);
  const last = [...points].reverse().find((p) => p.medians.overall !== null);
  const label =
    first && last
      ? `Median scores per ${by}: overall ${first.medians.overall} at the start, ${last.medians.overall} at the end`
      : `Median scores per ${by}`;

  const endOf = (m: Metric) => {
    const runs = runsOf(points, (p) => p.medians[m] !== null);
    const lastRun = runs[runs.length - 1];
    return lastRun ? lastRun[lastRun.length - 1] : null;
  };
  const labelY = dodge(
    show.flatMap((m) => {
      const e = endOf(m);
      return e ? [{ m, y: yOf(e.medians[m] ?? 0) + 4 }] : [];
    }),
  );

  return (
    <div className="overflow-x-auto" data-testid="trend-chart-box">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} className="block h-auto w-full min-w-[620px]" data-testid="trend-chart">
        {[0, 25, 50, 75, 100].map((v) => (
          <g key={v}>
            <line x1={L} x2={R} y1={yOf(v)} y2={yOf(v)} className="stroke-chat-rule" strokeWidth={1} />
            <text x={L - 8} y={yOf(v) + 4} fontSize={11} textAnchor="end" className="fill-chat-dim">
              {v}
            </text>
          </g>
        ))}
        {points.map((p, i) =>
          i % every === 0 ? (
            <text key={p.bucket} x={xOf(p.start)} y={B + 20} fontSize={11} textAnchor="middle" className="fill-chat-dim">
              {bucketLabel(p, by)}
            </text>
          ) : null,
        )}
        {marks.map((m) => (
          <g key={m.id} data-testid="version-marker" data-version={m.id}>
            <line x1={m.x} x2={m.x} y1={T} y2={B} strokeDasharray="4 4" className="stroke-chat-muted" opacity={0.6} />
            <text x={m.x + 4} y={T + 10} fontSize={11} className="fill-chat-muted">
              {m.label}
            </text>
          </g>
        ))}
        {runsOf(points, (p) => !!p.band).map((r, k) =>
          r.length > 1 ? (
            <path
              key={k}
              data-testid="trend-band"
              className="fill-chat-accent"
              opacity={0.12}
              d={`M${r.map((p) => `${xOf(p.start)},${yOf(p.band?.[1] ?? 0)}`).join("L")}L${[...r]
                .reverse()
                .map((p) => `${xOf(p.start)},${yOf(p.band?.[0] ?? 0)}`)
                .join("L")}Z`}
            />
          ) : null,
        )}
        {points
          .filter((p) => p.medians.overall === null)
          .map((p) => (
            <g key={p.bucket} data-testid="trend-gap" data-bucket={p.bucket}>
              <line x1={xOf(p.start)} x2={xOf(p.start)} y1={T} y2={B} strokeDasharray="1 5" className="stroke-chat-rule" />
              <title>{`${bucketLabel(p, by)}: not scored (${p.n} ${p.n === 1 ? "session" : "sessions"}, 3 needed)`}</title>
            </g>
          ))}
        {show.map((m) => {
          const runs = runsOf(points, (p) => p.medians[m] !== null);
          const end = endOf(m);
          const val = (p: TrendPoint) => p.medians[m] ?? 0;
          return (
            <g key={m} className={COLOR[m]} data-testid={`trend-line-${m}`}>
              {runs
                .filter((r) => r.length > 1)
                .map((r, k) => (
                  <path
                    key={k}
                    d={`M${r.map((p) => `${xOf(p.start)},${yOf(val(p))}`).join("L")}`}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={m === "overall" ? 2.5 : 2}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                ))}
              {runs.flat().map((p) => (
                <circle key={p.bucket} cx={xOf(p.start)} cy={yOf(val(p))} r={4} fill="currentColor" className="stroke-chat-surface" strokeWidth={2}>
                  <title>{`${METRIC_NAME[m]} · ${bucketLabel(p, by)}: ${p.medians[m]} (${p.n} sessions)`}</title>
                </circle>
              ))}
              {end && (
                <text x={R + 10} y={labelY.get(m) ?? yOf(val(end)) + 4} fontSize={12} className="fill-chat-fg">
                  {METRIC_NAME[m]} {end.medians[m]}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
