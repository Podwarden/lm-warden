"use client";

// "Your requests chart" (design 4, item 4): the key holder's own finished
// requests of the scope model. x = finish time, y = duration (log), size =
// generated tokens, shape = finish reason. With Prompt Lens on (D8, `promptLens`)
// a dot opens that request on the Prompt Lens page (`onView`).

import { type SpeedPoint, fmtSeconds, fmtTokens } from "@/lib/stats-redesign";

import { hueOf } from "./primitives";

const W = 800;
const H = 230;
const PAD = { l: 52, r: 12, t: 10, b: 26 };

export function KeyRequestsChart({
  points,
  sinceEpoch,
  nowEpoch,
  promptLens,
  onView,
}: {
  points: readonly SpeedPoint[];
  sinceEpoch: number;
  nowEpoch: number;
  promptLens: boolean;
  onView: (id: string) => void;
}) {
  const pts = points.filter((p) => p.duration != null && p.duration > 0);
  if (!pts.length) {
    return (
      <svg data-testid="key-requests-chart" viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full">
        <rect x={PAD.l} y={10} width={W - 64} height={H - 36} fill="none" stroke="currentColor" className="text-chat-rule" strokeDasharray="3 4" />
        <text x={W / 2} y={H / 2} textAnchor="middle" className="fill-chat-dim text-[13px]">
          No requests in window
        </text>
      </svg>
    );
  }
  const ds = pts.map((p) => p.duration as number);
  const lo = 10 ** Math.floor(Math.log10(Math.max(1e-3, Math.min(...ds))));
  const hi = 10 ** Math.ceil(Math.log10(Math.max(...ds) * 1.01));
  const y = (v: number) => PAD.t + (H - PAD.t - PAD.b) * (1 - (Math.log10(v) - Math.log10(lo)) / Math.max(1e-9, Math.log10(hi) - Math.log10(lo)));
  const span = Math.max(1, nowEpoch - sinceEpoch);
  const x = (t: number) => PAD.l + ((W - PAD.l - PAD.r) * (t - sinceEpoch)) / span;
  const ticks: number[] = [];
  for (let e = Math.log10(lo); e <= Math.log10(hi) + 1e-9; e++) ticks.push(10 ** e);
  return (
    <svg data-testid="key-requests-chart" viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label="your requests">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="currentColor" className="text-chat-rule/60" />
          <text x={PAD.l - 6} y={y(t) + 3} textAnchor="end" className="fill-chat-dim text-[10px]">
            {fmtSeconds(t)}
          </text>
        </g>
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <text key={f} x={x(sinceEpoch + f * span)} y={H - 8} textAnchor="middle" className="fill-chat-dim text-[10px]">
          {f === 1 ? "now" : `-${Math.round(((1 - f) * span) / 60)}m`}
        </text>
      ))}
      {pts.map((p) => {
        const cx = x(p.finished_at);
        const cy = y(p.duration as number);
        const s = 4 + Math.min(6, Math.log10(p.completion_tokens + 1) * 1.6);
        const c = hueOf(p.model_id);
        const props = promptLens
          ? {
              "data-testid": "chart-view",
              role: "button",
              "aria-label": "Open in Prompt Lens",
              style: { cursor: "pointer" },
              onClick: () => onView(p.id),
            }
          : {};
        const title = (
          <title>
            {fmtSeconds(p.duration)} · {fmtTokens(p.completion_tokens)} generated · {p.finish_reason ?? "-"}
            {promptLens ? " · click to open in Prompt Lens" : ""}
          </title>
        );
        if (p.finish_reason === "stop" || p.finish_reason === "tool_calls")
          return (
            <circle key={p.id} cx={cx} cy={cy} r={s} fill={c} stroke="rgb(var(--vw-surface, 42 31 14))" strokeWidth={2} {...props}>
              {title}
            </circle>
          );
        if (p.finish_reason === "length")
          return (
            <circle key={p.id} cx={cx} cy={cy} r={s} fill="none" stroke={c} strokeWidth={2} pointerEvents="all" {...props}>
              {title}
            </circle>
          );
        return (
          <path key={p.id} d={`M${cx - s} ${cy - s} L${cx + s} ${cy + s} M${cx + s} ${cy - s} L${cx - s} ${cy + s}`} stroke={c} strokeWidth={3} {...props}>
            {title}
          </path>
        );
      })}
    </svg>
  );
}
