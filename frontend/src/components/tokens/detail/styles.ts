// Class strings that reproduce the approved mockup
// (docs/superpowers/specs/2026-09-18-token-details-mockup.html) through theme
// tokens. Every string here is a COMPLETE literal so Tailwind's scanner sees it.
// The mockup CSS each helper copies is quoted next to it.

export const VW_TOKENS = [
  "rule-soft", "btn", "btn-hover", "on-btn", "danger", "danger-fg", "danger-bg",
  "amber-bg", "amber-fg", "band-fg", "ok-bg", "ok-fg", "live", "info-bg", "info-fg",
  "low-bg", "low-fg", "dock", "prompt", "completion", "requests", "queue", "ttft",
  "dur", "strip-chain", "strip-own",
  "model-1", "model-2", "model-3", "model-4", "model-5", "model-6",
] as const;

// .btn { border-radius: 6px; padding: 7px 14px; font-weight: 500; font-size: 13px;
//        border: 1px solid var(--rule); background: transparent; color: var(--fg); }
// .btn:hover { background: var(--surface); }  .btn:disabled { opacity: .45; cursor: not-allowed; }
const BTN_BASE =
  "inline-flex items-center justify-center whitespace-nowrap rounded-md border px-3.5 py-[7px] text-[13px] font-medium leading-[1.5] cursor-pointer disabled:cursor-not-allowed disabled:opacity-45";

export type BtnVariant = "default" | "primary" | "danger" | "pause";

const BTN: Record<BtnVariant, string> = {
  default: "border-chat-rule bg-transparent text-chat-fg hover:bg-chat-surface",
  // .btn-primary { background: var(--btn); border-color: var(--btn); } :hover { background: var(--btn-hover); }
  primary: "border-vw-btn bg-vw-btn text-vw-on-btn hover:bg-vw-btn-hover",
  // .btn-danger { border-color: rgba(248,113,113,.45); color: #fca5a5; } :hover { background: rgba(127,29,29,.35); }
  danger: "border-vw-danger/45 bg-transparent text-vw-danger-fg hover:bg-vw-danger-bg/35",
  // .btn-pause { border-color: rgba(251,191,36,.5); color: var(--amber); } :hover { background: rgba(120,53,15,.35); }
  pause: "border-chat-accent/50 bg-transparent text-chat-accent hover:bg-vw-amber-bg/35",
};

export function btn(variant: BtnVariant = "default", extra?: string): string {
  return extra ? `${BTN_BASE} ${BTN[variant]} ${extra}` : `${BTN_BASE} ${BTN[variant]}`;
}

// .badge { border-radius: 9999px; padding: 2px 10px; font-size: 12px; font-weight: 500; white-space: nowrap; }
const BADGE_BASE = "whitespace-nowrap rounded-full px-2.5 py-0.5 text-[12px] font-medium leading-[1.5]";

export type BadgeTone = "active" | "paused" | "grace" | "revoked";

const BADGE: Record<BadgeTone, string> = {
  active: "bg-vw-ok-bg/50 text-vw-ok-fg",          // .b-active  rgba(6,78,59,.5)   #6ee7b7
  paused: "bg-vw-amber-bg/60 text-vw-amber-fg",    // .b-paused  rgba(120,53,15,.6) #fcd34d
  grace: "bg-vw-amber-bg/35 text-vw-amber-fg",     // .b-grace   rgba(120,53,15,.35) #fcd34d
  revoked: "bg-vw-danger-bg/50 text-vw-danger-fg", // .b-revoked rgba(127,29,29,.5) #fca5a5
};

export function badge(tone: BadgeTone): string {
  return `${BADGE_BASE} ${BADGE[tone]}`;
}

// Mockup PRIO table: 0-2 #fcd9b6/rgba(92,58,24,.6), 3-6 #93c5fd/rgba(30,58,138,.5),
// 7-8 #fcd34d/rgba(120,53,15,.5), 9 #fca5a5/rgba(127,29,29,.5).
export function prioTone(p: number): { fg: string; bg: string } {
  if (p >= 9) return { fg: "rgb(var(--vw-danger-fg))", bg: "rgb(var(--vw-danger-bg) / .5)" };
  if (p >= 7) return { fg: "rgb(var(--vw-amber-fg))", bg: "rgb(var(--vw-amber-bg) / .5)" };
  if (p >= 3) return { fg: "rgb(var(--vw-info-fg))", bg: "rgb(var(--vw-info-bg) / .5)" };
  return { fg: "rgb(var(--vw-low-fg))", bg: "rgb(var(--vw-low-bg) / .6)" };
}

// SVG strokes/fills and inline styles. CSS variables resolve in SVG
// presentation attributes in every supported browser.
export const SERIES = {
  prompt: "rgb(var(--vw-prompt))",
  completion: "rgb(var(--vw-completion))",
  requests: "rgb(var(--vw-requests))",
  queue: "rgb(var(--vw-queue))",
  ttft: "rgb(var(--vw-ttft))",
  dur: "rgb(var(--vw-dur))",
  grid: "rgb(var(--chat-rule))",     // #4a3820
  axis: "rgb(var(--chat-dim))",      // #8a7c66
  muted: "rgb(var(--chat-muted))",   // #c4b49a
  dim: "rgb(var(--chat-dim))",
  stripChain: "rgb(var(--vw-strip-chain))",
  stripOwn: "rgb(var(--vw-strip-own))",
} as const;

// Per-model series colours (tokens chart lines, usage-by-model swatches and
// share bars), in `by_model` order; the seventh model reuses the first.
export const MODEL_COLORS = [
  "rgb(var(--vw-model-1))",
  "rgb(var(--vw-model-2))",
  "rgb(var(--vw-model-3))",
  "rgb(var(--vw-model-4))",
  "rgb(var(--vw-model-5))",
  "rgb(var(--vw-model-6))",
] as const;

export function modelColor(index: number): string {
  return MODEL_COLORS[index % MODEL_COLORS.length];
}

/** How the model at `index` is told apart once the six colours run out:
 *  models 1-6 solid, 7-12 dashed, 13-18 dotted (then it repeats -- a key that
 *  used nineteen models in one period is not a case worth a fourth style).
 *  `dash` is an SVG stroke-dasharray; `swatch` the card/legend marker. */
export type ModelMark = "solid" | "dashed" | "dotted";
const MARKS: readonly ModelMark[] = ["solid", "dashed", "dotted"];
const DASH: Record<ModelMark, string | undefined> = { solid: undefined, dashed: "5 4", dotted: "1.5 3" };

export function modelStyle(index: number): { color: string; mark: ModelMark; dash: string | undefined } {
  const mark = MARKS[Math.floor(index / MODEL_COLORS.length) % MARKS.length];
  return { color: modelColor(index), mark, dash: DASH[mark] };
}

// The mockup's non-mono text stack: "DM Sans", system-ui, sans-serif.
export const FONT_SANS = "font-['DM_Sans',system-ui,sans-serif]";

// .dock-bar / .stream horizontal padding: max(16px, calc((100vw - 1250px) / 2)).
export const GUTTER_X = "px-[max(16px,calc((100vw_-_1250px)/2))]";

// .card { background: var(--surface); border: 1px solid var(--rule-soft); border-radius: 8px; padding: 16px; }
export const CARD = "rounded-lg border border-vw-rule-soft/50 bg-chat-surface p-4";
// .card h2 { font-size: 14px; font-weight: 600; margin: 0 0 12px; }
export const CARD_TITLE = "mb-3 mt-0 text-[14px] font-semibold";
