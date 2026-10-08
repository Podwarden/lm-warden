// Class strings for the router overview and Activity pages (router page
// redesign, plan §5). They reproduce the approved mockup
// (scratchpad router-redesign/mockup.html) through theme tokens only — no
// slate/emerald literals, so retro and retro-dark need no overrides. Every
// string is a COMPLETE literal so Tailwind's scanner sees it. The mockup CSS
// each one copies is quoted next to it. Buttons reuse `btn()` from the token
// detail page.

import type { StripTone } from "@/lib/router";

export { btn } from "@/components/tokens/detail/styles";

export type Tone = "ok" | "amber" | "danger" | "info" | "idle";

// .pill { border-radius: 999px; padding: 1px 9px; font-size: 12px; font-weight: 600; gap: 6px }
const PILL_BASE =
  "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-[9px] py-px text-[12px] font-semibold leading-[1.5]";

const PILL: Record<Tone, string> = {
  ok: "bg-vw-ok-bg/55 text-vw-ok-fg", //          .pill.ok
  amber: "bg-vw-amber-bg/60 text-vw-amber-fg", // .pill.warn
  danger: "bg-vw-danger-bg/55 text-vw-danger-fg", // .pill.bad
  info: "bg-vw-info-bg/50 text-vw-info-fg", //    .pill.info
  idle: "bg-chat-surface-2 text-chat-muted", //    .pill.idle
};

export function pill(tone: Tone): string {
  return `${PILL_BASE} ${PILL[tone]}`;
}

// .pill .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor }
export const PILL_DOT = "h-[7px] w-[7px] flex-none rounded-full bg-current";

// Text colour per tone, for counts and route labels (.r.local / .r.pass / .r.fb / .r.ref).
export const TONE_FG: Record<Tone, string> = {
  ok: "text-vw-ok-fg",
  amber: "text-vw-amber-fg",
  danger: "text-vw-danger-fg",
  info: "text-vw-info-fg",
  idle: "text-chat-muted",
};

// Bar / swatch fill per tone (traffic split, legend, mini bars).
export const TONE_BAR: Record<Tone, string> = {
  ok: "bg-vw-live",
  amber: "bg-vw-amber-fg",
  danger: "bg-vw-danger",
  info: "bg-vw-info-fg",
  idle: "bg-chat-dim",
};

export interface RouteTone {
  tone: Tone;
  /** The word that always accompanies the colour. */
  label: string;
  fg: string;
  bar: string;
}

const ROUTES: Record<string, { tone: Tone; label: string }> = {
  local: { tone: "ok", label: "local" },
  passthrough: { tone: "info", label: "Anthropic" },
  fallback: { tone: "amber", label: "fell back" },
  refused: { tone: "danger", label: "refused" },
  error: { tone: "danger", label: "error" },
};

/** A decision route → tone + word: local = ok, Anthropic = info, fell back =
 *  amber, refused/error = danger. Unknown routes are idle and keep their name. */
export function routeTone(route: string): RouteTone {
  const r = ROUTES[route] ?? { tone: "idle" as Tone, label: route };
  return { ...r, fg: TONE_FG[r.tone], bar: TONE_BAR[r.tone] };
}

// .r { display: inline-flex; gap: 5px; font-weight: 600; font-size: 12px }
// .r::before { 8x8, radius 2px, currentColor }  -> render a ROUTE_SWATCH span
export const ROUTE_LABEL = "inline-flex items-center gap-[5px] text-[12px] font-semibold";
export const ROUTE_SWATCH = "h-2 w-2 flex-none rounded-[2px] bg-current";

// .strip { gap: 12px; padding: 12px 14px; border-radius: 10px; border: 1px solid rule-soft/.7;
//          background: surface/.6; font-size: 13.5px }
const STRIP_BASE =
  "flex items-start gap-3 rounded-[10px] border px-3.5 py-3 text-[13.5px] leading-normal";

const STRIP: Record<StripTone, string> = {
  neutral: "border-vw-rule-soft/70 bg-chat-surface/60",
  ok: "border-vw-live/35 bg-vw-ok-bg/20", //           .strip.ok
  amber: "border-vw-amber-fg/45 bg-vw-amber-bg/30", // settings redesign's conflict strip
  danger: "border-vw-danger/55 bg-vw-danger-bg/30", // .strip.bad
};

export function strip(tone: StripTone): string {
  return `${STRIP_BASE} ${STRIP[tone]}`;
}

// .strip .ico { 18x18, margin-top 1px }, coloured per tone.
export const STRIP_ICON: Record<StripTone, string> = {
  neutral: "mt-px h-[18px] w-[18px] flex-none text-chat-muted",
  ok: "mt-px h-[18px] w-[18px] flex-none text-vw-live",
  amber: "mt-px h-[18px] w-[18px] flex-none text-vw-amber-fg",
  danger: "mt-px h-[18px] w-[18px] flex-none text-vw-danger-fg",
};

// .strip strong — the headline sentence; danger headlines are danger-fg.
export const STRIP_TITLE: Record<StripTone, string> = {
  neutral: "font-semibold text-chat-fg",
  ok: "font-semibold text-chat-fg",
  amber: "font-semibold text-vw-amber-fg",
  danger: "font-semibold text-vw-danger-fg",
};
export const STRIP_TEXT = "m-0 text-chat-muted";
export const STRIP_ACTIONS = "mt-2 flex flex-wrap gap-2";

// .sec { border: 1px solid rule-soft/.7; border-radius: 12px; background: surface/.55; min-width: 0 }
export const SEC = "min-w-0 rounded-xl border border-vw-rule-soft/70 bg-chat-surface/55";
// .sec-head { gap: 10px; padding: 14px 16px; flex-wrap }   h2 15/600   .note 12.5 muted
export const SEC_HEAD = "flex flex-wrap items-center gap-2.5 px-4 py-3.5";
export const SEC_TITLE = "m-0 text-[15px] font-semibold";
export const SEC_NOTE = "text-[12.5px] text-chat-muted";
export const SEC_BODY = "px-4 pb-4";

// .card (rail): same surface as .sec, padded.
export const CARD = "rounded-xl border border-vw-rule-soft/70 bg-chat-surface/55 px-4 py-3.5";

// Band and column labels: 11.5-12 uppercase +0.06em, chat-dim.
export const BAND_LABEL = "text-[11.5px] font-semibold uppercase tracking-[0.06em] text-chat-dim";

// Map rows: divided by rule-soft/45 hairlines, not boxed.
// Rows set the mockup body size (14px); the app body is 16px.
export const ROW = "border-t border-vw-rule-soft/45 px-4 py-3 text-[14px]";
// .route.alarm { background: danger-bg/.16; box-shadow: inset 3px 0 0 danger }
export const ROW_ALARM = "bg-vw-danger-bg/[0.16] shadow-[inset_3px_0_0_rgb(var(--vw-danger))]";
// .route.off { opacity: .55 }  .route.off .pat { line-through, dim }
// opacity .55 (mockup) drops muted meta and the "rule paused" pill to ~2.9:1;
// .8 keeps the faded look and clears 4.5:1 in both themes (.75 is 4.37 on the dark pill).
export const ROW_OFF = "opacity-80";
export const ROW_OFF_PATTERN = "line-through decoration-chat-dim";
// .route.rest { background: page/.35; border-radius: 0 0 12px 12px }
export const ROW_REST = "rounded-b-xl bg-chat-page/35";
// .route .pat (mono, 13.5/600, wraps anywhere). `!font-mono`: the retro themes
// set `span, button, p, a… { font-family: DM Sans }` (0,1,1), which beats a
// plain `.font-mono` utility on those elements.
export const PATTERN = "min-w-0 !font-mono text-[13.5px] font-semibold [overflow-wrap:anywhere]";

// Muted meta line under a row / item: 12.5 muted.
export const META = "text-[12.5px] text-chat-muted";

// Focus ring shared by router controls (craft: accent only on CTA + focus).
export const FOCUS =
  "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent";

// Links are fg + underline (accent is reserved).
export const LINK = `font-medium text-chat-fg underline underline-offset-[3px] rounded-sm ${FOCUS}`;

// Icon buttons (↑ ↓ ⋯): 32x32 desktop, 40x40 under 760px.
export const ICON_BTN =
  `inline-grid h-10 w-10 place-items-center rounded-md text-chat-muted hover:bg-chat-surface-2/60 hover:text-chat-fg disabled:cursor-not-allowed disabled:opacity-40 min-[760px]:h-8 min-[760px]:w-8 ${FOCUS}`;
