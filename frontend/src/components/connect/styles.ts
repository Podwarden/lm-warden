// Class strings for the Connect a client page (plan §6). Same language as the
// router and settings redesigns: the router's tokens re-exported, plus the few
// list, picker and snippet classes this page adds. Every string is a COMPLETE
// literal so Tailwind's scanner sees it.

export {
  BAND_LABEL,
  CARD,
  FOCUS,
  ICON_BTN,
  LINK,
  META,
  PATTERN,
  SEC,
  SEC_BODY,
  SEC_HEAD,
  SEC_NOTE,
  SEC_TITLE,
  STRIP_ACTIONS,
  STRIP_ICON,
  STRIP_TEXT,
  STRIP_TITLE,
  btn,
  pill,
  strip,
} from "@/components/router/styles";

import { FOCUS } from "@/components/router/styles";

// .btn.sm — 30 px, 40 px on phones (router strip actions).
export const SMALL_BTN =
  "inline-flex min-h-[30px] items-center justify-center whitespace-nowrap rounded-md border border-chat-rule bg-transparent px-2.5 py-[3px] text-[12.5px] font-medium text-chat-fg hover:bg-chat-surface disabled:cursor-not-allowed disabled:opacity-45 max-[759px]:min-h-10 " +
  FOCUS;

// Native <select> / <input>: 36 px, hairline, surface fill.
export const FIELD =
  "h-9 w-full min-w-0 rounded-md border border-chat-rule bg-chat-surface px-2.5 text-[13.5px] text-chat-fg disabled:cursor-not-allowed disabled:opacity-60 max-[759px]:h-10 " +
  FOCUS;

// Field labels: 12.5/600 muted (sentence case, not a band label).
export const FIELD_LABEL = "text-[12.5px] font-semibold text-chat-muted";

// Tool list rows: 40 px, the selected one surface-2/60 with a 2 px fg bar
// (amber stays for the CTA and focus). Inside the horizontal strip (< 1100)
// the bar sits under the tab instead.
export const TOOL_TAB =
  "flex min-h-10 w-full min-w-0 items-center gap-2 bg-transparent px-4 py-1.5 text-left text-[14px] text-chat-fg hover:bg-chat-surface-2/40 max-[1099px]:w-auto max-[1099px]:flex-none max-[1099px]:whitespace-nowrap max-[1099px]:px-3 focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-chat-accent";
export const TOOL_TAB_ON =
  "bg-chat-surface-2/60 font-semibold shadow-[inset_2px_0_0_rgb(var(--chat-fg))] max-[1099px]:shadow-[inset_0_-2px_0_rgb(var(--chat-fg))]";

// Underline tabs (mode), as connect-claude-code.tsx.
export const TAB =
  "-mb-px whitespace-nowrap border-b-2 bg-transparent px-2.5 py-2 text-[13px] max-[759px]:min-h-10 " + FOCUS;
export const TAB_ON = "border-chat-accent font-semibold text-chat-fg";
export const TAB_OFF = "border-transparent text-chat-muted hover:text-chat-fg";

// Chip tabs (files inside a mode): a quieter second level.
export const CHIP =
  "whitespace-nowrap rounded-md border px-2.5 py-1 text-[12.5px] max-[759px]:min-h-10 " + FOCUS;
export const CHIP_ON = "border-vw-rule-soft bg-chat-surface-2/70 font-semibold text-chat-fg";
export const CHIP_OFF = "border-vw-rule-soft/60 bg-transparent text-chat-muted hover:text-chat-fg";

// The snippet <pre>: router-snippet's dock, hairline and mono 12.5/1.6, but
// lines do not wrap — config files stay readable and copyable; the block
// scrolls inside itself (focusable, with a hint when it overflows).
export const SNIPPET =
  "m-0 overflow-x-auto whitespace-pre rounded-lg border border-vw-rule-soft/60 bg-vw-dock py-3 pl-3 pr-12 !font-mono text-[12.5px] leading-[1.6] [scrollbar-width:thin] " +
  FOCUS;
