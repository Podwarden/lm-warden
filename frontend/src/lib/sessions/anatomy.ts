// The prompt-anatomy bar's segments: fixed order and colours, widths in percent.

import type { Anatomy } from "@/lib/sessions/types";

export const ANATOMY_ORDER = ["cached", "dead", "recomputed", "new", "unknown"] as const;

export const ANATOMY_CLASS: Record<keyof Anatomy, string> = {
  cached: "bg-emerald-500",
  dead: "anat-dead",
  recomputed: "bg-rose-500",
  new: "bg-stone-500",
  unknown: "bg-slate-400",
};

/** Percent widths in the fixed order, zero segments dropped; [] for an all-zero bar. */
export function anatomyWidths(a: Anatomy): { key: keyof Anatomy; pct: number }[] {
  const total = ANATOMY_ORDER.reduce((s, k) => s + Math.max(0, a[k] ?? 0), 0);
  if (!(total > 0)) return [];
  return ANATOMY_ORDER.filter((k) => (a[k] ?? 0) > 0).map((k) => ({ key: k, pct: (a[k] / total) * 100 }));
}
