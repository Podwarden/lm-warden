// Colours of the sessions pages, as literal Tailwind classes (the JIT only sees whole class names).
// Share fill: green ≥ 0.8 from cache, amber ≥ 0.2, red below, slate unknown. Cause stripes match the cache panel.

export function shareClass(share: number | null): string {
  if (share === null || !Number.isFinite(share)) return "bg-slate-400";
  if (share >= 0.8) return "bg-emerald-500";
  if (share >= 0.2) return "bg-amber-500";
  return "bg-rose-500";
}

export function causeClass(cause: string | null): string | null {
  if (cause === "compacted") return "bg-yellow-400";
  if (cause === "harness_rewrite") return "bg-orange-600";
  return null;
}

const SVG_FILL: Record<string, string> = {
  "bg-emerald-500": "fill-emerald-500",
  "bg-amber-500": "fill-amber-500",
  "bg-rose-500": "fill-rose-500",
  "bg-slate-400": "fill-slate-400",
  "bg-yellow-400": "fill-yellow-400",
  "bg-orange-600": "fill-orange-600",
};

/** The SVG fill class for a background class (SVG shapes take `fill`, not `background`). */
export function svgFill(bg: string): string {
  return SVG_FILL[bg] ?? "fill-slate-400";
}

export const LINK_STYLE: Record<"known" | "inferred" | "compaction", { stroke: string; dash?: string }> = {
  known: { stroke: "stroke-slate-400" },
  inferred: { stroke: "stroke-amber-500", dash: "4 3" },
  compaction: { stroke: "stroke-yellow-400", dash: "2 2" },
};
