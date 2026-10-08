"use client";

import type { CameraMode } from "@/components/forest/engine/camera";
import { FOREST_RANGES, type ForestRange } from "@/lib/forest/types";
import { cn } from "@/lib/utils";
import { GLASS } from "./glass";

const MODES: { mode: CameraMode; label: string; title: string }[] = [
  { mode: "cinema", label: "Cinematic", title: "Orbit the tree that is growing now" },
  { mode: "follow", label: "Wide", title: "Keep all growth in view" },
];

/**
 * Bottom centre (spec §6.4, §10.6): the camera toggle. No time controls in the live-only view: the range switch
 * (`RangeSwitch`, beside it) changes how much history is shown and framed, never the shown time.
 * A drag pauses the active camera; pressing a mode resumes it.
 */
export function Controls({ mode, onMode }: { mode: CameraMode; onMode: (m: CameraMode) => void }) {
  return (
    <div role="group" aria-label="Camera" className={cn(GLASS, "flex items-center gap-1 p-1 text-xs")}>
      <span className="px-2 text-white/60">live · 30 s delay</span>
      {MODES.map((m) => (
        <button
          key={m.mode}
          type="button"
          title={m.title}
          aria-pressed={mode === m.mode}
          onClick={() => onMode(m.mode)}
          className={cn(
            "rounded-md px-3 py-1.5 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#7cc4f4]",
            mode === m.mode ? "bg-white/15 text-[#e5a95b]" : "text-white/80 hover:bg-white/10",
          )}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}

/**
 * The full window's range switch (follow-up C, spec §6.4, §10.6): 1h / 6h / 24h / 7d of history, fetched and framed.
 * Still live: it is not a replay.
 */
export function RangeSwitch({ range, onRange }: { range: ForestRange; onRange: (r: ForestRange) => void }) {
  return (
    <div role="group" aria-label="Range" className={cn(GLASS, "flex items-center gap-1 p-1 text-xs")}>
      {FOREST_RANGES.map((r) => (
        <button
          key={r}
          type="button"
          title={`Show the last ${r}`}
          aria-pressed={range === r}
          onClick={() => onRange(r)}
          className={cn(
            "rounded-md px-2.5 py-1.5 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#7cc4f4]",
            range === r ? "bg-white/15 text-[#e5a95b]" : "text-white/80 hover:bg-white/10",
          )}
        >
          {r}
        </button>
      ))}
    </div>
  );
}
