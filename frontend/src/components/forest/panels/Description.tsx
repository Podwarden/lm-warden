"use client";

import { RANGE_S, type Range } from "@/lib/forest/types";
import { GLASS, GlassDetails, openWhenWide } from "./glass";

/** "24h" → "24 hours" / "24 h"; "7d" → "7 days" / "7 d". */
export function rangeWords(range: Range): { long: string; short: string } {
  const n = Number.parseInt(range, 10), day = range.endsWith("d");
  return { long: `${n} ${day ? (n === 1 ? "day" : "days") : n === 1 ? "hour" : "hours"}`, short: `${n} ${day ? "d" : "h"}` };
}

/** The server's name as the app shows it everywhere else (nav brand, document title): there is no per-install name. */
const SERVER_NAME = "LM Warden";

/** Left, top (spec §6.4): what the forest shows, the range and the delay. Collapsible, remembered. */
export function Description({
  trees,
  flowers,
  loaded,
  empty,
  range = "24h",
}: {
  /** Trees in the window, or null before the first response. */
  trees: number | null;
  flowers: number | null;
  loaded: boolean;
  /** Loaded, and nothing grew in the window (no trees, no flowers). */
  empty: boolean;
  /** The shown range (the window's switch). */
  range?: Range;
}) {
  const w = rangeWords(RANGE_S[range] ? range : "24h");
  return (
    <div className="flex flex-col gap-1.5">
      <GlassDetails title="Session forest" storageKey="forest.desc.open" defaultOpen={openWhenWide} testid="forest-desc">
        <p className="leading-relaxed text-[#d3dad3]">
          Real traffic of this server. One tree for each API key and working spell. One limb for each session. One
          twig for each request. The view shows the last {w.long}, live, 30 seconds late: the camera sees what grows
          next before it grows.
        </p>
        <p className="mt-1.5 text-white/60">
          {SERVER_NAME} · Last {w.short} · live · 30 s delay
          {loaded && trees !== null && (
            <>
              {" "}· {trees} {trees === 1 ? "tree" : "trees"}
              {flowers ? ` · ${flowers} ${flowers === 1 ? "flower" : "flowers"}` : ""}
            </>
          )}
        </p>
        <p className="mt-1.5 text-white/60">Drag to look around; pan left to see older trees.</p>
      </GlassDetails>
      {empty && (
        <p role="status" className={`${GLASS} px-3 py-2 text-xs`}>
          No traffic yet. Trees grow here as requests finish.
        </p>
      )}
    </div>
  );
}
