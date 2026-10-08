/**
 * Data for the full-window side panels (spec §6.4): per-minute token bars. The in-flight panel reads live
 * `/api/stats/requests` (spec §10.6: live view only).
 */
import type { ForestState, Session } from "./types";

export interface TokenBar {
  /** Wall-clock minute: floor(absolute seconds / 60) = floor((t + t0) / 60). */
  minute: number;
  /** Prompt tokens served from the prefix cache. */
  cached: number;
  /** Prompt tokens computed (ctx − cached). */
  computed: number;
  gen: number;
}

const walk = (sessions: Session[], f: (s: Session) => void) => {
  const go = (s: Session) => {
    f(s);
    (s.children ?? []).forEach(go);
  };
  sessions.forEach(go);
};

/**
 * One bar per COMPLETED wall-clock minute before the minute containing `atRel`: minutes m − `minutes` … m − 1 with
 * m = floor((atRel + t0) / 60), oldest first. A turn counts in the minute its start falls in, on the absolute clock
 * (floor((t + t0) / 60)), so a t0 re-anchor never re-buckets the bars. Subagents included.
 */
export function tokenBars(state: ForestState, atRel: number, minutes = 30): TokenBar[] {
  const m = Math.floor((atRel + state.t0) / 60);
  const first = m - minutes;
  const bars: TokenBar[] = Array.from({ length: minutes }, (_, i) => ({ minute: first + i, cached: 0, computed: 0, gen: 0 }));
  for (const tree of state.trees.values())
    walk(tree.sessions, (s) => {
      for (const [t, ctx, gen, cached] of s.turns) {
        const i = Math.floor((t + state.t0) / 60) - first;
        if (i < 0 || i >= minutes) continue;
        const hit = cached ?? 0;
        bars[i].cached += hit;
        bars[i].computed += Math.max(0, ctx - hit);
        bars[i].gen += gen;
      }
    });
  return bars;
}
