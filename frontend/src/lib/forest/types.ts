export type Range = "1h" | "6h" | "24h" | "48h" | "7d";

/** The ranges the forest offers (spec §6.3, §6.4, §10.6): the Stats page's own, so the card follows its switch. */
export const FOREST_RANGES = ["1h", "6h", "24h", "7d"] as const;
export type ForestRange = (typeof FOREST_RANGES)[number];
export const isForestRange = (x: unknown): x is ForestRange => typeof x === "string" && (FOREST_RANGES as readonly string[]).includes(x);

/** A range's length in seconds (the server's window is [now − range, now]). */
export const RANGE_S: Record<Range, number> = { "1h": 3600, "6h": 6 * 3600, "24h": 86400, "48h": 2 * 86400, "7d": 7 * 86400 };

/** [t, ctx, gen, cached, toolsIn, toolsOut, finish, mergedN] */
export type Turn = [
  t: number,
  ctx: number,
  gen: number,
  cached: number | null,
  toolsIn: [string, number, boolean][],
  toolsOut: string[],
  finish: string | null,
  mergedN: number,
];

export interface Session {
  id: string;
  variant: string | null;
  turns: Turn[];
  children: Session[];
  at?: number;
}

export interface Traits {
  sys0: number;
  ctx_gen: number;
  mean_gen: number;
  thrash: number;
  fail: number;
  fanout: number;
}

export interface Tree {
  id: string;
  key: string;
  start: number;
  end: number;
  last: number;
  last_at: number;
  traits: Traits;
  sessions: Session[];
}

export interface ModelStats {
  prefill_tps: number | null;
  decode_tps: number | null;
  max_model_len: number | null;
  n?: number;
}

/** `GET /api/stats/forest` as sent. Tree and flower times are relative to `t0`; `t0`, `now` and `cursor` are absolute. */
export interface ForestResponse {
  range: Range;
  t0: number;
  /** Absolute unix seconds on the wire; `applyResponse` stores it relative to t0 (`ForestState.now = now − t0`). */
  now: number;
  models: Record<string, ModelStats>;
  trees: Tree[];
  flowers: [number, number, number][];
  ids: string[];
  full: boolean;
  cursor: number;
}

/** Times are relative to `t0` (absolute unix seconds). */
export interface ForestState {
  t0: number;
  now: number;
  range: Range;
  trees: Map<string, Tree>;
  flowers: [number, number, number][];
  models: Record<string, ModelStats>;
  cursor: number;
}
