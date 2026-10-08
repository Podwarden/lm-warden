import { authFetchJSON } from "@/lib/auth-fetch";
import { readForestToken } from "./token";
import type { ForestResponse, ForestState, Range, Tree } from "./types";
export type { ForestResponse, ForestState } from "./types";

export class ForestAuthError extends Error {}

/** Shift every relative time in a tree by `d` seconds (t0 moved from a to a+d). */
function rebaseTree(t: Tree, d: number): Tree {
  if (d === 0) return t;
  const shiftS = (s: Tree["sessions"][number]): Tree["sessions"][number] => ({
    ...s,
    turns: s.turns.map((x) => [x[0] - d, ...x.slice(1)] as typeof x),
    children: s.children.map(shiftS),
  });
  return { ...t, start: t.start - d, end: t.end - d, last: t.last - d, sessions: t.sessions.map(shiftS) };
}

/**
 * Merge a forest response into the held state. Every ForestState time is relative to `state.t0`, `now` included: the
 * wire sends `now` absolute (unix seconds), so it is stored as `r.now − r.t0`.
 */
export function applyResponse(
  prev: ForestState | null,
  r: ForestResponse,
): { state: ForestState; refetchFull: boolean } {
  if (r.full || prev === null) {
    return {
      state: {
        t0: r.t0, now: r.now - r.t0, range: r.range, models: r.models, cursor: r.cursor,
        trees: new Map(r.trees.map((t) => [t.id, t])),
        flowers: r.flowers.slice(),
      },
      refetchFull: false,
    };
  }
  const incoming = new Map(r.trees.map((t) => [t.id, t]));
  if (r.ids.some((id) => !incoming.has(id) && !prev.trees.has(id))) {
    return { state: prev, refetchFull: true };
  }
  const d = r.t0 - prev.t0;
  const trees = new Map<string, Tree>();
  for (const id of r.ids) {
    const fromResponse = incoming.get(id);
    // Response trees are already relative to the new t0; held trees need rebasing.
    trees.set(id, fromResponse ?? rebaseTree(prev.trees.get(id)!, d));
  }
  // Server resends flowers whose absolute start is > since - 600 (since = prev.cursor).
  const cutAbs = prev.cursor - 600;
  const kept = prev.flowers
    .filter((f) => f[0] + prev.t0 <= cutAbs)
    .map((f) => [f[0] - d, f[1], f[2]] as [number, number, number]);
  return {
    state: {
      t0: r.t0, now: r.now - r.t0, range: r.range, models: r.models, cursor: r.cursor,
      trees, flowers: [...kept, ...r.flowers],
    },
    refetchFull: false,
  };
}

/**
 * Fetcher for the forest view's endpoints. "session": the admin UI session (`authFetchJSON`). "forest-token": the
 * key-scoped forest token from sessionStorage as a Bearer header; 401/403 throw `ForestAuthError`, and so does a missing
 * or expired token (with no request: an expired token is never sent). Also used for the live in-flight endpoint
 * (`/api/stats/requests`), which accepts the same forest token.
 */
export function forestFetcher<T = ForestResponse>(mode: "session" | "forest-token") {
  return async (url: string): Promise<T> => {
    if (mode === "session") return authFetchJSON<T>(url);
    const token = readForestToken();
    if (!token) throw new ForestAuthError("no forest token");
    const res = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
    if (res.status === 401 || res.status === 403) throw new ForestAuthError(`forest ${res.status}`);
    if (!res.ok) throw new Error(`forest ${res.status}`);
    return (await res.json()) as T;
  };
}

/**
 * The full window's default range (spec §6.4, §10.6: live view only; its range switch offers 1h / 6h / 24h / 7d). The
 * Stats card fetches the Stats page's range.
 */
export const FOREST_RANGE: Range = "24h";

/** `GET /api/stats/forest` for `range` (default: the full window's); with `since`, a delta from that cursor. */
export function forestUrl(since?: number, range: Range = FOREST_RANGE): string {
  const base = `/api/stats/forest?range=${range}`;
  return since === undefined ? base : `${base}&since=${since}`;
}

/**
 * One poll: a full fetch when nothing is held for `range` (a state of another range is never polled with its cursor: a
 * range switch is one full fetch), otherwise a delta from the held cursor. When the delta asks for a
 * refetch (`applyResponse` → `refetchFull`), exactly one full fetch follows; its result is taken as is (never a loop).
 */
export async function pollForest(
  prev: ForestState | null,
  get: (url: string) => Promise<ForestResponse>,
  range: Range = FOREST_RANGE,
): Promise<ForestState> {
  const held = prev !== null && prev.range === range ? prev : null;
  const r = await get(forestUrl(held?.cursor, range));
  const a = applyResponse(held, r);
  if (!a.refetchFull) return a.state;
  return applyResponse(null, await get(forestUrl(undefined, range))).state;
}
