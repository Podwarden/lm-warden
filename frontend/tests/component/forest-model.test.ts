import { describe, expect, it } from "vitest";
import { budgets, compress } from "@/lib/forest/normalize";
import { KNOBS, SPECIES_RULES, cronScore, pickSpecies } from "@/lib/forest/species";
import { buildTimeline } from "@/lib/forest/timeline";
import { tokenBars } from "@/lib/forest/panels";
import type { Session, Traits, Tree, Turn } from "@/lib/forest/types";

const turn = (t: number, ctx = 40000, gen = 200, cached: number | null = 38000) =>
  [t, ctx, gen, cached, [], ["read"], "tool_calls", 1] as const;

describe("normalize", () => {
  it("uses server speed and falls back when null", () => {
    const fast = budgets(turn(0) as never, { prefill_tps: 20000, decode_tps: 1000, max_model_len: 262144 });
    const slow = budgets(turn(0) as never, { prefill_tps: null, decode_tps: null, max_model_len: 32768 });
    expect(fast.len).toBeCloseTo(0.2);
    expect(slow.len).toBeCloseTo(0.5); // 200 / 400 fallback
    expect(fast.foliage).toBeCloseTo(40000 / 262144);
  });
  it("compresses logarithmically", () => {
    expect(compress(0, 1)).toBe(0);
    expect(compress(1, 1)).toBe(1);
  });
});

describe("species", () => {
  const base = { sys0: 37000, ctx_gen: 280, mean_gen: 134, thrash: 0, fail: 0, fanout: 0 };
  it("applies rules in order", () => {
    expect(pickSpecies({ ...base, fail: 0.5 }, {})).toBe("cactus");
    expect(pickSpecies({ ...base, thrash: 0.5 }, {})).toBe("willow");
    expect(pickSpecies({ ...base, sys0: 70000 }, {})).toBe("baobab");
    expect(pickSpecies({ ...base, sys0: 3000, mean_gen: 100 }, {})).toBe("shrub");
    expect(pickSpecies(base, {})).toBe("oak");
  });
});

describe("timeline", () => {
  it("X is monotone and slows when one tree grows alone", () => {
    const tree = (id: string, ts: number[]) => ({ id, key: id, start: ts[0], end: ts.at(-1)!, last: ts.at(-1)!, last_at: ts.at(-1)!,
      traits: { sys0: 1, ctx_gen: 1, mean_gen: 1, thrash: 0, fail: 0, fanout: 0 },
      sessions: [{ id: id + "s", variant: null, children: [], turns: ts.map((t) => turn(t)) as never }] });
    const tl = buildTimeline([tree("a", [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600])], 7200);
    const xs = [0, 600, 1200, 3600, 7200].map(tl.X);
    xs.slice(1).forEach((x, i) => expect(x).toBeGreaterThanOrEqual(xs[i]));
    expect(tl.X(600) - tl.X(0)).toBeLessThan(tl.X(7200) - tl.X(6600));
  });
});

describe("panels", () => {
  it("token bars sum per minute", () => {
    const state = { t0: 0, now: 600, range: "6h", models: {}, cursor: 600, flowers: [],
      trees: new Map([["a", { id: "a", sessions: [{ id: "s", turns: [turn(30, 1000, 10, 900), turn(50, 2000, 20, 1800)], children: [] }] }]]) } as never;
    const bars = tokenBars(state, 120, 3);
    expect(bars.at(-2)).toEqual({ minute: 0, cached: 2700, computed: 300, gen: 30 });
  });
});

// ---------------------------------------------------------------------------------------------
// Beyond the brief: rules worth pinning.

const T = (t: number, o: Partial<{ ctx: number; gen: number; cached: number | null; n: number }> = {}): Turn =>
  [t, o.ctx ?? 40000, o.gen ?? 200, o.cached === undefined ? 38000 : o.cached, [], ["read"], "tool_calls", o.n ?? 1];
const S = (id: string, turns: Turn[], children: Session[] = [], variant: string | null = null): Session => ({ id, variant, turns, children });
const traits: Traits = { sys0: 37000, ctx_gen: 280, mean_gen: 134, thrash: 0, fail: 0, fanout: 0 };
const mkTree = (id: string, sessions: Session[], key = id): Tree => {
  const ts = sessions.flatMap((s) => s.turns.map((t) => t[0]));
  const start = Math.min(...ts), end = Math.max(...ts);
  return { id, key, start, end, last: end, last_at: end, traits, sessions };
};

describe("normalize — fallback and cache weighting", () => {
  it("girth counts cache hits at 1/20 and falls back to 4000 tok/s prefill", () => {
    const g = budgets(T(0, { ctx: 40000, cached: 38000 }), { prefill_tps: null, decode_tps: null, max_model_len: null });
    expect(g.girth).toBeCloseTo(2000 / 4000 + 38000 / (20 * 4000));
    expect(g.foliage).toBeCloseTo(40000 / 32768); // max_model_len fallback
  });
  it("treats a null cached count as fully computed", () => {
    const g = budgets(T(0, { ctx: 8000, cached: null }), { prefill_tps: 8000, decode_tps: 100, max_model_len: 8192 });
    expect(g.girth).toBeCloseTo(1);
  });
  it("compress uses the reference scale", () => {
    expect(compress(3, 1)).toBe(2);
    expect(compress(30, 10)).toBe(2);
  });
});

describe("species — full rule table", () => {
  const base = traits;
  it("lists the 13 rules in spec §6.2 order", () => {
    expect(SPECIES_RULES.map((r) => r.name)).toEqual([
      "cactus", "bamboo", "willow", "banyan", "birch", "palm", "pine", "baobab", "bonsai", "shrub", "poplar", "oak", "round",
    ]);
  });
  it("has knobs for every species", () => {
    for (const r of SPECIES_RULES) expect(KNOBS[r.name]).toBeDefined();
    expect(KNOBS.pine.cone).toBe(true);
    expect(KNOBS.oak).toEqual({ girth: 1.15, tall: 0.8, el: 0.48 });
    expect(KNOBS.round).toEqual({}); // mockup default habit: no overrides
  });
  it("bamboo: periodic sessions, low thrash", () => {
    const sessions = [0, 3600, 7200, 10800, 14400].map((t, i) => S("c" + i, [T(t), T(t + 30)]));
    expect(cronScore(sessions)).toBeCloseTo(1);
    expect(pickSpecies(base, { sessions })).toBe("bamboo");
    expect(pickSpecies({ ...base, thrash: 0.1 }, { sessions })).toBe("oak");
  });
  it("cron is 0 with fewer than 4 sessions and low for irregular starts", () => {
    expect(cronScore([0, 3600, 7200].map((t, i) => S("c" + i, [T(t)])))).toBe(0);
    expect(cronScore([0, 10, 5000, 5010, 20000].map((t, i) => S("c" + i, [T(t)])))).toBeLessThan(0.7);
  });
  it("banyan: fanout ≥ 1.5", () => {
    expect(pickSpecies({ ...base, fanout: 1.5 }, {})).toBe("banyan");
  });
  it("birch never fires in v1 (no reasoning data on the wire)", () => {
    expect(SPECIES_RULES.find((r) => r.name === "birch")!.test(base, {})).toBe(false);
  });
  it("palm: ≥70% single-turn sessions and sys0 > 30k", () => {
    const sessions = [S("a", [T(0)]), S("b", [T(40)]), S("c", [T(900)]), S("d", [T(1000), T(1010)])]; // irregular starts: not cron
    expect(pickSpecies(base, { sessions })).toBe("palm");
    expect(pickSpecies({ ...base, sys0: 20000 }, { sessions })).not.toBe("palm");
  });
  it("pine: exactly one session, ≥40 turns, no children", () => {
    const long = S("p", Array.from({ length: 40 }, (_, i) => T(i * 10)));
    expect(pickSpecies(base, { sessions: [long] })).toBe("pine");
    expect(pickSpecies(base, { sessions: [{ ...long, children: [S("k", [T(5)])] }] })).toBe("oak");
    expect(pickSpecies(base, { sessions: [long, S("q", [T(900), T(910)])] })).toBe("oak");
    // turnless sessions are ignored, as for cron and single-turn share
    expect(pickSpecies(base, { sessions: [long, S("empty", [])] })).toBe("pine");
    // merged turns count as their raw rows
    expect(pickSpecies(base, { sessions: [S("m", [T(0, { n: 20 }), T(10, { n: 20 })])] })).toBe("pine");
  });
  it("bonsai: small local variant", () => {
    expect(pickSpecies({ ...base, sys0: 10000 }, { maxModelLen: 16384 })).toBe("bonsai");
    expect(pickSpecies({ ...base, sys0: 10000 }, { maxModelLen: 32768 })).toBe("round");
  });
  it("poplar and round crown", () => {
    expect(pickSpecies({ ...base, ctx_gen: 8 }, {})).toBe("poplar");
    expect(pickSpecies({ ...base, sys0: 10000 }, {})).toBe("round");
  });
});

describe("timeline — lanes, transplants, widths", () => {
  it("is deterministic for the same input", () => {
    const trees = [mkTree("a", [S("a1", [T(0), T(60)])]), mkTree("b", [S("b1", [T(0), T(60)])])];
    const A = buildTimeline(trees, 3600), B = buildTimeline(trees, 3600);
    expect(A.places("a")).toEqual(B.places("a"));
    expect(A.places("b")).toEqual(B.places("b"));
  });
  it("does not depend on input order: ties break by tree id", () => {
    const trees = ["c", "a", "b"].map((id) => mkTree(id, [S(id + "1", [T(0), T(60), T(60 + 7200)])]));
    const A = buildTimeline(trees, 10000), B = buildTimeline([trees[2], trees[0], trees[1]], 10000);
    for (const id of ["a", "b", "c"]) {
      expect(B.lane(id)).toBe(A.lane(id));
      expect(B.places(id)).toEqual(A.places(id));
    }
    expect(A.places("a")).toHaveLength(2);
    expect(["a", "b", "c"].map(A.lane)).toEqual([0, 1, -1]);
    for (const t of [0, 60, 3600, 7260, 10000]) expect(B.X(t)).toBe(A.X(t));
  });
  it("puts overlapping trees in successive lanes across the spit", () => {
    const trees = ["a", "b", "c"].map((id) => mkTree(id, [S(id + "1", [T(0), T(60)])]));
    const tl = buildTimeline(trees, 3600);
    expect(trees.map((t) => tl.lane(t.id))).toEqual([0, 1, -1]);
    expect(tl.places("b")[0].z).toBeCloseTo(4.4);
    expect(tl.places("a")[0].x).toBeCloseTo(tl.X(0) + 2);
  });
  it("transplants a resumed session forward, landing 16 units ahead", () => {
    const ses = S("s", [T(0), T(60), T(120), T(120 + 7200), T(120 + 7260)]);
    const tl = buildTimeline([mkTree("a", [ses])], 10000);
    const pl = tl.places("a");
    expect(pl).toHaveLength(2);
    expect(pl[1].from).toBe(7320);
    expect(pl[1].x).toBeCloseTo(tl.X(7320) + 16);
    expect(pl[1].x).toBeGreaterThan(pl[0].x);
  });
  it("does not move for a gap under 30 min or a hop under 20 units", () => {
    const short = S("s", [T(0), T(1700)]); // 28 min gap
    expect(buildTimeline([mkTree("a", [short])], 4000).places("a")).toHaveLength(1);
    // 35 min gap: idle speed is 1.0 → 2100 s × 0.0062 ≈ 13 units < 20
    const hop = S("s", [T(0), T(2100)]);
    expect(buildTimeline([mkTree("a", [hop])], 4000).places("a")).toHaveLength(1);
  });
  it("widens the spit only above 10 concurrent trees", () => {
    const many = (n: number) => Array.from({ length: n }, (_, i) => mkTree("t" + i, [S("s" + i, [T(0), T(30)])]));
    const ten = buildTimeline(many(10), 3600), base = buildTimeline([], 3600);
    const x = ten.places("t0")[0].x;
    expect(ten.width(x, -1)).toBeCloseTo(base.width(x, -1));
    const fourteen = buildTimeline(many(14), 3600);
    expect(fourteen.width(x, -1)).toBeGreaterThan(base.width(x, -1) + 4);
    expect(fourteen.width(x, 1)).toBeGreaterThan(base.width(x, 1) + 4);
  });
  it("returns neutral values for an unknown tree", () => {
    const tl = buildTimeline([], 600);
    expect(tl.lane("nope")).toBe(0);
    expect(tl.places("nope")).toEqual([]);
  });
});

describe("panels — buckets", () => {
  const stateOf = (trees: Tree[], models = {}) =>
    ({ t0: 0, now: 3600, range: "6h", models, cursor: 3600, flowers: [], trees: new Map(trees.map((t) => [t.id, t])) }) as never;
  it("returns exactly `minutes` completed buckets, oldest first, including subagents", () => {
    const tree = mkTree("a", [S("s", [T(65, { ctx: 100, gen: 1, cached: null })], [S("k", [T(70, { ctx: 50, gen: 2, cached: 40 })])])]);
    const bars = tokenBars(stateOf([tree]), 179, 2);
    expect(bars.map((b) => b.minute)).toEqual([0, 1]);
    expect(bars[1]).toEqual({ minute: 1, cached: 40, computed: 110, gen: 3 });
    expect(tokenBars(stateOf([tree]), 179)).toHaveLength(30);
  });
  it("(M1) buckets are wall-clock minutes, floor((t + t0) / 60), whatever the anchor t0", () => {
    const t0 = 1_700_000_030; // 50 s into a wall-clock minute (minutes start at …980, …040, …100)
    const turns = (d: number) => [T(0 - d, { ctx: 100, gen: 1, cached: 0 }), T(20 - d, { ctx: 200, gen: 2, cached: 0 })];
    const st = { ...(stateOf([mkTree("a", [S("s", turns(0))])]) as object), t0 } as never;
    const bars = tokenBars(st, 150, 3); // shown at 1_700_000_180
    const m = Math.floor((t0 + 150) / 60);
    expect(bars.map((b) => b.minute)).toEqual([m - 3, m - 2, m - 1]);
    // rel 0 and rel 20 (one t0-relative minute) are 1_700_000_030 and 1_700_000_050: two wall-clock minutes
    expect(bars.find((b) => b.minute === Math.floor(1_700_000_030 / 60))!.computed).toBe(100);
    expect(bars.find((b) => b.minute === Math.floor(1_700_000_050 / 60))!.computed).toBe(200);
    // the same turns after a re-anchor by 299.97 s land in the same buckets
    const st2 = { ...(stateOf([mkTree("a", [S("s", turns(299.97))])]) as object), t0: t0 + 299.97 } as never;
    expect(tokenBars(st2, 150 - 299.97, 3)).toEqual(bars);
  });
});
