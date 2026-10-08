// Quiet hours are compressed (coordinator ruling, 2026-10-06): idle stretches of the timeline (no tree active) add at
// most IDLE_CAP_U to X, globally (never per range), and each compressed stretch is reported for a marker.
import { describe, expect, it } from "vitest";
import { IDLE_CAP_U, XS, XSTEP, buildTimeline, quietLabel } from "@/lib/forest/timeline";
import type { Tree, Turn } from "@/lib/forest/types";

const TRAITS = { sys0: 1, ctx_gen: 1, mean_gen: 1, thrash: 0, fail: 0, fanout: 0 };
const tree = (id: string, times: number[]): Tree => {
  const turns = times.map((t): Turn => [t, 1000, 100, 0, [], [], "stop", 1]);
  return { id, key: "k", start: times[0], end: times.at(-1)!, last: times.at(-1)!, last_at: 0, traits: TRAITS, sessions: [{ id: `s-${id}`, variant: null, turns, children: [] }] };
};
const H = 3600, D = 86400;

/** A farm-like week: two keys, each working 09:00-18:00 every day in 3 h spells (a turn every 75 s); idle nights. */
export function farmWeek(days = 7): Tree[] {
  const out: Tree[] = [];
  for (let d = 0; d < days; d++)
    for (const k of [0, 1])
      for (let s = 0; s < 3; s++) {
        const t0 = d * D + 9 * H + s * 3 * H + k * 600;
        out.push(tree(`d${d}k${k}s${s}`, Array.from({ length: 130 }, (_, i) => t0 + i * 75)));
      }
  return out;
}

describe("quiet hours", () => {
  it("each idle stretch adds at most IDLE_CAP_U; X stays monotone; short pauses are unchanged", () => {
    // two trees 8 h apart (a long quiet stretch), and a 20 min pause inside the second one
    const a = tree("a", [600, 700, 800]), b = tree("b", [8 * H, 8 * H + 100, 8 * H + 1300 + 100, 8 * H + 1500]);
    const tl = buildTimeline([a, b], 9 * H);
    const quiet = tl.X(8 * H) - tl.X(800 + 600); // from 10 min after a's last turn to b's first
    expect(quiet).toBeLessThanOrEqual(IDLE_CAP_U + XS * XSTEP); // + the partly active minutes at its ends
    expect(quiet).toBeGreaterThan(IDLE_CAP_U * 0.5);
    let prev = -Infinity;
    for (let t = 0; t <= 9 * H; t += 30) {
      const x = tl.X(t);
      expect(x).toBeGreaterThanOrEqual(prev);
      prev = x;
    }
    // a 10 min gap (no idle minute at all: activity lasts 10 min after each turn) runs as before
    const c = tree("c", [0, 600, 1200]), d = tree("d", [0, 600, 1200]);
    expect(buildTimeline([c], 2 * H).X(1200)).toBeCloseTo(buildTimeline([d], 2 * H).X(1200), 9);
    // the long stretch is reported, with its length
    const gaps = tl.gaps();
    expect(gaps).toHaveLength(2); // the 8 h night, and the 32 min after b (longer than IDLE_CAP_U at normal speed)
    expect(gaps[0].to - gaps[0].from).toBeGreaterThan(7 * H);
    expect(quietLabel(gaps[0].to - gaps[0].from)).toBe("8 h quiet") // 00:24 (10 min after a) to 08:00;
  });

  it("a stretch shorter than the cap's worth of normal speed is not compressed (1h/6h views as before)", () => {
    const minutes = IDLE_CAP_U / (XS * XSTEP); // ≈ 21 min at normal speed
    const end = 60 + 600 + (minutes - 24) * 60;
    const a = tree("a", [0, 60]), b = tree("b", [60 + 600 + (minutes - 25) * 60, end]);
    expect(buildTimeline([a, b], end + 540).gaps()).toHaveLength(0);
  });

  it("a farm-like week: the spit is several times shorter than without the compression", () => {
    const week = farmWeek();
    const tl = buildTimeline(week, 7 * D);
    const len = tl.X(7 * D) - tl.X(0);
    // without compression every night would run 15 h at XS per second
    const nights = 7 * 15 * H * XS;
    console.log(`farm week: spit ${len.toFixed(0)} u (quiet nights uncompressed would add ~${nights.toFixed(0)} u)`);
    expect(len).toBeLessThan(700);
    expect(tl.gaps().length).toBeGreaterThanOrEqual(7);
  });

  it("a key working non-stop for 48 h: its 16 spell trees (3 h each) never clash in a lane (re-review N1)", () => {
    // one tree per 3 h spell (forest.py SPELL_SPAN_S), a turn every 60 s: one or two trees active at any time, more
    // spells than the 11 lanes
    const spells: Tree[] = [];
    for (let k = 0; k < 16; k++) spells.push(tree(`a${k}`, Array.from({ length: 179 }, (_, i) => k * 3 * H + i * 60)));
    const tl = buildTimeline(spells, 48 * H);
    const ps = spells.map((t) => tl.places(t.id)[0]);
    let clashes = 0;
    for (let i = 0; i < ps.length; i++) for (let j = i + 1; j < ps.length; j++)
      if (ps[i].z === ps[j].z && Math.abs(ps[i].x - ps[j].x) < 11 - 1e-9) clashes++;
    expect(clashes).toBe(0);
    // working hours are not compressed: 48 h of one tree at a time run at its speed (0.12 of normal)
    expect(tl.X(48 * H) - tl.X(0)).toBeGreaterThan(48 * 60 * XS * XSTEP * 0.12 * 0.9);
    expect(tl.gaps()).toHaveLength(0);
  });

  it("labels", () => {
    expect(quietLabel(50 * 60)).toBe("50 min quiet");
    expect(quietLabel(6 * H + 600)).toBe("6 h quiet");
    expect(quietLabel(2 * D + 3 * H)).toBe("2 d 3 h quiet");
  });
});


describe("the follow fit of a long span is tight", () => {
  it("a 760-unit spit is framed at the distance its width needs (the search reaches past 400)", async () => {
    const { CameraController } = await import("@/components/forest/engine/camera");
    const spit = Array.from({ length: 77 }, (_, i) => [{ x: i * 10, y: 0, z: -15 }, { x: i * 10, y: 8, z: 15 }]).flat();
    const safe = { l: -0.5, r: 0.95, t: 0.1, b: -0.85 };
    const cam = new CameraController({ fov: 36, aspect: 1.6 });
    cam.follow(spit, safe);
    // the width it needs: 760 units over 1.45 NDC at a 36° vertical fov and aspect 1.6 (≈ 1000 units off)
    const need = 760 / (Math.tan((18 * Math.PI) / 180) * 1.6 * (1.45 / 2) * 2) / 1;
    console.log(`long span fit: ${cam.fitDistance()!.toFixed(0)} (width needs ≈ ${need.toFixed(0)})`);
    expect(cam.fitDistance()!).toBeLessThan(need * 1.15);
  });
});

describe("the X table is prefix-stable (re-review N2)", () => {
  it("30 polls of a sliding window with trees entering and leaving: X(t < now − 20 min) never changes; new trees land ahead", async () => {
    const { SceneLayout, X_FREEZE_S } = await import("@/components/forest/engine/frame");
    expect(X_FREEZE_S).toBe(1200);
    const rnd = ((q: number) => () => (q = (q * 16807) % 2147483647) / 2147483647)(5);
    // scene time: trees start every 3-25 min (quiet stretches between some), each with 3-12 turns a minute or two apart
    const all: Tree[] = [];
    for (let t = -7200, k = 0; t < 7200; k++) {
      t += (k % 5 === 4 ? 90 : 3 + rnd() * 22) * 60;
      const n = 3 + Math.floor(rnd() * 10);
      all.push(tree(`t${k}`, Array.from({ length: n }, (_, i) => t + i * (60 + rnd() * 60))));
    }
    const layout = new SceneLayout();
    const H = 7 * 86400;
    let prev: Map<number, number> | null = null, worst = 0, behind = 0, stoneMove = 0;
    const stones = new Map<number, number>();
    const held = new Map<string, number>();
    for (let p = 0; p < 30; p++) {
      const now = p * 120, ws = now - 3 * 3600; // a 3 h window: long enough to hold a 90 min quiet stretch
      const trees = all
        .map((t) => ({ ...t, sessions: t.sessions.map((s) => ({ ...s, turns: s.turns.filter((x) => x[0] <= now) })) }))
        .filter((t) => t.sessions[0].turns.length && t.sessions[0].turns.at(-1)![0] >= ws);
      const tl = layout.update(trees as Tree[], now, H, ws - 3600);
      const grid = new Map<number, number>();
      for (let t = ws - 3600; t < now - 1200; t += 60) grid.set(t, tl.X(t));
      if (prev) for (const [t, x] of grid) if (prev.has(t)) worst = Math.max(worst, Math.abs(x - prev.get(t)!));
      prev = grid;
      // the quiet stones stand at their run's start (re-review N6): fixed once that is frozen
      for (const g of tl.gaps()) {
        if (g.from >= now - 1200 - 60) continue; // frozen: older than now − 20 min by a whole sample
        const x = tl.X(g.from) + 1;
        if (stones.has(g.from)) stoneMove = Math.max(stoneMove, Math.abs(x - stones.get(g.from)!));
        stones.set(g.from, x);
      }
      const maxHeld = Math.max(-Infinity, ...[...held].filter(([id]) => trees.some((t) => t.id === id)).map(([, x]) => x));
      for (const t of trees) {
        const x = tl.places(t.id)[0].x;
        if (!held.has(t.id) && Number.isFinite(maxHeld) && x < maxHeld - 1e-6) behind++;
        held.set(t.id, held.get(t.id) ?? x);
      }
    }
    console.log(`N2: max X change for t < now − 20 min over 30 polls: ${worst}; new trees behind a held one: ${behind}`);
    expect(worst).toBeLessThan(1e-9);
    expect(behind).toBe(0);
    expect(stones.size).toBeGreaterThan(0);
    expect(stoneMove).toBeLessThan(1e-9);
  });
});
