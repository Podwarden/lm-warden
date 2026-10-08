// frontend/tests/component/forest-camera.test.ts
import { describe, expect, it } from "vitest";
import { damp, ForestClock } from "@/components/forest/engine/camera";
import {
  CameraController,
  ARRIVE_MARGIN,
  FOLLOW_PAUSE_MS,
  ISO,
  LEAVE_AFTER,
  V_MAX,
  A_MAX,
  RETURN_S,
  RETURN_LIMIT_S,
  noOvershootRate,
  LOOK,
  framePoints,
  lookAheadAim,
  MIN_VISIT_S,
  planShot,
  REPLAN_MARGIN,
  project,
  safeRectFromPanels,
  type CamEvent,
  type Crown,
  type V3,
} from "@/components/forest/engine/camera";

describe("camera", () => {
  it("damped spring converges without overshoot", () => {
    const cur = { x: 0, y: 0, z: 0 }, vel = { x: 0, y: 0, z: 0 }, goal = { x: 10, y: 0, z: 0 };
    let maxX = 0;
    for (let i = 0; i < 600; i++) { damp(cur, goal, vel, 1.5, 1 / 60); maxX = Math.max(maxX, cur.x); }
    expect(cur.x).toBeCloseTo(10, 2);
    expect(maxX).toBeLessThanOrEqual(10.0001);
  });
  // the brief's nextEvent test is superseded by the shot-plan tests (controller ruling): nextEvent fed the removed lull
});

// Scope change (spec §10.6): live view only. The brief's lull test is gone with the lull; resume lands on the
// render-delayed bound, nowRel − LOOK.
describe("clock", () => {
  it("resume after hidden jumps without replay storm", () => {
    const c = new ForestClock(100);
    c.resumeAfterHidden(3700);
    expect(c.rel).toBe(3700 - LOOK);
  });
  it("(d) resumeAfterHidden is one discontinuous step to the bound, not a walk through the events in between", () => {
    const c = new ForestClock(100);
    c.setNow(100.5);
    c.tick(0.5);
    expect(c.lastStep).toEqual({ from: 70, to: 70.5, continuous: true });
    c.resumeAfterHidden(3700);
    expect(c.lastStep).toEqual({ from: 70.5, to: 3700 - LOOK, continuous: false });
    c.setNow(3700.1);
    c.tick(0.1);
    expect(c.rel).toBeCloseTo(3700.1 - LOOK, 9);
    expect(c.lastStep.continuous).toBe(true);
  });
  it("holds rel = nowRel − LOOK, advancing at 1× and never past the bound", () => {
    const c = new ForestClock(1000);
    expect(c.rel).toBe(1000 - LOOK);
    for (let i = 1; i <= 120; i++) {
      c.setNow(1000 + i / 60);
      c.tick(1 / 60);
      expect(c.rel).toBeLessThanOrEqual(c.nowRel - LOOK + 1e-9);
    }
    expect(c.rel).toBeCloseTo(1002 - LOOK, 6);
    // no now update: time holds at the bound
    c.tick(1);
    expect(c.rel).toBeCloseTo(1002 - LOOK, 6);
  });
  it("never runs backward when now is corrected back, and makes up a small lag at ≤ 2×", () => {
    const c = new ForestClock(1000);
    c.setNow(995);
    c.tick(0.5);
    expect(c.rel).toBe(970);
    c.setNow(1004);
    let prev = c.rel;
    for (let i = 0; i < 600; i++) {
      c.tick(1 / 60);
      expect(c.rel - prev).toBeLessThanOrEqual(2 / 60 + 1e-9);
      expect(c.rel).toBeLessThanOrEqual(1004 - LOOK + 1e-9);
      prev = c.rel;
    }
    expect(c.rel).toBeCloseTo(1004 - LOOK, 9);
  });
  it("a lag of more than 10 s is one discontinuous jump", () => {
    const c = new ForestClock(1000);
    c.setNow(1100);
    c.tick(1 / 60);
    expect(c.rel).toBe(1100 - LOOK);
    expect(c.lastStep.continuous).toBe(false);
  });
});

// ---------------------------------------------------------------- helpers

const dist = (a: V3, b: V3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

describe("look-ahead aim", () => {
  const events: CamEvent[] = [
    { time: 100, sessionId: "a", treeId: "A" },
    { time: 200, sessionId: "b", treeId: "B" },
  ];
  const tips = new Map<string, V3>([["a", { x: 1, y: 2, z: 3 }]]);
  const bases = new Map([["B", { x: 50, z: -4, s: 0.8 }]]);
  it("aims at the session tip, or the tree base for a new tree, only within LOOK seconds", () => {
    expect(LOOK).toBe(30);
    expect(lookAheadAim(events, 50, tips, (id) => bases.get(id), "A")).toBeNull();
    expect(lookAheadAim(events, 80, tips, (id) => bases.get(id), "B")).toBeNull(); // only that tree's turns count
    const a = lookAheadAim(events, 80, tips, (id) => bases.get(id), "A");
    expect(a?.p).toEqual({ x: 1, y: 2, z: 3 });
    expect(a?.treeId).toBe("A");
    expect(a?.T).toBeCloseTo(Math.max(1.5, 20 * 0.45), 9);
    const b = lookAheadAim(events, 199, tips, (id) => bases.get(id), "B");
    expect(b?.p).toEqual({ x: 50, y: 2.5 * 0.8, z: -4 });
    expect(b?.T).toBe(1.5);
  });
});

describe("framePoints (setCamGoal selection)", () => {
  const w = () => 15;
  const pt = (x: number): V3 => ({ x, y: 3, z: 0 });
  it("frames the growing trees near the front plus the shore", () => {
    const P = framePoints(
      [{ x: -20, active: true, points: [pt(-20)] }, { x: 95, active: true, points: [pt(95)] }, { x: 90, active: false, points: [pt(90)] }],
      100, w,
    );
    expect(P.filter((p) => p.y === 3).map((p) => p.x)).toEqual([95]); // x=−20 is > 110 behind the front
    expect(P.filter((p) => p.y === 0)).toHaveLength(8);
    expect(Math.min(...P.map((p) => p.x))).toBe(88); // xc − 12
  });
  it("in a lull frames the newest two near the front, never a tree far back", () => {
    const trees = [70, 80, 90, 101].map((x) => ({ x, active: false, points: [pt(x)] }));
    expect(framePoints(trees, 100, w).filter((p) => p.y === 3).map((p) => p.x)).toEqual([101, 90]);
    const far = framePoints([{ x: 10, active: false, points: [pt(10)] }], 100, w);
    expect(far.every((p) => p.y === 0)).toBe(true);
  });
});

// ---------------------------------------------------------------- motion (I3, I4)

/** Speed bound: 0.1 units per frame at 60 fps (the plan's e2e bound). */
const MAX_SPEED = 6;
/** Acceleration bound (units/s²), the same at every frame rate. */
const MAX_ACCEL = 8;

const A: Crown = { x: 0, z: 0, r: 4, top: 9 };
const M: Crown = { x: 45, z: -2, r: 3, top: 8 }; // stands in the transit path
const B: Crown = { x: 90, z: 6, r: 6, top: 12 };

/** Three trees; the aim alternates between A and B (90 apart) every 60 s, across M. */
function farSwitchRig() {
  const events: CamEvent[] = [];
  for (let t = 20; t < 480; t += 120) events.push({ time: t, sessionId: "a", treeId: "A" });
  for (let t = 80; t < 480; t += 120) events.push({ time: t, sessionId: "b", treeId: "B" });
  events.sort((p, q) => p.time - q.time);
  const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -12, y: 6, z: 4 }, target: { x: 0, y: 4, z: 0 } });
  cam.setCrowns(new Map([["A", A], ["M", M], ["B", B]]));
  cam.setEvents(events);
  cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }], ["b", { x: 91, y: 7, z: 4 }]]));
  return { cam, crowns: [A, M, B], seconds: 480 };
}

/** The camera starts low behind M and must reach B: the straight path crosses M's crown. */
function transitRig() {
  const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: 30, y: 5, z: -2 }, target: { x: 40, y: 4, z: 0 } });
  cam.setCrowns(new Map([["M", M], ["B", B]]));
  cam.setEvents([{ time: 20, sessionId: "b", treeId: "B" }]);
  cam.setTips(new Map([["b", { x: 91, y: 7, z: 4 }]]));
  return { cam, crowns: [M, B], seconds: 90 };
}

function lookOf(cam: CameraController): V3 {
  const { pos, target } = cam.pose();
  const d = dist(pos, target);
  return { x: (target.x - pos.x) / d, y: (target.y - pos.y) / d, z: (target.z - pos.z) / d };
}

/** A tall neighbour crown N stands on B's orbit ring: the orbit must climb 13 units over it, longer than the 3 s
 * velocity look-ahead covers at orbit speed, so the arc look-ahead and its speed gate have to hold the orbit back. */
function orbitRingRig() {
  const N: Crown = { x: 99.5, z: 6, r: 2.5, top: 20 };
  const events: CamEvent[] = [];
  for (let t = 5; t < 200; t += 10) events.push({ time: t, sessionId: "b", treeId: "B" });
  const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: 90, y: 7, z: 15.5 }, target: { x: 91, y: 7, z: 4 } });
  cam.setCrowns(new Map([["B", B], ["N", N]]));
  cam.setEvents(events);
  cam.setTips(new Map([["b", { x: 91, y: 7, z: 4 }]]));
  return { cam, crowns: [B, N], seconds: 180 };
}

type Rig = { cam: CameraController; crowns: Crown[]; seconds: number };
const RATES: [string, (t: number) => number][] = [
  ["30 fps", () => 1 / 30],
  ["60 fps", () => 1 / 60],
  ["144 fps", () => 1 / 144],
  ["60 fps + one 0.5 s hitch", (t) => (t >= 30 && t < 30 + 1 / 60 ? 0.5 : 1 / 60)],
];

function fly(rig: Rig, dtAt: (t: number) => number) {
  const { cam, crowns, seconds } = rig;
  let t = 0, prev = cam.pose().pos, prevV: V3 | null = null, prevDt = 0;
  let maxSpeed = 0, maxStep60 = 0, maxAccel = 0, clearance = Infinity, maxTurn = 0;
  let prevLook = lookOf(cam);
  const near = new Set<string>();
  while (t < seconds) {
    const dt = dtAt(t);
    t += dt;
    cam.step(dt, { rel: t, nowMs: t * 1000 });
    const p = cam.pose().pos;
    const v = { x: (p.x - prev.x) / dt, y: (p.y - prev.y) / dt, z: (p.z - prev.z) / dt };
    maxSpeed = Math.max(maxSpeed, Math.hypot(v.x, v.y, v.z));
    if (Math.abs(dt - 1 / 60) < 1e-9) maxStep60 = Math.max(maxStep60, dist(p, prev));
    if (prevV) maxAccel = Math.max(maxAccel, dist(v, prevV) / ((dt + prevDt) / 2));
    for (const c of crowns) {
      const d = Math.hypot(p.x - c.x, p.z - c.z);
      if (p.y < c.top) clearance = Math.min(clearance, d - c.r);
      if (d < c.r + 15) near.add(`${c.x}`);
    }
    const look = lookOf(cam);
    maxTurn = Math.max(maxTurn, (Math.acos(Math.min(1, look.x * prevLook.x + look.y * prevLook.y + look.z * prevLook.z)) * 180) / Math.PI / dt);
    prevLook = look;
    prev = p; prevV = v; prevDt = dt;
  }
  return { maxSpeed, maxStep60, maxAccel, clearance, near, maxTurn };
}

describe.each([["far aim switch", farSwitchRig], ["crown transit", transitRig], ["crown on the orbit ring", orbitRingRig]] as const)("cinema motion: %s", (_, rig) => {
  const runs = RATES.map(([name, dtAt]) => ({ name, ...fly(rig(), dtAt) }));
  it.each(RATES.map(([n]) => n))("%s: speed ≤ 6 u/s, step ≤ 0.1 per 60 fps frame, acceleration ≤ 8 u/s²", (name) => {
    const r = runs.find((x) => x.name === name)!;
    expect(r.maxSpeed).toBeLessThanOrEqual(MAX_SPEED);
    expect(r.maxStep60).toBeLessThanOrEqual(0.1);
    expect(r.maxAccel).toBeLessThanOrEqual(MAX_ACCEL);
  });
  it.each(RATES.map(([n]) => n))("%s: (b) never inside a crown below its top", (name) => {
    expect(runs.find((x) => x.name === name)!.clearance).toBeGreaterThanOrEqual(0.5);
  });
  it.each(RATES.map(([n]) => n))("%s: the look direction turns ≤ 90°/s", (name) => {
    expect(runs.find((x) => x.name === name)!.maxTurn).toBeLessThanOrEqual(90);
  });
  it("acceleration does not depend on the frame rate", () => {
    const a = runs.map((r) => r.maxAccel);
    expect(Math.max(...a) / Math.min(...a)).toBeLessThan(1.5);
  });
  it("(a) gets there: the camera reaches the destination tree", () => {
    for (const r of runs) expect(r.near.has(`${B.x}`)).toBe(true);
  });
});

describe("shot plan (N1: plan from the LOOK window, never shuttle)", () => {
  const crownOf = (id: string) => ({ A, B } as Record<string, Crown>)[id];

  it("a plan only uses events inside [rel, rel + LOOK]", () => {
    const events: CamEvent[] = [];
    for (let t = 0; t < 200; t += 1.5) events.push({ time: t, sessionId: t % 3 ? "a" : "b", treeId: t % 3 ? "A" : "B" });
    for (const rel of [0, 17, 50.2, 120]) {
      for (const from of [{ x: -8, y: 6, z: 0 }, { x: 95, y: 7, z: 14 }, { x: 45, y: 7, z: 30 }]) {
        const plan = planShot(events, rel, from, crownOf);
        expect(plan.legs.length).toBeGreaterThan(0);
        for (const leg of plan.legs) for (const e of leg.events) {
          expect(e.time).toBeGreaterThan(rel);
          expect(e.time).toBeLessThanOrEqual(rel + LOOK);
        }
      }
    }
  });

  it("drops turns the camera cannot reach in time, and never plans a shuttle", () => {
    const events: CamEvent[] = [];
    for (let t = 3; t < 60; t += 6) events.push({ time: t, sessionId: "a", treeId: "A" });
    for (let t = 6; t < 60; t += 6) events.push({ time: t, sessionId: "b", treeId: "B" });
    events.sort((p, q) => p.time - q.time);
    // at A: B's turns are 90 units (≈ 20 s) away
    const plan = planShot(events, 0, { x: -8, y: 6, z: 0 }, crownOf, { firstTree: "A", local: "A" });
    expect(plan.legs.map((l) => l.treeId)).toEqual(["A"]);
    const fresh = planShot(events, 0, { x: -8, y: 6, z: 0 }, crownOf);
    expect(fresh.legs.length).toBe(1);
    for (const l of fresh.legs) for (const e of l.events) expect(e.time).toBeGreaterThanOrEqual(l.arriveBy);
  });

  it("interleaved far-apart trees: watched one at a time, one move when it stops, ≤ 1 destination change per 30 s", () => {
    // A and B (90 apart) alternate every 3 s until 120 s; then only B; from 200 s both again
    const events: CamEvent[] = [];
    for (let t = 3; t < 400; t += 6) if (t < 120 || t >= 200) events.push({ time: t, sessionId: "a", treeId: "A" });
    for (let t = 6; t < 400; t += 6) events.push({ time: t, sessionId: "b", treeId: "B" });
    events.sort((p, q) => p.time - q.time);
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -12, y: 6, z: 4 }, target: { x: 0, y: 4, z: 0 } });
    const crowns = new Map([["A", A], ["B", B]]);
    cam.setCrowns(crowns);
    cam.setEvents(events);
    cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }], ["b", { x: 91, y: 7, z: 4 }]]));
    const arriveR = (c: Crown) => c.r + 3.2 + 1.5 + ARRIVE_MARGIN;
    const dt = 1 / 60;
    let dest: string | null = null, prevD = Infinity, arrived = false, pickedAt = 0, pickD = Infinity, first = true;
    let prevLook = lookOf(cam), maxTurn = 0;
    const changes: number[] = [], at: Record<number, string | null> = {};
    for (let i = 1; i <= 300 * 60; i++) {
      const t = i * dt;
      cam.step(dt, { rel: t, nowMs: t * 1000 });
      if (cam.destination !== dest) {
        if (dest !== null) changes.push(t);
        first = dest === null;
        dest = cam.destination; prevD = Infinity; arrived = false; pickedAt = t; pickD = Infinity;
      }
      if (i % (60 * 10) === 0) at[Math.round(t)] = dest;
      const look = lookOf(cam);
      maxTurn = Math.max(maxTurn, (Math.acos(Math.min(1, look.x * prevLook.x + look.y * prevLook.y + look.z * prevLook.z)) * 180) / Math.PI / dt);
      prevLook = look;
      const c = crowns.get(dest!)!;
      const p = cam.pose().pos, d = Math.hypot(p.x - c.x, p.z - c.z);
      if (!arrived) {
        // closing in. From rest (the first pick) strictly monotone; after a move the orbit's own velocity takes
        // ≤ 0.5 s to turn round, during which it never gains more than 0.1 on the distance at the pick
        if (pickD === Infinity) pickD = d;
        if (first || t - pickedAt > 0.5) expect(d).toBeLessThanOrEqual(prevD + 1e-6);
        else expect(d).toBeLessThanOrEqual(pickD + 0.1);
        prevD = d;
        if (d <= arriveR(c)) arrived = true;
      } else {
        // dwelling at a tree whose turns it is showing (or the window is empty)
        const win = events.filter((e) => e.time > t && e.time <= t + LOOK);
        if (win.length) expect(win.some((e) => e.treeId === dest)).toBe(true);
      }
    }
    for (let k = 1; k < changes.length; k++) expect(changes[k] - changes[k - 1]).toBeGreaterThanOrEqual(30);
    expect(at[60]).toBe("A"); // the interleaved stretch: stays on A, no shuttle
    expect(at[180]).toBe("B"); // A stopped at 120 s: one move to B
    expect(changes.length).toBeGreaterThanOrEqual(1);
    expect(maxTurn).toBeLessThanOrEqual(90);
  });

  it("fairness: two far-apart trees busy for 5 minutes are both visited, each visit ≥ 30 s, changes ≥ 30 s apart", () => {
    const events: CamEvent[] = [];
    for (let t = 3; t < 400; t += 6) events.push({ time: t, sessionId: "a", treeId: "A" });
    for (let t = 6; t < 400; t += 6) events.push({ time: t, sessionId: "b", treeId: "B" });
    events.sort((p, q) => p.time - q.time);
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -12, y: 6, z: 4 }, target: { x: 0, y: 4, z: 0 } });
    const crowns = new Map([["A", A], ["B", B]]);
    cam.setCrowns(crowns);
    cam.setEvents(events);
    cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }], ["b", { x: 91, y: 7, z: 4 }]]));
    const arriveR = (c: Crown) => c.r + 3.2 + 1.5 + ARRIVE_MARGIN;
    const dt = 1 / 60;
    let dest: string | null = null, arrivedAt: number | null = null;
    const changes: number[] = [], visits: { tree: string; s: number }[] = [];
    for (let i = 1; i <= 300 * 60; i++) {
      const t = i * dt;
      cam.step(dt, { rel: t, nowMs: t * 1000 });
      if (cam.destination !== dest) {
        if (dest !== null) {
          changes.push(t);
          if (arrivedAt !== null) visits.push({ tree: dest, s: t - arrivedAt });
        }
        dest = cam.destination; arrivedAt = null;
      }
      const c = crowns.get(dest!)!, p = cam.pose().pos;
      if (arrivedAt === null && Math.hypot(p.x - c.x, p.z - c.z) <= arriveR(c)) arrivedAt = t;
    }
    expect(new Set(visits.map((v) => v.tree))).toEqual(new Set(["A", "B"]));
    for (const v of visits) expect(v.s).toBeGreaterThanOrEqual(30);
    for (let k = 1; k < changes.length; k++) expect(changes[k] - changes[k - 1]).toBeGreaterThanOrEqual(30);
  });

  it("fairness: a tree that is the only one growing keeps the camera", () => {
    const events: CamEvent[] = [];
    for (let t = 3; t < 400; t += 6) events.push({ time: t, sessionId: "a", treeId: "A" });
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -12, y: 6, z: 4 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(new Map([["A", A], ["B", B]]));
    cam.setEvents(events);
    cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }]]));
    for (let t = 1 / 60; t < 300; t += 1 / 60) {
      cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
      expect(cam.destination).toBe("A");
    }
  });

  /** Runs a cinema camera; records visits (arrival → leaving) and destination changes, with the window at each. */
  function tour(cam: CameraController, crowns: Map<string, Crown>, events: CamEvent[], seconds: number) {
    const arriveR = (c: Crown) => c.r + 3.2 + 1.5 + ARRIVE_MARGIN;
    const dt = 1 / 60;
    let dest: string | null = null, arrivedAt: number | null = null;
    const changes: { t: number; dry: boolean }[] = [], visits: { tree: string; s: number; dry: boolean }[] = [];
    for (let i = 1; i <= seconds * 60; i++) {
      const t = i * dt;
      cam.step(dt, { rel: t, nowMs: t * 1000 });
      if (cam.destination !== dest) {
        // "dry": the tree being left had no turn left in the window (the one allowed early exit)
        const dry = dest !== null && !events.some((e) => e.treeId === dest && e.time > t - dt && e.time <= t - dt + LOOK);
        if (dest !== null) {
          changes.push({ t, dry });
          if (arrivedAt !== null) visits.push({ tree: dest, s: t - arrivedAt, dry });
        }
        dest = cam.destination; arrivedAt = null;
      }
      const c = crowns.get(dest!);
      const p = cam.pose().pos;
      if (c && arrivedAt === null && Math.hypot(p.x - c.x, p.z - c.z) <= arriveR(c)) arrivedAt = t;
    }
    return { changes, visits };
  }

  it("(R18) close trees with the busy one swapping every 20 s: every visit and every gap ≥ 30 s", () => {
    const C: Crown = { x: 12, z: 0, r: 4, top: 9 };
    const events: CamEvent[] = [];
    // both trees always have turns in the window (light every 7 s), the heavy stream swaps every 20 s
    for (let t = 1; t < 400; t += 7) events.push({ time: t, sessionId: "a", treeId: "A" }, { time: t + 3, sessionId: "c", treeId: "C" });
    for (let t = 0; t < 400; t += 1) events.push({ time: t + 0.5, sessionId: Math.floor(t / 20) % 2 ? "c" : "a", treeId: Math.floor(t / 20) % 2 ? "C" : "A", weight: 2 });
    events.sort((p, q) => p.time - q.time);
    const crowns = new Map([["A", A], ["C", C]]);
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -8, y: 6, z: 0 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(crowns);
    cam.setEvents(events);
    cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }], ["c", { x: 12, y: 5, z: 1 }]]));
    const { changes, visits } = tour(cam, crowns, events, 300);
    expect(changes.length).toBeGreaterThanOrEqual(2); // it does follow the heavy stream
    for (const v of visits) expect(v.s).toBeGreaterThanOrEqual(MIN_VISIT_S);
    for (let k = 1; k < changes.length; k++) expect(changes[k].t - changes[k - 1].t).toBeGreaterThanOrEqual(MIN_VISIT_S);
  });

  it("(R18) arriving at a tree with no turn left: moves on at once if another tree grows, else stays", () => {
    const C: Crown = { x: 12, z: 0, r: 4, top: 9 };
    const crowns = new Map([["A", A], ["C", C]]);
    // the camera starts far from A, so it arrives after A's only turn has played
    const mk = (withC: boolean) => {
      const events: CamEvent[] = [{ time: 4, sessionId: "a", treeId: "A" }];
      if (withC) for (let t = 40; t < 120; t += 2) events.push({ time: t, sessionId: "c", treeId: "C" });
      const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -40, y: 6, z: 10 }, target: { x: -30, y: 4, z: 0 } });
      cam.setCrowns(crowns);
      cam.setEvents(events);
      cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }], ["c", { x: 12, y: 5, z: 1 }]]));
      return { cam, events };
    };
    const stays = mk(false);
    const r0 = tour(stays.cam, crowns, stays.events, 90);
    expect(r0.changes).toHaveLength(0);
    expect(stays.cam.destination).toBe("A");
    const moves = mk(true);
    const r1 = tour(moves.cam, crowns, moves.events, 60);
    expect(r1.changes.length).toBeGreaterThanOrEqual(1);
    for (const v of r1.visits) if (v.s < MIN_VISIT_S) expect(v.dry).toBe(true); // only a dry tree is left early
  });

  it("(NEW-1) a new tree with no crown yet: the camera flies to its place and arrives before its first turn", () => {
    const events: CamEvent[] = [{ time: 25, sessionId: "n", treeId: "N" }];
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -8, y: 6, z: 0 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(new Map([["A", A]])); // N is not built yet
    cam.setBaseOf((id) => (id === "N" ? { x: 40, z: 0, s: 1 } : undefined));
    cam.setEvents(events);
    let arrived = -1;
    for (let t = 1 / 60; t < 25 && arrived < 0; t += 1 / 60) {
      cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
      const p = cam.pose().pos;
      if (Math.hypot(p.x - 40, p.z) <= 1.5 + 3.2 + 1.5 + ARRIVE_MARGIN) arrived = t;
    }
    expect(cam.destination).toBe("N");
    expect(arrived).toBeGreaterThan(0);
    expect(arrived).toBeLessThan(25);
  });

  it("(NEW-2) new turns are noticed when the events array is pruned at the front", () => {
    const C: Crown = { x: 12, z: 0, r: 4, top: 9 };
    let events: CamEvent[] = [];
    for (let t = 1; t < 200; t += 3) events.push({ time: t, sessionId: "a", treeId: "A" });
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -8, y: 6, z: 0 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(new Map([["A", A], ["C", C]]));
    cam.setEvents(events);
    cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }], ["c", { x: 12, y: 5, z: 1 }]]));
    let t = 0;
    // stop between A's turns entering the window (A@70 entered at 40 s, A@73 enters at 43 s)
    for (; t < 41.5; t += 1 / 60) cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
    expect(cam.destination).toBe("A");
    // 48 h retention: the front is pruned, and a short heavy burst on C starts in 2 s. The array is now shorter up to
    // the window end than before, so an index-based "did the window grow" check would miss the burst
    events = events.filter((e) => e.time > 35);
    for (let u = 42; u < 46; u += 0.5) events.push({ time: u, sessionId: "c", treeId: "C", weight: 10 });
    events.sort((p, q) => p.time - q.time);
    cam.setEvents(events);
    // the next allowed replan (≤ 1 s) sees it and moves to C; a missed change would hold A until the old leg's end (70 s)
    for (; t < 42.6; t += 1 / 60) cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
    expect(cam.destination).toBe("C");
  });

  it("a committed planned move is not postponed by later replans", () => {
    // at A (busy every 3 s); a steady stream on nearby C starts at 60 s. The plan "A until its next turn + 4 s, then
    // C" is made once C enters the window; C's turns keep entering every 0.5 s and trigger replans, which must not
    // keep pushing "leave after the next A turn" forward
    const C: Crown = { x: 12, z: 0, r: 4, top: 9 };
    const events: CamEvent[] = [];
    for (let t = 1; t < 300; t += 3) events.push({ time: t, sessionId: "a", treeId: "A" });
    for (let t = 60; t < 300; t += 0.5) events.push({ time: t, sessionId: "c", treeId: "C", weight: 2 });
    events.sort((p, q) => p.time - q.time);
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -8, y: 6, z: 0 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(new Map([["A", A], ["C", C]]));
    cam.setEvents(events);
    let moved = -1;
    for (let t = 1 / 60; t < 75 && moved < 0; t += 1 / 60) {
      cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
      if (cam.destination === "C") moved = t;
    }
    expect(moved).toBeGreaterThan(30); // planned, not immediate
    expect(moved).toBeLessThanOrEqual(60); // there before C's stream starts
  });

  it("(NEW-4) a replan treats the orbited tree as local: its imminent turn is not given up to a transit pad", () => {
    // at A since 0 s; A's last turn is at 40.5 s. At 39 s a turn on C (at 69 s) enters the window and forces a
    // replan: A's turn 1.5 s ahead is within TRAVEL_PAD, so only a local A still counts it
    const C: Crown = { x: 12, z: 0, r: 4, top: 9 };
    const events: CamEvent[] = [];
    for (let t = 1; t < 38; t += 3) events.push({ time: t, sessionId: "a", treeId: "A" });
    events.push({ time: 40.5, sessionId: "a", treeId: "A" }, { time: 69, sessionId: "c", treeId: "C" });
    events.sort((p, q) => p.time - q.time);
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -8, y: 6, z: 0 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(new Map([["A", A], ["C", C]]));
    cam.setEvents(events);
    for (let t = 1 / 60; t < 40.6; t += 1 / 60) cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
    expect(cam.destination).toBe("A"); // still there to see 40.5 grow; it leaves for C afterwards
  });

  it("(NEW-5) after the current tree's last turn, the camera stays LEAVE_AFTER to watch it grow, even with a busy tree", () => {
    // at A since 0 s; A's last turn is at 40 s. A heavy stream on nearby C starts at 71 s and enters the window at
    // 41 s, forcing a replan that sees no A turn ahead
    const C: Crown = { x: 12, z: 0, r: 4, top: 9 };
    const events: CamEvent[] = [];
    for (let t = 1; t <= 40; t += 3) events.push({ time: t, sessionId: "a", treeId: "A" });
    for (let t = 71; t < 200; t += 1) events.push({ time: t, sessionId: "c", treeId: "C", weight: 3 });
    events.sort((p, q) => p.time - q.time);
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -8, y: 6, z: 0 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(new Map([["A", A], ["C", C]]));
    cam.setEvents(events);
    let moved = -1;
    for (let t = 1 / 60; t < 60 && moved < 0; t += 1 / 60) {
      cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
      if (cam.destination === "C") moved = t;
    }
    expect(moved).toBeGreaterThanOrEqual(40 + LEAVE_AFTER);
    expect(moved).toBeLessThan(40 + LEAVE_AFTER + 1.1); // and then it goes (next replan slot)
  });

  it("(NEW-3) the planner runs at most once per second, even with a turn entering the window every frame", () => {
    const events: CamEvent[] = [];
    const trees = new Map<string, Crown>();
    for (let k = 0; k < 8; k++) trees.set(`T${k}`, { x: k * 15, z: (k % 3) * 8, r: 3, top: 8 });
    for (let i = 0; i < 100 * 60; i++) events.push({ time: i / 60, sessionId: `s${i % 8}`, treeId: `T${i % 8}` });
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -8, y: 6, z: 0 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(trees);
    cam.setEvents(events);
    for (let t = 1 / 60; t <= 60; t += 1 / 60) cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
    expect(cam.plannerRuns).toBeGreaterThan(0);
    expect(cam.plannerRuns).toBeLessThanOrEqual(61);
  });

  it("(NEW-3) a finished leg with nothing to plan does not replan every frame", () => {
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -8, y: 6, z: 0 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(new Map([["A", A]]));
    cam.setEvents([{ time: 2, sessionId: "a", treeId: "A" }]);
    for (let t = 1 / 60; t <= 60; t += 1 / 60) cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
    expect(cam.plannerRuns).toBeLessThanOrEqual(61);
  });

  it("turns the camera cannot reach in time count for nothing", () => {
    const events: CamEvent[] = [];
    for (let t = 3; t < 30; t += 6) events.push({ time: t, sessionId: "a", treeId: "A" });
    events.push({ time: 25, sessionId: "b", treeId: "B" });
    events.sort((p, q) => p.time - q.time);
    // from B, A is 90 units (≈ 20 s) away: A's turns at 3, 9, 15 cannot be seen
    const plan = planShot(events, 0, { x: 95, y: 7, z: 14 }, crownOf);
    const seen = plan.legs.flatMap((l) => l.events.map((e) => e.time));
    expect(seen).not.toContain(3);
    expect(seen).not.toContain(9);
    for (const l of plan.legs) for (const e of l.events) expect(e.time).toBeGreaterThanOrEqual(l.arriveBy);
  });

  it("never changes the destination of a leg in transit, even for a much better plan", () => {
    // committed to A (one turn); 2 s later ten heavy turns on B enter the window, while the camera is still flying
    const events: CamEvent[] = [{ time: 25, sessionId: "a", treeId: "A" }];
    for (let t = 32; t < 36; t += 0.5) events.push({ time: t, sessionId: "b", treeId: "B", weight: 10 });
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: 45, y: 7, z: 25 }, target: { x: 45, y: 4, z: 0 } });
    cam.setCrowns(new Map([["A", A], ["B", B]]));
    cam.setEvents(events);
    cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }], ["b", { x: 91, y: 7, z: 4 }]]));
    let t = 0;
    for (; t < 1; t += 1 / 60) cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
    expect(cam.destination).toBe("A");
    let arrived = false;
    for (; t < 20 && !arrived; t += 1 / 60) {
      cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
      const p = cam.pose().pos;
      arrived = Math.hypot(p.x - A.x, p.z - A.z) <= A.r + 3.2 + 1.5 + ARRIVE_MARGIN;
      if (!arrived) expect(cam.destination).toBe("A");
    }
    expect(arrived).toBe(true);
  });

  it("keeps its leg unless a new plan beats it by REPLAN_MARGIN", () => {
    // at A with one turn left (weight 1); at 6.5 s a turn on B (weight 1.4 < 1.5) enters the window
    const events: CamEvent[] = [
      { time: 5, sessionId: "a", treeId: "A" },
      { time: 20, sessionId: "a", treeId: "A" },
      { time: 36.5, sessionId: "b", treeId: "B", weight: 1.4 },
    ];
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -8, y: 6, z: 0 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(new Map([["A", A], ["B", B]]));
    cam.setEvents(events);
    cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }], ["b", { x: 91, y: 7, z: 4 }]]));
    for (let t = 1 / 60; t < 12; t += 1 / 60) cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
    expect(REPLAN_MARGIN).toBe(0.5);
    expect(cam.destination).toBe("A"); // 1.4 > 1, but not by 50 %
  });

  it("replans when the committed leg's turns have passed", () => {
    const events: CamEvent[] = [{ time: 20, sessionId: "a", treeId: "A" }, { time: 45, sessionId: "b", treeId: "B" }];
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: 12, y: 7, z: 14 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(new Map([["A", A], ["B", B]]));
    cam.setEvents(events);
    cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }], ["b", { x: 91, y: 7, z: 4 }]]));
    for (let t = 1 / 60; t < 19; t += 1 / 60) cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
    expect(cam.destination).toBe("A");
    for (let t = 19; t < 23.5; t += 1 / 60) cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
    expect(cam.destination).toBe("A"); // watching A's turn grow for LEAVE_AFTER
    for (let t = 23.5; t < 25.5; t += 1 / 60) cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
    expect(cam.destination).toBe("B"); // A's turn has grown; B's is in the window and reachable
  });

  it("an empty window keeps the last tree", () => {
    const cam = new CameraController({ fov: 45, aspect: 16 / 9, pos: { x: -8, y: 6, z: 0 }, target: { x: 0, y: 4, z: 0 } });
    cam.setCrowns(new Map([["A", A], ["B", B]]));
    cam.setEvents([{ time: 5, sessionId: "a", treeId: "A" }]);
    cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }]]));
    for (let t = 1 / 60; t < 120; t += 1 / 60) cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
    expect(cam.destination).toBe("A");
  });
});

describe("cinema motion: far aim switch visits both trees", () => {
  it("orbits A, then B, then A again", () => {
    const r = fly(farSwitchRig(), () => 1 / 60);
    expect(r.near.has(`${A.x}`) && r.near.has(`${B.x}`)).toBe(true);
  });
});

// ---------------------------------------------------------------- controller

describe("CameraController", () => {
  it("(c) follow fit puts all points inside the safe rect", () => {
    const W = 1600, H = 900;
    const panels = [
      { left: 1200, top: 80, right: 1590, bottom: 700 }, // legend / panel on the right
      { left: 0, top: 0, right: 1600, bottom: 60 }, // header
      { left: 0, top: 820, right: 1600, bottom: 900 }, // camera bar
    ];
    const S = safeRectFromPanels(W, H, panels);
    expect(S.r).toBeLessThan((2 * 1200) / W - 1);
    expect(S.t).toBeLessThan(1 - (2 * 60) / H);
    expect(S.b).toBeGreaterThan(1 - (2 * 820) / H);
    const pts: V3[] = [];
    for (let i = 0; i < 40; i++) pts.push({ x: -30 + i * 2, y: (i % 7) * 1.3, z: ((i * 13) % 31) - 15 });
    const cam = new CameraController({ fov: 45, aspect: W / H, pos: { x: 100, y: 60, z: -40 }, target: { x: 0, y: 0, z: 0 } });
    cam.setMode("follow");
    cam.follow(pts, S);
    for (let i = 0; i < 60 * 30; i++) cam.step(1 / 60, { rel: 0, nowMs: i * 16 });
    const { pos, target } = cam.pose();
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of pts) {
      const q = project(p, pos, target, 45, W / H);
      expect(q.x).toBeGreaterThanOrEqual(S.l - 1e-6);
      expect(q.x).toBeLessThanOrEqual(S.r + 1e-6);
      expect(q.y).toBeGreaterThanOrEqual(S.b - 1e-6);
      expect(q.y).toBeLessThanOrEqual(S.t + 1e-6);
      x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y);
    }
    // a fit, not a far-away shot
    expect(Math.max((x1 - x0) / (S.r - S.l), (y1 - y0) / (S.t - S.b))).toBeGreaterThan(0.85);
  });

  it.each([30, 60, 144])("(I5) at %i fps a follow resume from a far-side drag sweeps around: keeps its distance and turns ≤ 90°/s", (fps) => {
    const pts: V3[] = [{ x: -10, y: 0, z: -5 }, { x: 10, y: 4, z: 5 }];
    // the user left the camera 40 units north of the target; the follow goal is on the south
    const cam = new CameraController({ fov: 45, aspect: 1.5, pos: { x: 0, y: 8, z: -40 }, target: { x: 0, y: 0, z: 0 } });
    cam.setMode("follow");
    cam.follow(pts, { l: -0.9, r: 0.9, t: 0.9, b: -0.9 });
    const goalDist = (() => {
      const probe = new CameraController({ fov: 45, aspect: 1.5, pos: { x: 0, y: 8, z: -40 }, target: { x: 0, y: 0, z: 0 } });
      probe.setMode("follow");
      probe.follow(pts, { l: -0.9, r: 0.9, t: 0.9, b: -0.9 });
      for (let i = 0; i < 60 * 40; i++) probe.step(1 / 60, { rel: 0, nowMs: i * 16 });
      return dist(probe.pose().pos, probe.pose().target);
    })();
    const dt = 1 / fps;
    let prevDir: V3 | null = null, minD = Infinity, maxTurn = 0;
    for (let i = 0; i < fps * 20; i++) {
      cam.step(dt, { rel: 0, nowMs: i * dt * 1000 });
      const { pos, target } = cam.pose();
      const d = dist(pos, target);
      minD = Math.min(minD, d);
      const dir = { x: (target.x - pos.x) / d, y: (target.y - pos.y) / d, z: (target.z - pos.z) / d };
      if (prevDir) {
        const c = Math.min(1, dir.x * prevDir.x + dir.y * prevDir.y + dir.z * prevDir.z);
        maxTurn = Math.max(maxTurn, (Math.acos(c) * 180) / Math.PI / dt);
      }
      prevDir = dir;
    }
    expect(minD).toBeGreaterThanOrEqual(0.95 * Math.min(40.8, goalDist));
    expect(maxTurn).toBeLessThanOrEqual(90);
    expect(cam.pose().pos.z).toBeGreaterThan(0); // it did get round to the south
  });

  it("the cinema ↔ follow toggle carries velocity (no stop mid-flight)", () => {
    const { cam } = farSwitchRig();
    const dt = 1 / 60;
    let t = 0, prev = cam.pose().pos, speed = 0;
    while (t < 62) { t += dt; cam.step(dt, { rel: t, nowMs: t * 1000 }); const p = cam.pose().pos; speed = dist(p, prev) / dt; prev = p; }
    expect(speed).toBeGreaterThan(1); // mid-transit toward B
    cam.follow([{ x: 0, y: 0, z: 0 }, { x: 90, y: 5, z: 6 }], { l: -1, r: 1, t: 1, b: -1 });
    cam.setMode("follow");
    t += dt; cam.step(dt, { rel: t, nowMs: t * 1000 });
    const after = dist(cam.pose().pos, prev) / dt;
    expect(after).toBeGreaterThan(0.5 * speed); // zeroing velocity would give ≈ 0
    // the switch jolt is bounded (follow now also holds A_MAX, so this is far inside the bound)
    expect(Math.abs(after - speed) / dt).toBeLessThan(100);
  });

  it("(N2) a follow→cinema toggle at the peak of a follow sweep stays ≤ V_MAX and drifts ≤ 0.1·peak + 13 (follow is capped at V_MAX too)", () => {
    const cam = new CameraController({ fov: 45, aspect: 1.5, pos: { x: 0, y: 8, z: -80 }, target: { x: 0, y: 0, z: 0 } });
    cam.setMode("follow");
    cam.follow([{ x: -40, y: 0, z: -20 }, { x: 40, y: 6, z: 20 }], { l: -0.9, r: 0.9, t: 0.9, b: -0.9 });
    const dt = 1 / 60;
    let prev = cam.pose().pos, speed = 0, peak = 0, i = 0;
    for (; i < 60 * 20; i++) {
      cam.step(dt, { rel: 0, nowMs: i * 16 });
      const p = cam.pose().pos;
      speed = dist(p, prev) / dt;
      prev = p;
      if (speed > peak) peak = speed;
      else if (peak > 1 && speed < peak - 1e-6) break; // just past the peak (the start of the braking)
    }
    // follow has the cinema's hard limits (Plan 3 Task 1): its peak is V_MAX, so there is no excess to shed
    expect(peak).toBeGreaterThan(1);
    expect(peak).toBeLessThanOrEqual(V_MAX + 1e-6);
    const at = cam.pose().pos;
    cam.setMode("cinema");
    let maxDrift = 0;
    for (let k = 1; k <= 120; k++) {
      cam.step(dt, { rel: 0, nowMs: (i + k) * 16 });
      const p = cam.pose().pos;
      const v = dist(p, prev) / dt;
      prev = p;
      expect(v).toBeLessThanOrEqual(V_MAX + 1e-6);
      maxDrift = Math.max(maxDrift, dist(p, at));
    }
    expect(maxDrift).toBeLessThanOrEqual(0.1 * peak + 13); // brake overshoot τ·e0, plus ≤ 2 s at V_MAX
  });

  it("(N4) toggling to follow while a session is focused clears the focus", () => {
    const { cam } = farSwitchRig();
    cam.focusSession("b");
    for (let i = 0; i < 60; i++) cam.step(1 / 60, { rel: 0, nowMs: i * 16 });
    cam.follow([{ x: -100, y: 0, z: -10 }, { x: -80, y: 3, z: 10 }], { l: -1, r: 1, t: 1, b: -1 });
    cam.setMode("follow");
    for (let i = 0; i < 60 * 30; i++) cam.step(1 / 60, { rel: 0, nowMs: 1000 + i * 16 });
    expect(cam.pose().target.x).toBeLessThan(-80); // follow framing, not the focused tree at x = 90
  });

  it("a drag pauses the active camera for 9 s after it ends; a long drag never resumes mid-drag", () => {
    const cam = new CameraController({ fov: 45, aspect: 1.5, pos: { x: 0, y: 10, z: 60 }, target: { x: 0, y: 0, z: 0 } });
    cam.setMode("follow");
    cam.follow([{ x: 100, y: 0, z: 0 }, { x: 110, y: 2, z: 3 }], { l: -1, r: 1, t: 1, b: -1 });
    cam.setDragging(true, 1000);
    expect(cam.paused).toBe(true);
    expect(cam.setPose({ x: 5, y: 12, z: 50 }, { x: 1, y: 0, z: 0 })).toBe(true); // OrbitControls moved it
    expect(cam.step(1 / 60, { rel: 0, nowMs: 30_000 })).toBe(false); // 29 s into one drag: still paused
    expect(cam.pose().pos).toEqual({ x: 5, y: 12, z: 50 });
    cam.setDragging(false, 31_000);
    expect(cam.step(1 / 60, { rel: 0, nowMs: 31_000 + FOLLOW_PAUSE_MS - 1 })).toBe(false);
    cam.step(1 / 60, { rel: 0, nowMs: 31_000 + FOLLOW_PAUSE_MS + 1 });
    expect(cam.paused).toBe(false);
    expect(dist(cam.pose().pos, { x: 5, y: 12, z: 50 })).toBeLessThan(0.1);
  });

  it("userInteracted (wheel, click) pauses the cinematic camera too", () => {
    const { cam } = farSwitchRig();
    cam.userInteracted(0);
    const p = cam.pose().pos;
    expect(cam.step(1 / 60, { rel: 1, nowMs: 5000 })).toBe(false);
    expect(cam.pose().pos).toEqual(p);
    cam.step(1 / 60, { rel: 1, nowMs: FOLLOW_PAUSE_MS + 1 });
    expect(cam.paused).toBe(false);
  });

  it("setPose is ignored unless paused (sync during a user drag only)", () => {
    const cam = new CameraController({ fov: 45, aspect: 1, pos: { x: 1, y: 2, z: 3 }, target: { x: 0, y: 0, z: 0 } });
    expect(cam.setPose({ x: 9, y: 9, z: 9 }, { x: 0, y: 0, z: 0 })).toBe(false);
    expect(cam.pose().pos).toEqual({ x: 1, y: 2, z: 3 });
  });

  it("focusSession orbits that session's tree just outside its crown", () => {
    const { cam } = farSwitchRig();
    cam.userInteracted(0);
    cam.focusSession("b");
    expect(cam.paused).toBe(false);
    for (let i = 0; i < 60 * 60; i++) cam.step(1 / 60, { rel: 0, nowMs: i * 16 });
    const { pos, target } = cam.pose();
    const r = Math.hypot(pos.x - B.x, pos.z - B.z);
    expect(r).toBeGreaterThan(B.r + 0.5);
    expect(r).toBeLessThan(B.r + 10);
    expect(dist(target, { x: 91, y: 7, z: 4 })).toBeLessThan(0.5);
  });

  it("holds no shared state; ISO is frozen and copied", () => {
    expect(Object.isFrozen(ISO)).toBe(true);
    const a = farSwitchRig().cam;
    const b = new CameraController({ fov: 45, aspect: 1, pos: { x: 1, y: 2, z: 3 }, target: { x: 0, y: 0, z: 0 } });
    a.step(1 / 60, { rel: 5, nowMs: 0 });
    expect(b.pose().pos).toEqual({ x: 1, y: 2, z: 3 });
  });

  it("project() is finite when looking straight down", () => {
    const q = project({ x: 1, y: 0, z: 1 }, { x: 0, y: 10, z: 0 }, { x: 0, y: 0, z: 0 }, 45, 1);
    expect(Number.isFinite(q.x) && Number.isFinite(q.y)).toBe(true);
    expect(q.depth).toBeCloseTo(10, 9);
  });
});

describe("focus release (final review I3)", () => {
  it("a focused session is held LOOK + 10 s after the click, then while it has turns, then released (N3)", () => {
    const { cam } = farSwitchRig(); // b's turns at 80, 200, 320, 440
    cam.focusSession("b");
    expect(cam.focused).toBe("b");
    cam.step(1 / 60, { rel: 400, nowMs: 0 }); // the click, at shown time 400
    // b's last turn (440) landed 9 s ago — still growing, still focused
    cam.step(1 / 60, { rel: 449, nowMs: 16 });
    expect(cam.focused).toBe("b");
    // nothing of b in the last 40 s of the server's time, and the click is 51 s old
    cam.step(1 / 60, { rel: 451, nowMs: 32 });
    expect(cam.focused).toBeNull();
    expect(LOOK).toBe(30);
  });
  it("a click counts as activity: a session whose last turn is old is held LOOK + 10 s after the click (N3)", () => {
    const { cam } = farSwitchRig();
    cam.focusSession("b"); // its last turn, 440, is long past at 900: the row's own request is still in flight
    cam.step(1 / 60, { rel: 900, nowMs: 0 });
    expect(cam.focused).toBe("b");
    cam.step(1 / 60, { rel: 939.9, nowMs: 16 });
    expect(cam.focused).toBe("b");
    cam.step(1 / 60, { rel: 940.1, nowMs: 32 });
    expect(cam.focused).toBeNull();
  });
  it("a session the scene does not know is held only the click's LOOK + 10 s", () => {
    const { cam } = farSwitchRig();
    cam.focusSession("nobody");
    cam.step(1 / 60, { rel: 0, nowMs: 0 });
    expect(cam.focused).toBe("nobody");
    cam.step(1 / 60, { rel: 40.1, nowMs: 16 });
    expect(cam.focused).toBeNull();
  });
  it("toggling the camera mode releases the focus, in either direction; focusSession(null) releases it", () => {
    for (const mode of ["cinema", "follow"] as const) {
      const { cam } = farSwitchRig();
      cam.focusSession("b");
      cam.setMode(mode);
      expect(cam.focused, mode).toBeNull();
    }
    const { cam } = farSwitchRig();
    cam.focusSession("b");
    cam.focusSession(null);
    expect(cam.focused).toBeNull();
  });
  it("a row click in Wide switches the camera to cinematic and reports the change", () => {
    const { cam } = farSwitchRig();
    const seen: string[] = [];
    cam.onModeChange = (m) => seen.push(m);
    cam.setMode("follow");
    cam.focusSession("b");
    expect(cam.cameraMode).toBe("cinema");
    expect(cam.focused).toBe("b");
    expect(seen).toEqual(["follow", "cinema"]);
    cam.setMode("cinema"); // no change: not reported again
    expect(seen).toEqual(["follow", "cinema"]);
  });
});

describe("follow has the cinema's hard limits (hand-off jump, Plan 3 Task 1)", () => {
  const S = { l: -0.9, r: 0.9, t: 0.9, b: -0.9 };
  it.each([30, 60, 144])("at %i fps a far re-fit moves at ≤ V_MAX with acceleration ≤ A_MAX, and arrives without overshoot", (fps) => {
    const cam = new CameraController({ fov: 36, aspect: 1.6, pos: { x: 0, y: 20, z: 60 }, target: { x: 0, y: 3, z: 0 } });
    cam.setMode("follow");
    // the fit is ~860 units away (the card's 6 h layout vs the full window's 48 h one)
    cam.follow([{ x: 840, y: 0, z: -10 }, { x: 880, y: 8, z: 10 }], S);
    const dt = 1 / fps;
    let prev = cam.pose(), pv = 0, maxV = 0, maxA = 0, maxStep = 0;
    const steps: number[] = [];
    for (let i = 0; i < fps * 400; i++) {
      cam.step(dt, { rel: 0, nowMs: i * dt * 1000 });
      const p = cam.pose();
      const s = dist(p.pos, prev.pos), v = s / dt;
      maxStep = Math.max(maxStep, s);
      maxV = Math.max(maxV, v, dist(p.target, prev.target) / dt);
      if (v > pv) maxA = Math.max(maxA, (v - pv) / dt);
      pv = v;
      prev = p;
      steps.push(p.pos.x);
    }
    expect(maxV).toBeLessThanOrEqual(V_MAX + 1e-6);
    expect(maxA).toBeLessThanOrEqual(A_MAX + 1e-6);
    if (fps >= 60) expect(maxStep).toBeLessThanOrEqual(0.1);
    // arrived at the fit, and never went past it on x
    expect(cam.pose().target.x).toBeGreaterThan(840);
    expect(cam.pose().target.x).toBeLessThan(880);
    expect(Math.max(...steps) - steps.at(-1)!).toBeLessThan(0.5);
  });

  it("after importView the first data's far fit starts from rest: the first 30 frames at 60 fps move ≤ 0.1 each", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.6 });
    const v = { pos: { x: 10, y: 22, z: 70 }, target: { x: 5, y: 3, z: 0 } };
    cam.importView(v.pos, v.target, "follow");
    cam.placeInitial({ x: 900, y: 12, z: 40 }, { x: 880, y: 3, z: 0 }); // the first data: the import wins
    cam.follow([{ x: 840, y: 0, z: -10 }, { x: 880, y: 8, z: 10 }], S);
    let prev = cam.pose().pos;
    expect(dist(prev, v.pos)).toBeLessThan(1e-9);
    for (let i = 0; i < 30; i++) {
      cam.step(i === 0 ? 0.1 : 1 / 60, { rel: 0, nowMs: i * 17 }); // a long first frame (the data landing)
      const p = cam.pose().pos;
      expect(dist(p, prev), `frame ${i}`).toBeLessThanOrEqual(0.1);
      prev = p;
    }
  });
});

describe("follow: the first placement is the fit itself (the card opens framed, no 10 s glide)", () => {
  it("followPose is the fit's pose; placed there, the camera does not move", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.6 });
    expect(cam.followPose()).toBeNull();
    cam.setMode("follow");
    cam.follow([{ x: 100, y: 0, z: -10 }, { x: 140, y: 8, z: 10 }], { l: -0.9, r: 0.9, t: 0.9, b: -0.9 });
    const p = cam.followPose()!;
    expect(cam.followGoal()).toEqual(p.pos);
    cam.placeInitial(p.pos, p.target);
    const a = cam.pose();
    for (let i = 0; i < 120; i++) cam.step(1 / 60, { rel: 0, nowMs: i * 17 });
    expect(dist(cam.pose().pos, a.pos)).toBeLessThan(1e-6);
    expect(dist(cam.pose().target, a.target)).toBeLessThan(1e-6);
  });
});

describe("follow: cut after a jump, ≤ 3 s glide on a return (review I1)", () => {
  const S = { l: -0.9, r: 0.9, t: 0.9, b: -0.9 };
  const PTS = [{ x: 100, y: 0, z: -10 }, { x: 140, y: 8, z: 10 }];
  /** Steps at 60 fps for `maxS` s: when the camera came within 2 of its follow goal, the largest frame step, and the
   * overshoot (how far it moved away from the goal again after arriving). */
  const arrive = (cam: CameraController, t0: number, maxS = 20) => {
    const dt = 1 / 60, goal = cam.followGoal()!, from = cam.pose().pos;
    const d0 = dist(goal, from);
    let prev = from, maxStep = 0, firstStep = -1, over = 0, at: number | null = null, best = Infinity;
    for (let i = 1; i <= maxS * 60; i++) {
      cam.step(dt, { rel: 0, nowMs: t0 + i * dt * 1000 });
      const p = cam.pose().pos, s = dist(p, prev);
      if (firstStep < 0) firstStep = s;
      maxStep = Math.max(maxStep, s);
      prev = p;
      const dg = dist(p, cam.followGoal()!);
      if (at === null && dg < 2) at = i * dt;
      if (at !== null) (best = Math.min(best, dg)), (over = Math.max(over, dg - best));
    }
    return { at, maxStep, firstStep, over, d0 };
  };

  it("cutToFit puts a follow camera at its fit at once; it then holds still; refused in cinema or under a user hold", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.6, pos: { x: 0, y: 20, z: 60 }, target: { x: 0, y: 3, z: 0 } });
    cam.follow(PTS, S);
    expect(cam.cutToFit()).toBe(false); // cinema keeps its own rules
    cam.setMode("follow");
    cam.setDragging(true, 0);
    expect(cam.cutToFit()).toBe(false); // the user holds the camera
    cam.setDragging(false, 0);
    cam.resume();
    expect(cam.cutToFit()).toBe(true);
    const p = cam.followPose()!;
    expect(dist(cam.pose().pos, p.pos)).toBeLessThan(1e-9);
    const a = cam.pose();
    for (let i = 0; i < 60; i++) cam.step(1 / 60, { rel: 0, nowMs: 1000 + i * 17 });
    expect(dist(cam.pose().pos, a.pos)).toBeLessThan(1e-6);
  });

  it("a pan of 200 units left returns, after the 9 s hold, within 3 s; no frame step over d/30, no instant step, no overshoot", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.6 });
    cam.setMode("follow");
    cam.follow(PTS, S);
    expect(cam.cutToFit()).toBe(true);
    const at = cam.pose();
    // the user drags the view 200 units to the left (older trees), then lets go
    cam.setDragging(true, 0);
    cam.setPose({ ...at.pos, x: at.pos.x - 200 }, { ...at.target, x: at.target.x - 200 });
    cam.setDragging(false, 1000);
    // the 9 s hold: nothing moves
    for (let i = 0; i < 9 * 60; i++) cam.step(1 / 60, { rel: 0, nowMs: 1000 + i * (1000 / 60) });
    expect(cam.paused).toBe(true);
    expect(cam.pose().pos.x).toBeCloseTo(at.pos.x - 200, 6);
    const r = arrive(cam, 10_000 + 17);
    console.log(`pan return: ${JSON.stringify(r)}`);
    expect(r.at).not.toBeNull();
    expect(r.at!).toBeLessThanOrEqual(3);
    expect(r.maxStep).toBeLessThanOrEqual(r.d0 / 30);
    expect(r.firstStep).toBeLessThan(0.2); // it starts from rest
    expect(r.over).toBeLessThan(0.5);
    // arrived: it rests on the fit (the steady limits apply again)
    const p = cam.pose().pos;
    cam.step(1 / 60, { rel: 0, nowMs: 40_000 });
    expect(dist(cam.pose().pos, p)).toBeLessThan(0.1);
  });

  it("Cinematic → Wide arrives within 3 s", () => {
    const { cam } = farSwitchRig();
    const dt = 1 / 60;
    for (let t = 0; t < 30; t += dt) cam.step(dt, { rel: t, nowMs: t * 1000 });
    cam.follow([{ x: -10, y: 0, z: -12 }, { x: 160, y: 8, z: 12 }], S);
    cam.setMode("follow");
    const r = arrive(cam, 30_000);
    console.log(`cinema→wide: ${JSON.stringify(r)}`);
    expect(r.at).not.toBeNull();
    expect(r.at!).toBeLessThanOrEqual(3);
    expect(r.maxStep).toBeLessThanOrEqual(Math.max(0.1, r.d0 / 30));
    expect(r.over).toBeLessThan(0.5);
  });

  it("a steady re-fit (new data while following) keeps the steady limits: ≤ V_MAX", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.6 });
    cam.setMode("follow");
    cam.follow(PTS, S);
    cam.cutToFit();
    for (let i = 0; i < 300; i++) cam.step(1 / 60, { rel: 0, nowMs: i * 17 }); // past any window a cut opens
    cam.follow(PTS.map((p) => ({ ...p, x: p.x + 60 })), S); // the forest grew: the fit moves 60 units
    let prev = cam.pose().pos, maxV = 0;
    for (let i = 0; i < 60 * 20; i++) {
      cam.step(1 / 60, { rel: 0, nowMs: 6000 + i * 17 });
      const p = cam.pose().pos;
      maxV = Math.max(maxV, dist(p, prev) * 60);
      prev = p;
    }
    expect(maxV).toBeLessThanOrEqual(V_MAX + 1e-6);
  });
});

describe("follow returns: a moving fit keeps the deadline and never overshoots (re-review N1, N2)", () => {
  const S = { l: -0.9, r: 0.9, t: 0.9, b: -0.9 };
  const PTS = [{ x: 100, y: 0, z: -10 }, { x: 140, y: 8, z: 10 }];
  const fitAt = (cam: CameraController, dx: number) => cam.follow(PTS.map((p) => ({ ...p, x: p.x + dx })), S);
  /**
   * A 200-unit pan to the left, the 9 s hold, then the return at 60 fps with the fit moved by `moves` (s after the hold
   * ends, Δx). Overshoot: how far the camera ever got past the (current) fit along +x, the way it returns; `overAny`:
   * past the furthest fit it was ever given (a fit stepping back behind a camera that sits on it is not an overshoot).
   * Arrival: from when it stays within 2 of the final fit.
   */
  const panReturn = (moves: { at: number; dx: number }[]) => {
    const cam = new CameraController({ fov: 36, aspect: 1.6 });
    cam.setMode("follow");
    fitAt(cam, 0);
    cam.cutToFit();
    const a = cam.pose();
    cam.setDragging(true, 0);
    cam.setPose({ ...a.pos, x: a.pos.x - 200 }, { ...a.target, x: a.target.x - 200 });
    cam.setDragging(false, 1000);
    for (let i = 0; i < 9 * 60; i++) cam.step(1 / 60, { rel: 0, nowMs: 1000 + i * (1000 / 60) });
    const dt = 1 / 60, rest = [...moves].sort((p, q) => p.at - q.at);
    let at305 = Infinity;
    let shift = 0, over = -Infinity, overAny = -Infinity, gMax = -Infinity, arrived: number | null = null, maxStep = 0, prev = cam.pose().pos;
    for (let i = 1; i <= 6 * 60; i++) {
      const t = i * dt;
      while (rest.length && rest[0].at <= t) fitAt(cam, (shift += rest.shift()!.dx));
      cam.step(dt, { rel: 0, nowMs: 10_000 + i * dt * 1000 });
      const p = cam.pose().pos, g = cam.followGoal()!;
      over = Math.max(over, p.x - g.x);
      gMax = Math.max(gMax, g.x);
      overAny = Math.max(overAny, p.x - gMax);
      maxStep = Math.max(maxStep, dist(p, prev));
      prev = p;
      if (Math.abs(t - 3.05) < dt / 2) at305 = dist(p, g);
      if (dist(p, g) < 2) arrived ??= t;
      else arrived = null;
    }
    return { over: +over.toFixed(3), overAny: +overAny.toFixed(3), at305: +at305.toFixed(4), arrived: arrived === null ? null : +arrived.toFixed(3), maxStep: +maxStep.toFixed(3) };
  };

  it.each([
    [0.6, -3], [0.6, 3], [0.75, -3], [0.75, 3],
  ])("the fit moves at u = %s by %s units: no overshoot (< 0.5), arrival ≤ 3 s after the hold ends", (u, dx) => {
    const r = panReturn([{ at: u * RETURN_S, dx }]);
    console.log(`late move u=${u} dx=${dx}: ${JSON.stringify(r)}`);
    expect(r.over).toBeLessThan(0.5);
    expect(r.arrived).not.toBeNull();
    expect(r.arrived!).toBeLessThanOrEqual(3);
    expect(r.maxStep).toBeLessThanOrEqual(200 / 30);
  });

  it("repeated fit moves (±1 every 0.15 s through the return) cannot push arrival past 3 s", () => {
    const moves = Array.from({ length: 18 }, (_, k) => ({ at: 0.3 + k * 0.15, dx: k % 2 ? -1 : 1.2 }));
    const r = panReturn(moves);
    console.log(`repeated moves: ${JSON.stringify(r)}`);
    expect(r.arrived).not.toBeNull();
    expect(r.arrived!).toBeLessThanOrEqual(RETURN_LIMIT_S);
    expect(r.at305, "on the final fit by RETURN_LIMIT_S").toBeLessThan(0.05);
    expect(r.overAny).toBeLessThan(0.5);
  });

  it("noOvershootRate: a start rate toward the goal is held to 3·d/T; a rate away is kept", () => {
    const T = RETURN_S;
    // the reviewer's case: 2 units to go at V_MAX → without the clamp the path passes the goal by ~0.65
    const path = (v: number) => {
      let peak = -Infinity;
      for (let k = 0; k <= 1000; k++) {
        const u = k / 1000, u2 = u * u, u3 = u2 * u;
        peak = Math.max(peak, (u3 - 2 * u2 + u) * T * v + (3 * u2 - 2 * u3) * 2);
      }
      return peak - 2;
    };
    expect(path(V_MAX)).toBeGreaterThan(0.5);
    expect(noOvershootRate(0, 2, V_MAX, T)).toBeCloseTo(6 / T, 12);
    expect(path(noOvershootRate(0, 2, V_MAX, T))).toBeLessThan(1e-9);
    expect(noOvershootRate(0, 2, -4, T)).toBe(-4);
    expect(noOvershootRate(0, -2, -V_MAX, T)).toBeCloseTo(-6 / T, 12);
    expect(noOvershootRate(0, 50, 1, T)).toBe(1);
  });

  it("Cinematic → Wide while moving toward a near fit: the camera stops at the fit, no overshoot", () => {
    for (const lead of [2, 6, 15]) {
      // a camera moving at V_MAX in follow coordinates (straight along +x), handed to a fit `lead` units ahead
      const cam = new CameraController({ fov: 36, aspect: 1.6 });
      cam.setMode("follow");
      fitAt(cam, 0);
      cam.cutToFit();
      const a = cam.pose();
      cam.setMode("cinema");
      // replace the state: `lead` units short of the fit, moving toward it at V_MAX (what the cinema hands over)
      const st = cam as unknown as { pos: V3; target: V3; vel: V3; tvel: V3; lookDir: V3 };
      st.pos = { ...a.pos, x: a.pos.x - lead };
      st.target = { ...a.target, x: a.target.x - lead };
      st.vel = { x: V_MAX, y: 0, z: 0 };
      st.tvel = { x: V_MAX, y: 0, z: 0 };
      fitAt(cam, 0);
      cam.setMode("follow");
      let over = -Infinity, prevV = V_MAX, maxDv = 0, prev = cam.pose().pos;
      for (let i = 1; i <= 4 * 60; i++) {
        cam.step(1 / 60, { rel: 0, nowMs: 50_000 + i * 17 });
        const p = cam.pose().pos;
        over = Math.max(over, p.x - cam.followGoal()!.x);
        const v = (p.x - prev.x) * 60;
        maxDv = Math.max(maxDv, Math.abs(v - prevV));
        prevV = v;
        prev = p;
      }
      console.log(`toggle in motion, lead ${lead}: over ${over.toFixed(3)}, max Δv/frame ${maxDv.toFixed(2)}`);
      expect(over, `lead ${lead}`).toBeLessThan(0.5);
      expect(dist(cam.pose().pos, cam.followGoal()!), `lead ${lead}`).toBeLessThan(0.05);
    }
  });
});
