// frontend/tests/component/forest-camera-ground.test.ts
// Forest follow-ups A and B (brief 2026-10-06-forest-camera-range): the camera never goes underground, and "Reset view".
import { describe, expect, it } from "vitest";
import {
  CameraController,
  GROUND_CLEARANCE,
  RETURN_LIMIT_S,
  clampToGround,
  type CamEvent,
  type Crown,
  type GroundFn,
  type V3,
} from "@/components/forest/engine/camera";
import { cameraGround, groundHeight, landHeight, landVertex, terraceHeight } from "@/components/forest/engine/land";

const dist = (a: V3, b: V3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const lcg = (seed: number) => {
  let q = seed;
  return () => (q = (q * 16807) % 2147483647) / 2147483647;
};
/** A spit whose width swells and narrows along x, as the timeline's width does. */
const width = (x: number, side: -1 | 1) => 5 + 3 * Math.sin(x / 23 + (side > 0 ? 1 : 0)) + 1.5 * Math.cos(x / 7);
/** The real terrain function (the land's own profile) over that spit. */
const terrain: GroundFn = (x, z) => groundHeight(x, z, width);
/** The spit with its stepped terraces (Plan 4): a wide spit, the strip starting far back so the rig stands on terraces. */
const wide = (x: number, side: -1 | 1) => 15 + 1.2 * Math.sin(x * 0.05 + (side > 0 ? 2 : 0));
const terraced: GroundFn = (x, z) => groundHeight(x, z, wide, -1000, 5000);
/** The camera's floor over it: the terraces' continuous envelope (cameraGround ≥ groundHeight, no steps). */
const terracedFloor: GroundFn = (x, z) => cameraGround(x, z, wide, -1000, 5000);
/** A hill (for the planner and return paths): 9 units high at (45, 0). */
const hill: GroundFn = (x, z) => Math.max(terrain(x, z), 9 * Math.exp(-((x - 45) ** 2 + z ** 2) / (2 * 14 ** 2)));
const floorAt = (g: GroundFn, p: V3) => g(p.x, p.z) + GROUND_CLEARANCE;

describe("groundHeight: the land's own profile", () => {
  it("is the upper envelope of the strip's vertices (any noise), within 0.05, and never below the sea", () => {
    const r = lcg(7);
    for (let i = 0; i < 2000; i++) {
      const w = 1 + r() * 12, a = r(), nz = r();
      const v = landVertex(a, w, nz);
      for (const side of [-1, 1] as const) {
        const g = landHeight(40, side * v.z, () => w);
        expect(g).toBeGreaterThanOrEqual(Math.max(v.y, -0.3) - 1e-9);
        expect(g - Math.max(v.y, -0.3)).toBeLessThanOrEqual(0.05);
      }
    }
    expect(groundHeight(40, 500, () => 6)).toBe(-0.3); // open sea
    expect(groundHeight(-80, 0, () => 6)).toBe(-0.3); // before the strip starts
  });
});

describe("clampToGround", () => {
  it("lifts a point below ground + clearance to it, leaves others alone, and is continuous", () => {
    expect(clampToGround({ x: 0, y: -3, z: 0 }, hill).y).toBeCloseTo(floorAt(hill, { x: 0, y: 0, z: 0 }), 9);
    expect(clampToGround({ x: 0, y: 7, z: 0 }, hill)).toEqual({ x: 0, y: 7, z: 0 });
    // a raw path that dives through the hill in steps of 0.1: the clamped path never steps more than 0.3
    let prev = clampToGround({ x: 0, y: 2, z: 0 }, hill), maxStep = 0;
    for (let i = 1; i <= 900; i++) {
      const raw = { x: i * 0.1, y: 2 - i * 0.01, z: Math.sin(i / 40) * 3 };
      const p = clampToGround(raw, hill);
      expect(p.y).toBeGreaterThanOrEqual(floorAt(hill, p) - 1e-9);
      maxStep = Math.max(maxStep, dist(p, prev));
      prev = p;
    }
    expect(maxStep).toBeLessThanOrEqual(0.3);
  });
});

/** OrbitControls, as the scene drives it: rotate about the target (polar ≤ maxPolar), dolly, pan. */
function orbit(pos: V3, target: V3, dAz: number, dPol: number, scale: number, pan: V3, maxPolar: number) {
  const off = { x: pos.x - target.x, y: pos.y - target.y, z: pos.z - target.z };
  const r = Math.min(220, Math.max(6, Math.hypot(off.x, off.y, off.z) * scale));
  let az = Math.atan2(off.x, off.z), pol = Math.acos(Math.max(-1, Math.min(1, off.y / Math.hypot(off.x, off.y, off.z))));
  az += dAz;
  pol = Math.max(0.01, Math.min(maxPolar, pol + dPol));
  const t = { x: target.x + pan.x, y: target.y + pan.y, z: target.z + pan.z };
  return { pos: { x: t.x + r * Math.sin(pol) * Math.sin(az), y: t.y + r * Math.cos(pol), z: t.z + r * Math.sin(pol) * Math.cos(az) }, target: t };
}

function forestRig(ground: GroundFn) {
  const A: Crown = { x: 0, z: 0, r: 4, top: 9 }, B: Crown = { x: 90, z: 3, r: 5, top: 11 };
  const events: CamEvent[] = [];
  for (let t = 20; t < 4000; t += 60) events.push({ time: t, sessionId: "a", treeId: "A" }, { time: t + 30, sessionId: "b", treeId: "B" });
  const cam = new CameraController({ fov: 36, aspect: 1.6, pos: { x: -12, y: 6, z: 14 }, target: { x: 0, y: 4, z: 0 } });
  cam.setGround(ground);
  cam.setCrowns(new Map([["A", A], ["B", B]]));
  cam.setEvents(events);
  cam.setTips(new Map([["a", { x: 1, y: 5, z: 1 }], ["b", { x: 91, y: 7, z: 4 }]]));
  cam.follow([{ x: -10, y: 0, z: -8 }, { x: 100, y: 11, z: 8 }], { l: -0.9, r: 0.9, t: 0.9, b: -0.9 });
  return { cam, A, B };
}

describe("A. the camera never goes underground", () => {
  it.each([["the real terrain", terrain, terrain], ["a hill", hill, hill], ["stepped terraces", terraced, terracedFloor]] as const)(
    "%s: 1000 random drag / zoom / orbit / wait / mode / reset steps keep it ≥ ground + 0.5 and the target ≥ ground",
    (_, ground, floor) => {
      const r = lcg(ground === terrain ? 11 : ground === hill ? 29 : 37);
      const { cam } = forestRig(floor);
      let overTerrace = 0;
      let t = 0, now = 0, minClear = Infinity, minTarget = Infinity, maxDragStep = 0, lifted = 0;
      const dt = 1 / 60;
      const check = () => {
        const { pos, target } = cam.pose();
        minClear = Math.min(minClear, pos.y - ground(pos.x, pos.z));
        minTarget = Math.min(minTarget, target.y - ground(target.x, target.z));
        if (terraceHeight(pos.x, pos.z) > 0 && pos.y < 4) overTerrace++;
      };
      const frame = () => {
        t += dt;
        now += dt * 1000;
        cam.step(dt, { rel: t, nowMs: now });
        check();
      };
      for (let k = 0; k < 1000; k++) {
        const op = r();
        if (op < 0.45) {
          // a drag: OrbitControls moves the camera a little per frame (it may aim below the horizon, the scene's old
          // maxPolarAngle, or pan the target under the ground); the controller adopts each pose and clamps it
          cam.setDragging(true, now);
          const dAz = (r() - 0.5) * 0.06, dPol = (r() - 0.3) * 0.05, scale = 1 + (r() - 0.5) * 0.04;
          const pan = { x: (r() - 0.5) * 0.08, y: (r() - 0.6) * 0.05, z: (r() - 0.5) * 0.08 };
          for (let f = 0, n = 5 + Math.floor(r() * 40); f < n; f++) {
            const p = cam.pose();
            const o = orbit(p.pos, p.target, dAz, dPol, scale, pan, Math.PI * 0.56);
            // a slow drag: the raw pose moves ≤ 0.1 per frame (a far camera's orbit is scaled down to that)
            const s = Math.min(1, 0.1 / Math.max(1e-9, dist(o.pos, p.pos), dist(o.target, p.target)));
            const lerp = (a: V3, b: V3) => ({ x: a.x + (b.x - a.x) * s, y: a.y + (b.y - a.y) * s, z: a.z + (b.z - a.z) * s });
            const raw = lerp(p.pos, o.pos);
            if (raw.y < floorAt(floor, raw)) lifted++;
            cam.setPose(raw, lerp(p.target, o.target));
            const q = cam.pose();
            maxDragStep = Math.max(maxDragStep, dist(q.pos, p.pos));
            now += dt * 1000;
            check();
          }
          cam.setDragging(false, now);
        } else if (op < 0.6) {
          // a wheel zoom: one big dolly, toward or away
          cam.userInteracted(now);
          const p = cam.pose();
          const o = orbit(p.pos, p.target, 0, (r() - 0.2) * 0.4, r() < 0.5 ? 0.6 : 1.6, { x: 0, y: 0, z: 0 }, Math.PI * 0.56);
          cam.setDragging(true, now);
          if (o.pos.y < floorAt(floor, o.pos)) lifted++;
          cam.setPose(o.pos, o.target);
          cam.setDragging(false, now);
          check();
        } else if (op < 0.92) {
          for (let f = 0, n = Math.floor(r() * 900); f < n; f++) frame(); // up to 15 s: holds, resumes, glides, cinema
        } else if (op < 0.97) {
          cam.setMode(cam.cameraMode === "cinema" ? "follow" : "cinema");
        } else cam.resetView();
      }
      console.log(`fuzz (${_}): min camera clearance ${minClear.toFixed(4)}, min target clearance ${minTarget.toFixed(4)}, max drag step ${maxDragStep.toFixed(3)}, user poses lifted ${lifted}`);
      expect(lifted).toBeGreaterThan(20); // the sequences really do push the camera into the ground
      expect(minClear).toBeGreaterThanOrEqual(GROUND_CLEARANCE - 1e-6);
      expect(minTarget).toBeGreaterThanOrEqual(-1e-6);
      expect(maxDragStep).toBeLessThanOrEqual(0.3); // the clamp is continuous: a 0.1 raw step never becomes > 0.3
      if (ground === terraced) expect(overTerrace).toBeGreaterThan(100); // low over the terraces, not only above them
    },
  );

  it("a planned follow return over a hill stays above it, ≤ 3 s, no frame step > d/30", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.6 });
    cam.setGround(hill);
    cam.setMode("follow");
    cam.follow([{ x: 100, y: 0, z: -10 }, { x: 140, y: 8, z: 10 }], { l: -0.9, r: 0.9, t: 0.9, b: -0.9 });
    cam.cutToFit();
    // dragged far back, low, beyond the hill
    cam.setDragging(true, 0);
    cam.setPose({ x: -20, y: 2, z: 8 }, { x: -10, y: 1, z: 0 });
    cam.setDragging(false, 0);
    cam.resume();
    const goal = cam.followGoal()!, d0 = dist(cam.pose().pos, goal);
    let prev = cam.pose().pos, maxStep = 0, minClear = Infinity, at: number | null = null;
    for (let i = 1; i <= 6 * 60; i++) {
      cam.step(1 / 60, { rel: 0, nowMs: 100 + i * 16.7 });
      const p = cam.pose().pos;
      maxStep = Math.max(maxStep, dist(p, prev));
      minClear = Math.min(minClear, p.y - hill(p.x, p.z));
      if (at === null && dist(p, goal) < 2) at = i / 60;
      prev = p;
    }
    console.log(`return over a hill: d0 ${d0.toFixed(1)}, at ${at}, max step ${maxStep.toFixed(3)}, min clearance ${minClear.toFixed(3)}`);
    expect(minClear).toBeGreaterThanOrEqual(GROUND_CLEARANCE - 1e-6);
    expect(at).not.toBeNull();
    expect(at!).toBeLessThanOrEqual(RETURN_LIMIT_S);
    expect(maxStep).toBeLessThanOrEqual(d0 / 30);
  });

  it("the cinematic planner's flight from A to B over a hill stays above it, steps ≤ 0.3 per 60 fps frame", () => {
    const { cam } = forestRig(hill);
    let t = 0, prev = cam.pose().pos, maxStep = 0, minClear = Infinity;
    const seen = new Set<string | null>();
    while (t < 400) {
      t += 1 / 60;
      cam.step(1 / 60, { rel: t, nowMs: t * 1000 });
      const p = cam.pose().pos;
      maxStep = Math.max(maxStep, dist(p, prev));
      minClear = Math.min(minClear, p.y - hill(p.x, p.z));
      seen.add(cam.destination);
      prev = p;
    }
    console.log(`cinema over a hill: max step ${maxStep.toFixed(3)}, min clearance ${minClear.toFixed(3)}`);
    expect(seen.has("A") && seen.has("B")).toBe(true);
    expect(minClear).toBeGreaterThanOrEqual(GROUND_CLEARANCE - 1e-6);
    expect(maxStep).toBeLessThanOrEqual(0.3);
  });

  it("a hand-off imported underground (or a reset pose) lands above the ground; the look target too", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.6 });
    cam.setGround(hill);
    cam.importView({ x: 45, y: -5, z: 0 }, { x: 50, y: -2, z: 0 }, "follow");
    const p = cam.pose();
    expect(p.pos.y).toBeGreaterThanOrEqual(floorAt(hill, p.pos) - 1e-9);
    expect(p.target.y).toBeGreaterThanOrEqual(hill(p.target.x, p.target.z) - 1e-9);
    cam.placeInitial({ x: 45, y: 0, z: 3 }, { x: 40, y: 0, z: 0 });
    expect(cam.pose().pos.y).toBeGreaterThanOrEqual(floorAt(hill, cam.pose().pos) - 1e-9);
  });
});

describe("A. a held camera is clamped too (review M1)", () => {
  it("the ground rising under a camera the user holds lifts it at the next step; the frame core then applies the pose", async () => {
    const { FrameCore } = await import("@/components/forest/engine/frame");
    let level = 0;
    const cam = new CameraController({ fov: 36, aspect: 1.6, pos: { x: 0, y: 3, z: 20 }, target: { x: 0, y: 1, z: 0 } });
    cam.setGround(() => level);
    cam.setDragging(true, 0);
    cam.setPose({ x: 0, y: 1, z: 20 }, { x: 0, y: 0.5, z: 0 });
    cam.setDragging(false, 100);
    expect(cam.paused).toBe(true);
    level = 2; // the shore widened under it (or a switch made sea into land)
    expect(cam.step(1 / 60, { rel: 0, nowMs: 200 })).toBe(true);
    expect(cam.paused).toBe(true); // still the user's hold
    expect(cam.pose().pos.y).toBeGreaterThanOrEqual(2 + GROUND_CLEARANCE - 1e-9);
    expect(cam.pose().target.y).toBeGreaterThanOrEqual(2 - 1e-9);
    // the frame core hands a lifted pose to the scene even under the hold
    const core = new FrameCore(cam);
    core.intake({ t0: 0, now: 100, range: "6h", trees: new Map(), flowers: [], models: {}, cursor: 100 }, 0);
    level = 3;
    const out = core.frame(300, 1 / 60)!;
    expect(out.pose).not.toBeNull();
    expect(out.pose!.pos.y).toBeGreaterThanOrEqual(3 + GROUND_CLEARANCE - 1e-9);
  });
});

describe("B. Reset view", () => {
  const S = { l: -0.9, r: 0.9, t: 0.9, b: -0.9 };
  const PTS = [{ x: 100, y: 0, z: -10 }, { x: 140, y: 8, z: 10 }];

  it("follow: after a drag and a zoom, reset ends the hold and glides back within 3 s, no frame step > d/30", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.6 });
    cam.setGround(terrain);
    cam.setMode("follow");
    cam.follow(PTS, S);
    cam.cutToFit();
    const at = cam.pose();
    cam.setDragging(true, 0);
    cam.setPose({ ...at.pos, x: at.pos.x - 80 }, { ...at.target, x: at.target.x - 80 }); // drag
    cam.setDragging(false, 500);
    cam.userInteracted(600); // wheel
    cam.setDragging(true, 600);
    cam.setPose({ x: at.target.x - 80 + 4, y: 6, z: at.target.z + 9 }, { ...at.target, x: at.target.x - 80 }); // zoomed in
    cam.setDragging(false, 600);
    expect(cam.paused).toBe(true);
    cam.resetView();
    expect(cam.paused).toBe(false);
    const goal = cam.followGoal()!, d0 = dist(cam.pose().pos, goal);
    let prev = cam.pose().pos, maxStep = 0, back: number | null = null;
    for (let i = 1; i <= 5 * 60; i++) {
      cam.step(1 / 60, { rel: 0, nowMs: 700 + i * 16.7 }); // well inside the 9 s hold: only the reset moves it
      const p = cam.pose().pos;
      maxStep = Math.max(maxStep, dist(p, prev));
      if (back === null && dist(p, goal) < 0.05) back = i / 60;
      prev = p;
    }
    console.log(`reset (follow): d0 ${d0.toFixed(1)}, back in ${back} s, max step ${maxStep.toFixed(3)} (d/30 = ${(d0 / 30).toFixed(3)})`);
    expect(back).not.toBeNull();
    expect(back!).toBeLessThanOrEqual(3);
    expect(maxStep).toBeLessThanOrEqual(d0 / 30);
  });

  it("cinema: reset returns to the planner's current shot (the destination's orbit) within 3 s, no jump", () => {
    const { cam, A } = forestRig(terrain);
    const dt = 1 / 60;
    let t = 0;
    for (; t < 15; t += dt) cam.step(dt, { rel: t, nowMs: t * 1000 });
    expect(cam.destination).toBe("A");
    // the user drags far off and zooms out
    cam.setDragging(true, t * 1000);
    cam.setPose({ x: -60, y: 40, z: 70 }, { x: -20, y: 2, z: 10 });
    cam.setDragging(false, t * 1000);
    cam.resetView();
    expect(cam.paused).toBe(false);
    const d0 = dist(cam.pose().pos, { x: A.x, y: 5, z: A.z });
    let prev = cam.pose().pos, maxStep = 0, back: number | null = null;
    const ring = A.r + 3.2 + 1.5 + 2; // the arrival radius about the crown axis
    for (let i = 1; i <= 5 * 60; i++) {
      t += dt;
      cam.step(dt, { rel: t, nowMs: t * 1000 });
      const p = cam.pose().pos;
      maxStep = Math.max(maxStep, dist(p, prev));
      if (back === null && Math.hypot(p.x - A.x, p.z - A.z) <= ring && p.y < A.top + 4) back = i / 60;
      prev = p;
    }
    console.log(`reset (cinema): d0 ${d0.toFixed(1)}, back in ${back} s, max step ${maxStep.toFixed(3)}`);
    expect(cam.cameraMode).toBe("cinema");
    expect(back).not.toBeNull();
    expect(back!).toBeLessThanOrEqual(3);
    expect(maxStep).toBeLessThanOrEqual(d0 / 30);
    expect(cam.destination).toBe("A");
  });

  it("cinema: the reset glide never passes through another crown on its straight line (review I1)", () => {
    const { cam, A, B } = forestRig(terrain);
    const dt = 1 / 60;
    let t = 0;
    for (; t < 15; t += dt) cam.step(dt, { rel: t, nowMs: t * 1000 });
    expect(cam.destination).toBe("A");
    // dragged to the far side of B, low: the straight way back to A's orbit runs through B's crown
    cam.setDragging(true, t * 1000);
    cam.setPose({ x: B.x + 40, y: 5, z: B.z }, { x: B.x, y: 4, z: B.z });
    cam.setDragging(false, t * 1000);
    cam.resetView();
    const d0 = dist(cam.pose().pos, { x: A.x, y: 5, z: A.z });
    let prev = cam.pose().pos, maxStep = 0, back: number | null = null, inside = 0, crossed = false;
    const ring = A.r + 3.2 + 1.5 + 2;
    for (let i = 1; i <= 5 * 60; i++) {
      t += dt;
      cam.step(dt, { rel: t, nowMs: t * 1000 });
      const p = cam.pose().pos;
      maxStep = Math.max(maxStep, dist(p, prev));
      const h = Math.hypot(p.x - B.x, p.z - B.z);
      if (h < B.r + 1.2) crossed = true;
      if (h < B.r + 1.2 && p.y < B.top + 1 - 1e-6) inside++;
      if (back === null && Math.hypot(p.x - A.x, p.z - A.z) <= ring && p.y < A.top + 4) back = i / 60;
      prev = p;
    }
    console.log(`reset over a crown: crossed ${crossed}, frames inside ${inside}, back ${back} s, max step ${maxStep.toFixed(3)} (d0/30 ${(d0 / 30).toFixed(3)})`);
    expect(crossed).toBe(true); // the path does go over B
    expect(inside).toBe(0);
    expect(back).not.toBeNull();
    expect(back!).toBeLessThanOrEqual(3);
    expect(maxStep).toBeLessThanOrEqual(d0 / 30);
  });

  it("reset clears a focused session (and keeps the mode)", () => {
    const { cam } = forestRig(terrain);
    cam.focusSession("b");
    expect(cam.focused).toBe("b");
    cam.resetView();
    expect(cam.focused).toBeNull();
    expect(cam.cameraMode).toBe("cinema");
  });
});
