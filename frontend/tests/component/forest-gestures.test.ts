// frontend/tests/component/forest-gestures.test.ts
// Plan 4 Task 4: trackpad gestures. Two fingers pan, pinch (ctrlKey wheel) zooms to the pointer, Shift + two fingers
// orbits (above the horizon), Safari twist orbits, a real mouse wheel zooms; the card lets the page scroll until it is
// clicked into. Every gesture pauses the automatic camera (9 s) and stays above ground.
import { describe, expect, it } from "vitest";
import { CameraController, FOLLOW_PAUSE_MS, GROUND_CLEARANCE, type GroundFn, type V3 } from "@/components/forest/engine/camera";
import {
  HORIZON_MAX,
  WheelGate,
  applyOrbit,
  applyPan,
  applyZoom,
  isMouseWheel,
  panScale,
  twistOrbit,
  wheelIntent,
  type WheelLike,
} from "@/components/forest/engine/gestures";

const w = (o: Partial<WheelLike>): WheelLike => ({ deltaX: 0, deltaY: 0, deltaMode: 0, ctrlKey: false, shiftKey: false, ...o });
const dist = (a: V3, b: V3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const polar = (pos: V3, target: V3) => Math.acos((pos.y - target.y) / dist(pos, target));

describe("isMouseWheel: a real mouse wheel vs a trackpad", () => {
  it("line or page deltas are a mouse wheel", () => {
    expect(isMouseWheel(w({ deltaMode: 1, deltaY: 3 }))).toBe(true);
    expect(isMouseWheel(w({ deltaMode: 2, deltaY: 1 }))).toBe(true);
  });
  it("whole-pixel vertical notches (≥ 50 px, no x) are a mouse wheel", () => {
    expect(isMouseWheel(w({ deltaY: 100 }))).toBe(true);
    expect(isMouseWheel(w({ deltaY: -120 }))).toBe(true);
  });
  it("fractional, small or two-axis pixel deltas are a trackpad", () => {
    expect(isMouseWheel(w({ deltaY: 4.5 }))).toBe(false);
    expect(isMouseWheel(w({ deltaY: 12 }))).toBe(false);
    expect(isMouseWheel(w({ deltaX: 3, deltaY: 60 }))).toBe(false);
  });
  it("a pinch (ctrlKey) is never a mouse wheel", () => {
    expect(isMouseWheel(w({ ctrlKey: true, deltaY: 100 }))).toBe(false);
  });
});

describe("wheelIntent: what a wheel event does", () => {
  it("two-finger slide (no modifier, trackpad deltas) pans by the deltas in px", () => {
    expect(wheelIntent(w({ deltaX: 7.5, deltaY: -3.25 }), 800)).toEqual({ kind: "pan", dx: 7.5, dy: -3.25 });
  });
  it("pinch (ctrlKey) zooms: spreading the fingers (deltaY < 0) zooms in, factor < 1", () => {
    const i = wheelIntent(w({ ctrlKey: true, deltaY: -10 }), 800);
    expect(i.kind).toBe("zoom");
    if (i.kind === "zoom") expect(i.factor).toBeCloseTo(Math.exp(-0.1), 9);
    const o = wheelIntent(w({ ctrlKey: true, deltaY: 10 }), 800);
    if (o.kind === "zoom") expect(o.factor).toBeGreaterThan(1);
  });
  it("Shift + two fingers orbits (either axis: macOS swaps a shifted slide to x)", () => {
    const a = wheelIntent(w({ shiftKey: true, deltaX: 10, deltaY: 5 }), 800);
    expect(a.kind).toBe("orbit");
    if (a.kind === "orbit") (expect(a.dTheta).toBeCloseTo(-0.06, 9), expect(a.dPhi).toBeCloseTo(-0.03, 9));
  });
  it("a real mouse wheel zooms, gently, line deltas scaled to px", () => {
    const i = wheelIntent(w({ deltaY: 100 }), 800);
    expect(i.kind).toBe("zoom");
    if (i.kind === "zoom") expect(i.factor).toBeCloseTo(Math.exp(0.15), 9);
    const l = wheelIntent(w({ deltaMode: 1, deltaY: 3 }), 800);
    if (l.kind === "zoom") expect(l.factor).toBeCloseTo(Math.exp(3 * 16 * 0.0015), 9);
  });
  it("a line-mode wheel (Firefox mouse) zooms, never pans", () => {
    expect(wheelIntent(w({ deltaMode: 1, deltaY: 1 }), 800).kind).toBe("zoom");
  });
  it("page-mode deltas count a page as the view height", () => {
    const i = wheelIntent(w({ deltaMode: 2, deltaY: 1 }), 400);
    if (i.kind === "zoom") expect(i.factor).toBeCloseTo(Math.exp(400 * 0.0015), 9);
  });
});

describe("twistOrbit: Safari rotation", () => {
  it("turns the azimuth by the rotation change (degrees), no tilt", () => {
    const t = twistOrbit(10, 25);
    expect(t.dPhi).toBe(0);
    expect(t.dTheta).toBeCloseTo((-15 * Math.PI) / 180, 9);
  });
});

describe("applyPan: the slide moves camera and target together, scaled by distance", () => {
  const pos = { x: 0, y: 10, z: 30 }, target = { x: 0, y: 2, z: 0 };
  it("keeps the offset, moves sideways for dx and is proportional to the distance", () => {
    const p = applyPan(pos, target, 100, 0);
    expect(p.pos.x - pos.x).toBeCloseTo(p.target.x - target.x, 9);
    expect(dist(p.pos, p.target)).toBeCloseTo(dist(pos, target), 9);
    expect(Math.abs(p.target.x - target.x)).toBeCloseTo(100 * panScale(dist(pos, target)), 9);
    const far = { x: 0, y: 20, z: 60 }, q = applyPan(far, target, 100, 0);
    expect(Math.abs(q.target.x - target.x)).toBeGreaterThan(Math.abs(p.target.x - target.x) * 1.8);
  });
  it("dy moves along the camera's up (screen-vertical)", () => {
    const p = applyPan(pos, target, 0, 100);
    expect(p.target.x).toBeCloseTo(0, 9);
    expect(Math.abs(p.target.y - target.y)).toBeGreaterThan(0.1);
  });
});

describe("applyOrbit: Shift + two fingers / twist", () => {
  const target = { x: 5, y: 2, z: 0 };
  it("turns about the target at the same distance", () => {
    const pos = { x: 5, y: 10, z: 30 };
    const p = applyOrbit(pos, target, 0.5, 0);
    expect(p.target).toEqual(target);
    expect(dist(p.pos, target)).toBeCloseTo(dist(pos, target), 9);
    expect(p.pos.x).not.toBeCloseTo(pos.x, 3);
  });
  it("never tilts below the horizon, however far it is pushed", () => {
    let pos: V3 = { x: 5, y: 10, z: 30 };
    for (let i = 0; i < 200; i++) pos = applyOrbit(pos, target, 0.01, 0.05).pos;
    expect(polar(pos, target)).toBeLessThanOrEqual(HORIZON_MAX + 1e-9);
    expect(pos.y).toBeGreaterThan(target.y);
  });
  it("a camera already lower than the limit is not snapped up (it may only tilt up)", () => {
    const low = { x: 5, y: 2.1, z: 30 }; // polar ≈ 89.8°
    const p = applyOrbit(low, target, 0.2, 0);
    expect(polar(p.pos, target)).toBeCloseTo(polar(low, target), 9);
    const up = applyOrbit(low, target, 0, -0.2);
    expect(polar(up.pos, target)).toBeLessThan(polar(low, target));
    const down = applyOrbit(low, target, 0, 0.2);
    expect(polar(down.pos, target)).toBeCloseTo(polar(low, target), 9);
  });
});

describe("applyZoom: toward the pointer", () => {
  const pos = { x: 0, y: 10, z: 40 }, target = { x: 0, y: 5, z: 0 };
  const opts = { fovDeg: 36, aspect: 1.6, minDist: 6, maxDist: 220 };
  it("at the centre it dollies straight in by the factor", () => {
    const p = applyZoom(pos, target, 0, 0, 0.5, opts);
    expect(dist(p.pos, p.target)).toBeCloseTo(dist(pos, target) * 0.5, 6);
    expect(p.target.x).toBeCloseTo(0, 6);
    expect(p.target.y).toBeCloseTo(5, 6);
  });
  it("off-centre, the point under the pointer stays put: zooming in shifts the target toward it", () => {
    const p = applyZoom(pos, target, 0.8, 0, 0.5, opts);
    expect(p.target.x).toBeGreaterThan(1); // right of screen = +x for a camera looking down −z
  });
  it("the distance is clamped to [minDist, maxDist]", () => {
    expect(dist(applyZoom(pos, target, 0, 0, 0.01, opts).pos, target)).toBeCloseTo(6, 6);
    const out = applyZoom(pos, target, 0, 0, 100, opts);
    expect(dist(out.pos, out.target)).toBeCloseTo(220, 6);
  });
});

describe("WheelGate: the card lets the page scroll until it is clicked into", () => {
  it("the full window always captures", () => {
    const g = new WheelGate(false);
    expect(g.captures(w({ deltaY: 5 }))).toBe(true);
    expect(g.captures(w({ deltaY: 100 }))).toBe(true);
  });
  it("an unengaged card passes slides and wheels through, but a pinch still zooms the card", () => {
    const g = new WheelGate(true);
    expect(g.captures(w({ deltaY: 5.5 }))).toBe(false);
    expect(g.captures(w({ deltaY: 100 }))).toBe(false);
    expect(g.captures(w({ shiftKey: true, deltaX: 4 }))).toBe(false);
    expect(g.captures(w({ ctrlKey: true, deltaY: 3 }))).toBe(true);
  });
  it("a Safari twist orbits the card only once it is engaged, like Shift + two fingers (final review M7)", () => {
    expect(new WheelGate(false).capturesTwist()).toBe(true);
    const g = new WheelGate(true);
    expect(g.capturesTwist()).toBe(false);
    g.pointerDown(true);
    expect(g.capturesTwist()).toBe(true);
    g.key("Escape");
    expect(g.capturesTwist()).toBe(false);
  });
  it("a click or tap into the card engages it; a pointer-down elsewhere releases it", () => {
    const g = new WheelGate(true);
    g.pointerDown(true);
    expect(g.engaged).toBe(true);
    expect(g.captures(w({ deltaY: 5.5 }))).toBe(true);
    g.pointerDown(false);
    expect(g.engaged).toBe(false);
    expect(g.captures(w({ deltaY: 5.5 }))).toBe(false);
    g.pointerDown(true);
    g.release();
    expect(g.captures(w({ deltaY: 5.5 }))).toBe(false);
  });
  it("Esc and a window blur release an engaged card; other keys do not (Plan 4 T5: the focus cue goes with them)", () => {
    const g = new WheelGate(true);
    g.pointerDown(true);
    g.key("a");
    g.key("Enter");
    expect(g.engaged).toBe(true);
    g.key("Escape");
    expect(g.engaged).toBe(false);
    g.pointerDown(true);
    g.blur();
    expect(g.engaged).toBe(false);
  });
  it("reports each change of engagement once (the card's focus ring follows it)", () => {
    const seen: boolean[] = [];
    const g = new WheelGate(true, (e) => seen.push(e));
    g.pointerDown(true);
    g.pointerDown(true); // still engaged: no report
    g.key("Escape");
    g.key("Escape");
    g.pointerDown(false);
    g.pointerDown(true);
    g.blur();
    g.release();
    expect(seen).toEqual([true, false, true, false]);
  });
  it("the full window never reports engagement (it always captures; no focus cue)", () => {
    const seen: boolean[] = [];
    const g = new WheelGate(false, (e) => seen.push(e));
    g.pointerDown(true);
    g.key("Escape");
    expect(seen).toEqual([]);
    expect(g.engaged).toBe(false);
  });
});

describe("a gesture is a user interaction: the camera holds 9 s, stays above ground, and Reset view works", () => {
  const hill: GroundFn = (x, z) => 4 + 0 * x * z;
  it("userInteracted + setPose: adopted, clamped above ground, released after FOLLOW_PAUSE_MS", () => {
    const ctl = new CameraController({ fov: 36, aspect: 1.6, pos: { x: 0, y: 12, z: 40 }, target: { x: 0, y: 5, z: 0 } });
    ctl.setGround(hill);
    // a pan that would sink the camera into the hill
    const p = applyPan({ x: 0, y: 12, z: 40 }, { x: 0, y: 5, z: 0 }, 0, 4000);
    ctl.userInteracted(1000);
    expect(ctl.setPose(p.pos, p.target)).toBe(true);
    const q = ctl.pose();
    expect(q.pos.y).toBeGreaterThanOrEqual(hill(q.pos.x, q.pos.z) + GROUND_CLEARANCE - 1e-9);
    expect(ctl.paused).toBe(true);
    ctl.resetView();
    expect(ctl.paused).toBe(false);
    ctl.userInteracted(2000);
    expect(ctl.paused).toBe(true);
    expect(FOLLOW_PAUSE_MS).toBe(9000);
  });
});
