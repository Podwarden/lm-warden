// frontend/tests/component/forest-world.test.ts
// Plan 4 Task 3, the world: stepped terraces (the ground height, fixed in the world), boxy clouds above the horizon,
// an always-day sky. Pure helpers only (jsdom has no WebGL).
import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  CLOUDS, CLOUD_BOTTOM, TAPER, TERRACE_STEP, TERRACE_ZMAX, cameraGround, cloudPose, groundHeight, hillsIn, landHeight, terraceHeight, terraceSpan,
} from "@/components/forest/engine/land";
import { SKY_FS } from "@/components/forest/engine/sky";
import { createQuietMarks } from "@/components/forest/engine/quiet";
import { createGround } from "@/components/forest/engine/ground";
import type { ForestState } from "@/lib/forest/types";
import { BASE_W } from "@/lib/forest/timeline";

const lcg = (seed: number) => {
  let q = seed;
  return () => (q = (q * 16807) % 2147483647) / 2147483647;
};
/** The timeline's width shape: BASE_W ± 1.8, wider where many trees stand. */
const spit = (extra: number) => (x: number, side: -1 | 1) =>
  side < 0 ? BASE_W + extra + 1.2 * Math.sin(x * 0.05) + 0.6 * Math.sin(x * 0.17 + 1) : BASE_W + extra + 1.2 * Math.sin(x * 0.06 + 2) + 0.6 * Math.sin(x * 0.19);

describe("terraces: stepped hills fixed in the world", () => {
  const hills = hillsIn(-200, 2000);

  it("there are terraces: boxes of 1–3 tiers, each tier a 0.5 step up, inside the grass on either side", () => {
    expect(hills.length).toBeGreaterThan(80);
    expect(hills.some((h) => h.tiers.length === 3)).toBe(true);
    expect(hills.some((h) => h.tiers[0].z0 < 0) && hills.some((h) => h.tiers[0].z1 > 0)).toBe(true);
    for (const h of hills) {
      h.tiers.forEach((t, k) => {
        expect(t.top).toBeCloseTo(TERRACE_STEP * (k + 1), 12);
        expect(t.x1).toBeGreaterThan(t.x0);
        expect(t.z1).toBeGreaterThan(t.z0);
        // inside the grass of the narrowest spit (BASE_W − 1.8), with a margin
        expect(Math.max(Math.abs(t.z0), Math.abs(t.z1))).toBeLessThanOrEqual(TERRACE_ZMAX + 1e-9);
        // each tier sits on the one below
        if (k) {
          const b = h.tiers[k - 1];
          expect(t.x0 >= b.x0 && t.x1 <= b.x1 && t.z0 >= b.z0 && t.z1 <= b.z1).toBe(true);
        }
      });
    }
    expect(TERRACE_ZMAX).toBeLessThan(BASE_W - 1.8);
  });

  it("groundHeight matches the terraces exactly: a tier's top inside it, the land's own profile elsewhere", () => {
    const w = spit(0), x0 = -1000, x1 = 5000;
    const r = lcg(5);
    let onTop = 0;
    for (let i = 0; i < 20_000; i++) {
      const x = r() * 1500, z = (r() - 0.5) * 40;
      const tops = hills.flatMap((h) => h.tiers.filter((t) => x >= t.x0 && x < t.x1 && z >= t.z0 && z < t.z1).map((t) => t.top));
      const g = groundHeight(x, z, w, x0, x1);
      if (tops.length) {
        onTop++;
        expect(g).toBe(Math.max(...tops)); // exactly the step height
        expect(terraceHeight(x, z)).toBe(Math.max(...tops));
      } else {
        expect(terraceHeight(x, z)).toBe(0);
        expect(g).toBe(landHeight(x, z, w, x0));
      }
    }
    expect(onTop).toBeGreaterThan(1000);
  });

  it("terraces are drawn (and count as ground) only where the strip does not taper", () => {
    const [a, b] = terraceSpan(0, 1000);
    expect(a).toBe(TAPER);
    expect(b).toBe(1000 - 20 - TAPER);
    for (const h of hillsIn(a, b)) for (const t of h.tiers) expect(t.x0 >= a && t.x1 <= b).toBe(true);
    // a hill cut by the span's end is not a hill there
    const cut = hills.find((h) => h.tiers[0].x0 < 400 && h.tiers[0].x1 > 400)!;
    if (cut) {
      const t = cut.tiers[0], x = (t.x0 + t.x1) / 2, z = (t.z0 + t.z1) / 2;
      expect(groundHeight(x, z, spit(0), 400 - TAPER, 5000)).toBeLessThan(0.1);
      expect(groundHeight(x, z, spit(0), -1000, 5000)).toBeGreaterThanOrEqual(TERRACE_STEP);
    }
  });

  it("the terrace hash gives the same heights as the spit grows (longer, wider, another range start)", () => {
    const r = lcg(9);
    const short = hillsIn(-200, 600), long = hillsIn(-500, 3000);
    const key = (h: (typeof short)[number]) => JSON.stringify(h.tiers);
    const longKeys = new Set(long.map(key));
    for (const h of short) expect(longKeys.has(key(h))).toBe(true);
    for (let i = 0; i < 5000; i++) {
      const x = r() * 580, z = (r() - 0.5) * 2 * TERRACE_ZMAX;
      const a = groundHeight(x, z, spit(0), -1000, 600 + TAPER + 20);
      const b = groundHeight(x, z, spit(2 * 4.4), -1000, 4000); // wider (many trees) and much longer
      if (terraceHeight(x, z) > 0) expect(b).toBe(a);
    }
  });

  it("the camera's floor is the ground (≥ it everywhere, equal on terrace tops and flat land) and has no steps", () => {
    const w = spit(0), r = lcg(13), tiers = hillsIn(-200, 2000).flatMap((h) => h.tiers);
    const out = (t: (typeof tiers)[number], x: number, z: number) => Math.hypot(Math.max(t.x0 - x, 0, x - t.x1), Math.max(t.z0 - z, 0, z - t.z1));
    let equal = 0;
    for (let i = 0; i < 20_000; i++) {
      const x = r() * 800, z = (r() - 0.5) * 40;
      const g = groundHeight(x, z, w, -1000, 5000), c = cameraGround(x, z, w, -1000, 5000);
      expect(c).toBeGreaterThanOrEqual(g);
      // no tier above the ground here reaches within its own height difference: the floor is the ground itself
      if (tiers.every((t) => t.top <= g || out(t, x, z) >= t.top - g)) (expect(c).toBe(g), equal++);
      // Lipschitz ≤ 1 + the land's own slope: a 0.05 step never lifts it more than 0.06
      const c2 = cameraGround(x + 0.05, z, w, -1000, 5000);
      expect(Math.abs(c2 - c)).toBeLessThanOrEqual(0.06);
    }
    expect(equal).toBeGreaterThan(10_000);
  });
});

describe("clouds: boxy, above the horizon, parallax", () => {
  it("every cloud's lowest box stays above eye level (the horizon) for any camera height and time", () => {
    const target = new THREE.Vector3(120, 3, 0);
    for (const camY of [0, 0.5, 2, 13, 60, 400, 1500])
      for (const t of [0, 10, 333, 5000])
        for (let i = 0; i < CLOUDS.length; i++) {
          const p = cloudPose(i, t, { camY, target });
          expect(p.y - CLOUD_BOTTOM * p.s).toBeGreaterThan(camY);
        }
  });

  it("nearer layers are bigger and drift faster (parallax), right to left, and stay with the view", () => {
    const target = new THREE.Vector3(0, 3, 0);
    const near = CLOUDS.findIndex((c) => c.layer === 2), far = CLOUDS.findIndex((c) => c.layer === 0);
    const v = (i: number) => cloudPose(i, 0.5, { camY: 10, target }).x - cloudPose(i, 0, { camY: 10, target }).x;
    expect(v(near)).toBeLessThan(v(far));
    expect(v(far)).toBeLessThan(0);
    const z = (i: number) => cloudPose(i, 0, { camY: 10, target }).z;
    expect(z(near)).toBeGreaterThan(z(far)); // nearer the spit (the camera looks toward −z)
    const at = cloudPose(near, 0, { camY: 10, target: new THREE.Vector3(1000, 3, 0) });
    expect(Math.abs(at.x - 1000)).toBeLessThanOrEqual(350);
  });
});

describe("sky: always day", () => {
  it("the dome has no sun, moon, stars or clock", () => {
    expect(SKY_FS).not.toMatch(/uSun|uMoon|uDay|uTwi|star/i);
  });
});

describe("quiet stones", () => {
  it("are stone blocks (every face axis-aligned) on the ground under them", () => {
    const scene = new THREE.Scene();
    const q = createQuietMarks(scene, { shadows: false, baseAt: () => 1 });
    q.update([{ from: 100, to: 9000 }], (t) => t / 10, () => 15);
    const mesh = scene.children.find((o) => o instanceof THREE.InstancedMesh) as THREE.InstancedMesh;
    const n = mesh.geometry.attributes.normal;
    for (let i = 0; i < n.count; i++) expect([n.getX(i), n.getY(i), n.getZ(i)].filter((v) => Math.abs(v) > 1e-6).length).toBe(1);
    const m = new THREE.Matrix4(), p = new THREE.Vector3();
    mesh.getMatrixAt(0, m);
    p.setFromMatrixPosition(m);
    expect(p.y).toBe(1);
    q.dispose();
  });
});

describe("flora on the terraces", () => {
  it("a flower bed's daisies and a call's mushrooms stand on the ground under each of them", () => {
    const scene = new THREE.Scene();
    const g = createGround(scene, { shadows: false, baseAt: (x) => (x < 100 ? 1 : 0.5) });
    const st: ForestState = { t0: 0, now: 1000, range: "48h", trees: new Map(), flowers: [[100, 900, 350], [300, 50_000, 0]], models: {}, cursor: 1000 };
    g.setData(st, (t) => t, () => 15);
    g.tick(900, 0);
    g.grow(900, true);
    const ms: THREE.InstancedMesh[] = [];
    scene.traverse((o) => o instanceof THREE.InstancedMesh && ms.push(o));
    const m = new THREE.Matrix4(), p = new THREE.Vector3();
    let n = 0;
    for (const im of ms)
      for (let i = 0; i < im.count; i++) {
        im.getMatrixAt(i, m);
        p.setFromMatrixPosition(m);
        expect(p.y).toBe(p.x < 100 ? 1 : 0.5);
        n++;
      }
    expect(n).toBeGreaterThan(10);
    g.dispose();
  });
});
