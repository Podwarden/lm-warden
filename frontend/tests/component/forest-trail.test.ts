// The transplant sparks (engine/flights createTrail): idle slots are never drawn. They sit at the origin, black, and
// with fog (which mixes toward its own colour) and additive blending they summed to a glowing dot at (0, 0, 0)
// (re-review: the small bright rectangle in shots-1h).
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { createTrail } from "@/components/forest/engine/flights";

describe("transplant sparks", () => {
  it("draw only the live sparks, without fog", () => {
    const scene = new THREE.Scene();
    const trail = createTrail(scene);
    const pts = scene.children.find((o) => (o as THREE.Points).isPoints) as THREE.Points;
    expect((pts.material as THREE.PointsMaterial).fog).toBe(false);
    trail.step(0);
    expect(pts.geometry.drawRange.count).toBe(0);
    trail.burst({ x: 5, z: 2 }, 1, 1000);
    trail.step(1000);
    expect(pts.geometry.drawRange.count).toBe(80);
    trail.step(1_000_000);
    expect(pts.geometry.drawRange.count).toBe(0);
    trail.dispose();
  });
});
