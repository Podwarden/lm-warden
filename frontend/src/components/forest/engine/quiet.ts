/**
 * Quiet-hour markers (coordinator ruling 2026-10-06): the timeline compresses every quiet stretch (no tree active) to at
 * most IDLE_CAP_U units of spit, so a night is a short step. Each compressed stretch gets a small pale stone on the
 * spit's south (camera-side) shore at its start; hovering it shows "6 h quiet" (panels/Tooltip). In the voxel look
 * (Plan 4) the stone is a small stack of grey stone blocks, standing on the ground under it (`baseAt`).
 */
import * as THREE from "three";
import { boxes } from "./voxels";
import type { QuietGap } from "@/lib/forest/timeline";

/** A marker's pick: the quiet stretch, in scene time (the scene turns it absolute). */
export type QuietPick = { kind: "quiet"; time: number; to: number };

const MAX_MARKS = 400;

/** The stone: blocks [w, h, d, x, y, z, colour], knee-high to a young tree (a 2.4-unit post stood as tall as the trees). */
export const STONE = [
  [0.75, 0.4, 0.6, 0, 0.2, 0, "#b4b0a4"], [0.5, 0.3, 0.45, -0.05, 0.55, 0.02, "#c9c5b8"], [0.25, 0.2, 0.25, 0.25, 0.1, 0.3, "#9d998e"],
] as const;

/** `baseAt`: the ground's height under a stone (land.ts `groundHeight`; flat 0 by default). */
export function createQuietMarks(
  scene: THREE.Scene,
  opts: { shadows: boolean; baseAt?: (x: number, z: number) => number; toon?: (o: THREE.MeshToonMaterialParameters) => THREE.Material },
) {
  const geo = boxes(STONE);
  const ownMat = !opts.toon;
  const mat = opts.toon?.({ vertexColors: true }) ?? new THREE.MeshToonMaterial({ vertexColors: true });
  const mesh = new THREE.InstancedMesh(geo, mat, MAX_MARKS);
  mesh.castShadow = opts.shadows;
  mesh.count = 0;
  mesh.frustumCulled = false;
  scene.add(mesh);
  let gaps: QuietGap[] = [];
  let key = "";
  const m4 = new THREE.Matrix4();
  return {
    /** Places a post per compressed stretch; true when anything changed. */
    update(next: QuietGap[], X: (t: number) => number, width: (x: number, side: -1 | 1) => number): boolean {
      const shown = next.slice(-MAX_MARKS);
      const k = shown.map((g) => `${g.from}:${g.to}`).join("|");
      if (k === key) return false;
      key = k;
      gaps = shown;
      shown.forEach((g, i) => {
        // at the run's start (re-review N6): X there is frozen once it is 20 min old, so the stone never slides as the
        // run grows or its smoothing settles
        const x = X(g.from) + 1, z = width(x, 1) * 0.92;
        m4.makeTranslation(x, opts.baseAt?.(x, z) ?? 0, z);
        mesh.setMatrixAt(i, m4);
      });
      mesh.count = shown.length;
      mesh.instanceMatrix.needsUpdate = true;
      return true;
    },
    pick(ray: THREE.Raycaster): { result: QuietPick; distance: number } | null {
      if (!mesh.count) return null;
      const hit = ray.intersectObject(mesh, false)[0];
      const g = hit?.instanceId !== undefined ? gaps[hit.instanceId] : undefined;
      return g && hit ? { result: { kind: "quiet", time: g.from, to: g.to }, distance: hit.distance } : null;
    },
    count: () => mesh.count,
    dispose() {
      scene.remove(mesh);
      geo.dispose();
      if (ownMat) mat.dispose();
      mesh.dispose();
    },
  };
}
