/**
 * The geometry Web Worker: builds one tree per message and returns its buffers with every ArrayBuffer transferred.
 * Protocol: in `BuildJob` ({ id, key, tree, cut, knobs, norm, place, traits }), out `{ id, buffers }` or
 * `{ id, error }`. `traits` are the tree's traits frozen at first sight (R17).
 *
 * The scene draws the tree's voxel cells (voxel.ts). They only append as the cut moves on, so a rebuild swap needs no
 * morph: the next build shows the same cubes plus new ones, which scale in from their `born` in the shader. The reply
 * carries the cells, the nodes, the crown and the framing points; the voxelizer's inputs stay in the worker.
 */
import type { Knobs } from "../species";
import type { Tree, Traits } from "../types";
import { type LevelCells, shownLevels } from "../levels";
import { type Norm, type TreeBuffers, buildTreeGeometry } from "./index";

export interface BuildJob {
  /** Job id: a result whose id is not the tree's current job is stale and dropped. */
  id: number;
  /** Tree id, or the tree id, a separator and k for the ghost of the tree's k-th place. */
  key: string;
  tree: Tree;
  cut: number;
  knobs: Knobs;
  norm: Norm;
  place: { x: number; z: number; s: number };
  traits: Traits;
}

/**
 * What the main thread gets: per level of detail only the cells drawn at the cut (levels.ts; the main thread does no
 * O(cells) work at a swap), the voxelizer's cell count, and what the camera and picking need (not the old look's
 * meshes, not the full solid).
 */
export type BuiltBuffers = Omit<TreeBuffers, "vox" | "voxels"> & { cut: number; levels: LevelCells[]; cells: number };
export type BuildReply = { id: number; buffers: BuiltBuffers } | { id: number; error: string };

/** Builds the tree at the job's cut, as the scene shows it. */
export function buildShown(job: Omit<BuildJob, "id" | "key">): BuiltBuffers {
  const { vox: _v, voxels, ...shown } = buildTreeGeometry(job.tree, job.cut, job.knobs, job.norm, job.place, job.traits);
  return { ...shown, cut: job.cut, levels: shownLevels(voxels, job.cut), cells: voxels.count };
}

/** Every distinct ArrayBuffer in the buffers. */
export function transferables(b: BuiltBuffers): ArrayBuffer[] {
  const arrays = [b.points, ...b.levels.flatMap((l) => [l.coords, l.kind, l.shade, l.born, l.shown])];
  return [...new Set(arrays.map((a) => a.buffer as ArrayBuffer))];
}

export function handleBuild(job: BuildJob): { reply: BuildReply; transfer: ArrayBuffer[] } {
  try {
    const buffers = buildShown(job);
    return { reply: { id: job.id, buffers }, transfer: transferables(buffers) };
  } catch (e) {
    return { reply: { id: job.id, error: e instanceof Error ? e.message : String(e) }, transfer: [] };
  }
}

// Wire up only inside a dedicated worker (not when a test or the main thread imports this module).
declare const WorkerGlobalScope: unknown;
if (typeof WorkerGlobalScope !== "undefined" && typeof self !== "undefined") {
  const scope = self as unknown as {
    onmessage: ((e: MessageEvent<BuildJob>) => void) | null;
    postMessage(m: BuildReply, transfer: ArrayBuffer[]): void;
  };
  scope.onmessage = (e) => {
    const { reply, transfer } = handleBuild(e.data);
    scope.postMessage(reply, transfer);
  };
}
