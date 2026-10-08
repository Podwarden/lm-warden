/**
 * Pure, worker-safe tree geometry: `buildTreeGeometry` runs the mockup's `buildTree` (tree.ts), voxelizes it (the look
 * that is drawn, voxel.ts) and packs the result into typed arrays: the voxelizer's inputs and cells, the turn nodes,
 * the crown extent and the camera's framing points. No three.js here: the worker stays small and deterministic.
 */
import type { Knobs } from "../species";
import type { Traits, Tree } from "../types";
import { CHOSEN, type VoxelCells, voxelizeTree } from "../voxel";
import { type Norm, buildTree } from "./tree";
import { type VoxelInput, packVox } from "./voxin";

export type { Norm } from "./tree";
export type { VoxelInput } from "./voxin";

export interface NodeRef {
  /** The turn's joint on its limb, or its twig tip. */
  kind: "joint" | "twig";
  /** World position of the turn's joint or twig tip. */
  p: [number, number, number];
  /** Wire session id (shoots of one session share it). */
  sessionId: string;
  /** Turn index in that session. */
  turn: number;
  time: number;
}

export interface TreeBuffers {
  nodes: NodeRef[];
  /**
   * Horizontal reach from the tree's place (≥ 1.5) and top height (≥ 1): the extent of the voxel cells. A tree with
   * no cells yet falls back to the mockup's `crownOf` (its nodes).
   */
  crown: { r: number; top: number };
  /** Camera framing points, xyz: the voxel extent's box corners; a tree with no cells yet: mockup `P` (trunk base,
   * trunk top and every node). */
  points: Float32Array;
  /** The voxelizer's inputs: prefix-stable foliage points, blossom tips and wood centrelines with girth schedules. */
  vox: VoxelInput;
  /** The tree as voxel cells (`voxelizeTree` with the chosen look, seeded by the tree id). */
  voxels: VoxelCells;
}

/**
 * Builds one tree as of `cut` (history seconds relative to t0). Same inputs give bit-identical buffers, and a turn
 * after the cut changes nothing.
 * `traits` shape the habit (girth, height, limb angle). Pass the traits as first seen for this tree and keep passing
 * those: the server's whole-spell traits drift while the tree grows live, and a drifting habit would re-aim every
 * limb. Defaults to `tree.traits` (fine for finished trees and tests).
 */
export function buildTreeGeometry(
  tree: Tree, cut: number, knobs: Knobs, norm: Norm, place: { x: number; z: number; s: number }, traits: Traits = tree.traits,
): TreeBuffers {
  const t = buildTree(tree, cut, knobs, norm, place, traits);
  const vox = packVox(t.vox, place);
  const voxels = voxelizeTree({ vox }, CHOSEN, tree.id);
  let r = 1.5, top = 1, points: Float32Array | null = null;
  if (voxels.count) {
    // the voxel extent: cell corners about the place (the grid's origin); the framing points are its box's corners
    const e = CHOSEN.size, c = voxels.coords;
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    for (let i = 0; i < voxels.count; i++) {
      const x = c[i * 3] * e, z = c[i * 3 + 2] * e;
      r = Math.max(r, Math.hypot(Math.max(Math.abs(x), Math.abs(x + e)), Math.max(Math.abs(z), Math.abs(z + e))));
      top = Math.max(top, (c[i * 3 + 1] + 1) * e);
      (x0 = Math.min(x0, x)), (x1 = Math.max(x1, x + e)), (z0 = Math.min(z0, z)), (z1 = Math.max(z1, z + e));
    }
    const ox = vox.origin[0], oz = vox.origin[1], P: number[] = [];
    for (const x of [x0, x1]) for (const y of [0, top]) for (const z of [z0, z1]) P.push(ox + x, y, oz + z);
    points = new Float32Array(P);
  } else
    for (const p of t.P) {
      r = Math.max(r, Math.hypot(p.x - place.x, p.z - place.z));
      top = Math.max(top, p.y);
    }
  return {
    vox,
    voxels,
    nodes: t.nodes.map((n) => ({ kind: n.kind, p: [n.p.x, n.p.y, n.p.z], sessionId: n.ses.src, turn: n.t.i, time: n.t.time })),
    crown: { r, top },
    points: points ?? new Float32Array(t.P.flatMap((p) => [p.x, p.y, p.z])),
  };
}
