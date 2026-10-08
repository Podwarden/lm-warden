/**
 * README §4.3 — transplants, ghosts and emergence (wall-clock animations). Ported from forest-real.html
 * (`startFlight`, `flightPose`, `stepFlights` with its spark trail, `ghostFor`'s bare patch and fallen leaves,
 * `EMERGE`/`emergePose`). The pose maths is pure; the scene applies it to a tree's group.
 * - A transplanted tree lifts, glides along the spit and settles, only forward and never below ground; a long hop
 *   (> 30 units) dissolves into sparks at the old spot and enters the frame airborne, 30 units out.
 * - A tree that comes into being while time runs forward grows out of the ground over 2.5 s, ease-out.
 */
import * as THREE from "three";
import type { ClockStep } from "./clock";
import { Rng } from "@/lib/forest/geometry/random";
import { GHOST_SOIL, GHOST_STONE } from "./ghostColors";

export const FLIGHT_MS = 3200;
export const EMERGE_MS = 2500;
/** A hop longer than this (units) is shortened to it: the tree enters the frame airborne. */
export const FAR_HOP = 30;

/** Fly only when time moves forward in ordinary steps, to a later place (mockup `build`: `pi > shown`, small step). */
export function shouldFly(shownIdx: number | undefined, idx: number, step: ClockStep): boolean {
  return shownIdx !== undefined && idx > shownIdx && step.continuous && step.to > step.from && step.to - step.from < 1800;
}

export interface Flight {
  /** Start offset from the new place (from − to), clamped to ±FAR_HOP. */
  dx: number;
  dz: number;
  /** Tree scale. */
  s: number;
  far: boolean;
  t0: number;
  dur: number;
  /** The old base's height above the new one (terraces): the glide eases it out with the horizontal offset. */
  dy?: number;
}

/** `dy`: the old place's base height minus the new one's (the ground under each, land.ts terraces). */
export function makeFlight(from: { x: number; z: number }, to: { x: number; z: number }, s: number, nowMs: number, dy = 0): Flight {
  let dx = from.x - to.x;
  const far = Math.abs(dx) > FAR_HOP;
  if (far) dx = Math.sign(dx) * FAR_HOP;
  return { dx, dz: from.z - to.z, s, far, t0: nowMs, dur: FLIGHT_MS, dy: far ? 0 : dy };
}

export interface FlightPose {
  u: number;
  lift: number;
  /** Offset of the tree's base (the group origin: trees are built about their base) from its new place. */
  x: number;
  y: number;
  z: number;
  /** Bank about the base. */
  rotZ: number;
  done: boolean;
}

/** Mockup `flightPose`: ease-in-out glide, lift ≥ 0, a slight bank pivoting at the tree's base. */
export function flightPose(f: Flight, nowMs: number): FlightPose {
  const u = Math.min(1, Math.max(0, (nowMs - f.t0) / f.dur));
  const e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
  const lift = Math.max(
    0,
    f.far ? (7 * Math.pow(1 - e, 1.25) + 1.2 * Math.sin(Math.PI * u)) * f.s : Math.sin(Math.PI * Math.min(1, u * 1.05)) * (3 + Math.abs(f.dx) * 0.03) * f.s,
  );
  if (u >= 1) return { u, lift: 0, x: 0, y: 0, z: 0, rotZ: 0, done: true };
  return { u, lift, x: f.dx * (1 - e), y: lift + (f.dy ?? 0) * (1 - e), z: f.dz * (1 - e), rotZ: Math.sin(Math.PI * u) * 0.08 * Math.sign(f.dx), done: false };
}

/** Mockup `emergePose`: the tree scales from ~0 about its base (the group origin), ease-out over 2.5 s. */
export function emergePose(t0: number, nowMs: number): { s: number; done: boolean } {
  const u = Math.min(1, Math.max(0, (nowMs - t0) / EMERGE_MS));
  if (u >= 1) return { s: 1, done: true };
  return { s: Math.max(0.001, 1 - Math.pow(1 - u, 3)), done: false };
}

/** A tree group's wall-clock motion about its place: a transplant flight, an emergence and a glide. */
export interface Motion {
  flight?: Flight;
  emerge?: { t0: number };
  /** Decaying offset from the place (the timeline moved the place a little: the tree glides, no jump). */
  glide: { x: number; z: number };
  /** Where the base stands (the group origin). */
  place: { x: number; z: number } | null;
  /** "Bigger when far" (farscale.ts): a render-time scale about the base; 1 near. */
  far?: number;
  /** The base's height: the ground under the place (land.ts terraces). */
  base?: number;
  /** The ground's height at a world position: while a glide moves the tree, its base follows the ground under it
   * (final review I1: never floating over, never sunk into a terrace). Without it the base stays at `base`. */
  ground?: (x: number, z: number) => number;
}

/** A glide decays with this time constant (s). */
export const GLIDE_TAU = 0.6;

/** Puts `group` at its place plus this frame's motion (sparks under a flying base go to `trail`); true while moving. */
export function stepMotion(m: Motion, group: THREE.Object3D, nowMs: number, dt: number, trail: Trail): boolean {
  const pl = m.place ?? { x: 0, z: 0 };
  const moving = !!(m.flight || m.emerge || m.glide.x || m.glide.z);
  const k = Math.exp(-dt / GLIDE_TAU);
  m.glide.x = Math.abs(m.glide.x * k) < 1e-4 ? 0 : m.glide.x * k;
  m.glide.z = Math.abs(m.glide.z * k) < 1e-4 ? 0 : m.glide.z * k;
  const base = m.base ?? 0;
  let x = pl.x + m.glide.x, y = base, z = pl.z + m.glide.z, rot = 0, s = 1;
  // a glide eases the base with x and z: it stands on the ground under where it is drawn (a tier edge is a 0.5 step)
  if ((m.glide.x || m.glide.z) && m.ground) {
    const g = m.ground(x, z);
    if (Number.isFinite(g)) y = g;
  }
  if (m.flight) {
    const p = flightPose(m.flight, nowMs);
    x += p.x;
    y = base + p.y;
    z += p.z;
    rot = p.rotZ;
    if (p.done) m.flight = undefined;
    else trail.emit(x, y, z, nowMs);
  }
  if (m.emerge) {
    const p = emergePose(m.emerge.t0, nowMs);
    s = p.s;
    if (p.done) m.emerge = undefined;
  }
  group.position.set(x, y, z);
  group.rotation.z = rot;
  group.scale.setScalar(s * (m.far ?? 1));
  return moving;
}

const TRAIL_N = 420;
const TRAIL_MS = 1600;

/** The spark trail behind flying trees (mockup `TRAIL`, additive points that sink and fade over 1.6 s). */
export function createTrail(scene: THREE.Scene) {
  const pts: { x: number; y: number; z: number; t: number }[] = [];
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(TRAIL_N * 3), 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(TRAIL_N * 3), 3));
  // fog off and only the live sparks drawn: the idle slots sit at the origin, black, and with fog (which mixes toward
  // its own colour) and additive blending 38 of them summed to a glowing dot at (0, 0, 0) (re-review, shots-1h)
  const mesh = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.32, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
  geo.setDrawRange(0, 0);
  mesh.frustumCulled = false;
  scene.add(mesh);
  return {
    /** Sparks at the old spot of a long hop. */
    burst(at: { x: number; z: number }, s: number, nowMs: number) {
      for (let k = 0; k < 80; k++) pts.push({ x: at.x + (Math.random() - 0.5) * 3, y: Math.random() * 5 * s, z: at.z + (Math.random() - 0.5) * 3, t: nowMs });
    },
    /** Three sparks under a flying tree's base. */
    emit(x: number, lift: number, z: number, nowMs: number) {
      for (let k = 0; k < 3; k++) pts.push({ x: x + (Math.random() - 0.5) * 1.2, y: lift + 0.2 + Math.random() * 1.2, z: z + (Math.random() - 0.5) * 1.2, t: nowMs });
    },
    /** Ages the sparks; true while any are alive. */
    step(nowMs: number): boolean {
      while (pts.length > TRAIL_N || (pts.length && nowMs - pts[0].t > TRAIL_MS)) pts.shift();
      const P = geo.attributes.position.array as Float32Array, C = geo.attributes.color.array as Float32Array;
      for (let i = 0; i < TRAIL_N; i++) {
        const p = pts[i], k = i * 3;
        if (p) {
          const a = 1 - (nowMs - p.t) / TRAIL_MS;
          P.set([p.x, p.y - (1 - a) * 0.6, p.z], k);
          C.set([a, 0.82 * a, 0.55 * a], k);
        } else {
          P.fill(0, k, k + 3);
          C.fill(0, k, k + 3);
        }
      }
      geo.attributes.position.needsUpdate = true;
      geo.attributes.color.needsUpdate = true;
      geo.setDrawRange(0, Math.min(pts.length, TRAIL_N));
      return pts.length > 0;
    },
    dispose() {
      scene.remove(mesh);
      geo.dispose();
      (mesh.material as THREE.Material).dispose();
    },
  };
}
export type Trail = ReturnType<typeof createTrail>;

/** Ghost opacity by age (s since the tree left the spot): fades but never goes (mockup `0.42·exp(−age/5400)`, ≥ 0.1). */
export const ghostOpacity = (age: number) => Math.max(0.1, 0.42 * Math.exp(-Math.max(0, age) / 5400));

/** The ghost ground's materials: the shared unit cube and one toon material (white: the instance colours show). */
export interface GhostGroundMats {
  cube: THREE.BufferGeometry;
  ghostGroundMat: THREE.Material;
}

/**
 * Mockup `ghostFor` extras in the voxel look: a bare patch at the spot a tree left, drawn as flat soil blocks on the
 * 0.5 grid (a rough disc of radius 1.3·s, a few cells left out) with a few small stones on it. Axis-aligned toon cubes,
 * seeded by tree and place, so they never change; solid while the crown ghost fades.
 */
export function ghostGround(treeId: string, k: number, s: number, m: GhostGroundMats): THREE.Group {
  const g = new THREE.Group();
  const r = 1.3 * s, cell = 0.5 * s, tile = 0.125 * s, n = Math.ceil(r / cell);
  const rng = new Rng();
  rng.SEED(treeId, "ghost-ground", k);
  const R = () => rng.R();
  const items: { x: number; y: number; z: number; w: number; h: number; c: string }[] = [];
  for (let i = -n; i <= n; i++)
    for (let j = -n; j <= n; j++) {
      const x = i * cell, z = j * cell, keep = R();
      if (Math.hypot(x, z) > r || (keep > 0.8 && (i || j))) continue;
      items.push({ x, y: tile / 2, z, w: cell, h: tile, c: GHOST_SOIL[Math.floor(R() * GHOST_SOIL.length)] });
    }
  const soil = items.length;
  for (let q = 0; q < 5; q++) {
    const t = items[Math.floor(R() * soil)], st = 2 * tile, dx = R() < 0.5 ? -tile : tile, dz = R() < 0.5 ? -tile : tile;
    items.push({ x: t.x + dx, y: tile + st / 2, z: t.z + dz, w: st, h: st, c: GHOST_STONE[Math.floor(R() * GHOST_STONE.length)] });
  }
  const mesh = new THREE.InstancedMesh(m.cube, m.ghostGroundMat, items.length);
  const M = new THREE.Matrix4(), col = new THREE.Color();
  items.forEach((b, i) => {
    mesh.setMatrixAt(i, M.makeScale(b.w, b.h, b.w).setPosition(b.x, b.y, b.z));
    mesh.setColorAt(i, col.set(b.c));
  });
  mesh.receiveShadow = true;
  mesh.userData.sharedGeo = true;
  g.add(mesh);
  return g;
}

/** The ghost of a crown a tree left behind, with its bare patch of soil and stone blocks. */
export interface Ghost {
  group: THREE.Group;
  /** The crown's own material (its opacity fades with age). */
  mat: THREE.MeshBasicMaterial;
}

/** The ghost's group is built about the old spot's base; the caller places it at that spot (and keeps it there). */
export function makeGhost(
  treeId: string, k: number, crown: THREE.Mesh[], mat: THREE.MeshBasicMaterial, s: number,
  m: GhostGroundMats,
): Ghost {
  const group = new THREE.Group();
  group.add(...crown, ghostGround(treeId, k, s, m));
  return { group, mat };
}

/** Frees a ghost's geometry and its crown material (the shared cube and ground material stay). */
export function disposeGhost(g: Ghost) {
  g.group.traverse((o) => {
    if (!(o instanceof THREE.Mesh)) return;
    if (!o.userData.sharedGeo) o.geometry.dispose();
    if (o instanceof THREE.InstancedMesh) o.dispose();
  });
  g.mat.dispose();
  g.group.removeFromParent();
}
