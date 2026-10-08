/**
 * The trees: their builds and voxel meshes. Ported from forest-real.html `build` (the per-tree cache `TC`, freeze 30 min
 * after the last turn, transplants, ghosts, emergence); drawn in the voxel look (voxels.ts, Plan 4).
 * - R17: a tree's traits, species (→ knobs) and normalization reference are frozen at first sight (`FreezeTable`).
 * - Geometry is built off the main thread (scene.ts `createWorkerBuilder`), one job in flight per tree; a result whose
 *   job id is no longer the tree's current job is dropped (`BuildQueue`). Tests and browsers without workers build
 *   directly.
 * - Every tree is built about its base (local origin); its group stands at its place, so a place move (the timeline
 *   reshaped by new data) is a glide of the group, never a rebuild. Ghosts likewise stand at the spot left.
 * - Which trees are built and when is planned in frame.ts (`planTrees`): when a turn lands or every 20 s, requested
 *   early and swapped in exactly when the clock reaches the cut. Cells only append, so a swap shows the same cubes plus
 *   new ones scaling in from their born: no morph.
 * - One instanced mesh of cubes per tree, at a level of detail chosen by the camera's normalised distance (with
 *   hysteresis); a switch cross-fades the two levels over FADE_S. Budgets: ≤ 60 trees built (more when far), and
 *   ≤ 120 000 cubes shared with the ghosts and the ground flora: the farthest trees go coarser first (`fitBudget`).
 */
import * as THREE from "three";
import { type Norm } from "@/lib/forest/geometry";
import { MOCK_MEDIAN_CTX, MOCK_MEDIAN_GEN } from "@/lib/forest/geometry/tree";
import { type BuildJob, type BuildReply, type BuiltBuffers } from "@/lib/forest/geometry/worker";
import { FALLBACK_DECODE_TPS, FALLBACK_MAX_MODEL_LEN, budgets } from "@/lib/forest/normalize";
import { type Knobs, KNOBS, type SpeciesName, pickSpecies } from "@/lib/forest/species";
import { type Place, TREE_SCALE } from "@/lib/forest/timeline";
import type { ForestState, ModelStats, Session, Traits, Tree } from "@/lib/forest/types";
import { CHOSEN, KIND } from "@/lib/forest/voxel";
import type { ClockStep } from "./clock";
import { crownCaps, easeScale, effectiveDistance, farScale } from "./farscale";
import { type Ghost, type Motion, disposeGhost, ghostOpacity, makeFlight, makeGhost, shouldFly, stepMotion, type Trail } from "./flights";
import { BEHIND, MAX_BUILT, type ReqRecord, type TreeInfo, placeIdxAt, planTrees, staleBuild, treeInfo } from "./frame";
import type { Materials, VoxelMaterial } from "./materials";
import { FADE_S, type LevelCells, LOD_MAX, BudgetLevels, cubeColors, girthOf, levelFor, pickTurn, topOf, voxelMesh } from "./voxels";

export type { BuildJob, BuildReply } from "@/lib/forest/geometry/worker";

/** The scene's cube budget, full window (tree cubes, ghost crowns and the ground flora's voxels). */
export const CUBE_CAP = 120_000;
/** Frozen trees' buffers kept after they leave the view (so panning back needs no worker): FAR_BUILT's worth. */
export const FROZEN_CACHE = 200;
/** Once the budget forced coarser levels, it holds them until the total fits this share of the cap (no flicker). */
const BUDGET_RELEASE = 0.92;

// ---------------- pure: freeze, normalization, bookkeeping, budgets ----------------

export interface Frozen {
  traits: Traits;
  species: SpeciesName;
  knobs: Knobs;
  norm: Norm;
}

const walk = (ss: Session[], f: (s: Session) => void) => {
  const go = (s: Session) => (f(s), s.children.forEach(go));
  ss.forEach(go);
};

const median = (v: number[]) => {
  if (!v.length) return NaN;
  v.sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
};

const NO_STATS: ModelStats = { prefill_tps: null, decode_tps: null, max_model_len: null };

/** Spec §5: the installation's median turn (length and foliage budgets) over the window; a neutral reference if empty. */
export function normFor(state: ForestState): Norm {
  const len: number[] = [], fol: number[] = [];
  for (const t of state.trees.values())
    walk(t.sessions, (s) => {
      const m = (s.variant != null && state.models[s.variant]) || NO_STATS;
      for (const turn of s.turns) {
        const b = budgets(turn, m);
        if (b.len > 0) len.push(b.len);
        if (b.foliage > 0) fol.push(b.foliage);
      }
    });
  const l = median(len), f = median(fol);
  return {
    len: l > 0 ? l : MOCK_MEDIAN_GEN / FALLBACK_DECODE_TPS,
    foliage: f > 0 ? f : MOCK_MEDIAN_CTX / FALLBACK_MAX_MODEL_LEN,
    models: { ...state.models },
  };
}

/** R17: traits, species/knobs and the normalization reference, fixed the first time a tree is seen. */
export class FreezeTable {
  private m = new Map<string, Frozen>();
  private normOf = new WeakMap<ForestState, Norm>();

  get(tree: Tree, state: ForestState): Frozen {
    let f = this.m.get(tree.id);
    if (f) return f;
    let norm = this.normOf.get(state);
    if (!norm) this.normOf.set(state, (norm = normFor(state)));
    const variant = tree.sessions.find((s) => s.variant != null)?.variant;
    const maxModelLen = (variant != null && state.models[variant]?.max_model_len) || undefined;
    const traits = { ...tree.traits };
    const species = pickSpecies(traits, { maxModelLen, sessions: tree.sessions });
    f = { traits, species, knobs: KNOBS[species], norm };
    this.m.set(tree.id, f);
    return f;
  }

  /** Forget the trees that left the window. */
  retain(ids: ReadonlySet<string>): void {
    for (const id of [...this.m.keys()]) if (!ids.has(id)) this.m.delete(id);
  }
}

/** Where geometry is built: a worker pool in the browser, a direct call in tests. */
export interface Builder {
  onReply: ((r: BuildReply) => void) | null;
  post(job: BuildJob): void;
  dispose(): void;
}

export interface Built {
  key: string;
  job: BuildJob;
  buffers: BuiltBuffers;
}

/** One job in flight per key (tree id, or `id#k` for a ghost); newer requests coalesce; stale results are dropped. */
export class BuildQueue {
  readonly freeze = new FreezeTable();
  private seq = 0;
  private current = new Map<string, number>();
  private jobs = new Map<number, BuildJob>();
  private pending = new Map<string, Omit<BuildJob, "id">>();
  /** `whenIdle` waiters: the keys each still waits on. */
  private waiters: { keys: Set<string>; done: () => void }[] = [];

  constructor(private builder: Builder, private onBuilt: (b: Built) => void, private onFail?: (key: string) => void) {
    builder.onReply = (r) => this.receive(r);
  }

  /** Builds `tree` at `cut`, about its base (scale `s`). */
  request(key: string, tree: Tree, state: ForestState, cut: number, s: number): void {
    const f = this.freeze.get(tree, state);
    const input = { key, tree, cut, knobs: f.knobs, norm: f.norm, place: { x: 0, z: 0, s }, traits: f.traits };
    if (this.current.has(key)) this.pending.set(key, input);
    else this.send(input);
  }

  private send(input: Omit<BuildJob, "id">) {
    const job = { ...input, id: ++this.seq };
    this.current.set(job.key, job.id);
    this.jobs.set(job.id, job);
    this.builder.post(job);
  }

  receive(r: BuildReply): void {
    const job = this.jobs.get(r.id);
    this.jobs.delete(r.id);
    if (!job || this.current.get(job.key) !== r.id) return; // stale: cancelled or superseded
    this.current.delete(job.key);
    const next = this.pending.get(job.key);
    this.pending.delete(job.key);
    if ("buffers" in r) this.onBuilt({ key: job.key, job, buffers: r.buffers });
    else if (!next) this.onFail?.(job.key);
    if (next) this.send(next);
    this.idle(job.key);
  }

  /** Resolves once every key has been seen idle (nothing in flight or coalesced) after the call. */
  whenIdle(keys: Iterable<string>): Promise<void> {
    const w = { keys: new Set([...keys].filter((k) => this.busy(k))), done: () => {} };
    if (!w.keys.size) return Promise.resolve();
    return new Promise((resolve) => {
      w.done = resolve;
      this.waiters.push(w);
    });
  }

  /** Busy keys now: in flight or with a coalesced request. */
  busyKeys(): Set<string> {
    return new Set([...this.current.keys(), ...this.pending.keys()]);
  }

  private idle(key: string): void {
    if (this.busy(key) || !this.waiters.length) return;
    this.waiters = this.waiters.filter((w) => {
      w.keys.delete(key);
      if (w.keys.size) return true;
      w.done();
      return false;
    });
  }

  /** Forget a key: its in-flight result will be dropped, its pending request discarded. */
  cancel(key: string): void {
    this.current.delete(key);
    this.pending.delete(key);
    this.idle(key);
  }

  busy(key: string): boolean {
    return this.current.has(key) || this.pending.has(key);
  }

  dispose(): void {
    this.current.clear();
    this.pending.clear();
    this.jobs.clear();
    for (const w of this.waiters) w.done();
    this.waiters = [];
    this.builder.dispose();
  }
}

/** Ghost build keys: `${treeId}${GHOST}${k}` (a separator no tree id contains). */
const GHOST = "\u0001";

// ---------------- three.js: the tree layer ----------------

/** What a picked cube grew as. */
export type CellPart = "wood" | "leaf" | "blossom" | "failed";
const PART: Record<number, CellPart> = { [KIND.wood]: "wood", [KIND.leaf]: "leaf", [KIND.blossom]: "blossom", [KIND.failed]: "failed" };

export type TreePickResult =
  | { kind: "turn"; part: CellPart; treeId: string; sessionId: string; turn: number; time: number }
  | { kind: "ghost"; treeId: string; plantedAt: number; movedAt: number };

/**
 * A level switch in progress (review Q2): two meshes with complementary dithers, `a` keeping the pixels whose hash is
 * below `f` and `b` the rest. `f` runs toward 1 (`dir` +1: `a` wins) or back toward 0 (`dir` −1: `b` wins), so a switch
 * reversed half way continues from where it stands, never popping.
 */
interface Fade {
  a: THREE.InstancedMesh;
  b: THREE.InstancedMesh;
  aLevel: number;
  bLevel: number;
  aMat: VoxelMaterial;
  bMat: VoxelMaterial;
  f: number;
  dir: 1 | -1;
  /** Wall time of the last step (ms). */
  last: number;
  /** A third level asked for during the fade: started when it ends. */
  pending?: number;
}

interface Entry extends Motion {
  id: string;
  group: THREE.Group;
  /** The cubes on screen (at `level`), and a cross-fade from the previous level. */
  mesh: THREE.InstancedMesh | null;
  fade: Fade | null;
  /** The level drawn, and the level the distance alone asks for (with hysteresis). */
  level: number;
  natural?: number;
  /** Normalised camera distance at the last view. */
  dEff?: number;
  /** The build on screen, and results waiting for their cut (swapped in exactly when the clock reaches it). */
  buffers: BuiltBuffers | null;
  /** Each with the turn times of the data it was built from. */
  waiting: (BuiltBuffers & { from: readonly number[] })[];
  /** Cut of the latest request. */
  req?: ReqRecord;
  place: Place | null;
  shownIdx?: number;
  /** May grow out of the ground when first shown (born while time ran forward, never emerged before). */
  emergeOk: boolean;
  /** The far scale last reported to the camera (setView), and a change not reported yet. */
  farShown?: number;
  farDue?: boolean;
  ghosts: Map<number, Ghost & { buffers: BuiltBuffers; level: number }>;
}

export interface TreeLayerOpts {
  shadows: boolean;
  builder: Builder;
  /** Called when a result changed what is on screen (redraw, shadows, camera inputs). */
  onChange: () => void;
  /** Budgets (spec §6.5): full window 60 trees, the card (compact) 12 for 1h / 6h and 30 for 24h / 7d (read at every
   * tick: the range can change). */
  maxTrees?: number | (() => number);
  /** How far behind the view a tree may stand and still be built (default BEHIND; the scene widens it to the range). */
  behind?: () => number;
  /** The cubes the trees and ghosts may use (read every view: the ground flora's voxels are taken off it). */
  cubeCap?: () => number;
  /** The ground's height at a world position (land.ts terraces; pure in the position). Flat 0 by default. */
  baseAt?: (x: number, z: number) => number;
}

type P3 = { x: number; y: number; z: number };

/**
 * A tree first built at `rel` grows out of the ground only if it was born while time ran forward after the first data
 * (`primedAt`), within the last 30 min, and its wood was not already on screen under another tree id (`inherited`: a
 * regroup or a window slide renamed it; final review I1).
 */
export function canEmerge(rel: number, primedAt: number, continuous: boolean, info: Pick<TreeInfo, "start" | "inherited">): boolean {
  return !info.inherited && rel > primedAt && continuous && info.start > primedAt && rel - info.start < 1800;
}

export function createTreeLayer(scene: THREE.Scene, mats: Materials, trail: Trail, opts: TreeLayerOpts) {
  const root = new THREE.Group();
  scene.add(root);
  const entries = new Map<string, Entry>();
  const frozenCache = new Map<string, BuiltBuffers>();
  /** The ground's height under a place: a tree's base stands there (pure in the position, so a place that stays put
   * never changes height; a place the timeline moves takes the ground under its new spot: final review I1). */
  const groundAt = (x: number, z: number): number => {
    const v = opts.baseAt?.(x, z) ?? 0;
    return Number.isFinite(v) ? v : 0;
  };
  /** Trees that have been shown once: they never grow out of the ground again. */
  const emerged = new Set<string>();
  /** Every tree's last built crown radius (kept while it is laid out), and the caps computed over them. */
  const radii = new Map<string, number>();
  let caps = new Map<string, number>(), layoutVersion = 0, capsVersion = -1, lastFarSync = -Infinity;
  let infos = new Map<string, TreeInfo>();
  let state: ForestState | null = null;
  let primedAt: number | null = null, jumpAt = -Infinity, lastRel = -Infinity, hi: string | null = null;
  /** The budget's held level choice (per-tree hysteresis). */
  const budget = new BudgetLevels(BUDGET_RELEASE);
  /** The last view (`setView`): a tree's first build is drawn at its level for it at once. */
  let view: { cam: P3; viewH: number } | null = null;
  let markers: THREE.InstancedMesh | null = null;
  /** The highlight markers: small voxel cubes (0.25 u) in the tint's cyan. */
  const markerGeo = mats.cube.clone().scale(0.25, 0.25, 0.25);
  /** Keys whose build threw: not retried until new data arrives. */
  const failed = new Set<string>();
  const queue = new BuildQueue(opts.builder, (b) => onBuilt(b), (key) => failed.add(key));
  /** A build's drawn cells at level L (built in the worker: levels.ts). */
  const cellsOf = (_id: string, b: BuiltBuffers, L: number): LevelCells => b.levels[L];

  const disposeMesh = (m: THREE.InstancedMesh | null) => {
    if (!m) return;
    m.removeFromParent();
    m.geometry.dispose();
    m.dispose();
  };

  /** Level `L` of a build as one mesh of cubes (a ghost: its own material, no shadows). */
  const cubes = (id: string, b: BuiltBuffers, L: number, mat: THREE.Material, userData: object, ghost = false) => {
    const lc = cellsOf(id, b, L), m = voxelMesh(lc, cubeColors(lc, id), mat, mats.cube);
    m.castShadow = m.receiveShadow = opts.shadows && !ghost;
    if (!ghost) m.customDepthMaterial = mats.voxelDepth;
    m.userData = { ...userData, treeId: id, lc };
    return m;
  };

  /** Ends a cross-fade: the winning mesh rests on the shared material, the other goes. Then a pending level starts. */
  const endFade = (e: Entry, nowMs = 0) => {
    const F = e.fade;
    if (!F) return;
    const [keep, lose] = F.dir > 0 ? [F.a, F.b] : [F.b, F.a];
    disposeMesh(lose);
    keep.material = mats.voxel;
    e.mesh = keep;
    F.aMat.dispose();
    F.bMat.dispose();
    e.fade = null;
    if (F.pending !== undefined && F.pending !== e.level) setLevel(e, F.pending, nowMs);
  };

  /** Draws the build on screen at the entry's level (a new build: the same cubes plus new ones scaling in). A fade in
   * progress keeps running on the new build's cells. */
  const show = (e: Entry) => {
    const F = e.fade;
    if (F && e.buffers) {
      const a = cubes(e.id, e.buffers, F.aLevel, F.aMat, { kind: "tree" }), b = cubes(e.id, e.buffers, F.bLevel, F.bMat, { kind: "tree" });
      disposeMesh(F.a);
      disposeMesh(F.b);
      [F.a, F.b] = [a, b];
      e.group.add(a, b);
      e.mesh = F.dir > 0 ? a : b;
      return;
    }
    disposeMesh(e.mesh);
    e.mesh = e.buffers ? cubes(e.id, e.buffers, e.level, mats.voxel, { kind: "tree" }) : null;
    if (e.mesh) e.group.add(e.mesh);
  };

  /** Switches the entry to level `L` with a cross-fade; reversing a fade in progress runs it back from where it is. */
  function setLevel(e: Entry, L: number, nowMs: number) {
    const F = e.fade;
    if (F) {
      if (L === (F.dir > 0 ? F.aLevel : F.bLevel)) return void (F.pending = undefined);
      if (L === (F.dir > 0 ? F.bLevel : F.aLevel)) {
        F.dir = F.dir > 0 ? -1 : 1;
        F.pending = undefined;
        e.level = L;
        e.mesh = F.dir > 0 ? F.a : F.b;
        for (const [k, g] of e.ghosts) ghostCrown(e, k, g);
        return;
      }
      F.pending = L; // a third level: after this fade
      return;
    }
    if (L === e.level) return;
    const from = e.level;
    e.level = L;
    if (!e.buffers || !e.mesh) return;
    const aMat = mats.fading(true), bMat = mats.fading(false);
    const b = e.mesh;
    b.material = bMat;
    const a = cubes(e.id, e.buffers, L, aMat, { kind: "tree" });
    e.group.add(a);
    e.mesh = a;
    e.fade = { a, b, aLevel: L, bLevel: from, aMat, bMat, f: 0, dir: 1, last: nowMs };
    for (const [k, g] of e.ghosts) ghostCrown(e, k, g); // ghosts follow without a fade (faint, and seldom)
  }

  const ghostCrown = (e: Entry, k: number, g: Ghost & { buffers: BuiltBuffers; level: number }) => {
    for (const c of [...g.group.children]) if (c.userData.kind === "ghost") disposeMesh(c as THREE.InstancedMesh);
    g.level = e.level;
    g.group.add(cubes(e.id, g.buffers, e.level, g.mat, { kind: "ghost", k }, true));
  };

  function onBuilt({ key, job, buffers }: Built) {
    const [id, ks] = key.split(GHOST);
    const e = entries.get(id), info = infos.get(id);
    if (!e || !info) return;
    if (ks !== undefined) return void addGhost(e, info, Number(ks), buffers);
    if (job.cut >= info.freezeAt) {
      frozenCache.delete(id);
      frozenCache.set(id, buffers);
      while (frozenCache.size > FROZEN_CACHE) frozenCache.delete(frozenCache.keys().next().value as string);
    }
    // a build made from data that lacked a turn at or before its cut is stale: a newer request includes the turn
    const from = treeInfo(job.tree, []).times;
    if (e.buffers && staleBuild(from, info.times, job.cut)) return;
    e.waiting.push({ ...buffers, from });
    e.waiting.sort((a, b) => a.cut - b.cut);
    if (!e.buffers) swap(e, info, lastRel); // the first build shows at once
  }

  /** Shows the newest waiting build whose cut the clock has reached. */
  function swap(e: Entry, info: TreeInfo, rel: number) {
    if (e.buffers) e.waiting = e.waiting.filter((w) => !staleBuild(w.from, info.times, w.cut)); // stale for the data now
    let b: BuiltBuffers | undefined;
    while (e.waiting.length && (e.waiting[0].cut <= rel || !e.buffers && !b)) b = e.waiting.shift();
    if (!b) return;
    const now = performance.now(), prevCut = e.buffers?.cut ?? -Infinity;
    const idx = placeIdxAt(info.places, b.cut), pl = info.places[idx];
    if (e.place && e.shownIdx !== undefined && idx !== e.shownIdx) {
      if (shouldFly(e.shownIdx, idx, { from: prevCut, to: b.cut, continuous: prevCut >= jumpAt })) {
        e.flight = makeFlight(e.place, pl, TREE_SCALE, now, (e.base ?? 0) - groundAt(pl.x, pl.z));
        e.glide = { x: 0, z: 0 };
        if (e.flight.far) trail.burst(e.place, TREE_SCALE, now);
      } else e.flight = undefined;
    }
    e.shownIdx = idx;
    e.place = pl;
    e.base = groundAt(pl.x, pl.z);
    if (!e.buffers) {
      if (e.emergeOk && !emerged.has(e.id)) e.emerge = { t0: now };
      emerged.add(e.id);
      if (view) e.level = e.natural = levelFor(effectiveDistance(Math.hypot(view.cam.x - pl.x, view.cam.y, view.cam.z - pl.z), view.viewH));
    }
    e.buffers = b;
    show(e);
    if (hi) placeMarkers();
    opts.onChange();
  }

  function addGhost(e: Entry, info: TreeInfo, k: number, buffers: BuiltBuffers) {
    if (e.ghosts.has(k) || !info.places[k]) return;
    const mat = mats.ghost(), old = info.places[k - 1];
    const g = { ...makeGhost(e.id, k, [], mat, TREE_SCALE, mats), buffers, level: e.level };
    ghostCrown(e, k, g);
    g.group.position.set(old.x, groundAt(old.x, old.z), old.z);
    root.add(g.group);
    e.ghosts.set(k, g);
    opts.onChange();
  }

  const drop = (e: Entry) => {
    queue.cancel(e.id);
    if (e.fade) (disposeMesh(e.fade.a), disposeMesh(e.fade.b), e.fade.aMat.dispose(), e.fade.bMat.dispose(), (e.fade = null));
    disposeMesh(e.mesh);
    e.mesh = null;
    for (const [k, g] of e.ghosts) (queue.cancel(`${e.id}${GHOST}${k}`), disposeGhost(g));
    e.group.removeFromParent();
    entries.delete(e.id);
  };

  /** A tree's node in world coordinates (trees are built about their base). */
  const world = (e: Entry, p: readonly number[]): P3 => ({ x: p[0] + (e.place?.x ?? 0), y: p[1] + (e.base ?? 0), z: p[2] + (e.place?.z ?? 0) });

  /** Every session's newest joint (camera `tips`) and the newest growth overall. */
  function tips(): { tips: Map<string, P3>; newest: P3 | null } {
    const out = new Map<string, P3 & { t: number }>();
    let best: (P3 & { t: number }) | null = null;
    for (const e of entries.values())
      for (const n of e.buffers?.nodes ?? []) {
        const o = out.get(n.sessionId);
        const p = { ...world(e, n.p), t: n.time };
        if (!o || n.time >= o.t) out.set(n.sessionId, p);
        if (!best || n.time > best.t) best = p;
      }
    return { tips: out, newest: best };
  }

  function placeMarkers() {
    if (markers) (markers.removeFromParent(), markers.dispose(), (markers = null));
    if (!hi) return;
    const pts: P3[] = [];
    for (const e of entries.values()) for (const n of e.buffers?.nodes ?? []) if (n.sessionId === hi) pts.push(world(e, n.p));
    if (!pts.length) return;
    markers = new THREE.InstancedMesh(markerGeo, mats.tip, pts.length);
    const m = new THREE.Matrix4();
    pts.forEach((p, i) => markers!.setMatrixAt(i, m.makeTranslation(p.x, p.y, p.z)));
    markers.computeBoundingSphere();
    root.add(markers);
  }

  /** Cubes drawn now: every tree's level (and a fading one), every ghost crown. */
  const cubeCount = () => {
    let n = 0;
    for (const e of entries.values()) {
      n += e.fade ? e.fade.a.count + e.fade.b.count : (e.mesh?.count ?? 0);
      for (const g of e.ghosts.values()) n += cellsOf(e.id, g.buffers, g.level).shown.length;
    }
    return n;
  };

  return {
    /** New data (scene time): the timeline's places and every tree's turn times. Nothing is rebuilt here. */
    setData(s: ForestState, info: Map<string, TreeInfo>) {
      state = s;
      infos = info;
      layoutVersion++;
      for (const id of [...radii.keys()]) if (!info.has(id)) radii.delete(id);
      failed.clear();
      for (const id of [...frozenCache.keys()]) if (!infos.has(id)) frozenCache.delete(id);
      for (const id of [...emerged]) if (!infos.has(id)) emerged.delete(id);
      queue.freeze.retain(new Set(s.trees.keys()));
    },

    /** Picks, places and builds the trees for history time `rel`; `step` is the clock's merged step, `viewX` the view. */
    tick(rel: number, step: ClockStep, viewX: number) {
      if (!state) return;
      if (!step.continuous || rel < lastRel) jumpAt = rel;
      lastRel = rel;
      primedAt ??= rel;
      const req = new Map<string, ReqRecord>();
      for (const e of entries.values()) if (e.req) req.set(e.id, e.req);
      const max = typeof opts.maxTrees === "function" ? opts.maxTrees() : (opts.maxTrees ?? MAX_BUILT);
      const plan = planTrees(infos, req, rel, viewX, max, opts.behind?.() ?? BEHIND);
      for (const e of [...entries.values()]) if (!plan.keep.has(e.id)) drop(e);
      for (const id of plan.keep) {
        const info = infos.get(id)!;
        let e = entries.get(id);
        if (!e) {
          const emergeOk = canEmerge(rel, primedAt, step.continuous, info);
          e = {
            id, group: new THREE.Group(), mesh: null, fade: null, level: LOD_MAX, buffers: null, waiting: [], place: null, emergeOk,
            glide: { x: 0, z: 0 }, ghosts: new Map(), ground: groundAt,
          };
          root.add(e.group);
          entries.set(id, e);
        }
        // the timeline moved this tree's spot (new data): glide there, no rebuild
        if (e.place && e.shownIdx !== undefined) {
          const p = info.places[Math.min(e.shownIdx, info.places.length - 1)];
          if (p.x !== e.place.x || p.z !== e.place.z) {
            e.glide = { x: e.glide.x + e.place.x - p.x, z: e.glide.z + e.place.z - p.z };
            e.place = p;
            e.base = groundAt(p.x, p.z); // where the glide ends; on the way it follows the ground (stepMotion)
          }
        }
        // every spot the tree has left: a fading ghost of the crown as it stood on a patch of soil and stone blocks
        for (let k = 1; k <= (e.shownIdx ?? 0); k++) {
          const g = e.ghosts.get(k), key = `${id}${GHOST}${k}`, old = info.places[k - 1];
          if (g) {
            g.mat.opacity = ghostOpacity(rel - info.places[k].from);
            g.group.position.set(old.x, groundAt(old.x, old.z), old.z);
          } else if (!queue.busy(key) && !failed.has(key)) queue.request(key, info.tree, state, info.places[k].from - 1, TREE_SCALE);
        }
      }
      for (const b of plan.builds) {
        const e = entries.get(b.id)!, info = infos.get(b.id)!;
        if (failed.has(b.id)) continue;
        e.req = { cut: b.cut, times: info.times };
        const c = frozenCache.get(b.id);
        if (c && c.cut === b.cut) onBuilt({ key: b.id, job: { id: 0, key: b.id, tree: info.tree, cut: c.cut, place: { x: 0, z: 0, s: 1 }, ...queue.freeze.get(info.tree, state) }, buffers: c });
        else queue.request(b.id, info.tree, state, b.cut, TREE_SCALE);
      }
    },

    /** Per frame: swaps in builds whose cut the clock reached, then flights, emergence, glides and level cross-fades.
     * True if anything moved. */
    animate(nowMs: number, dt: number, rel: number): boolean {
      let moved = false;
      for (const e of entries.values()) {
        const info = infos.get(e.id);
        if (info && e.waiting.length && e.waiting[0].cut <= rel) swap(e, info, rel);
        if (stepMotion(e, e.group, nowMs, dt, trail)) moved = true;
        // a moved tree's ghosts are drawn at its scale too (re-review 3, N6)
        for (const g of e.ghosts.values()) g.group.scale.setScalar(e.far ?? 1);
        if (e.buffers && radii.get(e.id) !== e.buffers.crown.r) (radii.set(e.id, e.buffers.crown.r), layoutVersion++);
        const F = e.fade;
        if (F) {
          moved = true;
          F.f = Math.min(1, Math.max(0, F.f + (F.dir * Math.max(0, nowMs - F.last)) / (FADE_S * 1000)));
          F.last = nowMs;
          if ((F.dir > 0 && F.f >= 1) || (F.dir < 0 && F.f <= 0)) endFade(e, nowMs);
          else F.aMat.fade.uFade.value = F.bMat.fade.uFade.value = F.f;
        }
      }
      return moved;
    },

    /**
     * "Bigger when far" (farscale.ts) and the level of detail, for a camera at `cam` over a canvas `viewH` px high:
     * - each built tree's render scale, capped so no scaled crown overlaps a neighbour's; applied by the next `animate`
     *   (about the tree's base); true when any scale moved by more than 2 % since the last time it reported (the
     *   camera's crown checks then refresh);
     * - each tree's level from its normalised distance (hysteresis), coarsened from the farthest tree in while the
     *   cubes exceed the budget; a change cross-fades (`nowMs`: wall time).
     */
    setView(cam: P3, viewH: number, dt = 0, nowMs = 0): boolean {
      view = { cam: { x: cam.x, y: cam.y, z: cam.z }, viewH };
      // the caps, over every laid-out tree with a known crown (not only the built ones, so building or dropping a
      // neighbour does not change them), recomputed only when the layout or a crown changed (re-review 3, N4)
      if (capsVersion !== layoutVersion) {
        capsVersion = layoutVersion;
        const all: { id: string; x: number; z: number; r: number }[] = [];
        for (const [id, info] of infos) {
          const r = radii.get(id), p = info.places[info.places.length - 1];
          if (r !== undefined && p) all.push({ id, x: p.x, z: p.z, r });
        }
        caps = crownCaps(all);
      }
      let moved = false;
      const lod: { id: string; d: number; level: number; count: (L: number) => number }[] = [];
      for (const e of entries.values()) {
        if (!e.buffers || !e.place) continue;
        const d = Math.hypot(cam.x - e.place.x, cam.y, cam.z - e.place.z), dEff = effectiveDistance(d, viewH);
        const target = Math.min(farScale(dEff), caps.get(e.id) ?? 1);
        // a far zoom moves the target smoothly by itself; only a cap change (a neighbour's crown) is eased
        const f = e.far === undefined || dt <= 0 ? target : easeScale(e.far, target, dt);
        e.far = f;
        if (Math.abs(f - (e.farShown ?? 1)) > 0.02 * (e.farShown ?? 1)) e.farDue = true;
        e.dEff = dEff;
        e.natural = levelFor(dEff, e.natural);
        const b = e.buffers, id = e.id;
        lod.push({ id, d: dEff, level: e.natural, count: (L) => cellsOf(id, b, L).shown.length });
      }
      // the cube budget: what the flora leaves, less the ghost crowns
      let ghosts = 0;
      for (const e of entries.values()) for (const g of e.ghosts.values()) ghosts += cellsOf(e.id, g.buffers, g.level).shown.length;
      const cap = Math.max(0, (opts.cubeCap?.() ?? CUBE_CAP) - ghosts);
      const lv = budget.levels(lod, cap, nowMs);
      for (const t of lod) setLevel(entries.get(t.id)!, lv.get(t.id)!, nowMs);
      // the camera's crown checks follow at most every 250 ms
      if (nowMs - lastFarSync >= 250) {
        for (const e of entries.values()) if (e.farDue) (moved = true), (e.farDue = false), (e.farShown = e.far);
        if (moved) lastFarSync = nowMs;
      }
      return moved;
    },

    /** A built tree's true crown radius (tests). */
    crownR: (treeId: string): number | null => entries.get(treeId)?.buffers?.crown.r ?? null,

    /** A tree's render scale now (1 near; farscale.ts). */
    scaleOf: (treeId: string): number => entries.get(treeId)?.far ?? 1,

    /** A tree's level of detail drawn now (null: not built). */
    levelOf: (treeId: string): number | null => (entries.get(treeId)?.buffers ? entries.get(treeId)!.level : null),

    /** Camera inputs from the built trees: crowns at their places, session tips, the newest growth, framing. */
    cameraInputs(rel: number) {
      const crowns = new Map<string, { x: number; z: number; r: number; top: number }>();
      const frame: { x: number; active: boolean; points: P3[] }[] = [];
      for (const e of entries.values()) {
        if (!e.buffers || !e.place) continue;
        // crown checks see the scaled crown (1 near the trees); the framing points stay true to size (no feedback
        // between the fit's distance and the scale)
        const k = e.far ?? 1;
        crowns.set(e.id, { x: e.place.x, z: e.place.z, ...e.buffers.crown, r: e.buffers.crown.r * k, top: e.buffers.crown.top * k + (e.base ?? 0) });
        const P = e.buffers.points, points: P3[] = [];
        for (let i = 0; i < P.length; i += 3) points.push(world(e, [P[i], P[i + 1], P[i + 2]]));
        frame.push({ x: e.place.x, active: e.buffers.nodes.some((n) => n.time >= rel - 420), points });
      }
      return { crowns, frame, ...tips() };
    },

    highlight(sessionId: string | null) {
      hi = sessionId;
      placeMarkers();
    },

    /** The cube under the ray, mapped back to the turn that grew it (its born), or the ghost it belongs to. */
    pick(ray: THREE.Raycaster): { result: TreePickResult; distance: number } | null {
      const hit = ray.intersectObject(root, true).find((h) => h.object.userData.kind && h.instanceId !== undefined);
      if (!hit) return null;
      const ud = hit.object.userData as { kind: string; treeId: string; k?: number; lc: LevelCells };
      const info = infos.get(ud.treeId), distance = hit.distance;
      if (ud.kind === "ghost" && info && ud.k)
        return { result: { kind: "ghost", treeId: ud.treeId, plantedAt: info.places[ud.k - 1].from, movedAt: info.places[ud.k].from }, distance };
      const e = entries.get(ud.treeId);
      if (!e?.buffers) return null;
      const cell = ud.lc.shown[hit.instanceId!];
      // back into the tree's own (unscaled) frame: a far tree is drawn larger about its base
      const p = hit.point.clone().sub(e.group.position).divideScalar(e.group.scale.x || 1);
      const n = pickTurn(e.buffers.nodes, ud.lc.born[cell], p);
      return n && { result: { kind: "turn", part: PART[ud.lc.kind[cell]] ?? "leaf", treeId: e.id, sessionId: n.sessionId, turn: n.turn, time: n.time }, distance };
    },

    /** The trunk base girth on screen at `now`: the wood cells' cross-section, each at its scale-in (voxels.ts). */
    girthAt(treeId: string, now: number) {
      const b = entries.get(treeId)?.buffers;
      return b ? girthOf(b.levels[0], now, CHOSEN.size) : null;
    },
    /** The top of the fully grown cubes on screen at `now` (debug: height continuity across swaps). */
    heightAt(treeId: string, now: number) {
      const b = entries.get(treeId)?.buffers;
      return b ? topOf(b.levels[0], now, CHOSEN.size) : null;
    },
    /** The cubes the built trees would draw at each level (debug: what the levels cost). */
    levelCubes(): number[] {
      const out = [0, 0, 0];
      for (const e of entries.values()) if (e.buffers) for (let L = 0; L <= LOD_MAX; L++) out[L] += cellsOf(e.id, e.buffers, L).shown.length;
      return out;
    },
    /** Fine cells of a tree's build on screen (debug: growth). */
    cellsAt: (treeId: string): number | null => entries.get(treeId)?.buffers?.cells ?? null,
    /**
     * Resolves when every build in flight now has replied (or was cancelled or failed). That covers the trees present
     * now: a tree is requested in the same `tick` that adds it (or shown at once from the frozen cache), so a tree
     * without a build is always busy, unless its build failed. A result waiting for a later cut is not forced.
     */
    settle(): Promise<void> {
      return queue.whenIdle(queue.busyKeys());
    },
    flights: () => [...entries.values()].filter((e) => e.flight).map((e) => ({ treeId: e.id, dx: e.flight!.dx })),

    stats() {
      let trees = 0;
      const levelsDrawn = [0, 0, 0];
      for (const e of entries.values()) if (e.buffers) (trees++, levelsDrawn[e.level]++);
      return { trees, cubes: cubeCount(), levels: levelsDrawn };
    },

    dispose() {
      for (const e of [...entries.values()]) drop(e);
      if (markers) markers.dispose();
      markerGeo.dispose();
      queue.dispose();
      root.removeFromParent();
    },
  };
}

export type TreeLayer = ReturnType<typeof createTreeLayer>;
