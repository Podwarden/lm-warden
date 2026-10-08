/**
 * Ports the mockup's tree assembly (forest-real.html): `habit` (l.132), `ramify` (l.145) and `buildTree` (l.484) —
 * trunk height with `RAMP`, limbs on the axis with their lean fixed at birth, and the voxel trunk (the pipe-model taper
 * with bulge and root flare, as girth schedules). Also maps wire turns to the mockup's turn shape, with normalization
 * replacing raw tokens (spec §5). The old look's bark tubes, buttress and aerial roots and leaf instances are gone
 * (Plan 4): the build yields the voxelizer's inputs, the turn nodes and the camera points.
 */
import { budgets, compress } from "../normalize";
import type { Knobs } from "../species";
import type { ModelStats, Session, Traits, Tree, Turn } from "../types";
import { type Build, type GSes, type GTurn, type Hab, type Node, RAMP, SH, growLimb, pipeR } from "./limb";
import { Rng, fnv } from "./random";
import { V3 } from "./tube";
import { type Ev, type VoxRec, type WoodSample, newVoxRec, schedule } from "./voxin";

export interface Norm {
  /** Reference (median) length budget, seconds of full-server decode. */
  len: number;
  /** Reference (median) foliage budget, share of the context window. */
  foliage: number;
  models: Record<string, ModelStats>;
}

/**
 * The mockup's median turn (README §5: p50 prompt 37k, p50 output 134 tokens). A turn whose budget equals the
 * installation's reference (`compress(v, ref) = 1`) gets exactly the mockup's size for these raw tokens.
 */
export const MOCK_MEDIAN_GEN = 134;
export const MOCK_MEDIAN_CTX = 37000;
export const GOLD = Math.PI * (3 - Math.sqrt(5));

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
/** `compress` that never yields NaN: no reference → the median (1); NaN budgets (0/0) → 0. */
const squash = (v: number, ref: number) => {
  if (!(ref > 0) || !Number.isFinite(ref)) return 1;
  const c = compress(v, ref);
  return Number.isNaN(c) ? 0 : Math.max(0, c);
};

/**
 * Wire turn → mockup turn. Segment length: `gen = MOCK_MEDIAN_GEN · compress(budgets.len, norm.len)`, so the mockup's
 * `min(1.1, 0.30 + √gen/60)` gives 0.493 for a median turn. Foliage: `ctx = MOCK_MEDIAN_CTX · compress(budgets.foliage,
 * norm.foliage)`, so `7 + min(13, 2.6·log2(1 + ctx/2000))` gives 18 leaves for a median turn.
 * Tools: one leaf per tool call the turn made (`toolsOut`, by family); a leaf's result size and failure come from the
 * same-index entry of the results this turn carried (`toolsIn`), 0 / not failed when there is none.
 */
export function adaptTurn(turn: Turn, i: number, m: ModelStats, norm: Norm): GTurn {
  const [time, , , , toolsIn, toolsOut] = turn;
  // like to_forest.py:38, leaves come from this request alone: size/fail describe the PREVIOUS call's result (Q1 → a)
  const b = budgets(turn, m);
  return {
    i,
    // float32 once, here: the cut filter (`time <= cut`) and the recorded float32 `born` then agree, so two turns that
    // share a float32 time are always in a build together (voxel prefix stability)
    time: Math.fround(time),
    gen: MOCK_MEDIAN_GEN * squash(b.len, norm.len),
    ctx: MOCK_MEDIAN_CTX * squash(b.foliage, norm.foliage),
    tools: toolsOut.map((fam, j) => ({ fam, size: Math.max(0, toolsIn[j]?.[1] ?? 0), fail: toolsIn[j]?.[2] === true })),
  };
}

const NO_STATS: ModelStats = { prefill_tps: null, decode_tps: null, max_model_len: null };

function adaptSession(s: Session, norm: Norm): GSes {
  const m = (s.variant != null && norm.models[s.variant]) || NO_STATS;
  return {
    id: s.id,
    src: s.id,
    turns: s.turns.map((t, i) => adaptTurn(t, i, m, norm)),
    children: s.children.map((c) => adaptSession(c, norm)),
    at: s.at ?? 0,
    shoot: false,
  };
}

/**
 * Ports `ramify`: turns are cut into shoots of 5; shoot k grows from shoot ⌊(k−1)/2⌋ at turn
 * min(len−1, 1 + 2·((k−1) mod 2)), a binary crown. Subagents move to the shoot holding their fork turn.
 *
 * Deviation (prefix stability, controller ruling): the mockup only ramified sessions of more than 6 turns, so the 7th
 * turn re-cut the whole session and moved turn 6's wood onto a new shoot. Here EVERY session (subagents too) is cut by
 * turn index, so a new turn only appends to the last shoot or opens a new one; turns already grown never move.
 * Sessions of ≤ 5 turns are one limb exactly as in the mockup; a 6-turn session now forks its 6th turn off turn 1.
 * Builds new objects (the mockup mutated `c.at` in place, and its shoot 0 lost a ramified subagent's `at`).
 */
export function ramify(ses: GSes): GSes {
  const ch: GTurn[][] = [];
  ses.turns.forEach((t, i) => (ch[Math.floor(i / SH)] ??= []).push(t));
  if (!ch.length) return { ...ses, children: ses.children.map(ramify) };
  const shoots: GSes[] = ch.map((turns, k) => ({ id: ses.id + (k ? "/" + k : ""), src: ses.src, turns, children: [], at: 0, shoot: k > 0 }));
  for (let k = 1; k < shoots.length; k++) {
    const p = shoots[Math.floor((k - 1) / 2)];
    const sib = (k - 1) % 2;
    // the parent shoot index is < k, so it is already full (5 turns): `at` never changes as turns arrive
    shoots[k].at = Math.min(p.turns.length - 1, 1 + sib * 2);
    p.children.push(shoots[k]);
  }
  for (const c of ses.children) {
    const k = Math.min(shoots.length - 1, Math.floor(c.at / SH));
    shoots[k].children.push({ ...ramify(c), at: c.at % SH });
  }
  return { ...shoots[0], at: ses.at, shoot: ses.shoot };
}

/**
 * Ports `habit(tree)` from traits (same definitions as the mockup: mean first prompt, ctx/gen, mean gen), then lays
 * the species knobs over it (`KNOBS.round = {}` is the pure habit). The caller passes traits frozen at first sight,
 * so the habit never drifts while a tree grows live. Every knob key is typed through `Hab`; no casts.
 */
export function habit(traits: Traits, knobs: Knobs): Hab {
  const { sys0, ctx_gen, mean_gen } = traits;
  const g = Math.log10(ctx_gen);
  const { girth, bulge, tall, el, up, shrub, ...rest } = knobs;
  return {
    ...rest,
    girth: girth ?? clamp(0.75 + 0.55 * Math.log10(sys0 / 8000), 0.55, 1.5),
    bulge: bulge ?? clamp((sys0 - 45000) / 45000, 0, 1),
    tall: tall ?? clamp(1.9 - 0.5 * g, 0.6, 1.8),
    el: el ?? clamp(1.25 - 0.35 * g, 0.32, 1.15),
    up: up ?? (g < 1.1 ? 0.55 : 0),
    shrub: shrub ?? (sys0 < 5000 && mean_gen < 260),
  };
}

export interface BuiltTree {
  nodes: Node[];
  /** Mockup `P`: the trunk base and top plus every node, for the camera. */
  P: V3[];
  /** The voxelizer's inputs. */
  vox: VoxRec;
}

/** The voxel trunk's centreline sample spacing (world units): fixed heights, so a taller trunk only adds samples. */
export const TRUNK_STEP = 0.25;

interface Limb {
  ses: GSes;
  dir: V3;
  sy: number;
  seed: number;
  /** Crown species (palm): the limb's root as a share of the trunk height (`sy = trunkH · u`). */
  u: number | null;
  /** Voxel trunk tiers: (time, height) the voxels root this limb's turns at (see `VoxRec.shiftAt`). */
  roots: [t: number, y: number][];
  /** The limb's base load events (from `growLimb`). */
  ev: Ev[];
}

/**
 * Ports `buildTree(tree, ti, cut)` with the mockup's `b = { x, z, s }` given as `place`. `traits` drive the habit
 * (pass them frozen at first sight).
 */
export function buildTree(
  tree: Tree, cut: number, knobs: Knobs, norm: Norm, place: { x: number; z: number; s: number }, traits: Traits = tree.traits,
): BuiltTree {
  const HB = habit(traits, knobs);
  const sessions = tree.sessions.map((s) => ramify(adaptSession(s, norm)));
  const b = place;
  // the mockup's tree index `ti` only phases the axis wobble and the limb azimuths: a stable per-tree integer here
  const ti = fnv(tree.id) % 1000;
  const B: Build = {
    rng: new Rng(),
    HAB: HB,
    NODES: [],
    VOX: newVoxRec(),
    LAST_EV: [],
  };
  const R = () => B.rng.R();
  const hAt = (time: number) => {
    let hw = 0;
    for (const s2 of sessions) for (const t2 of s2.turns) if (t2.time <= time) hw += RAMP(time, t2.time);
    return HB.shrub || HB.ground ? (0.22 + 0.025 * Math.sqrt(hw)) * b.s : (0.25 + 0.36 * Math.sqrt(hw)) * b.s * HB.tall;
  };
  const trunkH = hAt(cut);
  const hCache = new Map<number, number>();
  const hAtC = (t: number) => {
    let h = hCache.get(t);
    if (h === undefined) hCache.set(t, (h = hAt(t)));
    return h;
  };
  // The trunk is the axis every limb leaves from; each limb starts ON the axis, inside the wood.
  const TW = HB.twist ?? 1;
  const axis = (y: number) =>
    new V3(b.x + 0.1 * TW * b.s * Math.sin(y * 0.9 * Math.sqrt(TW) + ti), y, b.z + 0.08 * TW * b.s * Math.cos(y * 0.7 * Math.sqrt(TW) + ti * 2));
  const limbs: Limb[] = [];
  sessions.forEach((ses, si) => {
    if (!ses.turns.length || ses.turns[0].time > cut) return;
    B.rng.SEED(ses.id, "place");
    const az = si * GOLD + ti, el = (HB.shrub ? 0.95 : HB.el) + (si % 4) * (HB.elSpread ?? 0.1);
    const dir = new V3(Math.cos(az) * Math.cos(el), Math.sin(el), Math.sin(az) * Math.cos(el));
    let sy: number, u: number | null = null;
    if (HB.ground) sy = 0.02;
    else if (HB.crown) {
      // the crown rides the trunk top (it rises with the cut); the voxels root each turn at the top of its own time
      u = 0.96 + 0.03 * R();
      sy = trunkH * u;
    } else sy = hAt(ses.turns[0].time) * (HB.shrub ? 0.15 + 0.8 * R() : 0.9 + 0.06 * R());
    limbs.push({ ses, dir, sy, u, roots: [], seed: B.rng.seed, ev: [] });
  });
  limbs.sort((a, c) => a.sy - c.sy);
  // lean fixed at birth: re-aiming with age swung whole limbs at every rebuild
  for (const L of limbs) L.dir.lerp(new V3(0, 1, 0), Math.max(HB.up, 0.3)).normalize();
  for (const L of limbs) {
    B.rng.seed = L.seed;
    const u = L.u, root = axis(L.sy);
    const tierAt = u == null ? null : (t: number) => hAtC(t) * u;
    B.VOX.shiftAt = tierAt && ((t: number) => axis(tierAt(t)).sub(root));
    growLimb(B, L.ses, root, L.dir, b.s, cut, false, 0);
    L.ev = B.LAST_EV;
    L.roots = tierAt ? L.ev.map(([t]) => [t, tierAt(t)]) : [[L.ses.turns[0].time, L.sy]];
  }
  B.VOX.shiftAt = null;
  if (!HB.ground) (B.VOX.trunk = B.VOX.lines.length), B.VOX.lines.push(voxelTrunk(limbs, HB, b.s, axis));
  const P = [new V3(b.x, 0, b.z), new V3(b.x, trunkH, b.z), ...B.NODES.map((n) => n.p)];
  return { nodes: B.NODES, P, vox: B.VOX };
}

/**
 * The voxel trunk: samples on the trunk axis at fixed heights (`TRUNK_STEP`) up to the highest limb root (palm: the
 * highest turn tier). A sample exists from the first root at or above it; its girth is the trunk taper above (pipe
 * model, bulge, root flare, or the palm's pole) counted as turns land, as a running maximum (see `schedule`).
 * Deviation from the mockup's bark trunk: no smoothing pass (it mixed neighbouring heights through the cut-dependent
 * sampling).
 */
function voxelTrunk(limbs: Limb[], HB: Hab, s: number, axis: (y: number) => V3): WoodSample[] {
  const roots = limbs.flatMap((L) => L.roots).sort((a, c) => a[0] - c[0]);
  const tops: [number, number][] = [];
  for (const [t, y] of roots) tops.push([t, Math.max(y, tops.length ? tops[tops.length - 1][1] : 0)]);
  const topAt = (t: number) => {
    let m = 0;
    for (let i = 0; i < tops.length && tops[i][0] <= t; i++) m = tops[i][1];
    return m;
  };
  const top = tops.length ? tops[tops.length - 1][1] : 0;
  const reach = limbs.map((L) => L.roots.reduce((m, r) => Math.max(m, r[1]), -Infinity));
  const out: WoodSample[] = [];
  for (let k = 0; k * TRUNK_STEP <= top; k++) {
    const y = k * TRUNK_STEP, above = limbs.filter((_, i) => reach[i] >= y);
    const born = roots.find((r) => r[1] >= y)![0];
    const flare = 1 + 0.5 * Math.exp(-y / (0.18 * s)) + 0.1 * Math.exp(-y / (0.8 * s));
    const radius = (l: number, t: number) => {
      const ty = Math.max(topAt(t), 1e-3);
      return HB.pole
        ? HB.pole * s * (1 - (0.25 * y) / ty)
        : pipeR(l + 0.5, s) * 0.85 * HB.girth * (1 + HB.bulge * 1.1 * Math.sin(Math.PI * (0.12 + 0.8 * Math.min(1, y / ty)))) * flare;
    };
    out.push({ p: axis(y), born, ...schedule(above.flatMap((L) => L.ev), born, radius) });
  }
  return out;
}
