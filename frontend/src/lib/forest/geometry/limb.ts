/**
 * Ports the mockup's limb growth (forest-real.html l.256-323): the pipe model (`PIPE_K`, `PIPE_E`, `pipeR`), `RAMP`
 * and `growLimb` with its twigs, foliage points and blossom tips — the voxelizer's inputs (voxin.ts) and the turn
 * nodes. The mockup's globals (`seed`, `HAB`, `NODES`) live on one `Build` per tree build. The old look's bark tubes,
 * leaf instances and bark colours are gone (Plan 4); their random draws are kept so the seeded shapes are unchanged.
 */
import type { Knobs } from "../species";
import type { Rng } from "./random";
import { V3 } from "./tube";
import { type Ev, type VoxRec, type WoodSample, lerpSched, schedule, shifted } from "./voxin";

/** A turn as the mockup's `growLimb` reads it, after normalization (see `adaptTurn` in tree.ts). */
export interface GTurn {
  /** Index in its session (mockup `t.i`; kept through `ramify`). */
  i: number;
  time: number;
  /** Mockup-equivalent generated tokens: drives segment length. */
  gen: number;
  /** Mockup-equivalent context tokens: drives the foliage count. */
  ctx: number;
  /** One entry per leaf: the tool calls the turn made. */
  tools: { fam: string; size: number; fail: boolean }[];
}

/** A session or a `ramify` shoot. */
export interface GSes {
  id: string;
  /** The wire session this shoot belongs to (for picking). */
  src: string;
  turns: GTurn[];
  children: GSes[];
  /** Parent turn this child forks from. */
  at: number;
  /** A ramify shoot (not a subagent). */
  shoot: boolean;
}

/** The habit fields every tree has (from `habit()`); the knobs may override them. */
type HabCore = "girth" | "bulge" | "tall" | "el" | "up" | "shrub";

/**
 * Mockup `HAB`: `habit()` merged with the species knobs; colours already linear. Every other field is typed from
 * `Knobs`, so a knob read or written here under a misspelt name is a type error.
 */
export type Hab = Omit<Knobs, HabCore> & Required<Pick<Knobs, HabCore>>;

export interface Node {
  /** The turn's joint on its limb, or its twig tip. */
  kind: "joint" | "twig";
  p: V3;
  t: GTurn;
  ses: GSes;
  /** Mockup `treeName.includes('subagent')`. */
  sub: boolean;
}

/** Everything the mockup kept in globals during one `buildTree` call. */
export interface Build {
  rng: Rng;
  HAB: Hab;
  NODES: Node[];
  /** The voxelizer's inputs (prefix-stable points, sizes and girth schedules). */
  VOX: VoxRec;
  /** The last `growLimb`'s base load events (its turns, its base and its subagents'), for the trunk's schedule. */
  LAST_EV: Ev[];
}

// Pipe model (da Vinci): the cross-section at any point carries everything that grows beyond it.
export const PIPE_K = 0.0135, PIPE_E = 0.5;
export const pipeR = (load: number, scale: number) =>
  Math.max(0.011 * scale, PIPE_K * scale * Math.pow(Math.max(load, 0.6), PIPE_E));
/** A turn's share of height and girth eases in over 90 s (smoothstep). */
export const RAMP = (now: number, t: number) => {
  const u = Math.min(1, Math.max(0, (now - t) / 90));
  return u * u * (3 - 2 * u);
};

const UP = () => new V3(0, 1, 0);

/** A `ramify` shoot length (mockup `SH`). Every session is cut into shoots of this many turns. */
export const SH = 5;

/**
 * Ports `growLimb(ses, start, dir, scale, cut, …, depth)`: one segment and one twig per turn, subagents first (their
 * load flows into this limb below the fork), then the bowed centreline. It records what the voxels are built from
 * (foliage points, blossom tips, the centreline with its girth schedules) and the turn nodes; the limb's base load
 * events are left in `B.LAST_EV`. The old look's tubes, leaf instances and bark colours are not built, but every random
 * draw they made is still taken, in order, so the seeded shapes are unchanged.
 */
export function growLimb(B: Build, ses: GSes, start: V3, dir: V3, scale: number, cut: number, sub: boolean, depth: number): void {
  const { rng, HAB } = B;
  const R = () => rng.R();
  rng.SEED(ses.id, "limb");
  const pts = [start.clone()], nodes: { p: V3; d: V3; t: GTurn }[] = [];
  let p = start.clone();
  const d = dir.clone().normalize();
  const wander = new V3().crossVectors(d, new V3(0, 1, 0.001)).normalize();
  const ph = R() * 6;
  const turns = ses.turns.filter((t) => t.time <= cut);
  if (!turns.length) {
    B.LAST_EV = [];
    return;
  }
  const twigLoad: number[] = [];
  for (const t of turns) {
    rng.SEED(ses.id, "turn", t.i);
    const len = Math.min(1.1, 0.3 + Math.sqrt(t.gen) / 60) * scale * (HAB.len ?? 1);
    wander.applyAxisAngle(d, (R() - 0.5) * 0.9);
    wander.lerp(UP().cross(d).normalize(), 0.15).normalize();
    d.addScaledVector(wander, (0.17 + 0.06 * Math.sin(t.i * 0.7 + ph)) * (HAB.wig ?? 1));
    if (R() < 0.18 * (HAB.wig ?? 1)) {
      const kink = new V3(R() - 0.5, (R() - 0.3) * 0.4, R() - 0.5).normalize();
      d.addScaledVector(kink, 0.32);
    }
    if (HAB.droop) d.lerp(new V3(0, -1, 0), HAB.droop * (depth ? 1.3 : 1)).normalize();
    else d.lerp(UP(), Math.min(0.9, (depth ? 0.12 : 0.07) * (HAB.photo ?? 1))).normalize();
    p = p.clone().addScaledVector(d, len);
    pts.push(p);
    nodes.push({ p: p.clone(), d: d.clone(), t });
    // twig: its own small pipe — thicker when the turn carried more tool calls
    const tl_ = 1 + 0.55 * t.tools.length;
    twigLoad.push(tl_);
    /** Where an answered turn's blossom sits: its twig tip, or its joint when the species has no twigs. */
    let bloomAt: V3 | null = null;
    if (HAB.twig !== 0) {
      R();
      const base = p.clone();
      const az = R() * Math.PI * 2;
      const out = new V3().crossVectors(d, new V3(Math.cos(az), 0.2, Math.sin(az))).normalize();
      const td = HAB.cone
        ? out.clone().setY(-0.08).normalize()
        : d.clone().multiplyScalar(0.55).add(out).add(new V3(0, HAB.droop ? -0.9 : 0.35, 0)).normalize();
      const toolTok = t.tools.reduce((a, b) => a + b.size, 0);
      // pine cone: the mockup tapered by t.i / max(turns, ses.turns.length). With every session cut into shoots that
      // went negative past turn 5 (twigs pointed back through the limb); taper along the shoot instead (prefix-stable)
      const tlen = (t.tools.length ? 0.28 + Math.min(1.1, Math.sqrt(toolTok) / 95) : 0.18) * scale * (depth ? 0.8 : 1) *
        (HAB.twig ?? 1) * (HAB.cone ? 0.5 + 3.2 * (1 - (t.i % SH) / SH) : 1);
      const n = 4, tp = [base.clone()];
      let q = base.clone();
      for (let m = 1; m <= n; m++) {
        td.add(new V3((R() - 0.5) * 0.35, 0.06, (R() - 0.5) * 0.35)).normalize();
        q = q.clone().addScaledVector(td, tlen / n);
        tp.push(q);
      }
      B.NODES.push({ kind: "twig", p: q.clone(), t, ses, sub });
      // tool calls: a foliage point each (failed ones marked), along the twig
      t.tools.forEach((tl, j) => {
        const f = 0.35 + (0.65 * (j + 1)) / (t.tools.length + 0.2);
        const k2 = Math.min(n - 1, Math.floor(f * n)), w2 = f * n - k2;
        const at = tp[k2].clone().lerp(tp[k2 + 1], w2);
        const sideA = R() * Math.PI * 2;
        const lside = new V3().crossVectors(td, new V3(Math.cos(sideA), 0.4, Math.sin(sideA))).normalize();
        // the leaf's size without LEAFK (which grows with the tree's work and would re-cover old cells)
        const raw = (0.6 + Math.min(0.8, Math.sqrt(tl.size) / 90)) * scale;
        at.addScaledVector(lside, 0.01);
        const va = shifted(B.VOX, at, t.time);
        B.VOX.foliage.push(va.x, va.y, va.z, raw, t.time, tl.fail ? 1 : 0);
        R(); // the old look's leaf spin: still drawn, so every later draw is unchanged
      });
      // the turn's context: foliage points in the outer half of the twig
      {
        const nf = Math.round((7 + Math.min(13, Math.log2(1 + t.ctx / 2000) * 2.6)) * (HAB.leafy ?? 1));
        for (let f = 0; f < nf; f++) {
          const u = 0.55 + 0.45 * Math.sqrt(R()), kf = Math.min(n - 1, Math.floor(u * n)), wf = u * n - kf;
          const at = tp[kf].clone().lerp(tp[kf + 1], wf)
            .add(new V3((R() - 0.5) * 0.5, (R() - 0.3) * 0.4, (R() - 0.5) * 0.5).multiplyScalar(scale * 0.55));
          R(), R(), R(); // the old look's leaf direction
          const raw = (0.55 + R() * 0.35) * scale;
          const va = shifted(B.VOX, at, t.time);
          B.VOX.foliage.push(va.x, va.y, va.z, raw, t.time, 0);
          R(), R(), R(), R(); // the old look's leaf spin and colour
        }
      }
      if (!t.tools.length) bloomAt = q;
    } else if (!t.tools.length) bloomAt = p;
    if (bloomAt) {
      const tip = shifted(B.VOX, bloomAt, t.time);
      B.VOX.tips.push(tip.x, tip.y, tip.z, t.time);
    }
    // mockup: compaction rings (`ringAt`) — the wire carries no compaction flag, so none are built
  }
  // subagents first: their load flows into this limb below the point where they branch off
  const childEv: { at: number; ev: Ev[] }[] = [];
  for (const ch of ses.children) {
    if (ch.at < nodes.length) {
      rng.SEED(ch.id, "fork");
      const nd = nodes[ch.at];
      const out = new V3().crossVectors(nd.d, new V3(R() - 0.5, 0, R() - 0.5)).normalize();
      growLimb(B, ch, nd.p, nd.d.clone().multiplyScalar(0.4).add(out).normalize(), scale * 0.78, cut, ch.shoot ? sub : true, depth + 1);
      childEv.push({ at: ch.at + 1, ev: B.LAST_EV });
    }
  }
  // voxel girth (pipe model): point i carries every segment, twig and subagent beyond it, counted as each turn lands
  const evAt = (i: number): Ev[] => [
    [turns[0].time, 0.6],
    ...turns.slice(i).map((t, k): Ev => [t.time, 1 + twigLoad[i + k]]),
    ...childEv.filter((c) => i <= c.at).flatMap((c) => c.ev),
  ];
  const vr = (l: number) => pipeR(l, scale) * (HAB.lg ?? 1) * 0.72;
  const jointBorn = (i: number) => turns[Math.max(0, i - 1)].time;
  const joints = pts.map((_, i) => schedule(evAt(i), jointBorn(i), vr));
  const line: WoodSample[] = [{ p: shifted(B.VOX, pts[0], jointBorn(0)), born: jointBorn(0), ...joints[0] }];
  {
    // wood is never straight between joints: two bowed midpoints per internode (seeded, stable in replay)
    rng.SEED(ses.id, "bow");
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1], b = pts[i], seg = b.clone().sub(a), L = seg.length();
      const side = new V3(R() - 0.5, R() - 0.5, R() - 0.5).cross(seg).normalize();
      for (const f of [1 / 3, 2 / 3]) {
        const q = a.clone().lerp(b, f)
          .addScaledVector(side, L * 0.09 * Math.sin(Math.PI * f) * (0.6 + R() * 0.8))
          .add(new V3(0, L * 0.04 * Math.sin(Math.PI * f), 0));
        line.push({ p: shifted(B.VOX, q, nodes[i - 1].t.time), born: nodes[i - 1].t.time, ...lerpSched(joints[i - 1], joints[i], f) });
      }
      line.push({ p: shifted(B.VOX, b, nodes[i - 1].t.time), born: nodes[i - 1].t.time, ...joints[i] });
    }
    B.VOX.lines.push(line);
  }
  for (const nd of nodes) B.NODES.push({ kind: "joint", p: nd.p, t: nd.t, ses, sub });
  // mockup: the live tip sphere (`ses.live`) is drawn by the renderer, not baked into the wood
  B.LAST_EV = evAt(0);
}
