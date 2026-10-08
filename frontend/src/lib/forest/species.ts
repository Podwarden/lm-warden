/**
 * Spec §6.2 — traits → species, and the knob set that shapes each species (README §4.2).
 * Computed once per tree from its sessions, so a tree never changes species in replay.
 */
import type { Session, Traits } from "./types";

export type SpeciesName =
  | "cactus" | "bamboo" | "willow" | "banyan" | "birch" | "palm" | "pine"
  | "baobab" | "bonsai" | "shrub" | "poplar" | "oak" | "round";

export interface SpeciesContext {
  /** The tree's model variant's context window. */
  maxModelLen?: number;
  /** The tree's top-level sessions (children nested), for the client-side traits. */
  sessions?: Session[];
}

/** Every species threshold in one table so they can be tuned (spec §6.2). */
export const SPECIES_THRESHOLDS = {
  cactusFail: 0.35,
  bambooCron: 0.7,
  bambooThrash: 0.05,
  willowThrash: 0.35,
  banyanFanout: 1.5,
  birchReasoning: 0.5,
  palmSingleTurnShare: 0.7,
  palmSys0: 30_000,
  pineTurns: 40,
  baobabSys0: 60_000,
  bonsaiMaxModelLen: 16_384,
  shrubSys0: 5_000,
  shrubMeanGen: 260,
  poplarCtxGen: 12,
  oakSys0: 20_000,
} as const;
const TH = SPECIES_THRESHOLDS;

/** Raw request rows behind a session's turns (a merged LOD turn stands for `mergedN` rows). */
const rawTurns = (s: Session) => s.turns.reduce((n, t) => n + Math.max(1, t[7]), 0);

/**
 * Cron-likeness of a tree's sessions, 0..1: 1 − coefficient of variation of the gaps between consecutive
 * top-level session start times (first-turn `t`), clamped to 0..1. Fewer than 4 sessions (under 3 gaps) is 0:
 * too few starts to call it periodic. Sessions with no turns are ignored.
 */
export function cronScore(sessions: Session[]): number {
  const starts = sessions.filter((s) => s.turns.length).map((s) => s.turns[0][0]).sort((a, b) => a - b);
  if (starts.length < 4) return 0;
  const gaps = starts.slice(1).map((t, i) => t - starts[i]);
  const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
  if (mean <= 0) return 0;
  const sd = Math.sqrt(gaps.reduce((a, g) => a + (g - mean) ** 2, 0) / gaps.length);
  return Math.min(1, Math.max(0, 1 - sd / mean));
}

/** Share of top-level sessions that are a single request (palm). 0 with no sessions. */
export function singleTurnShare(sessions: Session[]): number {
  const live = sessions.filter((s) => s.turns.length);
  if (!live.length) return 0;
  return live.filter((s) => rawTurns(s) === 1).length / live.length;
}

/** Exactly one top-level session (turnless ones ignored), with ≥ 40 raw turns and no subagents (pine). */
export function isPine(sessions: Session[]): boolean {
  const live = sessions.filter((s) => s.turns.length);
  return live.length === 1 && live[0].children.length === 0 && rawTurns(live[0]) >= TH.pineTurns;
}

export interface SpeciesRule {
  name: SpeciesName;
  test(traits: Traits, ctx: SpeciesContext): boolean;
}

/** Spec §6.2, evaluated in order; the first match wins. */
export const SPECIES_RULES: readonly SpeciesRule[] = [
  { name: "cactus", test: (t) => t.fail > TH.cactusFail },
  { name: "bamboo", test: (t, c) => cronScore(c.sessions ?? []) > TH.bambooCron && t.thrash < TH.bambooThrash },
  { name: "willow", test: (t) => t.thrash > TH.willowThrash },
  { name: "banyan", test: (t) => t.fanout >= TH.banyanFanout },
  // birch: reasoning > 0.5. The wire carries no reasoning-token count in v1, so reasoning is unknown (0) and this
  // rule never fires. Kept so the table matches the spec and lights up once the backend sends the trait.
  { name: "birch", test: () => 0 > TH.birchReasoning },
  {
    name: "palm",
    test: (t, c) => singleTurnShare(c.sessions ?? []) >= TH.palmSingleTurnShare && t.sys0 > TH.palmSys0,
  },
  { name: "pine", test: (_t, c) => isPine(c.sessions ?? []) },
  { name: "baobab", test: (t) => t.sys0 > TH.baobabSys0 },
  { name: "bonsai", test: (_t, c) => c.maxModelLen != null && c.maxModelLen <= TH.bonsaiMaxModelLen },
  { name: "shrub", test: (t) => t.sys0 < TH.shrubSys0 && t.mean_gen < TH.shrubMeanGen },
  { name: "poplar", test: (t) => t.ctx_gen < TH.poplarCtxGen },
  { name: "oak", test: (t) => t.sys0 > TH.oakSys0 },
  { name: "round", test: () => true },
];

export function pickSpecies(traits: Traits, ctx: SpeciesContext): SpeciesName {
  for (const r of SPECIES_RULES) if (r.test(traits, ctx)) return r.name;
  return "round";
}

/** The knobs that shape a tree (the old look's per-species foliage and bark colours are gone: Plan 4). */
export interface Knobs {
  girth?: number;
  tall?: number;
  el?: number;
  up?: number;
  leafy?: number;
  bulge?: number;
  shrub?: boolean;
  wig?: number;
  photo?: number;
  cone?: boolean;
  len?: number;
  droop?: number;
  twig?: number;
  lg?: number;
  crown?: boolean;
  pole?: number;
  elSpread?: number;
  ground?: boolean;
  twist?: number;
}

/**
 * Ported from the mockup's `SP` object (forest-real.html), values verbatim except the old look's colours and roots. `round` has no `SP` entry: the mockup
 * shapes a round crown purely from `habit()` with no species overrides, so its knob set is empty.
 */
export const KNOBS: Record<SpeciesName, Knobs> = {
  oak: { girth: 1.15, tall: 0.8, el: 0.48 },
  baobab: { leafy: 0.6, girth: 1.45, bulge: 0.9, tall: 0.6, el: 0.35 },
  poplar: { girth: 0.6, tall: 1.75, el: 1.1, up: 0.55 },
  shrub: { shrub: true },
  pine: { leafy: 1.6, girth: 1.0, tall: 1.0, el: 1.5, wig: 0.12, photo: 6, cone: true, len: 0.75 },
  banyan: { girth: 1.1, tall: 0.8, el: 0.3 },
  willow: { leafy: 1.4, girth: 0.95, tall: 1.05, el: 0.75, droop: 0.11, len: 1.05, twig: 1.3, lg: 0.85 },
  birch: { girth: 0.55, tall: 1.6, el: 0.95, up: 0.3 },
  palm: {
    leafy: 1.5, crown: true, pole: 0.11, tall: 2.4, el: 0.5, elSpread: 0.04, droop: 0.08, len: 1.5,
    twig: 0.6,
  },
  bamboo: {
    ground: true, el: 1.48, elSpread: 0.015, wig: 0.05, photo: 6, up: 0.95, len: 1.25, lg: 1.7,
    twig: 0.5,
  },
  bonsai: { twist: 4, el: 0.12, len: 0.6, wig: 2.2, girth: 1.4, tall: 0.55 },
  cactus: {
    leafy: 0, twig: 0, lg: 3.4, el: 0.05, elSpread: 0.02, photo: 10, wig: 0.15, girth: 2.4, tall: 1.0, len: 0.75,
  },
  round: {},
};
