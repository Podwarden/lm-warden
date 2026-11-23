// Pure layout of the session timeline: one lane per conversation (main first, each compaction continuation right
// under its parent, then the rest by start), one block per turn, connectors from a parent turn to its child lane.
// O(turns); 5,000 turns lay out in a few ms.

import { causeClass, shareClass } from "@/lib/sessions/colours";
import type { Conversation, LinkKind, SessionResponse } from "@/lib/sessions/types";

export type Selection = { conversationId: string | null; requestId: string | null };

export interface LaneBox {
  conversationId: string | null;
  label: string;
  kind: Conversation["kind"];
  y: number;
}

export interface Block {
  requestId: string;
  conversationId: string | null;
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
  stripe: string | null;
  relayed: boolean;
  /** A resume after a pause that found its cache evicted (drawn with a notch). */
  evicted: boolean;
}

export interface LinkPath {
  from: string;
  to: string | null;
  d: string;
  kind: LinkKind;
  labelX: number;
  labelY: number;
}

export interface Tick {
  x: number;
  t: number;
}

export const LABEL_W = 140;
export const MIN_BLOCK_W = 2;

export function laneOrder(convs: Conversation[]): Conversation[] {
  const byStart = [...convs].sort((a, b) => a.first_at - b.first_at);
  const comps = new Map<string | null, Conversation[]>();
  for (const c of byStart) {
    if (c.kind === "compaction" && c.parent) {
      const k = c.parent.conversation_id;
      comps.set(k, [...(comps.get(k) ?? []), c]);
    }
  }
  const out: Conversation[] = [];
  const seen = new Set<Conversation>();
  const visit = (c: Conversation) => {
    if (seen.has(c)) return;
    seen.add(c);
    out.push(c);
    for (const k of comps.get(c.id) ?? []) visit(k);
  };
  const main = byStart.find((c) => c.kind === "main");
  if (main) visit(main);
  for (const c of byStart) if (c.kind !== "compaction" && c.kind !== "unlinked") visit(c);
  for (const c of byStart) if (c.kind !== "unlinked") visit(c); // orphaned compactions
  for (const c of byStart) visit(c); // unlinked last
  return out;
}

export function layoutTimeline(
  s: SessionResponse,
  opts: { width: number; laneH: number; maxLanes: number; expanded: boolean },
): { lanes: LaneBox[]; blocks: Block[]; links: LinkPath[]; hidden: number; ticks: Tick[] } {
  const ordered = laneOrder(s.conversations);
  const shown = opts.expanded ? ordered : ordered.slice(0, opts.maxLanes);
  const laneIdx = new Map<string | null, number>();
  const lanes: LaneBox[] = shown.map((c, i) => {
    laneIdx.set(c.id, i);
    return { conversationId: c.id, label: c.label, kind: c.kind, y: i * opts.laneH };
  });
  let t0 = Infinity;
  let t1 = -Infinity;
  for (const t of s.turns) {
    if (t.started_at < t0) t0 = t.started_at;
    if (t.finished_at > t1) t1 = t.finished_at;
  }
  if (!Number.isFinite(t0)) {
    t0 = 0;
    t1 = 1;
  }
  const span = Math.max(t1 - t0, 1e-6);
  const plotW = Math.max(opts.width - LABEL_W, 10);
  const x = (t: number) => LABEL_W + ((t - t0) / span) * plotW;
  const pad = Math.max(2, Math.round(opts.laneH * 0.2));
  const h = opts.laneH - 2 * pad;
  const blocks: Block[] = [];
  const byReq = new Map<string, Block>();
  for (const t of s.turns) {
    const li = laneIdx.get(t.conversation_id);
    if (li === undefined) continue;
    const bx = x(t.started_at);
    const share = t.prompt_tokens > 0 && t.cached_tokens !== null ? t.cached_tokens / t.prompt_tokens : null;
    const b: Block = {
      requestId: t.request_id,
      conversationId: t.conversation_id,
      x: bx,
      y: li * opts.laneH + pad,
      w: Math.max(MIN_BLOCK_W, x(t.finished_at) - bx),
      h,
      fill: shareClass(share),
      stripe: causeClass(t.divergence_cause ?? t.divergence_cause_own),
      relayed: t.relayed,
      evicted: t.evicted_on_resume ?? false,
    };
    blocks.push(b);
    byReq.set(t.request_id, b);
  }
  const links: LinkPath[] = [];
  for (const c of shown) {
    if (!c.parent) continue;
    const from = byReq.get(c.parent.request_id);
    const li = laneIdx.get(c.id);
    if (!from || li === undefined) continue;
    const x1 = from.x + from.w / 2;
    const y1 = from.y + from.h;
    const x2 = x(c.first_at);
    const y2 = li * opts.laneH + opts.laneH / 2;
    links.push({
      from: c.parent.request_id,
      to: c.id,
      kind: c.parent.link,
      d: `M ${x1.toFixed(1)} ${y1.toFixed(1)} V ${y2.toFixed(1)} H ${x2.toFixed(1)}`,
      labelX: x1 + 4,
      labelY: y2 - 2,
    });
  }
  const n = 5;
  const ticks: Tick[] = Array.from({ length: n }, (_, i) => {
    const t = t0 + (span * i) / (n - 1);
    return { t, x: x(t) };
  });
  return { lanes, blocks, links, hidden: ordered.length - shown.length, ticks };
}
