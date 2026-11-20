import { describe, it, expect } from 'vitest';
import { laneOrder, layoutTimeline } from '@/lib/sessions/layout';
import { fixture, turn } from './session-fixture';
import type { SessionResponse } from '@/lib/sessions/types';

const O = { width: 1000, laneH: 20, maxLanes: 10, expanded: false };

describe('session timeline layout', () => {
  it('orders main, its compaction, then subagents by start', () => {
    expect(laneOrder(fixture().conversations).map((c) => c.id)).toEqual(['c-main', 'c-comp', 'c-sub1', 'c-sub2', 'c-sub3']);
  });

  it('maps time to x, and never draws a block narrower than 2px', () => {
    const s = fixture();
    s.turns.push(turn('tiny', 'c-main', 5, 46.99, { finished_at: 47 }));
    const L = layoutTimeline(s, O);
    const m1 = L.blocks.find((b) => b.requestId === 'm1')!;
    const m4 = L.blocks.find((b) => b.requestId === 'm4')!;
    expect(m1.x).toBeLessThan(m4.x);
    expect(Math.min(...L.blocks.map((b) => b.w))).toBeGreaterThanOrEqual(2);
    expect(L.blocks.find((b) => b.requestId === 'p2')!.relayed).toBe(true);
    expect(L.blocks.find((b) => b.requestId === 'm3')!.stripe).toBe('bg-orange-600');
    expect(L.ticks.length).toBeGreaterThan(1);
  });

  it('emits a dashed inferred link from the parent block to the child lane', () => {
    const L = layoutTimeline(fixture(), O);
    const link = L.links.find((l) => l.to === 'c-sub1')!;
    expect(link.from).toBe('m2');
    expect(link.kind).toBe('inferred');
    expect(link.d.startsWith('M')).toBe(true);
  });

  it('collapses lanes beyond the cap into hidden', () => {
    const s = fixture();
    const L = layoutTimeline(s, { ...O, maxLanes: 3 });
    expect(L.lanes).toHaveLength(3);
    expect(L.hidden).toBe(2);
    expect(layoutTimeline(s, { ...O, maxLanes: 3, expanded: true }).hidden).toBe(0);
  });

  it('marks evicted blocks from the turns\' evicted_on_resume', () => {
    const L = layoutTimeline(fixture(), O);
    expect(L.blocks.filter((b) => b.evicted).map((b) => b.requestId)).toEqual(['m4']);
  });

  it('lays out 5,000 turns in under 50 ms', () => {
    const s: SessionResponse = fixture();
    s.turns = Array.from({ length: 5000 }, (_, i) => turn(`x${i}`, 'c-main', i + 1, i * 3));
    // Process CPU time (user + system), not wall: a loaded CI runner stretches wall time, not
    // the layout. Node has no per-thread CPU clock; the layout is synchronous on this thread.
    const t0 = process.cpuUsage();
    layoutTimeline(s, O);
    const { user, system } = process.cpuUsage(t0);
    expect((user + system) / 1000).toBeLessThan(50);
  });
});
