import { describe, it, expect } from 'vitest';
import { causeClass, LINK_STYLE, shareClass, svgFill } from '@/lib/sessions/colours';

describe('sessions colours', () => {
  it.each([[0.95, 'bg-emerald-500'], [0.8, 'bg-emerald-500'], [0.5, 'bg-amber-500'], [0.2, 'bg-amber-500'], [0.1, 'bg-rose-500'], [null, 'bg-slate-400']] as const)(
    'share %s', (share, cls) => expect(shareClass(share)).toBe(cls),
  );
  it('stripes compaction yellow and harness rewrite orange', () => {
    expect(causeClass('compacted')).toBe('bg-yellow-400');
    expect(causeClass('harness_rewrite')).toBe('bg-orange-600');
    expect(causeClass(null)).toBeNull();
  });
  it('maps to svg fills and dashes inferred links', () => {
    expect(svgFill('bg-emerald-500')).toBe('fill-emerald-500');
    expect(LINK_STYLE.inferred.dash).toBe('4 3');
    expect(LINK_STYLE.known.dash).toBeUndefined();
  });
});
