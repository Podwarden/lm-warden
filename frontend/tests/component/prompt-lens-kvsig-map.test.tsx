import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';

import { KvsigFooter, KvsigMap } from '@/components/prompt-lens/dock/kvsig-map';
import { PartsTable } from '@/components/prompt-lens/dock/parts-table';
import { mapParts, type KvsigScope } from '@/lib/prompt-lens/kvsig';
import type { RequestView } from '@/lib/prompt-lens/store';

import { PROMPT, part, reqWith } from '../unit/prompt-lens-kvsig-fixtures';

afterEach(cleanup);

const turn17 = reqWith({
  parts: [
    part('sys', -1, -1, 'exact', { chars: 3120 }),
    part('tools', -1, -1, 'exact', { chars: 9840 }),
    part('nodes', 0, 13, 'exact', { chars: 26400, roles: { user: 5, assistant: 5, tool: 4 } }),
    part('nodes', 14, 14, 'loose', { chars: 1850, roles: { assistant: 1 } }),
    part('nodes', 15, 15, 'new', { chars: 2300, roles: { tool: 1 } }),
    part('nodes', 16, 16, 'new', { chars: 410, roles: { user: 1 } }),
  ],
  end: { prompt_tokens: 43920, cached_tokens: 39360, own_reused_tokens: 39360, cache: { cache_outcome: 'hit' } },
  prompt: PROMPT,
});

/** A whole prompt map block as the request screen composes it. */
function Block({ r, scope, phone = false }: { r: RequestView; scope: KvsigScope; phone?: boolean }) {
  const v = mapParts(r, scope);
  return (
    <div>
      <KvsigMap view={v} onSegment={() => {}} />
      <PartsTable parts={v.parts} phone={phone} />
      <KvsigFooter view={v} />
    </div>
  );
}

// The same conversation in two worlds. In the first another key had already
// sent the system prompt and tools, so the admin sees `other` and the engine
// cached more; the key holder must see exactly what they see in the second
// world, where no other key exists.
const withOtherKey = reqWith({
  parts: [
    part('sys', -1, -1, 'other', { chars: 2610 }),
    part('tools', -1, -1, 'other', { chars: 6130 }),
    part('nodes', 0, 0, 'other', { chars: 700, roles: { user: 1 } }),
    part('nodes', 1, 1, 'new', { chars: 5360, roles: { user: 1 }, exact8: 'd2c861f0', loose8: 'd2c861f0' }),
  ],
  kvsig: { reusable_own: 0, reusable_any: 9440, edge: 'new', turn: 1, parent_request_id: null },
  end: {
    prompt_tokens: 14800,
    cached_tokens: 9440,
    own_reused_tokens: 0,
    cache: { cache_outcome: 'cold', divergence_cause: null, other_key_label: 'dev-team' },
  },
});
const noOtherKey = reqWith({
  parts: [
    part('sys', -1, -1, 'new', { chars: 2610 }),
    part('tools', -1, -1, 'new', { chars: 6130 }),
    part('nodes', 0, 1, 'new', { chars: 6060, roles: { user: 2 }, exact8: 'd2c861f0', loose8: 'd2c861f0' }),
  ],
  kvsig: { reusable_own: 0, edge: 'new', turn: 1, parent_request_id: null },
  end: { prompt_tokens: 14800, own_reused_tokens: 0 },
});

describe('KvsigMap', () => {
  it('renders one segment per part with flex widths', () => {
    render(<KvsigMap view={mapParts(turn17, 'admin')} onSegment={() => {}} />);
    const segs = screen.getAllByTestId('kvsig-seg');
    expect(segs).toHaveLength(6);
    expect(segs[2].style.flexGrow).toBe('26400');
    expect(segs[3].className).toMatch(/loose/);
  });

  it('clicking a segment calls onSegment with its part', () => {
    const onSegment = vi.fn();
    render(<KvsigMap view={mapParts(turn17, 'admin')} onSegment={onSegment} />);
    fireEvent.click(screen.getAllByTestId('kvsig-seg')[3]);
    expect(onSegment).toHaveBeenCalledTimes(1);
    expect(onSegment.mock.calls[0][0].label).toBe('node 15');
  });

  it('is headed kvsig · prompt map (G-R13)', () => {
    render(<KvsigMap view={mapParts(turn17, 'key')} onSegment={() => {}} />);
    expect(screen.getByRole('heading').textContent).toBe('kvsig · prompt map');
    expect(screen.getByRole('region', { name: 'kvsig · prompt map' })).toBeInTheDocument();
  });

  it('legend new: admin says computed, key says no earlier match', () => {
    render(<KvsigMap view={mapParts(turn17, 'admin')} onSegment={() => {}} />);
    expect(screen.getByTestId('kvsig-legend').textContent).toContain('new: computed for this request');
    cleanup();
    render(<KvsigMap view={mapParts(turn17, 'key')} onSegment={() => {}} />);
    const t = screen.getByTestId('kvsig-legend').textContent;
    expect(t).toContain('new: no earlier request matches');
    expect(t).not.toContain('computed');
  });

  it('segments have a visible focus style', () => {
    render(<KvsigMap view={mapParts(turn17, 'admin')} onSegment={() => {}} />);
    expect(screen.getAllByTestId('kvsig-seg')[0].className).toMatch(/focus-visible:outline/);
  });

  it('the legend has no another-key item for a key holder', () => {
    render(<KvsigMap view={mapParts(withOtherKey, 'key')} onSegment={() => {}} />);
    const legend = screen.getByTestId('kvsig-legend');
    expect(legend.textContent).not.toMatch(/another key|other key|attribut/i);
    expect(within(legend).getAllByTestId('kvsig-swatch')).toHaveLength(3);
  });

  it('the admin legend and brackets name another key', () => {
    const { container } = render(<KvsigMap view={mapParts(withOtherKey, 'admin')} onSegment={() => {}} />);
    expect(screen.getByTestId('kvsig-legend').textContent).toMatch(/cached by another key/);
    expect(container.textContent).toMatch(/engine cached 9,440 tok/);
    expect(container.textContent).toMatch(/prefix cached by another key \(any-key index\) 9,440 tok/);
  });

  it('a key holder sees own reuse, never the engine figure', () => {
    const { container } = render(<KvsigMap view={mapParts(turn17, 'key')} onSegment={() => {}} />);
    expect(container.textContent).toMatch(/reused from your earlier requests 39,360 tok/);
    expect(container.textContent).not.toMatch(/engine cached/);
  });

  it('keeps its height before and after request_end (fixed size)', () => {
    const pending = reqWith({ parts: turn17.kvsig!.parts });
    const { rerender } = render(<KvsigMap view={mapParts(pending, 'admin')} onSegment={() => {}} />);
    const before = screen.getByTestId('kvsig-under').className;
    rerender(<KvsigMap view={mapParts(turn17, 'admin')} onSegment={() => {}} />);
    expect(screen.getByTestId('kvsig-under').className).toBe(before);
    expect(before).toMatch(/h-\[/);
  });

  it.each([
    ['not_chat', /not a chat request/],
    ['no_key', /no key/],
    ['error', /could not be built/],
    ['over_cap', /too large/],
  ] as const)('unavailable (%s) renders its empty state', (reason, text) => {
    render(<KvsigMap view={mapParts(reqWith({ kvsig: { available: false, reason, parts: [] } }), 'key')} onSegment={() => {}} />);
    expect(screen.getByTestId('kvsig-empty').textContent).toMatch(text);
    expect(screen.queryAllByTestId('kvsig-seg')).toHaveLength(0);
  });

  it('a queued request says the map comes after admission', () => {
    render(<KvsigMap view={mapParts(reqWith({ kvsig: null, phase: 'queued' }), 'admin')} onSegment={() => {}} />);
    expect(screen.getByTestId('kvsig-empty').textContent).toMatch(/prompt map after admission/);
  });
});

describe('key scope isolation (G-R11)', () => {
  it('a key render with other-key parts is byte-identical to a world with no other key', () => {
    const a = render(<Block r={withOtherKey} scope="key" />);
    const htmlA = a.container.innerHTML;
    a.unmount();
    const b = render(<Block r={noOtherKey} scope="key" />);
    expect(htmlA).toBe(b.container.innerHTML);
  });

  it('the same holds on a phone and with the table open', () => {
    const a = render(<Block r={withOtherKey} scope="key" phone />);
    fireEvent.click(screen.getByRole('button', { name: /Show the 3 parts/ }));
    const htmlA = a.container.innerHTML;
    a.unmount();
    const b = render(<Block r={noOtherKey} scope="key" phone />);
    fireEvent.click(screen.getByRole('button', { name: /Show the 3 parts/ }));
    expect(htmlA).toBe(b.container.innerHTML);
  });

  it('no another-key text, label or tooltip renders in key scope', () => {
    for (const r of [withOtherKey, turn17]) {
      const { container, unmount } = render(<Block r={r} scope="key" />);
      expect(container.innerHTML).not.toMatch(/another key|other key|other-key|any-key|attribut|dev-team/i);
      expect(container.querySelector('.other, [data-state="other"]')).toBeNull();
      unmount();
    }
  });

  it('no internal names render in either scope', () => {
    for (const scope of ['key', 'admin'] as const) {
      const { container, unmount } = render(<Block r={withOtherKey} scope={scope} />);
      expect(container.textContent).not.toMatch(/cobs2|loose\/1|god[ -]?mode/i);
      unmount();
    }
  });
});

describe('PartsTable', () => {
  it('has the columns and short ids', () => {
    const v = mapParts(turn17, 'admin');
    render(<PartsTable parts={v.parts} phone={false} />);
    const heads = screen.getAllByRole('columnheader').map((h) => h.textContent);
    expect(heads).toEqual(['Part', 'Covers', '≈ tok', 'exact id', 'loose id', 'vs earlier', 'Cache']);
    const rows = screen.getAllByRole('row').slice(1);
    expect(rows).toHaveLength(6);
    expect(rows[0].textContent).toContain('e000·0000');
    expect(rows[3].textContent).toContain('loose same, exact differs: the client dropped the reasoning');
    expect(rows[5].textContent).toContain('computed');
    expect(screen.queryByTestId('parts-box')).toBeNull();
    for (const th of screen.getAllByRole('columnheader')) expect(th).toHaveAttribute('scope', 'col');
  });

  it('marks the active row with aria-current', () => {
    const v = mapParts(turn17, 'admin');
    const { rerender } = render(<PartsTable parts={v.parts} phone={false} active={3} />);
    const rows = screen.getAllByRole('row').slice(1);
    expect(rows[3]).toHaveAttribute('aria-current', 'true');
    expect(rows.filter((r) => r.hasAttribute('aria-current'))).toHaveLength(1);
    rerender(<PartsTable parts={v.parts} phone active={2} />);
    fireEvent.click(screen.getByRole('button', { name: /Show the 6 parts/ }));
    expect(screen.getAllByTestId('parts-card')[2]).toHaveAttribute('aria-current', 'true');
  });

  it('over 8 rows sits in a fixed scroll box', () => {
    const parts = [part('sys', -1, -1, 'exact'), ...Array.from({ length: 9 }, (_, i) => part('nodes', i, i, i % 2 ? 'new' : 'exact'))];
    const v = mapParts(reqWith({ parts }), 'admin');
    render(<PartsTable parts={v.parts} phone={false} />);
    const box = screen.getByTestId('parts-box');
    expect(box.className).toMatch(/overflow-y-auto/);
    expect(box.className).toMatch(/h-\[/);
  });

  it('phone: behind a toggle, stacked cards, no id columns', () => {
    const v = mapParts(turn17, 'admin');
    const { container } = render(<PartsTable parts={v.parts} phone />);
    expect(screen.queryByTestId('parts-card')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Show the 6 parts ▸' }));
    expect(screen.getAllByTestId('parts-card')).toHaveLength(6);
    const card = within(screen.getAllByTestId('parts-card')[5]);
    for (const label of ['≈ tok', 'vs earlier', 'Cache']) expect(card.getByText(label)).toBeInTheDocument();
    expect(container.textContent).not.toMatch(/·0000|exact id|loose id/);
    fireEvent.click(screen.getByRole('button', { name: /Hide the 6 parts/ }));
    expect(screen.queryByTestId('parts-card')).toBeNull();
  });

  it('key cache column shows reused or a dash', () => {
    const v = mapParts(reqWith({ parts: turn17.kvsig!.parts, end: { prompt_tokens: 43920, own_reused_tokens: 13000 } }), 'key');
    render(<PartsTable parts={v.parts} phone={false} />);
    const cells = screen.getAllByTestId('parts-cache').map((c) => c.textContent);
    // the key's two new nodes are one run
    expect(cells).toEqual(['reused', 'reused', '—', '—', '—']);
  });

  it('desktop segments stay 26 px tall', () => {
    render(<KvsigMap view={mapParts(turn17, 'admin')} onSegment={() => {}} />);
    expect(screen.getByTestId('kvsig-bar')).toHaveClass('h-[26px]');
  });
});
