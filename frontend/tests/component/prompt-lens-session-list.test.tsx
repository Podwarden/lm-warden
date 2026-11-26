// The Sessions list row: title, sub line, worded scores, the not-scored line.
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SessionList } from '@/components/prompt-lens/sessions/session-list';
import type { SessionListItem } from '@/lib/prompt-lens/scorecard';

const item = (o: Partial<SessionListItem>): SessionListItem => ({
  session_id: 's_' + 'a'.repeat(32), model: 'fake-model', route: 'local', harness: 'opencode', turns: 12,
  started: 1_700_000_000, last_at: 1_700_000_000 + 5400, overall: null, e: null, c: 74, r: null, confidence: 'low',
  reasons: { E: '0 of 3 continuing turns', R: '0 of 5 assistant turns' }, score_version: 1, key_name: 'alpha-key', ...o,
});

let items: SessionListItem[] = [];
vi.mock('@/lib/prompt-lens/scorecard', async (orig) => ({
  ...(await orig<typeof import('@/lib/prompt-lens/scorecard')>()),
  useSessionList: () => ({ data: { sessions: items, next_cursor: null }, loading: false, error: null }),
}));

afterEach(cleanup);

describe('session list row', () => {
  it('admin: key name and harness lead, duration follows, no model, layers in words', () => {
    items = [item({})];
    render(<SessionList scope="admin" selected={null} onOpen={() => {}} />);
    expect(screen.getByTestId('row-title')).toHaveTextContent('alpha-key · OpenCode session');
    expect(screen.getByTestId('row-sub')).toHaveTextContent('12 turns · 1 h 30 min');
    const row = screen.getByTestId('lens-session-row');
    expect(row).not.toHaveTextContent('fake-model');
    expect(row).toHaveTextContent('Efficiency —');
    expect(row).toHaveTextContent('Construction 74');
    expect(row).toHaveTextContent('Response —');
    expect(row).toHaveTextContent('low confidence');
    expect(screen.queryByTestId('row-overall')).toBeNull();
    expect(screen.getByTestId('row-not-scored')).toHaveTextContent(
      'Not scored: Efficiency (0 of the 3 continuing turns it needs), Response (0 of the 5 measured replies it needs)',
    );
    expect(screen.getByTestId('row-E').getAttribute('title')).toMatch(/how much of each prompt the cache could reuse/);
  });

  it('admin: a deleted key reads "unnamed key"; key scope omits the key and a full score has no not-scored line', () => {
    items = [item({ key_name: null, overall: 78, e: 70, c: 74, r: 85, confidence: 'high', reasons: {} })];
    const { unmount } = render(<SessionList scope="admin" selected={null} onOpen={() => {}} />);
    expect(screen.getByTestId('row-title')).toHaveTextContent('unnamed key · OpenCode session');
    expect(screen.getByTestId('row-overall')).toHaveTextContent('Overall 78');
    expect(screen.queryByTestId('row-not-scored')).toBeNull();
    unmount();
    render(<SessionList scope="key" selected={null} onOpen={() => {}} />);
    expect(screen.getByTestId('row-title').textContent).toBe('OpenCode session');
  });
});
