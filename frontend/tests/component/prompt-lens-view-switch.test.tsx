// The Prompt Lens view switch (analysis Task 2.5, spec 16.1): Requests and
// Sessions always; Trends and the Leaderboard only when their routes exist; the
// Leaderboard never for a key holder, whatever the props say (D-4).

import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';

import { ViewSwitch, viewsFor } from '@/components/prompt-lens/view-switch';

afterEach(cleanup);

const tabs = () => screen.getAllByRole('tab').map((t) => t.textContent);

describe('ViewSwitch', () => {
  it('offers Requests and Sessions, marks the current one, and reports a change', () => {
    const change = vi.fn();
    render(<ViewSwitch scope="key" view="requests" onChange={change} />);
    expect(tabs()).toEqual(['Requests', 'Sessions']);
    expect(screen.getByRole('tab', { name: 'Requests' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Sessions' })).toHaveAttribute('aria-selected', 'false');
    fireEvent.click(screen.getByRole('tab', { name: 'Sessions' }));
    expect(change).toHaveBeenCalledWith('sessions');
  });

  it('Trends is hidden until its route answers; the Leaderboard is the admin\'s alone', () => {
    const { rerender } = render(<ViewSwitch scope="admin" view="sessions" onChange={vi.fn()} trends leaderboard />);
    expect(tabs()).toEqual(['Requests', 'Sessions', 'Trends', 'Leaderboard']);
    rerender(<ViewSwitch scope="key" view="sessions" onChange={vi.fn()} trends leaderboard />);
    expect(tabs()).toEqual(['Requests', 'Sessions', 'Trends']);
    expect(screen.queryByRole('tab', { name: 'Leaderboard' })).toBeNull();
    rerender(<ViewSwitch scope="admin" view="requests" onChange={vi.fn()} />);
    expect(tabs()).toEqual(['Requests', 'Sessions']);
    expect(viewsFor({ scope: 'key', leaderboard: true })).not.toContain('leaderboard');
  });
});
