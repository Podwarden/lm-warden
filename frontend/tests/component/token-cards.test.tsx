import { describe, it, expect, vi, afterEach } from 'vitest';
process.env.TZ = 'UTC';
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react';
import { LimitsCard } from '@/components/tokens/detail/limits-card';
import { DetailsCard } from '@/components/tokens/detail/details-card';
import { LineageCard, lineageMeta } from '@/components/tokens/detail/lineage-card';
import { NOW, tokenDetail } from './token-fixtures';

describe('LimitsCard', () => {
  afterEach(() => cleanup());

  it('is clean until priority changes, and saves priority only, in ONE call', async () => {
    const onSave = vi.fn().mockResolvedValue(null);
    render(<LimitsCard token={{ id: 't', priority: 9 }} onSave={onSave} />);
    const save = screen.getByRole('button', { name: 'Save changes' });
    expect(save).toBeDisabled();
    expect(screen.queryByText('Unsaved')).toBeNull();
    expect(screen.getByRole('button', { name: 'P9' })).toHaveAttribute('aria-pressed', 'true');

    fireEvent.click(screen.getByRole('button', { name: 'P5' }));
    expect(screen.getByText('Unsaved')).toBeInTheDocument();
    expect(screen.getByText('Priority').textContent).toBe('Priority 5');
    expect(save).toBeEnabled();

    await act(async () => { fireEvent.click(save); });
    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledWith({ priority: 5 });
  });

  // A poll that brings a new saved priority (changed in another tab, say):
  // a clean form follows it; a form with an edit in progress keeps the edit.
  it('a clean form adopts a polled priority', () => {
    const { rerender } = render(<LimitsCard token={{ id: 't', priority: 5 }} onSave={vi.fn()} />);
    rerender(<LimitsCard token={{ id: 't', priority: 7 }} onSave={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'P7' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('Priority').textContent).toBe('Priority 7');
    expect(screen.getByRole('button', { name: 'Save changes' })).toBeDisabled();
    expect(screen.queryByText('Unsaved')).toBeNull();
  });

  it('a dirty form keeps the edit when a poll brings a new priority', () => {
    const { rerender } = render(<LimitsCard token={{ id: 't', priority: 5 }} onSave={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'P2' }));
    rerender(<LimitsCard token={{ id: 't', priority: 7 }} onSave={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'P2' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('Unsaved')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save changes' })).toBeEnabled();

    // Going back to the NEW saved value makes it clean, and a later poll is adopted again.
    fireEvent.click(screen.getByRole('button', { name: 'P7' }));
    expect(screen.getByRole('button', { name: 'Save changes' })).toBeDisabled();
    rerender(<LimitsCard token={{ id: 't', priority: 9 }} onSave={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'P9' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('has no rate limit field any more', () => {
    render(<LimitsCard token={{ id: 't', priority: 9 }} onSave={vi.fn()} />);
    expect(screen.queryByRole('textbox')).toBeNull();
    expect(screen.queryByText(/rate limit/i)).toBeNull();
    expect(screen.queryByText('tokens / s')).toBeNull();
    expect(screen.queryByText(/Leave empty for no limit/)).toBeNull();
  });

  it('going back to the saved values makes it clean again', () => {
    render(<LimitsCard token={{ id: 't', priority: 9 }} onSave={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'P2' }));
    fireEvent.click(screen.getByRole('button', { name: 'P9' }));
    expect(screen.getByRole('button', { name: 'Save changes' })).toBeDisabled();
  });

  it('shows the server error next to the button', async () => {
    render(<LimitsCard token={{ id: 't', priority: 9 }} onSave={vi.fn().mockResolvedValue('priority out of range')} />);
    fireEvent.click(screen.getByRole('button', { name: 'P1' }));
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Save changes' })); });
    expect(screen.getByRole('alert')).toHaveTextContent('priority out of range');
  });

  it('carries the mockup copy', () => {
    render(<LimitsCard token={{ id: 't', priority: 9 }} onSave={vi.fn()} />);
    for (const s of ['Limits', 'served last', 'served first',
      'Strict: a waiting P9 request always goes before a P8 one. Low priorities can wait indefinitely on a busy box.']) {
      expect(screen.getByText(s)).toBeInTheDocument();
    }
  });
});

describe('DetailsCard', () => {
  afterEach(() => cleanup());
  it('shows last used, created, expires and the 24h line', () => {
    render(<DetailsCard token={tokenDetail()} nowSec={NOW} />);
    expect(screen.getByText('2 min ago')).toBeInTheDocument();
    expect(screen.getByText('Sep 6, 13:59')).toBeInTheDocument();
    expect(screen.getByText('Never')).toBeInTheDocument();
    expect(screen.getByText('1.5k req · 110M tok')).toBeInTheDocument();
  });
});

describe('LineageCard', () => {
  afterEach(() => cleanup());
  it('renders the mockup timeline, linking earlier keys', () => {
    render(<LineageCard lineage={tokenDetail().lineage} nowSec={NOW} />);
    expect(screen.getByText('Rotation history')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'opencode-ip-macbook (old 1)' })).toHaveAttribute('href', '/tokens/tok-old1');
    expect(screen.getByText('Created Sep 5 · rotated Sep 5, 13:55 · cut off')).toBeInTheDocument();
    expect(screen.getByText('This token')).toBeInTheDocument();
    expect(screen.getByText('Created Sep 6, 13:59 · in use')).toBeInTheDocument();
  });
  it('says "in grace" for a predecessor still in its window', () => {
    expect(lineageMeta({ id: 'a', name: 'a', created_at: '2026-09-05 12:16:00', rotated_at: '2026-09-18 10:00:00',
      is_revoked: false, in_grace: true, is_self: false }, NOW)).toBe('Created Sep 5 · rotated Sep 18, 10:00 · in grace');
  });
  it('is not rendered without earlier keys', () => {
    const { container } = render(<LineageCard lineage={[tokenDetail().lineage[2]]} nowSec={NOW} />);
    expect(container).toBeEmptyDOMElement();
  });
});
