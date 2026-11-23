/**
 * Task 14 review, fix round 1: wrapping marks, the weight clamped on blur, named and
 * keyboard-operable radio groups, no aria-selected on plain rows, the share bar's text,
 * and the admin's model list narrowed to the key's allowed models.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';

import { Mark, ModelIsolationList } from '@/components/kv/parts';
import { KvWeightCard } from '@/components/kv/kv-store-section';
import { TokenPromptCacheCard } from '@/components/kv/token-prompt-cache-card';
import { CacheUseCard } from '@/components/kv/cache-use-card';
import { KvUsagePanel } from '@/components/kv/kv-usage-panel';
import { modelsForKey } from '@/lib/kv';
import { KV_USAGE, WINDOWS } from './kv-fixtures';

afterEach(() => cleanup());

describe('marks wrap where they must', () => {
  it("the models list's refusal mark wraps instead of running off a phone screen", () => {
    render(<ModelIsolationList models={[{ id: 'gemma-3-27b', cache_isolation: 'unsupported' }]} isolated refusedOnly />);
    const mark = screen.getByText("cannot keep this key's cache separate: refused");
    expect(mark.className).toContain('whitespace-normal');
    expect(mark.className).not.toContain('whitespace-nowrap');
  });

  it('a plain mark stays on one line', () => {
    render(<Mark tone="info">isolated</Mark>);
    expect(screen.getByText('isolated').className).toContain('whitespace-nowrap');
  });
});

describe('weight field', () => {
  it('clamps on blur, not per keystroke', () => {
    const onWeight = vi.fn();
    render(<KvWeightCard weight={3} part={null} kvStore="ram" policy="fair" onWeight={onWeight} />);
    const input = screen.getByLabelText('Weight') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '' } });
    expect(input.value).toBe('');
    expect(onWeight).not.toHaveBeenCalled();
    fireEvent.change(input, { target: { value: '250' } });
    expect(onWeight).not.toHaveBeenCalledWith(100);
    fireEvent.blur(input);
    expect(onWeight).toHaveBeenLastCalledWith(100);
    expect(input.value).toBe('100');
    fireEvent.change(input, { target: { value: '' } });
    fireEvent.blur(input);
    expect(onWeight).toHaveBeenLastCalledWith(1);
  });

  it('a valid value goes straight through', () => {
    const onWeight = vi.fn();
    render(<KvWeightCard weight={3} part={null} kvStore="ram" policy="fair" onWeight={onWeight} />);
    fireEvent.change(screen.getByLabelText('Weight'), { target: { value: '7' } });
    expect(onWeight).toHaveBeenCalledWith(7);
  });
});

describe('admin Shared / Isolated control', () => {
  it('is a radiogroup the arrow keys move through, with one tab stop', () => {
    render(
      <TokenPromptCacheCard
        token={{ id: 't', kv_isolation: 'sharing', kv_max_pct_ram: null, kv_max_pct_nvme: null }}
        policy="lru"
        models={[]}
        onSave={async () => null}
        onReset={async () => null}
      />,
    );
    const group = screen.getByRole('radiogroup', { name: 'Sharing' });
    const [shared, isolated] = within(group).getAllByRole('radio');
    expect(shared.tabIndex).toBe(0);
    expect(isolated.tabIndex).toBe(-1);
    shared.focus();
    fireEvent.keyDown(shared, { key: 'ArrowRight' });
    expect(isolated).toHaveAttribute('aria-checked', 'true');
    expect(document.activeElement).toBe(isolated);
    expect(isolated.tabIndex).toBe(0);
    fireEvent.keyDown(isolated, { key: 'ArrowLeft' });
    expect(shared).toHaveAttribute('aria-checked', 'true');
  });
});

describe('cache use table', () => {
  it('marks the selected window with its button, not aria-selected on a row', () => {
    const { container } = render(<CacheUseCard windows={WINDOWS} />);
    expect(container.querySelector('tr[aria-selected]')).toBeNull();
    expect(screen.getByRole('button', { name: '24h' })).toHaveAttribute('aria-pressed', 'true');
  });
});

describe('use against share bar', () => {
  it('says the use, the share and whether it is over', () => {
    render(<KvUsagePanel usage={KV_USAGE} models={{ 'm-qwen': { name: 'qwen3-235b-a22b' } }} storeOn />);
    const bars = screen.getAllByRole('img', { name: /share/ });
    expect(bars.length).toBeGreaterThan(0);
    expect(bars.some((b) => /over its share/.test(b.getAttribute('aria-label') ?? ''))).toBe(true);
  });
});

describe("the admin's list of a key's models", () => {
  const models = [
    { id: 'qwen', cache_isolation: 'isolated' as const },
    { id: 'gemma', cache_isolation: 'unsupported' as const },
  ];
  it('null allowed models: every loaded model', () => {
    expect(modelsForKey(models, null).map((m) => m.id)).toEqual(['qwen', 'gemma']);
    expect(modelsForKey(models, undefined).map((m) => m.id)).toEqual(['qwen', 'gemma']);
  });
  it('a restricted key: only its models', () => {
    expect(modelsForKey(models, ['gemma']).map((m) => m.id)).toEqual(['gemma']);
  });
});
