/**
 * The admin's per-key cache controls on the token details page (plan 2, task 14):
 * sharing both ways with a warning before save, caps, reset with a confirmation,
 * the store use with "dropped by", the isolation cost and the models an isolated
 * key is refused on.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';

import { TokenPromptCacheCard } from '@/components/kv/token-prompt-cache-card';
import { TokenStoreUse } from '@/components/kv/token-store-use';
import { CacheUseCard } from '@/components/kv/cache-use-card';
import { ADMIN_STORE, KV_USAGE, TOKEN_CACHE, WINDOWS } from './kv-fixtures';

const MODELS = [
  { id: 'qwen3-235b-a22b', cache_isolation: 'isolated' as const },
  { id: 'gemma-3-27b', cache_isolation: 'unsupported' as const },
];

function card(over: Partial<Parameters<typeof TokenPromptCacheCard>[0]> = {}) {
  const onSave = vi.fn(async () => null as string | null);
  const onReset = vi.fn(async () => null as string | null);
  render(
    <TokenPromptCacheCard
      token={{ id: 'tok1', kv_isolation: 'sharing', kv_max_pct_ram: null, kv_max_pct_nvme: 40 }}
      policy="lru"
      models={MODELS}
      onSave={onSave}
      onReset={onReset}
      {...over}
    />,
  );
  return { onSave, onReset };
}

afterEach(() => cleanup());

describe('token prompt cache card', () => {
  it('warns before save, and saves only on Save', async () => {
    const { onSave } = card();
    expect(screen.queryByText(/next turn computes its prompt again/)).toBeNull();
    fireEvent.click(screen.getByRole('radio', { name: 'Isolated' }));
    expect(screen.getByText("Before you save: this key's next turn computes its prompt again.")).toBeInTheDocument();
    expect(screen.getByText('What it cached while shared stays shared and ages out as usual.')).toBeInTheDocument();
    expect(screen.getByText('Unsaved')).toBeInTheDocument();
    expect(onSave).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Save changes to the prompt cache' }));
    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ kv_isolation: 'isolated' }));
  });

  it('asks with a dialog before switching an isolated key back to shared', async () => {
    const { onSave } = card({
      token: { id: 'tok1', kv_isolation: 'isolated', kv_max_pct_ram: null, kv_max_pct_nvme: null },
    });
    fireEvent.click(screen.getByRole('radio', { name: 'Shared' }));
    expect(screen.getByText('Its own cached parts in RAM and on disk are dropped first.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Save changes to the prompt cache' }));
    const dlg = await screen.findByRole('dialog');
    expect(within(dlg).getByText('Switch this key back to shared?')).toBeInTheDocument();
    expect(onSave).not.toHaveBeenCalled();
    fireEvent.click(within(dlg).getByRole('button', { name: 'Switch to shared' }));
    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ kv_isolation: 'sharing' }));
  });

  it('reset asks for confirmation first', async () => {
    const { onReset } = card({
      token: { id: 'tok1', kv_isolation: 'isolated', kv_max_pct_ram: null, kv_max_pct_nvme: null },
    });
    fireEvent.click(screen.getByRole('button', { name: "Reset this key's cache" }));
    const dlg = await screen.findByRole('dialog');
    expect(within(dlg).getByText("Reset this key's cache?")).toBeInTheDocument();
    expect(onReset).not.toHaveBeenCalled();
    fireEvent.click(within(dlg).getByRole('button', { name: 'Cancel' }));
    expect(onReset).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: "Reset this key's cache" }));
    fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Reset cache' }));
    await waitFor(() => expect(onReset).toHaveBeenCalledTimes(1));
  });

  it('a shared key has nothing of its own to reset', () => {
    card();
    expect(screen.getByRole('button', { name: "Reset this key's cache" })).toBeDisabled();
    expect(screen.getByText('Only an isolated key has a cache of its own to reset.')).toBeInTheDocument();
  });

  it('caps are disabled with a note under oldest first, editable under fair share', async () => {
    card();
    expect(screen.getByLabelText('Most of the RAM tier')).toBeDisabled();
    expect(screen.getByText(/caps work only with the fair eviction policy/)).toBeInTheDocument();
    cleanup();
    const { onSave } = card({ policy: 'fair' });
    const ram = screen.getByLabelText('Most of the RAM tier');
    expect(ram).toBeEnabled();
    expect(screen.queryByText(/caps work only with the fair eviction policy/)).toBeNull();
    fireEvent.change(ram, { target: { value: '30' } });
    fireEvent.change(screen.getByLabelText('Most of the disk tier'), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save changes to the prompt cache' }));
    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ kv_max_pct_ram: 30, kv_max_pct_nvme: null }));
  });

  it("an isolated key's models list marks the models that cannot isolate", () => {
    card({ token: { id: 'tok1', kv_isolation: 'isolated', kv_max_pct_ram: null, kv_max_pct_nvme: null } });
    const list = screen.getByTestId('kv-model-isolation');
    const rows = within(list).getAllByRole('listitem');
    expect(rows[0]).toHaveTextContent('qwen3-235b-a22b');
    expect(rows[0]).toHaveTextContent('cache kept separate');
    expect(rows[1]).toHaveTextContent("gemma-3-27bcannot keep this key's cache separate: refused");
    cleanup();
    card();
    expect(screen.queryByTestId('kv-model-isolation')).toBeNull();
  });
});

describe('token cache figures', () => {
  it('cache use: windows from history, the selected window drives the headline', () => {
    render(<CacheUseCard windows={WINDOWS} />);
    expect(screen.getByTestId('kv-hit-headline')).toHaveTextContent('64 %');
    fireEvent.click(screen.getByRole('button', { name: '7d' }));
    expect(screen.getByTestId('kv-hit-headline')).toHaveTextContent('74 %');
    expect(screen.getByText('402k')).toBeInTheDocument();
    expect(screen.getByText('44.9M')).toBeInTheDocument();
  });

  it('store use is labelled since the store started and lists who dropped the key\'s parts', () => {
    render(<TokenStoreUse report={TOKEN_CACHE} usage={KV_USAGE} storeOn />);
    expect(screen.getByText(/since the store started, 3 d ago/)).toBeInTheDocument();
    expect(screen.getByText('3.4 GiB')).toBeInTheDocument();
    expect(screen.getByText('of a 2.9 GiB share')).toBeInTheDocument();
    const by = screen.getByTestId('kv-evicted-by');
    expect(within(by).getByText('nightly-evals')).toBeInTheDocument();
    expect(within(by).getByText('301')).toBeInTheDocument();
    expect(screen.getByTestId('kv-isolation-cost')).toHaveTextContent('1.24M');
  });

  it('a shared key gets no isolation cost; a store that does not answer says so', () => {
    render(
      <TokenStoreUse report={{ ...TOKEN_CACHE, isolation: 'sharing', store: null }} usage={{ ...KV_USAGE, available: false }} storeOn />,
    );
    expect(screen.queryByTestId('kv-isolation-cost')).toBeNull();
    expect(screen.getByText(/The KV cache store is not answering, so its use is unavailable/)).toBeInTheDocument();
  });

  it('no model with the store on: one line, no figures', () => {
    render(<TokenStoreUse report={{ ...TOKEN_CACHE, store: { ...ADMIN_STORE } }} usage={KV_USAGE} storeOn={false} />);
    expect(screen.getByText(/No model has the KV cache store on/)).toBeInTheDocument();
    expect(screen.queryByTestId('kv-evicted-by')).toBeNull();
  });
});
