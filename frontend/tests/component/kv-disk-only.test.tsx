/** The Disk only option of the KV cache store (kv_store "nvme"). */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';

import { KvStoreCard } from '@/components/kv/kv-store-section';
import { ModelCard } from '@/components/models/model-card';
import { KV_STORE_LABEL } from '@/lib/kv';
import { formatSettingValue } from '@/lib/model-settings';

afterEach(cleanup);

const card = (over: Partial<React.ComponentProps<typeof KvStoreCard>> = {}) => {
  const onStore = vi.fn();
  render(
    <KvStoreCard backend="vllm" loaded={false} kvStore="off" nvmeDir="/kv" cacheIsolation={undefined} onStore={onStore} {...over} />,
  );
  return onStore;
};

describe('Disk only option', () => {
  it('shows four options and choosing Disk only calls onStore("nvme")', () => {
    const onStore = card();
    expect(screen.getAllByRole('radio')).toHaveLength(4);
    expect(screen.getByText('Disk only')).toBeInTheDocument();
    expect(
      screen.getByText('Writes them to disk and keeps about 1 GiB of RAM to move them. Survives a restart. For hosts with little spare RAM.'),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole('radio', { name: /Disk only/ }));
    expect(onStore).toHaveBeenCalledWith('nvme');
  });

  it('with no disk folder both disk options are disabled and the note names both', () => {
    card({ nvmeDir: '' });
    expect(screen.getByRole('radio', { name: /RAM \+ disk/ })).toBeDisabled();
    expect(screen.getByRole('radio', { name: /Disk only/ })).toBeDisabled();
    expect(screen.getByText(/No disk folder is set, so RAM \+ disk and Disk only are not offered\./)).toBeInTheDocument();
  });

  it('the summary reads "disk only"', () => {
    card({ kvStore: 'nvme' });
    expect(screen.getByText('disk only')).toBeInTheDocument();
    expect(KV_STORE_LABEL.nvme).toBe('disk only');
    expect(formatSettingValue('kv_store', 'nvme')).toBe('Disk only');
  });

  it('the model card reads "KV cache store: disk only"', () => {
    const model = { id: 'm', served_model_name: 'x', status: 'stopped', gpu_indices: [], kv_store: 'nvme' };
    render(<ModelCard model={model as unknown as React.ComponentProps<typeof ModelCard>['model']} />);
    expect(screen.getByText(/KV cache store:/).textContent).toContain('disk only');
  });
});
