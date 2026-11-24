/** The KV cache store's pool and disk guards (Rulings R16, R17) in the UI. */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { SWRConfig } from 'swr';

import { ModelCard } from '@/components/models/model-card';
import { PromptCacheTab } from '@/components/settings/prompt-cache-tab';
import { setAccessToken } from '@/lib/auth-fetch';
import { KV_USAGE } from './kv-fixtures';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const POOL = 'RAM pool 8 GiB does not fit /dev/shm 2 GiB';
const DISK =
  'NVMe tier off: /tmp is on the root filesystem (overlay); mount a dedicated disk there, or set kv_store_nvme_allow_root_disk';
const NOTE = 'cap 250 GB clamped to 34.5 GB: the filesystem has 49.5 GB free and 15% is kept free';

type Model = React.ComponentProps<typeof ModelCard>['model'];
const model = (kv_store: string) =>
  ({ id: 'm', served_model_name: 'x', status: 'loaded', gpu_indices: [0], kv_store }) as unknown as Model;

describe('model card: store errors', () => {
  it('shows the pool error for a model with the store on', () => {
    render(<ModelCard model={model('ram')} kvErrors={{ pool_error: POOL, nvme_error: null }} />);
    expect(screen.getByTestId('model-kv-pool-error')).toHaveTextContent(`KV cache store off: ${POOL}`);
    expect(screen.queryByTestId('model-kv-nvme-error')).toBeNull();
  });

  it('shows the disk error for a model in a disk mode only', () => {
    render(<ModelCard model={model('nvme')} kvErrors={{ pool_error: null, nvme_error: DISK }} />);
    expect(screen.getByTestId('model-kv-nvme-error')).toHaveTextContent(DISK);
    cleanup();
    render(<ModelCard model={model('ram')} kvErrors={{ pool_error: null, nvme_error: DISK }} />);
    expect(screen.queryByTestId('model-kv-nvme-error')).toBeNull();
  });

  it('shows nothing for a model with the store off', () => {
    render(<ModelCard model={model('off')} kvErrors={{ pool_error: POOL, nvme_error: DISK }} />);
    expect(screen.queryByTestId('model-kv-pool-error')).toBeNull();
    expect(screen.queryByTestId('model-kv-nvme-error')).toBeNull();
  });
});

describe('settings: Prompt cache tab store errors', () => {
  beforeEach(() => setAccessToken('admin'));
  afterEach(() => setAccessToken(null));

  function tab(daemon: Record<string, unknown>) {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith('/api/settings/runtime'))
        return new Response(JSON.stringify({ kv_store_nvme_dir: '/tmp', kv_store_nvme_cap_gb: '250' }));
      if (url.startsWith('/api/system/kv-store'))
        return new Response(JSON.stringify({
          daemon: { healthy: true, generation: 1, ram_requested_bytes: 8 * 1024 ** 3, ram_granted_bytes: 0, ...daemon },
          models: {},
        }));
      if (url.startsWith('/api/kv/usage')) return new Response(JSON.stringify(KV_USAGE));
      return new Response('{}', { status: 404 });
    }));
    render(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <PromptCacheTab />
      </SWRConfig>,
    );
  }

  it('shows the disk error and the pool error', async () => {
    tab({ pool_error: POOL, nvme_error: DISK, nvme_cap_effective_bytes: 0, nvme_cap_note: null });
    expect(await screen.findByTestId('kv-nvme-error')).toHaveTextContent(DISK);
    expect(screen.getByTestId('kv-pool-error')).toHaveTextContent(POOL);
    expect(screen.queryByTestId('kv-nvme-cap-note')).toBeNull();
  });

  it('shows the cap note when the cap was clamped', async () => {
    tab({ pool_error: null, nvme_error: null, nvme_cap_effective_bytes: 34_500_000_000, nvme_cap_note: NOTE });
    expect(await screen.findByTestId('kv-nvme-cap-note')).toHaveTextContent(NOTE);
    expect(screen.queryByTestId('kv-nvme-error')).toBeNull();
    expect(screen.queryByTestId('kv-pool-error')).toBeNull();
  });
});
