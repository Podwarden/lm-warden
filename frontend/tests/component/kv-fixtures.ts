// Shared fixtures for the per-key KV cache UI tests (plan 2, task 14). Shapes follow
// src/lib/api-types.generated.ts: KvWindows, KvKeyReport, TokenCacheOut, KvUsage.
import type { components } from '@/lib/api-types.generated';

type S = components['schemas'];

const GiB = 1024 ** 3;
const GB = 1e9;

export const WINDOWS: S['KvWindows'] = {
  '1h': { gpu: 402_000, ram: 188_000, nvme: 0, recomputed: 311_000, hit_rate: 0.65 },
  '24h': { gpu: 7_000_000, ram: 2_800_000, nvme: 2_000_000, recomputed: 6_600_000, hit_rate: 0.64 },
  '7d': { gpu: 44_900_000, ram: 16_100_000, nvme: 14_200_000, recomputed: 27_300_000, hit_rate: 0.74 },
};

export const SINCE = new Date(Date.now() - 3 * 86_400_000).toISOString();

export const KEY_STORE: S['KvKeyStore'] = {
  ram_bytes: 3.4 * GiB,
  nvme_bytes: 52 * GB,
  ram_share: 0.18,
  nvme_share: 0.18,
  ram_evictions: 1208,
  nvme_evictions: 96,
  since: SINCE,
};

export const ADMIN_STORE: S['KvKeyStoreAdmin'] = {
  ...KEY_STORE,
  ram_share_bytes: 2.9 * GiB,
  nvme_share_bytes: 45 * GB,
  evicted_by: [
    { token_id: 't-nightly', name: 'nightly-evals', count: 301 },
    { token_id: 't-ci', name: 'ci-runner', count: 88 },
  ],
};

export const KEY_REPORT: S['KvKeyReport'] = { isolation: 'isolated', windows: WINDOWS, store: KEY_STORE };

export const TOKEN_CACHE: S['TokenCacheOut'] = {
  token_id: 'tok1',
  range: '24h',
  since_epoch: 0,
  models: [],
  isolation: 'isolated',
  windows: WINDOWS,
  store: ADMIN_STORE,
  isolation_cost_tokens: { '24h': 1_240_000 },
};

export const KV_USAGE: S['KvUsage'] = {
  available: true,
  policy: 'lru',
  since: SINCE,
  tiers: {
    ram: { capacity: 16 * GiB, used: 11.6 * GiB, dead: 0.9 * GiB },
    nvme: { capacity: 250 * GB, used: 231 * GB, dead: 13 * GB },
  },
  models: [
    {
      model_id: 'm-qwen',
      model_ids: ['m-qwen'],
      weight: 3,
      part: { ram: 0.75, nvme: 0.75 },
      used: { ram: 9.4 * GiB, nvme: 218 * GB, total: 9.4 * GiB + 218 * GB },
      tenants: [
        {
          token_id: 't-nightly',
          name: 'nightly-evals',
          used: { ram: 4 * GiB, nvme: 74 * GB, total: 4 * GiB + 74 * GB },
          share: { ram: 0.26, nvme: 0.26 },
          evictions: { ram: 31, nvme: 0 },
          evicted_by: [{ token_id: 't-ci', name: 'ci-runner', count: 31 }],
        },
        {
          token_id: 'tok1',
          name: 'dana-laptop',
          used: { ram: 2.1 * GiB, nvme: 38 * GB, total: 2.1 * GiB + 38 * GB },
          share: { ram: 0.26, nvme: 0.26 },
          evictions: { ram: 412, nvme: 0 },
          evicted_by: [{ token_id: 't-nightly', name: 'nightly-evals', count: 412 }],
        },
      ],
    },
    {
      model_id: 'm-coder',
      model_ids: ['m-coder'],
      weight: 1,
      part: { ram: 0.25, nvme: 0.25 },
      used: { ram: 1.3 * GiB, nvme: 0, total: 1.3 * GiB },
      tenants: [],
    },
  ],
};
