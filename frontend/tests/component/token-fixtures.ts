// Shared fixtures for the token details tests: the mockup's
// "opencode-laptop" key after two rotations, at the mockup's NOW.
import type { ModelUsage, ModelVariantUsage, TokenDetail, TokenSeries } from '@/lib/token-series';

export const NOW = Date.UTC(2026, 8, 18, 13, 21) / 1000;

export function tokenDetail(o: Partial<TokenDetail> = {}): TokenDetail {
  return {
    id: 'tok-self', name: 'opencode-laptop', prefix: 'vw_6gqfa', preview: 'vw_6gqfa',
    created_at: '2026-09-06 13:59:00', last_used_at: '2026-09-18 13:19:00', expires_at: null,
    rotated_at: null, rotated_from: 'tok-old2', successor_id: null, successor_deleted: false,
    is_expired: false, is_near_expiry: false, revoked_at: null, is_revoked: false,
    paused_at: null, is_paused: false, priority: 9,
    usage_24h: { requests: 1500, prompt_tokens: 108_000_000, completion_tokens: 2_000_000, total_tokens: 110_000_000 },
    lineage: [
      { id: 'tok-old1', name: 'opencode-laptop (old 1)', created_at: '2026-09-05 12:16:00', rotated_at: '2026-09-05 13:55:00', is_revoked: true, in_grace: false, is_self: false },
      { id: 'tok-old2', name: 'opencode-laptop (old 2)', created_at: '2026-09-05 13:55:00', rotated_at: '2026-09-06 13:59:00', is_revoked: true, in_grace: false, is_self: false },
      { id: 'tok-self', name: 'opencode-laptop', created_at: '2026-09-06 13:59:00', rotated_at: null, is_revoked: false, in_grace: false, is_self: true },
    ],
    ...o,
  };
}

export function seriesFixture(o: Partial<TokenSeries> = {}): TokenSeries {
  const to = Math.floor(NOW / 60);
  const from = to - 10080;
  return {
    token_ids: ['tok-old1', 'tok-old2', 'tok-self'],
    from_minute: from, to_minute: to, bin_minutes: 30,
    latency_since: Date.UTC(2026, 8, 18, 0, 0) / 1000,
    timing_sample: { total: 24, used: 24, stride: 1 },
    totals: { requests: 4200, prompt_tokens: 324_000_000, completion_tokens: 5_000_000 },
    bins: [
      { minute: Math.floor(from / 30) * 30 + 30, requests_per_min: 1.2, prompt_per_min: 95_000, completion_per_min: 1_000,
        peak_prompt: 210_000, peak_completion: 3_000, requests: 36, n: 0, queue_p50: null, queue_p95: null,
        ttft_p50: null, ttft_p95: null, duration_p50: null, duration_p95: null },
      { minute: Math.floor((to - 60) / 30) * 30, requests_per_min: 0.8, prompt_per_min: 60_000, completion_per_min: 900,
        peak_prompt: 120_000, peak_completion: 2_000, requests: 24, n: 24, queue_p50: 0.2, queue_p95: 4.1,
        ttft_p50: 1.1, ttft_p95: 3.9, duration_p50: 22, duration_p95: 96 },
    ],
    by_model_since: null,
    by_model: [],
    model_bins: [],
    ...o,
  };
}

export function variantUsage(o: Partial<ModelVariantUsage> = {}): ModelVariantUsage {
  return {
    variant_id: 'a1b2c3d4e5f60718', backend: 'vllm', engine_channel: 'stable',
    engine_vllm_version: '0.11.0', engine_version: null, engine_image_tag: 'v0.11.0', quantization: 'awq',
    dtype: 'auto', hf_revision: 'a1b2c3d4e5f6', hf_commit: null, max_model_len: 32768,
    first_seen: Date.UTC(2026, 8, 17, 9, 0) / 1000,
    requests: 10, prompt_tokens: 1000, completion_tokens: 100, total_tokens: 1100, share: 1,
    ...o,
  };
}

export function modelUsage(o: Partial<ModelUsage> = {}): ModelUsage {
  const base: ModelUsage = {
    model_id: 'id-qwen', model: 'qwen3-8b', requests: 10, prompt_tokens: 1000,
    completion_tokens: 100, total_tokens: 1100, share: 1, variants: [variantUsage()],
  };
  return { ...base, ...o };
}

/** Two models over the fixture window, per-model data from Sep 15 00:00 on
 *  (the window starts Sep 11, so it starts before the breakdown does). */
export function splitSeriesFixture(o: Partial<TokenSeries> = {}): TokenSeries {
  const s = seriesFixture();
  const [early, late] = s.bins;
  return seriesFixture({
    by_model_since: Date.UTC(2026, 8, 15, 0, 0) / 1000,
    by_model: [
      modelUsage({ model_id: 'id-qwen', model: 'qwen3-8b', requests: 20, prompt_tokens: 1_200_000,
        completion_tokens: 20_000, total_tokens: 1_220_000, share: 0.8,
        variants: [
          variantUsage({ variant_id: 'aaaaaaaaaaaaaaaa', requests: 15, prompt_tokens: 1_000_000, completion_tokens: 15_000, total_tokens: 1_015_000, share: 0.66 }),
          variantUsage({ variant_id: 'bbbbbbbbbbbbbbbb', engine_vllm_version: '0.9.0', engine_image_tag: 'v0.9.0', quantization: 'fp8', hf_revision: 'main', max_model_len: 8192, requests: 5, prompt_tokens: 200_000, completion_tokens: 5_000, total_tokens: 205_000, share: 0.14 }),
        ] }),
      modelUsage({ model_id: 'id-llama', model: 'llama-3.1-8b', requests: 4, prompt_tokens: 300_000,
        completion_tokens: 5_000, total_tokens: 305_000, share: 0.2,
        variants: [variantUsage({ variant_id: 'cccccccccccccccc', backend: 'llamacpp', engine_vllm_version: null, engine_image_tag: null, quantization: null, dtype: null, hf_revision: 'main', max_model_len: 4096 })] }),
    ],
    model_bins: [
      { minute: late.minute, models: {
        'id-qwen': { prompt_per_min: 45_000, completion_per_min: 700 },
        'id-llama': { prompt_per_min: 15_000, completion_per_min: 200 },
      } },
    ],
    bins: [early, late],
    ...o,
  });
}
