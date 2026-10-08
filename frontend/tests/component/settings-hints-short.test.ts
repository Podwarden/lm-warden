import { describe, it, expect } from 'vitest';
import { MODEL_HINTS } from '@/lib/settings-hints';

describe('MODEL_HINTS short hints', () => {
  it('extra_env short hint is right on both backends', () => {
    // Review minor: it listed only the vLLM prefixes, which is wrong on a
    // llama.cpp row (GGML_ only).
    const short = MODEL_HINTS.extra_env.short!;
    expect(short).toMatch(/VLLM_/);
    expect(short).toMatch(/GGML_/);
    expect(short.length).toBeLessThanOrEqual(90);
  });

});
