import { describe, it, expect } from 'vitest';
import {
  planLayout,
  layoutInvalidReason,
  gpuMemoryReadout,
} from '@/lib/settings-layout';

// ---------------------------------------------------------------------------
// Pure helpers behind the settings page's layout picture and GiB readout
// (plan 2026-10-04-settings-redesign §4.3, §4.6). Replica r gets
// gpu_indices.slice(r*tp, r*tp+tp) — the order vLLM assigns ranks over
// CUDA_VISIBLE_DEVICES.
// ---------------------------------------------------------------------------

describe('planLayout (vLLM)', () => {
  it('4 GPUs, dp 4, tp 1 → four single-GPU replicas', () => {
    const p = planLayout({ gpuIndices: [0, 1, 2, 3], dp: 4, tp: 1 });
    expect(p.valid).toBe(true);
    expect(p.replicas).toEqual([[0], [1], [2], [3]]);
    expect(p.sentence).toBe('4 GPUs = 4 replicas × 1 GPU each (tensor-parallel 1)');
  });

  it('4 GPUs, dp 2, tp 2 → two pairs', () => {
    const p = planLayout({ gpuIndices: [0, 1, 2, 3], dp: 2, tp: 2 });
    expect(p.replicas).toEqual([[0, 1], [2, 3]]);
    expect(p.sentence).toBe('4 GPUs = 2 replicas × 2 GPUs each (tensor-parallel 2)');
  });

  it('6 GPUs, dp 3, tp 2 keeps the selected indices in order', () => {
    const p = planLayout({ gpuIndices: [1, 2, 4, 5, 6, 7], dp: 3, tp: 2 });
    expect(p.replicas).toEqual([[1, 2], [4, 5], [6, 7]]);
  });

  it('dp 1 is "one replica"', () => {
    const p = planLayout({ gpuIndices: [0, 1], dp: 1, tp: 2 });
    expect(p.valid).toBe(true);
    expect(p.replicas).toEqual([[0, 1]]);
    expect(p.sentence).toBe('2 GPUs = 1 replica × 2 GPUs each (tensor-parallel 2)');
  });

  it('a single GPU reads in the singular', () => {
    expect(planLayout({ gpuIndices: [3], dp: 1, tp: 1 }).sentence).toBe(
      '1 GPU = 1 replica × 1 GPU each (tensor-parallel 1)',
    );
  });

  it('replicas that do not divide the GPUs are invalid, all GPUs unassigned', () => {
    const p = planLayout({ gpuIndices: [0, 1, 2, 3], dp: 3, tp: 1 });
    expect(p.valid).toBe(false);
    expect(p.replicas).toEqual([]);
    expect(p.unassigned).toEqual([0, 1, 2, 3]);
    expect(p.sentence).toBe("4 GPUs can't split into 3 equal replicas");
  });

  it('a tensor-parallel size that disagrees with GPUs ÷ replicas is flagged, not drawn', () => {
    const p = planLayout({ gpuIndices: [0, 1, 2, 3], dp: 2, tp: 1 });
    expect(p.valid).toBe(false);
    expect(p.unassigned).toEqual([0, 1, 2, 3]);
    expect(p.sentence).toBe(
      '4 GPUs ≠ 2 replicas × 1 GPU each (tensor-parallel 1); tensor-parallel should be 2',
    );
  });

  it('no GPUs is invalid', () => {
    const p = planLayout({ gpuIndices: [], dp: 1, tp: 1 });
    expect(p.valid).toBe(false);
    expect(p.sentence).toBe('No GPUs selected');
  });

  it('never uses the word "parallelism" (a llama.cpp test asserts no such label)', () => {
    for (const [g, dp, tp] of [[[0, 1, 2, 3], 2, 2], [[0, 1, 2, 3], 3, 1], [[], 1, 1]] as const) {
      expect(planLayout({ gpuIndices: [...g], dp, tp }).sentence).not.toMatch(/parallelism/i);
    }
  });
});

describe('planLayout (llama.cpp)', () => {
  it('one box, layers split across the GPUs, no replicas/TP wording', () => {
    const p = planLayout({ gpuIndices: [0, 1, 2], dp: 4, tp: 4, backend: 'llamacpp' });
    expect(p.valid).toBe(true);
    expect(p.replicas).toEqual([[0, 1, 2]]);
    expect(p.sentence).toBe('3 GPUs · layers split across them');
    expect(p.sentence).not.toMatch(/replica|tensor|parallelism|GPU layers/i);
  });

  it('single GPU llama.cpp', () => {
    expect(planLayout({ gpuIndices: [0], dp: 1, tp: 1, backend: 'llamacpp' }).sentence).toBe(
      '1 GPU · all layers on it',
    );
  });
});

describe('layoutInvalidReason', () => {
  it('null when valid', () => {
    expect(layoutInvalidReason([0, 1, 2, 3], 2)).toBeNull();
  });
  it('empty GPU set', () => {
    expect(layoutInvalidReason([], 1)).toBe('select at least one GPU');
  });
  it('dp does not divide', () => {
    expect(layoutInvalidReason([0, 1, 2, 3], 3)).toBe(
      'replicas must divide the 4 selected GPUs',
    );
  });
  it('tp × dp must equal the GPU count', () => {
    expect(layoutInvalidReason([0, 1], 1, 'vllm', 1)).toBe(
      'tensor-parallel size × replicas must equal the 2 selected GPUs',
    );
    expect(layoutInvalidReason([0, 1], 1, 'vllm', 2)).toBeNull();
    expect(layoutInvalidReason([0, 1], 1, 'llamacpp', 1)).toBeNull();
  });
  it('llama.cpp ignores dp', () => {
    expect(layoutInvalidReason([0, 1, 2], 2, 'llamacpp')).toBeNull();
  });
});

describe('gpuMemoryReadout', () => {
  const gpus = [
    { index: 0, memory_total_mib: 16376 },
    { index: 1, memory_total_mib: 16376 },
    { index: 2, memory_total_mib: 81559 },
  ];

  it('util × the smallest selected GPU, in GiB', () => {
    const r = gpuMemoryReadout(0.9, [0, 1], gpus);
    expect(r).not.toBeNull();
    expect(r!.usedGib).toBe('14.4');
    expect(r!.totalGib).toBe('16');
    expect(r!.text).toBe('≈ 14.4 GiB of 16 GiB per GPU');
  });

  it('uses the minimum across a mixed selection', () => {
    expect(gpuMemoryReadout(0.5, [0, 2], gpus)!.usedGib).toBe('8.0');
  });

  it('null when the probe has no data for the selection', () => {
    expect(gpuMemoryReadout(0.9, [5], gpus)).toBeNull();
    expect(gpuMemoryReadout(0.9, [], gpus)).toBeNull();
    expect(gpuMemoryReadout(0.9, [0], undefined)).toBeNull();
  });

  it('null when util is missing or out of range', () => {
    expect(gpuMemoryReadout(null, [0], gpus)).toBeNull();
    expect(gpuMemoryReadout(0, [0], gpus)).toBeNull();
    expect(gpuMemoryReadout(Number.NaN, [0], gpus)).toBeNull();
  });

  it('ignores GPUs whose total is unknown', () => {
    const r = gpuMemoryReadout(0.9, [0, 1], [
      { index: 0, memory_total_mib: 16376 },
      { index: 1, memory_total_mib: null },
    ]);
    expect(r!.usedGib).toBe('14.4');
  });
});
