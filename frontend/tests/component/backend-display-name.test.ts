import { describe, it, expect } from 'vitest';

// ---------------------------------------------------------------------------
// The model detail page names the engine with `backendDisplayName`.
//
// The wire value is the backend's machine name ('mlx'), but the UI has always
// spelled the house names vLLM / llama.cpp / MLX (live-panels.tsx). The Mac
// backend was added after that table was written, so the detail page read
// "Engine: mlx". This file pins the whole leg set of the lookup so the table
// and the wire values cannot drift apart again.
//
// `BACKEND_LABELS` is a NEW export: import it inside the test that needs it so
// the file still loads on a tree that only has `backendDisplayName`.
// ---------------------------------------------------------------------------

describe('backendDisplayName', () => {
  it('names the Mac backend MLX, not the wire value mlx', async () => {
    const { backendDisplayName } = await import('@/lib/backend-fields');
    expect(backendDisplayName('mlx')).toBe('MLX');
  });

  it('keeps the existing vLLM spelling', async () => {
    const { backendDisplayName } = await import('@/lib/backend-fields');
    expect(backendDisplayName('vllm')).toBe('vLLM');
  });

  it('keeps the existing llama.cpp spelling', async () => {
    const { backendDisplayName } = await import('@/lib/backend-fields');
    expect(backendDisplayName('llamacpp')).toBe('llama.cpp');
  });

  it('falls back to vLLM for null, undefined and empty (decision D6)', async () => {
    const { backendDisplayName } = await import('@/lib/backend-fields');
    expect(backendDisplayName(null)).toBe('vLLM');
    expect(backendDisplayName(undefined)).toBe('vLLM');
    expect(backendDisplayName('')).toBe('vLLM');
  });

  it('passes an unknown value through unchanged', async () => {
    const { backendDisplayName } = await import('@/lib/backend-fields');
    expect(backendDisplayName('sglang')).toBe('sglang');
  });

  it('is exact-case: no case folding', async () => {
    const { backendDisplayName } = await import('@/lib/backend-fields');
    expect(backendDisplayName('MLX')).toBe('MLX');
    expect(backendDisplayName('Mlx')).toBe('Mlx');
    expect(backendDisplayName('VLLM')).toBe('VLLM');
  });

  it('does not let inherited object keys masquerade as backends', async () => {
    const { backendDisplayName } = await import('@/lib/backend-fields');
    expect(backendDisplayName('toString')).toBe('toString');
    expect(backendDisplayName('constructor')).toBe('constructor');
    expect(backendDisplayName('hasOwnProperty')).toBe('hasOwnProperty');
  });

  it('exposes the lookup table under exactly the three known wire values', async () => {
    const { BACKEND_LABELS } = await import('@/lib/backend-fields');
    expect(Object.keys(BACKEND_LABELS).sort()).toEqual(['llamacpp', 'mlx', 'vllm']);
  });
});
