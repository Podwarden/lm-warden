// Pure helpers behind the model settings page's layout picture and the GPU
// memory readout (plan docs/superpowers/plans/2026-10-04-settings-redesign.md
// §4.3, §4.6). No React here, so the arithmetic is unit-tested on its own.
//
// Wording rule: never say "parallelism" or "GPU layers" in anything that can
// end up as an accessible name. The llama.cpp settings test asserts
// `queryByLabelText(/parallelism/i)` is null, and "GPU layers" is a real
// field label on llama.cpp rows.

export interface LayoutInput {
  gpuIndices: readonly number[];
  dp: number;
  tp: number;
  /** null/empty means vLLM (backend-fields.ts, decision D6). */
  backend?: string | null;
}

export interface LayoutPlan {
  valid: boolean;
  /** One entry per replica (one entry for llama.cpp), GPU indices in order. */
  replicas: number[][];
  /** GPUs that could not be placed (every selected GPU when invalid). */
  unassigned: number[];
  /** Plain-text equation; the diagram's accessible name. */
  sentence: string;
  /** Normalised numbers, for rendering the equation with emphasis. */
  gpuCount: number;
  dp: number;
  tp: number;
  llamacpp: boolean;
}

function plural(n: number, one: string, many: string): string {
  return `${n} ${n === 1 ? one : many}`;
}

function normInt(v: number): number {
  return Number.isFinite(v) ? Math.max(1, Math.floor(v)) : 1;
}

export function isLlamacpp(backend: string | null | undefined): boolean {
  return backend === "llamacpp";
}

/**
 * How the selected GPUs fall into replicas. Replica r gets
 * `gpuIndices.slice(r*tp, r*tp + tp)` — the order vLLM assigns ranks over
 * CUDA_VISIBLE_DEVICES. Valid only when tp × dp equals the GPU count, which
 * is what the backend enforces on save (#286).
 */
export function planLayout({ gpuIndices, dp: rawDp, tp: rawTp, backend }: LayoutInput): LayoutPlan {
  const gpus = [...gpuIndices];
  const n = gpus.length;
  const dp = normInt(rawDp);
  const tp = normInt(rawTp);
  const llamacpp = isLlamacpp(backend);
  const base = { gpuCount: n, dp, tp, llamacpp };

  if (n === 0) {
    return { ...base, valid: false, replicas: [], unassigned: [], sentence: "No GPUs selected" };
  }

  if (llamacpp) {
    return {
      ...base,
      valid: true,
      replicas: [gpus],
      unassigned: [],
      sentence: n === 1 ? "1 GPU · all layers on it" : `${n} GPUs · layers split across them`,
    };
  }

  if (n % dp !== 0) {
    return {
      ...base,
      valid: false,
      replicas: [],
      unassigned: gpus,
      sentence: `${plural(n, "GPU", "GPUs")} can't split into ${dp} equal replicas`,
    };
  }

  const equation =
    `${plural(dp, "replica", "replicas")} × ${plural(tp, "GPU", "GPUs")} each (tensor-parallel ${tp})`;

  if (tp * dp !== n) {
    return {
      ...base,
      valid: false,
      replicas: [],
      unassigned: gpus,
      sentence: `${plural(n, "GPU", "GPUs")} ≠ ${equation}; tensor-parallel should be ${n / dp}`,
    };
  }

  const replicas: number[][] = [];
  for (let r = 0; r < dp; r++) replicas.push(gpus.slice(r * tp, r * tp + tp));
  return {
    ...base,
    valid: true,
    replicas,
    unassigned: [],
    sentence: `${plural(n, "GPU", "GPUs")} = ${equation}`,
  };
}

/**
 * tensor_parallel_size × data_parallel_size must equal the GPU count (the
 * server 422s otherwise). Only meaningful once replicas divide the count and
 * never applies to llama.cpp, which has no TP/DP layout.
 */
export function tpDpMismatch(
  n: number,
  dp: number,
  tp: number,
  backend?: string | null,
): boolean {
  if (n === 0 || isLlamacpp(backend)) return false;
  const d = Math.max(1, dp);
  return n % d === 0 && tp * d !== n;
}

/**
 * The reason Save is blocked by the layout, phrased to follow "Can't save: ".
 * Mirrors the page's existing rules exactly (GPU set non-empty; replicas
 * divide the GPU count) — it adds no new validation.
 */
export function layoutInvalidReason(
  gpuIndices: readonly number[],
  dp: number,
  backend?: string | null,
  tp?: number,
): string | null {
  const n = gpuIndices.length;
  if (n === 0) return "select at least one GPU";
  if (isLlamacpp(backend)) return null;
  if (n % Math.max(1, dp) !== 0) return `replicas must divide the ${n} selected GPUs`;
  if (tp !== undefined && tpDpMismatch(n, dp, tp, backend)) {
    return `tensor-parallel size × replicas must equal the ${n} selected GPUs`;
  }
  return null;
}

export interface GpuMemoryInfo {
  index: number;
  memory_total_mib?: number | null;
}

export interface GpuMemoryReadout {
  /** util × smallest selected GPU, GiB, one decimal ("14.4"). */
  usedGib: string;
  /** Smallest selected GPU's total, GiB, at most one decimal ("16"). */
  totalGib: string;
  /** "≈ 14.4 GiB of 16 GiB per GPU" */
  text: string;
}

/**
 * The GiB readout under GPU memory utilization: `util × min(memory_total_mib)`
 * over the selected GPUs. vLLM applies the fraction per GPU, so the smallest
 * card is the binding one. Null when the probe has nothing for the selection.
 */
export function gpuMemoryReadout(
  util: number | null | undefined,
  selected: readonly number[],
  gpus: readonly GpuMemoryInfo[] | null | undefined,
): GpuMemoryReadout | null {
  if (util == null || !Number.isFinite(util) || util <= 0 || util > 1) return null;
  if (!gpus || selected.length === 0) return null;
  const totals = gpus
    .filter((g) => selected.includes(g.index))
    .map((g) => g.memory_total_mib)
    .filter((m): m is number => typeof m === "number" && Number.isFinite(m) && m > 0);
  if (totals.length === 0) return null;
  const minMib = Math.min(...totals);
  const usedGib = ((util * minMib) / 1024).toFixed(1);
  const totalGib = String(Number((minMib / 1024).toFixed(1)));
  return { usedGib, totalGib, text: `≈ ${usedGib} GiB of ${totalGib} GiB per GPU` };
}
