// Shared model-settings types and pure helpers for /ui/models/<id>/settings.
//
// Moved out of app/models/[id]/settings/page.tsx (settings redesign, Task 3)
// so the page and its extracted panels (components/models/settings/*) import
// ONE definition.

import { MODEL_HINTS } from "@/lib/settings-hints";

// ---------------------------------------------------------------------------
// Patchable allowlist — must stay in sync with backend
// app/settings/routes_api.py:_PATCHABLE_MODEL_FIELDS. Hard-coding here (vs.
// just iterating MODEL_HINTS) keeps the page from rendering inputs for the
// MODEL_HINTS entries that the backend would 400 on (quantization,
// kv_cache_dtype, block_size, swap_space, max_num_seqs,
// max_num_batched_tokens, enforce_eager, disable_log_requests).
// ---------------------------------------------------------------------------
export const PATCHABLE_KEYS = [
  "served_model_name",
  "hf_repo",
  "hf_revision",
  "gpu_indices",
  "tensor_parallel_size",
  // #286: first-class data-parallel layout + proxy-side routing knobs.
  "data_parallel_size",
  "dp_affinity_enabled",
  "dp_spill_threshold",
  "dtype",
  "max_model_len",
  "gpu_memory_utilization",
  "supports_vision",
  "supports_tools",
  "supports_reasoning",
  "trust_remote_code",
  "extra_args",
  "extra_env",
  // Sub-project C, migration 0028. llama.cpp only.
  "n_gpu_layers",
  "mmproj_filename",
] as const;

export type PatchableKey = (typeof PATCHABLE_KEYS)[number];

// llama.cpp dials with a control but NO column of their own: they live inside
// `extra_args` (see @/lib/llamacpp-args). Deliberately a separate list from
// PATCHABLE_KEYS: these must never appear in a PATCH body under their own names.
export const ARG_BACKED_KEYS = ["flash_attn", "cache_type_k", "cache_type_v"] as const;
export type ArgBackedKey = (typeof ARG_BACKED_KEYS)[number];

/** Anything the settings page draws a control for. */
export type FieldKey = PatchableKey | ArgBackedKey;

export interface ModelSettings {
  id: string;
  served_model_name: string;
  hf_repo: string;
  hf_revision: string;
  gpu_indices: number[];
  tensor_parallel_size: number;
  // #286. Optional on the wire: an older server omits all three, which reads
  // as 1 / on / auto. dp_affinity_enabled is the RAW column (1 / 0) here.
  data_parallel_size?: number;
  dp_affinity_enabled?: number | boolean;
  dp_spill_threshold?: number | null;
  dtype: string | null;
  max_model_len: number | null;
  gpu_memory_utilization: number;
  trust_remote_code: boolean;
  extra_args: string[];
  extra_env: Record<string, string>;
  // Read-only here; NULL decodes to vLLM (D6).
  backend: string | null;
  n_gpu_layers: number | null;
  mmproj_filename: string | null;
  // Capability flags: the RAW column (1 / 0 / null) so the third state
  // survives the trip; snapshotToDraft normalises to `boolean | null`.
  supports_vision: number | boolean | null;
  supports_tools: number | boolean | null;
  supports_reasoning: number | boolean | null;
  status:
    | "registered"
    | "pulling"
    | "pulled"
    | "loading"
    | "loaded"
    | "unloading"
    | "failed";
  pulled_bytes: number;
  pulled_total: number | null;
  last_error: string | null;
  layout_notice_level?: "info" | "warning" | null;
  layout_notice?: string | null;
}

/** The three tri-state capability flags (true / false / null = auto). */
export const TRISTATE_KEYS = ["supports_vision", "supports_tools", "supports_reasoning"] as const;
export type TristateKey = (typeof TRISTATE_KEYS)[number];

export type Draft = Omit<
  Pick<ModelSettings, PatchableKey>,
  TristateKey | "data_parallel_size" | "dp_affinity_enabled" | "dp_spill_threshold"
> & {
  [K in TristateKey]: boolean | null;
} & {
  data_parallel_size: number;
  dp_affinity_enabled: boolean;
  dp_spill_threshold: number | null;
};

/**
 * Keys that may be PATCHed on a LOADED model: the capability flags and the two
 * replica-routing knobs (mirrors the backend's loaded-editable set). The ONE
 * list behind both the field locking and the save bar's live / next-load split.
 */
export const LOADED_EDITABLE_KEYS: readonly string[] = [
  ...TRISTATE_KEYS,
  "dp_affinity_enabled",
  "dp_spill_threshold",
];

/** Normalise a raw tri-state column (null / 0 / 1 / bool) to `boolean | null`. */
export function triState(v: number | boolean | null | undefined): boolean | null {
  return v === null || v === undefined ? null : Boolean(v);
}

export function snapshotToDraft(s: ModelSettings): Draft {
  return {
    served_model_name: s.served_model_name,
    hf_repo: s.hf_repo,
    hf_revision: s.hf_revision,
    gpu_indices: s.gpu_indices,
    tensor_parallel_size: s.tensor_parallel_size,
    data_parallel_size: s.data_parallel_size ?? 1,
    dp_affinity_enabled:
      s.dp_affinity_enabled === undefined ? true : Boolean(s.dp_affinity_enabled),
    dp_spill_threshold: s.dp_spill_threshold ?? null,
    dtype: s.dtype,
    max_model_len: s.max_model_len,
    gpu_memory_utilization: s.gpu_memory_utilization,
    n_gpu_layers: s.n_gpu_layers ?? null,
    mmproj_filename: s.mmproj_filename ?? null,
    supports_vision: triState(s.supports_vision),
    supports_tools: triState(s.supports_tools),
    supports_reasoning: triState(s.supports_reasoning),
    trust_remote_code: s.trust_remote_code,
    extra_args: s.extra_args,
    extra_env: s.extra_env,
  };
}

/**
 * Per-key deep equality. Arrays compare elementwise; objects compare entry
 * sets; scalars use ===. No JSON.stringify shortcut: key order can differ
 * between snapshot and draft for `extra_env` and would falsely report dirty.
 */
export function eqValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!eqValue(a[i], b[i])) return false;
    return true;
  }
  if (
    a &&
    b &&
    typeof a === "object" &&
    typeof b === "object" &&
    !Array.isArray(a) &&
    !Array.isArray(b)
  ) {
    const ao = a as Record<string, unknown>;
    const bo = b as Record<string, unknown>;
    const ak = Object.keys(ao);
    const bk = Object.keys(bo);
    if (ak.length !== bk.length) return false;
    for (const k of ak) {
      if (!Object.prototype.hasOwnProperty.call(bo, k)) return false;
      if (!eqValue(ao[k], bo[k])) return false;
    }
    return true;
  }
  return false;
}

export function dirtyKeys(draft: Draft, snapshot: Draft): PatchableKey[] {
  return PATCHABLE_KEYS.filter((k) => !eqValue(draft[k], snapshot[k]));
}

/** Operator-facing label for a settings key (falls back to the key itself). */
export function labelFor(key: string): string {
  return MODEL_HINTS[key]?.label ?? key;
}

/**
 * Operator-facing text for a setting value. One wording everywhere a value is
 * shown as before/after: the save bar's review list and the preset / suggest
 * diffs. `null` reads as what the engine does without the value ("auto", or
 * "default" for dtype and the llama.cpp flags), never as JSON.
 */
export function formatSettingValue(key: string, v: unknown): string {
  if (key.startsWith("supports_")) return v === null || v === undefined ? "Auto" : v ? "Yes" : "No";
  if (typeof v === "boolean") return v ? "on" : "off";
  if (v === null || v === undefined) {
    if (key === "dtype" || (ARG_BACKED_KEYS as readonly string[]).includes(key)) return "default";
    return "auto";
  }
  if (Array.isArray(v)) return v.join(key === "gpu_indices" ? ", " : " ");
  if (typeof v === "object")
    return Object.entries(v as Record<string, unknown>)
      .map(([a, b]) => `${a}=${String(b)}`)
      .join(" ");
  return String(v);
}
