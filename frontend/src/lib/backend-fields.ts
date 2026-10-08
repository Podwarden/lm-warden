// ---------------------------------------------------------------------------
// Which model fields each backend actually consumes.
//
// This table used to live inside `app/models/[id]/settings/page.tsx` and was
// applied on that page alone. The detail page rendered the same row's fields
// unconditionally, so a llama.cpp model advertised a `Tensor parallel size` and
// a `gpu_memory_utilization` -- two vLLM-only concepts -- next to an `Engine`
// label. One table, one rule, both pages: a second copy is how the two drift.
//
// Hide, do not disable. A greyed-out box for a knob the engine has never heard
// of is an invitation to wonder what it does; the field is not "unavailable",
// it is meaningless (design spec §9.1, UI audit §4.5).
//
// The source of truth is `BackendCapabilities`, served on
// `/api/system/backends`; this map is the rendering half of it. A field named
// for NO backend here is shared and always shown -- that is the common case,
// and it is why the map lists only the exceptions.
//
// `parallelism_strategy` is deliberately absent from both lists: the control is
// create-time only, and if it ever reaches a detail or settings surface it must
// not be offered for llama.cpp -- plan() maps both 'tp' and 'pp' to
// --split-mode layer, so the question has no answer there.
// ---------------------------------------------------------------------------

export const BACKEND_ONLY_FIELDS: Record<string, readonly string[]> = {
  vllm: [
    "dtype",
    "gpu_memory_utilization",
    "trust_remote_code",
    // vLLM's --tensor-parallel-size. llama.cpp reports
    // supports_tensor_parallel: false and splits by layer instead, so the
    // number is meaningless on a llama.cpp row even though the column is
    // populated (ModelCreate derives it from len(gpu_indices) for every row).
    "tensor_parallel_size",
    // #286: data-parallel replicas and their proxy-side routing knobs. llama.cpp
    // has no data-parallel mode (supports_data_parallel: false).
    "data_parallel_size",
    "dp_affinity_enabled",
    "dp_spill_threshold",
  ],
  llamacpp: [
    "n_gpu_layers",
    "mmproj_filename",
    // Stored inside `extra_args` rather than in columns of their own -- see
    // @/lib/llamacpp-args for why. They are listed HERE anyway, because this
    // table is the one rule for per-backend visibility and a field's storage
    // is not what decides whether it is shown. A `--cache-type-k` control on a
    // vLLM row would be exactly the defect the table exists to prevent.
    "flash_attn",
    "cache_type_k",
    "cache_type_v",
  ],
};

/**
 * Is `key` shown for a row served by `backend`?
 *
 * A null/empty backend means vLLM (decision D6 — the column is NULL on every
 * pre-0027 row). The API now always sends a concrete value; the fallback stays
 * because this helper is also called with drafts that have not loaded yet.
 */
export function fieldAppliesTo(key: string, backend: string | null | undefined): boolean {
  const effective = backend || "vllm";
  return !Object.entries(BACKEND_ONLY_FIELDS).some(
    ([b, keys]) => b !== effective && keys.includes(key),
  );
}

// ---------------------------------------------------------------------------
// The same rule, one level up: whole PANELS a backend has no concept of.
//
// The Try-stack panel picks a CUDA channel and a vLLM version. Both are vLLM's
// image catalogue (app/runtime/backends/vllm/images.py), and there is no
// llama.cpp counterpart -- so on a llama.cpp row the panel was not merely
// offering an unavailable control, it was offering one that could only ever
// resolve a *vLLM* image and launch vLLM under the operator's llama.cpp model
// name.
//
// That is the field rule at panel scale, so it lives in this file rather than
// in a second gate somewhere else. What differs is where the answer comes from:
// a field's owner is a static fact about the engine and is listed above, while
// "can a version be pinned here" is a fact about the engine AND the running
// driver, which only the server knows. So the server sends the decision
// (`version_pin_reason_code` on GET /api/system/backends) and this function
// turns it into the one thing the client has to choose between.
// ---------------------------------------------------------------------------

/**
 * Why a version pin is unavailable, as reported by GET /api/system/backends.
 *
 * `null` means it IS available. The three non-null values differ in what the
 * operator could do about it, which is exactly what decides the rendering.
 */
export type VersionPinReasonCode = "engine" | "driver" | "backend" | null;

/**
 * Should a version-pin control be RENDERED AT ALL for this backend?
 *
 * `true` for an available pin and for a `"driver"` obstacle: the driver is a
 * deployment choice the operator can change, so the control stays visible,
 * disabled, with the server's sentence beside it -- they need to see that the
 * capability exists and what would unlock it (UI audit §2.7, the house
 * pattern).
 *
 * `false` for `"backend"` and `"engine"`: no driver removes those. The control
 * is not "unavailable here", it is meaningless for this engine, and this file's
 * standing rule applies -- hide, do not disable. A greyed-out channel selector
 * on a llama.cpp model is an invitation to wonder which channel it would use.
 *
 * An UNRECOGNISED code returns `true`, deliberately. It mirrors
 * /api/system/engine's own defensive default (`supports` is True for an unknown
 * driver so it never wrongly blocks): a code this build has not heard of is a
 * newer server, not a licence to silently remove a control that works.
 */
export function versionPinControlApplies(
  code: VersionPinReasonCode | string | undefined,
): boolean {
  return code !== "backend" && code !== "engine";
}

/** How to name a backend in the UI. `llamacpp` is a wire value, not a label. */
export function backendDisplayName(backend: string | null | undefined): string {
  const effective = backend || "vllm";
  return effective === "llamacpp" ? "llama.cpp" : effective === "vllm" ? "vLLM" : effective;
}
