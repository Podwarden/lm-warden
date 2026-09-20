"use client";

import Link from "next/link";
import { use, useEffect, useMemo, useRef, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { MODEL_HINTS } from "@/lib/settings-hints";
import { fieldAppliesTo } from "@/lib/backend-fields";
import {
  LLAMACPP_MANAGED_FLAGS,
  mergeManagedArgs,
  setManagedArg,
  splitManagedArgs,
} from "@/lib/llamacpp-args";
import { SettingField } from "@/components/settings/setting-field";
import { SettingsSection } from "@/components/settings-section";
import { type GpuInfo } from "@/components/gpu/gpu-checklist";
import { copyToClipboard } from "@/lib/utils";
import { useBreadcrumb } from "@/lib/use-breadcrumb";

// ---------------------------------------------------------------------------
// Patchable allowlist — must stay in sync with backend
// app/settings/routes_api.py:_PATCHABLE_MODEL_FIELDS. Hard-coding here (vs.
// just iterating MODEL_HINTS) keeps the page from rendering inputs for the
// 8 MODEL_HINTS entries that the backend would 400 on (quantization,
// kv_cache_dtype, block_size, swap_space, max_num_seqs,
// max_num_batched_tokens, enforce_eager, disable_log_requests). The order
// here is the visual order the operator sees — identity-y fields first,
// then runtime knobs, then escape hatches.
// ---------------------------------------------------------------------------
const PATCHABLE_KEYS = [
  "served_model_name",
  "hf_repo",
  "hf_revision",
  "gpu_indices",
  "tensor_parallel_size",
  "dtype",
  "max_model_len",
  "gpu_memory_utilization",
  "supports_vision",
  "supports_tools",
  "supports_reasoning",
  "trust_remote_code",
  "extra_args",
  "extra_env",
  // Sub-project C, migration 0028. llama.cpp only -- BACKEND_ONLY below hides
  // them for a vLLM row, and hides vLLM's own knobs for a llama.cpp one.
  "n_gpu_layers",
  "mmproj_filename",
] as const;

type PatchableKey = (typeof PATCHABLE_KEYS)[number];

// llama.cpp dials with a control but NO column of their own: they live inside
// `extra_args`, which is where a hand-typed one would go and what the argv
// builder already appends last (@/lib/llamacpp-args explains the choice).
//
// Deliberately a SEPARATE list from PATCHABLE_KEYS rather than an addition to
// it. PATCHABLE_KEYS is the backend's column allowlist and the draft's shape;
// these three are not columns and must never appear in a PATCH body under
// their own names -- the write they produce is an `extra_args` write, so
// nothing new had to be allowed server-side. Two lists, two meanings: what the
// API accepts, and what the page renders.
const ARG_BACKED_KEYS = ["flash_attn", "cache_type_k", "cache_type_v"] as const;
type ArgBackedKey = (typeof ARG_BACKED_KEYS)[number];

/** Anything the settings page draws a control for. */
type FieldKey = PatchableKey | ArgBackedKey;

// ---------------------------------------------------------------------------
// Section grouping (S4 design principle #1 + #2 — discrete instrument-panel
// sections of related knobs). Each key appears in exactly one section so the
// renderer can do a flat iteration per section without dedup. Order of sections
// + order of keys within each section is the visual order the operator sees.
// ---------------------------------------------------------------------------
const SECTION_GROUPS: ReadonlyArray<{
  title: string;
  testId: string;
  keys: ReadonlyArray<FieldKey>;
}> = [
  {
    title: "Identity",
    testId: "section-identity",
    keys: ["served_model_name", "hf_repo", "hf_revision"],
  },
  {
    title: "Memory",
    testId: "section-memory",
    keys: ["gpu_memory_utilization", "max_model_len", "dtype", "n_gpu_layers"],
  },
  {
    // Absent entirely on a vLLM row -- every key in it is llama.cpp's, so
    // fieldAppliesTo empties the section and SettingsSection renders nothing.
    // These are the knobs an operator actually reaches for on a small card,
    // and until now the only way to set any of them was to hand-type a flag
    // into Extra args.
    title: "llama.cpp engine",
    testId: "section-llamacpp",
    keys: ["flash_attn", "cache_type_k", "cache_type_v"],
  },
  {
    title: "Compute",
    testId: "section-compute",
    keys: ["gpu_indices", "tensor_parallel_size"],
  },
  {
    title: "Capabilities",
    testId: "section-capabilities",
    keys: ["supports_vision", "supports_tools", "supports_reasoning"],
  },
  {
    title: "Advanced",
    testId: "section-advanced",
    keys: ["trust_remote_code", "extra_args", "extra_env", "mmproj_filename"],
  },
];

// `fieldAppliesTo` and its table moved to @/lib/backend-fields so the model
// DETAIL page can apply the same rule. It used to live here and be applied on
// this page alone, which is why a llama.cpp row's detail view still advertised
// a Tensor parallel size and a gpu_memory_utilization.

interface ModelSettings {
  id: string;
  served_model_name: string;
  hf_repo: string;
  hf_revision: string;
  gpu_indices: number[];
  tensor_parallel_size: number;
  dtype: string | null;
  max_model_len: number | null;
  gpu_memory_utilization: number;
  trust_remote_code: boolean;
  extra_args: string[];
  extra_env: Record<string, string>;
  // Sub-project C. `backend` is read-only HERE -- switching a loaded model's
  // engine is a load-time decision made in the wizard, and the PATCH route
  // validates it separately (app/settings/routes_api.py::_coerce_backend).
  // NULL decodes to vLLM (D6).
  backend: string | null;
  n_gpu_layers: number | null;
  mmproj_filename: string | null;
  // Capability flags. The GET returns the RAW column (1 / 0 / null) rather
  // than a coerced bool precisely so the third state survives the trip — see
  // _MODEL_TRISTATE_FIELDS in app/settings/routes_api.py. They are widened to
  // `number | boolean | null` because the PATCH we send back uses booleans;
  // `snapshotToDraft` normalises both shapes to `boolean | null`.
  supports_vision: number | boolean | null;
  supports_tools: number | boolean | null;
  supports_reasoning: number | boolean | null;
  // Read-only fields included in GET response.
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
}

// The three tri-state capability flags. Held in the draft as `boolean | null`
// (never the raw 0/1 the GET hands back) so the PATCH body is already in the
// shape the backend wants: true / false / null, where null means "reset to
// auto-detection" and an omitted key means "don't touch".
const TRISTATE_KEYS = ["supports_vision", "supports_tools", "supports_reasoning"] as const;
type TristateKey = (typeof TRISTATE_KEYS)[number];

type Draft = Omit<Pick<ModelSettings, PatchableKey>, TristateKey> & {
  [K in TristateKey]: boolean | null;
};

/** Normalise a raw tri-state column (null / 0 / 1 / bool) to `boolean | null`. */
function triState(v: number | boolean | null | undefined): boolean | null {
  return v === null || v === undefined ? null : Boolean(v);
}

const DTYPE_OPTIONS = ["auto", "float16", "bfloat16", "float32"];

// ---------------------------------------------------------------------------
// Presets (GET /api/presets) — see app/presets/builtin.json. Each entry
// declares a sparse settings dict that we overlay on the current draft. The
// "target_archetype" is informational only — the user picks by name; we don't
// pre-filter based on GPU arch (the operator knows their hardware better than
// we do, and a misclassified arch would silently hide useful presets).
// ---------------------------------------------------------------------------
interface PresetEntry {
  id: string;
  name: string;
  description: string;
  target_archetype: string;
  settings: Record<string, unknown>;
}
interface PresetsResponse {
  presets: PresetEntry[];
}

// ---------------------------------------------------------------------------
// Suggest (GET /api/models/{id}/suggest-config) — server returns a flat blob
// (see app/models/suggest.py SuggestedConfig.to_dict). Each non-null key is a
// suggested starting point we surface to the operator; `disclaimer` is shown
// as the rationale string and intentionally NOT applied as a setting. Server
// may add new keys; unknown keys are ignored by applySparseSettings.
// ---------------------------------------------------------------------------
interface SuggestResponse {
  gpu_memory_utilization?: number | null;
  max_model_len?: number | null;
  kv_cache_dtype?: string | null;
  disclaimer?: string;
  // Permit future additions without breaking the FE.
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Effective argv (GET /api/models/{id}/effective-argv) — the FULL argv, from
// argv[0] onwards, that the engine would actually be invoked with for the
// current persisted settings (preview port 10000). The panel polls along with
// the draft so saves show up almost immediately.
// ---------------------------------------------------------------------------
interface EffectiveArgvResponse {
  argv: string[];
}

function snapshotToDraft(s: ModelSettings): Draft {
  return {
    served_model_name: s.served_model_name,
    hf_repo: s.hf_repo,
    hf_revision: s.hf_revision,
    gpu_indices: s.gpu_indices,
    tensor_parallel_size: s.tensor_parallel_size,
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

// Per-key deep equality. Arrays compare elementwise; objects compare entry
// sets; scalars use ===. This is what dirty-tracking uses to decide which
// keys to include in the PATCH body — see `dirtyKeys()` below. We avoid a
// JSON.stringify shortcut because key-order can differ across snapshot vs.
// draft for `extra_env` and would falsely report dirty.
function eqValue(a: unknown, b: unknown): boolean {
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

function dirtyKeys(draft: Draft, snapshot: Draft): PatchableKey[] {
  return PATCHABLE_KEYS.filter((k) => !eqValue(draft[k], snapshot[k]));
}

export default function ModelSettingsPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  // Same Next.js 15 unwrap pattern as /models/[id]/page.tsx — see ref at
  // file head 80-86 there.
  const { id } = use(params);
  const key = `/api/models/${id}/settings`;
  const { data, error, isLoading, mutate } = useSWR<ModelSettings>(
    key,
    authFetchJSON,
    {
      // No polling — this is an edit form, polling would fight the user's
      // typing by rewriting the snapshot mid-keystroke. Save/Reset go
      // through explicit mutate() calls below.
      revalidateOnFocus: false,
    },
  );
  // Name the parent crumb (/models/[id]) after the model — on a direct load
  // of this page nothing else has told the breadcrumb what it is called. The
  // page's own crumb is the registry's "Settings".
  useBreadcrumb({ path: `/models/${id}`, title: data?.served_model_name ?? (error ? id : undefined) });
  const { data: gpuData } = useSWR<{ gpus: GpuInfo[]; probed_at: string; probe_error: string | null }>(
    "/api/system/gpus",
    authFetchJSON,
  );
  const gpus = gpuData?.gpus ?? [];
  // Global mutate for cross-page cache coherence after a successful save —
  // the dashboard list (/api/models) and the model detail page
  // (/api/models/{id}) would otherwise show stale values for up to one
  // poll cycle. Pattern mirrors frontend/src/app/models/[id]/page.tsx.
  const { mutate: globalMutate } = useSWRConfig();

  const [draft, setDraft] = useState<Draft | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  // 409 is special: surfaced as a non-dismissable banner, not the inline
  // saveError red text. Keeping it separate also lets the pre-emptive
  // status guard reuse the same UI without conflating "I tried to save"
  // with "the page noticed status=loaded on first load".
  const [conflict, setConflict] = useState(false);

  // Initialize draft from snapshot exactly once per load cycle. A naive
  // useEffect dep on `data` would clobber the user's in-flight edits on
  // every SWR revalidation. The ref tracks "I've consumed this snapshot
  // identity" so subsequent revalidations (e.g. after a successful save)
  // re-seed only when the underlying row is genuinely new.
  const seededFor = useRef<string | null>(null);
  useEffect(() => {
    if (!data) return;
    // Use the served_model_name+updated_at-ish signal. The settings
    // response doesn't expose updated_at, so fall back to a structural
    // hash via JSON.stringify of the patchable fields. After a successful
    // PATCH we explicitly null seededFor.current so the next snapshot
    // re-seeds the draft.
    const sig = JSON.stringify(snapshotToDraft(data));
    if (seededFor.current === sig) return;
    seededFor.current = sig;
    setDraft(snapshotToDraft(data));
  }, [data]);

  // ---- 404 — model not found. Mirror the detail-page pattern. ----------
  const errStatus = (error as (Error & { status?: number }) | undefined)
    ?.status;
  if (errStatus === 404) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Model not found</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-400">
            The model <span className="font-mono">{id}</span> does not exist.
            It may have been deleted.
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isLoading || (!data && !error)) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="space-y-4">
        <div className="rounded-md border border-red-700 bg-red-900/30 p-4 text-sm text-red-200">
          Failed to load settings
          {error instanceof Error ? `: ${error.message}` : "."}
        </div>
      </div>
    );
  }

  if (!data || !draft) return null;

  // Pre-emptive guard: a loaded model would 409 on PATCH anyway, so we
  // disable all inputs + Save up front and surface the same banner copy
  // the 409 branch would. The form is rendered so the operator can see
  // what's currently in place, just can't change it without unloading.
  //
  // One carve-out, mirroring the backend's (app/settings/routes_api.py:
  // `capabilities_only`): a patch made up ENTIRELY of capability flags is
  // allowed on a loaded model. Those columns are inert metadata the load
  // runner never reads, and the loaded model is exactly the one an operator
  // is looking at when they notice vision is set wrong.
  const isLoaded = data.status === "loaded";
  const allDisabled = isLoaded || saving;
  const capabilitiesDisabled = saving;
  const dirty = dirtyKeys(draft, snapshotToDraft(data));
  const capabilitiesOnly =
    dirty.length > 0 &&
    dirty.every((k) => (TRISTATE_KEYS as readonly string[]).includes(k));
  // Per-model gpu_indices requires >=1 selection (spec: "Save disabled /
  // validation message when empty"). Unlike Settings → default_gpu_indices
  // (which may be empty = no preset), a model with zero GPUs would be handed
  // an empty CUDA_VISIBLE_DEVICES and never schedule. Block Save and surface
  // an inline message next to the field.
  const gpuIndicesEmpty = draft.gpu_indices.length === 0;
  const canSave =
    !saving && dirty.length > 0 && !conflict && !gpuIndicesEmpty &&
    (!isLoaded || capabilitiesOnly);

  async function onSave() {
    if (!data || !draft) return;
    // Compute dirty-set BEFORE flipping the spinner — if there's nothing to
    // save we return synchronously, skipping the try/finally entirely so
    // the Save button can't get stuck in the "Saving…" state when the
    // operator clicks Save with no changes.
    const dirtyNow = dirtyKeys(draft, snapshotToDraft(data));
    if (dirtyNow.length === 0) return;
    setSaving(true);
    setSaveError(null);
    try {
      const body: Partial<Draft> = {};
      for (const k of dirtyNow) {
        // Indexed assignment widens to `unknown`; the cast keeps the
        // typed shape on the wire without sacrificing the per-key
        // narrowing the rest of the file relies on.
        (body as Record<string, unknown>)[k] = draft[k];
      }
      const r = await authFetch(key, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          const j = await r.json();
          if (j && typeof j.detail === "string") detail = j.detail;
          // A 422 carries FastAPI's list of field errors. The PATCH now
          // validates the whole merged row against the same rules as register
          // (#262), so this is the ordinary shape of a refused save and the
          // message is the only thing that says which field is wrong.
          else if (Array.isArray(j?.detail)) {
            const msgs = (j.detail as { msg?: unknown }[])
              .map((e) => (typeof e?.msg === "string" ? e.msg : null))
              .filter((m): m is string => m !== null);
            if (msgs.length) detail = msgs.join("; ");
          }
        } catch {
          /* non-JSON body */
        }
        // Two things answer 409 now: the unload-first guard, and a
        // served_model_name that another row already holds. Only the first is
        // the "reload and unload" banner; the second is an ordinary save
        // error naming the field the operator just edited.
        if (r.status === 409 && !detail.includes("already exists")) {
          setConflict(true);
          return;
        }
        setSaveError(detail);
        return;
      }
      // Success — clear the seeded signal so the next revalidate re-seeds
      // the draft to the freshly-saved snapshot. mutate() with undefined
      // forces a re-fetch.
      seededFor.current = null;
      await mutate();
      // Fire-and-forget cross-page cache busts so the dashboard list and
      // the model detail page reflect the new served_model_name/etc.
      // the draft so saves show up almost immediately.on back-navigation. Awaiting these would make the
      // operator wait on two purely-cosmetic refreshes.
      void globalMutate("/api/models");
      void globalMutate(`/api/models/${id}`);
      // Also refresh the effective-argv panel — it's keyed off the saved
      // settings, so a successful PATCH should flash it.
      void globalMutate(`/api/models/${id}/effective-argv`);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  function onReset() {
    if (!data) return;
    seededFor.current = null; // allow the seeding effect to re-fire
    setDraft(snapshotToDraft(data));
    setSaveError(null);
    // Clear a 409-sourced banner too — the user explicitly chose to
    // discard their in-flight edits, so the "you tried to save while
    // loaded" warning is no longer relevant. The pre-emptive
    // status==='loaded' banner stays because `isLoaded` is derived from
    // server state, not local; only the next revalidate clears it.
    setConflict(false);
  }

  // Apply a sparse settings dict (from preset or suggest) on top of the
  // current draft. Only keys in PATCHABLE_KEYS are accepted; unknown keys
  // are silently ignored so a future backend addition doesn't crash an old
  // FE. Each value goes through the same per-kind coercion the SettingField
  // onChange handlers do, so an arbitrary JSON int for max_model_len lands
  // as a number not a string. Returns nothing — mutates draft via setDraft.
  function applySparseSettings(sparse: Record<string, unknown>) {
    setDraft((d) => {
      if (d === null) return d;
      const next: Draft = { ...d };
      for (const k of PATCHABLE_KEYS) {
        if (!Object.prototype.hasOwnProperty.call(sparse, k)) continue;
        const v = sparse[k];
        // The TS-safe cast is per-key: we know each key's runtime shape.
        // We do a narrow JS coercion (numbers as numbers, etc.) rather
        // than blindly assigning `v as Draft[K]` so bad inputs surface as
        // visible-on-the-page wrong values not silent type-confused state.
        switch (k) {
          case "served_model_name":
          case "hf_repo":
          case "hf_revision":
            if (typeof v === "string") next[k] = v;
            break;
          case "tensor_parallel_size":
            if (typeof v === "number" && Number.isFinite(v))
              next.tensor_parallel_size = Math.max(1, Math.floor(v));
            break;
          case "max_model_len":
            if (v === null) next.max_model_len = null;
            else if (typeof v === "number" && Number.isFinite(v))
              next.max_model_len = Math.max(1, Math.floor(v));
            break;
          case "gpu_memory_utilization":
            if (typeof v === "number" && Number.isFinite(v))
              next.gpu_memory_utilization = Math.min(1.0, Math.max(0.05, v));
            break;
          case "dtype":
            if (v === null) next.dtype = null;
            else if (typeof v === "string" && DTYPE_OPTIONS.includes(v))
              next.dtype = v;
            break;
          case "trust_remote_code":
            if (typeof v === "boolean") next.trust_remote_code = v;
            break;
          case "gpu_indices":
            if (
              Array.isArray(v) &&
              v.every((x) => typeof x === "number" && Number.isFinite(x))
            ) {
              next.gpu_indices = (v as number[]).map((x) => Math.floor(x));
              // Carry tensor_parallel_size with the placement, exactly as the
              // GPU picker's own onChange does. The PATCH validates the MERGED
              // row and holds it to register's cross-field rule
              // (tensor_parallel_size == len(gpu_indices)), so a preset that
              // moves the placement without the parallel size is a 422 -- and a
              // row where the two disagree could not load anyway.
              next.tensor_parallel_size = next.gpu_indices.length;
            }
            break;
          case "extra_args":
            if (Array.isArray(v) && v.every((x) => typeof x === "string"))
              next.extra_args = v as string[];
            break;
          case "extra_env":
            if (
              v &&
              typeof v === "object" &&
              !Array.isArray(v) &&
              Object.values(v as Record<string, unknown>).every(
                (x) => typeof x === "string",
              )
            )
              next.extra_env = v as Record<string, string>;
            break;
        }
      }
      return next;
    });
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <h1 className="text-2xl font-semibold">Settings</h1>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onReset}
            disabled={allDisabled || dirty.length === 0}
            data-testid="settings-reset"
          >
            Reset
          </Button>
          <Button
            size="sm"
            onClick={onSave}
            disabled={!canSave}
            data-testid="settings-save"
          >
            {saving ? "Saving…" : "Save"}
          </Button>
        </div>
      </div>

      {(isLoaded || conflict) && (
        <div
          role="alert"
          aria-live="polite"
          className="rounded-md border border-amber-700 bg-amber-900/30 p-3 text-sm text-amber-200"
          data-testid="settings-loaded-banner"
        >
          Model must be unloaded before editing settings —{" "}
          <Link href={`/models/${id}`} className="underline">
            go to the model page
          </Link>{" "}
          and unload it first. The capability flags below are the exception:
          they are metadata the engine never reads, so they can be corrected
          on a loaded model.
        </div>
      )}

      {saveError && (
        <div
          role="alert"
          className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
          data-testid="settings-save-error"
        >
          {saveError}
        </div>
      )}

      <PresetStrip
        disabled={allDisabled}
        draft={draft}
        onApply={applySparseSettings}
      />

      <Card>
        <CardHeader>
          <CardTitle>Model configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {SECTION_GROUPS.map((group) => {
            // A section every one of whose keys belongs to a DIFFERENT backend
            // is not rendered at all. Without this the "llama.cpp engine"
            // section would draw its header and a collapse triangle over an
            // empty body on every vLLM row -- a control that opens onto
            // nothing, which is the defect class this branch is about. Memory
            // is exempt because it also carries the Suggest panel.
            const visible = group.keys.filter(
              (k) => MODEL_HINTS[k] && fieldAppliesTo(k, data.backend),
            );
            if (visible.length === 0 && group.title !== "Memory") return null;
            return (
            <SettingsSection
              key={group.testId}
              title={group.title}
              testId={group.testId}
              subtitle={renderSectionSubtitle(group.keys, draft, data)}
            >
              {/* Memory section gets the inline Suggest values affordance
                  at the top — that's where gpu_memory_utilization and
                  max_model_len live, which are exactly what the suggester
                  drives off last_error analysis. */}
              {group.title === "Memory" && (
                <SuggestPanel
                  modelId={id}
                  disabled={allDisabled}
                  draft={draft}
                  onApply={applySparseSettings}
                />
              )}
              {/* Explain, do not offer a dead control. --split-mode and
                  --main-gpu are DERIVED from the GPU selection (decision D4
                  keeps the GPU count the invariant and lets plan() pick the
                  backend-appropriate flag), and --tensor-split is left to
                  llama.cpp's memory-proportional default, which is the better
                  answer on a heterogeneous box. A control for any of them
                  would be a second, competing source for a fact the Compute
                  section already owns. */}
              {group.testId === "section-llamacpp" && (
                <p
                  data-testid="llamacpp-derived-note"
                  className="text-xs text-slate-500"
                >
                  <code>--split-mode</code> and <code>--main-gpu</code> follow the
                  GPU selection in Compute and are not set here.{" "}
                  <code>--tensor-split</code> is left to llama.cpp&apos;s
                  memory-proportional default. Anything else goes in Extra args.
                </p>
              )}
              {group.keys.map((k) => {
                const hint = MODEL_HINTS[k];
                if (!hint) return null;
                if (!fieldAppliesTo(k, data.backend)) return null;
                return (
                  <SettingFieldFor
                    key={k}
                    fieldKey={k}
                    hint={hint}
                    draft={draft}
                    gpus={gpus}
                    setDraft={(updater) =>
                      setDraft((d) => (d === null ? d : updater(d)))
                    }
                    disabled={
                      (TRISTATE_KEYS as readonly string[]).includes(k)
                        ? capabilitiesDisabled
                        : allDisabled
                    }
                    gpuIndicesEmpty={gpuIndicesEmpty}
                    modelId={id}
                  />
                );
              })}
            </SettingsSection>
            );
          })}
        </CardContent>
      </Card>

      <EffectiveArgvPanel modelId={id} />
    </div>
  );
}

// Subtitle shown on the section header — counts dirty keys against the
// last-saved snapshot so the operator can collapse a section and still see
// "2 unsaved" next to its title.
function renderSectionSubtitle(
  keys: ReadonlyArray<FieldKey>,
  draft: Draft,
  snapshot: ModelSettings,
): string | null {
  const snap = snapshotToDraft(snapshot);
  // `extra_args` is ONE column serving two sections, so it is counted in
  // halves: the llama.cpp controls own the managed flags, Extra args owns the
  // rest. Counting the whole column in both would report a single edit twice,
  // in two different places, and counting it in neither would leave a changed
  // control with no unsaved marker at all.
  const draftArgs = splitManagedArgs(draft.extra_args);
  const snapArgs = splitManagedArgs(snap.extra_args);
  const dirty = keys.filter((k) => {
    if ((ARG_BACKED_KEYS as readonly string[]).includes(k)) {
      return draftArgs.managed[k] !== snapArgs.managed[k];
    }
    if (k === "extra_args") return !eqValue(draftArgs.rest, snapArgs.rest);
    return !eqValue(draft[k as PatchableKey], snap[k as PatchableKey]);
  });
  if (dirty.length === 0) return null;
  return `${dirty.length} unsaved`;
}

// ---------------------------------------------------------------------------
// PresetStrip — horizontal row of preset chips above the form (S4 design
// principle #4 — presets are first-class, named, and previewed before apply).
// Clicking a preset opens a tiny confirm popover that shows exactly which
// keys would change relative to the current draft. The popover is the same
// component shape as the Suggest panel so the diff-then-apply flow is
// consistent across both entry points.
// ---------------------------------------------------------------------------
function PresetStrip({
  disabled,
  draft,
  onApply,
}: {
  disabled: boolean;
  draft: Draft;
  onApply: (sparse: Record<string, unknown>) => void;
}) {
  const { data, error } = useSWR<PresetsResponse>(
    "/api/presets",
    authFetchJSON,
    { revalidateOnFocus: false },
  );
  const [pendingId, setPendingId] = useState<string | null>(null);
  const presets = data?.presets ?? [];
  const pending = pendingId
    ? presets.find((p) => p.id === pendingId) ?? null
    : null;
  const pendingDiff = useMemo(
    () => (pending ? computeDiff(draft, pending.settings) : []),
    [draft, pending],
  );

  if (error) {
    return (
      <div
        className="text-xs text-slate-500"
        data-testid="presets-error"
      >
        Could not load presets.
      </div>
    );
  }
  if (!data) {
    return (
      <div className="text-xs text-slate-500" data-testid="presets-loading">
        Loading presets…
      </div>
    );
  }

  return (
    <div data-testid="presets-strip" className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs uppercase tracking-wider text-slate-400">
          Apply preset
        </span>
        {presets.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => setPendingId(p.id)}
            disabled={disabled}
            data-testid={`preset-chip-${p.id}`}
            title={p.description}
            className="rounded-full border border-slate-600 px-3 py-1 text-xs text-slate-200 hover:border-amber-600 hover:text-amber-300 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {p.name}
          </button>
        ))}
      </div>
      {pending && (
        <div
          role="dialog"
          aria-label={`Apply preset ${pending.name}`}
          data-testid="preset-confirm"
          className="rounded-md border border-slate-700 bg-slate-900/60 p-3 text-sm"
        >
          <div className="flex items-center justify-between">
            <span className="text-slate-200">
              Apply <strong>{pending.name}</strong>?
            </span>
            <span className="text-[11px] text-slate-500">
              {pending.target_archetype}
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-400">{pending.description}</p>
          <DiffList rows={pendingDiff} testIdPrefix="preset-diff" />
          <div className="mt-3 flex justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPendingId(null)}
              data-testid="preset-cancel"
            >
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={() => {
                onApply(pending.settings);
                setPendingId(null);
              }}
              disabled={pendingDiff.length === 0}
              data-testid="preset-apply"
            >
              Apply
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SuggestPanel — inline "Suggest values" button rendered at the top of the
// Memory section (S4 design principle #5 — suggestions live next to the
// fields they touch, not as a global modal). On click the FE fetches
// /api/models/{id}/suggest-config, renders the diff against the current
// draft, and offers Apply/Dismiss.
// ---------------------------------------------------------------------------
function SuggestPanel({
  modelId,
  disabled,
  draft,
  onApply,
}: {
  modelId: string;
  disabled: boolean;
  draft: Draft;
  onApply: (sparse: Record<string, unknown>) => void;
}) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<SuggestResponse | null>(null);

  async function fetchSuggestion() {
    setLoading(true);
    setErr(null);
    setResult(null);
    try {
      const data = await authFetchJSON<SuggestResponse>(
        `/api/models/${modelId}/suggest-config`,
      );
      setResult(data);
      setOpen(true);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  // Strip the disclaimer + drop null values before diffing — the suggest
  // endpoint returns null for "I don't have an opinion" which would otherwise
  // show up as a misleading "→ null" diff row against a non-null draft.
  const applicable = useMemo<Record<string, unknown>>(() => {
    if (!result) return {};
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(result)) {
      if (k === "disclaimer") continue;
      if (v === null || v === undefined) continue;
      out[k] = v;
    }
    return out;
  }, [result]);

  const diff = useMemo(
    () => (result ? computeDiff(draft, applicable) : []),
    [draft, result, applicable],
  );

  return (
    <div data-testid="suggest-panel" className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400">
          Need a starting point? Suggest values based on model config and detected VRAM.
        </span>
        <Button
          size="sm"
          variant="outline"
          onClick={fetchSuggestion}
          disabled={disabled || loading}
          data-testid="suggest-fetch"
        >
          {loading ? "Loading…" : "Suggest values"}
        </Button>
      </div>
      {err && (
        <div
          className="text-xs text-red-400"
          data-testid="suggest-error"
        >
          {err}
        </div>
      )}
      {open && result && (
        <div
          role="dialog"
          aria-label="Suggested values"
          data-testid="suggest-result"
          className="rounded-md border border-slate-700 bg-slate-900/60 p-3"
        >
          {result.disclaimer && (
            <p className="text-xs text-slate-400" data-testid="suggest-rationale">
              {result.disclaimer}
            </p>
          )}
          <DiffList rows={diff} testIdPrefix="suggest-diff" />
          <div className="mt-2 flex justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setOpen(false)}
              data-testid="suggest-dismiss"
            >
              Dismiss
            </Button>
            <Button
              size="sm"
              onClick={() => {
                onApply(applicable);
                setOpen(false);
              }}
              disabled={diff.length === 0}
              data-testid="suggest-apply"
            >
              Apply
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// EffectiveArgvPanel — faux-terminal block at the bottom of the page that
// shows the FULL argv, argv[0] included, that the engine would be invoked
// with for the *persisted* settings (preview port 10000). Re-fetched on every
// save so the operator gets a one-second feedback loop: change → save → see it.
//
// Highlight on change uses the existing `row-flash-emerald` keyframe pattern
// from globals.css — gated by prefers-reduced-motion (S4 design principle #10).
// ---------------------------------------------------------------------------
function EffectiveArgvPanel({ modelId }: { modelId: string }) {
  const { data, error, isLoading } = useSWR<EffectiveArgvResponse>(
    `/api/models/${modelId}/effective-argv`,
    authFetchJSON,
    { revalidateOnFocus: false },
  );

  // Highlight tokens that changed vs. the previous render. We keep the prior
  // argv in a ref so the comparison survives across SWR mutations without
  // triggering an extra render.
  const prevArgv = useRef<string[] | null>(null);
  const [changedSet, setChangedSet] = useState<Set<number>>(new Set());
  useEffect(() => {
    if (!data) return;
    const prev = prevArgv.current;
    if (!prev) {
      prevArgv.current = data.argv;
      return;
    }
    const changed = new Set<number>();
    for (let i = 0; i < data.argv.length; i++) {
      if (data.argv[i] !== prev[i]) changed.add(i);
    }
    setChangedSet(changed);
    prevArgv.current = data.argv;
    // Clear the highlight after the flash animation completes. row-flash-emerald
    // is a 2s keyframe; we wait 2100ms to be safe. If a *second* save lands
    // mid-flash the new useEffect run replaces the set + restarts the timer.
    const t = setTimeout(() => setChangedSet(new Set()), 2100);
    return () => clearTimeout(t);
  }, [data]);

  const [copied, setCopied] = useState(false);
  function onCopy() {
    if (!data) return;
    // copyToClipboard (lib/utils.ts) tries navigator.clipboard then
    // falls back to a hidden-textarea + execCommand("copy") so the
    // button works on a LAN-HTTP deployment too (#149). The
    // success-flash is best-effort: we still flip `copied` even if the
    // call rejects, because there is no toast surface on this panel
    // and the textarea fallback leaves the argv selected as a
    // belt-and-suspenders for the operator.
    void copyToClipboard(data.argv.join(" ")).catch(() => {
      /* swallow — see comment above */
    });
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <Card data-testid="effective-argv-panel">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Effective argv</CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={onCopy}
            disabled={!data}
            data-testid="effective-argv-copy"
          >
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>
        <p className="text-xs text-slate-400" data-testid="effective-argv-subtitle">
          {/* The operator's question, verbatim: "why is Extra args empty but
              Effective argv has so many arguments?" The distinction was
              already correct and entirely invisible, which is the same defect
              family as a control whose effect is gated elsewhere -- so say it
              here, next to the thing that prompts the question. */}
          The <em>whole</em> command, generated from the fields above plus your
          Extra args — which is why it is longer than that box. Preview only:{" "}
          <code>--port 10000</code> is a placeholder; the supervisor binds a real
          ephemeral port at load time.
        </p>
      </CardHeader>
      <CardContent>
        {error && (
          <div
            className="text-sm text-red-400"
            data-testid="effective-argv-error"
          >
            Could not compute effective argv
            {error instanceof Error ? `: ${error.message}` : "."}
          </div>
        )}
        {isLoading && !data && (
          <Skeleton className="h-24 w-full" data-testid="effective-argv-skeleton" />
        )}
        {data && (
          <pre
            data-testid="effective-argv-pre"
            className="overflow-x-auto rounded-md border border-slate-700 bg-slate-950 p-3 text-xs leading-relaxed text-slate-200"
          >
            <code>
              {data.argv.map((tok, i) => (
                <span
                  key={i}
                  data-testid={
                    changedSet.has(i)
                      ? "effective-argv-tok-changed"
                      : "effective-argv-tok"
                  }
                  className={
                    changedSet.has(i) ? "row-flash-emerald" : undefined
                  }
                >
                  {i === 0 ? tok : ` ${tok}`}
                </span>
              ))}
            </code>
          </pre>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Shared diff helpers — used by both PresetStrip and SuggestPanel.
//
// computeDiff produces a stable ordered list of {key, before, after} rows;
// keys whose `after` value deep-equals `before` are skipped, so an empty
// diff means "nothing to apply" and we disable the Apply button.
// ---------------------------------------------------------------------------
interface DiffRow {
  key: string;
  before: unknown;
  after: unknown;
}

function computeDiff(
  draft: Draft,
  sparse: Record<string, unknown>,
): DiffRow[] {
  const rows: DiffRow[] = [];
  for (const k of PATCHABLE_KEYS) {
    if (!Object.prototype.hasOwnProperty.call(sparse, k)) continue;
    const before = draft[k];
    const after = sparse[k];
    if (eqValue(before, after)) continue;
    rows.push({ key: k, before, after });
  }
  return rows;
}

function DiffList({
  rows,
  testIdPrefix,
}: {
  rows: DiffRow[];
  testIdPrefix: string;
}) {
  if (rows.length === 0) {
    return (
      <p
        className="mt-2 text-xs text-slate-500"
        data-testid={`${testIdPrefix}-empty`}
      >
        No changes — your draft already matches.
      </p>
    );
  }
  return (
    <ul
      className="mt-2 space-y-1 text-xs"
      data-testid={`${testIdPrefix}-list`}
    >
      {rows.map((r) => (
        <li
          key={r.key}
          className="flex flex-wrap items-baseline gap-2 font-mono"
          data-testid={`${testIdPrefix}-row`}
          data-diff-key={r.key}
        >
          <span className="text-slate-400">{r.key}:</span>
          <span className="text-slate-500 line-through">{formatValue(r.before)}</span>
          <span className="text-slate-500">→</span>
          <span className="text-amber-300">{formatValue(r.after)}</span>
        </li>
      ))}
    </ul>
  );
}

function formatValue(v: unknown): string {
  if (v === null) return "null";
  if (typeof v === "string") return v === "" ? '""' : v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return JSON.stringify(v);
}

// ---------------------------------------------------------------------------
// SettingFieldFor — per-key dispatcher
// ---------------------------------------------------------------------------
//
// One switch maps a patchable key to the matching SettingField `kind` and
// the corresponding typed setter. Pulling this out of the parent JSX keeps
// the render block readable and confines the discriminated-union narrowing
// to one place. The `setDraft` calls all use the functional form so two
// keystrokes that land in the same render tick don't drop the first one.
//
function SettingFieldFor({
  fieldKey,
  hint,
  draft,
  setDraft,
  disabled,
  gpus,
  gpuIndicesEmpty,
  modelId,
}: {
  fieldKey: FieldKey;
  hint: import("@/lib/settings-hints").FieldHint;
  draft: Draft;
  setDraft: (updater: (d: Draft) => Draft) => void;
  disabled: boolean;
  gpus: GpuInfo[];
  gpuIndicesEmpty: boolean;
  /** This row's id, so its own loaded engine is not reported as a competing
   *  occupant of the card it is running on. */
  modelId: string;
}) {
  // Local updater factory — narrows TS to a per-key setter that the
  // SettingField onChange callbacks can accept directly.
  function set<K extends PatchableKey>(k: K, v: Draft[K]) {
    setDraft((d) => ({ ...d, [k]: v }));
  }

  switch (fieldKey) {
    case "served_model_name":
    case "hf_repo":
    case "hf_revision":
      return (
        <SettingField
          kind="text"
          field={hint}
          value={draft[fieldKey]}
          onChange={(v) => set(fieldKey, v)}
          disabled={disabled}
        />
      );
    case "mmproj_filename":
      return (
        <SettingField
          kind="text"
          field={hint}
          value={draft.mmproj_filename ?? ""}
          onChange={(v) => set("mmproj_filename", v.trim() === "" ? null : v)}
          disabled={disabled}
        />
      );
    case "n_gpu_layers":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.n_gpu_layers}
          // A cleared box means "omit the flag" -- llama.cpp's own auto/--fit
          // sizing decides -- which is NOT the same as 0 (everything on CPU).
          // Null has to survive the round trip or the default becomes
          // unreachable once an operator has typed a number.
          onChange={(v) => set("n_gpu_layers", v)}
          disabled={disabled}
        />
      );
    case "tensor_parallel_size":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.tensor_parallel_size}
          onChange={(v) =>
            // tensor_parallel_size is non-null in the DB schema (int NOT
            // NULL), so clamp null → 1 here. Backend would 422 on null.
            set("tensor_parallel_size", v === null ? 1 : v)
          }
          min={1}
          step={1}
          disabled={disabled}
        />
      );
    case "max_model_len":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.max_model_len}
          onChange={(v) => set("max_model_len", v)}
          min={1}
          step={1}
          disabled={disabled}
        />
      );
    case "gpu_memory_utilization":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.gpu_memory_utilization}
          onChange={(v) =>
            // Same nullability story as tensor_parallel_size — schema is
            // NOT NULL, so cleared input falls back to the vLLM default
            // of 0.9.
            set("gpu_memory_utilization", v === null ? 0.9 : v)
          }
          min={0.05}
          max={1.0}
          step={0.05}
          disabled={disabled}
        />
      );
    case "dtype":
      return (
        <SettingField
          kind="select"
          field={hint}
          value={draft.dtype}
          onChange={(v) => set("dtype", v)}
          options={DTYPE_OPTIONS}
          allowNull
          disabled={disabled}
        />
      );
    case "supports_vision":
    case "supports_tools":
    case "supports_reasoning":
      return (
        <SettingField
          kind="tristate"
          field={hint}
          value={draft[fieldKey]}
          onChange={(v) => set(fieldKey, v)}
          disabled={disabled}
        />
      );
    case "trust_remote_code":
      return (
        <SettingField
          kind="boolean"
          field={hint}
          value={draft.trust_remote_code}
          onChange={(v) => set("trust_remote_code", v)}
          disabled={disabled}
        />
      );
    case "gpu_indices":
      return (
        <div className="space-y-1">
          <SettingField
            kind="gpu-set"
            field={hint}
            value={draft.gpu_indices}
            gpus={gpus}
            excludeModelId={modelId}
            // Placement and parallel size move together. The PATCH validates
            // the MERGED row and holds it to register's cross-field rule
            // (tensor_parallel_size == len(gpu_indices)), so sending the
            // selection on its own is a 422 -- and a row where the two disagree
            // could not load anyway. Both keys are dirty afterwards, so both
            // are sent. Shown, not hidden: the tensor_parallel_size field sits
            // in this same Compute section and updates as you tick.
            onChange={(v) =>
              setDraft((d) => ({
                ...d,
                gpu_indices: v,
                tensor_parallel_size: v.length,
              }))
            }
            disabled={disabled}
          />
          {gpuIndicesEmpty && (
            <p
              role="alert"
              className="text-xs text-red-400"
              data-testid="gpu-indices-empty-error"
            >
              Select at least one GPU.
            </p>
          )}
        </div>
      );
    case "flash_attn":
    case "cache_type_k":
    case "cache_type_v": {
      // Backed by `extra_args`, not by a column. The control reads the flag
      // out of the list and writes it back into the same list, so the value
      // has exactly ONE home and cannot end up both here and in the free-text
      // box below. setManagedArg REPLACES; it never appends a second copy.
      const spec = LLAMACPP_MANAGED_FLAGS.find((f) => f.key === fieldKey)!;
      const { managed } = splitManagedArgs(draft.extra_args);
      return (
        <SettingField
          kind="select"
          field={hint}
          value={managed[fieldKey] ?? null}
          options={[...spec.values]}
          // "default" clears the flag entirely rather than writing the
          // engine's own default out. Emitting a value for an untouched
          // control would change the launch command of every existing row --
          // and a default the operator cannot get back to is the settings bug
          // this codebase keeps rediscovering.
          allowNull
          onChange={(v) =>
            set("extra_args", setManagedArg(draft.extra_args, fieldKey, v ?? undefined))
          }
          disabled={disabled}
        />
      );
    }
    case "extra_args":
      return (
        <SettingField
          kind="string-list"
          field={hint}
          // Only what the controls above do NOT own. A managed flag lives in
          // exactly one place, so an operator who types `--flash-attn on` here
          // finds it in the Flash attention control on the next load -- moved,
          // never duplicated and never silently dropped.
          value={splitManagedArgs(draft.extra_args).rest}
          onChange={(v) =>
            // Re-attach the managed flags the box never saw, so editing free
            // text cannot clear a control the operator did not touch.
            set(
              "extra_args",
              mergeManagedArgs(v, splitManagedArgs(draft.extra_args).managed),
            )
          }
          disabled={disabled}
        />
      );
    case "extra_env":
      return (
        <SettingField
          kind="kv-map"
          field={hint}
          value={draft.extra_env}
          onChange={(v) => set("extra_env", v)}
          disabled={disabled}
        />
      );
  }
}
