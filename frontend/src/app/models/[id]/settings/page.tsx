"use client";

// /ui/models/<id>/settings — the per-model engine + routing settings.
//
// Layout and attention flow: docs/superpowers/plans/2026-10-04-settings-redesign.md.
// The page answers "what can I change right now?" first. Two bands:
//   - ENGINE (Layout, Memory & context, llama.cpp engine, Name & source,
//     Advanced): applies on the next load; locked while the model is loaded.
//   - LIVE (Replica routing, Capabilities): applies immediately, even loaded.
// The band you can act on renders first (Live first when loaded, Engine first
// otherwise); field order inside a band never changes. The reload rule is said
// once per band and once per change in the save bar, not once per field.
//
// Every save / validation / carry rule is unchanged from the single-form
// page this replaced: dirty-subset PATCH, tp × dp == GPU count, the
// loaded-editable carve-out, the two kinds of 409, list-shaped 422s.

import Link from "next/link";
import { use, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import useSWR, { useSWRConfig } from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { Skeleton } from "@/components/ui/skeleton";
import { LayoutNotice } from "@/components/models/layout-notice";
import { MODEL_HINTS, type FieldHint } from "@/lib/settings-hints";
import { fieldAppliesTo } from "@/lib/backend-fields";
import {
  LLAMACPP_MANAGED_FLAGS,
  mergeManagedArgs,
  setManagedArg,
  splitManagedArgs,
} from "@/lib/llamacpp-args";
import {
  ARG_BACKED_KEYS,
  LOADED_EDITABLE_KEYS,
  PATCHABLE_KEYS,
  dirtyKeys,
  eqValue,
  formatSettingValue,
  labelFor,
  snapshotToDraft,
  type ArgBackedKey,
  type Draft,
  type FieldKey,
  type ModelSettings,
  type PatchableKey,
} from "@/lib/model-settings";
import { gpuMemoryReadout, layoutInvalidReason, tpDpMismatch } from "@/lib/settings-layout";
import { SettingField } from "@/components/settings/setting-field";
import { SettingsSection } from "@/components/settings-section";
import { type GpuInfo } from "@/components/gpu/gpu-checklist";
import { useBreadcrumb } from "@/lib/use-breadcrumb";
import { cn } from "@/lib/utils";
import { LayoutDiagram } from "@/components/models/settings/layout-diagram";
import { SaveBar, type SaveBarRow } from "@/components/models/settings/save-bar";
import { StatusStrip } from "@/components/models/settings/status-strip";
import { PresetStrip } from "@/components/models/settings/preset-strip";
import { SuggestPanel } from "@/components/models/settings/suggest-panel";
import { EffectiveArgvPanel } from "@/components/models/settings/effective-argv-panel";
import { computeDiff } from "@/components/models/settings/diff-list";

/**
 * Keep tensor_parallel_size × data_parallel_size == number of GPUs when the
 * GPU selection changes. If the current replica count still divides the new
 * count it is kept; otherwise it falls back to 1 replica (all GPUs in TP).
 */
function layoutForGpuCount(n: number, dp: number): { dp: number; tp: number } {
  if (n > 0 && dp >= 1 && n % dp === 0) return { dp, tp: n / dp };
  return { dp: 1, tp: n };
}

const DTYPE_OPTIONS = ["auto", "float16", "bfloat16", "float32"];

// The input the save bar's "Fix" focuses when the replica count is invalid.
const DP_INPUT_ID = "dp";
const TP_INPUT_ID = "tp";

// ---------------------------------------------------------------------------
// Section grouping. Each key appears in exactly one section. Section order and
// key order within a section are the visual order (plan §8). `title` is the
// toggle's accessible name; `displayTitle` the visible text when it differs.
// ---------------------------------------------------------------------------
type Band = "engine" | "live";

interface SummaryCtx {
  draft: Draft;
  data: ModelSettings;
  gpus: GpuInfo[];
}

interface SectionGroup {
  title: string;
  displayTitle?: string;
  testId: string;
  band: Band;
  keys: ReadonlyArray<FieldKey>;
  /** One-line value summary of the DRAFT, shown while collapsed. */
  summary: (ctx: SummaryCtx) => string;
  defaultOpen: (isLoaded: boolean) => boolean;
  /** Grid columns for the section's fields (default: auto-fit ≥ 200px). */
  grid?: string;
}

// Below 640px fields stack, except the three capability selects (2 + 1).
const GRID_DEFAULT = "grid-cols-[repeat(auto-fit,minmax(200px,1fr))] max-sm:grid-cols-1";
const GRID_CAPS = "grid-cols-[repeat(auto-fit,minmax(150px,1fr))] max-sm:grid-cols-2";

// Fields that take the whole row of their section's grid.
const WIDE_KEYS: ReadonlySet<FieldKey> = new Set<FieldKey>([
  "gpu_indices",
  "trust_remote_code",
  "extra_args",
  "extra_env",
  "mmproj_filename",
]);

const nf = new Intl.NumberFormat("en-US");
const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

function layoutSummary({ draft, data, gpus }: SummaryCtx): string {
  const n = draft.gpu_indices.length;
  if (n === 0) return "no GPUs selected";
  const names = new Set(
    draft.gpu_indices.map((i) => gpus.find((g) => g.index === i)?.name).filter(Boolean),
  );
  const which = names.size === 1 ? `${n} × ${[...names][0]}` : plural(n, "GPU", "GPUs");
  if (!fieldAppliesTo("data_parallel_size", data.backend)) return `${which} · layers split`;
  const dp = Math.max(1, draft.data_parallel_size);
  if (n % dp !== 0) return `${which} · invalid layout`;
  return `${which} · ${plural(dp, "replica", "replicas")} × ${plural(n / dp, "GPU", "GPUs")}`;
}

function memorySummary({ draft, data }: SummaryCtx): string {
  const parts: string[] = [];
  if (fieldAppliesTo("gpu_memory_utilization", data.backend))
    parts.push(`${draft.gpu_memory_utilization.toFixed(2)} of VRAM`);
  parts.push(
    draft.max_model_len === null
      ? "context from the model"
      : `${nf.format(draft.max_model_len)} tokens`,
  );
  if (fieldAppliesTo("dtype", data.backend)) parts.push(`dtype ${draft.dtype ?? "default"}`);
  if (fieldAppliesTo("n_gpu_layers", data.backend))
    parts.push(
      draft.n_gpu_layers === null ? "offload auto" : `${draft.n_gpu_layers} layers offloaded`,
    );
  return parts.join(" · ");
}

function llamacppSummary({ draft }: SummaryCtx): string {
  const { managed } = splitManagedArgs(draft.extra_args);
  return [
    `flash attention ${managed.flash_attn ?? "default"}`,
    `KV cache ${managed.cache_type_k ?? "default"} / ${managed.cache_type_v ?? "default"}`,
  ].join(" · ");
}

function advancedSummary({ draft, data }: SummaryCtx): string {
  const rest = splitManagedArgs(draft.extra_args).rest;
  const env = Object.keys(draft.extra_env).length;
  const parts = [
    rest.length ? plural(rest.length, "extra arg", "extra args") : "no extra args",
    env ? plural(env, "env var", "env vars") : "no env vars",
  ];
  if (fieldAppliesTo("trust_remote_code", data.backend))
    parts.push(draft.trust_remote_code ? "remote code trusted" : "remote code not trusted");
  if (fieldAppliesTo("mmproj_filename", data.backend))
    parts.push(draft.mmproj_filename ? "vision projector set" : "no vision projector");
  return parts.join(" · ");
}

const triWord = (v: boolean | null) => (v === null ? "auto" : v ? "yes" : "no");

const SECTION_GROUPS: ReadonlyArray<SectionGroup> = [
  {
    title: "Layout",
    testId: "section-compute",
    band: "engine",
    keys: ["gpu_indices", "data_parallel_size", "tensor_parallel_size"],
    summary: layoutSummary,
    defaultOpen: (loaded) => !loaded,
  },
  {
    // aria-label stays "Memory" (tests find `button /memory/i` in it).
    title: "Memory",
    displayTitle: "Memory & context",
    testId: "section-memory",
    band: "engine",
    keys: ["gpu_memory_utilization", "max_model_len", "dtype", "n_gpu_layers"],
    summary: memorySummary,
    defaultOpen: (loaded) => !loaded,
  },
  {
    // Absent entirely on a vLLM row -- every key in it is llama.cpp's, so the
    // section would open onto nothing.
    title: "llama.cpp engine",
    testId: "section-llamacpp",
    band: "engine",
    keys: ["flash_attn", "cache_type_k", "cache_type_v"],
    summary: llamacppSummary,
    defaultOpen: (loaded) => !loaded,
  },
  {
    title: "Name & source",
    testId: "section-identity",
    band: "engine",
    keys: ["served_model_name", "hf_repo", "hf_revision"],
    summary: ({ draft }) => `${draft.served_model_name} · ${draft.hf_repo} @ ${draft.hf_revision}`,
    defaultOpen: () => false,
  },
  {
    title: "Advanced",
    testId: "section-advanced",
    band: "engine",
    keys: ["trust_remote_code", "extra_args", "extra_env", "mmproj_filename"],
    summary: advancedSummary,
    defaultOpen: () => false,
  },
  {
    // Rendered only when data_parallel_size > 1; below that one line says how
    // to turn it on.
    title: "Replica routing",
    testId: "section-dp-routing",
    band: "live",
    keys: ["dp_affinity_enabled", "dp_spill_threshold"],
    summary: ({ draft }) =>
      `affinity ${draft.dp_affinity_enabled ? "on" : "off"} · spill ${draft.dp_spill_threshold ?? "auto"}`,
    defaultOpen: () => true,
  },
  {
    title: "Capabilities",
    testId: "section-capabilities",
    band: "live",
    keys: ["supports_vision", "supports_tools", "supports_reasoning"],
    summary: ({ draft }) =>
      `vision ${triWord(draft.supports_vision)} · tools ${triWord(draft.supports_tools)} · reasoning ${triWord(draft.supports_reasoning)}`,
    defaultOpen: () => true,
    grid: GRID_CAPS,
  },
];

/** Section testId → the section that renders `key`. */
function sectionOf(key: string): string | undefined {
  return SECTION_GROUPS.find((g) => (g.keys as readonly string[]).includes(key))?.testId;
}

/**
 * Is a field's draft value different from the snapshot? `extra_args` is ONE
 * column serving two sections, so it is split in halves: the llama.cpp
 * controls own the managed flags, Extra args owns the rest. Counting the whole
 * column in both would report a single edit twice, in two places.
 */
function fieldDirty(k: FieldKey, draft: Draft, snap: Draft): boolean {
  if ((ARG_BACKED_KEYS as readonly string[]).includes(k)) {
    const key = k as ArgBackedKey;
    return (
      splitManagedArgs(draft.extra_args).managed[key] !==
      splitManagedArgs(snap.extra_args).managed[key]
    );
  }
  if (k === "extra_args")
    return !eqValue(splitManagedArgs(draft.extra_args).rest, splitManagedArgs(snap.extra_args).rest);
  return !eqValue(draft[k as PatchableKey], snap[k as PatchableKey]);
}

/**
 * The save bar's rows, one per changed control, labelled. Built from
 * dirtyKeys with the same extra_args half-split as the section counts; `live`
 * comes from LOADED_EDITABLE_KEYS, the one list the backend mirrors.
 */
function saveBarRows(draft: Draft, snap: Draft): SaveBarRow[] {
  const rows: SaveBarRow[] = [];
  for (const k of dirtyKeys(draft, snap)) {
    if (k === "extra_args") {
      const d = splitManagedArgs(draft.extra_args);
      const s = splitManagedArgs(snap.extra_args);
      let any = false;
      for (const mk of ARG_BACKED_KEYS) {
        if (d.managed[mk] === s.managed[mk]) continue;
        any = true;
        rows.push({
          key: mk,
          label: labelFor(mk),
          before: formatSettingValue(mk, s.managed[mk] ?? null),
          after: formatSettingValue(mk, d.managed[mk] ?? null),
          live: false,
        });
      }
      if (any && eqValue(d.rest, s.rest)) continue;
      rows.push({
        key: k,
        label: labelFor(k),
        before: formatSettingValue(k, s.rest),
        after: formatSettingValue(k, d.rest),
        live: false,
      });
      continue;
    }
    rows.push({
      key: k,
      label: labelFor(k),
      before: formatSettingValue(k, snap[k]),
      after: formatSettingValue(k, draft[k]),
      live: LOADED_EDITABLE_KEYS.includes(k),
    });
  }
  return rows;
}

const STATUS_PILL: Record<ModelSettings["status"], { text: string; tone: "ok" | "idle" | "busy" | "bad" }> = {
  loaded: { text: "Loaded", tone: "ok" },
  registered: { text: "Not loaded", tone: "idle" },
  pulled: { text: "Not loaded", tone: "idle" },
  pulling: { text: "Pulling", tone: "busy" },
  loading: { text: "Loading…", tone: "busy" },
  unloading: { text: "Unloading…", tone: "busy" },
  failed: { text: "Failed", tone: "bad" },
};

const PILL_TONE = {
  ok: "bg-vw-ok-bg/55 text-vw-ok-fg",
  idle: "bg-chat-surface-2 text-chat-muted",
  busy: "bg-vw-amber-bg/60 text-vw-amber-fg",
  bad: "bg-vw-danger-bg/40 text-vw-danger-fg",
} as const;

export default function ModelSettingsPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  // Same Next.js 15 unwrap pattern as /models/[id]/page.tsx.
  const { id } = use(params);
  const key = `/api/models/${id}/settings`;
  const { data, error, isLoading, mutate } = useSWR<ModelSettings>(key, authFetchJSON, {
    // No polling — this is an edit form, polling would fight the user's
    // typing by rewriting the snapshot mid-keystroke. Save/Reset go
    // through explicit mutate() calls below.
    revalidateOnFocus: false,
  });
  // Name the parent crumb (/models/[id]) after the model — on a direct load
  // of this page nothing else has told the breadcrumb what it is called.
  useBreadcrumb({ path: `/models/${id}`, title: data?.served_model_name ?? (error ? id : undefined) });
  const { data: gpuData } = useSWR<{ gpus: GpuInfo[]; probed_at: string; probe_error: string | null }>(
    "/api/system/gpus",
    authFetchJSON,
  );
  const gpus = useMemo(() => gpuData?.gpus ?? [], [gpuData]);
  // Global mutate for cross-page cache coherence after a successful save.
  const { mutate: globalMutate } = useSWRConfig();

  const [draft, setDraft] = useState<Draft | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  // 409 is special: surfaced in the status strip (amber), not as the inline
  // save error. Keeping it separate also lets the pre-emptive status guard
  // reuse the same strip without conflating "I tried to save" with "the page
  // noticed status=loaded on first load".
  const [conflict, setConflict] = useState(false);
  // Per-section expansion the operator (or a preset / suggestion / an error)
  // chose. Absent → the section's state-dependent default.
  const [openState, setOpenState] = useState<Record<string, boolean>>({});

  // Initialize draft from snapshot exactly once per load cycle. A naive
  // useEffect dep on `data` would clobber the user's in-flight edits on
  // every SWR revalidation. After a successful PATCH seededFor is nulled so
  // the next snapshot re-seeds the draft.
  const seededFor = useRef<string | null>(null);
  useEffect(() => {
    if (!data) return;
    const sig = JSON.stringify(snapshotToDraft(data));
    if (seededFor.current === sig) return;
    seededFor.current = sig;
    setDraft(snapshotToDraft(data));
  }, [data]);

  // Fields are disabled while a save is in flight, and a browser drops focus
  // to <body> when the focused control becomes disabled — after Ctrl+S the
  // operator would lose their place. Remember the control and hand focus back
  // once the save settles, unless focus has since moved somewhere real.
  const refocusAfterSave = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (saving) return;
    const el = refocusAfterSave.current;
    refocusAfterSave.current = null;
    if (!el || !el.isConnected) return;
    const active = document.activeElement;
    if (active === null || active === document.body) el.focus();
  }, [saving]);

  // A section holding a validation error opens, so the error is never hidden
  // behind a collapsed header. Today both layout rules live in Layout.
  const layoutInvalid =
    draft !== null &&
    (draft.gpu_indices.length === 0 ||
      draft.gpu_indices.length % Math.max(1, draft.data_parallel_size) !== 0 ||
      tpDpMismatch(draft.gpu_indices.length, draft.data_parallel_size, draft.tensor_parallel_size, data?.backend));
  useEffect(() => {
    if (layoutInvalid)
      setOpenState((s) => (s["section-compute"] ? s : { ...s, "section-compute": true }));
  }, [layoutInvalid]);

  // ---- 404 — model not found. Mirror the detail-page pattern. ----------
  const errStatus = (error as (Error & { status?: number }) | undefined)?.status;
  if (errStatus === 404) {
    return (
      <div className="rounded-xl border border-vw-rule-soft/70 bg-chat-surface/55 px-4 py-3.5">
        <h1 className="m-0 text-[15px] font-semibold text-chat-fg">Model not found</h1>
        <p className="mt-1.5 text-sm text-chat-muted">
          The model <span className="font-mono">{id}</span> does not exist. It may have been
          deleted.
        </p>
      </div>
    );
  }

  if (isLoading || (!data && !error)) {
    // Shape-matched: header, strip, two cards beside the rail.
    return (
      <div className="space-y-4" data-testid="settings-skeleton">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-11 w-full" />
        <div className="grid gap-6 min-[1100px]:grid-cols-[minmax(0,1fr)_360px]">
          <div className="space-y-3">
            <Skeleton className="h-64 w-full" />
            <Skeleton className="h-40 w-full" />
          </div>
          <Skeleton className="h-72 w-full" />
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div
        role="alert"
        className="rounded-xl border border-chat-negative/60 bg-vw-danger-bg/25 px-4 py-3 text-sm text-vw-danger-fg"
      >
        Failed to load settings
        {error instanceof Error ? `: ${error.message}` : "."}
      </div>
    );
  }

  if (!data || !draft) return null;

  // Pre-emptive guard: a loaded model would 409 on PATCH anyway, so engine
  // inputs are disabled up front and the strip says so. The form still
  // renders so the operator can see what's in place.
  //
  // The carve-out mirrors the backend's (app/settings/routes_api.py): a patch
  // made up ENTIRELY of loaded-editable keys (capability flags + replica
  // routing) is allowed on a loaded model.
  const isLoaded = data.status === "loaded";
  const allDisabled = isLoaded || saving;
  const liveDisabled = saving;
  const snap = snapshotToDraft(data);
  const dirty = dirtyKeys(draft, snap);
  const loadedEditableOnly =
    dirty.length > 0 && dirty.every((k) => LOADED_EDITABLE_KEYS.includes(k));
  // Per-model gpu_indices requires >=1 selection: a model with zero GPUs
  // would be handed an empty CUDA_VISIBLE_DEVICES and never schedule.
  const gpuIndicesEmpty = draft.gpu_indices.length === 0;
  // #286: replicas must divide the GPU count (tp × dp == len(gpu_indices)).
  const dpInvalid = draft.gpu_indices.length % Math.max(1, draft.data_parallel_size) !== 0;
  // The server also requires tensor_parallel_size × dp == GPUs (422 otherwise).
  const tpInvalid = tpDpMismatch(
    draft.gpu_indices.length,
    draft.data_parallel_size,
    draft.tensor_parallel_size,
    data.backend,
  );
  const canSave =
    !saving && dirty.length > 0 && !conflict && !gpuIndicesEmpty && !dpInvalid && !tpInvalid &&
    (!isLoaded || loadedEditableOnly);

  // A loaded row can still hold an engine edit: the draft re-seeds only when
  // a patchable field changes, so a refetch that merely flips status to
  // loaded keeps it. Say why Save is off instead of a silent disabled button;
  // Reset (still enabled) is the way out, or unloading.
  const loadedEngineDraft = isLoaded && dirty.length > 0 && !loadedEditableOnly;
  const invalidReason =
    layoutInvalidReason(
      draft.gpu_indices,
      draft.data_parallel_size,
      data.backend,
      draft.tensor_parallel_size,
    ) ??
    (loadedEngineDraft ? "engine changes can't be saved while loaded — Reset or unload" : null);
  const fixTargetId = gpuIndicesEmpty
    ? gpus.length > 0
      ? `gpu-checklist-${gpus[0].index}`
      : undefined
    : dpInvalid
      ? DP_INPUT_ID
      : tpInvalid
        ? TP_INPUT_ID
        : undefined;

  async function onSave() {
    if (!data || !draft) return;
    // Compute the dirty-set BEFORE flipping the spinner — with nothing to
    // save we return synchronously, so Save can't get stuck on "Saving…".
    const dirtyNow = dirtyKeys(draft, snapshotToDraft(data));
    if (dirtyNow.length === 0) return;
    const active = document.activeElement;
    refocusAfterSave.current =
      active instanceof HTMLElement && active !== document.body ? active : null;
    setSaving(true);
    setSaveError(null);
    try {
      const body: Partial<Draft> = {};
      for (const k of dirtyNow) {
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
          // A 422 carries FastAPI's list of field errors: the PATCH validates
          // the whole merged row against register's rules (#262).
          else if (Array.isArray(j?.detail)) {
            const msgs = (j.detail as { msg?: unknown }[])
              .map((e) => (typeof e?.msg === "string" ? e.msg : null))
              .filter((m): m is string => m !== null);
            if (msgs.length) detail = msgs.join("; ");
          }
        } catch {
          /* non-JSON body */
        }
        // Two things answer 409: the unload-first guard, and a
        // served_model_name another row already holds. Only the first is the
        // "unload" strip; the second is an ordinary save error.
        if (r.status === 409 && !detail.includes("already exists")) {
          setConflict(true);
          return;
        }
        setSaveError(detail);
        return;
      }
      // Success — clear the seeded signal so the next revalidate re-seeds
      // the draft to the freshly-saved snapshot.
      seededFor.current = null;
      await mutate();
      // Fire-and-forget cross-page cache busts (dashboard list, detail page)
      // and the effective command, which is keyed off the saved settings.
      void globalMutate("/api/models");
      void globalMutate(`/api/models/${id}`);
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
    // Clear a 409-sourced strip too — the user chose to discard their edits.
    // The pre-emptive loaded strip stays: `isLoaded` is server state.
    setConflict(false);
  }

  // Apply a sparse settings dict (from preset or suggest) on top of the
  // current draft. Only keys in PATCHABLE_KEYS are accepted; unknown keys are
  // ignored so a future backend addition doesn't crash an old FE. Each value
  // goes through the same per-kind coercion the SettingField onChange
  // handlers do.
  function applySparseSettings(sparse: Record<string, unknown>) {
    setDraft((d) => {
      if (d === null) return d;
      const next: Draft = { ...d };
      for (const k of PATCHABLE_KEYS) {
        if (!Object.prototype.hasOwnProperty.call(sparse, k)) continue;
        const v = sparse[k];
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
            else if (typeof v === "string" && DTYPE_OPTIONS.includes(v)) next.dtype = v;
            break;
          case "trust_remote_code":
            if (typeof v === "boolean") next.trust_remote_code = v;
            break;
          case "gpu_indices":
            if (Array.isArray(v) && v.every((x) => typeof x === "number" && Number.isFinite(x))) {
              next.gpu_indices = (v as number[]).map((x) => Math.floor(x));
              // Carry the parallel layout with the placement, exactly as the
              // GPU picker's own onChange does: the PATCH validates the MERGED
              // row against tensor_parallel_size × dp == len(gpu_indices).
              const lay = layoutForGpuCount(next.gpu_indices.length, next.data_parallel_size);
              next.data_parallel_size = lay.dp;
              next.tensor_parallel_size = lay.tp;
            }
            break;
          case "data_parallel_size":
            if (typeof v === "number" && Number.isInteger(v) && v >= 1) {
              next.data_parallel_size = v;
              const n = next.gpu_indices.length;
              if (n > 0 && n % v === 0) next.tensor_parallel_size = n / v;
            }
            break;
          case "dp_affinity_enabled":
            if (typeof v === "boolean") next.dp_affinity_enabled = v;
            break;
          case "dp_spill_threshold":
            if (v === null) next.dp_spill_threshold = null;
            else if (typeof v === "number" && Number.isFinite(v))
              next.dp_spill_threshold = Math.max(1, Math.floor(v));
            break;
          case "supports_vision":
          case "supports_tools":
          case "supports_reasoning":
            if (v === null || typeof v === "boolean") next[k] = v;
            break;
          case "n_gpu_layers":
            if (v === null) next.n_gpu_layers = null;
            else if (typeof v === "number" && Number.isFinite(v))
              next.n_gpu_layers = Math.max(0, Math.floor(v));
            break;
          case "mmproj_filename":
            if (v === null) next.mmproj_filename = null;
            else if (typeof v === "string") next.mmproj_filename = v.trim() === "" ? null : v;
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
              Object.values(v as Record<string, unknown>).every((x) => typeof x === "string")
            )
              next.extra_env = v as Record<string, string>;
            break;
        }
      }
      return next;
    });
  }

  // Preset / suggestion entry point: apply, and open every section whose
  // keys the sparse dict actually changes so the operator sees the result.
  function applyAndReveal(sparse: Record<string, unknown>) {
    if (draft) {
      const touched = computeDiff(draft, sparse)
        .map((r) => sectionOf(r.key))
        .filter((s): s is string => s !== undefined);
      if (touched.length)
        setOpenState((s) => {
          const next = { ...s };
          for (const t of touched) next[t] = true;
          return next;
        });
    }
    applySparseSettings(sparse);
  }

  const pill = STATUS_PILL[data.status] ?? { text: data.status, tone: "idle" as const };
  const backendLabel = data.backend === "llamacpp" ? "llama.cpp" : "vLLM";

  function renderSection(group: SectionGroup): ReactNode {
    if (!data || !draft) return null;
    const visible = group.keys.filter((k) => MODEL_HINTS[k] && fieldAppliesTo(k, data.backend));
    // A section every one of whose keys belongs to a DIFFERENT backend is not
    // rendered at all: a header that opens onto nothing is a dead control.
    // Memory is exempt because it also carries the Suggest row.
    if (visible.length === 0 && group.testId !== "section-memory") return null;
    if (group.testId === "section-dp-routing" && draft.data_parallel_size <= 1) {
      return (
        <p
          key={group.testId}
          data-testid="dp-routing-needs-replicas"
          className="m-0 rounded-xl border border-dashed border-vw-rule-soft/70 px-4 py-3 text-[12.5px] leading-[1.45] text-chat-muted"
        >
          Set Data-parallel replicas above 1 to enable replica routing.
        </p>
      );
    }
    const dirtyCount = group.keys.filter((k) => fieldDirty(k, draft, snap)).length;
    const open = openState[group.testId] ?? group.defaultOpen(isLoaded);
    const readout = gpuMemoryReadout(
      draft.gpu_memory_utilization,
      draft.gpu_indices,
      gpus,
    );
    const readoutId = `${group.testId}-gib-readout`;

    return (
      <SettingsSection
        key={group.testId}
        title={group.title}
        displayTitle={group.displayTitle}
        testId={group.testId}
        variant="card"
        as="h3"
        summary={group.summary({ draft, data, gpus })}
        dirtyCount={dirtyCount}
        open={open}
        onOpenChange={(o) => setOpenState((s) => ({ ...s, [group.testId]: o }))}
      >
        {group.testId === "section-compute" && (
          <LayoutDiagram
            gpuIndices={draft.gpu_indices}
            dp={draft.data_parallel_size}
            tp={draft.tensor_parallel_size}
            backend={data.backend}
          />
        )}
        {group.testId === "section-memory" && (
          <SuggestPanel
            modelId={id}
            disabled={allDisabled}
            draft={draft}
            onApply={applyAndReveal}
          />
        )}
        {/* Explain, do not offer a dead control. --split-mode and --main-gpu
            are DERIVED from the GPU selection (decision D4), and
            --tensor-split is left to llama.cpp's memory-proportional default.
            A control for any of them would be a second, competing source for
            a fact the Layout section already owns. */}
        {group.testId === "section-llamacpp" && (
          <p
            data-testid="llamacpp-derived-note"
            className="m-0 text-[12.5px] leading-[1.45] text-chat-muted"
          >
            <code className="font-mono text-[12px]">--split-mode</code> and{" "}
            <code className="font-mono text-[12px]">--main-gpu</code> follow the GPU selection in
            Layout and are not set here.{" "}
            <code className="font-mono text-[12px]">--tensor-split</code> is left to
            llama.cpp&apos;s memory-proportional default. Anything else goes in Extra args.
          </p>
        )}
        {group.testId === "section-capabilities" && (
          <p className="m-0 text-[12.5px] leading-[1.45] text-chat-muted">
            What the chat offers for this model. Auto reads the model&apos;s config.
          </p>
        )}
        {visible.length > 0 && (
          <div className={cn("grid gap-x-5 gap-y-[18px]", group.grid ?? GRID_DEFAULT)}>
            {visible.map((k) => (
              <div key={k} className={cn("min-w-0", WIDE_KEYS.has(k) && "col-span-full")}>
                <SettingFieldFor
                  fieldKey={k}
                  // Plan §8: the GPU picker is labelled "GPUs" on this page.
                  hint={k === "gpu_indices" ? { ...MODEL_HINTS[k], label: "GPUs" } : MODEL_HINTS[k]}
                  draft={draft}
                  gpus={gpus}
                  setDraft={(updater) => setDraft((d) => (d === null ? d : updater(d)))}
                  disabled={LOADED_EDITABLE_KEYS.includes(k) ? liveDisabled : allDisabled}
                  dirty={fieldDirty(k, draft, snap)}
                  gpuIndicesEmpty={gpuIndicesEmpty}
                  modelId={id}
                  readout={
                    k === "gpu_memory_utilization" && readout ? (
                      <span
                        id={readoutId}
                        data-testid="gpu-memory-readout"
                        className="text-[12.5px] tabular-nums text-chat-muted"
                      >
                        ≈ <b className="font-semibold text-chat-fg">{readout.usedGib} GiB</b> of{" "}
                        {readout.totalGib} GiB per GPU
                      </span>
                    ) : undefined
                  }
                  readoutId={k === "gpu_memory_utilization" && readout ? readoutId : undefined}
                />
              </div>
            ))}
          </div>
        )}
      </SettingsSection>
    );
  }

  const engineBand = (
    <section
      key="engine"
      aria-labelledby="band-engine-h"
      data-testid="band-engine"
      className="flex flex-col"
    >
      <div className="mb-2.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2
          id="band-engine-h"
          className="m-0 flex items-center gap-2 text-xs font-semibold uppercase tracking-[.08em] text-chat-muted"
        >
          <svg
            width="13"
            height="13"
            viewBox="0 0 14 14"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            aria-hidden="true"
          >
            <rect x="2.5" y="6" width="9" height="6.5" rx="1.5" />
            <path d="M4.5 6V4.5a2.5 2.5 0 0 1 5 0V6" />
          </svg>
          Engine
        </h2>
        <span className="text-[12.5px] text-chat-muted">
          {isLoaded ? "Locked while loaded · applies on next load" : "Applies on next load"}
        </span>
      </div>
      <div className="flex flex-col gap-3">
        {SECTION_GROUPS.filter((g) => g.band === "engine").map(renderSection)}
      </div>
    </section>
  );

  const liveBand = (
    <section
      key="live"
      aria-labelledby="band-live-h"
      data-testid="band-live"
      className="flex flex-col"
    >
      <div className="mb-2.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2
          id="band-live-h"
          className="m-0 flex items-center gap-2 text-xs font-semibold uppercase tracking-[.08em] text-vw-ok-fg"
        >
          <span aria-hidden="true" className="h-[7px] w-[7px] rounded-full bg-vw-live" />
          Live
        </h2>
        <span className="text-[12.5px] text-chat-muted">Applies immediately, even while loaded</span>
      </div>
      <div className="flex flex-col gap-3">
        {SECTION_GROUPS.filter((g) => g.band === "live").map(renderSection)}
      </div>
    </section>
  );

  return (
    <div className="text-chat-fg">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h1 className="m-0 text-2xl font-semibold leading-tight tracking-[-0.01em]">Settings</h1>
          <div className="mt-1.5 flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-2 text-[13px] text-chat-muted">
            <span
              data-testid="settings-status-pill"
              className={cn(
                "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-semibold",
                PILL_TONE[pill.tone],
              )}
            >
              <span aria-hidden="true" className="h-[7px] w-[7px] rounded-full bg-current" />
              {pill.text}
            </span>
            <span className="inline-flex items-center whitespace-nowrap rounded-md border border-vw-rule-soft/80 px-2 py-px text-xs font-medium">
              {backendLabel}
            </span>
            {/* !font-mono: the retro themes force DM Sans onto every <span>
                (globals.css), which outranks a plain font-mono utility. */}
            <span className="min-w-0 !font-mono text-[12.5px] [overflow-wrap:anywhere]">
              {data.hf_repo}
              <span className="!font-mono"> @ </span>
              {data.hf_revision}
            </span>
          </div>
        </div>
        <Link
          href={`/models/${id}`}
          className="rounded-sm border-b border-vw-rule-soft text-[13px] text-chat-muted hover:text-chat-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent"
        >
          Model page
        </Link>
      </div>

      <div className="mt-4 flex flex-col gap-3">
        <StatusStrip status={data.status} conflict={conflict} modelId={id} />
        {data.layout_notice && data.layout_notice_level && (
          <LayoutNotice
            modelId={data.id}
            level={data.layout_notice_level}
            message={data.layout_notice}
            onDismissed={async () => {
              await mutate();
              // The model page shows the same notice from /api/models/{id}.
              void globalMutate(`/api/models/${id}`);
              void globalMutate("/api/models");
            }}
          />
        )}
      </div>

      <div className="mt-6 grid items-start gap-6 min-[1100px]:grid-cols-[minmax(0,1fr)_360px]">
        <div className="flex min-w-0 flex-col gap-8">
          {/* Presets could only produce locked keys on a loaded model, so the
              card is not rendered there at all (plan §2.1). */}
          {!isLoaded && (
            <PresetStrip disabled={saving} draft={draft} onApply={applyAndReveal} />
          )}
          {isLoaded ? [liveBand, engineBand] : [engineBand, liveBand]}
        </div>
        <aside aria-label="In effect" className="flex min-w-0 flex-col gap-4 min-[1100px]:sticky min-[1100px]:top-4">
          <EffectiveArgvPanel
            modelId={id}
            extraArgs={data.extra_args}
            stale={dirty.length > 0}
          />
        </aside>
      </div>

      <SaveBar
        rows={saveBarRows(draft, snap)}
        invalidReason={invalidReason}
        saving={saving}
        canSave={canSave}
        isLoaded={isLoaded}
        saveError={saveError}
        onSave={onSave}
        onReset={onReset}
        fixTargetId={fixTargetId}
        onFix={() => setOpenState((s) => ({ ...s, "section-compute": true }))}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// SettingFieldFor — per-key dispatcher
// ---------------------------------------------------------------------------
//
// One switch maps a field key to the matching SettingField `kind` and the
// corresponding typed setter. The `setDraft` calls all use the functional
// form so two keystrokes that land in the same render tick don't drop the
// first one. Every field uses the redesign anatomy: short hint with "More",
// no per-field reload badge (the band says it once), amber dirty dot.
//
function SettingFieldFor({
  fieldKey,
  hint,
  draft,
  setDraft,
  disabled,
  dirty,
  gpus,
  gpuIndicesEmpty,
  modelId,
  readout,
  readoutId,
}: {
  fieldKey: FieldKey;
  hint: FieldHint;
  draft: Draft;
  setDraft: (updater: (d: Draft) => Draft) => void;
  disabled: boolean;
  dirty: boolean;
  gpus: GpuInfo[];
  gpuIndicesEmpty: boolean;
  /** This row's id, so its own loaded engine is not reported as a competing
   *  occupant of the card it is running on. */
  modelId: string;
  /** Rendered between the control and its hint, linked as a description. */
  readout?: ReactNode;
  readoutId?: string;
}) {
  function set<K extends PatchableKey>(k: K, v: Draft[K]) {
    setDraft((d) => ({ ...d, [k]: v }));
  }
  const common = {
    field: hint,
    disabled,
    dirty,
    chrome: "token",
  } as const;

  switch (fieldKey) {
    case "served_model_name":
    case "hf_repo":
    case "hf_revision":
      return (
        <SettingField
          {...common}
          kind="text"
          value={draft[fieldKey]}
          onChange={(v) => set(fieldKey, v)}
        />
      );
    case "mmproj_filename":
      return (
        <SettingField
          {...common}
          kind="text"
          value={draft.mmproj_filename ?? ""}
          onChange={(v) => set("mmproj_filename", v.trim() === "" ? null : v)}
        />
      );
    case "n_gpu_layers":
      return (
        <SettingField
          {...common}
          kind="number"
          value={draft.n_gpu_layers}
          // A cleared box means "omit the flag" -- llama.cpp's own auto/--fit
          // sizing decides -- which is NOT the same as 0 (everything on CPU).
          onChange={(v) => set("n_gpu_layers", v)}
        />
      );
    case "tensor_parallel_size":
      return (
        <SettingField
          {...common}
          kind="number"
          id={TP_INPUT_ID}
          value={draft.tensor_parallel_size}
          // NOT NULL in the DB schema, so clamp null → 1 (backend 422s on null).
          onChange={(v) => set("tensor_parallel_size", v === null ? 1 : v)}
          min={1}
          step={1}
        />
      );
    case "data_parallel_size": {
      const n = draft.gpu_indices.length;
      const invalid = n % Math.max(1, draft.data_parallel_size) !== 0;
      return (
        <div className="flex flex-col gap-1.5">
          <SettingField
            {...common}
            kind="number"
            id={DP_INPUT_ID}
            value={draft.data_parallel_size}
            onChange={(v) =>
              setDraft((d) => {
                const next = v === null ? 1 : v;
                const count = d.gpu_indices.length;
                // tensor_parallel_size is derived: tp = GPUs / replicas. When
                // the count does not divide, tp is left alone and the inline
                // error below blocks Save.
                return {
                  ...d,
                  data_parallel_size: next,
                  tensor_parallel_size:
                    count > 0 && next >= 1 && count % next === 0 ? count / next : d.tensor_parallel_size,
                };
              })
            }
            min={1}
            step={1}
          />
          {invalid && (
            <p
              role="alert"
              className="m-0 text-[12.5px] font-medium text-vw-danger-fg"
              data-testid="dp-invalid-error"
            >
              Replicas must divide the number of selected GPUs ({n}).
            </p>
          )}
        </div>
      );
    }
    case "dp_affinity_enabled":
      return (
        <SettingField
          {...common}
          kind="boolean"
          variant="switch"
          value={draft.dp_affinity_enabled}
          onChange={(v) => set("dp_affinity_enabled", v)}
        />
      );
    case "dp_spill_threshold":
      return (
        <SettingField
          {...common}
          kind="number"
          value={draft.dp_spill_threshold}
          onChange={(v) => set("dp_spill_threshold", v)}
          min={1}
          step={1}
          placeholder="auto (max_num_seqs / 4)"
        />
      );
    case "max_model_len":
      return (
        <SettingField
          {...common}
          kind="number"
          value={draft.max_model_len}
          onChange={(v) => set("max_model_len", v)}
          min={1}
          step={1}
        />
      );
    case "gpu_memory_utilization":
      return (
        <SettingField
          {...common}
          kind="number"
          value={draft.gpu_memory_utilization}
          // NOT NULL in the schema, so a cleared input falls back to 0.9.
          onChange={(v) => set("gpu_memory_utilization", v === null ? 0.9 : v)}
          min={0.05}
          max={1.0}
          step={0.05}
          afterControl={readout}
          describedBy={readoutId}
        />
      );
    case "dtype":
      return (
        <SettingField
          {...common}
          kind="select"
          value={draft.dtype}
          onChange={(v) => set("dtype", v)}
          options={DTYPE_OPTIONS}
          allowNull
        />
      );
    case "supports_vision":
    case "supports_tools":
    case "supports_reasoning":
      return (
        <SettingField
          {...common}
          kind="tristate"
          value={draft[fieldKey]}
          onChange={(v) => set(fieldKey, v)}
        />
      );
    case "trust_remote_code":
      return (
        <SettingField
          {...common}
          kind="boolean"
          variant="switch"
          value={draft.trust_remote_code}
          onChange={(v) => set("trust_remote_code", v)}
        />
      );
    case "gpu_indices":
      return (
        <div className="flex flex-col gap-1.5">
          <SettingField
            {...common}
            kind="gpu-set"
            value={draft.gpu_indices}
            gpus={gpus}
            excludeModelId={modelId}
            // Placement and parallel size move together: the PATCH validates
            // the MERGED row (tensor_parallel_size × dp == len(gpu_indices)),
            // so sending the selection alone is a 422. Both keys are dirty
            // afterwards, so both are sent; the Replicas / Tensor-parallel
            // fields right below update as you tick.
            onChange={(v) =>
              setDraft((d) => {
                const lay = layoutForGpuCount(v.length, d.data_parallel_size);
                return { ...d, gpu_indices: v, data_parallel_size: lay.dp, tensor_parallel_size: lay.tp };
              })
            }
          />
          {gpuIndicesEmpty && (
            <p
              role="alert"
              className="m-0 text-[12.5px] font-medium text-vw-danger-fg"
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
      // has exactly ONE home. setManagedArg REPLACES; it never appends.
      const spec = LLAMACPP_MANAGED_FLAGS.find((f) => f.key === fieldKey)!;
      const { managed } = splitManagedArgs(draft.extra_args);
      return (
        <SettingField
          {...common}
          kind="select"
          value={managed[fieldKey] ?? null}
          options={[...spec.values]}
          // "default" clears the flag entirely rather than writing the
          // engine's own default out, so an untouched control changes no argv.
          allowNull
          onChange={(v) => set("extra_args", setManagedArg(draft.extra_args, fieldKey, v ?? undefined))}
        />
      );
    }
    case "extra_args":
      return (
        <SettingField
          {...common}
          kind="string-list"
          // Only what the controls above do NOT own: a managed flag lives in
          // exactly one place.
          value={splitManagedArgs(draft.extra_args).rest}
          onChange={(v) =>
            // Re-attach the managed flags the box never saw, so editing free
            // text cannot clear a control the operator did not touch.
            set("extra_args", mergeManagedArgs(v, splitManagedArgs(draft.extra_args).managed))
          }
        />
      );
    case "extra_env":
      return (
        <SettingField
          {...common}
          kind="kv-map"
          value={draft.extra_env}
          onChange={(v) => set("extra_env", v)}
        />
      );
  }
}
