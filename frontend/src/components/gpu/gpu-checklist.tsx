"use client";

// ---------------------------------------------------------------------------
// GpuChecklist — single source of truth for GPU-selection UX.
//
// Prop-driven and presentational: it never fetches. Callers pass the live
// inventory (`gpus`, from GET /api/system/gpus) and the configured selection
// (`selected`). The component renders one checkbox per present GPU and, for
// any configured index NOT in the inventory, a distinct "ghost row" so a
// missing GPU is surfaced and repairable rather than silently dropped.
//
// `onChange` always emits a sorted-ascending number[] so callers never have
// to re-sort, and dirty-tracking against a stored (sorted) value is stable.
// ---------------------------------------------------------------------------

/** One process holding memory on a GPU, as GET /api/system/gpus reports it. */
export interface GpuHolder {
  pid: number;
  memory_mib: number;
  process: string;
  /** `model` = a process this warden started and can name. */
  kind: "model" | "external";
  model_id: string | null;
  label: string | null;
}

export interface GpuInfo {
  index: number;
  name: string;
  memory_total_mib: number;
  memory_used_mib: number;
  utilization_pct: number;
  /** Optional: callers that fetch the full /api/system/gpus payload pass it
   *  through and get occupancy warnings; the settings page's slimmer probe
   *  omits it and simply gets none. */
  holders?: GpuHolder[];
}

interface GpuChecklistProps {
  gpus: GpuInfo[];
  selected: number[];
  onChange: (next: number[]) => void;
  disabled?: boolean;
  /** Forwarded to the component's root element as `aria-describedby`. Lets a
   *  caller (e.g. SettingField's chrome) wire the field hint to this control,
   *  since GpuChecklist owns its own per-GPU checkbox ids and can't consume an
   *  external input id. */
  describedById?: string;
  /** The setup wizard's `allowed_gpu_indices`, from GET /api/system/gpus.
   *
   *  `undefined`/`null` mean NO allowlist was recorded -- every GPU stays
   *  selectable. They do NOT mean "none allowed"; conflating the two would
   *  lock an operator out of a box whose setup draft predates the key.
   *
   *  A GPU outside the list is disabled rather than hidden, and told why:
   *  POST /api/models rejects it with a 400, so offering it as a live choice
   *  is a control that cannot work. An already-SELECTED out-of-list GPU stays
   *  enabled -- stored state can predate a narrowed allowlist, and the
   *  operator needs a way to clear the row the server is about to reject. */
  allowedIndices?: number[] | null;
  /** The model this checklist is editing, when there is one.
   *
   *  Its own engine holds the cards it is loaded on, and "GPU 0 is already
   *  serving qwen3.8-27b" while editing qwen3.8-27b is a false alarm. A
   *  warning that cries wolf is worse than none, because the operator learns
   *  to skip it — including on the occasion it is real. The Add-model wizard
   *  has no id yet and passes nothing. */
  excludeModelId?: string;
}

/** Models this warden started that currently hold memory on `gpu`. */
function occupyingModels(gpu: GpuInfo, excludeModelId?: string): string[] {
  return (gpu.holders ?? [])
    .filter((h) => h.kind === "model" && h.label && h.model_id !== excludeModelId)
    .map((h) => h.label as string);
}

function emit(onChange: (n: number[]) => void, set: Set<number>) {
  onChange(Array.from(set).sort((a, b) => a - b));
}

export function GpuChecklist({
  gpus,
  selected,
  onChange,
  disabled = false,
  describedById,
  allowedIndices,
  excludeModelId,
}: GpuChecklistProps) {
  const selectedSet = new Set(selected);
  const presentIndices = new Set(gpus.map((g) => g.index));
  // Configured indices with no matching present GPU — render as ghost rows.
  const missing = selected.filter((i) => !presentIndices.has(i)).sort((a, b) => a - b);
  // null/undefined = no allowlist recorded = no restriction. See the prop doc.
  const allowedSet = allowedIndices == null ? null : new Set(allowedIndices);
  const isAllowed = (index: number) => allowedSet === null || allowedSet.has(index);

  // Selected GPUs that another model is already serving from. Warn, never
  // block: co-locating two small models on one card is a legitimate thing to
  // do on purpose, and it is doing it by accident that is worth catching.
  const occupiedSelections = gpus
    .filter(
      (g) => selectedSet.has(g.index) && occupyingModels(g, excludeModelId).length > 0,
    )
    .map((g) => ({ index: g.index, models: occupyingModels(g, excludeModelId) }));

  function toggle(index: number) {
    const next = new Set(selectedSet);
    if (next.has(index)) next.delete(index);
    else next.add(index);
    emit(onChange, next);
  }

  if (gpus.length === 0 && missing.length === 0) {
    return (
      <p className="text-sm text-chat-muted" data-testid="gpu-empty" aria-describedby={describedById}>
        No GPUs detected — saving will still validate against allowed_gpu_indices server-side.
      </p>
    );
  }

  return (
    <div className="space-y-1" aria-describedby={describedById}>
      {missing.length > 0 && (
        <div
          role="alert"
          className="rounded-md border border-vw-amber-fg/45 bg-vw-amber-bg/30 px-2 py-1.5 text-xs text-vw-amber-fg"
        >
          {missing.length === 1 ? "GPU" : "GPUs"} {missing.join(", ")} configured but not present
          in the system. Uncheck to remove, or restore the card before loading.
        </div>
      )}
      {occupiedSelections.length > 0 && (
        <div
          role="alert"
          data-testid="gpu-occupied-warning"
          className="rounded-md border border-vw-amber-fg/45 bg-vw-amber-bg/30 px-2 py-1.5 text-xs text-vw-amber-fg"
        >
          {occupiedSelections
            .map(
              (o) =>
                `GPU ${o.index} is already serving ${o.models.join(", ")}`,
            )
            .join("; ")}
          . Both models will share the card&apos;s VRAM, and the second may fail
          to load or slow the first. Deliberate co-location is fine — this is
          only here so it is not accidental.
        </div>
      )}
      <ul className="grid grid-cols-1 gap-1 sm:grid-cols-2" data-testid="gpu-list">
        {gpus.map((g) => {
          const id = `gpu-checklist-${g.index}`;
          const freeGiB = (g.memory_total_mib - g.memory_used_mib) / 1024;
          const models = occupyingModels(g, excludeModelId);
          // Out of the allowlist AND not already chosen: the server would 400
          // on it, so it is not a live choice. Already chosen stays enabled so
          // the operator can clear a selection the allowlist has since
          // outlawed.
          const blocked = !isAllowed(g.index) && !selectedSet.has(g.index);
          return (
            <li
              key={g.index}
              className={
                "flex items-center gap-2 rounded-md border px-2 py-1.5 text-xs " +
                (blocked
                  ? "border-vw-rule-soft/50 bg-chat-surface-2/40 opacity-60"
                  : "border-vw-rule-soft bg-chat-surface")
              }
            >
              <input
                id={id}
                type="checkbox"
                className="h-3.5 w-3.5 accent-chat-accent"
                checked={selectedSet.has(g.index)}
                disabled={disabled || blocked}
                onChange={() => toggle(g.index)}
              />
              <label htmlFor={id} className="flex-1 cursor-pointer">
                <span className="!font-mono text-chat-fg">#{g.index}</span>{" "}
                <span>{g.name}</span>{" "}
                <span className="text-chat-muted">{freeGiB.toFixed(1)} GiB free</span>
                {models.length > 0 && (
                  <>
                    {" "}
                    <span className="text-vw-amber-fg" data-testid={`gpu-holder-${g.index}`}>
                      · serving {models.join(", ")}
                    </span>
                  </>
                )}
                {blocked && (
                  <>
                    {" "}
                    <span
                      className="text-chat-muted"
                      data-testid={`gpu-not-allowed-${g.index}`}
                    >
                      · not allowed by this deployment&apos;s setup
                    </span>
                  </>
                )}
              </label>
            </li>
          );
        })}
        {missing.map((index) => {
          const id = `gpu-checklist-missing-${index}`;
          return (
            <li
              key={`missing-${index}`}
              className="flex items-center gap-2 rounded-md border border-vw-amber-fg/45 bg-vw-amber-bg/30 px-2 py-1.5 text-xs"
            >
              <input
                id={id}
                type="checkbox"
                className="h-3.5 w-3.5 accent-chat-accent"
                checked={selectedSet.has(index)}
                disabled={disabled}
                onChange={() => toggle(index)}
              />
              <label htmlFor={id} className="flex-1 cursor-pointer text-vw-amber-fg">
                GPU {index} — not present
              </label>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
