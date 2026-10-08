"use client";
// The model selector used by /stats.
//
// One control, one selection, every surface — the operator picks once and
// every page answers about the same models. A per-page selector would let two
// open tabs disagree about what "the numbers" mean.
//
// THE LAST CHECKBOX IS DISABLED, not silently re-ticked. At least one model
// must stay selected (see @/lib/model-selection for why an empty selection is
// unanswerable), and there are two ways to enforce that: refuse the click
// invisibly, or say so. Refusing invisibly leaves the operator unable to tell
// whether the click registered at all — which is this codebase's defining
// defect class, an affordance whose effect is gated on a different condition
// than the affordance itself. So the input carries `disabled` and a title
// explaining it, and the gate and the control are the same fact.
//
// THE CONTROL ADAPTS. Inline chips read best for a handful of models and
// become a three-row wall at eight — model names run to things like
// `ista-daslab-qwen3.8-27b-gsq-rco-gguf`. Above INLINE_MAX the chips collapse
// into a single summary button with a filterable checklist popover, so the
// control stays ONE row at any model count. The rows inside the popover are
// the same labelled checkboxes as the inline chips (same testids, same
// disabled-last rule); only the layout collapses, never the behaviour.
import { useEffect, useId, useRef, useState } from "react";
import type { ModelSelection } from "@/lib/model-selection";

export interface SelectableModel {
  id: string;
  served_model_name: string;
  /** Optional: which cards this model occupies, for the tooltip. */
  gpu_indices?: number[];
}

/** Above this many models the inline chips collapse into the popover. */
export const INLINE_MAX = 4;

function OptionChip({
  m,
  selection,
}: {
  m: SelectableModel;
  selection: ModelSelection;
}) {
  const checked = selection.selected.includes(m.id);
  const locked = checked && selection.isLocked(m.id);
  const cards =
    m.gpu_indices && m.gpu_indices.length > 0
      ? ` · GPU ${m.gpu_indices.join(", ")}`
      : "";
  return (
    <label
      data-testid="model-selector-option"
      data-model-id={m.id}
      data-checked={checked}
      data-locked={locked}
      title={
        locked
          ? "At least one model must stay selected."
          : `${m.served_model_name}${cards}`
      }
      className={[
        "inline-flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs transition-colors",
        checked
          ? "border-chat-accent/60 bg-chat-accent/10 text-chat-fg"
          : "border-chat-rule bg-chat-surface/60 text-chat-muted hover:text-chat-fg",
        locked ? "cursor-not-allowed opacity-80" : "",
      ].join(" ")}
    >
      <input
        type="checkbox"
        className="h-3 w-3 accent-emerald-500 disabled:cursor-not-allowed"
        checked={checked}
        // The gate and the control, one condition. See the header note.
        disabled={locked}
        onChange={() => selection.toggle(m.id)}
        aria-label={m.served_model_name}
      />
      {/* Long names truncate; the full name stays in the title above. */}
      <span className="max-w-[18ch] truncate font-mono">
        {m.served_model_name}
      </span>
    </label>
  );
}

function summaryText(models: SelectableModel[], selection: ModelSelection): string {
  const n = selection.selected.length;
  if (n === models.length) return `All ${n} models`;
  if (n === 1) {
    const only = models.find((m) => m.id === selection.selected[0]);
    return only?.served_model_name ?? "1 model";
  }
  return `${n} of ${models.length} models`;
}

function CollapsedControl({
  models,
  selection,
}: {
  models: SelectableModel[];
  selection: ModelSelection;
}) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const rootRef = useRef<HTMLDivElement | null>(null);
  const listId = useId();

  // Outside click + Escape close — same conventions as the nav menu.
  useEffect(() => {
    if (!open) return;
    function onPointerDown(e: PointerEvent) {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    function onKeydown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeydown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeydown);
    };
  }, [open]);

  const q = filter.trim().toLowerCase();
  const rows = q
    ? models.filter((m) => m.served_model_name.toLowerCase().includes(q))
    : models;
  const allSelected = selection.selected.length === models.length;
  const selectedNames = models
    .filter((m) => selection.selected.includes(m.id))
    .map((m) => m.served_model_name);

  return (
    <div className="relative inline-flex" ref={rootRef}>
      <button
        type="button"
        data-testid="model-selector-summary"
        aria-expanded={open}
        aria-controls={listId}
        aria-haspopup="listbox"
        title={allSelected ? "" : selectedNames.join(", ")}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex max-w-[32ch] items-center gap-1.5 truncate rounded-full border border-chat-rule bg-chat-surface/60 px-2.5 py-0.5 font-mono text-xs text-chat-fg transition-colors hover:border-chat-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-chat-accent"
      >
        <span className="truncate">{summaryText(models, selection)}</span>
        <span aria-hidden="true">▾</span>
      </button>
      {open && (
        <div
          id={listId}
          data-testid="model-selector-popover"
          className="absolute left-0 top-full z-30 mt-1.5 w-80 rounded-lg border border-chat-rule bg-chat-surface p-2 shadow-lg"
        >
          <input
            type="search"
            data-testid="model-selector-filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter models"
            aria-label="Filter models"
            // eslint-disable-next-line jsx-a11y/no-autofocus
            autoFocus
            className="mb-2 w-full rounded-md border border-chat-rule bg-chat-page px-2 py-1 text-xs text-chat-fg placeholder:text-chat-dim focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-chat-accent"
          />
          <div className="mb-2 flex gap-1.5">
            <button
              type="button"
              data-testid="model-selector-all"
              onClick={selection.selectAll}
              disabled={allSelected}
              className="rounded border border-chat-rule px-2 py-0.5 text-xs text-chat-muted transition-colors hover:text-chat-fg disabled:cursor-not-allowed disabled:opacity-40"
            >
              Select all
            </button>
          </div>
          <div className="flex max-h-64 flex-col gap-1 overflow-y-auto" role="listbox">
            {rows.length === 0 ? (
              <p className="px-1 py-1.5 text-xs text-chat-dim">
                No model matches &ldquo;{filter}&rdquo;.
              </p>
            ) : (
              rows.map((m) => <OptionChip key={m.id} m={m} selection={selection} />)
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function ModelSelector({
  models,
  selection,
  label = "Models",
}: {
  models: SelectableModel[];
  selection: ModelSelection;
  label?: string;
}) {
  // Nothing loaded: render nothing rather than an empty box. There is no
  // selection to make, and the pages already have their own "nothing loaded"
  // states that say more than a disabled fieldset would.
  if (models.length === 0) return null;

  const allSelected = selection.selected.length === models.length;
  const inline = models.length <= INLINE_MAX;

  return (
    <div
      role="group"
      aria-label={label}
      data-testid="model-selector"
      className="flex flex-wrap items-center gap-2"
    >
      <span className="text-xs uppercase tracking-wider text-chat-dim">
        {label}
      </span>
      {inline ? (
        <>
          {models.map((m) => (
            <OptionChip key={m.id} m={m} selection={selection} />
          ))}
          <button
            type="button"
            data-testid="model-selector-all"
            onClick={selection.selectAll}
            // Disabled when it would do nothing — the same rule as the
            // checkboxes, applied to the shortcut.
            disabled={allSelected}
            className="rounded border border-chat-rule px-2 py-0.5 text-xs text-chat-muted transition-colors hover:text-chat-fg disabled:cursor-not-allowed disabled:opacity-40"
          >
            All
          </button>
        </>
      ) : (
        <CollapsedControl models={models} selection={selection} />
      )}
    </div>
  );
}
