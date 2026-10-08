"use client";

// "Start from a preset" card (plan §4.4). Presets come from GET /api/presets
// (app/presets/builtin.json); each declares a sparse settings dict that is
// overlaid on the current DRAFT -- nothing is saved until the operator saves.
// Clicking a chip opens a confirm popover with the exact diff. Each chip shows
// the name and, on a second line, the target hardware (`target_archetype`,
// informational only: we never pre-filter by GPU arch).
//
// `presets-strip` wraps the chips and NOTHING else -- a test asserts it holds
// exactly four buttons, so the confirm popover renders outside it.

import { useEffect, useId, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import type { Draft } from "@/lib/model-settings";
import {
  DiffList,
  PANEL_BUTTON_CLASS,
  PANEL_BUTTON_STRONG_CLASS,
  computeDiff,
} from "./diff-list";

export interface PresetEntry {
  id: string;
  name: string;
  description: string;
  target_archetype: string;
  settings: Record<string, unknown>;
}
export interface PresetsResponse {
  presets: PresetEntry[];
}

export interface PresetStripProps {
  disabled: boolean;
  draft: Draft;
  onApply: (sparse: Record<string, unknown>) => void;
}

export function PresetStrip({ disabled, draft, onApply }: PresetStripProps) {
  const { data, error } = useSWR<PresetsResponse>("/api/presets", authFetchJSON, {
    revalidateOnFocus: false,
  });
  const headingId = useId();
  const [pendingId, setPendingId] = useState<string | null>(null);
  const chipRefs = useRef(new Map<string, HTMLButtonElement>());
  const dialogRef = useRef<HTMLDivElement>(null);
  // Focus moves into the confirm when it opens, so Escape and Tab work at
  // once; close() hands it back to the chip.
  useEffect(() => {
    if (pendingId) dialogRef.current?.focus();
  }, [pendingId]);
  const presets = data?.presets ?? [];
  const pending = pendingId ? (presets.find((p) => p.id === pendingId) ?? null) : null;
  const pendingDiff = useMemo(
    () => (pending ? computeDiff(draft, pending.settings) : []),
    [draft, pending],
  );

  function close() {
    const chip = pendingId ? chipRefs.current.get(pendingId) : undefined;
    setPendingId(null);
    // Focus goes back to the chip that opened the confirm (plan §7).
    chip?.focus();
  }

  let body: React.ReactNode;
  if (error) {
    body = (
      <p className="text-[12.5px] text-chat-muted" data-testid="presets-error">
        Could not load presets.
      </p>
    );
  } else if (!data) {
    body = (
      <p className="text-[12.5px] text-chat-muted" data-testid="presets-loading">
        Loading presets…
      </p>
    );
  } else {
    body = (
      <>
        <div data-testid="presets-strip" className="flex flex-wrap gap-2">
          {presets.map((p) => (
            <button
              key={p.id}
              ref={(el) => {
                if (el) chipRefs.current.set(p.id, el);
                else chipRefs.current.delete(p.id);
              }}
              type="button"
              onClick={() => setPendingId(p.id)}
              disabled={disabled}
              aria-expanded={pendingId === p.id}
              data-testid={`preset-chip-${p.id}`}
              title={p.description}
              className="flex min-h-6 flex-col items-start gap-px rounded-lg border border-vw-rule-soft bg-transparent px-3 py-[7px] text-left hover:border-chat-accent/70 hover:bg-chat-surface-2/50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent disabled:cursor-not-allowed disabled:opacity-[.45] disabled:hover:border-vw-rule-soft disabled:hover:bg-transparent"
            >
              <span
                data-testid="preset-chip-name"
                className="text-[13px] font-semibold text-chat-fg"
              >
                {p.name}
              </span>
              <span
                data-testid="preset-chip-archetype"
                className="text-[11.5px] text-chat-muted"
              >
                {p.target_archetype}
              </span>
            </button>
          ))}
        </div>
        {pending && (
          <div
            ref={dialogRef}
            tabIndex={-1}
            role="dialog"
            aria-label={`Apply preset ${pending.name}`}
            data-testid="preset-confirm"
            onKeyDown={(e) => {
              if (e.key === "Escape") close();
            }}
            className="rounded-lg border border-vw-rule-soft/70 bg-chat-surface-2/60 p-3 text-sm focus:outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent"
          >
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="text-chat-fg">
                Apply <strong className="font-semibold">{pending.name}</strong>?
              </span>
              <span className="text-[11.5px] text-chat-muted">
                {pending.target_archetype}
              </span>
            </div>
            <p className="mt-1 text-[12.5px] leading-[1.45] text-chat-muted">
              {pending.description}
            </p>
            <DiffList rows={pendingDiff} testIdPrefix="preset-diff" />
            <div className="mt-3 flex justify-end gap-2">
              <button
                type="button"
                className={PANEL_BUTTON_CLASS}
                onClick={close}
                data-testid="preset-cancel"
              >
                Cancel
              </button>
              <button
                type="button"
                className={PANEL_BUTTON_STRONG_CLASS}
                onClick={() => {
                  onApply(pending.settings);
                  close();
                }}
                disabled={pendingDiff.length === 0}
                data-testid="preset-apply"
              >
                Apply
              </button>
            </div>
          </div>
        )}
      </>
    );
  }

  return (
    <section
      aria-labelledby={headingId}
      data-testid="presets-card"
      className="flex flex-col gap-2.5 rounded-xl border border-vw-rule-soft/70 bg-chat-surface/55 px-4 pb-4 pt-3.5"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 id={headingId} className="m-0 text-[15px] font-semibold text-chat-fg">
          Start from a preset
        </h2>
        <p className="m-0 text-[12.5px] text-chat-muted">
          Fills in the form only. You see the changes before anything is saved.
        </p>
      </div>
      {body}
    </section>
  );
}
