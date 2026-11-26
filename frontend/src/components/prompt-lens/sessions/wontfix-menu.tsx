"use client";

// Won't fix on a finding (analysis spec 11.4, decision D-3): "Won't fix" asks
// whether the mark is for every prompt version or only this one; a muted finding
// says who marked it and offers "Undo". A mark leaves the finding out of the
// scores and the top 3 but keeps it listed; it is reversible and audited by the
// server. A secret in the prompt (R-09) cannot be marked, so it gets no menu.

import { useState } from "react";

import type { ScorecardFinding } from "@/lib/prompt-lens/scorecard";

/** What a mark applies to: every version, or the session's own. */
export type MarkScope = "all" | "version";

export interface WontfixActions {
  /** The session's prompt version label ("v4"), or null when it has none. */
  versionLabel: string | null;
  mark: (rule: string, scope: MarkScope) => void;
  unmark: (rule: string) => void;
}

const WHO: Record<string, string> = { key: "key holder", admin: "admin" };

export const UNMARKABLE = new Set(["R-09"]);

export function WontfixMenu({ finding: f, actions }: { finding: ScorecardFinding; actions: WontfixActions | null }) {
  const [open, setOpen] = useState(false);
  if (UNMARKABLE.has(f.rule)) return null;
  if (f.muted) {
    return (
      <span data-testid="wontfix-muted" className="inline-flex items-center gap-1.5 text-[11px] text-chat-dim">
        won&apos;t fix · by {WHO[f.muted_by ?? ""] ?? "key holder"}
        {actions && (
          <button
            type="button"
            data-testid="wontfix-undo"
            onClick={() => actions.unmark(f.rule)}
            className="cursor-pointer bg-transparent p-0 text-[11px] text-chat-accent hover:underline"
          >
            Undo
          </button>
        )}
      </span>
    );
  }
  if (!actions) return null;
  if (!open) {
    return (
      <button
        type="button"
        data-testid="wontfix-open"
        aria-label={`Won't fix ${f.rule}`}
        onClick={() => setOpen(true)}
        className="cursor-pointer bg-transparent p-0 text-[11px] text-chat-muted hover:text-chat-fg hover:underline"
      >
        Won&apos;t fix…
      </button>
    );
  }
  const choose = (scope: MarkScope) => {
    setOpen(false);
    actions.mark(f.rule, scope);
  };
  const btn = "cursor-pointer rounded border border-chat-rule bg-transparent px-1.5 py-0.5 text-[11px] text-chat-fg hover:border-chat-accent/60";
  return (
    <span data-testid="wontfix-choices" role="group" aria-label={`Won't fix ${f.rule}`} className="inline-flex flex-wrap items-center gap-1">
      <button type="button" className={btn} onClick={() => choose("all")}>
        Won&apos;t fix for every version
      </button>
      {actions.versionLabel && (
        <button type="button" className={btn} onClick={() => choose("version")}>
          for {actions.versionLabel} only
        </button>
      )}
      <button type="button" className="cursor-pointer bg-transparent p-0 text-[11px] text-chat-dim hover:underline" onClick={() => setOpen(false)}>
        Cancel
      </button>
    </span>
  );
}
