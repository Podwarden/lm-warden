"use client";

// Limits card (spec §4.2; mockup `#settings`): P0–P9 and ONE Save button
// that is enabled only when the priority changed and sends one PATCH.
// (Per-token rate limits were removed; priority is the only control left.)

import { useId, useState } from "react";
import type { TokenDetail } from "@/lib/token-series";
import { btn, CARD, CARD_TITLE, prioTone } from "./styles";

export interface LimitsPatch {
  priority?: number;
}

export interface LimitsCardProps {
  token: Pick<TokenDetail, "id" | "priority">;
  onSave: (patch: LimitsPatch) => Promise<string | null>;
}

const PRIORITIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const;

export function LimitsCard({ token, onSave }: LimitsCardProps) {
  const [prio, setPrio] = useState(token.priority);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const prioLabelId = useId();

  // A poll that brings a new saved value resets a CLEAN form; an edit in
  // progress is never overwritten under the operator's cursor. "Clean" is
  // judged against the saved value the form last saw, not the new one: by
  // the time the new value arrives, `prio !== token.priority` holds for a
  // clean form too, which is how a clean form used to ignore polls (#251).
  // Adjusted during render (React's "storing information from previous
  // renders" pattern), so no frame ever shows the stale value.
  const [seen, setSeen] = useState(token.priority);
  if (token.priority !== seen) {
    setSeen(token.priority);
    if (prio === seen) setPrio(token.priority);
  }
  const dirty = prio !== token.priority;

  async function save() {
    const patch: LimitsPatch = { priority: prio };
    setSaving(true);
    setError(null);
    try {
      setError(await onSave(patch));
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className={CARD} aria-label="Limits">
      <h2 className={CARD_TITLE}>Limits</h2>

      <div>
        <div id={prioLabelId} className="mb-1.5 text-[13px] text-chat-muted">
          Priority <span className="font-semibold text-chat-fg">{prio}</span>
        </div>
        <div role="group" aria-labelledby={prioLabelId} className="grid grid-cols-10 gap-[3px]">
          {PRIORITIES.map((i) => {
            const on = i === prio;
            const tone = prioTone(i);
            return (
              <button
                key={i}
                type="button"
                aria-pressed={on}
                onClick={() => {
                  setPrio(i);
                  setError(null);
                }}
                className={
                  on
                    ? "cursor-pointer rounded border py-1.5 text-[12px] font-semibold tabular-nums"
                    : "cursor-pointer rounded border border-transparent bg-chat-page py-1.5 text-[12px] font-semibold tabular-nums text-chat-dim hover:text-chat-fg"
                }
                style={on ? { color: tone.fg, background: tone.bg, borderColor: tone.fg } : undefined}
              >
                P{i}
              </button>
            );
          })}
        </div>
        <div className="mt-1 flex justify-between text-[11px] text-chat-dim">
          <span>served last</span>
          <span>served first</span>
        </div>
        <div className="mt-1.5 text-[12px] leading-[1.45] text-chat-dim">
          Strict: a waiting P9 request always goes before a P8 one. Low priorities can wait indefinitely on a busy box.
        </div>
      </div>

      <div className="mt-[18px] flex items-center gap-2">
        <button type="button" className={btn("primary")} disabled={!dirty || saving} onClick={save}>
          Save changes
        </button>
        {error ? (
          <span role="alert" className="text-[12px] text-vw-danger-fg">{error}</span>
        ) : (
          dirty && <span className="text-[12px] text-chat-accent">Unsaved</span>
        )}
      </div>
    </section>
  );
}
