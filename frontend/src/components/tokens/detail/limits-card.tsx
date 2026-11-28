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

/** #348: the key's cap on thinking tokens and what happens at the cap (null = the model's). */
export interface ReasoningPatch {
  reasoning_budget: number | null;
  reasoning_budget_mode: "force_answer" | "stop" | null;
}

export interface LimitsCardProps {
  token: Pick<TokenDetail, "id" | "priority"> &
    Partial<
      Pick<
        TokenDetail,
        "anthropic_relay" | "thinking_policy" | "reasoning_budget" | "reasoning_budget_mode"
      >
    >;
  onSave: (patch: LimitsPatch) => Promise<string | null>;
  /** Flips the Claude Code router relay flag; resolves true on success. */
  onRelay?: (on: boolean) => Promise<boolean>;
  relayError?: string | null;
  relaying?: boolean;
  /** Sets the key's thinking policy; resolves true on success. */
  onThinking?: (policy: "client" | "off") => Promise<boolean>;
  thinkingError?: string | null;
  thinkingBusy?: boolean;
  /** Saves the key's reasoning budget; resolves to an error message, or null on success. */
  onReasoning?: (patch: ReasoningPatch) => Promise<string | null>;
  /** The cache store's eviction policy: under fair share the priority is also the key's weight. */
  policy?: "lru" | "fair";
}

const PRIORITIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const;

export function LimitsCard({
  token,
  onSave,
  onRelay,
  relayError,
  relaying,
  onThinking,
  thinkingError,
  thinkingBusy,
  onReasoning,
  policy,
}: LimitsCardProps) {
  const [prio, setPrio] = useState(token.priority);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const prioLabelId = useId();
  const relayLabelId = useId();
  const thinkingLabelId = useId();
  const forcedOff = token.thinking_policy === "off";

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

  // Reasoning budget (#348): the same "reset a clean form when a poll brings a new saved
  // value" rule as the priority above, judged against the saved value the form last saw.
  const savedBudget = token.reasoning_budget == null ? "" : String(token.reasoning_budget);
  const savedMode = token.reasoning_budget_mode ?? "";
  const [rbBudget, setRbBudget] = useState(savedBudget);
  const [rbMode, setRbMode] = useState<string>(savedMode);
  const [rbSeen, setRbSeen] = useState(`${savedBudget}|${savedMode}`);
  const [rbSaving, setRbSaving] = useState(false);
  const [rbError, setRbError] = useState<string | null>(null);
  if (rbSeen !== `${savedBudget}|${savedMode}`) {
    const [seenBudget, seenMode] = rbSeen.split("|");
    setRbSeen(`${savedBudget}|${savedMode}`);
    if (rbBudget === seenBudget && rbMode === seenMode) {
      setRbBudget(savedBudget);
      setRbMode(savedMode);
    }
  }
  const rbDirty = rbBudget !== savedBudget || rbMode !== savedMode;
  const rbNumber = rbBudget.trim() === "" ? null : Number(rbBudget);
  const rbValid = rbNumber === null || (Number.isInteger(rbNumber) && rbNumber >= 1);
  const reasoningBudgetId = useId();
  const reasoningModeId = useId();

  async function saveReasoning() {
    if (!onReasoning || !rbValid) return;
    setRbSaving(true);
    setRbError(null);
    try {
      setRbError(
        await onReasoning({
          reasoning_budget: rbNumber,
          reasoning_budget_mode: rbMode === "stop" || rbMode === "force_answer" ? rbMode : null,
        }),
      );
    } finally {
      setRbSaving(false);
    }
  }

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
          {policy === "fair" && (
            <> The priority also sets this key&apos;s part of the KV cache store: P{prio} counts {prio + 1}, P0 counts 1.</>
          )}
          {policy === "lru" && <> With the fair eviction policy it also sets this key&apos;s part of the KV cache store.</>}
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

      <div className="mt-[18px] border-t border-chat-rule pt-3.5">
        <div className="flex items-center justify-between gap-2">
          <span id={relayLabelId} className="text-[13px] text-chat-muted">
            Relay to Anthropic:{" "}
            <span className="font-semibold text-chat-fg">{!!token.anthropic_relay ? "on" : "off"}</span>
          </span>
          <button
            type="button"
            role="switch"
            aria-checked={!!token.anthropic_relay}
            aria-label="Relay to Anthropic"
            disabled={!onRelay || relaying}
            onClick={() => void onRelay?.(!token.anthropic_relay)}
            className={btn(token.anthropic_relay ? "primary" : "default")}
          >
            {token.anthropic_relay ? "Disable" : "Enable"}
          </button>
        </div>
        <div className="mt-1.5 text-[12px] leading-[1.45] text-chat-dim">
          When on, this key may pass the client&apos;s own Anthropic credential through
          this warden to Anthropic (Claude Code router mode). Off, the key can still use
          local rules but is never relayed.
        </div>
        {relayError && (
          <div role="alert" className="mt-1 text-[12px] text-vw-danger-fg">{relayError}</div>
        )}
      </div>

      <div className="mt-[18px] border-t border-chat-rule pt-3.5">
        <div className="flex items-center justify-between gap-2">
          <span id={thinkingLabelId} className="text-[13px] text-chat-muted">
            Thinking:{" "}
            <span className="font-semibold text-chat-fg">{forcedOff ? "forced off" : "client decides"}</span>
          </span>
          <button
            type="button"
            role="switch"
            aria-checked={forcedOff}
            aria-label="Force thinking off"
            disabled={!onThinking || thinkingBusy}
            onClick={() => void onThinking?.(forcedOff ? "client" : "off")}
            className={btn(forcedOff ? "primary" : "default")}
          >
            {forcedOff ? "Let the client decide" : "Force off"}
          </button>
        </div>
        <div className="mt-1.5 text-[12px] leading-[1.45] text-chat-dim">
          Off: requests with this key never think, whatever the client asks. Takes effect on
          the next request, no reload. Client decides: the request&apos;s own setting applies
          (default). If the model&apos;s own policy is off, it stays off.
        </div>
        {thinkingError && (
          <div role="alert" className="mt-1 text-[12px] text-vw-danger-fg">{thinkingError}</div>
        )}
      </div>

      <div className="mt-[18px] border-t border-chat-rule pt-3.5" data-testid="reasoning-budget">
        <div className="mb-1.5 text-[13px] text-chat-muted">
          Reasoning budget:{" "}
          <span className="font-semibold text-chat-fg">
            {token.reasoning_budget == null ? "no cap" : `${token.reasoning_budget} tokens`}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor={reasoningBudgetId} className="sr-only">
            Reasoning budget (tokens)
          </label>
          <input
            id={reasoningBudgetId}
            type="number"
            min={1}
            step={1}
            inputMode="numeric"
            placeholder="no cap"
            value={rbBudget}
            disabled={!onReasoning}
            onChange={(e) => {
              setRbBudget(e.target.value);
              setRbError(null);
            }}
            className="w-28 rounded border border-chat-rule bg-chat-page px-2 py-1 text-[13px] text-chat-fg"
          />
          <label htmlFor={reasoningModeId} className="sr-only">
            When the budget is spent
          </label>
          <select
            id={reasoningModeId}
            value={rbMode}
            disabled={!onReasoning}
            onChange={(e) => {
              setRbMode(e.target.value);
              setRbError(null);
            }}
            className="rounded border border-chat-rule bg-chat-page px-2 py-1 text-[13px] text-chat-fg"
          >
            <option value="">model&apos;s mode</option>
            <option value="force_answer">force_answer</option>
            <option value="stop">stop</option>
          </select>
          <button
            type="button"
            className={btn("primary")}
            disabled={!onReasoning || !rbDirty || !rbValid || rbSaving}
            onClick={saveReasoning}
          >
            Save budget
          </button>
        </div>
        <div className="mt-1.5 text-[12px] leading-[1.45] text-chat-dim">
          The most tokens a request with this key may spend thinking. A request can ask for less
          (Anthropic <code>thinking.budget_tokens</code>, <code>reasoning_effort</code>,{" "}
          <code>lmw_reasoning_budget</code>), never more; the model&apos;s own budget applies too,
          and the smallest wins. <code>force_answer</code> closes the thinking and lets the model
          answer; <code>stop</code> ends the request. Either way the client gets{" "}
          <code>lmw_stop_reason: &quot;reasoning_budget&quot;</code>. Takes effect on the next request.
        </div>
        {rbError ? (
          <div role="alert" className="mt-1 text-[12px] text-vw-danger-fg">{rbError}</div>
        ) : (
          rbDirty && <div className="mt-1 text-[12px] text-chat-accent">Unsaved</div>
        )}
      </div>
    </section>
  );
}
