"use client";

// Add / edit a routing rule (plan §4.4), laid out top-down as a sentence:
// "When Claude Code asks for [pattern] → answer with [local model]", then
// "If the local model fails: (•) Fall back to Anthropic ( ) Refuse with 529",
// then a collapsed "Model quirks" disclosure. A live preview under the pattern
// shows which known Claude ids it would catch (the glob mirrors the server).
// Kept: the "Pattern" / "Local model" labels, "Create rule" / "Save rule",
// PATCH-only-changed, the server 422 in role="alert" with input preserved.

import { useEffect, useId, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { errorDetail } from "@/components/tokens/use-token-actions";
import { Modal } from "@/components/ui/modal";
import { Select } from "@/components/ui/select";
import { BAND_LABEL, META, btn } from "@/components/router/styles";
import { cn } from "@/lib/utils";
import type { components } from "@/lib/api-types.generated";
import { KNOWN_CLAUDE_IDS, globMatches } from "@/lib/router";
import type { RuleIn, RuleOut, RulePatch } from "@/lib/router";

type ModelList = components["schemas"]["ModelList"];

export interface RuleDialogProps {
  open: boolean;
  /** null creates; a rule edits. */
  rule: RuleOut | null;
  /** Prefills the pattern of a NEW rule (the claude-haiku* / sonnet / opus presets). */
  initialPattern?: string;
  onClose: () => void;
  onSaved: () => void;
}

interface Draft {
  pattern: string;
  target_model_id: string | null;
  fallback: boolean;
  strip_thinking: boolean;
  min_max_tokens: string;
}

function initial(rule: RuleOut | null, initialPattern?: string): Draft {
  return {
    pattern: rule?.pattern ?? initialPattern ?? "",
    target_model_id: rule?.target_model_id ?? null,
    fallback: rule?.fallback ?? true,
    strip_thinking: rule?.strip_thinking ?? true,
    min_max_tokens: String(rule?.min_max_tokens ?? 0),
  };
}

const PREVIEW_MAX = 3;

/** "Matches e.g. a, b, c (+2 more)" / "Exact match" / no-match warning. */
export function patternPreview(pattern: string): { text: string; warn: boolean } {
  const p = pattern.trim();
  if (p === "") return { text: "Glob: * any run, ? one character; exact when no wildcard; case-sensitive.", warn: false };
  if (!/[*?]/.test(p)) return { text: `Exact match: only ${p}.`, warn: false };
  const hits = KNOWN_CLAUDE_IDS.filter((id) => globMatches(p, id));
  if (hits.length === 0) {
    return {
      text: "Matches none of the known Claude ids — check the spelling (case-sensitive).",
      warn: true,
    };
  }
  const more = hits.length - PREVIEW_MAX;
  return {
    text: `Matches e.g. ${hits.slice(0, PREVIEW_MAX).join(", ")}${more > 0 ? ` (+${more} more)` : ""}`,
    warn: false,
  };
}

const LEAD = "text-[13.5px] font-medium text-chat-fg";
// Token-only field (no slate literals): .input in the mockup.
// Mirrors MIN_MAX_TOKENS_CAP in app/router/routes_api.py.
const MIN_MAX_TOKENS_CAP = 131072;

const INPUT =
  "block h-9 w-full rounded-md border border-chat-rule bg-chat-page/60 px-3 text-[13.5px] text-chat-fg placeholder:text-chat-dim focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-chat-accent";
const HELP = "text-[12.5px] text-chat-muted";
const RADIO =
  "flex cursor-pointer items-start gap-2.5 rounded-[10px] border border-vw-rule-soft/70 px-3 py-2.5 text-[13.5px] has-[:checked]:border-vw-live/60 has-[:checked]:bg-vw-ok-bg/15";
const FIELD_ERR_FOCUS: Record<string, "pattern" | "min"> = {
  pattern: "pattern",
  min_max_tokens: "min",
};

export function RuleDialog({ open, rule, initialPattern, onClose, onSaved }: RuleDialogProps) {
  const { data } = useSWR<ModelList>(open ? "/api/models" : null, authFetchJSON);
  const [draft, setDraft] = useState<Draft>(() => initial(rule, initialPattern));
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const ids = useId();
  const patternRef = useRef<HTMLInputElement | null>(null);
  const minRef = useRef<HTMLInputElement | null>(null);
  const quirksDefault = !!rule && (!rule.strip_thinking || rule.min_max_tokens > 0);
  const [quirksOpen, setQuirksOpen] = useState(quirksDefault);

  useEffect(() => {
    if (open) {
      setDraft(initial(rule, initialPattern));
      setError(null);
      setQuirksOpen(!!rule && (!rule.strip_thinking || rule.min_max_tokens > 0));
    }
  }, [open, rule, initialPattern]);

  // Loaded models first (plan §4.4); stable otherwise.
  const options = useMemo(
    () =>
      [...(data?.models ?? [])]
        .sort((a, b) => Number(b.status === "loaded") - Number(a.status === "loaded"))
        .map((m) => ({ value: m.id, label: `${m.served_model_name} (${m.status})` })),
    [data],
  );
  const preview = patternPreview(draft.pattern);
  const minTokens = Number(draft.min_max_tokens);
  const minTooBig = minTokens > MIN_MAX_TOKENS_CAP;
  const valid =
    draft.pattern.trim() !== "" &&
    draft.target_model_id !== null &&
    Number.isInteger(minTokens) &&
    minTokens >= 0 &&
    minTokens <= MIN_MAX_TOKENS_CAP;

  function focusInvalid(detail: string) {
    const field = Object.keys(FIELD_ERR_FOCUS).find((k) => detail.startsWith(k));
    if (!field) return;
    if (FIELD_ERR_FOCUS[field] === "min") {
      setQuirksOpen(true);
      queueMicrotask(() => minRef.current?.focus());
    } else {
      patternRef.current?.focus();
    }
  }

  async function save() {
    if (!valid || draft.target_model_id === null) return;
    setBusy(true);
    setError(null);
    try {
      const next: RuleIn = {
        pattern: draft.pattern.trim(),
        target_model_id: draft.target_model_id,
        enabled: rule?.enabled ?? true,
        fallback: draft.fallback,
        strip_thinking: draft.strip_thinking,
        min_max_tokens: minTokens,
      };
      let path = "/api/router/rules";
      let method = "POST";
      let payload: RuleIn | RulePatch = next;
      if (rule) {
        path = `/api/router/rules/${encodeURIComponent(rule.id)}`;
        method = "PATCH";
        const changed: Record<string, unknown> = {};
        for (const k of Object.keys(next) as (keyof RuleIn)[]) {
          if (next[k] !== rule[k]) changed[k] = next[k];
        }
        payload = changed as RulePatch;
      }
      const r = await authFetch(path, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) {
        const detail = await errorDetail(r, `Failed to save rule (HTTP ${r.status})`);
        setError(detail);
        focusInvalid(detail);
        return;
      }
      onSaved();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  const patternId = `${ids}-pattern`;
  const previewId = `${ids}-preview`;
  const failId = `${ids}-fail`;
  const minId = `${ids}-min`;
  const thinkId = `${ids}-think`;

  return (
    <Modal open={open} onClose={onClose} title={rule ? "Edit rule" : "Add rule"}>
      <div className="space-y-5">
        <div className="space-y-1.5">
          <p className={LEAD}>When Claude Code asks for</p>
          <label htmlFor={patternId} className={BAND_LABEL}>
            Pattern
          </label>
          <input
            ref={patternRef}
            id={patternId}
            className={cn(INPUT, "!font-mono")}
            placeholder="claude-haiku*"
            autoComplete="off"
            spellCheck={false}
            value={draft.pattern}
            aria-describedby={previewId}
            onChange={(e) => setDraft({ ...draft, pattern: e.target.value })}
          />
          <p
            id={previewId}
            data-testid="pattern-preview"
            aria-live="polite"
            className={cn(HELP, "[overflow-wrap:anywhere]", preview.warn && "text-vw-amber-fg")}
          >
            {preview.text}
          </p>
        </div>

        <div className="space-y-1.5">
          <p className={LEAD}>
            <span aria-hidden="true" className="text-chat-dim">
              →{" "}
            </span>
            answer with
          </p>
          <span className={cn(BAND_LABEL, "block")}>Local model</span>
          <Select
            ariaLabel="Local model"
            options={options}
            value={draft.target_model_id}
            onChange={(v) => setDraft({ ...draft, target_model_id: v })}
            placeholder="Select a model…"
          />
        </div>

        <div role="radiogroup" aria-labelledby={failId} className="space-y-1.5">
          <p id={failId} className={LEAD}>
            If the local model fails
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            <label className={RADIO}>
              <input
                type="radio"
                name={failId}
                className="mt-1"
                checked={draft.fallback}
                onChange={() => setDraft({ ...draft, fallback: true })}
                aria-label="Fall back to Anthropic"
              />
              <span>
                <span className="block font-semibold">
                  Fall back to Anthropic
                </span>
                <span className={cn(HELP, "block")}>Needs a key that may relay.</span>
              </span>
            </label>
            <label className={RADIO}>
              <input
                type="radio"
                name={failId}
                className="mt-1"
                checked={!draft.fallback}
                onChange={() => setDraft({ ...draft, fallback: false })}
                aria-label="Refuse with 529"
              />
              <span>
                <span className="block font-semibold">
                  Refuse with 529
                </span>
                <span className={cn(HELP, "block")}>Claude Code retries.</span>
              </span>
            </label>
          </div>
        </div>

        <details
          open={quirksOpen}
          onToggle={(e) => setQuirksOpen((e.currentTarget as HTMLDetailsElement).open)}
          className="rounded-[10px] border border-vw-rule-soft/70"
        >
          <summary className="cursor-pointer select-none px-3 py-2.5 text-[13.5px] font-medium">
            Model quirks
            <span className={cn(META, "ml-2")}>
              {draft.strip_thinking ? "thinking off" : "thinking passed through"}
              {Number(draft.min_max_tokens) > 0 ? ` · max_tokens ≥ ${draft.min_max_tokens}` : ""}
            </span>
          </summary>
          <div className="space-y-3 border-t border-vw-rule-soft/45 px-3 py-3">
            <div className="flex items-start gap-2">
              <input
                id={thinkId}
                type="checkbox"
                className="mt-1"
                checked={draft.strip_thinking}
                aria-describedby={`${thinkId}-help`}
                onChange={(e) => setDraft({ ...draft, strip_thinking: e.target.checked })}
              />
              <span>
                <label htmlFor={thinkId} className="text-[13.5px]">
                  Disable thinking
                </label>
                <span id={`${thinkId}-help`} className={cn(HELP, "block")}>
                  Strip the client&apos;s thinking settings before the local model sees them.
                </span>
              </span>
            </div>
            <div className="space-y-1">
              <label htmlFor={minId} className="text-[13.5px] font-medium">
                Minimum max_tokens
              </label>
              <input
                ref={minRef}
                id={minId}
                type="number"
                min={0}
                max={MIN_MAX_TOKENS_CAP}
                aria-invalid={minTooBig || undefined}
                aria-describedby={minTooBig ? `${minId}-err` : undefined}
                className={INPUT}
                value={draft.min_max_tokens}
                onChange={(e) => setDraft({ ...draft, min_max_tokens: e.target.value })}
              />
              {minTooBig && (
                <p id={`${minId}-err`} role="alert" className="text-[12.5px] font-medium text-vw-danger-fg">
                  Must be at most {MIN_MAX_TOKENS_CAP.toLocaleString("en-US")}.
                </p>
              )}
              <p className={HELP}>
                Raise small max_tokens values to this floor; 0 disables. Tiny limits can make a
                reasoning model spend them all thinking and return an empty reply.
              </p>
            </div>
          </div>
        </details>

        {error && (
          <div
            role="alert"
            className="rounded-[10px] border border-vw-danger/55 bg-vw-danger-bg/30 px-3 py-2 text-[13px] text-vw-danger-fg"
          >
            {error}
          </div>
        )}
        <div className="flex justify-end gap-2">
          <button type="button" className={btn()} onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className={btn("primary")}
            onClick={() => void save()}
            disabled={!valid || busy}
          >
            {rule ? "Save rule" : "Create rule"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
