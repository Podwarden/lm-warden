"use client";

// Suggest row at the top of Memory & context (plan §4.4 -- proximity: the
// suggestion sits next to the fields it drives). On click it fetches
// /api/models/{id}/suggest-config, shows the diff against the current draft,
// and offers Apply / Dismiss. `disclaimer` is shown as the rationale and is
// never applied; null values mean "no opinion" and are dropped before diffing.

import { useEffect, useMemo, useRef, useState } from "react";
import { authFetchJSON } from "@/lib/auth-fetch";
import type { Draft } from "@/lib/model-settings";
import {
  DiffList,
  PANEL_BUTTON_CLASS,
  PANEL_BUTTON_STRONG_CLASS,
  computeDiff,
} from "./diff-list";

export interface SuggestResponse {
  gpu_memory_utilization?: number | null;
  max_model_len?: number | null;
  kv_cache_dtype?: string | null;
  disclaimer?: string;
  // Permit future additions without breaking the FE.
  [key: string]: unknown;
}

export interface SuggestPanelProps {
  modelId: string;
  disabled: boolean;
  draft: Draft;
  onApply: (sparse: Record<string, unknown>) => void;
}

export function SuggestPanel({ modelId, disabled, draft, onApply }: SuggestPanelProps) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<SuggestResponse | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  // Focus moves into the result when it opens, so Escape and Tab work at
  // once; close() hands it back to the button.
  useEffect(() => {
    if (open) dialogRef.current?.focus();
  }, [open]);

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

  function close() {
    setOpen(false);
    // Focus goes back to the button that opened the result (plan §7).
    buttonRef.current?.focus();
  }

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
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
        <span className="text-[12.5px] leading-[1.45] text-chat-muted">
          Not sure? Get values sized from the model config and the VRAM on these GPUs.
        </span>
        <button
          ref={buttonRef}
          type="button"
          className={PANEL_BUTTON_CLASS}
          onClick={fetchSuggestion}
          disabled={disabled || loading}
          aria-expanded={open}
          data-testid="suggest-fetch"
        >
          {loading ? "Loading…" : "Suggest values"}
        </button>
      </div>
      {err && (
        <p className="text-[12.5px] text-vw-danger-fg" data-testid="suggest-error">
          {err}
        </p>
      )}
      {open && result && (
        <div
          ref={dialogRef}
          tabIndex={-1}
          role="dialog"
          aria-label="Suggested values"
          data-testid="suggest-result"
          onKeyDown={(e) => {
            if (e.key === "Escape") close();
          }}
          className="rounded-lg border border-vw-rule-soft/70 bg-chat-surface-2/60 p-3 focus:outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent"
        >
          {result.disclaimer && (
            <p
              className="text-[12.5px] leading-[1.45] text-chat-muted"
              data-testid="suggest-rationale"
            >
              {result.disclaimer}
            </p>
          )}
          <DiffList rows={diff} testIdPrefix="suggest-diff" />
          <div className="mt-3 flex justify-end gap-2">
            <button
              type="button"
              className={PANEL_BUTTON_CLASS}
              onClick={close}
              data-testid="suggest-dismiss"
            >
              Dismiss
            </button>
            <button
              type="button"
              className={PANEL_BUTTON_STRONG_CLASS}
              onClick={() => {
                onApply(applicable);
                close();
              }}
              disabled={diff.length === 0}
              data-testid="suggest-apply"
            >
              Apply
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
