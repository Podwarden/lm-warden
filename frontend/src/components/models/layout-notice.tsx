"use client";

import { useState } from "react";
import { authFetch } from "@/lib/auth-fetch";

/**
 * Boot-reconcile outcome for a model's parallel layout (#286).
 *
 * A warning means `extra_args` still carry a layout the boot pass could not
 * fix; only editing them clears it. An info note (the row was promoted or
 * cleaned) can be dismissed, which calls DELETE /api/models/{id}/layout-notice.
 */
export function LayoutNotice({
  modelId,
  level,
  message,
  onDismissed,
}: {
  modelId: string;
  level: "info" | "warning";
  message: string;
  onDismissed: () => unknown;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const warning = level === "warning";

  async function dismiss() {
    setBusy(true);
    setError(null);
    try {
      const r = await authFetch(
        `/api/models/${encodeURIComponent(modelId)}/layout-notice`,
        { method: "DELETE" },
      );
      if (!r.ok) {
        setError(`Could not dismiss (HTTP ${r.status})`);
        return;
      }
      await onDismissed();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      role={warning ? "alert" : "status"}
      data-testid="layout-notice"
      data-level={level}
      className={
        warning
          ? "rounded-[10px] border border-vw-amber-fg/45 bg-vw-amber-bg/30 px-3.5 py-3 text-sm text-chat-fg"
          : "flex items-start justify-between gap-3 rounded-[10px] border border-vw-rule-soft/70 bg-chat-surface/60 px-3.5 py-3 text-sm text-chat-fg"
      }
    >
      <div>
        <span className="font-semibold">
          {warning ? "Parallel layout needs attention:" : "Parallel layout updated:"}
        </span>{" "}
        {message}
        {error && (
          <div role="alert" className="mt-1 text-vw-danger-fg">
            {error}
          </div>
        )}
      </div>
      {!warning && (
        <button
          type="button"
          disabled={busy}
          onClick={() => void dismiss()}
          className="inline-flex h-8 flex-none items-center rounded-lg border border-vw-rule-soft px-3 text-xs font-medium text-chat-fg hover:bg-chat-surface-2/60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent disabled:cursor-not-allowed disabled:opacity-50"
        >
          Dismiss
        </button>
      )}
    </div>
  );
}
