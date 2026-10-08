import Link from "next/link";
import { cn } from "@/lib/utils";

/**
 * One line that answers "what can I change right now?" (plan
 * 2026-10-04-settings-redesign §2, §6, §7). Replaces the old wall-of-text
 * loaded banner.
 *
 * - loaded   → green, role=alert, testid `settings-loaded-banner`
 * - conflict → amber, role=alert, same testid (a save hit the 409
 *              unload-first guard); wins over loaded
 * - anything else (registered / pulled / failed / loading / unloading /
 *   pulling) → neutral role=status, so a field error can hold the page's
 *   single alert slot.
 *
 * Only `status === "loaded"` counts as loaded — the same rule that locks the
 * engine fields.
 */
export function StatusStrip({
  status,
  conflict,
  modelId,
}: {
  status: string;
  conflict: boolean;
  modelId: string;
}) {
  const href = `/models/${encodeURIComponent(modelId)}`;
  const variant = conflict ? "conflict" : status === "loaded" ? "loaded" : "unloaded";
  const linkClass =
    "whitespace-nowrap font-medium text-chat-fg underline underline-offset-[3px] rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent";

  const shell =
    "flex items-start gap-3 rounded-[10px] border px-3.5 py-3 text-[13.5px] leading-normal";

  if (variant === "unloaded") {
    return (
      <div
        role="status"
        data-testid="settings-status-strip"
        data-variant="unloaded"
        className={cn(shell, "border-vw-rule-soft/70 bg-chat-surface/60")}
      >
        <svg
          viewBox="0 0 18 18"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
          aria-hidden="true"
          className="mt-px h-[18px] w-[18px] flex-none text-chat-muted"
        >
          <circle cx="9" cy="9" r="7" />
        </svg>
        <p className="m-0 text-chat-muted">
          <strong className="font-semibold text-chat-fg">Everything is editable.</strong> Engine
          changes take effect the next time you load the model.{" "}
          <Link href={href} className={linkClass}>
            Load on the model page <span aria-hidden="true">→</span>
          </Link>
        </p>
      </div>
    );
  }

  if (variant === "conflict") {
    return (
      <div
        role="alert"
        aria-live="polite"
        data-testid="settings-loaded-banner"
        data-variant="conflict"
        className={cn(shell, "border-vw-amber-fg/45 bg-vw-amber-bg/30")}
      >
        <svg
          viewBox="0 0 18 18"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          aria-hidden="true"
          className="mt-px h-[18px] w-[18px] flex-none text-vw-amber-fg"
        >
          <circle cx="9" cy="9" r="7" />
          <path d="M9 5.2v4.6M9 12.6v.1" />
        </svg>
        <p className="m-0 text-chat-muted">
          <strong className="font-semibold text-chat-fg">Not saved — the model is loaded.</strong>{" "}
          Engine settings must be unloaded before editing them. Unload it, then save again, or
          Reset to drop your changes.{" "}
          <Link href={href} className={linkClass}>
            Unload on the model page <span aria-hidden="true">→</span>
          </Link>
        </p>
      </div>
    );
  }

  return (
    <div
      role="alert"
      aria-live="polite"
      data-testid="settings-loaded-banner"
      data-variant="loaded"
      className={cn(shell, "border-vw-live/35 bg-vw-ok-bg/20")}
    >
      <svg
        viewBox="0 0 18 18"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        aria-hidden="true"
        className="mt-px h-[18px] w-[18px] flex-none text-vw-live"
      >
        <circle cx="9" cy="9" r="3.2" fill="currentColor" stroke="none" />
        <circle cx="9" cy="9" r="7" />
      </svg>
      <p className="m-0 text-chat-muted">
        <strong className="font-semibold text-chat-fg">Live changes only.</strong> Replica routing
        and capabilities apply immediately. Engine settings are locked — the model must be
        unloaded before editing them.{" "}
        <Link href={href} className={linkClass}>
          Unload on the model page <span aria-hidden="true">→</span>
        </Link>
      </p>
    </div>
  );
}
