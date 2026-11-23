"use client";

// Small pieces the per-key cache screens share: the marks, the confirmation
// dialog, the models list with its isolation marks, and the use-against-share bar.

import { useRef, type ReactNode } from "react";
import { Modal } from "@/components/ui/modal";
import { btn } from "@/components/tokens/detail/styles";
import { cn } from "@/lib/utils";
import { MARK, shareBar, type ModelIsolation } from "@/lib/kv";

export type MarkTone = "info" | "warn" | "ok" | "bad" | "idle";

const MARK_TONE: Record<MarkTone, string> = {
  info: "border-vw-info-fg/40 bg-vw-info-bg/50 text-vw-info-fg",
  warn: "border-chat-accent/45 bg-vw-amber-bg/50 text-vw-amber-fg",
  ok: "border-transparent bg-vw-ok-bg/55 text-vw-ok-fg",
  bad: "border-transparent bg-vw-danger-bg/45 text-vw-danger-fg",
  idle: "border-transparent bg-chat-surface-2 text-chat-muted",
};

/** A small pill: "isolated", "isolated keys refused", "not isolated at the provider". */
export function Mark({
  tone,
  children,
  title,
  testId,
  className,
  wrap = false,
}: {
  tone: MarkTone;
  children: ReactNode;
  title?: string;
  testId?: string;
  className?: string;
  /** Let a long mark wrap (a list cell on a phone) instead of staying on one line. */
  wrap?: boolean;
}) {
  return (
    <span
      data-testid={testId}
      title={title}
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-px text-[11px] font-medium leading-[1.5]",
        wrap ? "whitespace-normal [overflow-wrap:anywhere]" : "whitespace-nowrap",
        MARK_TONE[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/** A confirmation dialog: a title, what happens, Cancel (focused) and the action. */
export function ConfirmDialog({
  open,
  title,
  lead,
  points,
  confirm,
  danger,
  busy,
  error,
  onCancel,
  onConfirm,
}: {
  open: boolean;
  title: string;
  lead?: ReactNode;
  points?: ReactNode[];
  confirm: string;
  danger?: boolean;
  busy?: boolean;
  error?: string | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  return (
    <Modal open={open} onClose={onCancel} title={title} initialFocusRef={cancelRef}>
      <div className="text-[13px] leading-[1.5] text-chat-muted">
        {lead && <p className="m-0">{lead}</p>}
        {points && points.length > 0 && (
          <ul className="mb-0 mt-2 list-disc space-y-1 pl-5">
            {points.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        )}
        {error && (
          <p role="alert" className="mb-0 mt-2 text-vw-danger-fg">
            {error}
          </p>
        )}
        <div className="mt-4 flex justify-end gap-2">
          <button ref={cancelRef} type="button" className={btn()} onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button type="button" className={btn(danger ? "danger" : "primary")} onClick={onConfirm} disabled={busy}>
            {confirm}
          </button>
        </div>
      </div>
    </Modal>
  );
}

/**
 * The models a key may use. For an isolated key each one says whether its engine keeps
 * the key's cache apart; `refusedOnly` marks only the ones that refuse the key.
 */
export function ModelIsolationList({
  models,
  isolated,
  refusedOnly = false,
  className,
}: {
  models: ModelIsolation[];
  isolated: boolean;
  refusedOnly?: boolean;
  className?: string;
}) {
  return (
    <ul data-testid={isolated ? "kv-model-isolation" : "kv-model-list"} className={cn("m-0 list-none space-y-1 p-0", className)}>
      {models.map((m) => (
        <li key={m.id} className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="!font-mono text-[13px]">{m.id}</span>
          {isolated && m.cache_isolation === "unsupported" && <Mark tone="warn" wrap>{MARK.cannotSeparate}</Mark>}
          {isolated && !refusedOnly && m.cache_isolation === "isolated" && <Mark tone="info" wrap>{MARK.keptSeparate}</Mark>}
        </li>
      ))}
    </ul>
  );
}

/** Use against share: the fill is the use, the white tick the share; red past the share. */
export function ShareBar({
  used,
  share,
  label,
  className,
}: {
  used: number;
  share: number;
  /** Use and share in words ("2.1 GiB of a 2.9 GiB share"); "over its share" is added when it is. */
  label: string;
  className?: string;
}) {
  const b = shareBar(used, share);
  return (
    <div
      role="img"
      aria-label={b.over ? `${label}, over its share` : label}
      className={cn("relative h-2 rounded-full bg-chat-surface-2", className)}
    >
      <div
        className={cn("h-full rounded-full", b.over ? "bg-vw-danger" : "bg-chat-accent-strong")}
        style={{ width: `${(b.fill * 100).toFixed(1)}%` }}
      />
      {b.tick !== null && (
        <span className="absolute -top-1 h-4 w-[2px] bg-chat-fg" style={{ left: `calc(${(b.tick * 100).toFixed(1)}% - 1px)` }} />
      )}
    </div>
  );
}

/** A one-line notice inside a card: the store does not answer, or nothing uses it. */
export function Notice({ tone, children, testId }: { tone: "bad" | "info"; children: ReactNode; testId?: string }) {
  return (
    <p
      role={tone === "bad" ? "status" : undefined}
      data-testid={testId}
      className={cn(
        "m-0 rounded-md border px-3 py-2 text-[13px] leading-[1.45]",
        tone === "bad"
          ? "border-vw-danger/40 bg-vw-danger-bg/30 text-vw-danger-fg"
          : "border-vw-rule-soft/60 bg-chat-surface-2/50 text-chat-muted",
      )}
    >
      {children}
    </p>
  );
}
