"use client";

// The state strip (plan §4.1, §7): one sentence that says the router's state
// and its consequence, then the next action. The container is in the DOM from
// the first render so its live region announces in-place changes: role
// "status" for the calm states, "alert" for danger/amber issues.

import { useId, useState } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import type { RouterIssue, RouterStrip, StripAction, StripPart, StripTone } from "@/lib/router";
import { btn, FOCUS, STRIP_ACTIONS, STRIP_ICON, STRIP_TEXT, STRIP_TITLE, strip as stripClass } from "./styles";

// .btn.sm — 30 px, 40 px tall on phones (plan §7 targets).
const ACTION_BTN = btn("default", `min-h-[30px] px-2.5 py-[3px] text-[12.5px] max-[759px]:min-h-10 ${FOCUS}`);

export function Parts({ parts }: { parts: StripPart[] }) {
  return (
    <>
      {parts.map((p, i) =>
        typeof p === "string" ? (
          <span key={i}>{p}</span>
        ) : (
          <code key={i} className="font-mono text-[0.92em]">
            {p.code}
          </code>
        ),
      )}
    </>
  );
}

function StripIcon({ tone, pulse }: { tone: StripTone; pulse?: boolean }) {
  const cls = STRIP_ICON[tone];
  if (tone === "danger" || tone === "amber") {
    return (
      <svg className={cls} viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
        <path d="M9 2 1.8 15h14.4L9 2Z" />
        <path d="M9 7v4M9 13.2v.1" />
      </svg>
    );
  }
  if (tone === "ok") {
    return (
      <svg className={cls} viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
        <circle cx="9" cy="9" r="3.2" fill="currentColor" stroke="none" />
        <circle cx="9" cy="9" r="7" />
      </svg>
    );
  }
  return (
    <svg className={cls} viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
      <circle cx="9" cy="9" r="7" />
      {pulse && (
        <circle
          cx="9"
          cy="9"
          r="3"
          fill="currentColor"
          stroke="none"
          className="motion-safe:animate-pulse"
        />
      )}
    </svg>
  );
}

interface ActionProps {
  action: StripAction;
  onAction: (a: StripAction) => void;
  busy: boolean;
}

function Action({ action, onAction, busy }: ActionProps) {
  if ("href" in action) {
    if (action.href.startsWith("#")) {
      return (
        <a href={action.href} className={ACTION_BTN}>
          {action.label}
        </a>
      );
    }
    return (
      <Link href={action.href} className={ACTION_BTN}>
        {action.label}
      </Link>
    );
  }
  return (
    <button type="button" className={ACTION_BTN} disabled={busy} onClick={() => onAction(action)}>
      {action.label}
    </button>
  );
}

function Actions({ actions, onAction, busy }: { actions: StripAction[]; onAction: (a: StripAction) => void; busy: boolean }) {
  if (actions.length === 0) return null;
  return (
    <div className={STRIP_ACTIONS}>
      {actions.map((a) => (
        <Action key={`${a.kind}-${a.label}`} action={a} onAction={onAction} busy={busy} />
      ))}
    </div>
  );
}

function MoreIssues({ issues, onAction, busy }: { issues: RouterIssue[]; onAction: (a: StripAction) => void; busy: boolean }) {
  const [open, setOpen] = useState(false);
  const listId = useId();
  return (
    <div className="mt-2">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={listId}
        onClick={() => setOpen((o) => !o)}
        className={cn("rounded-sm text-[13px] font-medium text-chat-fg underline underline-offset-[3px]", FOCUS)}
      >
        and {issues.length} more
      </button>
      {open && (
        <ul id={listId} className="m-0 mt-2 grid list-none gap-2.5 border-t border-vw-rule-soft/45 p-0 pt-2.5">
          {issues.map((it, i) => (
            <li key={`${it.code}-${it.ruleId ?? i}`}>
              <p className={STRIP_TEXT}>
                <strong className={STRIP_TITLE[it.tone]}>
                  <Parts parts={it.title} />
                </strong>{" "}
                <Parts parts={it.body} />
              </p>
              <Actions actions={it.actions} onAction={onAction} busy={busy} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

interface Props {
  strip: RouterStrip | null;
  /** A failed action (switch, pause): its server detail, as an alert. */
  error: string | null;
  /** Clears `error` (its Dismiss button). */
  onDismissError?: () => void;
  onAction: (a: StripAction) => void;
  /** An action is in flight: its buttons are disabled. */
  busy?: boolean;
}

export function RouterStatusStrip({ strip, error, onDismissError, onAction, busy = false }: Props) {
  return (
    <div data-testid="router-strip" className="mt-4">
      {error && (
        <div role="alert" className={cn(stripClass("danger"), "mb-2")}>
          <StripIcon tone="danger" />
          <p className={cn(STRIP_TEXT, "min-w-0 flex-1")}>
            <strong className={STRIP_TITLE.danger}>That didn&apos;t work.</strong> {error}
          </p>
          {onDismissError && (
            <button type="button" className={ACTION_BTN} onClick={onDismissError}>
              Dismiss
            </button>
          )}
        </div>
      )}
      <div role={strip?.role ?? "status"} className={strip ? stripClass(strip.tone) : undefined}>
        {strip && (
          <>
            <StripIcon tone={strip.tone} pulse={strip.pulse} />
            <div className="min-w-0 flex-1">
              <p className={STRIP_TEXT}>
                <strong className={STRIP_TITLE[strip.tone]}>
                  <Parts parts={strip.title} />
                </strong>{" "}
                <Parts parts={strip.body} />
              </p>
              {strip.note && <p className={cn(STRIP_TEXT, "mt-1 text-[12.5px]")}>{strip.note}</p>}
              <Actions actions={strip.actions} onAction={onAction} busy={busy} />
              {strip.more.length > 0 && <MoreIssues issues={strip.more} onAction={onAction} busy={busy} />}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
