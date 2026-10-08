"use client";

// The requirements checklist (plan §5.3): one row per requirement with a mark,
// the text and the measured value. The state is always in words ("met",
// "below", "not met", "check yourself"), never colour alone. Rows never block
// Copy.

import { cn } from "@/lib/utils";
import type { CheckState, RequirementCheck } from "@/lib/connect";

const WORD: Record<CheckState, string> = {
  ok: "met",
  warn: "below",
  fail: "not met",
  unknown: "unknown",
};

const TONE: Record<CheckState, string> = {
  ok: "text-vw-ok-fg",
  warn: "text-vw-amber-fg",
  fail: "text-vw-danger-fg",
  unknown: "text-chat-muted",
};

function Mark({ state, note }: { state: CheckState; note: boolean }) {
  const cls = cn("mt-[3px] h-[14px] w-[14px] flex-none", note ? "text-chat-dim" : TONE[state]);
  if (note) {
    return (
      <svg viewBox="0 0 14 14" className={cls} aria-hidden="true">
        <circle cx="7" cy="7" r="2" fill="currentColor" />
      </svg>
    );
  }
  if (state === "ok") {
    return (
      <svg viewBox="0 0 14 14" className={cls} fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
        <path d="M2.5 7.5 5.5 10.5 11.5 3.5" />
      </svg>
    );
  }
  if (state === "fail") {
    return (
      <svg viewBox="0 0 14 14" className={cls} fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
        <path d="M3.5 3.5l7 7M10.5 3.5l-7 7" />
      </svg>
    );
  }
  if (state === "warn") {
    return (
      <svg viewBox="0 0 14 14" className={cls} aria-hidden="true">
        <circle cx="7" cy="7" r="4" fill="currentColor" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 14 14" className={cls} fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
      <circle cx="7" cy="7" r="5.5" />
      <path d="M5.4 5.6a1.7 1.7 0 1 1 2.3 1.6c-.5.2-.7.5-.7 1v.3M7 10.3v.1" />
    </svg>
  );
}

export function Requirements({ checks }: { checks: RequirementCheck[] }) {
  if (checks.length === 0) return null;
  return (
    <ul aria-label="Requirements" className="m-0 grid list-none gap-2 p-0 text-[13px]">
      {checks.map((c) => {
        // A text-only requirement (client version, reachability) has nothing the
        // page can measure: it is a note for the operator, not an unknown.
        const note = c.state === "unknown" && !c.detail;
        return (
          <li key={c.id} data-testid={`req-${c.id}`} data-state={c.state} className="flex min-w-0 gap-2">
            <Mark state={c.state} note={note} />
            <span className="min-w-0">
              <span>{c.text}</span>
              {note ? (
                <span className="block text-[12.5px] text-chat-muted">Check this yourself.</span>
              ) : (
                <span className="block text-[12.5px] text-chat-muted [overflow-wrap:anywhere]">
                  {c.detail}
                  {c.state !== "unknown" && (
                    <>
                      {c.detail ? " — " : ""}
                      <span className={cn("font-semibold", TONE[c.state])}>{WORD[c.state]}</span>
                    </>
                  )}
                </span>
              )}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
