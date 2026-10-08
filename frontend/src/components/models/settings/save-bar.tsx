"use client";

import { useEffect, useId, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { cn } from "@/lib/utils";

export interface SaveBarRow {
  /** Field key; rendered as `data-diff-key` on the diff row. */
  key?: string;
  /** Human label (MODEL_HINTS label), never the raw key. */
  label: string;
  before: string;
  after: string;
  /** In LOADED_EDITABLE_KEYS: applies immediately, even while loaded. */
  live: boolean;
}

export interface SaveBarProps {
  rows: SaveBarRow[];
  /** Phrase that follows "Can't save: " (see layoutInvalidReason). */
  invalidReason: string | null;
  saving: boolean;
  canSave: boolean;
  isLoaded: boolean;
  saveError: string | null;
  onSave: () => void;
  onReset: () => void;
  /** id of the input "Fix" focuses (e.g. "dp"); no Fix button without it. */
  fixTargetId?: string;
  /** Runs before Fix focuses the target, e.g. to expand the collapsed
   *  section that holds it. Flushed synchronously so the target is visible. */
  onFix?: () => void;
}

const btnBase =
  "inline-flex min-h-[34px] items-center justify-center gap-1.5 whitespace-nowrap rounded-lg border px-3.5 py-1.5 text-[13px] font-medium " +
  "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent " +
  "disabled:cursor-not-allowed disabled:opacity-45 max-sm:min-h-10 max-sm:flex-1";

const tagBase = "whitespace-nowrap rounded-full px-[7px] py-px text-[11.5px] font-semibold";
const tagNext = "bg-vw-amber-bg/60 text-vw-amber-fg";
const tagLive = "bg-vw-ok-bg/60 text-vw-ok-fg";

/**
 * Sticky save bar for the model settings page (plan
 * 2026-10-04-settings-redesign §4.1). One place for the unsaved total, the
 * diff, the reload signal and the actions.
 *
 * Contract the page tests rely on: exactly ONE button whose name matches
 * /save/i exists at all times — the "N unsaved changes" text is a plain
 * <strong>, never a button. The only alert it can render is the save error;
 * the invalid reason sits in the polite live summary.
 *
 * Owns the ⌘S / Ctrl+S listener (window keydown, added on mount, removed on
 * unmount): it always prevents the browser's save-page dialog and calls
 * `onSave` only when `canSave`.
 */
export function SaveBar({
  rows,
  invalidReason,
  saving,
  canSave,
  isLoaded,
  saveError,
  onSave,
  onReset,
  fixTargetId,
  onFix,
}: SaveBarProps) {
  const reviewId = useId();
  const [reviewOpen, setReviewOpen] = useState(false);
  const [isMac, setIsMac] = useState(true);
  const barRef = useRef<HTMLDivElement>(null);

  const dirty = rows.length;
  const live = rows.filter((r) => r.live).length;
  const next = dirty - live;
  const showReview = !invalidReason && !saving && dirty > 0;
  const expanded = showReview && reviewOpen;

  // Latest props for the once-registered key listener.
  const latest = useRef({ canSave, onSave });
  useEffect(() => {
    latest.current = { canSave, onSave };
  });

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!(e.metaKey || e.ctrlKey) || e.altKey || e.shiftKey) return;
      if (e.key.toLowerCase() !== "s") return;
      e.preventDefault();
      if (latest.current.canSave) latest.current.onSave();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    const p = typeof navigator !== "undefined" ? navigator.platform || navigator.userAgent : "";
    setIsMac(/mac|iphone|ipad/i.test(p));
  }, []);

  // Keep keyboard focus from landing behind the sticky bar (WCAG 2.4.11):
  // reserve its height as the document's bottom scroll padding.
  useEffect(() => {
    const el = barRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const html = document.documentElement;
    const prev = html.style.scrollPaddingBottom;
    const ro = new ResizeObserver(() => {
      html.style.scrollPaddingBottom = `${Math.ceil(el.getBoundingClientRect().height) + 8}px`;
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      html.style.scrollPaddingBottom = prev;
    };
  }, []);

  function fix() {
    if (!fixTargetId) return;
    if (onFix) flushSync(onFix);
    const target = document.getElementById(fixTargetId);
    if (!target) return;
    target.focus();
    target.scrollIntoView?.({ block: "center" });
  }

  // "applies now" only when every change is live; a loaded row holding an
  // engine edit cannot save at all, so it must not promise anything.
  const saveLabel =
    saving ? "Saving…" : isLoaded && dirty > 0 && live === dirty ? "Save — applies now" : "Save";

  let summary;
  if (saving) {
    summary = <span>Saving…</span>;
  } else if (invalidReason) {
    summary = (
      <span className="font-medium text-vw-danger-fg">
        Can&apos;t save: {invalidReason}.
        {fixTargetId && (
          <>
            {" "}
            <button
              type="button"
              onClick={fix}
              className="rounded-sm underline underline-offset-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent"
            >
              Fix
            </button>
          </>
        )}
      </span>
    );
  } else if (dirty === 0) {
    summary = <span>No unsaved changes</span>;
  } else {
    summary = (
      <>
        <strong className="font-semibold text-chat-fg">
          {dirty} unsaved {dirty === 1 ? "change" : "changes"}
        </strong>
        {next > 0 && (
          <span data-testid="save-bar-tag-next" className={cn(tagBase, tagNext)}>
            {next} on next load
          </span>
        )}
        {live > 0 && (
          <span data-testid="save-bar-tag-live" className={cn(tagBase, tagLive)}>
            {live} live
          </span>
        )}
      </>
    );
  }

  return (
    <div
      ref={barRef}
      data-testid="settings-save-bar"
      className="sticky bottom-0 z-20 mt-8"
    >
      <div className="rounded-t-xl border border-b-0 border-vw-rule-soft bg-chat-surface shadow-[0_-6px_18px_rgb(120_80_20/0.10)] dark:shadow-[0_-8px_24px_rgb(0_0_0/0.25)]">
        {saveError && (
          <div
            role="alert"
            data-testid="settings-save-error"
            className="mx-4 mt-2.5 rounded-lg border border-chat-negative/60 bg-vw-danger-bg/25 px-3 py-2 text-[13px] text-vw-danger-fg [overflow-wrap:anywhere]"
          >
            {saveError}
          </div>
        )}
        <div className="flex flex-wrap items-center gap-3 px-4 py-2.5">
          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2.5 text-[13.5px] text-chat-muted tabular-nums">
            <div
              data-testid="settings-save-summary"
              aria-live="polite"
              className="flex min-w-0 flex-wrap items-center gap-2.5"
            >
              {summary}
            </div>
            {showReview && (
              <button
                type="button"
                aria-expanded={expanded}
                aria-controls={reviewId}
                onClick={() => setReviewOpen((o) => !o)}
                className={cn(
                  btnBase,
                  "min-h-7 border-vw-rule-soft px-2.5 py-0.5 text-[12.5px] text-chat-fg hover:bg-chat-surface-2/60 max-sm:min-h-7 max-sm:flex-none",
                )}
              >
                Review
              </button>
            )}
          </div>
          <div className="ml-auto flex gap-2 max-sm:w-full">
            <button
              type="button"
              data-testid="settings-reset"
              onClick={onReset}
              disabled={saving || dirty === 0}
              className={cn(btnBase, "border-vw-rule-soft bg-transparent text-chat-fg hover:bg-chat-surface-2/60")}
            >
              Reset
            </button>
            <button
              type="button"
              data-testid="settings-save"
              onClick={onSave}
              disabled={!canSave}
              aria-keyshortcuts="Meta+S Control+S"
              className={cn(
                btnBase,
                "border-vw-btn bg-vw-btn font-semibold text-vw-on-btn hover:bg-vw-btn-hover disabled:hover:bg-vw-btn",
              )}
            >
              {saveLabel}
              <kbd
                aria-hidden="true"
                className="rounded border border-current px-1 font-sans text-[11px] opacity-80 max-sm:hidden"
              >
                {isMac ? "⌘S" : "Ctrl S"}
              </kbd>
            </button>
          </div>
        </div>
        <div
          id={reviewId}
          hidden={!expanded}
          className="max-h-[40vh] overflow-auto border-t border-vw-rule-soft/60 px-4 pb-3 pt-2.5"
        >
          <ul className="m-0 grid list-none gap-1.5 p-0" aria-label="Unsaved changes">
            {rows.map((r) => (
              <li
                key={r.key ?? r.label}
                data-testid="save-bar-diff-row"
                data-diff-key={r.key}
                className="grid grid-cols-[minmax(140px,220px)_1fr_auto] items-baseline gap-2.5 text-[13px] max-sm:grid-cols-[minmax(0,1fr)_auto]"
              >
                <span className="font-medium text-chat-fg">{r.label}</span>
                <span className="!font-mono text-[12.5px] text-chat-muted [overflow-wrap:anywhere] max-sm:col-span-full max-sm:row-start-2">
                  <span className="sr-only">from </span>
                  <s>{r.before === "" ? "(empty)" : r.before}</s>{" "}
                  <span aria-hidden="true">→</span>
                  <span className="sr-only"> to </span>{" "}
                  <b className="font-medium text-vw-amber-fg">{r.after === "" ? "(empty)" : r.after}</b>
                </span>
                <span className={cn(tagBase, r.live ? tagLive : tagNext)}>
                  {r.live ? "live" : "next load"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
