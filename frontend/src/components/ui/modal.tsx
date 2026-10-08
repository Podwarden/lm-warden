"use client";

import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "./button";

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  className?: string;
  /**
   * Optional ref to the element that should receive focus when the modal opens.
   * Useful for destructive confirm dialogs that should focus the safe (Cancel)
   * action by default rather than the first DOM-order focusable (the header
   * Close button).
   */
  initialFocusRef?: React.RefObject<HTMLElement | null>;
  /**
   * Optional content rendered to the right of the title, before the Close button.
   * Used by the override / doctor modals to show a context badge in the header.
   */
  headerExtra?: React.ReactNode;
  /** "default" → max-w-lg (32rem); "lg" → max-w-xl (36rem). */
  size?: "default" | "lg";
}

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]):not([aria-disabled="true"]), textarea:not([disabled]):not([aria-disabled="true"]), input:not([disabled]):not([aria-disabled="true"]), select:not([disabled]):not([aria-disabled="true"]), [contenteditable=""], [contenteditable="true"], [tabindex]:not([tabindex="-1"]):not([aria-disabled="true"])';

export function Modal({
  open,
  onClose,
  title,
  children,
  className,
  initialFocusRef,
  headerExtra,
  size = "default",
}: ModalProps) {
  const [mounted, setMounted] = useState(false);
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);
  const titleId = useId();

  // Stash the latest onClose / initialFocusRef in refs so the open-effect's
  // dep array can stay limited to [open]. Without this, every parent re-render
  // that changes the inline `onClose` arrow (which is essentially every
  // re-render at every existing call site) would tear down and re-run the
  // effect, restoring focus to the opener and then re-focusing into the dialog
  // — a focus-thrash that drops keystrokes and breaks IME composition while
  // the user is typing in a Modal-hosted form.
  const onCloseRef = useRef(onClose);
  const initialFocusRefRef = useRef(initialFocusRef);
  useEffect(() => {
    onCloseRef.current = onClose;
    initialFocusRefRef.current = initialFocusRef;
  });

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!open) return;

    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;

    const handleKeydown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        onCloseRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const root = dialogRef.current;
      if (!root) return;
      const focusable = Array.from(
        root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)
      ).filter((el) => !el.hasAttribute("aria-hidden"));
      if (focusable.length === 0) {
        e.preventDefault();
        root.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeydown);
    document.body.style.overflow = "hidden";
    // Prevent screen-reader virtual-cursor (VoiceOver/NVDA arrow-key navigation)
    // from reaching content behind the modal. The keyboard Tab trap alone is not
    // sufficient — browse-mode can still wander outside the dialog without inert.
    document.getElementById("app-root")?.setAttribute("inert", "");

    // Focus the dialog (or first focusable descendant) on next tick so the
    // portal child has actually mounted. If the caller provided an explicit
    // initialFocusRef and that element is focusable, prefer it.
    queueMicrotask(() => {
      const root = dialogRef.current;
      if (!root) return;
      const explicit = initialFocusRefRef.current?.current ?? null;
      if (explicit && typeof explicit.focus === "function") {
        explicit.focus();
        return;
      }
      const firstFocusable = root.querySelector<HTMLElement>(FOCUSABLE_SELECTOR);
      (firstFocusable ?? root).focus();
    });

    return () => {
      document.removeEventListener("keydown", handleKeydown);
      document.body.style.overflow = "";
      document.getElementById("app-root")?.removeAttribute("inert");
      // Restore focus to the element that opened the modal.
      const prev = previouslyFocusedRef.current;
      if (prev && typeof prev.focus === "function") {
        prev.focus();
      }
    };
  }, [open]);

  if (!open || !mounted) return null;

  return createPortal(
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4" data-modal-overlay>
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden
      />
      <div
        ref={dialogRef}
        tabIndex={-1}
        className={cn(
          "relative z-[1] flex max-h-[85vh] w-full flex-col rounded-lg border border-slate-700 bg-slate-900 shadow-xl outline-none",
          size === "lg" ? "max-w-xl" : "max-w-lg",
          className
        )}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-slate-700 p-4">
          <div className="flex items-center gap-3">
            <h2 id={titleId} className="text-lg font-semibold">
              {title}
            </h2>
            {headerExtra}
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-8 w-8 p-0"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">{children}</div>
      </div>
    </div>,
    document.body
  );
}
