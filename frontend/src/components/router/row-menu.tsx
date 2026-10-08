"use client";

// The routing map's ⋯ menu (plan §4.4, §7): a menu button with a role="menu"
// list. Keyboard: Enter/Space/ArrowDown open on the first item, ArrowUp on the
// last; ArrowUp/Down/Home/End move; Escape closes and returns focus to the
// button; Tab closes. No dependency — a few lines of focus management.

import { forwardRef, useEffect, useId, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { FOCUS, ICON_BTN } from "@/components/router/styles";

export interface RowMenuItem {
  key: string;
  /** Visible text. */
  label: string;
  /** Accessible name when it must differ from the text (kept test labels). */
  ariaLabel?: string;
  onSelect: () => void;
  disabled?: boolean;
  danger?: boolean;
  /** Extra classes, e.g. `min-[760px]:hidden` for the mobile-only items. */
  className?: string;
}

export interface RowMenuProps {
  /** Accessible name of the trigger, e.g. "Actions for claude-haiku*". */
  label: string;
  items: RowMenuItem[];
  disabled?: boolean;
}

// .menu: a small popover card under the trigger, right-aligned.
const MENU =
  "absolute right-0 top-full z-30 mt-1 min-w-[180px] rounded-[10px] border border-vw-rule-soft/70 bg-chat-surface p-1 shadow-lg";
const ITEM =
  "flex w-full items-center rounded-md px-3 py-2 text-left text-[13px] text-chat-fg hover:bg-chat-surface-2/70 focus:bg-chat-surface-2/70 focus:outline-none aria-disabled:cursor-not-allowed aria-disabled:opacity-45";

function Dots() {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" className="h-4 w-4">
      <circle cx="3" cy="8" r="1.4" />
      <circle cx="8" cy="8" r="1.4" />
      <circle cx="13" cy="8" r="1.4" />
    </svg>
  );
}

export const RowMenu = forwardRef<HTMLButtonElement, RowMenuProps>(function RowMenu(
  { label, items, disabled },
  ref,
) {
  const [open, setOpen] = useState(false);
  const [initial, setInitial] = useState<"first" | "last">("first");
  const wrapRef = useRef<HTMLSpanElement | null>(null);
  const btnRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const menuId = useId();

  function setBtn(el: HTMLButtonElement | null) {
    btnRef.current = el;
    if (typeof ref === "function") ref(el);
    else if (ref) ref.current = el;
  }

  function itemEls(): HTMLElement[] {
    // Skip items hidden by CSS (the mobile-only ones at desktop width).
    return Array.from(
      menuRef.current?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [],
    ).filter((el) => getComputedStyle(el).display !== "none");
  }

  useEffect(() => {
    if (!open) return;
    const els = itemEls();
    (initial === "last" ? els[els.length - 1] : els[0])?.focus();
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open, initial]);

  function show(at: "first" | "last") {
    setInitial(at);
    setOpen(true);
  }

  function close(returnFocus = true) {
    setOpen(false);
    if (returnFocus) btnRef.current?.focus();
  }

  function onTriggerKey(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      show("first");
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      show("last");
    }
  }

  function onMenuKey(e: React.KeyboardEvent) {
    const els = itemEls();
    const i = els.indexOf(document.activeElement as HTMLElement);
    const go = (n: number) => els[(n + els.length) % els.length]?.focus();
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        go(i + 1);
        break;
      case "ArrowUp":
        e.preventDefault();
        go(i - 1);
        break;
      case "Home":
        e.preventDefault();
        go(0);
        break;
      case "End":
        e.preventDefault();
        go(els.length - 1);
        break;
      case "Escape":
        e.preventDefault();
        e.stopPropagation();
        close();
        break;
      case "Tab":
        close(false);
        break;
    }
  }

  function choose(item: RowMenuItem) {
    if (item.disabled) return;
    // Focus goes back to the trigger first, so a dialog opened by the item
    // restores focus to the row's ⋯ when it closes (plan §7).
    close();
    item.onSelect();
  }

  return (
    <span ref={wrapRef} className="relative inline-flex">
      <button
        ref={setBtn}
        type="button"
        className={ICON_BTN}
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        disabled={disabled}
        onClick={() => (open ? close() : show("first"))}
        onKeyDown={onTriggerKey}
      >
        <Dots />
      </button>
      {open && (
        <div
          ref={menuRef}
          id={menuId}
          role="menu"
          aria-label={label}
          className={MENU}
          onKeyDown={onMenuKey}
        >
          {items.map((item) => (
            <button
              key={item.key}
              type="button"
              role="menuitem"
              tabIndex={-1}
              aria-label={item.ariaLabel}
              aria-disabled={item.disabled || undefined}
              className={cn(ITEM, FOCUS, item.danger && "text-vw-danger-fg", item.className)}
              onClick={() => choose(item)}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
    </span>
  );
});
