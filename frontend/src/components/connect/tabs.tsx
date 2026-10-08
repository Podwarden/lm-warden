"use client";

// Tab strip with the keyboard handling of connect-claude-code.tsx (plan §6,
// §7): roving tabindex, arrows wrap, Home/End, selection follows focus. The
// caller renders the tabpanel with `id={panelId(idBase)}` and
// `aria-labelledby={tabId(idBase, selected)}`.

import { useRef } from "react";
import { cn } from "@/lib/utils";
import { CHIP, CHIP_OFF, CHIP_ON, TAB, TAB_OFF, TAB_ON } from "./styles";

export const tabId = (base: string, id: string) => `${base}-tab-${id}`;
export const panelId = (base: string) => `${base}-panel`;

/** Arrow/Home/End → the next index, or null when the key is not ours. Both
 *  axes are accepted so a list that changes orientation by width keeps working. */
export function nextTabIndex(key: string, i: number, n: number): number | null {
  if (key === "Home") return 0;
  if (key === "End") return n - 1;
  const d = key === "ArrowRight" || key === "ArrowDown" ? 1 : key === "ArrowLeft" || key === "ArrowUp" ? -1 : 0;
  return d ? (i + d + n) % n : null;
}

interface Item {
  id: string;
  label: string;
}

interface Props {
  items: Item[];
  selected: string;
  onSelect: (id: string) => void;
  label: string;
  idBase: string;
  variant?: "underline" | "chip";
  className?: string;
}

export function Tabs({ items, selected, onSelect, label, idBase, variant = "underline", className }: Props) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const chip = variant === "chip";

  function onKey(e: React.KeyboardEvent, i: number) {
    const next = nextTabIndex(e.key, i, items.length);
    if (next === null) return;
    e.preventDefault();
    onSelect(items[next].id);
    refs.current[next]?.focus();
  }

  return (
    <div
      role="tablist"
      aria-label={label}
      className={cn(
        "flex overflow-x-auto",
        chip ? "gap-1.5 p-px" : "gap-1 border-b border-vw-rule-soft/60",
        className,
      )}
    >
      {items.map((t, i) => {
        const on = t.id === selected;
        return (
          <button
            key={t.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            id={tabId(idBase, t.id)}
            aria-selected={on}
            aria-controls={panelId(idBase)}
            tabIndex={on ? 0 : -1}
            onClick={() => onSelect(t.id)}
            onKeyDown={(e) => onKey(e, i)}
            className={chip ? cn(CHIP, on ? CHIP_ON : CHIP_OFF) : cn(TAB, on ? TAB_ON : TAB_OFF)}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}
