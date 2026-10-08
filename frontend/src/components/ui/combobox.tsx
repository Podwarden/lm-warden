"use client";

import { useEffect, useId, useRef, useState } from "react";
import { cn } from "@/lib/utils";

export interface ComboboxSuggestion {
  value: string;
  label: string;
}

interface ComboboxProps {
  suggestions: ComboboxSuggestion[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  /** Optional: extra class on the input itself. */
  inputClassName?: string;
  /** aria-label for the combobox input. */
  ariaLabel?: string;
  /** data-testid forwarded onto the underlying input element. */
  "data-testid"?: string;
}

export function Combobox({
  suggestions,
  value,
  onChange,
  placeholder,
  disabled = false,
  className,
  inputClassName,
  ariaLabel,
  "data-testid": dataTestId,
}: ComboboxProps) {
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState<number>(-1);
  // Internal text state for the input. We can't rely on the `value` prop alone
  // because parents commonly wire `onChange` to a setter that lags by a render
  // (or, in tests, to a vi.fn() that never updates the prop). Tracking input
  // text locally gives us snappy free-typing and correct filtering regardless
  // of how the parent reflects the value back.
  const [inputValue, setInputValue] = useState<string>(value);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const listboxId = useId();

  // Re-sync local input state when the controlled `value` changes from outside
  // (e.g., parent reset, suggestion commit reflected back through props).
  useEffect(() => {
    setInputValue(value);
  }, [value]);

  const filtered = inputValue
    ? suggestions.filter(
        (s) =>
          s.value.toLowerCase().includes(inputValue.toLowerCase()) ||
          s.label.toLowerCase().includes(inputValue.toLowerCase()),
      )
    : suggestions;

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (!wrapperRef.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  function commit(idx: number) {
    const sug = filtered[idx];
    if (!sug) return;
    setInputValue(sug.value);
    onChange(sug.value);
    setOpen(false);
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (disabled) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!open) setOpen(true);
      setHighlight((h) => Math.min(filtered.length - 1, h + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlight((h) => Math.max(0, h - 1));
    } else if (e.key === "Enter" && open && highlight >= 0) {
      e.preventDefault();
      commit(highlight);
    } else if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
    }
  }

  return (
    <div ref={wrapperRef} className={cn("relative", className)}>
      <input
        ref={inputRef}
        type="text"
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={open}
        aria-controls={open ? listboxId : undefined}
        aria-activedescendant={
          open && highlight >= 0 ? `${listboxId}-opt-${highlight}` : undefined
        }
        aria-label={ariaLabel}
        data-testid={dataTestId}
        value={inputValue}
        disabled={disabled}
        placeholder={placeholder}
        onChange={(e) => {
          setInputValue(e.target.value);
          onChange(e.target.value);
          setOpen(true);
          setHighlight(0);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
        className={cn(
          "flex h-9 w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-1 font-mono text-sm text-slate-100 placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 disabled:cursor-not-allowed disabled:opacity-50",
          inputClassName,
        )}
      />
      {open && filtered.length > 0 && (
        <ul
          id={listboxId}
          role="listbox"
          className="absolute left-0 right-0 top-full z-50 mt-1 max-h-60 overflow-auto rounded-md border border-slate-700 bg-slate-900 py-1 shadow-lg"
        >
          {filtered.map((sug, idx) => (
            <li
              key={sug.value}
              id={`${listboxId}-opt-${idx}`}
              role="option"
              aria-selected={value === sug.value}
              className={cn(
                "cursor-pointer px-3 py-1.5 text-sm",
                idx === highlight
                  ? "bg-slate-800 text-slate-100"
                  : "text-slate-300 hover:bg-slate-800",
              )}
              onMouseEnter={() => setHighlight(idx)}
              onMouseDown={(e) => {
                e.preventDefault();
                commit(idx);
              }}
            >
              {sug.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
