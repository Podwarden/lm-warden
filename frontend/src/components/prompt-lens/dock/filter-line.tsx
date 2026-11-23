"use client";

// The Prompt Lens filter line (spec 2026-11-22 section 2.1, mockup states
// 1-2): one row, 38 px, scrolling sideways, never wrapping.
//
//   Key (admin only, applied by the server: the stream's ?token_ids=)
//   Session · Model · Client IP (admin only) · All / Live / Finished
//   Finish · Has · search · "N of M requests · …" on the right
//
// The search covers captured text only (Q10), so its tooltip says so. The
// key holder's line has no Key and no Client IP filter (G-R11, D5) and its
// count never shows the ring's minutes (Q8: they reveal the server's volume).
// Matching and persistence are lib/prompt-lens/filters.ts.
//
// Phone (spec section 6, mockup state 7): the line keeps one "Filters (n)"
// button, the search and the count; the button opens a full-screen sheet
// with every other filter as 44 px choices (Key and Client IP for the admin
// only). n counts the sheet's filters (the search is in the line). The back
// gesture closes the sheet (lib/prompt-lens/back-layer).

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

import { useBackLayer } from "@/lib/prompt-lens/back-layer";
import { FINISH_VALUES, HAS_FLAGS, activeCount, type DockFilters, type HasFlag } from "@/lib/prompt-lens/filters";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { LiveNumber } from "@/components/ui/live-number";
import { cn } from "@/lib/utils";

export const FILTER_LINE_PX = 38;
/** The phone line: 44 px controls and their margin. */
export const FILTER_LINE_PHONE_PX = 52;
export const SEARCH_TOOLTIP = "searches captured text; long prompts are trimmed in the middle";

const HAS_WORD: Record<HasFlag, string> = {
  tool_calls: "tool calls",
  thinking: "thinking",
  images: "images",
  tool_results: "tool results",
};

export interface FilterOption {
  value: string;
  label: string;
}

export interface FilterOptions {
  keys: FilterOption[];
  sessions: FilterOption[];
  models: FilterOption[];
  ips: FilterOption[];
}

const SEL =
  "inline-flex h-[26px] flex-none cursor-pointer items-center gap-1 rounded-[5px] border border-chat-rule bg-transparent px-2 text-[12px] text-chat-muted hover:bg-chat-surface hover:text-chat-fg";
const SET = "border-chat-accent/60 text-chat-fg";

/** A multi-select: null is "all"; picking from "all" narrows to that one;
 *  emptying the list goes back to "all".
 *
 *  The menu is position: fixed (the line scrolls sideways and would clip
 *  it), placed under its button and moved with it on any scroll or resize.
 *  Opening focuses the first item; arrows, Home and End move; Escape closes
 *  and returns focus to the button; Tab or focus leaving closes it. */
function Multi({
  name,
  options,
  value,
  onChange,
  allText = "all",
}: {
  name: string;
  options: FilterOption[];
  value: string[] | null;
  onChange: (v: string[] | null) => void;
  allText?: string;
}) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const btn = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const set = value !== null && value.length > 0;
  const labelOf = (v: string) => options.find((o) => o.value === v)?.label ?? v;
  const text = !set ? allText : value.length === 1 ? labelOf(value[0]) : `${value.length} picked`;

  const place = useCallback(() => {
    if (!btn.current) return;
    const r = btn.current.getBoundingClientRect();
    setPos((p) => (p.top === r.bottom + 4 && p.left === r.left ? p : { top: r.bottom + 4, left: r.left }));
  }, []);

  useLayoutEffect(() => {
    if (!open) return;
    place();
    menu.current?.querySelector<HTMLElement>('[role="menuitemcheckbox"]')?.focus();
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent | MouseEvent) => {
      const t = e.target as Node;
      if (menu.current?.contains(t) || btn.current?.contains(t)) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    // Capture: a scroll of any ancestor (the filter line, the page) moves the button.
    document.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open, place]);

  const close = (refocus: boolean) => {
    setOpen(false);
    if (refocus) btn.current?.focus();
  };

  const onMenuKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const items = [...(menu.current?.querySelectorAll<HTMLElement>('[role="menuitemcheckbox"]') ?? [])];
    const at = items.indexOf(document.activeElement as HTMLElement);
    const go = (k: number) => {
      e.preventDefault();
      items[(k + items.length) % items.length]?.focus();
    };
    switch (e.key) {
      case "ArrowDown":
        return go(at + 1);
      case "ArrowUp":
        return go(at < 0 ? items.length - 1 : at - 1);
      case "Home":
        return go(0);
      case "End":
        return go(items.length - 1);
      case "Escape":
        e.preventDefault();
        return close(true);
      case "Tab":
        return close(false);
    }
  };

  const toggle = (v: string) => {
    if (!set) return onChange([v]);
    const next = value.includes(v) ? value.filter((x) => x !== v) : [...value, v];
    onChange(next.length ? next : null);
  };

  const ITEM = "flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-chat-surface-2 focus:bg-chat-surface-2 focus:outline-none";

  return (
    <>
      <button
        ref={btn}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        className={cn(SEL, set && SET)}
        onClick={() => setOpen((o) => !o)}
      >
        {name} <b className="max-w-[160px] overflow-hidden text-ellipsis font-semibold">{text}</b>
      </button>
      {open && (
        <div
          ref={menu}
          role="menu"
          aria-label={name}
          onKeyDown={onMenuKey}
          onBlur={(e) => {
            const to = e.relatedTarget as Node | null;
            if (to && (menu.current?.contains(to) || btn.current?.contains(to))) return;
            if (to) setOpen(false);
          }}
          className="fixed z-[60] max-h-[260px] min-w-[180px] overflow-y-auto rounded-md border border-chat-rule bg-chat-surface p-1 text-[12px] shadow-lg"
          style={{ top: pos.top, left: pos.left }}
        >
          <button type="button" role="menuitemcheckbox" aria-checked={!set} className={ITEM} onClick={() => onChange(null)}>
            <span aria-hidden className="w-3">{!set ? "✓" : ""}</span>All
          </button>
          {options.length === 0 && <div className="px-2 py-1 text-chat-dim">Nothing in the list yet</div>}
          {options.map((o) => {
            const on = set && value.includes(o.value);
            return (
              <button key={o.value} type="button" role="menuitemcheckbox" aria-checked={on} className={ITEM} onClick={() => toggle(o.value)}>
                <span aria-hidden className="w-3">{on ? "✓" : ""}</span>
                {o.label}
              </button>
            );
          })}
        </div>
      )}
    </>
  );
}

/** Picking from "all" narrows to that one; emptying the list is "all" again. */
function toggled(value: string[] | null, v: string): string[] | null {
  if (value === null || value.length === 0) return [v];
  const next = value.includes(v) ? value.filter((x) => x !== v) : [...value, v];
  return next.length ? next : null;
}

const CHOICE =
  "inline-flex min-h-11 min-w-11 items-center justify-center gap-2 rounded-md border border-chat-rule bg-transparent px-3 text-left text-[13px] text-chat-muted aria-checked:border-chat-accent/70 aria-checked:text-chat-fg aria-pressed:border-chat-accent/70 aria-pressed:bg-chat-surface-2 aria-pressed:text-chat-fg";

/** One filter of the phone sheet: "All" and a 44 px choice per option. */
function Choices({
  name,
  options,
  value,
  onChange,
  allText = "All",
}: {
  name: string;
  options: FilterOption[];
  value: string[] | null;
  onChange: (v: string[] | null) => void;
  allText?: string;
}) {
  const set = value !== null && value.length > 0;
  return (
    <fieldset role="group" aria-label={name} className="m-0 min-w-0 border-0 p-0">
      <legend className="mb-1.5 p-0 text-[11px] font-semibold uppercase tracking-[.07em] text-chat-dim">{name}</legend>
      <div className="flex flex-wrap gap-2">
        <button type="button" role="checkbox" aria-checked={!set} className={CHOICE} onClick={() => onChange(null)}>
          {allText}
        </button>
        {options.length === 0 && <span className="self-center text-[12px] text-chat-dim">Nothing in the list yet</span>}
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            role="checkbox"
            aria-checked={set && value.includes(o.value)}
            className={cn(CHOICE, "max-w-full overflow-hidden text-ellipsis whitespace-nowrap")}
            onClick={() => onChange(toggled(value, o.value))}
          >
            {o.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

function FilterSheet({
  scope,
  filters: f,
  onChange,
  options,
  onClose,
}: {
  scope: LensScope;
  filters: DockFilters;
  onChange: (f: DockFilters) => void;
  options: FilterOptions;
  onClose: () => void;
}) {
  const admin = scope === "admin";
  const patch = (p: Partial<DockFilters>) => onChange({ ...f, ...p });
  const done = useRef<HTMLButtonElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    done.current?.focus();
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      closeRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const n = activeCount({ ...f, q: "" });
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Filters"
      data-testid="prompt-lens-filter-sheet"
      className="fixed inset-0 z-[70] flex flex-col bg-vw-dock pb-[env(safe-area-inset-bottom)] font-sans"
    >
      <div className="flex min-h-[52px] flex-none items-center gap-2 border-b border-chat-rule px-3">
        <h2 className="m-0 text-[14px] font-semibold text-chat-fg">Filters</h2>
        <span className="text-[12px] text-chat-dim">{n ? `${n} on` : "none on"}</span>
        <button
          type="button"
          disabled={n === 0}
          onClick={() =>
            onChange({ ...f, keys: null, sessions: null, models: null, ips: null, phase: "all", finish: null, has: [] })
          }
          className="ml-auto inline-flex min-h-11 items-center rounded-md border border-chat-rule bg-transparent px-3 text-[13px] text-chat-muted disabled:opacity-40"
        >
          Clear all
        </button>
        <button
          ref={done}
          type="button"
          onClick={onClose}
          className="inline-flex min-h-11 items-center rounded-md border border-chat-accent bg-chat-accent px-4 text-[13px] font-semibold text-chat-on-accent"
        >
          Done
        </button>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto overscroll-contain px-3 py-3">
        {admin && (
          <Choices name="Key" options={options.keys} value={f.keys} onChange={(keys) => patch({ keys })} />
        )}
        <Choices name="Session" options={options.sessions} value={f.sessions} onChange={(sessions) => patch({ sessions })} />
        <Choices name="Model" options={options.models} value={f.models} onChange={(models) => patch({ models })} />
        {admin && <Choices name="Client IP" options={options.ips} value={f.ips} onChange={(ips) => patch({ ips })} />}
        <fieldset role="group" aria-label="Phase" className="m-0 min-w-0 border-0 p-0">
          <legend className="mb-1.5 p-0 text-[11px] font-semibold uppercase tracking-[.07em] text-chat-dim">Phase</legend>
          <div className="flex flex-wrap gap-2">
            {(["all", "live", "finished"] as const).map((p) => (
              <button key={p} type="button" aria-pressed={f.phase === p} className={CHOICE} onClick={() => patch({ phase: p })}>
                {p === "all" ? "All" : p === "live" ? "Live" : "Finished"}
              </button>
            ))}
          </div>
        </fieldset>
        <Choices
          name="Finish"
          allText="Any"
          options={FINISH_VALUES.map((v) => ({ value: v, label: v }))}
          value={f.finish}
          onChange={(finish) => patch({ finish })}
        />
        <Choices
          name="Has"
          allText="Any"
          options={HAS_FLAGS.map((v) => ({ value: v, label: HAS_WORD[v] }))}
          value={f.has.length ? f.has : null}
          onChange={(has) => patch({ has: (has ?? []) as HasFlag[] })}
        />
      </div>
    </div>
  );
}

export interface FilterLineProps {
  scope: LensScope;
  /** Phone width: one "Filters (n)" button and a sheet (spec section 6). */
  phone?: boolean;
  filters: DockFilters;
  onChange: (f: DockFilters) => void;
  options: FilterOptions;
  /** Requests after the filters, and in the dock. */
  shown: number;
  total: number;
  /** Admin only: minutes the server's ring holds (from the `ring` frame). */
  ringMinutes?: number | null;
}

export function FilterLine(props: FilterLineProps) {
  return props.phone ? <PhoneFilterLine {...props} /> : <DesktopFilterLine {...props} />;
}

/** "12 of 40 requests · ...": the counts sit in reserved slots
 *  (ui.live-numbers), so the words after them never move. */
/** Up to "300" requests (the client bound). */
const COUNT_CH = 3;

function countText(scope: LensScope, shown: number, total: number, ringMinutes: number | null): ReactNode {
  const admin = scope === "admin";
  const tail =
    admin && ringMinutes !== null ? (
      <>
        {" "}
        · ring keeps ≈ <LiveNumber value={ringMinutes} reserve={2} /> min
      </>
    ) : admin ? null : (
      " · older requests leave server memory"
    );
  return (
    <>
      <LiveNumber value={shown} reserve={COUNT_CH} /> of <LiveNumber value={total} reserve={COUNT_CH} /> requests{tail}
    </>
  );
}

function PhoneFilterLine({ scope, filters: f, onChange, options, shown, total, ringMinutes = null }: FilterLineProps) {
  const [sheet, setSheet] = useState(false);
  const btn = useRef<HTMLButtonElement>(null);
  const close = useCallback(() => {
    setSheet(false);
    btn.current?.focus();
  }, []);
  useBackLayer(sheet, close);
  const n = activeCount({ ...f, q: "" });
  return (
    <>
      <div
        data-testid="prompt-lens-filter-line"
        style={{ height: FILTER_LINE_PHONE_PX }}
        className="flex flex-none items-center gap-2 overflow-x-auto overflow-y-hidden whitespace-nowrap border-b border-chat-rule px-3"
      >
        <button
          ref={btn}
          type="button"
          aria-haspopup="dialog"
          aria-expanded={sheet}
          onClick={() => setSheet(true)}
          className={cn(
            "inline-flex min-h-11 flex-none items-center gap-1 rounded-md border border-chat-rule bg-transparent px-3 text-[13px] text-chat-muted",
            n > 0 && SET,
          )}
        >
          Filters ({n})
        </button>
        <input
          type="search"
          aria-label="Search captured text"
          placeholder="Search captured text"
          title={SEARCH_TOOLTIP}
          value={f.q}
          onChange={(e) => onChange({ ...f, q: e.target.value })}
          className="h-11 w-[150px] min-w-0 flex-none rounded-md border border-chat-rule bg-transparent px-2 text-[16px] text-chat-fg placeholder:text-[13px] placeholder:text-chat-dim"
        />
        <span className="flex-none pl-1 text-[12px] tabular-nums text-chat-dim">
          {countText(scope, shown, total, ringMinutes)}
        </span>
      </div>
      {sheet && <FilterSheet scope={scope} filters={f} onChange={onChange} options={options} onClose={close} />}
    </>
  );
}

function DesktopFilterLine({ scope, filters: f, onChange, options, shown, total, ringMinutes = null }: FilterLineProps) {
  const admin = scope === "admin";
  const patch = (p: Partial<DockFilters>) => onChange({ ...f, ...p });
  const keysAll = options.keys.length ? `all ${options.keys.length}` : "all";
  return (
    <div
      data-testid="prompt-lens-filter-line"
      style={{ height: FILTER_LINE_PX }}
      className="flex h-[38px] flex-none items-center gap-1.5 overflow-x-auto overflow-y-hidden whitespace-nowrap border-b border-chat-rule px-3"
    >
      {admin && (
        <Multi name="Key" options={options.keys} value={f.keys} allText={keysAll} onChange={(keys) => patch({ keys })} />
      )}
      <Multi name="Session" options={options.sessions} value={f.sessions} onChange={(sessions) => patch({ sessions })} />
      <Multi name="Model" options={options.models} value={f.models} onChange={(models) => patch({ models })} />
      {admin && <Multi name="Client IP" options={options.ips} value={f.ips} onChange={(ips) => patch({ ips })} />}
      <span role="group" aria-label="Phase" className="inline-flex flex-none overflow-hidden rounded-[5px] border border-chat-rule">
        {(["all", "live", "finished"] as const).map((p) => (
          <button
            key={p}
            type="button"
            aria-pressed={f.phase === p}
            onClick={() => patch({ phase: p })}
            className={cn(
              "h-[24px] px-2 text-[12px] text-chat-muted hover:text-chat-fg",
              f.phase === p && "bg-chat-surface-2 text-chat-fg",
            )}
          >
            {p === "all" ? "All" : p === "live" ? "Live" : "Finished"}
          </button>
        ))}
      </span>
      <Multi
        name="Finish"
        allText="any"
        options={FINISH_VALUES.map((v) => ({ value: v, label: v }))}
        value={f.finish}
        onChange={(finish) => patch({ finish })}
      />
      <Multi
        name="Has"
        allText="any"
        options={HAS_FLAGS.map((v) => ({ value: v, label: HAS_WORD[v] }))}
        value={f.has.length ? f.has : null}
        onChange={(has) => patch({ has: (has ?? []) as HasFlag[] })}
      />
      <input
        type="search"
        aria-label="Search captured text"
        placeholder="Search captured text"
        title={SEARCH_TOOLTIP}
        value={f.q}
        onChange={(e) => patch({ q: e.target.value })}
        className="h-[26px] w-[200px] flex-none rounded-[5px] border border-chat-rule bg-transparent px-2 text-[12px] text-chat-fg placeholder:text-chat-dim"
      />
      <span className="ml-auto flex-none pl-2 text-[12px] tabular-nums text-chat-dim">
        {countText(scope, shown, total, ringMinutes)}
      </span>
    </div>
  );
}
