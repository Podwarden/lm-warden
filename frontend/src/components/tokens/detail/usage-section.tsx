"use client";

// Usage section (spec §4.3–4.4): head (toggle + presets), the strip card with
// the range text and the custom From/To row, the summary line, then the charts
// passed in as children.

import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import {
  binLabel, DEFAULT_RANGE, fillBins, PRESETS, spanWords,
  type Preset, type RangeSel, type TokenSeries,
} from "@/lib/token-series";
import { compact, day, fromInputValue, hhmm, toInputValue } from "@/lib/token-format";
import { cn } from "@/lib/utils";
import { HistoryStrip, type StripWindow } from "./history-strip";
import { btn } from "./styles";

export interface UsageSectionProps {
  sel: RangeSel;
  window: StripWindow;
  /** The key's lifetime: typed times are clamped to it. */
  bounds: StripWindow;
  /** The strip's drawn axis. Omitted → `bounds` (the whole life). A custom
   *  Apply zooms it to the applied period. */
  view?: StripWindow;
  /** The strip is zoomed in (its copy says so and how to get back). */
  zoomed?: boolean;
  showChainToggle: boolean;
  chain: boolean;
  onChainChange: (on: boolean) => void;
  onPreset: (p: Preset) => void;
  onCustom: () => void;
  /** Back to the default period, zoomed out. */
  onReset: () => void;
  /** Already at the default period, zoomed out: Reset has nothing to do. */
  resetDisabled: boolean;
  onWindowChange: (w: StripWindow) => void;
  /** The custom row's Apply. Unlike a drag it must always refetch, even for
   *  an unchanged window (the page does that). */
  onApply: (w: StripWindow) => void;
  rangeError: string | null;
  strip: { chain: TokenSeries | null; own: TokenSeries | null; rotations: number[] };
  series: TokenSeries | null;
  children?: React.ReactNode;
}

const period = (from: number, to: number) => `${day(from)} ${hhmm(from)} to ${day(to)} ${hhmm(to)}`;
/** A preset window ends at `now` (mid-minute) and starts on the minute after
 *  `now - span`; name its end by the minute it runs to, so the text reads like
 *  the mockup's `Sep 17 13:21 to Sep 18 13:21` and matches the chart's last
 *  tick. Whole-minute ends (custom, dragged) are unchanged. */
const windowText = (w: { from: number; to: number }) => period(w.from, Math.ceil(w.to / 60) * 60);

/** Mockup: `${words ?? "<from> to <to>"} · ${binLabel(w)} · ${bins.length} points`. */
export function summaryCaption(sel: RangeSel, series: TokenSeries): string {
  const words = spanWords(sel) ?? period(series.from_minute * 60, series.to_minute * 60);
  return `${words} · ${binLabel(series.bin_minutes)} · ${fillBins(series).length} points`;
}

function segClass(first: boolean, pressed: boolean): string {
  return cn(
    "cursor-pointer bg-transparent px-[13px] py-1.5 text-[13px]",
    !first && "border-l border-chat-rule",
    pressed ? "bg-chat-surface-2 font-semibold text-chat-fg" : "text-chat-muted hover:bg-chat-surface hover:text-chat-fg",
  );
}

// Mockup: `button, input { font: inherit; color: inherit; }` — the input
// inherits the <label>'s `--muted` colour, not `--fg`.
const INPUT =
  "rounded-md border border-chat-rule bg-chat-page px-2 py-[5px] text-[13px] tabular-nums text-chat-muted [color-scheme:var(--vw-color-scheme)]";

export function UsageSection(p: UsageSectionProps) {
  const headingId = useId();
  const fromRef = useRef<HTMLInputElement>(null);
  const toRef = useRef<HTMLInputElement>(null);
  const [fromStr, setFromStr] = useState(() => toInputValue(p.window.from));
  const [toStr, setToStr] = useState(() => toInputValue(p.window.to));
  const [localErr, setLocalErr] = useState<string | null>(null);
  // Set when Apply had to clamp the typed times to the key's lifetime; the
  // fields then show the clamped times and this says why. Cleared by the
  // next edit.
  const [adjusted, setAdjusted] = useState(false);
  const custom = p.sel.kind === "custom";

  useEffect(() => {
    if (!custom) setAdjusted(false);
  }, [custom]);

  // Window → fields, except the field the operator is typing in.
  useEffect(() => {
    if (document.activeElement !== fromRef.current) setFromStr(toInputValue(p.window.from));
    if (document.activeElement !== toRef.current) setToStr(toInputValue(p.window.to));
  }, [p.window.from, p.window.to]);

  function apply(e?: FormEvent) {
    e?.preventDefault();
    if (!fromStr || !toStr) {
      setLocalErr("Enter both a start and an end.");
      return;
    }
    let from = fromInputValue(fromStr);
    let to = fromInputValue(toStr);
    if (from == null || to == null) {
      setLocalErr("Enter both a start and an end.");
      return;
    }
    if (from >= to) {
      setLocalErr("Start must be before end.");
      return;
    }
    // Values outside the key's lifetime are clamped (spec §4.3).
    const typed = { from, to };
    from = Math.max(from, p.bounds.from);
    to = Math.min(to, p.bounds.to);
    if (from >= to) {
      setLocalErr("Start must be before end.");
      return;
    }
    setLocalErr(null);
    // Show what was actually applied: when the clamped window equals the
    // current one, the window->fields effect never fires, so without this
    // the fields would keep showing times that were not used.
    const clamped = from !== typed.from || to !== typed.to;
    if (clamped) {
      setFromStr(toInputValue(from));
      setToStr(toInputValue(to));
    }
    setAdjusted(clamped);
    (document.activeElement as HTMLElement | null)?.blur?.();
    p.onApply({ from, to });
  }

  const s = p.series;
  const totals: [number | null, string][] = [
    [s ? s.totals.requests : null, "requests"],
    [s ? s.totals.prompt_tokens : null, "prefill tokens"],
    [s ? s.totals.completion_tokens : null, "generation tokens"],
  ];

  return (
    <section aria-labelledby={headingId}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id={headingId} className="m-0 text-[18px] font-semibold">Usage</h2>
        <div className="flex flex-wrap items-center gap-3.5">
          {p.showChainToggle && (
            <label className="inline-flex cursor-pointer select-none items-center gap-2 text-[13px] text-chat-muted">
              <input
                type="checkbox"
                checked={p.chain}
                onChange={(e) => p.onChainChange(e.target.checked)}
                // The mockup's checkbox keeps the browser's default margin
                // (3px 3px 3px 4px), which Tailwind's preflight removes.
                className="m-[3px_3px_3px_4px] h-[15px] w-[15px] accent-chat-accent-strong"
              />
              Include earlier keys
            </label>
          )}
          <div role="group" aria-label="Time range" className="inline-flex overflow-hidden rounded-md border border-chat-rule">
            {PRESETS.map((preset, i) => (
              <button
                key={preset}
                type="button"
                aria-pressed={p.sel.kind === "preset" && p.sel.preset === preset}
                className={segClass(i === 0, p.sel.kind === "preset" && p.sel.preset === preset)}
                onClick={() => p.onPreset(preset)}
              >
                {preset}
              </button>
            ))}
            <button
              type="button"
              aria-pressed={custom}
              className={segClass(false, custom)}
              onClick={() => {
                p.onCustom();
                // Mockup focuses From once the row is shown.
                requestAnimationFrame(() => fromRef.current?.focus());
              }}
            >
              Custom
            </button>
          </div>
          <button
            type="button"
            className={btn("default", "py-1.5 font-normal")}
            disabled={p.resetDisabled}
            title={`Back to the ${spanWords(DEFAULT_RANGE)}, whole history`}
            onClick={p.onReset}
          >
            Reset
          </button>
        </div>
      </div>

      <div className="mt-3.5 rounded-lg border border-vw-rule-soft/50 bg-chat-surface px-3 pb-2 pt-2.5">
        <div className="mb-1.5 flex flex-wrap justify-between gap-3 text-[12px] text-chat-dim">
          <span>
            {p.zoomed
              ? "Zoomed to the applied period. Drag the window's edges to narrow it, or Reset to see the whole history."
              : "Whole history of this key. Drag the window or its edges, or choose Custom to type exact times."}
          </span>
          <span className="font-medium tabular-nums text-chat-fg">{windowText(p.window)}</span>
        </div>

        {custom && (
          <form className="mb-2.5 mt-0.5 flex flex-wrap items-end gap-x-3.5 gap-y-2.5" onSubmit={apply} noValidate>
            <label className="flex flex-col gap-[3px] text-[12px] text-chat-muted">
              From
              <input
                ref={fromRef}
                type="datetime-local"
                step={60}
                min={toInputValue(p.bounds.from)}
                max={toInputValue(p.bounds.to)}
                value={fromStr}
                onChange={(e) => {
                  setFromStr(e.target.value);
                  setAdjusted(false);
                }}
                className={INPUT}
              />
            </label>
            <label className="flex flex-col gap-[3px] text-[12px] text-chat-muted">
              To
              <input
                ref={toRef}
                type="datetime-local"
                step={60}
                min={toInputValue(p.bounds.from)}
                max={toInputValue(p.bounds.to)}
                value={toStr}
                onChange={(e) => {
                  setToStr(e.target.value);
                  setAdjusted(false);
                }}
                className={INPUT}
              />
            </label>
            <button type="submit" className={btn("primary")}>Apply</button>
            <span className="self-center text-[12px] text-chat-dim">Your local time</span>
            {adjusted && (
              <span role="status" className="self-center text-[12px] text-chat-dim">
                Adjusted to this key&apos;s lifetime.
              </span>
            )}
            <span role="alert" className="self-center text-[12px] text-vw-danger-fg">
              {localErr ?? p.rangeError ?? ""}
            </span>
          </form>
        )}

        <HistoryStrip
          startSec={(p.view ?? p.bounds).from}
          endSec={(p.view ?? p.bounds).to}
          chain={p.strip.chain}
          own={p.strip.own}
          showChain={p.chain}
          rotations={p.strip.rotations}
          window={p.window}
          onWindowChange={(w) => {
            // A drag or arrow key rewrites the fields, so the clamping note
            // would describe times they no longer show.
            setAdjusted(false);
            p.onWindowChange(w);
          }}
        />
      </div>

      <div className="mx-0.5 mb-3 mt-4 flex flex-wrap items-baseline gap-7">
        {totals.map(([v, label]) => (
          <div key={label}>
            <div className="text-[22px] font-semibold tabular-nums">{v == null ? "–" : compact(v)}</div>
            <div className="text-[12px] text-chat-dim">{label}</div>
          </div>
        ))}
        <div className="ml-auto text-[12px] text-chat-muted">{s ? summaryCaption(p.sel, s) : ""}</div>
      </div>

      {p.children}
    </section>
  );
}
