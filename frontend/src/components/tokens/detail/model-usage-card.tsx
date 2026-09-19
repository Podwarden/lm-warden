"use client";

// "Usage by model" (migration 0036): prefill and generation per model for the
// selected period and token set, busiest first. A model served as more than
// one variant (the try-stack or the settings form changed its engine,
// quantization, context...) expands into one row per variant; a single
// variant's settings show under the model's name.

import { useId, useState } from "react";
import {
  startsBeforeByModel, variantLabels,
  type ModelUsage, type ModelVariantUsage, type TokenSeries,
} from "@/lib/token-series";
import { compact, fmtDateTime } from "@/lib/token-format";
import { cn } from "@/lib/utils";
import { CARD, modelStyle, type ModelMark } from "./styles";

/** "37%", "<1%" for a sliver, "0%" for nothing. */
export function sharePct(share: number): string {
  if (!(share > 0)) return "0%";
  if (share < 0.01) return "<1%";
  return `${Math.round(share * 100)}%`;
}

const full = (n: number) => n.toLocaleString("en-US");

const TH = "pb-1.5 text-[12px] font-normal text-chat-dim";
const NUM = "py-1.5 pl-3 text-right tabular-nums";

function Num({ v, sub }: { v: number; sub?: boolean }) {
  return (
    <td className={cn(NUM, sub && "py-1 text-[12px] text-chat-muted")} title={full(v)}>
      {compact(v)}
    </td>
  );
}

function Share({ share, color, sub }: { share: number; color: string; sub?: boolean }) {
  return (
    <td className={cn("py-1.5 pl-4", sub && "py-1")}>
      <div className="flex items-center gap-2">
        <div aria-hidden className="h-1.5 w-[80px] shrink-0 overflow-hidden rounded-full bg-chat-surface-2">
          <div
            className="h-full rounded-full"
            style={{
              width: `${Math.min(100, share * 100)}%`,
              minWidth: share > 0 ? 2 : 0,
              background: color,
              opacity: sub ? 0.6 : 1,
            }}
          />
        </div>
        <span className={cn("w-[34px] text-right tabular-nums", sub ? "text-[12px] text-chat-muted" : "text-chat-muted")}>
          {sharePct(share)}
        </span>
      </div>
    </td>
  );
}

/** The model's colour, marked like its chart line once the six colours run
 *  out: filled square (solid line), hollow square (dashed), dot (dotted). */
function Swatch({ color, mark }: { color: string; mark: ModelMark }) {
  return (
    <span
      aria-hidden
      data-mark={mark}
      className={cn(
        "inline-block h-2 w-2 shrink-0",
        mark === "dotted" ? "rounded-full" : "rounded-[2px]",
        mark === "dashed" && "border-[1.5px] border-solid",
      )}
      style={mark === "dashed" ? { borderColor: color } : { background: color }}
    />
  );
}

function variantTitle(v: ModelVariantUsage, nowSec: number): string {
  const seen = v.first_seen != null ? `first served ${fmtDateTime(v.first_seen, nowSec)} · ` : "";
  return `${seen}variant ${v.variant_id}`;
}

function ModelRows({ m, index, nowSec }: { m: ModelUsage; index: number; nowSec: number }) {
  const [open, setOpen] = useState(false);
  const { color, mark } = modelStyle(index);
  const labels = variantLabels(m.variants);
  const many = m.variants.length > 1;
  return (
    <>
      <tr className="border-t border-vw-rule-soft/50 align-top">
        <td className="max-w-0 py-1.5 pr-2">
          <div className="flex items-center gap-2">
            <Swatch color={color} mark={mark} />
            <span className="truncate font-medium" title={m.model}>{m.model}</span>
          </div>
          <div className="pl-4 text-[12px] text-chat-dim">
            {many ? (
              <button
                type="button"
                aria-expanded={open}
                aria-label={`${open ? "Hide" : "Show"} the ${m.variants.length} variants of ${m.model}`}
                onClick={() => setOpen((o) => !o)}
                className="cursor-pointer border-0 bg-transparent p-0 text-[12px] text-chat-muted hover:text-chat-fg"
              >
                <span aria-hidden className="mr-1 inline-block w-2">{open ? "▾" : "▸"}</span>
                {m.variants.length} variants
              </button>
            ) : (
              <span className="block truncate" title={variantTitle(m.variants[0], nowSec)}>{labels[0]}</span>
            )}
          </div>
        </td>
        <Num v={m.requests} />
        <Num v={m.prompt_tokens} />
        <Num v={m.completion_tokens} />
        <Num v={m.total_tokens} />
        <Share share={m.share} color={color} />
      </tr>
      {many && open && m.variants.map((v, i) => (
        <tr key={v.variant_id} data-testid="variant-row">
          <td className="max-w-0 py-1 pl-4 pr-2 text-[12px] text-chat-muted">
            <span className="block truncate" title={variantTitle(v, nowSec)}>{labels[i]}</span>
          </td>
          <Num v={v.requests} sub />
          <Num v={v.prompt_tokens} sub />
          <Num v={v.completion_tokens} sub />
          <Num v={v.total_tokens} sub />
          <Share share={v.share} color={color} sub />
        </tr>
      ))}
    </>
  );
}

export function ModelUsageCard({ series, nowSec }: { series: TokenSeries; nowSec: number }) {
  const headingId = useId();
  const models = series.by_model;
  const since = series.by_model_since;
  const note = startsBeforeByModel(series) && since != null
    ? `Per-model breakdown recorded since ${fmtDateTime(since, nowSec)}.`
    : null;
  return (
    <section className={cn(CARD, "mb-3.5 pb-3 pt-3")} aria-labelledby={headingId}>
      <h3 id={headingId} className="m-0 mb-2 text-[14px] font-semibold">Usage by model</h3>
      {models.length === 0 ? (
        <p className="m-0 text-[13px] text-chat-muted">No per-model data in this period yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] border-collapse text-[13px]">
            <thead>
              <tr>
                <th scope="col" className={cn(TH, "w-[44%] text-left")}>Model</th>
                <th scope="col" className={cn(TH, "pl-3 text-right")}>Requests</th>
                <th scope="col" className={cn(TH, "pl-3 text-right")}>Prefill</th>
                <th scope="col" className={cn(TH, "pl-3 text-right")}>Generation</th>
                <th scope="col" className={cn(TH, "pl-3 text-right")}>Total</th>
                <th scope="col" className={cn(TH, "w-[140px] pl-4 text-left")}>Share</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m, i) => <ModelRows key={m.model_id} m={m} index={i} nowSec={nowSec} />)}
            </tbody>
          </table>
        </div>
      )}
      {note && <div className="mt-2 text-[12px] text-chat-dim">{note}</div>}
    </section>
  );
}
