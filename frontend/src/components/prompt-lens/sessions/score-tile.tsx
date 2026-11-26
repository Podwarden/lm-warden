"use client";

// One score tile of the scorecard (mockup `scorecard.html`): the number with
// its range, the confidence, the turns, the one-line meaning of 100 and an
// "inputs" toggle. A layer that was not scored shows "not scored: <reason>"
// and no number (A-7). The model review tile is a different thing: a model's
// opinion, dashed, never inside the overall (A-8).

import { useState } from "react";

import { LayerBadge } from "@/components/prompt-lens/findings/finding-chip";
import {
  inputsText,
  LAYER_NAME,
  meaning,
  rangeText,
  type Confidence,
  type ScoreOut,
} from "@/lib/prompt-lens/scorecard";
import type { LensScope } from "@/lib/prompt-lens/stream";
import type { LensLayer } from "@/lib/prompt-lens/findings";
import { cn } from "@/lib/utils";

const BAR: Record<LensLayer | "overall", string> = {
  E: "bg-vw-ttft",
  C: "bg-vw-dur",
  R: "bg-vw-completion",
  overall: "bg-chat-accent",
};

const CONF: Record<Confidence, string> = {
  high: "text-vw-ok-fg",
  medium: "text-vw-amber-fg",
  low: "text-chat-dim",
};

export function ConfidenceTag({ confidence }: { confidence: Confidence | null | undefined }) {
  if (!confidence) return null;
  return (
    <span data-testid="confidence" className={cn("ml-auto text-[11px] font-normal", CONF[confidence])}>
      {confidence === "high" ? "high confidence" : confidence}
    </span>
  );
}

export interface ScoreTileProps {
  layer: LensLayer | "overall";
  score: ScoreOut | undefined;
  scope: LensScope;
  /** Findings marked won't fix that this score leaves out (stage 3), with the
   *  score as it would be with them counted. */
  wontfix?: { count: number; withThem: number } | null;
  weights?: Record<string, number>;
}

export function ScoreTile({ layer, score, scope, weights, wontfix = null }: ScoreTileProps) {
  const [open, setOpen] = useState(false);
  const scored = !!score && score.scored && typeof score.value === "number";
  const title = layer === "overall" ? "Overall" : LAYER_NAME[layer];
  const range = score ? rangeText(score) : null;
  const meant =
    layer === "overall"
      ? `${weights?.E ?? 0.45} × Efficiency + ${weights?.C ?? 0.15} × Construction + ${weights?.R ?? 0.4} × Response. Model review not included.`
      : meaning(layer, scope);
  const inputs = score && scored ? inputsText(score, scope) : "";
  return (
    <div
      data-testid={`score-tile-${layer}`}
      data-scored={scored ? "true" : "false"}
      className={cn(
        "min-w-0 rounded-lg border border-chat-rule bg-chat-surface p-3",
        layer === "overall" && "col-span-2 sm:col-span-1",
      )}
    >
      <h3 className="m-0 flex items-center gap-1.5 text-[12px] font-semibold text-chat-muted">
        {layer !== "overall" && <LayerBadge layer={layer} />}
        {title}
        {scored && <ConfidenceTag confidence={score?.confidence} />}
      </h3>
      {scored ? (
        <>
          <div className="mt-1 flex items-baseline gap-2 tabular-nums">
            <span data-testid="score-value" className="text-[28px] font-semibold leading-none text-chat-fg">
              {score?.value}
            </span>
            {range && <small className="text-[11px] text-chat-dim">range {range}</small>}
          </div>
          <div className="relative mt-2 h-1.5 overflow-hidden rounded bg-chat-rule/60" aria-hidden>
            <span className={cn("absolute inset-y-0 left-0 rounded", BAR[layer])} style={{ width: `${score?.value ?? 0}%` }} />
          </div>
        </>
      ) : (
        <p data-testid="not-scored" className="mt-1 text-[13px] leading-snug text-chat-muted">
          not scored: {score?.reason ?? "not enough data"}
        </p>
      )}
      {scored && wontfix && wontfix.count > 0 && (
        <p data-testid="wontfix-note" className="mt-2 text-[11px] leading-snug text-chat-muted">
          {wontfix.count} {wontfix.count === 1 ? "finding marked" : "findings marked"} won&apos;t fix {wontfix.count === 1 ? "is" : "are"} not counted · with them: {wontfix.withThem}
        </p>
      )}
      <p className="mt-2 text-[11px] leading-snug text-chat-dim">{meant}</p>
      {scored && layer !== "overall" && (
        <>
          <button
            type="button"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
            className="mt-1 cursor-pointer bg-transparent p-0 text-[11px] text-chat-accent hover:underline"
          >
            {open ? "hide inputs" : "inputs"}
          </button>
          {open && (
            <p data-testid="score-inputs" className="mt-1 text-[11px] leading-snug text-chat-muted">
              {inputs}
              {score?.excluded_rules && score.excluded_rules.length > 0 && (
                <span> · left out: {score.excluded_rules.join(", ")}</span>
              )}
            </p>
          )}
        </>
      )}
      {scored && layer === "overall" && score?.inputs && Array.isArray(score.inputs.from) && (
        <p data-testid="score-from" className="mt-1 text-[11px] text-chat-muted">
          from {(score.inputs.from as string[]).join(", ")}
        </p>
      )}
    </div>
  );
}

/** "a model's opinion": its own, differently styled tile; "off" until the
 *  review ships and is switched on for the key. Not in the overall (A-8). */
export function ReviewTile() {
  return (
    <div
      data-testid="score-tile-review"
      className="col-span-2 min-w-0 rounded-lg border border-dashed border-chat-rule bg-transparent p-3 sm:col-span-1"
    >
      <h3 className="m-0 flex items-center gap-1.5 text-[12px] font-semibold text-chat-muted">
        Model review
        <span className="ml-auto text-[11px] font-normal text-chat-dim">a model&apos;s opinion</span>
      </h3>
      <p className="mt-1 text-[13px] text-chat-muted">off</p>
      <p className="mt-2 text-[11px] leading-snug text-chat-dim">An opinion, kept apart: never part of the overall or the trends.</p>
    </div>
  );
}
