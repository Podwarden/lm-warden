"use client";

// One finding beside what it judged (analysis spec 16.1, mockup
// `request-findings.html`): an E-01 on the sys part with its line, a C-01 on
// the tools part, an R-04 under the tool call. The copy is the catalogue's
// (`ruleCopy`); a notice (`counts.info`) is muted and shows no fix. Text only:
// nothing here is ever parsed as HTML.

import { forScope, isNotice, ruleCopy, type LensFinding, type LensLayer } from "@/lib/prompt-lens/findings";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { cn } from "@/lib/utils";

/** The layer badge colours (mockup lens.css `.lay.E/.C/.R`). */
export const LAYER_CLASS: Record<LensLayer, string> = {
  E: "bg-vw-ttft/15 text-vw-ttft",
  C: "bg-vw-dur/15 text-vw-dur",
  R: "bg-vw-completion/15 text-vw-completion",
};

export function LayerBadge({ layer }: { layer: LensLayer }) {
  return (
    <span className={cn("flex-none rounded px-[5px] text-[10px] font-semibold leading-4", LAYER_CLASS[layer])}>
      {layer}
    </span>
  );
}

export function RuleTag({ rule, muted = false }: { rule: string; muted?: boolean }) {
  return (
    <span
      className={cn(
        "mr-1 inline-block flex-none whitespace-nowrap rounded border border-chat-rule px-[5px] font-mono text-[10px] leading-4",
        muted ? "text-chat-dim" : "text-chat-muted",
      )}
    >
      {rule}
    </span>
  );
}

export interface FindingChipProps {
  finding: LensFinding;
  names?: Record<string, string>;
  scope: LensScope;
  /** Shown before the title, e.g. "tool call 1 · search_docs". */
  lead?: string;
}

export function FindingChip({ finding: f, names = {}, scope, lead }: FindingChipProps) {
  const notice = isNotice(f);
  const { title, fix } = ruleCopy(f.rule, f, names, scope);
  return (
    <span
      data-testid="finding-chip"
      data-rule={f.rule}
      data-notice={notice ? "true" : undefined}
      className={cn(
        "mt-1.5 block min-w-0 break-words rounded-md px-2 py-1.5 text-[12px] leading-[1.45]",
        notice ? "border border-dashed border-chat-rule text-chat-dim" : "bg-chat-surface-2 text-chat-fg",
      )}
    >
      <RuleTag rule={f.rule} muted={notice} />
      {notice && <span className="mr-1 text-[11px] uppercase tracking-[.06em]">notice</span>}
      {lead && <span className="mr-1 font-mono text-[11px] text-chat-muted">{lead}:</span>}
      {forScope(title, scope)}.
      {!notice && fix && <span className="text-chat-muted"> Fix: {forScope(fix, scope)}.</span>}
      {f.confidence !== "high" && !notice && (
        <span className="ml-1 text-[11px] text-chat-dim">({f.confidence} confidence)</span>
      )}
    </span>
  );
}
