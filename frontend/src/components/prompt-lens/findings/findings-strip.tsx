"use client";

// "Findings on this request" (analysis spec 16.1, mockup
// `request-findings.html`): one pill per finding that names this request's
// turn, under the request header. A pill is the layer, the rule and a few
// words; the whole copy is its tooltip and sits beside the part it judged
// (FindingChip). Notices (`counts.info`) are muted. Wraps on a phone; nothing
// scrolls the page sideways.

import { forScope, isNotice, layerOf, ruleCopy, shortLabel, type LensFinding } from "@/lib/prompt-lens/findings";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { cn } from "@/lib/utils";

import { LayerBadge, RuleTag } from "./finding-chip";

export interface FindingsStripProps {
  findings: LensFinding[];
  names?: Record<string, string>;
  scope: LensScope;
}

export function FindingsStrip({ findings, names = {}, scope }: FindingsStripProps) {
  if (findings.length === 0) return null;
  return (
    <div
      data-testid="findings-strip"
      role="list"
      aria-label="Findings on this request"
      className="mb-2 mt-2 flex min-w-0 flex-wrap items-center gap-1.5 text-[12px]"
    >
      <span className="text-chat-dim">Findings on this request</span>
      {findings.map((f, i) => {
        const notice = isNotice(f);
        const { title } = ruleCopy(f.rule, f, names);
        return (
          <span
            key={`${f.rule}:${i}`}
            role="listitem"
            data-testid="findings-pill"
            data-rule={f.rule}
            title={forScope(title, scope)}
            className={cn(
              "inline-flex min-w-0 max-w-full items-center gap-1.5 rounded-md border border-chat-rule px-2 py-[3px]",
              notice ? "border-dashed text-chat-dim" : "bg-chat-surface text-chat-fg",
            )}
          >
            <LayerBadge layer={layerOf(f.rule)} />
            <RuleTag rule={f.rule} muted={notice} />
            <span className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">
              {forScope(shortLabel(f, names), scope)}
            </span>
          </span>
        );
      })}
    </div>
  );
}
