"use client";

// "All findings" (spec 12.1 item 4): every fired rule grouped by layer,
// collapsible, with "not scored: <reason>" where a layer lacked data. A notice
// (info, muted, R-09) is listed, never a fix. Text only.

import { useState } from "react";

import { LayerBadge, RuleTag } from "@/components/prompt-lens/findings/finding-chip";
import { forScope, isNotice, ruleCopy, type LensLayer } from "@/lib/prompt-lens/findings";
import { LAYER_NAME, savingText, turnsText, type Scorecard } from "@/lib/prompt-lens/scorecard";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { cn } from "@/lib/utils";
import { ShowTurns } from "./fix-card";
import { WontfixMenu, type WontfixActions } from "./wontfix-menu";

const LAYERS: LensLayer[] = ["E", "C", "R"];

export function FindingsList({ card, scope, onShowTurn, actions = null }: { card: Scorecard; scope: LensScope; onShowTurn: (reqId: string) => void; actions?: WontfixActions | null }) {
  const [closed, setClosed] = useState<Record<string, boolean>>({});
  const currency = card.cost.card?.currency ?? null;
  return (
    <div data-testid="findings-list" className="rounded-lg border border-chat-rule bg-chat-surface">
      {LAYERS.map((layer) => {
        const fs = card.findings.filter((f) => f.layer === layer);
        const reason = card.not_scored[layer];
        const shut = closed[layer] === true;
        return (
          <div key={layer} data-testid={`findings-${layer}`} className="border-b border-chat-rule/60 last:border-b-0">
            <button
              type="button"
              aria-expanded={!shut}
              onClick={() => setClosed((c) => ({ ...c, [layer]: !shut }))}
              className="flex w-full cursor-pointer items-center gap-2 bg-transparent px-3 py-2 text-left text-[12px] font-semibold text-chat-fg"
            >
              <span aria-hidden className="text-chat-muted">{shut ? "▸" : "▾"}</span>
              <LayerBadge layer={layer} />
              {LAYER_NAME[layer]}
              <span className="ml-auto text-[11px] font-normal text-chat-dim">
                {reason ? `not scored: ${reason}` : fs.length === 0 ? "nothing to fix" : `${fs.length} fired`}
              </span>
            </button>
            {!shut &&
              fs.map((f, i) => {
                const { title } = ruleCopy(f.rule, f, card.names, scope);
                const notice = isNotice(f);
                return (
                  <div key={`${f.rule}-${f.tool ?? ""}-${i}`} data-testid="finding-row" data-rule={f.rule} data-muted={f.muted ? "true" : "false"} className={cn("border-t border-chat-rule/40 px-3 py-2 text-[12px] leading-[1.45]", f.muted && "opacity-60")}>
                    <div className="flex items-start gap-1.5">
                      <RuleTag rule={f.rule} muted={notice} />
                      <span className={notice ? "min-w-0 flex-1 text-chat-dim" : "min-w-0 flex-1 text-chat-fg"}>
                        {forScope(title, scope)}
                      </span>
                      <span className="flex-none whitespace-nowrap text-[11px] text-chat-muted">
                        {notice ? "not counted" : savingText(f, currency)}
                      </span>
                    </div>
                    <div className="mt-0.5 flex flex-wrap items-center gap-x-3 pl-1 text-[11px] text-chat-dim">
                      <span>turns {turnsText(f.turns)}</span>
                      <ShowTurns finding={f} card={card} onShowTurn={onShowTurn} />
                      <WontfixMenu finding={f} actions={actions} />
                    </div>
                  </div>
                );
              })}
          </div>
        );
      })}
    </div>
  );
}
