"use client";

// One of the top fixes (mockup `scorecard.html`, spec 12.1 item 3): rank, layer,
// rule, the leanprompt id, the finding in words, the evidence (turns, tool), the
// expected saving per session and per 100 sessions, the fix, and "show turns"
// (jumps to the request view on that turn).

import { LayerBadge, RuleTag } from "@/components/prompt-lens/findings/finding-chip";
import { forScope, ruleCopy } from "@/lib/prompt-lens/findings";
import { savingText, turnsText, type Scorecard, type ScorecardFinding } from "@/lib/prompt-lens/scorecard";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { WontfixMenu, type WontfixActions } from "./wontfix-menu";

/** The leanprompt rule ids a finding's fix points to (app/lens/catalogue.py `lp`). */
export const LEANPROMPT: Record<string, string> = {
  "E-01": "PC-03, PC-02",
  "E-02": "PC-04",
  "E-03": "PC-05",
  "E-04": "PC-06",
  "E-05": "PC-09",
  "E-06": "PC-01",
  "E-07": "PC-08",
  "E-09": "PC-10",
  "C-01": "PC-05",
};

export interface FixCardProps {
  rank: number;
  finding: ScorecardFinding;
  card: Scorecard;
  scope: LensScope;
  onShowTurn: (reqId: string) => void;
  actions?: WontfixActions | null;
}

export function ShowTurns({ finding, card, onShowTurn }: { finding: ScorecardFinding; card: Scorecard; onShowTurn: (reqId: string) => void }) {
  const first = finding.turns[0];
  const bar = card.turns.find((t) => t.n === first);
  if (!bar) return null;
  return (
    <button
      type="button"
      data-testid="show-turns"
      onClick={() => onShowTurn(bar.req_id)}
      className="cursor-pointer bg-transparent p-0 text-[12px] text-chat-accent hover:underline"
    >
      Show turns {turnsText(finding.turns)}
    </button>
  );
}

export function FixCard({ rank, finding: f, card, scope, onShowTurn, actions = null }: FixCardProps) {
  const { title, fix } = ruleCopy(f.rule, f, card.names, scope);
  const currency = card.cost.card?.currency ?? null;
  const lp = LEANPROMPT[f.rule];
  return (
    <div data-testid="fix-card" data-rule={f.rule} className="min-w-0 rounded-lg border border-chat-rule bg-chat-surface p-3 text-[12px] leading-[1.45]">
      <div className="flex items-center gap-1.5 text-chat-muted">
        <b className="text-[13px] text-chat-fg">{rank}</b>
        <LayerBadge layer={f.layer} />
        <RuleTag rule={f.rule} />
        {lp && <span className="font-mono text-[10px] text-chat-dim">LP-{lp.split(",")[0].trim()}</span>}
      </div>
      <h4 className="m-0 mt-1.5 text-[13px] font-semibold text-chat-fg">{forScope(title, scope)}</h4>
      <p data-testid="fix-saving" className="mt-1.5 text-chat-muted">
        <b className="text-chat-fg">≈ {savingText(f, currency)}</b> per session
        <span className="text-chat-dim"> · {savingText(f, currency, 100)} per 100 sessions</span>
      </p>
      {f.confidence !== "high" && <p className="mt-0.5 text-[11px] text-chat-dim">{f.confidence} confidence</p>}
      <p className="mt-1.5 text-chat-muted">{forScope(fix, scope)}.</p>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
        <ShowTurns finding={f} card={card} onShowTurn={onShowTurn} />
        <WontfixMenu finding={f} actions={actions} />
      </div>
    </div>
  );
}
