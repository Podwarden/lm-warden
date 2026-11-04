// Dead weight in a session's prompts: the share, the anatomy bar and one row per kind. Unknown is "—", never 0.

import { S } from "@/lib/sessions/strings";
import { ANATOMY_CLASS, ANATOMY_ORDER } from "@/lib/sessions/anatomy";
import type { Anatomy, DeadWeight, DeadWeightKind, Tools } from "@/lib/sessions/types";

import { AnatomyBar } from "./anatomy-bar";
import { fmtTokens, pct } from "./format";

const KINDS: DeadWeightKind[] = ["unused_tools", "dup", "boil", "reason"];

export function DeadWeightPanel({ dw, anatomy, tools }: { dw: DeadWeight; anatomy: Anatomy; tools: Tools }) {
  const share = pct(dw.share);
  const legend = ANATOMY_ORDER.filter((k) => k !== "unknown" || anatomy.unknown > 0);
  return (
    <section data-testid="dead-weight-panel" className="space-y-3">
      <h3 className="text-sm font-semibold text-chat-fg">{S.deadWeight.title}</h3>
      {dw.turns_measured === 0 || share === null ? (
        <p className="text-sm text-chat-muted">{S.deadWeight.notMeasured}</p>
      ) : (
        <p className="flex flex-wrap items-baseline gap-x-3">
          <span className="text-3xl font-semibold tabular-nums text-chat-fg">{share}%</span>
          <span className="text-sm text-chat-muted">
            {S.deadWeight.of(fmtTokens(dw.total_tokens), fmtTokens(dw.prompt_tokens))}
          </span>
        </p>
      )}
      <AnatomyBar a={anatomy} size="lg" />
      <ul aria-label={S.deadWeight.legendAria} className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-chat-muted">
        {legend.map((k) => (
          <li key={k} className="inline-flex items-center gap-1.5">
            <i className={`inline-block h-2.5 w-2.5 rounded-sm ${ANATOMY_CLASS[k]}`} />
            {S.anatomy[k]}
          </li>
        ))}
      </ul>
      <div className="divide-y divide-chat-rule border-t border-chat-rule">
        {KINDS.map((k) => {
          const v = dw.kinds[k];
          const kp = pct(v.share);
          const def = S.deadWeight.kinds[k];
          return (
            <div key={k} data-testid={`dw-${k}`} className="flex items-baseline justify-between gap-3 py-2 text-sm">
              <div className="min-w-0">
                <div className="text-chat-fg">{def.label}</div>
                <div className="text-xs text-chat-dim">{def.sub(tools.unused, tools.offered)}</div>
              </div>
              <div className="shrink-0 text-right tabular-nums">
                {v.tokens === null ? (
                  <span className="text-chat-muted">—</span>
                ) : (
                  <>
                    <span className="text-chat-fg">≈ {fmtTokens(v.tokens)}</span>
                    {kp !== null && <span className="ml-2 text-xs text-chat-muted">{kp}%</span>}
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
      {dw.turns_measured > 0 && dw.turns_measured < dw.turns && (
        <p className="text-xs text-chat-dim">{S.deadWeight.measured(dw.turns_measured, dw.turns)}</p>
      )}
    </section>
  );
}
