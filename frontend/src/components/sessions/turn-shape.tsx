"use client";

import { S } from "@/lib/sessions/strings";
import type { Turn } from "@/lib/sessions/types";

export function TurnShape({ t }: { t: Turn }) {
  const sh = t.shape;
  const dc = t.divergence_cause ?? t.divergence_cause_own; // key holders: own lens only
  const cause = dc ? S.turn.causeShort[dc] ?? dc : "";
  return (
    <div data-testid="turn-shape" className="grid grid-cols-2 gap-x-6 gap-y-1 px-2 py-2 text-xs sm:grid-cols-3">
      {sh?.sys !== undefined && (
        <div>
          {S.turn.sys}: {S.turn.chars(sh.sys)}
        </div>
      )}
      {sh?.tools !== undefined && (
        <div>
          {S.turn.tools}: {sh.tools} ({S.turn.chars(sh.tools_chars ?? 0)})
        </div>
      )}
      {sh?.msgs && (
        <div>
          {S.turn.msgs}: {sh.msgs.user + sh.msgs.assistant + sh.msgs.tool}
        </div>
      )}
      {sh?.last_user !== undefined && sh.last_user !== null && (
        <div>
          {S.turn.lastUser}: {S.turn.chars(sh.last_user)}
        </div>
      )}
      {sh?.brk !== undefined && sh.brk !== null && (
        <div className="text-orange-400">{S.turn.broke(sh.brk, cause)}</div>
      )}
      <div>
        {S.turn.replica}: {t.dp_rank ?? "—"}
      </div>
      <div className="col-span-full text-chat-muted">
        {t.dead_weight ? S.turn.deadWeight(t.dead_weight) : S.turn.deadWeightUnknown}
      </div>
    </div>
  );
}
