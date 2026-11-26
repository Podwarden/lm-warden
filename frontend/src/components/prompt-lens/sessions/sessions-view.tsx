"use client";

// The Sessions view: the scored sessions beside the scorecard of the picked one
// (analysis spec 16.1). Phone: the list, or the scorecard in its place with a
// "‹ Sessions" button (its own history entry, so the back gesture returns).

import { useCallback } from "react";

import { useBackLayer } from "@/lib/prompt-lens/back-layer";
import { useScorecard } from "@/lib/prompt-lens/scorecard";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { cn } from "@/lib/utils";
import { Scorecard } from "./scorecard";
import { SessionList } from "./session-list";

const LIST_WIDTH_PX = 360;

export interface SessionsViewProps {
  scope: LensScope;
  phone: boolean;
  keyNames?: Record<string, string>;
  /** Admin: only these keys' sessions (the bar's Key filter; none: every key). */
  keys?: string[] | null;
  /** Only these models' and clients' sessions (the bar's Model and Client filters). */
  models?: string[] | null;
  harnesses?: string[] | null;
  /** The list tells the bar how many sessions it shows (and whether more exist). */
  onCount?: (shown: number | null, more: boolean) => void;
  /** The session id (`s_...`) open in the scorecard, or null. */
  session: string | null;
  onSession: (id: string | null) => void;
  onShowTurn: (reqId: string) => void;
}

function Pane({ scope, session, onShowTurn }: { scope: LensScope; session: string; onShowTurn: (reqId: string) => void }) {
  const { data, loading, missing, error, reload } = useScorecard(scope, session);
  if (loading) return <p className="text-[13px] text-chat-dim">Loading the scorecard…</p>;
  if (missing)
    return (
      <p data-testid="scorecard-missing" className="text-[13px] text-chat-muted">
        This session is not available: it is older than the kept history, or it is not yours to read.
      </p>
    );
  if (error || !data || !data.scores) return <p data-testid="scorecard-error" className="text-[13px] text-vw-amber-fg">The scorecard could not be loaded.</p>;
  return <Scorecard data={data} scope={scope} onShowTurn={onShowTurn} onChanged={reload} session={session} />;
}

export function SessionsView({ scope, phone, keyNames, keys, models, harnesses, onCount, session, onSession, onShowTurn }: SessionsViewProps) {
  const back = useCallback(() => onSession(null), [onSession]);
  useBackLayer(phone && session !== null, back);
  const showList = !phone || session === null;
  const showPane = !phone || session !== null;
  return (
    <div data-testid="lens-sessions-view" className="relative flex min-h-0 flex-1">
      {showList && (
        <div
          data-testid="lens-sessions-list"
          className={cn("overflow-y-auto", phone ? "min-w-0 flex-1 overscroll-contain" : "flex-none border-r border-chat-rule")}
          style={phone ? undefined : { width: LIST_WIDTH_PX }}
        >
          <SessionList scope={scope} keyNames={keyNames} keys={keys} models={models} harnesses={harnesses} onCount={onCount} selected={session} onOpen={onSession} />
        </div>
      )}
      {showPane && (
        <div data-testid="lens-sessions-pane" className={cn("min-w-0 flex-1 overflow-y-auto", phone ? "overscroll-contain px-3 pb-6 pt-2" : "p-3")}>
          {phone && (
            <button type="button" onClick={back} className="mb-2 min-h-11 min-w-11 cursor-pointer rounded-[5px] border border-chat-rule bg-transparent px-[9px] py-[3px] text-[12px] text-chat-fg">
              ‹ Sessions
            </button>
          )}
          {session ? (
            <Pane scope={scope} session={session} onShowTurn={onShowTurn} />
          ) : (
            <p className="text-[13px] text-chat-dim">Pick a session in the list to see its scorecard.</p>
          )}
        </div>
      )}
    </div>
  );
}
