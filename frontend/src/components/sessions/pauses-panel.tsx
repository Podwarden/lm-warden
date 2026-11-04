// A session's resumes after a pause: how many found their cache evicted, by pause length, and what a disk tier
// would have recovered.

import { Panel } from "@/components/stats/live-panels";
import { S } from "@/lib/sessions/strings";
import type { Pauses } from "@/lib/sessions/types";

import { PauseNotes, PauseTable } from "./pause-table";

export function PausesPanel({ p }: { p: Pauses }) {
  return (
    <Panel title={S.pauses.title} testid="pauses-panel">
      {p.resumes === 0 ? (
        <p className="text-sm text-chat-muted">{S.pauses.none}</p>
      ) : (
        <div className="space-y-3">
          <p className="text-sm text-chat-fg">{S.pauses.summary(p.resumes, p.evicted, p.lost_s)}</p>
          <PauseTable p={p} />
          <PauseNotes p={p} />
        </div>
      )}
    </Panel>
  );
}
