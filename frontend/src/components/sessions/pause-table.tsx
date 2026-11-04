// "How long caches survive a pause": resumes after a pause, by pause length, and how many had to prefill again.

import { S } from "@/lib/sessions/strings";
import type { Pauses } from "@/lib/sessions/types";

import { fmtApprox, pct } from "./format";

export function PauseTable({ p }: { p: Pauses }) {
  const c = S.pauses.cols;
  return (
    <div className="overflow-x-auto">
      <table className="w-full max-w-2xl text-sm tabular-nums">
        <thead className="text-left text-xs text-chat-dim">
          <tr>
            <th className="py-1.5 pr-4 font-normal">{c.pause}</th>
            <th className="py-1.5 pr-4 text-right font-normal">{c.resumes}</th>
            <th className="py-1.5 pr-4 text-right font-normal">{c.evicted}</th>
            <th className="py-1.5 pr-4 text-right font-normal">{c.share}</th>
            <th className="py-1.5 text-right font-normal">{c.lost}</th>
          </tr>
        </thead>
        <tbody>
          {p.buckets.map((b) => {
            const share = b.resumes > 0 ? pct(b.evicted / b.resumes) : null;
            return (
              <tr key={b.label} className="border-t border-chat-rule">
                <td className="py-1.5 pr-4 text-chat-fg">{S.pauses.bucket[b.label] ?? b.label}</td>
                <td className="py-1.5 pr-4 text-right">{b.resumes.toLocaleString()}</td>
                <td className="py-1.5 pr-4 text-right">{b.evicted.toLocaleString()}</td>
                <td className="py-1.5 pr-4 text-right">{share === null ? "—" : `${share}%`}</td>
                <td className="py-1.5 text-right">{fmtApprox(b.lost_s)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** The disk-tier hint (or that nothing was evicted), the unassessed count, and what a resume is. */
export function PauseNotes({ p }: { p: Pauses }) {
  const h = p.disk_tier_hint;
  return (
    <div className="space-y-1.5">
      <p className="text-sm text-chat-fg">{h ? S.pauses.hint(h.keep_s, h.recovered_s) : S.pauses.noEvictions}</p>
      {p.unassessed > 0 && <p className="text-xs text-chat-muted">{S.pauses.unassessed(p.unassessed)}</p>}
      <p className="max-w-[72ch] text-xs text-chat-dim">{S.pauses.explain}</p>
    </div>
  );
}
