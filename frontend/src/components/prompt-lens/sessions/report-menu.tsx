"use client";

// "Download report" on the scorecard (analysis Task 4.4): HTML or JSON, with an "include tool names"
// box. The file holds numbers and codes only: no key, session or request ids, no prompt text.

import { useId, useState } from "react";

import { downloadReport, type ReportFormat } from "@/lib/prompt-lens/report";
import type { LensScope } from "@/lib/prompt-lens/stream";

export function ReportMenu({ scope, session }: { scope: LensScope; session: string }) {
  const [open, setOpen] = useState(false);
  const [names, setNames] = useState(false);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const boxId = useId();
  async function go(format: ReportFormat) {
    setBusy(true);
    setFailed(false);
    try {
      const ok = await downloadReport(scope, session, { format, toolNames: names });
      setFailed(!ok);
      if (ok) setOpen(false);
    } catch {
      setFailed(true);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div data-testid="report-menu" className="relative inline-block">
      <button
        type="button"
        aria-haspopup="true"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="rounded-md border border-chat-rule bg-chat-surface px-2.5 py-1 text-[12px] text-chat-fg hover:bg-chat-surface-hi"
      >
        Download report
      </button>
      {open && (
        <div role="menu" className="absolute right-0 z-20 mt-1 w-56 rounded-md border border-chat-rule bg-chat-surface p-2 text-[12px] shadow-lg">
          <button type="button" role="menuitem" disabled={busy} onClick={() => void go("html")} className="block w-full rounded px-2 py-1 text-left hover:bg-chat-surface-hi">
            HTML (one page)
          </button>
          <button type="button" role="menuitem" disabled={busy} onClick={() => void go("json")} className="block w-full rounded px-2 py-1 text-left hover:bg-chat-surface-hi">
            JSON (for CI)
          </button>
          <label htmlFor={boxId} className="mt-1 flex items-start gap-2 px-2 py-1 text-chat-muted">
            <input id={boxId} type="checkbox" checked={names} onChange={(e) => setNames(e.target.checked)} className="mt-0.5" />
            <span>Include tool names</span>
          </label>
          <p className="px-2 pb-1 text-[11px] leading-snug text-chat-dim">
            Numbers and codes only: no prompt text, no key or session ids. Tools are &quot;tool #1&quot; unless you include names.
          </p>
          {failed && <p role="alert" className="px-2 text-[11px] text-vw-danger-fg">The report could not be downloaded.</p>}
        </div>
      )}
    </div>
  );
}
