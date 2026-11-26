// Download a session report (analysis Task 4.4, spec 12.3): HTML or JSON, numbers and codes with
// report-local labels (no key, session or request ids, no prompt text). The route needs the
// caller's credential, so it is fetched with it and saved from a blob.

import { authFetch } from '@/lib/auth-fetch';
import { readKeyToken } from '@/lib/sessions/key-token';

import type { LensScope } from './stream';

export type ReportFormat = 'html' | 'json';

export function reportUrl(scope: LensScope, ref: string, o: { format: ReportFormat; toolNames: boolean }): string {
  const base = scope === 'admin' ? '/api/admin/prompt-lens' : '/api/me/prompt-lens';
  return `${base}/sessions/${encodeURIComponent(ref)}/report?format=${o.format}&tool_names=${o.toolNames ? 1 : 0}`;
}

export const reportFileName = (format: ReportFormat) => `prompt-lens-report.${format}`;

/** Fetch the report and save it. Resolves true when the file was handed to the browser. */
export async function downloadReport(scope: LensScope, ref: string, o: { format: ReportFormat; toolNames: boolean }): Promise<boolean> {
  const url = reportUrl(scope, ref, o);
  let r: Response;
  if (scope === 'admin') {
    r = await authFetch(url);
  } else {
    const token = readKeyToken();
    if (!token) return false;
    r = await fetch(url, { headers: { Authorization: `Bearer ${token}` }, credentials: 'omit' });
  }
  if (!r.ok) return false;
  const blob = await r.blob();
  const href = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = href;
  a.download = reportFileName(o.format);
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
  return true;
}
