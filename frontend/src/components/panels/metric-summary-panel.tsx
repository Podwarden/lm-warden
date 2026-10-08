// Generic stat-card row for the /stats page (and any future dashboard).
//
// The plan §11.7 says "adapt the podwarden `metric-summary-panel.tsx`",
// but no podwarden source ships in this repo to copy from. Implement a
// lightweight panel from scratch — title + a row of label/value cards.
// Driven entirely by props so it can sit above any chart.

import { cn } from "@/lib/utils";

export interface Metric {
  label: string;
  value: string;
  /** optional hint shown under the value in muted text (e.g. "last 1h") */
  hint?: string;
}

interface Props {
  title: string;
  metrics: Metric[];
  className?: string;
}

export function MetricSummaryPanel({ title, metrics, className }: Props) {
  return (
    <section
      className={cn(
        "rounded-lg border border-slate-700 bg-slate-900/50 p-4",
        className,
      )}
    >
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
        {title}
      </h2>
      {metrics.length === 0 ? (
        <p className="mt-3 text-sm text-slate-500">No data yet.</p>
      ) : (
        <dl className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-3">
          {metrics.map((m) => (
            <div key={m.label}>
              <dt className="text-xs uppercase tracking-wide text-slate-500">
                {m.label}
              </dt>
              <dd className="mt-1 text-xl font-semibold text-slate-100">
                {m.value}
              </dd>
              {m.hint && (
                <dd className="text-xs text-slate-500">{m.hint}</dd>
              )}
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}
