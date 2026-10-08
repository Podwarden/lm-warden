// Single-readout tile for the /stats "current" row.
//
// Reusable surface for any future "one big number" widget — kept
// deliberately simple so it doesn't drift into the territory of
// MetricSummaryPanel (which renders a row of metrics inside one card).
// A StatCard is one card with one metric.
//
// Visual language matches the operator-cockpit aesthetic of the rest of
// the app: subdued slate panel, uppercase muted label, big tabular
// digits, optional hint underneath. No animations on the value — when
// the operator glances at /stats they want the number, not motion.

import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props {
  label: string;
  /** The number / string to display prominently. Use a string so the
   *  caller controls formatting (locale, thousands separators, fixed
   *  decimals). Pass "—" when the value is unavailable. */
  value: ReactNode;
  /** Optional unit ("%", "W", "GiB", …). Rendered after the value in a
   *  muted secondary color so the digits stay the focal point. */
  unit?: string;
  /** Optional one-line context under the value, e.g. "last 1h" or
   *  "max across GPUs". Keep it short. */
  hint?: string;
  /** Optional tooltip shown on hover — title attribute. Useful for
   *  surfacing data the headline value omits. */
  title?: string;
  className?: string;
}

export function StatCard({ label, value, unit, hint, title, className }: Props) {
  return (
    <div
      data-testid="stat-card"
      title={title}
      className={cn(
        "rounded-lg border border-slate-700 bg-slate-900/50 p-4",
        className,
      )}
    >
      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
        {label}
      </p>
      <p className="mt-2 flex items-baseline gap-1.5">
        <span className="text-2xl font-semibold tabular-nums text-slate-100">
          {value}
        </span>
        {unit ? (
          <span
            data-testid="stat-card-unit"
            className="text-sm font-medium text-slate-400"
          >
            {unit}
          </span>
        ) : null}
      </p>
      {hint ? (
        <p
          data-testid="stat-card-hint"
          className="mt-1 text-xs text-slate-500"
        >
          {hint}
        </p>
      ) : null}
    </div>
  );
}
