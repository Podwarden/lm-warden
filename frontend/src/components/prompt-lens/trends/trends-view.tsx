"use client";

// The Trends view (analysis spec 13.3, mockup `trends.html`): the medians of the
// key's scored sessions per day, week or prompt version with the version
// boundaries marked, the "Did the last change help?" card, and the rules that
// fired per version with what was resolved. A key holder sees their own key
// only (no fleet figure, D-4); the admin picks one key (the page's Key filter, or
// the picker in this view's message), since versions are per key. Numbers only (A-10).

import { useState } from "react";

import { forScope } from "@/lib/prompt-lens/findings";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { RULE_LABEL, ruleStatus, percent, useTrends, type TrendBy } from "@/lib/prompt-lens/trends";
import { cn } from "@/lib/utils";
import { CompareCard } from "./compare-card";
import { TrendChart } from "./trend-chart";

const BYS: TrendBy[] = ["day", "week", "version"];
const TONE = { ok: "text-vw-ok-fg", bad: "text-vw-danger-fg", muted: "text-chat-muted" };

export interface TrendsViewProps {
  scope: LensScope;
  /** Admin: the one key to read (the page's Key filter). A key holder's is the sign-in's. */
  keyId?: string | null;
  /** Admin: how many keys the Key filter holds (several is as good as none here). */
  pickedKeys?: number;
  /** Admin: every key that can be picked, and the pick (it sets the page's Key filter). */
  keyOptions?: { value: string; label: string }[];
  onPickKey?: (id: string) => void;
  /** The page's Model filter (one model); null: the newest session's. */
  model?: string | null;
  onModel?: (m: string | null) => void;
  phone?: boolean;
}

export function TrendsView({
  scope,
  keyId = null,
  pickedKeys = 0,
  keyOptions = [],
  onPickKey,
  model: modelProp = null,
  onModel,
  phone = false,
}: TrendsViewProps) {
  const [by, setBy] = useState<TrendBy>("week");
  // The page's Model filter when it hands one over; else this view's own pick.
  const [own, setOwn] = useState<string | null>(null);
  const model = onModel ? modelProp : own;
  const setModel = onModel ?? setOwn;
  const got = useTrends(scope, keyId, by, model);
  // An answer without its lists is no answer (an old server, a proxy page): the error line.
  const data = got.data && Array.isArray(got.data.points) && Array.isArray(got.data.versions) ? got.data : null;
  const { loading } = got;
  const error = got.error || (got.data !== null && data === null);

  if (scope === "admin" && !keyId) {
    return (
      <div data-testid="lens-trends-view" className="flex-1 overflow-y-auto p-3">
        <p data-testid="trends-pick-key" className="flex flex-wrap items-center gap-2 text-[13px] text-chat-muted">
          <span>
            Trends are per key, because prompt versions are.
            {pickedKeys > 1 ? ` ${pickedKeys} keys are picked in the Key filter; trends show one.` : ""}
          </span>
          {keyOptions.length > 0 ? (
            <label className="inline-flex items-center gap-2">
              <span className="text-chat-fg">Pick a key:</span>
              <select
                aria-label="Pick a key"
                value=""
                onChange={(e) => e.target.value && onPickKey?.(e.target.value)}
                className={cn(
                  "rounded border border-chat-rule bg-chat-surface px-1.5 py-1 text-[12px] text-chat-fg",
                  phone && "min-h-11 text-[16px]",
                )}
              >
                <option value="">Choose…</option>
                {keyOptions.map((k) => (
                  <option key={k.value} value={k.value}>
                    {k.label}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <span>There are no keys to pick yet.</span>
          )}
        </p>
      </div>
    );
  }

  const scored = data ? data.points.reduce((n, p) => n + (p.medians.overall !== null ? p.n : 0), 0) : 0;
  const total = data ? data.points.reduce((n, p) => n + p.n, 0) : 0;
  return (
    <div data-testid="lens-trends-view" className={cn("min-h-0 flex-1 overflow-y-auto", phone ? "px-3 pb-6 pt-2" : "p-3")}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-[12px] text-chat-muted">
        {data && data.models.length > 0 && (
          <select
            aria-label="Model"
            value={data.model ?? ""}
            onChange={(e) => setModel(e.target.value || null)}
            className="rounded border border-chat-rule bg-chat-surface px-1.5 py-1 text-[12px] text-chat-fg"
          >
            {data.models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        )}
        <div role="group" aria-label="Group by" className="inline-flex overflow-hidden rounded-md border border-chat-rule">
          {BYS.map((b) => (
            <button
              key={b}
              type="button"
              aria-pressed={by === b}
              onClick={() => setBy(b)}
              className={cn(
                "cursor-pointer border-0 bg-transparent px-2.5 py-1 text-[12px] text-chat-muted",
                by === b && "bg-chat-rule/40 text-chat-accent",
                phone && "min-h-9",
              )}
            >
              {b}
            </button>
          ))}
        </div>
        {data && (
          <span className="text-chat-dim">
            {scored} scored sessions · {total - scored} in buckets too small to score
          </span>
        )}
      </div>

      {loading && <p className="text-[13px] text-chat-dim">Loading trends…</p>}
      {error && <p data-testid="trends-error" className="text-[13px] text-vw-amber-fg">The trends could not be loaded.</p>}
      {data && data.points.length === 0 && (
        <p data-testid="trends-empty" className="text-[13px] text-chat-muted">
          No scored sessions yet. A session is scored once it has been quiet for ten minutes.
        </p>
      )}
      {data && data.points.length > 0 && (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <div className="min-w-0 rounded-lg border border-chat-rule bg-chat-surface p-3">
            <div className="mb-2 flex flex-wrap items-baseline gap-x-3">
              <h2 className="m-0 text-[13px] font-semibold text-chat-fg">Session scores per {by}</h2>
              <span className="text-[11px] text-chat-dim">
                median per {by} · band = 25th-75th percentile of the overall · dashed lines = new prompt version · a gap = under 3 scored sessions
              </span>
            </div>
            <TrendChart points={data.points} versions={data.versions} by={by} />
          </div>
          <CompareCard scope={scope} keyId={keyId} trends={data} />
        </div>
      )}

      {data && data.versions.length > 0 && (
        <section className="mt-5" data-testid="findings-by-version">
          <div className="mb-2 flex flex-wrap items-baseline gap-x-3">
            <h2 className="m-0 text-[13px] font-semibold text-chat-fg">Findings by prompt version</h2>
            <span className="text-[11px] text-chat-dim">share of sessions where the rule fired</span>
          </div>
          <div className="overflow-x-auto rounded-lg border border-chat-rule bg-chat-surface">
            <table className="w-full border-collapse text-[12px]">
              <thead>
                <tr className="text-left text-chat-muted">
                  <th className="px-3 py-2 font-medium">Rule</th>
                  <th className="px-3 py-2 font-medium">Finding</th>
                  {data.versions.map((v) => (
                    <th key={v.id} className="whitespace-nowrap px-3 py-2 font-medium">
                      {v.label}
                    </th>
                  ))}
                  <th className="px-3 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {[...new Set(data.versions.flatMap((v) => Object.keys(v.rules)))].sort().map((rule) => {
                  const st = ruleStatus(rule, data.versions);
                  return (
                    <tr key={rule} data-testid="version-rule-row" data-rule={rule} className="border-t border-chat-rule/50">
                      <td className="whitespace-nowrap px-3 py-1.5 font-mono text-[11px] text-chat-muted">{rule}</td>
                      <td className="min-w-[140px] px-3 py-1.5 text-chat-fg">{forScope(RULE_LABEL[rule] ?? rule, scope)}</td>
                      {data.versions.map((v) => (
                        <td key={v.id} className="whitespace-nowrap px-3 py-1.5 tabular-nums text-chat-muted">
                          {(v.rules[rule] ?? 0) > 0 ? percent(v.rules[rule], v.n) : "—"}
                        </td>
                      ))}
                      <td className={cn("whitespace-nowrap px-3 py-1.5", TONE[st.tone])}>{st.text}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
      <p className="mt-4 text-[11px] text-chat-dim">
        Versions are detected from the system prompt (volatile lines left out) and the tool list, or named by your app with the X-Prompt-Version header.{" "}
        {scope === "key" ? "Only your key's sessions are shown." : "One key's sessions are shown."}
      </p>
    </div>
  );
}
