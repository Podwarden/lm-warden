"use client";

// "Did the last change help?" (mockup `trends.html`, spec 13.3): version B
// against the version before it, same model, at least five scored sessions a
// side: a plain verdict from a rank test on the per-session overall, the median
// difference per score, a note when the two sides are not like for like, and the
// rules that stopped (or started) firing. Below five sessions a side it says so
// and shows no difference (A-7).

import { useMemo, useState } from "react";

import {
  diffText,
  METRICS,
  METRIC_NAME,
  RULE_LABEL,
  useCompare,
  verdictText,
  verdictTone,
  type Metric,
  type Trends,
} from "@/lib/prompt-lens/trends";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { cn } from "@/lib/utils";

const TONE = {
  ok: "border-vw-ok-fg/50 bg-vw-ok-bg text-vw-ok-fg",
  wait: "border-vw-amber-fg/50 bg-vw-amber-bg text-vw-amber-fg",
  bad: "border-vw-danger-fg/50 bg-vw-danger-bg text-vw-danger-fg",
};

export interface CompareCardProps {
  scope: LensScope;
  keyId: string | null;
  trends: Trends;
}

export function CompareCard({ scope, keyId, trends }: CompareCardProps) {
  const vs = trends.versions;
  const [picked, setPicked] = useState<{ a: string; b: string } | null>(null);
  const pair = useMemo(() => {
    if (picked && vs.some((v) => v.id === picked.a) && vs.some((v) => v.id === picked.b)) return picked;
    return vs.length >= 2 ? { a: vs[vs.length - 2].id, b: vs[vs.length - 1].id } : null;
  }, [picked, vs]);
  const { data, loading, missing } = useCompare(scope, keyId, pair?.a ?? null, pair?.b ?? null, trends.model ?? null);
  const va = vs.find((v) => v.id === pair?.a);
  const vb = vs.find((v) => v.id === pair?.b);

  return (
    <div data-testid="compare-card" className="min-w-0 rounded-lg border border-chat-rule bg-chat-surface p-3">
      <h2 className="m-0 text-[13px] font-semibold text-chat-fg">Did the last change help?</h2>
      {!pair ? (
        <p data-testid="compare-none" className="mt-2 text-[12px] text-chat-dim">
          Only one prompt version so far. Once your prompt or tool list changes, the new version is set against this one here.
        </p>
      ) : (
        <>
          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[12px] text-chat-muted">
            <select
              aria-label="Newer version"
              value={pair.b}
              onChange={(e) => setPicked({ a: pair.a, b: e.target.value })}
              className="rounded border border-chat-rule bg-chat-surface px-1 py-0.5 text-[12px] text-chat-fg"
            >
              {vs.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label}
                </option>
              ))}
            </select>
            against
            <select
              aria-label="Earlier version"
              value={pair.a}
              onChange={(e) => setPicked({ a: e.target.value, b: pair.b })}
              className="rounded border border-chat-rule bg-chat-surface px-1 py-0.5 text-[12px] text-chat-fg"
            >
              {vs.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label}
                </option>
              ))}
            </select>
            <span className="text-chat-dim">· same model{trends.model ? ` (${trends.model})` : ""}</span>
          </div>
          {loading && <p className="mt-2 text-[12px] text-chat-dim">Comparing…</p>}
          {missing && <p className="mt-2 text-[12px] text-chat-dim">One of these versions has no scored session in range.</p>}
          {data && (
            <>
              <div className="my-2">
                <span
                  data-testid="verdict"
                  data-verdict={data.verdict}
                  className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[12px]", TONE[verdictTone(data.verdict)])}
                >
                  ● {verdictText(data)}
                </span>
              </div>
              {data.verdict !== "not_enough" && (
                <div data-testid="compare-diffs" className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {METRICS.map((m: Metric) => (
                    <div key={m} className="rounded-md bg-chat-rule/30 px-2.5 py-2 text-[12px] text-chat-muted">
                      {METRIC_NAME[m]}
                      <b className="block text-[20px] tabular-nums text-chat-fg">{diffText(data.diffs[m])}</b>
                      <span className="tabular-nums text-[11px]">
                        {va?.medians[m] ?? "—"} → {vb?.medians[m] ?? "—"}
                      </span>
                    </div>
                  ))}
                </div>
              )}
              {data.note && (
                <p data-testid="compare-note" className="mt-2 text-[12px] text-chat-muted">
                  {data.like_for_like ? "" : "Not like for like: "}
                  {data.note}.
                </p>
              )}
              {data.like_for_like && data.verdict !== "not_enough" && (
                <p className="mt-2 text-[11px] text-chat-dim">Like for like: harness, route and session length are similar on both sides.</p>
              )}
              {(data.resolved.length > 0 || data.new.length > 0) && (
                <p data-testid="compare-rules" className="mt-2 text-[12px] text-chat-muted">
                  {data.resolved.length > 0 && (
                    <span>
                      Stopped firing: {data.resolved.map((r) => `${r} ${RULE_LABEL[r] ?? ""}`.trim()).join("; ")}.{" "}
                    </span>
                  )}
                  {data.new.length > 0 && <span>Started firing: {data.new.map((r) => `${r} ${RULE_LABEL[r] ?? ""}`.trim()).join("; ")}.</span>}
                </p>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
