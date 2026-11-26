"use client";

// The session scorecard (analysis spec 12.1, mockup `scorecard.html`): header
// facts, the score tiles (overall first, the model review apart), a secret
// notice, the top 3 fixes, the turn strip, every finding by layer and the cost
// of the session. Codes and numbers only (A-10). Phone: two tiles a row.

import { useMemo } from "react";

import { fmtTok, withMarks, type Scorecard as ScorecardData } from "@/lib/prompt-lens/scorecard";
import { setWontfix } from "@/lib/prompt-lens/trends";
import { forScope } from "@/lib/prompt-lens/findings";
import { ruleCopy } from "@/lib/prompt-lens/findings";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { FindingsList } from "./findings-list";
import { FixCard } from "./fix-card";
import { ReportMenu } from "./report-menu";
import { ReviewTile, ScoreTile } from "./score-tile";
import { TurnStrip } from "./turn-strip";
import type { WontfixActions } from "./wontfix-menu";

const HARNESS: Record<string, string> = { other: "other client" };

function duration(s: number): string {
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

function Section({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <section className="mt-5">
      <div className="mb-2 flex flex-wrap items-baseline gap-x-3">
        <h2 className="m-0 text-[13px] font-semibold text-chat-fg">{title}</h2>
        {note && <span className="text-[11px] text-chat-dim">{note}</span>}
      </div>
      {children}
    </section>
  );
}

export interface ScorecardProps {
  data: ScorecardData;
  scope: LensScope;
  /** "Show turns": open the request view on that request. */
  onShowTurn: (reqId: string) => void;
  /** After a won't-fix mark or undo: reload the scorecard (the scores moved). */
  onChanged?: () => void;
  /** The session id (`s_...`): with it the scorecard offers "Download report". */
  session?: string | null;
}

export function Scorecard({ data, scope, onShowTurn, onChanged, session = null }: ScorecardProps) {
  const s = data.session;
  const key = s.key_id ?? null;
  // Won't fix (spec 11.4): a key holder's marks name no key; the admin's name the session's.
  const actions = useMemo<WontfixActions | null>(() => {
    if (!onChanged || (scope === "admin" && !key)) return null;
    const versionId = s.version_id ?? null;
    const done = (ok: boolean) => {
      if (ok) onChanged();
    };
    return {
      versionLabel: versionId ? (s.version ?? versionId) : null,
      mark: (rule, sc) => void setWontfix(scope, { rule, on: true, version: sc === "version" ? versionId : null, key }).then(done, () => undefined),
      unmark: (rule) =>
        void (async () => {
          // The mark may be for every version or for this one: clear both.
          const a = await setWontfix(scope, { rule, on: false, version: null, key });
          const b = versionId ? await setWontfix(scope, { rule, on: false, version: versionId, key }) : true;
          done(a && b);
        })().catch(() => undefined),
    };
  }, [scope, key, onChanged, s.version, s.version_id]);
  const muted = data.findings.filter((f) => f.muted);
  const wontfixOf = (layer: "overall" | "E" | "C" | "R") => {
    const withThem = withMarks(data.scores[layer]);
    const count = layer === "overall" ? muted.length : muted.filter((f) => f.layer === layer).length;
    return withThem === null ? null : { count, withThem };
  };
  const unit = data.cost.card ? `money (${data.cost.card.currency})` : data.top.some((f) => f.saving.prefill_s !== null || f.saving.decode_s !== null) ? "GPU-seconds (no price card on this route)" : "tokens (no price card or learned rate)";
  const n = data.notice;
  const c = data.cost;
  return (
    <div data-testid="scorecard" className="min-w-0 pb-6">
      <div data-testid="scorecard-facts" className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-chat-muted">
        <span>harness <b className="font-semibold text-chat-fg">{HARNESS[s.harness] ?? s.harness}</b></span>
        <span>model <b className="font-semibold text-chat-fg">{s.model}</b></span>
        <span>route <b className="font-semibold text-chat-fg">{s.route === "relay" ? "relayed" : "own engine"}</b></span>
        <span><b className="font-semibold text-chat-fg">{s.turns}</b> turns</span>
        <span><b className="font-semibold text-chat-fg">{duration(s.duration_s)}</b></span>
        {s.version && <span>prompt <b className="font-semibold text-chat-fg">{s.version}</b></span>}
        <span className="text-chat-dim">score version {data.score_version}</span>
        {session && (
          <span className="ml-auto">
            <ReportMenu scope={scope} session={session} />
          </span>
        )}
      </div>

      <div data-testid="score-tiles" className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
        <ScoreTile layer="overall" score={data.scores.overall} scope={scope} weights={data.weights} wontfix={wontfixOf("overall")} />
        <ScoreTile layer="E" score={data.scores.E} scope={scope} wontfix={wontfixOf("E")} />
        <ScoreTile layer="C" score={data.scores.C} scope={scope} wontfix={wontfixOf("C")} />
        <ScoreTile layer="R" score={data.scores.R} scope={scope} wontfix={wontfixOf("R")} />
        <ReviewTile />
      </div>

      {n && (
        <p data-testid="secret-notice" role="note" className="mt-4 rounded-md border border-vw-amber-fg/40 bg-vw-amber-bg px-3 py-2 text-[12px] text-vw-amber-fg">
          <b>Secret in the prompt:</b> {forScope(ruleCopy(n.rule, n, data.names, scope).title, scope)}. Keep them out at the source.{" "}
          <span className="font-mono text-[10px]">R-09</span>
        </p>
      )}

      <Section title="Top 3 fixes" note={`ranked by expected saving × confidence · ${unit}`}>
        {data.top.length === 0 ? (
          <p data-testid="no-fixes" className="text-[12px] text-chat-dim">Nothing to fix in this session.</p>
        ) : (
          <div data-testid="top-fixes" className="grid grid-cols-1 gap-2 md:grid-cols-3">
            {data.top.map((f, i) => (
              <FixCard key={`${f.rule}-${f.tool ?? ""}-${i}`} rank={i + 1} finding={f} card={data} scope={scope} onShowTurn={onShowTurn} actions={actions} />
            ))}
          </div>
        )}
      </Section>

      <Section title="Turns" note="prompt tokens per turn · letters mark turns with findings">
        <TurnStrip turns={data.turns} scope={scope} onShowTurn={onShowTurn} />
      </Section>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <Section title="All findings" note={`${data.findings.length} fired`}>
          <FindingsList card={data} scope={scope} onShowTurn={onShowTurn} actions={actions} />
        </Section>
        <Section title="Cost of this session">
          <div data-testid="scorecard-cost" className="rounded-lg border border-chat-rule bg-chat-surface p-3 text-[12px] text-chat-muted">
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              <span>prompt <b className="text-chat-fg">{fmtTok(c.prompt_tokens)}</b></span>
              <span>{forScope("reused from cache", scope)} <b className="text-chat-fg">{fmtTok(c.reused_tokens)}</b></span>
              <span>recomputed <b className="text-chat-fg">{fmtTok(c.recomputed_tokens)}</b></span>
              <span>lost to fixable misses <b className="text-chat-fg">{fmtTok(c.fixable_lost_tokens)}</b></span>
            </div>
            {c.card && c.actual_input != null && (
              <p className="mt-2 text-[11px] text-chat-dim">
                Input cost ≈ {c.actual_input.toFixed(4)} {c.card.currency}
                {c.saving != null && c.saving_pct != null && <> · fixing the misses saves ≈ {c.saving_pct} %</>}
                {" "}· price card {c.card.provider} {c.card.model}, effective {c.card.effective_from}
              </p>
            )}
          </div>
        </Section>
      </div>
    </div>
  );
}
