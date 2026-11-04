"use client";

// The session page head (mockup sessions-redesign-v1): who and which harness, when it started and on what, then six
// facts. Every time figure is an estimate (≈); an unknown one reads "—" with its cause as the tooltip.

import { S } from "@/lib/sessions/strings";
import type { SessionResponse } from "@/lib/sessions/types";

import { fmtApprox, fmtDuration, fmtStart, fmtTokens, pct } from "./format";

const LIVE_S = 120;

function Fact({
  id,
  label,
  value,
  sub,
  title,
  testid,
}: {
  id: string;
  label: string;
  value: string;
  sub: string;
  title?: string;
  testid?: string;
}) {
  return (
    <div data-testid={`fact-${id}`} className="min-w-0">
      <div className="text-xs text-chat-muted">{label}</div>
      <div className="text-lg font-semibold tabular-nums text-chat-fg" title={title} data-testid={testid}>
        {value}
      </div>
      <div className="text-xs text-chat-dim">{sub}</div>
    </div>
  );
}

function startText(t: number | null): string {
  if (t === null) return "—";
  const d = new Date(t * 1000);
  if (d.toDateString() === new Date().toDateString()) {
    return S.header.today(`${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`);
  }
  return fmtStart(t);
}

export function SessionHeader({ s }: { s: SessionResponse }) {
  const t = s.totals;
  const prompt = t.prompt_tokens ?? t.prefill_tokens;
  const share = pct(t.cache_share ?? null);
  const rel = t.relayed;
  const relShare = rel && rel.prefill_tokens > 0 ? pct(rel.cached_tokens / rel.prefill_tokens) : null;
  const live = t.last_at !== null && Date.now() / 1000 - t.last_at <= LIVE_S;
  const wall = fmtDuration(t.wall_s);
  const cacheKnown = s.turns.some((x) => !x.relayed && (x.cache_known ?? x.cached_tokens !== null));
  // Why a time figure is unknown, most specific first.
  const cause =
    t.where === "relayed"
      ? S.header.unknown.relayed
      : !cacheKnown
        ? S.header.unknown.cache
        : t.replicas === null
          ? S.header.unknown.model
          : S.header.unknown.rate;
  const time = (v: number | null | undefined) => (v === null || v === undefined ? "—" : fmtApprox(v));
  const why = (v: number | null | undefined) => (v === null || v === undefined ? cause : undefined);
  const L = S.header.labels;
  return (
    <section data-testid="session-header" className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-chat-fg">
          {S.header.title(s.token_name, S.harness[s.harness ?? "other"] ?? S.harness.other)}
        </h1>
        <p className="text-sm text-chat-muted">
          {S.header.meta(
            startText(t.started_at),
            live ? S.header.runningFor(wall) : S.header.ranFor(wall),
            s.models.join(", "),
            t.replicas ?? null,
          )}
          <span className="ml-2 font-mono text-xs text-chat-dim">{s.session_id.slice(2, 10)}</span>
        </p>
      </div>
      <div className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-3">
        <Fact id="requests" label={L.requests} value={t.requests.toLocaleString()} sub={S.header.subs.requests(t.conversations, t.compactions)} />
        <Fact
          id="prompt"
          label={L.prompt}
          value={fmtTokens(prompt)}
          sub={share === null ? S.header.subs.promptUnknown : S.header.subs.prompt(share)}
        />
        <Fact id="generated" label={L.generated} value={fmtTokens(t.generated_tokens)} sub={S.header.subs.generated} />
        <Fact
          id="served"
          label={L.served}
          value={time(t.cache_served_s)}
          title={why(t.cache_served_s) ?? ""}
          sub={S.header.subs.served}
        />
        <Fact
          id="saved"
          label={L.saved}
          value={time(t.saved_vs_vanilla_s)}
          title={why(t.saved_vs_vanilla_s) ?? S.list.savedNote}
          sub={S.header.subs.saved}
          testid="saved-vs-vanilla"
        />
        <Fact
          id="recomputed"
          label={L.recomputed}
          value={time(t.recomputed_s)}
          title={why(t.recomputed_s) ?? ""}
          sub={S.header.subs.recomputed}
        />
      </div>
      {(rel || s.truncated || s.oldest_data_at !== null) && (
        <div className="space-y-0.5">
          {rel && <p className="text-xs text-chat-muted">{S.header.relayed(rel.requests, relShare)}</p>}
          {s.truncated && <p className="text-xs text-chat-warn">{S.header.truncated}</p>}
          {s.oldest_data_at !== null && (
            <p className="text-xs text-chat-dim">{S.header.oldest(new Date(s.oldest_data_at * 1000).toLocaleDateString())}</p>
          )}
        </div>
      )}
    </section>
  );
}
