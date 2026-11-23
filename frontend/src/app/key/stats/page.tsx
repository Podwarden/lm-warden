"use client";

// /key/stats — the key holder's Stats view (stats redesign, design section 4).
//
// Only this key's own requests. The owner decided key holders see nothing
// fleet-wide (D2, D6): no server status, install names, engine or host data, no
// other key's figures, no "requests ahead". Every number on this page comes from
// /api/me/* routes that bind the key from the sign-in (app/tokens/routes_me_stats.py);
// the page never opens the admin's stats feeds or the live engine stream.
// The session identifies a request, not the client IP (D5).
//
// Prompt Lens (spec 2026-11-22, Q1): whenever it is on for the platform
// (me.prompt_lens_view_available), an eye on each of the key holder's own
// requests (table and chart) opens that request on the Prompt Lens page
// (/key/prompt-lens?req=<id>; it was a dock along the bottom of this page until
// 2026-11-23). Otherwise there is no eye column.

import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";

import { promptLensHref } from "@/components/prompt-lens/prompt-lens-page";
import { KeyShell } from "@/components/key-area/key-shell";
import { OwnCacheCard } from "@/components/stats/redesign/cache-card";
import { KeyRequestsChart } from "@/components/stats/redesign/key-requests-chart";
import { FixedCard, Seg, Section } from "@/components/stats/redesign/primitives";
import { KeyRequestsTable } from "@/components/stats/redesign/requests-table";
import { NoticeLine, ScopeBar, type ScopeTab } from "@/components/stats/redesign/scope-bar";
import { LatencyScatter, SpeedTiles } from "@/components/stats/redesign/speed-tiles";
import { TokensChart } from "@/components/stats/v2-charts";
import { type MeKey, useMe } from "@/lib/key-area/api";
import { K } from "@/lib/key-area/strings";
import { WINDOW_META, tokenRateBuckets } from "@/lib/live-history";
import {
  FIXED,
  type MeCache,
  type MeRequests,
  type MeTokens,
  type SpeedResponse,
  anchoredEnd,
  fmtAgo,
} from "@/lib/stats-redesign";
import type { StatsRange } from "@/lib/stats-v2";
import { usePersistedRange } from "@/lib/use-persisted-range";

const RANGE_KEY = "vw.key.stats.range";
const RANGE_S: Record<StatsRange, number> = { "1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800 };
const RANGE_WORDS: Record<StatsRange, { last: string; span: string }> = {
  "1h": { last: "last hour", span: "hour" },
  "6h": { last: "last 6 hours", span: "6 hours" },
  "24h": { last: "last 24 hours", span: "24 hours" },
  "7d": { last: "last 7 days", span: "7 days" },
};

function KeyStats() {
  const [range, setRange] = usePersistedRange(RANGE_KEY, "1h");
  const meta = WINDOW_META[range];
  const [chosen, setChosen] = useState<string | null>(null);

  const me = useMe<MeKey>("/api/me/key");
  const live = useMe<MeRequests>("/api/me/stats/requests", 1500);
  // The window ends now, unless it holds none of this key's requests: then it is
  // anchored to the key's last request, so an idle key never shows empty
  // panels (owner request 2026-11-22). "Back to now" undoes it.
  const listNow = useMe<SpeedResponse>(`/api/me/stats/speed?range=${range}`, 30_000);
  const [atNow, setAtNow] = useState(false);
  const lastAt = listNow.data?.last_request_at ?? null;
  const anchor = anchoredEnd(listNow.data, atNow);
  const uq = anchor != null ? `&until=${encodeURIComponent(new Date(anchor * 1000).toISOString())}` : "";
  const listAnchored = useMe<SpeedResponse>(anchor != null ? `/api/me/stats/speed?range=${range}${uq}` : null, 30_000);
  const list = anchor != null ? listAnchored : listNow;
  const models = list.data?.models ?? [];
  const liveModels = (live.data?.requests ?? []).map((r) => ({ model_id: r.model_id, model: r.model }));
  const all = [...models, ...liveModels.filter((l) => !models.some((m) => m.model_id === l.model_id))];
  const seen = new Set<string>();
  const uniq = all.filter((m) => (seen.has(m.model_id) ? false : (seen.add(m.model_id), true)));
  const model = chosen && uniq.some((m) => m.model_id === chosen) ? chosen : (uniq[0]?.model_id ?? null);
  const mq = model ? `&model=${encodeURIComponent(model)}` : "";

  const speed = useMe<SpeedResponse>(model ? `/api/me/stats/speed?range=${range}${mq}${uq}` : null, 30_000);
  const tokens = useMe<MeTokens>(model ? `/api/me/stats/tokens?range=${range}${mq}${uq}` : null, 30_000);
  const cache = useMe<MeCache>(model ? `/api/me/stats/cache?range=${range}${mq}${uq}` : null, 30_000);
  // SWR keeps the previous data when a key turns null; with no model in the
  // window there is nothing to show, not the last window's figures.
  const speedData = model ? speed.data : undefined;
  const tokensData = model ? tokens.data : undefined;
  const cacheData = model ? cache.data : undefined;
  const lensOn = !!me.data?.prompt_lens_view_available;
  const router = useRouter();
  const openInLens = useCallback((id: string) => router.push(promptLensHref("key", { req: id })), [router]);

  const state = live.data?.state ?? {};
  const tabs: ScopeTab[] = uniq.map((m) => ({
    id: m.model_id,
    name: m.model,
    state: state[m.model] === "waiting" ? "Waiting" : state[m.model] === "running" ? "Running" : "Idle",
  }));
  const scopeName = uniq.find((m) => m.model_id === model)?.model ?? "no requests yet";
  const scope = `window · ${scopeName}`;
  const buckets = tokenRateBuckets(
    (tokensData?.minutes ?? []).map((m) => ({
      minute: m.minute,
      prompt: m.prompt_tokens,
      completion: m.completion_tokens,
      cached: 0,
      cached_measured_requests: 0,
      requests: m.requests,
    })),
    {
      bucketMinutes: meta.bucketMinutes,
      sinceMinute: tokensData?.since_minute ?? 0,
      nowMinute: tokensData?.now_minute ?? 0,
      measuredFromMinute: tokensData?.since_minute ?? null,
    },
  );
  const nowEpoch = anchor ?? Date.now() / 1000;
  const sinceEpoch = nowEpoch - RANGE_S[range];
  const quietNow = !!listNow.data && listNow.data.count === 0 && lastAt != null;
  const notice = quietNow
    ? anchor != null
      ? {
          strong: `No requests in the ${RANGE_WORDS[range].last}`,
          rest: `showing the ${RANGE_WORDS[range].span} up to your last request, ${fmtAgo(lastAt)}`,
        }
      : { strong: `No requests in the ${RANGE_WORDS[range].last}`, rest: `your last request was ${fmtAgo(lastAt)}` }
    : null;

  return (
    <div data-testid="key-stats-page" className="pb-6">
      <div className="flex h-8 items-center gap-2.5 whitespace-nowrap">
        <h2 className="min-w-0 flex-1 overflow-hidden text-ellipsis text-lg font-semibold">Your usage · {me.data?.name ?? ""}</h2>
        <span className="hidden flex-none text-xs text-emerald-400 sm:inline">Live</span>
        <span className="flex-none">
          <Seg
            label="Window"
            value={range}
            options={(["1h", "6h", "24h", "7d"] as const).map((r) => ({ v: r, label: r }))}
            onChange={setRange}
          />
        </span>
      </div>
      <ScopeBar tabs={tabs} selected={model} onSelect={setChosen} emptyLabel="no requests from this key yet" emptyState="Idle" />
      <NoticeLine
        notices={notice ? [notice] : []}
        tone="info"
        ok="Signed in with your API key · you see only your own requests"
        action={
          quietNow
            ? anchor != null
              ? { label: "Back to now", onClick: () => setAtNow(true) }
              : { label: "Show last activity", onClick: () => setAtNow(false) }
            : undefined
        }
      />

      <Section title="Your speed" scope={scope} note="your requests only · prompt speed includes cache effects" testid="latency">
        <SpeedTiles data={speedData} view="key" typicalPrefill={null} />
        <LatencyScatter data={speedData} view="key" sinceEpoch={sinceEpoch} nowEpoch={nowEpoch} />
      </Section>

      <Section title="Your tokens / second" scope={scope} note="no cached / computed split" testid="tokens">
        <FixedCard height={FIXED.tokensChart} testid="tokens-card">
          <TokensChart buckets={buckets} bucketMinutes={meta.bucketMinutes} range={range} split={false} />
        </FixedCard>
      </Section>

      <Section title="Your cache" scope={`${scope} · own lens`} testid="cache">
        <OwnCacheCard data={cacheData} />
      </Section>

      <Section title="Your requests" scope={scope} note="no key, IP or install column · nobody else's rows" testid="requests">
        <div className="rounded-lg border border-chat-rule bg-chat-surface p-3">
          <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-chat-muted">Your requests chart</h3>
          <KeyRequestsChart
            points={speedData?.points ?? []}
            sinceEpoch={sinceEpoch}
            nowEpoch={nowEpoch}
            promptLens={lensOn}
            onView={openInLens}
          />
        </div>
        <div className="mt-4">
          <KeyRequestsTable
            rows={live.data?.requests ?? []}
            multiModel={uniq.length > 1}
            onEye={lensOn ? openInLens : undefined}
            lastFinishedAt={lastAt}
          />
        </div>
      </Section>

      <p data-testid="key-privacy-note" className="mt-6 border-t border-chat-rule pt-2.5 text-xs text-chat-dim">
        {K.stats.privacy}
      </p>
    </div>
  );
}

export default function KeyStatsPage() {
  return (
    <KeyShell>
      <KeyStats />
    </KeyShell>
  );
}
