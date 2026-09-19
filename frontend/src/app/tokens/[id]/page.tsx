"use client";

// /ui/tokens/{id} — token details (spec docs/superpowers/specs/2026-09-18-token-details-design.md).
// Manage the key (rename, limits, pause/resume, test, rotate, delete), chart its
// usage for any period, and watch its traffic in the god-mode dock. The look is
// the approved mockup's, verbatim (§4.6); see components/tokens/detail/*.

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, use, useCallback, useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { parseSqliteUtc } from "@/lib/token-format";
import {
  fetchSeries, parseRange, pollIntervalMs, resolveWindow, SeriesError, seriesUrl,
  type Preset, type RangeSel, type TokenDetail, type TokenSeries,
} from "@/lib/token-series";
import { RotateTokenDialog } from "@/components/tokens/rotate-token-dialog";
import { errorDetail, useTokenActions } from "@/components/tokens/use-token-actions";
import { TokenHeader } from "@/components/tokens/detail/token-header";
import { LimitsCard, type LimitsPatch } from "@/components/tokens/detail/limits-card";
import { DetailsCard } from "@/components/tokens/detail/details-card";
import { LineageCard } from "@/components/tokens/detail/lineage-card";
import { UsageSection } from "@/components/tokens/detail/usage-section";
import { UsageCharts } from "@/components/tokens/detail/usage-charts";
import { ModelUsageCard } from "@/components/tokens/detail/model-usage-card";
import type { StripWindow } from "@/components/tokens/detail/history-strip";
import { GodModeDock, useDockHeight, useDockOpen, useGodModeEnabled } from "@/components/tokens/detail/godmode-dock";
import { useWindowSelection } from "@/components/tokens/detail/use-window-selection";
import { Toast, useToast } from "@/components/tokens/detail/toast";
import { useBreadcrumb } from "@/lib/use-breadcrumb";
import { btn, CARD } from "@/components/tokens/detail/styles";
import { cn } from "@/lib/utils";

// Lands the content box on the mockup's pixels inside the app's
// `main.container.p-6` (see Task F13 notes in the plan): 1218 px wide, 5 px
// higher, 16 px gutters on phones.
const ROOT = "vw-td mx-auto -mt-[5px] max-w-[1218px] text-[14px] leading-[1.5] text-chat-fg antialiased max-sm:-mx-2";
const MAX_SPAN_S = 366 * 86400; // series 422s beyond this (spec §3.4)
const STRIP_MAX_BINS = 360; // the server ceiling for max_bins
const DOCK_BAR = 40;
const DOCK_GRIP = 6;

// The token poll: a plain number. Not an inline arrow — SWR restarts its
// polling timer whenever `refreshInterval` changes identity, so on 1h/6h,
// where `useNow` re-renders the page every 10 s, the token was never
// re-polled. And not a function returning 0 while the tab is hidden either:
// SWR stops its loop for good on a 0 (it only reschedules a non-zero
// interval), so a tab hidden once never polled again. SWR already skips a
// tick while hidden (`refreshWhenHidden` defaults to false) and keeps the
// loop alive.
const TOKEN_REFRESH_MS = 10_000;

type HttpError = Error & { status?: number };
const statusOf = (e: unknown): number | undefined => (e as HttpError | undefined)?.status;
// A 404 is final (the key was deleted elsewhere): no back-off retries.
const retryUnlessGone = (e: unknown) => statusOf(e) !== 404;

function useNow(intervalMs: number): number {
  const [now, setNow] = useState(() => Math.floor(Date.now() / 1000));
  useEffect(() => {
    const t = setInterval(() => {
      if (typeof document === "undefined" || !document.hidden) setNow(Math.floor(Date.now() / 1000));
    }, intervalMs);
    return () => clearInterval(t);
  }, [intervalMs]);
  return now;
}

// ---- series polling (controller ruling, fix round 1) -------------------------
//
// For a preset, `resolveWindow` gives {start of the minute span-1 minutes
// before the current one, now} — `to` is second-precision, so the window moves
// with every poll and always includes the current partial minute. If the SWR
// key baked that in directly, it would change on every tick of `nowSec` (an
// unbounded, ever-growing cache) while `refreshInterval` ALSO fired on its
// own cadence — double fetches. Instead, for a preset the key only encodes
// the preset's identity (stable between polls); the actual from/to window is
// resolved inside the fetcher, at the moment SWR actually calls it — so a
// poll always asks for a fresh "now", without the key ever changing. Custom
// ranges have no ambiguity (the operator typed exact numbers) and keep them
// in the key directly. `refreshInterval: pollIntervalMs(sel)` is then the
// ONLY thing driving a preset's re-fetch cadence.
type SeriesKey = readonly [tag: "series", id: string, kind: string, chain: boolean, maxBins: number, from: number, to: number];

function seriesSwrKey(id: string, sel: RangeSel, chain: boolean, maxBins: number): SeriesKey {
  return sel.kind === "custom"
    ? (["series", id, "custom", chain, maxBins, sel.from, sel.to] as const)
    : (["series", id, `preset:${sel.preset}`, chain, maxBins, 0, 0] as const);
}

async function seriesFetcher([, id, kind, chain, maxBins, from, to]: SeriesKey): Promise<TokenSeries> {
  const w = kind === "custom"
    ? { from, to }
    : resolveWindow({ kind: "preset", preset: kind.slice("preset:".length) as Preset }, Math.floor(Date.now() / 1000));
  return fetchSeries(seriesUrl(id, { from: w.from, to: w.to, chain, maxBins }));
}

// ---- history strip (controller ruling, fix round 2) --------------------------
//
// The strip covers the key's whole life, so its window only ever moves
// forward with "now" — baking the minute-floored `to` into the SWR key (as
// fix round 1 did) meant a NEW key every minute the page stayed open, and
// SWR's cache never evicts old keys: hours of an open tab meant hundreds of
// stale strip entries pinned in memory. The key here carries only the
// key's identity and the own/chain distinction (stable for the page's whole
// life); `refreshInterval: 60_000` is the sole re-fetch driver, and the
// window (`to = now`, `from` clamped to the key's lineage and the server's
// 366-day cap) is resolved inside the fetcher at fetch time, exactly like a
// preset series request. `to` is deliberately NOT floored to the minute
// (unlike fix round 1's page-local `bounds`) — a floored `to` could sit
// behind the committed preset window's own (unfloored) `to`, which drew the
// selection brush hanging off the strip's right edge.
type StripKey = readonly [tag: "strip", id: string, chain: boolean];

// `from`, extracted so the fetcher below is a plain inline arrow function
// (ESLint's exhaustive-deps needs that shape to check `lineageStart` itself,
// rather than a factory call it can't see into).
function stripFrom(lineageStart: number, to: number): number {
  return Math.min(Math.max(lineageStart, to - MAX_SPAN_S), to - 60);
}

export default function TokenDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  // useSearchParams opts the tree into client-side rendering; Suspense is
  // required around it (same as chat2/page.tsx).
  return (
    <Suspense fallback={null}>
      <TokenDetailView id={id} />
    </Suspense>
  );
}

function NotFound() {
  return (
    <div className={ROOT} style={{ paddingBottom: 40 }}>
      <section className={cn(CARD, "max-w-[460px]")}>
        <h1 className="m-0 mb-2 text-[18px] font-semibold">Token not found</h1>
        <p className="m-0 text-[13px] text-chat-muted">It may have been deleted, or the link is wrong.</p>
        <Link href="/tokens" className={btn("default", "mt-3.5 no-underline")}>Back to API tokens</Link>
      </section>
    </div>
  );
}

function LoadError({ message }: { message: string }) {
  return (
    <div className={ROOT} style={{ paddingBottom: 40 }}>
      <section role="alert" className={cn(CARD, "max-w-[460px]")}>
        <h1 className="m-0 mb-2 text-[18px] font-semibold">Couldn&apos;t load this token</h1>
        <p className="m-0 text-[13px] text-vw-danger-fg">{message}</p>
        <Link href="/tokens" className={btn("default", "mt-3.5 no-underline")}>Back to API tokens</Link>
      </section>
    </div>
  );
}

function Loading() {
  return (
    <div className={ROOT} style={{ paddingBottom: 40 }} aria-busy="true">
      {/* The title row's placeholder; the crumbs row it used to sit under is
          the app-wide breadcrumb strip now. */}
      <div className="h-[39px] w-80 animate-pulse rounded bg-chat-surface" />
      <div className="mt-[22px] grid grid-cols-[300px_minmax(0,1fr)] gap-5 max-[960px]:grid-cols-1">
        <div className="h-[380px] animate-pulse rounded-lg bg-chat-surface" />
        <div className="h-[380px] animate-pulse rounded-lg bg-chat-surface" />
      </div>
    </div>
  );
}

function TokenDetailView({ id }: { id: string }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const sel = useMemo(() => parseRange(searchParams), [searchParams]);
  const nowSec = useNow(pollIntervalMs(sel) || 60_000);
  const toast = useToast();

  const key = `/api/tokens/${encodeURIComponent(id)}`;
  // Once the token 404s (deleted elsewhere) the page stops asking (#251): no
  // retries, no polls, no focus/reconnect revalidation, and the series and
  // strip requests are unmounted (their keys go null below). Remembered per
  // key, so navigating to another token starts fresh.
  const [goneKey, setGoneKey] = useState<string | null>(null);
  const gone = goneKey === key;
  const { data: token, error, isLoading, mutate } = useSWR<TokenDetail>(key, authFetchJSON, {
    refreshInterval: gone ? 0 : TOKEN_REFRESH_MS,
    revalidateOnFocus: !gone,
    revalidateOnReconnect: !gone,
    shouldRetryOnError: retryUnlessGone,
  });
  const status = statusOf(error);
  useEffect(() => {
    if (status === 404) setGoneKey(key);
  }, [status, key]);
  const refresh = useCallback(() => {
    mutate().catch(() => {});
  }, [mutate]);
  const actions = useTokenActions({ id, name: token?.name ?? "" }, refresh);
  // The breadcrumb strip and the back button name the token, not its id. A
  // token that failed to load falls back to the id so the crumb isn't stuck
  // on the italic "Token" placeholder.
  useBreadcrumb({ title: token?.name ?? (error ? id : undefined) });

  // ---- token set --------------------------------------------------------------
  const [chainOn, setChainOn] = useState(true);
  const lineage = token?.lineage ?? [];
  const selfIdx = lineage.findIndex((e) => e.is_self);
  const predecessors = selfIdx > 0 ? lineage.slice(0, selfIdx) : [];
  const chain = chainOn && predecessors.length > 0;
  const tokenIds = chain ? [...predecessors.map((e) => e.id), id] : [id];
  const rotations = predecessors
    .map((e) => parseSqliteUtc(e.rotated_at))
    .filter((t): t is number => t != null);

  // ---- window -------------------------------------------------------------------
  const created = parseSqliteUtc(token?.created_at) ?? nowSec;
  const lineageStart = Math.min(created, ...lineage.map((e) => parseSqliteUtc(e.created_at) ?? created));

  // ---- history strip ---------------------------------------------------------------
  // The strip asks for the finest bins the server allows (the ladder rung for
  // `max_bins = width / 4` is usually wider than the mockup's 4 px buckets)
  // and re-buckets them into floor(width / 4) bars itself (HistoryStrip). It
  // draws token volume only, so it asks the server to skip the latency
  // timings (`timings=0`, #251) and the per-model split (`by_model=0`). See the `stripFetcher` below for how the key
  // and window are resolved.
  const stripFetcher = useCallback(
    async ([, tid, chain]: StripKey): Promise<TokenSeries> => {
      const to = Math.floor(Date.now() / 1000);
      return fetchSeries(seriesUrl(tid, { from: stripFrom(lineageStart, to), to, chain, maxBins: STRIP_MAX_BINS, timings: false, byModel: false }));
    },
    [lineageStart],
  );
  const stripOpts = { keepPreviousData: true, revalidateOnFocus: false, shouldRetryOnError: false, refreshInterval: 60_000 };
  const stripKeyChain: StripKey | null = token && !gone && predecessors.length > 0 ? ["strip", id, true] : null;
  const stripKeyOwn: StripKey | null = token && !gone ? ["strip", id, false] : null;
  const { data: stripChain, mutate: mutateStripChain } = useSWR<TokenSeries, Error, StripKey | null>(stripKeyChain, stripFetcher, stripOpts);
  const { data: stripOwn, mutate: mutateStripOwn } = useSWR<TokenSeries, Error, StripKey | null>(stripKeyOwn, stripFetcher, stripOpts);
  // Drawn from the strip's OWN response (`own` is always requested, `chain`
  // only when there are earlier keys) — never from a page-local "now", so
  // the strip's drawn axis always matches the window it actually fetched
  // (controller ruling, fix round 2). Falls back to the same "key's whole
  // life, clamped to 366d, from < to" calc before that response has landed.
  const fallbackTo = Math.floor(nowSec / 60) * 60;
  const fallbackBounds: StripWindow = {
    from: Math.min(Math.max(lineageStart, fallbackTo - MAX_SPAN_S), fallbackTo - 60),
    to: fallbackTo,
  };
  const bounds: StripWindow = stripOwn
    ? { from: stripOwn.from_minute * 60, to: stripOwn.to_minute * 60 }
    : fallbackBounds;

  const committed = resolveWindow(sel, nowSec);
  const replaceQuery = useCallback(
    (q: string) => router.replace(`/tokens/${encodeURIComponent(id)}?${q}`, { scroll: false }),
    [router, id],
  );
  const { shownWindow, shownSel, onWindowChange, onApply: applyWindow, onPreset, onCustom } =
    useWindowSelection(sel, committed, replaceQuery);

  // ---- series ---------------------------------------------------------------------
  // Controller ruling (fix round 1): a preset's SWR key stays stable between
  // polls (see `seriesSwrKey`/`seriesFetcher` above) — `refreshInterval:
  // pollIntervalMs(sel)` is the ONLY thing driving a preset's re-fetch
  // cadence (10s for 1h/6h, 60s for 24h/7d, none for Custom).
  const seriesKey = token && !gone ? seriesSwrKey(id, sel, chain, 360) : null;
  const { data: seriesData, error: seriesErr, mutate: mutateSeries } = useSWR<TokenSeries, Error, typeof seriesKey>(seriesKey, seriesFetcher, {
    keepPreviousData: true,
    revalidateOnFocus: false,
    shouldRetryOnError: false,
    refreshInterval: pollIntervalMs(sel),
  });
  const [lastGood, setLastGood] = useState<TokenSeries | null>(null);
  useEffect(() => {
    if (seriesData) setLastGood(seriesData);
  }, [seriesData]);
  const series = seriesData ?? lastGood;
  const rangeError = seriesErr instanceof SeriesError && seriesErr.status === 422 ? seriesErr.detail : null;

  // Custom-row Apply. A custom range does not poll, so an Apply that leaves
  // the URL unchanged (the same times, or times clamped back to the current
  // window) would otherwise neither refetch nor show anything. A changed
  // window refetches through its new series key; an unchanged one is
  // revalidated here. The strip is refetched either way, so its "now" end —
  // and so the lifetime the next Apply clamps to — is fresh too.
  const onApply = useCallback((w: StripWindow) => {
    if (!applyWindow(w)) mutateSeries().catch(() => {});
    mutateStripOwn().catch(() => {});
    mutateStripChain().catch(() => {});
  }, [applyWindow, mutateSeries, mutateStripOwn, mutateStripChain]);

  // ---- dock ---------------------------------------------------------------------------
  const gmEnabled = useGodModeEnabled();
  const [dockOpen, setDockOpen] = useDockOpen(gmEnabled); // never remembered, never auto-opens
  const [dockH, setDockH] = useDockHeight();

  const showToast = toast.show; // stable (useCallback) — named so the effect's deps are exact
  useEffect(() => {
    if (actions.deleteError) showToast(actions.deleteError);
  }, [actions.deleteError, showToast]);

  // ---- actions --------------------------------------------------------------------------
  async function patch(body: Record<string, unknown>): Promise<string | null> {
    const r = await authFetch(key, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) return errorDetail(r, `HTTP ${r.status}`);
    // A 2xx whose body isn't JSON (a proxy's HTML page, a truncated reply)
    // must not reject `onRename`/`onSaveLimits` with no message (#251). The
    // change may well have been applied, so re-read the token as well.
    let fresh: TokenDetail;
    try {
      fresh = (await r.json()) as TokenDetail;
    } catch {
      refresh();
      return `the server's reply could not be read (HTTP ${r.status})`;
    }
    await mutate(fresh, { revalidate: false });
    return null;
  }
  async function onRename(name: string): Promise<boolean> {
    const err = await patch({ name });
    toast.show(err ? `Rename failed: ${err}` : "Name saved");
    return err == null;
  }
  async function onSaveLimits(p: LimitsPatch): Promise<string | null> {
    // Resolves only once the PATCH response is applied to the SWR cache (see
    // `patch` above), so a second click on Save can never re-send the same
    // patch while the first is still in flight.
    const err = await patch({ ...p });
    if (!err) toast.show("Limits saved");
    return err;
  }
  async function onPauseToggle() {
    if (!token) return;
    const pausing = !token.is_paused;
    if (await actions.setPaused(pausing)) toast.show(pausing ? "Token paused" : "Token resumed");
  }
  async function onTest() {
    const r = await actions.runTest();
    if (!r) return; // a test is already running; its own toast will follow
    const n = r.models ?? 0;
    toast.show(
      r.ok ? `Test passed: ${n} model${n === 1 ? "" : "s"} reachable, proxy up`
        : r.paused ? "Test: key is paused, requests get 403"
          : `Test failed: ${r.detail ?? `HTTP ${r.status}`}`,
    );
  }
  async function onDelete() {
    const deleted = await actions.remove();
    if (deleted) router.push("/tokens");
    else if (deleted === false) refresh(); // null: nothing was sent
  }

  // ---- render ----------------------------------------------------------------------------
  // A poll that 404s (the token was deleted elsewhere) lands here exactly
  // like an unknown id on first load — SWR keeps the last good `token`
  // around, but `error.status` always wins.
  if (status === 404) return <NotFound />;
  if (isLoading || (!token && !error)) return <Loading />;
  if (!token) return <LoadError message={error instanceof Error ? error.message : "Unknown error"} />;

  const paddingBottom = gmEnabled ? (dockOpen ? dockH + DOCK_BAR + 44 - 24 : 64 + 44 - 24) : 40;
  const toastBottom = gmEnabled ? (dockOpen ? dockH + DOCK_GRIP + DOCK_BAR + 20 : DOCK_BAR + 20) : 20;

  return (
    <div className={ROOT} style={{ paddingBottom }}>
      <TokenHeader
        token={token}
        nowSec={nowSec}
        pausing={actions.pausing}
        testing={actions.testing}
        deleting={actions.busy}
        pauseError={actions.pauseError}
        onRename={onRename}
        onPauseToggle={onPauseToggle}
        onTest={onTest}
        onRotate={actions.openRotate}
        onDelete={onDelete}
      />

      <div className="mt-[22px] grid grid-cols-[300px_minmax(0,1fr)] items-start gap-5 max-[960px]:grid-cols-1">
        <aside className="space-y-3.5">
          <LimitsCard token={token} onSave={onSaveLimits} />
          <DetailsCard token={token} nowSec={nowSec} />
          <LineageCard lineage={lineage} nowSec={nowSec} />
        </aside>

        <UsageSection
          sel={shownSel}
          window={shownWindow}
          bounds={bounds}
          showChainToggle={predecessors.length > 0}
          chain={chainOn}
          onChainChange={setChainOn}
          onPreset={onPreset}
          onCustom={onCustom}
          onWindowChange={onWindowChange}
          onApply={onApply}
          rangeError={rangeError}
          strip={{ chain: stripChain ?? null, own: stripOwn ?? null, rotations }}
          series={series}
        >
          {series && (
            <>
              <ModelUsageCard series={series} nowSec={nowSec} />
              <UsageCharts series={series} nowSec={nowSec} />
            </>
          )}
        </UsageSection>
      </div>

      <RotateTokenDialog
        open={actions.rotateOpen}
        tokenId={id}
        onClose={actions.closeRotate}
        onRotated={(successorId) => router.push(`/tokens/${encodeURIComponent(successorId)}`)}
      />

      {gmEnabled && (
        <GodModeDock
          tokenName={token.name}
          tokenIds={tokenIds}
          includesEarlier={chain}
          open={dockOpen}
          onOpenChange={setDockOpen}
          height={dockH}
          onHeightChange={setDockH}
        />
      )}

      <Toast api={toast} bottomPx={toastBottom} />
    </div>
  );
}
