"use client";

// /router/stats → "Activity" (router page redesign, plan §2.4 / §4.8): four
// KPI cards, the by-rule table (with each target's breaker and pass-through as
// the last row), local models and breakers, fallback reasons and the last 100
// decisions with a route filter. Counters are per process (a restart zeroes
// them); with no request since start or reset, one empty state replaces the
// whole block.

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { fmtClock, fmtInt, type RouterStatsOut, type TargetStats } from "@/lib/router";
import { StatCard } from "@/components/stat-card";
import { Modal } from "@/components/ui/modal";
import {
  LINK, PILL_DOT, SEC, SEC_BODY, SEC_HEAD, SEC_NOTE, SEC_TITLE,
  STRIP_TEXT, STRIP_TITLE, btn, pill, strip, type Tone,
} from "./styles";
import {
  DecisionsTable, RouteFilterChips, TABLE, TABLE_WRAP, TD, TD_EMPTY, TD_HEAD, TD_N, TH, TH_N, TR,
  fmtMsPair, type Decision, type RouteFilter,
} from "./decisions-table";

const STATS_KEY = "/api/router/stats";
const DECISIONS_KEY = "/api/router/decisions?limit=100";
// A stable number: SWR restarts its timer when the option changes identity.
const POLL_MS = 2000;
const POLL = { refreshInterval: POLL_MS } as const;

const BREAKER_TONE: Record<string, Tone> = { closed: "ok", half_open: "amber", open: "danger" };

function BreakerPill({ state }: { state: string }) {
  return <span className={pill(BREAKER_TONE[state] ?? "idle")}>{state}</span>;
}

function openUntil(iso: string | null, nowMs: number): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const s = Math.round((t - nowMs) / 1000);
  if (s <= 0) return "now";
  if (s < 60) return `in ${s} s`;
  if (s < 3600) return `in ${Math.round(s / 60)} min`;
  return `in ${Math.round(s / 3600)} h`;
}

/** "09:12" today, "3 Oct 09:12" otherwise. */
function fmtSince(iso: string, nowMs: number): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const sameDay = d.toDateString() === new Date(nowMs).toDateString();
  return sameDay
    ? fmtClock(iso)
    : `${d.toLocaleDateString("en-GB", { day: "numeric", month: "short" })} ${fmtClock(iso)}`;
}

const PROCESS_NOTE = "counters are per process; a restart zeroes them.";

export function RouterStats({
  route,
  onRouteChange,
}: {
  /** The decisions filter to start from (the page reads it from `?route=`). */
  route?: RouteFilter;
  /** Told about chip clicks; `null` means "All" (the page writes `?route=`). */
  onRouteChange?: (route: Exclude<RouteFilter, "all"> | null) => void;
} = {}) {
  const stats = useSWR<RouterStatsOut>(STATS_KEY, authFetchJSON, POLL);
  const decisions = useSWR<{ decisions: Decision[] }>(DECISIONS_KEY, authFetchJSON, POLL);
  const [filter, setFilter] = useState<RouteFilter>(route ?? "all");
  const [confirming, setConfirming] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);
  const [resetting, setResetting] = useState(false);
  const cancelRef = useRef<HTMLButtonElement | null>(null);

  // The URL is the source: back/forward or a deep link changes `route`.
  useEffect(() => {
    setFilter(route ?? "all");
  }, [route]);

  function chooseFilter(f: RouteFilter) {
    setFilter(f);
    onRouteChange?.(f === "all" ? null : f);
  }

  async function refetch() {
    await Promise.all([stats.mutate(), decisions.mutate()]);
  }

  async function reset() {
    setResetting(true);
    setResetError(null);
    try {
      const r = await authFetch("/api/router/stats/reset", { method: "POST" });
      if (!r.ok) {
        setResetError(`Reset failed (HTTP ${r.status})`);
        return;
      }
      setConfirming(false);
      await refetch();
    } catch (err) {
      setResetError(err instanceof Error ? err.message : "Network error");
    } finally {
      setResetting(false);
    }
  }

  const s = stats.data;
  const rows = decisions.data?.decisions ?? [];
  const nowMs = Date.now();
  // SWR keeps the last good data next to a refetch error; show both.
  const error = stats.error ?? decisions.error;

  const errorReasons = s
    ? Object.entries(s.error_reasons ?? {})
        .filter(([, n]) => n > 0)
        .sort((a, b) => b[1] - a[1])
    : [];
  const fmtReasons = (rs: [string, number][]) => rs.map(([r, n]) => `${r} ${fmtInt(n)}`).join(", ");
  // error_reasons are reasons recorded on local, pass-through and error routes (e.g. a
  // local request the engine rejected with 400), so they need not match the counts on
  // this card; label them as what they are.
  const errorReasonsHint =
    errorReasons.length > 0 ? `engine/upstream errors: ${fmtReasons(errorReasons.slice(0, 3))}` : undefined;
  const errorReasonsTitle =
    errorReasons.length > 0
      ? `Engine and upstream error reasons, also recorded on local and pass-through requests, so not limited to the counts above: ${fmtReasons(errorReasons)}`
      : undefined;

  const decisionsSection = (
    <section aria-labelledby="act-decisions-h" className={SEC}>
      <div className={SEC_HEAD}>
        <h2 id="act-decisions-h" className={SEC_TITLE}>Recent decisions</h2>
        <span className={SEC_NOTE}>last 100 · newest first</span>
        <div className="ml-auto max-[759px]:ml-0">
          <RouteFilterChips decisions={rows} value={filter} onChange={chooseFilter} />
        </div>
      </div>
      <div className={SEC_BODY}>
        <DecisionsTable decisions={rows} filter={filter} />
      </div>
    </section>
  );

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div className="min-w-0">
          <h1 className="m-0 text-2xl font-semibold">Router activity</h1>
          <p className="mt-1 text-[13px] text-chat-muted">
            {s?.since
              ? `Since ${fmtSince(s.since, nowMs)} · ${PROCESS_NOTE}`
              : s
                ? `No requests yet · ${PROCESS_NOTE}`
                : `Counters are per process; a restart zeroes them.`}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 max-[759px]:w-full">
          <Link href="/router" className={`${LINK} text-[13px] font-normal text-chat-muted`}>
            ← Router
          </Link>
          {s && (
            <span className={pill(s.enabled ? "ok" : "idle")}>
              <span aria-hidden="true" className={PILL_DOT} />
              {s.enabled ? "Routing on" : "Routing off"}
            </span>
          )}
          <button
            type="button"
            className={btn("default", "max-[759px]:min-h-10")}
            onClick={() => {
              setResetError(null);
              setConfirming(true);
            }}
            disabled={resetting}
          >
            Reset counters
          </button>
        </div>
      </header>

      {error && (
        <div role="alert" className={strip("danger")}>
          <div className="min-w-0 flex-1">
            <p className={`m-0 ${STRIP_TITLE.danger}`}>
              Could not load router stats{s ? " (showing the last good data)" : ""}.
            </p>
            <p className={STRIP_TEXT}>The page keeps polling every 2 s.</p>
          </div>
          <button type="button" className={btn("default", "max-[759px]:min-h-10")} onClick={() => void refetch()}>
            Retry
          </button>
        </div>
      )}

      {!s ? (
        // Stats failed but the decisions loaded: still show them.
        error ? decisions.data && decisionsSection : <Skeleton />
      ) : s.since === null ? (
        <>
          <EmptyState enabled={s.enabled} />
          {/* A reset clears breakers too, but never hide an incident. */}
          {s.targets.some((t) => t.breaker !== "closed") && <TargetsSection targets={s.targets} nowMs={nowMs} />}
        </>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <StatCard className={KPI} label="Local" value={fmtInt(s.totals.local ?? 0)} />
            <StatCard className={KPI} label="Anthropic (pass-through)" value={fmtInt(s.totals.passthrough ?? 0)} />
            <StatCard className={KPI} label="Fell back" value={fmtInt(s.totals.fallback ?? 0)} />
            <StatCard
              className={KPI}
              label="Refused · errors"
              value={`${fmtInt(s.totals.refused ?? 0)} · ${fmtInt(s.totals.error ?? 0)}`}
              hint={errorReasonsHint}
              title={errorReasonsTitle}
            />
          </div>

          <RulesSection stats={s} />
          <TargetsSection targets={s.targets} nowMs={nowMs} />

          {Object.keys(s.by_reason).length > 0 && (
            <section aria-labelledby="act-reasons-h" className={SEC}>
              <div className={SEC_HEAD}>
                <h2 id="act-reasons-h" className={SEC_TITLE}>Fallback reasons</h2>
                <span className={SEC_NOTE}>why a rule went to Anthropic or refused</span>
              </div>
              <ul className={`${SEC_BODY} m-0 flex list-none flex-wrap gap-2 text-[12.5px]`}>
                {Object.entries(s.by_reason)
                  .sort((a, b) => b[1] - a[1])
                  .map(([reason, n]) => (
                    <li key={reason} className={pill("amber")}>
                      <code className="font-mono">{reason}</code>
                      <span className="tabular-nums">{fmtInt(n)}</span>
                    </li>
                  ))}
              </ul>
            </section>
          )}

          {decisionsSection}
        </>
      )}

      <Modal
        open={confirming}
        onClose={() => setConfirming(false)}
        title="Reset counters"
        initialFocusRef={cancelRef}
      >
        <p className="mb-4 text-sm">
          Reset all router counters, breakers and the recent-decisions list? Counting starts again
          with the next request.
        </p>
        {resetError && (
          <p role="alert" className="mb-3 text-sm text-vw-danger-fg">{resetError}</p>
        )}
        <div className="flex justify-end gap-2">
          <button ref={cancelRef} type="button" className={btn("default")} onClick={() => setConfirming(false)}>
            Cancel
          </button>
          <button type="button" className={btn("danger")} onClick={() => void reset()} disabled={resetting}>
            Reset
          </button>
        </div>
      </Modal>
    </div>
  );
}

// .kpi { border rule-soft/.6; radius 10px; padding 10px 12px }
const KPI = "rounded-[10px] border-vw-rule-soft/60 bg-chat-surface/55 px-3 py-2.5";

function Skeleton() {
  const bar = "rounded-[10px] bg-chat-surface-2/60 motion-safe:animate-pulse";
  return (
    <div aria-busy="true" className="space-y-5">
      <span className="sr-only">Loading…</span>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {[0, 1, 2, 3].map((i) => <div key={i} className={`h-[74px] ${bar}`} />)}
      </div>
      <div className={`h-40 ${bar}`} />
      <div className={`h-64 ${bar}`} />
    </div>
  );
}

function EmptyState({ enabled }: { enabled: boolean }) {
  return (
    <section aria-label="No activity" className={`${SEC} px-5 py-8 text-center`}>
      <p className="m-0 text-[15px] font-semibold">No requests yet.</p>
      <p className="mx-auto mt-1 max-w-prose text-[13.5px] text-chat-muted">
        Counters start with the first request after routing is on.
      </p>
      <p className="mt-4">
        <Link href="/router" className={enabled ? LINK : btn("primary")}>
          {enabled ? "Open the router page" : "Turn routing on"}
        </Link>
      </p>
    </section>
  );
}

function RulesSection({ stats: s }: { stats: RouterStatsOut }) {
  const breakerOf = new Map(s.targets.map((t) => [t.model_id, t.breaker]));
  const pt = s.passthrough;
  return (
    <section aria-labelledby="act-rules-h" className={SEC}>
      <div className={SEC_HEAD}>
        <h2 id="act-rules-h" className={SEC_TITLE}>By rule</h2>
        <span className={SEC_NOTE}>latency and time to first byte, p50 / p95</span>
      </div>
      <div className={`${SEC_BODY} ${TABLE_WRAP}`}>
        <table data-testid="router-stats-rules" className={TABLE}>
          <thead>
            <tr>
              <th scope="col" className={TH}>Rule</th>
              <th scope="col" className={TH_N}>Local</th>
              <th scope="col" className={TH_N}>Fell back</th>
              <th scope="col" className={TH_N}>Refused</th>
              <th scope="col" className={TH_N}>Latency</th>
              <th scope="col" className={TH_N}>TTFB</th>
              <th scope="col" className={TH}>Breaker</th>
            </tr>
          </thead>
          <tbody>
            {s.rules.length === 0 && (
              <tr className="max-[759px]:block">
                <td colSpan={7} className={TD_EMPTY}>No rules.</td>
              </tr>
            )}
            {s.rules.map((r) => {
              const breaker = r.target_model_id ? breakerOf.get(r.target_model_id) : undefined;
              return (
                <tr key={r.rule_id} className={TR}>
                  <td className={`${TD_HEAD} font-mono`}>
                    <b className="font-semibold">{r.pattern}</b>
                    {" → "}
                    {r.target_served_name ? (
                      <span>{r.target_served_name}</span>
                    ) : (
                      <span className="font-sans text-vw-danger-fg">model deleted</span>
                    )}
                  </td>
                  <td className={TD_N} data-label="Local">{fmtInt(r.local)}</td>
                  <td className={TD_N} data-label="Fell back">{fmtInt(r.fallback)}</td>
                  <td className={TD_N} data-label="Refused">{fmtInt(r.refused)}</td>
                  <td className={TD_N} data-label="Latency">{fmtMsPair(r.latency_ms.p50, r.latency_ms.p95)}</td>
                  <td className={TD_N} data-label="TTFB">{fmtMsPair(r.ttfb_ms.p50, r.ttfb_ms.p95)}</td>
                  <td className={TD} data-label="Breaker">{breaker ? <BreakerPill state={breaker} /> : "—"}</td>
                </tr>
              );
            })}
            {/* Pass-through: its numbers are requests / errors / tokens, so
                each cell names its unit instead of borrowing the headers. */}
            <tr data-testid="router-stats-passthrough" className={TR}>
              <td className={TD_HEAD}><b className="font-semibold">Pass-through to Anthropic</b></td>
              <td className={TD_N}>
                <span className="tabular-nums">{fmtInt(pt.requests)}</span>{" "}
                <span className={UNIT}>requests</span>
              </td>
              <td className={TD_N}>
                <span className="tabular-nums">{fmtInt(pt.errors)}</span>{" "}
                <span className={UNIT}>{pt.errors === 1 ? "error" : "errors"}</span>
              </td>
              <td className={TD_N}>
                <span className="tabular-nums">{fmtInt(pt.input_tokens)}</span>
                {" / "}
                <span className="tabular-nums">{fmtInt(pt.output_tokens)}</span>{" "}
                <span className={UNIT}>tokens in / out</span>
              </td>
              <td className={TD_N} data-label="Latency">{fmtMsPair(pt.latency_ms.p50, pt.latency_ms.p95)}</td>
              <td className={TD_N} data-label="TTFB">{fmtMsPair(pt.ttfb_ms.p50, pt.ttfb_ms.p95)}</td>
              <td className={`${TD} max-[759px]:hidden`} />
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}

const UNIT = "text-[11.5px] text-chat-dim";

function TargetsSection({ targets, nowMs }: { targets: TargetStats[]; nowMs: number }) {
  return (
    <section aria-labelledby="act-targets-h" className={SEC}>
      <div className={SEC_HEAD}>
        <h2 id="act-targets-h" className={SEC_TITLE}>Local models and breakers</h2>
        <span className={SEC_NOTE}>one breaker per local model; when open, one probe request decides</span>
      </div>
      <div className={`${SEC_BODY} ${TABLE_WRAP}`}>
        <table data-testid="router-stats-targets" className={TABLE}>
          <thead>
            <tr>
              <th scope="col" className={TH}>Model</th>
              <th scope="col" className={TH}>Breaker</th>
              <th scope="col" className={TH_N}>Failures in a row</th>
              <th scope="col" className={TH_N}>Failures</th>
              <th scope="col" className={TH}>Open until</th>
              <th scope="col" className={TH}>Last reason</th>
            </tr>
          </thead>
          <tbody>
            {targets.length === 0 ? (
              <tr className="max-[759px]:block">
                <td colSpan={6} className={TD_EMPTY}>No targets.</td>
              </tr>
            ) : targets.map((t) => (
              <tr key={t.model_id} className={TR}>
                <td className={`${TD_HEAD} font-mono`}>
                  {t.served_name ?? <span className="font-sans text-vw-danger-fg">model deleted</span>}
                </td>
                <td className={TD} data-label="Breaker"><BreakerPill state={t.breaker} /></td>
                <td className={TD_N} data-label="In a row">{fmtInt(t.consecutive_failures)}</td>
                <td className={TD_N} data-label="Failures">{fmtInt(t.failures)}</td>
                <td className={TD} data-label="Open until">{openUntil(t.open_until, nowMs)}</td>
                <td className={TD} data-label="Last reason">
                  {t.last_reason ? <code className="font-mono">{t.last_reason}</code> : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
