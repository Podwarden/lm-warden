"use client";

// Claude Code router overview (router page redesign, plan §2–§8). The page
// opens on the router's state — one strip computed by deriveRouterState — and
// then the next action for that state: the setup checklist on first run, the
// routing map + traffic day to day, the incident evidence when something is
// failing. Editing happens here; /router/stats is the Activity drill-down.
//
// Layout: two columns (work area + rail) from 1100 px; below that one column
// whose order is set with CSS `order` on flattened (display: contents)
// columns: strip → checklist → map → traffic → latest → connect → settings →
// explainer.

import { useCallback, useEffect, useState } from "react";
import { authFetch } from "@/lib/auth-fetch";
import { errorDetail } from "@/components/tokens/use-token-actions";
import { CreateTokenDialog } from "@/components/tokens/create-token-dialog";
import { useRouterOverview } from "@/components/router/use-router-data";
import { RouterHeader, useRoutingSwitch } from "@/components/router/router-header";
import { RouterStatusStrip } from "@/components/router/router-status-strip";
import { RouterSetup, RoutingExplainer } from "@/components/router/router-setup";
import { TrafficPanel } from "@/components/router/traffic-panel";
import { LatestRequests } from "@/components/router/latest-requests";
import { ConnectClaudeCode } from "@/components/router/connect-claude-code";
import { RouterSettingsCard } from "@/components/router/router-settings-card";
import { RuleDialog } from "@/components/router/rule-dialog";
import { RoutingMap } from "@/components/router/routing-map";
import { btn, FOCUS, SEC } from "@/components/router/styles";
import { cn } from "@/lib/utils";
import {
  deriveRouterState,
  setupSteps,
  showSetupChecklist,
  type RuleOut,
  type StripAction,
} from "@/lib/router";

// Re-render the "ago" copy between polls; a stable number (frontend-gotchas).
const CLOCK_MS = 5000;
// A failed action's alert clears itself after this long (or on Dismiss, or
// on the next action that succeeds).
const ACTION_ERROR_MS = 30_000;
const SLOW_MS = 15_000;

function useNow(): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), CLOCK_MS);
    return () => clearInterval(t);
  }, []);
  return now;
}

function useSlow(loading: boolean): boolean {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    if (!loading) {
      setSlow(false);
      return;
    }
    const t = setTimeout(() => setSlow(true), SLOW_MS);
    return () => clearTimeout(t);
  }, [loading]);
  return slow;
}

// Order in the one-column layout (< 1100 px); none in two columns.
const O = {
  setup: "order-1 min-[1100px]:order-none",
  map: "order-2 min-[1100px]:order-none",
  traffic: "order-3 min-[1100px]:order-none",
  latest: "order-4 min-[1100px]:order-none",
  connect: "order-5 min-[1100px]:order-none",
  settings: "order-6 min-[1100px]:order-none",
  explain: "order-7 min-[1100px]:order-none",
} as const;

function LoadingSkeleton({ slow }: { slow: boolean }) {
  return (
    <div className={cn(SEC, O.map)} aria-busy="true">
      <span className="sr-only">Loading the router…</span>
      <div className="h-12 px-4 py-3.5">
        <div aria-hidden="true" className="h-4 w-56 animate-pulse rounded bg-chat-surface-2/70 motion-reduce:animate-none" />
      </div>
      {[0, 1, 2].map((i) => (
        <div key={i} aria-hidden="true" className="flex items-center gap-3 border-t border-vw-rule-soft/45 px-4 py-3">
          <div className="h-6 w-6 animate-pulse rounded-md bg-chat-surface-2/70 motion-reduce:animate-none" />
          <div className="h-4 flex-1 animate-pulse rounded bg-chat-surface-2/70 motion-reduce:animate-none" />
          <div className="h-4 w-24 animate-pulse rounded bg-chat-surface-2/70 motion-reduce:animate-none" />
        </div>
      ))}
      {slow && <p className="m-0 px-4 pb-3 text-[13px] text-chat-muted">Taking longer than expected…</p>}
    </div>
  );
}

export default function RouterPage() {
  const ov = useRouterOverview();
  const now = useNow();
  const [actionError, setActionError] = useState<string | null>(null);
  useEffect(() => {
    if (!actionError) return;
    const t = setTimeout(() => setActionError(null), ACTION_ERROR_MS);
    return () => clearTimeout(t);
  }, [actionError]);
  const [acting, setActing] = useState(false);
  const [ruleDialog, setRuleDialog] = useState<{ rule: RuleOut | null; pattern?: string } | null>(null);
  const [tokenDialog, setTokenDialog] = useState(false);
  // The plaintext of a key created here (and whether it may relay):
  // component state only, never stored or logged; it goes on unmount.
  const [newKey, setNewKey] = useState<{ plaintext: string; relay: boolean } | null>(null);

  const { mutate, refresh } = ov;
  const sw = useRoutingSwitch({ decisions: ov.decisions, onSaved: mutate.settings, onError: setActionError });

  const state = deriveRouterState({
    settings: ov.settings,
    rules: ov.rules,
    stats: ov.stats,
    statsError: ov.errors.stats || ov.errors.decisions,
    decisions: ov.decisions,
    relay: ov.relay,
    now,
  });
  const steps = setupSteps({
    settings: ov.settings,
    rules: ov.rules,
    stats: ov.stats,
    decisions: ov.decisions,
    relay: ov.relay,
  });

  const loadFailed = (!ov.settings && ov.errors.settings) || (!ov.rules && ov.errors.rules);
  const loading = !loadFailed && (!ov.settings || !ov.rules);
  const slow = useSlow(loading);
  // Durable facts only (review #1): a restart, a counter reset or the kill
  // switch must never swap the traffic evidence for the checklist.
  const showSetup = showSetupChecklist({
    settings: ov.settings,
    rules: ov.rules,
    stats: ov.stats,
    decisions: ov.decisions,
  });
  const hasRules = (ov.rules?.length ?? 0) > 0;
  const seen = (ov.stats?.since ?? null) !== null || (ov.decisions?.length ?? 0) > 0;

  const pauseRule = useCallback(
    async (ruleId: string) => {
      setActing(true);
      setActionError(null);
      try {
        const r = await authFetch(`/api/router/rules/${encodeURIComponent(ruleId)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: false }),
        });
        if (!r.ok) {
          setActionError(await errorDetail(r, `Couldn't pause the rule (HTTP ${r.status})`));
          return;
        }
        await Promise.all([mutate.rules(), mutate.stats()]);
      } catch (err) {
        setActionError(err instanceof Error ? err.message : "Network error");
      } finally {
        setActing(false);
      }
    },
    [mutate],
  );

  function onAction(a: StripAction) {
    switch (a.kind) {
      case "retry":
        void refresh();
        return;
      case "turn_on":
        sw.request(true);
        return;
      case "pause_rule":
        void pauseRule(a.ruleId);
        return;
      case "edit_rule":
        setRuleDialog({ rule: ov.rules?.find((r) => r.id === a.ruleId) ?? null });
        return;
      case "create_relay_key":
        setTokenDialog(true);
        return;
      default:
        // Links render as anchors; nothing to do.
        return;
    }
  }

  const addRule = useCallback((pattern?: string) => setRuleDialog({ rule: null, pattern }), []);

  // A dialog opened from a checklist step returns focus to its opener — but
  // finishing the step removes that button once the data re-fetches, which
  // drops focus to <body>. Land on the checklist (or the map) heading instead.
  const rescueFocus = useCallback(() => {
    requestAnimationFrame(() => {
      const a = document.activeElement;
      if (a && a !== document.body && a.isConnected) return;
      (document.getElementById("router-setup-h") ?? document.getElementById("router-map-h"))?.focus();
    });
  }, []);

  return (
    <div className="mx-auto w-full max-w-[1232px] pb-12">
      <RouterHeader
        settings={ov.settings}
        rules={ov.rules}
        sw={sw}
        showActivity={!!ov.settings && state.kind !== "setup"}
      />
      <RouterStatusStrip
        strip={state.strip}
        error={actionError}
        onDismissError={() => setActionError(null)}
        onAction={onAction}
        busy={acting}
      />

      <div className="mt-6 flex flex-col gap-6 min-[1100px]:grid min-[1100px]:grid-cols-[minmax(0,1fr)_340px] min-[1100px]:items-start">
        <div className="contents min-[1100px]:flex min-[1100px]:min-w-0 min-[1100px]:flex-col min-[1100px]:gap-6">
          {loadFailed ? (
            <div role="alert" data-testid="router-load-error" className={cn(SEC, "px-4 py-3.5 text-[13.5px]", O.map)}>
              <p className="m-0">
                <strong className="font-semibold text-vw-danger-fg">Couldn&apos;t load the router&apos;s rules or settings.</strong>{" "}
                <span className="text-chat-muted">Nothing was changed.</span>
              </p>
              <button type="button" className={btn("default", `mt-2 ${FOCUS}`)} onClick={() => void refresh()}>
                Retry
              </button>
            </div>
          ) : loading || !ov.settings || !ov.rules ? (
            <LoadingSkeleton slow={slow} />
          ) : (
            <>
              {showSetup && (
                <RouterSetup
                  className={O.setup}
                  steps={steps}
                  decisions={ov.decisions}
                  onAddRule={addRule}
                  onCreateRelayKey={() => setTokenDialog(true)}
                  onTurnOn={() => sw.request(true)}
                  turning={sw.pending === true}
                />
              )}
              <div className={cn("min-w-0", O.map)}>
                <RoutingMap
                  rules={ov.rules}
                  settings={ov.settings}
                  stats={ov.stats}
                  onChange={() => void Promise.all([mutate.rules(), mutate.stats()])}
                  onSettingsChange={() => void mutate.settings()}
                  onAddRule={addRule}
                  now={now}
                />
              </div>
              <ConnectClaudeCode
                className={O.connect}
                settings={ov.settings}
                rules={ov.rules}
                decisions={ov.decisions}
                relay={ov.relay}
                plaintext={newKey?.plaintext ?? null}
                plaintextRelay={newKey?.relay ?? true}
                defaultOpen={ov.stats || ov.decisions ? !seen : showSetup}
                onCreateRelayKey={() => setTokenDialog(true)}
              />
              <RouterSettingsCard
                className={O.settings}
                settings={ov.settings}
                onSaved={() => void mutate.settings()}
              />
            </>
          )}
        </div>

        <div className="contents min-[1100px]:flex min-[1100px]:min-w-0 min-[1100px]:flex-col min-[1100px]:gap-4">
          {/* Once a rule exists the traffic evidence always renders — on,
              off (the kill switch mid-incident) or after a restart. The
              explainer joins it only while setting up. */}
          {ov.settings && ov.rules && hasRules && (
            <>
              <TrafficPanel className={O.traffic} stats={ov.stats} alarm={state.kind === "danger"} />
              {!ov.errors.decisions && <LatestRequests className={O.latest} decisions={ov.decisions} />}
            </>
          )}
          {(showSetup || (ov.settings && ov.rules && !hasRules)) && (
            <RoutingExplainer className={O.explain} headerName={ov.settings?.header_name ?? "X-LMWarden-Key"} />
          )}
        </div>
      </div>

      <RuleDialog
        open={ruleDialog !== null}
        rule={ruleDialog?.rule ?? null}
        initialPattern={ruleDialog?.pattern}
        onClose={() => setRuleDialog(null)}
        onSaved={() => {
          setRuleDialog(null);
          setActionError(null);
          void Promise.all([mutate.rules(), mutate.stats()]).then(rescueFocus);
        }}
      />
      <CreateTokenDialog
        open={tokenDialog}
        initialName="claude-code"
        initialRelay
        onCreated={(p, info) => {
          setNewKey({ plaintext: p, relay: info?.anthropicRelay ?? true });
          setActionError(null);
          void mutate.tokens();
        }}
        onClose={() => {
          setTokenDialog(false);
          void Promise.resolve(mutate.tokens()).then(rescueFocus);
        }}
      />
    </div>
  );
}
