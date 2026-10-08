"use client";

// Everything the router overview reads (plan §4.7, §9), from public endpoints
// only. Rules and settings load once and are re-fetched after edits; stats and
// the latest 20 decisions poll every 5 s (SWR pauses hidden tabs by default);
// the token list answers "may any key relay to Anthropic?".

import { useCallback } from "react";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import {
  relayFromTokens,
  type DecisionOut,
  type RelayInfo,
  type RouterSettingsOut,
  type RouterStatsOut,
  type RuleOut,
  type RulesOut,
  type TokenListPage,
} from "@/lib/router";

export const ROUTER_KEYS = {
  rules: "/api/router/rules",
  settings: "/api/router/settings",
  stats: "/api/router/stats",
  decisions: "/api/router/decisions?limit=20",
  tokens: "/api/tokens?limit=500&sort=created",
} as const;

// A stable number: SWR restarts its timer when the option changes identity
// (frontend-gotchas: function-valued refreshInterval must be module level).
export const OVERVIEW_POLL_MS = 5000;
const POLL = { refreshInterval: OVERVIEW_POLL_MS } as const;

export interface RouterOverview {
  rules: RuleOut[] | undefined;
  settings: RouterSettingsOut | undefined;
  stats: RouterStatsOut | undefined;
  /** Newest first. */
  decisions: DecisionOut[] | undefined;
  relay: RelayInfo;
  /** SWR keeps the last good data next to an error: both can be set. */
  errors: {
    rules: boolean;
    settings: boolean;
    stats: boolean;
    decisions: boolean;
    tokens: boolean;
  };
  mutate: {
    rules: () => Promise<unknown>;
    settings: () => Promise<unknown>;
    stats: () => Promise<unknown>;
    decisions: () => Promise<unknown>;
    tokens: () => Promise<unknown>;
  };
  /** Re-fetch every source (Retry, after a failed action). */
  refresh: () => Promise<void>;
}

export function useRouterOverview(): RouterOverview {
  const rules = useSWR<RulesOut>(ROUTER_KEYS.rules, authFetchJSON);
  const settings = useSWR<RouterSettingsOut>(ROUTER_KEYS.settings, authFetchJSON);
  const stats = useSWR<RouterStatsOut>(ROUTER_KEYS.stats, authFetchJSON, POLL);
  const decisions = useSWR<{ decisions: DecisionOut[] }>(ROUTER_KEYS.decisions, authFetchJSON, POLL);
  const tokens = useSWR<TokenListPage>(ROUTER_KEYS.tokens, authFetchJSON);

  const mRules = rules.mutate;
  const mSettings = settings.mutate;
  const mStats = stats.mutate;
  const mDecisions = decisions.mutate;
  const mTokens = tokens.mutate;
  const refresh = useCallback(async () => {
    await Promise.all([mRules(), mSettings(), mStats(), mDecisions(), mTokens()]);
  }, [mRules, mSettings, mStats, mDecisions, mTokens]);

  return {
    rules: rules.data?.rules,
    settings: settings.data,
    stats: stats.data,
    decisions: decisions.data?.decisions,
    relay: relayFromTokens(tokens.data, !!tokens.error),
    errors: {
      rules: !!rules.error,
      settings: !!settings.error,
      stats: !!stats.error,
      decisions: !!decisions.error,
      tokens: !!tokens.error,
    },
    mutate: {
      rules: () => mRules(),
      settings: () => mSettings(),
      stats: () => mStats(),
      decisions: () => mDecisions(),
      tokens: () => mTokens(),
    },
    refresh,
  };
}
