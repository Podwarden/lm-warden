"use client";

// Everything the Connect page reads (plan §5.1). The catalogue polls every
// 15 s so a model that finishes loading appears; the token list and the
// public URL use the router page's keys, so the SWR cache is shared.

import { useCallback } from "react";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { ROUTER_KEYS } from "@/components/router/use-router-data";
import type { ConnectClientsOut } from "@/lib/connect";
import type { TokenListPage } from "@/lib/router";

export const CONNECT_KEYS = {
  clients: "/api/connect/clients",
  tokens: ROUTER_KEYS.tokens,
  runtime: "/api/settings/runtime",
} as const;

// Module level: SWR restarts its timer when the option changes identity.
export const CONNECT_POLL_MS = 15_000;
const POLL = { refreshInterval: CONNECT_POLL_MS } as const;

export interface ConnectData {
  clients: ConnectClientsOut | undefined;
  tokens: TokenListPage | undefined;
  /** null = unset, undefined = not loaded yet. */
  publicUrl: string | null | undefined;
  errors: { clients: boolean; tokens: boolean };
  refreshClients: () => Promise<unknown>;
  refreshTokens: () => Promise<unknown>;
}

export function useConnectData(): ConnectData {
  const clients = useSWR<ConnectClientsOut>(CONNECT_KEYS.clients, authFetchJSON, POLL);
  const tokens = useSWR<TokenListPage>(CONNECT_KEYS.tokens, authFetchJSON);
  const runtime = useSWR<{ public_url?: string | null }>(CONNECT_KEYS.runtime, authFetchJSON);

  const mClients = clients.mutate;
  const mTokens = tokens.mutate;
  const refreshClients = useCallback(() => mClients(), [mClients]);
  const refreshTokens = useCallback(() => mTokens(), [mTokens]);

  return {
    clients: clients.data,
    tokens: tokens.data,
    publicUrl: runtime.data ? (runtime.data.public_url ?? null) : runtime.error ? null : undefined,
    // SWR keeps the last good data next to an error: only a missing payload is a failure.
    errors: { clients: !!clients.error && !clients.data, tokens: !!tokens.error && !tokens.data },
    refreshClients,
    refreshTokens,
  };
}
