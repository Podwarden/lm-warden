"use client";

// The key's prompt-cache use through the OWN lens (spec §5.2): matched only
// against this key's earlier prompts, so it can never reveal another key's.

import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { CachePanel } from "@/components/stats/cache-panel";
import type { CacheResponse } from "@/lib/cache-obs";

export function CacheCard({
  tokenId,
  range,
  unsupportedSelection = false,
}: {
  tokenId: string;
  range: "1h" | "24h" | "7d";
  /** True when the page's selection (6h, custom) has no cache equivalent. */
  unsupportedSelection?: boolean;
}) {
  const { data, error } = useSWR<CacheResponse>(
    `/api/tokens/${encodeURIComponent(tokenId)}/cache?range=${range}`,
    authFetchJSON,
    { keepPreviousData: true, revalidateOnFocus: false, shouldRetryOnError: false },
  );
  return (
    <section className="rounded-lg border border-chat-rule p-4" aria-label="Prompt cache" data-testid="token-cache-card">
      <h3 className="mb-2 text-sm text-chat-fg">Prompt cache · last {range}</h3>
      {unsupportedSelection && (
        <p className="mb-2 text-xs text-chat-dim" data-testid="cache-range-note">
          showing the last 24h; this card supports 1h, 24h and 7d
        </p>
      )}
      {data ? (
        <CachePanel data={data} lens="own" />
      ) : error ? (
        <p className="text-xs text-chat-dim" data-testid="cache-error">Cache data unavailable.</p>
      ) : null}
    </section>
  );
}
