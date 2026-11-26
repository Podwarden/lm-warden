"use client";

// /prompt-lens — Prompt Lens for the admin: every key's requests, live and
// recent, full screen (owner request 2026-11-23; it was a dock along the
// bottom of /stats and the key details page).
//
// ?req=<request id> opens that request (the eye on a Stats Requests row);
// ?key=<id>[,<id>...] presets the Key filter (the key details page: the key
// and its earlier keys) and names them. Shown while the platform switch is on
// (GET /api/admin/prompt-lens/status), and also, for ?key=, while it is off
// if that key keeps captures (they stay readable on disk). Otherwise the page
// says it is off and where to turn it on.

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";
import useSWR from "swr";

import { FullScreen, useLensDeepLink, useWritePick } from "@/components/prompt-lens/prompt-lens-page";
import { PromptLensView, usePromptLensStatus } from "@/components/prompt-lens/prompt-lens-view";
import { authFetchJSON } from "@/lib/auth-fetch";
import type { TokenDetail } from "@/lib/token-series";

function AdminPromptLens() {
  const params = useSearchParams();
  const { select, keys, view, session } = useLensDeepLink(params);
  const status = usePromptLensStatus();
  const onPick = useWritePick();
  // The key the link came from is the last of the set (its earlier keys first).
  const self = keys ? keys[keys.length - 1] : null;
  const token = useSWR<TokenDetail>(self ? `/api/tokens/${encodeURIComponent(self)}` : null, authFetchJSON, {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
  });
  const keyNames = useMemo(() => {
    const out: Record<string, string> = {};
    for (const e of token.data?.lineage ?? []) out[e.id] = e.name;
    if (token.data) out[token.data.id] = token.data.name;
    return out;
  }, [token.data]);

  const kept = !!keys && token.data?.keep_captures === true;
  const available = status.enabled || kept;
  if (status.loading || (!status.enabled && self && token.isLoading)) {
    return <p className="text-sm text-chat-dim">Loading Prompt Lens…</p>;
  }
  if (!available) {
    return (
      <div data-testid="prompt-lens-off" className="max-w-[640px] rounded-lg border border-chat-rule bg-chat-surface p-4 text-sm text-chat-muted">
        <h1 className="mb-2 text-base font-semibold text-chat-fg">Prompt Lens is off</h1>
        <p>
          Prompt Lens shows what clients send and what the model answers, as it happens. Turn it on in{" "}
          <Link href="/settings" className="text-chat-accent underline">
            Settings, General, Privacy
          </Link>
          .
        </p>
      </div>
    );
  }
  return (
    <FullScreen>
      <PromptLensView
        key={keys ? keys.join(",") : "all"}
        scope="admin"
        pageKey="admin-stats"
        presetKeys={keys}
        keyNames={keyNames}
        select={select}
        onPick={onPick}
        view={view}
        session={session}
      />
    </FullScreen>
  );
}

export default function Page() {
  return (
    <Suspense fallback={null}>
      <AdminPromptLens />
    </Suspense>
  );
}
