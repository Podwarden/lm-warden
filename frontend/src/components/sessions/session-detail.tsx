"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";

import { SessionsShell } from "@/components/sessions/sessions-shell";
import { sessionHref } from "@/lib/routes";
import {
  KeyAuthError,
  SessionsHttpError,
  sessionsFetcher,
  sessionsKey,
  sessionUrl,
  useSessionsMode,
} from "@/lib/sessions/client";
import { SESSIONS_LOGIN } from "@/lib/sessions/key-token";
import type { Selection } from "@/lib/sessions/layout";
import { S } from "@/lib/sessions/strings";
import type { SessionResponse } from "@/lib/sessions/types";

import { ConversationTree } from "./conversation-tree";
import { DeadWeightPanel } from "./dead-weight-panel";
import { EngineComparison } from "./engine-comparison";
import { PausesPanel } from "./pauses-panel";
import { SessionHeader } from "./session-header";
import { Timeline } from "./timeline";
import { ToolsPanel } from "./tools-panel";
import { TurnTable } from "./turn-table";

const LIVE_S = 120;

type Sel = { conversationId: string | null | undefined; requestId: string | null };

function Detail({ id }: { id: string }) {
  const mode = useSessionsMode();
  const router = useRouter();
  const fetcher = sessionsFetcher<SessionResponse>(mode);
  const [live, setLive] = useState(false);
  const { data, error } = useSWR(sessionsKey(mode, sessionUrl(id)), ([, , u]) => fetcher(u), {
    // A quiet session still refreshes, slowly: it may resume (verified finding #60).
    refreshInterval: live ? 5_000 : 30_000,
  });
  const [sel, setSel] = useState<Sel>({ conversationId: undefined, requestId: null });

  useEffect(() => {
    if (data && data.session_id !== id) router.replace(sessionHref(data.session_id));
  }, [data, id, router]);
  useEffect(() => {
    if (error instanceof KeyAuthError) router.replace(SESSIONS_LOGIN);
  }, [error, router]);
  useEffect(() => {
    const last = data?.totals.last_at;
    setLive(last != null && Date.now() / 1000 - last <= LIVE_S);
  }, [data]);

  const turns = useMemo(() => {
    if (!data) return [];
    return sel.conversationId === undefined
      ? data.turns
      : data.turns.filter((t) => t.conversation_id === sel.conversationId);
  }, [data, sel.conversationId]);

  if (error) {
    if (error instanceof KeyAuthError) return null;
    const msg =
      error instanceof SessionsHttpError && error.status === 404
        ? error.detail === "not recorded yet"
          ? S.page.notRecordedYet
          : S.page.notFound
        : S.page.unavailable;
    return <p className="p-4 text-sm text-chat-muted">{msg}</p>;
  }
  if (!data) return <p className="p-4 text-sm text-chat-muted">{S.page.loading}</p>;

  const conv = sel.conversationId === undefined ? null : data.conversations.find((c) => c.id === sel.conversationId) ?? null;
  const selection: Selection = { conversationId: sel.conversationId ?? null, requestId: sel.requestId };

  return (
    <div className="space-y-4">
      <div className="space-y-5 rounded-[10px] border border-chat-rule bg-chat-surface p-4 sm:p-5">
        <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
          <SessionHeader s={data} />
          {data.dead_weight && data.anatomy && (
            <DeadWeightPanel dw={data.dead_weight} anatomy={data.anatomy} tools={data.tools} />
          )}
        </div>
        {data.engine_replay && (
          <div className="border-t border-chat-rule pt-4">
            <EngineComparison r={data.engine_replay} />
          </div>
        )}
      </div>
      {data.pauses && <PausesPanel p={data.pauses} />}
      <Timeline
        s={data}
        selection={selection}
        onBlock={(requestId, conversationId) => setSel({ conversationId, requestId })}
        onLane={(conversationId) => setSel({ conversationId, requestId: null })}
      />
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="min-w-0 lg:col-span-1">
          <ConversationTree
            s={data}
            selected={sel.conversationId}
            onSelect={(conversationId) => setSel({ conversationId, requestId: null })}
          />
        </div>
        <div className="min-w-0 space-y-4 lg:col-span-2">
          <TurnTable
            turns={turns}
            title={conv ? `${S.turn.title} · ${conv.label}` : S.turn.title}
            selection={selection}
            onSelect={(t) => setSel((s) => ({ conversationId: s.conversationId, requestId: t.request_id }))}
          />
          <ToolsPanel tools={conv ? conv.tools : data.tools} />
        </div>
      </div>
    </div>
  );
}

export function SessionDetail({ id }: { id: string }) {
  const mode = useSessionsMode();
  return mode === "key" ? (
    <SessionsShell>
      <Detail id={id} />
    </SessionsShell>
  ) : (
    <Detail id={id} />
  );
}
