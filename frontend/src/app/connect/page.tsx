"use client";

// Connect a client (plan docs/superpowers/plans/2026-10-04-connect-clients.md
// §3, §5). Pick the tool, pick or create a key, pick a model, copy, and send a
// test request over the exact path the tool will use. Everything renders from
// GET /api/connect/clients; the page only fills the templates with this tab's
// picks. A key created here lives in component state only (never the URL,
// SWR, storage or logs) and is gone on unmount or "Forget key".

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { CreateTokenDialog } from "@/components/tokens/create-token-dialog";
import { useConnectData } from "@/components/connect/use-connect-data";
import { ToolList } from "@/components/connect/tool-list";
import { PickerBar, CREATE_KEY, NEW_KEY } from "@/components/connect/picker-bar";
import { ClientPanel } from "@/components/connect/client-panel";
import { FOCUS, SEC, SMALL_BTN, STRIP_ICON, STRIP_TEXT, STRIP_TITLE, strip as stripClass } from "@/components/connect/styles";
import { getPublicBaseUrl } from "@/lib/public-url";
import { cn } from "@/lib/utils";
import {
  deriveConnectState,
  filesFor,
  hardMinContext,
  isLiveKey,
  isLoaded,
  isRouterMode,
  keyComment,
  parseConnectParams,
  renderTemplate,
  requirementChecks,
  verifyModelFor,
  type ConnectFile,
  type ConnectStrip,
  type TemplateVars,
} from "@/lib/connect";

export default function ConnectPage() {
  // useSearchParams opts the tree into client-side rendering; Suspense is
  // required around it (as /router/stats and /tokens).
  return (
    <Suspense fallback={null}>
      <ConnectView />
    </Suspense>
  );
}

const HEAD_LINK = cn(
  "whitespace-nowrap rounded-sm border-b border-vw-rule-soft text-[13px] text-chat-muted hover:text-chat-fg",
  FOCUS,
);

function hrefFor(tool: string, mode?: string | null): string {
  const q = new URLSearchParams({ tool });
  if (mode) q.set("mode", mode);
  return `/connect?${q.toString()}`;
}

function StripIcon({ tone }: { tone: ConnectStrip["tone"] }) {
  const cls = STRIP_ICON[tone];
  if (tone === "neutral") {
    return (
      <svg className={cls} viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
        <circle cx="9" cy="9" r="7" />
        <path d="M9 8v4.5M9 5.6v.1" />
      </svg>
    );
  }
  return (
    <svg className={cls} viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M9 2 1.8 15h14.4L9 2Z" />
      <path d="M9 7v4M9 13.2v.1" />
    </svg>
  );
}

function Strip({ s, onRetry, onRetryTokens, onCreateKey }: {
  s: ConnectStrip | null;
  onRetry: () => void;
  onRetryTokens: () => void;
  onCreateKey: () => void;
}) {
  return (
    <div data-testid="connect-strip">
      <div role={s?.role ?? "status"} className={s ? cn(stripClass(s.tone), "mt-4") : undefined}>
        {s && (
          <>
            <StripIcon tone={s.tone} />
            <div className="min-w-0 flex-1">
              <p className={STRIP_TEXT}>
                <strong className={STRIP_TITLE[s.tone]}>{s.text}</strong>
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                {s.action === "retry" && (
                  <button type="button" className={SMALL_BTN} onClick={onRetry}>
                    Retry
                  </button>
                )}
                {s.action === "retry_tokens" && (
                  <button type="button" className={SMALL_BTN} onClick={onRetryTokens}>
                    Retry
                  </button>
                )}
                {s.action === "models" && (
                  <Link href="/models" className={SMALL_BTN}>
                    Models
                  </Link>
                )}
                {s.action === "create_key" && (
                  <button type="button" className={SMALL_BTN} onClick={onCreateKey}>
                    Create a key
                  </button>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="mt-4 flex flex-col gap-6 min-[1100px]:grid min-[1100px]:grid-cols-[280px_minmax(0,1fr)]" aria-busy="true">
      <span className="sr-only">Loading the client catalogue…</span>
      <div aria-hidden="true" className={cn(SEC, "grid gap-2 p-4")}>
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="h-5 animate-pulse rounded bg-chat-surface-2/70 motion-reduce:animate-none" />
        ))}
      </div>
      <div aria-hidden="true" className={cn(SEC, "h-80 animate-pulse motion-reduce:animate-none")} />
    </div>
  );
}

function ConnectView() {
  const nav = useRouter();
  const params = useSearchParams();
  const data = useConnectData();
  const out = data.clients;
  const clients = useMemo(() => out?.clients ?? [], [out]);

  // Picks. The URL seeds tool/mode (?tool=&mode=); clicks mirror it with replace.
  const [toolPick, setToolPick] = useState<string | null>(null);
  const [modePick, setModePick] = useState<string | null>(null);
  const [modelPick, setModelPick] = useState<string | null>(null);
  const [keyValue, setKeyValue] = useState("");
  // Secrets: page state only.
  const [newKey, setNewKey] = useState<{ plaintext: string; relay: boolean } | null>(null);
  const [pasted, setPasted] = useState("");
  const [dialog, setDialog] = useState(false);
  const keySelectRef = useRef<HTMLSelectElement | null>(null);

  const search = params.toString();
  const parsed = useMemo(() => parseConnectParams(new URLSearchParams(search), clients), [search, clients]);

  // An unknown ?tool= / ?mode= is corrected once, without a history step.
  const corrected = useRef(false);
  useEffect(() => {
    if (corrected.current || !parsed.corrected || !parsed.tool) return;
    corrected.current = true;
    nav.replace(hrefFor(parsed.tool, parsed.mode), { scroll: false });
  }, [parsed, nav]);

  const client = clients.find((c) => c.id === (toolPick ?? parsed.tool)) ?? clients[0] ?? null;
  const wantMode = toolPick ? modePick : (modePick ?? parsed.mode);
  const mode = client ? (client.modes.find((m) => m.id === wantMode) ?? client.modes[0] ?? null) : null;

  const selectTool = useCallback(
    (id: string) => {
      setToolPick(id);
      setModePick(null);
      nav.replace(hrefFor(id), { scroll: false });
    },
    [nav],
  );
  const selectMode = useCallback(
    (id: string) => {
      if (!client) return;
      setToolPick(client.id);
      setModePick(id);
      nav.replace(hrefFor(client.id, id), { scroll: false });
    },
    [client, nav],
  );

  // Model: the operator's pick while it is still loaded, else the server's
  // default (the first loaded model).
  const models = out?.models ?? [];
  const loaded = models.filter(isLoaded);
  const model =
    loaded.find((m) => m.served_name === modelPick) ??
    loaded.find((m) => m.served_name === out?.selected_model) ??
    loaded[0] ??
    null;

  const tokens = data.tokens;
  const pickedToken = keyValue && keyValue !== NEW_KEY ? tokens?.items.find((t) => t.id === keyValue && isLiveKey(t)) : undefined;

  const router = out?.router ?? null;
  const routerMode = !!mode && isRouterMode(mode);
  const publicUrl = data.publicUrl;
  const origin = getPublicBaseUrl(publicUrl);
  const header = out?.header_name ?? "X-LMWarden-Key";

  const vars: TemplateVars = {
    origin,
    model: model?.served_name ?? null,
    header,
    context: model?.context_window ?? null,
    verify_model: client && mode ? verifyModelFor(client, mode, router, model?.served_name ?? null) : null,
  };

  const renderFile = (f: ConnectFile): string => {
    const text = renderTemplate(f.template, { ...vars, key: newKey?.plaintext ?? null });
    if (newKey || !pickedToken) return text;
    const c = keyComment(f.language, pickedToken.name, pickedToken.preview);
    return c ? `${c}\n${text}` : text;
  };

  const state = deriveConnectState({ clients: out, tokens, errors: data.errors });

  const openCreate = useCallback(() => setDialog(true), []);
  const onKeyChange = useCallback((v: string) => {
    if (v === CREATE_KEY) {
      setDialog(true);
      return;
    }
    setKeyValue(v);
    setPasted("");
  }, []);
  const forget = useCallback(() => {
    setNewKey(null);
    setPasted("");
    setKeyValue("");
  }, []);

  const files = mode ? filesFor(mode, model) : [];
  const checks =
    client && mode
      ? requirementChecks(client, mode, {
          model,
          key: newKey ? { anthropic_relay: newKey.relay } : (pickedToken ?? null),
          router,
          models,
        })
      : [];

  return (
    <div className="mx-auto w-full max-w-[1232px] pb-12">
      <div className="flex flex-wrap items-start justify-between gap-4 max-[759px]:flex-col max-[759px]:gap-2">
        <div className="min-w-0">
          <h1 className="m-0 text-2xl font-semibold leading-tight tracking-[-0.01em]">Connect a client</h1>
          <p className="mb-0 mt-1.5 max-w-[62ch] text-[13.5px] text-chat-muted">
            Point a coding tool or SDK at this warden: pick the tool, a key and a model, copy the setup, then
            send a test request to prove it works.
          </p>
        </div>
        <div className="flex items-center gap-4 pt-1.5">
          <Link href="/tokens" className={HEAD_LINK}>
            API Keys →
          </Link>
          <Link href="/router" className={HEAD_LINK}>
            Router →
          </Link>
        </div>
      </div>

      <Strip
        s={state.strip}
        onRetry={() => void data.refreshClients()}
        onRetryTokens={() => void data.refreshTokens()}
        onCreateKey={openCreate}
      />

      {state.kind === "error" ? null : !out || !client ? (
        <Skeleton />
      ) : (
        <>
          <div className="z-10 -mx-1 mt-4 px-1 min-[1100px]:sticky min-[1100px]:top-0 min-[1100px]:bg-chat-page min-[1100px]:pb-1 min-[1100px]:pt-2">
            <PickerBar
              tokens={tokens}
              tokensError={data.errors.tokens}
              keyValue={newKey ? NEW_KEY : keyValue}
              newKey={newKey ? { relay: newKey.relay } : null}
              onKeyChange={onKeyChange}
              onForget={forget}
              keySelectRef={keySelectRef}
              models={models}
              model={model?.served_name ?? null}
              onModelChange={setModelPick}
              rules={routerMode && router ? router.rules : null}
              minContext={hardMinContext(client, mode)}
              toolName={client.name}
            />
          </div>

          <div className="mt-4 flex flex-col gap-4 min-[1100px]:mt-5 min-[1100px]:grid min-[1100px]:grid-cols-[280px_minmax(0,1fr)] min-[1100px]:items-start min-[1100px]:gap-6">
            <ToolList clients={clients} selected={client.id} onSelect={selectTool} />
            <ClientPanel
              client={client}
              mode={mode}
              onModeChange={selectMode}
              files={files}
              renderFile={renderFile}
              checks={checks}
              router={router}
              vars={vars}
              createdKey={newKey?.plaintext ?? null}
              pasted={pasted}
              onPaste={setPasted}
              resetKey={`${client.id}|${mode?.id}|${model?.served_name}`}
              originNote={
                publicUrl === null
                  ? "Using this tab's address; set a public URL in Settings › Networking if clients connect from elsewhere."
                  : null
              }
            />
          </div>

          <p className="mb-0 mt-6 max-w-[90ch] text-[12.5px] text-chat-muted">
            Not here yet: Gemini CLI, GitHub Copilot CLI, Roo Code, Kilo Code, Zed and JetBrains AI. Any client with
            an OpenAI-compatible base URL setting works like the OpenAI SDK recipe.
          </p>
        </>
      )}

      {/* Keyed by tool and relay so the dialog's own fields start from this tool's defaults. */}
      <CreateTokenDialog
        key={`${client?.id ?? "none"}-${routerMode}`}
        open={dialog}
        initialName={client?.id ?? ""}
        initialRelay={!!mode?.needs_relay_key}
        onCreated={(plaintext, info) => {
          setNewKey({ plaintext, relay: info?.anthropicRelay ?? false });
          setPasted("");
          void data.refreshTokens();
        }}
        onClose={() => {
          setDialog(false);
          // The opener (the select's "Create a key…" or the strip button)
          // may have re-rendered away: land on the key picker.
          requestAnimationFrame(() => keySelectRef.current?.focus());
        }}
      />
    </div>
  );
}
