"use client";

// The two values every snippet needs (plan §3 #2–#3, §5.6): the key and the
// model. Native selects with visible labels. Router mode replaces the model
// picker with the rules, because the rules decide the model there.

import Link from "next/link";
import { forwardRef } from "react";
import { cn } from "@/lib/utils";
import { isLiveKey, isLoaded, type ConnectModel, type ConnectRule } from "@/lib/connect";
import type { TokenListPage } from "@/lib/router";
import { CARD, FIELD, FIELD_LABEL, LINK, META, PATTERN, SMALL_BTN, pill } from "./styles";

type TokenItem = TokenListPage["items"][number];

export const CREATE_KEY = "__create";
export const NEW_KEY = "__new";

const fmt = (n: number) => n.toLocaleString("en-US");

function deadWord(t: TokenItem): string {
  return t.is_revoked ? "revoked" : t.is_expired ? "expired" : "paused";
}

export function modelLabel(m: ConnectModel, minContext: number | null, toolName: string): string {
  if (!isLoaded(m)) return `${m.served_name} (not loaded)`;
  const parts = [m.served_name, m.context_window != null ? `${fmt(m.context_window)} tokens` : "context unknown"];
  if (m.supports_tools) parts.push("tools");
  if (minContext != null && m.context_window != null && m.context_window < minContext) {
    parts.push(`too small for ${toolName}`);
  }
  return parts.join(" · ");
}

interface KeyProps {
  tokens: TokenListPage | undefined;
  tokensError: boolean;
  value: string;
  hasNewKey: boolean;
  onChange: (value: string) => void;
}

const KeySelect = forwardRef<HTMLSelectElement, KeyProps>(function KeySelect(
  { tokens, tokensError, value, hasNewKey, onChange },
  ref,
) {
  const loading = !tokens && !tokensError;
  const items = tokens?.items ?? [];
  const live = items.filter(isLiveKey);
  const dead = items.filter((t) => !isLiveKey(t));
  return (
    <select
      ref={ref}
      id="connect-key"
      className={FIELD}
      value={value}
      disabled={loading}
      onChange={(e) => onChange(e.target.value)}
    >
      {loading ? (
        <option value="">Loading keys…</option>
      ) : (
        <>
          <option value="">{live.length || hasNewKey ? "Choose a key" : "No keys yet"}</option>
          {hasNewKey && <option value={NEW_KEY}>New key (shown once)</option>}
          {live.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name} ({t.preview})
            </option>
          ))}
          {dead.map((t) => (
            <option key={t.id} value={t.id} disabled>
              {t.name} ({t.preview}), {deadWord(t)}
            </option>
          ))}
          <option value={CREATE_KEY}>Create a key…</option>
        </>
      )}
    </select>
  );
});

interface Props {
  tokens: TokenListPage | undefined;
  tokensError: boolean;
  keyValue: string;
  /** A key created on this page is in state (its plaintext fills the snippets). */
  newKey: { relay: boolean } | null;
  onKeyChange: (value: string) => void;
  onForget: () => void;
  keySelectRef: React.Ref<HTMLSelectElement>;
  models: ConnectModel[];
  model: string | null;
  onModelChange: (served: string) => void;
  /** Router mode: the rules decide the model. */
  rules: ConnectRule[] | null;
  minContext: number | null;
  toolName: string;
  className?: string;
}

export function PickerBar({
  tokens,
  tokensError,
  keyValue,
  newKey,
  onKeyChange,
  onForget,
  keySelectRef,
  models,
  model,
  onModelChange,
  rules,
  minContext,
  toolName,
  className,
}: Props) {
  const anyLoaded = models.some(isLoaded);
  const ordered = [...models.filter(isLoaded), ...models.filter((m) => !isLoaded(m))];
  const truncated = tokens && tokens.total > tokens.items.length;

  return (
    <div data-testid="connect-pickers" className={cn(CARD, "grid gap-x-6 gap-y-3 min-[760px]:grid-cols-2", className)}>
      <div className="grid min-w-0 content-start gap-1.5">
        <label htmlFor="connect-key" className={FIELD_LABEL}>
          Key
        </label>
        <div className="flex min-w-0 items-center gap-2">
          <KeySelect
            ref={keySelectRef}
            tokens={tokens}
            tokensError={tokensError}
            value={keyValue}
            hasNewKey={!!newKey}
            onChange={onKeyChange}
          />
          {newKey && (
            <button type="button" className={SMALL_BTN} onClick={onForget}>
              Forget key
            </button>
          )}
        </div>
        <p className={cn(META, "m-0")}>
          {newKey ? (
            <>
              <strong className="font-semibold text-chat-fg">Shown once — copy it now.</strong> It lives in this tab
              only and is gone when you leave the page.
            </>
          ) : (
            "The warden keeps only a hash of each key, so snippets show a placeholder unless you create a key here."
          )}
          {truncated && ` You have ${fmt(tokens.total)} keys; the first 500 shown.`}
        </p>
      </div>

      <div className="grid min-w-0 content-start gap-1.5">
        {rules ? (
          <>
            <span className={FIELD_LABEL}>Model</span>
            <div data-testid="connect-rules" className="min-w-0">
              {rules.length === 0 ? (
                <p className={cn(META, "m-0")}>
                  No enabled rule yet, so every Claude model goes to Anthropic.{" "}
                  <Link href="/router" className={LINK}>
                    Add a rule
                  </Link>
                </p>
              ) : (
                <ul className="m-0 grid list-none gap-1 p-0 text-[13px]">
                  {rules.map((r) => (
                    <li key={r.pattern} className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5">
                      <code className={PATTERN}>{r.pattern}</code>
                      <span aria-hidden="true" className="text-chat-muted">
                        →
                      </span>
                      <span className="sr-only">runs on</span>
                      <code className="!font-mono text-[13px] [overflow-wrap:anywhere]">
                        {r.target_served_name ?? "a deleted model"}
                      </code>
                      {r.target_status && r.target_status !== "loaded" && (
                        <span className={pill("amber")}>{r.target_status}</span>
                      )}
                    </li>
                  ))}
                </ul>
              )}
              <p className={cn(META, "m-0 mt-1")}>
                Router mode: these rules pick the model, so there is nothing to choose here.
              </p>
            </div>
          </>
        ) : (
          <>
            <label htmlFor="connect-model" className={FIELD_LABEL}>
              Model
            </label>
            <select
              id="connect-model"
              className={FIELD}
              value={anyLoaded ? (model ?? "") : ""}
              disabled={!anyLoaded}
              onChange={(e) => onModelChange(e.target.value)}
            >
              {!anyLoaded ? (
                <option value="">No model loaded</option>
              ) : (
                ordered.map((m) => (
                  <option key={m.id} value={m.served_name} disabled={!isLoaded(m)}>
                    {modelLabel(m, minContext, toolName)}
                  </option>
                ))
              )}
            </select>
            <p className={cn(META, "m-0")}>
              {anyLoaded
                ? "Snippets name this model; it must match a served name exactly."
                : "Snippets use a placeholder model name until a model is loaded."}
            </p>
          </>
        )}
      </div>
    </div>
  );
}
