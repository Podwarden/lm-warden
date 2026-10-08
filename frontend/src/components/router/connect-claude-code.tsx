"use client";

// Connect Claude Code (plan §4.5): open on first run, collapsed with a "Seen
// from N keys" summary once requests arrive. A key created inline is handed
// in as `plaintext` (page state only — never stored or logged) and fills
// every snippet in place of the placeholder; it is gone on unmount.

import { useId, useRef, useState } from "react";
import Link from "next/link";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { getPublicBaseUrl } from "@/lib/public-url";
import { cn, copyToClipboard } from "@/lib/utils";
import {
  KEY_PLACEHOLDER,
  envSnippet,
  localOnlySnippet,
  settingsJsonSnippet,
  type DecisionOut,
  type RelayInfo,
  type RouterSettingsOut,
  type RuleOut,
} from "@/lib/router";
import { btn, FOCUS, ICON_BTN, LINK, pill, SEC, SEC_BODY, SEC_NOTE, SEC_TITLE } from "./styles";

type TabId = "shell" | "json" | "local";

const TABS: { id: TabId; label: string }[] = [
  { id: "shell", label: "Shell" },
  { id: "json", label: "settings.json" },
  { id: "local", label: "Local only (no Anthropic account)" },
];

interface Props {
  settings: RouterSettingsOut;
  rules: RuleOut[];
  /** Newest first; distinct key names feed the collapsed summary. */
  decisions: DecisionOut[] | undefined;
  relay: RelayInfo;
  /** A just-created key's plaintext, or null. */
  plaintext: string | null;
  /** Whether that key may relay to Anthropic (the operator can untick it). */
  plaintextRelay?: boolean;
  /** Open on first render (setup: no request seen yet). */
  defaultOpen: boolean;
  onCreateRelayKey: () => void;
  className?: string;
}

function summary(decisions: DecisionOut[] | undefined): string {
  const names = [...new Set((decisions ?? []).map((d) => d.token_name).filter((n): n is string => !!n))];
  if (!decisions || decisions.length === 0) return "No requests yet";
  if (names.length === 0) return "Requests seen · shell, settings.json and local-only snippets";
  const head = `Seen from ${names.length} key${names.length === 1 ? "" : "s"}: ${names.slice(0, 3).join(", ")}${names.length > 3 ? ", …" : ""}`;
  return `${head} · shell, settings.json and local-only snippets`;
}

function RelayLine({
  relay,
  plaintext,
  plaintextRelay,
  onCreate,
}: {
  relay: RelayInfo;
  plaintext: string | null;
  plaintextRelay: boolean;
  onCreate: () => void;
}) {
  let badge: React.ReactNode = null;
  let note: React.ReactNode;
  if (plaintext && plaintextRelay) {
    badge = <span className={pill("ok")}>New relay key</span>;
    note = <strong className="font-semibold text-chat-fg">Shown once — copy it now.</strong>;
  } else if (plaintext) {
    badge = <span className={pill("idle")}>New key</span>;
    note = (
      <>
        <strong className="font-semibold text-chat-fg">Shown once — copy it now.</strong> It can&apos;t relay
        to Anthropic, so fallbacks and unmatched models are refused (403) for it.
      </>
    );
  } else if (relay.relayLoading) {
    note = "Checking your API keys…";
  } else if (relay.relayKnown && relay.relayKeys > 0) {
    badge = (
      <span className={pill("ok")}>
        {relay.relayKeys === 1 ? "1 key may relay" : `${relay.relayKeys} keys may relay`}
      </span>
    );
    note = "Paste its plaintext in place of the placeholder below.";
  } else if (relay.relayKnown) {
    badge = <span className={pill("amber")}>No relay key yet</span>;
    note = (
      <>
        The key needs <b className="font-medium text-chat-fg">May relay to Anthropic</b>. Create one to
        fill the snippet with a real key — it is shown once.
      </>
    );
  } else {
    badge = <span className={pill("idle")}>Relay keys not checked</span>;
    note = "Use a key with the relay flag on.";
  }
  return (
    <div className="mb-3 flex flex-wrap items-center gap-2.5 text-[13px]">
      {badge}
      <span className="text-chat-muted">{note}</span>
      <button
        type="button"
        onClick={onCreate}
        className={btn("default", `min-h-[30px] px-2.5 py-[3px] text-[12.5px] max-[759px]:min-h-10 ${FOCUS}`)}
      >
        Create a relay key
      </button>
    </div>
  );
}

export function ConnectClaudeCode({
  settings,
  rules,
  decisions,
  relay,
  plaintext,
  plaintextRelay = true,
  defaultOpen,
  onCreateRelayKey,
  className,
}: Props) {
  const { data: runtime } = useSWR<{ public_url?: string | null }>("/api/settings/runtime", authFetchJSON);
  const [userOpen, setUserOpen] = useState<boolean | null>(null);
  const [tab, setTab] = useState<TabId>("shell");
  const [copy, setCopy] = useState<"idle" | "copied" | "failed">("idle");
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const bodyId = useId();
  const tabsId = useId();

  // A fresh key must be visible: it is shown once.
  const open = userOpen ?? (plaintext ? true : defaultOpen);

  const baseUrl = getPublicBaseUrl(runtime?.public_url);
  const header = settings.header_name;
  const key = plaintext ?? KEY_PLACEHOLDER;
  const firstServed = [...rules]
    .filter((r) => r.enabled)
    .sort((a, b) => a.position - b.position)
    .find((r) => r.target_served_name)?.target_served_name;

  const snippet =
    tab === "shell"
      ? envSnippet(baseUrl, header, key)
      : tab === "json"
        ? settingsJsonSnippet(baseUrl, header, key)
        : localOnlySnippet(baseUrl, firstServed ?? "your-served-model-name", key);

  async function doCopy() {
    try {
      await copyToClipboard(snippet);
      setCopy("copied");
      setTimeout(() => setCopy("idle"), 1500);
    } catch {
      setCopy("failed");
    }
  }

  function onTabKey(e: React.KeyboardEvent, i: number) {
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!d && e.key !== "Home" && e.key !== "End") return;
    e.preventDefault();
    const n = TABS.length;
    const next = e.key === "Home" ? 0 : e.key === "End" ? n - 1 : (i + d + n) % n;
    setTab(TABS[next].id);
    tabRefs.current[next]?.focus();
  }

  return (
    <section
      id="connect"
      aria-labelledby="router-connect-h"
      data-testid="router-connect"
      className={cn(SEC, "scroll-mt-4", className)}
    >
      <h2 className={cn(SEC_TITLE, "px-4 py-3.5")}>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={() => setUserOpen(!open)}
          className={cn(
            "flex w-full min-w-0 items-center gap-2.5 rounded-md bg-transparent p-0 text-left max-[759px]:flex-wrap",
            FOCUS,
          )}
        >
          <svg
            viewBox="0 0 16 16"
            className={cn("h-4 w-4 flex-none text-chat-muted motion-safe:transition-transform", open && "rotate-90")}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            aria-hidden="true"
          >
            <path d="M6 3l5 5-5 5" />
          </svg>
          <span id="router-connect-h" className="whitespace-nowrap">
            Connect Claude Code
          </span>
          {open ? (
            <span className={cn(SEC_NOTE, "font-normal")}>
              Keep your normal <code className="font-mono">claude</code> login
            </span>
          ) : (
            <span className="min-w-0 truncate text-[12.5px] font-normal text-chat-muted max-[759px]:basis-full max-[759px]:whitespace-normal max-[759px]:pl-[26px]">
              {summary(decisions)}
            </span>
          )}
        </button>
      </h2>
      {open && (
        <div id={bodyId} className={SEC_BODY}>
          <RelayLine relay={relay} plaintext={plaintext} plaintextRelay={plaintextRelay} onCreate={onCreateRelayKey} />
          <div
            role="tablist"
            aria-label="Snippet format"
            className="mb-2.5 flex gap-1 overflow-x-auto border-b border-vw-rule-soft/60"
          >
            {TABS.map((t, i) => (
              <button
                key={t.id}
                ref={(el) => {
                  tabRefs.current[i] = el;
                }}
                type="button"
                role="tab"
                id={`${tabsId}-${t.id}`}
                aria-selected={tab === t.id}
                aria-controls={`${tabsId}-panel`}
                tabIndex={tab === t.id ? 0 : -1}
                onClick={() => setTab(t.id)}
                onKeyDown={(e) => onTabKey(e, i)}
                className={cn(
                  "-mb-px whitespace-nowrap border-b-2 bg-transparent px-2.5 py-2 text-[13px]",
                  tab === t.id
                    ? "border-chat-accent font-semibold text-chat-fg"
                    : "border-transparent text-chat-muted hover:text-chat-fg",
                  FOCUS,
                )}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div
            role="tabpanel"
            id={`${tabsId}-panel`}
            aria-labelledby={`${tabsId}-${tab}`}
            className="relative"
          >
            <pre
              data-testid="router-snippet"
              className="m-0 whitespace-pre-wrap rounded-lg border border-vw-rule-soft/60 bg-vw-dock py-3 pl-3 pr-12 font-mono text-[12.5px] leading-[1.6] [overflow-wrap:anywhere]"
            >
              {snippet}
            </pre>
            <button type="button" aria-label="Copy snippet" onClick={doCopy} className={cn(ICON_BTN, "absolute right-2 top-2")}>
              {copy === "copied" ? (
                <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                  <path d="M3.5 8.5 6.5 11.5 12.5 4.5" />
                </svg>
              ) : (
                <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                  <rect x="5" y="5" width="9" height="9" rx="1.5" />
                  <path d="M11 5V3.5A1.5 1.5 0 0 0 9.5 2h-6A1.5 1.5 0 0 0 2 3.5v6A1.5 1.5 0 0 0 3.5 11H5" />
                </svg>
              )}
            </button>
            <span role="status" className="sr-only">
              {copy === "copied" ? "Copied" : ""}
            </span>
            {copy === "failed" && (
              <p className="mb-0 mt-1.5 text-[12.5px] text-vw-danger-fg">Copy failed — select the text.</p>
            )}
          </div>
          <ul className="mb-0 mt-2.5 grid list-disc gap-1 pl-4 text-[12.5px] text-chat-muted">
            <li>
              Needs Claude Code 2.1.227 or newer (for <code className="font-mono">ANTHROPIC_CUSTOM_HEADERS</code>).
              Older clients: use the Local only tab.
            </li>
            <li>
              Don&apos;t put the warden key in <code className="font-mono">ANTHROPIC_API_KEY</code> or{" "}
              <code className="font-mono">ANTHROPIC_AUTH_TOKEN</code> — that would replace your Claude login.
            </li>
          </ul>
          <p className="mb-0 mt-2.5 text-[12.5px] text-chat-muted">
            <Link href="/connect?tool=opencode" className={LINK}>
              Using another tool? Connect Codex, OpenCode, Aider… →
            </Link>
          </p>
        </div>
      )}
    </section>
  );
}
