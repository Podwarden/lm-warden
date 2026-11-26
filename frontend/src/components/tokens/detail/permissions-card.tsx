"use client";

// Per-key permissions (owner decision 2026-11-26, migration 0066), set by the admin: whether the
// key's holder may use Prompt Lens on their own traffic (pages, API, MCP tools), and whether the
// key may read engine facts (GET /api/me/engine). Each switch saves on its own.

import { useId, useState } from "react";
import { btn, CARD, CARD_TITLE } from "./styles";

export interface PermissionsCardProps {
  promptLens: boolean;
  engineFacts: boolean;
  /** Sets one permission; resolves null on success, else the error text. */
  onChange?: (flag: "prompt_lens" | "engine_facts", on: boolean) => Promise<string | null>;
  /** Whether Prompt Lens is on for the platform: the holder also needs it. */
  platformOn?: boolean;
}

function Row({
  flag,
  label,
  on,
  help,
  busy,
  onToggle,
}: {
  flag: string;
  label: string;
  on: boolean;
  help: string;
  busy: boolean;
  onToggle?: (on: boolean) => void;
}) {
  const id = useId();
  return (
    <div className="mt-2 first:mt-0">
      <div className="flex items-center justify-between gap-2">
        <span id={id} className="text-[13px] text-chat-muted">
          {label}: <span className="font-semibold text-chat-fg">{on ? "allowed" : "off"}</span>
        </span>
        <button
          type="button"
          role="switch"
          aria-checked={on}
          aria-label={label}
          data-testid={`perm-${flag}`}
          disabled={!onToggle || busy}
          onClick={() => onToggle?.(!on)}
          className={btn(on ? "primary" : "default")}
        >
          {on ? "Turn off" : "Turn on"}
        </button>
      </div>
      <div className="mt-0.5 text-[12px] leading-[1.45] text-chat-dim">{help}</div>
    </div>
  );
}

export function PermissionsCard({ promptLens, engineFacts, onChange, platformOn = true }: PermissionsCardProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function set(flag: "prompt_lens" | "engine_facts", on: boolean) {
    setBusy(true);
    setError(null);
    try {
      setError((await onChange?.(flag, on)) ?? null);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className={CARD} aria-label="Permissions">
      <h2 className={CARD_TITLE}>Permissions</h2>
      <Row
        flag="prompt_lens"
        label="Prompt Lens"
        on={promptLens}
        busy={busy}
        onToggle={(on) => void set("prompt_lens", on)}
        help="The holder may use Prompt Lens on this key's own traffic: the key area's pages, the API and the MCP tools. Needs Prompt Lens on in Settings."
      />
      {promptLens && !platformOn && (
        <div data-testid="perm-platform-off" className="mt-1 text-[12px] leading-[1.45] text-chat-accent">
          Prompt Lens is off on this server, so the holder cannot use it yet.
        </div>
      )}
      <Row
        flag="engine_facts"
        label="Engine facts"
        on={engineFacts}
        busy={busy}
        onToggle={(on) => void set("engine_facts", on)}
        help="The key may read the cache block size, match unit and context length of the models it may use (GET /api/me/engine)."
      />
      {error && (
        <div role="alert" className="mt-1 text-[12px] text-vw-danger-fg">
          {error}
        </div>
      )}
    </section>
  );
}
