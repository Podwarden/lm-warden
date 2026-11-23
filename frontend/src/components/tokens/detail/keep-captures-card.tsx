"use client";

// Prompt Lens "keep captures" (ruling G-R4): an admin-only switch per key.
// On, the key's last 50 finished requests -- prompts included -- are written to
// this server's database, so they outlive memory and a restart. Off deletes
// them (app/proxy/prompt_lens_kept.py).

import Link from "next/link";
import { useId } from "react";
import { btn, CARD, CARD_TITLE } from "./styles";

export interface KeepCapturesCardProps {
  on: boolean;
  /** Flips the switch; resolves true on success. */
  onToggle?: (on: boolean) => Promise<boolean>;
  busy?: boolean;
  error?: string | null;
  /** Whether Prompt Lens is on for the platform (the Settings switch, else VW_PROMPT_LENS_ENABLED). */
  platformOn?: boolean;
  /** The Prompt Lens page with this key (and its earlier keys) preset, when
   *  there is something to see there (Prompt Lens on, or captures kept). */
  lensHref?: string | null;
}

export function KeepCapturesCard({
  on,
  onToggle,
  busy,
  error,
  platformOn = true,
  lensHref = null,
}: KeepCapturesCardProps) {
  const labelId = useId();
  return (
    <section className={CARD} aria-label="Prompt Lens">
      <h2 className={CARD_TITLE}>Prompt Lens</h2>
      {lensHref && (
        <Link
          href={lensHref}
          data-testid="token-prompt-lens-link"
          className="mb-2 inline-flex items-center gap-1 text-[13px] font-medium text-chat-accent hover:underline"
        >
          Open this key&apos;s requests in Prompt Lens →
        </Link>
      )}
      <div className="flex items-center justify-between gap-2">
        <span id={labelId} className="text-[13px] text-chat-muted">
          Keep captures: <span className="font-semibold text-chat-fg">{on ? "on" : "off"}</span>
        </span>
        <button
          type="button"
          role="switch"
          aria-checked={on}
          aria-label="Keep captures"
          disabled={!onToggle || busy}
          onClick={() => void onToggle?.(!on)}
          className={btn(on ? "primary" : "default")}
        >
          {on ? "Turn off" : "Turn on"}
        </button>
      </div>
      <div data-testid="keep-captures-note" className="mt-1.5 text-[12px] leading-[1.45] text-chat-dim">
        When on, Prompt Lens keeps this key&apos;s last 50 finished requests on disk: their
        prompts, replies and tool calls are stored in this server&apos;s database, so they
        survive a restart. The key&apos;s holder sees them in their own Prompt Lens. Turning
        this off deletes what was kept, for this key and the keys it was rotated from;
        deleting the key deletes it too.
      </div>
      {!platformOn && (
        <div data-testid="keep-captures-off" className="mt-1.5 text-[12px] leading-[1.45] text-chat-accent">
          Prompt Lens is off on this server (Settings, General, Privacy), so nothing is captured or
          kept now. Captures already kept stay on disk: you can still open them in Prompt Lens
          (read-only); the key&apos;s holder cannot until it is on again.
        </div>
      )}
      {error && (
        <div role="alert" className="mt-1 text-[12px] text-vw-danger-fg">{error}</div>
      )}
    </section>
  );
}
