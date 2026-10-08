"use client";

// Overview head (plan §2, §4.2): title, one-sentence purpose, "Activity →",
// and the master switch. The switch applies immediately (no Save): it is the
// incident kill switch. Turning it off while Claude Code is actively using
// the router asks first.

import { useCallback, useId, useRef, useState } from "react";
import Link from "next/link";
import { authFetch } from "@/lib/auth-fetch";
import { errorDetail } from "@/components/tokens/use-token-actions";
import { Modal } from "@/components/ui/modal";
import { cn } from "@/lib/utils";
import {
  ACTIVITY_HREF,
  type DecisionOut,
  type RouterSettingsOut,
  type RuleOut,
} from "@/lib/router";
import { RouterSwitch } from "./switch";
import { btn, FOCUS } from "./styles";

/** A decision this recent means someone is using the router right now. */
export const CONFIRM_WINDOW_MS = 10 * 60 * 1000;

export function hasRecentDecision(decisions: DecisionOut[] | undefined, now: number): boolean {
  return (decisions ?? []).some((d) => {
    const t = Date.parse(d.ts);
    return !Number.isNaN(t) && now - t < CONFIRM_WINDOW_MS;
  });
}

export interface RoutingSwitch {
  /** The value being saved, or null when idle. */
  pending: boolean | null;
  /** The off-confirm dialog is open. */
  confirming: boolean;
  request: (on: boolean) => void;
  confirm: () => void;
  cancel: () => void;
}

interface SwitchArgs {
  decisions: DecisionOut[] | undefined;
  /** Re-fetch settings; awaited so the switch never flickers back. */
  onSaved: () => Promise<unknown>;
  /** null clears the previous error. */
  onError: (detail: string | null) => void;
}

/** The one handler behind the master switch, the strip's "Turn on" and the
 *  checklist's step 4. */
export function useRoutingSwitch({ decisions, onSaved, onError }: SwitchArgs): RoutingSwitch {
  const [pending, setPending] = useState<boolean | null>(null);
  const [confirming, setConfirming] = useState(false);

  const send = useCallback(
    async (on: boolean) => {
      setPending(on);
      onError(null);
      try {
        const r = await authFetch("/api/router/settings", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: on }),
        });
        if (!r.ok) {
          onError(await errorDetail(r, `Couldn't turn routing ${on ? "on" : "off"} (HTTP ${r.status})`));
          return;
        }
        await onSaved();
      } catch (err) {
        onError(err instanceof Error ? err.message : "Network error");
      } finally {
        setPending(null);
      }
    },
    [onSaved, onError],
  );

  return {
    pending,
    confirming,
    request: (on) => {
      if (pending !== null) return;
      if (!on && hasRecentDecision(decisions, Date.now())) {
        setConfirming(true);
        return;
      }
      void send(on);
    },
    confirm: () => {
      setConfirming(false);
      void send(false);
    },
    cancel: () => setConfirming(false),
  };
}

interface Props {
  settings: RouterSettingsOut | undefined;
  rules: RuleOut[] | undefined;
  sw: RoutingSwitch;
  showActivity: boolean;
}

function subLabel(on: boolean, settings: RouterSettingsOut, rules: RuleOut[] | undefined): string {
  if (!on) return "Claude model names are not routed";
  const n = rules?.length;
  const r = n === undefined ? "" : n === 0 ? "no rules · " : `${n} rule${n === 1 ? "" : "s"} · `;
  return `${r}pass-through ${settings.passthrough_unmatched ? "on" : "off"}`;
}

export function RouterHeader({ settings, rules, sw, showActivity }: Props) {
  const descId = useId();
  const cancelRef = useRef<HTMLButtonElement | null>(null);
  const on = sw.pending ?? settings?.enabled ?? false;
  const saving = sw.pending !== null;

  return (
    <div className="flex flex-wrap items-start justify-between gap-4 max-[759px]:flex-col">
      <div className="min-w-0">
        <h1 className="m-0 text-2xl font-semibold leading-tight tracking-[-0.01em]">Claude Code router</h1>
        <p className="mb-0 mt-1.5 max-w-[62ch] text-[13.5px] text-chat-muted">
          Claude Code points at this warden. The Claude models you choose run on your GPUs; every
          other request goes to Anthropic on the user&apos;s own login.
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-3.5 max-[759px]:w-full max-[759px]:flex-nowrap max-[759px]:justify-between">
        {showActivity && (
          <Link
            href={ACTIVITY_HREF}
            className={cn(
              "whitespace-nowrap rounded-sm border-b border-vw-rule-soft text-[13px] text-chat-muted hover:text-chat-fg",
              FOCUS,
            )}
          >
            Activity →
          </Link>
        )}
        {settings ? (
          <label
            className={cn(
              "flex cursor-pointer items-center gap-3 rounded-xl border bg-chat-surface/55 py-2 pl-3.5 pr-3 max-[759px]:flex-1",
              on ? "border-vw-live/45" : "border-vw-rule-soft/80",
            )}
          >
            <span id={descId} className="flex min-w-0 flex-1 flex-col leading-tight">
              <b className="text-[14px] font-semibold">{on ? "Routing is on" : "Routing is off"}</b>
              <span className="text-[12px] text-chat-muted">
                {saving ? "Saving…" : subLabel(on, settings, rules)}
              </span>
            </span>
            <RouterSwitch
              checked={on}
              onChange={(v) => sw.request(v)}
              label="Routing"
              describedBy={descId}
              disabled={saving}
              testId="router-enabled"
            />
          </label>
        ) : (
          <div
            aria-hidden="true"
            className="h-[50px] w-[240px] animate-pulse rounded-xl border border-vw-rule-soft/60 bg-chat-surface/55 motion-reduce:animate-none max-[759px]:w-full"
          />
        )}
      </div>

      <Modal open={sw.confirming} onClose={sw.cancel} title="Turn routing off?" initialFocusRef={cancelRef}>
        <p className="m-0 text-sm">
          Claude Code sessions using this warden will get 404 for Claude models until it is back on.
          Nothing will be relayed.
        </p>
        <div className="mt-4 flex justify-end gap-2">
          <button ref={cancelRef} type="button" className={btn("default", FOCUS)} onClick={sw.cancel}>
            Cancel
          </button>
          <button type="button" className={btn("danger", FOCUS)} onClick={sw.confirm}>
            Turn off
          </button>
        </div>
      </Modal>
    </div>
  );
}
