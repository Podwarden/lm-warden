"use client";

// First-run checklist (plan §2.1, §4.3): the empty state is the onboarding.
// Five steps; the first one neither done nor loading is "current". Steps whose
// data is still loading get a neutral mark, never a false tick. Step 5 is a
// live region that flips from "Waiting…" to the first request it sees.

import Link from "next/link";
import { cn } from "@/lib/utils";
import { fmtInt, type DecisionOut, type SetupStep, type SetupStepId } from "@/lib/router";
import { btn, CARD, FOCUS, LINK, routeTone, ROUTE_LABEL, ROUTE_SWATCH, SEC, SEC_HEAD, SEC_NOTE, SEC_TITLE } from "./styles";

const SM = `min-h-[30px] px-2.5 py-[3px] text-[12.5px] max-[759px]:min-h-10 ${FOCUS}`;
export const PRESETS = ["claude-haiku*", "claude-sonnet*", "claude-opus*"] as const;

interface Props {
  steps: SetupStep[];
  decisions: DecisionOut[] | undefined;
  onAddRule: (pattern?: string) => void;
  onCreateRelayKey: () => void;
  onTurnOn: () => void;
  /** The switch is saving. */
  turning: boolean;
  className?: string;
}

const TITLES: Record<SetupStepId, string> = {
  rule: "Choose which Claude models run here",
  relay: "Give Claude Code a key that may relay to Anthropic",
  connect: "Point Claude Code at this warden",
  enable: "Turn routing on",
  first_request: "See the first request land",
};

function Mark({ step, n }: { step: SetupStep; n: number }) {
  const base =
    "mt-px grid h-6 w-6 place-items-center rounded-full border-[1.5px] text-[12px] font-bold";
  if (step.done) {
    return (
      <span className={cn(base, "border-vw-live/70 bg-vw-ok-bg text-vw-ok-fg")}>
        <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true">
          <path d="M3.5 8.5 6.5 11.5 12.5 4.5" />
        </svg>
        <span className="sr-only">Done:</span>
      </span>
    );
  }
  return (
    <span
      aria-hidden="true"
      className={cn(
        base,
        step.current ? "border-chat-accent text-chat-accent" : "border-vw-rule-soft text-chat-muted",
      )}
    >
      {step.loading ? "·" : n}
    </span>
  );
}

function FirstRequest({ decisions }: { decisions: DecisionOut[] | undefined }) {
  const d = decisions?.[0];
  return (
    <div role="status" className="inline-flex flex-wrap items-center gap-2 text-[13px] text-chat-muted">
      {d ? (
        <>
          <span className="font-semibold text-chat-fg">First request:</span>
          <code className="font-mono text-[12.5px] [overflow-wrap:anywhere]">
            {d.model_in ?? d.path}
            {d.route === "local" && d.model_out ? ` → ${d.model_out}` : ""}
          </code>
          <span aria-hidden="true">·</span>
          <span className={cn(ROUTE_LABEL, routeTone(d.route).fg)}>
            <span className={ROUTE_SWATCH} aria-hidden="true" />
            {routeTone(d.route).label}
          </span>
          <span aria-hidden="true">·</span>
          <span className="tabular-nums">{fmtInt(d.latency_ms)} ms</span>
        </>
      ) : (
        <>
          <span
            aria-hidden="true"
            className="h-2 w-2 flex-none rounded-full bg-chat-dim motion-safe:animate-pulse"
          />
          Waiting for the first request…
        </>
      )}
    </div>
  );
}

export function RouterSetup({ steps, decisions, onAddRule, onCreateRelayKey, onTurnOn, turning, className }: Props) {
  const done = steps.filter((s) => s.done).length;
  const by = (id: SetupStepId) => steps.find((s) => s.id === id)!;
  const relay = by("relay");

  function body(step: SetupStep) {
    switch (step.id) {
      case "rule":
        return (
          <>
            <p className="mb-0 mt-0.5 text-[13px] text-chat-muted">
              A rule maps a model name Claude Code asks for to one of your local models. Start with
              Haiku: it is Claude Code&apos;s background model and the cheapest to replace.
            </p>
            {!step.done && (
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <button type="button" className={btn("primary", SM)} onClick={() => onAddRule()}>
                  Add a rule
                </button>
                <div className="flex flex-wrap gap-1.5" role="group" aria-label="Start from a pattern">
                  {PRESETS.map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => onAddRule(p)}
                      className={cn(
                        "min-h-[30px] rounded-lg border border-dashed border-vw-rule-soft bg-transparent px-2.5 py-1 !font-mono text-[12.5px] text-chat-fg hover:bg-chat-surface-2/60 max-[759px]:min-h-10",
                        FOCUS,
                      )}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        );
      case "relay":
        return (
          <>
            <p className="mb-0 mt-0.5 text-[13px] text-chat-muted">
              {step.done ? (
                <>
                  A key with <b className="font-semibold text-chat-fg">May relay to Anthropic</b> exists.
                  Use it in the snippet below.
                </>
              ) : step.unverified ? (
                <>
                  We couldn&apos;t check every key. Make sure the key Claude Code uses has{" "}
                  <b className="font-semibold text-chat-fg">May relay to Anthropic</b>; without it,
                  unrouted models and fallbacks are refused with 403{" "}
                  <code className="font-mono">relay_not_allowed</code>.
                </>
              ) : step.loading ? (
                <>Checking your API keys…</>
              ) : (
                <>
                  No key has <b className="font-semibold text-chat-fg">May relay to Anthropic</b> yet.
                  Without it, models you don&apos;t route here (and fallbacks) are refused with 403{" "}
                  <code className="font-mono">relay_not_allowed</code>.
                </>
              )}
            </p>
            {!step.done && (
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  className={btn(relay.current ? "primary" : "default", SM)}
                  onClick={onCreateRelayKey}
                >
                  Create a relay key
                </button>
                <Link href="/tokens" className={cn(LINK, "text-[12.5px]")}>
                  or turn it on for an existing key
                </Link>
              </div>
            )}
          </>
        );
      case "connect":
        return (
          <p className="mb-0 mt-0.5 text-[13px] text-chat-muted">
            Two environment variables; your Claude login stays as it is. Needs Claude Code 2.1.227 or
            newer.{" "}
            <a href="#connect" className={LINK}>
              Snippets below ↓
            </a>
          </p>
        );
      case "enable":
        return (
          <>
            <p className="mb-0 mt-0.5 text-[13px] text-chat-muted">
              The switch at the top. It takes effect on the next request; no restart.
            </p>
            {!step.done && !step.loading && (
              <div className="mt-2">
                <button
                  type="button"
                  className={btn(step.current ? "primary" : "default", SM)}
                  onClick={onTurnOn}
                  disabled={turning}
                >
                  {turning ? "Turning on…" : "Turn on"}
                </button>
              </div>
            )}
          </>
        );
      case "first_request":
        return (
          <>
            <p className="mb-0 mt-0.5 text-[13px] text-chat-muted">
              Run <code className="font-mono">claude</code> and send any prompt.
            </p>
            <div className="mt-2">
              <FirstRequest decisions={decisions} />
            </div>
          </>
        );
    }
  }

  return (
    <section aria-labelledby="router-setup-h" data-testid="router-setup" className={cn(SEC, className)}>
      <div className={SEC_HEAD}>
        <h2 id="router-setup-h" tabIndex={-1} className={cn(SEC_TITLE, "focus:outline-none")}>
          Set up in five steps
        </h2>
        <span className={SEC_NOTE}>{`${done} of ${steps.length} done`}</span>
      </div>
      <ol className="m-0 list-none px-4 pb-2">
        {steps.map((s, i) => (
          <li
            key={s.id}
            aria-current={s.current ? "step" : undefined}
            className="grid grid-cols-[28px_minmax(0,1fr)] gap-x-3 gap-y-1 border-t border-vw-rule-soft/45 py-3 first:border-t-0"
          >
            <Mark step={s} n={i + 1} />
            <div className="min-w-0">
              <h3 className={cn("m-0 text-[14px]", s.done ? "font-medium text-chat-muted" : "font-semibold")}>
                {TITLES[s.id]}
              </h3>
              {body(s)}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

/** Rail card shown with the checklist: three facts instead of a paragraph. */
export function RoutingExplainer({ headerName, className }: { headerName: string; className?: string }) {
  return (
    <section aria-labelledby="router-how-h" className={cn(CARD, className)}>
      <h2 id="router-how-h" className="m-0 text-[15px] font-semibold">
        How routing works
      </h2>
      <ol className="mb-0 mt-2 grid list-decimal gap-2 pl-[18px] text-[13px] text-chat-muted">
        <li>
          <b className="font-semibold text-chat-fg">Two credentials, two headers.</b> Your Claude login
          stays in <code className="font-mono">Authorization</code>; the warden key rides in{" "}
          <code className="font-mono">{headerName}</code> and never reaches Anthropic.
        </li>
        <li>
          <b className="font-semibold text-chat-fg">First matching rule wins.</b> A match runs on your
          local model; anything else goes to Anthropic byte for byte.
        </li>
        <li>
          <b className="font-semibold text-chat-fg">Fails toward Anthropic.</b> If the local model fails
          before its first byte, the request falls back — or is refused with 529, per rule.
        </li>
      </ol>
    </section>
  );
}
