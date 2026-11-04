"use client";

// A key's sensitive data guard choice: use the platform setting (with what that means), off for
// this key, or its own settings (the shared form). Used by the key page's card and the Create API
// token dialog, so both offer the same thing.

import { useId } from "react";

import { GuardForm, type FormMode } from "@/components/guard/guard-form";
import { G } from "@/lib/guard/strings";
import { type GuardSettings, type KeyGuard, type PacksResponse, payload } from "@/lib/guard/types";
import { cn } from "@/lib/utils";

export type GuardChoice = "inherit" | "off" | "own";

export function choiceOf(g: KeyGuard | undefined): GuardChoice {
  if (!g) return "inherit";
  return "off" in g ? "off" : "own";
}

/** What PATCH / POST /api/tokens takes for this choice. */
export function keyGuardBody(choice: GuardChoice, own: GuardSettings): KeyGuard {
  return choice === "inherit" ? null : choice === "off" ? { off: true } : { override: payload(own) };
}

export function guardSummary(
  eff: (Partial<GuardSettings> & { source?: unknown }) | null | undefined,
  packs: PacksResponse | undefined,
): string {
  if (!eff || !eff.mode) return G.key.platformOff;
  const labels = (eff.packs ?? []).map((id) => packs?.packs?.find((p) => p.id === id)?.label ?? id);
  const parts = [eff.mode === "block" ? G.mode.block : G.mode.sanitize, ...labels];
  if (eff.rules?.length) parts.push(G.key.customRules(eff.rules.length));
  return parts.join(" · ");
}

interface Props {
  choice: GuardChoice;
  onChoice: (c: GuardChoice) => void;
  /** The key's own settings (shown when `choice` is "own"). */
  own: GuardSettings;
  onOwn: (s: GuardSettings) => void;
  /** What "use the platform setting" means for this key right now. */
  inherited: (Partial<GuardSettings> & { source?: unknown }) | null | undefined;
  packs: PacksResponse | undefined;
}

export function KeyGuardChoice({ choice, onChoice, own, onOwn, inherited, packs }: Props) {
  const name = useId();
  const opt = (c: GuardChoice, label: string) => (
    <label
      className={cn(
        "flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1 text-[13px]",
        choice === c ? "border-chat-accent" : "border-chat-rule",
      )}
    >
      <input type="radio" name={name} checked={choice === c} onChange={() => onChoice(c)} />
      {label}
    </label>
  );
  return (
    <div className="space-y-3">
      <fieldset>
        <legend className="mb-1.5 text-[13px] font-semibold">{G.key.legend}</legend>
        <div className="flex flex-wrap gap-2">
          {opt("inherit", G.key.inherit)}
          {opt("off", G.key.off)}
          {opt("own", G.key.own)}
        </div>
        {choice === "inherit" && (
          <p className="mt-2 text-[12px]">
            <span className="text-chat-dim">{G.key.effective}</span> {guardSummary(inherited, packs)}
          </p>
        )}
      </fieldset>
      {choice === "own" && (
        <GuardForm
          value={own}
          mode={own.mode}
          packs={packs}
          onChange={(v, m: FormMode) => onOwn(m === "off" ? v : { ...v, mode: m })}
        />
      )}
    </div>
  );
}
