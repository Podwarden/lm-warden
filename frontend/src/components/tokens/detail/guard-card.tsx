"use client";

// Token page → Sensitive data: the key uses the platform setting, switches the guard off for itself,
// or has its own settings (PATCH /api/tokens/{id} with guard null / {"off": true} /
// {"override": {...}}). Shows what is in effect, a try-it box and this key's log.

import { useId, useState } from "react";
import useSWR from "swr";

import { GuardLog } from "@/components/guard/guard-log";
import { choiceOf, type GuardChoice, KeyGuardChoice, keyGuardBody } from "@/components/guard/key-guard-choice";
import { TryIt } from "@/components/guard/try-it";
import { authFetchJSON } from "@/lib/auth-fetch";
import { sendJSON } from "@/lib/guard/api";
import { G } from "@/lib/guard/strings";
import { type GuardSettings, type KeyGuard, normalise, type PacksResponse } from "@/lib/guard/types";

import { btn, CARD, CARD_TITLE } from "./styles";

export interface GuardCardProps {
  token: {
    id: string;
    guard?: KeyGuard;
    guard_effective?: (Partial<GuardSettings> & { source?: unknown }) | null;
  };
  onSaved: () => void;
}

export function GuardCard({ token, onSaved }: GuardCardProps) {
  const name = useId();
  const packs = useSWR<PacksResponse>("/api/guard/packs", authFetchJSON);
  const [choice, setChoice] = useState<GuardChoice>(choiceOf(token.guard));
  const [own, setOwn] = useState<GuardSettings | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);

  // Own settings start from the key's override on top of what is in effect today.
  const base: GuardSettings = normalise(
    {
      ...(token.guard_effective ?? {}),
      ...(token.guard && "override" in token.guard ? token.guard.override : {}),
    },
    packs.data,
  );
  const value = own ?? base;
  const effective = token.guard_effective ? normalise(token.guard_effective, packs.data) : null;

  async function save() {
    setBusy(true);
    setNote(null);
    const r = await sendJSON(`/api/tokens/${encodeURIComponent(token.id)}`, "PATCH", {
      guard: keyGuardBody(choice, value),
    });
    setBusy(false);
    setNote(r.ok ? { ok: true, text: G.saved } : { ok: false, text: r.error ?? "" });
    if (r.ok) onSaved();
  }

  return (
    <section className={CARD} aria-labelledby={`${name}-title`}>
      <h2 id={`${name}-title`} className={CARD_TITLE}>
        {G.title}
      </h2>
      <p className="mb-3 text-[12px] text-chat-dim">{G.intro}</p>
      <KeyGuardChoice
        choice={choice}
        onChoice={setChoice}
        own={value}
        onOwn={setOwn}
        inherited={token.guard_effective}
        packs={packs.data}
      />
      <div className="mt-3 flex items-center gap-3">
        <button type="button" className={btn("primary")} disabled={busy} onClick={() => void save()}>
          {busy ? G.saving : G.save}
        </button>
        {note && (
          <span role={note.ok ? "status" : "alert"} className={note.ok ? "text-[12px] text-chat-dim" : "text-[12px] text-vw-danger-fg"}>
            {note.text}
          </span>
        )}
      </div>
      {(choice === "own" || (choice === "inherit" && effective)) && (
        <div className="mt-4">
          <TryIt settings={choice === "own" ? value : (effective as GuardSettings)} />
        </div>
      )}
      <div className="mt-4">
        <GuardLog tokenId={token.id} />
      </div>
    </section>
  );
}
