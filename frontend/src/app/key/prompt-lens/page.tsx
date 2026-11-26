"use client";

// /key/prompt-lens — Prompt Lens for a key holder: their own key's requests
// only, full screen (owner request 2026-11-23; it was a dock along the bottom
// of /key/stats). Shown only while the platform switch makes it available to
// key holders (GET /api/me/key `prompt_lens_view_available`), the same rule
// the dock followed; the stream and section routes enforce the scope (G-R11).
// ?req=<request id> opens that request (the eye on a Stats row or chart dot).

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

import { KeyShell } from "@/components/key-area/key-shell";
import { useLensDeepLink, useWritePick } from "@/components/prompt-lens/prompt-lens-page";
import { PromptLensView } from "@/components/prompt-lens/prompt-lens-view";
import { type MeKey, useMe } from "@/lib/key-area/api";

function KeyPromptLens() {
  const params = useSearchParams();
  const { select, view, session } = useLensDeepLink(params);
  const onPick = useWritePick();
  const me = useMe<MeKey>("/api/me/key");
  const available = me.data?.prompt_lens_view_available === true;
  if (!me.data) return <KeyShell>{null}</KeyShell>;
  if (!available) {
    return (
      <KeyShell>
        <div data-testid="prompt-lens-off" className="text-sm text-slate-400">
          <h2 className="mb-2 text-base font-semibold text-slate-200">Prompt Lens is not available</h2>
          <p>
            {me.data.prompt_lens === false
              ? "The administrator has not allowed Prompt Lens for this key."
              : "The administrator has not turned Prompt Lens on."}{" "}
            Your requests are still on the{" "}
            <Link href="/key/stats" className="underline">
              Stats
            </Link>{" "}
            page.
          </p>
        </div>
      </KeyShell>
    );
  }
  return (
    <KeyShell fill crumb="Prompt Lens">
      <PromptLensView scope="key" pageKey="key-stats" select={select} onPick={onPick} view={view} session={session} />
    </KeyShell>
  );
}

export default function Page() {
  return (
    <Suspense fallback={null}>
      <KeyPromptLens />
    </Suspense>
  );
}
