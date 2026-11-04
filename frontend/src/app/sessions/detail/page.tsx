"use client";
// /sessions/detail?id=<s_… or r_…> — one session (see lib/routes.ts). An r_ ref is replaced by the canonical s_ id
// once the API answers.
import { Suspense } from "react";

import { DetailIdGate } from "@/components/detail-id-gate";
import { SessionDetail } from "@/components/sessions/session-detail";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <DetailIdGate listHref="/sessions">{(id) => <SessionDetail id={id} />}</DetailIdGate>
    </Suspense>
  );
}
