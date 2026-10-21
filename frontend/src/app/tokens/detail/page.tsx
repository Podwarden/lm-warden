"use client";
// /tokens/detail?id=<id> — the token detail page (see lib/routes.ts). The
// range selection shares the query with the id. Old /tokens/<id> bookmarks
// are redirected here server-side (next.config.ts, app/webui/redirects.py).
import { Suspense } from "react";
import { DetailIdGate } from "@/components/detail-id-gate";
import { TokenDetailPage } from "@/components/tokens/token-detail-page";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <DetailIdGate listHref="/tokens">{(id) => <TokenDetailPage id={id} />}</DetailIdGate>
    </Suspense>
  );
}
