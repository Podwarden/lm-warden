"use client";
// /models/detail?id=<id> — the model detail page. The id is runtime data, so
// it rides in the query rather than a /models/[id] segment (static export,
// BACKLOG-macos P4; see lib/routes.ts). Old /models/<id> bookmarks are
// redirected here server-side (next.config.ts, app/webui/redirects.py).
import { Suspense } from "react";
import { DetailIdGate } from "@/components/detail-id-gate";
import { ModelDetailPage } from "@/components/models/model-detail-page";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <DetailIdGate listHref="/models">{(id) => <ModelDetailPage id={id} />}</DetailIdGate>
    </Suspense>
  );
}
