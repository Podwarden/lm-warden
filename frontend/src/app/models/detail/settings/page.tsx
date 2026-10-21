"use client";
// /models/detail/settings?id=<id> — the per-model settings page (see
// models/detail/page.tsx and lib/routes.ts). Old /models/<id>/settings
// bookmarks are redirected here server-side.
import { Suspense } from "react";
import { DetailIdGate } from "@/components/detail-id-gate";
import { ModelSettingsPage } from "@/components/models/model-settings-page";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <DetailIdGate listHref="/models">{(id) => <ModelSettingsPage id={id} />}</DetailIdGate>
    </Suspense>
  );
}
