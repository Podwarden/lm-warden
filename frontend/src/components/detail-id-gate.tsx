"use client";

/**
 * DetailIdGate — the query-string half of a detail route (BACKLOG-macos P4).
 *
 * /models/detail, /models/detail/settings and /tokens/detail take the id from
 * `?id=` (a static export cannot pre-render a page per runtime id; see
 * lib/routes.ts). This reads it and renders the page body keyed by it, so
 * moving from one id to another remounts the body exactly as the old
 * `/models/[id]` segment change did. Without an id there is nothing to show:
 * the gate client-redirects to the list page instead.
 *
 * `useSearchParams` opts the tree into client-side rendering, so every caller
 * wraps this in <Suspense> (same as chat2/page.tsx).
 */

import { Fragment, useEffect, type ReactNode } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export function DetailIdGate({
  listHref,
  children,
}: {
  /** Where an id-less URL goes (the list page: /models or /tokens). */
  listHref: string;
  children: (id: string) => ReactNode;
}) {
  const id = useSearchParams()?.get("id") ?? "";
  const router = useRouter();

  useEffect(() => {
    if (!id) router.replace(listHref);
  }, [id, listHref, router]);

  if (!id) return null;
  return <Fragment key={id}>{children(id)}</Fragment>;
}
