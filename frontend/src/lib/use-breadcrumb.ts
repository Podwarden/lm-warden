"use client";

import { useContext, useEffect } from "react";
import { NavStackContext } from "@/lib/nav-stack";

/**
 * Register a human-readable title for a page in the breadcrumb trail. Use on
 * dynamic detail pages whose URL segment is an opaque id — calling
 * `useBreadcrumb({title: model.served_model_name})` makes the trail read
 * `Models › Qwen3-8B` instead of `Models › 3f9c…`. The same title is what the
 * back button shows (`← Qwen3-8B`) after the user moves on.
 *
 * - `title` may be `undefined` while the page is loading; the crumb shows the
 *   route's placeholder label (in the pending style) until it arrives.
 * - `path` names a page other than the current one — a child page naming its
 *   dynamic parent (the model settings page titles `/models/[id]`). Defaults
 *   to the current pathname.
 * - `parent` overrides the URL-derived parent chain.
 *
 * Overrides are keyed by exact pathname and garbage-collected by
 * `NavStackProvider` once the user navigates outside that path. Outside a
 * provider (a page rendered on its own in a unit test) this is a no-op.
 *
 * Ported from PodWarden Core's `hooks/use-breadcrumb.ts`; `path`, reading the
 * pathname from the provider and the provider-less no-op are LM Warden
 * additions.
 */
export function useBreadcrumb(opts: {
  title: string | null | undefined;
  path?: string;
  parent?: string;
}) {
  // The provider's pathname rather than `usePathname()`: the hook then needs
  // nothing from next/navigation, so a page's unit test that mocks only
  // `useRouter` keeps working.
  const ctx = useContext(NavStackContext);
  const setBreadcrumbTitle = ctx?.setBreadcrumbTitle;
  const { title, parent } = opts;
  const target = opts.path ?? ctx?.pathname ?? "/";

  useEffect(() => {
    if (!title || !setBreadcrumbTitle) return;
    // No teardown on purpose: clearing on every change would flicker an
    // async-loaded title back to the placeholder between renders. The
    // provider drops the override when the user leaves the path.
    setBreadcrumbTitle(target, title, parent);
  }, [target, title, parent, setBreadcrumbTitle]);
}
