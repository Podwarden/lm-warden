"use client";

/**
 * NavStackProvider — the in-memory navigation stack behind the app-shell
 * back button and breadcrumb header. Ported from PodWarden Core's
 * `contexts/NavStackContext.tsx` (#1187).
 *
 * - The stack is **session-scoped** and lives entirely in React state. It is
 *   deliberately NOT persisted: a hard reload, a new tab or an external link
 *   starts with an empty stack, and the back button falls back to `← Home`.
 * - Each entry captures the page title (resolved against the breadcrumb
 *   registry + per-page overrides) and the window `scrollY` at the moment
 *   the user left it. On a forward navigation the *outgoing* page is pushed;
 *   `goBack` pops it and restores its scroll.
 * - Pages without the app shell (/login, /setup/*) never go on the stack,
 *   and entering one clears it: signing out and back in starts a fresh
 *   trail instead of offering `← login`.
 *
 * Adapted from PodWarden: stored hrefs are app-router paths (no `/ui`
 * basePath — `router.push` adds it), the inner-pane scroll registry
 * (`useRestorableScroll`) is left out because no LM Warden page needs it,
 * and `goBack` navigates outside the state updater (an updater must stay
 * pure; StrictMode runs it twice).
 *
 * The pathname the provider tracks (and exposes) is the *logical* one: on a
 * detail page, whose id rides in the query (/models/detail?id=abc,
 * BACKLOG-macos P4), it is /models/abc — see lib/routes.ts. The query is read
 * by a small child inside its own <Suspense> (useSearchParams needs one in a
 * static export) and reported up; until it has reported for the current
 * detail URL the provider is not `ready` and its effects hold still, so a
 * half-known path never lands on the stack.
 */

import {
  createContext,
  Suspense,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  HOME_HREF,
  navTarget,
  resolvePageTitle,
  shouldHideBreadcrumb,
  type BreadcrumbOverrides,
} from "@/lib/breadcrumbs";
import { isQueryDetailPath, logicalPath } from "@/lib/routes";

/** A single entry in the back navigation stack. */
export interface NavStackEntry {
  /** Logical app-router path (no basePath), e.g. `/tokens/abc`; `navTarget`
   *  turns it into the URL to push. */
  href: string;
  /** Human-readable page title rendered in `← <title>`. */
  title: string;
  /** Window scroll at the moment of outbound navigation. */
  scrollY: number;
}

export interface NavStackContextValue {
  /** The current logical app-router pathname (no basePath): /models/abc on
   *  /models/detail?id=abc, see lib/routes.ts. */
  pathname: string;
  /** Frozen view of the back-stack (most-recent last). */
  stack: NavStackEntry[];
  /** Snapshot the current window scroll (call right before navigating). */
  captureScroll: () => void;
  /** True when the back stack is non-empty (i.e. show `← <prev title>`). */
  canGoBack: boolean;
  /** The most-recent stack entry, used to render `← <prev.title>`. */
  topOfStack: NavStackEntry | null;
  /** Navigate back (pop + restore scroll). Falls back to Home (HOME_HREF) when empty. */
  goBack: () => void;
  /**
   * Per-path title override. Called by `useBreadcrumb({title})` so the
   * breadcrumb reads `Qwen3-8B` instead of the model's opaque id.
   * `null` removes the override.
   */
  setBreadcrumbTitle: (pathname: string, title: string | null, parent?: string) => void;
  /** Map of per-pathname overrides used by `buildBreadcrumbs`. */
  overrides: BreadcrumbOverrides;
}

/** A session that wanders for hours must not grow the stack without bound. */
export const MAX_STACK = 50;

export const NavStackContext = createContext<NavStackContextValue | null>(null);

function deriveFallbackTitle(pathname: string): string {
  if (pathname === "/") return "Home";
  const last = pathname.split("/").filter(Boolean).pop() ?? "";
  try {
    return decodeURIComponent(last) || "Home";
  } catch {
    return last || "Home";
  }
}

/** Reports the query string of the URL `path` was read with. */
function SearchReporter({ onSearch }: { onSearch: (path: string, search: string) => void }) {
  const path = usePathname() ?? "/";
  const search = useSearchParams()?.toString() ?? "";
  useEffect(() => {
    onSearch(path, search);
  }, [path, search, onSearch]);
  return null;
}

export function NavStackProvider({ children }: { children: ReactNode }) {
  const rawPath = usePathname() ?? "/";
  const router = useRouter();

  // The last query string SearchReporter saw, with the path it saw it on: a
  // detail page's logical path is known only once that path is `rawPath`.
  const [seen, setSeen] = useState<{ path: string; search: string } | null>(null);
  const onSearch = useCallback((path: string, search: string) => {
    setSeen((cur) => (cur && cur.path === path && cur.search === search ? cur : { path, search }));
  }, []);
  const queryRoute = isQueryDetailPath(rawPath);
  const ready = !queryRoute || seen?.path === rawPath;
  const pathname = queryRoute && ready ? logicalPath(rawPath, seen!.search) : rawPath;

  const [stack, setStack] = useState<NavStackEntry[]>([]);
  const [overrides, setOverrides] = useState<BreadcrumbOverrides>({});

  /** Latest stack, read by `goBack` without going through an updater. */
  const stackRef = useRef<NavStackEntry[]>(stack);
  stackRef.current = stack;

  /** Set by `goBack`: the window scroll the destination should restore. */
  const restoreScrollRef = useRef<number | null>(null);
  /** The pathname we last saw — `null` until first mount. */
  const lastPathRef = useRef<string | null>(null);
  /** The most recent window scroll of the *current* page. */
  const currentScrollRef = useRef(0);
  /** The current page's title, captured into the stack as we leave. */
  const currentTitleRef = useRef("Home");

  useEffect(() => {
    function onScroll() {
      currentScrollRef.current = window.scrollY;
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Detect outbound nav: push the *previous* page onto the back-stack.
  //
  // ORDERING INVARIANT (PodWarden #1187 AC-3): this effect MUST be declared
  // BEFORE the title-sync effect below. Both depend on `pathname` and React
  // runs same-commit effects in declaration order, so declaring this one
  // first guarantees `currentTitleRef.current` still holds the *leaving*
  // page's title when it is snapshotted. Reordering them makes the back
  // button show the destination's own title.
  useEffect(() => {
    if (!ready) return; // detail URL whose id is not read yet
    const prev = lastPathRef.current;
    lastPathRef.current = pathname;
    if (prev === null || prev === pathname) return; // first mount / no-op

    const restoreY = restoreScrollRef.current;
    if (restoreY !== null) {
      // Arrived via `goBack`: the entry was already popped; restore scroll
      // on the next frame, after Next has painted the page.
      restoreScrollRef.current = null;
      requestAnimationFrame(() => window.scrollTo({ top: restoreY, behavior: "auto" }));
      currentScrollRef.current = restoreY;
      return;
    }

    if (shouldHideBreadcrumb(pathname)) {
      // Entering a no-shell page (sign-out, setup): the trail ends here.
      setStack([]);
    } else if (!shouldHideBreadcrumb(prev)) {
      const leaving: NavStackEntry = {
        href: prev,
        title: currentTitleRef.current,
        scrollY: currentScrollRef.current,
      };
      setStack((s) => {
        // Collapse a duplicate consecutive entry into the fresher snapshot.
        const base = s.length > 0 && s[s.length - 1].href === leaving.href ? s.slice(0, -1) : s;
        return [...base, leaving].slice(-MAX_STACK);
      });
    }
    currentScrollRef.current = 0;
  }, [pathname, ready]);

  // Keep currentTitleRef in sync with the active override / route registry.
  // MUST be declared AFTER the push effect above — see ORDERING INVARIANT.
  useEffect(() => {
    if (!ready) return;
    currentTitleRef.current = resolvePageTitle(pathname, overrides) ?? deriveFallbackTitle(pathname);
  }, [pathname, overrides, ready]);

  // Drop overrides for paths that aren't the current page or one of its URL
  // ancestors, so the map doesn't accumulate stale titles. The new page's
  // own `useBreadcrumb` effect runs before this one (children first) and its
  // path is kept.
  useEffect(() => {
    if (!ready) return;
    setOverrides((prev) => {
      const keep: BreadcrumbOverrides = {};
      for (const key of Object.keys(prev)) {
        if (pathname === key || pathname.startsWith(key + "/") || key === "/") keep[key] = prev[key];
      }
      return Object.keys(keep).length === Object.keys(prev).length ? prev : keep;
    });
  }, [pathname, ready]);

  const captureScroll = useCallback(() => {
    currentScrollRef.current = window.scrollY;
  }, []);

  const goBack = useCallback(() => {
    // Read the latest stack from the ref and navigate here, not inside a
    // `setStack` updater: updaters must be pure (StrictMode runs them twice,
    // which would push twice). The top entry is never the current page —
    // pushes collapse equal neighbours and always push the page being left.
    const s = stackRef.current;
    if (s.length === 0) {
      router.push(HOME_HREF);
      return;
    }
    const top = s[s.length - 1];
    const rest = s.slice(0, -1);
    stackRef.current = rest; // a second click before the router lands goes one further back
    setStack(rest);
    restoreScrollRef.current = top.scrollY;
    // `scroll: false` — we restore the saved position ourselves; letting
    // Next scroll to the top first would flash the page's head.
    router.push(navTarget(top.href), { scroll: false });
  }, [router]);

  const setBreadcrumbTitle = useCallback((path: string, title: string | null, parent?: string) => {
    setOverrides((prev) => {
      const cur = prev[path];
      if (title === null) {
        if (!cur) return prev;
        const next = { ...prev };
        delete next[path];
        return next;
      }
      if (cur && cur.title === title && cur.parent === parent) return prev;
      return { ...prev, [path]: { title, parent } };
    });
  }, []);

  const value = useMemo<NavStackContextValue>(
    () => ({
      pathname,
      stack,
      captureScroll,
      canGoBack: stack.length > 0,
      topOfStack: stack.length > 0 ? stack[stack.length - 1] : null,
      goBack,
      setBreadcrumbTitle,
      overrides,
    }),
    [pathname, stack, captureScroll, goBack, setBreadcrumbTitle, overrides],
  );

  return (
    <NavStackContext.Provider value={value}>
      <Suspense fallback={null}>
        <SearchReporter onSearch={onSearch} />
      </Suspense>
      {children}
    </NavStackContext.Provider>
  );
}

export function useNavStack(): NavStackContextValue {
  const ctx = useContext(NavStackContext);
  if (!ctx) throw new Error("useNavStack must be used inside <NavStackProvider>");
  return ctx;
}
