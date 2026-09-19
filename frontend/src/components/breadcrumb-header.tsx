"use client";

/**
 * BreadcrumbHeader — the one-row strip under the nav bar on every app-shell
 * page. Ported from PodWarden Core's `components/breadcrumb-header.tsx`
 * (#1187 / #1687). Two things in one row:
 *
 *  1. **Smart back button** (left). `← <previous page>` when the session has
 *     history (see `NavStackProvider`), `← Home` otherwise — and not shown at
 *     all when it would lead to the page you are on (Models, on a fresh load:
 *     Home navigates to /models, see HOME_HREF).
 *  2. **Route-derived breadcrumb**, from the registry in `lib/breadcrumbs.ts`
 *     with titles supplied by pages through `useBreadcrumb`.
 *
 * Hidden where the nav bar is hidden: /login and the /setup wizard.
 *
 * Every href is an app-router path and goes through next/link or
 * `router.push`, so the `/ui` basePath is added by Next, never by hand.
 * Colours are the chat-* theme tokens, so the strip follows Retro and Retro
 * Dark. It is exactly BREADCRUMB_BAR_PX tall — chat2's full-height shell
 * subtracts it (keep the two in step).
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMemo, useState } from "react";
import { ArrowLeft, ChevronRight, Ellipsis, House } from "lucide-react";
import { useNavStack } from "@/lib/nav-stack";
import {
  buildBreadcrumbs,
  HOME_HREF,
  navTarget,
  shouldHideBreadcrumb,
  type BreadcrumbEntry,
} from "@/lib/breadcrumbs";
import { cn } from "@/lib/utils";

/** Total height of the strip, rule included; `h-[30px]` below and chat2's
 *  `100vh` calc spell it out (Tailwind needs literal class names). */
export const BREADCRUMB_BAR_PX = 30;

const DEFAULT_MAX_CHARS = 28;
const BACK_MAX_CHARS = 24;

export function truncate(label: string, maxChars: number | undefined): string {
  const limit = maxChars ?? DEFAULT_MAX_CHARS;
  if (label.length <= limit) return label;
  return label.slice(0, Math.max(0, limit - 1)) + "…";
}

const FOCUS = "focus:outline-none focus-visible:ring-1 focus-visible:ring-chat-accent";
const CHEVRON = <ChevronRight className="h-3 w-3 shrink-0 text-chat-dim/70" aria-hidden="true" />;

export function BreadcrumbHeader() {
  const pathname = usePathname() ?? "/";
  const { canGoBack, topOfStack, goBack, captureScroll, overrides } = useNavStack();
  const hidden = shouldHideBreadcrumb(pathname);

  const chain = useMemo(
    () => (hidden ? [] : buildBreadcrumbs(pathname, overrides)),
    [hidden, pathname, overrides],
  );

  if (hidden) return null;

  const backLabel = canGoBack && topOfStack ? topOfStack.title : "Home";
  // A back button that lands on this very page is dead weight: hide it.
  const backTarget = canGoBack && topOfStack ? topOfStack.href : HOME_HREF;
  const showBack = backTarget !== pathname;
  const backAria = `Back to ${backLabel}`;
  const backClass = cn(
    "-ml-1.5 inline-flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-chat-muted",
    "hover:bg-chat-surface-2/60 hover:text-chat-accent",
    FOCUS,
  );
  const backInner = (
    <>
      <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
      <span className="max-w-[12rem] truncate">{truncate(backLabel, BACK_MAX_CHARS)}</span>
    </>
  );

  return (
    // The 30 px includes the 1 px rule (border-box), so the strip's total
    // height is exactly BREADCRUMB_BAR_PX.
    <div data-testid="breadcrumb-header" className="h-[30px] border-b border-chat-rule/50">
      <div className="container mx-auto flex h-full items-center gap-2.5 px-6 text-[13px] leading-none">
        {showBack && (
          <>
            {canGoBack ? (
              <button type="button" onClick={goBack} aria-label={backAria} title={backAria} className={backClass}>
                {backInner}
              </button>
            ) : (
              <Link href={HOME_HREF} onClick={captureScroll} aria-label={backAria} title={backAria} className={backClass}>
                {backInner}
              </Link>
            )}
            <div className="h-3.5 w-px shrink-0 bg-chat-rule" aria-hidden="true" />
          </>
        )}

        <BreadcrumbTrail chain={chain} pathname={pathname} onNavigate={captureScroll} />
      </div>
    </div>
  );
}

function BreadcrumbTrail({
  chain,
  pathname,
  onNavigate,
}: {
  chain: BreadcrumbEntry[];
  pathname: string;
  onNavigate: () => void;
}) {
  const [mobileExpanded, setMobileExpanded] = useState(false);

  if (chain.length === 0) return null;

  // Mobile collapses the middle segments behind `…`. Two renderings
  // (sm:hidden + hidden sm:flex) keep the markup semantic at both sizes.
  const first = chain[0];
  const last = chain[chain.length - 1];
  const middle = chain.slice(1, -1);

  return (
    <nav aria-label="Breadcrumb" className="flex min-w-0 flex-1 items-center text-chat-dim">
      {/* Desktop: the full chain */}
      <ol className="hidden min-w-0 items-center gap-1 sm:flex">
        {chain.map((seg, idx) => (
          <li key={seg.href} className="flex min-w-0 items-center gap-1">
            {idx > 0 && CHEVRON}
            <BreadcrumbSegment pathname={pathname} entry={seg} isCurrent={idx === chain.length - 1} onNavigate={onNavigate} />
          </li>
        ))}
      </ol>

      {/* Mobile: Home › … › Current */}
      <ol className="flex min-w-0 items-center gap-1 sm:hidden">
        <li className="flex items-center gap-1">
          <BreadcrumbSegment pathname={pathname} entry={first} isCurrent={chain.length === 1} onNavigate={onNavigate} iconOnly />
        </li>
        {middle.length > 0 && (
          <li className="flex items-center gap-1">
            {CHEVRON}
            {mobileExpanded ? (
              middle.map((seg, idx) => (
                <span key={seg.href} className="flex items-center gap-1">
                  {idx > 0 && CHEVRON}
                  <BreadcrumbSegment pathname={pathname} entry={seg} isCurrent={false} onNavigate={onNavigate} />
                </span>
              ))
            ) : (
              <button
                type="button"
                onClick={() => setMobileExpanded(true)}
                aria-label="Show hidden breadcrumb segments"
                className={cn("inline-flex items-center rounded px-1 py-0.5 text-chat-dim hover:text-chat-fg", FOCUS)}
              >
                <Ellipsis className="h-3.5 w-3.5" aria-hidden="true" />
              </button>
            )}
          </li>
        )}
        {chain.length > 1 && (
          <li className="flex min-w-0 items-center gap-1">
            {CHEVRON}
            <BreadcrumbSegment pathname={pathname} entry={last} isCurrent onNavigate={onNavigate} />
          </li>
        )}
      </ol>
    </nav>
  );
}

function BreadcrumbSegment({
  entry,
  pathname,
  isCurrent,
  onNavigate,
  iconOnly,
}: {
  entry: BreadcrumbEntry;
  pathname: string;
  isCurrent: boolean;
  onNavigate: () => void;
  iconOnly?: boolean;
}) {
  const text = truncate(entry.label, entry.maxChars);
  // Home's link goes to /models (HOME_HREF); on /models itself that would be
  // a link to the current page, so it renders as plain text there.
  const target = navTarget(entry.href);
  const selfLink = target === pathname;

  if (iconOnly && entry.href === "/") {
    const icon = (
      <>
        <House className="h-3.5 w-3.5" aria-hidden="true" />
        <span className="sr-only">{entry.label}</span>
      </>
    );
    if (isCurrent || selfLink) {
      return (
        <span
          aria-current={isCurrent ? "page" : undefined}
          title={entry.label}
          className={cn("inline-flex items-center", isCurrent ? "text-chat-fg" : "text-chat-dim")}
        >
          {icon}
        </span>
      );
    }
    return (
      <Link
        href={target}
        onClick={onNavigate}
        title={entry.label}
        className={cn("inline-flex items-center rounded px-0.5 py-0.5 text-chat-muted hover:text-chat-accent", FOCUS)}
      >
        {icon}
      </Link>
    );
  }

  if (isCurrent) {
    return (
      <span
        aria-current="page"
        title={entry.label}
        className={cn(
          "truncate",
          entry.pending ? "italic text-chat-dim" : "font-medium text-chat-fg",
        )}
      >
        {text}
      </span>
    );
  }

  // A crumb whose path has no page.tsx is plain text — no link, no 404. So
  // is one that would only link to the page you are on (Home, on /models).
  if (entry.navigable === false || selfLink) {
    return (
      <span title={entry.label} className="truncate text-chat-dim">
        {text}
      </span>
    );
  }

  return (
    <Link
      href={target}
      onClick={onNavigate}
      title={entry.label}
      className={cn("truncate rounded text-chat-muted hover:text-chat-accent hover:underline", FOCUS)}
    >
      {text}
    </Link>
  );
}
