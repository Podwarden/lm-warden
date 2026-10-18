'use client';
// ---------------------------------------------------------------------------
// SessionGate — don't paint the app until we know who is looking at it.
// ---------------------------------------------------------------------------
//
// Before this gate, a signed-out visitor opening /ui/ saw the Models page —
// heading, skeleton cards, the lot — for the length of one round trip, and was
// then bounced to /ui/login. The first impression of the product was a page
// that flashed empty and threw the visitor out; "it's broken" is a fair
// reading of that.
//
// The cause is ordering, not auth: the shell renders immediately, and the
// session is only discovered later by whichever SWR fetcher gets a 401 first.
// So resolve it up front instead. `ensureSession()` performs the one
// `POST /api/auth/refresh` that `authFetch`'s eager-refresh guard was going to
// make anyway before its first request, and `refresh()` de-dupes concurrent
// callers — so this costs no extra network, it just moves a decision earlier.
//
// Three outcomes:
//   'ok'        — render the shell.
//   'transient' — render the shell anyway. The backend is restarting or the
//                 link flapped; the cookie was never rejected. Wedging the
//                 whole UI behind a spinner during a backend bounce would be
//                 a worse failure than the one being fixed, and each page's
//                 own error handling is already written for this.
//   'invalid'   — the session really is dead. Redirect and stay closed, so
//                 nothing behind the gate ever paints.
//
// /login and the /setup/* wizard render unconditionally: they are the pages
// you reach WITHOUT a session, and gating them would deadlock the product on
// first run. The route list mirrors `isUnauthRoute` in nav-bar.tsx — the same
// question ("is this page for someone not signed in?") asked by the two
// components that both have to know the answer.

import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';

import { ensureSession, getAccessToken, redirectToLogin } from '@/lib/auth-fetch';

// Exact match, not `startsWith('/login')` — see nav-bar.tsx's #39 note: a
// hypothetical /login-help page must NOT inherit the unauthenticated
// treatment. The wizard genuinely owns its whole subtree.
const UNAUTH_EXACT_PATHS: ReadonlySet<string> = new Set([
  '/login',
  '/login/',
  '/setup',
  '/setup/',
]);

export function isUnauthRoute(path: string): boolean {
  return UNAUTH_EXACT_PATHS.has(path) || path.startsWith('/setup/');
}

export function SessionGate({ children }: { children: React.ReactNode }) {
  // `usePathname` reports the path WITHOUT the '/ui' basePath, so these
  // compare against '/login', not '/ui/login'.
  const path = usePathname() ?? '';
  const openToEveryone = isUnauthRoute(path);
  // An admin session opens the gate for every page, for the life of the
  // layout. Once opened it stays opened — a client-side navigation does not
  // re-check the session.
  const [opened, setOpened] = useState(false);

  useEffect(() => {
    if (openToEveryone || opened) return;
    // A token already in memory means a client-side navigation, not a fresh
    // load; there is nothing to wait for.
    if (getAccessToken() !== null) {
      setOpened(true);
      return;
    }
    let cancelled = false;
    void ensureSession().then((check) => {
      if (cancelled) return;
      if (check === 'invalid') {
        // Stay closed. The navigation is what ends this render.
        redirectToLogin();
        return;
      }
      setOpened(true);
    });
    return () => {
      cancelled = true;
    };
  }, [openToEveryone, opened, path]);

  if (openToEveryone || opened) return <>{children}</>;

  return (
    <div
      data-testid="session-gate-pending"
      className="flex items-center justify-center py-24"
      role="status"
      aria-label="Checking your session"
    >
      <span className="h-6 w-6 animate-spin rounded-full border-2 border-slate-600 border-t-emerald-500" />
    </div>
  );
}
